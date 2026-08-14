"""Transparent face-quality evidence tests; no native model is loaded."""

from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from retouch.detection import FaceData, _LandmarkCompat
from retouch.face_quality import (
    FACE_QUALITY_VERSION,
    SHARPNESS_METHOD,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    FaceQualityAnalyzer,
)
from retouch.shoot_intelligence import inspect_asset
from retouch.shoot_review import FaceQualityEvidence, build_review_manifest


def _landmarks():
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(468)]
    for center_x, indices in ((0.65, LEFT_EYE_INDICES), (0.35, RIGHT_EYE_INDICES)):
        for position, index in enumerate(indices):
            angle = 2.0 * np.pi * position / len(indices)
            points[index] = SimpleNamespace(
                x=center_x + 0.065 * float(np.cos(angle)),
                y=0.43 + 0.028 * float(np.sin(angle)),
                z=0.0,
            )
    return _LandmarkCompat(points)


def _face(*, confidence=0.91, source="retinaface", bbox=(16, 10, 96, 108)):
    return FaceData(
        landmarks=_landmarks(),
        bbox=bbox,
        ied=38.0,
        confidence=confidence,
        confidence_source=source,
    )


def test_face_quality_records_versioned_geometry_focus_and_provenance():
    yy, xx = np.indices((128, 128))
    checker = (((xx // 3 + yy // 3) % 2) * 255).astype(np.uint8)
    image = cv2.cvtColor(checker, cv2.COLOR_GRAY2BGR)

    evidence = FaceQualityAnalyzer().analyze(image, [_face()])[0]

    assert evidence.measurement_version == FACE_QUALITY_VERSION
    assert evidence.sharpness_method == SHARPNESS_METHOD
    assert evidence.bbox is not None
    assert 0.0 < evidence.coverage <= 1.0
    assert evidence.detector_confidence == 0.91
    assert evidence.detector_confidence_source == "retinaface"
    assert evidence.face_sharpness > 0.0
    assert evidence.left_eye_sharpness > 0.0
    assert evidence.right_eye_sharpness > 0.0
    assert evidence.eyes_open == "uncertain"
    assert "blink_analysis_deferred" in evidence.uncertainty
    assert evidence.measurement_details["face_bbox_pixels"] == [16, 10, 96, 108]


def test_face_quality_sharpness_decreases_under_strong_blur():
    rng = np.random.default_rng(7)
    sharp = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (0, 0), sigmaX=5.0, sigmaY=5.0)
    analyzer = FaceQualityAnalyzer()

    sharp_evidence = analyzer.analyze(sharp, [_face()])[0]
    blurred_evidence = analyzer.analyze(blurred, [_face()])[0]

    assert sharp_evidence.face_sharpness > blurred_evidence.face_sharpness
    assert sharp_evidence.left_eye_sharpness > blurred_evidence.left_eye_sharpness
    assert sharp_evidence.right_eye_sharpness > blurred_evidence.right_eye_sharpness


def test_unmeasured_detector_confidence_is_not_promoted_from_compat_default():
    image = np.full((128, 128, 3), 100, dtype=np.uint8)
    evidence = FaceQualityAnalyzer().analyze(image, [_face(source="unknown")])[0]

    assert evidence.detector_confidence is None
    assert "detector_confidence_unavailable" in evidence.uncertainty
    assert "face_local_contrast_unavailable" in evidence.uncertainty


def test_tiny_face_and_missing_landmarks_are_explicitly_uncertain():
    image = np.full((100, 100, 3), 80, dtype=np.uint8)
    face = FaceData(
        landmarks=None,
        bbox=(10, 10, 20, 20),
        ied=3.0,
        confidence_source="mediapipe_presence_unavailable",
    )

    evidence = FaceQualityAnalyzer().analyze(image, [face])[0]

    assert "face_crop_too_small" in evidence.uncertainty
    assert "left_eye_landmarks_unavailable" in evidence.uncertainty
    assert "right_eye_landmarks_unavailable" in evidence.uncertainty
    assert evidence.left_eye_sharpness is None
    assert evidence.right_eye_sharpness is None


def _manifest_face(observation_id, bbox, sharpness):
    return FaceQualityEvidence(
        face_id=observation_id,
        bbox=bbox,
        coverage=bbox[2] * bbox[3],
        face_sharpness=sharpness,
        eyes_open="uncertain",
        uncertainty=["blink_analysis_deferred"],
    )


def test_manifest_reconciles_face_ids_independently_of_detector_order(tmp_path):
    source = tmp_path / "group.jpg"
    Image.new("RGB", (120, 80), (120, 100, 80)).save(source)
    assets = [inspect_asset(source)]
    path_key = str(source.resolve())
    first_faces = [
        _manifest_face("observation-left", (0.10, 0.15, 0.25, 0.55), 4.0),
        _manifest_face("observation-right", (0.60, 0.15, 0.25, 0.55), 8.0),
    ]
    first = build_review_manifest(
        tmp_path, assets, [], {}, face_evidence_by_path={path_key: first_faces},
    )
    first_record = next(iter(first.assets.values()))
    ids_by_side = {"left" if face.bbox[0] < 0.5 else "right": face.face_id for face in first_record.faces}

    rescanned_faces = [
        _manifest_face("observation-new-right", (0.605, 0.15, 0.25, 0.55), 7.5),
        _manifest_face("observation-new-left", (0.105, 0.15, 0.25, 0.55), 3.5),
    ]
    second = build_review_manifest(
        tmp_path, assets, [], {}, face_evidence_by_path={path_key: rescanned_faces},
    )
    second_record = next(iter(second.assets.values()))
    rescanned_ids = {"left" if face.bbox[0] < 0.5 else "right": face.face_id for face in second_record.faces}

    assert rescanned_ids == ids_by_side


def test_metadata_only_rescan_preserves_same_content_face_evidence(tmp_path):
    source = tmp_path / "portrait.jpg"
    Image.new("RGB", (80, 80), (100, 100, 100)).save(source)
    assets = [inspect_asset(source)]
    path_key = str(source.resolve())
    first = build_review_manifest(
        tmp_path,
        assets,
        [],
        {},
        face_evidence_by_path={
            path_key: [_manifest_face("observation-one", (0.2, 0.1, 0.5, 0.7), 3.0)]
        },
    )
    first_id = next(iter(first.assets.values())).faces[0].face_id

    second = build_review_manifest(tmp_path, assets, [], {})
    record = next(iter(second.assets.values()))

    assert record.faces[0].face_id == first_id
    assert "face_quality_not_refreshed" in record.uncertainty
    assert "face_quality_unavailable" not in record.uncertainty


def test_content_change_does_not_carry_face_identity_or_measurements(tmp_path):
    source = tmp_path / "portrait.jpg"
    Image.new("RGB", (80, 80), (100, 100, 100)).save(source)
    path_key = str(source.resolve())
    first = build_review_manifest(
        tmp_path,
        [inspect_asset(source)],
        [],
        {},
        face_evidence_by_path={
            path_key: [_manifest_face("observation-one", (0.2, 0.1, 0.5, 0.7), 3.0)]
        },
    )
    assert next(iter(first.assets.values())).faces

    Image.new("RGB", (80, 80), (220, 220, 220)).save(source)
    second = build_review_manifest(tmp_path, [inspect_asset(source)], [], {})
    record = next(iter(second.assets.values()))

    assert record.faces == []
    assert "face_quality_unavailable" in record.uncertainty
    assert record.identity_history[-1]["kind"] == "content_changed"


def test_ambiguous_geometry_does_not_silently_reuse_face_identity(tmp_path):
    source = tmp_path / "overlap.jpg"
    Image.new("RGB", (100, 100), (90, 90, 90)).save(source)
    assets = [inspect_asset(source)]
    path_key = str(source.resolve())
    first = build_review_manifest(
        tmp_path,
        assets,
        [],
        {},
        face_evidence_by_path={path_key: [
            _manifest_face("observation-a", (0.10, 0.10, 0.50, 0.60), 4.0),
            _manifest_face("observation-b", (0.18, 0.10, 0.50, 0.60), 5.0),
        ]},
    )
    prior_ids = {face.face_id for face in next(iter(first.assets.values())).faces}

    second = build_review_manifest(
        tmp_path,
        assets,
        [],
        {},
        face_evidence_by_path={path_key: [
            _manifest_face("observation-current", (0.14, 0.10, 0.50, 0.60), 4.5),
        ]},
    )
    current = next(iter(second.assets.values())).faces[0]

    assert current.face_id not in prior_ids
    assert "face_identity_ambiguous" in current.uncertainty
