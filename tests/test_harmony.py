"""Regression tests for read-only portrait harmony QA."""

import cv2
import numpy as np
import pytest

from retouch.harmony import build_face_anchored_body_mask, evaluate_harmony
from retouch.qa_detectors import run_all


def _portrait_fixture():
    rng = np.random.RandomState(7)
    img = np.full((160, 128, 3), (105, 145, 190), dtype=np.float32)
    img += rng.normal(0, 12, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    face = np.zeros(img.shape[:2], dtype=np.float32)
    body = np.zeros_like(face)
    face[20:76, 36:92] = 1.0
    body[84:152, 20:108] = 1.0
    return img, face, body


def test_harmony_detects_relative_texture_parity_drift():
    original, face, body = _portrait_fixture()
    processed = original.copy()
    processed[face > 0.5] = cv2.GaussianBlur(processed, (0, 0), 3.0)[face > 0.5]

    result = evaluate_harmony(
        processed,
        face_skin_mask=face,
        body_skin_mask=body,
        reference_img_bgr=original,
    )

    assert result["available"] is True
    assert result["texture_parity_drift"] > 0.2
    # Calibration is intentionally observational until the real corpus exists.
    assert result["flagged"] is False


def test_face_anchored_body_mask_is_tone_relative():
    for bgr in ((189, 208, 244), (143, 180, 231), (109, 152, 208),
                (87, 114, 165), (62, 82, 122), (38, 51, 80)):
        img = np.full((160, 128, 3), bgr, dtype=np.uint8)
        face = np.zeros(img.shape[:2], dtype=np.float32)
        person = np.ones_like(face)
        face[16:76, 36:92] = 1.0
        body = build_face_anchored_body_mask(img, face, person)
        assert int((body > 0.5).sum()) >= 512
        assert float(body[face > 0.5].max()) == 0.0


# Fitzpatrick I-VI-like skin values (BGR), same sweep as tests/test_freckle.py.
_FITZPATRICK_BGR = ((189, 208, 244), (143, 180, 231), (109, 152, 208),
                    (87, 114, 165), (62, 82, 122), (38, 51, 80))


def _textured_portrait(bgr):
    """Face/body swatch pair with multiplicative (tone-scaled) skin texture."""
    rng = np.random.RandomState(11)
    base = np.full((160, 128, 3), bgr, dtype=np.float32)
    img = base * (1.0 + rng.normal(0, 0.08, base.shape).astype(np.float32))
    img = np.clip(img, 0, 255).astype(np.uint8)
    face = np.zeros(img.shape[:2], dtype=np.float32)
    body = np.zeros_like(face)
    face[16:76, 32:96] = 1.0
    body[88:152, 16:112] = 1.0
    return img, face, body


@pytest.mark.parametrize("bgr", _FITZPATRICK_BGR)
def test_texture_parity_drift_discriminates_at_every_tone(bgr):
    """Face-only smoothing must flag, matched smoothing must pass, I-VI."""
    original, face, body = _textured_portrait(bgr)
    blurred = cv2.GaussianBlur(original, (0, 0), 3.0)

    face_only = original.copy()
    face_only[face > 0.5] = blurred[face > 0.5]
    matched = original.copy()
    both = np.maximum(face, body) > 0.5
    matched[both] = blurred[both]

    drift_face_only = evaluate_harmony(
        face_only, face_skin_mask=face, body_skin_mask=body,
        reference_img_bgr=original,
    )["texture_parity_drift"]
    drift_matched = evaluate_harmony(
        matched, face_skin_mask=face, body_skin_mask=body,
        reference_img_bgr=original,
    )["texture_parity_drift"]

    # Spike-calibrated behavior: face-only smoothing drifts past the 0.45
    # provisional gate; consistent smoothing lands well inside it (real-photo
    # range 0.10-0.41), at every tone in the sweep.
    assert drift_face_only > 0.45, (bgr, drift_face_only)
    assert drift_matched < 0.30, (bgr, drift_matched)
    assert drift_matched < drift_face_only / 2.0, (bgr, drift_matched, drift_face_only)


def test_mark_retention_detects_erased_identity_mark():
    img = np.full((240, 240, 3), (140, 170, 200), dtype=np.uint8)
    face = np.zeros(img.shape[:2], dtype=np.float32)
    body = np.zeros_like(face)
    face[20:140, 40:200] = 1.0
    body[160:232, 20:220] = 1.0

    reference = img.copy()
    cv2.circle(reference, (120, 80), 4, (20, 15, 12), -1)  # dark flat mole

    kept = evaluate_harmony(
        reference.copy(), face_skin_mask=face, body_skin_mask=body,
        reference_img_bgr=reference,
    )
    assert kept["marks_before"] >= 1
    assert kept["mark_retention"] == 1.0

    erased = evaluate_harmony(
        img, face_skin_mask=face, body_skin_mask=body,
        reference_img_bgr=reference,
    )
    assert erased["marks_before"] >= 1
    assert erased["mark_retention"] == 0.0
    assert erased["flagged"] is True


def test_harmony_policy_counts_only_preserve_class_marks():
    img = np.full((240, 240, 3), (140, 170, 200), dtype=np.uint8)
    face = np.zeros(img.shape[:2], dtype=np.float32)
    body = np.zeros_like(face)
    face[20:140, 40:200] = 1.0
    body[160:232, 20:220] = 1.0
    reference = img.copy()
    cv2.circle(reference, (120, 80), 4, (20, 15, 12), -1)

    result = evaluate_harmony(
        img, face_skin_mask=face, body_skin_mask=body,
        reference_img_bgr=reference,
        mark_policy={"mole": {"action": "remove"}, "unknown": {"action": "remove"}},
    )

    assert result["marks_before"] == 0
    assert np.isnan(result["mark_retention"])
    assert result["flagged"] is False


def test_banding_delta_is_differential_not_absolute():
    # Flat reference scores zero; the processed copy gains posterization
    # stripes on the body. Only the *introduced* steps may raise the delta.
    # Steps are 2 gray levels: detect_banding targets subtle quantization
    # contours (its smooth-variance gate excludes large, real edges).
    reference = np.full((160, 128, 3), 128, dtype=np.uint8)
    posterized = reference.copy()
    for y in range(88, 152, 16):
        posterized[y:y + 8] += 2

    face = np.zeros(reference.shape[:2], dtype=np.float32)
    body = np.zeros_like(face)
    face[16:76, 32:96] = 1.0
    body[88:152, 16:112] = 1.0

    identical = evaluate_harmony(
        reference, face_skin_mask=face, body_skin_mask=body,
        reference_img_bgr=reference,
    )
    assert identical["face_banding_delta"] == 0.0
    assert identical["body_banding_delta"] == 0.0

    stepped = evaluate_harmony(
        posterized, face_skin_mask=face, body_skin_mask=body,
        reference_img_bgr=reference,
    )
    assert stepped["body_banding_delta"] > 0.01
    assert stepped["face_banding_delta"] == 0.0


def test_run_all_includes_read_only_harmony_result():
    img, face, body = _portrait_fixture()
    result = run_all(
        img,
        skin_mask=face,
        reference_img_bgr=img,
        person_mask=np.maximum(face, body),
        face_skin_mask=face,
        body_skin_mask=body,
    )
    assert result["harmony"]["available"] is True
    assert result["harmony"]["flagged"] is False
