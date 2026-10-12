"""Focused contracts for the opt-in class-aware mark policy layer."""

import cv2
import numpy as np
import pytest

from retouch.marks import (
    MarkRecord,
    adapt_detector_mask,
    compile_mark_policy,
    detect_marks,
    resolve_mark_policy,
    resolve_mark_action,
)


def _record(mark_class="mole", confidence=0.9, *, on_body=False):
    return MarkRecord(
        mark_id=0,
        mark_class=mark_class,
        confidence=confidence,
        centroid=(20.0, 20.0),
        area_norm=0.01,
        bbox=(16, 16, 8, 8),
        features={"mel_rel": 0.0, "hb_rel": 0.0},
        on_body=on_body,
    )


def test_no_policy_compiles_empty_masks_and_cannot_change_legacy_output():
    masks = compile_mark_policy([_record("freckle")], (48, 48, 3), None)

    assert not masks.preserve.any()
    assert not masks.heal.any()
    assert not masks.attenuate.any()
    assert not masks.enhance.any()


def test_policy_is_class_aware_and_low_confidence_is_safe_unknown():
    records = [_record("freckle"), _record("acne_blemish", confidence=0.3)]
    records[1] = MarkRecord(**{**records[1].__dict__, "mark_id": 1, "centroid": (34.0, 34.0), "bbox": (30, 30, 8, 8)})
    masks = compile_mark_policy(records, (64, 64), {
        "freckle": {"action": "attenuate", "strength": 60},
        "acne_blemish": {"action": "remove"},
        "min_confidence": 0.6,
        "unknown": {"action": "preserve"},
    })

    assert masks.attenuate[20, 20] == pytest.approx(0.6)
    assert masks.heal[34, 34] == 0
    assert masks.preserve[34, 34] == 255
    assert resolve_mark_action(records[0], None) is None


def test_detector_mask_adapter_preserves_scale_free_area_and_body_locus():
    detector = np.zeros((80, 120), dtype=np.uint8)
    cv2.circle(detector, (60, 40), 3, 255, -1)
    records = adapt_detector_mask(
        detector, mark_class="stray_hair", face_width=100.0, on_body=True)

    assert len(records) == 1
    assert records[0].on_body is True
    assert records[0].area_norm == pytest.approx(np.count_nonzero(detector) / 10000.0)
    assert "eccentricity" in records[0].features


def test_detect_marks_emits_relative_feature_schema_for_existing_freckle_detector():
    image = np.full((180, 180, 3), (140, 170, 200), dtype=np.uint8)
    face = np.ones(image.shape[:2], dtype=np.float32)
    cv2.circle(image, (90, 90), 5, (20, 15, 12), -1)
    mel = np.full(image.shape[:2], 0.2, dtype=np.float32)
    hb = np.full(image.shape[:2], 0.3, dtype=np.float32)
    mel[85:96, 85:96] = 0.5
    hb[85:96, 85:96] = 0.1

    records = detect_marks(image, face_mask=face, face_width=120.0, mel_map=mel, hb_map=hb)

    assert records
    assert records[0].mark_class == "mole"
    for key in ("l_margin", "a_margin", "mel_rel", "hb_rel", "eccentricity", "cluster_density"):
        assert key in records[0].features


def test_named_policy_is_copied_and_legacy_is_a_true_noop():
    assert resolve_mark_policy("legacy") is None
    policy = resolve_mark_policy("protect_identity")
    assert policy is not None
    assert policy["mole"]["action"] == "preserve"
    policy["mole"]["action"] = "remove"
    assert resolve_mark_policy("protect_identity")["mole"]["action"] == "preserve"
