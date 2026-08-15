"""P2 Safe Auto contract tests."""

import numpy as np
import pytest

from retouch.engine import ProcessingContext, ProcessingResult, RetouchEngine
from retouch.safe_auto import (
    apply_decision,
    confidence_evidence,
    decide,
    decide_mask_stage,
    record_decision,
)


def test_safe_auto_actions_follow_confidence_bands():
    assert decide("blemish", confidence=0.95).action == "apply"
    assert decide("blemish", confidence=0.75).action == "dampen"
    assert decide("blemish", confidence=0.50).action == "review"
    assert decide("blemish", confidence=0.10).action == "skip"


def test_occlusion_forces_conservative_action_and_records_evidence():
    decision = decide_mask_stage(
        "eye",
        mask_coverage=0.98,
        landmark_stability=0.98,
        model_confidence=0.98,
        occluded=True,
    )

    assert decision.action == "skip"
    assert decision.reason == "occluded region"
    assert decision.evidence["occluded"] is True


def test_unavailable_detector_confidence_never_enters_auto_apply_band():
    evidence = confidence_evidence(1.0, "mediapipe_presence_unavailable")
    decision = decide_mask_stage(
        "face",
        mask_coverage=1.0,
        landmark_stability=evidence["safe_auto_confidence"],
        model_confidence=evidence["safe_auto_confidence"],
    )

    assert evidence["confidence_measured"] is False
    assert evidence["reported_confidence"] == 1.0
    assert decision.action == "review"


def test_safe_auto_skip_is_byte_identical():
    original = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    candidate = np.full_like(original, 255)
    decision = decide("reshape", confidence=0.1)

    result = apply_decision(original, candidate, decision)

    assert np.array_equal(result, original)
    assert result is not original


def test_safe_auto_dampen_is_bounded_and_preserves_dtype():
    original = np.zeros((2, 2, 3), dtype=np.uint8)
    candidate = np.full_like(original, 100)
    decision = decide("smooth", confidence=0.75)

    result = apply_decision(original, candidate, decision)

    assert result.dtype == np.uint8
    assert np.all(result == 50)


def test_safe_auto_rejects_invalid_thresholds_and_shapes():
    with pytest.raises(ValueError):
        decide("stage", confidence=0.8, review_at=0.8, dampen_at=0.7)
    with pytest.raises(ValueError):
        apply_decision(np.zeros((2, 2)), np.zeros((3, 3)), decide("stage", confidence=1.0))


def test_safe_auto_decisions_are_recorded_on_processing_context_and_result():
    ctx = ProcessingContext()
    decision = decide("body_reshape", confidence=0.1)
    record_decision(ctx, decision)

    result = ProcessingResult(
        np.zeros((2, 2, 3), dtype=np.uint8),
        params=ctx,
        safe_auto_decisions=ctx._safe_auto_decisions,
    )

    assert result.safe_auto_decisions[0]["stage"] == "body_reshape"
    assert result.safe_auto_decisions[0]["action"] == "skip"


def test_auto_body_reshape_skips_uncertain_pose(monkeypatch):
    class _Pose:
        detected = True
        body_visible = False
        landmarks = [(0.5, 0.5)] * 33
        visibility = [0.2] * 33
        feature_flags = {}

    class _Detector:
        def detect(self, _image):
            return _Pose()

    class _Reshaper:
        def __init__(self):
            self.detector = _Detector()

        def reshape(self, image, **_kwargs):
            return np.full_like(image, 255)

    monkeypatch.setattr("retouch.body_reshape.BodyReshaper", _Reshaper)
    monkeypatch.setattr(
        "retouch.body_reshape.suggest_body_reshape",
        lambda _pose: {name: 60.0 for name in (
            "arm_length", "leg_length", "torso_width", "shoulder_width", "hip_width"
        )},
    )

    engine = RetouchEngine.__new__(RetouchEngine)
    ctx = ProcessingContext(auto_body_reshape=100.0)
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    result = engine._stage_body_reshape(image, ctx)

    np.testing.assert_array_equal(result, image)
    assert ctx._safe_auto_decisions[-1]["stage"] == "body_reshape"
    assert ctx._safe_auto_decisions[-1]["action"] == "skip"


def test_auto_body_reshape_skip_preserves_manual_result_when_pose_unavailable(monkeypatch):
    class _Pose:
        detected = False
        body_visible = False
        landmarks = None
        visibility = []
        feature_flags = {}

    class _Detector:
        def detect(self, _image):
            return _Pose()

    class _Reshaper:
        def __init__(self):
            self.detector = _Detector()

        def reshape(self, image, **kwargs):
            # Manual and automatic renders are deliberately distinct so a
            # regression that returns the original frame cannot pass.
            amount = int(abs(kwargs["arm_length"]) * 2)
            return np.clip(image.astype(np.int16) + amount, 0, 255).astype(np.uint8)

    monkeypatch.setattr("retouch.body_reshape.BodyReshaper", _Reshaper)
    engine = RetouchEngine.__new__(RetouchEngine)
    ctx = ProcessingContext(auto_body_reshape=100.0, body_reshape_arm_length=60.0)
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    result = engine._stage_body_reshape(image, ctx)

    assert np.all(result == 20)
    assert ctx._safe_auto_decisions[-1]["action"] == "skip"
