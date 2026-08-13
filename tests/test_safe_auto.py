"""P2 Safe Auto contract tests."""

import numpy as np
import pytest

from retouch.safe_auto import apply_decision, decide, decide_mask_stage


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
