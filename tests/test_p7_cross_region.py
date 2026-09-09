"""Focused contract tests for the opt-in P7 cross-region skin stage."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from retouch.cross_region_skin import (
    infer_same_person_skin_support,
    propagate_face_edit_delta,
)
from retouch.engine import ProcessingContext, RetouchEngine, build_context
from retouch.params import resolve_recipe
from retouch.stage_wrappers import build_global_registry


def _masks(height: int = 80, width: int = 100):
    """Return deterministic face, body, and protected supports."""
    face = np.zeros((height, width), dtype=np.float32)
    face[10:30, 10:30] = 1.0
    target = np.zeros((height, width), dtype=np.float32)
    target[40:70, 20:70] = 1.0
    protect = np.zeros((height, width), dtype=np.float32)
    protect[40:70, 20:35] = 1.0
    return face, target, protect


def _edited_pair(dtype=np.float32):
    """Build an image whose face has a deliberately bounded tone edit."""
    face, target, protect = _masks()
    source = np.full((80, 100, 3), 0.5, dtype=np.float32)
    edited = source.copy()
    edited[face > 0.5] = (0.6, 0.55, 0.5)
    if dtype is np.uint8:
        return (
            np.rint(source * 255.0).astype(np.uint8),
            np.rint(edited * 255.0).astype(np.uint8),
            face,
            target,
            protect,
        )
    return source, edited, face, target, protect


def test_propagation_applies_bounded_delta_and_preserves_outside_support():
    source, edited, face, target, protect = _edited_pair()
    result = propagate_face_edit_delta(
        source,
        edited,
        face,
        target,
        protect_mask=protect,
        min_face_pixels=1,
        min_target_pixels=1,
    )

    assert result.applied is True
    assert result.abstained is False
    assert result.reason == "applied"
    assert result.changed_pixels > 0
    assert result.image.dtype == edited.dtype
    np.testing.assert_array_equal(result.image[target == 0], edited[target == 0])
    np.testing.assert_array_equal(result.image[protect > 0.5], edited[protect > 0.5])
    applied_delta = np.abs(np.asarray(result.applied_delta_lab255))
    assert applied_delta[0] <= 6.0 + 1e-6
    assert np.max(applied_delta[1:]) <= 4.0 + 1e-6
    # The body receives the tone delta without flattening its existing pixels.
    assert np.any(np.abs(result.image[40:70, 35:70] - edited[40:70, 35:70]) > 1e-5)


def test_uint8_propagation_and_abstention_keep_public_dtype_and_identity():
    source, edited, face, target, _ = _edited_pair(dtype=np.uint8)
    applied = RetouchEngine.apply_cross_region_skin(
        source, edited, face, target, strength=0.5
    )
    assert applied.image.dtype == np.uint8
    assert applied.image.shape == edited.shape

    missing_target = RetouchEngine.apply_cross_region_skin(
        source, edited, face, None
    )
    assert missing_target.abstained is True
    assert missing_target.reason == "target_support_required"
    assert missing_target.image.dtype == np.uint8
    np.testing.assert_array_equal(missing_target.image, edited)


def test_no_face_delta_abstains_without_round_trip_drift():
    source, edited, face, target, _ = _edited_pair(dtype=np.uint8)
    result = propagate_face_edit_delta(source, source.copy(), face, target)
    assert result.abstained is True
    assert result.reason == "no_approved_face_delta"
    np.testing.assert_array_equal(result.image, source)


def test_inferred_support_stays_on_face_owner_component(monkeypatch):
    height, width = 80, 100
    source = np.full((height, width, 3), 0.5, dtype=np.float32)
    person = np.zeros((height, width), dtype=np.float32)
    person[8:72, 8:42] = 1.0
    person[8:72, 58:92] = 1.0  # unrelated disconnected subject
    face = np.zeros((height, width), dtype=np.float32)
    face[18:30, 18:30] = 1.0

    # Isolate ownership/connected-components behavior from the color model.
    monkeypatch.setattr(
        "retouch.cross_region_skin.skin_mask_lch",
        lambda lch, **kwargs: np.ones(lch.shape[:2], dtype=np.float32),
    )
    support = infer_same_person_skin_support(source, person, face)
    assert support.dtype == np.float32
    assert np.count_nonzero(support[35:65, 18:30]) > 0
    assert np.count_nonzero(support[:, 58:92]) == 0
    assert np.count_nonzero(support[18:30, 18:30]) == 0


def test_engine_stage_requires_one_face_and_records_abstention():
    source, edited, face, target, _ = _edited_pair()
    ctx = ProcessingContext(
        cross_region_skin=100.0,
        cross_region_skin_mask=target,
        _p7_diagnostics={},
    )
    ctx._p7_source = source
    engine = RetouchEngine.__new__(RetouchEngine)
    output = engine._stage_cross_region_skin(
        edited,
        ctx,
        np.ones(face.shape, dtype=np.float32),
        face,
        np.zeros_like(face),
        np.zeros_like(face),
        [object(), object()],
    )
    np.testing.assert_array_equal(output, edited)
    assert ctx._p7_diagnostics["reason"] == "single_face_required"
    assert ctx._p7_diagnostics["abstained"] is True


def test_engine_stage_applies_and_exposes_diagnostics():
    source, edited, face, target, _ = _edited_pair()
    ctx = ProcessingContext(
        cross_region_skin=50.0,
        cross_region_skin_mask=target,
        _p7_diagnostics={},
    )
    ctx._p7_source = source
    engine = RetouchEngine.__new__(RetouchEngine)
    output = engine._stage_cross_region_skin(
        edited,
        ctx,
        np.ones(face.shape, dtype=np.float32),
        face,
        np.zeros_like(face),
        np.zeros_like(face),
        [object()],
    )
    assert np.any(np.abs(output[40:70, 35:70] - edited[40:70, 35:70]) > 1e-5)
    assert ctx._p7_diagnostics["applied"] is True
    assert ctx._p7_diagnostics["reason"] == "applied"


def test_engine_stage_abstains_when_legacy_absolute_body_match_is_enabled():
    # Both stages independently apply a face-to-body-skin LAB correction over
    # the same LCH skin-candidate region, and CrossRegionSkinStage runs right
    # after BodySkinStage in the global registry, so enabling both would stack
    # two corrections instead of one (same failure shape as the undereye
    # dark_circles/undereye_darken_removal double-apply, e73d3ba). The stage
    # abstains rather than raising, matching its other abstention contracts.
    source, edited, face, target, _ = _edited_pair()
    ctx = ProcessingContext(
        cross_region_skin=100.0,
        cross_region_skin_mask=target,
        body_match_face=40.0,
        _p7_diagnostics={},
    )
    ctx._p7_source = source
    engine = RetouchEngine.__new__(RetouchEngine)
    output = engine._stage_cross_region_skin(
        edited,
        ctx,
        np.ones(face.shape, dtype=np.float32),
        face,
        np.zeros_like(face),
        np.zeros_like(face),
        [object()],
    )
    np.testing.assert_array_equal(output, edited)
    assert ctx._p7_diagnostics["abstained"] is True
    assert ctx._p7_diagnostics["reason"] == "body_match_face_conflict"


def test_p7_context_and_registry_are_opt_in():
    default = build_context("natural", resolve_recipe("natural"), {})
    assert default.cross_region_skin == 0.0
    assert default.cross_region_skin_mask is None
    assert default.cross_region_protect_mask is None

    target = np.ones((4, 4), dtype=np.float32)
    protected = np.zeros((4, 4), dtype=np.float32)
    configured = build_context(
        "natural",
        resolve_recipe("natural"),
        {
            "cross_region_skin": 45,
            "cross_region_skin_mask": target,
            "cross_region_protect_mask": protected,
        },
    )
    assert configured.cross_region_skin == 45.0
    assert configured.cross_region_skin_mask is target
    assert configured.cross_region_protect_mask is protected
    assert "cross_region_skin" in inspect.signature(RetouchEngine.process).parameters

    names = build_global_registry(object()).names()
    assert names.index("cross_region_skin") == names.index("body_skin") + 1


def test_negative_stage_strength_is_rejected():
    source, edited, face, target, _ = _edited_pair()
    ctx = ProcessingContext(cross_region_skin=-1.0, _p7_diagnostics={})
    ctx._p7_source = source
    engine = RetouchEngine.__new__(RetouchEngine)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        engine._stage_cross_region_skin(
            edited,
            ctx,
            np.ones(face.shape, dtype=np.float32),
            face,
            np.zeros_like(face),
            np.zeros_like(face),
            [object()],
        )


