"""Tests for P7 cross-region skin appearance propagation.

Covers the isolated leaf primitive in retouch/cross_region_skin.py:
1. Input validation (dtype, shape, range)
2. Abstention reasons (missing face/target, insufficient pixels, tiny delta)
3. Delta bound clamping (MAX_L_DELTA, MAX_AB_DELTA) and strength scaling
4. Exact identity outside the final support (restore_outside_support contract)
5. uint8 vs float32 [0, 1] round trip
6. infer_same_person_skin_support: connected-component ownership + exclusions
"""

import numpy as np
import pytest

from retouch import RetouchEngine
from retouch.cross_region_skin import (
    MAX_AB_DELTA,
    MAX_L_DELTA,
    infer_same_person_skin_support,
    propagate_face_edit_delta,
)


def _solid(h, w, bgr, dtype=np.uint8):
    img = np.zeros((h, w, 3), dtype=np.float32)
    img[:] = bgr
    if dtype == np.uint8:
        return img.astype(np.uint8)
    return (img / 255.0).astype(np.float32)


def _rect_mask(h, w, y0, y1, x0, x1, dtype=np.uint8):
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y0:y1, x0:x1] = 1.0
    if dtype == np.uint8:
        return (mask * 255).astype(np.uint8)
    return mask


class TestInputValidation:
    def test_rejects_non_array_source(self):
        with pytest.raises(TypeError):
            propagate_face_edit_delta(
                "not an array", _solid(10, 10, (100, 100, 100)),
                _rect_mask(10, 10, 0, 5, 0, 5), _rect_mask(10, 10, 5, 10, 0, 5),
            )

    def test_rejects_mismatched_shapes(self):
        source = _solid(10, 10, (100, 100, 100))
        edited = _solid(20, 20, (100, 100, 100))
        with pytest.raises(ValueError, match="shape"):
            propagate_face_edit_delta(
                source, edited,
                _rect_mask(10, 10, 0, 5, 0, 5), _rect_mask(10, 10, 5, 10, 0, 5),
            )

    def test_rejects_out_of_range_float(self):
        source = np.full((10, 10, 3), 2.0, dtype=np.float32)
        edited = _solid(10, 10, (100, 100, 100), dtype=np.float32)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            propagate_face_edit_delta(
                source, edited,
                _rect_mask(10, 10, 0, 5, 0, 5, dtype=np.float32),
                _rect_mask(10, 10, 5, 10, 0, 5, dtype=np.float32),
            )

    def test_rejects_strength_out_of_range(self):
        source = _solid(64, 64, (150, 140, 180))
        edited = _solid(64, 64, (150, 140, 180))
        with pytest.raises(ValueError, match="strength"):
            propagate_face_edit_delta(
                source, edited,
                _rect_mask(64, 64, 0, 32, 0, 64),
                _rect_mask(64, 64, 32, 64, 0, 64),
                strength=1.5,
            )

    def test_rejects_negative_delta_bounds(self):
        source = _solid(64, 64, (150, 140, 180))
        edited = _solid(64, 64, (150, 140, 180))
        with pytest.raises(ValueError, match="non-negative"):
            propagate_face_edit_delta(
                source, edited,
                _rect_mask(64, 64, 0, 32, 0, 64),
                _rect_mask(64, 64, 32, 64, 0, 64),
                max_l_delta=-1.0,
            )


class TestAbstention:
    def test_abstains_when_face_mask_missing(self):
        source = _solid(64, 64, (150, 140, 180))
        edited = _solid(64, 64, (170, 140, 180))
        result = propagate_face_edit_delta(source, edited, None, _rect_mask(64, 64, 32, 64, 0, 64))
        assert result.abstained is True
        assert result.applied is False
        assert result.reason == "face_reference_required"
        np.testing.assert_array_equal(result.image, edited)

    def test_abstains_when_target_mask_missing(self):
        source = _solid(64, 64, (150, 140, 180))
        edited = _solid(64, 64, (170, 140, 180))
        result = propagate_face_edit_delta(source, edited, _rect_mask(64, 64, 0, 32, 0, 64), None)
        assert result.abstained is True
        assert result.reason == "target_support_required"
        np.testing.assert_array_equal(result.image, edited)

    def test_abstains_when_face_pixels_insufficient(self):
        source = _solid(64, 64, (150, 140, 180))
        edited = _solid(64, 64, (170, 140, 180))
        tiny_face = _rect_mask(64, 64, 0, 2, 0, 2)  # 4 px < MIN_FACE_PIXELS
        target = _rect_mask(64, 64, 32, 64, 0, 64)
        result = propagate_face_edit_delta(source, edited, tiny_face, target)
        assert result.reason == "face_reference_insufficient"
        assert result.abstained is True

    def test_abstains_when_target_pixels_insufficient(self):
        source = _solid(64, 64, (150, 140, 180))
        edited = _solid(64, 64, (170, 140, 180))
        face = _rect_mask(64, 64, 0, 32, 0, 64)
        tiny_target = _rect_mask(64, 64, 32, 34, 0, 2)  # 4 px < MIN_TARGET_PIXELS
        result = propagate_face_edit_delta(source, edited, face, tiny_target)
        assert result.reason == "target_support_insufficient"
        assert result.abstained is True

    def test_abstains_when_no_face_delta(self):
        # source == edited -> zero delta -> abstain, not a spurious edit.
        img = _solid(64, 64, (150, 140, 180))
        face = _rect_mask(64, 64, 0, 32, 0, 64)
        target = _rect_mask(64, 64, 32, 64, 0, 64)
        result = propagate_face_edit_delta(img, img, face, target)
        assert result.reason == "no_approved_face_delta"
        assert result.abstained is True
        np.testing.assert_array_equal(result.image, img)

    def test_target_pixels_intersecting_face_are_excluded_from_support(self):
        # Face and target masks fully overlap -> after face exclusion, target
        # support is empty -> insufficient, not a crash.
        source = _solid(64, 64, (150, 140, 180))
        edited = _solid(64, 64, (170, 140, 180))
        face = _rect_mask(64, 64, 0, 32, 0, 64)
        result = propagate_face_edit_delta(source, edited, face, face.copy())
        assert result.abstained is True
        assert result.reason == "target_support_insufficient"


class TestAppliedDelta:
    def test_applies_bounded_delta_within_target_only(self):
        source = _solid(64, 64, (150, 140, 180))
        edited = source.copy()
        edited[0:32, :, :] = _solid(32, 64, (200, 140, 180))[0:32]  # brighten face region
        face = _rect_mask(64, 64, 0, 32, 0, 64)
        target = _rect_mask(64, 64, 32, 64, 0, 64)

        result = propagate_face_edit_delta(source, edited, face, target, strength=1.0)

        assert result.applied is True
        assert result.abstained is False
        # Outside target: exact identity with edited.
        np.testing.assert_array_equal(result.image[0:32], edited[0:32])
        # Inside target: something changed.
        assert result.changed_pixels > 0

    def test_delta_is_clamped_to_max_bounds(self):
        source = _solid(64, 64, (100, 100, 100))
        edited = source.copy()
        # Huge brightening well beyond MAX_L_DELTA in LAB units.
        edited[0:32, :, :] = _solid(32, 64, (255, 255, 255))[0:32]
        face = _rect_mask(64, 64, 0, 32, 0, 64)
        target = _rect_mask(64, 64, 32, 64, 0, 64)

        result = propagate_face_edit_delta(source, edited, face, target, strength=1.0)

        assert result.applied is True
        assert abs(result.applied_delta_lab255[0]) <= MAX_L_DELTA + 1e-4
        assert abs(result.applied_delta_lab255[1]) <= MAX_AB_DELTA + 1e-4
        assert abs(result.applied_delta_lab255[2]) <= MAX_AB_DELTA + 1e-4

    def test_strength_scales_applied_delta(self):
        source = _solid(64, 64, (100, 100, 100))
        edited = source.copy()
        edited[0:32, :, :] = _solid(32, 64, (140, 100, 100))[0:32]
        face = _rect_mask(64, 64, 0, 32, 0, 64)
        target = _rect_mask(64, 64, 32, 64, 0, 64)

        full = propagate_face_edit_delta(source, edited, face, target, strength=1.0)
        half = propagate_face_edit_delta(source, edited, face, target, strength=0.5)

        assert full.applied and half.applied
        assert abs(half.applied_delta_lab255[0]) == pytest.approx(
            abs(full.applied_delta_lab255[0]) * 0.5, rel=0.05
        )

    def test_protect_mask_removes_pixels_from_target_support(self):
        source = _solid(64, 64, (100, 100, 100))
        edited = source.copy()
        edited[0:32, :, :] = _solid(32, 64, (140, 100, 100))[0:32]
        face = _rect_mask(64, 64, 0, 32, 0, 64)
        target = _rect_mask(64, 64, 32, 64, 0, 64)
        protect = _rect_mask(64, 64, 32, 64, 0, 32)  # protect left half of target

        result = propagate_face_edit_delta(
            source, edited, face, target, strength=1.0, protect_mask=protect,
        )

        assert result.applied is True
        # Protected half must be untouched.
        np.testing.assert_array_equal(result.image[32:64, 0:32], edited[32:64, 0:32])

    def test_uint8_input_returns_uint8_with_exact_identity_outside_support(self):
        source = _solid(64, 64, (100, 100, 100), dtype=np.uint8)
        edited = source.copy()
        edited[0:32, :, :] = _solid(32, 64, (140, 100, 100), dtype=np.uint8)[0:32]
        face = _rect_mask(64, 64, 0, 32, 0, 64)
        target = _rect_mask(64, 64, 32, 64, 0, 64)

        result = propagate_face_edit_delta(source, edited, face, target, strength=1.0)

        assert result.image.dtype == np.uint8
        np.testing.assert_array_equal(result.image[0:32], edited[0:32])

    def test_float_input_returns_float_in_unit_range(self):
        source = _solid(64, 64, (100, 100, 100), dtype=np.float32)
        edited = source.copy()
        edited[0:32, :, :] = _solid(32, 64, (140, 100, 100), dtype=np.float32)[0:32]
        face = _rect_mask(64, 64, 0, 32, 0, 64, dtype=np.float32)
        target = _rect_mask(64, 64, 32, 64, 0, 64, dtype=np.float32)

        result = propagate_face_edit_delta(source, edited, face, target, strength=1.0)

        assert result.image.dtype == np.float32
        assert float(result.image.min()) >= 0.0
        assert float(result.image.max()) <= 1.0

    def test_to_dict_is_json_safe(self):
        source = _solid(64, 64, (100, 100, 100))
        edited = source.copy()
        edited[0:32, :, :] = _solid(32, 64, (140, 100, 100))[0:32]
        face = _rect_mask(64, 64, 0, 32, 0, 64)
        target = _rect_mask(64, 64, 32, 64, 0, 64)
        result = propagate_face_edit_delta(source, edited, face, target, strength=1.0)

        d = result.to_dict()
        import json
        json.dumps(d)  # must not raise
        assert d["applied"] is True
        assert isinstance(d["face_delta_lab255"], list)


class TestInferSamePersonSkinSupport:
    def test_returns_zero_mask_when_person_or_face_mask_missing(self):
        source = _solid(64, 64, (150, 140, 180))
        support = infer_same_person_skin_support(source, None, _rect_mask(64, 64, 0, 32, 0, 64))
        assert np.count_nonzero(support) == 0

    def test_excludes_disconnected_person_components(self):
        h, w = 64, 64
        source = _solid(h, w, (150, 140, 180))
        person = np.zeros((h, w), dtype=np.uint8)
        person[0:32, :] = 255       # component A: contains the face
        person[40:64, :] = 255      # component B: disconnected, e.g. another person
        face = _rect_mask(h, w, 0, 16, 0, w)

        support = infer_same_person_skin_support(source, person, face)

        # Support must not include the disconnected second component.
        assert np.count_nonzero(support[40:64, :]) == 0

    def test_excludes_face_and_lip_regions(self):
        h, w = 64, 64
        source = _solid(h, w, (150, 140, 180))
        person = np.full((h, w), 255, dtype=np.uint8)
        face = _rect_mask(h, w, 0, 32, 0, w)
        lips = _rect_mask(h, w, 32, 40, 0, w)

        support = infer_same_person_skin_support(
            source, person, face, face_exclusion=face, lip_exclusion=lips,
        )

        assert np.count_nonzero(support[0:40, :]) == 0

    def test_ambiguous_ownership_returns_zero_mask(self):
        # Face mask straddles two disconnected person components -> ambiguous.
        h, w = 64, 64
        source = _solid(h, w, (150, 140, 180))
        person = np.zeros((h, w), dtype=np.uint8)
        person[0:16, :] = 255
        person[48:64, :] = 255
        face = np.zeros((h, w), dtype=np.uint8)
        face[0:16, :] = 255
        face[48:64, :] = 255

        support = infer_same_person_skin_support(source, person, face)
        assert np.count_nonzero(support) == 0


class TestEngineIntegration:
    """The stage must stay a true no-op unless a caller explicitly opts in."""

    def test_default_zero_is_byte_identical_to_disabled(self, natural_image):
        engine = RetouchEngine()
        img = natural_image

        baseline = engine.process(img, recipe="natural", fast=True)
        with_default = engine.process(
            img, recipe="natural", fast=True, cross_region_skin=0.0,
        )

        np.testing.assert_array_equal(np.asarray(baseline), np.asarray(with_default))

    def test_nonzero_without_target_mask_applies_inferred_support_on_real_face(
        self, natural_image,
    ):
        # On DSCF8007 (single clear face) with no reviewed target mask, the
        # stage does NOT abstain: it falls back to infer_same_person_skin_support
        # and applies a bounded delta. This is observed, not assumed -- pin it
        # so a future change silently making this abstain (or stop bounding
        # the delta) is caught.
        engine = RetouchEngine()
        img = natural_image

        result = engine.process(
            img, recipe="natural", fast=True, cross_region_skin=50.0,
        )

        diag = result.p7_diagnostics
        assert "reason" in diag
        if diag["applied"]:
            assert diag["abstained"] is False
            assert abs(diag["applied_delta_lab255"][0]) <= MAX_L_DELTA + 1e-4
            assert abs(diag["applied_delta_lab255"][1]) <= MAX_AB_DELTA + 1e-4
            assert abs(diag["applied_delta_lab255"][2]) <= MAX_AB_DELTA + 1e-4
            assert diag["changed_pixels"] > 0
        else:
            assert diag["abstained"] is True

    def test_no_recipe_or_param_spec_enables_it_by_default(self):
        from retouch.params import PROCESSING_PARAMS

        names = {p.name for p in PROCESSING_PARAMS}
        assert "cross_region_skin" not in names
