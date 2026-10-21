"""Tests for retouch/skin_protect.py — skin-tone protection API."""

import numpy as np

from retouch.color_space import bgr_to_lch, skin_mask_lch
from retouch.skin_protect import (
    _blend_skin,
    protect_skin,
    protect_skin_chromatic,
    skin_aware_apply,
)


SKIN_BGR = np.array([[[80, 120, 180]]], dtype=np.uint8)
SKIN_BGR_STRONG = np.array([[[121, 123, 201]]], dtype=np.uint8)
NON_SKIN_BGR = np.array([[[255, 0, 0]]], dtype=np.uint8)


class TestSkinMaskIdentification:
    def test_skin_tone_bgr_is_detected(self):
        lch = bgr_to_lch(SKIN_BGR)
        mask = skin_mask_lch(lch)
        assert mask[0, 0] > 0.05

    def test_pure_blue_is_not_skin(self):
        lch = bgr_to_lch(NON_SKIN_BGR)
        mask = skin_mask_lch(lch)
        assert mask[0, 0] < 0.1

    def test_gray_is_not_skin(self):
        gray = np.array([[[128, 128, 128]]], dtype=np.uint8)
        lch = bgr_to_lch(gray)
        mask = skin_mask_lch(lch)
        assert mask[0, 0] == 0.0

    def test_mask_is_float32_in_unit_range(self):
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[0, 0] = (80, 120, 180)
        bgr[0, 1] = (255, 0, 0)
        mask = skin_mask_lch(bgr_to_lch(bgr))
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0


class TestRelativeChromaGate:
    """House rule: no absolute chroma floors — the gate scales with L*."""

    def test_deep_skin_low_chroma_now_detected(self):
        # Deep skin: L* 30, chroma 6 (below the old absolute 8.0 floor),
        # hue at the skin centre. Previously excluded, now included.
        lch = np.array([[[30.0, 6.0, 25.0]]], dtype=np.float32)
        old_floor = 8.0
        assert not (lch[0, 0, 1] > old_floor)  # sanity: old gate excluded it
        mask = skin_mask_lch(lch, chroma_ref_lightness=60.0)
        assert mask[0, 0] > 0.9

    def test_deep_skin_gray_still_excluded(self):
        # Near-neutral deep-toned pixel (grey wall/hair in shadow): chroma ~0
        # stays excluded at every lightness.
        lch = np.array([[[30.0, 1.0, 25.0]]], dtype=np.float32)
        mask = skin_mask_lch(lch, chroma_ref_lightness=60.0)
        assert mask[0, 0] == 0.0

    def test_mid_tone_unchanged(self):
        # L* at/above the 60 reference: full 8.0 floor, same verdicts.
        kept = np.array([[[60.0, 8.5, 25.0]]], dtype=np.float32)
        dropped = np.array([[[60.0, 7.5, 25.0]]], dtype=np.float32)
        assert skin_mask_lch(kept, chroma_ref_lightness=60.0)[0, 0] > 0.9
        assert skin_mask_lch(dropped, chroma_ref_lightness=60.0)[0, 0] == 0.0

    def test_protect_skin_covers_deep_skin(self):
        # End-to-end: a deep-skin face patch (L* 30, chroma 6 — under the old
        # absolute 8.0 floor) is now protected by protect_skin at full strength.
        from retouch.color_space import lch_to_bgr

        deep = lch_to_bgr(np.array([[[30.0, 6.0, 25.0]]], dtype=np.float32))
        out = protect_skin(
            deep, lambda x: np.full_like(x, 255), strength=1.0
        )
        assert np.allclose(out[0, 0], deep[0, 0], atol=6)


class TestBlendSkin:
    def test_full_mask_preserves_image(self):
        img = np.full((4, 4, 3), 100, dtype=np.uint8)
        op_result = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        out = _blend_skin(img, op_result, mask, strength=1.0)
        assert np.array_equal(out, img)

    def test_zero_mask_returns_op(self):
        img = np.full((4, 4, 3), 100, dtype=np.uint8)
        op_result = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=np.float32)
        out = _blend_skin(img, op_result, mask, strength=1.0)
        assert np.array_equal(out, op_result)

    def test_zero_strength_returns_op(self):
        img = np.full((4, 4, 3), 100, dtype=np.uint8)
        op_result = np.full((4, 4, 3), 200, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        out = _blend_skin(img, op_result, mask, strength=0.0)
        assert np.array_equal(out, op_result)

    def test_output_dtype_and_shape(self):
        img = np.zeros((3, 5, 3), dtype=np.uint8)
        op_result = np.zeros((3, 5, 3), dtype=np.uint8)
        mask = np.zeros((3, 5), dtype=np.float32)
        out = _blend_skin(img, op_result, mask, strength=0.5)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


class TestProtectSkin:
    def test_zero_strength_returns_op_result(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0] = (80, 120, 180)
        img[0, 1] = (255, 0, 0)

        def op(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 42)

        out = protect_skin(img, op, strength=0.0)
        assert np.array_equal(out, op(img))

    def test_full_strength_preserves_skin_pixels(self):
        img = np.zeros((1, 2, 3), dtype=np.uint8)
        img[0, 0] = (121, 123, 201)
        img[0, 1] = (255, 0, 0)
        skin_lch = bgr_to_lch(img)
        skin_mask = skin_mask_lch(skin_lch)
        assert skin_mask[0, 0] > 0.99
        assert skin_mask[0, 1] < 0.1

        def op(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 0)

        out = protect_skin(img, op, strength=1.0)
        assert np.allclose(out[0, 0], img[0, 0], atol=2)
        assert np.array_equal(out[0, 1], op(img)[0, 1])

    def test_zero_op_returns_input_everywhere(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0, 0] = (80, 120, 180)
        img[0, 1] = (255, 0, 0)
        out = protect_skin(img, lambda x: x, strength=1.0)
        assert np.array_equal(out, img)

    def test_output_shape_and_dtype(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        out = protect_skin(img, lambda x: x, strength=0.7)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_non_skin_region_receives_op(self):
        img = np.zeros((1, 2, 3), dtype=np.uint8)
        img[0, 0] = (121, 123, 201)
        img[0, 1] = (255, 0, 0)
        skin_lch = bgr_to_lch(img)
        skin_mask = skin_mask_lch(skin_lch)
        assert skin_mask[0, 0] > 0.99
        assert skin_mask[0, 1] < 0.1

        def op(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 0)

        out = protect_skin(img, op, strength=1.0)
        assert np.allclose(out[0, 0], img[0, 0], atol=2)
        assert np.array_equal(out[0, 1], op(img)[0, 1])


class TestProtectSkinChromatic:
    def test_full_strength_skin_pixel_unchanged(self):
        bgr = SKIN_BGR_STRONG.copy()
        lch_orig = bgr_to_lch(bgr)
        out = protect_skin_chromatic(
            bgr,
            hue_shift_deg=45.0,
            sat_factor=1.5,
            strength=1.0,
        )
        lch_out = bgr_to_lch(out)
        hue_change = float((lch_out[0, 0, 2] - lch_orig[0, 0, 2]) % 360.0)
        hue_change = min(hue_change, 360.0 - hue_change)
        assert hue_change < 5.0

    def test_zero_strength_applies_full_shift(self):
        bgr = SKIN_BGR_STRONG.copy()
        lch_orig = bgr_to_lch(bgr)
        out = protect_skin_chromatic(
            bgr,
            hue_shift_deg=30.0,
            sat_factor=1.0,
            strength=0.0,
        )
        lch_out = bgr_to_lch(out)
        hue_change = float((lch_out[0, 0, 2] - lch_orig[0, 0, 2]) % 360.0)
        assert 25.0 < hue_change < 35.0

    def test_intermediate_strength_reduces_shift(self):
        bgr = SKIN_BGR_STRONG.copy()
        lch_orig = bgr_to_lch(bgr)
        out_full = protect_skin_chromatic(
            bgr,
            hue_shift_deg=60.0,
            sat_factor=1.0,
            strength=0.0,
        )
        out_partial = protect_skin_chromatic(
            bgr,
            hue_shift_deg=60.0,
            sat_factor=1.0,
            strength=0.5,
        )
        lch_full = bgr_to_lch(out_full)
        lch_partial = bgr_to_lch(out_partial)

        def hue_diff(a: float, b: float) -> float:
            d = abs(a - b) % 360.0
            return min(d, 360.0 - d)

        full_change = hue_diff(float(lch_full[0, 0, 2]), float(lch_orig[0, 0, 2]))
        partial_change = hue_diff(float(lch_partial[0, 0, 2]), float(lch_orig[0, 0, 2]))
        assert partial_change < full_change

    def test_non_skin_pixel_gets_full_hue_shift(self):
        bgr = NON_SKIN_BGR.copy()
        lch_orig = bgr_to_lch(bgr)
        out = protect_skin_chromatic(
            bgr,
            hue_shift_deg=30.0,
            sat_factor=1.0,
            strength=1.0,
        )
        lch_out = bgr_to_lch(out)
        hue_change = float((lch_out[0, 0, 2] - lch_orig[0, 0, 2]) % 360.0)
        assert 25.0 < hue_change < 35.0

    def test_sat_factor_applied_to_non_skin(self):
        bgr = np.array([[[200, 100, 50]]], dtype=np.uint8)
        lch_orig = bgr_to_lch(bgr)
        out = protect_skin_chromatic(
            bgr,
            hue_shift_deg=0.0,
            sat_factor=1.5,
            strength=1.0,
        )
        lch_out = bgr_to_lch(out)
        assert lch_out[0, 0, 1] > lch_orig[0, 0, 1]

    def test_output_shape_and_dtype(self):
        bgr = np.zeros((6, 6, 3), dtype=np.uint8)
        bgr[0, 0] = (80, 120, 180)
        bgr[0, 1] = (255, 0, 0)
        out = protect_skin_chromatic(bgr, 10.0, 1.2, 0.7)
        assert out.shape == bgr.shape
        assert out.dtype == np.uint8


class TestSkinAwareApply:
    def test_skin_runs_op_skin_non_skin_runs_op_other(self):
        img = np.zeros((2, 2, 3), dtype=np.uint8)
        mask = np.array(
            [[1.0, 1.0], [0.0, 0.0]],
            dtype=np.float32,
        )

        def op_skin(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 200)

        def op_other(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 50)

        out = skin_aware_apply(img, op_skin, op_other, skin_mask=mask)
        assert int(out[0, 0, 0]) == 200
        assert int(out[0, 1, 0]) == 200
        assert int(out[1, 0, 0]) == 50
        assert int(out[1, 1, 0]) == 50

    def test_mask_none_computes_from_image(self):
        bgr = np.zeros((4, 4, 3), dtype=np.uint8)
        bgr[0, 0] = (80, 120, 180)
        bgr[0, 1] = (255, 0, 0)

        def op_skin(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 250)

        def op_other(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 10)

        out = skin_aware_apply(bgr, op_skin, op_other)
        assert out.shape == bgr.shape
        assert out.dtype == np.uint8

    def test_output_shape_and_dtype(self):
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        out = skin_aware_apply(
            img,
            lambda x: x,
            lambda x: x,
            skin_mask=np.ones((4, 4), dtype=np.float32),
        )
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_identity_ops_pass_through(self):
        img = np.full((2, 2, 3), 77, dtype=np.uint8)
        out = skin_aware_apply(
            img,
            lambda x: x,
            lambda x: x,
            skin_mask=np.array([[1.0, 0.0], [0.5, 0.5]], dtype=np.float32),
        )
        assert np.array_equal(out, img)
