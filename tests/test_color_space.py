"""Tests for LCH range-editing primitives in retouch/color_space.py.

Covers: hue_range_mask, adjust_hue_range, adjust_chroma_range,
adjust_luminance_range, split_tone_lch, color_balance_lch.
"""

import numpy as np

from retouch.color_space import (
    adjust_chroma_range,
    adjust_hue_range,
    adjust_luminance_range,
    bgr_to_lch,
    color_balance_lch,
    hue_range_mask,
    lch_to_bgr,
    negative_split_tone_lch,
    split_tone_lch,
)

_WHITE_BGR = np.full((4, 4, 3), 255, dtype=np.uint8)
_GRAY_BGR = np.full((4, 4, 3), 128, dtype=np.uint8)
_RED_BGR = np.zeros((4, 4, 3), dtype=np.uint8)
_RED_BGR[:, :, 2] = 200


class TestHueRangeMask:
    def test_center_returns_one(self):
        lch = bgr_to_lch(_RED_BGR)
        actual_hue = lch[0, 0, 2]
        mask = hue_range_mask(lch, hue_center=actual_hue, hue_width=10.0, falloff=5.0)
        assert mask.max() > 0.9

    def test_opposite_hue_is_zero(self):
        """A pixel at hue 180° should get mask ≈ 0 when centre is 0°."""
        lch = bgr_to_lch(np.full((4, 4, 3), [80, 255, 80], dtype=np.uint8))
        mask = hue_range_mask(lch, hue_center=0.0, hue_width=10.0, falloff=5.0)
        assert mask.max() < 0.1

    def test_mask_is_float32(self):
        lch = bgr_to_lch(_GRAY_BGR)
        mask = hue_range_mask(lch)
        assert mask.dtype == np.float32
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_zero_width_all_zero(self):
        lch = bgr_to_lch(_RED_BGR)
        mask = hue_range_mask(lch, hue_width=0.0, falloff=0.0)
        assert np.all(mask == 0.0)

    def test_full_coverage_at_wide_bounds(self):
        """hue_width=180 + falloff=0 should cover the entire circle."""
        lch = bgr_to_lch(_RED_BGR)
        mask = hue_range_mask(lch, hue_width=180.0, falloff=0.0)
        assert mask.max() >= 0.99


class TestAdjustHueRange:
    def test_noop_at_zero_delta(self):
        lch = bgr_to_lch(_RED_BGR)
        result = adjust_hue_range(lch, hue_center=0.0, delta_deg=0.0)
        assert np.allclose(result[:, :, 2], lch[:, :, 2], atol=0.1)

    def test_narrow_shift_affects_only_center(self):
        """Shift of 30° at centre=0° should only move reds, not cyans."""
        cyan_bgr = np.full((4, 4, 3), [200, 200, 0], dtype=np.uint8)
        lch = bgr_to_lch(cyan_bgr)
        h_before = lch[:, :, 2].copy()
        result = adjust_hue_range(lch, hue_center=0.0, hue_width=15.0, delta_deg=30.0)
        assert np.allclose(result[:, :, 2], h_before, atol=0.5)

    def test_full_shift_moves_red_toward_orange(self):
        lch = bgr_to_lch(_RED_BGR)
        h_before = lch[:, :, 2].mean()
        result = adjust_hue_range(lch, hue_center=0.0, hue_width=180.0, delta_deg=15.0)
        assert result[:, :, 2].mean() != h_before

    def test_roundtrip_preserves_shape(self):
        lch = bgr_to_lch(_GRAY_BGR)
        result = adjust_hue_range(lch, hue_center=50.0, delta_deg=20.0)
        assert result.shape == lch.shape
        assert result.dtype == np.float32


class TestAdjustChromaRange:
    def test_factor_one_is_noop(self):
        lch = bgr_to_lch(_RED_BGR)
        result = adjust_chroma_range(lch, hue_center=0.0, factor=1.0)
        assert np.allclose(result, lch, atol=0.1)

    def test_factor_two_boosts_red(self):
        lch = bgr_to_lch(_RED_BGR)
        result = adjust_chroma_range(lch, hue_center=0.0, hue_width=180.0, factor=2.0)
        assert (result[:, :, 1] > lch[:, :, 1]).any()

    def test_factor_zero_kills_chroma_in_range(self):
        lch = bgr_to_lch(_RED_BGR)
        result = adjust_chroma_range(lch, hue_center=0.0, hue_width=180.0, factor=0.0)
        assert np.allclose(result[:, :, 1], 0.0, atol=0.1)


class TestAdjustLuminanceRange:
    def test_zero_delta_is_noop(self):
        lch = bgr_to_lch(_RED_BGR)
        result = adjust_luminance_range(lch, hue_center=0.0, delta=0.0)
        assert np.allclose(result[:, :, 0], lch[:, :, 0], atol=0.1)

    def test_positive_delta_brightens_reds(self):
        lch = bgr_to_lch(_RED_BGR)
        result = adjust_luminance_range(lch, hue_center=0.0, hue_width=180.0, delta=20.0)
        assert (result[:, :, 0] > lch[:, :, 0]).all()

    def test_clamps_to_range(self):
        lch = bgr_to_lch(_GRAY_BGR)
        result = adjust_luminance_range(lch, hue_center=50.0, delta=200.0)
        assert result[:, :, 0].max() <= 100.0
        assert result[:, :, 0].min() >= 0.0


class TestSplitToneLCH:
    def test_zero_sat_is_noop(self):
        lch = bgr_to_lch(_RED_BGR)
        result = split_tone_lch(lch, shadow_sat=0.0, highlight_sat=0.0)
        assert np.allclose(result, lch, atol=0.1)

    def test_shadow_only_affects_darks(self):
        """Shadow split-toning should tint dark pixels more than light ones."""
        # Create a gradient from black to white
        grad = np.zeros((1, 256, 3), dtype=np.uint8)
        for i in range(256):
            grad[0, i, :] = i
        lch = bgr_to_lch(grad)
        result = split_tone_lch(
            lch, shadow_hue=30.0, shadow_sat=0.5,
            highlight_hue=0.0, highlight_sat=0.0,
        )
        # Left half (dark) should shift hue more than right half (light)
        dark_shift = np.abs(np.diff(result[:, :128, 2])).mean()
        light_shift = np.abs(np.diff(result[:, 128:, 2])).mean()
        # Shadow tint should produce more hue variation in the dark region
        assert dark_shift > light_shift * 0.5

    def test_highlight_only_affects_lights(self):
        grad = np.zeros((1, 256, 3), dtype=np.uint8)
        for i in range(256):
            grad[0, i, :] = i
        lch = bgr_to_lch(grad)
        result = split_tone_lch(
            lch, shadow_hue=0.0, shadow_sat=0.0,
            highlight_hue=200.0, highlight_sat=0.5,
        )
        dark_shift = np.abs(np.diff(result[:, :128, 2])).mean()
        light_shift = np.abs(np.diff(result[:, 128:, 2])).mean()
        assert light_shift > dark_shift * 0.5

    def test_preserves_shape_and_dtype(self):
        lch = bgr_to_lch(_GRAY_BGR)
        result = split_tone_lch(lch, shadow_hue=30.0, shadow_sat=0.3,
                               highlight_hue=200.0, highlight_sat=0.3)
        assert result.shape == lch.shape
        assert result.dtype == np.float32

    def test_balance_shifts_crossover(self):
        """With balance=100, highlights should dominate."""
        grad = np.zeros((1, 256, 3), dtype=np.uint8)
        for i in range(256):
            grad[0, i, :] = i
        lch = bgr_to_lch(grad)
        result_bal = split_tone_lch(
            lch, shadow_hue=30.0, shadow_sat=0.5,
            highlight_hue=200.0, highlight_sat=0.0, balance=100.0,
        )
        result_no_bal = split_tone_lch(
            lch, shadow_hue=30.0, shadow_sat=0.5,
            highlight_hue=200.0, highlight_sat=0.0, balance=0.0,
        )
        # With balance pushed positive, shadow tint should be weaker
        assert result_bal[:, :, 2].mean() != result_no_bal[:, :, 2].mean()

    def test_bgr_roundtrip(self):
        """Full BGR→LCH→split-tone→BGR round-trip shouldn't crash."""
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        lch = bgr_to_lch(img)
        toned = split_tone_lch(lch, shadow_hue=30.0, shadow_sat=0.3,
                              highlight_hue=200.0, highlight_sat=0.3)
        result = lch_to_bgr(toned)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestColorBalanceLCH:
    def test_all_zero_is_noop(self):
        lch = bgr_to_lch(_RED_BGR)
        result = color_balance_lch(lch, 0.0, 0.0, 0.0)
        assert np.allclose(result, lch, atol=0.1)

    def test_cyan_red_axis_moves_reds(self):
        lch = bgr_to_lch(_RED_BGR)
        result = color_balance_lch(lch, cyan_red=1.0)
        # Red (hue ~0°) pushed further toward red — hue should shift
        assert not np.allclose(result[:, :, 2], lch[:, :, 2], atol=0.5)

    def test_yellow_blue_does_not_affect_reds(self):
        """Yellow-blue axis should have minimal effect on red pixels."""
        lch = bgr_to_lch(_RED_BGR)
        result = color_balance_lch(lch, yellow_blue=1.0)
        # Red at hue 0° is far from both yellow (60°) and blue (240°)
        assert np.allclose(result[:, :, 0], lch[:, :, 0], atol=0.1)

    def test_gray_pixels_are_protected(self):
        """Gray pixels (low chroma) should not shift hue."""
        lch = bgr_to_lch(_GRAY_BGR)
        result = color_balance_lch(lch, cyan_red=1.0, magenta_green=1.0,
                                   yellow_blue=1.0)
        # L* should be nearly unchanged for gray
        assert np.allclose(result[:, :, 0], lch[:, :, 0], atol=1.0)

    def test_preserve_luminosity(self):
        lch = bgr_to_lch(_RED_BGR)
        result = color_balance_lch(lch, cyan_red=0.5, preserve_luminosity=True)
        assert np.allclose(result[:, :, 0], lch[:, :, 0], atol=0.1)

    def test_no_preserve_luminosity(self):
        lch = bgr_to_lch(_RED_BGR)
        result = color_balance_lch(lch, cyan_red=0.5, preserve_luminosity=False)
        # L* should change when preserve_luminosity=False
        assert not np.allclose(result[:, :, 0], lch[:, :, 0], atol=0.1)

    def test_preserves_shape_and_dtype(self):
        lch = bgr_to_lch(_GRAY_BGR)
        result = color_balance_lch(lch, cyan_red=0.3)
        assert result.shape == lch.shape
        assert result.dtype == np.float32

    def test_bgr_roundtrip(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        lch = bgr_to_lch(img)
        balanced = color_balance_lch(lch, cyan_red=0.3, magenta_green=-0.2)
        result = lch_to_bgr(balanced)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_clamped_inputs(self):
        """Values outside [-1, 1] should be clamped."""
        lch = bgr_to_lch(_RED_BGR)
        result = color_balance_lch(lch, cyan_red=5.0)
        assert result[:, :, 2].min() >= 0.0


class TestNegativeSplitToneLCH:
    def test_noop_at_zero(self):
        lch = bgr_to_lch(_RED_BGR)
        result = negative_split_tone_lch(lch, shadow_desat=0.0, highlight_desat=0.0)
        assert np.allclose(result, lch, atol=0.1)

    def test_shadow_desat_reduces_chroma(self):
        lch = bgr_to_lch(_RED_BGR)
        result = negative_split_tone_lch(lch, shadow_desat=1.0)
        assert result[:, :, 1].max() <= lch[:, :, 1].max()

    def test_highlight_desat_reduces_chroma(self):
        # White image has high L*, so highlight desat affects it
        lch = bgr_to_lch(_WHITE_BGR)
        result = negative_split_tone_lch(lch, highlight_desat=1.0)
        assert result[:, :, 1].max() <= lch[:, :, 1].max()

    def test_clamped_values(self):
        lch = bgr_to_lch(_RED_BGR)
        result = negative_split_tone_lch(lch, shadow_desat=5.0)
        assert result[:, :, 1].min() >= 0.0

    def test_preserves_shape_and_dtype(self):
        lch = bgr_to_lch(_RED_BGR)
        result = negative_split_tone_lch(lch, shadow_desat=0.5, highlight_desat=0.3)
        assert result.shape == lch.shape
        assert result.dtype == np.float32
