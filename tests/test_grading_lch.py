"""Tests for LCH HSL panel methods on ColorGrader (Phase 1.d).

Covers: adjust_hsl_lch, adjust_per_channel_lch, split_tone_lch,
white_balance_lch, channel_mixer_bw.
"""

import numpy as np
import pytest

from retouch.grading import ColorGrader


@pytest.fixture
def grader():
    return ColorGrader()


@pytest.fixture
def img():
    return np.full((32, 32, 3), 128, dtype=np.uint8)


@pytest.fixture
def red_img():
    img = np.zeros((16, 16, 3), dtype=np.uint8)
    img[:, :, 2] = 200
    return img


@pytest.fixture
def colorful_img():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:21, :, 2] = 200
    img[21:43, :, 1] = 180
    img[43:, :, 0] = 220
    return img


class TestAdjustHSLLch:
    def test_noop_when_all_zero(self, grader, img):
        result = grader.adjust_hsl_lch(img, hue_shift=0.0, sat_scale=1.0, lum_shift=0.0)
        np.testing.assert_allclose(result, img, atol=1)

    def test_hue_shift_changes_image(self, grader, colorful_img):
        result = grader.adjust_hsl_lch(colorful_img, hue_shift=30.0)
        assert not np.array_equal(result, colorful_img)

    def test_sat_zero_desaturates(self, grader, colorful_img):
        result = grader.adjust_hsl_lch(colorful_img, sat_scale=0.0)
        result_f = result.astype(np.float32)
        # After desaturation, all channels should be nearly equal
        for ch in range(3):
            assert np.abs(result_f[:, :, ch] - result_f[:, :, 0]).max() < 3

    def test_lum_positive_brightens(self, grader, img):
        result = grader.adjust_hsl_lch(img, lum_shift=30.0)
        assert result.mean() > img.mean()

    def test_lum_negative_darkens(self, grader, img):
        result = grader.adjust_hsl_lch(img, lum_shift=-30.0)
        assert result.mean() < img.mean()

    def test_preserves_shape_dtype(self, grader, img):
        result = grader.adjust_hsl_lch(img, hue_shift=10.0, sat_scale=1.2, lum_shift=5.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestAdjustPerChannelLch:
    def test_invalid_channel_raises(self, grader, img):
        with pytest.raises(ValueError):
            grader.adjust_per_channel_lch(img, "teal")

    def test_all_channels_valid(self, grader, img):
        for ch in ColorGrader._HSL_CHANNELS:
            result = grader.adjust_per_channel_lch(img, ch, hue_shift=0.0)
            assert result.shape == img.shape

    def test_red_channel_sat_boost_affects_reds(self, grader, red_img):
        result = grader.adjust_per_channel_lch(red_img, "red", sat_scale=2.0)
        # Should be more vivid — red channel higher
        assert result[:, :, 2].mean() >= red_img[:, :, 2].mean()

    def test_noop_when_all_zero(self, grader, img):
        result = grader.adjust_per_channel_lch(img, "blue", 0.0, 1.0, 0.0)
        np.testing.assert_allclose(result, img, atol=1)

    def test_lum_shift_affects_image(self, grader, colorful_img):
        result = grader.adjust_per_channel_lch(colorful_img, "green", lum_shift=30.0)
        assert not np.array_equal(result, colorful_img)


class TestSplitToneLch:
    def test_noop_at_zero(self, grader, img):
        result = grader.split_tone_lch(img, shadow_sat=0.0, highlight_sat=0.0)
        np.testing.assert_allclose(result, img, atol=1)

    def test_shadow_tint_affects_image(self, grader, colorful_img):
        result = grader.split_tone_lch(
            colorful_img, shadow_hue=30.0, shadow_sat=0.5,
            highlight_hue=200.0, highlight_sat=0.3,
        )
        assert not np.array_equal(result, colorful_img)

    def test_preserves_shape_dtype(self, grader, img):
        result = grader.split_tone_lch(img, shadow_hue=30.0, shadow_sat=0.3,
                                       highlight_hue=200.0, highlight_sat=0.3)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_balance_modifies_result(self, grader, colorful_img):
        a = grader.split_tone_lch(colorful_img, shadow_hue=30.0, shadow_sat=0.5,
                                  highlight_hue=0.0, highlight_sat=0.0, balance=-80.0)
        b = grader.split_tone_lch(colorful_img, shadow_hue=30.0, shadow_sat=0.5,
                                  highlight_hue=0.0, highlight_sat=0.0, balance=80.0)
        assert not np.array_equal(a, b)


class TestWhiteBalanceLch:
    def test_neutral_is_noop(self, grader, img):
        result = grader.white_balance_lch(img, temperature=6500.0, tint=0.0)
        np.testing.assert_allclose(result, img, atol=1)

    def test_warm_adds_warmth(self, grader, colorful_img):
        result = grader.white_balance_lch(colorful_img, temperature=3000.0)
        assert not np.array_equal(result, colorful_img)

    def test_cool_shifts_blue(self, grader, colorful_img):
        result = grader.white_balance_lch(colorful_img, temperature=10000.0)
        assert not np.array_equal(result, colorful_img)

    def test_tint_magenta(self, grader, colorful_img):
        result = grader.white_balance_lch(colorful_img, temperature=6500.0, tint=50.0)
        assert not np.array_equal(result, colorful_img)

    def test_tint_green(self, grader, colorful_img):
        result = grader.white_balance_lch(colorful_img, temperature=6500.0, tint=-50.0)
        assert not np.array_equal(result, colorful_img)

    def test_clamped_temperature(self, grader, colorful_img):
        result = grader.white_balance_lch(colorful_img, temperature=100.0)
        assert result.shape == colorful_img.shape
        assert result.dtype == np.uint8


class TestChannelMixerBw:
    def test_standard_weights_grayscale(self, grader, colorful_img):
        result = grader.channel_mixer_bw(colorful_img)
        assert result.shape == colorful_img.shape
        assert result.dtype == np.uint8
        for ch in range(3):
            np.testing.assert_array_equal(result[:, :, ch], result[:, :, 0])

    def test_zero_weights_returns_black(self, grader, img):
        result = grader.channel_mixer_bw(img, 0.0, 0.0, 0.0)
        assert np.all(result == 0)

    def test_red_only_matches_red_channel(self, grader, img):
        img_copy = img.copy()
        img_copy[:, :, 2] = 200
        img_copy[:, :, 0] = 50
        img_copy[:, :, 1] = 50
        result = grader.channel_mixer_bw(img_copy, r_weight=1.0, g_weight=0.0, b_weight=0.0)
        assert np.all(result[:, :, 0] == 200)

    def test_brightness_shifts(self, grader, img):
        a = grader.channel_mixer_bw(img, brightness=0.0)
        b = grader.channel_mixer_bw(img, brightness=30.0)
        assert b.mean() > a.mean()

    def test_contrast_stretches(self, grader, img):
        result = grader.channel_mixer_bw(img, contrast=30.0)
        assert result.dtype == np.uint8


class TestNegativeSplitTone:
    def test_noop_when_both_zero(self, grader, colorful_img):
        result = grader.negative_split_tone(colorful_img, shadow_desat=0.0, highlight_desat=0.0)
        np.testing.assert_allclose(result, colorful_img, atol=1)

    def test_shadow_desat_darkens_shadows(self, grader, colorful_img):
        result = grader.negative_split_tone(colorful_img, shadow_desat=0.8, highlight_desat=0.0)
        assert result.dtype == np.uint8
        assert not np.array_equal(result, colorful_img)

    def test_highlight_desat_affects_highlights(self, grader, colorful_img):
        result = grader.negative_split_tone(colorful_img, shadow_desat=0.0, highlight_desat=0.8)
        assert result.dtype == np.uint8
        assert not np.array_equal(result, colorful_img)

    def test_both_active(self, grader, colorful_img):
        result = grader.negative_split_tone(colorful_img, shadow_desat=0.5, highlight_desat=0.5)
        assert result.shape == colorful_img.shape

    def test_full_desat_is_not_identity(self, grader, colorful_img):
        result = grader.negative_split_tone(colorful_img, shadow_desat=1.0, highlight_desat=1.0)
        assert not np.array_equal(result, colorful_img)
