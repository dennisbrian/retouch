"""Tests for retouch/color_space.py — LCH color space utilities."""

import numpy as np
import cv2

from retouch.color_space import (
    bgr_to_lab,
    lab_to_bgr,
    lab_to_lch,
    lch_to_lab,
    bgr_to_lch,
    lch_to_bgr,
    adjust_luminance,
    adjust_chroma,
    adjust_hue,
    skin_mask_lch,
)


class TestBgrLabRoundTrip:
    def test_single_pixel_round_trip(self):
        bgr = np.array([[[40, 90, 200]]], dtype=np.uint8)
        lab = bgr_to_lab(bgr)
        assert lab.shape == (1, 1, 3)
        assert lab.dtype == np.float32
        assert 0.0 <= lab[0, 0, 0] <= 100.0
        assert -128.0 <= lab[0, 0, 1] <= 127.0
        assert -128.0 <= lab[0, 0, 2] <= 127.0
        recovered = lab_to_bgr(lab)
        assert recovered.shape == bgr.shape
        assert recovered.dtype == np.uint8
        assert np.allclose(recovered, bgr, atol=2)

    def test_grayscale_keeps_equal_channels(self):
        bgr = np.full((4, 4, 3), 128, dtype=np.uint8)
        lab = bgr_to_lab(bgr)
        assert np.allclose(lab[:, :, 1], 0.0, atol=0.5)
        assert np.allclose(lab[:, :, 2], 0.0, atol=0.5)
        assert 50.0 < lab[0, 0, 0] < 60.0
        recovered = lab_to_bgr(lab)
        assert np.allclose(recovered, bgr, atol=2)


class TestBgrLchRoundTrip:
    def test_image_round_trip(self):
        rng = np.random.default_rng(42)
        bgr = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        lch = bgr_to_lch(bgr)
        assert lch.shape == bgr.shape
        assert lch.dtype == np.float32
        assert lch[:, :, 0].min() >= 0.0
        assert lch[:, :, 0].max() <= 100.0
        assert lch[:, :, 1].min() >= 0.0
        assert lch[:, :, 2].min() >= 0.0
        assert lch[:, :, 2].max() < 360.0
        recovered = lch_to_bgr(lch)
        assert recovered.shape == bgr.shape
        assert recovered.dtype == np.uint8
        assert np.allclose(recovered, bgr, atol=25)

    def test_lab_lch_inverse(self):
        lch = np.array([[[50.0, 30.0, 90.0]]], dtype=np.float32)
        lab = lch_to_lab(lch)
        recovered = lab_to_lch(lab)
        assert np.allclose(recovered, lch, atol=1e-3)


class TestAdjustLuminance:
    def test_gray_delta_20(self):
        bgr = np.full((1, 1, 3), 128, dtype=np.uint8)
        lch = bgr_to_lch(bgr)
        out = adjust_luminance(lch, 20.0)
        assert abs(out[0, 0, 0] - (lch[0, 0, 0] + 20.0)) < 0.1

    def test_clamp_upper(self):
        lch = np.array([[[95.0, 20.0, 30.0]]], dtype=np.float32)
        out = adjust_luminance(lch, 20.0)
        assert out[0, 0, 0] == 100.0

    def test_clamp_lower(self):
        lch = np.array([[[5.0, 20.0, 30.0]]], dtype=np.float32)
        out = adjust_luminance(lch, -20.0)
        assert out[0, 0, 0] == 0.0

    def test_zero_delta_is_identity(self):
        lch = np.array([[[50.0, 25.0, 45.0]]], dtype=np.float32)
        out = adjust_luminance(lch, 0.0)
        assert np.allclose(out, lch)

    def test_does_not_mutate_input(self):
        lch = np.array([[[50.0, 20.0, 30.0]]], dtype=np.float32)
        _ = adjust_luminance(lch, 10.0)
        assert lch[0, 0, 0] == 50.0


class TestAdjustChroma:
    def test_gray_stays_gray(self):
        bgr = np.full((1, 1, 3), 128, dtype=np.uint8)
        lch = bgr_to_lch(bgr)
        out = adjust_chroma(lch, 1.5)
        assert out[0, 0, 1] == 0.0

    def test_saturated_pixel_increases(self):
        bgr = np.array([[[60, 140, 220]]], dtype=np.uint8)
        lch = bgr_to_lch(bgr)
        c_before = lch[0, 0, 1]
        out = adjust_chroma(lch, 1.2)
        c_after = out[0, 0, 1]
        assert c_after > c_before
        assert abs(c_after - c_before * 1.2) < 0.1

    def test_factor_one_is_identity(self):
        lch = np.array([[[50.0, 25.0, 45.0]]], dtype=np.float32)
        out = adjust_chroma(lch, 1.0)
        assert np.allclose(out, lch)


class TestAdjustHue:
    def test_red_shift_180_is_cyan(self):
        bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
        lch = bgr_to_lch(bgr)
        out = adjust_hue(lch, 180.0)
        recovered = lch_to_bgr(out)
        b, g, r = int(recovered[0, 0, 0]), int(recovered[0, 0, 1]), int(recovered[0, 0, 2])
        assert b > r
        assert b > g

    def test_wrap_around(self):
        lch = np.array([[[50.0, 20.0, 350.0]]], dtype=np.float32)
        out = adjust_hue(lch, 30.0)
        assert 0.0 <= out[0, 0, 2] < 360.0
        assert abs(out[0, 0, 2] - 20.0) < 1e-3

    def test_zero_delta_is_identity(self):
        lch = np.array([[[50.0, 25.0, 200.0]]], dtype=np.float32)
        out = adjust_hue(lch, 0.0)
        assert np.allclose(out, lch)

    def test_does_not_mutate_input(self):
        lch = np.array([[[50.0, 20.0, 30.0]]], dtype=np.float32)
        _ = adjust_hue(lch, 50.0)
        assert lch[0, 0, 2] == 30.0


class TestSkinMaskLch:
    def test_skin_tone_detected(self):
        lch = np.array([[[60.0, 20.0, 25.0]]], dtype=np.float32)
        mask = skin_mask_lch(lch)
        assert mask.shape == (1, 1)
        assert mask.dtype == np.float32
        assert mask[0, 0] > 0.5

    def test_skin_tone_from_bgr(self):
        bgr = np.array([[[80, 120, 180]]], dtype=np.uint8)
        lch = bgr_to_lch(bgr)
        mask = skin_mask_lch(lch)
        assert mask[0, 0] > 0.05

    def test_blue_pixel_not_detected(self):
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
        lch = bgr_to_lch(bgr)
        mask = skin_mask_lch(lch)
        assert mask[0, 0] < 0.1

    def test_gray_pixel_not_detected(self):
        bgr = np.array([[[128, 128, 128]]], dtype=np.uint8)
        lch = bgr_to_lch(bgr)
        mask = skin_mask_lch(lch)
        assert mask[0, 0] == 0.0

    def test_hue_far_from_center(self):
        lch = np.array([[[50.0, 20.0, 200.0]]], dtype=np.float32)
        mask = skin_mask_lch(lch)
        assert mask[0, 0] < 0.1

    def test_low_chroma_excluded(self):
        lch = np.array([[[50.0, 1.0, 25.0]]], dtype=np.float32)
        mask = skin_mask_lch(lch, chroma_min=8.0)
        assert mask[0, 0] == 0.0
