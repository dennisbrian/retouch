"""Tests for retouch/frequency.py — frequency separation and recombination."""

import numpy as np
import cv2
import pytest

from retouch.frequency import (
    FrequencyLayers,
    separate,
    combine,
    DEFAULT_FEATHER_MIN,
)


class TestFrequencyLayers:
    def test_reconstruct(self):
        low = np.zeros((10, 10, 3), dtype=np.float32)
        mid = np.zeros((10, 10, 3), dtype=np.float32)
        high = np.zeros((10, 10, 3), dtype=np.float32)
        layers = FrequencyLayers(low, mid, high)
        recon = layers.reconstruct()
        assert recon.dtype == np.uint8
        assert recon.shape == (10, 10, 3)

    def test_reconstruction_preserves_value(self):
        img = np.full((20, 20, 3), 100, dtype=np.uint8)
        layers = separate(img, face_width=100)
        recon = layers.reconstruct()
        assert np.allclose(recon.astype(np.float32), img.astype(np.float32), atol=2)


class TestSeparate:
    def test_returns_frequency_layers(self):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=100)
        assert isinstance(layers, FrequencyLayers)
        assert layers.low.shape == (50, 50, 3)
        assert layers.mid.shape == (50, 50, 3)
        assert layers.high.shape == (50, 50, 3)

    def test_separate_is_lossless(self):
        np.random.seed(42)
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        layers = separate(img, face_width=80)
        recon = layers.reconstruct()
        err = np.abs(recon.astype(np.float32) - img.astype(np.float32)).max()
        assert err < 2.0, f"Reconstruction error too large: {err}"

    def test_low_is_smoothed(self):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        img[20:30, 20:30] = 255
        layers = separate(img, face_width=100)
        assert np.all(layers.low[20, 20] < 255)
        assert np.any(layers.high[20, 20] != 0)


class TestCombine:
    def test_none_mask_reconstructs(self):
        img = np.full((40, 40, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=80)
        result = combine(layers, skin_mask=None)
        assert np.allclose(result.astype(np.float32), img.astype(np.float32), atol=2)

    def test_zero_smooth_returns_original(self):
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        layers = separate(img, face_width=80)
        mask = np.ones((40, 40), dtype=np.float32)
        result = combine(layers, skin_mask=mask, smooth_strength=0, mid_reduction=0)
        assert np.allclose(result.astype(np.float32), img.astype(np.float32), atol=2)

    def test_mid_reduction_works(self):
        img = np.full((40, 40, 3), 128, dtype=np.uint8)
        img[20:22, 20:22] = 200  # small blemish-like spot
        layers = separate(img, face_width=80)
        mask = np.ones((40, 40), dtype=np.float32)
        result_before = combine(layers, skin_mask=mask, smooth_strength=0, mid_reduction=0)
        result_after = combine(layers, skin_mask=mask, smooth_strength=0, mid_reduction=1)
        assert not np.array_equal(result_before, result_after)

    def test_texture_opacity_reduces_high(self):
        img = checkerboard_100()
        layers = separate(img, face_width=100)
        mask = np.ones((100, 100), dtype=np.float32)
        result = combine(layers, skin_mask=mask, smooth_strength=0, mid_reduction=0, texture_opacity=0)
        # With 0 texture opacity, high frequencies should be suppressed
        assert not np.array_equal(result, img)

    def test_feather_mask_shape(self):
        """Ensure combine returns HxWx3 even with 2D mask."""
        img = np.full((30, 40, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=60)
        mask = np.ones((30, 40), dtype=np.float32)
        result = combine(layers, skin_mask=mask, smooth_strength=0.5, face_width=60)
        assert result.shape == (30, 40, 3)
        assert result.dtype == np.uint8

    def test_strength_zero_returns_input(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        layers = separate(img, face_width=64)
        mask = np.ones((32, 32), dtype=np.float32)
        result = combine(layers, skin_mask=mask, smooth_strength=0,
                         mid_reduction=0, texture_opacity=1.0)
        assert np.allclose(result.astype(np.float32), img.astype(np.float32), atol=2)

    def test_combine_output_dtype_shape(self):
        img = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        layers = separate(img, face_width=96)
        mask = np.ones((48, 64), dtype=np.float32)
        result = combine(layers, skin_mask=mask, smooth_strength=0.5)
        assert result.dtype == np.uint8
        assert result.shape == (48, 64, 3)


# Helpers
def checkerboard_100():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(0, 100, 10):
        for x in range(0, 100, 10):
            val = 255 if (x // 10 + y // 10) % 2 == 0 else 0
            img[y:y+10, x:x+10] = val
    return img


class TestPoreSynthesis:
    def test_pore_synthesis_changes_image(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=100)
        mask = np.ones((100, 100), dtype=np.float32)
        # Without pore synthesis
        res_no_pore = combine(layers, skin_mask=mask, smooth_strength=0, mid_reduction=0, pore_synthesis=0.0)
        # With pore synthesis
        res_with_pore = combine(
            layers, skin_mask=mask, smooth_strength=0, mid_reduction=0,
            pore_synthesis=0.5, face_width=100.0, roi_coords=(10, 20)
        )
        assert not np.array_equal(res_no_pore, res_with_pore)
        
    def test_pore_synthesis_is_deterministic(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=100)
        mask = np.ones((100, 100), dtype=np.float32)
        res1 = combine(
            layers, skin_mask=mask, smooth_strength=0, mid_reduction=0,
            pore_synthesis=0.5, face_width=100.0, roi_coords=(10, 20)
        )
        res2 = combine(
            layers, skin_mask=mask, smooth_strength=0, mid_reduction=0,
            pore_synthesis=0.5, face_width=100.0, roi_coords=(10, 20)
        )
        assert np.array_equal(res1, res2)

    def test_pore_synthesis_different_coords_different_noise(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=100)
        mask = np.ones((100, 100), dtype=np.float32)
        res1 = combine(
            layers, skin_mask=mask, smooth_strength=0, mid_reduction=0,
            pore_synthesis=0.5, face_width=100.0, roi_coords=(10, 20)
        )
        res2 = combine(
            layers, skin_mask=mask, smooth_strength=0, mid_reduction=0,
            pore_synthesis=0.5, face_width=100.0, roi_coords=(10, 21)
        )
        assert not np.array_equal(res1, res2)
