"""Tests for retouch/utils.py — pure helper functions."""

import numpy as np
import cv2
import pytest

from retouch.utils import (
    feather_mask,
    blend_masked,
    create_polygon_mask,
    vibrance,
    apply_curve,
    correct_exposure,
    adaptive_ksize,
    apply_global_bloom,
)


class TestFeatherMask:
    def test_none_mask(self):
        assert feather_mask(None, radius=5) is None

    def test_empty_mask(self):
        m = np.zeros((0, 0), dtype=np.float32)
        result = feather_mask(m, radius=5)
        assert result.size == 0

    def test_no_radius_no_sigma(self):
        m = np.zeros((10, 10), dtype=np.float32)
        m[4:6, 4:6] = 1.0
        result = feather_mask(m)
        assert np.allclose(result, m)

    def test_normalize_from_uint8(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        m[4:6, 4:6] = 255
        result = feather_mask(m, radius=3)
        assert result.max() <= 1.0
        assert result.dtype == np.float32

    def test_feather_reduces_max(self):
        m = np.zeros((30, 30), dtype=np.float32)
        m[10:20, 10:20] = 1.0
        result = feather_mask(m, radius=5)
        assert result.max() < 1.0
        assert result.shape == (30, 30)


class TestBlendMasked:
    def test_none_mask_returns_processed(self):
        orig = np.full((10, 10, 3), 0, dtype=np.uint8)
        proc = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = blend_masked(orig, proc, None)
        assert np.all(result == 255)

    def test_full_mask_returns_processed(self):
        orig = np.full((10, 10, 3), 0, dtype=np.uint8)
        proc = np.full((10, 10, 3), 255, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        result = blend_masked(orig, proc, mask)
        assert np.all(result == 255)

    def test_zero_mask_returns_original(self):
        orig = np.full((10, 10, 3), 128, dtype=np.uint8)
        proc = np.full((10, 10, 3), 0, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.float32)
        result = blend_masked(orig, proc, mask)
        assert np.all(result == 128)

    def test_half_mask(self):
        orig = np.full((10, 10, 3), 0, dtype=np.uint8)
        proc = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[:, :5] = 1.0
        result = blend_masked(orig, proc, mask)
        assert np.all(result[:, :5] == 100)
        assert np.all(result[:, 5:] == 0)

    def test_3d_mask_accepted(self):
        orig = np.full((10, 10, 3), 0, dtype=np.uint8)
        proc = np.full((10, 10, 3), 255, dtype=np.uint8)
        mask = np.ones((10, 10, 1), dtype=np.float32)
        result = blend_masked(orig, proc, mask)
        assert np.all(result == 255)


class TestCreatePolygonMask:
    def test_basic_polygon(self):
        pts = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.int32)
        mask = create_polygon_mask(pts, (40, 40), feather_radius=0)
        assert mask.shape == (40, 40)
        assert mask[10:30, 10:30].mean() > 0.5
        assert mask[0, 0] == 0.0

    def test_with_feathering(self):
        pts = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.int32)
        mask = create_polygon_mask(pts, (40, 40), feather_radius=3)
        assert mask.shape == (40, 40)
        assert mask.max() <= 1.0


class TestVibrance:
    def test_zero_strength(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        result = vibrance(img, mask, 0.0)
        assert np.all(result == img)

    def test_increases_saturation_masked(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 50  # low saturation green
        mask = np.ones((10, 10), dtype=np.float32)
        result = vibrance(img, mask, 1.0)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() > 50

    def test_mask_restricts_area(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 50
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[:, :5] = 1.0
        result = vibrance(img, mask, 1.0)
        assert not np.allclose(result, img)


class TestApplyCurve:
    def test_identity_curve(self):
        channel = np.arange(256, dtype=np.uint8).reshape(256, 1)
        curve = [(0, 0), (255, 255)]
        result = apply_curve(channel, curve)
        assert np.allclose(result.ravel(), np.arange(256))

    def test_invert_curve(self):
        channel = np.array([0, 128, 255], dtype=np.uint8)
        curve = [(0, 255), (255, 0)]
        result = apply_curve(channel, curve)
        assert result[0] == 255
        assert result[2] == 0

    def test_clamp_output(self):
        channel = np.array([128], dtype=np.uint8)
        curve = [(0, 0), (128, 300), (255, 255)]
        result = apply_curve(channel, curve)
        assert result[0] == 255


class TestCorrectExposure:
    def test_already_correct(self):
        img = np.full((50, 50, 3), 120, dtype=np.uint8)
        result, corrected = correct_exposure(img)
        assert not corrected
        assert np.all(result == img)

    def test_too_dark_global(self):
        img = np.full((50, 50, 3), 30, dtype=np.uint8)
        result, corrected = correct_exposure(img, min_threshold=50)
        assert corrected

    def test_too_bright_global(self):
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        result, corrected = correct_exposure(img, max_threshold=180)
        assert corrected

    def test_with_face_bbox(self):
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        bbox = (30, 30, 40, 40)
        result, corrected = correct_exposure(
            img, face_bboxes=[bbox], face_min_threshold=60, target_mean=120
        )
        assert corrected


class TestAdaptiveKsize:
    def test_odd_output(self):
        assert adaptive_ksize(100, factor=0.1, minimum=3) % 2 == 1

    def test_minimum_enforced(self):
        assert adaptive_ksize(10, factor=0.1, minimum=5) == 5

    def test_proportional(self):
        k1 = adaptive_ksize(100, factor=0.1, minimum=3)
        k2 = adaptive_ksize(200, factor=0.1, minimum=3)
        assert k2 >= k1


class TestGlobalBloom:
    def test_zero_strength_returns_original(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = apply_global_bloom(img, strength=0.0, threshold=200.0, softness=30.0)
        assert np.all(result == img)

    def test_bloom_disperses_highlights(self):
        # Create a dark image with a bright square in the center
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[45:55, 45:55] = 255  # bright highlight
        
        # Apply bloom with 100 strength, 200 threshold, and 10 softness
        result = apply_global_bloom(img, strength=100.0, threshold=200.0, softness=10.0)
        
        # Verify that the center is still bright
        assert np.all(result[49, 49] > 200)
        # Verify that pixels outside the original highlight are now non-zero (bloom dispersion)
        assert np.any(result[40, 40] > 0)
        assert np.any(result[60, 60] > 0)

    def test_low_threshold_glows_all(self):
        img = np.full((50, 50, 3), 160, dtype=np.uint8)
        result = apply_global_bloom(img, strength=50.0, threshold=150.0, softness=20.0)
        # Low threshold means the entire image glows, screen blending increases brightness
        assert result.mean() > img.mean()

    def test_downsampled_large_image(self):
        # Create a large image (> 2000px min_dim)
        img = np.zeros((2200, 2200, 3), dtype=np.uint8)
        img[1000:1200, 1000:1200] = 255
        result = apply_global_bloom(img, strength=50.0, threshold=200.0, softness=30.0)
        assert result.shape == (2200, 2200, 3)
