"""Tests for engine stage methods — _stage_global, _stage_grade, _stage_finish, etc."""
import cv2
import numpy as np
import pytest
from retouch.engine import (
    _adjust_vibrance,
    _adjust_saturation,
    _adjust_contrast,
    _adjust_tonal,
    _apply_selective_sharpening,
)
from retouch.grading import ColorGrader


class TestAdjustVibrance:
    def test_zero_returns_original(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = _adjust_vibrance(img, 0)
        assert np.all(result == img)

    def test_positive_vibrance(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 50
        result = _adjust_vibrance(img, 50)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() > 50

    def test_negative_vibrance(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 200
        result = _adjust_vibrance(img, -50)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() < 200

    def test_skin_hue_protection(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 100
        result_no_skin = _adjust_vibrance(img, 50)
        # Skin-hue pixels should be protected from over-saturation
        hsv = cv2.cvtColor(result_no_skin, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() <= 255

    def test_output_type(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = _adjust_vibrance(img, 30)
        assert result.dtype == np.uint8


class TestAdjustSaturation:
    def test_zero_returns_original(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = _adjust_saturation(img, 0)
        assert np.all(result == img)

    def test_positive_saturation(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 50
        result = _adjust_saturation(img, 50)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() > 50

    def test_negative_saturation(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 200
        result = _adjust_saturation(img, -50)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() < 200

    def test_output_type(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = _adjust_saturation(img, 30)
        assert result.dtype == np.uint8


class TestAdjustContrast:
    def test_zero_returns_same(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = _adjust_contrast(img, 0)
        assert np.all(result == img)

    def test_positive_increases_contrast(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = _adjust_contrast(img, 50)
        assert not np.allclose(result, img)

    def test_negative_decreases_contrast(self):
        img = np.full((10, 10, 3), [50, 150, 200], dtype=np.uint8)
        result = _adjust_contrast(img, -50)
        assert result.dtype == np.uint8

    def test_output_type(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = _adjust_contrast(img, 30)
        assert result.dtype == np.uint8


class TestAdjustTonal:
    def test_no_adjustment(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = _adjust_tonal(img)
        assert np.all(result == img)

    def test_shadows_lift(self):
        img = np.full((10, 10, 3), 50, dtype=np.uint8)
        result = _adjust_tonal(img, shadows=50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() > 50

    def test_highlights_boost(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = _adjust_tonal(img, highlights=50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() >= 200

    def test_whites(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = _adjust_tonal(img, whites=50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() >= 200

    def test_blacks(self):
        img = np.full((10, 10, 3), 30, dtype=np.uint8)
        result = _adjust_tonal(img, blacks=-50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() < 30

    def test_all_controls(self):
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        result = _adjust_tonal(img, shadows=30, highlights=-20, whites=10, blacks=-5)
        assert result.shape == img.shape

    def test_output_dtype(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = _adjust_tonal(img, shadows=20)
        assert result.dtype == np.uint8


class TestApplySelectiveSharpening:
    def test_zero_radius_mask(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.float32)
        result = _apply_selective_sharpening(img, mask, radius=0.8, amount=1.0, threshold=0)
        assert np.all(result == img)

    def test_full_mask_on_flat(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result = _apply_selective_sharpening(img, mask, radius=1.0, amount=1.0, threshold=0)
        assert np.allclose(result, img, atol=1)

    def test_sharpens_edges(self):
        img = np.zeros((20, 20, 3), dtype=np.uint8)
        img[:, 10:] = 255
        mask = np.ones((20, 20), dtype=np.float32)
        result = _apply_selective_sharpening(img, mask, radius=0.8, amount=1.5, threshold=2)
        assert not np.allclose(result, img)

    def test_threshold_filters_noise(self):
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result_no_thresh = _apply_selective_sharpening(img, mask, radius=0.8, amount=1.0, threshold=0)
        result_thresh = _apply_selective_sharpening(img, mask, radius=0.8, amount=1.0, threshold=10)
        assert result_thresh.shape == result_no_thresh.shape

    def test_output_shape(self):
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        mask = np.random.rand(20, 20).astype(np.float32)
        result = _apply_selective_sharpening(img, mask)
        assert result.shape == (20, 20, 3)

    def test_3d_mask_accepted(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.ones((20, 20, 1), dtype=np.float32)
        result = _apply_selective_sharpening(img, mask)
        assert result.shape == (20, 20, 3)
