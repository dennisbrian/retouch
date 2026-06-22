"""Tests for retouch/teeth.py — TeethWhitener."""
import cv2
import numpy as np
import pytest
from retouch.teeth import TeethWhitener


@pytest.fixture
def whitener():
    return TeethWhitener()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


class TestWhiten:
    def test_zero_strength(self, whitener, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener.whiten(img, mask, strength=0)
        assert np.all(result == img)

    def test_none_mask(self, whitener, img):
        result = whitener.whiten(img, None, strength=50)
        assert np.all(result == img)

    def test_empty_mask(self, whitener, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = whitener.whiten(img, mask, strength=50)
        assert np.all(result == img)

    def test_output_shape(self, whitener, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener.whiten(img, mask, strength=50)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_changes_image_with_teeth(self, whitener):
        img = np.full((64, 64, 3), 200, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener.whiten(img, mask, strength=80)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() >= orig_lab[:, :, 0].mean()


class TestDetectTeeth:
    def test_small_mouth_returns_empty(self, whitener, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[32:34, 32:34] = 1.0
        result = whitener._detect_teeth(img, mask)
        assert result.shape == (64, 64)
        assert result.max() == 0.0

    def test_output_shape(self, whitener, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener._detect_teeth(img, mask)
        assert result.shape == (64, 64)
        assert result.dtype == np.float32

    def test_detects_bright_pixels(self, whitener):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        img[:, :, 1] = 80
        img[20:40, 20:40] = [220, 215, 210]
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener._detect_teeth(img, mask)
        assert result[30, 30] > 0

    def test_skips_dark_pixels(self, whitener, img):
        img[:] = 30
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener._detect_teeth(img, mask)
        assert result.max() == 0.0

    def test_skips_saturated_pixels(self, whitener):
        img = np.full((64, 64, 3), 50, dtype=np.uint8)
        img[:, :, 1] = 10
        mask = np.ones((64, 64), dtype=np.float32)
        result = whitener._detect_teeth(img, mask)
        assert result.max() < 0.5
