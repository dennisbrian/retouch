"""Tests for retouch/blemish.py — BlemishRemover."""
import cv2
import numpy as np
import pytest
from retouch.blemish import BlemishRemover, compute_skin_quality_map


@pytest.fixture
def remover():
    return BlemishRemover()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def skin_mask():
    m = np.zeros((64, 64), dtype=np.float32)
    m[16:48, 16:48] = 1.0
    return m


class TestRemove:
    def test_zero_strength(self, remover, img, skin_mask):
        result = remover.remove(img, skin_mask, strength=0)
        assert np.all(result == img)

    def test_no_blemish_returns_original(self, remover, img, skin_mask):
        result = remover.remove(img, skin_mask, strength=50)
        assert np.all(result == img)

    def test_output_shape(self, remover, img, skin_mask):
        result = remover.remove(img, skin_mask, strength=50)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_removes_dark_spot(self, remover):
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        img[124:132, 124:132] = [30, 30, 30]
        mask = np.ones((256, 256), dtype=np.float32)
        result = remover.remove(img, mask, strength=80)
        assert not np.allclose(result[128, 128], [30, 30, 30], atol=10)

    def test_very_small_skin_region(self, remover, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[32, 32] = 1.0
        result = remover.remove(img, mask, strength=50)
        assert result.shape == (64, 64, 3)


class TestDetect:
    def test_returns_binary_mask(self, remover, img, skin_mask):
        result = remover._detect(img, skin_mask, 0.5, 200)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)
        assert result.max() <= 255

    def test_flat_skin_no_blemish(self, remover, img, skin_mask):
        result = remover._detect(img, skin_mask, 0.5, 200)
        assert result.max() == 0

    def test_detect_dark_spot(self, remover):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[30:35, 30:35] = [30, 30, 30]
        mask = np.ones((64, 64), dtype=np.float32)
        result = remover._detect(img, mask, 0.9, 200)
        assert result[32, 32] > 0

    def test_high_sensitivity_detects_more(self, remover):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[32, 32] = [80, 80, 80]
        mask = np.ones((64, 64), dtype=np.float32)
        result_low = remover._detect(img, mask, 0.3, 200)
        result_high = remover._detect(img, mask, 0.9, 200)
        assert result_high.sum() >= result_low.sum()

    def test_scale_affects_ksize(self, remover, img, skin_mask):
        result_small = remover._detect(img, skin_mask, 0.5, 100)
        result_large = remover._detect(img, skin_mask, 0.5, 500)
        assert result_small.shape == result_large.shape


class TestSkinQualityMap:
    def test_returns_float32(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.float32)
        result = compute_skin_quality_map(img, mask)
        assert result.dtype == np.float32
        assert result.shape == (32, 32)

    def test_flat_image_low_quality(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.float32)
        result = compute_skin_quality_map(img, mask)
        assert result.max() <= 1.0

    def test_no_skin_returns_zeros(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.float32)
        result = compute_skin_quality_map(img, mask)
        assert result.sum() == 0.0

    def test_varied_skin_quality(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.float32)
        result = compute_skin_quality_map(img, mask)
        assert result.max() <= 1.0
        assert result.min() >= 0.0
