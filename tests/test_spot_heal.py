"""Tests for retouch/spot_heal.py — SpotHealer (F4 v0)."""
import cv2
import numpy as np
import pytest

from retouch.spot_heal import SpotHealer


@pytest.fixture
def flat_img():
    return np.full((96, 96, 3), 128, dtype=np.uint8)


@pytest.fixture
def blemish_img():
    img = np.full((96, 96, 3), 140, dtype=np.uint8)
    img[40:56, 40:56] = [20, 20, 20]
    return img


@pytest.fixture
def disk_mask():
    m = np.zeros((96, 96), dtype=np.uint8)
    cv2.circle(m, (48, 48), 8, 255, -1)
    return m


class TestHeal:
    def test_removes_blemish(self, blemish_img, disk_mask):
        healer = SpotHealer()
        result = healer.heal(blemish_img, disk_mask, method="telea", radius=5)
        assert result.dtype == np.uint8
        center = result[48, 48]
        assert center.mean() > 80

    def test_dtype_preserved_uint8(self, flat_img, disk_mask):
        result = SpotHealer().heal(flat_img, disk_mask)
        assert result.dtype == np.uint8

    def test_dtype_preserved_float32(self, flat_img, disk_mask):
        img_f = flat_img.astype(np.float32)
        result = SpotHealer().heal(img_f, disk_mask)
        assert result.dtype == np.float32
        assert result.max() <= 255.0

    def test_telea_method(self, flat_img, disk_mask):
        result = SpotHealer().heal(flat_img, disk_mask, method="telea")
        assert result.shape == flat_img.shape

    def test_ns_method(self, flat_img, disk_mask):
        result = SpotHealer().heal(flat_img, disk_mask, method="ns")
        assert result.shape == flat_img.shape
        assert result.dtype == np.uint8

    def test_zero_mask_is_noop_uint8(self, flat_img):
        zero = np.zeros((96, 96), dtype=np.uint8)
        result = SpotHealer().heal(flat_img, zero)
        assert np.array_equal(result, flat_img)

    def test_zero_mask_is_noop_float32(self, flat_img):
        img_f = flat_img.astype(np.float32)
        zero = np.zeros((96, 96), dtype=np.float32)
        result = SpotHealer().heal(img_f, zero)
        assert np.array_equal(result, img_f)

    def test_float32_mask_accepted(self, flat_img, disk_mask):
        m_f = (disk_mask.astype(np.float32) / 255.0)
        result = SpotHealer().heal(flat_img, m_f, radius=4)
        assert result.shape == flat_img.shape

    def test_feathered_boundary_smooth(self, blemish_img, disk_mask):
        healer = SpotHealer()
        result = healer.heal(blemish_img, disk_mask, method="telea", radius=5)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ring_inner = gray[44:52, 44:52]
        ring_outer = gray[36:44, 36:44]
        assert abs(ring_inner.mean() - ring_outer.mean()) < 40

    def test_shape_preserved(self, flat_img, disk_mask):
        result = SpotHealer().heal(flat_img, disk_mask, radius=7)
        assert result.shape == flat_img.shape

    def test_invalid_dtype_raises(self, flat_img, disk_mask):
        img_i16 = flat_img.astype(np.int16)
        with pytest.raises(ValueError):
            SpotHealer().heal(img_i16, disk_mask)


class TestHealObjectRemoval:
    def test_removes_large_object(self):
        img = np.full((128, 128, 3), 150, dtype=np.uint8)
        img[40:88, 40:88] = [10, 200, 10]
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[40:88, 40:88] = 255
        result = SpotHealer().heal_object_removal(img, mask, method="telea")
        assert result.dtype == np.uint8
        center = result[64, 64]
        assert abs(int(center[0]) - 150) < 60
        assert abs(int(center[1]) - 150) < 80

    def test_dtype_preserved_float32(self):
        img = np.full((96, 96, 3), 150, dtype=np.float32)
        img[40:56, 40:56] = [10.0, 200.0, 10.0]
        mask = np.zeros((96, 96), dtype=np.uint8)
        mask[40:56, 40:56] = 255
        result = SpotHealer().heal_object_removal(img, mask)
        assert result.dtype == np.float32
        assert result.max() <= 255.0

    def test_zero_mask_is_noop(self, flat_img):
        zero = np.zeros((96, 96), dtype=np.uint8)
        result = SpotHealer().heal_object_removal(flat_img, zero)
        assert np.array_equal(result, flat_img)

    def test_float_mask_accepted(self):
        img = np.full((96, 96, 3), 150, dtype=np.uint8)
        img[40:56, 40:56] = [10, 200, 10]
        mask = np.zeros((96, 96), dtype=np.float32)
        mask[40:56, 40:56] = 1.0
        result = SpotHealer().heal_object_removal(img, mask)
        assert result.shape == img.shape
