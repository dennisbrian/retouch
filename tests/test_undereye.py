"""Tests for retouch/undereye.py — UnderEyeRepairer."""
import numpy as np
import pytest
from retouch.undereye import UnderEyeRepairer


class MockFaceRegions:
    def __init__(self):
        self.left_under_eye = None
        self.right_under_eye = None


@pytest.fixture
def repairer():
    return UnderEyeRepairer()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


class TestRepair:
    def test_zero_strength(self, repairer, img):
        regions = MockFaceRegions()
        result = repairer.repair(img, regions, strength=0)
        assert np.all(result == img)

    def test_none_masks(self, repairer, img):
        regions = MockFaceRegions()
        regions.left_under_eye = None
        regions.right_under_eye = None
        result = repairer.repair(img, regions, strength=50)
        assert np.all(result == img)

    def test_empty_masks(self, repairer, img):
        regions = MockFaceRegions()
        regions.left_under_eye = np.zeros((64, 64), dtype=np.float32)
        regions.right_under_eye = np.zeros((64, 64), dtype=np.float32)
        result = repairer.repair(img, regions, strength=50)
        assert np.all(result == img)

    def test_output_shape(self, repairer, img):
        regions = MockFaceRegions()
        regions.left_under_eye = np.ones((64, 64), dtype=np.float32)
        regions.right_under_eye = np.ones((64, 64), dtype=np.float32)
        result = repairer.repair(img, regions, strength=50)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_changes_image(self, repairer):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[20:44, 20:44] = [80, 80, 80]
        regions = MockFaceRegions()
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0
        regions.left_under_eye = mask
        regions.right_under_eye = np.zeros((64, 64), dtype=np.float32)
        result = repairer.repair(img, regions, strength=80)
        assert not np.allclose(result, img)


    def test_strength_zero_with_masks(self, repairer, img):
        regions = MockFaceRegions()
        regions.left_under_eye = np.ones((64, 64), dtype=np.float32)
        regions.right_under_eye = np.ones((64, 64), dtype=np.float32)
        result = repairer.repair(img, regions, strength=0)
        assert np.array_equal(result, img)

    def test_output_dtype_shape_preserved(self, repairer, img):
        regions = MockFaceRegions()
        regions.left_under_eye = np.ones((64, 64), dtype=np.float32) * 0.5
        regions.right_under_eye = np.ones((64, 64), dtype=np.float32) * 0.5
        result = repairer.repair(img, regions, strength=75)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

class TestRepairRegion:
    def test_small_surround_returns_original(self, repairer, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = repairer._repair_region(img, mask, 0.5)
        assert np.all(result == img)

    def test_changes_dark_region(self, repairer):
        img = np.full((64, 64, 3), 50, dtype=np.uint8)
        img[:32, :] = 128
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[32:, :] = 1.0
        result = repairer._repair_region(img, mask, 0.8)
        assert not np.allclose(result, img)

    def test_output_type(self, repairer, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = repairer._repair_region(img, mask, 0.3)
        assert result.dtype == np.uint8
