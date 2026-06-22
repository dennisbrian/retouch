"""Tests for retouch/undereye.py — UnderEyeRepairer."""
import cv2
import numpy as np
import pytest
from retouch.undereye import UnderEyeRepairer


class MockFaceRegions:
    pass


@pytest.fixture
def repairer():
    return UnderEyeRepairer()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


class TestRepair:
    def test_zero_strength(self, repairer, img):
        regions = MockFaceRegions()
        regions.left_under_eye = np.ones((64, 64), dtype=np.float32)
        regions.right_under_eye = np.ones((64, 64), dtype=np.float32)
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
        img = np.full((64, 64, 3), [100, 100, 100], dtype=np.uint8)
        regions = MockFaceRegions()
        regions.left_under_eye = np.ones((64, 64), dtype=np.float32)
        regions.right_under_eye = np.ones((64, 64), dtype=np.float32)
        result = repairer.repair(img, regions, strength=80)
        assert not np.allclose(result, img)


class TestRepairRegion:
    def test_small_surround_returns_original(self, repairer, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = repairer._repair_region(img, mask, 0.5)
        assert np.all(result == img)

    def test_changes_dark_region(self, repairer):
        img = np.full((64, 64, 3), 50, dtype=np.uint8)
        img[:32, :] = 128
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[40:60, 40:60] = 1.0
        result = repairer._repair_region(img, mask, 0.5)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        assert lab[50, 50, 0] >= orig_lab[50, 50, 0]

    def test_output_type(self, repairer, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = repairer._repair_region(img, mask, 0.5)
        assert result.dtype == np.uint8
