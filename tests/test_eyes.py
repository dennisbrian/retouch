"""Tests for retouch/eyes.py — EyeEnhancer."""
import cv2
import numpy as np
import pytest
from retouch.eyes import EyeEnhancer


class MockFaceRegions:
    pass


@pytest.fixture
def enhancer():
    return EyeEnhancer()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def regions():
    r = MockFaceRegions()
    r.left_eye = np.zeros((64, 64), dtype=np.float32)
    r.right_eye = np.zeros((64, 64), dtype=np.float32)
    r.left_iris = np.zeros((64, 64), dtype=np.float32)
    r.right_iris = np.zeros((64, 64), dtype=np.float32)
    r.left_eye[20:30, 20:30] = 1.0
    r.right_eye[20:30, 34:44] = 1.0
    r.left_iris[24:27, 24:27] = 1.0
    r.right_iris[24:27, 37:40] = 1.0
    return r


class TestEnhance:
    def test_zero_strength(self, enhancer, img, regions):
        result = enhancer.enhance(img, regions, strength=0)
        assert np.all(result == img)

    def test_output_shape(self, enhancer, img, regions):
        result = enhancer.enhance(img, regions, strength=50)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_changes_image(self, enhancer, img, regions):
        result = enhancer.enhance(img, regions, strength=80)
        assert not np.allclose(result, img)


class TestEnhanceWhites:
    def test_none_mask(self, enhancer, img):
        result = enhancer._enhance_whites(img, None, 0.5)
        assert np.all(result == img)

    def test_empty_mask(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer._enhance_whites(img, mask, 0.5)
        assert np.all(result == img)

    def test_bright_sclera_only(self, enhancer, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._enhance_whites(img, mask, 0.5)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        # The a channel should be reduced (less red) in bright areas
        assert lab[:, :, 1].mean() <= 128

    def test_output_type(self, enhancer, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._enhance_whites(img, mask, 1.0)
        assert result.dtype == np.uint8


class TestSculptIris:
    def test_none_mask(self, enhancer, img):
        result = enhancer._sculpt_iris(img, None, 0.5)
        assert np.all(result == img)

    def test_empty_mask(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer._sculpt_iris(img, mask, 0.5)
        assert np.all(result == img)

    def test_small_iris_returns_original(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[32, 32] = 1.0
        result = enhancer._sculpt_iris(img, mask, 0.5)
        assert np.all(result == img)

    def test_changes_image(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0
        result = enhancer._sculpt_iris(img, mask, 0.5)
        assert not np.allclose(result, img)

    def test_output_shape(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0
        result = enhancer._sculpt_iris(img, mask, 0.5)
        assert result.shape == (64, 64, 3)


class TestEnhanceCatchlights:
    def test_none_mask(self, enhancer, img):
        result = enhancer._enhance_catchlights(img, None, 0.5)
        assert np.all(result == img)

    def test_empty_mask(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer._enhance_catchlights(img, mask, 0.5)
        assert np.all(result == img)

    def test_no_catchlights_returns_original(self, enhancer, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._enhance_catchlights(img, mask, 0.5)
        assert np.all(result == img)

    def test_with_bright_catchlight(self, enhancer):
        img = np.full((64, 64, 3), 50, dtype=np.uint8)
        img[30:33, 30:33] = 240
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[25:40, 25:40] = 1.0
        result = enhancer._enhance_catchlights(img, mask, 1.0)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() > orig_lab[:, :, 0].mean()

    def test_output_type(self, enhancer):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        img[30:33, 30:33] = 240
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._enhance_catchlights(img, mask, 0.5)
        assert result.dtype == np.uint8
