"""Tests for retouch/makeup.py — MakeupEngine."""
import math
import cv2
import numpy as np
import pytest
from retouch.makeup import MakeupEngine


class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.z = 0.0


class MockLandmarks:
    def __init__(self):
        self.landmark = [MockLandmark(0.5, 0.5) for _ in range(500)]
        self.landmark[117] = MockLandmark(0.7, 0.6)
        self.landmark[346] = MockLandmark(0.3, 0.6)
        self.landmark[4] = MockLandmark(0.5, 0.7)


class MockFaceRegions:
    pass


@pytest.fixture
def engine():
    return MakeupEngine()


@pytest.fixture
def img():
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def landmarks():
    return MockLandmarks()


class TestApplyBlush:
    def test_zero_strength(self, engine, img, landmarks):
        result = engine.apply_blush(img, landmarks, 100, strength=0)
        assert np.all(result == img)

    def test_output_shape(self, engine, img, landmarks):
        result = engine.apply_blush(img, landmarks, 100, strength=50)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_changes_image(self, engine, img, landmarks):
        result = engine.apply_blush(img, landmarks, 100, strength=50)
        assert not np.allclose(result, img)

    def test_small_face_radius(self, engine, img, landmarks):
        result = engine.apply_blush(img, landmarks, 20, strength=50)
        assert result.shape == (100, 100, 3)

    def test_nose_blush(self, engine, img, landmarks):
        result = engine.apply_blush(img, landmarks, 100, strength=50, nose_blush=True)
        assert result.shape == (100, 100, 3)

    def test_under_eye_blush(self, engine, img, landmarks):
        regions = MockFaceRegions()
        regions.left_under_eye = np.ones((100, 100), dtype=np.float32)
        regions.right_under_eye = np.ones((100, 100), dtype=np.float32)
        regions.lips = np.zeros((100, 100), dtype=np.float32)
        result = engine.apply_blush(img, landmarks, 100, strength=50, regions=regions, under_eye_blush=True)
        assert result.shape == (100, 100, 3)

    def test_blush_with_lip_exclusion(self, engine, img, landmarks):
        regions = MockFaceRegions()
        regions.lips = np.ones((100, 100), dtype=np.float32)
        regions.left_under_eye = None
        regions.right_under_eye = None
        result = engine.apply_blush(img, landmarks, 100, strength=50, regions=regions)
        assert result.shape == (100, 100, 3)

    def test_both_nose_and_under_eye(self, engine, img, landmarks):
        regions = MockFaceRegions()
        regions.left_under_eye = np.ones((100, 100), dtype=np.float32)
        regions.right_under_eye = np.ones((100, 100), dtype=np.float32)
        regions.lips = np.zeros((100, 100), dtype=np.float32)
        result = engine.apply_blush(img, landmarks, 100, strength=50, regions=regions,
                                    nose_blush=True, under_eye_blush=True)
        assert not np.allclose(result, img)

    def test_lip_mask_normalization(self, engine, img, landmarks):
        regions = MockFaceRegions()
        regions.lips = np.ones((100, 100), dtype=np.uint8) * 255
        regions.left_under_eye = None
        regions.right_under_eye = None
        result = engine.apply_blush(img, landmarks, 100, strength=50, regions=regions)
        assert result.shape == (100, 100, 3)
