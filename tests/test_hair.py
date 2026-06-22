"""Tests for retouch/hair.py — HairEnhancer."""
import cv2
import numpy as np
import pytest
from retouch.hair import HairEnhancer


@pytest.fixture
def enhancer():
    return HairEnhancer()


@pytest.fixture
def img():
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def person_mask():
    return np.ones((100, 100), dtype=np.float32)


@pytest.fixture
def face_oval():
    m = np.zeros((100, 100), dtype=np.float32)
    m[30:70, 25:75] = 1.0
    return m


@pytest.fixture
def bbox():
    return (20, 30, 60, 60)


class TestEnhance:
    def test_zero_strength(self, enhancer, img, person_mask, face_oval, bbox):
        result = enhancer.enhance(img, person_mask, face_oval, bbox, strength=0)
        assert np.all(result == img)

    def test_no_person_mask_returns_original(self, enhancer, img, face_oval, bbox):
        result = enhancer.enhance(img, None, face_oval, bbox, strength=50)
        assert np.all(result == img)

    def test_no_face_oval_returns_original(self, enhancer, img, person_mask, bbox):
        result = enhancer.enhance(img, person_mask, None, bbox, strength=50)
        assert np.all(result == img)

    def test_output_shape(self, enhancer, img, person_mask, face_oval, bbox):
        result = enhancer.enhance(img, person_mask, face_oval, bbox, strength=50)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_changes_image(self, enhancer, img, person_mask, face_oval, bbox):
        result = enhancer.enhance(img, person_mask, face_oval, bbox, strength=80)
        assert not np.allclose(result, img)

    def test_with_hair_mask(self, enhancer, img, person_mask, face_oval, bbox):
        hair_mask = np.zeros((100, 100), dtype=np.float32)
        hair_mask[10:30, 25:75] = 1.0
        result = enhancer.enhance(img, person_mask, face_oval, bbox, strength=50, hair_mask=hair_mask)
        assert result.shape == (100, 100, 3)

    def test_tiny_bbox_returns_original(self, enhancer, img, person_mask, face_oval):
        bbox = (0, 0, 1, 1)
        result = enhancer.enhance(img, person_mask, face_oval, bbox, strength=50)
        assert np.all(result == img)

    def test_hair_mask_squeeze_3d(self, enhancer, img, bbox):
        pm = np.ones((100, 100, 1), dtype=np.float32)
        face_oval = np.ones((100, 100), dtype=np.float32)
        result = enhancer.enhance(img, pm, face_oval, bbox, strength=50)
        assert result.shape == (100, 100, 3)

    def test_hair_mask_normalization(self, enhancer, img, bbox):
        pm = np.ones((100, 100), dtype=np.uint8) * 255
        face_oval = np.ones((100, 100), dtype=np.float32)
        result = enhancer.enhance(img, pm, face_oval, bbox, strength=50)
        assert result.shape == (100, 100, 3)
