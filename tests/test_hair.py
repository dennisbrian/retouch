"""Tests for retouch/hair.py — HairEnhancer."""
import cv2
import numpy as np
import pytest
from retouch.hair import HairEnhancer, cleanup_flyaway_strands


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


def test_hair_enhance_zero_strength(enhancer, img, person_mask, face_oval, bbox):
    result = enhancer.enhance(img, person_mask, face_oval, bbox, strength=0)
    assert np.all(result == img)


def test_hair_enhance_none_mask(enhancer, img, face_oval, bbox):
    result = enhancer.enhance(img, None, face_oval, bbox, strength=50)
    assert np.all(result == img)


def test_hair_enhance_output_range(enhancer, img, person_mask, face_oval, bbox):
    result = enhancer.enhance(img, person_mask, face_oval, bbox, strength=80)
    assert result.dtype == np.uint8
    assert result.min() >= 0
    assert result.max() <= 255


class TestFlyawayStrandCleanup:
    """AB3: Flyaway hair strand cleanup outside hair mass silhouette."""

    def test_zero_strength_noop(self):
        img = np.full((100, 100, 3), 180, dtype=np.uint8)
        hair_mask = np.zeros((100, 100), dtype=np.float32)
        hair_mask[10:40, 20:80] = 1.0
        result = cleanup_flyaway_strands(img, hair_mask, strength=0.0)
        assert np.all(result == img)

    def test_hair_mass_region_byte_identical(self):
        """Main hair mass silhouette must be byte-identical (100% scope guarded)."""
        img = np.full((100, 100, 3), 180, dtype=np.uint8)
        hair_mask = np.zeros((100, 100), dtype=np.float32)
        hair_mask[10:40, 20:80] = 1.0

        # Draw a synthetic strand in background outside hair mass
        img[60:80, 50] = 20  # Dark strand

        result = cleanup_flyaway_strands(img, hair_mask, strength=1.0)
        # Hair body pixels (row 15..35, col 25..75) must be 100% byte-identical
        assert np.all(result[15:35, 25:75] == img[15:35, 25:75])

    def test_removes_background_stray_strand(self):
        """Stray hair strand on background outside hair mass is cleaned up."""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        hair_mask = np.zeros((100, 100), dtype=np.float32)
        hair_mask[10:30, 20:80] = 1.0

        # Draw a thin dark strand at y=60..70, x=50
        img[60:70, 50] = 20

        result = cleanup_flyaway_strands(img, hair_mask, strength=1.0)
        # Inprinted strand region (row 60..70, col 50) should be restored toward background lightness (200)
        strand_diff = float(np.mean(np.abs(result[60:70, 50].astype(float) - 200.0)))
        assert strand_diff < 50.0, f"Strand not inpainted properly: diff={strand_diff:.1f}"

