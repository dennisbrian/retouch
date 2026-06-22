"""Tests for retouch/lips.py — LipEnhancer."""
import cv2
import numpy as np
import pytest
from retouch.lips import LipEnhancer, LIP_TINTS


@pytest.fixture
def enhancer():
    return LipEnhancer()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def lip_mask():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[30:40, 25:40] = 1.0
    return mask


class TestEnhance:
    def test_zero_strength(self, enhancer, img, lip_mask):
        result = enhancer.enhance(img, lip_mask, strength=0)
        assert np.all(result == img)

    def test_none_mask(self, enhancer, img):
        result = enhancer.enhance(img, None, strength=50)
        assert np.all(result == img)

    def test_empty_mask(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer.enhance(img, mask, strength=50)
        assert np.all(result == img)

    def test_output_shape(self, enhancer, img, lip_mask):
        result = enhancer.enhance(img, lip_mask, strength=50)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_changes_image_default(self, enhancer, img, lip_mask):
        result = enhancer.enhance(img, lip_mask, strength=50)
        assert not np.allclose(result, img)

    def test_changes_image_matte(self, enhancer, img, lip_mask):
        result = enhancer.enhance(img, lip_mask, strength=50, finish="matte")
        assert not np.allclose(result, img)

    def test_changes_image_velvet(self, enhancer, img, lip_mask):
        result = enhancer.enhance(img, lip_mask, strength=50, finish="velvet")
        assert not np.allclose(result, img)

    def test_with_tint(self, enhancer, img, lip_mask):
        result = enhancer.enhance(img, lip_mask, strength=50, tint="pink")
        assert not np.allclose(result, img)

    def test_with_cosplay_tint(self, enhancer, img, lip_mask):
        result = enhancer.enhance(img, lip_mask, strength=50, tint="cosplay")
        assert not np.allclose(result, img)

    def test_tint_as_tuple(self, enhancer, img, lip_mask):
        result = enhancer.enhance(img, lip_mask, strength=50, tint=(100, 150, 200))
        assert not np.allclose(result, img)


class TestAddLipGloss:
    def test_zero_strength(self, enhancer, img, lip_mask):
        result = enhancer._add_lip_gloss(img, lip_mask, 0)
        assert np.all(result == img)

    def test_no_lip_pixels_returns_original(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer._add_lip_gloss(img, mask, 0.5)
        assert np.all(result == img)

    def test_with_lip_highlights(self, enhancer):
        img = np.full((64, 64, 3), 50, dtype=np.uint8)
        img[32:36, 30:34] = 200
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[30:40, 25:40] = 1.0
        result = enhancer._add_lip_gloss(img, mask, 0.5)
        assert not np.allclose(result, img)

    def test_output_type(self, enhancer, img, lip_mask):
        result = enhancer._add_lip_gloss(img, lip_mask, 0.5)
        assert result.dtype == np.uint8


class TestExtractTexture:
    def test_output_shape(self, enhancer, img, lip_mask):
        texture = enhancer._extract_texture(img, lip_mask, 200)
        assert texture.shape == (64, 64)

    def test_zero_output_with_flat_image(self, enhancer, lip_mask):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        texture = enhancer._extract_texture(img, lip_mask, 200)
        assert np.allclose(texture, 0, atol=1)

    def test_nonzero_with_varied_image(self, enhancer, lip_mask):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        texture = enhancer._extract_texture(img, lip_mask, 200)
        assert np.any(texture != 0)


class TestReapplyTexture:
    def test_opacity_zero(self, enhancer, img, lip_mask):
        texture = np.zeros((64, 64), dtype=np.float32)
        result = enhancer._reapply_texture(img, texture, lip_mask, 200, opacity=0)
        assert np.allclose(result, img)

    def test_with_texture(self, enhancer, img, lip_mask):
        texture = np.ones((64, 64), dtype=np.float32) * 10
        result = enhancer._reapply_texture(img, texture, lip_mask, 200, opacity=0.5)
        assert not np.allclose(result, img)


class TestSmooth:
    def test_zero_strength(self, enhancer, img, lip_mask):
        result = enhancer._smooth(img, lip_mask, 0)
        assert np.all(result == img)

    def test_changes_image(self, enhancer, img, lip_mask):
        result = enhancer._smooth(img, lip_mask, 1.0)
        assert not np.allclose(result, img)


class TestApplyTint:
    def test_unknown_tint_falls_back_to_nude(self, enhancer, img, lip_mask):
        result = enhancer._apply_tint(img, lip_mask, "nonexistent", 0.5)
        assert result.shape == (64, 64, 3)

    def test_changes_with_tint(self, enhancer, img, lip_mask):
        result = enhancer._apply_tint(img, lip_mask, "rose", 0.5)
        assert not np.allclose(result, img)

    def test_tuple_tint(self, enhancer, img, lip_mask):
        result = enhancer._apply_tint(img, lip_mask, (200, 150, 100), 0.5)
        assert not np.allclose(result, img)

    def test_zero_strength(self, enhancer, img, lip_mask):
        result = enhancer._apply_tint(img, lip_mask, "pink", 0)
        assert np.all(result == img)

    def test_all_tints(self, enhancer, img, lip_mask):
        for tint_name in LIP_TINTS:
            result = enhancer._apply_tint(img, lip_mask, tint_name, 0.5)
            assert not np.allclose(result, img), f"Tint {tint_name} did not change image"
