"""Tests for retouch/lips.py — LipEnhancer."""
import numpy as np
import pytest
from retouch.lips import LipEnhancer


@pytest.fixture
def enhancer():
    return LipEnhancer()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def lip_mask():
    m = np.zeros((64, 64), dtype=np.float32)
    m[24:40, 20:44] = 1.0
    return m


@pytest.fixture
def lip_img():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    img[24:40, 20:44] = [100, 60, 180]
    return img


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

    def test_changes_image_default(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50)
        assert not np.allclose(result, lip_img)

    def test_changes_image_matte(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50, finish="matte")
        assert not np.allclose(result, lip_img)

    def test_changes_image_velvet(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50, finish="velvet")
        assert not np.allclose(result, lip_img)

    def test_with_tint(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50, tint="pink")
        assert not np.allclose(result, lip_img)

    def test_with_cosplay_tint(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50, tint="cosplay")
        assert not np.allclose(result, lip_img)

    def test_tint_as_tuple(self, enhancer, lip_img, lip_mask):
        result = enhancer.enhance(lip_img, lip_mask, strength=50, tint=(100, 150, 200))
        assert not np.allclose(result, lip_img)


class TestAddLipGloss:
    def test_zero_strength(self, enhancer, img, lip_mask):
        result = enhancer._add_lip_gloss(img, lip_mask, 0)
        assert np.all(result == img)

    def test_no_lip_pixels_returns_original(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer._add_lip_gloss(img, mask, 0.5)
        assert np.all(result == img)

    def test_with_lip_highlights(self, enhancer):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        img[24:40, 20:44] = [180, 130, 200]
        img[28:32, 28:32] = [240, 230, 250]
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[24:40, 20:44] = 1.0
        result = enhancer._add_lip_gloss(img, mask, 0.8)
        assert not np.allclose(result, img)

    def test_output_type(self, enhancer, img, lip_mask):
        result = enhancer._add_lip_gloss(img, lip_mask, 0.3)
        assert result.dtype == np.uint8


class TestExtractTexture:
    def test_output_shape(self, enhancer, lip_img, lip_mask):
        result = enhancer._extract_texture(lip_img, lip_mask, face_width=500)
        assert result.shape == (64, 64)

    def test_zero_output_with_flat_image(self, enhancer, img, lip_mask):
        result = enhancer._extract_texture(img, lip_mask, face_width=500)
        assert np.all(result == 0)

    def test_nonzero_with_varied_image(self, enhancer):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[:, :, 0] = np.tile(np.arange(0, 64, dtype=np.uint8), (64, 1))
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._extract_texture(img, mask, face_width=500)
        assert np.abs(result).sum() > 0


class TestReapplyTexture:
    def test_opacity_zero(self, enhancer, lip_img, lip_mask):
        texture = np.random.rand(64, 64).astype(np.float32) * 30
        result = enhancer._reapply_texture(lip_img, texture, lip_mask, face_width=500, opacity=0)
        assert np.allclose(result, lip_img, atol=1)

    def test_with_texture(self, enhancer, lip_img, lip_mask):
        texture = np.random.rand(64, 64).astype(np.float32) * 20
        result = enhancer._reapply_texture(lip_img, texture, lip_mask, face_width=500, opacity=0.5)
        assert result.shape == (64, 64, 3)


class TestSmooth:
    def test_zero_strength(self, enhancer, img, lip_mask):
        result = enhancer._smooth(img, lip_mask, 0)
        assert np.all(result == img)

    def test_changes_image(self, enhancer):
        x = np.tile(np.linspace(100, 200, 64, dtype=np.uint8), (64, 1))
        img = np.stack([x, x, x], axis=-1)
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._smooth(img, mask, 0.5)
        assert not np.allclose(result, img)


class TestApplyTint:
    def test_unknown_tint_falls_back_to_nude(self, enhancer, lip_img, lip_mask):
        result = enhancer._apply_tint(lip_img, lip_mask, "nonexistent", 0.25)
        assert result.shape == (64, 64, 3)

    def test_changes_with_tint(self, enhancer, lip_img, lip_mask):
        result = enhancer._apply_tint(lip_img, lip_mask, "pink", 0.25)
        assert not np.allclose(result, lip_img)

    def test_tuple_tint(self, enhancer, lip_img, lip_mask):
        result = enhancer._apply_tint(lip_img, lip_mask, (100, 150, 200), 0.25)
        assert not np.allclose(result, lip_img)

    def test_zero_strength(self, enhancer, lip_img, lip_mask):
        result = enhancer._apply_tint(lip_img, lip_mask, "pink", 0)
        assert np.all(result == lip_img)

    def test_all_tints(self, enhancer, lip_img, lip_mask):
        for tint in ["nude", "pink", "rose", "coral", "berry", "red", "cosplay"]:
            result = enhancer._apply_tint(lip_img, lip_mask, tint, 0.25)
            assert not np.allclose(result, lip_img), f"Tint {tint} did not change image"
