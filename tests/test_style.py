"""Tests for retouch/style.py — StyleProfile, helpers."""

import numpy as np
import pytest

from retouch.style import (
    StyleProfile,
    weighted_mean_std,
    reinhard_transfer_masked,
)


class TestStyleProfile:
    def test_default_values(self):
        sp = StyleProfile()
        assert sp.brightness_delta == 0.0
        assert sp.contrast_delta == 0.0
        assert sp.skin_smooth_strength == 0.0
        assert sp.skin_texture_opacity == 1.0

    def test_to_dict(self):
        sp = StyleProfile(brightness_delta=10.5, contrast_delta=5.0)
        d = sp.to_dict()
        assert d["brightness_delta"] == 10.5
        assert d["contrast_delta"] == 5.0

    def test_to_json(self):
        sp = StyleProfile(skin_l_mean_delta=3.0)
        j = sp.to_json()
        assert "skin_l_mean_delta" in j

    def test_from_dict(self):
        d = {"brightness_delta": 8.0, "contrast_delta": -3.0}
        sp = StyleProfile.from_dict(d)
        assert sp.brightness_delta == 8.0
        assert sp.contrast_delta == -3.0

    def test_from_json(self):
        j = '{"brightness_delta": 5.0, "contrast_delta": 2.0}'
        sp = StyleProfile.from_json(j)
        assert sp.brightness_delta == 5.0

    def test_save_and_load(self, tmp_path):
        sp = StyleProfile(brightness_delta=12.0)
        path = str(tmp_path / "profile.json")
        sp.save(path)
        loaded = StyleProfile.load(path)
        assert loaded.brightness_delta == 12.0

    def test_serialization_roundtrip(self):
        sp = StyleProfile(
            brightness_delta=5.0,
            contrast_delta=3.2,
            saturation_delta=10.0,
            skin_l_mean_delta=2.5,
            skin_a_mean_delta=1.0,
            skin_b_mean_delta=-2.0,
            skin_smooth_strength=0.75,
            skin_mid_reduction=0.5,
            skin_texture_opacity=0.8,
        )
        d = sp.to_dict()
        sp2 = StyleProfile.from_dict(d)
        for field in d:
            assert getattr(sp, field) == getattr(sp2, field)

    def test_style_profile_dataclass(self):
        """StyleProfile is a dataclass: construction + attribute access work."""
        sp = StyleProfile(
            brightness_delta=1.5,
            contrast_delta=-2.5,
            saturation_delta=3.0,
            skin_l_mean_delta=4.0,
            skin_a_mean_delta=5.0,
            skin_b_mean_delta=-6.0,
            skin_smooth_strength=0.5,
            skin_mid_reduction=0.4,
            skin_texture_opacity=0.7,
        )
        # Each field is accessible and round-trips through repr-like access
        assert sp.brightness_delta == 1.5
        assert sp.contrast_delta == -2.5
        assert sp.saturation_delta == 3.0
        assert sp.skin_l_mean_delta == 4.0
        assert sp.skin_a_mean_delta == 5.0
        assert sp.skin_b_mean_delta == -6.0
        assert sp.skin_smooth_strength == 0.5
        assert sp.skin_mid_reduction == 0.4
        assert sp.skin_texture_opacity == 0.7
        # Equality between identical instances (dataclass __eq__)
        assert sp == StyleProfile(
            brightness_delta=1.5,
            contrast_delta=-2.5,
            saturation_delta=3.0,
            skin_l_mean_delta=4.0,
            skin_a_mean_delta=5.0,
            skin_b_mean_delta=-6.0,
            skin_smooth_strength=0.5,
            skin_mid_reduction=0.4,
            skin_texture_opacity=0.7,
        )


class TestWeightedMeanStd:
    def test_uniform_weights(self):
        data = np.random.randn(10, 10, 3).astype(np.float32) * 30 + 128
        weights = np.ones((10, 10), dtype=np.float32)
        mean, std = weighted_mean_std(data, weights)
        assert mean.shape == (3,)
        assert std.shape == (3,)
        assert np.all(std > 0)

    def test_weighted_mean_std_uniform_weights(self):
        """Uniform weights should reproduce numpy mean/std (with small epsilon on std)."""
        rng = np.random.default_rng(0)
        data = rng.random((16, 16, 3), dtype=np.float32) * 255.0
        weights = np.ones((16, 16), dtype=np.float32)

        mean, std = weighted_mean_std(data, weights)

        expected_mean = np.mean(data, axis=(0, 1))
        # weighted_mean_std adds 1e-5 to the std for numerical stability
        expected_std = np.std(data, axis=(0, 1)) + 1e-5
        assert np.allclose(mean, expected_mean, atol=1e-4)
        assert np.allclose(std, expected_std, atol=1e-3)

    def test_weighted_mean_std_zero_weights_region(self):
        """Mask with a zero region should compute stats over the non-zero region only."""
        data = np.zeros((10, 10, 3), dtype=np.float32)
        data[:, :5, :] = 50.0  # left half = 50
        data[:, 5:, :] = 200.0  # right half = 200
        weights = np.zeros((10, 10), dtype=np.float32)
        weights[:, :5] = 1.0  # only the left half contributes

        mean, std = weighted_mean_std(data, weights)
        assert mean.shape == (3,)
        # Mean over left half = 50, std ≈ 0 + epsilon
        assert np.allclose(mean, 50.0, atol=1e-3)
        assert np.all(std > 0)
        assert np.all(std < 1.0)

    def test_zero_weights(self):
        data = np.random.randn(10, 10, 3).astype(np.float32) * 30 + 128
        weights = np.zeros((10, 10), dtype=np.float32)
        mean, std = weighted_mean_std(data, weights)
        assert mean.shape == (3,)

    def test_downsample_large(self):
        data = np.random.randn(2048, 2048, 3).astype(np.float32) * 30 + 128
        weights = np.ones((2048, 2048), dtype=np.float32)
        mean, std = weighted_mean_std(data, weights)
        assert mean.shape == (3,)


class TestReinhardTransferMasked:
    def test_same_images(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result = reinhard_transfer_masked(img, img, mask, mask)
        assert np.allclose(result, img, atol=2)

    def test_different_images(self):
        rng = np.random.default_rng(42)
        src = np.clip(rng.normal(100, 15, (20, 20, 3)), 0, 255).astype(np.uint8)
        ref = np.clip(rng.normal(180, 15, (20, 20, 3)), 0, 255).astype(np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result = reinhard_transfer_masked(src, ref, mask, mask)
        assert not np.allclose(result, src)

    def test_output_shape(self):
        src = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        ref = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result = reinhard_transfer_masked(src, ref, mask, mask)
        assert result.shape == (20, 20, 3)

    def test_diff_masks(self):
        src = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        ref = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        src_mask = np.zeros((20, 20), dtype=np.float32)
        ref_mask = np.ones((20, 20), dtype=np.float32)
        result = reinhard_transfer_masked(src, ref, src_mask, ref_mask)
        assert np.all(result == src)

    def test_reinhard_transfer_masked_basic(self):
        """Synthetic 64x64 BGR images: output must be uint8 in the 0-255 range."""
        rng = np.random.default_rng(7)
        src = np.clip(rng.normal(80, 30, (64, 64, 3)), 0, 255).astype(np.uint8)
        ref = np.clip(rng.normal(180, 25, (64, 64, 3)), 0, 255).astype(np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = reinhard_transfer_masked(src, ref, mask, mask)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_reinhard_transfer_masked_preserves_shape(self):
        """Output shape must match the source image regardless of ref image size."""
        rng = np.random.default_rng(7)
        src = np.clip(rng.normal(120, 30, (64, 64, 3)), 0, 255).astype(np.uint8)
        ref = np.clip(rng.normal(150, 30, (64, 64, 3)), 0, 255).astype(np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = reinhard_transfer_masked(src, ref, mask, mask)
        assert result.shape == src.shape
        assert result.shape == (64, 64, 3)
