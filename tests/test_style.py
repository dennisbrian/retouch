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


class TestWeightedMeanStd:
    def test_uniform_weights(self):
        data = np.random.randn(10, 10, 3).astype(np.float32) * 30 + 128
        weights = np.ones((10, 10), dtype=np.float32)
        mean, std = weighted_mean_std(data, weights)
        assert mean.shape == (3,)
        assert std.shape == (3,)
        assert np.all(std > 0)

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
        src = np.full((20, 20, 3), 100, dtype=np.uint8)
        ref = np.full((20, 20, 3), 200, dtype=np.uint8)
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
