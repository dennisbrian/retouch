"""Tests for retouch/regions.py — unified mask system."""

import numpy as np
import pytest

from retouch.parsing import FaceRegions
from retouch.regions import (
    _blend_multiply,
    _blend_normal,
    _blend_overlay,
    _blend_screen,
    _blend_soft_light,
    apply_to_region,
    combine_masks,
    dilate_mask,
    erode_mask,
    feather_mask,
    threshold_mask,
)
from retouch.utils import normalize_mask


class TestApplyToRegion:
    def test_zero_mask_returns_input(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.float32)

        def op(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 200)

        out = apply_to_region(img, mask, op)
        assert np.array_equal(out, img)

    def test_full_mask_applies_op(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)

        def op(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 200)

        out = apply_to_region(img, mask, op)
        assert np.all(out == 200)

    def test_half_strength_is_50_percent_blend(self):
        img = np.full((10, 10, 3), 0, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)

        def op(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 200)

        out = apply_to_region(img, mask, op, strength=0.5)
        assert np.allclose(out, 100, atol=1)

    def test_half_mask_is_half_apply(self):
        img = np.full((10, 10, 3), 0, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[:, :5] = 1.0

        def op(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 200)

        out = apply_to_region(img, mask, op)
        assert np.all(out[:, :5] == 200)
        assert np.all(out[:, 5:] == 0)

    def test_invalid_blend_mode_raises(self):
        img = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            apply_to_region(img, mask, lambda x: x, blend_mode="bogus")

    def test_invalid_strength_raises(self):
        img = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError):
            apply_to_region(img, mask, lambda x: x, strength=1.5)

    def test_three_dim_mask_accepted(self):
        img = np.full((10, 10, 3), 50, dtype=np.uint8)
        mask = np.ones((10, 10, 1), dtype=np.float32)

        def op(x: np.ndarray) -> np.ndarray:
            return np.full_like(x, 200)

        out = apply_to_region(img, mask, op)
        assert np.all(out == 200)


class TestCombineMasks:
    def test_union_takes_max(self):
        a = np.zeros((10, 10), dtype=np.float32)
        a[2:5, 2:5] = 0.5
        b = np.zeros((10, 10), dtype=np.float32)
        b[6:9, 6:9] = 0.8
        out = combine_masks([a, b], mode="union")
        assert out[3, 3] == pytest.approx(0.5)
        assert out[7, 7] == pytest.approx(0.8)
        assert out[0, 0] == 0.0

    def test_intersection_takes_min(self):
        a = np.full((10, 10), 0.6, dtype=np.float32)
        b = np.full((10, 10), 0.4, dtype=np.float32)
        out = combine_masks([a, b], mode="intersection")
        assert np.allclose(out, 0.4)

    def test_average_takes_mean(self):
        a = np.full((10, 10), 0.2, dtype=np.float32)
        b = np.full((10, 10), 0.6, dtype=np.float32)
        out = combine_masks([a, b], mode="average")
        assert np.allclose(out, 0.4)

    def test_three_mask_average(self):
        a = np.full((4, 4), 0.0, dtype=np.float32)
        b = np.full((4, 4), 0.5, dtype=np.float32)
        c = np.full((4, 4), 1.0, dtype=np.float32)
        out = combine_masks([a, b, c], mode="average")
        assert np.allclose(out, 0.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            combine_masks([], mode="union")

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            combine_masks([np.zeros((4, 4), dtype=np.float32)], mode="bogus")


class TestFeatherMask:
    def test_larger_radius_smooths_more(self):
        m = np.zeros((30, 30), dtype=np.float32)
        m[10:20, 10:20] = 1.0
        soft = feather_mask(m, radius=5)
        hard = feather_mask(m, radius=1)
        assert soft.max() < hard.max()
        assert soft.shape == m.shape
        assert soft.dtype == np.float32

    def test_preserves_zero_region(self):
        m = np.zeros((20, 20), dtype=np.float32)
        m[5:15, 5:15] = 1.0
        out = feather_mask(m, radius=3)
        assert out[0, 0] == 0.0
        assert out[-1, -1] == 0.0

    def test_output_clipped_to_unit_range(self):
        m = np.zeros((20, 20), dtype=np.float32)
        m[8:12, 8:12] = 1.0
        out = feather_mask(m, radius=2)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestThresholdMask:
    def test_keeps_values_in_band(self):
        m = np.array(
            [[0.1, 0.3, 0.5], [0.6, 0.8, 0.95]],
            dtype=np.float32,
        )
        out = threshold_mask(m, low=0.3, high=0.8)
        assert out[0, 0] == 0.0
        assert out[0, 1] == pytest.approx(0.3)
        assert out[1, 2] == 0.0

    def test_preserves_shape_and_dtype(self):
        m = np.full((8, 8), 0.5, dtype=np.float32)
        out = threshold_mask(m, low=0.2, high=0.8)
        assert out.shape == m.shape
        assert out.dtype == np.float32

    def test_default_high_is_one(self):
        m = np.array([[0.5, 0.9]], dtype=np.float32)
        out = threshold_mask(m, low=0.4)
        assert out[0, 0] == pytest.approx(0.5)
        assert out[0, 1] == pytest.approx(0.9)


class TestDilateErode:
    def test_dilate_grows_mask(self):
        m = np.zeros((20, 20), dtype=np.float32)
        m[8:12, 8:12] = 1.0
        out = dilate_mask(m, iterations=1)
        assert out.sum() >= m.sum()
        assert out.shape == m.shape
        assert out.dtype == np.float32

    def test_erode_shrinks_mask(self):
        m = np.zeros((20, 20), dtype=np.float32)
        m[8:12, 8:12] = 1.0
        out = erode_mask(m, iterations=1)
        assert out.sum() <= m.sum()
        assert out.shape == m.shape
        assert out.dtype == np.float32

    def test_dilate_in_unit_range(self):
        m = np.zeros((20, 20), dtype=np.float32)
        m[8:12, 8:12] = 1.0
        out = dilate_mask(m, iterations=2)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_erode_in_unit_range(self):
        m = np.zeros((20, 20), dtype=np.float32)
        m[8:12, 8:12] = 1.0
        out = erode_mask(m, iterations=2)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


class TestBlendModes:
    @pytest.fixture
    def img(self) -> np.ndarray:
        return np.full((10, 10, 3), 100, dtype=np.uint8)

    @pytest.fixture
    def op_img(self) -> np.ndarray:
        return np.full((10, 10, 3), 200, dtype=np.uint8)

    def _all_blend_modes(self):
        return {
            "normal": _blend_normal,
            "multiply": _blend_multiply,
            "screen": _blend_screen,
            "soft_light": _blend_soft_light,
            "overlay": _blend_overlay,
        }

    def test_all_blend_modes_run_on_synthetic(self, img, op_img):
        for mode, fn in self._all_blend_modes().items():
            out = fn(img, op_img)
            assert out.shape == img.shape
            assert out.dtype == np.uint8
            assert out.min() >= 0
            assert out.max() <= 255

    def test_normal_returns_op(self, img, op_img):
        assert np.array_equal(_blend_normal(img, op_img), op_img)

    def test_multiply_darkens(self):
        a = np.full((4, 4, 3), 200, dtype=np.uint8)
        b = np.full((4, 4, 3), 200, dtype=np.uint8)
        out = _blend_multiply(a, b)
        assert out.max() < 200

    def test_screen_lightens(self):
        a = np.full((4, 4, 3), 50, dtype=np.uint8)
        b = np.full((4, 4, 3), 50, dtype=np.uint8)
        out = _blend_screen(a, b)
        assert out.min() > 50

    def test_overlay_split_at_midpoint(self):
        dark = np.full((4, 4, 3), 50, dtype=np.uint8)
        bright = np.full((4, 4, 3), 200, dtype=np.uint8)
        layer = np.full((4, 4, 3), 200, dtype=np.uint8)
        out_dark = _blend_overlay(dark, layer)
        out_bright = _blend_overlay(bright, layer)
        assert out_dark.max() < 200
        assert out_bright.min() > 200


class TestFaceRegions:
    def test_no_faces_all_attrs_none(self):
        regions = FaceRegions()
        for slot in FaceRegions.__slots__:
            assert getattr(regions, slot) is None

    def test_regions_are_float32_after_normalization(self):
        regions = FaceRegions()
        mask_uint8 = np.full((20, 20), 180, dtype=np.uint8)
        regions.skin = mask_uint8
        normalized = normalize_mask(regions.skin)
        assert normalized.dtype == np.float32
        assert normalized.min() >= 0.0
        assert normalized.max() <= 1.0
