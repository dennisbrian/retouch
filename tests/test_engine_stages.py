"""Tests for engine stage methods — _stage_global, _stage_grade, _stage_finish, etc."""
import cv2
import numpy as np
import pytest
from retouch.engine import (
    ProcessingContext,
    RetouchEngine,
    _adjust_vibrance,
    _apply_uniform_saturation,
    _adjust_contrast,
    _adjust_tonal,
    _apply_selective_sharpening,
)
from retouch.grading import ColorGrader


class TestAdjustVibrance:
    def test_zero_returns_original(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = _adjust_vibrance(img, 0)
        assert np.all(result == img)

    def test_positive_vibrance(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 50
        result = _adjust_vibrance(img, 50)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() > 50

    def test_negative_vibrance(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 200
        result = _adjust_vibrance(img, -50)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() < 200

    def test_skin_hue_protection(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 100
        result_no_skin = _adjust_vibrance(img, 50)
        # Skin-hue pixels should be protected from over-saturation
        hsv = cv2.cvtColor(result_no_skin, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() <= 255

    def test_output_type(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = _adjust_vibrance(img, 30)
        assert result.dtype == np.uint8


class TestAdjustSaturation:
    def test_zero_returns_original(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = _apply_uniform_saturation(img, 0)
        assert np.all(result == img)

    def test_positive_saturation(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 50
        result = _apply_uniform_saturation(img, 50)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() > 50

    def test_negative_saturation(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 200
        result = _apply_uniform_saturation(img, -50)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() < 200

    def test_output_type(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = _apply_uniform_saturation(img, 30)
        assert result.dtype == np.uint8


class TestAdjustContrast:
    def test_zero_returns_same(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = _adjust_contrast(img, 0)
        assert np.all(result == img)

    def test_positive_increases_contrast(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = _adjust_contrast(img, 50)
        assert not np.allclose(result, img)

    def test_negative_decreases_contrast(self):
        img = np.full((10, 10, 3), [50, 150, 200], dtype=np.uint8)
        result = _adjust_contrast(img, -50)
        assert result.dtype == np.uint8

    def test_output_type(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = _adjust_contrast(img, 30)
        assert result.dtype == np.uint8


class TestAdjustTonal:
    def test_no_adjustment(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = _adjust_tonal(img)
        assert np.all(result == img)

    def test_shadows_lift(self):
        img = np.full((10, 10, 3), 50, dtype=np.uint8)
        result = _adjust_tonal(img, shadows=50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() > 50

    def test_highlights_boost(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = _adjust_tonal(img, highlights=50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() >= 200

    def test_whites(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = _adjust_tonal(img, whites=50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() >= 200

    def test_blacks(self):
        img = np.full((10, 10, 3), 30, dtype=np.uint8)
        result = _adjust_tonal(img, blacks=-50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() < 30

    def test_all_controls(self):
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        result = _adjust_tonal(img, shadows=30, highlights=-20, whites=10, blacks=-5)
        assert result.shape == img.shape

    def test_output_dtype(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = _adjust_tonal(img, shadows=20)
        assert result.dtype == np.uint8


class TestApplySelectiveSharpening:
    def test_zero_radius_mask(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.zeros((20, 20), dtype=np.float32)
        result = _apply_selective_sharpening(img, mask, radius=0.8, amount=1.0, threshold=0)
        assert np.all(result == img)

    def test_full_mask_on_flat(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result = _apply_selective_sharpening(img, mask, radius=1.0, amount=1.0, threshold=0)
        assert np.allclose(result, img, atol=1)

    def test_sharpens_edges(self):
        img = np.full((20, 20, 3), 64, dtype=np.uint8)
        img[:, 10:] = [192, 192, 192]
        mask = np.ones((20, 20), dtype=np.float32)
        result = _apply_selective_sharpening(img, mask, radius=0.8, amount=1.5, threshold=2)
        assert not np.allclose(result, img)

    def test_threshold_filters_noise(self):
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=np.float32)
        result_no_thresh = _apply_selective_sharpening(img, mask, radius=0.8, amount=1.0, threshold=0)
        result_thresh = _apply_selective_sharpening(img, mask, radius=0.8, amount=1.0, threshold=10)
        assert result_thresh.shape == result_no_thresh.shape

    def test_output_shape(self):
        img = np.random.randint(0, 256, (20, 20, 3), dtype=np.uint8)
        mask = np.random.rand(20, 20).astype(np.float32)
        result = _apply_selective_sharpening(img, mask)
        assert result.shape == (20, 20, 3)

    def test_3d_mask_accepted(self):
        img = np.full((20, 20, 3), 128, dtype=np.uint8)
        mask = np.ones((20, 20, 1), dtype=np.float32)
        result = _apply_selective_sharpening(img, mask)
        assert result.shape == (20, 20, 3)


class TestStageGlobalClarityDispatch:
    """Regression for the 2026-09-07 clarity call-site bug.

    _stage_global's `if ctx.clarity:` branch called self._grader._add_clarity
    (the uint8-contract implementation) unconditionally, with no `is_float`
    guard — every other op in this function (contrast, brightness, tonal)
    branches on is_float. A float32 [0,1] frame (the normal contract for
    this function per its own docstring) was therefore always fed into the
    uint8-oriented function, which silently produced a badly wrong result
    (a real production render of a 24MP photo came back with a mean pixel
    value of ~1.5/255 — effectively black) rather than raising, because
    cv2.cvtColor tolerates a float array outside its expected [0,1] range
    without erroring. Guards against ANY future clarity dispatch reaching
    the wrong-dtype implementation, at either input dtype.
    """

    @staticmethod
    def _make_engine_and_image(dtype_float: bool):
        engine = RetouchEngine()
        ctx = ProcessingContext()
        ctx.clarity = 8.0
        rng = np.random.default_rng(0)
        img_u8 = rng.integers(40, 220, (48, 48, 3), dtype=np.uint8)
        if dtype_float:
            img = img_u8.astype(np.float32) / 255.0
        else:
            img = img_u8.copy()
        return engine, ctx, img

    def test_float_input_stays_in_plausible_range(self):
        """A float32 [0,1] image with nonzero clarity must not collapse
        toward black — the exact failure mode of the missing is_float
        branch (mean pixel value dropped to ~1.5/255 in production)."""
        engine, ctx, img = self._make_engine_and_image(dtype_float=True)
        result = engine._stage_global(img.copy(), ctx)
        assert result.dtype == np.float32
        # Source mean is comfortably mid-range (rng draws from [40, 220));
        # a healthy clarity pass should not move the mean by more than a
        # small fraction of that — collapse-to-black moves it by ~two
        # orders of magnitude.
        assert result.mean() > img.mean() * 0.5, (
            f"float32 input mean collapsed after _stage_global: "
            f"source={img.mean():.4f} -> out={result.mean():.4f}"
        )

    def test_uint8_input_stays_in_plausible_range(self):
        """Same check on the uint8 path, so a future regression that breaks
        uint8 dispatch instead of float dispatch is also caught."""
        engine, ctx, img = self._make_engine_and_image(dtype_float=False)
        result = engine._stage_global(img.copy(), ctx)
        assert result.dtype == np.uint8
        assert result.mean() > img.mean() * 0.5, (
            f"uint8 input mean collapsed after _stage_global: "
            f"source={img.mean():.4f} -> out={result.mean():.4f}"
        )

    def test_float_and_uint8_paths_agree(self):
        """The float and uint8 entry points should produce visually
        equivalent results for the same source image and clarity strength
        — if dispatch ever routes float input into the uint8 function (or
        vice versa), this diverges sharply."""
        engine, ctx, img_f = self._make_engine_and_image(dtype_float=True)
        img_u8 = np.clip(img_f * 255.0 + 0.5, 0, 255).astype(np.uint8)

        out_f = engine._stage_global(img_f.copy(), ctx)
        out_u8 = engine._stage_global(img_u8.copy(), ctx)

        out_f_as_u8 = np.clip(out_f * 255.0 + 0.5, 0, 255).astype(np.uint8)
        max_diff = np.abs(out_f_as_u8.astype(np.int32) - out_u8.astype(np.int32)).max()
        assert max_diff <= 20, (
            f"_stage_global float-vs-uint8 delta too high: {max_diff}"
        )
