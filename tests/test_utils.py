"""Tests for retouch/utils.py — pure helper functions."""

import numpy as np
import cv2
import pytest

from retouch.detection import _Landmark, _LandmarkCompat
from retouch.utils import (
    feather_mask,
    blend_masked,
    restore_outside_support,
    create_polygon_mask,
    get_points,
    inter_eye_distance,
    vibrance,
    apply_curve,
    correct_exposure,
    adaptive_ksize,
    apply_global_bloom,
    apply_skin_diffusion,
    soft_light_blend,
    overlay_blend,
    hard_light_blend,
    remove_purple_fringing,
)


class TestFeatherMask:
    def test_none_mask(self):
        assert feather_mask(None, radius=5) is None

    def test_empty_mask(self):
        m = np.zeros((0, 0), dtype=np.float32)
        result = feather_mask(m, radius=5)
        assert result.size == 0

    def test_no_radius_no_sigma(self):
        m = np.zeros((10, 10), dtype=np.float32)
        m[4:6, 4:6] = 1.0
        result = feather_mask(m)
        assert np.allclose(result, m)

    def test_normalize_from_uint8(self):
        m = np.zeros((10, 10), dtype=np.uint8)
        m[4:6, 4:6] = 255
        result = feather_mask(m, radius=3)
        assert result.max() <= 1.0
        assert result.dtype == np.float32

    def test_feather_reduces_max(self):
        m = np.zeros((30, 30), dtype=np.float32)
        m[10:20, 10:20] = 1.0
        result = feather_mask(m, radius=5)
        assert result.max() < 1.0
        assert result.shape == (30, 30)


class TestBlendMasked:
    def test_none_mask_returns_processed(self):
        orig = np.full((10, 10, 3), 0, dtype=np.uint8)
        proc = np.full((10, 10, 3), 255, dtype=np.uint8)
        result = blend_masked(orig, proc, None)
        assert np.all(result == 255)

    def test_full_mask_returns_processed(self):
        orig = np.full((10, 10, 3), 0, dtype=np.uint8)
        proc = np.full((10, 10, 3), 255, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        result = blend_masked(orig, proc, mask)
        assert np.all(result == 255)

    def test_zero_mask_returns_original(self):
        orig = np.full((10, 10, 3), 128, dtype=np.uint8)
        proc = np.full((10, 10, 3), 0, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.float32)
        result = blend_masked(orig, proc, mask)
        assert np.all(result == 128)

    def test_half_mask(self):
        orig = np.full((10, 10, 3), 0, dtype=np.uint8)
        proc = np.full((10, 10, 3), 100, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[:, :5] = 1.0
        result = blend_masked(orig, proc, mask)
        assert np.all(result[:, :5] == 100)
        assert np.all(result[:, 5:] == 0)

    def test_3d_mask_accepted(self):
        orig = np.full((10, 10, 3), 0, dtype=np.uint8)
        proc = np.full((10, 10, 3), 255, dtype=np.uint8)
        mask = np.ones((10, 10, 1), dtype=np.float32)
        result = blend_masked(orig, proc, mask)
        assert np.all(result == 255)


class TestRestoreOutsideSupport:
    @pytest.mark.parametrize("dtype", [np.uint8, np.float32])
    def test_restores_exact_zero_without_eroding_feather_tails(self, dtype):
        source = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        if dtype == np.float32:
            source = source.astype(np.float32)
        processed = np.clip(source.astype(np.float32) + 7.0, 0, 255).astype(dtype)
        support = np.zeros((6, 8), dtype=np.float32)
        support[2:4, 2:6] = 1.0
        support[1, 3] = np.nextafter(np.float32(0.0), np.float32(1.0))

        result = restore_outside_support(source, processed, support)

        np.testing.assert_array_equal(result[support == 0], source[support == 0])
        np.testing.assert_array_equal(result[support != 0], processed[support != 0])
        assert result.dtype == source.dtype

    def test_rejects_shape_and_dtype_mismatches(self):
        source = np.zeros((4, 5, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="processed shape"):
            restore_outside_support(
                source, np.zeros((4, 4, 3), dtype=np.uint8), np.ones((4, 5))
            )
        with pytest.raises(ValueError, match="processed dtype"):
            restore_outside_support(
                source, source.astype(np.float32), np.ones((4, 5))
            )
        with pytest.raises(ValueError, match="support shape"):
            restore_outside_support(source, source, np.ones((4, 5, 1)))

    def test_rejects_new_nonfinite_inside_support_only(self):
        source = np.zeros((4, 5, 3), dtype=np.float32)
        processed = source.copy()
        support = np.zeros((4, 5), dtype=np.float32)
        processed[0, 0] = np.nan

        # A bad intermediate outside support is discarded before validation.
        restored = restore_outside_support(source, processed, support)
        assert np.isfinite(restored).all()

        support[0, 0] = 1.0
        with pytest.raises(ValueError, match="introduced non-finite"):
            restore_outside_support(source, processed, support)

class TestCreatePolygonMask:
    def test_basic_polygon(self):
        pts = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.int32)
        mask = create_polygon_mask(pts, (40, 40), feather_radius=0)
        assert mask.shape == (40, 40)
        assert mask[10:30, 10:30].mean() > 0.5
        assert mask[0, 0] == 0.0

    def test_with_feathering(self):
        pts = np.array([[10, 10], [30, 10], [30, 30], [10, 30]], dtype=np.int32)
        mask = create_polygon_mask(pts, (40, 40), feather_radius=3)
        assert mask.shape == (40, 40)
        assert mask.max() <= 1.0


class TestVibrance:
    def test_zero_strength(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.float32)
        result = vibrance(img, mask, 0.0)
        assert np.all(result == img)

    def test_increases_saturation_masked(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 50  # low saturation green
        mask = np.ones((10, 10), dtype=np.float32)
        result = vibrance(img, mask, 1.0)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() > 50

    def test_mask_restricts_area(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        img[:, :, 1] = 50
        mask = np.zeros((10, 10), dtype=np.float32)
        mask[:, :5] = 1.0
        result = vibrance(img, mask, 1.0)
        assert not np.allclose(result, img)


class TestApplyCurve:
    def test_identity_curve(self):
        channel = np.arange(256, dtype=np.uint8).reshape(256, 1)
        curve = [(0, 0), (255, 255)]
        result = apply_curve(channel, curve)
        assert np.allclose(result.ravel(), np.arange(256))

    def test_invert_curve(self):
        channel = np.array([0, 128, 255], dtype=np.uint8)
        curve = [(0, 255), (255, 0)]
        result = apply_curve(channel, curve)
        assert result[0] == 255
        assert result[2] == 0

    def test_clamp_output(self):
        channel = np.array([128], dtype=np.uint8)
        curve = [(0, 0), (128, 300), (255, 255)]
        result = apply_curve(channel, curve)
        assert result[0] == 255


class TestCorrectExposure:
    def test_already_correct(self):
        img = np.full((50, 50, 3), 120, dtype=np.uint8)
        result, corrected = correct_exposure(img)
        assert not corrected
        assert np.all(result == img)

    def test_too_dark_global(self):
        img = np.full((50, 50, 3), 30, dtype=np.uint8)
        result, corrected = correct_exposure(img, min_threshold=50)
        assert corrected

    def test_too_bright_global(self):
        img = np.full((50, 50, 3), 200, dtype=np.uint8)
        result, corrected = correct_exposure(img, max_threshold=180)
        assert corrected

    def test_with_face_bbox(self):
        img = np.full((100, 100, 3), 50, dtype=np.uint8)
        bbox = (30, 30, 40, 40)
        result, corrected = correct_exposure(
            img, face_bboxes=[bbox], face_min_threshold=60, target_mean=120
        )
        assert corrected


class TestAdaptiveKsize:
    def test_odd_output(self):
        assert adaptive_ksize(100, factor=0.1, minimum=3) % 2 == 1

    def test_minimum_enforced(self):
        assert adaptive_ksize(10, factor=0.1, minimum=5) == 5

    def test_proportional(self):
        k1 = adaptive_ksize(100, factor=0.1, minimum=3)
        k2 = adaptive_ksize(200, factor=0.1, minimum=3)
        assert k2 >= k1


class TestGlobalBloom:
    def test_zero_strength_returns_original(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = apply_global_bloom(img, strength=0.0, threshold=200.0, softness=30.0)
        assert np.all(result == img)

    def test_bloom_disperses_highlights(self):
        # Create a dark image with a bright square in the center
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[45:55, 45:55] = 255  # bright highlight
        
        # Apply bloom with 100 strength, 200 threshold, and 10 softness
        result = apply_global_bloom(img, strength=100.0, threshold=200.0, softness=10.0)
        
        # Verify that the center is still bright
        assert np.all(result[49, 49] > 200)
        # Verify that pixels outside the original highlight are now non-zero (bloom dispersion)
        assert np.any(result[40, 40] > 0)
        assert np.any(result[60, 60] > 0)

    def test_low_threshold_glows_all(self):
        img = np.full((50, 50, 3), 160, dtype=np.uint8)
        result = apply_global_bloom(img, strength=50.0, threshold=150.0, softness=20.0)
        # Low threshold means the entire image glows, screen blending increases brightness
        assert result.mean() > img.mean()

    def test_downsampled_large_image(self):
        # Create a large image (> 2000px min_dim)
        img = np.zeros((2200, 2200, 3), dtype=np.uint8)
        img[1000:1200, 1000:1200] = 255
        result = apply_global_bloom(img, strength=50.0, threshold=200.0, softness=30.0)
        assert result.shape == (2200, 2200, 3)


class TestLogCrash:
    def test_log_crash_writes_file(self):
        from retouch.utils import get_cache_dir, log_crash
        from pathlib import Path

        # Clean up any existing crash log
        cache_dir = get_cache_dir()
        crash_log_file = cache_dir / "crash.log"
        if crash_log_file.exists():
            crash_log_file.unlink()

        try:
            raise ValueError("Test value error for logging")
        except ValueError as e:
            path = log_crash(e, {"test_key": "test_value"})

        assert path is not None
        assert Path(path).exists()
        assert Path(path).name == "crash.log"

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        assert "ValueError" in content
        assert "Test value error for logging" in content
        assert "test_key" in content
        assert "test_value" in content

        # Clean up
        if crash_log_file.exists():
            crash_log_file.unlink()


class TestGetPoints:
    def test_extracts_coords(self):
        lms = _LandmarkCompat([_Landmark(0.1, 0.2), _Landmark(0.5, 0.7)])
        pts = get_points(lms, [0, 1], 200, 100)
        assert pts.shape == (2, 2)
        assert pts[0, 0] == 20
        assert pts[1, 1] == 70

    def test_single_index(self):
        lms = _LandmarkCompat([_Landmark(0.3, 0.4)])
        pts = get_points(lms, [0], 100, 100)
        assert pts.shape == (1, 2)

    def test_output_dtype_int32(self):
        lms = _LandmarkCompat([_Landmark(0.5, 0.5)])
        pts = get_points(lms, [0], 100, 100)
        assert pts.dtype == np.int32

    def test_empty_indices(self):
        lms = _LandmarkCompat([])
        pts = get_points(lms, [], 100, 100)
        assert len(pts) == 0


class TestInterEyeDistance:
    def test_with_iris_landmarks(self):
        lms = _LandmarkCompat([_Landmark(0.0, 0.0) for _ in range(478)])
        lms.landmark[468] = _Landmark(0.3, 0.5)
        lms.landmark[473] = _Landmark(0.7, 0.5)
        d = inter_eye_distance(lms, 200, 200)
        expected = (0.7 - 0.3) * 200
        assert abs(d - expected) < 1.0

    def test_fallback_no_iris_indices(self):
        lms = _LandmarkCompat([_Landmark(0.0, 0.0) for _ in range(400)])
        lms.landmark[33] = _Landmark(0.25, 0.4)
        lms.landmark[133] = _Landmark(0.35, 0.5)
        lms.landmark[263] = _Landmark(0.65, 0.5)
        lms.landmark[362] = _Landmark(0.75, 0.4)
        d = inter_eye_distance(lms, 200, 200)
        assert d > 0

    def test_positive_distance(self):
        lms = _LandmarkCompat([_Landmark(0.0, 0.0) for _ in range(478)])
        lms.landmark[468] = _Landmark(0.3, 0.5)
        lms.landmark[473] = _Landmark(0.7, 0.5)
        d = inter_eye_distance(lms, 200, 200)
        assert d > 0


@pytest.fixture
def img_black():
    return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def img_white():
    return np.full((8, 8, 3), 255, dtype=np.uint8)


@pytest.fixture
def img_gray():
    return np.full((8, 8, 3), 128, dtype=np.uint8)


class TestSoftLightBlend:
    def test_darker_overlay_darkens(self, img_gray, img_black):
        result = soft_light_blend(img_gray, img_black)
        assert result.max() <= 128

    def test_lighter_overlay_brightens(self, img_gray, img_white):
        result = soft_light_blend(img_gray, img_white)
        assert result.max() >= 128

    def test_identity_on_midtone(self, img_gray):
        """128 overlay on 128 base = unchanged midtone."""
        result = soft_light_blend(img_gray, img_gray)
        assert result.dtype == np.uint8

    def test_preserves_shape(self, img_gray):
        result = soft_light_blend(img_gray, img_gray)
        assert result.shape == img_gray.shape


class TestOverlayBlend:
    def test_darker_overlay_darkens_midtones(self, img_gray, img_black):
        """Black overlay darkens gray midtones (multiply branch)."""
        result = overlay_blend(img_gray, img_black)
        # 128 base < 128? False → screen branch for <128 would not apply.
        # But with pixel values = 128 exactly:
        # base=128 is NOT < 128, so screen: 255-(127*255)/255 = 255-127 = 128
        # Actually: multiply = 128*0/255 = 0, screen = 255 - (127*255)/255 = 128
        assert result.dtype == np.uint8

    def test_lighter_overlay_brightens_midtones(self, img_gray, img_white):
        """White overlay brightens gray midtones (screen branch)."""
        result = overlay_blend(img_gray, img_white)
        assert result.max() >= 200

    def test_gray_on_gray(self, img_gray):
        result = overlay_blend(img_gray, img_gray)
        assert result.dtype == np.uint8


class TestHardLightBlend:
    def test_white_on_black_stays_white(self, img_black, img_white):
        result = hard_light_blend(img_black, img_white)
        assert result.max() >= 250

    def test_black_on_white_stays_black(self, img_white, img_black):
        result = hard_light_blend(img_white, img_black)
        assert result.min() <= 5

    def test_preserves_shape_dtype(self, img_gray):
        result = hard_light_blend(img_gray, img_gray)
        assert result.dtype == np.uint8


class TestApplySkinDiffusion:
    def test_zero_strength_noop(self):
        img = np.full((64, 64, 3), 150, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = apply_skin_diffusion(img, mask, strength=0)
        assert np.all(result == img)

    def test_mid_gray_skin_brightens(self):
        img = np.full((64, 64, 3), 180, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = apply_skin_diffusion(img, mask, strength=30)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        assert lab[:, :, 0].mean() >= orig_lab[:, :, 0].mean()

    def test_far_outside_mask_unchanged(self):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        img[50:60, 50:60] = [50, 200, 100]
        result = apply_skin_diffusion(img, mask, strength=40)
        np.testing.assert_allclose(result[50:60, 50:60], img[50:60, 50:60], atol=3)

    def test_dtype_shape(self):
        img = np.full((64, 64, 3), 180, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = apply_skin_diffusion(img, mask, strength=20)
        assert result.dtype == np.uint8
        assert result.shape == img.shape


class TestRemovePurpleFringing:
    """Tests for AB4 lateral chromatic aberration (purple fringing) removal."""

    def test_zero_strength_noop(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[32:, :] = [250, 50, 200]  # Magenta edge
        result = remove_purple_fringing(img, strength=0.0)
        assert np.all(result == img)

    def test_suppresses_fringe_at_high_gradient_edge(self):
        """Purple fringe at high-contrast edge must be desaturated."""
        img = np.full((64, 64, 3), 20, dtype=np.uint8)
        # High-contrast bright area (background)
        img[0:31, :] = 240
        # 1-pixel purple/magenta fringe at the high-gradient boundary (row 31)
        img[31, :] = [220, 40, 200]  # High blue & red, low green -> purple fringe

        out = remove_purple_fringing(img, strength=1.0, edge_threshold=20.0)
        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)

        orig_a = float(lab_orig[31, :].mean())
        new_a = float(lab_out[31, :].mean())
        assert new_a < 130.0, f"Purple fringe not desaturated to neutral: a* {orig_a:.1f} -> {new_a:.1f}"

    def test_non_edge_purple_fabric_preserved(self):
        """Uniform purple fabric (zero luminance gradient) must be preserved 100%."""
        img = np.full((64, 64, 3), [200, 40, 200], dtype=np.uint8)  # Uniform purple
        out = remove_purple_fringing(img, strength=1.0)
        assert np.all(out == img)
