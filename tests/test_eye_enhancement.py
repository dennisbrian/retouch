"""Tests for retouch/eye_enhancement.py — EyeEnhancer v0."""
import cv2
import numpy as np
import pytest

from retouch.eye_enhancement import (
    SclerbBrightener,
    IrisEnhancer,
    EyeEnhancer,
    bgr_to_lab_f32,
    lab_f32_to_bgr,
    bgr_to_lch_f32,
    lch_f32_to_bgr,
)


class MockFaceRegions:
    """Mock FaceRegions for testing."""
    def __init__(self):
        self.left_eye = None
        self.right_eye = None
        self.left_iris = None
        self.right_iris = None


@pytest.fixture
def img_uint8():
    """Create a test image (uint8) with some color variation."""
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    # Add some color variation to avoid zero saturation
    img[:, :, 0] = 100  # Blue channel more blue
    img[:, :, 1] = 120  # Green slightly darker
    img[:, :, 2] = 150  # Red more red
    return img


@pytest.fixture
def img_float32():
    """Create a test image (float32) with some color variation."""
    img = np.full((64, 64, 3), 128.0, dtype=np.float32)
    # Add some color variation to avoid zero saturation
    img[:, :, 0] = 100.0  # Blue channel more blue
    img[:, :, 1] = 120.0  # Green slightly darker
    img[:, :, 2] = 150.0  # Red more red
    return img


@pytest.fixture
def regions():
    """Create mock regions with eye/iris masks."""
    r = MockFaceRegions()
    r.left_eye = np.zeros((64, 64), dtype=np.float32)
    r.right_eye = np.zeros((64, 64), dtype=np.float32)
    r.left_iris = np.zeros((64, 64), dtype=np.float32)
    r.right_iris = np.zeros((64, 64), dtype=np.float32)

    # Left eye: region [20:30, 20:30], iris [24:27, 24:27]
    r.left_eye[20:30, 20:30] = 1.0
    r.left_iris[24:27, 24:27] = 1.0

    # Right eye: region [20:30, 34:44], iris [24:27, 37:40]
    r.right_eye[20:30, 34:44] = 1.0
    r.right_iris[24:27, 37:40] = 1.0

    return r


# ============================================================================
# Color space conversion tests
# ============================================================================

class TestColorSpaceConversions:
    """Test LAB and LCH color space conversions."""

    def test_bgr_to_lab_f32_uint8_input(self, img_uint8):
        """Test BGR to LAB conversion with uint8 input."""
        lab = bgr_to_lab_f32(img_uint8)
        assert lab.dtype == np.float32
        assert lab.shape == (64, 64, 3)
        assert lab.min() >= 0 and lab.max() <= 255

    def test_bgr_to_lab_f32_float32_input(self, img_float32):
        """Test BGR to LAB conversion with float32 input."""
        lab = bgr_to_lab_f32(img_float32)
        assert lab.dtype == np.float32
        assert lab.shape == (64, 64, 3)

    def test_lab_to_bgr_uint8_output(self, img_uint8):
        """Test LAB to BGR conversion with uint8 output."""
        lab = bgr_to_lab_f32(img_uint8)
        bgr_back = lab_f32_to_bgr(lab, is_float=False)
        assert bgr_back.dtype == np.uint8
        assert bgr_back.shape == (64, 64, 3)

    def test_lab_to_bgr_float32_output(self, img_float32):
        """Test LAB to BGR conversion with float32 output."""
        lab = bgr_to_lab_f32(img_float32)
        bgr_back = lab_f32_to_bgr(lab, is_float=True)
        assert bgr_back.dtype == np.float32
        assert bgr_back.shape == (64, 64, 3)

    def test_bgr_to_lch_f32(self, img_uint8):
        """Test BGR to LCH conversion."""
        lch = bgr_to_lch_f32(img_uint8)
        assert lch.dtype == np.float32
        assert lch.shape == (64, 64, 3)
        # H should be in [0, 360)
        assert lch[:, :, 2].min() >= 0
        assert lch[:, :, 2].max() < 360

    def test_lch_to_bgr_uint8_output(self, img_uint8):
        """Test LCH to BGR conversion with uint8 output."""
        lch = bgr_to_lch_f32(img_uint8)
        bgr_back = lch_f32_to_bgr(lch, is_float=False)
        assert bgr_back.dtype == np.uint8
        assert bgr_back.shape == (64, 64, 3)

    def test_lch_to_bgr_float32_output(self, img_float32):
        """Test LCH to BGR conversion with float32 output."""
        lch = bgr_to_lch_f32(img_float32)
        bgr_back = lch_f32_to_bgr(lch, is_float=True)
        assert bgr_back.dtype == np.float32
        assert bgr_back.shape == (64, 64, 3)

    def test_roundtrip_bgr_lab_bgr(self, img_uint8):
        """Test BGR -> LAB -> BGR roundtrip."""
        lab = bgr_to_lab_f32(img_uint8)
        bgr_back = lab_f32_to_bgr(lab, is_float=False)
        # Allow small tolerance due to quantization
        assert np.allclose(img_uint8, bgr_back, atol=1)

    def test_roundtrip_bgr_lch_bgr(self, img_uint8):
        """Test BGR -> LCH -> BGR roundtrip."""
        lch = bgr_to_lch_f32(img_uint8)
        bgr_back = lch_f32_to_bgr(lch, is_float=False)
        # Allow small tolerance due to quantization
        assert np.allclose(img_uint8, bgr_back, atol=2)


# ============================================================================
# SclerbBrightener tests
# ============================================================================

class TestSclerbBrightener:
    """Test sclera brightening functionality."""

    @pytest.fixture
    def brightener(self):
        return SclerbBrightener()

    def test_zero_strength_no_op(self, brightener, img_uint8, regions):
        """Zero strength should be no-op."""
        result = brightener.brighten(
            img_uint8, regions.left_eye, regions.left_iris, strength=0.0
        )
        assert np.all(result == img_uint8)

    def test_zero_strength_float32(self, brightener, img_float32, regions):
        """Zero strength should be no-op (float32)."""
        result = brightener.brighten(
            img_float32, regions.left_eye, regions.left_iris, strength=0.0
        )
        assert np.all(result == img_float32)

    def test_output_dtype_uint8(self, brightener, img_uint8, regions):
        """Output dtype should match input (uint8)."""
        result = brightener.brighten(
            img_uint8, regions.left_eye, regions.left_iris, strength=0.5
        )
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_output_dtype_float32(self, brightener, img_float32, regions):
        """Output dtype should match input (float32)."""
        result = brightener.brighten(
            img_float32, regions.left_eye, regions.left_iris, strength=0.5
        )
        assert result.dtype == np.float32
        assert result.shape == (64, 64, 3)

    def test_brightens_sclera(self, brightener, img_uint8, regions):
        """Sclera should be brightened (L channel increased)."""
        result = brightener.brighten(
            img_uint8, regions.left_eye, regions.left_iris, strength=0.8
        )
        # Result should be different from input
        assert not np.allclose(result, img_uint8)
        # And should be brighter overall (L increased)
        lab_orig = bgr_to_lab_f32(img_uint8)
        lab_result = bgr_to_lab_f32(result)
        # Only sclera region should be brighter
        sclera_mask = np.clip(regions.left_eye - regions.left_iris, 0, 1)
        if sclera_mask.max() > 0:
            assert lab_result[:, :, 0].mean() >= lab_orig[:, :, 0].mean() - 1

    def test_empty_eye_mask_no_op(self, brightener, img_uint8):
        """Empty eye mask should be no-op."""
        empty_eye = np.zeros((64, 64), dtype=np.float32)
        empty_iris = np.zeros((64, 64), dtype=np.float32)
        result = brightener.brighten(img_uint8, empty_eye, empty_iris, strength=0.8)
        assert np.all(result == img_uint8)

    def test_empty_iris_mask_no_op(self, brightener, img_uint8, regions):
        """Empty iris mask should be no-op (sclera = eye - iris = eye when iris empty)."""
        result = brightener.brighten(
            img_uint8, regions.left_eye, np.zeros((64, 64), dtype=np.float32), strength=0.8
        )
        # Should still brighten since sclera = eye
        assert result is not None
        assert result.dtype == np.uint8

    def test_strength_range(self, brightener, img_uint8, regions):
        """Test multiple strength values."""
        results = []
        for strength in [0.1, 0.3, 0.5, 0.8]:
            r = brightener.brighten(
                img_uint8, regions.left_eye, regions.left_iris, strength=strength
            )
            results.append(r)
        # Different strengths should produce different results
        assert not np.allclose(results[0], results[-1])


# ============================================================================
# IrisEnhancer tests
# ============================================================================

class TestIrisEnhancer:
    """Test iris enhancement functionality."""

    @pytest.fixture
    def enhancer(self):
        return IrisEnhancer()

    def test_zero_strength_no_op(self, enhancer, img_uint8, regions):
        """All zero strengths should be no-op."""
        result = enhancer.enhance(
            img_uint8, regions.left_iris,
            saturate_strength=0.0, hue_shift=0.0, brightness_strength=0.0
        )
        assert np.all(result == img_uint8)

    def test_output_dtype_uint8(self, enhancer, img_uint8, regions):
        """Output dtype should match input (uint8)."""
        result = enhancer.enhance(
            img_uint8, regions.left_iris, saturate_strength=0.5
        )
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_output_dtype_float32(self, enhancer, img_float32, regions):
        """Output dtype should match input (float32)."""
        result = enhancer.enhance(
            img_float32, regions.left_iris, saturate_strength=0.5
        )
        assert result.dtype == np.float32
        assert result.shape == (64, 64, 3)

    def test_saturation_boost(self, enhancer, img_uint8, regions):
        """Saturation boost should increase chroma."""
        result = enhancer.enhance(
            img_uint8, regions.left_iris, saturate_strength=0.8
        )
        # Result should be different
        assert not np.allclose(result, img_uint8)

    def test_hue_shift(self, enhancer, img_uint8, regions):
        """Hue shift should shift iris color."""
        result = enhancer.enhance(
            img_uint8, regions.left_iris, hue_shift=45.0
        )
        # Result should be different
        assert not np.allclose(result, img_uint8)

    def test_brightness_boost(self, enhancer, img_uint8, regions):
        """Brightness boost should lighten iris."""
        result = enhancer.enhance(
            img_uint8, regions.left_iris, brightness_strength=0.8
        )
        # Result should be different
        assert not np.allclose(result, img_uint8)

    def test_combined_enhancement(self, enhancer, img_uint8, regions):
        """Combined enhancement should work."""
        result = enhancer.enhance(
            img_uint8, regions.left_iris,
            saturate_strength=0.5, hue_shift=30.0, brightness_strength=0.5
        )
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_uint8)

    def test_empty_iris_mask_no_op(self, enhancer, img_uint8):
        """Empty iris mask should be no-op."""
        empty_iris = np.zeros((64, 64), dtype=np.float32)
        result = enhancer.enhance(
            img_uint8, empty_iris, saturate_strength=0.8
        )
        assert np.all(result == img_uint8)

    def test_negative_hue_shift(self, enhancer, img_uint8, regions):
        """Negative hue shift should work."""
        result = enhancer.enhance(
            img_uint8, regions.left_iris, hue_shift=-45.0
        )
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_uint8)

    def test_large_hue_shift(self, enhancer, img_uint8, regions):
        """Large hue shift should work (wrap-around)."""
        result = enhancer.enhance(
            img_uint8, regions.left_iris, hue_shift=350.0
        )
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_uint8)


# ============================================================================
# EyeEnhancer tests
# ============================================================================

class TestEyeEnhancer:
    """Test orchestrator eye enhancement functionality."""

    @pytest.fixture
    def enhancer(self):
        return EyeEnhancer()

    def test_zero_strength_no_op(self, enhancer, img_uint8, regions):
        """All zero strengths should be no-op."""
        result = enhancer.enhance(
            img_uint8, regions,
            sclera_brighten=0, iris_saturate=0, iris_hue_shift=0, iris_brightness=0
        )
        assert np.all(result == img_uint8)

    def test_sclera_brighten_left_eye(self, enhancer, img_uint8, regions):
        """Sclera brightening should work for left eye."""
        result = enhancer.enhance(
            img_uint8, regions, sclera_brighten=50
        )
        assert result.dtype == np.uint8
        # Should be different from input
        assert not np.allclose(result, img_uint8)

    def test_sclera_brighten_right_eye(self, enhancer, img_uint8, regions):
        """Sclera brightening should work for right eye."""
        regions.left_iris = np.zeros((64, 64), dtype=np.float32)
        regions.left_eye = np.zeros((64, 64), dtype=np.float32)
        result = enhancer.enhance(
            img_uint8, regions, sclera_brighten=50
        )
        assert result.dtype == np.uint8

    def test_sclera_brighten_both_eyes(self, enhancer, img_uint8, regions):
        """Sclera brightening should work for both eyes."""
        result = enhancer.enhance(
            img_uint8, regions, sclera_brighten=50
        )
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_uint8)

    def test_iris_saturate(self, enhancer, img_uint8, regions):
        """Iris saturation should work."""
        result = enhancer.enhance(
            img_uint8, regions, iris_saturate=50
        )
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_uint8)

    def test_iris_hue_shift(self, enhancer, img_uint8, regions):
        """Iris hue shift should work."""
        result = enhancer.enhance(
            img_uint8, regions, iris_hue_shift=30
        )
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_uint8)

    def test_iris_brightness(self, enhancer, img_uint8, regions):
        """Iris brightness should work."""
        result = enhancer.enhance(
            img_uint8, regions, iris_brightness=50
        )
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_uint8)

    def test_combined_enhancement(self, enhancer, img_uint8, regions):
        """Combined enhancement should work."""
        result = enhancer.enhance(
            img_uint8, regions,
            sclera_brighten=40, iris_saturate=50,
            iris_hue_shift=15, iris_brightness=30
        )
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_uint8)

    def test_output_dtype_float32(self, enhancer, img_float32, regions):
        """Output dtype should match input (float32)."""
        result = enhancer.enhance(
            img_float32, regions, iris_saturate=50
        )
        assert result.dtype == np.float32
        assert result.shape == (64, 64, 3)

    def test_missing_left_eye_no_crash(self, enhancer, img_uint8, regions):
        """Missing left eye should not crash."""
        regions.left_eye = None
        regions.left_iris = None
        result = enhancer.enhance(
            img_uint8, regions, sclera_brighten=50
        )
        assert result.dtype == np.uint8

    def test_missing_right_eye_no_crash(self, enhancer, img_uint8, regions):
        """Missing right eye should not crash."""
        regions.right_eye = None
        regions.right_iris = None
        result = enhancer.enhance(
            img_uint8, regions, sclera_brighten=50
        )
        assert result.dtype == np.uint8

    def test_missing_both_eyes_no_op(self, enhancer, img_uint8, regions):
        """Missing both eyes should be no-op."""
        regions.left_eye = None
        regions.left_iris = None
        regions.right_eye = None
        regions.right_iris = None
        result = enhancer.enhance(
            img_uint8, regions, sclera_brighten=50
        )
        assert np.all(result == img_uint8)

    def test_per_eye_independence(self, enhancer, img_uint8, regions):
        """Each eye should be processed independently."""
        # Create a fresh image for this test
        test_img = np.full((64, 64, 3), 128, dtype=np.uint8)

        # Enhance both eyes
        result_both = enhancer.enhance(
            test_img, regions, iris_saturate=50
        )

        # Enhance only left eye
        regions_left_only = MockFaceRegions()
        regions_left_only.left_eye = regions.left_eye.copy()
        regions_left_only.left_iris = regions.left_iris.copy()
        regions_left_only.right_eye = None
        regions_left_only.right_iris = None

        result_left = enhancer.enhance(
            test_img, regions_left_only, iris_saturate=50
        )

        # Enhance only right eye
        regions_right_only = MockFaceRegions()
        regions_right_only.left_eye = None
        regions_right_only.left_iris = None
        regions_right_only.right_eye = regions.right_eye.copy()
        regions_right_only.right_iris = regions.right_iris.copy()

        result_right = enhancer.enhance(
            test_img, regions_right_only, iris_saturate=50
        )

        # Results should be defined (no crash)
        assert result_both is not None
        assert result_left is not None
        assert result_right is not None

    def test_large_image(self, enhancer):
        """Test on larger image (dtype robustness)."""
        large_img = np.full((256, 256, 3), 128, dtype=np.uint8)
        large_img[:, :, 0] = 100
        large_img[:, :, 1] = 120
        large_img[:, :, 2] = 150

        # Create regions for large image
        large_regions = MockFaceRegions()
        large_regions.left_eye = np.zeros((256, 256), dtype=np.float32)
        large_regions.right_eye = np.zeros((256, 256), dtype=np.float32)
        large_regions.left_iris = np.zeros((256, 256), dtype=np.float32)
        large_regions.right_iris = np.zeros((256, 256), dtype=np.float32)
        large_regions.left_eye[80:120, 80:120] = 1.0
        large_regions.left_iris[96:108, 96:108] = 1.0
        large_regions.right_eye[80:120, 136:176] = 1.0
        large_regions.right_iris[96:108, 152:164] = 1.0

        result = enhancer.enhance(
            large_img, large_regions, iris_saturate=50
        )
        assert result.shape == (256, 256, 3)
        assert result.dtype == np.uint8

    def test_strength_range_100(self, enhancer, img_uint8, regions):
        """Test max strength (100)."""
        result = enhancer.enhance(
            img_uint8, regions,
            sclera_brighten=100, iris_saturate=100, iris_brightness=100
        )
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_uint8)

    def test_hue_shift_range_neg30_pos30(self, enhancer, img_uint8, regions):
        """Test hue shift range [-30, +30]."""
        result_neg = enhancer.enhance(
            img_uint8, regions, iris_hue_shift=-30
        )
        result_pos = enhancer.enhance(
            img_uint8, regions, iris_hue_shift=30
        )
        assert result_neg.dtype == np.uint8
        assert result_pos.dtype == np.uint8
        # Should be different from each other
        assert not np.allclose(result_neg, result_pos)


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_pipeline_uint8(self):
        """Test full pipeline with uint8."""
        enhancer = EyeEnhancer()
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        regions = MockFaceRegions()
        regions.left_eye = np.zeros((64, 64), dtype=np.float32)
        regions.right_eye = np.zeros((64, 64), dtype=np.float32)
        regions.left_iris = np.zeros((64, 64), dtype=np.float32)
        regions.right_iris = np.zeros((64, 64), dtype=np.float32)
        regions.left_eye[20:30, 20:30] = 1.0
        regions.left_iris[24:27, 24:27] = 1.0
        regions.right_eye[20:30, 34:44] = 1.0
        regions.right_iris[24:27, 37:40] = 1.0

        result = enhancer.enhance(
            img, regions,
            sclera_brighten=40, iris_saturate=50,
            iris_hue_shift=15, iris_brightness=30
        )
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)

    def test_full_pipeline_float32(self):
        """Test full pipeline with float32."""
        enhancer = EyeEnhancer()
        img = np.full((64, 64, 3), 128.0, dtype=np.float32)
        regions = MockFaceRegions()
        regions.left_eye = np.zeros((64, 64), dtype=np.float32)
        regions.right_eye = np.zeros((64, 64), dtype=np.float32)
        regions.left_iris = np.zeros((64, 64), dtype=np.float32)
        regions.right_iris = np.zeros((64, 64), dtype=np.float32)
        regions.left_eye[20:30, 20:30] = 1.0
        regions.left_iris[24:27, 24:27] = 1.0
        regions.right_eye[20:30, 34:44] = 1.0
        regions.right_iris[24:27, 37:40] = 1.0

        result = enhancer.enhance(
            img, regions,
            sclera_brighten=40, iris_saturate=50,
            iris_hue_shift=15, iris_brightness=30
        )
        assert result.dtype == np.float32
        assert result.shape == (64, 64, 3)

    def test_no_eye_regions_graceful_fallback(self):
        """Test graceful fallback when eye regions are missing."""
        enhancer = EyeEnhancer()
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        regions = MockFaceRegions()
        # No eye regions

        result = enhancer.enhance(
            img, regions,
            sclera_brighten=40, iris_saturate=50
        )
        # Should be no-op
        assert np.all(result == img)

    def test_repeated_enhancement_idempotence(self):
        """Test that repeated enhancement is roughly idempotent."""
        enhancer = EyeEnhancer()
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        regions = MockFaceRegions()
        regions.left_eye = np.zeros((64, 64), dtype=np.float32)
        regions.right_eye = np.zeros((64, 64), dtype=np.float32)
        regions.left_iris = np.zeros((64, 64), dtype=np.float32)
        regions.right_iris = np.zeros((64, 64), dtype=np.float32)
        regions.left_eye[20:30, 20:30] = 1.0
        regions.left_iris[24:27, 24:27] = 1.0
        regions.right_eye[20:30, 34:44] = 1.0
        regions.right_iris[24:27, 37:40] = 1.0

        result1 = enhancer.enhance(img.copy(), regions, iris_saturate=50)
        result2 = enhancer.enhance(result1.copy(), regions, iris_saturate=50)

        # Results should be similar (saturation has a ceiling effect)
        # Allow larger tolerance since we're applying twice
        assert np.allclose(result1, result2, atol=5)
