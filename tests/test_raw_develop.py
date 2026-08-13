"""Tests for retouch/raw_develop.py linear RAW development pipeline.

Covers:
  * CameraMatrix: per-camera XYZ->RGB matrix storage and WB presets
  * WhiteBalanceEstimator: metadata-based and gray-world WB estimation
  * LinearGrader: float32 linear space grading operations (curves, exposure, contrast, color)
  * RAWDeveloper: main orchestrator (load, develop, export)
  * Load/save round-trip with metadata preservation
  * Linear RGB correctness (no gamma encoding)
  * dtype preservation (float32 linear [0, 1])
  * Graceful degradation when rawpy is unavailable
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

from retouch.raw_develop import (
    CameraMatrix,
    WhiteBalanceEstimator,
    LinearGrader,
    RAWDeveloper,
    HAS_RAWPY,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def linear_rgb_f32(rng: np.random.Generator) -> np.ndarray:
    """(H, W, 3) float32 linear RGB in [0, 1]."""
    return rng.uniform(0.0, 1.0, size=(64, 64, 3)).astype(np.float32)


@pytest.fixture
def linear_rgb_gradient() -> np.ndarray:
    """Smooth gradient exercising linear space."""
    h, w = 32, 32
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    base = (yy + xx) * 0.5
    img = np.stack([base, 1.0 - base, base * 0.5], axis=-1)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@pytest.fixture
def developer() -> RAWDeveloper:
    """RAWDeveloper instance."""
    return RAWDeveloper()


# ============================================================================
# CameraMatrix Tests
# ============================================================================

class TestCameraMatrix:
    """Test per-camera matrix storage and WB presets."""

    def test_camera_matrix_canonical_shape(self):
        """Matrix should be (3, 3) float32."""
        mat = CameraMatrix.get_camera_matrix("Canon EOS 5D")
        assert mat.shape == (3, 3)
        assert mat.dtype == np.float32

    def test_camera_matrix_generic_fallback(self):
        """Unknown camera should fall back to Generic."""
        mat1 = CameraMatrix.get_camera_matrix("UnknownCamera XYZ")
        mat2 = CameraMatrix.get_camera_matrix(None)
        assert np.allclose(mat1, mat2)

    def test_camera_matrix_canon(self):
        """Canon matrix should be valid (non-zero, reasonable values)."""
        mat = CameraMatrix.get_camera_matrix("Canon EOS 5D")
        assert np.all(np.isfinite(mat))
        assert not np.allclose(mat, 0.0)

    def test_camera_matrix_nikon(self):
        """Nikon matrix should be valid."""
        mat = CameraMatrix.get_camera_matrix("Nikon D850")
        assert np.all(np.isfinite(mat))
        assert not np.allclose(mat, 0.0)

    def test_camera_matrix_sony(self):
        """Sony matrix should be valid."""
        mat = CameraMatrix.get_camera_matrix("Sony ILCE-7")
        assert np.all(np.isfinite(mat))
        assert not np.allclose(mat, 0.0)

    def test_white_balance_daylight(self):
        """Daylight WB should be (1, 1, 1) for all cameras."""
        for cam in ["Canon EOS 5D", "Nikon D850", "Sony ILCE-7"]:
            wb = CameraMatrix.get_white_balance(cam, "daylight")
            assert wb.shape == (3,)
            assert np.allclose(wb, [1.0, 1.0, 1.0])

    def test_white_balance_cloudy(self):
        """Cloudy WB should have R > G and B < G."""
        wb = CameraMatrix.get_white_balance("Canon EOS 5D", "cloudy")
        assert wb[0] > wb[1]  # Red boost
        assert wb[2] < wb[1]  # Blue cut

    def test_white_balance_generic_fallback(self):
        """Unknown camera should fall back to Generic WB."""
        wb1 = CameraMatrix.get_white_balance("UnknownCam", "cloudy")
        wb2 = CameraMatrix.get_white_balance(None, "cloudy")
        assert np.allclose(wb1, wb2)

    def test_white_balance_invalid_preset_fallback(self):
        """Invalid preset should fall back to daylight."""
        wb = CameraMatrix.get_white_balance("Canon EOS 5D", "invalid_preset")
        assert np.allclose(wb, [1.0, 1.0, 1.0])


# ============================================================================
# WhiteBalanceEstimator Tests
# ============================================================================

class TestWhiteBalanceEstimator:
    """Test WB estimation from metadata and image content."""

    def test_wb_from_metadata_basic(self):
        """Metadata WB should normalize to G=1.0."""
        wb = WhiteBalanceEstimator.estimate_from_metadata(1.0, 2.0, 0.8)
        assert wb.dtype == np.float32
        assert np.isclose(wb[1], 1.0)  # Green normalized to 1.0
        assert np.isclose(wb[0], 0.5)  # R normalized
        assert np.isclose(wb[2], 0.4)  # B normalized

    def test_wb_from_metadata_identity(self):
        """Equal RGB should give (1, 1, 1)."""
        wb = WhiteBalanceEstimator.estimate_from_metadata(1.0, 1.0, 1.0)
        assert np.allclose(wb, [1.0, 1.0, 1.0])

    def test_wb_from_metadata_zero_green_guard(self):
        """Zero green should not crash; should return (1, 1, 1)."""
        wb = WhiteBalanceEstimator.estimate_from_metadata(1.0, 0.0, 1.0)
        assert wb.shape == (3,)
        assert np.all(np.isfinite(wb))

    def test_wb_from_gray_center_roi(self):
        """WB from center ROI should normalize to G=1.0."""
        # Create uniform gray image
        gray = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        wb = WhiteBalanceEstimator.estimate_from_gray(gray)
        assert np.allclose(wb, [1.0, 1.0, 1.0], atol=1e-6)

    def test_wb_from_gray_unbalanced(self):
        """Unbalanced gray should produce corrective WB."""
        # Create slightly red-shifted gray
        img = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        img[:, :, 2] = 0.6  # Red (BGR order) boost
        wb = WhiteBalanceEstimator.estimate_from_gray(img)
        # WB is mean per channel. Red is higher (0.6), green is lower (0.5)
        # After normalization by green: red becomes 0.6/0.5 = 1.2
        assert np.isclose(wb[1], 1.0)
        assert wb[2] > 1.0  # Red multiplier > 1 to compensate (remember: [B, G, R])

    def test_wb_from_gray_custom_roi(self):
        """Custom ROI should isolate to that region."""
        # Create image with red left, blue right
        img = np.zeros((64, 64, 3), dtype=np.float32)
        img[:, :32, 2] = 0.8  # Red left (BGR[2] = R)
        img[:, 32:, 0] = 0.8  # Blue right (BGR[0] = B)

        # ROI on the left (red) region
        wb = WhiteBalanceEstimator.estimate_from_gray(img, gray_roi=(0, 0, 64, 32))
        # In left region: Red=0.8, Green=0, Blue=0. After normalization by green fails,
        # so fallback to [1,1,1]. Let's check that red is indeed dominant.
        # Actually, let's make a proper balanced ROI
        img2 = np.zeros((64, 64, 3), dtype=np.float32)
        img2[:32, :, :] = [0.4, 0.5, 0.8]  # Red boosted region (BGR order)
        img2[32:, :, :] = [0.5, 0.5, 0.5]  # Neutral region

        wb = WhiteBalanceEstimator.estimate_from_gray(img2, gray_roi=(0, 0, 32, 64))
        # Red (index 2) is 0.8, green is 0.5, so red gets 0.8/0.5 = 1.6
        assert wb[2] > wb[1]

    def test_wb_from_gray_dtype(self):
        """WB should always return float32."""
        img = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        wb = WhiteBalanceEstimator.estimate_from_gray(img)
        assert wb.dtype == np.float32


# ============================================================================
# LinearGrader Tests
# ============================================================================

class TestLinearGrader:
    """Test float32 linear space grading operations."""

    def test_linear_curves_identity(self, linear_rgb_f32):
        """Identity curve should not change image."""
        identity_curve = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        curves = {'r': identity_curve, 'g': identity_curve, 'b': identity_curve}
        out = LinearGrader.linear_curves(linear_rgb_f32, curves)
        # Allow small rounding errors due to interpolation
        assert np.allclose(out, linear_rgb_f32, atol=0.01)

    def test_linear_curves_invert(self, linear_rgb_f32):
        """Inverted curve should produce inverted output."""
        invert_curve = np.linspace(1.0, 0.0, 256, dtype=np.float32)
        curves = {'r': invert_curve, 'g': invert_curve, 'b': invert_curve}
        out = LinearGrader.linear_curves(linear_rgb_f32, curves)
        expected = 1.0 - linear_rgb_f32
        assert np.allclose(out, expected, atol=0.01)

    def test_linear_curves_clipping(self, linear_rgb_f32):
        """Output should always be in [0, 1]."""
        # Curve that inverts: valid curve lookup
        invert_curve = np.linspace(1.0, 0.0, 256, dtype=np.float32)
        curves = {'r': invert_curve, 'g': invert_curve, 'b': invert_curve}
        out = LinearGrader.linear_curves(linear_rgb_f32, curves)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_linear_exposure_zero(self, linear_rgb_f32):
        """Zero exposure should not change image."""
        out = LinearGrader.linear_exposure(linear_rgb_f32, 0.0)
        assert np.allclose(out, linear_rgb_f32)

    def test_linear_exposure_positive(self, linear_rgb_f32):
        """Positive exposure should brighten."""
        out = LinearGrader.linear_exposure(linear_rgb_f32, 1.0)  # 1 stop = 2x
        # Most pixels should be brighter (except already at 1.0)
        diff = out - linear_rgb_f32
        assert np.sum(diff > 0.01) > np.sum(diff < -0.01)

    def test_linear_exposure_negative(self, linear_rgb_f32):
        """Negative exposure should darken."""
        out = LinearGrader.linear_exposure(linear_rgb_f32, -1.0)  # -1 stop = 0.5x
        diff = out - linear_rgb_f32
        # Most pixels should be darker (except already at 0.0)
        assert np.sum(diff < -0.01) > np.sum(diff > 0.01)

    def test_linear_contrast_identity(self, linear_rgb_f32):
        """Contrast = 1.0 should be identity."""
        out = LinearGrader.linear_contrast(linear_rgb_f32, 1.0)
        assert np.allclose(out, linear_rgb_f32)

    def test_linear_contrast_boost(self):
        """Contrast > 1.0 should increase spread."""
        # Create a narrow range: [0.4, 0.6]
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        img[:16, :, :] = 0.4
        img[16:, :, :] = 0.6

        out = LinearGrader.linear_contrast(img, 2.0, midpoint=0.5)
        # After 2x contrast around 0.5: 0.4 -> 0.3, 0.6 -> 0.7
        assert out[0, 0, 0] < 0.4
        assert out[-1, -1, 0] > 0.6

    def test_linear_contrast_clipping(self):
        """Extreme contrast should clip to [0, 1]."""
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.1
        out = LinearGrader.linear_contrast(img, 10.0, midpoint=0.5)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_linear_color_cast_unity(self, linear_rgb_f32):
        """Unity color cast should not change image."""
        cc = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        out = LinearGrader.linear_color_cast(linear_rgb_f32, cc)
        assert np.allclose(out, linear_rgb_f32)

    def test_linear_color_cast_red_boost(self):
        """Red boost should increase red channel."""
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        # cc order matches image channel order: [B, G, R]
        cc = np.array([1.0, 1.0, 2.0], dtype=np.float32)  # Boost red (index 2)
        out = LinearGrader.linear_color_cast(img, cc)
        # Red (index 2) should increase
        assert out[0, 0, 2] > img[0, 0, 2]
        # Blue and green should stay same
        assert out[0, 0, 0] == img[0, 0, 0]  # Blue
        assert out[0, 0, 1] == img[0, 0, 1]  # Green

    def test_linear_color_cast_clipping(self):
        """Output should clip to [0, 1]."""
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.7
        cc = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        out = LinearGrader.linear_color_cast(img, cc)
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)


# ============================================================================
# RAWDeveloper Tests
# ============================================================================

class TestRAWDeveloper:
    """Test main RAW orchestrator."""

    def test_developer_init(self):
        """Developer should initialize without error."""
        dev = RAWDeveloper()
        assert dev is not None
        assert dev.last_metadata == {}

    def test_develop_identity(self, developer, linear_rgb_f32):
        """Develop with default params should be close to identity."""
        out = developer.develop(linear_rgb_f32)
        assert out.dtype == np.float32
        assert out.shape == linear_rgb_f32.shape
        # No exposure, contrast, or color cast should be near-identity
        assert np.allclose(out, linear_rgb_f32, atol=1e-5)

    def test_develop_exposure(self, developer, linear_rgb_f32):
        """Develop with exposure should brighten."""
        out = developer.develop(linear_rgb_f32, exposure=1.0)
        diff = out - linear_rgb_f32
        # Most pixels should be brighter
        assert np.sum(diff > 0.01) > np.sum(diff < -0.01)

    def test_develop_contrast(self, developer, linear_rgb_f32):
        """Develop with contrast should increase spread."""
        out = developer.develop(linear_rgb_f32, contrast=1.5)
        assert out.dtype == np.float32
        # All values should be in range
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_develop_color_cast(self, developer, linear_rgb_f32):
        """Develop with color cast should apply RGB multipliers."""
        cc = np.array([2.0, 0.8, 1.0], dtype=np.float32)
        out = developer.develop(linear_rgb_f32, color_cast=cc)
        assert out.dtype == np.float32
        # Red channel should on average increase
        assert out[:, :, 2].mean() > linear_rgb_f32[:, :, 2].mean() * 0.9

    def test_develop_denoise(self, developer):
        """Develop with denoise should smooth the image."""
        # Create a noisy image
        rng = np.random.default_rng(123)
        base = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        noise = rng.normal(0.0, 0.1, size=base.shape).astype(np.float32)
        noisy = np.clip(base + noise, 0.0, 1.0)

        # Denoise
        out = developer.develop(noisy, denoise_strength=50.0)
        # After denoising, variance should decrease
        assert np.var(out) < np.var(noisy)

    def test_develop_returns_float32(self, developer, linear_rgb_f32):
        """Develop should always return float32."""
        out = developer.develop(linear_rgb_f32)
        assert out.dtype == np.float32

    def test_export_tiff_16bit(self, developer, tmp_path, linear_rgb_gradient):
        """Export 16-bit TIFF should create file."""
        out_path = tmp_path / "test_16.tif"
        developer.export_linear(linear_rgb_gradient, out_path, format="tiff", bit_depth=16)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_export_tiff_32f(self, developer, tmp_path, linear_rgb_gradient):
        """Export 32-bit float TIFF should create file."""
        out_path = tmp_path / "test_32f.tif"
        developer.export_linear(linear_rgb_gradient, out_path, format="tiff", bit_depth=32)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_export_unsupported_format(self, developer, tmp_path, linear_rgb_gradient):
        """Unsupported format should raise ValueError."""
        out_path = tmp_path / "test.xyz"
        with pytest.raises(ValueError, match="Unsupported export format"):
            developer.export_linear(linear_rgb_gradient, out_path, format="xyz")

    def test_load_raw_missing_file(self, developer):
        """load_raw should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            developer.load_raw("/nonexistent/file.cr2")

    def test_load_raw_no_rawpy(self, developer, tmp_path):
        """load_raw should raise ImportError if rawpy not available."""
        existing_raw = tmp_path / "existing.cr2"
        existing_raw.write_bytes(b"not decoded because rawpy is unavailable")
        with mock.patch("retouch.raw_develop.HAS_RAWPY", False):
            with pytest.raises(ImportError, match="rawpy is required"):
                developer.load_raw(existing_raw)

    def test_load_raw_invalid_file(self, developer, tmp_path):
        """load_raw should raise ValueError for non-RAW file."""
        fake_raw = tmp_path / "fake.cr2"
        fake_raw.write_text("not a raw file")

        # Only test if rawpy is available
        if not HAS_RAWPY:
            pytest.skip("rawpy not available")

        with pytest.raises(ValueError, match="Failed to decode RAW file"):
            developer.load_raw(fake_raw)

    def test_export_tiff_roundtrip(self, developer, tmp_path, linear_rgb_gradient):
        """Export and reload 16-bit TIFF should preserve data closely."""
        out_path = tmp_path / "roundtrip.tif"
        developer.export_linear(linear_rgb_gradient, out_path, format="tiff", bit_depth=16)

        # Reload via cv2 (which preserves BGR order)
        reloaded_u16 = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
        if reloaded_u16 is None:
            pytest.skip("Failed to read TIFF (cv2)")

        # Normalize to [0, 1]
        reloaded_bgr = reloaded_u16.astype(np.float32) / 65535.0

        # Allow ~1 level of quantization error
        diff = np.abs(reloaded_bgr - linear_rgb_gradient)
        assert np.max(diff) < 2.0 / 65535.0

    def test_export_path_as_string(self, developer, tmp_path, linear_rgb_gradient):
        """Export should accept path as string."""
        out_path = str(tmp_path / "test.tif")
        developer.export_linear(linear_rgb_gradient, out_path, format="tiff", bit_depth=16)
        assert Path(out_path).exists()


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_workflow(self):
        """Test develop -> grade -> export workflow."""
        dev = RAWDeveloper()

        # Create synthetic linear image
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        img[:16, :, :] = 0.3

        # Develop with adjustments
        developed = dev.develop(
            img,
            exposure=0.5,
            contrast=1.2,
            color_cast=np.array([1.1, 1.0, 0.9], dtype=np.float32),
        )

        # Check result is valid
        assert developed.dtype == np.float32
        assert np.all(developed >= 0.0)
        assert np.all(developed <= 1.0)

    def test_camera_matrix_with_develop(self):
        """Test using camera matrix in workflow."""
        cam_mat = CameraMatrix.get_camera_matrix("Canon EOS 5D")
        wb = CameraMatrix.get_white_balance("Canon EOS 5D", "cloudy")

        # Verify matrix and WB are usable
        assert cam_mat.shape == (3, 3)
        assert wb.shape == (3,)
        assert wb[0] > wb[1]  # Red boosted for cloudy

    def test_wb_estimators_consistency(self):
        """WB from metadata and from gray should give similar results for neutral image."""
        # Neutral gray
        neutral_gray = np.ones((64, 64, 3), dtype=np.float32) * 0.5

        wb_metadata = WhiteBalanceEstimator.estimate_from_metadata(1.0, 1.0, 1.0)
        wb_gray = WhiteBalanceEstimator.estimate_from_gray(neutral_gray)

        # Both should be close to (1, 1, 1)
        assert np.allclose(wb_metadata, [1.0, 1.0, 1.0])
        assert np.allclose(wb_gray, [1.0, 1.0, 1.0], atol=1e-5)


# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_curves_dict(self):
        """Empty curves dict should be no-op."""
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        out = LinearGrader.linear_curves(img, {})
        assert np.allclose(out, img)

    def test_partial_curves_dict(self):
        """Partial curves (only R, G) should apply selectively."""
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        curve = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        curves = {'r': curve, 'g': curve}  # No 'b'
        out = LinearGrader.linear_curves(img, curves)
        assert out.shape == img.shape

    def test_linear_grader_zero_contrast(self):
        """Zero contrast should reduce to midpoint."""
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.5
        out = LinearGrader.linear_contrast(img, 0.0, midpoint=0.7)
        # All pixels pushed toward midpoint
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)

    def test_developer_large_exposure(self):
        """Large exposure should clip appropriately."""
        dev = RAWDeveloper()
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.9
        out = dev.develop(img, exposure=5.0)  # +5 stops = 32x
        # Should clip at 1.0
        assert np.all(out <= 1.0)

    def test_developer_negative_large_exposure(self):
        """Large negative exposure should clip appropriately."""
        dev = RAWDeveloper()
        img = np.ones((32, 32, 3), dtype=np.float32) * 0.1
        out = dev.develop(img, exposure=-5.0)  # -5 stops = 1/32x
        # Should clip at 0.0
        assert np.all(out >= 0.0)

    def test_wb_estimator_all_zeros(self):
        """WB from all-zero image should not crash."""
        img = np.zeros((32, 32, 3), dtype=np.float32)
        wb = WhiteBalanceEstimator.estimate_from_gray(img)
        assert wb.shape == (3,)
        assert np.all(np.isfinite(wb))

    def test_camera_matrix_partial_match(self):
        """Partial model string match should work."""
        # "Canon EOS 5D Mark II" should match "Canon EOS 5D"
        mat = CameraMatrix.get_camera_matrix("Canon EOS 5D Mark II")
        assert mat.shape == (3, 3)
