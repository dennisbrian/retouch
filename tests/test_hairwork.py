"""Tests for retouch/hairwork.py — Strand flow field via structure tensor.

Tests verify orientation and coherence computation on synthetic patterns:
  - Horizontal/vertical/diagonal stripes → correct orientation
  - Noise → low coherence
  - Flat regions → zero coherence
  - High-res proxy path → output shape and tolerance preserved
  - Mask application → coherence zeroed outside mask
"""

import numpy as np
import pytest
from retouch.hairwork import hair_flow


class TestHairFlowBasics:
    """Basic orientation and coherence computation."""

    def test_returns_two_arrays(self):
        """hair_flow should return (orientation, coherence) tuple."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        orient, cohere = hair_flow(img)
        assert isinstance(orient, np.ndarray)
        assert isinstance(cohere, np.ndarray)
        assert orient.shape == (64, 64)
        assert cohere.shape == (64, 64)

    def test_output_dtype_float32(self):
        """Both outputs should be float32."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        orient, cohere = hair_flow(img)
        assert orient.dtype == np.float32
        assert cohere.dtype == np.float32

    def test_no_nan_or_inf(self):
        """Output should contain no NaN or inf values."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        orient, cohere = hair_flow(img)
        assert np.all(np.isfinite(orient))
        assert np.all(np.isfinite(cohere))

    def test_coherence_in_range(self):
        """Coherence should be in [0, 1]."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        orient, cohere = hair_flow(img)
        assert cohere.min() >= 0.0
        assert cohere.max() <= 1.0

    def test_orientation_in_valid_range(self):
        """Orientation should be in [-π/2, π/2] (arctan2 half-angle range)."""
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        orient, cohere = hair_flow(img)
        assert orient.min() >= -np.pi / 2.0 - 0.01  # small tolerance
        assert orient.max() <= np.pi / 2.0 + 0.01


class TestHorizontalStripes:
    """Horizontal stripe patterns should have coherent horizontal orientation."""

    def test_horizontal_stripes_basic(self):
        """Horizontal stripes → coherent horizontal strand direction."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Create horizontal stripes: alternating white/black bands
        stripe_height = 8
        for y in range(0, 256, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255
        orient, cohere = hair_flow(img)

        # Extract center region (avoid edge effects)
        center = orient[64:192, 64:192]
        center_cohere = cohere[64:192, 64:192]

        # Horizontal direction should be near 0 (or ±π due to arctan2 symmetry)
        # Coherence should be high
        mean_orient = np.mean(center)
        mean_cohere = np.mean(center_cohere)

        # For horizontal stripes, gradient is vertical → orientation ≈ ±π/2
        # So we expect orientation near ±π/2 (not 0)
        # Actually, let me reconsider: horizontal stripes have strong ∂/∂y gradients,
        # weak ∂/∂x. So gy is large, gx is small.
        # The principal eigenvector of the structure tensor points in direction
        # of maximum gradient power, which is vertical. So orientation ≈ π/2.
        assert mean_cohere > 0.75, f"Expected high coherence, got {mean_cohere}"

    def test_horizontal_stripes_orientation_5deg_tolerance(self):
        """Orientation within ~5° of expected value."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        stripe_height = 8
        for y in range(0, 256, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255
        orient, cohere = hair_flow(img)

        center = orient[64:192, 64:192]
        mean_orient = np.mean(center)

        # Horizontal stripes → vertical gradients → orientation ≈ ±π/2
        # Allow 5° tolerance = 5 * π/180 ≈ 0.087 radians
        tolerance = np.deg2rad(5)
        target = np.pi / 2.0
        # Check distance from π/2 or -π/2
        dist_to_half_pi = min(
            abs(mean_orient - target), abs(mean_orient + target)
        )
        assert dist_to_half_pi < tolerance, f"Mean orientation {np.rad2deg(mean_orient):.1f}° not within 5° of ±90°"


class TestVerticalStripes:
    """Vertical stripe patterns should have coherent vertical orientation."""

    def test_vertical_stripes_coherence(self):
        """Vertical stripes → high coherence."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        stripe_width = 8
        for x in range(0, 256, 2 * stripe_width):
            img[:, x : x + stripe_width] = 255
        orient, cohere = hair_flow(img)

        center = cohere[64:192, 64:192]
        mean_cohere = np.mean(center)

        assert mean_cohere > 0.75, f"Expected high coherence for vertical stripes, got {mean_cohere}"

    def test_vertical_stripes_orientation(self):
        """Vertical stripes should have orientation near 0."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        stripe_width = 8
        for x in range(0, 256, 2 * stripe_width):
            img[:, x : x + stripe_width] = 255
        orient, cohere = hair_flow(img)

        center = orient[64:192, 64:192]
        mean_orient = np.mean(center)

        # Vertical stripes → horizontal gradients → orientation ≈ 0
        tolerance = np.deg2rad(5)
        assert abs(mean_orient) < tolerance, f"Expected orientation near 0°, got {np.rad2deg(mean_orient):.1f}°"


class TestDiagonalStripes:
    """45° diagonal stripes should have ~45° orientation."""

    def test_diagonal_45deg_coherence(self):
        """45° diagonal stripes → high coherence."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        # Create 45° stripes by iterating along diagonal
        stripe_width = 8
        for offset in range(-256, 256, 2 * stripe_width):
            for y in range(256):
                x = y - offset
                if 0 <= x < 256:
                    img[y, x] = 255
        orient, cohere = hair_flow(img)

        center = cohere[64:192, 64:192]
        mean_cohere = np.mean(center)

        assert mean_cohere > 0.7, f"Expected high coherence for diagonal stripes, got {mean_cohere}"

    def test_diagonal_45deg_orientation(self):
        """45° diagonal stripes should have orientation ≈ 45°."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        stripe_width = 8
        for offset in range(-256, 256, 2 * stripe_width):
            for y in range(256):
                x = y - offset
                if 0 <= x < 256:
                    img[y, x] = 255
        orient, cohere = hair_flow(img)

        center = orient[64:192, 64:192]
        mean_orient = np.mean(center)

        # 45° diagonal → orientation ≈ 45° or -45° (π/4 or -π/4)
        tolerance = np.deg2rad(5)
        target = np.pi / 4.0
        dist_to_45 = min(abs(mean_orient - target), abs(mean_orient + target))
        assert dist_to_45 < tolerance, f"Expected ≈45° or ≈-45°, got {np.rad2deg(mean_orient):.1f}°"


class TestNoiseImage:
    """Random noise image should have low mean coherence (isotropic)."""

    def test_noise_low_coherence(self):
        """Gaussian noise → mean coherence < 0.4."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        orient, cohere = hair_flow(img)

        center = cohere[64:192, 64:192]
        mean_cohere = np.mean(center)

        assert mean_cohere < 0.4, f"Expected low coherence for noise, got {mean_cohere}"

    def test_noise_no_artifacts(self):
        """Noise image should produce finite, valid outputs (no NaN)."""
        np.random.seed(42)
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        orient, cohere = hair_flow(img)

        assert np.all(np.isfinite(orient))
        assert np.all(np.isfinite(cohere))
        assert cohere.min() >= 0.0
        assert cohere.max() <= 1.0


class TestFlatImage:
    """Flat gray image (no gradients) → coherence ≈ 0."""

    def test_flat_image_zero_coherence(self):
        """Flat gray image → coherence ≈ 0 (no structure)."""
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        orient, cohere = hair_flow(img)

        # All pixels should have near-zero coherence (no gradient structure)
        mean_cohere = np.mean(cohere)
        assert mean_cohere < 0.05, f"Expected coherence ≈ 0 for flat image, got {mean_cohere}"

    def test_flat_image_no_inf_nan(self):
        """Flat image should not produce inf or nan."""
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        orient, cohere = hair_flow(img)

        assert np.all(np.isfinite(orient))
        assert np.all(np.isfinite(cohere))


class TestProxyResolution:
    """High-resolution images should trigger downscaling and upsampling."""

    def test_high_res_triggers_proxy(self):
        """Image > max_dim (1200px) should be downscaled internally."""
        # Create 3000×3000 horizontal stripes
        img = np.zeros((3000, 3000, 3), dtype=np.uint8)
        stripe_height = 16
        for y in range(0, 3000, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255

        orient, cohere = hair_flow(img, max_dim=1200)

        # Output shape should match input (not downscaled in output)
        assert orient.shape == (3000, 3000)
        assert cohere.shape == (3000, 3000)

    def test_high_res_upsampled_output_valid(self):
        """Upsampled output should be valid (no NaN, coherence in [0,1])."""
        img = np.zeros((3000, 3000, 3), dtype=np.uint8)
        stripe_height = 16
        for y in range(0, 3000, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255

        orient, cohere = hair_flow(img, max_dim=1200)

        assert np.all(np.isfinite(orient))
        assert np.all(np.isfinite(cohere))
        assert cohere.min() >= 0.0
        assert cohere.max() <= 1.0

    def test_high_res_non_square_shape_and_tolerance(self):
        """3000x1500 non-square horizontal stripes: shape preserved, orientation/coherence hold."""
        img = np.zeros((1500, 3000, 3), dtype=np.uint8)
        stripe_height = 16
        for y in range(0, 1500, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255

        orient, cohere = hair_flow(img, max_dim=1200)

        assert orient.shape == (1500, 3000)
        assert cohere.shape == (1500, 3000)
        assert np.all(np.isfinite(orient))
        assert np.all(np.isfinite(cohere))

        center_o = orient[400:1100, 800:2200]
        center_c = cohere[400:1100, 800:2200]
        assert np.mean(center_c) > 0.7

        # Horizontal stripes → vertical gradients → orientation ≈ ±90° (axial)
        tolerance = np.deg2rad(5)
        d = np.abs(np.abs(center_o) - np.pi / 2.0)
        assert np.mean(d) < tolerance, f"Mean axial deviation {np.rad2deg(np.mean(d)):.1f} deg exceeds 5"

    def test_high_res_axial_wrap_88deg(self):
        """~88 deg stripes near the axial wrap must survive proxy upsampling within 5 deg."""
        h, w = 1500, 3000
        phi = np.deg2rad(88.0)  # gradient direction ~88 deg (near the +/-90 wrap)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Sinusoidal stripes with gradient along direction phi
        pattern = 0.5 + 0.5 * np.sin(2 * np.pi / 24.0 * (xx * np.cos(phi) + yy * np.sin(phi)))
        img = np.repeat((pattern * 255).astype(np.uint8)[:, :, None], 3, axis=2)

        orient, cohere = hair_flow(img, max_dim=1200)

        assert orient.shape == (h, w)
        center_o = orient[400:1100, 800:2200]
        center_c = cohere[400:1100, 800:2200]
        assert np.mean(center_c) > 0.7

        # Axial wrap-around distance: d = min(|delta|, pi - |delta|)
        delta = np.abs(center_o - phi)
        d = np.minimum(delta, np.pi - delta)
        tolerance = np.deg2rad(5)
        assert np.mean(d) < tolerance, f"Mean axial distance {np.rad2deg(np.mean(d)):.2f} deg exceeds 5"
        # No sin/cos-averaging collapse toward 0 deg: very few pixels far off-axis
        assert np.mean(d > np.deg2rad(20)) < 0.01

    def test_high_res_coherence_still_high(self):
        """High-res horizontal stripes should still have high center coherence."""
        img = np.zeros((3000, 3000, 3), dtype=np.uint8)
        stripe_height = 16
        for y in range(0, 3000, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255

        orient, cohere = hair_flow(img, max_dim=1200)

        center = cohere[750:2250, 750:2250]
        mean_cohere = np.mean(center)

        # Should still have high coherence after upsampling
        assert mean_cohere > 0.7, f"Expected coherence > 0.7 after upsampling, got {mean_cohere}"


class TestMaskApplication:
    """Hair mask should gate coherence outside the mask."""

    def test_mask_none_computes_everywhere(self):
        """With mask=None, compute orientation and coherence everywhere."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        stripe_height = 8
        for y in range(0, 256, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255

        orient, cohere = hair_flow(img, hair_mask=None)

        # Should compute at all pixels
        assert orient.shape == (256, 256)
        assert cohere.shape == (256, 256)

    def test_mask_applied_zeros_outside(self):
        """Coherence should be zero outside mask > 0.05."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        stripe_height = 8
        for y in range(0, 256, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255

        # Create a mask covering only center region
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[64:192, 64:192] = 1.0

        orient, cohere = hair_flow(img, hair_mask=mask)

        # Coherence outside mask should be zero
        outside_cohere = cohere[:64, :].max(), cohere[192:, :].max()
        assert np.all(cohere[:64, :] == 0.0)
        assert np.all(cohere[192:, :] == 0.0)

    def test_mask_uint8_normalized(self):
        """uint8 masks (0-255) should be normalized to [0, 1]."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        stripe_height = 8
        for y in range(0, 256, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255

        # Create uint8 mask (0-255 range)
        mask = np.zeros((256, 256), dtype=np.uint8)
        mask[64:192, 64:192] = 255

        orient, cohere = hair_flow(img, hair_mask=mask)

        # Should work the same as float32 [0, 1]
        assert cohere.shape == (256, 256)
        assert np.all(np.isfinite(cohere))
        # Outside should be zero
        assert np.allclose(cohere[:64, :], 0.0)
        assert np.allclose(cohere[192:, :], 0.0)

    def test_mask_threshold_005(self):
        """Pixels with mask ≤ 0.05 should have zero coherence."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        stripe_height = 8
        for y in range(0, 256, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255

        # Create a gradient mask with threshold at 0.05
        mask = np.zeros((256, 256), dtype=np.float32)
        mask[64:192, 64:192] = 1.0
        # Edge pixels just below threshold
        mask[63:64, :] = 0.03
        mask[192:193, :] = 0.03

        orient, cohere = hair_flow(img, hair_mask=mask)

        # Pixels with mask < 0.05 should have zero coherence
        assert np.all(cohere[63, :] == 0.0)
        assert np.all(cohere[192, :] == 0.0)
        # Inside high-mask region should have non-zero coherence
        assert np.mean(cohere[100:150, 100:150]) > 0.5


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_very_small_image(self):
        """Very small images (16×16) should still work."""
        img = np.full((16, 16, 3), 128, dtype=np.uint8)
        orient, cohere = hair_flow(img)

        assert orient.shape == (16, 16)
        assert cohere.shape == (16, 16)
        assert np.all(np.isfinite(orient))
        assert np.all(np.isfinite(cohere))

    def test_rectangular_image(self):
        """Non-square image should work."""
        img = np.full((300, 500, 3), 128, dtype=np.uint8)
        orient, cohere = hair_flow(img)

        assert orient.shape == (300, 500)
        assert cohere.shape == (300, 500)

    def test_uint8_input_preserved(self):
        """Function should handle uint8 input correctly."""
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        orient, cohere = hair_flow(img)

        assert orient.shape == (256, 256)
        assert cohere.shape == (256, 256)
        assert np.all(np.isfinite(orient))
        assert np.all(np.isfinite(cohere))


class TestConsistency:
    """Consistency and determinism."""

    def test_deterministic_output(self):
        """Same input should produce same output."""
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        stripe_height = 8
        for y in range(0, 256, 2 * stripe_height):
            img[y : y + stripe_height, :] = 255

        orient1, cohere1 = hair_flow(img)
        orient2, cohere2 = hair_flow(img)

        assert np.allclose(orient1, orient2)
        assert np.allclose(cohere1, cohere2)

    def test_same_pattern_different_position(self):
        """Same pattern at different position should have similar stats."""
        # Pattern at top
        img1 = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(0, 100, 16):
            img1[y : y + 8, :] = 255

        # Pattern at bottom (same pattern, different location)
        img2 = np.zeros((256, 256, 3), dtype=np.uint8)
        for y in range(156, 256, 16):
            img2[y : y + 8, :] = 255

        _, cohere1 = hair_flow(img1)
        _, cohere2 = hair_flow(img2)

        # Both should have similar coherence statistics
        mean1 = np.mean(cohere1[cohere1 > 0.5])
        mean2 = np.mean(cohere2[cohere2 > 0.5])
        # Allow some tolerance due to edge effects
        assert abs(mean1 - mean2) < 0.2
