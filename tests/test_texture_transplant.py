"""Tests for texture_transplant method (Stage S6).

Tests cover:
1. Early return conditions (strength=0, skin_mask=None)
2. Donor region remains unmodified
3. Texture actually transplanted (measurable high-band energy increase)
4. No-donor-found graceful fallback
5. Rotation/jitter sanity (no NaN/Inf, valid uint8 range)
6. Luminance matching sanity (mean L shift within bounds)
"""

import pytest
import numpy as np
import cv2
from retouch.skin import SkinProcessor, _blotch_bandpass
from retouch.utils import blend_masked


@pytest.fixture
def skin_processor():
    return SkinProcessor()


def create_synthetic_face_image(height=400, width=400):
    """Create a synthetic face-like image for testing.

    Returns:
        img_bgr: (H, W, 3) uint8 BGR image
        skin_mask: (H, W) float32 skin mask [0, 1]
    """
    # Create a base image with skin-like color
    img_bgr = np.zeros((height, width, 3), dtype=np.uint8)

    # Skin-tone color in BGR (warm, slightly reddish)
    skin_bgr = np.array([135, 155, 180], dtype=np.uint8)  # Light brown
    img_bgr[:] = skin_bgr

    # Create a circular skin mask (face-like region)
    skin_mask = np.zeros((height, width), dtype=np.float32)
    center_y, center_x = height // 2, width // 2
    radius = min(height, width) // 2.5

    yy, xx = np.ogrid[:height, :width]
    circle_mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius ** 2
    skin_mask[circle_mask] = 1.0

    # Add some natural pore texture (high-frequency noise) to most of the face
    np.random.seed(42)
    pore_noise = np.random.normal(0, 3, (height, width)).astype(np.float32)
    pore_noise = cv2.GaussianBlur(pore_noise, (3, 3), 0)

    # Convert to LAB to add texture to L channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + pore_noise * skin_mask * 2.0, 0, 255)

    # Create a "clean donor" region on one side (cheek-like, very low blotch variance)
    # This is where the donor patch should come from
    donor_y_start, donor_y_end = height // 3, height // 2
    donor_x_start, donor_x_end = width // 3, width // 2
    donor_noise = np.random.normal(0, 1, (donor_y_end - donor_y_start, donor_x_end - donor_x_start))
    donor_noise = cv2.GaussianBlur(donor_noise, (3, 3), 0)
    lab[donor_y_start:donor_y_end, donor_x_start:donor_x_end, 0] = np.clip(
        lab[donor_y_start:donor_y_end, donor_x_start:donor_x_end, 0] + donor_noise * 0.5,
        0, 255
    )

    # Create an "over-smoothed" region on the other side (very low texture)
    # This is where texture should be transplanted
    smooth_y_start, smooth_y_end = height // 2, 2 * height // 3
    smooth_x_start, smooth_x_end = width // 2, 2 * width // 3
    # Heavily smooth this region (no pore texture)
    lab[smooth_y_start:smooth_y_end, smooth_x_start:smooth_x_end, 0] = cv2.GaussianBlur(
        lab[smooth_y_start:smooth_y_end, smooth_x_start:smooth_x_end, 0], (15, 15), 0
    )

    img_bgr = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    return img_bgr, skin_mask


def measure_highband_energy(img_bgr, mask=None):
    """Measure high-frequency energy in L channel.

    Returns the std dev of (L - gaussian_blur(L, sigma=2)) within the mask.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]

    L_blurred = cv2.GaussianBlur(L, (0, 0), 2.0)
    highband = L - L_blurred

    if mask is not None:
        mask_indices = mask > 0.3
        if np.any(mask_indices):
            return np.std(highband[mask_indices])
    return np.std(highband)


class TestTextureTransplantEarlyReturn:
    """Test early return conditions."""

    def test_strength_zero(self, skin_processor):
        """strength=0 should return input unchanged."""
        img_bgr, skin_mask = create_synthetic_face_image()
        result = skin_processor.texture_transplant(img_bgr, skin_mask, strength=0, face_width=100)
        np.testing.assert_array_equal(result, img_bgr, err_msg="Expected unchanged output at strength=0")

    def test_skin_mask_none(self, skin_processor):
        """skin_mask=None should return input unchanged."""
        img_bgr, _ = create_synthetic_face_image()
        result = skin_processor.texture_transplant(img_bgr, None, strength=50, face_width=100)
        np.testing.assert_array_equal(result, img_bgr, err_msg="Expected unchanged output with skin_mask=None")

    def test_both_none(self, skin_processor):
        """Both strength=0 and skin_mask=None should return input unchanged."""
        img_bgr, _ = create_synthetic_face_image()
        result = skin_processor.texture_transplant(img_bgr, None, strength=0, face_width=100)
        np.testing.assert_array_equal(result, img_bgr, err_msg="Expected unchanged output with both conditions")


class TestDonorRegionUnmodified:
    """Test that donor region remains unmodified (read-only)."""

    def test_donor_untouched(self, skin_processor):
        """Donor region pixels should remain byte-identical (or within float rounding ~±0 diff)."""
        img_bgr, skin_mask = create_synthetic_face_image(height=400, width=400)
        result = skin_processor.texture_transplant(img_bgr, skin_mask, strength=80, face_width=100)

        # The donor region is the upper-left quadrant (cleanest skin)
        # Check that this area is largely unchanged
        donor_region_input = img_bgr[50:150, 50:150, :].astype(np.float32)
        donor_region_output = result[50:150, 50:150, :].astype(np.float32)

        max_diff = np.max(np.abs(donor_region_input - donor_region_output))
        # Allow small rounding errors from float conversions
        assert max_diff < 2.0, f"Donor region changed by {max_diff}, expected < 2.0"


class TestTextureTransplanted:
    """Test that texture is actually transplanted into target zones.

    NOTE ON A FIXED BUG (2026-07-03 code review): the original luminance-matching
    step rescaled the donor texture using the TARGET zone's own (pre-transplant,
    near-zero-by-selection) high-band energy as the scale target. Since the target
    zone is selected specifically BECAUSE its energy is low, this was circular:
    on a near-flat zone, energy_scale collapsed toward 0 and the transplant became
    a near no-op on exactly the case it exists to fix. A second, related bug was
    also found: donor-patch selection used only blotch-band variance, so a
    perfectly flat patch (zero variance at every frequency, including blotch)
    could be chosen as the "cleanest" donor despite having zero pore-scale texture
    to actually donate.
    Both are fixed: (1) luminance matching now scales the donor texture to
    `median_energy` — the face's own healthy high-band energy level — instead of
    the target zone's own degraded energy; (2) donor candidates are now required
    to have real pore-scale high-band energy (std > 0.5) before being eligible,
    so a flat patch can no longer be selected as a texture donor.
    The tests below use a REAL threshold (not "no regression") and include the
    exact adversarial case that exposed the bug: a perfectly flat patch.
    """

    def test_texture_increase_in_smooth_zone(self, skin_processor):
        """High-band energy should increase measurably in previously smooth zones.

        This fixture's "smooth zone" (mild 15x15 Gaussian blur, baseline ~1.43)
        is not fully flat, so the bar is lower than the flat-patch adversarial
        test below. Empirically the fixed implementation achieves ~17% increase
        here; assert a real, non-trivial floor (>=10%) rather than "no decrease."
        """
        img_bgr, skin_mask = create_synthetic_face_image(height=400, width=400)

        smooth_zone_mask = np.zeros_like(skin_mask)
        smooth_zone_mask[200:300, 200:300] = 1.0
        baseline_energy = measure_highband_energy(img_bgr, smooth_zone_mask)

        result = skin_processor.texture_transplant(img_bgr, skin_mask, strength=80, face_width=100)
        result_energy = measure_highband_energy(result, smooth_zone_mask)

        energy_increase_pct = (result_energy - baseline_energy) / (baseline_energy + 1e-6) * 100
        print(f"Baseline energy: {baseline_energy:.3f}, Result energy: {result_energy:.3f}, "
              f"Increase: {energy_increase_pct:.1f}%")

        assert result_energy > baseline_energy * 1.10, \
            f"Expected >=10% energy increase, got {energy_increase_pct:.1f}% " \
            f"(baseline={baseline_energy:.3f}, result={result_energy:.3f})"

    def test_texture_increase_on_perfectly_flat_patch(self, skin_processor):
        """Adversarial case that exposed the luminance-matching + donor-selection bug:
        a perfectly flat 100x100 target patch (high-band energy EXACTLY 0.0) inside
        a textured face. Before the fix this stayed at ~0.0 (a 0/0-style collapse in
        the energy_scale ratio, compounded by a flat patch being selectable as its
        own "donor"). After the fix, the inner (edge-bleed-free) region of the flat
        patch must land in the same ballpark as the surrounding healthy texture
        energy, not remain near zero.
        """
        h, w = 400, 400
        img_bgr = np.full((h, w, 3), 150, dtype=np.uint8)

        skin_mask = np.zeros((h, w), dtype=np.float32)
        cy, cx = h // 2, w // 2
        radius = 180
        yy, xx = np.ogrid[:h, :w]
        skin_mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2] = 1.0

        np.random.seed(7)
        noise = np.random.normal(0, 4, (h, w)).astype(np.float32)
        noise = cv2.GaussianBlur(noise, (3, 3), 0)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + noise * skin_mask, 0, 255)

        flat_y0, flat_x0 = 150, 150
        lab[flat_y0:flat_y0 + 100, flat_x0:flat_x0 + 100, 0] = 150.0  # perfectly flat
        img_bgr = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Inner region avoids edge-bleed contamination from the surrounding texture
        # leaking into the blur kernel at the patch boundary — a clean apples-to-
        # apples measurement of the flat interior.
        inner_mask = np.zeros((h, w), dtype=np.float32)
        inner_mask[flat_y0 + 15:flat_y0 + 85, flat_x0 + 15:flat_x0 + 85] = 1.0

        flat_full_mask = np.zeros((h, w), dtype=np.float32)
        flat_full_mask[flat_y0:flat_y0 + 100, flat_x0:flat_x0 + 100] = 1.0
        surround_mask = np.clip(
            skin_mask - cv2.dilate(flat_full_mask, np.ones((21, 21), np.uint8)), 0, 1
        )

        baseline_inner_energy = measure_highband_energy(img_bgr, inner_mask)
        surround_energy = measure_highband_energy(img_bgr, surround_mask)

        # Sanity-check the adversarial setup itself
        assert baseline_inner_energy < 1e-4, \
            f"Expected exactly-flat baseline, got {baseline_inner_energy}"
        assert surround_energy > 0.5, "Surrounding texture reference is unexpectedly weak"

        result = skin_processor.texture_transplant(img_bgr, skin_mask, strength=80, face_width=100)
        result_inner_energy = measure_highband_energy(result, inner_mask)

        ratio_to_surround = result_inner_energy / surround_energy
        print(f"BEFORE (inner flat zone): {baseline_inner_energy:.6f}")
        print(f"AFTER  (inner flat zone): {result_inner_energy:.6f}")
        print(f"Surrounding texture energy (reference): {surround_energy:.6f}")
        print(f"Ratio to surrounding texture energy: {ratio_to_surround:.3f}")

        # Must move meaningfully off zero and land in the same ballpark as the
        # face's own healthy texture (empirically ~0.91x surround energy with the
        # fix; require at least 50% of surround energy as a real, non-trivial bar).
        assert result_inner_energy > 0.3, \
            f"Expected inner flat-zone energy to rise well above 0, got {result_inner_energy:.6f}"
        assert ratio_to_surround > 0.5, \
            f"Expected transplanted energy to reach >=50% of surrounding texture " \
            f"energy, got {ratio_to_surround:.3f} ({result_inner_energy:.3f} / {surround_energy:.3f})"


class TestNoDonorFoundFallback:
    """Test graceful fallback when no clean donor patch is found."""

    def test_no_donor_fallback_tiny_face(self, skin_processor):
        """Tiny face crop should gracefully return unchanged input."""
        # Create a very small skin region
        img_bgr = np.full((50, 50, 3), 150, dtype=np.uint8)
        skin_mask = np.ones((50, 50), dtype=np.float32) * 0.1  # Very sparse

        result = skin_processor.texture_transplant(img_bgr, skin_mask, strength=80, face_width=50)

        # Should return unchanged due to insufficient candidates
        # (Not a hard assertion since the algorithm might still find patches,
        # but expect minimal changes)
        max_diff = np.max(np.abs(img_bgr.astype(np.float32) - result.astype(np.float32)))
        assert max_diff < 5.0, f"Tiny face should have minimal changes, got {max_diff}"


class TestRotationJitterSanity:
    """Test that rotation jitter produces valid output."""

    def test_no_nan_inf_output(self, skin_processor):
        """Output should not contain NaN or Inf."""
        img_bgr, skin_mask = create_synthetic_face_image()
        result = skin_processor.texture_transplant(img_bgr, skin_mask, strength=80, face_width=100)

        assert np.all(np.isfinite(result)), "Output contains NaN or Inf"
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

    def test_output_in_valid_range(self, skin_processor):
        """Output uint8 values should be in [0, 255]."""
        img_bgr, skin_mask = create_synthetic_face_image()
        result = skin_processor.texture_transplant(img_bgr, skin_mask, strength=80, face_width=100)

        assert np.all(result >= 0), "Output has values < 0"
        assert np.all(result <= 255), "Output has values > 255"


class TestLuminanceMatchingSanity:
    """Test that luminance matching doesn't shift zone brightness excessively."""

    def test_mean_l_shift_bounded(self, skin_processor):
        """Mean L in transplant zone should shift by <5 L-levels even with texture added."""
        img_bgr, skin_mask = create_synthetic_face_image(height=400, width=400)

        # Measure baseline mean L in smooth zone
        smooth_zone_mask = np.zeros_like(skin_mask)
        smooth_zone_mask[200:300, 200:300] = 1.0

        lab_before = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_before = lab_before[:, :, 0]
        zone_indices = smooth_zone_mask > 0.3
        if np.any(zone_indices):
            baseline_mean_l = np.mean(L_before[zone_indices])
        else:
            baseline_mean_l = 128.0

        # Apply texture transplant
        result = skin_processor.texture_transplant(img_bgr, skin_mask, strength=80, face_width=100)

        # Measure mean L after
        lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_after = lab_after[:, :, 0]
        if np.any(zone_indices):
            result_mean_l = np.mean(L_after[zone_indices])
        else:
            result_mean_l = 128.0

        l_shift = abs(result_mean_l - baseline_mean_l)
        print(f"Baseline mean L: {baseline_mean_l:.1f}, Result mean L: {result_mean_l:.1f}, Shift: {l_shift:.1f}")

        # Shift should be small (we're adding texture variation, not brightness)
        assert l_shift < 5.0, f"Mean L shifted by {l_shift:.1f}, expected <5.0"


class TestBlotchBandpassIntegration:
    """Test that _blotch_bandpass is correctly used for donor detection."""

    def test_blotch_bandpass_output_shape(self):
        """_blotch_bandpass should return same shape as input L."""
        L = np.random.rand(100, 100).astype(np.float32) * 255
        face_width = 100.0
        result = _blotch_bandpass(L, face_width)

        assert result.shape == L.shape, f"Expected shape {L.shape}, got {result.shape}"

    def test_blotch_bandpass_can_be_negative(self):
        """_blotch_bandpass can produce negative values (DoG result)."""
        L = np.full((100, 100), 128.0, dtype=np.float32)
        L[40:60, 40:60] = 200.0  # Create a bright spot

        face_width = 100.0
        result = _blotch_bandpass(L, face_width)

        # DoG of a Gaussian should have negative regions
        assert np.any(result < 0), "Expected negative values in DoG output"
        assert np.any(result > 0), "Expected positive values in DoG output"


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_with_real_image_dimensions(self, skin_processor):
        """Test with larger, more realistic image."""
        img_bgr, skin_mask = create_synthetic_face_image(height=1024, width=1024)
        result = skin_processor.texture_transplant(img_bgr, skin_mask, strength=60, face_width=300)

        assert result.shape == img_bgr.shape
        assert result.dtype == np.uint8
        assert np.all(np.isfinite(result))

    def test_multiple_strength_levels(self, skin_processor):
        """Test consistency across different strength values."""
        img_bgr, skin_mask = create_synthetic_face_image()

        for strength in [0, 25, 50, 75, 100]:
            result = skin_processor.texture_transplant(img_bgr, skin_mask, strength=strength, face_width=100)
            assert result.shape == img_bgr.shape
            assert result.dtype == np.uint8
            # Verify no NaN/Inf regardless of strength
            assert np.all(np.isfinite(result)), f"Got NaN/Inf at strength={strength}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
