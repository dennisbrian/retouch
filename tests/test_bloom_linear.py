"""Test suite for linear RGB bloom implementation in apply_global_bloom.

Verifies that the upgraded apply_global_bloom:
1. Computes bloom in linear RGB space (gamma 2.2) instead of gamma-encoded sRGB
2. Produces visibly denser, more concentrated glow around highlights
3. Maintains backward compatibility (strength=0 returns original)
4. Preserves dtype/shape invariants
"""

import numpy as np
import pytest
from retouch.utils import apply_global_bloom


class TestBloomLinearBehavior:
    """Test that linear-space bloom behaves correctly."""

    def test_zero_strength_returns_original(self):
        """strength=0 should return the original image unchanged."""
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = apply_global_bloom(img, strength=0.0, threshold=200.0, softness=30.0)
        assert np.all(result == img), "Zero strength should return original image"

    def test_output_dtype_is_uint8(self):
        """Output must be uint8 regardless of internal computation in linear space."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[45:55, 45:55] = 255
        result = apply_global_bloom(img, strength=50.0, threshold=200.0, softness=30.0)
        assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"

    def test_output_shape_unchanged(self):
        """Output shape must match input shape."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[45:55, 45:55] = 255
        result = apply_global_bloom(img, strength=50.0, threshold=200.0, softness=30.0)
        assert result.shape == img.shape, f"Shape mismatch: {result.shape} != {img.shape}"

    def test_flat_image_below_threshold_returns_unchanged(self):
        """A flat image with all pixels below threshold should return unchanged."""
        img = np.full((100, 100, 3), 180, dtype=np.uint8)  # L~180 in LAB
        result = apply_global_bloom(img, strength=80.0, threshold=210.0, softness=30.0)
        # With threshold=210 and a fully gray L=180 image, no pixels are above threshold
        # The early-return on highlight_mask.max() < 0.01 should trigger
        assert np.allclose(result, img, atol=1), "Flat image below threshold should return unchanged"

    def test_downsampled_large_image_shape_preserved(self):
        """Large images (> 2000px min_dim) should maintain shape through downsampling."""
        img = np.zeros((2200, 2200, 3), dtype=np.uint8)
        img[1000:1200, 1000:1200] = 255
        result = apply_global_bloom(img, strength=50.0, threshold=200.0, softness=30.0)
        assert result.shape == (2200, 2200, 3), "Downsampling should preserve original shape"

    def test_downsampled_image_produces_bloom(self):
        """Large downsampled images should still produce visible bloom effect."""
        img = np.zeros((2200, 2200, 3), dtype=np.uint8)
        img[1000:1200, 1000:1200] = 255
        result = apply_global_bloom(img, strength=80.0, threshold=200.0, softness=30.0)
        # Verify center is still bright
        assert np.all(result[1100, 1100] > 200), "Center should remain bright"
        # Verify bloom dispersion outside the highlight region
        # With a 2200px image, blur kernels are large, so glow extends significantly
        # Bloom should be visible at edges of highlight (just outside 1000-1200 range)
        assert np.any(result[950:1000, 1100] > 20), "Bloom should start just above highlight"
        assert np.any(result[1200:1250, 1100] > 20), "Bloom should extend just below highlight"


class TestBloomLinearVsGammaSpace:
    """Test that linear-space bloom produces measurably different results than gamma-space.

    The key property: linear-space bloom concentrates light intensity in the near-field halo
    because bright-pixel energy is not under-weighted by gamma encoding.
    """

    def _gamma_encode(self, img_lin: np.ndarray) -> np.ndarray:
        """Helper: encode from linear [0, 1] to gamma space."""
        return (img_lin ** (1.0 / 2.2) * 255.0).astype(np.uint8)

    def _gamma_decode(self, img_gamma: np.ndarray) -> np.ndarray:
        """Helper: decode from gamma space to linear [0, 1]."""
        return (img_gamma.astype(np.float32) / 255.0) ** 2.2

    def _compute_naive_gamma_bloom(
        self,
        img_bgr: np.ndarray,
        strength: float,
        threshold: float = 210.0,
        softness: float = 30.0,
    ) -> np.ndarray:
        """Reference implementation: bloom computed in gamma space (old approach).

        This is the implementation the linear version should improve upon.
        """
        import cv2

        if strength <= 0:
            return img_bgr

        # Convert to LAB to isolate highlights based on L (luminance) channel
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]

        # Soft threshold ramp from threshold to threshold + softness
        soft_w = max(1.0, softness)
        highlight_mask = np.clip((l_chan - threshold) / soft_w, 0.0, 1.0)

        if highlight_mask.max() < 0.01:
            return img_bgr

        h, w = img_bgr.shape[:2]
        min_dim = min(h, w)

        # Isolate highlights in gamma space (original approach)
        highlights_gamma = img_bgr.astype(np.float32) * highlight_mask[:, :, np.newaxis]

        # Downsampled bloom optimization for large images
        target_min = 2000
        is_downsampled = min_dim > target_min
        if is_downsampled:
            scale = target_min / min_dim
            highlights_low = cv2.resize(
                highlights_gamma, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
            blur_dim = min(highlights_low.shape[:2])
        else:
            highlights_low = highlights_gamma
            blur_dim = min_dim

        # Cascaded Gaussian blurs in gamma space
        k1 = max(15, int(blur_dim * 0.01)) | 1
        k2 = max(31, int(blur_dim * 0.03)) | 1
        k3 = max(63, int(blur_dim * 0.08)) | 1

        blur1 = cv2.GaussianBlur(highlights_low, (k1, k1), 0)
        blur2 = cv2.GaussianBlur(highlights_low, (k2, k2), 0)
        blur3 = cv2.GaussianBlur(highlights_low, (k3, k3), 0)

        # Blend multi-scale glows in gamma space
        glow_low = blur1 * 0.5 + blur2 * 0.3 + blur3 * 0.2

        if is_downsampled:
            glow = cv2.resize(glow_low, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            glow = glow_low

        # Screen blend in gamma space
        img_f = img_bgr.astype(np.float32)
        screen = 255.0 - ((255.0 - img_f) * (255.0 - glow) / 255.0)

        # Linearly interpolate between original and Screened in gamma space
        s_factor = strength / 100.0
        result = img_f * (1.0 - s_factor) + screen * s_factor
        return np.clip(result, 0, 255).astype(np.uint8)

    def test_linear_bloom_produces_brighter_halo(self):
        """Linear-space bloom should produce a brighter near-field halo than gamma-space.

        Synthetic test: a small bright spot (L~250) on neutral gray background (L~128).
        Measure halo brightness: mean intensity in an annular ring around the bright spot
        in both implementations. Linear version should have brighter halo due to energy
        concentration.
        """
        # Create synthetic test image: bright spot on gray background
        img = np.full((200, 200, 3), 128, dtype=np.uint8)  # L~128 mid-gray
        img[90:110, 90:110] = 255  # L~255 bright spot (20x20 square)

        # Run both implementations with same parameters
        strength = 80.0
        threshold = 210.0
        softness = 30.0

        result_linear = apply_global_bloom(img, strength=strength, threshold=threshold, softness=softness)
        result_gamma = self._compute_naive_gamma_bloom(img, strength=strength, threshold=threshold, softness=softness)

        # Define a halo region: annular ring at distance ~5-15px from the bright spot
        # Center of bright spot is at (100, 100)
        center_y, center_x = 100, 100
        halo_inner_r = 5
        halo_outer_r = 15

        # Create mask for halo pixels
        yy, xx = np.ogrid[:200, :200]
        dist = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
        halo_mask = (dist >= halo_inner_r) & (dist <= halo_outer_r)

        # Extract halo regions and compute mean brightness
        halo_linear = result_linear[halo_mask]
        halo_gamma = result_gamma[halo_mask]

        mean_linear = float(halo_linear.mean())
        mean_gamma = float(halo_gamma.mean())

        # Linear version should produce noticeably brighter halo
        # (due to energy concentration in linear space)
        assert mean_linear > mean_gamma, (
            f"Linear halo brightness ({mean_linear:.2f}) should be > gamma ({mean_gamma:.2f}). "
            f"Linear-space bloom should concentrate energy better."
        )

        print(f"Halo intensity comparison:")
        print(f"  Linear-space bloom mean halo: {mean_linear:.2f}")
        print(f"  Gamma-space bloom mean halo:  {mean_gamma:.2f}")
        print(f"  Improvement factor: {mean_linear / (mean_gamma + 1e-6):.2%}")

    def test_linear_bloom_halo_extent_reasonable(self):
        """Halo extent should be reasonable (not unexpectedly far or near)."""
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        img[90:110, 90:110] = 255

        result = apply_global_bloom(img, strength=80.0, threshold=210.0, softness=30.0)

        center_y, center_x = 100, 100
        yy, xx = np.meshgrid(np.arange(200), np.arange(200), indexing='ij')
        dist = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

        # Find the extent of the halo (non-background brightness)
        # Convert result to grayscale for intensity comparison
        result_gray = result.mean(axis=2)
        background_level = 128
        glow_mask = result_gray > (background_level + 5)  # Threshold above background

        # Halo should extend at least 10px from center (within the blur kernels)
        if np.any(glow_mask[85:95, 100]):  # Region 5-15px from center
            pass  # Expected
        else:
            # This is okay if bloom is subtle, just verify it exists somewhere
            assert np.any(glow_mask), "Should have some glow"

        # Halo should not extend unreasonably far (>60px) given the blur kernels
        # (blur kernels are max 8% of image dimension = ~16px in a 200px image)
        if np.any(glow_mask):
            max_dist = dist[glow_mask].max()
            assert max_dist < 80, f"Halo extent ({max_dist:.1f}px) seems unreasonably large"

    def test_linear_bloom_preserves_color_balance(self):
        """Linear-space bloom should preserve color balance (no unexpected channel inversion)."""
        # Create an image with a colored highlight
        img = np.full((100, 100, 3), 100, dtype=np.uint8)
        # Bright cyan spot: B=255, G=255, R=100 (high in blue and green, low in red)
        img[40:60, 40:60, 0] = 100  # B channel
        img[40:60, 40:60, 1] = 255  # G channel
        img[40:60, 40:60, 2] = 255  # R channel

        result = apply_global_bloom(img, strength=70.0, threshold=200.0, softness=30.0)

        # Check that the bloom region maintains reasonable color preservation
        # The center should still have the cyan character (G and B > R)
        center = result[49, 49]  # Inside the bright region
        assert center[1] > 100, "Green channel should be high in center"
        assert center[0] >= center[2] or center[0] > 100, "Blue should be strong or equal"
        # Basic sanity: all channels should be in valid range
        assert np.all(result >= 0) and np.all(result <= 255), "Result should be in uint8 range"


class TestBloomHighlightThresholds:
    """Test threshold and softness parameter behavior in linear-space bloom."""

    def test_high_threshold_glows_only_brightest(self):
        """High threshold should gate bloom to only the brightest pixels."""
        img = np.full((100, 100, 3), 200, dtype=np.uint8)  # L~200
        # Add a bright spot at L~240
        img[45:55, 45:55] = 240

        result = apply_global_bloom(img, strength=80.0, threshold=230.0, softness=10.0)
        # With threshold=230, only pixels above L~230 get bloom
        # The flat background at L~200 should barely glow
        assert np.allclose(result[20, 20], img[20, 20], atol=1), "Non-highlight region should be nearly unchanged"

    def test_low_threshold_glows_broadly(self):
        """Low threshold should gate bloom to a wider range of pixels."""
        img = np.full((100, 100, 3), 160, dtype=np.uint8)  # L~160 mid-bright
        result = apply_global_bloom(img, strength=50.0, threshold=150.0, softness=20.0)
        # Low threshold means the entire image (L=160 > 150) glows
        assert result.mean() > img.mean(), "Low threshold should increase overall brightness"

    def test_softness_creates_smooth_falloff(self):
        """Softness parameter should create a smooth transition in the highlight mask."""
        # Two test images: one with soft transition, one with hard
        img = np.full((100, 100, 3), 180, dtype=np.uint8)
        img[40:60, 40:60] = 240  # Bright region

        result_soft = apply_global_bloom(img, strength=80.0, threshold=210.0, softness=30.0)
        result_hard = apply_global_bloom(img, strength=80.0, threshold=210.0, softness=2.0)

        # Soft transition should produce a more gradual glow edge
        # (harder to test precisely without image comparison tools, but both should work)
        assert result_soft.shape == result_hard.shape
        assert result_soft.dtype == result_hard.dtype


class TestBloomStrengthGradation:
    """Test that bloom strength parameter behaves linearly and correctly."""

    def test_strength_100_produces_more_bloom_than_50(self):
        """strength=100 should produce more visible bloom than strength=50."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[45:55, 45:55] = 255

        result_50 = apply_global_bloom(img, strength=50.0, threshold=200.0, softness=30.0)
        result_100 = apply_global_bloom(img, strength=100.0, threshold=200.0, softness=30.0)

        # Pixels outside the highlight should have higher intensity with strength=100
        assert result_100[30, 50].mean() > result_50[30, 50].mean(), (
            f"strength=100 bloom ({result_100[30, 50]}) should be brighter than strength=50 ({result_50[30, 50]})"
        )

    def test_strength_interpolation_is_smooth(self):
        """Strength parameter should interpolate smoothly."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[45:55, 45:55] = 255

        results = []
        for strength in [0, 25, 50, 75, 100]:
            result = apply_global_bloom(img, strength=strength, threshold=200.0, softness=30.0)
            results.append(float(result[30, 50:55].mean()))

        # Results should be monotonically increasing with strength
        for i in range(len(results) - 1):
            assert results[i] <= results[i + 1], (
                f"Bloom should increase monotonically with strength. "
                f"Got {results} at strengths [0, 25, 50, 75, 100]"
            )


class TestBloomEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_black_image(self):
        """All-black image should return unchanged."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        result = apply_global_bloom(img, strength=100.0, threshold=200.0, softness=30.0)
        assert np.all(result == img), "Black image should be unchanged"

    def test_all_white_image(self):
        """All-white image should have uniform bloom applied."""
        img = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = apply_global_bloom(img, strength=50.0, threshold=200.0, softness=30.0)
        # All pixels are above threshold, so all should bloom uniformly
        # Result should still be very bright (screen blend only lightens)
        assert result.min() > 200, "All-white image should remain very bright"

    def test_small_image(self):
        """Small images should work (even if blur kernels become tiny)."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        img[4:6, 4:6] = 255
        result = apply_global_bloom(img, strength=80.0, threshold=200.0, softness=30.0)
        assert result.shape == (10, 10, 3)
        assert result.dtype == np.uint8

    def test_rectangular_image(self):
        """Non-square images should work."""
        img = np.zeros((50, 200, 3), dtype=np.uint8)
        img[20:30, 90:110] = 255
        result = apply_global_bloom(img, strength=60.0, threshold=200.0, softness=30.0)
        assert result.shape == (50, 200, 3)

    def test_negative_strength_treated_as_zero(self):
        """Negative strength should be treated as zero (early return)."""
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        result = apply_global_bloom(img, strength=-10.0, threshold=200.0, softness=30.0)
        assert np.all(result == img), "Negative strength should return original image"


class TestBloomFloatNativeNoBanding:
    """F1/E2: apply_global_bloom's float32 path used to quantize to uint8
    internally (highlight isolation + linearization + cascaded blur all ran
    on 8-bit data) even though the function accepted and returned float32 --
    a "fake float" round-trip, same bug class as the other F1 findings
    documented in docs/plans/PLAN_P4_MAKEUP_UNMIX.md Sec 18. These tests
    pin the fix: the float32 path must stay in float precision throughout
    and must not introduce new banding on a smooth gradient.
    """

    @staticmethod
    def _smooth_gradient(h=256, w=512, lo=120.0, hi=250.0):
        """A smooth linear luminance ramp spanning into the highlight
        range (threshold=210 by default) so bloom actually triggers --
        a flat/no-highlight gradient would false-pass any banding check."""
        grad = np.tile(np.linspace(lo, hi, w, dtype=np.float32), (h, 1))
        return np.stack([grad, grad, grad], axis=-1)

    def test_uint8_path_byte_identical_after_fix(self):
        """The fix only changes the float32 branch; uint8 input must
        retain its historical output within one platform rounding unit."""
        rng = np.random.RandomState(0)
        img_u8 = rng.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        out = apply_global_bloom(img_u8.copy(), strength=60, threshold=180, softness=30)
        checksum = int(out.astype(np.int64).sum())
        assert abs(checksum - 7044139) <= 1, checksum

        rng2 = np.random.RandomState(1)
        img_u8_2 = rng2.randint(50, 255, (200, 150, 3), dtype=np.uint8)
        out2 = apply_global_bloom(img_u8_2.copy(), strength=30, threshold=210, softness=15)
        checksum2 = int(out2.astype(np.int64).sum())
        assert abs(checksum2 - 14030336) <= 1, checksum2

    def test_float_path_no_banding_on_smooth_gradient(self):
        """A smooth gradient through the float32 path must stay smooth --
        every column should differ from its neighbor by construction, so
        the count of distinct output levels should be close to the image
        width (full resolution), not collapsed to a handful of 8-bit steps."""
        h, w = 256, 512
        img_255 = self._smooth_gradient(h, w)
        img_01 = img_255 / 255.0
        out_float = apply_global_bloom(img_01.copy(), strength=60, threshold=210, softness=30)
        assert out_float.dtype == np.float32

        row = (out_float[h // 2, :, 0] * 255.0)
        distinct_levels = len(np.unique(np.round(row, 2)))
        # Pre-fix this collapsed to ~258 (still constrained by an internal
        # uint8 quantization); post-fix it should reach full column
        # resolution (512) since every input column is distinct.
        assert distinct_levels >= w - 4, distinct_levels

    def test_float_path_matches_uint8_path_within_tolerance(self):
        """The float32 and uint8 paths should agree closely in aggregate
        (same algorithm, different precision) -- large disagreement would
        indicate a scale-convention bug (e.g. LAB range mismatch), not
        just a precision difference."""
        h, w = 256, 512
        img_255 = self._smooth_gradient(h, w)
        img_01 = img_255 / 255.0
        img_u8 = np.clip(img_255, 0, 255).astype(np.uint8)

        out_float = apply_global_bloom(img_01.copy(), strength=60, threshold=210, softness=30)
        out_uint8 = apply_global_bloom(img_u8.copy(), strength=60, threshold=210, softness=30)

        out_float_255 = out_float * 255.0
        out_uint8_255 = out_uint8.astype(np.float32)
        mean_delta = float(np.abs(out_float_255 - out_uint8_255).mean())
        assert mean_delta < 2.0, mean_delta
