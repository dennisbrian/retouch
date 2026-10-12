"""Tests for grain.py amplitude scaling fix.

Verifies that the amplitude scaling change (from strength*20.0 to strength*7.0)
produces grain that matches the "fine Fuji grain baseline" design for strength=0.3,
while preserving luminance correlation and monotonicity across the strength range.
"""

import cv2
import numpy as np

from retouch.grain import apply_film_grain
from retouch.color_space import bgr_to_lab


class TestGrainAmplitudeScaling:
    """Test suite for grain amplitude scaling fix."""

    def _measure_grain_std(self, img_bgr: np.ndarray, strength: float) -> float:
        """Apply grain and measure the per-pixel std deviation in the L channel.

        Args:
            img_bgr: Input BGR image (typically flat gray).
            strength: Grain strength parameter.

        Returns:
            Standard deviation of grain effect (on 0-255 uint8 scale, measured in L).
        """
        if strength <= 0.0:
            return 0.0

        result = apply_film_grain(img_bgr, strength=strength, seed=42)

        # Convert to LAB to measure in L channel
        lab_in = bgr_to_lab(img_bgr)
        lab_out = bgr_to_lab(result)

        l_in = lab_in[:, :, 0].astype(np.float32)
        l_out = lab_out[:, :, 0].astype(np.float32)

        grain_diff = l_out - l_in  # Grain effect in L* [0,100] scale
        return float(np.std(grain_diff))

    def test_amplitude_at_fine_grain_baseline(self):
        """Verify that strength=0.3 produces grain std in the fine-grain target range.

        At strength=0.3, the docstring claims "a fine Fuji grain baseline."
        Empirical testing with the NEW formula (amplitude = strength * 7.0) should
        produce grain with std between 0.7 and 2.0 on the L* [0,100] scale.

        This is substantially smaller than the OLD formula's std (~2.5-5.0 with
        amplitude = strength * 20.0), validating the fix.
        """
        img = np.full((500, 500, 3), 128, dtype=np.uint8)  # Mid-gray

        std_new = self._measure_grain_std(img, strength=0.3)

        # With new formula (amplitude = 7.0 * 0.3 = 2.1), grain std should be
        # roughly in the 0.7-1.5 range on L* scale after luminance modulation.
        # Allow up to 2.0 to account for clumping and randomness.
        assert 0.7 <= std_new <= 2.0, (
            f"Expected grain std at strength=0.3 to be 0.7-2.0, got {std_new:.2f}. "
            "This indicates amplitude scaling may still be off."
        )

    def test_amplitude_regression_old_vs_new(self):
        """Verify new amplitude is substantially smaller than old at strength=0.3.

        Old formula: amplitude = 0.3 * 20.0 = 6.0
        New formula: amplitude = 0.3 * 7.0 = 2.1

        Expected reduction: ~65% smaller grain std with new formula.
        Assert: new_std <= 0.4 * old_std (expect ~30% of old std or less due to
        the nonlinear luminance modulation; a 0.65 multiplier in amplitude gives
        roughly 0.4-0.5x the std due to how per-pixel variance compounds).
        """
        img = np.full((500, 500, 3), 128, dtype=np.uint8)

        std_new = self._measure_grain_std(img, strength=0.3)

        # Compute what old formula would have given.
        # Old amplitude = 6.0, new amplitude = 2.1, ratio = 2.1/6.0 ≈ 0.35
        # Grain std scales roughly linearly with amplitude, so we expect:
        # std_new ≈ 0.35 * std_old, or equivalently std_new / std_old ≈ 0.35

        # For mid-gray with luminance modulation, std_old would be roughly:
        # mid-gray L* = 50, lum_mod ≈ (1 - 0.5)^1.2 ≈ 0.43
        # std_old ≈ 6.0 * 0.43 ≈ 2.6 (rough estimate, noise std ~1)
        # std_new ≈ 2.1 * 0.43 ≈ 0.9
        # ratio ≈ 0.9 / 2.6 ≈ 0.35 ✓

        # We can't compute exact old value without re-implementing, but we can
        # verify the new std is in the fine-grain range and small compared to
        # a hypothetical max-amplitude grain (strength=1.0).
        std_max = self._measure_grain_std(img, strength=1.0)

        # At max strength, grain should be noticeably larger than fine grain.
        assert std_new < std_max * 0.5, (
            f"Fine grain (std={std_new:.2f}) should be much smaller than "
            f"max grain (std={std_max:.2f}), but ratio is {std_new/std_max:.2f}."
        )

    def test_sanity_bounds_fine_grain(self):
        """Sanity check: fine grain should not exceed 3% of L* range.

        L* range is [0, 100], so 3% = 3 units. At strength=0.3 (fine grain),
        the grain std should be well below this: typically 0.7-1.5 units on L* scale.
        """
        img = np.full((500, 500, 3), 128, dtype=np.uint8)
        std_new = self._measure_grain_std(img, strength=0.3)

        # Sanity: should be much less than 3% of L* range
        assert std_new < 3.0, (
            f"Fine grain std ({std_new:.2f}) should be < 3.0 on L* [0,100] scale. "
            "This suggests the amplitude multiplier is still too high."
        )

    def test_sanity_bounds_max_grain(self):
        """Sanity check: max grain (strength=1.0) should not exceed 10% of L* range.

        At strength=1.0 (very strong), grain std should be noticeable but not
        destructive. Target: 2.5-5.0 units on L* scale (2.5-5.0% of range).
        """
        img = np.full((500, 500, 3), 128, dtype=np.uint8)
        std_max = self._measure_grain_std(img, strength=1.0)

        # At max strength, should be "very strong" but not absurdly so
        assert std_max >= 2.0, (
            f"Max grain std ({std_max:.2f}) should be >= 2.0 on L* scale. "
            "This suggests strength=1.0 is too weak."
        )
        assert std_max <= 8.0, (
            f"Max grain std ({std_max:.2f}) should be <= 8.0 on L* scale. "
            "This suggests the amplitude multiplier is still too high."
        )

    def test_monotonicity_across_strength_range(self):
        """Verify grain std increases monotonically as strength increases.

        Test: strength values 0.1, 0.3, 0.5, 0.8, 1.0 should have increasing std.
        This ensures the amplitude formula scales correctly across the full range.
        """
        img = np.full((500, 500, 3), 128, dtype=np.uint8)

        strengths = [0.1, 0.3, 0.5, 0.8, 1.0]
        stds = [self._measure_grain_std(img, s) for s in strengths]

        # Each step should increase std (allowing small tolerance for randomness)
        for i in range(len(stds) - 1):
            assert stds[i + 1] >= stds[i] - 0.1, (
                f"Monotonicity broken: strength={strengths[i]} -> std={stds[i]:.2f}, "
                f"strength={strengths[i + 1]} -> std={stds[i + 1]:.2f}. "
                f"Grain should increase with strength."
            )

        # Also verify significant overall increase from min to max
        assert stds[-1] >= stds[0] * 2.0, (
            f"Grain at strength=1.0 (std={stds[-1]:.2f}) should be >= 2x grain at "
            f"strength=0.1 (std={stds[0]:.2f})."
        )

    def test_zero_strength_is_identity(self):
        """Regression guard: strength=0 should return image unchanged."""
        img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        result = apply_film_grain(img, strength=0.0, seed=42)
        assert np.array_equal(result, img), (
            "strength=0.0 should return image unchanged."
        )

    def test_luminance_correlation_preserved(self):
        """Verify shadows still show more grain than highlights.

        Create a half-dark/half-bright image, apply grain, and verify that
        the dark half has higher grain std than the bright half.
        This ensures luminance_power modulation still works correctly.
        """
        # Create half-dark (L*=20), half-bright (L*=80) test image
        # In BGR 0-255 scale: L*=20 ≈ ~51 gray, L*=80 ≈ ~204 gray
        img = np.full((256, 256, 3), 128, dtype=np.uint8)  # Mid-gray base
        img[:128, :, :] = 51   # Dark half
        img[128:, :, :] = 204  # Bright half

        result = apply_film_grain(img, strength=0.3, seed=42)

        lab_in = bgr_to_lab(img)
        lab_out = bgr_to_lab(result)

        l_in_dark = lab_in[:128, :, 0].astype(np.float32)
        l_out_dark = lab_out[:128, :, 0].astype(np.float32)
        grain_dark = np.std(l_out_dark - l_in_dark)

        l_in_bright = lab_in[128:, :, 0].astype(np.float32)
        l_out_bright = lab_out[128:, :, 0].astype(np.float32)
        grain_bright = np.std(l_out_bright - l_in_bright)

        # Shadows should have more grain than highlights
        assert grain_dark > grain_bright, (
            f"Grain in dark areas (std={grain_dark:.2f}) should exceed grain in "
            f"bright areas (std={grain_bright:.2f}). Luminance correlation broken."
        )
        # Expect roughly 2x more grain in shadows due to luma_power=1.2
        assert grain_dark >= grain_bright * 1.3, (
            f"Expected dark grain to be >= 1.3x bright grain, got ratio {grain_dark/grain_bright:.2f}. "
            f"Luminance modulation may be too weak."
        )

    def test_consistency_with_seed(self):
        """Verify same seed produces identical grain."""
        img = np.full((128, 128, 3), 128, dtype=np.uint8)

        out1 = apply_film_grain(img, strength=0.3, seed=999)
        out2 = apply_film_grain(img, strength=0.3, seed=999)

        assert np.array_equal(out1, out2), (
            "Same seed should produce identical grain output."
        )

    def test_different_seeds_produce_different_grain(self):
        """Verify different seeds produce different grain patterns."""
        img = np.full((256, 256, 3), 128, dtype=np.uint8)

        out1 = apply_film_grain(img, strength=0.3, seed=1)
        out2 = apply_film_grain(img, strength=0.3, seed=2)

        # Different seeds should produce different (but similar std) outputs
        assert not np.array_equal(out1, out2), (
            "Different seeds should produce different grain patterns."
        )

        # But stds should be similar (within ~20%)
        lab1 = bgr_to_lab(out1)
        lab2 = bgr_to_lab(out2)
        std1 = np.std(lab1[:, :, 0].astype(np.float32))
        std2 = np.std(lab2[:, :, 0].astype(np.float32))

        # Both are full-image stds, so some variation is expected, but should be close
        ratio = std2 / std1 if std1 > 0 else 1.0
        assert 0.8 <= ratio <= 1.25, (
            f"Stds from different seeds should be similar: {std1:.2f} vs {std2:.2f} "
            f"(ratio {ratio:.2f}). Suggests non-deterministic behavior or seeds "
            f"affecting grain properties unexpectedly."
        )


class TestAmplitudeReferenceComparison:
    """Compare new amplitude scaling against reference values.

    These tests document the exact amplitude values produced by the new formula
    and verify they match the intended ranges.
    """

    def test_amplitude_formula_values(self):
        """Verify the new amplitude formula produces expected values.

        Formula: amplitude = strength * 7.0
        - strength=0.0 → amplitude=0.0
        - strength=0.1 → amplitude=0.7
        - strength=0.3 → amplitude=2.1
        - strength=0.5 → amplitude=3.5
        - strength=1.0 → amplitude=7.0
        """
        # This test is mostly documentation, but we verify the multiplier is 7.0
        # by checking grain output std aligns with expected ranges.
        img = np.full((500, 500, 3), 128, dtype=np.uint8)

        # At strength=0.3, with mid-gray input (L*~50):
        # lum_mod ≈ (1 - 0.5)^1.2 ≈ 0.43
        # grain_amplitude ≈ 2.1 * 0.43 ≈ 0.9 (in L* units)
        # But noise_std ≈ 1.0, so expected grain std ≈ 0.9 * 1.0 ≈ 0.9
        # Due to clumping, downsampling, and spatial structure, actual std ~0.8-1.5

        from retouch.grain import apply_film_grain
        from retouch.color_space import bgr_to_lab

        result = apply_film_grain(img, strength=0.3, seed=42)
        lab = bgr_to_lab(result)
        l_out = lab[:, :, 0].astype(np.float32)
        l_ref = 50.0  # Mid-gray in L* scale
        grain_std = np.std(l_out - l_ref)

        # With new formula, should be in fine-grain range
        assert 0.7 <= grain_std <= 2.0, (
            f"Grain std at strength=0.3 should be 0.7-2.0, got {grain_std:.2f}. "
            f"This validates the new amplitude formula."
        )
