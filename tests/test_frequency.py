"""Tests for retouch/frequency.py — frequency separation and recombination."""

import numpy as np
import cv2
import pytest

from retouch.frequency import (
    FrequencyLayers,
    separate,
    combine,
    DEFAULT_FEATHER_MIN,
)


class TestFrequencyLayers:
    def test_reconstruct(self):
        low = np.zeros((10, 10, 3), dtype=np.float32)
        mid = np.zeros((10, 10, 3), dtype=np.float32)
        high = np.zeros((10, 10, 3), dtype=np.float32)
        layers = FrequencyLayers(low, mid, high)
        recon = layers.reconstruct()
        assert recon.dtype == np.uint8
        assert recon.shape == (10, 10, 3)

    def test_reconstruction_preserves_value(self):
        img = np.full((20, 20, 3), 100, dtype=np.uint8)
        layers = separate(img, face_width=100)
        recon = layers.reconstruct()
        assert np.allclose(recon.astype(np.float32), img.astype(np.float32), atol=2)


class TestSeparate:
    def test_returns_frequency_layers(self):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=100)
        assert isinstance(layers, FrequencyLayers)
        assert layers.low.shape == (50, 50, 3)
        assert layers.mid.shape == (50, 50, 3)
        assert layers.high.shape == (50, 50, 3)

    def test_separate_is_lossless(self):
        np.random.seed(42)
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        layers = separate(img, face_width=80)
        recon = layers.reconstruct()
        err = np.abs(recon.astype(np.float32) - img.astype(np.float32)).max()
        assert err < 2.0, f"Reconstruction error too large: {err}"

    def test_low_is_smoothed(self):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        img[20:30, 20:30] = 255
        layers = separate(img, face_width=100)
        assert np.all(layers.low[20, 20] < 255)
        assert np.any(layers.high[20, 20] != 0)


class TestCombine:
    def test_none_mask_reconstructs(self):
        img = np.full((40, 40, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=80)
        result = combine(layers, skin_mask=None)
        assert np.allclose(result.astype(np.float32), img.astype(np.float32), atol=2)

    def test_zero_smooth_returns_original(self):
        img = np.random.randint(0, 256, (40, 40, 3), dtype=np.uint8)
        layers = separate(img, face_width=80)
        mask = np.ones((40, 40), dtype=np.float32)
        result = combine(layers, skin_mask=mask, smooth_strength=0, mid_reduction=0)
        assert np.allclose(result.astype(np.float32), img.astype(np.float32), atol=2)

    def test_mid_reduction_works(self):
        img = np.full((40, 40, 3), 128, dtype=np.uint8)
        img[20:22, 20:22] = 200  # small blemish-like spot
        layers = separate(img, face_width=80)
        mask = np.ones((40, 40), dtype=np.float32)
        result_before = combine(layers, skin_mask=mask, smooth_strength=0, mid_reduction=0)
        result_after = combine(layers, skin_mask=mask, smooth_strength=0, mid_reduction=1)
        assert not np.array_equal(result_before, result_after)

    def test_texture_opacity_reduces_high(self):
        img = checkerboard_100()
        layers = separate(img, face_width=100)
        mask = np.ones((100, 100), dtype=np.float32)
        result = combine(layers, skin_mask=mask, smooth_strength=0, mid_reduction=0, texture_opacity=0)
        # With 0 texture opacity, high frequencies should be suppressed
        assert not np.array_equal(result, img)

    def test_feather_mask_shape(self):
        """Ensure combine returns HxWx3 even with 2D mask."""
        img = np.full((30, 40, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=60)
        mask = np.ones((30, 40), dtype=np.float32)
        result = combine(layers, skin_mask=mask, smooth_strength=0.5, face_width=60)
        assert result.shape == (30, 40, 3)
        assert result.dtype == np.uint8

    def test_strength_zero_returns_input(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        layers = separate(img, face_width=64)
        mask = np.ones((32, 32), dtype=np.float32)
        result = combine(layers, skin_mask=mask, smooth_strength=0,
                         mid_reduction=0, texture_opacity=1.0)
        assert np.allclose(result.astype(np.float32), img.astype(np.float32), atol=2)

    def test_combine_output_dtype_shape(self):
        img = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        layers = separate(img, face_width=96)
        mask = np.ones((48, 64), dtype=np.float32)
        result = combine(layers, skin_mask=mask, smooth_strength=0.5)
        assert result.dtype == np.uint8
        assert result.shape == (48, 64, 3)


# Helpers
def checkerboard_100():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for y in range(0, 100, 10):
        for x in range(0, 100, 10):
            val = 255 if (x // 10 + y // 10) % 2 == 0 else 0
            img[y:y+10, x:x+10] = val
    return img


class TestPoreSynthesis:
    def test_pore_synthesis_changes_image(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=100)
        mask = np.ones((100, 100), dtype=np.float32)
        # Without pore synthesis
        res_no_pore = combine(layers, skin_mask=mask, smooth_strength=0, mid_reduction=0, pore_synthesis=0.0)
        # With pore synthesis
        res_with_pore = combine(
            layers, skin_mask=mask, smooth_strength=0, mid_reduction=0,
            pore_synthesis=0.5, face_width=100.0, roi_coords=(10, 20)
        )
        assert not np.array_equal(res_no_pore, res_with_pore)
        
    def test_pore_synthesis_is_deterministic(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=100)
        mask = np.ones((100, 100), dtype=np.float32)
        res1 = combine(
            layers, skin_mask=mask, smooth_strength=0, mid_reduction=0,
            pore_synthesis=0.5, face_width=100.0, roi_coords=(10, 20)
        )
        res2 = combine(
            layers, skin_mask=mask, smooth_strength=0, mid_reduction=0,
            pore_synthesis=0.5, face_width=100.0, roi_coords=(10, 20)
        )
        assert np.array_equal(res1, res2)

    def test_pore_synthesis_different_coords_different_noise(self):
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        layers = separate(img, face_width=100)
        mask = np.ones((100, 100), dtype=np.float32)
        res1 = combine(
            layers, skin_mask=mask, smooth_strength=0, mid_reduction=0,
            pore_synthesis=0.5, face_width=100.0, roi_coords=(10, 20)
        )
        res2 = combine(
            layers, skin_mask=mask, smooth_strength=0, mid_reduction=0,
            pore_synthesis=0.5, face_width=100.0, roi_coords=(10, 21)
        )
        assert not np.array_equal(res1, res2)


class TestAdaptiveTexturePreservation:
    """Adaptive protection of pore texture on flat, front-lit skin.

    The mechanism measures masked high-band energy in ``combine()`` and, when
    it is low, scales back effective smoothing and floors ``texture_opacity``
    upward. It must be smooth (no threshold pop), asymmetric (only ever reduce
    smoothing), resolution-independent (amplitude in intensity units), and a
    no-op when texture energy is normal/high.
    """

    @staticmethod
    def _robust_std(vals):
        med = np.median(vals)
        return float(np.median(np.abs(vals - med)) * 1.4826)

    def _make_layers(self, size, amp, seed=0):
        """Build FrequencyLayers with a controlled pore-texture amplitude.

        Real pore texture spans the mid ("blemish"-scale) and high ("pore"-
        scale) bands. We place correlated band-limited noise in BOTH bands and
        scale both so their robust std equals ``amp`` (intensity units). This
        mirrors reality: the guided filter smooths low+mid, so mid-band texture
        is the part that gets stripped, while the high band drives the detector.
        Low band is a flat mid-gray.
        """
        rng = np.random.default_rng(seed)
        low = np.full((size, size, 3), 150.0, dtype=np.float32)

        def band(sigma_lo, sigma_hi):
            noise = rng.standard_normal((size, size, 3)).astype(np.float32)
            hi = noise if sigma_hi <= 0 else cv2.GaussianBlur(noise, (0, 0), sigma_hi)
            lo = noise if sigma_lo <= 0 else cv2.GaussianBlur(noise, (0, 0), sigma_lo)
            b = hi - lo
            mag = np.abs(b).mean(axis=2)
            cur = self._robust_std(mag)
            if cur > 1e-6:
                b = b * (amp / cur)
            return b.astype(np.float32)

        mid = band(1.5, 4.0)   # medium-scale texture
        high = band(0.0, 1.2)  # fine pore texture
        # Rescale high so its own robust std is exactly `amp` (band() normalizes
        # to `amp` already, but the two bands are independent draws).
        return FrequencyLayers(low, mid, high)

    def _high_std(self, result, mask):
        """Std of retained high-frequency (pore) texture in the output, in mask.

        Higher means more pore texture preserved. This is the quantity the
        adaptive mechanism protects on flat/front-lit skin.
        """
        r = result.astype(np.float32)
        highband = r - cv2.GaussianBlur(r, (0, 0), 1.2)
        sel = mask > 0.5
        return float(highband[sel].std())

    def test_low_texture_retains_more_high_band(self):
        """Flat/front-lit (low-energy) skin keeps more pore texture than the
        non-adaptive baseline would leave (baseline emulated with adapt=1).

        Uses texture_opacity=0.6 (a realistic 'smoother' recipe) so the
        opacity-floor lever is exercised: on flat skin the adaptation raises
        opacity toward 1.0, preserving the high band the recipe would otherwise
        attenuate.
        """
        size = 200
        layers_low = self._make_layers(size, amp=0.6, seed=1)
        mask = np.ones((size, size), dtype=np.float32)

        import retouch.frequency as freq
        adaptive = combine(
            layers_low, skin_mask=mask, smooth_strength=0.5,
            mid_reduction=0.35, texture_opacity=0.6, face_width=float(size),
        )
        # Baseline: force adapt to 1.0 (old behaviour) via monkeypatch.
        orig = freq._texture_adaptation_factor
        freq._texture_adaptation_factor = lambda h, m: 1.0
        try:
            baseline = combine(
                layers_low, skin_mask=mask, smooth_strength=0.5,
                mid_reduction=0.35, texture_opacity=0.6, face_width=float(size),
            )
        finally:
            freq._texture_adaptation_factor = orig

        adaptive_tex = self._high_std(adaptive, mask)
        baseline_tex = self._high_std(baseline, mask)
        assert adaptive_tex > baseline_tex * 1.05, (
            f"adaptive={adaptive_tex:.4f} not >5% over baseline={baseline_tex:.4f}"
        )

    def test_high_texture_unchanged(self):
        """Normal/high-texture skin must be pixel-identical to non-adaptive
        output (adapt clamps to 1.0)."""
        size = 200
        layers_high = self._make_layers(size, amp=6.0, seed=2)
        mask = np.ones((size, size), dtype=np.float32)

        import retouch.frequency as freq
        adaptive = combine(
            layers_high, skin_mask=mask, smooth_strength=0.5,
            mid_reduction=0.35, texture_opacity=0.6, face_width=float(size),
        )
        orig = freq._texture_adaptation_factor
        freq._texture_adaptation_factor = lambda h, m: 1.0
        try:
            baseline = combine(
                layers_high, skin_mask=mask, smooth_strength=0.5,
                mid_reduction=0.35, texture_opacity=0.6, face_width=float(size),
            )
        finally:
            freq._texture_adaptation_factor = orig

        # High-energy → adapt clamps to 1.0 → identical output.
        assert np.array_equal(adaptive, baseline)

    def test_adapt_factor_monotonic_and_asymmetric(self):
        """Factor is monotonic in energy and never exceeds 1.0 (never boosts
        smoothing)."""
        import retouch.frequency as freq
        size = 160
        mask = np.ones((size, size), dtype=np.float32)
        energies = [0.3, 0.8, 1.5, 2.5, 4.0, 8.0]
        factors = []
        for i, amp in enumerate(energies):
            layers = self._make_layers(size, amp=amp, seed=100 + i)
            f = freq._texture_adaptation_factor(layers.high, mask)
            factors.append(f)
            assert f <= 1.0 + 1e-9, f"factor {f} exceeds 1.0 (would boost smoothing)"
            assert f >= freq.TEXTURE_ADAPT_FLOOR - 1e-9

        # Non-decreasing in energy (allowing tiny numerical noise).
        for a, b in zip(factors, factors[1:]):
            assert b >= a - 1e-6, f"factor not monotonic: {factors}"
        # Extremes behave as designed.
        assert factors[0] == pytest.approx(freq.TEXTURE_ADAPT_FLOOR, abs=1e-6)
        assert factors[-1] == pytest.approx(1.0, abs=1e-6)

    def test_empty_mask_no_adaptation(self):
        """Degenerate/empty mask → factor 1.0 (no crash, no adaptation)."""
        import retouch.frequency as freq
        size = 64
        layers = self._make_layers(size, amp=3.0, seed=7)
        empty = np.zeros((size, size), dtype=np.float32)
        assert freq._texture_adaptation_factor(layers.high, empty) == 1.0


class TestAdaptFactorCustomThresholds:
    """The engine's F8.0 reinjection calls _texture_adaptation_factor with
    custom energy_low/energy_high/floor. Verify the parametrization behaves."""

    def _high_with_energy(self, size, amp, seed):
        rng = np.random.default_rng(seed)
        return (rng.standard_normal((size, size, 3)).astype(np.float32) * amp)

    def test_custom_floor_zero_reaches_zero(self):
        import retouch.frequency as freq
        size = 128
        mask = np.ones((size, size), dtype=np.float32)
        # Very low energy, floor=0.0 → factor drives to 0.0.
        high = self._high_with_energy(size, amp=0.05, seed=1)
        f = freq._texture_adaptation_factor(
            high, mask, energy_low=1.0, energy_high=3.0, floor=0.0
        )
        assert f == pytest.approx(0.0, abs=1e-6)

    def test_custom_high_threshold_leaves_textured_untouched(self):
        import retouch.frequency as freq
        size = 128
        mask = np.ones((size, size), dtype=np.float32)
        # Energy above energy_high → factor 1.0 (no adaptation).
        high = self._high_with_energy(size, amp=20.0, seed=2)
        f = freq._texture_adaptation_factor(
            high, mask, energy_low=1.0, energy_high=3.0, floor=0.0
        )
        assert f == pytest.approx(1.0, abs=1e-6)

    def test_custom_thresholds_monotonic(self):
        import retouch.frequency as freq
        size = 128
        mask = np.ones((size, size), dtype=np.float32)
        factors = []
        for i, amp in enumerate([0.3, 1.2, 2.0, 4.0]):
            high = self._high_with_energy(size, amp=amp, seed=10 + i)
            factors.append(
                freq._texture_adaptation_factor(
                    high, mask, energy_low=1.0, energy_high=3.0, floor=0.0
                )
            )
        for a, b in zip(factors, factors[1:]):
            assert b >= a - 1e-6, f"not monotonic: {factors}"
        # Lower-energy input must yield a strictly smaller factor than a
        # clearly-textured one (asymmetric lift is energy-driven).
        assert factors[0] < factors[-1]
