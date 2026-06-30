"""Tests for retouch/tonal.py — H&D film response curve.

Promoted from the AUDIT_REPORT.md §1.4 deep algorithmic verification harness.
"""

import numpy as np

from retouch.tonal import apply_hd_curve, apply_lift_gamma_gain, hd_curve_lut


class TestHDCurveLut:
    """Properties of the 256-entry H&D tone-curve LUT."""

    def test_strength_zero_is_identity(self):
        lut = hd_curve_lut(strength=0.0)
        expected = np.arange(256, dtype=np.uint8)
        assert np.array_equal(lut, expected)

    def test_strength_very_small_is_identity(self):
        lut = hd_curve_lut(strength=1e-9)
        expected = np.arange(256, dtype=np.uint8)
        assert np.array_equal(lut, expected)

    def test_monotonic_non_decreasing_across_grid(self):
        """Verify the curve never inverts for any toe/shoulder in [0, 0.5]."""
        midpoints = [0.3, 0.5, 0.7]
        gammas = [0.5, 1.0, 2.0]
        strengths = [0.3, 0.7, 1.0]
        for toe in (0.0, 0.05, 0.10, 0.25, 0.50):
            for shoulder in (0.0, 0.05, 0.10, 0.25, 0.50):
                for mp in midpoints:
                    for gm in gammas:
                        for st in strengths:
                            lut = hd_curve_lut(
                                strength=st, toe=toe, shoulder=shoulder,
                                midpoint=mp, gamma=gm,
                            )
                            violations = np.sum(np.diff(lut) < 0)
                            assert violations == 0, (
                                f"Non-monotonic at toe={toe} shoulder={shoulder} "
                                f"mp={mp} gamma={gm} strength={st}"
                            )

    def test_full_strength_differs_from_identity(self):
        lut = hd_curve_lut(strength=1.0, toe=0.10, shoulder=0.10)
        identity = np.arange(256, dtype=np.uint8)
        assert not np.array_equal(lut, identity)

    def test_output_is_uint8_256(self):
        lut = hd_curve_lut(strength=0.7)
        assert lut.dtype == np.uint8
        assert len(lut) == 256

    def test_clipped_range(self):
        """LUT values must stay in [0, 255]."""
        lut = hd_curve_lut(strength=1.0, toe=0.1, shoulder=0.1)
        assert lut.min() >= 0
        assert lut.max() <= 255


class TestApplyHDCurve:
    """Integration: apply_hd_curve on a real image array."""

    def test_noop_at_strength_zero(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = apply_hd_curve(img, strength=0.0)
        assert np.array_equal(result, img)

    def test_luma_only_preserves_shape(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = apply_hd_curve(img, strength=0.7, luma_only=True)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_per_channel_preserves_shape(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = apply_hd_curve(img, strength=0.7, luma_only=False)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_rejects_bad_shape(self):
        import pytest
        with pytest.raises(ValueError):
            apply_hd_curve(np.zeros((32, 32), dtype=np.uint8))

    def test_gray_input_rejected(self):
        import pytest
        with pytest.raises(ValueError):
            apply_hd_curve(np.zeros((32, 32, 1), dtype=np.uint8))


class TestLiftGammaGain:
    def test_all_zero_is_noop(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = apply_lift_gamma_gain(img, lift=0.0, gamma=1.0, gain=1.0)
        assert np.array_equal(result, img)

    def test_preserves_shape(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        result = apply_lift_gamma_gain(img, lift=0.1)
        assert result.shape == img.shape

    def test_lift_alone_modifies_image(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = apply_lift_gamma_gain(img, lift=0.2, luma_only=True)
        # Should brighten shadows, not be identity
        assert not np.array_equal(result, img)

    def test_gain_alone_modifies_image(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = apply_lift_gamma_gain(img, gain=1.5, luma_only=True)
        assert not np.array_equal(result, img)
