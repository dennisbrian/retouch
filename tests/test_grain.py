"""Tests for retouch/grain.py — organic, clumped, luminance-correlated film grain."""

import cv2
import numpy as np

from retouch.grain import (
    _generate_luminance_mask,
    apply_film_grain,
    generate_film_grain,
    grain_autocorrelation_check,
)


class TestGenerateFilmGrain:
    def test_output_shape_matches_input(self):
        np.random.seed(0)
        g = generate_film_grain((128, 256), grain_strength=0.5, clumping_sigma=1.2)
        assert g.shape == (128, 256)
        assert g.dtype == np.float32

    def test_zero_grain_strength_returns_zeros(self):
        g = generate_film_grain((64, 64), grain_strength=0.0)
        assert np.array_equal(g, np.zeros((64, 64), dtype=np.float32))
        assert g.dtype == np.float32

    def test_output_has_expected_std(self):
        np.random.seed(1)
        g = generate_film_grain((256, 256), grain_strength=0.5, clumping_sigma=1.2)
        assert 0.05 < g.std() < 0.6
        assert -1.0 <= g.min() <= -0.5
        assert 0.5 <= g.max() <= 1.0

    def test_white_noise_at_zero_clumping(self):
        np.random.seed(2)
        white = generate_film_grain((256, 256), grain_strength=0.5, clumping_sigma=0.0)
        r = grain_autocorrelation_check(white)
        assert r < 0.1

    def test_clumpy_at_higher_clumping_sigma(self):
        np.random.seed(3)
        clumpy = generate_film_grain((256, 256), grain_strength=0.5, clumping_sigma=2.0)
        r = grain_autocorrelation_check(clumpy)
        assert r > 0.2


class TestApplyFilmGrain:
    def test_zero_strength_is_identity(self):
        np.random.seed(4)
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out = apply_film_grain(img, strength=0.0)
        assert np.array_equal(out, img)
        assert out.dtype == np.uint8

    def test_output_shape_matches_input(self):
        img = np.random.randint(0, 256, (73, 97, 3), dtype=np.uint8)
        out = apply_film_grain(img, strength=0.5, seed=42)
        assert out.shape == (73, 97, 3)

    def test_output_is_uint8(self):
        img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out = apply_film_grain(img, strength=0.5, seed=99)
        assert out.dtype == np.uint8

    def test_nonzero_strength_changes_pixels(self):
        np.random.seed(5)
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        out = apply_film_grain(img, strength=1.0)
        assert not np.array_equal(out, img)
        assert out.shape == img.shape
        assert out.dtype == np.uint8

    def test_grain_only_affects_luminance(self):
        np.random.seed(6)
        img = np.full((128, 128, 3), 128, dtype=np.uint8)
        out = apply_film_grain(img, strength=1.0)
        lab_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        assert np.allclose(lab_in[:, :, 1], lab_out[:, :, 1], atol=2)
        assert np.allclose(lab_in[:, :, 2], lab_out[:, :, 2], atol=2)


class TestGrainAutocorrelationCheck:
    def test_clumpy_scores_higher_than_white(self):
        np.random.seed(7)
        white = generate_film_grain(
            (256, 256), grain_strength=0.5, clumping_sigma=0.0, shadow_boost=1.0
        )
        clumpy = generate_film_grain(
            (256, 256), grain_strength=0.5, clumping_sigma=2.0, shadow_boost=1.0
        )
        r_white = grain_autocorrelation_check(white)
        r_clumpy = grain_autocorrelation_check(clumpy)
        assert r_clumpy > r_white
        assert r_clumpy > 0.2

    def test_returns_float(self):
        np.random.seed(8)
        g = generate_film_grain((64, 64), grain_strength=0.5, clumping_sigma=1.0)
        r = grain_autocorrelation_check(g)
        assert isinstance(r, float)


class TestLuminanceMask:
    def test_mask_shape_and_range(self):
        np.random.seed(9)
        m = _generate_luminance_mask((100, 200))
        assert m.shape == (100, 200)
        assert m.dtype == np.float32
        assert 0.0 <= m.min()
        assert m.max() <= 1.0
