"""Tests for retouch/color_space.py wide-gamut conversion utilities."""

from __future__ import annotations

import numpy as np

from retouch.color_space import (
    bgr_to_prophoto,
    prophoto_to_bgr,
    bgr_to_adobe_rgb,
    adobe_rgb_to_bgr,
    estimate_gamut,
    srgb_to_xyz_matrix,
    xyz_to_prophoto_matrix,
)


class TestMatrixHelpers:
    def test_srgb_to_xyz_shape_and_dtype(self):
        m = srgb_to_xyz_matrix()
        assert m.shape == (3, 3)
        assert m.dtype == np.float32

    def test_xyz_to_prophoto_shape_and_dtype(self):
        m = xyz_to_prophoto_matrix()
        assert m.shape == (3, 3)
        assert m.dtype == np.float32

    def test_srgb_to_xyz_spec_values(self):
        m = srgb_to_xyz_matrix()
        assert abs(float(m[0, 0]) - 0.4124564) < 1e-6
        assert abs(float(m[1, 1]) - 0.7151522) < 1e-6
        assert abs(float(m[2, 2]) - 0.9503041) < 1e-6

    def test_xyz_to_prophoto_spec_values(self):
        m = xyz_to_prophoto_matrix()
        assert abs(float(m[0, 0]) - 1.3459) < 1e-3
        assert abs(float(m[1, 1]) - 1.5082) < 1e-3
        assert abs(float(m[2, 2]) - 1.2118) < 1e-3

    def test_matrices_are_copies(self):
        m1 = srgb_to_xyz_matrix()
        m2 = srgb_to_xyz_matrix()
        m1[0, 0] = 99.0
        assert m2[0, 0] != 99.0


class TestProPhotoRoundTrip:
    def test_bgr_to_prophoto_shape_and_dtype(self):
        bgr = np.random.default_rng(7).integers(0, 256, (10, 12, 3), dtype=np.uint8)
        pp = bgr_to_prophoto(bgr)
        assert pp.shape == bgr.shape
        assert pp.dtype == np.float32

    def test_prophoto_to_bgr_shape_and_dtype(self):
        pp = np.random.default_rng(7).uniform(0.0, 1.0, (10, 12, 3)).astype(np.float32)
        bgr = prophoto_to_bgr(pp)
        assert bgr.shape == pp.shape
        assert bgr.dtype == np.uint8

    def test_round_trip_random_uint8(self):
        rng = np.random.default_rng(42)
        bgr = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        pp = bgr_to_prophoto(bgr)
        recovered = prophoto_to_bgr(pp)
        assert np.allclose(recovered, bgr, atol=2)

    def test_white_point_round_trip(self):
        bgr = np.array([[[255, 255, 255]]], dtype=np.uint8)
        recovered = prophoto_to_bgr(bgr_to_prophoto(bgr))
        assert np.allclose(recovered, bgr, atol=1)

    def test_black_point_round_trip(self):
        bgr = np.array([[[0, 0, 0]]], dtype=np.uint8)
        recovered = prophoto_to_bgr(bgr_to_prophoto(bgr))
        assert np.allclose(recovered, bgr, atol=1)

    def test_pure_blue_round_trip(self):
        bgr = np.array([[[255, 0, 0]]], dtype=np.uint8)
        recovered = prophoto_to_bgr(bgr_to_prophoto(bgr))
        assert np.allclose(recovered, bgr, atol=2)

    def test_pure_red_round_trip(self):
        bgr = np.array([[[0, 0, 255]]], dtype=np.uint8)
        recovered = prophoto_to_bgr(bgr_to_prophoto(bgr))
        assert np.allclose(recovered, bgr, atol=2)

    def test_does_not_mutate_input(self):
        bgr = np.array([[[100, 150, 200]]], dtype=np.uint8)
        snapshot = bgr.copy()
        _ = bgr_to_prophoto(bgr)
        assert np.array_equal(bgr, snapshot)

    def test_accepts_float_input(self):
        fimg = np.array([[[0.5, 0.25, 0.75]]], dtype=np.float32)
        pp = bgr_to_prophoto(fimg)
        assert pp.shape == fimg.shape
        assert pp.dtype == np.float32
        recovered = prophoto_to_bgr(pp)
        assert np.allclose(recovered, np.array([[[128, 64, 191]]], dtype=np.uint8), atol=2)

    def test_accepts_uint16_input(self):
        u16 = np.array([[[32768, 16384, 49152]]], dtype=np.uint16)
        pp = bgr_to_prophoto(u16)
        assert pp.shape == u16.shape
        assert pp.dtype == np.float32


class TestAdobeRGBRoundTrip:
    def test_bgr_to_adobe_shape_and_dtype(self):
        bgr = np.random.default_rng(7).integers(0, 256, (10, 12, 3), dtype=np.uint8)
        ar = bgr_to_adobe_rgb(bgr)
        assert ar.shape == bgr.shape
        assert ar.dtype == np.float32

    def test_adobe_to_bgr_shape_and_dtype(self):
        ar = np.random.default_rng(7).uniform(0.0, 1.0, (10, 12, 3)).astype(np.float32)
        bgr = adobe_rgb_to_bgr(ar)
        assert bgr.shape == ar.shape
        assert bgr.dtype == np.uint8

    def test_round_trip_random_uint8(self):
        rng = np.random.default_rng(42)
        bgr = rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)
        ar = bgr_to_adobe_rgb(bgr)
        recovered = adobe_rgb_to_bgr(ar)
        assert np.allclose(recovered, bgr, atol=2)

    def test_white_point_round_trip(self):
        bgr = np.array([[[255, 255, 255]]], dtype=np.uint8)
        recovered = adobe_rgb_to_bgr(bgr_to_adobe_rgb(bgr))
        assert np.allclose(recovered, bgr, atol=1)

    def test_black_point_round_trip(self):
        bgr = np.array([[[0, 0, 0]]], dtype=np.uint8)
        recovered = adobe_rgb_to_bgr(bgr_to_adobe_rgb(bgr))
        assert np.allclose(recovered, bgr, atol=1)


class TestWhitePointConsistency:
    def test_prophoto_white_is_finite_and_positive(self):
        bgr = np.array([[[255, 255, 255]]], dtype=np.uint8)
        pp = bgr_to_prophoto(bgr)
        assert np.all(np.isfinite(pp))
        assert np.all(pp > 0.0)

    def test_adobe_white_is_finite_and_positive(self):
        bgr = np.array([[[255, 255, 255]]], dtype=np.uint8)
        ar = bgr_to_adobe_rgb(bgr)
        assert np.all(np.isfinite(ar))
        assert np.all(ar > 0.0)

    def test_white_point_preserved_after_round_trip_prophoto(self):
        bgr = np.array([[[255, 255, 255]]], dtype=np.uint8)
        pp = bgr_to_prophoto(bgr)
        recovered = prophoto_to_bgr(pp)
        assert recovered[0, 0, 0] >= 253
        assert recovered[0, 0, 1] >= 253
        assert recovered[0, 0, 2] >= 253

    def test_white_point_preserved_after_round_trip_adobe(self):
        bgr = np.array([[[255, 255, 255]]], dtype=np.uint8)
        ar = bgr_to_adobe_rgb(bgr)
        recovered = adobe_rgb_to_bgr(ar)
        assert recovered[0, 0, 0] >= 253
        assert recovered[0, 0, 1] >= 253
        assert recovered[0, 0, 2] >= 253


class TestEstimateGamut:
    def test_uint8_normal_image_returns_srgb(self):
        rng = np.random.default_rng(42)
        bgr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
        assert estimate_gamut(bgr) == "srgb"

    def test_uint8_grayscale_returns_srgb(self):
        bgr = np.full((8, 8, 3), 128, dtype=np.uint8)
        assert estimate_gamut(bgr) == "srgb"

    def test_uint8_pure_colors_return_srgb(self):
        for pixel in [[[0, 0, 0]], [[255, 255, 255]], [[255, 0, 0]], [[0, 255, 0]], [[0, 0, 255]]]:
            bgr = np.array(pixel, dtype=np.uint8)
            assert estimate_gamut(bgr) == "srgb"

    def test_uint16_image_returns_prophoto(self):
        rng = np.random.default_rng(42)
        bgr = rng.integers(0, 65536, (16, 16, 3), dtype=np.uint16)
        assert estimate_gamut(bgr) == "prophoto"

    def test_uint16_uniform_white_returns_prophoto(self):
        bgr = np.full((8, 8, 3), 32768, dtype=np.uint16)
        assert estimate_gamut(bgr) == "prophoto"

    def test_float_clipped_returns_prophoto(self):
        bgr = np.array([[[1.5, 0.5, 0.5]]], dtype=np.float32)
        assert estimate_gamut(bgr) == "prophoto"

    def test_float_in_range_returns_srgb(self):
        bgr = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        assert estimate_gamut(bgr) == "srgb"

    def test_float_saturated_returns_srgb(self):
        bgr = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        assert estimate_gamut(bgr) == "srgb"

    def test_returns_known_string(self):
        for dtype, shape in [
            (np.uint8, (4, 4, 3)),
            (np.uint16, (4, 4, 3)),
            (np.float32, (4, 4, 3)),
        ]:
            img = np.zeros(shape, dtype=dtype)
            result = estimate_gamut(img)
            assert result in ("srgb", "adobe_rgb", "prophoto")
