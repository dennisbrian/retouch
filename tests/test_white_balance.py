"""Regression coverage for the CAT16 white-balance control."""

from __future__ import annotations

import numpy as np

from retouch.white_balance import (
    _D65_XYZ,
    _RGB_TO_XYZ,
    _XYZ_TO_RGB,
    cat16_matrix,
    linear_to_srgb,
    source_white_xyz,
    srgb_to_linear,
    white_balance_cat16,
)


def _render_under_source(
    bgr: np.ndarray,
    temperature: float,
    tint: float = 0.0,
) -> np.ndarray:
    """Render a D65-referred BGR patch through a simulated source white."""
    rgb_linear = srgb_to_linear(bgr[..., ::-1].astype(np.float32) / 255.0)
    xyz = rgb_linear @ _RGB_TO_XYZ.T
    # D65-referred object values rendered under the source illuminant.
    source_matrix = cat16_matrix(_D65_XYZ, source_white_xyz(temperature, tint))
    source_xyz = xyz @ source_matrix.T
    source_rgb = source_xyz @ _XYZ_TO_RGB.T
    return (
        linear_to_srgb(source_rgb)[..., ::-1] * 255.0
    ).astype(np.float32)


class TestCat16WhiteBalance:
    def test_default_is_exact_identity_for_uint8_and_float(self):
        image_u8 = np.random.default_rng(7).integers(0, 256, (17, 19, 3), dtype=np.uint8)
        image_f = image_u8.astype(np.float32)

        out_u8 = white_balance_cat16(image_u8, temperature=6500.0, tint=0.0)
        out_f = white_balance_cat16(image_f, temperature=6500.0, tint=0.0)

        np.testing.assert_array_equal(out_u8, image_u8)
        np.testing.assert_array_equal(out_f, image_f)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_neutral_moves_in_the_kelvin_correction_direction(self):
        gray = np.full((1, 1, 3), 128, dtype=np.uint8)
        warm_source = white_balance_cat16(gray, temperature=3000.0)
        cool_source = white_balance_cat16(gray, temperature=9000.0)

        # A warm source requires a cool correction; a cool source the inverse.
        assert warm_source[0, 0, 0] > warm_source[0, 0, 2]
        assert cool_source[0, 0, 2] > cool_source[0, 0, 0]
        assert not np.array_equal(warm_source, gray)
        assert not np.array_equal(cool_source, gray)

    def test_corrects_a_simulated_neutral_cast(self):
        neutral = np.full((9, 11, 3), 146, dtype=np.uint8)
        for temperature in (3000.0, 4500.0, 8000.0, 12000.0):
            cast = _render_under_source(neutral, temperature)
            corrected = white_balance_cat16(cast, temperature)
            np.testing.assert_allclose(corrected, neutral.astype(np.float32), atol=0.35)

    def test_corrects_skin_hue_across_kelvin_sweep(self):
        # A moderately saturated skin-like patch, expressed under D65.
        skin = np.full((7, 7, 3), (92, 139, 191), dtype=np.uint8)
        for temperature in (3000.0, 4000.0, 5500.0, 8000.0, 12000.0):
            cast = _render_under_source(skin, temperature)
            corrected = white_balance_cat16(cast, temperature)
            # This measures the useful invariant: equivalent illuminants return
            # the same skin colour after their respective correction.
            np.testing.assert_allclose(corrected, skin.astype(np.float32), atol=0.55)

    def test_positive_tint_is_magenta_and_negative_is_green(self):
        gray = np.full((1, 1, 3), 128, dtype=np.uint8)
        green = white_balance_cat16(gray, temperature=6500.0, tint=-50.0)[0, 0].astype(np.int16)
        magenta = white_balance_cat16(gray, temperature=6500.0, tint=50.0)[0, 0].astype(np.int16)

        assert green[1] > green[0] and green[1] > green[2]
        assert magenta[0] + magenta[2] > 2 * magenta[1]

    def test_gamut_dtype_and_finiteness_at_control_extremes(self):
        image = np.random.default_rng(42).uniform(0.0, 255.0, (23, 29, 3)).astype(np.float32)
        for temperature, tint in ((1000.0, -200.0), (2000.0, 100.0), (50000.0, -100.0), (100000.0, 200.0)):
            out = white_balance_cat16(image, temperature=temperature, tint=tint)
            assert out.dtype == np.float32
            assert np.isfinite(out).all()
            assert float(out.min()) >= 0.0
            assert float(out.max()) <= 255.0
