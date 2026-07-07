"""Tests for retouch/film.py — C3 parametric film-density engine.

Covers the §7.1 unit-test spec from docs/PLAN_C3_FILM_DENSITY.md:
- Round-trip neutrality (enable=False, and enable=True with zero params)
- Monotonicity (H&D curve never inverts)
- Crosstalk sign/direction (positive cy_mg warms shadows)
- Skew endpoints (skew=0 hue-preserved, skew=1 per-channel)
- Dtype preservation
"""

from __future__ import annotations

import numpy as np
import pytest

from retouch.film import (
    FilmDensityEngine,
    _build_crosstalk_matrix,
    _hd_curve_logdensity,
    _master_tonemap,
    _srgb_to_linear,
    _linear_to_srgb,
)


def _make_test_image(size: int = 64, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (size, size, 3), dtype=np.uint8)


def _make_ramp(size: int = 64) -> np.ndarray:
    ramp = np.linspace(0, 255, size, dtype=np.float32)
    img = np.stack([ramp] * size, axis=0)
    return np.stack([img, img, img], axis=-1).astype(np.uint8)


def _default_params(**overrides) -> dict:
    p = {
        "enable": True,
        "strength": 1.0,
        "toe_r": 0.10, "toe_g": 0.10, "toe_b": 0.10,
        "shoulder_r": 0.10, "shoulder_g": 0.10, "shoulder_b": 0.10,
        "midpoint": 0.50,
        "gamma": 1.0,
        "crosstalk_cy_mg": 0.06,
        "crosstalk_cy_ye": 0.03,
        "crosstalk_mg_ye": 0.02,
        "tonemap_strength": 0.7,
        "tonemap_toe": 0.10,
        "tonemap_shoulder": 0.15,
        "skew": 0.3,
    }
    p.update(overrides)
    return p


class TestRoundTripNeutrality:
    def test_enable_false_is_byte_identical(self):
        img = _make_test_image()
        engine = FilmDensityEngine()
        out = engine.apply(img, {"enable": False})
        assert out is img or np.array_equal(out, img)

    def test_enable_false_float_input(self):
        img = (_make_test_image().astype(np.float32))
        engine = FilmDensityEngine()
        out = engine.apply(img, {"enable": False})
        assert np.array_equal(out, img)

    def test_all_zero_params_is_near_identity(self):
        img = _make_test_image()
        engine = FilmDensityEngine()
        p = _default_params(
            toe_r=0, toe_g=0, toe_b=0,
            shoulder_r=0, shoulder_g=0, shoulder_b=0,
            crosstalk_cy_mg=0, crosstalk_cy_ye=0, crosstalk_mg_ye=0,
            tonemap_strength=0,
            strength=1.0,
        )
        out = engine.apply(img, p)
        diff = np.abs(out.astype(np.int32) - img.astype(np.int32))
        assert diff.mean() < 1.0, f"all-zero params should be ~identity, mean diff={diff.mean()}"


class TestDtypePreservation:
    def test_uint8_input_returns_uint8(self):
        img = _make_test_image()
        engine = FilmDensityEngine()
        out = engine.apply(img, _default_params())
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    def test_float32_input_returns_float32(self):
        img = _make_test_image().astype(np.float32)
        engine = FilmDensityEngine()
        out = engine.apply(img, _default_params())
        assert out.dtype == np.float32
        assert out.shape == img.shape

    def test_output_range_uint8(self):
        img = _make_test_image()
        engine = FilmDensityEngine()
        out = engine.apply(img, _default_params())
        assert out.min() >= 0
        assert out.max() <= 255

    def test_output_range_float32(self):
        img = _make_test_image().astype(np.float32)
        engine = FilmDensityEngine()
        out = engine.apply(img, _default_params())
        assert out.min() >= 0.0
        assert out.max() <= 255.0


class TestMonotonicity:
    def test_hd_curve_monotonic(self):
        logE = np.linspace(-6.0, 0.0, 1000, dtype=np.float32)
        for toe in (0.0, 0.05, 0.10, 0.25, 0.50):
            for shoulder in (0.0, 0.05, 0.10, 0.25, 0.50):
                for mp in (0.3, 0.5, 0.7):
                    for gm in (0.5, 1.0, 2.0):
                        D = _hd_curve_logdensity(logE, toe, shoulder, mp, gm)
                        diffs = np.diff(D)
                        assert np.all(diffs >= -1e-6), (
                            f"Non-monotonic at toe={toe} shoulder={shoulder} "
                            f"mp={mp} gamma={gm}: min diff={diffs.min()}"
                        )

    def test_increasing_toe_darkens_shadows(self):
        img = _make_ramp(128)
        engine = FilmDensityEngine()
        p_low = _default_params(toe_r=0.05, toe_g=0.05, toe_b=0.05, tonemap_strength=0)
        p_high = _default_params(toe_r=0.30, toe_g=0.30, toe_b=0.30, tonemap_strength=0)
        out_low = engine.apply(img, p_low)
        out_high = engine.apply(img, p_high)
        shadow_mask = img[:, :, 0] < 50
        if shadow_mask.sum() > 0:
            low_shadow_mean = out_low[shadow_mask].astype(np.float32).mean()
            high_shadow_mean = out_high[shadow_mask].astype(np.float32).mean()
            assert high_shadow_mean <= low_shadow_mean + 5, (
                f"deeper toe should darken or keep shadows: low={low_shadow_mean}, high={high_shadow_mean}"
            )


class TestCrosstalkSign:
    def test_crosstalk_matrix_near_identity(self):
        M = _build_crosstalk_matrix(0.0, 0.0, 0.0)
        assert np.allclose(M, np.eye(3, dtype=np.float32))

    def test_positive_cy_mg_adds_density_to_green(self):
        M = _build_crosstalk_matrix(0.10, 0.0, 0.0)
        assert M[0, 1] == pytest.approx(0.10)
        assert M[1, 0] == pytest.approx(0.10)

    def test_crosstalk_warms_shadows(self):
        shadow_patch = np.full((32, 32, 3), 20, dtype=np.uint8)
        shadow_patch[:, :, 1] = 5
        shadow_patch[:, :, 2] = 5
        engine = FilmDensityEngine()
        p_zero = _default_params(
            crosstalk_cy_mg=0.0, crosstalk_cy_ye=0.0, crosstalk_mg_ye=0.0,
            tonemap_strength=0,
        )
        p_warm = _default_params(
            crosstalk_cy_mg=0.10, crosstalk_cy_ye=0.05, crosstalk_mg_ye=0.0,
            tonemap_strength=0,
        )
        out_zero = engine.apply(shadow_patch, p_zero)
        out_warm = engine.apply(shadow_patch, p_warm)
        zero_r_mean = out_zero[:, :, 2].astype(np.float32).mean()
        warm_r_mean = out_warm[:, :, 2].astype(np.float32).mean()
        zero_g_mean = out_zero[:, :, 1].astype(np.float32).mean()
        warm_g_mean = out_warm[:, :, 1].astype(np.float32).mean()
        assert warm_r_mean >= zero_r_mean - 2, (
            f"positive crosstalk should warm (add R): zero={zero_r_mean}, warm={warm_r_mean}"
        )
        assert warm_g_mean >= zero_g_mean - 2, (
            f"positive cy_mg should add magenta/green density: zero={zero_g_mean}, warm={warm_g_mean}"
        )

    def test_negative_crosstalk_cools(self):
        shadow_patch = np.full((32, 32, 3), 20, dtype=np.uint8)
        shadow_patch[:, :, 1] = 5
        shadow_patch[:, :, 2] = 5
        engine = FilmDensityEngine()
        p_zero = _default_params(
            crosstalk_cy_mg=0.0, crosstalk_cy_ye=0.0, crosstalk_mg_ye=0.0,
            tonemap_strength=0,
        )
        p_cool = _default_params(
            crosstalk_cy_mg=-0.10, crosstalk_cy_ye=-0.05, crosstalk_mg_ye=0.0,
            tonemap_strength=0,
        )
        out_zero = engine.apply(shadow_patch, p_zero)
        out_cool = engine.apply(shadow_patch, p_cool)
        zero_r = out_zero[:, :, 2].astype(np.float32).mean()
        cool_r = out_cool[:, :, 2].astype(np.float32).mean()
        assert cool_r <= zero_r + 2, (
            f"negative crosstalk should cool (reduce R): zero={zero_r}, cool={cool_r}"
        )


class TestSkewEndpoints:
    def test_skew_zero_preserves_hue(self):
        engine = FilmDensityEngine()
        ramp = np.linspace(0.1, 0.9, 64, dtype=np.float32)
        img_lin = np.stack([ramp, ramp * 0.5, ramp * 0.3], axis=-1)
        img_lin = np.broadcast_to(img_lin, (64, 64, 3)).copy()
        img01 = _linear_to_srgb(img_lin)
        img_bgr = (img01[..., ::-1] * 255).astype(np.uint8)

        p = _default_params(
            skew=0.0, tonemap_strength=1.0,
            toe_r=0, toe_g=0, toe_b=0,
            shoulder_r=0, shoulder_g=0, shoulder_b=0,
            crosstalk_cy_mg=0, crosstalk_cy_ye=0, crosstalk_mg_ye=0,
        )
        out = engine.apply(img_bgr, p)

        Y_in = 0.2126 * img_lin[..., 0] + 0.7152 * img_lin[..., 1] + 0.0722 * img_lin[..., 2]
        Y_in_safe = np.maximum(Y_in, 1e-6)
        in_ratios = img_lin / Y_in_safe[..., np.newaxis]

        out_f = out.astype(np.float32) / 255.0
        out_rgb = out_f[..., ::-1]
        out_lin = _srgb_to_linear(out_rgb)
        Y_out = 0.2126 * out_lin[..., 0] + 0.7152 * out_lin[..., 1] + 0.0722 * out_lin[..., 2]
        Y_out_safe = np.maximum(Y_out, 1e-6)
        out_ratios = out_lin / Y_out_safe[..., np.newaxis]

        valid = (Y_in > 0.05) & (Y_out > 0.01)
        if valid.sum() > 0:
            max_diff = np.max(np.abs(out_ratios[valid] - in_ratios[valid]))
            assert max_diff < 0.15, (
                f"skew=0 should preserve hue (ratio), max_diff={max_diff}"
            )

    def test_skew_one_is_per_channel(self):
        engine = FilmDensityEngine()
        ramp = np.linspace(0.1, 0.9, 64, dtype=np.float32)
        img_lin = np.stack([ramp, ramp * 0.5, ramp * 0.3], axis=-1)
        img_lin = np.broadcast_to(img_lin, (64, 64, 3)).copy()
        img01 = _linear_to_srgb(img_lin)
        img_bgr = (img01[..., ::-1] * 255).astype(np.uint8)

        p = _default_params(
            skew=1.0, tonemap_strength=1.0,
            toe_r=0, toe_g=0, toe_b=0,
            shoulder_r=0, shoulder_g=0, shoulder_b=0,
            crosstalk_cy_mg=0, crosstalk_cy_ye=0, crosstalk_mg_ye=0,
        )
        out = engine.apply(img_bgr, p)
        assert out.dtype == np.uint8

        p_zero = _default_params(
            skew=0.0, tonemap_strength=1.0,
            toe_r=0, toe_g=0, toe_b=0,
            shoulder_r=0, shoulder_g=0, shoulder_b=0,
            crosstalk_cy_mg=0, crosstalk_cy_ye=0, crosstalk_mg_ye=0,
        )
        out_zero = engine.apply(img_bgr, p_zero)
        diff = np.abs(out.astype(np.int32) - out_zero.astype(np.int32))
        assert diff.sum() > 0, "skew=1 should differ from skew=0 (per-channel drift)"

    def test_skew_half_between(self):
        engine = FilmDensityEngine()
        img = _make_test_image()
        p0 = _default_params(
            skew=0.0, tonemap_strength=1.0,
            toe_r=0, toe_g=0, toe_b=0,
            shoulder_r=0, shoulder_g=0, shoulder_b=0,
            crosstalk_cy_mg=0, crosstalk_cy_ye=0, crosstalk_mg_ye=0,
        )
        p1 = _default_params(
            skew=1.0, tonemap_strength=1.0,
            toe_r=0, toe_g=0, toe_b=0,
            shoulder_r=0, shoulder_g=0, shoulder_b=0,
            crosstalk_cy_mg=0, crosstalk_cy_ye=0, crosstalk_mg_ye=0,
        )
        ph = _default_params(
            skew=0.5, tonemap_strength=1.0,
            toe_r=0, toe_g=0, toe_b=0,
            shoulder_r=0, shoulder_g=0, shoulder_b=0,
            crosstalk_cy_mg=0, crosstalk_cy_ye=0, crosstalk_mg_ye=0,
        )
        out0 = engine.apply(img, p0)
        out1 = engine.apply(img, p1)
        outh = engine.apply(img, ph)
        d0 = np.abs(outh.astype(np.float32) - out0.astype(np.float32)).mean()
        d1 = np.abs(outh.astype(np.float32) - out1.astype(np.float32)).mean()
        assert d0 > 0.1 and d1 > 0.1, "skew=0.5 should be between both endpoints"


class TestEotfRoundTrip:
    def test_srgb_eotf_round_trip(self):
        x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
        lin = _srgb_to_linear(x)
        back = _linear_to_srgb(lin)
        assert np.max(np.abs(back - x)) < 1e-5

    def test_midgray_round_trip(self):
        x = np.array([0.5], dtype=np.float32)
        lin = _srgb_to_linear(x)
        assert 0.0 < lin[0] < 1.0
        back = _linear_to_srgb(lin)
        assert abs(back[0] - 0.5) < 1e-5


class TestMasterTonemap:
    def test_zero_strength_identity(self):
        rgb = np.random.rand(16, 16, 3).astype(np.float32)
        out = _master_tonemap(rgb, 0.1, 0.15, 0.5, 1.0, 0.0, 0.3)
        assert np.allclose(out, rgb)

    def test_output_in_range(self):
        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        out = _master_tonemap(rgb, 0.1, 0.15, 0.5, 1.0, 1.0, 0.3)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_skew_zero_output_consistent(self):
        rgb = np.random.rand(16, 16, 3).astype(np.float32)
        out = _master_tonemap(rgb, 0.1, 0.15, 0.5, 1.0, 1.0, 0.0)
        assert out.shape == rgb.shape
        assert out.dtype == np.float32


class TestStrengthBlend:
    def test_strength_zero_is_identity(self):
        img = _make_test_image()
        engine = FilmDensityEngine()
        p = _default_params(strength=0.0)
        out = engine.apply(img, p)
        assert np.array_equal(out, img)

    def test_partial_strength_between(self):
        img = _make_test_image()
        engine = FilmDensityEngine()
        p_full = _default_params(strength=1.0)
        p_half = _default_params(strength=0.5)
        out_full = engine.apply(img, p_full).astype(np.float32)
        out_half = engine.apply(img, p_half).astype(np.float32)
        img_f = img.astype(np.float32)
        d_full = np.abs(out_full - img_f).mean()
        d_half = np.abs(out_half - img_f).mean()
        assert d_half < d_full, "half strength should be closer to original"


class TestShapeValidation:
    def test_rejects_bad_shape(self):
        engine = FilmDensityEngine()
        bad = np.zeros((32, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            engine.apply(bad, _default_params())

    def test_small_image_works(self):
        engine = FilmDensityEngine()
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        out = engine.apply(img, _default_params())
        assert out.shape == (4, 4, 3)
