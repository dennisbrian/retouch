"""Regression tests for X4 scene-linear highlight-purity compression."""

from __future__ import annotations

import numpy as np

from retouch.film import FilmDensityEngine, _master_tonemap


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _chroma_length(rgb: np.ndarray) -> np.ndarray:
    y = _luminance(rgb)[..., np.newaxis]
    return np.linalg.norm(rgb - y, axis=-1)


def _highlight_ramp() -> np.ndarray:
    """Constant-chroma, increasing-luminance ramp inside linear RGB gamut."""
    y = np.linspace(0.65, 0.90, 16, dtype=np.float32)
    # Zero-luminance direction: changing this vector cannot change Y.
    direction = np.array([1.0, -0.2, -0.9634349], dtype=np.float32)
    rgb = y[:, np.newaxis] + 0.08 * direction
    return rgb[np.newaxis, ...]


def test_below_knee_is_identity():
    rgb = np.array([[[0.40, 0.22, 0.08], [0.55, 0.47, 0.32]]], dtype=np.float32)
    assert np.all(_luminance(rgb) < 0.65)

    out = _master_tonemap(
        rgb, 0.1, 0.15, 0.5, 1.0, strength=0.0, skew=0.3, highlight_purity=1.0
    )
    assert np.array_equal(out, rgb)


def test_saturated_highlight_ramp_loses_chroma_monotonically():
    rgb = _highlight_ramp()
    out = _master_tonemap(
        rgb, 0.1, 0.15, 0.5, 1.0, strength=0.0, skew=0.0, highlight_purity=1.0
    )

    chroma = _chroma_length(out)[0]
    assert np.all(np.diff(chroma) <= 1e-6), chroma
    assert chroma[-1] < chroma[0] * 0.25


def test_highlight_purity_preserves_luminance_and_hue_direction():
    rgb = _highlight_ramp()
    out = _master_tonemap(
        rgb, 0.1, 0.15, 0.5, 1.0, strength=0.0, skew=0.7, highlight_purity=0.8
    )

    np.testing.assert_allclose(_luminance(out), _luminance(rgb), rtol=0.0, atol=2e-6)

    base_chroma = rgb - _luminance(rgb)[..., np.newaxis]
    out_chroma = out - _luminance(out)[..., np.newaxis]
    base_unit = base_chroma / np.linalg.norm(base_chroma, axis=-1, keepdims=True)
    out_unit = out_chroma / np.linalg.norm(out_chroma, axis=-1, keepdims=True)
    assert np.all(np.sum(base_unit * out_unit, axis=-1) > 0.99999)


def test_output_is_finite_and_bounded_for_out_of_range_input():
    rgb = np.array([[[-2.0, 0.3, 2.0], [1.5, 0.9, -0.1]]], dtype=np.float32)
    out = _master_tonemap(
        rgb, 0.1, 0.15, 0.5, 1.0, strength=1.0, skew=1.0, highlight_purity=1.0
    )
    assert np.isfinite(out).all()
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_default_highlight_purity_is_byte_identical_to_explicit_zero():
    rng = np.random.default_rng(8)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    params = {
        "enable": True,
        "strength": 1.0,
        "toe_r": 0.10,
        "toe_g": 0.10,
        "toe_b": 0.10,
        "shoulder_r": 0.10,
        "shoulder_g": 0.10,
        "shoulder_b": 0.10,
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
    engine = FilmDensityEngine()
    implicit_default = engine.apply(img, params)
    explicit_zero = engine.apply(img, {**params, "highlight_purity": 0.0})
    assert np.array_equal(implicit_default, explicit_zero)
