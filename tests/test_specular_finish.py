"""Tests for R12 — specular/diffuse re-render finish (retouch/specular.py)."""

from __future__ import annotations

import numpy as np
import pytest

from retouch.specular import extract_specular, render_finish, specular_recolor


def _blob_spec(h: int = 100, w: int = 100, peak: float = 200.0, sigma: float = 8.0) -> np.ndarray:
    """Synthetic single-channel specular blob (Gaussian) centered in the image."""
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2.0, w / 2.0
    d2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return (peak * np.exp(-d2 / (2.0 * sigma * sigma))).astype(np.float32)


def _blob_image(base: float = 100.0, peak: float = 150.0, sigma: float = 8.0) -> np.ndarray:
    """Mid-gray canvas with a bright (white) Gaussian highlight blob."""
    spec = _blob_spec(peak=peak, sigma=sigma)
    img = np.full((100, 100, 3), base, dtype=np.float32)
    img[..., 0] += spec
    img[..., 1] += spec
    img[..., 2] += spec
    return np.clip(img, 0.0, 255.0)


# ---------------------------------------------------------------------------
# extract_specular
# ---------------------------------------------------------------------------


def test_extract_specular_ordering():
    img = np.zeros((80, 80, 3), dtype=np.float32)
    # Bright near-white highlight region (top band), well separated from red.
    img[0:20, :] = 240.0
    # Saturated red pixel region (bottom band, far from the white band so the
    # feather blur doesn't bleed the white specular into it).
    img[60:80, :] = 0.0
    img[60:80, :, 2] = 255.0  # BGR: channel 2 = R

    spec = extract_specular(img)
    assert spec.shape == (80, 80)
    assert spec.dtype == np.float32

    white_mean = spec[5:15, :].mean()
    red_mean = spec[65:75, :].mean()
    # White highlight reads as HIGH specular; saturated red reads LOW.
    assert white_mean > 50.0
    assert red_mean < 1.0
    assert white_mean > red_mean


# ---------------------------------------------------------------------------
# render_finish modes
# ---------------------------------------------------------------------------


def test_render_finish_matte_reduces_peak():
    img = _blob_image()
    spec = _blob_spec()
    out = render_finish(img, spec, "matte", 0.5)

    assert out.dtype == np.float32
    assert out.shape == img.shape
    assert out.min() >= 0.0 and out.max() <= 255.0

    # Matte attenuates the highlight: center is pulled down toward base.
    base_center = img[50, 50].mean()
    out_center = out[50, 50].mean()
    assert out_center < base_center


def test_render_finish_dewy_glass_reboost_vs_matte():
    img = _blob_image()
    spec = _blob_spec()

    out_matte = render_finish(img, spec, "matte", 0.5)
    out_dewy = render_finish(img, spec, "dewy", 0.5)
    out_glass = render_finish(img, spec, "glass_skin", 0.5)

    peak_matte = out_matte[50, 50].mean()
    peak_dewy = out_dewy[50, 50].mean()
    peak_glass = out_glass[50, 50].mean()

    # dewy / glass re-boost the highlight above the matte-attenuated level.
    assert peak_dewy > peak_matte
    assert peak_glass > peak_matte


def test_render_finish_glass_higher_contrast_than_dewy():
    img = _blob_image()
    spec = _blob_spec()

    out_dewy = render_finish(img, spec, "dewy", 0.5)
    out_glass = render_finish(img, spec, "glass_skin", 0.5)

    mask = spec > 10.0
    # Contrast of the re-added contribution within the highlight blob.
    contrib_dewy = (out_dewy - img)[mask].std()
    contrib_glass = (out_glass - img)[mask].std()

    assert contrib_glass > contrib_dewy


def test_render_finish_unknown_mode_falls_back_to_matte():
    img = _blob_image()
    spec = _blob_spec()
    out_unknown = render_finish(img, spec, "bogus", 0.5)
    out_matte = render_finish(img, spec, "matte", 0.5)
    assert np.allclose(out_unknown, out_matte, atol=1e-3)


def test_render_finish_dtype_shape_range():
    img = _blob_image()
    spec = _blob_spec()
    for mode in ("matte", "powder", "dewy", "glass_skin"):
        out = render_finish(img, spec, mode, 0.7)
        assert out.dtype == np.float32
        assert out.shape == img.shape
        assert out.min() >= 0.0
        assert out.max() <= 255.0


# ---------------------------------------------------------------------------
# specular_recolor
# ---------------------------------------------------------------------------


def test_specular_recolor_reduces_blue_cast():
    # Mid-gray background with a blue-cast highlight region.
    # BGR order: index0=B, index1=G, index2=R. (B=255,G=200,R=200) => blue cast.
    img = np.full((40, 40, 3), 120.0, dtype=np.float32)
    hr = slice(10, 30)
    img[hr, hr, 0] = 255.0  # B
    img[hr, hr, 1] = 200.0  # G
    img[hr, hr, 2] = 200.0  # R

    spec = np.zeros((40, 40), dtype=np.float32)
    spec[hr, hr] = 220.0  # highlight region marked

    out = specular_recolor(img, spec, 1.0)

    in_br = img[hr, hr, 0] - img[hr, hr, 2]   # B - R  (blue cast, positive)
    out_br = out[hr, hr, 0] - out[hr, hr, 2]
    assert out_br.mean() < in_br.mean()       # blue cast reduced toward neutral
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# golden path
# ---------------------------------------------------------------------------


def test_render_finish_golden_matte_equivalence():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (60, 80, 3)).astype(np.float32)
    spec = rng.uniform(0, 200, (60, 80)).astype(np.float32)

    out = render_finish(img, spec, "matte", 0.5)
    expected = np.clip(img - 0.5 * spec[..., np.newaxis], 0.0, 255.0)
    assert np.allclose(out, expected, atol=1e-3)
