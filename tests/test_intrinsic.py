"""Tests for R9 — Intrinsic decomposition (albedo x shading).

Proves form (shading) and pigment (albedo) are separated, that the decomposition
closes (energy conservation), that even_albedo evens pigment while preserving
form, and that the API is dtype/shape correct.
"""

from __future__ import annotations

import numpy as np
import cv2

from retouch.intrinsic import decompose_intrinsic, even_albedo


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    da = a - a.mean()
    db = b - b.mean()
    denom = np.sqrt((da * da).sum() * (db * db).sum()) + 1e-12
    return float((da * db).sum() / denom)


def _make_checker_x_ramp(h: int = 256, w: int = 256) -> np.ndarray:
    """uint8 BGR = alternating-gray checker albedo x smooth horizontal ramp.

    Albedo is constant *within* each cell (pigment). Shading is a smooth
    left-dark -> right-bright ramp with no high-frequency content (form).
    """
    yy, xx = np.mgrid[0:h, 0:w]
    cells = 4
    cell_h, cell_w = h // cells, w // cells
    checker = np.zeros((h, w), dtype=np.float32)
    for i in range(cells):
        for j in range(cells):
            val = 0.6 if (i + j) % 2 == 0 else 0.8
            checker[
                i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w
            ] = val
    ramp = np.linspace(0.4, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    img = np.clip(checker * ramp, 0.0, 1.0)
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return (img3 * 255.0).astype(np.uint8)


def test_decompose_separates_form_from_pigment():
    img = _make_checker_x_ramp()
    albedo, shading = decompose_intrinsic(img)

    # True shading is the horizontal ramp (varies only with x).
    true_ramp = np.linspace(0.4, 1.0, 256, dtype=np.float32)[None, :].repeat(
        256, axis=0
    )
    # Shading is returned mean-normalized; correlation is scale-invariant.
    corr = _pearson(shading, true_ramp)
    assert corr > 0.9, f"shading not correlated with ramp (r={corr:.3f})"

    # Recovered albedo must be ~constant within a single checker cell.
    # Cell (0,0) interior, away from edges where WLS can halo.
    cell = albedo[16:48, 16:48, 0]
    assert float(cell.std()) < 0.05, (
        f"albedo not constant within cell (std={cell.std():.4f})"
    )


def test_energy_conservation_closes():
    img = _make_checker_x_ramp()
    albedo, shading = decompose_intrinsic(img)

    # Luminance of the input (float [0,1]).
    img_f = img.astype(np.float32) / 255.0
    rgb = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
    L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    # luminance(albedo) * shading should reconstruct L.
    alb_rgb = cv2.cvtColor(albedo, cv2.COLOR_BGR2RGB)
    alb_luma = (
        0.2126 * alb_rgb[..., 0]
        + 0.7152 * alb_rgb[..., 1]
        + 0.0722 * alb_rgb[..., 2]
    )
    recon = alb_luma * shading
    assert float(np.mean(np.abs(recon - L))) < 0.01, "decomposition does not close"


def _make_blotch_x_ramp(h: int = 256, w: int = 256) -> np.ndarray:
    """uint8 BGR = broad central albedo blotch x smooth horizontal ramp."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blotch = 0.25 * np.exp(-d2 / (2.0 * (w * 0.10) ** 2))
    albedo = 0.7 + blotch  # smooth pigment patch (low-freq)
    ramp = np.linspace(0.4, 1.0, w, dtype=np.float32)[None, :].repeat(h, axis=0)
    img = np.clip(albedo * ramp, 0.0, 1.0)
    img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return (img3 * 255.0).astype(np.uint8)


def test_even_albedo_reduces_blotch_preserves_ramp():
    img = _make_blotch_x_ramp()
    _, shading0 = decompose_intrinsic(img)

    # Albedo before (per channel) — central blotch present.
    albedo0, _ = decompose_intrinsic(img)
    cy, cx = 128, 128
    bg = float(albedo0[20, 20, 0])  # away from blotch
    center0 = float(albedo0[cy, cx, 0])
    blotch0 = center0 - bg

    out = even_albedo(img, 1.0)
    albedo1, shading1 = decompose_intrinsic(out)
    center1 = float(albedo1[cy, cx, 0])
    blotch1 = center1 - bg

    # Blotch amplitude must drop.
    assert blotch1 < blotch0 * 0.8, (
        f"blotch not evened (before={blotch0:.3f}, after={blotch1:.3f})"
    )

    # Low-frequency shading ramp must be preserved (re-multiplied, untouched).
    assert _pearson(shading0, shading1) > 0.95, "form ramp altered by even_albedo"


def test_dtype_shape_and_uint8_input():
    img_u8 = _make_checker_x_ramp()
    albedo, shading = decompose_intrinsic(img_u8)

    assert albedo.dtype == np.float32
    assert shading.dtype == np.float32
    assert albedo.shape == (256, 256, 3)
    assert shading.shape == (256, 256)
    assert albedo.min() >= 0.0 and albedo.max() <= 1.0

    # float32 input also accepted without raising.
    img_f = img_u8.astype(np.float32) / 255.0
    a2, s2 = decompose_intrinsic(img_f)
    assert a2.shape == (256, 256, 3)
    assert s2.shape == (256, 256)


def test_even_albedo_zero_is_identity():
    img = _make_blotch_x_ramp()
    out = even_albedo(img, 0.0)
    ref = img.astype(np.float32) / 255.0
    assert float(np.mean(np.abs(out - ref))) < 1e-4
