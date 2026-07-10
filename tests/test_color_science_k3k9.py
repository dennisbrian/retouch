"""Tests for K3 (gamut-aware chroma compression) and K9 (subtractive saturation).

Run: python3 -m pytest tests/test_color_science_k3k9.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from retouch.color_science import (
    apply_subtractive_saturation,
    bgr_to_oklab,
    gamut_compress,
    oklab_to_bgr,
    oklab_to_oklch,
    oklch_to_oklab,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_from_oklch(L, C, h, shape=(64, 64)):
    H, W = shape
    oklch = np.zeros((H, W, 3), dtype=np.float32)
    oklch[..., 0] = L
    oklch[..., 1] = C
    oklch[..., 2] = h
    oklab = oklch_to_oklab(oklch)
    return oklab_to_bgr(oklab)


def _hue_diff_deg(h0, h1):
    d = np.abs(((h0 - h1 + 180.0) % 360.0) - 180.0)
    return np.max(d)


def _lin_luma(bgr_float01):
    # sRGB -> linear -> CIE luma
    lin = np.where(
        bgr_float01 <= 0.04045,
        bgr_float01 / 12.92,
        ((bgr_float01 + 0.055) / 1.055) ** 2.4,
    )
    r, g, b = lin[..., 2], lin[..., 1], lin[..., 0]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# ---------------------------------------------------------------------------
# K3 — gamut-aware chroma compression
# ---------------------------------------------------------------------------

def test_k3_hue_preserved_on_saturated_push():
    """Hard-pushed out-of-gamut colors keep their hue after compression."""
    for h in [0.0, 30.0, 90.0, 210.0, 300.0]:
        bgr = _bgr_from_oklch(L=0.7, C=0.5, h=h)  # C=0.5 is far out of gamut
        oklch = oklab_to_oklch(bgr_to_oklab(bgr))
        out = gamut_compress(oklch)
        dh = _hue_diff_deg(oklch[..., 2], out[..., 2])
        assert dh < 1e-3, f"hue shifted by {dh} deg at h={h}"


def test_k3_no_clip_hue_shift_across_saturation_sweep():
    """Across a chroma sweep, gamut_compress preserves hue (unlike hard clip)."""
    H, W = 64, 256
    hs = np.linspace(0, 360, W, endpoint=False)
    sweep = np.zeros((H, W, 3), dtype=np.float32)
    sweep[..., 0] = 0.65
    sweep[..., 1] = np.linspace(0.02, 0.5, W)  # sweep chroma past the boundary
    sweep[..., 2] = hs[None, :]

    oklch = sweep
    out = gamut_compress(oklch)
    dh = _hue_diff_deg(oklch[..., 2], out[..., 2])
    assert dh < 1e-3, f"max hue drift across sweep = {dh} deg"

    # Hard clip (naive) WOULD shift hue -- show the contrast.
    bgr = oklab_to_bgr(oklch_to_oklab(oklch))
    clipped_oklch = oklab_to_oklch(bgr_to_oklab(bgr))
    dh_clip = _hue_diff_deg(oklch[..., 2], clipped_oklch[..., 2])
    # Naive clip must produce a measurable hue shift somewhere (sanity of test).
    assert dh_clip > 1e-2


def test_k3_identity_on_ingamut_image():
    """An already-in-gamut image is ~unchanged (max diff < 1e-3)."""
    rng = np.random.default_rng(1)
    # A random sRGB image in [0,1] is, by definition, fully in-gamut.
    H, W = 128, 128
    bgr = (rng.random((H, W, 3))).astype(np.float32)
    oklch = oklab_to_oklch(bgr_to_oklab(bgr))

    out = gamut_compress(oklch)
    dL = np.max(np.abs(out[..., 0] - oklch[..., 0]))
    dC = np.max(np.abs(out[..., 1] - oklch[..., 1]))
    dh = _hue_diff_deg(oklch[..., 2], out[..., 2])
    assert max(dL, dC, dh) < 1e-3, f"identity violated: dL={dL} dC={dC} dh={dh}"


def test_k3_thr1_is_identity():
    """thr=1.0 triggers no compression -> byte-identical to input."""
    rng = np.random.default_rng(2)
    x = rng.random((32, 32, 3)).astype(np.float32)
    bgr = (np.clip(x, 0, 1) * 255).astype(np.uint8)
    oklch = oklab_to_oklch(bgr_to_oklab(bgr))
    out = gamut_compress(oklch, thr=1.0)
    d = np.max(np.abs(out - oklch))
    assert d < 1e-6, f"thr=1.0 not identity: {d}"


# ---------------------------------------------------------------------------
# K9 — subtractive (film-density) saturation
# ---------------------------------------------------------------------------

def test_k9_amount_zero_is_identity():
    rng = np.random.default_rng(3)
    bgr = (rng.random((64, 64, 3)) * 255).astype(np.uint8)
    out = apply_subtractive_saturation(bgr, 0.0)
    assert np.array_equal(out, bgr)


def test_k9_luma_drops_as_chroma_rises():
    """Subtractive saturation darkens saturated colors as chroma increases."""
    chromas = np.linspace(0.04, 0.35, 8)
    lums = []
    for C in chromas:
        bgr = _bgr_from_oklch(L=0.7, C=float(C), h=30.0).astype(np.float32) / 255.0
        out = apply_subtractive_saturation(bgr, 0.6)
        lums.append(float(np.mean(_lin_luma(out))))
    lums = np.array(lums)
    # Luminance drops overall as chroma rises (film look), with a clear
    # negative trend from the lowest to the highest chroma.
    assert lums[-1] < lums[0], f"luma did not drop with chroma: {lums}"
    assert np.mean(np.diff(lums)) < 0, f"luma trend not downward: {lums}"


def test_k9_additive_holds_luma_flat():
    """Additive saturation (LAB/HSV scaling) holds luminance roughly flat."""
    from retouch.grading import ColorGrader
    grader = ColorGrader()
    chromas = np.linspace(0.04, 0.35, 8)
    lums = []
    for C in chromas:
        bgr = _bgr_from_oklch(L=0.7, C=float(C), h=30.0)
        out = grader._F_adjust_saturation(bgr.astype(np.float32) / 255.0, 0.6)
        out = np.clip(out, 0, 1).astype(np.float32)
        lums.append(float(np.mean(_lin_luma(out))))
    lums = np.array(lums)
    # Additive should not produce a strong downward luma trend with chroma.
    assert abs(np.mean(np.diff(lums))) < 0.02, f"additive luma drift: {lums}"


def test_k9_negative_amount_lightens():
    """Negative subtractive amount lightens a saturated color."""
    bgr = _bgr_from_oklch(L=0.7, C=0.3, h=30.0).astype(np.float32) / 255.0
    base_lum = float(np.mean(_lin_luma(bgr)))
    light = apply_subtractive_saturation(bgr, -0.5)
    light_lum = float(np.mean(_lin_luma(light)))
    assert light_lum > base_lum, f"negative amount did not lighten: {base_lum} -> {light_lum}"
