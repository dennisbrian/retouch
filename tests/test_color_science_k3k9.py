"""Tests for K3 (gamut-aware chroma compression) and K9 (subtractive saturation).

Run: python3 -m pytest tests/test_color_science_k3k9.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from retouch.color_science import (
    apply_subtractive_saturation,
    bgr_to_oklab,
    compress_chroma_gamut,
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


def test_k9_negative_amount_desaturates():
    """Negative subtractive amount pulls chroma toward the neutral (gray) axis.

    Subtractive saturation scales chroma in OKLCh about the neutral axis while
    holding hue fixed, so a negative amount reduces chroma (desaturates) without
    any hue slide -- unlike the earlier per-channel log-density implementation,
    which pinned the brightest channel and cast warm/neutral pixels green.
    """
    # In-gamut saturated probe (uint8 BGR -- bgr_to_oklab expects [0, 255]).
    im = np.full((8, 8, 3), (40, 40, 200), np.uint8)  # saturated red
    base_oklch = oklab_to_oklch(bgr_to_oklab(im))
    out_oklch = oklab_to_oklch(bgr_to_oklab(apply_subtractive_saturation(im, -0.5)))
    base_c = float(np.mean(base_oklch[..., 1]))
    out_c = float(np.mean(out_oklch[..., 1]))
    assert out_c < base_c, f"negative amount did not desaturate: C {base_c} -> {out_c}"
    # Hue must stay put (no cast) on saturated colour.
    dh = float(np.mean(_hue_diff_deg(base_oklch[..., 2], out_oklch[..., 2])))
    assert dh < 3.0, f"negative amount shifted hue by {dh} deg"


def test_k9_no_green_cast():
    """Regression: subtractive saturation must not throw a green/hue cast.

    The earlier implementation anchored on the brightest channel's density,
    pinning that channel and sliding hue toward it -- turning warm skin and
    neutral backgrounds green. Guard the two properties that actually matter:
    (1) a neutral gray stays *exactly* neutral, and (2) saturated colours keep
    their hue. (Near-neutral skin has undefined hue at C~=0, so its large
    relative hue "drift" is imperceptible and is deliberately not asserted.)
    """
    # A neutral gray must remain exactly neutral -- the strongest anti-cast test.
    gray = np.full((8, 8, 3), 128, np.uint8)
    assert np.array_equal(apply_subtractive_saturation(gray, 0.6), gray), \
        "neutral gray was not preserved (cast introduced)"

    # Saturated, in-gamut colours must hold hue within a tight tolerance.
    # (Work in uint8 BGR -- bgr_to_oklab expects [0, 255] scale.)
    probes = {
        "red": (40, 40, 200),
        "green": (40, 180, 40),
        "blue": (190, 60, 60),
        "skin": (150, 180, 220),
    }
    for name, bgr8 in probes.items():
        im = np.full((8, 8, 3), bgr8, np.uint8)
        base_h = oklab_to_oklch(bgr_to_oklab(im))[..., 2]
        out_h = oklab_to_oklch(bgr_to_oklab(
            apply_subtractive_saturation(im, 0.6)))[..., 2]
        dh = float(np.mean(_hue_diff_deg(base_h, out_h)))
        assert dh < 3.0, f"hue slid {dh} deg on {name} (cast)"


def test_k3_finish_stage_no_longer_calls_any_gamut_mapper():
    """CS-05 regression: _stage_finish's second gamut-mapping call site was
    removed, not swapped, because everything upstream of it (selective
    sharpening, impact finish, purple-fringing removal) already clips to
    [0,1]/uint8 -- so a mapper there is a guaranteed no-op (verified
    empirically: zero pixels changed across natural/cosplay/porcelain/
    outdoor-harsh-sun recipes and a forced +100 global saturation push).
    The retired compress_chroma_gamut it used to call had an 85% knee that
    rolled off valid, already-in-gamut colors near the boundary on every
    render -- confirmed to mangle a pure BGR blue primary [255,0,0] to
    [228,49,0] (see test_gamut_compress_preserves_pure_primary below).
    grading.ColorGrader.grade()'s internal _apply_gamut_compress call is now
    the single gamut-mapping site, ahead of quantization, per CS-05's gate.
    Reading engine.py's source (rather than only calling the public API) is
    deliberate: a regression here is silent unless specifically checked for.
    """
    import inspect
    from retouch import engine as engine_mod

    src = inspect.getsource(engine_mod.RetouchEngine._stage_finish)
    assert "compress_chroma_gamut(" not in src and "import compress_chroma_gamut" not in src, (
        "_stage_finish must not call the retired 85%-knee compress_chroma_gamut"
    )
    assert "_apply_gamut_compress(" not in src, (
        "_stage_finish must not carry a second, guaranteed-no-op gamut-mapping call site"
    )


def test_gamut_compress_preserves_pure_primary_unlike_compress_chroma_gamut():
    """Direct reproduction of the CS-05 finding: on a pure BGR blue primary,
    the old compress_chroma_gamut visibly desaturates an in-gamut color,
    while gamut_compress (the unified mapper) preserves it."""
    img = np.array([[[255, 0, 0]]], dtype=np.uint8)  # pure blue, BGR
    oklch = oklab_to_oklch(bgr_to_oklab(img))

    # knee passed explicitly: this test pins compress_chroma_gamut's
    # documented default-knee behavior, not whatever its default happens to
    # be if CS-06 ever retunes/deprecates it (it has no production callers
    # after this fix).
    old_out = oklab_to_bgr(oklch_to_oklab(compress_chroma_gamut(oklch, knee=0.85)))[0, 0]
    new_out = oklab_to_bgr(oklch_to_oklab(gamut_compress(oklch)))[0, 0]

    # The old mapper visibly desaturates (green channel lifts well above 0).
    assert int(old_out[1]) > 30, "expected old compress_chroma_gamut to desaturate the primary"
    # The unified mapper preserves it within uint8 round-trip tolerance.
    assert abs(int(new_out[0]) - 255) <= 2
    assert int(new_out[1]) <= 2
    assert int(new_out[2]) <= 2


def test_ctx_gamut_compress_still_reaches_grade_after_stage_finish_removal():
    """ctx.gamut_compress must remain a live ParamSpec: removing the
    _stage_finish gamut call site must not leave the CLI flag / GUI checkbox
    controlling nothing. It is routed into settings["gamut_compress"] at two
    sites in _run_global_phases / _stage_grade, both feeding
    ColorGrader.grade() -> _apply_gamut_compress."""
    import inspect
    from retouch import engine as engine_mod

    src = inspect.getsource(engine_mod)
    assert src.count('settings["gamut_compress"] = ctx.gamut_compress') >= 1, (
        "ctx.gamut_compress must still be wired into grade()'s settings dict"
    )
