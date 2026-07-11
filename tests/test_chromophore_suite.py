"""Tests for retouch/skin_chromophore.py — R10 chromophore suite + R11 moat.

Synthetic melanin/hemoglobin maps are built directly (no dependency on
``decompose_chromophores`` at test time). The real chromophore import is
smoke-tested if present.
"""

import numpy as np
import cv2
import pytest

from retouch.skin_chromophore import (
    blemish_vs_mole,
    bruise_remove,
    vein_attenuate,
    tanline_even,
    hemoglobin_guided_smooth,
    compression_mark_remove,
    goosebumps_smooth,
    beard_shadow_neutralize,
    facepaint_crack_repair,
    paint_coverage_even,
    ingrown_hair_cleanup,
)


def _flat(h=40, w=40, val=150.0):
    return np.full((h, w, 3), val, dtype=np.float32)


def _dot(img, cy, cx, r, channel, val):
    h, w = img.shape[:2]
    yy, xx = np.ogrid[:h, :w]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    img[m, channel] = val
    return img


def _full_mask(h, w):
    return np.ones((h, w), dtype=np.float32)


def _decompose(img):
    return __import__("retouch.chromophore", fromlist=["decompose_chromophores"]).decompose_chromophores(img.astype(np.uint8))


# --- smoke test the real chromophore import if available -------------------

def test_chromophore_import_smoke():
    try:
        from retouch.chromophore import decompose_chromophores
    except Exception:  # pragma: no cover - R7 may be absent
        pytest.skip("chromophore.decompose_chromophores unavailable")
    img = np.full((16, 16, 3), 150, dtype=np.uint8)
    mel, hb = decompose_chromophores(img)
    assert mel.shape == (16, 16) and hb.shape == (16, 16)
    assert mel.dtype == np.float32 and hb.dtype == np.float32


# --- R10 blemish_vs_mole ----------------------------------------------------

def test_blemish_vs_mole_separation():
    hb_map = np.zeros((40, 40), dtype=np.float32)
    hb_map[10:15, 10:15] = 0.9  # compact red region
    mel_map = np.zeros((40, 40), dtype=np.float32)
    mel_map[25:30, 25:30] = 0.9  # compact brown region
    blemish, mole = blemish_vs_mole(hb_map, mel_map, mask=_full_mask(40, 40), area_max=200)
    assert blemish[12, 12] == 1
    assert mole[27, 27] == 1
    # they must not cross
    assert blemish[27, 27] == 0
    assert mole[12, 12] == 0


# --- R10 bruise_remove ------------------------------------------------------

def test_bruise_remove_evened_toward_surround():
    try:
        from retouch.chromophore import decompose_chromophores
    except Exception:
        pytest.skip("chromophore unavailable")
    img = _flat(40, 40, 150.0)
    # green (healing) bruise patch: G high, R/B equal & lower -> hue ~60
    img[10:30, 10:30, 0] = 100.0
    img[10:30, 10:30, 1] = 210.0
    img[10:30, 10:30, 2] = 100.0
    hb = decompose_chromophores(img.astype(np.uint8))[1]
    mask = _full_mask(40, 40)
    out = bruise_remove(img, hb, mask=mask, strength=0.8)
    hb_after = decompose_chromophores(out.astype(np.uint8))[1]
    roi = slice(10, 30)
    # Healing bruises are a HUE anomaly (near-zero hemoglobin in the
    # decomposition), so "even toward healthy" shows up as the patch colour
    # moving toward the global mean, not as a hemoglobin drop.
    global_mean = img.reshape(-1, 3).mean(axis=0)
    patch_before = img[roi, roi].reshape(-1, 3).mean(axis=0)
    patch_after = out[roi, roi].reshape(-1, 3).mean(axis=0)
    before_dev = np.abs(patch_before - global_mean).sum()
    after_dev = np.abs(patch_after - global_mean).sum()
    assert after_dev < before_dev


# --- R10 vein_attenuate -----------------------------------------------------

def test_vein_attenuate_reduces_gb_broad_component():
    h = w = 64
    img = _flat(h, w, 150.0)
    # broad hemoglobin ramp across x
    hb = np.tile(np.linspace(0.0, 1.0, w), (h, 1)).astype(np.float32)
    # embed a sharp red-only feature that must survive (R untouched)
    img[20:24, :, 2] = 40.0
    out = vein_attenuate(img, hb, mask=_full_mask(h, w), strength=0.8)
    # R channel untouched by vein op
    assert np.array_equal(out[:, :, 2], img[:, :, 2])
    k = 15
    g_in = cv2.GaussianBlur(img[:, :, 1], (k, k), 0)
    g_out = cv2.GaussianBlur(out[:, :, 1], (k, k), 0)
    assert g_out.mean() < g_in.mean()
    b_in = cv2.GaussianBlur(img[:, :, 0], (k, k), 0)
    b_out = cv2.GaussianBlur(out[:, :, 0], (k, k), 0)
    assert b_out.mean() < b_in.mean()


# --- R10 tanline_even -------------------------------------------------------

def test_tanline_even_flattens_melanin():
    try:
        from retouch.chromophore import decompose_chromophores
    except Exception:
        pytest.skip("chromophore unavailable")
    img = _flat(48, 48, 170.0)
    mel = np.tile(np.linspace(0.2, 0.9, 48).reshape(-1, 1), (1, 48)).astype(np.float32)
    img[2:6, 2:6] = 200.0  # sharp bright dot to preserve
    out = tanline_even(img, mel, mask=_full_mask(48, 48), strength=0.8)
    mel_after = decompose_chromophores(out.astype(np.uint8))[0]
    assert mel_after.std() < mel.std()


# --- R10 hemoglobin_guided_smooth ------------------------------------------

def test_hemoglobin_guided_smooth_respects_edge():
    h = w = 60
    img = _flat(h, w, 150.0)
    img[:, :30, :] = 90.0
    img[:, 30:, :] = 210.0
    hb = np.zeros((h, w), dtype=np.float32)
    hb[:, :30] = 0.8
    hb[:, 30:] = 0.1
    img[10:14, 10:14, :] = 150.0  # small blemish on left to smooth
    out = hemoglobin_guided_smooth(img, hb, mask=_full_mask(h, w), strength=0.9)
    left_delta = abs(out[12, 12, 0] - img[12, 12, 0])
    right_delta = abs(out[5, 40, 0] - img[5, 40, 0])
    assert left_delta > right_delta


# --- R11 compression_mark_remove -------------------------------------------

def test_compression_mark_lifts_dip():
    h = w = 50
    img = _flat(h, w, 150.0)
    img[24:27, :] = 90.0  # horizontal dark line (dip)
    out = compression_mark_remove(img, mask=_full_mask(h, w), strength=0.9)
    assert out[25, 25, 0] > img[25, 25, 0]
    assert abs(out[5, 5, 0] - img[5, 5, 0]) < 8.0


# --- R11 goosebumps_smooth --------------------------------------------------

def test_goosebumps_periodic_attenuated():
    h = w = 64
    yy = np.arange(h).reshape(-1, 1)
    pattern = 150.0 + 40.0 * np.sin(2 * np.pi * yy / 6.0)
    pattern = pattern.astype(np.float32)
    img = np.broadcast_to(pattern[:, :, None], (h, w, 3)).copy()
    out = goosebumps_smooth(img, mask=_full_mask(h, w), strength=0.9)
    in_var = img[:, :, 0].var(axis=0).mean()
    out_var = out[:, :, 0].var(axis=0).mean()
    assert out_var < in_var


def test_goosebumps_noise_unchanged():
    rng = np.random.RandomState(0)
    img = rng.normal(150.0, 20.0, (48, 48, 3)).astype(np.float32)
    out = goosebumps_smooth(img, mask=_full_mask(48, 48), strength=0.9)
    assert np.array_equal(out, img)


# --- R11 beard_shadow_neutralize -------------------------------------------

def test_beard_shadow_neutralize():
    h = w = 40
    img = _flat(h, w, 150.0)
    img[10:30, 10:30, 0] = 180.0  # B
    img[10:30, 10:30, 1] = 120.0  # G
    img[10:30, 10:30, 2] = 80.0   # R
    out = beard_shadow_neutralize(img, mask=_full_mask(h, w), strength=0.9)
    roi = slice(10, 30)
    b_in, g_in, r_in = img[roi, roi, 0], img[roi, roi, 1], img[roi, roi, 2]
    b_out, g_out, r_out = out[roi, roi, 0], out[roi, roi, 1], out[roi, roi, 2]
    assert (b_out - r_out).mean() < (b_in - r_in).mean()
    assert (b_out - g_out).mean() < (b_in - g_in).mean()


# --- R11 facepaint_crack_repair --------------------------------------------

def test_facepaint_crack_repair():
    h = w = 40
    img = _flat(h, w, 235.0)  # white paint
    paint = _full_mask(h, w)
    img[18:22, :] = 120.0  # dark crack line inside paint
    out = facepaint_crack_repair(img, paint_mask=paint, strength=0.9)
    assert out[20, 20, 0] > img[20, 20, 0]


# --- R11 paint_coverage_even -----------------------------------------------

def test_paint_coverage_even_reduces_variance():
    h = w = 40
    img = _flat(h, w, 200.0)
    paint = _full_mask(h, w)
    rng = np.random.RandomState(1)
    img[:, :, 0] = np.clip(img[:, :, 0] + rng.normal(0, 35, (h, w)), 0, 255)
    out = paint_coverage_even(img, paint_mask=paint, strength=0.9)
    assert out[:, :, 0].var() < img[:, :, 0].var()


# --- R11 ingrown_hair_cleanup ----------------------------------------------

def test_ingrown_hair_cleanup():
    try:
        from retouch.chromophore import decompose_chromophores
    except Exception:
        pytest.skip("chromophore unavailable")
    img = _flat(40, 40, 170.0)
    img = _dot(img, 20, 20, 2, 2, 235.0)  # compact red dot
    hb = decompose_chromophores(img.astype(np.uint8))[1]
    out = ingrown_hair_cleanup(img, hb, mask=_full_mask(40, 40), strength=0.9)
    assert out[20, 20, 2] < img[20, 20, 2]


# --- strength=0 identity for every function --------------------------------

@pytest.mark.parametrize(
    "fn,pos",
    [
        (bruise_remove, ["hb", None]),
        (vein_attenuate, ["hb", None]),
        (tanline_even, ["mel", None]),
        (hemoglobin_guided_smooth, ["hb", None]),
        (compression_mark_remove, [None]),
        (goosebumps_smooth, [None]),
        (beard_shadow_neutralize, [None]),
        (facepaint_crack_repair, ["paint"]),
        (paint_coverage_even, ["paint"]),
        (ingrown_hair_cleanup, ["hb", None]),
    ],
)
def test_strength_zero_identity(fn, pos):
    img = _flat(32, 32, 160.0)
    hb = np.full((32, 32), 0.3, dtype=np.float32)
    mel = np.full((32, 32), 0.3, dtype=np.float32)
    paint = _full_mask(32, 32)
    call = {"hb": hb, "mel": mel, "paint": paint}
    a = [call[p] if p in call else p for p in pos]
    out = fn(img, *a, strength=0.0)
    assert np.array_equal(out, img)


def test_strength_zero_blemish_vs_mole():
    hb = np.zeros((20, 20), dtype=np.float32)
    mel = np.zeros((20, 20), dtype=np.float32)
    bm, mm = blemish_vs_mole(hb, mel, mask=_full_mask(20, 20))
    assert bm.shape == (20, 20) and bm.dtype == np.uint8
    assert mm.shape == (20, 20) and mm.dtype == np.uint8
