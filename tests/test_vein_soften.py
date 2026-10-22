"""Vein Softening (retouch/vein_soften.py).

Synthetic skin with planted veins: absorption in linear light (red absorbed
most), so the same vein reads fainter on darker skin, as it does in life.
"""
import cv2
import numpy as np
import pytest

from retouch.vein_soften import soften_veins

H = W = 420
FW = 300.0  # face width: vein scales 1.8-6 px


def _srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _skin(gain=1.0, colour=(0.30, 0.38, 0.62), seed=0):
    """Float BGR 0-1 skin with fine texture and a soft lighting gradient."""
    rng = np.random.default_rng(seed)
    lin = np.ones((H, W, 3), np.float32) * np.array(colour, np.float32)
    shade = np.linspace(0.85, 1.1, W, dtype=np.float32)[None, :, None]
    tex = 1.0 + 0.03 * cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.8)[..., None]
    return _lin_to_srgb(lin * shade * tex * gain).astype(np.float32)


def _line(points, sigma, width=1.0):
    prof = np.zeros((H, W), np.float32)
    pts = np.int32(np.round(np.array(points, np.float64) * 4))
    cv2.polylines(prof, [pts], False, 1.0, thickness=max(1, int(4 * sigma * width)),
                  lineType=cv2.LINE_AA, shift=2)
    prof = cv2.GaussianBlur(prof, (0, 0), sigma * 0.8)
    return prof / max(float(prof.max()), 1e-6)


def _vein_profile():
    xs = np.linspace(60, 360, 60)
    ys = 210 + 25 * np.sin(xs / 50.0)
    return _line(np.column_stack([xs, ys]), 2.2)


def _absorb(img, prof, transmit):
    """Multiply linear light by ``transmit`` (B, G, R) at the profile's peak."""
    t = 1.0 - prof[..., None] * (1.0 - np.array(transmit, np.float32))
    return _lin_to_srgb(_srgb_to_lin(img) * t).astype(np.float32)


VEIN = (0.99, 0.93, 0.86)


def _lab(x):
    return cv2.cvtColor(np.clip(x, 0, 1).astype(np.float32), cv2.COLOR_BGR2Lab)


def _signal(x, base, where):
    d = cv2.GaussianBlur(_lab(x) - _lab(base), (0, 0), 1.5)
    return float(np.linalg.norm(d, axis=2)[where].mean())


MASK = np.ones((H, W), np.float32)


def _removed(base, strength=1.0, mask=MASK, **kw):
    prof = _vein_profile()
    planted = _absorb(base, prof, VEIN)
    out = soften_veins(planted, mask, strength, FW, value_scale=1.0, **kw)
    where = prof > 0.6
    return 1.0 - _signal(out, base, where) / _signal(planted, base, where), out, planted


def test_planted_vein_is_softened_and_strength_scales():
    base = _skin()
    half, _, _ = _removed(base, 0.5)
    full, _, _ = _removed(base, 1.0)
    assert full >= 0.35
    assert 0.1 < half < full


def test_same_share_on_darker_skin():
    light, _, _ = _removed(_skin())
    dark, _, _ = _removed(_skin(gain=0.45))
    assert abs(light - dark) < 0.12


def test_unplanted_skin_is_left_alone():
    base = _skin()
    out = soften_veins(base, MASK, 1.0, FW, value_scale=1.0)
    assert np.abs(out - base).max() * 255 < 1.0


@pytest.mark.parametrize('kind', ['mole', 'crease', 'tattoo', 'speck'])
def test_marks_creases_tattoos_and_specks_are_kept(kind):
    base = _skin()
    if kind == 'mole':
        prof = np.zeros((H, W), np.float32)
        cv2.circle(prof, (210, 210), 5, 1.0, -1)
        prof = cv2.GaussianBlur(prof, (0, 0), 1.0)
        img = _absorb(base, prof / prof.max(), (0.75, 0.62, 0.6))  # brown
    elif kind == 'crease':
        img = _absorb(base, _vein_profile(), (0.8, 0.8, 0.8))  # neutral fold shadow
    elif kind == 'tattoo':
        img = _absorb(base, _vein_profile(), (0.7, 0.35, 0.15))  # strong blue ink
    else:
        prof = np.zeros((H, W), np.float32)
        for x in range(60, 360, 40):
            cv2.line(prof, (x, 200), (x + 6, 203), 1.0, 2)
        img = _absorb(base, cv2.GaussianBlur(prof, (0, 0), 1.0), VEIN)
    out = soften_veins(img, MASK, 1.0, FW, value_scale=1.0)
    assert np.abs(out - img).max() * 255 < 2.0


@pytest.mark.parametrize('colour', [(0.78, 0.78, 0.80), (0.86, 0.74, 0.70)])
def test_white_and_bluish_white_paint_are_left_alone(colour):
    # Neutral white, and the bunny photos' bluish white (b* about -9, C/L*
    # above the neutral gate, hue outside natural skin's).
    paint = _skin(colour=colour)
    _, out, planted = _removed(paint)
    assert np.abs(out - planted).max() * 255 < 1.0


def test_outside_mask_and_excluded_pixels_unchanged():
    mask = MASK.copy()
    mask[:, 220:] = 0
    exclude = np.zeros((H, W), np.float32)
    exclude[:, :80] = 1
    _, out, planted = _removed(_skin(), mask=mask, exclude_mask=exclude)
    assert np.array_equal(out[:, 220:], planted[:, 220:])
    assert np.array_equal(out[:, :80], planted[:, :80])
    assert not np.array_equal(out[:, 80:220], planted[:, 80:220])


def test_dtypes_and_off():
    base = _skin()
    planted = _absorb(base, _vein_profile(), VEIN)
    assert soften_veins(planted, MASK, 0.0, FW) is planted
    u8 = np.clip(planted * 255 + 0.5, 0, 255).astype(np.uint8)
    out_u8 = soften_veins(u8, MASK * 255, 1.0, FW)
    assert out_u8.dtype == np.uint8 and not np.array_equal(out_u8, u8)
    f255 = planted * 255.0
    out_255 = soften_veins(f255, MASK, 1.0, FW, value_scale=255.0)
    out_1 = soften_veins(planted, MASK, 1.0, FW, value_scale=1.0)
    assert np.abs(out_255 / 255.0 - out_1).max() < 1e-4


def test_param_is_a_0_100_slider_and_recipe_stays_off():
    from retouch.params import PROCESSING_PARAMS
    from retouch.recipes import RECIPES
    spec = next(p for p in PROCESSING_PARAMS if p.name == 'vein_attenuate')
    assert (spec.min_val, spec.max_val, spec.default, spec.conversion) == (0, 100, 0, 'recipe_pct')
    assert RECIPES['vascular_refine_v1']['skin']['vein_attenuate'] == 0.0
