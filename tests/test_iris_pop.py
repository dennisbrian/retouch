"""Tests for the opt-in Iris Pop (retouch/iris_pop.py).

Synthetic eyes: a sclera ellipse, a textured coloured lens wider than the
landmark iris (as circle lenses are), a dark pupil and a catchlight. Darker
and lighter skin/exposure are simulated with a linear-light gain; no real
darker-skin photo was available.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from retouch.iris_pop import iris_pop, iris_pop_eye
from retouch.params import PROCESSING_PARAMS

H, W = 120, 240
EYES = ((70.0, 60.0), (170.0, 60.0))   # iris centres (x, y)
R = 10.0                               # landmark iris radius
R_LENS = 14.0                          # visible lens radius (1.4x)


def _lin(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) * 255.0


def _landmarks():
    lm = [SimpleNamespace(x=0.5, y=0.02) for _ in range(478)]
    for (cx, cy), c in zip(EYES, (468, 473)):
        lm[c] = SimpleNamespace(x=cx / W, y=cy / H)
        for i, (dx, dy) in enumerate(((R, 0), (0, -R), (-R, 0), (0, R))):
            lm[c + 1 + i] = SimpleNamespace(x=(cx + dx) / W, y=(cy + dy) / H)
    return SimpleNamespace(landmark=lm)


def _eye_masks(closed=()):
    masks = []
    for i, (cx, cy) in enumerate(EYES):
        m = np.zeros((H, W), np.uint8)
        if i not in closed:
            cv2.ellipse(m, (int(cx), int(cy)), (40, 17), 0, 0, 360, 255, -1)
        masks.append(m)
    return masks


def _scene(gain=1.0):
    """Skin, sclera, a dull textured pink lens, pupil and catchlight (BGR)."""
    rng = np.random.default_rng(3)
    img = np.ones((H, W, 3), np.float32) * np.array([150, 165, 200], np.float32)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    for (cx, cy), m in zip(EYES, _eye_masks()):
        inside = m > 0
        img[inside] = (225, 228, 232)
        d = np.hypot(xx - cx, yy - cy)
        lens = inside & (d < R_LENS)
        tex = 1.0 + 0.15 * np.sin(np.arctan2(yy - cy, xx - cx) * 14) + rng.normal(0, 0.04, (H, W))
        img[lens] = (np.array([118, 96, 138], np.float32) * tex[lens][:, None])
        img[inside & (d < 4.5)] = (18, 16, 18)                      # pupil
        img[inside & (np.hypot(xx - cx - 3, yy - cy + 3) < 1.6)] = (250, 250, 250)  # catchlight
    if gain != 1.0:
        img = _srgb(_lin(img) * gain).astype(np.float32)
    return img


def _chroma(img):
    lab = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.hypot(lab[..., 1] - 128, lab[..., 2] - 128), np.arctan2(lab[..., 2] - 128, lab[..., 1] - 128)


def _ring(cx, cy, r0, r1):
    yy, xx = np.mgrid[:H, :W]
    d = np.hypot(xx - cx, yy - cy)
    return (d >= r0) & (d < r1)


def _log_detail(img, mask):
    y = _lin(img) @ np.array([0.0722, 0.7152, 0.2126])
    ly = np.log(np.maximum(y, 1e-5))
    return float((ly - cv2.GaussianBlur(ly, (0, 0), 2.0))[mask].std())


def test_strength_zero_missing_landmarks_or_closed_eyes_is_identity():
    img = _scene()
    assert iris_pop(img, _landmarks(), 0, _eye_masks()) is img
    assert iris_pop(img, None, 100, _eye_masks()) is img
    assert iris_pop(img, _landmarks(), 100, _eye_masks(closed=(0, 1))) is img


def test_only_the_visible_iris_changes():
    img = _scene()
    out = iris_pop(img, _landmarks(), 100, _eye_masks())
    changed = np.abs(out - img).max(axis=2) > 0
    allowed = np.zeros((H, W), bool)
    for cx, cy in EYES:
        allowed |= _ring(cx, cy, 0, R_LENS + 1.5)
    assert changed.any()
    assert not (changed & ~allowed).any()          # sclera, skin bit-identical


def test_closed_eye_is_skipped_but_open_eye_is_popped():
    img = _scene()
    out = iris_pop(img, _landmarks(), 100, _eye_masks(closed=(1,)))
    right = _ring(*EYES[1], 0, 30)
    assert np.array_equal(out[right], img[right])
    assert not np.array_equal(out[_ring(*EYES[0], 0, 30)], img[_ring(*EYES[0], 0, 30)])


def test_colour_rises_hue_kept_and_wide_lens_included():
    img = _scene()
    out = iris_pop(img, _landmarks(), 100, _eye_masks())
    c0, h0 = _chroma(img)
    c1, h1 = _chroma(out)
    for cx, cy in EYES:
        body = _ring(cx, cy, 5.5, R - 1)
        outer = _ring(cx, cy, R + 0.5, R_LENS - 2)   # lens past the landmark iris
        assert c1[body].mean() > 1.2 * c0[body].mean()
        assert c1[outer].mean() > 1.1 * c0[outer].mean()
        dh = np.angle(np.exp(1j * (h1[body] - h0[body])))
        assert np.abs(np.degrees(dh)).mean() < 6.0


def test_texture_and_limbal_depth():
    img = _scene()
    out = iris_pop(img, _landmarks(), 100, _eye_masks())
    for cx, cy in EYES:
        body = _ring(cx, cy, 5.5, R - 1)
        assert _log_detail(out, body) > 1.15 * _log_detail(img, body)
        rim = _ring(cx, cy, 0.86 * R_LENS, 0.96 * R_LENS)
        y0 = _lin(img) @ np.array([0.0722, 0.7152, 0.2126])
        y1 = _lin(out) @ np.array([0.0722, 0.7152, 0.2126])
        assert y1[rim].mean() < 0.95 * y0[rim].mean()   # darker limbal ring


def test_pupil_and_catchlight_kept():
    img = _scene()
    out = iris_pop(img, _landmarks(), 100, _eye_masks())
    for cx, cy in EYES:
        pupil = _ring(cx, cy, 0, 3.0)
        glint = _ring(cx + 3, cy - 3, 0, 1.0)
        assert np.abs(out[pupil] - img[pupil]).max() < 2.0
        assert np.abs(out[glint] - img[glint]).max() < 2.0


@pytest.mark.parametrize("gain", [0.45, 1.5])
def test_relative_effect_is_tone_invariant(gain):
    """Simulated darker/lighter exposure: the same relative chroma and detail gain."""
    def gains(img):
        out = iris_pop(img, _landmarks(), 100, _eye_masks())
        body = _ring(*EYES[0], 5.5, R - 1)
        c0, _ = _chroma(img)
        c1, _ = _chroma(out)
        return (c1[body].mean() / c0[body].mean(),
                _log_detail(out, body) / _log_detail(img, body))

    ref_c, ref_d = gains(_scene())
    c, d = gains(_scene(gain))
    assert abs(c - ref_c) < 0.12
    assert abs(d - ref_d) < 0.12


def test_strength_scales_and_support_is_reported():
    img = _scene()
    support = np.zeros((H, W), np.float32)
    half = iris_pop(img, _landmarks(), 50, _eye_masks(), support_out=support)
    full = iris_pop(img, _landmarks(), 100, _eye_masks())
    d50 = np.abs(half - img).sum()
    d100 = np.abs(full - img).sum()
    assert 0.3 * d100 < d50 < 0.8 * d100
    assert support.max() > 0.9
    assert not (np.abs(half - img).max(axis=2) > 0)[support == 0].any()


def test_float_mask_and_single_eye_call():
    img = _scene()
    mask = _eye_masks()[0].astype(np.float32) / 255.0
    out = iris_pop_eye(img, mask, EYES[0], R, 100)
    assert out is not img and not np.array_equal(out, img)
    assert iris_pop_eye(img, mask, EYES[0], 1.0, 100) is img       # too small


def test_param_registered_off_by_default():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "iris_pop")
    assert spec.default == 0
    assert spec.recipe_key == "eyes.iris_pop"
    assert spec.cli_flag == "iris-pop"


def test_iris_support_is_composited_at_full_weight():
    """The face composite must include Iris Pop's support, not only the
    (contact-lens scaled) eye-sharpen mask."""
    from retouch.engine import RetouchEngine
    from retouch.perf_optimizations import _FaceResult

    z = np.zeros((4, 4), np.float32)
    extra = z.copy()
    extra[1, 1] = 1.0
    fr = _FaceResult(canvas=np.zeros((4, 4, 3), np.uint8), skin_mask=z, skin_hair_mask=z,
                     lips_mask=z, sharpen_mask=z, roi_box=(0, 0, 4, 4), eye_edit_mask=extra)
    alpha = np.maximum.reduce(RetouchEngine._composite_parts(fr))
    assert alpha[1, 1] == 1.0
    fr.eye_edit_mask = None
    assert np.maximum.reduce(RetouchEngine._composite_parts(fr)).max() == 0.0
