"""Tests for the opt-in nose tip blush (retouch/nose_tip_blush.py).

Darker skin is simulated (a linear-light exposure gain on a flat patch);
no real darker-skin photo was available.
"""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from retouch.nose_tip_blush import add_nose_tip_blush, tip_mask
from retouch.params import PROCESSING_PARAMS

H = W = 200
IED = 100.0
TIP = (100, 120)  # (x, y) of the blush centre in the synthetic layout


def _landmarks(cx: float = 100.0):
    """478 landmarks at the centre; nose axis vertical, tip at y=120."""
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]

    def put(i, x, y):
        lm[i] = SimpleNamespace(x=x / W, y=y / H)

    put(168, cx, 40)    # between the brows
    put(4, cx, 115)     # just above the tip
    put(1, cx, 125)     # tip
    put(129, cx - 25, 122)  # alar wings: 50 px wide
    put(358, cx + 25, 122)
    return SimpleNamespace(landmark=lm)


def _flat(bgr=(150.0, 165.0, 200.0)):
    return np.ones((H, W, 3), np.float32) * np.array(bgr, np.float32)


def _lab(img):
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)


def _lin(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055) * 255.0


def test_strength_zero_or_missing_landmarks_is_identity():
    img = _flat()
    assert add_nose_tip_blush(img, _landmarks(), 0, IED) is img
    assert add_nose_tip_blush(img, None, 100, IED) is img


def test_tints_the_tip_pink_and_leaves_the_rest_bit_identical():
    img = _flat()
    out = add_nose_tip_blush(img, _landmarks(), 100, IED)
    d = _lab(out) - _lab(img)
    x, y = TIP
    assert d[y, x, 1] >= 6  # a* up: pinker
    assert abs(d[y, x, 0]) <= 3  # lightness about the same
    assert out.dtype == np.float32
    # Far from the nose: untouched, exactly.
    assert np.array_equal(out[:60], img[:60])
    assert np.array_equal(out[:, :40], img[:, :40])
    assert np.array_equal(out[:, 161:], img[:, 161:])


def test_falls_off_faster_toward_the_lips_than_up_the_bridge():
    m = tip_mask(_landmarks(), (H, W), IED)
    x, y = TIP
    assert m[y, x] > 0.95
    assert m[y + 15, x] < 0.5 * m[y - 15, x]  # 15 px down vs 15 px up
    assert m[y + 30, x] < 0.05  # philtrum / upper lip


def test_strength_scales_monotonically():
    img = _flat()
    x, y = TIP
    a = [(_lab(add_nose_tip_blush(img, _landmarks(), s, IED)) - _lab(img))[y, x, 1]
         for s in (0, 25, 50, 75, 100)]
    assert all(b >= a_ for a_, b in zip(a, a[1:]))
    assert a[-1] > a[1]


def test_similar_relative_tint_on_darker_skin():
    """Simulated darker skin (linear gain 0.2) keeps a comparable shift.

    The pigment model is a ratio, so the a* shift scales roughly with L*;
    it must neither vanish nor jump to a fixed pink.
    """
    light = _flat()
    dark = _srgb(_lin(light) * 0.2).astype(np.float32)
    x, y = TIP
    d_light = (_lab(add_nose_tip_blush(light, _landmarks(), 100, IED)) - _lab(light))[y, x]
    d_dark = (_lab(add_nose_tip_blush(dark, _landmarks(), 100, IED)) - _lab(dark))[y, x]
    L_light, L_dark = _lab(light)[y, x, 0], _lab(dark)[y, x, 0]
    ratio = (d_dark[1] / L_dark) / (d_light[1] / L_light)
    assert d_dark[1] >= 3
    assert 0.6 < ratio < 1.6


def test_nostrils_are_left_alone():
    img = _flat()
    nostril = (TIP[0] + 10, TIP[1] + 4)
    cv2.circle(img, nostril, 4, (25.0, 25.0, 35.0), -1)
    out = add_nose_tip_blush(img, _landmarks(), 100, IED)
    d = _lab(out) - _lab(img)
    assert abs(d[nostril[1], nostril[0], 1]) <= 1
    assert d[TIP[1], TIP[0], 1] >= 6


def test_skin_and_lips_masks_limit_the_tint():
    img = _flat()
    skin = np.ones((H, W), np.uint8) * 255
    skin[:, TIP[0]:] = 0  # right half isn't skin (hair, a strap...)
    out = add_nose_tip_blush(img, _landmarks(), 100, IED, skin_mask=skin)
    assert np.array_equal(out[:, TIP[0]:], img[:, TIP[0]:])
    lips = np.zeros((H, W), np.float32)
    lips[TIP[1]:, :] = 1.0
    out = add_nose_tip_blush(img, _landmarks(), 100, IED, lips_mask=lips)
    assert np.array_equal(out[TIP[1]:], img[TIP[1]:])


def test_white_paint_does_not_clip_to_a_flat_patch():
    img = _flat((250.0, 250.0, 250.0))
    out = add_nose_tip_blush(img, _landmarks(), 100, IED)
    d = _lab(out) - _lab(img)
    assert d[TIP[1], TIP[0], 1] >= 3  # painted noses still get a pink tip
    assert np.all(out <= 255.0)


def test_param_spec_is_opt_in():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "nose_tip_blush")
    assert spec.default == 0
    assert spec.cli_flag == "nose-tip-blush"
    assert spec.recipe_key == "skin.nose_tip_blush"


class TestEngineWiring:
    """The op must reach the per-face pipeline and stay on the nose."""

    @pytest.fixture(scope="class")
    def run(self):
        from retouch.engine import RetouchEngine
        from tests.golden_face_fixture import (
            make_face_context,
            make_synthetic_face_image,
        )

        img = make_synthetic_face_image()
        h, w = img.shape[:2]
        fc = make_face_context(w, h, img)
        eng = RetouchEngine()
        off = np.asarray(eng.process(img, recipe="natural", face_contexts=[fc]))
        on = np.asarray(eng.process(img, recipe="natural", face_contexts=[fc], nose_tip_blush=100))
        return img, fc, off, on

    def test_default_off_changes_nothing(self, run):
        from retouch.engine import RetouchEngine

        img, fc, off, _ = run
        again = np.asarray(RetouchEngine().process(img, recipe="natural", face_contexts=[fc], nose_tip_blush=0))
        assert np.array_equal(off, again)

    def test_on_tints_only_around_the_nose_tip(self, run):
        img, fc, off, on = run
        lab_off, lab_on = _lab(off), _lab(on)
        diff = lab_on - lab_off
        mag = np.abs(diff).max(axis=2)
        assert mag.max() > 0, "nose_tip_blush=100 had no effect: op not wired into the face path"
        h, w = img.shape[:2]
        lm = fc.face_data.landmarks.landmark
        tip = np.array([lm[1].x * w, lm[1].y * h])
        ys, xs = np.nonzero(mag > 1)
        r = 0.8 * fc.face_data.ied
        assert np.all(np.hypot(xs - tip[0], ys - tip[1]) < r)
        assert diff[ys, xs, 1].mean() > 0  # pinker, not greener


def test_slider_replaces_the_legacy_nose_blush_disc():
    """With the slider on, the old nose_blush disc inside apply_blush is skipped."""
    import inspect

    from retouch import perf_optimizations

    src = inspect.getsource(perf_optimizations._process_face_core)
    assert "nose_blush=ctx.nose_blush and _nose_tip_blush <= 0" in src
