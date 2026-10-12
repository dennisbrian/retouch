"""Tests for the opt-in nose bridge highlight (retouch/nose_highlight.py)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from retouch.nose_highlight import RIDGE, add_nose_highlight, ridge_mask
from retouch.params import PROCESSING_PARAMS

H = W = 200
IED = 100.0


def _landmarks(x: float = 100.0, top: float = 40.0, bottom: float = 150.0):
    """478 landmarks at the centre, with the ridge points on a vertical line."""
    lm = [SimpleNamespace(x=0.5, y=0.5) for _ in range(478)]
    for i, idx in enumerate(RIDGE):
        y = top + (bottom - top) * i / (len(RIDGE) - 1)
        lm[idx] = SimpleNamespace(x=x / W, y=y / H)
    return SimpleNamespace(landmark=lm)


def _flat(bgr=(150.0, 160.0, 190.0)):
    return np.ones((H, W, 3), np.float32) * np.array(bgr, np.float32)


def test_strength_zero_or_missing_landmarks_is_identity():
    img = _flat()
    assert add_nose_highlight(img, _landmarks(), 0, IED) is img
    assert add_nose_highlight(img, None, 100, IED) is img
    assert add_nose_highlight(img, _landmarks(), 100, 0) is img


def test_brightens_the_ridge_not_the_sides():
    img = _flat()
    out = add_nose_highlight(img, _landmarks(), 100, IED)
    d = (out - img).mean(axis=2)
    assert d[95, 100] > 8.0  # ridge centre, mid bridge
    assert d[95, 100] > 5 * d[95, 112]  # 0.12 IED to the side
    assert np.allclose(d[:, :80], 0.0, atol=1e-3)
    assert np.allclose(d[:, 121:], 0.0, atol=1e-3)
    assert out.dtype == np.float32


def test_fades_at_both_ends():
    img = _flat()
    d = (add_nose_highlight(img, _landmarks(), 100, IED) - img).mean(axis=2)
    mid = d[95, 100]
    assert d[42, 100] < 0.15 * mid  # between the brows
    assert d[148, 100] < 0.15 * mid  # just above the tip
    assert np.allclose(d[160:], 0.0, atol=1e-3)


def test_strength_scales_monotonically():
    img = _flat()
    lifts = [
        (add_nose_highlight(img, _landmarks(), s, IED) - img)[95, 100].mean()
        for s in (25, 50, 100)
    ]
    assert 0 < lifts[0] < lifts[1] < lifts[2]


def test_keeps_skin_hue():
    """Linear-light gain: chromaticity of the ridge stays the skin's own."""
    img = _flat()
    out = add_nose_highlight(img, _landmarks(), 100, IED)

    def lin(v):
        v = v / 255.0
        return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)

    a, b = lin(img[95, 100]), lin(out[95, 100])
    # Same channel ratios up to the shoulder's small per-channel difference.
    assert np.allclose(a / a.sum(), b / b.sum(), atol=0.02)


def test_similar_relative_lift_on_darker_skin():
    """No intensity threshold: darker skin gets a comparable lift in stops."""
    light = _flat((150.0, 160.0, 190.0))
    dark = _flat((45.0, 60.0, 85.0))
    lm = _landmarks()

    def stops(img):
        out = add_nose_highlight(img, lm, 100, IED)
        g = lambda v: np.where(v / 255 <= 0.04045, v / 255 / 12.92, ((v / 255 + 0.055) / 1.055) ** 2.4)
        return float(np.log2(g(out[95, 100]).mean() / g(img[95, 100]).mean()))

    s_light, s_dark = stops(light), stops(dark)
    assert s_light > 0.25 and s_dark > 0.25
    assert s_dark == pytest.approx(s_light, rel=0.5)


def test_bright_paint_does_not_clip():
    img = _flat((250.0, 250.0, 250.0))
    out = add_nose_highlight(img, _landmarks(), 100, IED)
    assert out.max() <= 255.0
    assert out[95, 100].mean() > 250.0


def test_skin_mask_limits_highlight():
    img = _flat()
    skin = np.ones((H, W), np.float32)
    skin[:90] = 0.0  # e.g. glasses or hair over the upper bridge
    out = add_nose_highlight(img, _landmarks(), 100, IED, skin_mask=(skin * 255).astype(np.uint8))
    d = (out - img).mean(axis=2)
    assert np.allclose(d[:90], 0.0, atol=1e-3)
    assert d[110, 100] > 5.0


def test_ridge_mask_follows_a_tilted_face():
    pts = np.array([[60, 40], [140, 150]], np.float32)
    m = ridge_mask(pts, (H, W), IED)
    assert m[95, 100] > 0.9  # on the diagonal
    assert m[95, 60] < 0.01 and m[95, 140] < 0.01


def test_param_spec_is_opt_in():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "nose_highlight")
    assert spec.default == 0
    assert spec.cli_flag == "nose-highlight"
    assert spec.recipe_key == "skin.nose_highlight"


class TestEngineWiring:
    """The op must reach the per-face pipeline and stay on the nose."""

    @pytest.fixture(scope="class")
    def run(self):
        from retouch.engine import RetouchEngine
        from tests.golden_face_fixture import make_face_context, make_synthetic_face_image

        img = make_synthetic_face_image()
        h, w = img.shape[:2]
        fc = make_face_context(w, h, img)
        eng = RetouchEngine()
        off = np.asarray(eng.process(img, recipe="natural", face_contexts=[fc]))
        on = np.asarray(eng.process(img, recipe="natural", face_contexts=[fc], nose_highlight=100))
        return img, fc, off, on

    def test_default_off_changes_nothing(self, run):
        from retouch.engine import RetouchEngine

        img, fc, off, _ = run
        again = np.asarray(RetouchEngine().process(img, recipe="natural", face_contexts=[fc], nose_highlight=0))
        assert np.array_equal(off, again)

    def test_on_brightens_only_along_the_ridge(self, run):
        img, fc, off, on = run
        diff = on.astype(np.int16) - off.astype(np.int16)
        mag = np.abs(diff).max(axis=2)
        assert mag.max() > 0, "nose_highlight=100 had no effect: op not wired into the face path"
        h, w = img.shape[:2]
        lm = fc.face_data.landmarks.landmark
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in RIDGE], np.float32)
        pad = 0.25 * fc.face_data.ied
        ys, xs = np.nonzero(mag > 1)
        assert xs.min() >= pts[:, 0].min() - pad and xs.max() <= pts[:, 0].max() + pad
        assert ys.min() >= pts[:, 1].min() - pad and ys.max() <= pts[:, 1].max() + pad
        assert diff[ys, xs].mean() > 0  # a highlight, not a shadow
