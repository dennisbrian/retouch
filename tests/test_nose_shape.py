"""Tests for the opt-in Keep Nose Shape op (retouch/nose_shape.py)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.nose_shape import keep_nose_shape
from retouch.params import PROCESSING_PARAMS


def _scene(scale: float = 1.0):
    """A nose-shaped bump with fine texture, and the same after heavy smoothing."""
    h, w = 160, 160
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bump = 40.0 * np.exp(-((xx - 80) ** 2 + (yy - 80) ** 2) / (2 * 12.0 ** 2))
    rng = np.random.default_rng(0)
    texture = rng.normal(0.0, 6.0, (h, w)).astype(np.float32)
    base = 150.0 + bump + texture
    before = np.repeat(base[:, :, None], 3, axis=2) * scale
    # "Smoothing" that flattens the bump and removes texture inside the face.
    after = np.full_like(before, 150.0 * scale + 10.0 * scale)
    nose = np.zeros((h, w), np.float32)
    cv2.circle(nose, (80, 80), 30, 1.0, -1)
    return before.astype(np.float32), after.astype(np.float32), nose


def _highpass_std(img, mask, sigma=4.0):
    L = img[:, :, 1]
    return float((L - cv2.GaussianBlur(L, (0, 0), sigma))[mask].std())


def test_strength_zero_is_identity():
    before, after, nose = _scene()
    out = keep_nose_shape(after, before, nose, 0, ied=100)
    assert out is after


def test_missing_or_empty_mask_is_identity():
    before, after, nose = _scene()
    assert keep_nose_shape(after, before, None, 100, ied=100) is after
    assert keep_nose_shape(after, before, np.zeros_like(nose), 100, ied=100) is after


def test_restores_broad_shape_not_fine_texture():
    before, after, nose = _scene()
    out = keep_nose_shape(after, before, nose, 100, ied=100)
    core = cv2.erode(nose, np.ones((15, 15), np.uint8)) > 0.5
    # The bump's peak comes back most of the way ...
    peak_before = before[80, 80, 1] - 150.0
    peak_out = out[78:83, 78:83, 1].mean() - after[80, 80, 1] + 10.0
    assert peak_out > 0.7 * (peak_before - 6.0)
    # ... while fine texture stays well below the original.
    assert _highpass_std(out, core) < 0.5 * _highpass_std(before, core)


def test_strength_scales_restore():
    before, after, nose = _scene()
    half = keep_nose_shape(after, before, nose, 50, ied=100)
    full = keep_nose_shape(after, before, nose, 100, ied=100)
    d_half = half[80, 80, 1] - after[80, 80, 1]
    d_full = full[80, 80, 1] - after[80, 80, 1]
    assert d_full > 0
    assert d_half == pytest.approx(d_full / 2, rel=0.02)


def test_leaves_pixels_away_from_nose_alone():
    before, after, nose = _scene()
    out = keep_nose_shape(after, before, nose, 100, ied=100)
    assert np.allclose(out[:20, :20], after[:20, :20], atol=0.01)


def test_tone_invariant():
    """The restore is a difference of the same pixels, so it scales with tone."""
    b1, a1, nose = _scene(1.0)
    b2, a2, _ = _scene(0.4)
    d1 = keep_nose_shape(a1, b1, nose, 100, ied=100) - a1
    d2 = keep_nose_shape(a2, b2, nose, 100, ied=100) - a2
    assert np.allclose(d2, 0.4 * d1, atol=0.05)


def test_skin_mask_limits_restore():
    before, after, nose = _scene()
    skin = np.ones(nose.shape, np.float32)
    skin[:, 80:] = 0.0  # e.g. hair falling over half the nose
    out = keep_nose_shape(after, before, nose, 100, ied=100, skin_mask=skin)
    assert np.allclose(out[:, 81:], after[:, 81:], atol=0.01)
    assert abs(out[80, 70, 1] - after[80, 70, 1]) > 5.0


def test_accepts_0_255_masks():
    before, after, nose = _scene()
    a = keep_nose_shape(after, before, nose, 100, ied=100)
    b = keep_nose_shape(after, before, (nose * 255).astype(np.uint8), 100, ied=100)
    assert np.allclose(a, b, atol=0.5)


def test_param_spec_is_opt_in():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "nose_shape")
    assert spec.default == 0
    assert spec.cli_flag == "nose-shape"
    assert spec.recipe_key == "skin.nose_shape"


class TestEngineWiring:
    """The op must reach the per-face pipeline and stay inside the nose."""

    @pytest.fixture(scope="class")
    def run(self):
        from retouch.engine import RetouchEngine
        from tests.golden_face_fixture import make_face_context, make_synthetic_face_image

        img = make_synthetic_face_image()
        h, w = img.shape[:2]
        fc = make_face_context(w, h, img)
        eng = RetouchEngine()
        off = np.asarray(eng.process(img, recipe="natural", face_contexts=[fc]))
        on = np.asarray(eng.process(img, recipe="natural", face_contexts=[fc], nose_shape=100))
        return img, fc, off, on

    def test_default_off_changes_nothing(self, run):
        from retouch.engine import RetouchEngine

        img, fc, off, _ = run
        again = np.asarray(RetouchEngine().process(img, recipe="natural", face_contexts=[fc], nose_shape=0))
        assert np.array_equal(off, again)

    def test_on_changes_only_the_nose_area(self, run):
        from retouch.parsing import NOSE

        img, fc, off, on = run
        diff = np.abs(on.astype(np.int16) - off.astype(np.int16)).max(axis=2)
        assert diff.max() > 0, "nose_shape=100 had no effect: op not wired into the face path"
        h, w = img.shape[:2]
        lm = fc.face_data.landmarks.landmark
        pts = np.array([[lm[i].x * w, lm[i].y * h] for i in NOSE], np.float32)
        x0, y0 = pts.min(0)
        x1, y1 = pts.max(0)
        pad = 0.5 * (x1 - x0)
        ys, xs = np.nonzero(diff > 1)
        assert xs.min() >= x0 - pad and xs.max() <= x1 + pad
        assert ys.min() >= y0 - pad and ys.max() <= y1 + pad
