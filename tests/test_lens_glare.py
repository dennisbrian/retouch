"""Glasses / goggle / visor glare removal (retouch/lens_glare.py).

Uses the frozen real landmark set from ``golden_face_fixture`` on a flat
skin canvas with a drawn dark frame, so every case is deterministic and
needs no detector or model.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.lens_glare import lens_support, remove_lens_glare, remove_lens_glare_faces
from retouch.params import PROCESSING_PARAMS
from retouch.utils import bgr_f32_to_lab_f32
from tests.golden_face_fixture import make_face_context

W, H = 640, 750

# Light, medium and dark skin (BGR).
SKIN_TONES = {
    "light": (170, 190, 230),
    "medium": (105, 140, 190),
    "dark": (45, 65, 95),
}


def _face(tone=SKIN_TONES["medium"], frame=True, seed=0):
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.float32)
    img[:] = tone
    img += rng.normal(0, 3, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    face = make_face_context(W, H, img).face_data
    ellipses, ied = lens_support(face.landmarks, W, H)
    if frame:
        for e in ellipses:
            cv2.ellipse(img, e, (25, 25, 25), max(2, int(0.03 * ied)))
    return img, face, ellipses, ied


def _plant(img, ellipses, ied, amount=70.0):
    """Additive soft white reflection in the lower half of each lens."""
    g = np.zeros(img.shape[:2], np.float32)
    for (cx, cy), (ew, eh), ang in ellipses:
        blob = np.zeros_like(g)
        cv2.ellipse(blob, ((cx, cy + 0.28 * eh), (ew * 0.45, eh * 0.22), ang + 25), 1.0, -1)
        g = np.maximum(g, cv2.GaussianBlur(blob, (0, 0), ied * 0.03))
    out = np.clip(img.astype(np.float32) + g[..., None] * amount, 0, 255).astype(np.uint8)
    return out, g


def _L(img):
    return bgr_f32_to_lab_f32(img.astype(np.float32))[..., 0]


def test_zero_strength_is_identity():
    img, face, _, _ = _face()
    out, diag = remove_lens_glare(img, face.landmarks, 0)
    assert out is img
    assert diag["glare_fraction"] == 0.0


@pytest.mark.parametrize("tone", list(SKIN_TONES))
def test_removes_planted_glare_on_every_skin_tone(tone):
    clean, face, ellipses, ied = _face(SKIN_TONES[tone])
    glared, g = _plant(clean, ellipses, ied)
    fixed, diag = remove_lens_glare(glared, face.landmarks, 100)
    m = g > 0.1
    before = np.abs(_L(glared) - _L(clean))[m].mean()
    after = np.abs(_L(fixed) - _L(clean))[m].mean()
    assert before > 20.0
    assert after < 0.35 * before, (tone, before, after)
    assert diag["glare_fraction"] > 0.0


def test_leaves_clean_glasses_alone():
    """A frame with no glare: the lens area barely moves."""
    for tone in SKIN_TONES.values():
        clean, face, ellipses, ied = _face(tone)
        out, _ = remove_lens_glare(clean, face.landmarks, 100)
        lens = np.zeros(clean.shape[:2], np.uint8)
        for e in ellipses:
            cv2.ellipse(lens, e, 1, -1)
        assert np.abs(_L(out) - _L(clean))[lens > 0].mean() < 1.0


def test_strength_scales_the_lift():
    clean, face, ellipses, ied = _face()
    glared, g = _plant(clean, ellipses, ied)
    m = g > 0.1
    half, _ = remove_lens_glare(glared, face.landmarks, 50)
    full, _ = remove_lens_glare(glared, face.landmarks, 100)
    lift_half = (_L(glared) - _L(half))[m].mean()
    lift_full = (_L(glared) - _L(full))[m].mean()
    assert 0.3 * lift_full < lift_half < 0.7 * lift_full


def test_pixels_away_from_the_eyes_are_bit_exact():
    clean, face, ellipses, ied = _face()
    glared, _ = _plant(clean, ellipses, ied)
    fixed, _ = remove_lens_glare(glared, face.landmarks, 100)
    ys = [e[0][1] for e in ellipses]
    far = int(max(ys) + 1.2 * ied)
    assert np.array_equal(fixed[far:], glared[far:])


def test_float32_in_float32_out():
    clean, face, ellipses, ied = _face()
    glared, _ = _plant(clean, ellipses, ied)
    out, _ = remove_lens_glare(glared.astype(np.float32), face.landmarks, 80)
    assert out.dtype == np.float32
    assert out.shape == glared.shape


def test_faces_helper_skips_faces_without_landmarks():
    clean, face, _, _ = _face()

    class _NoLm:
        landmarks = None

    out, diags = remove_lens_glare_faces(clean, [_NoLm(), face], 60)
    assert len(diags) == 2
    assert out.shape == clean.shape


def test_param_is_registered_and_off_by_default():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "lens_glare")
    assert spec.default == 0
    assert (spec.min_val, spec.max_val) == (0, 100)
    assert spec.cli_flag == "lens-glare"


def test_engine_stage_is_a_no_op_when_off_and_runs_when_on():
    from retouch.engine import ProcessingContext, RetouchEngine

    clean, face, ellipses, ied = _face()
    glared, _ = _plant(clean, ellipses, ied)
    engine = RetouchEngine.__new__(RetouchEngine)

    ctx = ProcessingContext()
    timings = {}
    assert engine._stage_lens_glare(glared, [face], ctx, timings) is glared
    assert "lens_glare" not in timings

    ctx.lens_glare = 100.0
    out = engine._stage_lens_glare(glared, [face], ctx, timings)
    assert "lens_glare" in timings
    assert ctx._runtime_diagnostics["lens_glare"][0]["glare_fraction"] > 0.0
    assert not np.array_equal(out, glared)
