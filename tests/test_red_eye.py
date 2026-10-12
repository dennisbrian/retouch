"""Flash red-eye removal (retouch/red_eye.py).

Uses the frozen real landmark set from ``golden_face_fixture`` on a flat skin
canvas with drawn eyes (sclera, iris, pupil, catchlight), so every case is
deterministic and needs no detector or model.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.params import PROCESSING_PARAMS
from retouch.parsing import LEFT_EYE, LEFT_IRIS, RIGHT_EYE, RIGHT_IRIS
from retouch.red_eye import remove_red_eye, remove_red_eye_faces
from tests.golden_face_fixture import make_face_context

W, H = 640, 750

# Light, medium and dark skin (BGR).
SKIN_TONES = {
    "light": (170, 190, 230),
    "medium": (105, 140, 190),
    "dark": (45, 65, 95),
}
RED_PUPIL = (40, 45, 205)          # flash red-eye (BGR)
DARK_PUPIL = (18, 20, 24)          # normal pupil
BROWN_IRIS = (40, 70, 110)
BLUE_IRIS = (150, 120, 80)
RED_LENS = (30, 30, 190)           # cosplay red contact lens


def _eye_geom(face, iris_idx):
    lm = face.landmarks.landmark
    c = np.array([lm[iris_idx[0]].x * W, lm[iris_idx[0]].y * H])
    ring = np.array([[lm[i].x * W, lm[i].y * H] for i in iris_idx[1:]])
    return c, float(np.linalg.norm(ring - c, axis=1).mean())


def _face(tone=SKIN_TONES["medium"], pupil=RED_PUPIL, iris=BROWN_IRIS,
          pupil_frac=0.5, background=None, seed=0):
    """Skin canvas with two drawn eyes. Returns (img, face, pupil_mask)."""
    rng = np.random.default_rng(seed)
    img = np.empty((H, W, 3), np.float32)
    img[:] = tone if background is None else background
    img += rng.normal(0, 2, img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    face = make_face_context(W, H, img).face_data
    lm = face.landmarks.landmark
    pupil_mask = np.zeros((H, W), np.uint8)
    for iris_idx, eye_idx in ((LEFT_IRIS, LEFT_EYE), (RIGHT_IRIS, RIGHT_EYE)):
        poly = np.array([[lm[i].x * W, lm[i].y * H] for i in eye_idx], np.int32)
        opening = np.zeros((H, W), np.uint8)
        cv2.fillPoly(opening, [poly], 1)
        layer = img.copy()
        layer[opening > 0] = (225, 228, 232)            # sclera
        c, r = _eye_geom(face, iris_idx)
        ctr = (int(round(c[0])), int(round(c[1])))
        cv2.circle(layer, ctr, int(round(r)), iris, -1)
        cv2.circle(layer, ctr, int(round(pupil_frac * r)), pupil, -1)
        cv2.circle(pupil_mask, ctr, int(round(pupil_frac * r)), 1, -1)
        # Catchlight up and left of the pupil centre.
        cl = (int(round(c[0] - 0.2 * r)), int(round(c[1] - 0.2 * r)))
        cv2.circle(layer, cl, max(1, int(round(0.1 * r))), (250, 250, 250), -1)
        img = np.where(opening[..., None] > 0, layer, img)
        pupil_mask &= opening
    return img, face, pupil_mask.astype(bool)


def _redness(img, mask):
    px = img[mask].astype(np.float32)
    return float((px[:, 2] - np.maximum(px[:, 0], px[:, 1])).mean())


def test_zero_strength_is_identity():
    img, face, _ = _face()
    out, diag = remove_red_eye(img, face.landmarks, 0)
    assert out is img
    assert diag["eyes_fixed"] == 0.0


@pytest.mark.parametrize("tone", list(SKIN_TONES))
def test_fixes_red_pupils_on_every_skin_tone(tone):
    img, face, pupil = _face(SKIN_TONES[tone])
    out, diag = remove_red_eye(img, face.landmarks, 100)
    assert diag["eyes_fixed"] == 2.0, diag
    assert _redness(img, pupil) > 100.0
    assert _redness(out, pupil) < 10.0, (tone, _redness(out, pupil))


@pytest.mark.parametrize("tone", list(SKIN_TONES))
@pytest.mark.parametrize("iris", [BROWN_IRIS, BLUE_IRIS])
def test_leaves_normal_dark_pupils_alone(tone, iris):
    img, face, _ = _face(SKIN_TONES[tone], pupil=DARK_PUPIL, iris=iris)
    out, diag = remove_red_eye(img, face.landmarks, 100)
    assert diag["eyes_fixed"] == 0.0
    assert out is img


@pytest.mark.parametrize("tone", list(SKIN_TONES))
def test_red_contact_lens_with_dark_pupil_is_kept(tone):
    """A red cosplay lens has a dark pupil in the middle: not red-eye."""
    img, face, _ = _face(SKIN_TONES[tone], pupil=DARK_PUPIL, iris=RED_LENS,
                         pupil_frac=0.35)
    out, diag = remove_red_eye(img, face.landmarks, 100)
    assert diag["eyes_fixed"] == 0.0
    assert out is img


def test_red_surround_is_not_treated_as_red_eye():
    """Red all around the eye (false face on a red wall / red wig)."""
    img, face, _ = _face(background=(40, 45, 205))
    out, diag = remove_red_eye(img, face.landmarks, 100)
    assert diag["eyes_fixed"] == 0.0


def test_iris_and_catchlight_are_kept():
    img, face, pupil = _face(iris=BLUE_IRIS)
    out, _ = remove_red_eye(img, face.landmarks, 100)
    for iris_idx in (LEFT_IRIS, RIGHT_IRIS):
        c, r = _eye_geom(face, iris_idx)
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.hypot(xx - c[0], yy - c[1]) / r
        ring = (d > 0.7) & (d < 0.9)
        assert np.abs(out[ring].astype(int) - img[ring]).max() <= 1
        cl = (int(round(c[1] - 0.2 * r)), int(round(c[0] - 0.2 * r)))
        assert out[cl].min() >= 245          # catchlight still white


def test_pupil_becomes_neutral_and_dark():
    img, face, pupil = _face()
    out, _ = remove_red_eye(img, face.landmarks, 100)
    px = out[pupil].astype(np.float32)
    assert float(np.ptp(np.median(px, axis=0))) < 8.0
    assert float(np.median(px.max(axis=1))) < 60.0


def test_strength_scales_the_fix():
    img, face, pupil = _face()
    half, _ = remove_red_eye(img, face.landmarks, 50)
    full, _ = remove_red_eye(img, face.landmarks, 100)
    before = _redness(img, pupil)
    assert _redness(full, pupil) < _redness(half, pupil) < before
    assert 0.3 * before < _redness(half, pupil) < 0.7 * before


def test_pixels_away_from_the_eyes_are_bit_exact():
    img, face, _ = _face()
    out, _ = remove_red_eye(img, face.landmarks, 100)
    changed = np.abs(out.astype(int) - img).max(axis=2) > 0
    assert changed.any()
    for iris_idx in (LEFT_IRIS, RIGHT_IRIS):
        c, r = _eye_geom(face, iris_idx)
        yy, xx = np.mgrid[0:H, 0:W]
        changed &= np.hypot(xx - c[0], yy - c[1]) > 1.2 * r
    assert not changed.any()


def test_float32_in_float32_out():
    img, face, _ = _face()
    out, diag = remove_red_eye(img.astype(np.float32), face.landmarks, 80)
    assert out.dtype == np.float32
    assert out.shape == img.shape
    assert diag["eyes_fixed"] == 2.0


def test_needs_iris_landmarks():
    img, face, _ = _face()

    class _Short:
        landmark = face.landmarks.landmark[:468]

    out, diag = remove_red_eye(img, _Short(), 100)
    assert out is img
    assert diag["eyes_fixed"] == 0.0


def test_faces_helper_skips_faces_without_landmarks():
    img, face, _ = _face()

    class _NoLm:
        landmarks = None

    out, diags = remove_red_eye_faces(img, [_NoLm(), face], 100)
    assert [d["eyes_fixed"] for d in diags] == [0.0, 2.0]
    assert out.shape == img.shape


def test_param_is_registered_and_off_by_default():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "red_eye")
    assert spec.default == 0
    assert (spec.min_val, spec.max_val) == (0, 100)
    assert spec.cli_flag == "red-eye"
    assert spec.recipe_key == "eyes.red_eye"


def test_engine_stage_is_a_no_op_when_off_and_runs_when_on():
    from retouch.engine import ProcessingContext, RetouchEngine

    img, face, _ = _face()
    engine = RetouchEngine.__new__(RetouchEngine)

    ctx = ProcessingContext()
    timings = {}
    assert engine._stage_red_eye(img, [face], ctx, timings) is img
    assert "red_eye" not in timings

    ctx.red_eye = 100.0
    out = engine._stage_red_eye(img, [face], ctx, timings)
    assert "red_eye" in timings
    assert ctx._runtime_diagnostics["red_eye"][0]["eyes_fixed"] == 2.0
    assert not np.array_equal(out, img)
