"""Neck Tone Match (retouch/neck_tone_match.py).

Synthetic scenes: an oval face over a neck/chest block, with a shadow band
under the chin. The "retouch" is simulated by brightening or recolouring the
face in the current frame relative to the pre-edit reference. Darker and
lighter skin are simulated with a linear-light gain on the whole scene; that
is a stand-in, not a real darker-skin corpus.
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from retouch.neck_tone_match import neck_support, neck_tone_match
from retouch.params import PROCESSING_PARAMS

H, W = 420, 320
CX, CY, AX, AY = 160, 115, 50, 65  # face ellipse
CHIN_Y = CY + AY
NECK = (slice(CHIN_Y - 15, H), slice(60, 260))  # tucked under the jaw
SHADOW_ROWS = slice(CHIN_Y + 2, CHIN_Y + 28)
LIT = (slice(CHIN_Y + 60, CHIN_Y + 140), slice(110, 210))


def _srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(x):
    x = np.maximum(x, 0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _lab_to_bgr(L, a, b):
    lab = np.array([[[L, a, b]]], np.float32)
    return cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)[0, 0]


def _landmarks():
    """478 normalised points on and inside the face ellipse; 10 = top, 152 = chin."""
    rng = np.random.default_rng(0)
    pts = []
    for i in range(478):
        t = 2 * np.pi * i / 478
        r = 1.0 if i % 2 == 0 else rng.uniform(0.2, 0.9)
        pts.append((CX + AX * r * np.sin(t), CY - AY * r * np.cos(t)))
    pts[10] = (CX, CY - AY)
    pts[152] = (CX, CY + AY)
    lms = [SimpleNamespace(x=x / W, y=y / H) for x, y in pts]
    return SimpleNamespace(landmarks=SimpleNamespace(landmark=lms))


def _face_mask():
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (CX, CY), (AX - 3, AY - 3), 0, 0, 360, 1, -1)
    return m.astype(np.float32)


def _scene(gain=1.0, face_lab=(66, 12, 16), neck_lab=(64, 12, 16), seed=1):
    rng = np.random.default_rng(seed)
    img = np.zeros((H, W, 3), np.float32)
    img[:] = _lab_to_bgr(30, 0, -10)  # cool grey background
    img[NECK] = _lab_to_bgr(*neck_lab)
    # Natural shadow under the chin: 0.55x linear light.
    band = img[SHADOW_ROWS, 60:260]
    img[SHADOW_ROWS, 60:260] = _lin_to_srgb(_srgb_to_lin(band) * 0.55)
    face = _face_mask() > 0
    img[face] = _lab_to_bgr(*face_lab)
    img += rng.normal(0, 0.004, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    if gain != 1.0:
        img = np.clip(_lin_to_srgb(_srgb_to_lin(img) * gain), 0, 1).astype(np.float32)
    person = np.zeros((H, W), np.float32)
    person[NECK] = 1
    person[face] = 1
    return img.astype(np.float32), person


def _brighten_face(img, factor):
    out = img.copy()
    face = _face_mask() > 0
    out[face] = np.clip(_lin_to_srgb(_srgb_to_lin(img[face]) * factor), 0, 1)
    return out


def _lum(img, sl):
    lin = _srgb_to_lin(img[sl])
    return float(np.median(0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]))


def _run(ref, cur, person, strength=100, **kw):
    return neck_tone_match(cur, [_landmarks()], _face_mask(), strength,
                           reference=ref, person_mask=person, **kw)


def test_strength_zero_is_identity():
    ref, person = _scene()
    cur = _brighten_face(ref, 1.5)
    assert _run(ref, cur, person, strength=0) is cur


def test_neck_follows_face_brightening():
    ref, person = _scene()
    cur = _brighten_face(ref, 1.5)
    out = _run(ref, cur, person)
    lit_gain = _lum(out, LIT) / _lum(cur, LIT)
    assert lit_gain == pytest.approx(1.5, rel=0.08)
    # Half strength gives half the change in log terms.
    half = _run(ref, cur, person, strength=50)
    half_gain = _lum(half, LIT) / _lum(cur, LIT)
    assert np.log(half_gain) == pytest.approx(0.5 * np.log(lit_gain), rel=0.15)


def test_shadow_under_chin_is_kept():
    ref, person = _scene()
    cur = _brighten_face(ref, 1.5)
    out = _run(ref, cur, person)
    shadow = (slice(SHADOW_ROWS.start + 4, SHADOW_ROWS.stop - 4), slice(120, 200))
    before = _lum(cur, shadow) / _lum(cur, LIT)
    after = _lum(out, shadow) / _lum(out, LIT)
    assert after == pytest.approx(before, rel=0.05)
    assert after < 0.7  # still clearly a shadow


def test_face_and_background_untouched():
    ref, person = _scene()
    cur = _brighten_face(ref, 1.5)
    out = _run(ref, cur, person)
    face = _face_mask() > 0
    face_in = cv2.erode(face.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    np.testing.assert_allclose(out[face_in], cur[face_in], atol=1e-6)
    np.testing.assert_allclose(out[:, :40], cur[:, :40], atol=1e-6)


def test_colour_follows_face():
    ref, person = _scene()
    cur = ref.copy()
    face = _face_mask() > 0
    lab = cv2.cvtColor(cur, cv2.COLOR_BGR2Lab)
    lab[..., 1][face] -= 6.0  # retouch took redness out of the face
    cur = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)
    out = _run(ref, cur, person)
    a_cur = np.median(cv2.cvtColor(cur, cv2.COLOR_BGR2Lab)[LIT][..., 1])
    a_out = np.median(cv2.cvtColor(out, cv2.COLOR_BGR2Lab)[LIT][..., 1])
    assert a_out - a_cur == pytest.approx(-6.0, abs=1.2)


def test_painted_face_leaves_neck_alone():
    # White face paint: near-neutral face colour. The neck must not be dragged
    # toward it, however the "retouch" moved the painted face.
    # A pale, low-chroma neck keeps it inside the colour sample's reach, so
    # only the paint test itself stops the match.
    ref, person = _scene(face_lab=(90, 1, 2), neck_lab=(64, 5, 6))
    cur = _brighten_face(ref, 1.3)
    out = _run(ref, cur, person)
    np.testing.assert_array_equal(out, cur)


def test_body_painted_neck_left_alone():
    ref, person = _scene(neck_lab=(55, -10, -30))  # blue body paint
    cur = _brighten_face(ref, 1.5)
    out = _run(ref, cur, person)
    np.testing.assert_array_equal(out, cur)


def test_costume_patch_in_other_colour_untouched():
    ref, person = _scene()
    patch = (slice(CHIN_Y + 70, CHIN_Y + 110), slice(200, 250))
    ref[patch] = _lab_to_bgr(40, 45, 25)  # red fabric
    cur = _brighten_face(ref, 1.5)
    out = _run(ref, cur, person)
    inner = (slice(patch[0].start + 6, patch[0].stop - 6), slice(patch[1].start + 6, patch[1].stop - 6))
    assert np.abs(out[inner] - cur[inner]).max() < 0.01
    assert _lum(out, LIT) / _lum(cur, LIT) > 1.3


def test_excluded_hair_untouched():
    ref, person = _scene()
    cur = _brighten_face(ref, 1.5)
    hair = np.zeros((H, W), np.float32)
    hair[:, 60:100] = 1.0
    out = _run(ref, cur, person, exclude_mask=hair)
    np.testing.assert_allclose(out[CHIN_Y + 10:, 62:98], cur[CHIN_Y + 10:, 62:98], atol=1e-6)


@pytest.mark.parametrize("gain", [0.45, 1.3])
def test_tone_invariance_simulated(gain):
    """Same relative effect on a darker or lighter version of the scene.

    Simulated by a linear-light gain; not a substitute for real darker-skin
    photos.
    """
    ref, person = _scene()
    base = np.log(_lum(_run(ref, _brighten_face(ref, 1.4), person), LIT)
                  / _lum(_brighten_face(ref, 1.4), LIT))
    ref2, _ = _scene(gain=gain)
    cur2 = _brighten_face(ref2, 1.4)
    shifted = np.log(_lum(_run(ref2, cur2, person), LIT) / _lum(cur2, LIT))
    assert shifted == pytest.approx(base, rel=0.12)


def test_uint8_in_uint8_out():
    ref, person = _scene()
    cur = _brighten_face(ref, 1.5)
    ref8 = (ref * 255 + 0.5).astype(np.uint8)
    cur8 = (cur * 255 + 0.5).astype(np.uint8)
    out = neck_tone_match(cur8, [_landmarks()], _face_mask(), 100,
                          reference=ref8, person_mask=person)
    assert out.dtype == np.uint8
    assert out[LIT].astype(int).mean() > cur8[LIT].astype(int).mean() + 10


def test_no_reference_no_brightness_change():
    # Without the pre-edit frame the edits can't be measured: brightness is
    # left alone (only a foundation colour pull could act; none here).
    ref, person = _scene()
    cur = _brighten_face(ref, 1.5)
    out = neck_tone_match(cur, [_landmarks()], _face_mask(), 100, person_mask=person)
    assert _lum(out, LIT) == pytest.approx(_lum(cur, LIT), rel=0.01)


def test_support_covers_neck_not_background():
    ref, person = _scene()
    sup = neck_support(ref, [_landmarks()], _face_mask(), reference=ref, person_mask=person)
    assert float(sup[LIT].mean()) > 0.9
    assert float(sup[:, :50].max()) == 0.0


def test_missing_inputs_are_noops():
    ref, _ = _scene()
    cur = _brighten_face(ref, 1.5)
    assert neck_tone_match(cur, [], _face_mask(), 100, reference=ref) is cur
    assert neck_tone_match(cur, [_landmarks()], None, 100, reference=ref) is cur
    assert neck_tone_match(cur, [_landmarks()], np.zeros((H, W), np.float32), 100,
                           reference=ref) is cur


def test_param_spec_registered():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "neck_tone_match")
    assert spec.cli_flag == "neck-tone-match"
    assert spec.recipe_key == "skin.neck_tone_match"
    assert spec.default == 0


def test_stage_runs_after_body_skin_before_body_paint_and_grade():
    from retouch.stage_wrappers import build_global_registry

    names = build_global_registry(object()).names()
    i = names.index("neck_tone_match")
    assert names.index("body_skin") < i < names.index("body_paint")
