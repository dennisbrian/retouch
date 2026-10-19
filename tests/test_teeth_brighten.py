"""Teeth Brightening: finds teeth by colour on the inner-lip band, lifts them.

Darker skin and exposure are simulated with a linear-light gain on the whole
synthetic face (no real darker-skin smile in the corpus yet).
"""
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from retouch import teeth_brighten as tb

W, H = 400, 300
SKIN = (150, 170, 205)       # BGR, light skin
LIP = (95, 85, 190)          # BGR, red lips
TEETH = (120, 160, 172)      # BGR, ivory-yellow teeth in mouth shade
MOUTH = (40, 35, 60)         # dark mouth


def _landmarks(open_mouth=True):
    pts = [NS(x=0.5, y=0.2) for _ in range(478)]
    pts[468] = NS(x=0.30, y=0.2)
    pts[473] = NS(x=0.70, y=0.2)          # IED = 160 px
    cx, cy = 200, 200
    for i, idx in enumerate(tb._OUTER):
        a = 2 * np.pi * i / len(tb._OUTER)
        pts[idx] = NS(x=(cx + 80 * np.cos(a)) / W, y=(cy + 34 * np.sin(a)) / H)
    half = 14 if open_mouth else 1
    for i, idx in enumerate(tb._INNER):
        a = 2 * np.pi * i / len(tb._INNER)
        pts[idx] = NS(x=(cx + 60 * np.cos(a)) / W, y=(cy + half * np.sin(a)) / H)
    return pts


def _face(open_mouth=True, upper_lip=LIP, rng_seed=0):
    img = np.empty((H, W, 3), np.float32)
    img[:] = SKIN
    cv2.ellipse(img, (200, 200), (80, 34), 0, 0, 360, LIP, -1)
    if upper_lip != LIP:
        cv2.ellipse(img, (200, 200), (80, 34), 0, 180, 360, upper_lip, -1)
    if open_mouth:
        cv2.ellipse(img, (200, 200), (60, 14), 0, 0, 360, MOUTH, -1)
        for x in range(150, 250, 25):           # four teeth with gaps
            cv2.rectangle(img, (x + 2, 190), (x + 22, 205), TEETH, -1)
    noise = np.random.default_rng(rng_seed).normal(0, 1.5, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.float32)


def _teeth_mask():
    mask = np.zeros((H, W), bool)
    for x in range(150, 250, 25):
        mask[192:204, x + 4:x + 21] = True
    return mask


def _lab(img):
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.float32) / 255, cv2.COLOR_BGR2LAB)


def _exposed(img, gain):
    return (tb._linear_to_srgb(tb._srgb_to_linear(img / 255) * gain) * 255).astype(np.float32)


def test_strength_zero_and_missing_landmarks_are_no_ops():
    img = _face()
    assert tb.brighten_teeth(img, _landmarks(), 0) is img
    assert tb.brighten_teeth(img, None, 80) is img
    assert tb.brighten_teeth(img, [NS(x=0, y=0)] * 10, 80) is img


def test_teeth_get_brighter_and_less_yellow_lips_untouched():
    img = _face()
    support = np.zeros((H, W), np.float32)
    out = tb.brighten_teeth(img, _landmarks(), 100, support_out=support)
    teeth = _teeth_mask()
    before, after = _lab(img)[teeth], _lab(out)[teeth]
    assert after[:, 0].mean() > before[:, 0].mean() + 4
    assert after[:, 2].mean() < before[:, 2].mean() - 0.5 * (before[:, 2].mean() - 0.04 * before[:, 0].mean())
    assert support[teeth].mean() > 0.9
    # Lips, skin, and the dark gaps stay bit-identical.
    lips = np.zeros((H, W), np.uint8)
    cv2.ellipse(lips, (200, 200), (80, 34), 0, 0, 360, 1, -1)
    cv2.ellipse(lips, (200, 200), (64, 18), 0, 0, 360, 0, -1)
    np.testing.assert_array_equal(out[lips > 0], img[lips > 0])
    np.testing.assert_array_equal(out[support == 0], img[support == 0])
    assert (support > 0).sum() < 2.0 * teeth.sum()


def test_closed_mouth_changes_nothing():
    img = _face(open_mouth=False)
    out = tb.brighten_teeth(img, _landmarks(open_mouth=False), 100)
    np.testing.assert_array_equal(out, img)


def test_pale_painted_upper_lip_is_not_teeth():
    pale = (195, 200, 222)   # face-paint pale upper lip, a little pink
    img = _face(upper_lip=pale)
    support = np.zeros((H, W), np.float32)
    tb.brighten_teeth(img, _landmarks(), 100, support_out=support)
    upper = np.zeros((H, W), np.uint8)
    cv2.ellipse(upper, (200, 200), (78, 32), 0, 180, 360, 1, -1)
    cv2.ellipse(upper, (200, 200), (62, 18), 0, 0, 360, 0, -1)
    assert support[upper > 0].max() < 0.05
    assert support[_teeth_mask()].mean() > 0.9


@pytest.mark.parametrize('gain', [0.45, 1.4])
def test_same_teeth_and_lift_under_simulated_exposure(gain):
    """Simulated darker/brighter exposure (linear gain), not real skin tones.

    The lift rolls off toward the encoding's white, so brighter teeth get a
    little less; the gates themselves are ratios and pick the same teeth.
    """
    def lift(img):
        support = np.zeros((H, W), np.float32)
        out = tb.brighten_teeth(img, _landmarks(), 60, support_out=support)
        teeth = _teeth_mask()
        y0 = tb._srgb_to_linear(img[teeth] / 255) @ [0.0722, 0.7152, 0.2126]
        y1 = tb._srgb_to_linear(np.clip(out[teeth], 0, 255) / 255) @ [0.0722, 0.7152, 0.2126]
        return support > 0.5, np.log2(y1.mean() / y0.mean())
    base_found, base_stops = lift(_face())
    found, stops = lift(_exposed(_face(), gain))
    overlap = (found & base_found).sum() / max(base_found.sum(), 1)
    assert overlap > 0.9
    assert abs(stops - base_stops) < 0.15
    assert stops > 0.15


def test_bright_teeth_never_clip():
    img = _face()
    img[_teeth_mask()] = (240, 248, 250)
    out = tb.brighten_teeth(img, _landmarks(), 100)
    assert out.max() <= 255.0 + 1e-3
    after = out[_teeth_mask()]
    assert (after.max(axis=1) < 255.0).mean() > 0.95
