"""Tests for Lint & Dust Cleanup (retouch/lint_dust.py).

Synthetic scenes with hand-built multiclass confidences, so no model is
needed. Darker exposure is simulated with a linear-light gain (no real
darker-skin cosplay photo with lint in the corpus yet); that is a stand-in,
not real-corpus validation.
"""

import cv2
import numpy as np
import pytest

from retouch.lint_dust import apply_lint_dust
from retouch.params import PROCESSING_PARAMS

H, W = 360, 480
FACE = [(20, 20, 200, 240)]  # face width 200 px: specks up to ~4 px, fibres up to 20 px
SKIN_BGR = np.array([150, 170, 215], np.float32) / 255.0
VINYL_BGR = np.array([0.06, 0.05, 0.07], np.float32)
BACKDROP_BGR = np.array([0.80, 0.80, 0.78], np.float32)


def _lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _srgb(y):
    y = np.clip(y, 0.0, 1.0)
    return np.where(y <= 0.0031308, 12.92 * y, 1.055 * np.power(y, 1 / 2.4) - 0.055)


def _scene(gain=1.0):
    """Skin top-left, black vinyl costume bottom, pale backdrop top-right.

    Returns (img, probs, person). Rows >= 200 are costume, x < 240 above it
    is skin, the rest is backdrop (outside the person).
    """
    img = np.empty((H, W, 3), np.float32)
    img[:] = BACKDROP_BGR
    img[:200, :240] = SKIN_BGR
    img[200:] = VINYL_BGR
    rng = np.random.default_rng(3)
    grain = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), 0.8)
    img = img * (1.0 + 0.02 * grain[..., None])
    probs = np.zeros((H, W, 6), np.float32)
    probs[..., 0] = 0.95
    probs[:200, :240] = 0.0
    probs[:200, :240, 2] = 0.95
    probs[200:] = 0.0
    probs[200:, :, 4] = 0.95
    person = np.zeros((H, W), np.float32)
    person[:200, :240] = 1.0
    person[200:] = 1.0
    if gain != 1.0:
        img = _srgb(_lin(np.clip(img, 0, 1)) * gain)
    return np.clip(img, 0, 1).astype(np.float32), probs, person


def _fibre(img, pts, col, thick=2):
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.polylines(m, [np.array(pts, np.int32)], False, 1, thickness=thick)
    out = img.copy()
    out[m > 0] = col
    return out, m > 0


def _dot(img, c, r, col):
    m = np.zeros(img.shape[:2], np.uint8)
    cv2.circle(m, c, r, 1, -1)
    out = img.copy()
    out[m > 0] = col
    return out, m > 0


def _run(img, probs, person, strength=0.6, face_skin=None):
    return apply_lint_dust(
        img,
        strength,
        FACE,
        segment_classes=lambda u8: cv2.resize(probs, (u8.shape[1], u8.shape[0])),
        hair_full=None,
        person_mask=person,
        face_skin=face_skin,
    )


def _removed(clean, planted, out, m):
    e0 = np.abs(planted - clean).max(axis=2)[m].mean()
    e1 = np.abs(out - clean).max(axis=2)[m].mean()
    return 1.0 - e1 / max(e0, 1e-6)


def test_param_spec_is_opt_in():
    spec = next(p for p in PROCESSING_PARAMS if p.name == "backdrop_cleanup")
    assert spec.default == 0
    assert spec.cli_flag == "backdrop-cleanup"
    assert spec.recipe_key == "background.backdrop_cleanup"


def test_zero_strength_is_identity():
    img, probs, person = _scene()
    out, diag = _run(img, probs, person, strength=0.0)
    assert out is img
    assert diag["reason"] == "off"


def test_no_specks_leaves_image_bit_identical():
    img, probs, person = _scene()
    out, diag = _run(img, probs, person)
    assert np.array_equal(out, img)
    assert not diag["applied"]


@pytest.mark.parametrize("gain", [1.0, 0.45])
def test_light_lint_on_black_costume_removed(gain):
    clean, probs, person = _scene(gain)
    lint = np.array([0.75, 0.75, 0.78], np.float32) * (gain ** (1 / 2.4))
    img, m1 = _fibre(clean, [(300, 260), (305, 268), (310, 274), (316, 279)], lint)
    img, m2 = _dot(img, (150, 320), 2, lint)
    out, diag = _run(img, probs, person)
    assert diag["applied"]
    assert _removed(clean, img, out, m1) > 0.7
    assert _removed(clean, img, out, m2) > 0.7


def test_dark_dust_on_backdrop_removed():
    clean, probs, person = _scene()
    img, m = _dot(clean, (400, 80), 2, np.array([0.2, 0.2, 0.2], np.float32))
    out, _ = _run(img, probs, person)
    assert _removed(clean, img, out, m) > 0.7


def test_skin_specks_are_left_alone():
    clean, probs, person = _scene()
    img, m = _dot(clean, (120, 120), 2, np.array([0.3, 0.3, 0.4], np.float32))
    out, _ = _run(img, probs, person, strength=1.0)
    assert np.array_equal(out[:200, :240], img[:200, :240])


def test_dense_texture_kept():
    """A fishnet-like lattice of bright knots is texture, not lint."""
    clean, probs, person = _scene()
    img = clean.copy()
    for y in range(230, 350, 12):
        for x in range(260, 470, 12):
            img = _dot(img, (x, y), 1, np.array([0.6, 0.6, 0.6], np.float32))[0]
    out, _ = _run(img, probs, person, strength=1.0)
    assert np.array_equal(out[220:, 250:], img[220:, 250:])


def test_glint_on_sheen_kept():
    """A small highlight on a broader sheen is part of the shine."""
    clean, probs, person = _scene()
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    sheen = 0.35 * np.exp(-((xx - 360) ** 2 + (yy - 290) ** 2) / (2 * 14.0 ** 2))
    img = np.clip(clean + sheen[..., None], 0, 1).astype(np.float32)
    img, m = _dot(img, (360, 290), 2, np.array([0.95, 0.95, 0.95], np.float32))
    out, _ = _run(img, probs, person, strength=1.0)
    assert np.array_equal(out[m], img[m])


def test_edges_untouched():
    """The costume/backdrop boundary and a seam line are not specks."""
    clean, probs, person = _scene()
    img, m = _fibre(clean, [(250, 330), (470, 330)], np.array([0.4, 0.4, 0.4], np.float32), 2)
    out, _ = _run(img, probs, person, strength=1.0)
    band = np.zeros((H, W), bool)
    band[190:210] = True
    band |= cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    assert np.array_equal(out[band], img[band])


def test_coloured_speck_takes_surround_colour():
    clean, probs, person = _scene()
    img, m = _dot(clean, (380, 300), 2, np.array([0.1, 0.1, 0.8], np.float32))  # red speck
    out, _ = _run(img, probs, person)
    assert np.abs(out[m] - clean[m]).max() < 0.08


def test_no_segmentation_is_skipped():
    img, _, _ = _scene()
    out, diag = apply_lint_dust(img, 0.6, FACE)
    assert out is img
    assert diag["reason"] == "no_segmentation"
