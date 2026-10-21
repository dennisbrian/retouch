"""Tests for retouch/halation.py (point-light halation)."""

import numpy as np
import pytest

from retouch.grading import ColorGrader
from retouch.halation import apply_point_light_halation, halation_sources


def _scene(w=1500, h=1000, bg=0.08):
    img = np.full((h, w, 3), bg, np.float32)
    lamp = (h // 2, w // 3)
    r = max(2, w // 150)
    img[lamp[0] - r:lamp[0] + r, lamp[1] - r:lamp[1] + r] = 1.0
    return img, lamp, r


def test_glow_is_red_orange_around_a_lamp():
    img, (y, x), r = _scene()
    out = apply_point_light_halation(img, 0.5)
    d = out[y, x + 4 * r] - img[y, x + 4 * r]
    b, g, rr = d
    assert rr > 0.03
    assert rr > g > b >= 0


def test_glow_size_follows_the_photo_not_pixels():
    radii = []
    for w in (1000, 6000):
        img, (y, x), _ = _scene(w=w, h=w * 2 // 3)
        d = (apply_point_light_halation(img, 0.5) - img)[y, :, 2]
        radii.append((d > 0.02).sum() / w)
    assert radii[0] == pytest.approx(radii[1], rel=0.25)


def test_unclipped_pale_area_and_skin_never_seed_a_glow():
    img = np.full((800, 1200, 3), 0.08, np.float32)
    img[200:400, 200:400] = 0.85          # lit white wig, below the ceiling
    img[400:600, 700:900] = 1.0           # clipped highlight on skin
    skin = np.zeros(img.shape[:2], np.float32)
    skin[400:600, 700:900] = 1.0
    out = apply_point_light_halation(img, 1.0, skin_mask=skin)
    assert np.array_equal(out, img)
    assert halation_sources(img, skin).max() == 0.0


def test_large_blown_area_does_not_seed_a_band():
    # A blown window is never several times brighter than its own surround
    # (it is most of that surround), so it does not paint an orange band;
    # small lights next to it still glow.
    img = np.full((800, 1200, 3), 0.05, np.float32)
    img[:, 600:] = 1.0
    img[400:408, 300:308] = 1.0
    src = halation_sources(img)
    assert src[:, 600:].max() < 0.05
    assert src[404, 304] > 0.5


def test_untouched_pixels_are_bit_identical_and_dtype_kept():
    img, _, _ = _scene()
    u8 = np.round(img * 255).astype(np.uint8)
    out = apply_point_light_halation(u8, 0.5)
    assert out.dtype == np.uint8
    assert np.array_equal(out[:20, -20:], u8[:20, -20:])
    assert apply_point_light_halation(img, 0.0) is img


def test_grade_dispatches_point_light_mode():
    img, (y, x), r = _scene()
    grader = ColorGrader()
    out = grader.grade(img, {"halation": {"mode": "point_light", "intensity": 0.5}},
                       1.0, return_float=True, skin_mask=np.zeros(img.shape[:2], np.float32))
    assert out[y, x + 4 * r, 2] > img[y, x + 4 * r, 2] + 0.03
