"""Tests for Body Skin Evening (retouch/body_skin_even.py).

Synthetic skin patches stand in for real body skin here. Darker skin is
simulated (a darker, browner base colour), not a real darker-skin sample:
the repo has no real darker-skin body photo yet.
"""

import cv2
import numpy as np
import pytest

from retouch.body_skin_even import even_body_skin
from retouch.engine import RetouchEngine

H = W = 640
FACE_W = 200.0


def _skin(L=65.0, a=14.0, b=16.0, noise=1.5, seed=0):
    """Flat Lab skin with fine luminance texture."""
    rng = np.random.default_rng(seed)
    lab = np.zeros((H, W, 3), np.float32)
    lab[..., 0] = L + rng.normal(0, noise, (H, W)).astype(np.float32)
    lab[..., 1] = a
    lab[..., 2] = b
    return lab


def _blob(cy=320, cx=320, r=40.0):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)).astype(np.float32)


def _bgr(lab):
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1)


def _lab(bgr):
    return cv2.cvtColor(bgr.astype(np.float32), cv2.COLOR_BGR2Lab)


def _plant_red(lab, blob, scale=1.0):
    out = lab.copy()
    out[..., 1] += 7.0 * scale * blob
    out[..., 0] -= 3.0 * scale * blob
    return out


def _redness_removed(clean_lab, strength, scale=1.0):
    blob = _blob()
    mask = np.ones((H, W), np.float32)
    planted = _bgr(_plant_red(clean_lab, blob, scale))
    clean = _bgr(clean_lab)
    out_p = _lab(even_body_skin(planted, mask, strength, FACE_W))
    out_c = _lab(even_body_skin(clean, mask, strength, FACE_W))
    core = blob > 0.8
    before = (_lab(planted) - _lab(clean))[..., 1][core].mean()
    after = (out_p - out_c)[..., 1][core].mean()
    return 1.0 - after / before


class TestNoOp:
    def test_strength_zero_returns_input(self):
        img = _bgr(_skin())
        assert even_body_skin(img, np.ones((H, W), np.float32), 0.0, FACE_W) is img

    def test_empty_mask_returns_input(self):
        img = _bgr(_skin())
        assert even_body_skin(img, np.zeros((H, W), np.float32), 1.0, FACE_W) is img

    def test_none_mask_returns_input(self):
        img = _bgr(_skin())
        assert even_body_skin(img, None, 1.0, FACE_W) is img

    def test_outside_mask_bit_identical(self):
        img = _bgr(_plant_red(_skin(), _blob()))
        mask = np.zeros((H, W), np.float32)
        mask[:, : W // 2] = 1.0
        out = even_body_skin(img, mask, 1.0, FACE_W)
        np.testing.assert_array_equal(out[:, W // 2 + 2:], img[:, W // 2 + 2:])


class TestEvening:
    def test_removes_most_planted_redness_at_full(self):
        assert _redness_removed(_skin(), 1.0) > 0.6

    def test_half_strength_removes_about_half(self):
        full = _redness_removed(_skin(), 1.0)
        half = _redness_removed(_skin(), 0.5)
        assert 0.35 * full < half < 0.65 * full

    def test_simulated_darker_skin_gets_comparable_relative_effect(self):
        light = _redness_removed(_skin(L=68, a=12, b=15), 1.0)
        # Simulated darker skin: lower L*, browner; the planted patch is
        # scaled with the skin's lightness, as real redness shows less.
        dark = _redness_removed(_skin(L=35, a=12, b=20), 1.0, scale=35 / 68)
        assert dark > 0.6
        assert abs(light - dark) < 0.15

    def test_fine_texture_kept(self):
        img = _bgr(_plant_red(_skin(noise=2.0), _blob()))
        out = even_body_skin(img, np.ones((H, W), np.float32), 1.0, FACE_W)

        def tex(x):
            L = _lab(x)[..., 0]
            return float(np.std(L - cv2.GaussianBlur(L, (0, 0), 1.2)))

        assert tex(out) == pytest.approx(tex(img), rel=0.05)

    def test_mole_contrast_kept(self):
        lab = _plant_red(_skin(), _blob())
        mole = _blob(330, 335, 3.0)
        lab_m = lab.copy()
        lab_m[..., 0] -= 25.0 * mole
        mask = np.ones((H, W), np.float32)
        out = _lab(even_body_skin(_bgr(lab_m), mask, 1.0, FACE_W))
        ref = _lab(even_body_skin(_bgr(lab), mask, 1.0, FACE_W))
        core = mole > 0.8
        before = (lab - lab_m)[..., 0][core].mean()
        after = (ref - out)[..., 0][core].mean()
        assert after > 0.9 * before

    def test_plain_shading_untouched(self):
        # A lightness gradient with constant chromaticity is form, not a
        # blotch: lightness and chromaticity must stay.
        lab = _skin(noise=0.5)
        yy, xx = np.mgrid[0:H, 0:W]
        shade = 0.6 + 0.4 * np.exp(-((xx - 320) ** 2) / (2 * 120.0 ** 2))
        lab_s = lab.copy()
        lab_s[..., 0] *= shade
        lab_s[..., 1] *= shade
        lab_s[..., 2] *= shade
        img = _bgr(lab_s)
        out = even_body_skin(img, np.ones((H, W), np.float32), 1.0, FACE_W)
        assert np.abs(_lab(out) - _lab(img)).max() < 1.0

    def test_white_paint_left_alone(self):
        lab = _plant_red(_skin(L=85, a=0.5, b=1.0), _blob(), scale=0.2)
        img = _bgr(lab)
        out = even_body_skin(img, np.ones((H, W), np.float32), 1.0, FACE_W)
        assert np.abs(_lab(out) - _lab(img)).max() < 0.3

    def test_uint8_in_uint8_out(self):
        img = (_bgr(_plant_red(_skin(), _blob())) * 255 + 0.5).astype(np.uint8)
        out = even_body_skin(img, np.ones((H, W), np.float32), 1.0, FACE_W)
        assert out.dtype == np.uint8 and out.shape == img.shape
        assert np.abs(out.astype(int) - img.astype(int)).max() > 0


class TestMaskGrowth:
    def test_grows_same_coloured_skin_not_costume(self):
        lab = _skin(noise=0.5)
        lab[:, 400:] = (20.0, 0.0, 0.0)  # black costume
        img = (_bgr(lab) * 255 + 0.5).astype(np.uint8)
        probs = np.zeros((H, W, 6), np.float32)
        probs[:, :200, 2] = 1.0  # segmenter only found part of the skin
        body = probs[:, :, 2].copy()
        grown = RetouchEngine._grow_body_skin_by_colour(
            img, probs, body, np.ones((H, W), np.float32)
        )
        assert grown[:, 210:390].mean() > 0.9
        assert grown[:, 410:].max() < 0.05

    def test_unconnected_skin_coloured_region_not_grown(self):
        lab = _skin(noise=0.5)
        lab[:, 250:300] = (20.0, 0.0, 0.0)  # costume strip separates them
        img = (_bgr(lab) * 255 + 0.5).astype(np.uint8)
        probs = np.zeros((H, W, 6), np.float32)
        probs[:, :200, 2] = 1.0
        grown = RetouchEngine._grow_body_skin_by_colour(
            img, probs, probs[:, :, 2].copy(), np.ones((H, W), np.float32)
        )
        assert grown[:, 320:].max() < 0.05
