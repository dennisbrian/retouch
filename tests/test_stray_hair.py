"""Tests for Stray Hair Cleanup (``retouch/stray_hair.py``, ``hair_remove_flyaways``).

The face is synthetic: a skin patch with its own grain and pores, a wig
region along the top, and thin curved strands of the wig's colour drawn
across the skin with anti-aliasing. Darker skin is simulated by scaling the
whole image in linear light (no real darker-skin photo with strands was
available), which is the stand-in this repo uses for tone checks.
"""
from __future__ import annotations

import numpy as np
import cv2
import pytest

from retouch.stray_hair import (
    detect_stray_hairs,
    exclusion_from_regions,
    remove_stray_hairs,
)

SIZE = 512
FACE_WIDTH = 400.0
SKIN_BGR = (150.0, 170.0, 215.0)
HAIR_BGR = (95.0, 150.0, 190.0)
HAIR_ROWS = 110


def _to_lin(x):
    x = x / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _to_srgb(y):
    y = np.clip(y, 0.0, 1.0)
    return 255.0 * np.where(y <= 0.0031308, 12.92 * y, 1.055 * y ** (1 / 2.4) - 0.055)


def _strand_alpha(p0, p1, p2, width=2.5, opacity=0.85):
    s = 4
    big = np.zeros((SIZE * s, SIZE * s), np.uint8)
    t = np.linspace(0, 1, 120)[:, None]
    pts = (1 - t) ** 2 * np.array(p0) + 2 * (1 - t) * t * np.array(p1) + t ** 2 * np.array(p2)
    cv2.polylines(big, [np.round(pts * s).astype(np.int32)], False, 255,
                  int(round(width * s)), cv2.LINE_AA)
    a = cv2.resize(big.astype(np.float32) / 255.0, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(a, (0, 0), 0.6) * opacity


STRANDS = [
    ((120, 200), (200, 260), (300, 330)),
    ((380, 170), (330, 260), (300, 400)),
    ((90, 420), (180, 380), (260, 440)),
]


def _scene(gain=1.0, strands=True, seed=0):
    rng = np.random.default_rng(seed)
    img = np.empty((SIZE, SIZE, 3), np.float32)
    img[:] = SKIN_BGR
    # broad shading + fine grain + a few pores
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    img *= (0.92 + 0.08 * (xx / SIZE))[..., None]
    grain = cv2.GaussianBlur(rng.normal(0, 3.0, (SIZE, SIZE)).astype(np.float32), (0, 0), 0.8)
    img += grain[..., None]
    for _ in range(60):
        cy, cx = rng.integers(HAIR_ROWS + 10, SIZE - 10), rng.integers(10, SIZE - 10)
        cv2.circle(img, (int(cx), int(cy)), 1, tuple(float(v) * 0.94 for v in SKIN_BGR), -1)
    hair = np.zeros((SIZE, SIZE), np.float32)
    hair[:HAIR_ROWS] = 1.0
    hair_tex = HAIR_BGR + rng.normal(0, 6.0, (HAIR_ROWS, SIZE, 1)).astype(np.float32)
    img[:HAIR_ROWS] = hair_tex
    clean = img.copy()
    alphas = []
    if strands:
        for p0, p1, p2 in STRANDS:
            a = _strand_alpha(p0, p1, p2)
            img = img * (1 - a[..., None]) + np.array(HAIR_BGR, np.float32) * a[..., None]
            alphas.append(a)
    skin = 1.0 - hair
    if gain != 1.0:
        img = _to_srgb(_to_lin(img) * gain).astype(np.float32)
        clean = _to_srgb(_to_lin(clean) * gain).astype(np.float32)
    return img.astype(np.float32), clean.astype(np.float32), skin, hair, alphas


def _removed_fraction(planted, clean, out, alphas):
    fr = []
    for a in alphas:
        sel = a > 0.3
        before = np.abs(planted - clean).max(2)[sel].mean()
        after = np.abs(out - clean).max(2)[sel].mean()
        fr.append(1.0 - after / before)
    return np.array(fr)


class TestNoOp:
    def test_strength_zero_returns_input(self):
        img, _, skin, hair, _ = _scene()
        out, diags = remove_stray_hairs(img, skin, FACE_WIDTH, 0, hair_mask=hair)
        assert out is img
        assert diags["strands"] == 0

    def test_no_skin_mask_is_noop(self):
        img, _, _, hair, _ = _scene()
        out, _ = remove_stray_hairs(img, None, FACE_WIDTH, 80, hair_mask=hair)
        assert out is img

    def test_empty_skin_mask_is_noop(self):
        img, _, _, hair, _ = _scene()
        out, _ = remove_stray_hairs(img, np.zeros(img.shape[:2], np.float32),
                                    FACE_WIDTH, 80, hair_mask=hair)
        assert out is img

    def test_clean_skin_is_left_alone(self):
        img, _, skin, hair, _ = _scene(strands=False)
        out, diags = remove_stray_hairs(img, skin, FACE_WIDTH, 100, hair_mask=hair)
        assert diags["strands"] == 0
        assert np.array_equal(out, img)


class TestRemoval:
    @pytest.mark.parametrize("strength", [50, 100])
    def test_removes_planted_strands(self, strength):
        img, clean, skin, hair, alphas = _scene()
        out, diags = remove_stray_hairs(img, skin, FACE_WIDTH, strength, hair_mask=hair)
        fr = _removed_fraction(img, clean, out, alphas)
        assert diags["strands"] >= 1
        assert fr.min() > 0.6, fr

    def test_wig_and_far_skin_untouched(self):
        img, _, skin, hair, alphas = _scene()
        out, _ = remove_stray_hairs(img, skin, FACE_WIDTH, 100, hair_mask=hair)
        changed = np.abs(out - img).max(2) > 0.5
        # the wig itself never changes
        assert not changed[:HAIR_ROWS - 2].any()
        # skin away from the strands keeps its grain and pores
        near = cv2.dilate((np.maximum.reduce(alphas) > 0.02).astype(np.uint8),
                          np.ones((25, 25), np.uint8)) > 0
        far = (skin > 0.5) & ~near
        assert changed[far].mean() < 0.002

    def test_keeps_skin_grain_inside_healed_strand(self):
        img, clean, skin, hair, alphas = _scene()
        out, _ = remove_stray_hairs(img, skin, FACE_WIDTH, 100, hair_mask=hair)
        sel = alphas[0] > 0.5
        hp = lambda im: im[..., 1] - cv2.GaussianBlur(im[..., 1], (0, 0), 2.0)
        ref = float(np.std(hp(clean)[(skin > 0.5)]))
        healed = float(np.std(hp(out)[sel]))
        # healed strand is neither a flat smear nor noisier than the skin
        assert 0.35 * ref < healed < 2.0 * ref

    def test_uint8_input_keeps_dtype(self):
        img, clean, skin, hair, alphas = _scene()
        out, _ = remove_stray_hairs(np.clip(img, 0, 255).astype(np.uint8), skin,
                                    FACE_WIDTH, 80, hair_mask=hair)
        assert out.dtype == np.uint8
        fr = _removed_fraction(img, clean, out.astype(np.float32), alphas)
        assert fr.min() > 0.5


class TestKeeps:
    def test_skin_crease_is_kept(self):
        """A dark line of the skin's own hue (a smile line, a neck ring) is
        not a strand."""
        img, _, skin, hair, _ = _scene(strands=False)
        a = _strand_alpha((100, 300), (250, 340), (420, 300), width=3.0, opacity=1.0)
        crease = img * (1.0 - 0.18 * a[..., None])
        out, diags = remove_stray_hairs(crease, skin, FACE_WIDTH, 100, hair_mask=hair)
        assert diags["strands"] == 0
        assert np.array_equal(out, crease)

    def test_feature_mask_is_never_touched(self):
        img, _, skin, hair, alphas = _scene()
        feat = np.zeros(img.shape[:2], np.float32)
        feat[:, :256] = 1.0
        out, _ = remove_stray_hairs(img, skin, FACE_WIDTH, 100, hair_mask=hair,
                                    feature_mask=feat)
        assert np.array_equal(out[:, :240], img[:, :240])

    def test_thick_lock_is_kept(self):
        """A lock of hair many fibres wide lying on the skin stays."""
        img, _, skin, hair, _ = _scene(strands=False)
        a = _strand_alpha((100, 250), (250, 300), (420, 260), width=28.0, opacity=1.0)
        lock = img * (1 - a[..., None]) + np.array(HAIR_BGR, np.float32) * a[..., None]
        out, _ = remove_stray_hairs(lock, skin, FACE_WIDTH, 100, hair_mask=hair)
        core = a > 0.9
        assert np.abs(out - lock).max(2)[core].mean() < 1.0

    def test_wig_outline_is_kept(self):
        """The wig's outermost strand, lying along its edge, is the wig's own
        outline (hair on one side), not a stray."""
        img, _, skin, hair, _ = _scene(strands=False)
        a = _strand_alpha((30, HAIR_ROWS + 1), (256, HAIR_ROWS + 2), (480, HAIR_ROWS + 1))
        edge = img * (1 - a[..., None]) + np.array(HAIR_BGR, np.float32) * a[..., None]
        out, _ = remove_stray_hairs(edge, skin, FACE_WIDTH, 100, hair_mask=hair)
        assert np.abs(out - edge).max(2)[a > 0.3].mean() < 2.0


class TestToneInvariance:
    def test_darker_exposure_removes_the_same_share(self):
        """Simulated darker skin (whole image x0.45 in linear light): the
        share of each strand removed stays close to the lighter version."""
        light = _scene(gain=1.0)
        dark = _scene(gain=0.45)
        res = []
        for img, clean, skin, hair, alphas in (light, dark):
            out, _ = remove_stray_hairs(img, skin, FACE_WIDTH, 60, hair_mask=hair)
            res.append(_removed_fraction(img, clean, out, alphas))
        assert res[1].min() > 0.6
        assert np.abs(res[0] - res[1]).max() < 0.2

    def test_without_hair_mask_still_finds_contrasting_strands(self):
        img, clean, skin, _, alphas = _scene()
        out, _ = remove_stray_hairs(img, skin, FACE_WIDTH, 100)
        assert _removed_fraction(img, clean, out, alphas).mean() > 0.5


class TestDetectAndExclusion:
    def test_detect_returns_masks_of_input_shape(self):
        img, _, skin, hair, _ = _scene()
        m, nx, ny, diags = detect_stray_hairs(img, skin, FACE_WIDTH, 60, hair_mask=hair)
        assert m.shape == img.shape[:2] and m.dtype == np.float32
        assert set(np.unique(m)) <= {0.0, 1.0}
        assert diags["strands"] >= 1

    def test_exclusion_grows_feature_masks(self):
        class R:
            pass

        r = R()
        eye = np.zeros((300, 300), np.float32)
        eye[140:150, 140:160] = 1.0
        for name in ("left_eye", "right_eye", "left_iris", "right_iris",
                     "left_eyebrow", "right_eyebrow", "left_under_eye",
                     "right_under_eye", "lips", "mouth_interior", "nose",
                     "nasolabial_l", "nasolabial_r"):
            setattr(r, name, None)
        r.left_eye = eye
        ex = exclusion_from_regions(r, 200.0)
        assert ex is not None
        assert ex.sum() > 4 * eye.sum()
        assert ex[145, 150] == 1.0

    def test_exclusion_none_without_regions(self):
        class R:
            pass

        assert exclusion_from_regions(R(), 200.0) is None
