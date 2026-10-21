"""Tests for retouch/skin_objects.py (non-skin objects cut out of the skin mask)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.skin_objects import exclude_foreign_objects, foreign_object_mask

H, W, IED = 240, 200, 80.0


def _face(skin_bgr=(120, 150, 200), seed=0):
    """A skin-toned ellipse with mild noise and shading, on a grey backdrop."""
    rng = np.random.default_rng(seed)
    img = np.full((H, W, 3), 90, np.float32)
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (W // 2, H // 2), (70, 95), 0, 0, 360, 1, -1)
    shade = np.linspace(1.05, 0.85, H, dtype=np.float32)[:, None, None]
    face = np.asarray(skin_bgr, np.float32)[None, None, :] * shade + rng.normal(0, 2.0, (H, W, 3))
    img[mask > 0] = face[mask > 0]
    return img, mask.astype(np.float32)


def _u8(img):
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def _plant_clip(img, center=(150, 50)):
    """Saturated pink clip crossing the face outline (hard edge)."""
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, center, (22, 8), -30, 0, 360, 1, -1)
    out = img.copy()
    out[m > 0] = (170, 90, 235)
    return out, m > 0


def _scale_linear(img, gain):
    f = np.clip(img, 0, 255) / 255.0
    lin = np.where(f <= 0.04045, f / 12.92, ((f + 0.055) / 1.055) ** 2.4) * gain
    lin = np.clip(lin, 0, 1)
    return np.where(lin <= 0.0031308, lin * 12.92, 1.055 * lin ** (1 / 2.4) - 0.055) * 255.0


def test_clip_on_face_outline_is_cut():
    img, skin = _face()
    img, clip = _plant_clip(img)
    obj = foreign_object_mask(_u8(img), skin, IED)
    inside_clip = clip & (skin > 0.5)
    assert inside_clip.sum() > 50
    assert (obj[inside_clip] > 0.5).mean() > 0.9


@pytest.mark.parametrize("gain", [0.45, 1.0, 1.5])
def test_clip_cut_same_on_darker_and_lighter_capture(gain):
    img, skin = _face()
    img, clip = _plant_clip(img)
    obj = foreign_object_mask(_u8(_scale_linear(img, gain)), skin, IED)
    assert (obj[clip & (skin > 0.5)] > 0.5).mean() > 0.9
    # The rest of the skin stays skin.
    rest = (skin > 0.5) & ~cv2.dilate(clip.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    assert (obj[rest] > 0.5).mean() < 0.01


@pytest.mark.parametrize("skin_bgr", [(120, 150, 200), (45, 70, 110), (215, 220, 228)])
def test_blush_inside_the_face_stays_skin(skin_bgr):
    """Cheek colour (any skin tone, incl. near-white face paint) is kept."""
    img, skin = _face(skin_bgr)
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    g = np.exp(-(((xx - 70) ** 2 + (yy - 140) ** 2) / (2 * 14.0 ** 2)))
    img[..., 2] += 40 * g
    img[..., 1] -= 15 * g
    obj = foreign_object_mask(_u8(img), skin, IED)
    assert (obj[skin > 0.5] > 0.5).mean() < 0.005


def test_soft_makeup_reaching_the_outline_stays_skin():
    """Colour that fades into the skin (eyeshadow wing, contour) is not an object."""
    img, skin = _face((215, 220, 228))  # white face paint
    yy, xx = np.mgrid[:H, :W].astype(np.float32)
    g = np.clip(1.0 - np.hypot(xx - 170, yy - 90) / 45.0, 0, 1) ** 1.5
    img[..., 2] += 30 * g
    img[..., 0] -= 25 * g
    obj = foreign_object_mask(_u8(img), skin, IED)
    assert (obj[skin > 0.5] > 0.5).mean() < 0.01


def test_spot_inside_the_face_is_not_cut():
    """A mole or a red spot does not touch the face outline: kept."""
    img, skin = _face()
    cv2.circle(img, (100, 130), 5, (40, 40, 150), -1)
    obj = foreign_object_mask(_u8(img), skin, IED)
    assert obj.max() == 0.0


def test_exclude_updates_regions_in_place():
    img, skin = _face()
    img, clip = _plant_clip(img)

    class R:
        pass

    regions = R()
    regions.skin = skin.copy()
    exclude_foreign_objects(regions, _u8(img), IED)
    assert regions.skin.dtype == np.float32
    assert regions.skin[clip & (skin > 0.5)].mean() < 0.1
    assert regions.skin[(skin > 0.5) & ~clip].mean() > 0.95


def test_tiny_or_missing_skin_is_a_no_op():
    img, _ = _face()
    assert foreign_object_mask(_u8(img), np.zeros((H, W), np.float32), IED).max() == 0.0
    assert foreign_object_mask(None, np.ones((H, W), np.float32), IED).max() == 0.0
