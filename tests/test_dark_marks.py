"""Tests for retouch/dark_marks.py (beauty marks kept through matting ops)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.dark_marks import fine_spread, find_dark_marks, restore_dark_marks

IED = 100.0


def _skin(level=180.0, size=200, seed=0):
    rng = np.random.default_rng(seed)
    L = level + cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 1.2) * 3.0
    return L.astype(np.float32), np.ones((size, size), bool)


def _disc(size, cy, cx, r):
    yy, xx = np.mgrid[0:size, 0:size]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


@pytest.mark.parametrize("level,depth", [(180.0, 40.0), (60.0, 14.0)])
def test_finds_a_small_mark_on_light_and_dark_skin(level, depth):
    L, core = _skin(level)
    mole = _disc(200, 100, 100, 2)
    L[mole] -= depth
    marks = find_dark_marks(L, core, IED, fine_spread(L, core, IED))
    assert marks is not None
    assert marks.weight[mole].min() == 1.0
    assert marks.weight.sum() < 60  # nothing else on plain skin


def test_large_dark_area_is_not_a_mark():
    L, core = _skin()
    shade = cv2.GaussianBlur(_disc(200, 100, 100, 25).astype(np.float32), (0, 0), 1.5)
    L -= 40.0 * shade  # a cast shadow / nostril, soft-edged like a real one
    marks = find_dark_marks(L, core, IED, fine_spread(L, core, IED))
    # At most a stray grain on its rim (keeping that grain's own depth is
    # harmless); the shadow itself stays with the op.
    assert marks is None or marks.weight.sum() < 0.01 * shade.sum()


def test_tear_mole_next_to_dark_eye_makeup_is_found():
    """The rednote shrine photo's tear-mole sits ~5 px under red eye
    makeup and the lower lash line; it must still count as a mark."""
    L, core = _skin()
    L[88:94, 70:130] -= 35.0  # lower lid makeup band just above
    mole = _disc(200, 100, 100, 2)
    L[mole] -= 30.0
    marks = find_dark_marks(L, core, IED, fine_spread(L, core, IED))
    assert marks is not None and marks.weight[mole].min() == 1.0


def test_marks_next_to_the_eyes_are_ignored():
    L, core = _skin()
    L[_disc(200, 100, 100, 2)] -= 40.0
    eyes = _disc(200, 85, 100, 8).astype(np.float32)
    assert find_dark_marks(L, core, IED, fine_spread(L, core, IED), eyes_mask=eyes) is None


def test_restore_puts_the_depth_back_after_matting():
    L, core = _skin()
    mole = _disc(200, 100, 100, 2)
    L[mole] -= 40.0
    marks = find_dark_marks(L, core, IED, fine_spread(L, core, IED))
    ring = _disc(200, 100, 100, 9) & ~_disc(200, 100, 100, 5)
    before = float(np.median(L[ring]) - L[mole].mean())
    # A matting op pulls the lit skin down by 12 but leaves the mark.
    matted = np.where(mole, L, L - 12.0).astype(np.float32)
    out = restore_dark_marks(matted, marks)
    after = float(np.median(out[ring]) - out[mole].mean())
    assert after == pytest.approx(before, abs=1.5)
    assert np.array_equal(out[~marks.weight.astype(bool)], matted[~marks.weight.astype(bool)])


def test_restore_never_lightens():
    L, core = _skin()
    L[_disc(200, 100, 100, 2)] -= 40.0
    marks = find_dark_marks(L, core, IED, fine_spread(L, core, IED))
    darker = L - 5.0  # an op that darkened everything, mark included
    assert np.all(restore_dark_marks(darker, marks) <= darker + 1e-4)


def test_shine_removal_keeps_a_mark_on_a_highlight():
    from retouch.skin import SkinProcessor

    size = 240
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    base = np.array([150.0, 170.0, 215.0], np.float32)
    hot = np.exp(-((yy - 120) ** 2 + (xx - 120) ** 2) / (2 * 14.0 ** 2))
    img = base[None, None, :] + hot[..., None] * 38.0
    img = np.clip(img, 0, 255).astype(np.float32)
    mole = _disc(size, 120, 124, 2)
    img[mole] *= 0.55
    mask = np.ones((size, size), np.float32)
    sp = SkinProcessor()
    L = lambda x: cv2.cvtColor(np.clip(x, 0, 255).astype(np.uint8), cv2.COLOR_BGR2Lab)[..., 0].astype(float)
    ring = _disc(size, 120, 124, 9) & ~_disc(size, 120, 124, 5)
    depth = lambda x: float(np.median(L(x)[ring]) - L(x)[mole].mean())
    old = sp.shine_removal(img, mask, 100)
    new = sp.shine_removal(img, mask, 100, ied=IED)
    assert depth(new) >= 0.9 * depth(img)
    assert depth(new) > depth(old)
