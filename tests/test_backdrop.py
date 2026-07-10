"""Tests for retouch/backdrop.py — auto backdrop cleanup."""
import numpy as np
import pytest

from retouch.backdrop import clean_backdrop


def _make():
    """64x64 float [0,1] image: gray bg, colored person blob, defects."""
    img = np.full((64, 64, 3), 200 / 255.0, dtype=np.float32)
    pm = np.zeros((64, 64), dtype=np.float32)
    # person blob (must survive)
    img[10:30, 10:30] = (30 / 255.0, 80 / 255.0, 200 / 255.0)
    pm[10:30, 10:30] = 1.0
    # dust speck on background
    img[40:42, 40:42] = 30 / 255.0
    # crease line on background
    img[5:60, 50] = 150 / 255.0
    return img, pm


def test_zero_strength_noop():
    img, pm = _make()
    out = clean_backdrop(img, pm, 0)
    assert np.allclose(out, img)


def test_removes_dust_and_crease():
    img, pm = _make()
    out = clean_backdrop(img, pm, 100)
    # dust + crease regions should move toward the ~200/255 background
    assert not np.allclose(out[40:42, 40:42], img[40:42, 40:42])
    assert not np.allclose(out[5:60, 50], img[5:60, 50])
    bg = np.full(3, 200 / 255.0)
    assert np.allclose(out[40:42, 40:42].mean(axis=(0, 1)), bg, atol=0.06)


def test_person_untouched():
    img, pm = _make()
    out = clean_backdrop(img, pm, 100)
    assert np.allclose(out[10:30, 10:30], img[10:30, 10:30])


def test_smooth_background_unchanged():
    # No high-frequency defects -> nothing should be inpainted
    img = np.full((64, 64, 3), 180 / 255.0, dtype=np.float32)
    # gentle low-frequency gradient (NOT a defect)
    img = img + np.linspace(0, 40 / 255.0, 64).reshape(-1, 1, 1)
    pm = np.zeros((64, 64), dtype=np.float32)
    pm[10:30, 10:30] = 1.0
    out = clean_backdrop(img, pm, 100)
    assert np.allclose(out, img)


def test_dtype_preserved():
    img, pm = _make()
    out = clean_backdrop(img, pm, 50)
    assert out.dtype == np.float32


def test_none_mask_noop():
    img, _ = _make()
    out = clean_backdrop(img, None, 100)
    assert np.allclose(out, img)
