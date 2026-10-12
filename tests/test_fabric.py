"""Tests for retouch/fabric.py — auto fabric wrinkle smoothing."""
import numpy as np

from retouch.fabric import smooth_fabric_wrinkles


def _make():
    """64x64 float [0,1] image: cloth-colored region + skin region + fold lines."""
    img = np.full((64, 64, 3), 0.7, dtype=np.float32)
    cloth = np.zeros((64, 64), dtype=np.float32)
    # cloth region (right half)
    cloth[:, 32:] = 1.0
    # skin/face region (left half), distinct tone, must survive
    img[:, :32] = (0.9, 0.6, 0.4)
    # dark vertical fold lines inside the cloth region
    img[8:56, 40] = 0.25
    img[8:56, 48] = 0.25
    return img, cloth


def test_zero_strength_noop():
    img, cloth = _make()
    out = smooth_fabric_wrinkles(img, cloth, 0)
    assert np.allclose(out, img)


def test_removes_dark_fold_lines():
    img, cloth = _make()
    out = smooth_fabric_wrinkles(img, cloth, 100)
    # fold line pixels should lighten toward the ~0.7 cloth tone
    assert not np.allclose(out[8:56, 40], img[8:56, 40])
    assert out[8:56, 40].mean() > img[8:56, 40].mean()
    assert out[8:56, 48].mean() > img[8:56, 48].mean()


def test_skin_region_untouched():
    img, cloth = _make()
    out = smooth_fabric_wrinkles(img, cloth, 100)
    # skin region is the left half; the 2px cloth-mask feather touches only
    # the two columns at the boundary, so check clear of that.
    assert np.allclose(out[:, :30], img[:, :30])


def test_dtype_preserved():
    img, cloth = _make()
    out = smooth_fabric_wrinkles(img, cloth, 50)
    assert out.dtype == np.float32


def test_none_mask_noop():
    img, _ = _make()
    out = smooth_fabric_wrinkles(img, None, 100)
    assert np.allclose(out, img)


def test_empty_mask_noop():
    img, _ = _make()
    empty = np.zeros((64, 64), dtype=np.float32)
    out = smooth_fabric_wrinkles(img, empty, 100)
    assert np.allclose(out, img)
