import numpy as np

from retouch.matting import build_auto_trimap, refine_alpha_matte


def test_auto_trimap_has_known_and_unknown_regions():
    mask = np.zeros((96, 96), np.float32)
    mask[18:78, 20:76] = 1.0
    trimap = build_auto_trimap(mask, band_radius=3)
    assert np.any(trimap == 0)
    assert np.any(trimap == 128)
    assert np.any(trimap == 255)


def test_refine_matte_locks_known_regions_and_stays_bounded():
    image = np.full((96, 96, 3), 120, dtype=np.uint8)
    image[:, 48:] = (180, 80, 40)
    mask = np.zeros((96, 96), np.float32)
    mask[18:78, 20:76] = 1.0
    result = refine_alpha_matte(image, mask, band_radius=3)
    assert result.alpha.dtype == np.float32
    assert np.isfinite(result.alpha).all()
    assert result.alpha.min() >= 0.0 and result.alpha.max() <= 1.0
    assert np.all(result.alpha[result.trimap == 0] == 0.0)
    assert np.all(result.alpha[result.trimap == 255] == 1.0)


def test_unusable_trimap_returns_input_mask_as_fallback():
    image = np.full((32, 32, 3), 100, dtype=np.uint8)
    mask = np.ones((32, 32), np.float32)
    result = refine_alpha_matte(image, mask)
    assert result.used_fallback
    assert np.array_equal(result.alpha, mask)
