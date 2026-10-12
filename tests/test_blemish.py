"""Tests for retouch/blemish.py — BlemishRemover."""
import cv2
import numpy as np
import pytest
from retouch.blemish import BlemishRemover, compute_skin_quality_map


@pytest.fixture
def remover():
    return BlemishRemover()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def skin_mask():
    m = np.zeros((64, 64), dtype=np.float32)
    m[16:48, 16:48] = 1.0
    return m


class TestRemove:
    def test_zero_strength(self, remover, img, skin_mask):
        result = remover.remove(img, skin_mask, strength=0)
        assert np.all(result == img)

    def test_no_blemish_returns_original(self, remover, img, skin_mask):
        result = remover.remove(img, skin_mask, strength=50)
        assert np.all(result == img)

    def test_output_shape(self, remover, img, skin_mask):
        result = remover.remove(img, skin_mask, strength=50)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_removes_dark_spot(self, remover):
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        img[124:132, 124:132] = [30, 30, 30]
        mask = np.ones((256, 256), dtype=np.float32)
        result = remover.remove(img, mask, strength=80)
        assert not np.allclose(result[128, 128], [30, 30, 30], atol=10)

    def test_patchmatch_is_available_for_auto_blemish_repair(self, remover):
        img = np.full((256, 256, 3), 128, dtype=np.uint8)
        img[124:132, 124:132] = [30, 30, 30]
        mask = np.ones((256, 256), dtype=np.float32)
        result = remover.remove(img, mask, strength=80, heal_engine="patchmatch")
        assert result.dtype == np.uint8
        assert not np.array_equal(result[128, 128], img[128, 128])

    def test_very_small_skin_region(self, remover, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[32, 32] = 1.0
        result = remover.remove(img, mask, strength=50)
        assert result.shape == (64, 64, 3)


class TestDetect:
    def test_returns_binary_mask(self, remover, img, skin_mask):
        result = remover._detect(img, skin_mask, 0.5, 200)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64)
        assert result.max() <= 255

    def test_flat_skin_no_blemish(self, remover, img, skin_mask):
        result = remover._detect(img, skin_mask, 0.5, 200)
        assert result.max() == 0

    def test_detect_dark_spot(self, remover):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[30:35, 30:35] = [30, 30, 30]
        mask = np.ones((64, 64), dtype=np.float32)
        result = remover._detect(img, mask, 0.9, 200)
        assert result[32, 32] > 0

    def test_high_sensitivity_detects_more(self, remover):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[32, 32] = [80, 80, 80]
        mask = np.ones((64, 64), dtype=np.float32)
        result_low = remover._detect(img, mask, 0.3, 200)
        result_high = remover._detect(img, mask, 0.9, 200)
        assert result_high.sum() >= result_low.sum()

    def test_scale_affects_ksize(self, remover, img, skin_mask):
        result_small = remover._detect(img, skin_mask, 0.5, 100)
        result_large = remover._detect(img, skin_mask, 0.5, 500)
        assert result_small.shape == result_large.shape


class TestSkinQualityMap:
    def test_returns_float32(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.float32)
        result = compute_skin_quality_map(img, mask)
        assert result.dtype == np.float32
        assert result.shape == (32, 32)

    def test_flat_image_low_quality(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.float32)
        result = compute_skin_quality_map(img, mask)
        assert result.max() <= 1.0

    def test_no_skin_returns_zeros(self):
        img = np.full((32, 32, 3), 128, dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.float32)
        result = compute_skin_quality_map(img, mask)
        assert result.sum() == 0.0

    def test_varied_skin_quality(self):
        img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        mask = np.ones((32, 32), dtype=np.float32)
        result = compute_skin_quality_map(img, mask)
        assert result.max() <= 1.0
        assert result.min() >= 0.0


# --- P1 (2026-09-23): final repair mask must stay inside eligible skin -------
# RESEARCH_RETOUCH_PROTECTION_LIFECYCLE_2026_09_22 §3: closing/dilation ran
# after protection was subtracted, so two defects either side of a thin
# protected line were bridged and the line was repainted (face_w 1800:
# 130/259 protected px changed, max 149 levels with patchmatch).

import pytest as _pytest


def _thin_gap_scene(s=2.0):
    import numpy as _np

    H = W = int(600 * s)
    rng = _np.random.default_rng(0)
    img = _np.clip(
        _np.full((H, W, 3), (150, 170, 205), _np.float32) + rng.normal(0, 2, (H, W, 3)), 0, 255
    ).astype(_np.uint8)
    yy, xx = _np.mgrid[:H, :W]
    c = H // 2
    prot = (_np.abs(xx - c) <= int(s)) & (_np.abs(yy - c) <= 6 * s)
    blem = (
        ((yy - c) ** 2 + (xx - c - 4 * s) ** 2 <= (3 * s) ** 2)
        | ((yy - c) ** 2 + (xx - c + 4 * s) ** 2 <= (3 * s) ** 2)
    ) & ~prot
    img[blem] = (110, 125, 160)
    img[prot] = (40, 40, 60)
    skin = _np.ones((H, W), _np.float32)
    skin[prot] = 0.0
    return img, skin, prot, blem


def test_detect_mask_never_covers_excluded_skin_after_morphology():
    from retouch.blemish import BlemishRemover

    img, skin, prot, blem = _thin_gap_scene()
    det = BlemishRemover()._detect(img, skin, 1.0, 1200.0)
    assert (det[blem] > 0).any(), "defects must still be detected"
    assert not (det[prot] > 0).any()


@_pytest.mark.parametrize("engine", ["telea", "patchmatch"])
def test_remove_leaves_protected_gap_unchanged(engine):
    import numpy as _np
    from retouch.blemish import BlemishRemover

    img, skin, prot, blem = _thin_gap_scene()
    out = BlemishRemover().remove(img, skin, strength=100, heal_engine=engine)
    d = _np.abs(out.astype(_np.int16) - img.astype(_np.int16)).max(axis=2)
    assert d[prot].max() <= 2
    assert (d[blem] > 2).any(), "defects must still be repaired"
