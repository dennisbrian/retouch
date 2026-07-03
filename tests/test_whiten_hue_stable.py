"""Tests for whiten hue_stable parameter (Q4 C1 finish)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.skin import SkinProcessor
from retouch.color_science import bgr_to_oklab, oklab_to_oklch


@pytest.fixture
def sp() -> SkinProcessor:
    return SkinProcessor()


@pytest.fixture
def skin_img() -> np.ndarray:
    rand = np.random.RandomState(42)
    return rand.randint(100, 220, (100, 100, 3), dtype=np.uint8)


@pytest.fixture
def skin_mask() -> np.ndarray:
    return np.ones((100, 100), dtype=np.float32)


def test_hue_stable_false_byte_identical(sp: SkinProcessor, skin_img: np.ndarray, skin_mask: np.ndarray):
    """hue_stable=False should match default whiten behavior."""
    result_default = sp.whiten(skin_img, skin_mask, strength=50, tone="rosy")
    result_explicit = sp.whiten(skin_img, skin_mask, strength=50, tone="rosy", hue_stable=False)
    assert np.array_equal(result_default, result_explicit)


def test_hue_stable_true_lifts_luminance(sp: SkinProcessor, skin_img: np.ndarray, skin_mask: np.ndarray):
    """hue_stable=True should increase L in OKLCh."""
    result = sp.whiten(skin_img, skin_mask, strength=50, tone="rosy", hue_stable=True)
    oklab_orig = bgr_to_oklab(skin_img)
    oklab_result = bgr_to_oklab(result)
    L_orig = oklab_orig[..., 0]
    L_result = oklab_result[..., 0]
    skin_idx = skin_mask > 0.3
    mean_L_orig = np.mean(L_orig[skin_idx])
    mean_L_result = np.mean(L_result[skin_idx])
    assert mean_L_result > mean_L_orig + 0.005, "Luminance should increase with positive strength"


def test_hue_stable_true_preserves_hue(sp: SkinProcessor, skin_img: np.ndarray, skin_mask: np.ndarray):
    """hue_stable=True should preserve hue angle (within ~1)."""
    result = sp.whiten(skin_img, skin_mask, strength=50, tone="rosy", hue_stable=True)
    lch_orig = oklab_to_oklch(bgr_to_oklab(skin_img))
    lch_result = oklab_to_oklch(bgr_to_oklab(result))
    h_orig = lch_orig[..., 2]
    h_result = lch_result[..., 2]
    skin_idx = skin_mask > 0.3
    mean_h_diff = np.mean(np.abs((h_result[skin_idx] - h_orig[skin_idx] + 180) % 360 - 180))
    assert mean_h_diff < 2.0, f"Hue should be preserved (diff={mean_h_diff:.2f})"


def test_hue_stable_preserves_chroma(sp: SkinProcessor, skin_img: np.ndarray, skin_mask: np.ndarray):
    """hue_stable=True should keep chroma nearly unchanged."""
    result = sp.whiten(skin_img, skin_mask, strength=50, tone="rosy", hue_stable=True)
    lch_orig = oklab_to_oklch(bgr_to_oklab(skin_img))
    lch_result = oklab_to_oklch(bgr_to_oklab(result))
    C_orig = lch_orig[..., 1]
    C_result = lch_result[..., 1]
    skin_idx = skin_mask > 0.3
    mean_C_diff = np.mean(np.abs(C_result[skin_idx] - C_orig[skin_idx]))
    assert mean_C_diff < 0.005, f"Chroma should be preserved (diff={mean_C_diff:.5f})"


def test_hue_stable_strength_zero(sp: SkinProcessor, skin_img: np.ndarray, skin_mask: np.ndarray):
    result = sp.whiten(skin_img, skin_mask, strength=0, tone="rosy", hue_stable=True)
    assert np.array_equal(result, skin_img)


def test_hue_stable_none_mask(sp: SkinProcessor, skin_img: np.ndarray):
    result = sp.whiten(skin_img, None, strength=50, tone="rosy", hue_stable=True)
    assert np.array_equal(result, skin_img)


def test_hue_stable_valid_output(sp: SkinProcessor, skin_img: np.ndarray, skin_mask: np.ndarray):
    result = sp.whiten(skin_img, skin_mask, strength=50, tone="rosy", hue_stable=True)
    assert result.dtype == np.uint8
    assert result.shape == skin_img.shape
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))
