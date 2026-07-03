"""Tests for H3 hair color unification."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.hairwork import unify_hair_color, hair_flow
from retouch.color_science import bgr_to_oklab, oklab_to_oklch


@pytest.fixture
def hair_img() -> np.ndarray:
    """Synthetic hair image: warm-toned gray on top half."""
    rand = np.random.RandomState(42)
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    # Top half: warm-toned 'hair' (yellow cast)
    for y in range(80):
        img[y, :, :] = [180, 160, 140] + rand.randint(-10, 10, 3)
    return img


@pytest.fixture
def hair_mask() -> np.ndarray:
    mask = np.zeros((200, 200), dtype=np.float32)
    mask[:80, :] = 1.0
    return mask


def test_strength_zero_unchanged(hair_img: np.ndarray, hair_mask: np.ndarray):
    result = unify_hair_color(hair_img, hair_mask, strength=0)
    assert np.array_equal(result, hair_img)


def test_none_mask_unchanged(hair_img: np.ndarray):
    result = unify_hair_color(hair_img, None, strength=50)
    assert np.array_equal(result, hair_img)


def test_unify_reduces_hue_std(hair_img: np.ndarray, hair_mask: np.ndarray):
    """Hue std within hair mask should decrease after unification."""
    result = unify_hair_color(hair_img, hair_mask, strength=80)
    lch_orig = oklab_to_oklch(bgr_to_oklab(hair_img))
    lch_result = oklab_to_oklch(bgr_to_oklab(result))
    mask_bin = hair_mask > 0.05
    h_std_before = float(np.std(lch_orig[..., 2][mask_bin]))
    h_std_after = float(np.std(lch_result[..., 2][mask_bin]))
    assert h_std_after < h_std_before * 0.95, f"Hue std should decrease: {h_std_before:.4f} -> {h_std_after:.4f}"


def test_luminance_unchanged(hair_img: np.ndarray, hair_mask: np.ndarray):
    """L channel should be unchanged (shading preservation)."""
    result = unify_hair_color(hair_img, hair_mask, strength=80)
    lch_orig = oklab_to_oklch(bgr_to_oklab(hair_img))
    lch_result = oklab_to_oklch(bgr_to_oklab(result))
    mask_bin = hair_mask > 0.05
    L_diff = np.max(np.abs(
        lch_result[..., 0][mask_bin].astype(float) - lch_orig[..., 0][mask_bin].astype(float)
    ))
    assert L_diff < 0.01, f"Luminance should be preserved, max diff={L_diff:.4f}"


def test_valid_output_type_and_shape(hair_img: np.ndarray, hair_mask: np.ndarray):
    result = unify_hair_color(hair_img, hair_mask, strength=50)
    assert result.dtype == np.uint8
    assert result.shape == hair_img.shape
    assert not np.any(np.isnan(result))
    assert not np.any(np.isinf(result))


def test_target_hue_override():
    """Explicit target_hue should pull toward it."""
    from retouch.color_science import bgr_to_oklab, oklab_to_oklch

    rand = np.random.RandomState(42)
    red_img = np.zeros((200, 200, 3), dtype=np.uint8)
    for y in range(80):
        red_img[y, :, :] = [100, 80, 220] + rand.randint(-5, 5, 3)
    red_mask = np.zeros((200, 200), dtype=np.float32)
    red_mask[:80, :] = 1.0

    result = unify_hair_color(red_img, red_mask, strength=80, target_hue=240.0)
    lch_result = oklab_to_oklch(bgr_to_oklab(result))
    mask_bin = red_mask > 0.05
    mean_h_before = float(np.mean(oklab_to_oklch(bgr_to_oklab(red_img))[..., 2][mask_bin]))
    mean_h_after = float(np.mean(lch_result[..., 2][mask_bin]))
    dist_before = min(abs(mean_h_before - 240), 360 - abs(mean_h_before - 240))
    dist_after = min(abs(mean_h_after - 240), 360 - abs(mean_h_after - 240))
    assert dist_after < dist_before * 0.95, \
        f"Distance to target should decrease: {dist_before:.1f} -> {dist_after:.1f}"


def test_target_chroma_override(hair_img: np.ndarray, hair_mask: np.ndarray):
    """Explicit target_chroma should pull chroma toward it."""
    result = unify_hair_color(hair_img, hair_mask, strength=80, target_chroma=0.01)
    lch_result = oklab_to_oklch(bgr_to_oklab(result))
    mask_bin = hair_mask > 0.05
    mean_C_after = float(np.mean(lch_result[..., 1][mask_bin]))
    assert mean_C_after < 0.05, f"Chroma should be pulled toward 0.01, got {mean_C_after:.4f}"


def test_too_few_hair_pixels_returns_unchanged():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.float32)
    mask[0, 0] = 1.0  # Only 1 pixel
    result = unify_hair_color(img, mask, strength=50)
    assert np.array_equal(result, img)
