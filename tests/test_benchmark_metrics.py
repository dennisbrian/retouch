"""Tests for skin quality metrics (blotch_std and chroma_std).

Tests the functions used by scripts/bench/benchmark.py to measure skin quality
before and after processing.
"""

from __future__ import annotations

import numpy as np
import cv2
import pytest


def test_blotchy_vs_clean_synthetic_skin():
    """Test (a): synthetic blotchy skin has strictly higher blotch_std than clean.

    Creates:
    - A clean smooth-gradient image
    - A blotchy version with medium-scale luminance blobs
    Uses full-ones skin mask and face_width≈200.
    """
    from scripts.bench.benchmark import skin_quality_metrics

    img_h, img_w = 300, 300
    face_width = 200.0

    # Create clean smooth gradient: L goes from 50 to 200 in LAB
    clean_img_bgr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    for y in range(img_h):
        gray_val = int(50 + (y / img_h) * 150)
        clean_img_bgr[y, :] = [gray_val, 128, 128]  # BGR with constant CB/CR-like

    # Create blotchy version: add medium-scale luminance blobs
    blotchy_img_bgr = clean_img_bgr.copy().astype(np.float32)
    # Convert to LAB to manipulate L channel
    blotchy_lab = cv2.cvtColor(blotchy_img_bgr.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    # Add some blobs via circular patterns
    cx, cy = img_w // 2, img_h // 2
    yy, xx = np.ogrid[:img_h, :img_w]
    # Blob 1
    mask1 = ((xx - cx + 30) ** 2 + (yy - cy) ** 2) < 40 ** 2
    blotchy_lab[mask1, 0] += 20
    # Blob 2
    mask2 = ((xx - cx - 30) ** 2 + (yy - cy) ** 2) < 40 ** 2
    blotchy_lab[mask2, 0] -= 15
    # Blob 3
    mask3 = ((xx - cx) ** 2 + (yy - cy + 40) ** 2) < 35 ** 2
    blotchy_lab[mask3, 0] += 25
    blotchy_lab = np.clip(blotchy_lab, 0, 255).astype(np.uint8)
    blotchy_img_bgr = cv2.cvtColor(blotchy_lab, cv2.COLOR_LAB2BGR).astype(np.uint8)

    # Full-ones skin mask
    skin_mask = np.ones((img_h, img_w), dtype=np.float32)

    # Compute metrics
    clean_metrics = skin_quality_metrics(clean_img_bgr, skin_mask, face_width)
    blotchy_metrics = skin_quality_metrics(blotchy_img_bgr, skin_mask, face_width)

    # Blotchy should have strictly higher blotch_std
    assert blotchy_metrics["blotch_std"] > clean_metrics["blotch_std"], (
        f"Blotchy blotch_std {blotchy_metrics['blotch_std']:.4f} "
        f"should be > clean {clean_metrics['blotch_std']:.4f}"
    )


def test_chroma_std_returns_finite():
    """Test (b): chroma_std returns a finite float for typical skin."""
    from scripts.bench.benchmark import skin_quality_metrics

    img_h, img_w = 300, 300
    face_width = 200.0

    # Create a realistic-ish skin-tone image
    skin_img_bgr = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    # Typical skin tone in BGR (neutral skin)
    skin_img_bgr[:, :] = [145, 110, 100]  # B, G, R values

    # Add some variation
    noise = np.random.RandomState(42).randint(-10, 11, (img_h, img_w, 3))
    skin_img_bgr = np.clip(skin_img_bgr.astype(np.int32) + noise, 0, 255).astype(np.uint8)

    # Full-ones skin mask
    skin_mask = np.ones((img_h, img_w), dtype=np.float32)

    # Compute metrics
    metrics = skin_quality_metrics(skin_img_bgr, skin_mask, face_width)

    # Check that chroma_std is finite
    assert np.isfinite(metrics["chroma_std"]), (
        f"chroma_std should be finite, got {metrics['chroma_std']}"
    )
    assert isinstance(metrics["chroma_std"], float)
    # Typical range is 0.01–0.15
    assert 0.0 <= metrics["chroma_std"] <= 0.5, (
        f"chroma_std {metrics['chroma_std']} outside reasonable range"
    )


def test_empty_mask_handled_gracefully():
    """Test (c): skin_mask=None or empty mask handled gracefully.

    Should return zeros or match skin_chroma_std's convention.
    """
    from scripts.bench.benchmark import skin_quality_metrics

    img_h, img_w = 300, 300
    face_width = 200.0

    # Create a simple test image
    test_img_bgr = np.random.randint(0, 255, (img_h, img_w, 3), dtype=np.uint8)

    # Test 1: None mask
    metrics_none = skin_quality_metrics(test_img_bgr, None, face_width)
    assert metrics_none["blotch_std"] == 0.0
    # chroma_std on None mask should also be handled gracefully
    assert np.isfinite(metrics_none.get("chroma_std", 0.0))

    # Test 2: Empty mask
    empty_mask = np.array([], dtype=np.float32).reshape(0, 0)
    metrics_empty = skin_quality_metrics(test_img_bgr, empty_mask, face_width)
    assert metrics_empty["blotch_std"] == 0.0
    assert np.isfinite(metrics_empty.get("chroma_std", 0.0))

    # Test 3: All-zeros mask (no valid pixels > 0.3)
    zero_mask = np.zeros((img_h, img_w), dtype=np.float32)
    metrics_zero = skin_quality_metrics(test_img_bgr, zero_mask, face_width)
    assert metrics_zero["blotch_std"] == 0.0
    # chroma_std on zero mask
    assert np.isfinite(metrics_zero.get("chroma_std", 0.0))
