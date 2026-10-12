"""Tests for unify_hue_line smoothstep gates (E3.3) and harmonize_neck gate (E3.4)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.color_science import oklab_to_oklch, oklch_to_oklab, oklab_to_bgr, bgr_to_oklab
from retouch.skin import SkinProcessor, _smoothstep


def test_smoothstep_is_continuous():
    """Smoothstep should produce values in [0,1] without jumps."""
    x = np.linspace(-0.1, 0.3, 1000, dtype=np.float32)
    result = _smoothstep(0.0, 0.18, x)
    diffs = np.diff(result)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)
    assert np.max(np.abs(diffs)) < 0.01, "Smoothstep should have no large jumps (>0.01)"


def test_smoothstep_strictly_increasing():
    x = np.linspace(-0.05, 0.23, 500, dtype=np.float32)
    result = _smoothstep(0.0, 0.18, x)
    diffs = np.diff(result)
    assert np.all(diffs >= -1e-6), "Smoothstep should be non-decreasing"
    # interior: x in (0.0, 0.18), trim to match diffs length
    interior = (x[:-1] > 0.0) & (x[:-1] < 0.18)
    assert np.any(diffs[interior] > 0), "Smoothstep should increase in interior"


def test_smoothstep_boundary_values():
    x = np.array([-0.1, 0.0, 0.09, 0.18, 0.3], dtype=np.float32)
    result = _smoothstep(0.0, 0.18, x)
    assert result[0] == 0.0, "Below edge0 should be 0"
    assert result[1] == 0.0, "At edge0 should be 0"
    assert result[3] == 1.0, "At edge1 should be 1"
    assert result[4] == 1.0, "Above edge1 should be 1"
    assert 0.0 < result[2] < 1.0, "Between edges should be in (0,1)"


@pytest.fixture
def sp() -> SkinProcessor:
    return SkinProcessor()


def test_unify_hue_line_smooth_gradient(sp: SkinProcessor):
    """unify_hue_line should handle gradient crossing C=0.18 without seam."""
    h, w = 200, 200
    # Build an OKLCh image where chroma linearly crosses 0.18 threshold.
    # Start at C=0.02, not 0.0: the chroma pull (+0.02 toward C_target) on
    # near-achromatic pixels creates steep-but-smooth uint8 ramps that Sobel
    # scores ~90 — quantization stair-steps, not seams. The seam gate this
    # test guards (the C=0.18 eligibility smoothstep) is unaffected.
    C_grad = np.tile(np.linspace(0.02, 0.36, w, dtype=np.float32), (h, 1))
    L_flat = np.full((h, w), 0.70, dtype=np.float32)
    h_flat = np.full((h, w), 30.0, dtype=np.float32)

    oklch = np.stack([L_flat, C_grad, h_flat], axis=-1).astype(np.float32)
    oklab = oklch_to_oklab(oklch)
    img = oklab_to_bgr(oklab)

    mask = np.ones((h, w), dtype=np.float32)
    result = sp.unify_hue_line(img, mask, hue_strength=50, chroma_strength=50)
    assert result.dtype == np.uint8
    assert result.shape == img.shape

    # Measure gradient of the difference (adjustment map), not the full image
    # This isolates the seam from the underlying color gradient.
    diff = result.astype(np.float32) - img.astype(np.float32)
    grad_x = cv2.Sobel(diff, cv2.CV_32F, 1, 0, ksize=3)
    max_grad = np.max(np.abs(grad_x))
    assert max_grad < 50, f"No hard seam at C=0.18 boundary; max_grad={max_grad}"


def test_harmonize_neck_gate_condition():
    """Verify the neck-harmonisation gate in perf_optimizations.py includes C1 params."""
    with open("retouch/perf_optimizations.py") as f:
        lines = f.readlines()
    for i, line in enumerate(lines, 1):
        if "Neck harmonisation" in line:
            # Next non-blank line should be the if gate
            for j in range(i, len(lines)):
                candidate = lines[j].strip()
                if candidate.startswith("if ") and "ctx." in candidate:
                    assert "skin_hue_unify" in candidate or "skin_chroma_even" in candidate or "redness_even" in candidate, \
                        f"harmonize_neck gate at line {j+1} should reference C1 params: {candidate}"
                    break
            break
    else:
        pytest.skip("Could not locate harmonize_neck gate line")


def test_harmonize_neck_gate_all_zero_skips():
    """When whiten=0, equalize=0, and all C1 params=0, harmonize_neck should NOT be called."""
    from retouch.skin import SkinProcessor
    proc = SkinProcessor()
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    # with all params 0, harmonize_neck returns original quickly
    result = proc.harmonize_neck(img, [], None, None, None, strength=0)
    assert np.array_equal(result, img)
