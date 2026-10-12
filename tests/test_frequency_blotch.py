"""Tests for R4 — D&B v2: dedicated blotch-band (frequency.combine).

The dedicated blotch band must:
  * be a strict no-op (byte-identical output) when blotch_reduction == 0, and
  * even a broad mid-frequency blotch WITHOUT touching the low-frequency
    form/shading ramp or fine high-frequency pore texture.
"""

from __future__ import annotations

import numpy as np
import cv2

from retouch.frequency import FrequencySeparator


def _make_image(h: int = 256, w: int = 256) -> np.ndarray:
    """BGR uint8 image: low-freq form ramp + broad blotch + fine pore noise."""
    yy, xx = np.mgrid[0:h, 0:w]
    # Low-frequency form ramp (facial shading — must survive untouched).
    ramp = (yy.astype(np.float32) / max(1, h - 1)) * 40.0
    # Broad mid-frequency blotch (redness/pigment patch — should be evened).
    cy, cx = h // 2, w // 2
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    blotch = 55.0 * np.exp(-d2 / (2.0 * 8.0 ** 2))
    # Fine high-frequency pore noise (must survive untouched).
    rng = np.random.default_rng(0)
    pore = rng.standard_normal((h, w)).astype(np.float32) * 12.0

    gray = 128.0 + ramp + blotch + pore
    img = np.clip(gray, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _full_mask(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=np.float32)


def test_blotch_zero_is_byte_identical():
    img = _make_image()
    sep = FrequencySeparator()
    layers = sep.separate(img, face_width=100.0)
    out_default = sep.combine(layers, skin_mask=_full_mask(256, 256), face_width=100.0)
    out_zero = sep.combine(
        layers, skin_mask=_full_mask(256, 256), face_width=100.0, blotch_reduction=0.0
    )
    assert np.array_equal(out_default, out_zero)


def test_blotch_evens_blotch_preserves_form_and_pores():
    img = _make_image()
    sep = FrequencySeparator()
    layers = sep.separate(img, face_width=100.0)
    mask = _full_mask(256, 256)

    out0 = sep.combine(
        layers, skin_mask=mask, smooth_strength=0.0, mid_reduction=0.0,
        blotch_reduction=0.0, face_width=100.0,
    )
    out1 = sep.combine(
        layers, skin_mask=mask, smooth_strength=0.0, mid_reduction=0.0,
        blotch_reduction=1.0, face_width=100.0,
    )

    # --- Blotch is actually reduced at the patch center ---
    cy, cx = 128, 128
    # Local form-ramp baseline at this row (form survives in both outputs).
    ramp_at_row = (cy / 255.0) * 40.0
    base_level = 128.0 + ramp_at_row
    r0 = float(out0[cy, cx, 0]) - base_level
    r1 = float(out1[cy, cx, 0]) - base_level
    assert r1 < r0 * 0.8, f"blotch not sufficiently evened (r0={r0:.1f}, r1={r1:.1f})"

    # --- Change is localized: far from the patch the two outputs agree ---
    # Probe a corner far from the central blotch (form ramp only, no blotch).
    corner = out0[10:20, 10:20].astype(np.int32) - out1[10:20, 10:20].astype(np.int32)
    assert np.abs(corner).max() <= 2, "form ramp altered outside the blotch region"

    # --- Fine pore texture preserved: high-freq energy unchanged ---
    def _hf_energy(arr):
        g = arr[..., 0].astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.std(np.sqrt(gx ** 2 + gy ** 2)))

    e0 = _hf_energy(out0)
    e1 = _hf_energy(out1)
    assert abs(e1 - e0) / max(1e-6, e0) < 0.05, "pore texture energy changed by blotch band"
