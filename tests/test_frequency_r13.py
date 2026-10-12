"""Tests for R13 texture v2 — directional (oriented) wrinkle attenuation.

`directional_wrinkle_attenuate` must:
  * soften *anisotropic* (oriented) structure such as wrinkles, while
  * preserving *isotropic* fine texture (pores / noise), and
  * never remove more than (1 - retention_floor) of the band at any pixel.
"""

from __future__ import annotations

import numpy as np
import cv2

from retouch.frequency import directional_wrinkle_attenuate


def _stripe_band(h: int = 128, w: int = 128, freq: float = 0.12, amp: float = 40.0) -> np.ndarray:
    """Vertical sine stripes -> strong horizontal gradient -> anisotropic structure."""
    xx = np.arange(w, dtype=np.float32)
    return (amp * np.sin(2.0 * np.pi * freq * xx)).astype(np.float32)[None, :].repeat(h, axis=0)


def _noise_band(h: int = 128, w: int = 128, amp: float = 40.0, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((h, w)).astype(np.float32) * amp)


def test_wrinkle_attenuated_more_than_noise():
    stripe = _stripe_band()
    noise = _noise_band()
    out_s = directional_wrinkle_attenuate(stripe, strength=1.0, retention_floor=0.3)
    out_n = directional_wrinkle_attenuate(noise, strength=1.0, retention_floor=0.3)

    # Wrinkle (anisotropic) should be softened substantially...
    assert np.mean(np.abs(out_s)) < 0.7 * np.mean(np.abs(stripe)), "wrinkle not attenuated"
    # ...while isotropic noise is preserved far better.
    assert np.mean(np.abs(out_n)) > 0.9 * np.mean(np.abs(noise)), "isotropic texture over-attenuated"


def test_retention_floor_holds_everywhere():
    stripe = _stripe_band(amp=60.0)
    out = directional_wrinkle_attenuate(stripe, strength=1.0, retention_floor=0.3)
    # Never remove more than 70% where real structure exists (skip
    # zero-crossings, where |stripe| ~ 0 and the ratio is undefined).
    mask = np.abs(stripe) > 5.0
    ratio = np.abs(out[mask]) / np.abs(stripe[mask])
    assert np.all(ratio >= 0.3 - 1e-3), "retention floor violated"


def test_strength_zero_is_identity():
    noise = _noise_band()
    out = directional_wrinkle_attenuate(noise, strength=0.0, retention_floor=0.3)
    assert np.allclose(out, noise, atol=1e-3)
