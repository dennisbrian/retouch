"""P4 skin↔makeup layer helpers (minimal product slice).

strength=0 paths are identity. Full inverse unmix remains spike territory;
this module exposes multi-cue α + coverage evening for engine wiring.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .chromophore import decompose_chromophores, reconstruct_from_chromophores
from .color_science import bgr_to_oklab, oklab_to_oklch


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _prep_mask(mask: Optional[np.ndarray], h: int, w: int) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = mask
    if m.ndim == 3:
        m = m[..., 0]
    if m.shape[:2] != (h, w):
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    m = m.astype(np.float32)
    if m.max() > 1.0:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0)


def estimate_makeup_alpha(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    mole_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Multi-cue soft α prior (chroma ∨ white-paint ∨ reconstruct residual)."""
    h, w = img_bgr.shape[:2]
    m = _prep_mask(skin_mask, h, w)
    if m is None:
        return np.zeros((h, w), dtype=np.float32)

    work = img_bgr
    if work.dtype != np.uint8:
        work = np.clip(work, 0, 255).astype(np.uint8)

    oklab = bgr_to_oklab(work.astype(np.float32))
    oklch = oklab_to_oklch(oklab)
    L, C = oklch[..., 0], oklch[..., 1]

    mel, hb = decompose_chromophores(work)
    recon = reconstruct_from_chromophores(mel, hb, img_bgr=work, skin_mask=m)
    resid = np.abs(work.astype(np.float32) - recon).mean(axis=-1) / 255.0

    cue_chroma = _smoothstep(0.10, 0.20, C)
    cue_white = _smoothstep(0.85, 0.95, L) * (1.0 - _smoothstep(0.02, 0.08, C))
    cue_resid = _smoothstep(0.04, 0.12, resid)

    hint = np.maximum(np.maximum(cue_chroma, cue_white), cue_resid) * m
    mm = _prep_mask(mole_mask, h, w)
    if mm is not None:
        hint = hint * (1.0 - mm)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    u8 = np.clip(hint * 255, 0, 255).astype(np.uint8)
    u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k)
    return (u8.astype(np.float32) / 255.0)


def even_coverage(
    img_bgr: np.ndarray,
    alpha: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Low-pass luminance inside α mask toward regional mean (coverage evening)."""
    if strength <= 0:
        return img_bgr
    s = float(np.clip(strength, 0.0, 1.0))
    h, w = img_bgr.shape[:2]
    a = _prep_mask(alpha, h, w)
    if a is None or a.max() < 1e-6:
        return img_bgr

    src = img_bgr.astype(np.float32)
    sigma = max(3.0, min(h, w) * 0.04)
    blur = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma)
    a3 = (a * s)[..., None]
    out = src * (1.0 - a3) + blur * a3
    if img_bgr.dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.clip(out, 0, 255).astype(np.float32)


def apply_makeup_coverage_even(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    strength: float,
    mole_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Public engine entry: estimate α then even coverage. strength≤0 = identity."""
    if strength is None or float(strength) <= 0:
        return img_bgr
    alpha = estimate_makeup_alpha(img_bgr, skin_mask, mole_mask=mole_mask)
    return even_coverage(img_bgr, alpha, float(strength))
