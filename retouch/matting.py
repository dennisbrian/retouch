"""Conservative automatic-trimap alpha-matting spike.

This is an evaluation-stage edge refinement, not a replacement for the
project's statistical person masks.  It only refines an unknown boundary band
and returns the original soft mask when the trimap is not confident enough.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .utils import guided_filter


@dataclass(frozen=True)
class MatteResult:
    alpha: np.ndarray
    trimap: np.ndarray
    confidence: float
    used_fallback: bool


def build_auto_trimap(
    foreground_mask: np.ndarray,
    *,
    hair_mask: Optional[np.ndarray] = None,
    band_radius: int = 4,
) -> np.ndarray:
    """Create a 0/128/255 trimap from a soft foreground and optional hair mask."""
    fg = _mask(foreground_mask)
    if band_radius < 1:
        raise ValueError("band_radius must be >= 1")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_radius * 2 + 1, band_radius * 2 + 1))
    core = cv2.erode((fg > 0.80).astype(np.uint8), kernel) > 0
    background = cv2.erode((fg < 0.08).astype(np.uint8), kernel) > 0
    trimap = np.full(fg.shape, 128, dtype=np.uint8)
    trimap[background] = 0
    trimap[core] = 255
    if hair_mask is not None:
        hair = _mask(hair_mask)
        if hair.shape != fg.shape:
            raise ValueError("hair_mask must match foreground_mask")
        hair_band = cv2.dilate((hair > 0.05).astype(np.uint8), kernel) > 0
        trimap[hair_band & ~core & ~background] = 128
    return trimap


def refine_alpha_matte(
    img_bgr: np.ndarray,
    foreground_mask: np.ndarray,
    *,
    hair_mask: Optional[np.ndarray] = None,
    band_radius: int = 4,
) -> MatteResult:
    """Refine only an automatic trimap's unknown band with guided alpha filtering.

    The closed-form sparse solver belongs behind this spike's synthetic halo
    gate.  This conservative implementation establishes trimap contracts,
    keeps known foreground/background exact, and fails back to the input mask
    on an unusable unknown band.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be an HxWx3 BGR image")
    fg = _mask(foreground_mask)
    if fg.shape != img_bgr.shape[:2]:
        raise ValueError("foreground_mask must match img_bgr")
    trimap = build_auto_trimap(fg, hair_mask=hair_mask, band_radius=band_radius)
    unknown = trimap == 128
    unknown_fraction = float(np.mean(unknown))
    if unknown_fraction < 0.002 or unknown_fraction > 0.45:
        return MatteResult(fg.copy(), trimap, 0.0, True)

    gray = cv2.cvtColor(np.clip(img_bgr, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    try:
        alpha = guided_filter(fg.astype(np.float32), radius=max(2, band_radius * 2), eps=9.0, guide=gray)
    except (cv2.error, ValueError):
        return MatteResult(fg.copy(), trimap, 0.0, True)
    alpha = np.clip(alpha, 0.0, 1.0).astype(np.float32)
    alpha[trimap == 0] = 0.0
    alpha[trimap == 255] = 1.0
    confidence = float(np.clip(1.0 - abs(unknown_fraction - 0.08) / 0.25, 0.0, 1.0))
    return MatteResult(alpha, trimap, confidence, False)


def _mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be a 2D array")
    if not np.isfinite(arr).all():
        raise ValueError("mask must contain finite values")
    arr = arr.astype(np.float32)
    if arr.size and float(arr.max()) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)
