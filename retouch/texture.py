"""Micro-Texture Grain Synthesis & Skin Pore Detail Retention.

Synthesizes tone-adaptive high-frequency skin micro-grain matching natural pore statistics,
preventing flat plastic skin appearance on strong smoothing presets.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def synthesize_micro_grain(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    intensity: float = 0.05,
    grain_size: float = 1.0,
) -> np.ndarray:
    """Synthesize tone-adaptive micro-texture film/skin grain.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 BGR image.
        skin_mask: Optional (H, W) float skin mask [0, 1].
        intensity: Grain amplitude scale [0, 1].
        grain_size: Spatial grain particle scale.

    Returns:
        (H, W, 3) BGR image with natural micro-grain texture injected.
    """
    if intensity <= 0.0:
        return img_bgr

    is_float = img_bgr.dtype == np.float32
    img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8) if is_float else img_bgr

    H, W = img_bgr.shape[:2]

    # Generate Gaussian high-frequency noise grid
    rng = np.random.RandomState(42)
    noise = rng.normal(0.0, 12.0 * intensity, (H, W)).astype(np.float32)

    if grain_size > 1.1:
        k = int(grain_size * 2) | 1
        noise = cv2.GaussianBlur(noise, (k, k), 0)

    # Tone-adaptation: attenuate noise in extreme blacks (<20) and blown highlights (>235)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
    tone_weight = np.clip(gray / 60.0, 0.0, 1.0) * np.clip((240.0 - gray) / 40.0, 0.0, 1.0)

    # Apply skin mask if provided
    if skin_mask is not None:
        sm = skin_mask.astype(np.float32)
        if sm.shape[:2] != (H, W):
            sm = cv2.resize(sm, (W, H))
        tone_weight *= sm

    grain_map = noise * tone_weight
    m = grain_map[:, :, np.newaxis]

    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 0] = np.clip(lab[:, :, 0] + grain_map, 0, 255)

    out_u8 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    if is_float:
        return out_u8.astype(np.float32)
    return out_u8
