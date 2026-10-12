"""Film-like highlight roll-off and tone mapping operators.

Replaces hard digital clipping (255) with a soft shoulder that mimics the
H&D curve of analog film stock. The Fuji JPEG look is dominated by a soft
shoulder on the highlight roll-off — see docs/FUJI_COLOR_RESEARCH.md §3.2.
"""

from __future__ import annotations

import cv2
import numpy as np


def soft_clip_highlights(
    img_bgr: np.ndarray,
    threshold: float = 230.0,
    rolloff_start: float = 200.0,
) -> np.ndarray:
    """Compress highlights above ``rolloff_start`` toward ``threshold``.

    For ``x <= rolloff_start`` the output equals the input (linear region).
    For ``x > rolloff_start`` the value is mapped through an exponential
    approach to ``threshold``::

        out = rolloff_start
            + (threshold - rolloff_start) * (1 - exp(-k * (x - rolloff_start)))

    with ``k = 1 / (threshold - rolloff_start)`` chosen so the curve is
    C¹-continuous at ``rolloff_start`` (slope 1.0 matches the linear region
    below, so the join is smooth).

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
        threshold: Asymptotic upper bound. Output is clamped to this value.
        rolloff_start: Pixel value where the soft shoulder begins.

    Returns:
        Same dtype as input: (H, W, 3) BGR image with no value above ``threshold``.
    """
    if rolloff_start >= threshold:
        return np.clip(img_bgr, 0, threshold).astype(img_bgr.dtype)

    x = img_bgr.astype(np.float32)
    headroom = float(threshold - rolloff_start)
    k = 1.0 / headroom
    excess = np.maximum(x - rolloff_start, 0.0)
    compressed = rolloff_start + headroom * (1.0 - np.exp(-k * excess))
    out = np.where(x <= rolloff_start, x, compressed)
    return np.clip(out, 0, threshold).astype(img_bgr.dtype)


def apply_highlight_rolloff(
    img_bgr: np.ndarray,
    strength: float = 1.0,
) -> np.ndarray:
    """Blend between input and soft-clipped output by ``strength``.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
        strength: 0.0 (passthrough) to 1.0 (full rolloff).

    Returns:
        Same dtype as input: (H, W, 3) BGR image.
    """
    s = float(np.clip(strength, 0.0, 1.0))
    if s <= 0.0:
        return img_bgr
    soft = soft_clip_highlights(img_bgr)
    if img_bgr.dtype == np.float32:
        return img_bgr * (1.0 - s) + soft * s
    return cv2.addWeighted(img_bgr, 1.0 - s, soft, s, 0)


def recover_highlights(
    img_bgr: np.ndarray,
    threshold: float = 240.0,
    amount: float = 0.3,
) -> np.ndarray:
    """Lift values below ``threshold`` partway toward ``threshold``.

    For each pixel ``v`` with ``v < threshold`` the function applies::

        v_new = v + amount * (threshold - v)

    Pixels at or above ``threshold`` are unchanged.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        threshold: Ceiling above which pixels are not lifted.
        amount: [0, 1] lift strength.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    if amount <= 0.0:
        return img_bgr
    x = img_bgr.astype(np.float32)
    below = x < threshold
    lifted = x + amount * (threshold - x)
    out = np.where(below, lifted, x)
    return np.clip(out, 0, 255).astype(np.uint8)


def tone_map_reinhard(
    img_bgr: np.ndarray,
    exposure: float = 1.0,
) -> np.ndarray:
    """Per-channel Reinhard tone mapping.

    Maps ``out = exposure * x / (1 + exposure * x)`` on a [0, 1] input
    range. The standard film-like compression operator.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        exposure: Linear exposure multiplier applied before compression.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    x = img_bgr.astype(np.float32) / 255.0
    x = x * float(exposure)
    out = x / (1.0 + x)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _hable_filmic(x: np.ndarray) -> np.ndarray:
    A = 0.15
    B = 0.50
    C = 0.10
    D = 0.20
    E = 0.02
    F = 0.30
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F


def tone_map_filmic(img_bgr: np.ndarray) -> np.ndarray:
    """Hable / Uncharted2-style filmic tone mapping (simplified).

    The Hable operator is normalized so that an input of 1.0 maps to 1.0,
    giving a clean filmic S-curve with a soft toe and a soft shoulder.
    Operates per-channel.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    x = img_bgr.astype(np.float32) / 255.0
    white_scale = 1.0 / _hable_filmic(np.array([1.0], dtype=np.float32))[0]
    out = _hable_filmic(x) * white_scale
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)
