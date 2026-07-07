"""Unified mask system for region-targeted image operations.

Provides a single, composable API for applying any image operation
to a soft region of an image, with a choice of blend modes and
fine-grained strength control. The companion mask utilities — combine,
feather, threshold, dilate, erode — make it easy to build multi-layer
masks for face-region-aware colour grading, skin-tone protection, and
regional face editing.
"""

from __future__ import annotations

from typing import Callable, Dict, List

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Blend mode primitives
# ---------------------------------------------------------------------------

def _blend_normal(base: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Identity blend — return the op layer untouched.

    Args:
        base: H×W×3 uint8 BGR base image. Unused.
        op: H×W×3 uint8 BGR op image to use as the result.

    Returns:
        H×W×3 uint8 BGR image equal to ``op``.
    """
    return op.copy()


def _blend_multiply(base: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Multiply blend — darkens the base by the op layer.

    Formula: ``base * op / 255`` per channel, clipped to uint8.

    Args:
        base: H×W×3 uint8 or float32 BGR base image.
        op: H×W×3 BGR op image, same dtype as ``base``.

    Returns:
        H×W×3 BGR image, same dtype as ``base``.
    """
    if base.dtype == np.float32:
        base_f = base.astype(np.float32, copy=False)
        op_f = op.astype(np.float32, copy=False)
        return np.clip(base_f * op_f / 255.0, 0.0, 255.0).astype(np.float32)
    base_f = base.astype(np.float32)
    op_f = op.astype(np.float32)
    return np.clip(base_f * op_f / 255.0, 0.0, 255.0).astype(np.uint8)


def _blend_screen(base: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Screen blend — lightens the base by the op layer.

    Formula: ``255 - ((255 - base) * (255 - op) / 255)`` per channel.

    Args:
        base: H×W×3 uint8 or float32 BGR base image.
        op: H×W×3 BGR op image, same dtype as ``base``.

    Returns:
        H×W×3 BGR image, same dtype as ``base``.
    """
    if base.dtype == np.float32:
        base_f = base.astype(np.float32, copy=False)
        op_f = op.astype(np.float32, copy=False)
        return np.clip(
            255.0 - ((255.0 - base_f) * (255.0 - op_f) / 255.0),
            0.0, 255.0,
        ).astype(np.float32)
    base_f = base.astype(np.float32)
    op_f = op.astype(np.float32)
    return np.clip(255.0 - ((255.0 - base_f) * (255.0 - op_f) / 255.0), 0.0, 255.0).astype(np.uint8)


def _blend_soft_light(base: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Soft-light blend — gentle, contrast-preserving overlay.

    Uses the Pegtop approximation
    ``(1 - 2*op) * base^2 + 2*op * base`` in normalised [0, 1] space.
    Matches the formula used in :mod:`retouch.grading`.

    Args:
        base: H×W×3 uint8 or float32 BGR base image.
        op: H×W×3 BGR op image, same dtype as ``base``.

    Returns:
        H×W×3 BGR image, same dtype as ``base``.
    """
    if base.dtype == np.float32:
        base_n = base.astype(np.float32, copy=False) * (1.0 / 255.0)
        op_n = op.astype(np.float32, copy=False) * (1.0 / 255.0)
        result = (1.0 - 2.0 * op_n) * base_n * base_n + 2.0 * op_n * base_n
        return np.clip(result * 255.0, 0.0, 255.0).astype(np.float32)
    base_f = base.astype(np.float32) / 255.0
    op_f = op.astype(np.float32) / 255.0
    result = (1.0 - 2.0 * op_f) * base_f * base_f + 2.0 * op_f * base_f
    return np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8)


def _blend_overlay(base: np.ndarray, op: np.ndarray) -> np.ndarray:
    """Overlay blend — multiply on dark, screen on light.

    For each channel:
    - if ``base < 128``: use ``base * op / 255`` (multiply)
    - else: use ``255 - ((255 - base) * (255 - op) / 255)`` (screen)

    Args:
        base: H×W×3 uint8 or float32 BGR base image.
        op: H×W×3 BGR op image, same dtype as ``base``.

    Returns:
        H×W×3 BGR image, same dtype as ``base``.
    """
    if base.dtype == np.float32:
        base_f = base.astype(np.float32, copy=False)
        op_f = op.astype(np.float32, copy=False)
        multiply = base_f * op_f / 255.0
        screen = 255.0 - ((255.0 - base_f) * (255.0 - op_f) / 255.0)
        result = np.where(base_f < 128.0, multiply, screen)
        return np.clip(result, 0.0, 255.0).astype(np.float32)
    base_f = base.astype(np.float32)
    op_f = op.astype(np.float32)
    multiply = base_f * op_f / 255.0
    screen = 255.0 - ((255.0 - base_f) * (255.0 - op_f) / 255.0)
    result = np.where(base_f < 128.0, multiply, screen)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


_BLEND_MODES: Dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "normal": _blend_normal,
    "multiply": _blend_multiply,
    "screen": _blend_screen,
    "soft_light": _blend_soft_light,
    "overlay": _blend_overlay,
}


# ---------------------------------------------------------------------------
# Mask operations
# ---------------------------------------------------------------------------

def combine_masks(
    masks: List[np.ndarray],
    mode: str = "union",
) -> np.ndarray:
    """Combine multiple same-shape masks into a single float32 mask.

    Args:
        masks: List of H×W float32 (or coercible) masks of the same shape.
        mode: ``"union"`` (max), ``"intersection"`` (min), or
            ``"average"`` (mean).

    Returns:
        H×W float32 mask in [0, 1].

    Raises:
        ValueError: If ``masks`` is empty or ``mode`` is not recognised.
    """
    if not masks:
        raise ValueError("combine_masks: at least one mask is required")
    if mode not in ("union", "intersection", "average"):
        raise ValueError(
            f"combine_masks: unknown mode '{mode}'. "
            "Expected one of 'union', 'intersection', 'average'."
        )

    stack = np.stack([np.asarray(m, dtype=np.float32) for m in masks], axis=0)
    if mode == "union":
        return np.max(stack, axis=0)
    if mode == "intersection":
        return np.min(stack, axis=0)
    return np.mean(stack, axis=0)


def feather_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Gaussian-blur a mask to produce soft edges.

    Args:
        mask: H×W float32 mask in [0, 1].
        radius: Feather radius in pixels. Used to derive the kernel size
            (kernel = ``2 * radius + 1``, minimum 3).

    Returns:
        H×W float32 feathered mask, clipped to [0, 1].
    """
    m = mask.astype(np.float32)
    ksize = max(int(radius) * 2 + 1, 3)
    return np.clip(cv2.GaussianBlur(m, (ksize, ksize), 0), 0.0, 1.0).astype(np.float32)


def threshold_mask(
    mask: np.ndarray,
    low: float,
    high: float = 1.0,
) -> np.ndarray:
    """Zero out mask values outside the ``[low, high]`` band.

    Values inside the inclusive band are preserved; values outside
    are set to 0.0.

    Args:
        mask: H×W float32 mask in [0, 1].
        low: Lower bound (inclusive).
        high: Upper bound (inclusive).

    Returns:
        H×W float32 mask with the same shape as ``mask``.
    """
    m = mask.astype(np.float32)
    return np.where((m >= float(low)) & (m <= float(high)), m, 0.0).astype(np.float32)


def dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphologically dilate a float32 mask using a 3×3 kernel.

    Args:
        mask: H×W float32 mask in [0, 1].
        iterations: Number of 3×3 dilations to apply.

    Returns:
        H×W float32 dilated mask in [0, 1].
    """
    m_u8 = np.clip(mask.astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    out_u8 = cv2.dilate(m_u8, kernel, iterations=int(iterations))
    return (out_u8.astype(np.float32) / 255.0).astype(np.float32)


def erode_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Morphologically erode a float32 mask using a 3×3 kernel.

    Args:
        mask: H×W float32 mask in [0, 1].
        iterations: Number of 3×3 erosions to apply.

    Returns:
        H×W float32 eroded mask in [0, 1].
    """
    m_u8 = np.clip(mask.astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    out_u8 = cv2.erode(m_u8, kernel, iterations=int(iterations))
    return (out_u8.astype(np.float32) / 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Region application
# ---------------------------------------------------------------------------

def apply_to_region(
    img: np.ndarray,
    mask: np.ndarray,
    op: Callable[[np.ndarray], np.ndarray],
    blend_mode: str = "normal",
    strength: float = 1.0,
) -> np.ndarray:
    """Apply an image operation to a soft region of ``img``.

    The pipeline is:

    1. ``op_result = op(img)``
    2. ``blended = blend(img, op_result)`` using the chosen ``blend_mode``
    3. ``final = img * (1 - mask * strength) + blended * (mask * strength)``

    Args:
        img: H×W×3 uint8 or float32 BGR image.
        mask: H×W float32 mask in [0, 1]. 1 = full apply, 0 = untouched.
        op: Function from H×W×3 BGR to H×W×3 BGR, preserving dtype.
        blend_mode: One of ``"normal"``, ``"multiply"``, ``"screen"``,
            ``"soft_light"``, ``"overlay"``.
        strength: 0–1 scalar, fraction of the op to apply.

    Returns:
        H×W×3 BGR image, same dtype as ``img``.

    Raises:
        ValueError: If ``blend_mode`` is not recognised or ``strength``
            is outside [0, 1].
    """
    if blend_mode not in _BLEND_MODES:
        raise ValueError(
            f"apply_to_region: unknown blend_mode '{blend_mode}'. "
            f"Expected one of {sorted(_BLEND_MODES)}."
        )
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(
            f"apply_to_region: strength must be in [0, 1], got {strength}."
        )

    is_float = img.dtype == np.float32
    op_result = op(img)
    blended = _BLEND_MODES[blend_mode](img, op_result)

    m = mask.astype(np.float32)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m.squeeze(-1)
    if m.ndim != 2:
        raise ValueError(
            f"apply_to_region: mask must be 2D or (H, W, 1), got shape {mask.shape}."
        )
    m = np.clip(m, 0.0, 1.0) * float(strength)
    m3 = m[:, :, np.newaxis]

    out = img.astype(np.float32) * (1.0 - m3) + blended.astype(np.float32) * m3
    out = np.clip(out, 0.0, 255.0)
    if is_float:
        return out.astype(np.float32)
    return out.astype(np.uint8)
