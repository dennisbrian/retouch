"""Unified mask system for region-targeted image operations.

Provides a single, composable API for applying any image operation
to a soft region of an image, with a choice of blend modes and
fine-grained strength control. The companion mask utilities — combine,
feather, threshold, dilate, erode — make it easy to build multi-layer
masks for face-region-aware colour grading, skin-tone protection, and
regional face editing.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import cv2
import numpy as np

from retouch.utils import (
    bgr_f32_to_lab_f32,
    guided_filter,
    lab_f32_to_bgr_f32,
    normalize_mask,
)


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


# ---------------------------------------------------------------------------
# Local mask generators
# ---------------------------------------------------------------------------

def radial_mask(
    h: int,
    w: int,
    center_y: float,
    center_x: float,
    radius: float,
    feather: float = 0.2,
) -> np.ndarray:
    """Generate a soft radial gradient mask.

    Args:
        h: Output height in pixels.
        w: Output width in pixels.
        center_y: Center row in [0, 1] relative to image height.
        center_x: Center column in [0, 1] relative to image width.
        radius: Radius in [0, 1] relative to ``min(h, w)``.
        feather: Soft-edge width as a fraction of ``radius``. ``0`` gives a
            hard edge; values up to ``1`` are accepted.

    Returns:
        H×W float32 mask in [0, 1]. Inside the radius the value is 1, outside
        it falls off to 0 across the feather band.
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"radial_mask: h and w must be positive, got h={h}, w={w}")
    if not 0.0 <= feather <= 1.0:
        raise ValueError(f"radial_mask: feather must be in [0, 1], got {feather}")

    cy = float(center_y) * h
    cx = float(center_x) * w
    r_px = max(1.0, float(radius) * min(h, w))
    feather_px = max(1e-6, float(feather) * r_px)

    ys = np.arange(h, dtype=np.float32)[:, np.newaxis]
    xs = np.arange(w, dtype=np.float32)[np.newaxis, :]
    dist = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)

    inner = r_px - feather_px
    m = np.clip((r_px - dist) / feather_px, 0.0, 1.0)
    if inner > 0.0:
        m = np.where(dist <= inner, 1.0, m)
    return m.astype(np.float32)


def linear_mask(
    h: int,
    w: int,
    start_y: float,
    start_x: float,
    end_y: float,
    end_x: float,
    feather: float = 0.1,
) -> np.ndarray:
    """Generate a linear gradient mask from start to end point.

    The mask is 1 at and beyond the ``end`` point and 0 at and beyond the
    ``start`` point, with a soft transition controlled by ``feather``.

    Args:
        h: Output height in pixels.
        w: Output width in pixels.
        start_y: Start row in [0, 1] relative to image height.
        start_x: Start column in [0, 1] relative to image width.
        end_y: End row in [0, 1] relative to image height.
        end_x: End column in [0, 1] relative to image width.
        feather: Soft-edge width as a fraction of the line length.

    Returns:
        H×W float32 mask in [0, 1].
    """
    if h <= 0 or w <= 0:
        raise ValueError(f"linear_mask: h and w must be positive, got h={h}, w={w}")
    if not 0.0 <= feather <= 1.0:
        raise ValueError(f"linear_mask: feather must be in [0, 1], got {feather}")

    sy = float(start_y) * h
    sx = float(start_x) * w
    ey = float(end_y) * h
    ex = float(end_x) * w

    dx = ex - sx
    dy = ey - sy
    length = float(np.sqrt(dx * dx + dy * dy))
    if length < 1e-6:
        return np.zeros((h, w), dtype=np.float32)

    ux = dx / length
    uy = dy / length
    feather_px = max(1e-6, float(feather) * length)

    ys = np.arange(h, dtype=np.float32)[:, np.newaxis]
    xs = np.arange(w, dtype=np.float32)[np.newaxis, :]
    proj = (xs - sx) * ux + (ys - sy) * uy

    # Linear ramp 0 -> 1 across the line, with soft feather bands of width
    # feather_px at both extremes. feather=0 gives a pure linear ramp.
    t = np.clip(proj / length, 0.0, 1.0)
    if feather_px >= length:
        m = t
    else:
        lo = feather_px
        hi = length - feather_px
        m = np.clip((proj - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return m.astype(np.float32)


# ---------------------------------------------------------------------------
# Local-adjustment ops
# ---------------------------------------------------------------------------

def _local_exposure(img_f32: np.ndarray, mask_f32: np.ndarray, strength: float) -> np.ndarray:
    """Shift brightness via LAB L channel, proportional to mask."""
    if strength == 0.0:
        return img_f32
    lab = bgr_f32_to_lab_f32(img_f32)
    L = lab[:, :, 0]
    m = mask_f32[:, :, np.newaxis]
    delta = 60.0 * float(strength)
    L_new = L + (delta * m[:, :, 0])
    lab[:, :, 0] = np.clip(L_new, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


def _local_warmth(img_f32: np.ndarray, mask_f32: np.ndarray, strength: float) -> np.ndarray:
    """Warm/cool shift via LAB a/b channels, proportional to mask.

    Positive strength warms (more red/yellow), negative cools.
    """
    if strength == 0.0:
        return img_f32
    lab = bgr_f32_to_lab_f32(img_f32)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    m = mask_f32
    delta = 25.0 * float(strength)
    a_new = a + delta * m
    b_new = b + delta * m
    lab[:, :, 1] = np.clip(a_new, 0.0, 255.0)
    lab[:, :, 2] = np.clip(b_new, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


def _local_saturation(img_f32: np.ndarray, mask_f32: np.ndarray, strength: float) -> np.ndarray:
    """Scale saturation via LCH C channel, proportional to mask.

    ``strength`` in [-1, 1]: 0 = no change, 1 = doubled, -1 = zeroed.
    """
    if strength == 0.0:
        return img_f32
    lab = bgr_f32_to_lab_f32(img_f32)
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    C = np.sqrt(a * a + b * b)
    H = np.arctan2(b, a)
    m = mask_f32
    scale = np.clip(1.0 + float(strength) * m, 0.0, 3.0)
    C_new = C * scale
    a_new = C_new * np.cos(H) + 128.0
    b_new = C_new * np.sin(H) + 128.0
    lab[:, :, 1] = np.clip(a_new, 0.0, 255.0)
    lab[:, :, 2] = np.clip(b_new, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


def _local_clarity(img_f32: np.ndarray, mask_f32: np.ndarray, strength: float) -> np.ndarray:
    """Boost local contrast via guided-filter detail band, proportional to mask."""
    if strength == 0.0:
        return img_f32
    h, w = img_f32.shape[:2]
    r = max(1, int(min(h, w) * 0.01))
    lab = bgr_f32_to_lab_f32(img_f32)
    L = lab[:, :, 0]
    base = guided_filter(L.astype(np.float32), r, 50.0)
    detail = L - base
    m = mask_f32
    L_new = L + detail * (float(strength) * m)
    lab[:, :, 0] = np.clip(L_new, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


def _local_smooth(img_f32: np.ndarray, mask_f32: np.ndarray, strength: float) -> np.ndarray:
    """Smooth via guided filter on L channel, proportional to mask.

    Skin-safe: preserves high-frequency detail scaled by ``(1 - mask*strength)``,
    so the detail band is reduced (not destroyed) under the mask.
    """
    if strength == 0.0:
        return img_f32
    h, w = img_f32.shape[:2]
    r = max(1, int(min(h, w) * 0.02))
    lab = bgr_f32_to_lab_f32(img_f32)
    L = lab[:, :, 0]
    base = guided_filter(L.astype(np.float32), r, 3.0)
    m = mask_f32
    detail_keep = np.clip(1.0 - float(strength) * m, 0.0, 1.0)
    L_new = base + (L - base) * detail_keep
    lab[:, :, 0] = np.clip(L_new, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


def _local_dodge(img_f32: np.ndarray, mask_f32: np.ndarray, strength: float) -> np.ndarray:
    """Lighten via LAB L channel boost, proportional to mask."""
    if strength == 0.0:
        return img_f32
    lab = bgr_f32_to_lab_f32(img_f32)
    L = lab[:, :, 0]
    m = mask_f32
    gain = 1.0 + 0.3 * float(strength) * m
    L_new = L * gain
    lab[:, :, 0] = np.clip(L_new, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


def _local_burn(img_f32: np.ndarray, mask_f32: np.ndarray, strength: float) -> np.ndarray:
    """Darken via LAB L channel cut, proportional to mask."""
    if strength == 0.0:
        return img_f32
    lab = bgr_f32_to_lab_f32(img_f32)
    L = lab[:, :, 0]
    m = mask_f32
    gain = 1.0 - 0.33 * float(strength) * m
    L_new = L * gain
    lab[:, :, 0] = np.clip(L_new, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


LOCAL_ADJUSTMENT_OPS: Dict[str, Callable[[np.ndarray, np.ndarray, float], np.ndarray]] = {
    "exposure": _local_exposure,
    "warmth": _local_warmth,
    "saturation": _local_saturation,
    "clarity": _local_clarity,
    "smooth": _local_smooth,
    "dodge": _local_dodge,
    "burn": _local_burn,
}


# ---------------------------------------------------------------------------
# Local-adjustment application
# ---------------------------------------------------------------------------

def apply_local_adjustment(
    img: np.ndarray,
    mask: np.ndarray,
    op_name: str,
    strength: float,
    semantic_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply a named local-adjustment op with the given mask and strength.

    Args:
        img: H×W×3 uint8 or float32 BGR image in [0, 255] (float32) or
            [0, 255] (uint8).
        mask: H×W float32 (or coercible) mask in [0, 1] (or [0, 255]).
        op_name: Key into :data:`LOCAL_ADJUSTMENT_OPS`.
        strength: Op strength. Most ops interpret this in [-1, 1] or [0, 1];
            see the per-op docstring.
        semantic_mask: Optional H×W mask to intersect with ``mask`` (e.g. a
            skin mask). Where ``semantic_mask > 0`` the painted mask is kept;
            elsewhere it is zeroed.

    Returns:
        H×W×3 BGR image, same dtype as ``img``.

    Raises:
        ValueError: If ``op_name`` is not registered.
    """
    if op_name not in LOCAL_ADJUSTMENT_OPS:
        raise ValueError(
            f"apply_local_adjustment: unknown op '{op_name}'. "
            f"Expected one of {sorted(LOCAL_ADJUSTMENT_OPS)}."
        )

    is_float = img.dtype == np.float32
    if is_float:
        img_f32 = img.astype(np.float32, copy=True)
    else:
        img_f32 = img.astype(np.float32)

    m = normalize_mask(mask)
    if m is None:
        m = np.ones(img_f32.shape[:2], dtype=np.float32)
    if m.ndim == 3 and m.shape[-1] == 1:
        m = m.squeeze(-1)
    if m.shape != img_f32.shape[:2]:
        m = cv2.resize(m, (img_f32.shape[1], img_f32.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = np.clip(m.astype(np.float32), 0.0, 1.0)

    if semantic_mask is not None:
        sem = normalize_mask(semantic_mask)
        if sem is not None:
            if sem.ndim == 3 and sem.shape[-1] == 1:
                sem = sem.squeeze(-1)
            if sem.shape != m.shape:
                sem = cv2.resize(sem, (m.shape[1], m.shape[0]), interpolation=cv2.INTER_LINEAR)
            m = m * np.clip(sem.astype(np.float32), 0.0, 1.0)
            m = np.clip(m, 0.0, 1.0)

    if float(strength) == 0.0 or m.max() < 1e-6:
        if is_float:
            return img.astype(np.float32, copy=True)
        return img.copy()

    op = LOCAL_ADJUSTMENT_OPS[op_name]
    # Each op applies the mask internally and returns a full-frame result:
    # masked pixels are adjusted, unmasked pixels are unchanged (see the
    # *_local_* ops). Compositing again here would square soft-mask coverage
    # (m -> m^2), steepening falloff for every soft brush. So return the op
    # result directly.
    result = op(img_f32, m, float(strength))
    result = np.clip(result, 0.0, 255.0)

    if is_float:
        return result.astype(np.float32)
    return result.astype(np.uint8)
