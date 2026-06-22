"""Shared utilities for the face retouching pipeline.

Provides mask normalization, blending helpers, landmark extraction utilities,
and colour/tone primitives used across the retouch modules.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Mask utilities
# ---------------------------------------------------------------------------

def normalize_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return a float32 mask in [0, 1]. Returns None if input is None.

    Accepts uint8 (0-255) or already-float (0-1) masks and normalizes
    the former by dividing by 255.
    """
    if mask is None:
        return None
    m = mask.astype(np.float32)
    if m.max() > 1.0:
        m /= 255.0
    return m


def squeeze_mask(mask: np.ndarray) -> np.ndarray:
    """Squeeze trailing singleton dim from a mask array.

    Masks are often stored as (H, W, 1) for broadcasting with (H, W, 3) images.
    This converts them to (H, W) for stage-internal use.

    Args:
        mask: Array of shape (H, W) or (H, W, 1).

    Returns:
        Array of shape (H, W). Pass-through if already 2D.
    """
    if mask.ndim == 3 and mask.shape[-1] == 1:
        return mask.squeeze(-1)
    return mask


def estimate_face_width(
    skin_mask: Optional[np.ndarray] = None,
    lip_mask: Optional[np.ndarray] = None,
    img_shape: Optional[Tuple[int, int]] = None,
    fallback_ratio: float = 0.25,
) -> int:
    """Estimate face width in pixels from facial region masks.

    Tries the most reliable signal first (skin mask x-extent), then falls back
    to lip-mask-derived width, then to image-shape x ratio.

    Args:
        skin_mask: Full-image skin mask in [0, 1] or [0, 255].
        lip_mask: Lip region mask, used as fallback (x 3.3 heuristic).
        img_shape: (height, width) used as final fallback.
        fallback_ratio: Fraction of image width to use as last-resort estimate.

    Returns:
        Face width in pixels, clamped to a minimum of 32.
    """
    if skin_mask is not None:
        cols = np.where(skin_mask.max(axis=0) > 0.5)[0]
        if len(cols) > 0:
            return max(32, int(cols[-1] - cols[0]))
    if lip_mask is not None:
        cols = np.where(lip_mask.max(axis=0) > 0.5)[0]
        if len(cols) > 0:
            return max(32, int((cols[-1] - cols[0]) * 3.3))
    if img_shape is not None:
        return max(32, int(img_shape[1] * fallback_ratio))
    return 64


def screen_blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Screen-blend two images in the 0-255 range.

    screen = 255 - ((255 - a) * (255 - b) / 255)

    This non-clipping blend lightens without saturating; equivalent to the
    "Screen" layer mode in Photoshop.

    Args:
        a: First image or scalar in [0, 255].
        b: Second image or scalar in [0, 255].

    Returns:
        Screen-blended result with the broadcast shape.
    """
    return 255.0 - ((255.0 - a) * (255.0 - b) / 255.0)


def feather_mask(
    mask: Optional[np.ndarray],
    radius: Optional[int] = None,
    sigma: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Apply Gaussian feathering to a mask for soft edges.

    Args:
        mask: Float mask (H, W), values 0.0–1.0. May be None.
        radius: Feather radius in pixels.
        sigma: Gaussian sigma. Derived from radius if not given.

    Returns:
        Feathered float mask (H, W), values 0.0–1.0. Returns the input as-is
        if it is None or empty.
    """
    if mask is None or mask.size == 0:
        return mask

    mask_f = normalize_mask(mask)

    if sigma is None and radius is None:
        return mask_f

    if sigma is None:
        sigma = max(radius / 3.0, 1.0)
    if radius is None:
        radius = int(sigma * 3)

    ksize = max(radius * 2 + 1, 3)
    return cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)


def blend_masked(
    original: np.ndarray,
    processed: np.ndarray,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Alpha-blend *processed* onto *original* using a soft mask.

    Args:
        original: (H, W, C) uint8 image.
        processed: (H, W, C) uint8 image.
        mask: (H, W) float mask, 0.0–1.0. If None, ``processed`` is returned.

    Returns:
        Blended (H, W, C) uint8 image.
    """
    if mask is None:
        return processed

    m = mask.astype(np.float32)
    if m.ndim == 2:
        m = m[:, :, np.newaxis]

    out = original.astype(np.float32) * (1.0 - m) + processed.astype(np.float32) * m
    return np.clip(out, 0, 255).astype(np.uint8)


def create_polygon_mask(
    points: np.ndarray,
    img_shape: Tuple[int, ...],
    feather_radius: int = 0,
) -> np.ndarray:
    """Create a filled-polygon mask from an ordered set of (x, y) points.

    Args:
        points: (N, 2) int32 array of pixel coordinates.
        img_shape: (H, W) or (H, W, C).
        feather_radius: Gaussian feather radius. 0 = hard edge.

    Returns:
        Float mask (H, W), 0.0–1.0.
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [points.reshape(-1, 1, 2)], 255)
    mask_f = mask.astype(np.float32) / 255.0
    if feather_radius > 0:
        mask_f = feather_mask(mask_f, radius=feather_radius)
    return mask_f


# ---------------------------------------------------------------------------
# Landmark helpers
# ---------------------------------------------------------------------------

def get_points(
    landmarks: Any,
    indices: Sequence[int],
    w: int,
    h: int,
) -> np.ndarray:
    """Extract pixel (x, y) coords from MediaPipe landmarks.

    Args:
        landmarks: MediaPipe NormalizedLandmarkList.
        indices: List[int] of landmark indices.
        w: Image width in pixels.
        h: Image height in pixels.

    Returns:
        (N, 2) int32 numpy array of pixel coordinates.
    """
    pts = []
    for idx in indices:
        lm = landmarks.landmark[idx]
        pts.append([int(lm.x * w), int(lm.y * h)])
    return np.array(pts, dtype=np.int32)


def inter_eye_distance(landmarks: Any, w: int, h: int) -> float:
    """Distance between iris centers (or eye-corner midpoints as fallback).

    Args:
        landmarks: MediaPipe NormalizedLandmarkList.
        w: Image width in pixels.
        h: Image height in pixels.

    Returns:
        Inter-eye distance in pixels.
    """
    try:
        left = landmarks.landmark[468]   # left iris center
        right = landmarks.landmark[473]  # right iris center
    except (IndexError, AttributeError):
        # Fallback: midpoint of eye corners
        li, lo = landmarks.landmark[133], landmarks.landmark[33]
        ri, ro = landmarks.landmark[362], landmarks.landmark[263]
        lx, ly = (li.x + lo.x) / 2, (li.y + lo.y) / 2
        rx, ry = (ri.x + ro.x) / 2, (ri.y + ro.y) / 2
        dx, dy = (rx - lx) * w, (ry - ly) * h
        return float(np.sqrt(dx * dx + dy * dy))

    dx = (right.x - left.x) * w
    dy = (right.y - left.y) * h
    return float(np.sqrt(dx * dx + dy * dy))


def adaptive_ksize(face_width: float, factor: float = 0.1, minimum: int = 3) -> int:
    """Return an odd kernel size proportional to face width.

    Args:
        face_width: Approximate face width in pixels.
        factor: Scaling factor for kernel size.
        minimum: Lower bound for the kernel size.

    Returns:
        Odd integer kernel size.
    """
    size = max(int(face_width * factor), minimum)
    return size | 1  # ensure odd


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def vibrance(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray],
    strength: float,
) -> np.ndarray:
    """Smart saturation: boosts under-saturated pixels more.

    When *mask* is None (full-image mode), red-orange skin hues are
    protected from over-saturation. *strength* is the raw intensity in
    [-1.0, 1.0]; 1.0 corresponds to the engine's vibrance=100 parameter.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        mask: (H, W) float mask 0–1. None applies the effect to the full image.
        strength: -1.0–1.0 intensity (0 is a no-op).

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    if strength == 0:
        return img_bgr

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    factor = 1.0 + strength * (1.0 - s / 255.0)
    if mask is None:
        skin_hue = ((h > 0) & (h < 25)) | (h > 160)
        skin_factor = np.clip(1.0 - strength * 0.5, 0.5, 1.0)
        factor = np.where(skin_hue, np.minimum(factor, skin_factor), factor)
    hsv[:, :, 1] = np.clip(s * factor, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return blend_masked(img_bgr, result, mask)


def apply_curve(
    channel: np.ndarray,
    curve_points: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Apply a piecewise-linear tone curve via LUT.

    Args:
        channel: Single-channel uint8 array.
        curve_points: Sequence of (input, output) sample points in 0–255.

    Returns:
        Tone-mapped channel (same shape), uint8.
    """
    xs = [p[0] for p in curve_points]
    ys = [p[1] for p in curve_points]
    lut = np.clip(np.interp(np.arange(256), xs, ys), 0, 255).astype(np.uint8)
    return cv2.LUT(channel, lut)


def correct_exposure(
    img_bgr: np.ndarray,
    face_bboxes: Optional[Sequence[Tuple[int, int, int, int]]] = None,
    target_mean: float = 120,
    min_threshold: float = 95,
    max_threshold: float = 165,
    face_target_mean: float = 145,
    face_min_threshold: float = 105,
    face_max_threshold: float = 185,
) -> Tuple[np.ndarray, bool]:
    """Normalize the image exposure using LAB space and a bounded Gamma curve.
    Prioritizes face luminance if face_bboxes are provided.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        face_bboxes: List of Tuple (x, y, w, h) bounding boxes.
        target_mean: Target global mean luminance (0-255).
        min_threshold: Lower bound for global mean luminance.
        max_threshold: Upper bound for global mean luminance.
        face_target_mean: Target face mean luminance (0-255).
        face_min_threshold: Lower bound for face mean luminance.
        face_max_threshold: Upper bound for face mean luminance.

    Returns:
        ((H, W, 3) uint8 BGR image, was_corrected bool).
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]

    use_face = False
    current_mean = 0.0

    if face_bboxes is not None and len(face_bboxes) > 0:
        # Sort face bboxes by area to find the largest (primary) face
        try:
            sorted_boxes = sorted(face_bboxes, key=lambda b: b[2] * b[3], reverse=True)
            primary_box = sorted_boxes[0]
            xf, yf, wf, hf = primary_box
            h, w = img_bgr.shape[:2]
            x1, y1, x2, y2 = max(0, int(xf)), max(0, int(yf)), min(w, int(xf+wf)), min(h, int(yf+hf))
            face_crop = l_chan[y1:y2, x1:x2]
            if face_crop.size > 0:
                current_mean = np.mean(face_crop)
                use_face = True
        except Exception:
            pass

    if not use_face:
        current_mean = np.mean(l_chan)
        t_min = min_threshold
        t_max = max_threshold
        t_mean = target_mean
    else:
        t_min = face_min_threshold
        t_max = face_max_threshold
        t_mean = face_target_mean

    # Only adjust if exposure falls outside the safe range [t_min, t_max]
    if current_mean < t_min or current_mean > t_max:
        if current_mean < 1.0:
            current_mean = 1.0
        elif current_mean > 254.0:
            current_mean = 254.0

        # Calculate gamma using log ratio
        # L_target / 255.0 = (L_current / 255.0)^gamma
        # gamma = ln(L_target / 255.0) / ln(L_current / 255.0)
        log_target = np.log(t_mean / 255.0)
        log_current = np.log(current_mean / 255.0)
        gamma = log_target / log_current

        # Safety bound the gamma shift to prevent extreme artifacts
        gamma = np.clip(gamma, 0.75, 1.25)

        # Apply gamma curve
        l_norm = l_chan / 255.0
        l_new = np.power(l_norm, gamma) * 255.0
        lab[:, :, 0] = np.clip(l_new, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR), True

    return img_bgr, False


def apply_global_bloom(
    img_bgr: np.ndarray,
    strength: float,
    threshold: float = 210.0,
    softness: float = 30.0,
) -> np.ndarray:
    """Apply a multi-scale atmospheric glow/bloom effect to the entire image.

    Resembles Composite Nation's Oniric Photoshop plugin.
    Isolates highlight regions above the threshold, blurs at 3 scales, and screen-blends.
    """
    if strength <= 0:
        return img_bgr

    # Convert to LAB to isolate highlights based on L (luminance) channel
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]

    # Soft threshold ramp from threshold to threshold + softness
    soft_w = max(1.0, softness)
    highlight_mask = np.clip((l_chan - threshold) / soft_w, 0.0, 1.0)
    
    if highlight_mask.max() < 0.01:
        return img_bgr

    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)
    
    # Isolate highlights in float32 (color-preserving, avoid early quantization to uint8)
    highlights = img_bgr.astype(np.float32) * highlight_mask[:, :, np.newaxis]

    # Downsampled bloom optimization for large images
    target_min = 2000
    is_downsampled = min_dim > target_min
    if is_downsampled:
        scale = target_min / min_dim
        highlights_low = cv2.resize(
            highlights, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        blur_dim = min(highlights_low.shape[:2])
    else:
        highlights_low = highlights
        blur_dim = min_dim

    # Cascaded Gaussian blurs representing different scales of atmospheric glow (1%, 3%, 8%)
    k1 = max(15, int(blur_dim * 0.01)) | 1
    k2 = max(31, int(blur_dim * 0.03)) | 1
    k3 = max(63, int(blur_dim * 0.08)) | 1

    blur1 = cv2.GaussianBlur(highlights_low, (k1, k1), 0)
    blur2 = cv2.GaussianBlur(highlights_low, (k2, k2), 0)
    blur3 = cv2.GaussianBlur(highlights_low, (k3, k3), 0)

    # Blend multi-scale glows to form realistic falloff
    glow_low = blur1 * 0.5 + blur2 * 0.3 + blur3 * 0.2
    
    if is_downsampled:
        glow = cv2.resize(glow_low, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        glow = glow_low
    
    # Screen blend to avoid clipping:
    # screen = 255 - ((255 - img) * (255 - glow) / 255)
    img_f = img_bgr.astype(np.float32)
    screen = screen_blend(img_f, glow)

    # Linearly interpolate between original BGR and Screened BGR based on strength factor
    s_factor = strength / 100.0
    result = img_f * (1.0 - s_factor) + screen * s_factor
    return np.clip(result, 0, 255).astype(np.uint8)


def log_crash(exc: Exception, context_info: Optional[dict] = None) -> str:
    """Log details of a crash (traceback, timestamp, context params) to ``~/.cache/retouch/crash.log``.

    Args:
        exc: The exception instance to log.
        context_info: Optional dict of contextual metadata to include in the log.

    Returns:
        The path of the crash log file, or an empty string if writing failed.
    """
    import os
    import sys
    import traceback
    from datetime import datetime
    from pathlib import Path

    try:
        cache_dir = Path.home() / ".cache" / "retouch"
        cache_dir.mkdir(parents=True, exist_ok=True)
        crash_log_file = cache_dir / "crash.log"

        timestamp = datetime.now().isoformat()
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

        entry = []
        entry.append(f"=== CRASH RECORDED AT {timestamp} ===")
        if context_info:
            entry.append("Context Metadata:")
            for k, v in context_info.items():
                entry.append(f"  {k}: {v}")
        entry.append("Traceback:")
        entry.append(tb_str)
        entry.append("=====================================\n\n")

        with open(crash_log_file, "a", encoding="utf-8") as f:
            f.write("\n".join(entry))

        return str(crash_log_file)
    except Exception as log_err:
        sys.stderr.write(f"Failed to write crash log: {log_err}\n")
        return ""

