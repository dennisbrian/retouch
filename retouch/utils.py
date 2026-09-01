"""Shared utilities for the face retouching pipeline.

Provides mask normalization, blending helpers, landmark extraction utilities,
and colour/tone primitives used across the retouch modules.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

import cv2
import logging
import numpy as np

logger = logging.getLogger(__name__)


def get_cache_dir() -> Path:
    """Return the writable Retouch cache directory.

    ``RETOUCH_CACHE_DIR`` is an application-specific override. Otherwise honor
    the standard ``XDG_CACHE_HOME`` base before falling back to
    ``~/.cache/retouch``. The directory is not created by this resolver.
    """
    explicit = os.environ.get("RETOUCH_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg_root = os.environ.get("XDG_CACHE_HOME")
    if xdg_root:
        return Path(xdg_root).expanduser() / "retouch"
    return Path.home() / ".cache" / "retouch"


def offline_mode_enabled() -> bool:
    """Return whether user-requested offline/privacy mode is enabled."""
    return os.environ.get("RETOUCH_OFFLINE", "").strip().lower() in {
        "1", "true", "yes", "on",
    }


_PATH_CONTEXT_KEYS = {
    "file_path", "image_path", "source_path", "input_path", "output_path",
    "file", "image", "source", "output", "input", "filename", "filepath",
}
_UNIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?!/)(?:[^\s:'\"]+/)*[^\s:'\"]+")
_WINDOWS_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\s:'\"]+")


def redact_diagnostics_text(value: Any) -> str:
    """Redact common local path forms before diagnostics leave the machine."""
    text = str(value)
    replacements = [str(Path.cwd()), str(Path.home())]
    for path in replacements:
        if path and path != "/":
            text = text.replace(path, "<redacted-path>")
    text = _WINDOWS_PATH_RE.sub("<redacted-path>", text)
    text = _UNIX_PATH_RE.sub("<redacted-path>", text)
    return text


def redact_diagnostics_context(context_info: Optional[dict]) -> dict:
    """Return crash context with path-bearing fields removed or redacted."""
    if not context_info:
        return {}
    redacted = {}
    for key, value in context_info.items():
        key_text = str(key)
        if key_text.lower() in _PATH_CONTEXT_KEYS or key_text.lower().endswith("_path"):
            redacted[key_text] = "<redacted-path>"
        else:
            redacted[key_text] = redact_diagnostics_text(value)
    return redacted


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


def restore_outside_support(
    original: np.ndarray,
    processed: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    """Restore source pixels exactly where an operation has zero support.

    This is a mechanical containment helper only. It deliberately does not
    normalize, blur, threshold, or otherwise reinterpret ``support``; callers
    remain responsible for defining their final effective operation support.

    Args:
        original: Source image before the local operation.
        processed: Result after the local operation.
        support: Two-dimensional final support. Exact zero means the operation
            has no authority to change that pixel.

    Returns:
        A copy of ``processed`` with source pixels restored outside support.

    Raises:
        TypeError: If image inputs are not NumPy arrays.
        ValueError: If shapes/dtypes do not match, support is invalid, or the
            operation introduced a non-finite value inside support.
    """
    if not isinstance(original, np.ndarray) or not isinstance(processed, np.ndarray):
        raise TypeError("original and processed must be NumPy arrays")
    if original.shape != processed.shape:
        raise ValueError(
            f"processed shape {processed.shape} does not match original {original.shape}"
        )
    if original.dtype != processed.dtype:
        raise ValueError(
            f"processed dtype {processed.dtype} does not match original {original.dtype}"
        )
    if original.ndim < 2:
        raise ValueError(f"images must have at least two dimensions, got {original.shape}")

    support_array = np.asarray(support)
    if support_array.ndim != 2 or support_array.shape != original.shape[:2]:
        raise ValueError(
            f"support shape {support_array.shape} does not match image spatial shape "
            f"{original.shape[:2]}"
        )
    if not (
        np.issubdtype(support_array.dtype, np.number)
        or np.issubdtype(support_array.dtype, np.bool_)
    ):
        raise ValueError(f"support must be numeric or boolean, got {support_array.dtype}")
    if not np.isfinite(support_array).all():
        raise ValueError("support contains non-finite values")

    result = processed.copy()
    outside = support_array == 0
    result[outside] = original[outside]

    newly_nonfinite = ~np.isfinite(result) & np.isfinite(original)
    if newly_nonfinite.any():
        raise ValueError("processed image introduced non-finite values inside support")
    return result


# --- Yaw gate (shared by geometry slimming/jaw/chin, relight, sculpt) ---------
# yaw_ratio = max/min of the nose-bridge (lm 6) -> temple (lm 234 / 454)
# x-distances. Calibrated 2026-09-02 on the 83-image DSCF corpus
# (docs/plans/RESEARCH_YAW_GATE_CALIBRATION_2026_09_02.md): the ratio tracks
# MediaPipe-depth yaw with Spearman 0.92, but the previous bands (geometry
# 1.3->1.6, relight/sculpt 1.5->1.7) mapped to only ~2-10 degrees of head
# turn and zeroed slimming on 58/83 ordinary portraits. Slimming at strength
# 70 with the gate bypassed rendered artifact-free through ratio ~4.0
# (~25 degrees), so the ramp now runs 2.5 -> 4.0.
YAW_GATE_START = 2.5
YAW_GATE_END = 4.0


def yaw_ratio(landmarks) -> float:
    """Nose-bridge/temple x-asymmetry ratio (>= 1.0; 1.0 = frontal)."""
    lm = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    d_left = abs(lm[6].x - lm[234].x)
    d_right = abs(lm[454].x - lm[6].x)
    return max(d_left, d_right) / (min(d_left, d_right) + 1e-5)


def yaw_gate_factor(ratio: float, start: float = YAW_GATE_START, end: float = YAW_GATE_END) -> float:
    """Smoothstep 1.0 -> 0.0 strength factor as ``ratio`` runs start -> end."""
    if ratio <= start:
        return 1.0
    if ratio >= end:
        return 0.0
    t = (ratio - start) / (end - start)
    return float(t * t * (3.0 - 2.0 * t))


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


def soft_light_blend(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Soft-light blend — gentle contrast-preserving overlay.

    Uses the Pegtop approximation in normalised [0, 1] space:
    ``(1 - 2*layer) * base² + 2*layer * base``.

    Preserves original detail while applying tonal shifts — ideal for
    non-destructive grading layers.

    Args:
        base: (H, W, 3) uint8 BGR base image.
        layer: (H, W, 3) uint8 BGR overlay layer.

    Returns:
        (H, W, 3) uint8 BGR blended result.
    """
    base_f = base.astype(np.float32) / 255.0
    layer_f = layer.astype(np.float32) / 255.0
    result = (1.0 - 2.0 * layer_f) * base_f * base_f + 2.0 * layer_f * base_f
    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def overlay_blend(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Overlay blend — multiply on dark pixels, screen on light pixels.

    For each channel, pixels below 128 use multiply, above 128 use screen.
    This is the classic "contrast boost" blend mode.

    Args:
        base: (H, W, 3) uint8 BGR base image.
        layer: (H, W, 3) uint8 BGR overlay layer.

    Returns:
        (H, W, 3) uint8 BGR blended result.
    """
    base_f = base.astype(np.float32)
    layer_f = layer.astype(np.float32)
    multiply = 2.0 * base_f * layer_f / 255.0
    screen = 255.0 - 2.0 * (255.0 - base_f) * (255.0 - layer_f) / 255.0
    return np.where(base_f < 128.0, multiply, screen).astype(np.uint8)


def hard_light_blend(base: np.ndarray, layer: np.ndarray) -> np.ndarray:
    """Hard-light blend — overlay but driven by the layer, not the base.

    For each channel, pixels in *layer* below 128 use multiply, above 128
    use screen.  This is the "hard contrast" version — stronger than overlay,
    preserves no original detail in highlights/shadows.

    Args:
        base: (H, W, 3) uint8 BGR base image.
        layer: (H, W, 3) uint8 BGR overlay layer.

    Returns:
        (H, W, 3) uint8 BGR blended result.
    """
    base_f = base.astype(np.float32)
    layer_f = layer.astype(np.float32)
    multiply = 2.0 * base_f * layer_f / 255.0
    screen = 255.0 - 2.0 * (255.0 - base_f) * (255.0 - layer_f) / 255.0
    return np.where(layer_f < 128.0, multiply, screen).astype(np.uint8)


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
    from .acceleration import accelerated_gaussian_blur
    return accelerated_gaussian_blur(mask_f, ksize, sigma)



def apply_u8_op_float(img_f32: np.ndarray, op, *args, **kwargs) -> np.ndarray:
    """Run a uint8-contract op on a float32 [0, 255] canvas via delta application.

    E1 adapter for ops built on inherently-uint8 primitives (e.g. cv2.inpaint):
    the op runs on a uint8 snapshot of the float canvas and its *delta*
    (op output minus snapshot) is added back to the float canvas. Pixels the op
    does not change keep full float precision; quantization is confined to the
    pixels the op actually modifies.

    Args:
        img_f32: (H, W, 3) float32 BGR canvas in [0, 255].
        op: Callable taking a uint8 image as first argument and returning uint8.
        *args, **kwargs: Forwarded to ``op`` after the image.

    Returns:
        (H, W, 3) float32 BGR canvas in [0, 255].
    """
    # Truncating snapshot (not rounding) matches the legacy uint8 chain's
    # per-op `np.clip(x, 0, 255).astype(np.uint8)` convention, so threshold
    # detectors inside `op` see values consistent with the pre-E1 pipeline.
    u8 = np.clip(img_f32, 0, 255).astype(np.uint8)
    out = op(u8, *args, **kwargs)
    delta = out.astype(np.float32) - u8.astype(np.float32)
    return np.clip(img_f32 + delta, 0.0, 255.0)


def bgr_f32_to_lab_f32(img_bgr_f32: np.ndarray) -> np.ndarray:
    """Convert float32 BGR [0, 255] to float32 LAB in the uint8-scale convention.

    cv2's float LAB path returns L in [0, 100] and a/b in [-127, 127]; this helper
    rescales to the uint8 convention (L in [0, 255], a/b offset by +128) so that
    code written against ``cv2.cvtColor(uint8, COLOR_BGR2LAB)`` values works
    unchanged — but without the uint8 quantization step (E1 float path).

    Args:
        img_bgr_f32: (H, W, 3) float32 BGR image in [0, 255].

    Returns:
        (H, W, 3) float32 LAB image, L in [0, 255], a/b centered at 128.
    """
    lab = cv2.cvtColor(np.clip(img_bgr_f32, 0.0, 255.0) * (1.0 / 255.0), cv2.COLOR_BGR2LAB)
    lab[:, :, 0] *= 255.0 / 100.0
    lab[:, :, 1] += 128.0
    lab[:, :, 2] += 128.0
    return lab


def lab_f32_to_bgr_f32(lab_f32: np.ndarray) -> np.ndarray:
    """Inverse of :func:`bgr_f32_to_lab_f32` — float32 LAB (uint8-scale) to float32 BGR [0, 255].

    Args:
        lab_f32: (H, W, 3) float32 LAB image, L in [0, 255], a/b centered at 128.

    Returns:
        (H, W, 3) float32 BGR image clipped to [0, 255].
    """
    lab = lab_f32.astype(np.float32, copy=True)
    lab[:, :, 0] = np.clip(lab[:, :, 0], 0.0, 255.0) * (100.0 / 255.0)
    lab[:, :, 1] = np.clip(lab[:, :, 1], 0.0, 255.0) - 128.0
    lab[:, :, 2] = np.clip(lab[:, :, 2], 0.0, 255.0) - 128.0
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(bgr * 255.0, 0.0, 255.0)


def blend_masked(
    original: np.ndarray,
    processed: np.ndarray,
    mask: Optional[np.ndarray],
) -> np.ndarray:
    """Alpha-blend *processed* onto *original* using a soft mask.

    Supports both uint8 [0, 255] and float32 [0, 255] images. If input is float32,
    output is float32; if input is uint8, output is uint8.

    Args:
        original: (H, W, C) uint8 or float32 image.
        processed: (H, W, C) uint8 or float32 image. Must match original dtype.
        mask: (H, W) float mask, 0.0–1.0. If None, ``processed`` is returned.

    Returns:
        Blended (H, W, C) image, same dtype as input (uint8 or float32).
    """
    if mask is None:
        return processed

    m = mask.astype(np.float32)
    if m.ndim == 2:
        m = m[:, :, np.newaxis]

    # Check if input is float32; if so, keep output float32
    is_float = original.dtype == np.float32

    out = original.astype(np.float32) * (1.0 - m) + processed.astype(np.float32) * m

    if is_float:
        return np.clip(out, 0, 255).astype(np.float32)
    else:
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


def guided_filter(src: np.ndarray, radius: int, eps: float, guide: Optional[np.ndarray] = None, max_dim: Optional[int] = 1200) -> np.ndarray:
    """Edge-preserving guided filter for smoothing while maintaining structure.

    Implements the guided filter algorithm where the output is a linear combination
    of the guide image. When guide is None, performs self-guided filtering (equivalent
    to bilateral filtering with edge preservation).

    Args:
        src: Input image, float32 single-channel array (H, W).
        radius: Kernel radius in pixels (kernel size = 2*radius+1).
        eps: Regularization parameter controlling edge preservation. Higher values
             preserve more edges (less smoothing). Typically in range [0.1, 100.0].
        guide: Guide image for filtering. If None, uses src as guide (self-guided).
               Must be float32 single-channel (H, W), same shape as src.
        max_dim: For large images (min(h,w) > max_dim), compute coefficients at
                 downsampled scale, then upsample them. Pass None to disable.
                 Improves performance on high-res images without quality loss.

    Returns:
        Filtered output image, float32 single-channel (H, W), same shape as src.

    Notes:
        For self-guided case (guide=None), the algorithm reduces to:
            mean_I = blur(src)
            var_I = blur(src*src) - mean_I*mean_I
            a = var_I / (var_I + eps)
            b = mean_I * (1 - a)
            output = blur(a) * src + blur(b)

        Downsampling is applied only to the coefficient computation (a, b), while
        the final blend is always done at full resolution.
    """
    if src.ndim != 2 or src.dtype != np.float32:
        raise ValueError(f"src must be float32 single-channel, got shape {src.shape}, dtype {src.dtype}")

    if guide is None:
        guide = src
    else:
        if guide.shape != src.shape or guide.dtype != np.float32:
            raise ValueError(f"guide shape/dtype mismatch: got {guide.shape}/{guide.dtype}, expected {src.shape}/float32")

    h, w = src.shape[:2]
    r = radius

    # Downsample guard: if image is too large, compute coefficients at smaller scale
    min_dim = min(h, w)
    is_downsampled = max_dim is not None and min_dim > max_dim

    if is_downsampled:
        scale = float(max_dim) / min_dim
        h_small = max(1, int(h * scale))
        w_small = max(1, int(w * scale))
        guide_small = cv2.resize(guide, (w_small, h_small), interpolation=cv2.INTER_AREA)
        src_small = cv2.resize(src, (w_small, h_small), interpolation=cv2.INTER_AREA)
        r_small = max(1, int(r * scale))
    else:
        guide_small = guide
        src_small = src
        r_small = r

    # Compute statistics on guide
    mean_guide = cv2.blur(guide_small, (r_small, r_small))
    mean_src = cv2.blur(src_small, (r_small, r_small))
    mean_guide_src = cv2.blur(guide_small * src_small, (r_small, r_small))
    mean_guide_guide = cv2.blur(guide_small * guide_small, (r_small, r_small))

    # Covariance and variance
    cov_guide_src = mean_guide_src - mean_guide * mean_src
    var_guide = mean_guide_guide - mean_guide * mean_guide

    # Coefficients a, b
    a = cov_guide_src / np.clip(var_guide + eps, eps, None)
    b = mean_src - a * mean_guide

    # Upsample coefficients if downsampled
    if is_downsampled:
        a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)

    # Final filtering: blur the coefficients, then blend
    mean_a = cv2.blur(a, (r, r))
    mean_b = cv2.blur(b, (r, r))
    output = mean_a * guide + mean_b

    return output


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

    Accepts uint8 [0, 255] or float32 [0, 255] BGR input; output dtype
    matches input. float32 input is processed entirely in float32 (no
    uint8 round-trip).

    Args:
        img_bgr: (H, W, 3) uint8 or float32 BGR image, range [0, 255].
        mask: (H, W) float mask 0–1. None applies the effect to the full image.
        strength: -1.0–1.0 intensity (0 is a no-op).

    Returns:
        (H, W, 3) BGR image, same dtype as *img_bgr*.
    """
    if strength == 0:
        return img_bgr

    is_float = img_bgr.dtype == np.float32

    if is_float:
        # OpenCV float32 HSV convention: H∈[0,360), S∈[0,1], V∈[0,255].
        # uint8 convention:                H∈[0,180), S∈[0,255], V∈[0,255].
        # Match the uint8 algorithm exactly in float, with S normalised to [0,1].
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        h, s = hsv[:, :, 0], hsv[:, :, 1]
        factor = 1.0 + strength * (1.0 - s)
        if mask is None:
            skin_hue = ((h > 0) & (h < 50)) | (h > 320)
            skin_factor = np.clip(1.0 - strength * 0.5, 0.5, 1.0)
            factor = np.where(skin_hue, np.minimum(factor, skin_factor), factor)
        hsv[:, :, 1] = np.clip(s * factor, 0.0, 1.0)
        result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)  # stays float32 [0,255]
    else:
        # uint8 path: H in [0,180) — preserved byte-identical to legacy behaviour.
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
            logger.warning("Primary-face detection failed; falling back to global mean", exc_info=True)

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
    Isolates highlight regions above the threshold, blurs at 3 scales, and screen-blends
    in LINEAR RGB space to produce denser, more concentrated bloom with hot cores and
    long soft tails (unlike gamma-space bloom which under-weights bright pixels).
    """
    if strength <= 0:
        return img_bgr

    is_float = img_bgr.dtype == np.float32

    # Convert to LAB to isolate highlights based on L (luminance) channel
    # Threshold and softness logic operates in gamma-space L domain (photographer-intuitive).
    #
    # F1/E2: previously this always quantized to uint8 here (even when given
    # float32 input) before computing the LAB highlight mask and the linear
    # base image used for blending -- a "fake float" round-trip: the
    # function accepted and returned float32, but did all of its actual math
    # (highlight isolation, linearization, cascaded blur) on 8-bit data. Now
    # the float path stays in float32 [0,255] throughout via
    # ``bgr_f32_to_lab_f32`` (uint8-scale-convention LAB, no quantization),
    # matching the pattern already used by ``apply_skin_diffusion`` and the
    # other float-native grading ops. See docs/plans/PLAN_TIERE_ENGINE_FIDELITY.md
    # Sec 18 for the before/after banding measurement.
    if is_float:
        img_f255 = np.clip(img_bgr * 255.0, 0.0, 255.0).astype(np.float32)
        lab = bgr_f32_to_lab_f32(img_f255)
    else:
        img_f255 = img_bgr.astype(np.float32)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]

    # Soft threshold ramp from threshold to threshold + softness
    soft_w = max(1.0, softness)
    highlight_mask = np.clip((l_chan - threshold) / soft_w, 0.0, 1.0)

    if highlight_mask.max() < 0.01:
        return img_bgr

    h, w = img_f255.shape[:2]
    min_dim = min(h, w)

    # Isolate highlights in float32 (color-preserving, avoid early quantization to uint8)
    highlights_gamma = img_f255 * highlight_mask[:, :, np.newaxis]

    # Linearize highlights and base image for bloom computation (gamma 2.2)
    highlights_lin = (highlights_gamma / 255.0) ** 2.2
    img_lin = (img_f255 / 255.0) ** 2.2

    # Downsampled bloom optimization for large images (operates in linear space)
    target_min = 2000
    is_downsampled = min_dim > target_min
    if is_downsampled:
        scale = target_min / min_dim
        highlights_lin_low = cv2.resize(
            highlights_lin, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        blur_dim = min(highlights_lin_low.shape[:2])
    else:
        highlights_lin_low = highlights_lin
        blur_dim = min_dim

    # Cascaded Gaussian blurs in linear space representing different scales of atmospheric glow (1%, 3%, 8%)
    k1 = max(15, int(blur_dim * 0.01)) | 1
    k2 = max(31, int(blur_dim * 0.03)) | 1
    k3 = max(63, int(blur_dim * 0.08)) | 1

    blur1 = cv2.GaussianBlur(highlights_lin_low, (k1, k1), 0)
    blur2 = cv2.GaussianBlur(highlights_lin_low, (k2, k2), 0)
    blur3 = cv2.GaussianBlur(highlights_lin_low, (k3, k3), 0)

    # Blend multi-scale glows in linear space to form realistic falloff
    glow_lin_low = blur1 * 0.5 + blur2 * 0.3 + blur3 * 0.2

    if is_downsampled:
        glow_lin = cv2.resize(glow_lin_low, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        glow_lin = glow_lin_low

    # Screen blend in linear space: 1 - (1 - a) * (1 - b)
    # Both img_lin and glow_lin are in [0, 1] range after linearization
    screen_lin = 1.0 - (1.0 - img_lin) * (1.0 - glow_lin)

    # Linearly interpolate between original (linear) and screen-blended (linear)
    s_factor = strength / 100.0
    result_lin = img_lin * (1.0 - s_factor) + screen_lin * s_factor

    # Clip to [0, 1] in linear space before delinearizing
    result_lin = np.clip(result_lin, 0.0, 1.0)

    # Delinearize back to gamma space (inverse gamma 2.2)
    result = result_lin ** (1.0 / 2.2)
    if is_float:
        return result.astype(np.float32)
    return (result * 255.0).astype(np.uint8)


def apply_skin_diffusion(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Skin-scoped atmospheric glow / light-wrap.

    Like ``apply_global_bloom`` but gated to luminous skin mid-tones rather
    than specular highlights, and weighted by a blurred skin mask so the
    wrap extends slightly past the silhouette.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        skin_mask: (H, W) float32 mask in [0, 1].
        strength: 0–100 glow intensity.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    if strength <= 0 or skin_mask is None or skin_mask.max() < 0.01:
        return img_bgr

    is_float = img_bgr.dtype == np.float32
    if is_float:
        lab = bgr_f32_to_lab_f32(img_bgr)
    else:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_chan = lab[:, :, 0]

    skin_gate = np.clip((l_chan - 110.0) / 90.0, 0.0, 1.0)
    highlight_mask = skin_gate

    if highlight_mask.max() < 0.01:
        return img_bgr

    h, w = img_bgr.shape[:2]
    min_dim = min(h, w)

    k = max(15, int(min_dim * 0.015)) | 1
    blurred_skin = cv2.GaussianBlur(skin_mask, (k, k), 0)
    wrap = np.clip(blurred_skin * 1.3, 0.0, 1.0)
    highlight_mask = highlight_mask * wrap

    if highlight_mask.max() < 0.01:
        return img_bgr

    img_f = img_bgr.astype(np.float32) if not is_float else img_bgr
    highlights = img_f * highlight_mask[:, :, np.newaxis]

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

    k1 = max(15, int(blur_dim * 0.01)) | 1
    k2 = max(31, int(blur_dim * 0.03)) | 1
    k3 = max(63, int(blur_dim * 0.08)) | 1

    blur1 = cv2.GaussianBlur(highlights_low, (k1, k1), 0)
    blur2 = cv2.GaussianBlur(highlights_low, (k2, k2), 0)
    blur3 = cv2.GaussianBlur(highlights_low, (k3, k3), 0)

    glow_low = blur1 * 0.5 + blur2 * 0.3 + blur3 * 0.2

    if is_downsampled:
        glow = cv2.resize(glow_low, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        glow = glow_low

    screen = screen_blend(img_f, glow)

    s_factor = strength / 100.0
    result = img_f * (1.0 - s_factor) + screen * s_factor
    if is_float:
        return np.clip(result, 0, 255).astype(np.float32)
    return np.clip(result, 0, 255).astype(np.uint8)


def log_crash(exc: Exception, context_info: Optional[dict] = None) -> str:
    """Log details of a crash to the configured Retouch cache directory.

    Args:
        exc: The exception instance to log.
        context_info: Optional dict of contextual metadata to include in the log.

    Returns:
        The path of the crash log file, or an empty string if writing failed.
    """
    import sys
    import traceback
    from datetime import datetime

    try:
        cache_dir = get_cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        crash_log_file = cache_dir / "crash.log"

        timestamp = datetime.now().isoformat()
        tb_str = redact_diagnostics_text(
            "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        )

        entry = []
        entry.append(f"=== CRASH RECORDED AT {timestamp} ===")
        if context_info:
            entry.append("Context Metadata:")
            for k, v in redact_diagnostics_context(context_info).items():
                entry.append(f"  {k}: {v}")
        entry.append("Traceback:")
        entry.append(tb_str)
        entry_text = "\n".join(entry) + "=====================================\n\n"

        try:
            max_bytes = max(64 * 1024, int(os.environ.get("RETOUCH_CRASH_MAX_BYTES", "5000000")))
        except (TypeError, ValueError):
            max_bytes = 5_000_000
        if crash_log_file.exists() and crash_log_file.stat().st_size + len(entry_text.encode("utf-8")) > max_bytes:
            rotated = crash_log_file.with_name("crash.log.1")
            try:
                os.replace(crash_log_file, rotated)
            except OSError:
                pass

        with open(crash_log_file, "a", encoding="utf-8") as f:
            f.write(entry_text)

        try:
            retention_days = max(1, int(os.environ.get("RETOUCH_CRASH_RETENTION_DAYS", "30")))
        except (TypeError, ValueError):
            retention_days = 30
        cutoff = datetime.now().timestamp() - (retention_days * 86400)
        for candidate in cache_dir.glob("crash.log*"):
            try:
                if candidate.stat().st_mtime < cutoff:
                    candidate.unlink()
            except OSError:
                pass

        return str(crash_log_file)
    except Exception as log_err:
        sys.stderr.write(f"Failed to write crash log: {log_err}\n")
        return ""


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fallback: float = 0.0,
) -> np.ndarray:
    """Element-wise division with zero-denominator protection.

    Args:
        numerator: (H, W) or (H, W, C) float32 array.
        denominator: Same shape as numerator.
        fallback: Value to use where |denominator| < 1e-8.

    Returns:
        Division result with safe fallback for near-zero denominators.
    """
    denom = np.where(np.abs(denominator) < 1e-8, 1.0, denominator)
    result = numerator / denom
    return np.where(np.abs(denominator) < 1e-8, fallback, result)


def remove_purple_fringing(
    img_bgr: np.ndarray,
    strength: float = 1.0,
    edge_threshold: float = 35.0,
) -> np.ndarray:
    """Detect and desaturate lateral chromatic aberration (purple fringing).

    Identifies purple/violet chroma spikes immediately adjacent to high-contrast
    luminance edges (e.g. backlit windows, dark wigs against bright backgrounds).
    Non-edge pixels (uniform purple fabrics, irises) are strictly protected by the
    luminance gradient gate.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 BGR image.
        strength: Desaturation intensity [0, 1].
        edge_threshold: Luminance gradient threshold for edge detection.

    Returns:
        (H, W, 3) BGR image with purple fringing suppressed.
    """
    if strength <= 0.0:
        return img_bgr

    is_float = img_bgr.dtype == np.float32
    img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8) if is_float else img_bgr

    # Convert to LAB for luminance gradient and chroma evaluation
    lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    # Luminance spatial gradient magnitude
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)

    # Edge gate: high-contrast luminance edge
    edge_gate = np.clip((grad - edge_threshold) / 20.0, 0.0, 1.0)

    # Purple/violet color gate in LAB space: positive a (magenta), negative b (blue)
    purple_gate = np.clip((a - 8.0) / 10.0, 0.0, 1.0) * np.clip((-b - 5.0) / 10.0, 0.0, 1.0)

    fringe_mask = edge_gate * purple_gate * min(strength, 1.0)

    if fringe_mask.max() < 0.01:
        return img_bgr

    # Desaturate a and b channels on fringe mask
    lab[:, :, 1] -= fringe_mask * a
    lab[:, :, 2] -= fringe_mask * b

    lab_u8 = np.clip(lab, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(lab_u8, cv2.COLOR_LAB2BGR)

    if is_float:
        return out.astype(np.float32)
    return out
