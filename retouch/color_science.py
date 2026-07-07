"""Oklab color space converters and skin-tone measurement utilities.

Provides vectorized Oklab/OKLCh conversions using the standard Björn Ottosson
matrices, plus skin-tone classification and chroma variance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# Oklab conversion matrices (Björn Ottosson standard from oklab.com)
# Linear RGB to LMS (M1)
_OKLAB_M1 = np.array([
    [0.4121656, 0.5362752, 0.0514575],
    [0.2118591, 0.6807189, 0.1074220],
    [0.1933712, 0.0883814, 0.8736469],
], dtype=np.float32)

# Cube-root LMS to Oklab (M2)
_OKLAB_M2 = np.array([
    [0.2104542, 0.7936177, -0.0040720],
    [1.9779985, -2.4285922, 0.4505937],
    [0.0259040, 0.7827717, -0.8086757],
], dtype=np.float32)

# Precompute inverses
_OKLAB_M1_INV = np.linalg.inv(_OKLAB_M1)
_OKLAB_M2_INV = np.linalg.inv(_OKLAB_M2)


def bgr_to_oklab(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR to float32 Oklab color space.

    Uses the standard Oklab matrices from Björn Ottosson's work.
    Applies sRGB EOTF, linear-to-LMS, then cube-root transform.

    Dtype-aware: accepts uint8 BGR or float32 BGR in [0, 255]. The float32
    path avoids the uint8 quantization step (E1 float path); the uint8 path
    is byte-identical to the previous implementation.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image, or float32 BGR in [0, 255].

    Returns:
        (H, W, 3) float32 Oklab image with L ∈ [0, 1], a/b ∈ [-0.4, 0.4].
    """
    is_float = img_bgr.dtype == np.float32
    if is_float:
        img_rgb = np.clip(img_bgr[..., ::-1], 0.0, 255.0).astype(np.float32) * (1.0 / 255.0)
    else:
        # BGR -> RGB and normalize to [0, 1]
        img_rgb = img_bgr[..., ::-1].astype(np.float32) / 255.0

    # sRGB EOTF (inverse companding)
    img_linear = np.where(
        img_rgb <= 0.04045,
        img_rgb / 12.92,
        np.power((img_rgb + 0.055) / 1.055, 2.4),
    )

    # Linear RGB -> LMS (via M1)
    lms = np.dot(img_linear, _OKLAB_M1.T)

    # Cube root
    lms_cbrt = np.cbrt(np.maximum(lms, 0.0))

    # cbrt(LMS) -> Oklab (via M2)
    oklab = np.dot(lms_cbrt, _OKLAB_M2.T)

    return oklab


def oklab_to_bgr(oklab: np.ndarray, float32_out: bool = False) -> np.ndarray:
    """Convert float32 Oklab to BGR color space.

    Inverts bgr_to_oklab via M2_inv, cube, M1_inv, and sRGB companding.

    Args:
        oklab: (H, W, 3) float32 Oklab image.
        float32_out: If True, return float32 [0, 255] (E1 float path — no
            uint8 quantization). Default False returns uint8.

    Returns:
        (H, W, 3) uint8 BGR image clipped to [0, 255], or float32 if
        ``float32_out=True``.
    """
    # Oklab -> cbrt(LMS) (via M2_inv)
    lms_cbrt = np.dot(oklab, _OKLAB_M2_INV.T)

    # Cube (inverse of cbrt)
    lms = np.power(lms_cbrt, 3.0)

    # LMS -> linear RGB (via M1_inv)
    img_linear = np.dot(lms, _OKLAB_M1_INV.T)
    img_linear = np.clip(img_linear, 0.0, 1.0)

    # sRGB companding
    img_rgb = np.where(
        img_linear <= 0.0031308,
        12.92 * img_linear,
        1.055 * np.power(img_linear, 1.0 / 2.4) - 0.055,
    )

    # RGB -> BGR and scale to [0, 255]
    img_bgr = img_rgb[..., ::-1]
    out = np.clip(img_bgr * 255.0, 0, 255)
    if float32_out:
        return np.ascontiguousarray(out, dtype=np.float32)
    return out.astype(np.uint8)


def oklab_to_oklch(oklab: np.ndarray) -> np.ndarray:
    """Convert Oklab (L, a, b) to OKLCh (L, C, h).

    Args:
        oklab: (H, W, 3) float32 Oklab with L, a, b channels.

    Returns:
        (H, W, 3) float32 OKLCh with L, C, h (hue in degrees [0, 360)).
    """
    L = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]

    C = np.sqrt(a**2 + b**2)
    h = np.degrees(np.arctan2(b, a)) % 360.0

    oklch = np.stack([L, C, h], axis=-1)
    return oklch.astype(np.float32)


def oklch_to_oklab(oklch: np.ndarray) -> np.ndarray:
    """Convert OKLCh (L, C, h) to Oklab (L, a, b).

    Args:
        oklch: (H, W, 3) float32 OKLCh with L, C, h (hue in degrees).

    Returns:
        (H, W, 3) float32 Oklab.
    """
    L = oklch[..., 0]
    C = oklch[..., 1]
    h = np.radians(oklch[..., 2])

    a = C * np.cos(h)
    b = C * np.sin(h)

    oklab = np.stack([L, a, b], axis=-1)
    return oklab.astype(np.float32)


@dataclass
class SkinState:
    """Measured skin tone state in OKLCh color space.

    Attributes:
        L_mean: Mean lightness [0, 1].
        C_mean: Mean chroma [0, ~0.4].
        C_std: Standard deviation of chroma.
        h_mean: Circular mean hue [0, 360) degrees.
    """

    L_mean: float
    C_mean: float
    C_std: float
    h_mean: float


def measure_skin_state(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray],
    thresh: float = 0.3,
) -> SkinState:
    """Measure skin tone state (L, C, h, σ_C) from a masked region.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        skin_mask: (H, W) float mask [0, 1]. If None, uses whole image.
        thresh: Mask threshold (pixels with mask > thresh are included).

    Returns:
        SkinState with L_mean, C_mean, C_std, h_mean.
    """
    oklab = bgr_to_oklab(img_bgr)
    oklch = oklab_to_oklch(oklab)

    if skin_mask is None:
        skin_idx = np.ones(img_bgr.shape[:2], dtype=bool)
    else:
        skin_idx = skin_mask > thresh

    if not np.any(skin_idx):
        # No skin; return neutral state
        return SkinState(L_mean=0.5, C_mean=0.085, C_std=0.0, h_mean=50.0)

    L_skin = oklch[..., 0][skin_idx]
    C_skin = oklch[..., 1][skin_idx]
    h_skin = oklch[..., 2][skin_idx]

    L_mean = float(np.mean(L_skin))
    C_mean = float(np.mean(C_skin))
    C_std = float(np.std(C_skin))

    # Circular mean for hue: convert to unit vectors, average, convert back
    h_rad = np.radians(h_skin)
    sin_mean = np.mean(np.sin(h_rad))
    cos_mean = np.mean(np.cos(h_rad))
    h_mean = float(np.degrees(np.arctan2(sin_mean, cos_mean)) % 360.0)

    return SkinState(L_mean=L_mean, C_mean=C_mean, C_std=C_std, h_mean=h_mean)


# Preferred skin-color loci per tone class (from literature on preferred reproduction)
SKIN_LOCI: Dict[str, Dict[str, float]] = {
    "fair": {
        "L_min": 0.72,
        "h_target": 45.0,
        "C_target": 0.085,
    },
    "tan": {
        "L_min": 0.55,
        "L_max": 0.72,
        "h_target": 52.0,
        "C_target": 0.105,
    },
    "deep": {
        "L_max": 0.55,
        "h_target": 58.0,
        "C_target": 0.125,
    },
}


def skin_chroma_std(img_bgr: np.ndarray, skin_mask: Optional[np.ndarray]) -> float:
    """Compute chroma standard deviation (σ_C) metric for skin uniformity.

    Lower σ_C = more uniform chroma = "milk skin" / "water glow" appearance.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        skin_mask: (H, W) float mask [0, 1].

    Returns:
        Chroma standard deviation, typically 0.01–0.15 for normal skin.
    """
    state = measure_skin_state(img_bgr, skin_mask)
    return state.C_std


def resolve_locus_override(
    locus: Dict[str, float],
    skin_state: Optional["SkinState"] = None,
) -> Dict[str, float]:
    """Resolve a partial locus override dict to a full target specification.

    Args:
        locus: Dict with optional keys 'h_target', 'C_target', 'L_min'.
               Missing keys are filled from SKIN_LOCI defaults.
        skin_state: Optional SkinState for auto-selecting the base locus class.

    Returns:
        Full locus dict with 'h_target', 'C_target', 'L_min' keys.
    """
    out: Dict[str, float] = {}
    out["h_target"] = float(locus.get("h_target", SKIN_LOCI["fair"]["h_target"]))
    out["C_target"] = float(locus.get("C_target", SKIN_LOCI["fair"]["C_target"]))
    out["L_min"] = float(locus.get("L_min", SKIN_LOCI["fair"]["L_min"]))
    return out
