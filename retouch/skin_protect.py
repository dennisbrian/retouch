"""Skin-tone protection API for color grading.

Wraps arbitrary color operations with an LCH-based skin mask so that
skin pixels survive aggressive grading. Used by the Fuji-quality
color recipe system to honour the Astia-quality portrait requirement:
skin tones must survive grading.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from retouch.color_space import bgr_to_lch, lch_to_bgr, skin_mask_lch
from retouch.regions import apply_to_region


def _blend_skin(
    img_bgr: np.ndarray,
    op_result: np.ndarray,
    skin_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Blend ``img_bgr`` and ``op_result`` weighted by the skin mask.

    Formula: ``img * (skin_mask * strength) + op_result * (1 - skin_mask * strength)``.

    At ``skin_mask = 1, strength = 1`` the result equals ``img`` (skin fully
    preserved); at ``skin_mask = 0`` the result equals ``op_result`` (op
    fully applied).

    Args:
        img_bgr: H×W×3 uint8 BGR base image.
        op_result: H×W×3 uint8 BGR image produced by the wrapped op.
        skin_mask: H×W float32 mask in [0, 1].
        strength: Protection strength in [0, 1].

    Returns:
        H×W×3 uint8 BGR blended image.
    """
    m = np.clip(skin_mask.astype(np.float32), 0.0, 1.0) * float(strength)
    m3 = m[:, :, np.newaxis]
    base_f = img_bgr.astype(np.float32)
    op_f = op_result.astype(np.float32)
    out = base_f * m3 + op_f * (1.0 - m3)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def protect_skin(
    img_bgr: np.ndarray,
    op: Callable[[np.ndarray], np.ndarray],
    strength: float = 0.7,
) -> np.ndarray:
    """Apply ``op`` to ``img_bgr`` while protecting skin tones.

    Skin pixels (per the LCH skin mask) are preserved while non-skin
    pixels receive the full effect of ``op``. With ``strength = 0`` the
    result equals ``op(img)``; with ``strength = 1`` skin pixels are
    unchanged.

    Args:
        img_bgr: H×W×3 uint8 BGR image.
        op: Function from H×W×3 uint8 BGR to H×W×3 uint8 BGR.
        strength: Protection strength in [0, 1]. 0 = no protection,
            1 = skin pixels unchanged.

    Returns:
        H×W×3 uint8 BGR image with skin protection applied.
    """
    lch = bgr_to_lch(img_bgr)
    mask = skin_mask_lch(lch)
    op_result = op(img_bgr)
    return _blend_skin(img_bgr, op_result, mask, strength)


def protect_skin_chromatic(
    img_bgr: np.ndarray,
    hue_shift_deg: float,
    sat_factor: float = 1.0,
    strength: float = 0.7,
) -> np.ndarray:
    """Apply a hue shift and chroma scale with skin-tone protection.

    The full effect is applied outside the skin region; inside the
    skin region the effect is scaled down by ``skin_mask * strength``.
    At ``strength = 1`` and ``skin_mask = 1`` the skin pixel is
    unchanged; at ``strength = 0`` the full effect reaches every pixel.

    Args:
        img_bgr: H×W×3 uint8 BGR image.
        hue_shift_deg: Hue rotation in degrees applied to non-skin pixels.
        sat_factor: Multiplicative chroma scale applied to non-skin pixels
            (1.0 = no change, 1.2 = +20% saturation).
        strength: Protection strength in [0, 1].

    Returns:
        H×W×3 uint8 BGR image with skin-protected chromatic edits.
    """
    lch = bgr_to_lch(img_bgr)
    mask = skin_mask_lch(lch)
    weight = np.clip(mask.astype(np.float32), 0.0, 1.0) * float(strength)
    keep = 1.0 - weight

    out = lch.copy()
    out[:, :, 2] = np.mod(out[:, :, 2] + float(hue_shift_deg) * keep, 360.0)
    out[:, :, 1] = out[:, :, 1] * (1.0 + (float(sat_factor) - 1.0) * keep)
    return lch_to_bgr(out)


def skin_aware_apply(
    img_bgr: np.ndarray,
    op_skin: Callable[[np.ndarray], np.ndarray],
    op_other: Callable[[np.ndarray], np.ndarray],
    skin_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply different operations to skin vs non-skin regions.

    ``op_skin`` runs inside the skin region, ``op_other`` runs outside.
    Soft masks are honoured: a mask value of 0.5 blends the original
    pixel half-and-half with the corresponding op result.

    Args:
        img_bgr: H×W×3 uint8 BGR image.
        op_skin: Op applied within the skin region.
        op_other: Op applied outside the skin region.
        skin_mask: Optional pre-computed H×W float32 mask. If None,
            computed from the LCH skin mask.

    Returns:
        H×W×3 uint8 BGR image.
    """
    if skin_mask is None:
        lch = bgr_to_lch(img_bgr)
        skin_mask = skin_mask_lch(lch)
    else:
        skin_mask = skin_mask.astype(np.float32)

    after_skin = apply_to_region(img_bgr, skin_mask, op_skin, strength=1.0)
    other_mask = 1.0 - np.clip(skin_mask, 0.0, 1.0)
    return apply_to_region(after_skin, other_mask, op_other, strength=1.0)
