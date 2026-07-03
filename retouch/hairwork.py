"""Strand flow field analysis for hair retouching.

Computes the structure tensor (per-pixel gradient outer product) to extract
local hair strand orientation and coherence. These primitives power H1–H4
hair retouching stages.

Theory:
    Hair specular reflection is anisotropic — the highlight runs **perpendicular
    to strand direction** (Kajiya-Kay model). By computing the structure tensor,
    we extract two key attributes per pixel:

    1. **Orientation**: The direction of maximum gradient (perpendicular to strands).
       The actual strand direction is 90° rotated from this.
    2. **Coherence**: A measure of local anisotropy, in [0, 1].
       High coherence = well-defined strand direction (smooth gradient aligned region).
       Low coherence = isotropic (matte, fuzzy, or noise).

    The structure tensor J is the outer product of gradients:
        J = [gx, gy] ⊗ [gx, gy] = [gx², gx·gy; gx·gy, gy²]

    After Gaussian smoothing, its eigenvalues λ1, λ2 encode:
        Coherence = (λ1 - λ2) / (λ1 + λ2 + eps)   [anisotropy in [0, 1]]
        Orientation = 0.5 * arctan2(2·Jxy, Jxx - Jyy)  [principal axis direction]

Convention:
    Orientation in radians, range [−π/2, π/2] (via arctan2 half-angle convention).
    This gives the direction of maximum gradient. Strand flow is perpendicular to
    the gradient, so to get strand direction from orientation, rotate by 90°
    (add π/2).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import blend_masked


def hair_flow(
    img_bgr: np.ndarray,
    hair_mask: Optional[np.ndarray] = None,
    max_dim: int = 1200,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute strand orientation and coherence via structure tensor.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        hair_mask: Optional (H, W) float mask in [0, 1]. If provided, coherence
                   is zeroed outside mask>0.05. None = compute everywhere.
        max_dim: Maximum image dimension. If max(H, W) > max_dim, downscale,
                 compute, then upsample results. (Default: 1200px.)

    Returns:
        (orientation, coherence): Both (H, W) float32 arrays.
            orientation: In radians, range [−π/2, π/2], per-pixel gradient direction
                        (strand direction is ⊥ to this, i.e., +π/2 rotation).
            coherence: In [0, 1], per-pixel anisotropy. 1.0 = perfect alignment,
                      0.0 = isotropic (noise, matte, or low-contrast region).
                      Outside hair_mask (if provided), coherence is 0.
    """
    h, w = img_bgr.shape[:2]
    is_downscaled = False
    scale = 1.0

    # Determine if downscaling is needed
    if max(h, w) > max_dim:
        is_downscaled = True
        scale = max_dim / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_work = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if hair_mask is not None:
            hair_mask_work = cv2.resize(
                hair_mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR
            )
        else:
            hair_mask_work = None
    else:
        img_work = img_bgr
        hair_mask_work = hair_mask

    # Compute structure tensor components
    # Convert to grayscale for gradient computation
    gray = cv2.cvtColor(img_work, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    # Compute gradients via Sobel
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)  # ∂/∂x
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)  # ∂/∂y

    # Structure tensor components: Jxx = gx², Jyy = gy², Jxy = gx·gy
    jxx = gx * gx
    jyy = gy * gy
    jxy = gx * gy

    # Gaussian smoothing (sigma ≈ 4 pixels)
    sigma = 4.0
    ksize = int(2 * np.ceil(3 * sigma) + 1)  # Typical OpenCV rule: ~6σ in kernel
    jxx = cv2.GaussianBlur(jxx, (ksize, ksize), sigma)
    jyy = cv2.GaussianBlur(jyy, (ksize, ksize), sigma)
    jxy = cv2.GaussianBlur(jxy, (ksize, ksize), sigma)

    # Compute orientation and coherence from structure tensor
    # orientation = 0.5 * arctan2(2·Jxy, Jxx - Jyy)  [half-angle formula]
    # coherence = sqrt((Jxx - Jyy)² + 4·Jxy²) / (Jxx + Jyy + eps)
    #           = (λ1 - λ2) / (λ1 + λ2 + eps)

    orientation = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)

    eps = 1e-8
    numerator = np.sqrt((jxx - jyy) ** 2 + 4.0 * jxy ** 2)
    denominator = jxx + jyy + eps
    coherence = numerator / denominator

    # Clamp coherence to [0, 1]
    coherence = np.clip(coherence, 0.0, 1.0)

    # If hair_mask provided, zero coherence outside mask
    if hair_mask_work is not None:
        # Normalize mask to [0, 1] if needed
        mask_norm = hair_mask_work.astype(np.float32)
        if mask_norm.max() > 1.0:
            mask_norm /= 255.0
        # Zero coherence outside mask > 0.05 threshold
        coherence = np.where(mask_norm > 0.05, coherence, 0.0)

    # Upsample back to original resolution if downscaled
    if is_downscaled:
        # Orientation is AXIAL (θ and θ+π denote the same strand axis, range
        # [−π/2, π/2]). Interpolating sin(θ)/cos(θ) directly breaks at the
        # ±π/2 wrap (−89° and +89° would average toward 0°). Standard fix:
        # interpolate the DOUBLED angle — sin(2θ)/cos(2θ) are continuous
        # across the axial wrap — then halve after arctan2.
        sin2 = np.sin(2.0 * orientation)
        cos2 = np.cos(2.0 * orientation)
        sin2_up = cv2.resize(sin2, (w, h), interpolation=cv2.INTER_LINEAR)
        cos2_up = cv2.resize(cos2, (w, h), interpolation=cv2.INTER_LINEAR)
        # Recombine: orientation = 0.5 * arctan2(sin2, cos2)
        orientation = 0.5 * np.arctan2(sin2_up, cos2_up)

        # Coherence: direct upsampling (it's a scalar field, no angle issues)
        coherence = cv2.resize(coherence, (w, h), interpolation=cv2.INTER_LINEAR)
        coherence = np.clip(coherence, 0.0, 1.0)

        # Re-apply mask constraint at original resolution
        if hair_mask is not None:
            mask_norm = hair_mask.astype(np.float32)
            if mask_norm.max() > 1.0:
                mask_norm /= 255.0
            coherence = np.where(mask_norm > 0.05, coherence, 0.0)

    # Ensure no NaN or inf values
    orientation = np.nan_to_num(orientation, nan=0.0, posinf=0.0, neginf=0.0)
    coherence = np.nan_to_num(coherence, nan=0.0, posinf=1.0, neginf=0.0)

    return orientation.astype(np.float32), coherence.astype(np.float32)


def unify_hair_color(
    img_bgr: np.ndarray,
    hair_mask: np.ndarray,
    strength: int = 50,
    target_hue: Optional[float] = None,
    target_chroma: Optional[float] = None,
) -> np.ndarray:
    """Unify hair color by pulling toward median hue/chroma within hair mask.

    Removes venue-light color casts (especially on white/silver wigs) by
    gently pulling each hair pixel's hue toward the hair-region median hue
    and compressing chroma variance. Luminance is untouched so shading remains.

    Uses the same OKLCh hue-line math as C1 (skin color science).

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        hair_mask: (H, W) float mask [0, 1] indicating hair region.
        strength: 0-100 pull strength. 0 returns input unchanged.
        target_hue: Optional target hue in degrees. If None, computed as
                    chroma-weighted median hue within hair_mask.
        target_chroma: Optional target chroma. If None, computed as median
                       chroma within hair_mask.

    Returns:
        (H, W, 3) uint8 BGR image with unified hair color.
    """
    if strength <= 0 or hair_mask is None:
        return img_bgr

    from .color_science import bgr_to_oklab, oklab_to_oklch, oklch_to_oklab

    s = strength / 100.0

    # Convert to OKLCh
    oklab = bgr_to_oklab(img_bgr)
    oklch = oklab_to_oklch(oklab)

    L = oklch[..., 0].copy()
    C = oklch[..., 1].copy()
    h = oklch[..., 2].copy()

    # Determine target hue and chroma from hair region
    mask_binary = (hair_mask > 0.05).astype(np.float32)
    if mask_binary.sum() < 100:
        return img_bgr  # Too few hair pixels

    if target_hue is None:
        # Chroma-weighted circular mean hue
        h_rad = np.deg2rad(h)
        sin_sum = (mask_binary * np.sin(h_rad)).sum()
        cos_sum = (mask_binary * np.cos(h_rad)).sum()
        target_hue = float(np.rad2deg(np.arctan2(sin_sum, cos_sum))) % 360.0

    if target_chroma is None:
        target_chroma = float(np.median(C[mask_binary > 0.5]))

    # Pull hue toward target (shortest arc, clipped to +/-10 deg)
    delta_h = (target_hue - h + 180.0) % 360.0 - 180.0
    delta_h_clipped = np.clip(delta_h, -10.0, 10.0)
    h_new = (h + delta_h_clipped * s * hair_mask) % 360.0

    # Pull chroma toward target (clipped to +/-0.03)
    delta_C = np.clip(target_chroma - C, -0.03, 0.03)
    C_new = C + delta_C * s * hair_mask

    # Reconstruct
    oklch_out = oklch.copy()
    oklch_out[..., 0] = L
    oklch_out[..., 1] = np.clip(C_new, 0.0, None)
    oklch_out[..., 2] = h_new

    result_oklab = oklch_to_oklab(oklch_out)
    from .color_science import oklab_to_bgr
    result = oklab_to_bgr(result_oklab)
    return blend_masked(img_bgr, result, hair_mask)
