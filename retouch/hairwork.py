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

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .utils import (
    blend_masked,
    bgr_f32_to_lab_f32,
    lab_f32_to_bgr_f32,
    normalize_mask,
)


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


# ---------------------------------------------------------------------------
# H2 — Wig shine shaping (deglare + anisotropic angel ring)
# ---------------------------------------------------------------------------
#
# See docs/PLAN_H2_WIG_SHINE.md for the full design. Both sub-ops are gated by
# the hair mask and steered by the H0 (orientation, coherence) flow field so
# the highlight/detected glare follow strand structure rather than reading as
# isotropic blobs — the failure mode of the legacy HairEnhancer.enhance().
# All LAB arithmetic is float32; the is_float path preserves float32 in/out
# (matches skin.shine_removal and hair.HairEnhancer conventions).


_COHERENCE_FLOOR: float = 0.25
_RING_ALPHA: float = 8.0
_RING_LIGHT_AZIMUTH_DEG: float = 0.0
_RING_LIGHT_ELEVATION_DEG: float = 70.0


def _to_lab(img_bgr: np.ndarray, is_float: bool) -> np.ndarray:
    if is_float:
        return bgr_f32_to_lab_f32(img_bgr)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab: np.ndarray, is_float: bool) -> np.ndarray:
    clipped = np.clip(lab, 0, 255)
    if is_float:
        return lab_f32_to_bgr_f32(clipped)
    return cv2.cvtColor(clipped.astype(np.uint8), cv2.COLOR_LAB2BGR)


def deglare_wig(
    img_bgr: np.ndarray,
    hair_mask: Optional[np.ndarray],
    orientation: np.ndarray,
    coherence: np.ndarray,
    strength: int = 0,
    face_width: float = 200.0,
    eyebrow_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Soften synthetic wig shine into the wig's own colour ("Wig Shine").

    Delegates to :func:`retouch.wig_shine.matte_wig_shine`, which measures
    shine against a local baseline of the wig itself (tone-invariant) and
    removes it as neutral linear light, so the fibre colour comes back. The
    first version of this function gated on low chroma against the whole
    wig's median and was inert on real wigs (0.5-2 L at strength 60).

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
        hair_mask: (H, W) float mask in [0, 1]. None or empty -> no-op.
        orientation: (H, W) H0 orientation field (unused; kept for the
            dispatch signature shared with the other H-stage ops).
        coherence: (H, W) H0 coherence field in [0, 1]; fuzzy and parting
            areas (low coherence) are left alone.
        strength: 0-100. 0 returns input unchanged.
        face_width: Face width in pixels; all scales derive from it.
        eyebrow_mask: Optional eyebrow mask, never touched.

    Returns:
        (H, W, 3) image, same dtype as input.
    """
    from .wig_shine import matte_wig_shine

    return matte_wig_shine(
        img_bgr, hair_mask, strength, face_width,
        coherence=coherence, exclude_mask=eyebrow_mask,
    )


def add_angel_ring(
    img_bgr: np.ndarray,
    hair_mask: Optional[np.ndarray],
    orientation: np.ndarray,
    coherence: np.ndarray,
    strength: int = 0,
    position: int = 30,
    tint: int = 40,
    face_width: float = 200.0,
    landmarks: Any = None,
) -> np.ndarray:
    """Add an anisotropic Kajiya-Kay angel-ring highlight along the crown.

    Lays a coherent highlight band on the upper crown, perpendicular to
    strand flow, tinted toward the hair's own chroma (never pure white).
    The band is modulated by ``cos(t · L_perp) ** α`` so it is brightest
    where strands run perpendicular to the light and dims to nothing where
    they align — the anisotropic behaviour real hair exhibits. A coherence
    floor breaks the ring at partings/occlusions and suppresses it on
    fuzzy/short wigs. Highlight protection prevents clipping.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
        hair_mask: (H, W) float mask in [0, 1]. None or empty → no-op.
        orientation: (H, W) float32 H0 orientation field (radians).
        coherence: (H, W) float32 H0 coherence field in [0, 1].
        strength: 0–100 ring strength (the reinterpreted ``hair_enhance``).
        position: 0–100 vertical band position (0 = top, 100 = bottom of
            hair). Used only when ``landmarks`` is None.
        tint: 0–100 how strongly the ring is pulled toward hair chroma vs
            neutral. 0 = neutral, 100 = full hair-color match.
        face_width: Face width in pixels; band width scales from it.
        landmarks: Optional MediaPipe landmarks object. When supplied, the
            crown arc is derived from the head ellipse; otherwise the
            ``position`` slider is used.

    Returns:
        (H, W, 3) image, same dtype as input.
    """
    if strength <= 0 or hair_mask is None:
        return img_bgr

    is_float = img_bgr.dtype == np.float32
    lab = _to_lab(img_bgr, is_float)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    h_img, w_img = L.shape

    mask_f = normalize_mask(hair_mask)
    if mask_f is None or mask_f.max() < 0.05:
        return img_bgr
    mask_f = mask_f.astype(np.float32, copy=False)
    if mask_f.shape != L.shape:
        mask_f = cv2.resize(mask_f, (w_img, h_img),
                            interpolation=cv2.INTER_LINEAR)

    hair_idx = mask_f > 0.3
    if not np.any(hair_idx):
        return img_bgr

    # H2 §3.2 — crown band. Landmark arc is the primary; position slider the
    # fallback. We approximate the landmark arc with a horizontal band
    # centred on the hair region's vertical extent when landmarks are
    # unavailable (the common test/CI path), and refine toward the landmark
    # forehead-top y when supplied.
    ys = np.where(mask_f.max(axis=1) > 0.3)[0]
    if len(ys) == 0:
        return img_bgr
    hair_top = float(ys[0])
    hair_bot = float(ys[-1])
    if landmarks is not None:
        try:
            lm = landmarks.landmark
            cy_band = lm[10].y * h_img
        except (AttributeError, IndexError, TypeError):
            cy_band = hair_top + (hair_bot - hair_top) * (position / 100.0)
    else:
        cy_band = hair_top + (hair_bot - hair_top) * (position / 100.0)

    band_half = max(2.0, face_width * 0.15)
    yy = np.arange(h_img, dtype=np.float32)[:, None]
    band_v = np.exp(-((yy - cy_band) ** 2) / (2.0 * band_half * band_half))
    band_v = np.broadcast_to(band_v, (h_img, w_img)).astype(np.float32, copy=True)
    crown_band_m = band_v * mask_f

    # H2 §3.3 — Kajiya-Kay modulation. cos_θ = |t · L_perp|.
    theta = np.deg2rad(_RING_LIGHT_AZIMUTH_DEG)
    phi = np.deg2rad(_RING_LIGHT_ELEVATION_DEG)
    # L_perp = in-image-plane component of the light vector.
    Lx = np.cos(phi) * np.sin(theta)
    Ly = -np.sin(phi)
    L_norm = float(np.hypot(Lx, Ly)) + 1e-8
    Lx /= L_norm
    Ly /= L_norm
    # Strand tangent: t = (cos(orientation + π/2), sin(orientation + π/2)).
    tan_angle = orientation + (np.pi * 0.5)
    tx = np.cos(tan_angle)
    ty = np.sin(tan_angle)
    cos_theta = np.abs(tx * Lx + ty * Ly)
    ring_profile = np.power(np.clip(cos_theta, 0.0, 1.0), _RING_ALPHA)

    coh = np.where(coherence > _COHERENCE_FLOOR, coherence, 0.0).astype(np.float32)
    ring_m = crown_band_m * ring_profile * coh
    # Soft feather so the band doesn't have a hard edge.
    feather_k = max(3, int(face_width * 0.05)) | 1
    ring_m = cv2.GaussianBlur(ring_m, (feather_k, feather_k), 0)
    ring_m = np.clip(ring_m, 0.0, 1.0)

    if ring_m.max() < 1e-3:
        return img_bgr

    s = strength / 100.0
    t_frac = tint / 100.0

    # H2 §3.4 — tint toward hair median chroma, highlight-protected L lift.
    median_a = float(np.median(a[hair_idx]))
    median_b = float(np.median(b[hair_idx]))

    # Highlight protection: prot = clip(1 - (L - 220)/30, 0, 1).
    prot = np.clip(1.0 - (L - 220.0) / 30.0, 0.0, 1.0)
    L_ring = L + s * ring_m * (255.0 - L) * 0.12 * prot
    a_ring = a + (median_a - a) * ring_m * (0.15 * t_frac)
    b_ring = b + (median_b - b) * ring_m * (0.15 * t_frac)

    lab_out = lab.copy()
    lab_out[:, :, 0] = np.clip(L_ring, 0, 255)
    lab_out[:, :, 1] = np.clip(a_ring, 0, 255)
    lab_out[:, :, 2] = np.clip(b_ring, 0, 255)
    result = _from_lab(lab_out, is_float)
    return blend_masked(img_bgr, result, mask_f)


# ---------------------------------------------------------------------------
# H1 — Flyaway / stray-hair removal
# ---------------------------------------------------------------------------
#
# Moved to retouch/stray_hair.py (Stray Hair Cleanup, hair_remove_flyaways).
# The first version searched the wig itself and normalised its line response
# against the band's strongest line (lash lines, trim), so it removed no real
# strands and Telea-inpainted catchlights and pupils instead.
