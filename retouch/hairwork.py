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
    guided_filter,
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
_DEGLARE_MAX_REDUCTION: float = 0.70
_DEGLARE_MIDBAND_FLOOR: float = 0.85
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


def _midband_energy(L: np.ndarray, face_width: float) -> float:
    """Mid-band luminance energy of a region — the deglare dimensionality gauge.

    Reuses skin._blotch_bandpass's difference-of-Gaussians bandpass to measure
    the shading structure that must survive the deglare (per PLAN_H2 §2.5 the
    post-deglare value must stay ≥ 85% of the pre-deglare value).
    """
    sigma_small = max(1.0, face_width / 40.0)
    sigma_large = max(1.0, face_width / 12.0)
    low_small = cv2.GaussianBlur(L, (0, 0), sigma_small)
    low_large = cv2.GaussianBlur(L, (0, 0), sigma_large)
    band = low_small - low_large
    return float(np.mean(band * band))


def _anisotropic_feather(
    mask: np.ndarray,
    orientation: np.ndarray,
    face_width: float,
) -> np.ndarray:
    """Feather a mask along the strand tangent (orientation + π/2).

    A separable two-pass Gaussian with σ_along = 4·σ_across elongates the
    mask along the strand so the deglare reads as a band, not a blob. Where
    the flow field is undefined (orientation == 0 everywhere) this collapses
    to a near-isotropic blur, which is the safe fallback.
    """
    sigma_across = max(1.0, face_width / 120.0)
    sigma_along = 4.0 * sigma_across
    # Two 1D passes. The across-strand direction is the gradient principal
    # axis (orientation); the along-strand direction is orientation + π/2.
    # We approximate by blurring with σ_along along the dominant strand axis
    # (vertical for near-horizontal strands, horizontal for near-vertical),
    # then σ_across along the perpendicular axis. A pixel-wise anisotropic
    # kernel is impractical per-pixel, so we use the angle to pick the axis
    # pair via a soft blend of the two separable orderings.
    # Vertical σ_along pass
    v = cv2.GaussianBlur(mask, (0, int(2 * np.ceil(3 * sigma_along)) + 1), sigma_along)
    v = cv2.GaussianBlur(v, (int(2 * np.ceil(3 * sigma_across)) + 1, 0), sigma_across)
    # Horizontal σ_along pass
    h = cv2.GaussianBlur(mask, (int(2 * np.ceil(3 * sigma_along)) + 1, 0), sigma_along)
    h = cv2.GaussianBlur(h, (0, int(2 * np.ceil(3 * sigma_across)) + 1), sigma_across)
    # Blend by how vertical the strand is. Strand tangent angle:
    #   tan_angle = orientation + π/2
    # weight_v = sin²(tan_angle) → 1 when strands run vertically.
    tan_angle = orientation + (np.pi * 0.5)
    w_v = np.sin(tan_angle) ** 2
    w_v = w_v.astype(np.float32)
    out = v * w_v + h * (1.0 - w_v)
    return np.clip(out, 0.0, 1.0)


def deglare_wig(
    img_bgr: np.ndarray,
    hair_mask: Optional[np.ndarray],
    orientation: np.ndarray,
    coherence: np.ndarray,
    strength: int = 0,
    face_width: float = 200.0,
    eyebrow_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Tame synthetic wig specular (deglare) without killing dimensionality.

    Adapts SkinProcessor.shine_removal to hair scale: detect bright
    low-chroma pixels within ``hair_mask``, reconstruct their chroma via a
    guided filter, and compress L toward a local non-glare target with an
    exponential soft rolloff capped at 70%. The detection mask is feathered
    anisotropically along the strand tangent (from the H0 flow field) and
    gated by a coherence floor so fuzzy/parting regions are left alone. A
    mid-band energy floor scales the compression back if it would flatten
    the wig's shading dimensionality below 85% of pre-deglare.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
        hair_mask: (H, W) float mask in [0, 1]. None or empty → no-op.
        orientation: (H, W) float32 H0 orientation field (radians).
        coherence: (H, W) float32 H0 coherence field in [0, 1].
        strength: 0–100 deglare intensity. 0 returns input unchanged.
        face_width: Face width in pixels; all kernel radii scale from it.
        eyebrow_mask: Optional union of left+right eyebrow masks, dilated
            and subtracted from the glare mask so eyebrow hairs are never
            deglared (mirrors skin.dodge_burn's eyebrow exclusion).

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

    mask_f = normalize_mask(hair_mask)
    if mask_f is None or mask_f.max() < 0.05:
        return img_bgr
    mask_f = mask_f.astype(np.float32, copy=False)
    if mask_f.shape != L.shape:
        mask_f = cv2.resize(mask_f, (L.shape[1], L.shape[0]),
                            interpolation=cv2.INTER_LINEAR)

    hair_idx = mask_f > 0.3
    if not np.any(hair_idx):
        return img_bgr

    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)
    median_L = float(np.median(L[hair_idx]))
    median_chroma = float(np.median(chroma[hair_idx]))

    # H2 §2.2 — retuned thresholds vs S4 (skin).
    shine_L_threshold = median_L + 12.0
    chroma_thresh = max(5.0, median_chroma * 0.25)

    L_gate = np.clip((L - shine_L_threshold) / 15.0, 0.0, 1.0)
    chroma_gate = 1.0 - np.clip(chroma / chroma_thresh, 0.0, 1.0)
    glare_m = L_gate * chroma_gate * mask_f

    # H2 §2.3 — coherence gate + anisotropic feather along strand tangent.
    coh = np.where(coherence > _COHERENCE_FLOOR, coherence, 0.0).astype(np.float32)
    glare_m = glare_m * coh
    glare_m = _anisotropic_feather(glare_m, orientation, face_width)
    glare_m = np.clip(glare_m, 0.0, 1.0)

    # H2 §6.3 — eyebrow/eyelash exclusion.
    if eyebrow_mask is not None:
        eb = normalize_mask(eyebrow_mask)
        if eb is not None:
            eb = eb.astype(np.float32, copy=False)
            if eb.shape != L.shape:
                eb = cv2.resize(eb, (L.shape[1], L.shape[0]),
                                interpolation=cv2.INTER_LINEAR)
            eb_dilated = cv2.dilate(eb, np.ones((5, 5), np.uint8))
            glare_m = glare_m * (1.0 - eb_dilated)

    if glare_m.max() < 0.01:
        return img_bgr

    # H2 §2.4 — chroma inpaint + local non-glare L target (radii scale with
    # face_width, NOT the S4 hardcoded 30 / (31,31)).
    gf_radius = max(4, int(face_width / 8.0))
    a_inpainted = guided_filter(a.astype(np.float32, copy=False),
                                radius=gf_radius, eps=100.0, guide=None)
    b_inpainted = guided_filter(b.astype(np.float32, copy=False),
                                radius=gf_radius, eps=100.0, guide=None)
    a_new = a * (1.0 - glare_m) + a_inpainted * glare_m
    b_new = b * (1.0 - glare_m) + b_inpainted * glare_m

    # Local non-glare L target. Use a *hard* non-shine mask for the target
    # field so bright glare pixels are fully replaced by the regional median
    # before blurring — otherwise a broad glare band leaks its own L back
    # into the target via the soft mask and the compression has nothing to
    # pull toward. The feathered ``glare_m`` is still what blends the final
    # result, so the visible transition stays soft.
    hard_shine = (glare_m > 0.05).astype(np.float32)
    non_shine_hard = 1.0 - hard_shine
    L_masked = L * non_shine_hard + median_L * hard_shine
    target_sigma = max(1.0, face_width / 8.0)
    L_target_smooth = cv2.GaussianBlur(L_masked, (0, 0), target_sigma)
    L_target = L * (1.0 - glare_m) + L_target_smooth * glare_m

    s = strength / 100.0
    L_excess = np.maximum(L - L_target, 0.0)
    excess_max = float(np.max(L_excess))
    if excess_max <= 0.0:
        return img_bgr
    normalized_excess = L_excess / (excess_max + 1e-6)
    compression_factor = 1.0 - np.exp(-1.5 * normalized_excess)
    compression_factor = np.clip(compression_factor * s, 0.0,
                                 _DEGLARE_MAX_REDUCTION)

    # H2 §2.5 — mid-band dimensionality floor. Scale back compression if the
    # deglare would drop mid-band energy below 85% of pre-deglare.
    energy_pre = _midband_energy(L * mask_f, face_width)
    L_trial = L - L_excess * compression_factor * glare_m
    energy_post = _midband_energy(L_trial * mask_f, face_width)
    if energy_post < _DEGLARE_MIDBAND_FLOOR * energy_pre and energy_post > 1e-8:
        scale_back = (energy_pre * _DEGLARE_MIDBAND_FLOOR) / energy_post
        scale_back = min(scale_back, 1.0)
        compression_factor = compression_factor * scale_back

    L_new = L - L_excess * compression_factor * glare_m

    lab_out = lab.copy()
    lab_out[:, :, 0] = np.clip(L_new, 0, 255)
    lab_out[:, :, 1] = a_new
    lab_out[:, :, 2] = b_new
    result = _from_lab(lab_out, is_float)
    return blend_masked(img_bgr, result, mask_f)


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
# See PLAN_TIERH_HAIR.md §Stage H1 for the spec. Detects thin curvilinear
# structures in the hair silhouette band and along the hairline that deviate
# significantly from the dominant H0 strand flow (a flyaway disagrees with
# the local strand orientation; a smooth edge strand agrees), then heals
# them via the F4 SpotHealer / heal_region machinery. Guards: never touch
# eyebrow/eyelash regions; strand-thickness cap (≤4px) so intentional hair
# locks are preserved. All pixel arithmetic is float32 internally; the
# output dtype matches the input image dtype (uint8 BGR or float32 [0,255]).

# Detection thresholds — tuned per the plan §H1.1.
_FLY_MAX_STRAND_PX: int = 4           # ≤4px = stray; thicker = intentional lock
_FLY_COHERENCE_FLOOR: float = 0.15    # below this, flow is undefined → skip deviation gate
_FLY_ALIGN_RAD: float = 0.10          # <~6° deviation = aligned edge strand → keep (not flyaway)
_FLY_TOPHAT_SCALES: tuple = (2, 3, 4) # px scales for black-hat line detection
_FLY_MIN_AREA_PX: int = 2            # drop specks smaller than this


def _flyaway_mask(
    img_bgr: np.ndarray,
    hair_mask: np.ndarray,
    orientation: np.ndarray,
    coherence: np.ndarray,
    strength: int,
    exclude_mask: Optional[np.ndarray],
    face_width: float,
) -> np.ndarray:
    """Detect flyaway/stray hairs as a float32 [0,1] mask.

    Combines two signals:
      1. Morphological thin-line detection via multi-scale black-hat on L.
      2. Flow-deviation gate: a detected thin structure is only kept if its
         local orientation disagrees with the dominant strand flow by more
         than ``_FLY_DEVIATION_RAD`` (where coherence is high enough to
         trust the flow). Smooth edge strands that *agree* with the flow
         are left alone.

    The search is restricted to a silhouette band (dilated hair mask minus
    eroded hair mask) so the detector never runs on the hair interior where
    every strand is a thin line.
    """
    h, w = img_bgr.shape[:2]
    mask_f = normalize_mask(hair_mask)
    if mask_f is None or mask_f.max() < 0.05:
        return np.zeros((h, w), dtype=np.float32)
    mask_f = mask_f.astype(np.float32, copy=False)
    if mask_f.shape != (h, w):
        mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)

    # Build the silhouette band: dilate minus erode hair mask, widened by
    # a fraction of face_width so flyaways protruding outward are caught.
    band_w = max(3, int(face_width * 0.06))
    k_dilate = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_w * 2 + 1, band_w * 2 + 1)
    )
    k_erode = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (band_w + 1, band_w + 1)
    )
    dilated = cv2.dilate(mask_f, k_dilate)
    eroded = cv2.erode(mask_f, k_erode)
    silhouette_band = np.clip(dilated - eroded, 0.0, 1.0).astype(np.float32)

    # The thin-line detector would flag every interior strand if run on the
    # full hair mask. Instead, the silhouette band catches protruding
    # flyaways, and the flow-deviation gate (below) is what makes the
    # detection selective inside the hair region. We therefore search the
    # union of the silhouette band and the full hair mask interior — the
    # flow gate + thickness cap keep interior strands from being flagged.
    search_band = np.clip(silhouette_band + mask_f, 0.0, 1.0).astype(np.float32)

    # Exclude eyebrow/eyelash regions if provided.
    if exclude_mask is not None:
        ex = normalize_mask(exclude_mask)
        if ex is not None:
            ex = ex.astype(np.float32, copy=False)
            if ex.shape != (h, w):
                ex = cv2.resize(ex, (w, h), interpolation=cv2.INTER_LINEAR)
            ex_dilated = cv2.dilate(
                ex, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            )
            search_band = search_band * (1.0 - ex_dilated)

    if search_band.max() < 0.01:
        return np.zeros((h, w), dtype=np.float32)

    # --- Thin-line detection via multi-scale black-hat on luminance ---
    is_float = img_bgr.dtype == np.float32
    lab = _to_lab(img_bgr, is_float)
    L = lab[:, :, 0].astype(np.float32, copy=False)

    line_response = np.zeros((h, w), dtype=np.float32)
    for scale in _FLY_TOPHAT_SCALES:
        ksize = scale * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        blackhat = cv2.morphologyEx(L, cv2.MORPH_BLACKHAT, kernel)
        tophat = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, kernel)
        # Flyaways can be either darker or lighter than surroundings
        combined = np.maximum(blackhat, tophat)
        line_response = np.maximum(line_response, combined)

    # Normalize line response to [0,1] within the search band. Flyaways
    # are sparse (typically <5% of the band), so the 98th percentile equals
    # the median — we normalize against the max response instead, with a
    # floor so a flat zero-response image (no lines) yields an all-zero mask.
    band_vals = line_response[search_band > 0.05]
    if band_vals.size == 0:
        return np.zeros((h, w), dtype=np.float32)
    p_lo = float(np.percentile(band_vals, 50))
    p_hi = float(band_vals.max())
    if p_hi - p_lo < 1.0:
        return np.zeros((h, w), dtype=np.float32)
    line_norm = np.clip((line_response - p_lo) / (p_hi - p_lo), 0.0, 1.0)

    # Strength slider → detection threshold. Higher strength = lower
    # threshold = more flyaways detected.
    s = max(0, min(100, strength)) / 100.0
    detect_thresh = 0.55 - 0.35 * s  # 0.55 at s=0, 0.20 at s=1
    line_mask = (line_norm > detect_thresh).astype(np.float32) * search_band

    # --- Thickness cap: keep only thin structures (≤ _FLY_MAX_STRAND_PX) ---
    # Erode by the cap radius; what disappears is thin enough to be a flyaway.
    cap_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (_FLY_MAX_STRAND_PX * 2 + 1, _FLY_MAX_STRAND_PX * 2 + 1),
    )
    thick = cv2.erode(line_mask, cap_k)
    thin_only = line_mask * (1.0 - thick)

    # --- Flow-alignment gate (edge-strand preservation) ---
    # The structure tensor's Gaussian smoothing (σ≈4px) averages over a
    # neighborhood larger than a 2–4px flyaway, so the per-pixel orientation
    # at a flyaway reads as the surrounding strand direction. The gate
    # therefore cannot reliably *detect* a flyaway by deviation. It is used
    # only to spare **thick** edge strands that run along the dominant flow:
    # we compute the dominant flow, and where a detected structure is thick
    # (survives the cap erosion) AND aligned with the flow, we reject it as
    # an intentional edge strand. Thin structures always pass — the
    # thickness cap is their discriminator, not the flow gate.
    trusted = (coherence > _FLY_COHERENCE_FLOOR).astype(np.float32)
    if trusted.max() > 0:
        sin2 = np.sin(2.0 * orientation)
        cos2 = np.cos(2.0 * orientation)
        dom_sigma = max(2.0, face_width / 20.0)
        ksize_dom = int(2 * np.ceil(3 * dom_sigma) + 1)
        sin2_dom = cv2.GaussianBlur(sin2 * trusted, (ksize_dom, ksize_dom), dom_sigma)
        cos2_dom = cv2.GaussianBlur(cos2 * trusted, (ksize_dom, ksize_dom), dom_sigma)
        dom_orient = 0.5 * np.arctan2(sin2_dom, cos2_dom)
        delta_2 = 2.0 * orientation - 2.0 * dom_orient
        delta_wrapped = np.arctan2(np.sin(delta_2), np.cos(delta_2))
        deviation = 0.5 * np.abs(delta_wrapped)
        aligned = (deviation < _FLY_ALIGN_RAD).astype(np.float32)
        # Only reject thick aligned structures; thin flyaways (already
        # isolated by ``thin_only``) pass regardless.
        reject = (thick > 0.5) * aligned * trusted
        thin_only = thin_only * (1.0 - reject)

    # --- Clean up specks ---
    if _FLY_MIN_AREA_PX > 0:
        thin_u8 = (thin_only > 0.5).astype(np.uint8) * 255
        n, labels, stats, _ = cv2.connectedComponentsWithStats(thin_u8, connectivity=8)
        clean = np.zeros_like(thin_u8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= _FLY_MIN_AREA_PX:
                clean[labels == i] = 255
        thin_only = (clean.astype(np.float32) / 255.0)

    # Feather the mask edges so the heal blends cleanly.
    feather_k = max(3, int(face_width * 0.03)) | 1
    out = cv2.GaussianBlur(thin_only, (feather_k, feather_k), 0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def remove_flyaways(
    img_bgr: np.ndarray,
    hair_mask: Optional[np.ndarray],
    orientation: np.ndarray,
    coherence: np.ndarray,
    strength: int = 0,
    face_width: float = 200.0,
    eyebrow_mask: Optional[np.ndarray] = None,
    skin_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Remove flyaway / stray hairs from the hair silhouette and hairline.

    Detects thin curvilinear structures in the hair silhouette band (and a
    thin inside-hairline band) that deviate from the dominant H0 strand
    flow, then heals them via the F4 :class:`SpotHealer` / ``heal_region``
    machinery. Intentional hair locks (thicker than
    ``_FLY_MAX_STRAND_PX``) and edge strands that agree with the flow are
    preserved. Eyebrow/eyelash regions are never touched.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
        hair_mask: (H, W) float mask in [0, 1]. None or empty → no-op.
        orientation: (H, W) float32 H0 orientation field (radians).
        coherence: (H, W) float32 H0 coherence field in [0, 1].
        strength: 0–100 detection strength. 0 returns input unchanged.
            Higher values lower the detection threshold so more flyaways
            are caught (at the risk of eating intentional wisps).
        face_width: Face width in pixels; silhouette band and feather
            radii scale from it.
        eyebrow_mask: Optional union of left+right eyebrow (and eyelash)
            masks, dilated and subtracted from the search band so those
            regions are never healed.
        skin_mask: Optional skin mask. When provided, face-crossing
            flyaways that overlap skin are also removed; without it only
            the hair silhouette band is searched.

    Returns:
        (H, W, 3) image, same dtype as input.
    """
    if strength <= 0 or hair_mask is None:
        return img_bgr

    h, w = img_bgr.shape[:2]
    mask_f = normalize_mask(hair_mask)
    if mask_f is None or mask_f.max() < 0.05:
        return img_bgr
    mask_f = mask_f.astype(np.float32, copy=False)
    if mask_f.shape != (h, w):
        mask_f = cv2.resize(mask_f, (w, h), interpolation=cv2.INTER_LINEAR)

    # Build the exclude mask (eyebrows + optionally skin-free zone guard).
    exclude = None
    if eyebrow_mask is not None:
        exclude = normalize_mask(eyebrow_mask)
        if exclude is not None:
            exclude = exclude.astype(np.float32, copy=False)
            if exclude.shape != (h, w):
                exclude = cv2.resize(
                    exclude, (w, h), interpolation=cv2.INTER_LINEAR
                )

    fly_mask = _flyaway_mask(
        img_bgr, mask_f, orientation, coherence,
        strength=strength, exclude_mask=exclude, face_width=face_width,
    )

    # Optionally extend the search to face-crossing flyaways over skin.
    if skin_mask is not None:
        skin_f = normalize_mask(skin_mask)
        if skin_f is not None:
            skin_f = skin_f.astype(np.float32, copy=False)
            if skin_f.shape != (h, w):
                skin_f = cv2.resize(skin_f, (w, h), interpolation=cv2.INTER_LINEAR)
            face_fly = _flyaway_mask(
                img_bgr, skin_f, orientation, coherence,
                strength=strength, exclude_mask=exclude, face_width=face_width,
            )
            fly_mask = np.clip(fly_mask + face_fly, 0.0, 1.0).astype(np.float32)

    if fly_mask.max() < 0.01:
        return img_bgr

    # Heal via the F4 SpotHealer. Small-radius single-pass Telea is the
    # right tool for thin flyaways — multi-pass object removal is overkill
    # and smears.
    from .spot_heal import SpotHealer

    healer = SpotHealer()
    radius = max(2, int(face_width * 0.02))
    healed = healer.heal(img_bgr, fly_mask, method="telea", radius=radius)
    return healed
