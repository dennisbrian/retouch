"""3-level frequency separation — the core of professional retouching.

Splits an image into three frequency bands:

    Low  — colour / overall tone  (large-scale)
    Mid  — blemishes, wrinkles    (medium-scale)
    High — pores, fine hairs      (small-scale)

Processing strategy:
    • Smooth the LOW layer to even out skin tone.
    • Reduce the MID layer to remove blemishes.
    • Keep the HIGH layer mostly untouched → natural texture preserved.
    • Recombine: Result = SmoothedLow + ReducedMid + High.

Key design choice: compositing is done on the **final reconstructed pixel**
values, not on individual layers. This avoids tonal discontinuities at
feathered mask boundaries where different layers have been modified
by different amounts.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import adaptive_ksize, blend_masked, estimate_face_width, guided_filter, squeeze_mask

logger = logging.getLogger(__name__)

# Constants for adaptive sizing and parameters
DEFAULT_FEATHER_FACTOR = 0.015
FALLBACK_FEATHER_FACTOR = 0.0075
DEFAULT_FEATHER_MIN = 5
APPROX_FACE_WIDTH_RATIO = 0.4
SMOOTH_K_FACTOR = 0.22
SMOOTH_K_MIN = 9
SIGMA_BASE = 20.0
SIGMA_STRENGTH_FACTOR = 60.0
GAUSSIAN_BLEND_FACTOR = 0.25

# --- Adaptive texture preservation (flat/front-lit skin protection) ---
# Flat, front-lit skin has intrinsically low high-band amplitude, so fixed
# smoothing parameters strip real pore texture as if it were noise. We measure
# the robust high-band energy inside the skin mask and, when it is low, scale
# back the effective smoothing and floor texture_opacity upward. The response
# is smooth (no threshold pop), asymmetric (only ever *reduces* smoothing), and
# expressed in intensity units (resolution-independent for amplitude).
#
# ENERGY_LOW/ENERGY_HIGH are robust-std (MAD-derived) thresholds on the masked
# high band. Below ENERGY_LOW the adaptation is at its strongest (factor floor);
# at/above ENERGY_HIGH there is no adaptation (factor 1.0 → identical output).
TEXTURE_ENERGY_LOW = 0.5
TEXTURE_ENERGY_HIGH = 3.0
# Minimum multiplier applied to smooth/mid levers when texture is maximally
# flat. Never 0 — we still want gentle smoothing, just far less aggressive.
TEXTURE_ADAPT_FLOOR = 0.5
# MAD → std scaling for a normal distribution (1 / 0.6745).
_MAD_TO_STD = 1.4826


def _texture_adaptation_factor(
    high: np.ndarray,
    mask_2d: np.ndarray,
    energy_low: float = TEXTURE_ENERGY_LOW,
    energy_high: float = TEXTURE_ENERGY_HIGH,
    floor: float = TEXTURE_ADAPT_FLOOR,
) -> float:
    """Return a smooth, asymmetric adaptation factor in [floor, 1].

    Measures the robust high-band energy (MAD-based std) inside the skin mask.
    High energy (textured / directional light) → 1.0 (no change). Low energy
    (flat / front-lit) → toward ``floor``, so callers can scale back smoothing
    and preserve pore texture.

    The mapping is a smoothstep between ``energy_low`` and ``energy_high``, so
    there is no hard threshold and the response is monotonic in energy. The
    factor only ever *reduces* smoothing (it is capped at 1.0), keeping the
    adaptation asymmetric.

    Args:
        high: (H, W, 3) float32 high-frequency band (can be negative).
        mask_2d: (H, W) float mask; pixels > 0.5 count toward the measurement.
        energy_low: Energy at/below which adaptation is strongest (→ floor).
        energy_high: Energy at/above which there is no adaptation (→ 1.0).
        floor: Minimum returned factor when texture is maximally flat.

    Returns:
        Scalar float in [floor, 1.0]. Returns 1.0 (no adaptation) when the
        mask is empty or degenerate.
    """
    sel = mask_2d > 0.5
    n = int(np.count_nonzero(sel))
    if n < 16:
        return 1.0

    # Per-pixel high-band magnitude (mean over channels), measured only inside
    # the mask. MAD is robust to specular highlights / stray hairs that would
    # otherwise inflate a plain std.
    high_mag = np.abs(high).mean(axis=2)
    vals = high_mag[sel]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    energy = mad * _MAD_TO_STD

    # Smoothstep from LOW→HIGH: 0 at/below LOW, 1 at/above HIGH.
    span = energy_high - energy_low
    t = (energy - energy_low) / span
    t = min(1.0, max(0.0, t))
    smooth_t = t * t * (3.0 - 2.0 * t)  # smoothstep

    factor = floor + (1.0 - floor) * smooth_t
    return min(1.0, max(floor, factor))


# --- Region-aware smoothing strength modulation ---
# Some face regions naturally carry more (nose bridge wrinkles) or less
# (cheeks) high-frequency structure. A single global smooth_strength over-
# smooths flat regions and under-protects detailed ones. We measure the
# MAD-based high-band energy inside each known region and nudge the local
# smooth_strength toward a per-region TARGET:
#   • detail regions (target < 1.0) pull the factor DOWN when that region
#     actually has high energy (don't strip real wrinkles);
#   • smooth regions (target > 1.0) pull the factor UP when that region is
#     genuinely flat (safe to smooth more).
# The nudge is scaled by ``regional_modulation`` and clamped to [0.5, 2.5].
# When ``regional_modulation == 0`` or no regions are supplied the caller's
# path is a strict no-op (byte-identical output).
_REGION_MOD_CONFIG: Dict[str, Dict[str, float]] = {
    # Wide target spread so regional_modulation=1.0 is clearly visible. Flat
    # regions (cheeks/forehead) smooth up to ~2.5x; detail regions (nose bridge,
    # crows-feet, jawline) keep more structure. Bounded by the [0.5, 2.5] clamp;
    # the high band is never touched so pores survive.
    "nose_bridge":        {"target": 0.5,  "e_low": 1.0, "e_high": 4.0},
    "forehead":           {"target": 2.4,  "e_low": 1.0, "e_high": 4.0},
    "forehead_center":    {"target": 2.4,  "e_low": 1.0, "e_high": 4.0},
    "cheek_highlights_l": {"target": 2.5,  "e_low": 1.0, "e_high": 4.0},
    "cheek_highlights_r": {"target": 2.5,  "e_low": 1.0, "e_high": 4.0},
    "left_cheek":         {"target": 2.3,  "e_low": 1.0, "e_high": 4.0},
    "right_cheek":        {"target": 2.3,  "e_low": 1.0, "e_high": 4.0},
    "jawline_contour":    {"target": 0.6,  "e_low": 1.0, "e_high": 4.0},
    "crows_feet_l":       {"target": 0.6,  "e_low": 1.0, "e_high": 4.0},
    "crows_feet_r":       {"target": 0.6,  "e_low": 1.0, "e_high": 4.0},
    "nasolabial_l":       {"target": 2.3,  "e_low": 1.0, "e_high": 4.0},
    "nasolabial_r":       {"target": 2.3,  "e_low": 1.0, "e_high": 4.0},
}
_REGION_MOD_FLOOR = 0.5
_REGION_MOD_CEIL = 2.5


def _region_mask_crop(
    region_mask: np.ndarray,
    shape: Tuple[int, int],
    crop: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    """Return a (H, W) float32 region mask aligned to the (already cropped) ``high`` band.

    Args:
        region_mask: Full-image region mask (any float/uint, (H, W) or (H, W, 1)).
        shape: (H, W) of the cropped high band the masks must align to.
        crop: Optional (y1, y2, x1, x2) slice applied to a full-image mask.

    Returns:
        (H, W) float32 mask in [0, 1].
    """
    m = squeeze_mask(region_mask.astype(np.float32, copy=False) if region_mask.dtype != np.float32 else region_mask)
    if crop is not None:
        y1, y2, x1, x2 = crop
        m = m[y1:y2, x1:x2]
    if m.shape != shape:
        # Region mask did not align (e.g. wrong resolution); safely no-op.
        return np.zeros(shape, dtype=np.float32)
    return np.clip(m, 0.0, 1.0)


def _regional_modulation_factors(
    high: np.ndarray,
    regions: Any,
    shape: Tuple[int, int],
    regional_modulation: float = 0.0,
    crop: Optional[Tuple[int, int, int, int]] = None,
) -> Dict[str, float]:
    """Compute per-region smoothing-strength modulation factors.

    Args:
        high: (H, W, 3) float32 high-frequency band (may be negative).
        regions: FaceRegions-like object exposing the attributes in
            ``_REGION_MOD_CONFIG`` as float/uint masks.
        shape: (H, W) shape of ``high`` the region masks must align to.
        regional_modulation: Global modulation strength (0.0 = no-op → all 1.0).
        crop: Optional (y1, y2, x1, x2) slice applied to full-image masks.

    Returns:
        Dict mapping region name → factor clamped to [0.5, 2.5]. Regions
        absent from ``regions`` or with too few pixels return 1.0 (no-op).
    """
    if regional_modulation <= 0.0 or regions is None:
        return {}

    # MAD-based robust energy of the high band — computed once, reused for
    # every region (avoids 12 redundant full-frame abs+mean passes on 4K).
    high_mag = np.abs(high).mean(axis=2)

    factors: Dict[str, float] = {}
    for name, cfg in _REGION_MOD_CONFIG.items():
        region_mask = getattr(regions, name, None)
        if region_mask is None:
            factors[name] = 1.0
            continue

        m_f = _region_mask_crop(region_mask, shape, crop)
        sel = m_f > 0.5
        n = int(np.count_nonzero(sel))
        if n < 16:
            factors[name] = 1.0
            continue

        vals = high_mag[sel]
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med)))
        energy = mad * _MAD_TO_STD

        # Smoothstep from e_low → e_high: 0 below, 1 above.
        span = cfg["e_high"] - cfg["e_low"]
        t = (energy - cfg["e_low"]) / span if span > 0 else 0.0
        t = min(1.0, max(0.0, t))
        smooth_t = t * t * (3.0 - 2.0 * t)

        target = cfg["target"]
        # Direction of the nudge depends on whether this is a "detail" region
        # (target < 1: reduce smoothing when energetic) or a "smooth" region
        # (target > 1: increase smoothing when flat).
        if target < 1.0:
            m = (target - 1.0) * smooth_t
        else:
            m = (target - 1.0) * (1.0 - smooth_t)

        factor = 1.0 + regional_modulation * m
        factor = min(_REGION_MOD_CEIL, max(_REGION_MOD_FLOOR, factor))
        factors[name] = factor

    return factors


# --- Anisotropic (orientation-aware) smoothing ---
# Skin grain, wrinkles and pores follow directional structure. Isotropic
# guided/bilateral filtering blurs equally in every direction, flattening
# detail that runs across the grain. We estimate a per-pixel orientation
# field from the luminance structure tensor and smooth preferentially ALONG
# the local dominant grain direction while preserving structure across it.
# The high-frequency band is never touched (callers only pass low+mid here).
# Flat / ambiguous regions (low gradient magnitude) fall back to the existing
# isotropic guided filter so they are not destabilised by a noisy angle.
def _compute_orientation_field(L: np.ndarray, sigma_smooth: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Structure-tensor orientation field + confidence from a luminance map.

    Args:
        L: (H, W) float32 luminance in [0, 255].
        sigma_smooth: Gaussian sigma used to stabilise the tensor.

    Returns:
        theta: (H, W) float32 angle in [0, π) of MAXIMUM intensity change
            (gradient direction). The edge-tangent / grain direction is
            ``theta + π/2``.
        conf: (H, W) float32 confidence in [0, 1] from smoothed gradient
            magnitude (low magnitude → unreliable orientation → 0).
    """
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)

    Ixx = gx * gx
    Iyy = gy * gy
    Ixy = gx * gy

    s = max(0.5, float(sigma_smooth))
    Ixx = cv2.GaussianBlur(Ixx, (0, 0), s)
    Iyy = cv2.GaussianBlur(Iyy, (0, 0), s)
    Ixy = cv2.GaussianBlur(Ixy, (0, 0), s)

    # Principal gradient direction (0..π).
    theta = 0.5 * np.arctan2(2.0 * Ixy, Ixx - Iyy)
    theta = np.where(theta < 0.0, theta + np.pi, theta)

    grad_mag = np.sqrt(Ixx + Iyy)
    gm_smooth = cv2.GaussianBlur(grad_mag, (0, 0), s)
    # Normalise: ~0 on flat areas → 1 on strong edges. SIGMA_BASE sets scale.
    conf = gm_smooth / (gm_smooth + SIGMA_BASE * 0.5)
    conf = np.clip(conf, 0.0, 1.0)

    return theta.astype(np.float32), conf.astype(np.float32)


def _bilinear_sample(img: np.ndarray, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Vectorised bilinear sample of a 2D float32 image at float coordinates."""
    hh, ww = img.shape
    px = np.clip(px, 0.0, ww - 1)
    py = np.clip(py, 0.0, hh - 1)
    x0 = np.floor(px).astype(np.int32)
    y0 = np.floor(py).astype(np.int32)
    x1 = np.minimum(x0 + 1, ww - 1)
    y1 = np.minimum(y0 + 1, hh - 1)
    fx = (px - x0).astype(np.float32)
    fy = (py - y0).astype(np.float32)
    v00 = img[y0, x0]
    v01 = img[y0, x1]
    v10 = img[y1, x0]
    v11 = img[y1, x1]
    return (
        v00 * (1.0 - fx) * (1.0 - fy)
        + v01 * fx * (1.0 - fy)
        + v10 * (1.0 - fx) * fy
        + v11 * fx * fy
    )


def _smooth_anisotropic(
    low_mid: np.ndarray,
    smooth_strength: float,
    downsample_for_large: int = 2048,
) -> np.ndarray:
    """Orientation-aware smoothing of low+mid bands (high band untouched).

    Smooths along the local grain direction via a per-pixel steered 1D
    Gaussian; blends toward the isotropic guided filter where the gradient
    magnitude is too low to trust the orientation. Works entirely in float32.

    Args:
        low_mid: (H, W, 3) float32 BGR in [0, 255].
        smooth_strength: (0–1) smoothing intensity (drives sigma + fallbacks).
        downsample_for_large: Crops larger than this are processed at this
            max dimension for speed, then upsampled.

    Returns:
        (H, W, 3) float32 smoothed low+mid.
    """
    h, w = low_mid.shape[:2]
    min_dim = min(h, w)
    is_ds = downsample_for_large and min_dim > downsample_for_large
    if is_ds:
        scale = float(downsample_for_large) / min_dim
        h2 = max(1, int(round(h * scale)) | 1)
        w2 = max(1, int(round(w * scale)) | 1)
        lm = cv2.resize(low_mid, (w2, h2), interpolation=cv2.INTER_LINEAR)
    else:
        lm = low_mid

    # Explicit colorspace boundary: BGR → grayscale luminance (float32).
    L = cv2.cvtColor(lm, cv2.COLOR_BGR2GRAY)

    sigma_smooth = max(0.5, smooth_strength * 2.0)
    theta, conf = _compute_orientation_field(L, sigma_smooth=sigma_smooth)

    # Grain / edge-tangent direction = perpendicular to gradient.
    ang = theta + np.pi / 2.0
    ux = np.cos(ang)
    uy = np.sin(ang)

    sigma_s = 3.0 + smooth_strength * 12.0  # spatial extent along grain
    K = max(1, int(round(3.0 * sigma_s)))

    hh, ww = lm.shape[:2]
    ys = np.arange(hh, dtype=np.float32)[:, None]
    xs = np.arange(ww, dtype=np.float32)[None, :]

    out = np.zeros_like(lm)
    wsum = np.zeros((hh, ww), dtype=np.float32)
    for t in range(-K, K + 1):
        wt = np.exp(-0.5 * (t / sigma_s) ** 2)
        if wt < 1e-3:
            continue
        px = xs + t * ux
        py = ys + t * uy
        for c in range(lm.shape[2]):
            out[:, :, c] += wt * _bilinear_sample(lm[:, :, c], px, py)
        wsum += wt
    out /= wsum[:, :, None]

    # Isotropic guided fallback for flat / ambiguous areas.
    radius = max(2, int(round(SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR)))
    eps = (SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR) ** 2
    guided = np.zeros_like(lm)
    for c in range(lm.shape[2]):
        guided[:, :, c] = guided_filter(
            lm[:, :, c], radius=radius, eps=eps, guide=None, max_dim=None
        )

    conf_3d = conf[:, :, np.newaxis]
    out = out * conf_3d + guided * (1.0 - conf_3d)

    if is_ds:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float32)


def _directional_wrinkle_2d(
    b: np.ndarray,
    strength: float,
    retention_floor: float,
    sigma: float,
) -> np.ndarray:
    """Per-channel oriented wrinkle attenuation (R13 texture v2 core).

    Wrinkles are *anisotropic* mid-frequency structure (roughly straight
    edges); pores / stray hair are *isotropic* noise. We measure local
    anisotropy from the structure tensor and attenuate only the anisotropic,
    oriented energy — so wrinkles soften while pores and fine texture
    survive. A per-pixel retention floor guarantees at least
    ``retention_floor`` of the original band is always kept (no plastic
    wipe-out of legitimate skin microstructure).
    """
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    s = max(0.5, float(sigma))
    Ixx = cv2.GaussianBlur(gx * gx, (0, 0), s)
    Iyy = cv2.GaussianBlur(gy * gy, (0, 0), s)
    Ixy = cv2.GaussianBlur(gx * gy, (0, 0), s)
    tr = Ixx + Iyy
    det = Ixx * Iyy - Ixy * Ixy
    disc = np.sqrt(np.maximum(tr * tr / 4.0 - det, 0.0))
    l1 = tr / 2.0 + disc
    l2 = tr / 2.0 - disc
    # Anisotropy: 0 = isotropic (pores / noise), 1 = fully oriented
    # (a wrinkle edge). This is the real wrinkle signal — it is
    # amplitude-independent, so isotropic fine texture survives even
    # when its gradient magnitude is large.
    aniso = (l1 - l2) / (tr + 1e-6)
    # Attenuate only STRONGLY-oriented structure (wrinkles). Isotropic
    # noise/pores sit below the knee and pass through untouched.
    t = (aniso - 0.55) / 0.40
    t = np.clip(t, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)  # smoothstep
    max_att = 1.0 - retention_floor
    att = np.clip(strength * t, 0.0, max_att)
    return (b * (1.0 - att)).astype(np.float32)


def directional_wrinkle_attenuate(
    band: np.ndarray,
    strength: float = 0.5,
    retention_floor: float = 0.3,
    sigma: float = 1.0,
) -> np.ndarray:
    """Oriented wrinkle attenuation with a retention floor (R13 texture v2).

    Promotes S5 (wrinkle softening) from "blur" to "pro standard":
    only anisotropic, oriented structure (wrinkles) is attenuated, and
    never by more than ``1 - retention_floor`` at any pixel. Isotropic
    fine texture (pores) is untouched.

    Args:
        band: (H, W) or (H, W, 3) float32 frequency band (e.g. the
            wrinkle-scale layer). Processed per channel when 3D.
        strength: 0–1 attenuation strength.
        retention_floor: minimum fraction of the original band kept at every
            pixel (0.3 = never remove more than 70%).
        sigma: tensor-smoothing sigma (stabilises orientation).

    Returns:
        (H, W) or (H, W, 3) float32 attenuated band.
    """
    if band.ndim == 2:
        return _directional_wrinkle_2d(band, strength, retention_floor, sigma)
    out = np.empty_like(band)
    for c in range(band.shape[2]):
        out[..., c] = _directional_wrinkle_2d(band[..., c], strength, retention_floor, sigma)
    return out


def _guided_smooth(low_mid: np.ndarray, sigma_color: float, sigma_space: float) -> np.ndarray:
    """Self-guided edge-preserving smoothing (per channel), float32 in/out."""
    radius = max(2, int(round(sigma_space)))
    eps = sigma_color ** 2
    smoothed = np.zeros_like(low_mid)
    for c in range(low_mid.shape[2]):
        smoothed[:, :, c] = guided_filter(
            low_mid[:, :, c],
            radius=radius,
            eps=eps,
            guide=None,  # self-guided
            max_dim=None,  # crops are already bounded
        )
    return smoothed


def _region_smooth(
    low_mid: np.ndarray,
    smooth_engine: str,
    smooth_strength: float,
    factor: float,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    """Region-scaled smoothing used by per-region modulation.

    Scales the effective smoothing by ``factor`` for the active engine.
    Only ever processes low+mid; the high band is handled by the caller.

    Args:
        low_mid: (H, W, 3) float32 BGR in [0, 255].
        smooth_engine: "guided", "bilateral" or "anisotropic".
        smooth_strength: Base 0–1 smoothing strength.
        factor: Region modulation factor (already clamped, in [0.5, 2.5]).
        sigma_color: Base sigma (color) for guided/bilateral.
        sigma_space: Base sigma (space) for guided/bilateral.

    Returns:
        (H, W, 3) float32 smoothed low+mid.
    """
    if smooth_engine == "anisotropic":
        return _smooth_anisotropic(low_mid, smooth_strength * factor)
    if smooth_engine == "guided":
        s_color = SIGMA_BASE + smooth_strength * factor * SIGMA_STRENGTH_FACTOR
        s_space = s_color
        return _guided_smooth(low_mid, s_color, s_space)
    # bilateral
    s_color = SIGMA_BASE + smooth_strength * factor * SIGMA_STRENGTH_FACTOR
    s_space = s_color
    return cv2.bilateralFilter(low_mid, -1, s_color, s_space)


class FrequencyLayers:
    """Container for the three frequency bands."""
    __slots__ = ["low", "mid", "high"]

    def __init__(self, low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> None:
        self.low = low    # float32, [0, 255]
        self.mid = mid    # float32, can be negative
        self.high = high  # float32, can be negative

    def reconstruct(self) -> np.ndarray:
        """Return low + mid + high as uint8.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        return np.clip(self.low + self.mid + self.high, 0, 255).astype(np.uint8)


class FrequencySeparator:
    """3-level frequency separation stage.

    Splits an image into three frequency bands:

        Low  — colour / overall tone  (large-scale)
        Mid  — blemishes, wrinkles    (medium-scale)
        High — pores, fine hairs      (small-scale)

    Processing strategy:
        • Smooth the LOW layer to even out skin tone.
        • Reduce the MID layer to remove blemishes.
        • Keep the HIGH layer mostly untouched → natural texture preserved.
        • Recombine: Result = SmoothedLow + ReducedMid + High.

    Key design choice: compositing is done on the **final reconstructed pixel**
    values, not on individual layers. This avoids tonal discontinuities at
    feathered mask boundaries where different layers have been modified
    by different amounts.

    The class is stateless; instances are cheap to create and may be shared
    across threads safely.
    """

    def separate(self, img_bgr: np.ndarray, face_width: float) -> "FrequencyLayers":
        """Split image into low / mid / high frequency layers.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            face_width: Approximate face width in pixels (used to size kernels).
                        Typically inter_eye_distance * 2.5.

        Returns:
            FrequencyLayers with .low, .mid, .high (all float32).
        """
        img_f = img_bgr.astype(np.float32)

        k_low = adaptive_ksize(face_width, factor=0.12, minimum=5)
        k_mid = adaptive_ksize(face_width, factor=0.04, minimum=3)

        low = cv2.GaussianBlur(img_f, (k_low, k_low), 0)
        med_blur = cv2.GaussianBlur(img_f, (k_mid, k_mid), 0)

        mid = med_blur - low
        high = img_f - med_blur

        if os.getenv("FREQ_DEBUG"):
            reconstruction = low + mid + high
            err = np.abs(reconstruction - img_f).max()
            assert err < 1e-3, f"Separation lossy: max_err={err:.4f}"

        return FrequencyLayers(low, mid, high)

    def combine(
        self,
        layers: "FrequencyLayers",
        skin_mask: Optional[np.ndarray] = None,
        smooth_strength: float = 0.5,
        mid_reduction: float = 0.4,
        blotch_reduction: float = 0.0,
        texture_opacity: float = 1.0,
        face_width: Optional[float] = None,
        pore_synthesis: float = 0.0,
        roi_coords: Optional[Tuple[int, int]] = None,
        smooth_engine: str = "guided",
        float32_out: bool = False,
        regions: Optional[Any] = None,
        regional_modulation: float = 0.0,
    ) -> np.ndarray:
        """Re-combine layers after selective processing.

        Compositing is done on final pixel values to avoid tonal discontinuities
        at feathered mask boundaries.

        Args:
            layers: FrequencyLayers from separate().
            skin_mask: (H, W) float mask. Processing applied only here.
            smooth_strength: How much additional smoothing on the low layer (0–1).
            mid_reduction: How much to reduce mid-frequency (0 = keep, 1 = remove).
            texture_opacity: High-frequency reprojection opacity (0–1).
                             Recommended range 0.4–1.0. Lower = smoother/waxier.
            face_width: Face width in pixels (for adaptive mask feathering).
            pore_synthesis: Strength of micro-pore texture synthesis (0-1).
            roi_coords: Tuple of (roi_x1, roi_y1) coordinates for deterministic spatial seeding.
            smooth_engine: Smoothing method: "guided" (guided filter, default),
                "bilateral" (legacy isotropic) or "anisotropic" (orientation-aware,
                smooths along local skin-grain direction). Default "guided".
            float32_out: If True, return float32 [0, 255] instead of uint8. Default False.
            regions: Optional FaceRegions-like object for per-region smoothing
                modulation. Ignored unless ``regional_modulation > 0``.
            regional_modulation: Strength of per-region modulation (0.0 = no-op,
                byte-identical to pre-feature output). Range 0.0–1.0. Default 0.0.

        Returns:
            (H, W, 3) uint8 BGR result, or float32 if float32_out=True.
        """
        if skin_mask is None:
            result = layers.reconstruct()
            if float32_out:
                return result.astype(np.float32)
            return result

        m_raw = skin_mask.astype(np.float32)

        # --- Bounding Box Optimization ---
        # Crop to the mask region to avoid running heavy ops (like bilateral filter)
        # on the entire image when the skin only covers a small fraction.
        ys, xs = np.where(m_raw > 0.01)
        if len(xs) == 0:
            result = layers.reconstruct()
            if float32_out:
                return result.astype(np.float32)
            return result

        fw_approx = face_width if face_width else estimate_face_width(img_shape=layers.low.shape[:2], fallback_ratio=APPROX_FACE_WIDTH_RATIO)
        pad = max(10, int(fw_approx * 0.1))
        x1, x2 = max(0, xs.min() - pad), min(m_raw.shape[1], xs.max() + pad + 1)
        y1, y2 = max(0, ys.min() - pad), min(m_raw.shape[0], ys.max() + pad + 1)

        low = layers.low[y1:y2, x1:x2].copy()
        mid_original = layers.mid[y1:y2, x1:x2]
        mid = mid_original.copy()
        high = layers.high[y1:y2, x1:x2].copy()
        m_raw = m_raw[y1:y2, x1:x2]

        texture_opacity = max(0.0, min(1.0, texture_opacity))

        if face_width:
            feather_r = max(DEFAULT_FEATHER_MIN, int(face_width * DEFAULT_FEATHER_FACTOR) | 1)
        else:
            h_full, w_full = layers.low.shape[:2]
            feather_r = max(DEFAULT_FEATHER_MIN, int(min(h_full, w_full) * FALLBACK_FEATHER_FACTOR) | 1)

        m_2d = cv2.GaussianBlur(m_raw, (feather_r, feather_r), 0)
        m_3d = m_2d[:, :, np.newaxis]

        # --- Adaptive texture preservation ---
        # Flat/front-lit skin has low intrinsic high-band energy, so the fixed,
        # content-blind smoothing parameters strip real pore texture as if it
        # were noise. Measure masked high-band energy and, when low, back off the
        # levers that erode pore-scale structure:
        #   (a) smooth_strength — drives the guided-filter eps/radius on low+mid;
        #       this is the dominant texture-killer, so it is scaled directly.
        #   (b) texture_opacity — floored upward so the high band is fully kept.
        # The response is smooth (no threshold pop), monotonic in energy, and
        # strictly asymmetric: every lever is only ever *reduced* (smoothing
        # never gets stronger than the recipe asks). On normal/high-texture
        # faces adapt == 1.0 → output is byte-for-byte identical to before.
        # (mid_reduction is intentionally NOT scaled: the guided-filter
        # interaction makes the output mid band non-monotonic in mid_reduction,
        # so touching it would not reliably preserve texture.)
        adapt = _texture_adaptation_factor(high, m_2d)
        if adapt < 1.0:
            smooth_strength = smooth_strength * adapt
            # Raise opacity toward 1.0 (keep more of the surviving high band).
            texture_opacity = texture_opacity + (1.0 - adapt) * (1.0 - texture_opacity)
            texture_opacity = max(0.0, min(1.0, texture_opacity))

        # Texture opacity — attenuate high band inside the mask
        if texture_opacity < 1.0:
            high = high * (1.0 - m_3d * (1.0 - texture_opacity))

        # Pore synthesis — inject synthetic high-frequency noise
        if face_width and roi_coords and pore_synthesis > 0:
            roi_x1, roi_y1 = roi_coords
            seed = abs(hash((int(face_width * 100), roi_x1, roi_y1))) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            h, w = low.shape[:2]

            # Generate 2D (1-channel) noise to avoid chroma noise.
            noise = rng.standard_normal((h, w)).astype(np.float32) * 15.0

            sigma = max(0.5, face_width / 120.0)
            noise_blur = cv2.GaussianBlur(noise, (0, 0), sigma)

            P = (noise - noise_blur)[:, :, np.newaxis]
            high = high + (pore_synthesis * P * m_3d)

        # Reduce mid layer
        if mid_reduction > 0:
            mid = mid * (1.0 - m_3d * mid_reduction)

        # --- Dedicated blotch band (R4 — D&B v2: dedicated blotch-band) ---
        # Broad pigment/redness blotches (wavelengths ~13-40px) live in the LOW
        # band, which S2 only *smooths* (blending blotches together) rather than
        # *evens*. We carve a dedicated broad band as blur(k_a) - low, where
        # k_a > k_low. Because both terms pass the lowest frequencies (true
        # form/shading) almost equally, the difference cancels them *by
        # construction* — so removing this band from LOW evens blotches without
        # flattening facial form. Pores (below k_low) are absent from LOW
        # entirely, so they survive untouched. This gives independent control
        # of broad blotch evening (this lever) versus fine blemish reduction
        # (mid_reduction), which until now shared one band. No-op when 0.
        if blotch_reduction > 0 and face_width:
            k_a = adaptive_ksize(face_width, factor=0.32, minimum=17)
            base = low + mid_original + high
            blur_a = cv2.GaussianBlur(base, (k_a, k_a), 0)
            # Band = LOW minus a broader blur → the broad component that the
            # broader blur removed. Pushing LOW toward blur_a evens it.
            blotch_band = layers.low[y1:y2, x1:x2] - blur_a
            low = low - m_3d * blotch_reduction * blotch_band

        # Early exit for no-smoothing case
        if smooth_strength <= 0:
            processed_crop = low + mid + high
            orig_crop = layers.low[y1:y2, x1:x2] + layers.mid[y1:y2, x1:x2] + layers.high[y1:y2, x1:x2]
            result_crop = blend_masked(orig_crop, processed_crop, m_2d)

            if float32_out:
                # E1 float path: keep the full result float32 — no uint8 paste target.
                full_result = np.clip(layers.low + layers.mid + layers.high, 0, 255).astype(np.float32)
                full_result[y1:y2, x1:x2] = np.clip(result_crop, 0, 255)
                return full_result
            full_result = layers.reconstruct()
            full_result[y1:y2, x1:x2] = result_crop
            return full_result

        # Smooth low + mid layers
        f_width = face_width if face_width else fw_approx
        k_smooth = adaptive_ksize(f_width, factor=SMOOTH_K_FACTOR, minimum=SMOOTH_K_MIN)
        smoothed_low_gaussian = cv2.GaussianBlur(low, (k_smooth, k_smooth), 0)

        # Smoothing filter (guided / bilateral / anisotropic)
        # Keep in float32 to avoid quantization banding on gradients (uint8 artifacts).
        low_mid_f32 = np.clip(low + mid_original, 0, 255)
        sigma_color = SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR
        sigma_space = SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR

        if smooth_engine == "anisotropic":
            # Orientation-aware smoothing along local skin-grain direction.
            # Graceful fallback to guided inside _smooth_anisotropic for flat/
            # ambiguous regions, so this never replaces the high band (it only
            # ever receives low+mid) and never crashes on degenerate input.
            try:
                smoothed_f32 = _smooth_anisotropic(low_mid_f32, smooth_strength)
            except cv2.error as e:  # pragma: no cover - defensive
                logger.warning("Anisotropic smoothing failed (%s); falling back to guided.", e)
                smoothed_f32 = _guided_smooth(low_mid_f32, sigma_color, sigma_space)
            smoothed_low_bilateral = smoothed_f32 - mid_original
        elif smooth_engine == "guided":
            smoothed_f32 = _guided_smooth(low_mid_f32, sigma_color, sigma_space)
            smoothed_low_bilateral = smoothed_f32 - mid_original
        else:
            # Bilateral filter (legacy path)
            # Use d=-1 to let OpenCV compute an optimal, efficient kernel size.
            smoothed_f32 = cv2.bilateralFilter(low_mid_f32, -1, sigma_color, sigma_space)
            smoothed_low_bilateral = smoothed_f32 - mid_original

        # Per-region modulation: blend region-scaled smoothing into the base.
        # Strict no-op when regional_modulation == 0.0 or regions is None (or
        # every region factor collapsed to 1.0), preserving byte-identical output.
        if regional_modulation != 0.0 and regions is not None:
            crop = (y1, y2, x1, x2)
            regional_factors = _regional_modulation_factors(
                high, regions, high.shape[:2], regional_modulation, crop=crop
            )
            base_smoothed = smoothed_f32
            for name, factor in regional_factors.items():
                if abs(factor - 1.0) < 1e-6:
                    continue
                region_mask = getattr(regions, name, None)
                if region_mask is None:
                    continue
                rm = _region_mask_crop(region_mask, low_mid_f32.shape[:2], crop)
                sel = rm > 0.01
                if int(np.count_nonzero(sel)) < 16:
                    continue
                # Feather region boundary (allowed: mask feathering, not skin smoothing).
                rm_f = cv2.GaussianBlur(rm, (15, 15), 0)
                rm_3d = rm_f[:, :, np.newaxis]

                scaled = _region_smooth(low_mid_f32, smooth_engine, smooth_strength, factor, sigma_color, sigma_space)
                base_smoothed = base_smoothed * (1.0 - rm_3d) + scaled * rm_3d
            smoothed_low_bilateral = base_smoothed - mid_original

        # Hybrid blend (cv2.addWeighted is slightly faster and purely SIMD optimized)
        blend_gaussian = min(1.0, smooth_strength * GAUSSIAN_BLEND_FACTOR)
        smoothed_low_final = cv2.addWeighted(
            smoothed_low_bilateral, 1.0 - blend_gaussian,
            smoothed_low_gaussian, blend_gaussian,
            0.0
        )

        low = low * (1.0 - m_3d) + smoothed_low_final * m_3d

        # Composite on final pixel values
        processed_crop = low + mid + high
        orig_crop = layers.low[y1:y2, x1:x2] + layers.mid[y1:y2, x1:x2] + layers.high[y1:y2, x1:x2]
        result_crop = blend_masked(orig_crop, processed_crop, m_2d)

        if float32_out:
            # E1 float path: keep the full result float32 — no uint8 paste target.
            full_result = np.clip(layers.low + layers.mid + layers.high, 0, 255).astype(np.float32)
            full_result[y1:y2, x1:x2] = np.clip(result_crop, 0, 255)
            return full_result

        # Paste back into the full image
        full_result = layers.reconstruct()
        full_result[y1:y2, x1:x2] = result_crop
        return full_result


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (DEPRECATED — use FrequencySeparator)
# ---------------------------------------------------------------------------

def separate(img_bgr: np.ndarray, face_width: float) -> "FrequencyLayers":
    """Deprecated. Use ``FrequencySeparator().separate()`` instead."""
    return FrequencySeparator().separate(img_bgr, face_width)


def combine(
    layers: "FrequencyLayers",
    skin_mask: Optional[np.ndarray] = None,
    smooth_strength: float = 0.5,
    mid_reduction: float = 0.4,
    blotch_reduction: float = 0.0,
    texture_opacity: float = 1.0,
    face_width: Optional[float] = None,
    pore_synthesis: float = 0.0,
    roi_coords: Optional[Tuple[int, int]] = None,
    smooth_engine: str = "guided",
    regions: Optional[Any] = None,
    regional_modulation: float = 0.0,
) -> np.ndarray:
    """Deprecated. Use ``FrequencySeparator().combine()`` instead."""
    return FrequencySeparator().combine(
        layers,
        skin_mask=skin_mask,
        smooth_strength=smooth_strength,
        mid_reduction=mid_reduction,
        blotch_reduction=blotch_reduction,
        texture_opacity=texture_opacity,
        face_width=face_width,
        pore_synthesis=pore_synthesis,
        roi_coords=roi_coords,
        smooth_engine=smooth_engine,
        regions=regions,
        regional_modulation=regional_modulation,
    )
