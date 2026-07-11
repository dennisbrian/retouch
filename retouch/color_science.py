"""Oklab color space converters and skin-tone measurement utilities.

Provides vectorized Oklab/OKLCh conversions using the standard Björn Ottosson
matrices, plus skin-tone classification and chroma variance metrics.
"""

from __future__ import annotations

import cv2
from dataclasses import dataclass
from typing import Any, Dict, Optional

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


def _oklab_to_linear_rgb(oklab: np.ndarray) -> np.ndarray:
    """Oklab -> linear RGB (unclipped, float32). Internal helper for gamut tests."""
    lms_cbrt = np.dot(oklab, _OKLAB_M2_INV.T)
    lms = np.power(lms_cbrt, 3.0)
    rgb = np.dot(lms, _OKLAB_M1_INV.T)
    return rgb.astype(np.float32)


def find_gamut_intersection(oklab: np.ndarray, eps: float = 1e-6, max_iter: int = 20) -> np.ndarray:
    """Per-pixel max in-gamut chroma C_max(L, h) in OKLab.

    For each pixel, holding L and hue fixed, bisection searches the largest
    chroma whose OKLab->linear-sRGB conversion stays inside [0, 1]. This is the
    standard Björn Ottosson reference method: monoticity of out-of-gamut-ness
    in chroma makes bisection exact. Upper bound 0.5 covers the full sRGB
    gamut (max sRGB chroma ~0.43), so the bracket is always valid.

    Args:
        oklab: (H, W, 3) float32 Oklab.
        eps: Bisection tolerance on chroma.
        max_iter: Maximum bisection iterations.

    Returns:
        (H, W) float32 C_max per pixel (0.0 for extreme L).
    """
    L = oklab[..., 0]
    a = oklab[..., 1]
    b = oklab[..., 2]
    hue = np.arctan2(b, a)
    cos_h = np.cos(hue).astype(np.float32)
    sin_h = np.sin(hue).astype(np.float32)

    lo = np.zeros_like(L, dtype=np.float32)
    hi = np.full_like(L, 0.5, dtype=np.float32)
    valid = (L > 0.0) & (L < 1.0)

    def _in_gamut(C: np.ndarray) -> np.ndarray:
        cand = np.stack([L, C * cos_h, C * sin_h], axis=-1).astype(np.float32)
        rgb = _oklab_to_linear_rgb(cand)
        return np.all((rgb >= -1e-4) & (rgb <= 1.0 + 1e-4), axis=-1)

    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        ing = _in_gamut(mid)
        lo = np.where(ing & valid, mid, lo)
        hi = np.where((~ing) & valid, mid, hi)
        span = float(np.max(hi - lo))
        if span < eps:
            break
    return lo.astype(np.float32)


def gamut_compress(oklch: np.ndarray, thr: float = 0.8, power: float = 0.6) -> np.ndarray:
    """Gamut-aware chroma compression in OKLCh (preserves hue & lightness).

    For in-gamut colors (C <= C_max) this is the exact identity — no change —
    which keeps the golden path byte-identical. For out-of-gamut colors it
    smoothly rolls chroma back toward the gamut boundary along a constant-hue,
    constant-lightness line (no hue shift, no posterization). ``thr`` shapes how
    aggressively the rolloff bites and ``power`` its softness.

    Args:
        oklch: (H, W, 3) float32 OKLCh (L, C, h).
        thr: Knee sharpness (larger = less compression).
        power: Rolloff softness (larger = softer near the boundary).

    Returns:
        (H, W, 3) float32 OKLCh with compressed chroma.
    """
    L = oklch[..., 0]
    C = oklch[..., 1]
    h = oklch[..., 2]
    oklab = oklch_to_oklab(oklch)
    Cmax = find_gamut_intersection(oklab)
    Cmax_safe = np.where(Cmax > 1e-6, Cmax, 1.0)
    t = C / Cmax_safe
    # In-gamut (t <= 1): exact identity (no hue/L/C change).
    # Out-of-gamut (t > 1): roll the excess chroma back toward the boundary.
    x = np.clip(t - 1.0, 0.0, None)
    denom = 1.0 + (x / max(float(thr), 1e-3)) ** power
    new_x = x * (1.0 / denom) ** (1.0 / power)
    t_new = np.where(t <= 1.0, t, 1.0 + new_x)
    # Clamp so the output is never beyond the gamut boundary (far out-of-gamut
    # colors roll all the way to C_max; near-boundary colors get the soft knee).
    C_new = np.clip(Cmax_safe * t_new, 0.0, Cmax_safe)
    return np.stack([L, C_new, h], axis=-1).astype(np.float32)


def apply_subtractive_saturation(img_bgr: np.ndarray, amount: float) -> np.ndarray:
    """Film-density (subtractive) saturation in perceptual OKLCh space.

    Boosts chroma while *darkening in proportion to chroma* -- the defining
    trait of subtractive (CMY dye) saturation, where denser dye layers both
    intensify colour and absorb more light. The work is done in OKLCh so that:

    * **Hue is locked.** Only C (chroma) and L (lightness) move; ``h`` is
      carried through untouched, so there is no hue slide toward whichever RGB
      channel happens to be brightest (the earlier log-density implementation
      pinned that channel and turned warm skin / neutral backgrounds green).
    * **Neutrals are a fixed point.** At C == 0 the lightness term vanishes, so
      grays are returned unchanged.

    Chroma scales by ``1 + amount``; lightness drops by a fraction proportional
    to ``amount * C`` (positive amount = the film look: saturated colours gain
    density and darken; negative amount desaturates and lightens toward gray).
    ``amount == 0`` is byte-identical to the input.

    Args:
        img_bgr: (H, W, 3) uint8 BGR or float32 BGR in [0, 1].
        amount: Saturation amount in roughly [-1, 1] (0 = identity).

    Returns:
        Same dtype/range as ``img_bgr``.
    """
    if amount == 0:
        return img_bgr
    is_float = img_bgr.dtype == np.float32
    # bgr_to_oklab expects float BGR in [0, 255] or uint8 BGR.
    if is_float:
        work = np.clip(img_bgr, 0.0, 1.0).astype(np.float32) * 255.0
    else:
        work = img_bgr

    oklab = bgr_to_oklab(work)
    oklch = oklab_to_oklch(oklab)
    L = oklch[..., 0]
    C = oklch[..., 1]

    amt = float(amount)
    # Chroma boost (hue held fixed by construction -- h is untouched).
    C_new = np.maximum(C * (1.0 + amt), 0.0)
    # Subtractive density: darken (positive amount) or lighten (negative) in
    # proportion to chroma, so neutrals (C == 0) are unaffected. The 0.5 gain
    # keeps the lightness move gentle relative to the chroma move.
    L_new = np.clip(L - amt * C * 0.5, 0.0, 1.0)

    oklch_new = np.stack([L_new, C_new, oklch[..., 2]], axis=-1).astype(np.float32)
    bgr = oklab_to_bgr(oklch_to_oklab(oklch_new), float32_out=True)  # [0, 255] float

    # Near-neutral pixels have numerically unstable hue and carry a small
    # achromatic residual (uint8 grays land at C up to ~0.046 in OKLCh purely
    # from quantization). Blend the transform back toward the source over a
    # soft chroma ramp so true grays / near-neutral skin stay put while
    # genuinely coloured pixels (C > ~0.09) are fully transformed. This keeps
    # neutrals a fixed point without freezing low-chroma real colours.
    C_LO, C_HI = 0.05, 0.09
    w = np.clip((C - C_LO) / (C_HI - C_LO), 0.0, 1.0)[..., None]
    if is_float:
        src = np.clip(img_bgr, 0.0, 1.0).astype(np.float32) * 255.0
    else:
        src = img_bgr.astype(np.float32)
    bgr = src * (1.0 - w) + bgr * w

    if is_float:
        return np.ascontiguousarray(np.clip(bgr / 255.0, 0.0, 1.0).astype(np.float32))
    return np.clip(np.round(bgr), 0, 255).astype(np.uint8)


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


def ita_value(lab_patch: np.ndarray) -> float:
    """Individual Typology Angle (ITA) of a CIELAB skin patch, in DEGREES.

    ITA = arctan((L* - 50) / b*) measured per pixel, then averaged over the
    patch. Pixels with near-zero b* are excluded from the mean (the angle is
    numerically unstable there and carries no useful skin-tone information).

    Args:
        lab_patch: (H, W, 3) float32 CIELAB, L* in [0, 100], a*/b* in [-128, 128].

    Returns:
        Mean ITA in degrees (float). Positive = lighter skin, negative = darker.
    """
    lab = np.asarray(lab_patch, dtype=np.float32)
    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError("lab_patch must be (H, W, 3) CIELAB")
    L = lab[..., 0].astype(np.float32)
    b = lab[..., 2].astype(np.float32)
    valid = np.abs(b) >= 1e-3
    if not np.any(valid):
        return 0.0
    ita = np.degrees(np.arctan((L[valid] - 50.0) / b[valid]))
    return float(np.mean(ita))


_FITZ_TONE: Dict[int, Dict[str, Any]] = {
    1: {"smooth_scale": 1.00, "highlight_scale": 1.10, "locus_target": 45.0},
    2: {"smooth_scale": 0.95, "highlight_scale": 1.05, "locus_target": 48.0},
    3: {"smooth_scale": 0.90, "highlight_scale": 1.00, "locus_target": 52.0},
    4: {"smooth_scale": 0.80, "highlight_scale": 0.95, "locus_target": 55.0},
    5: {"smooth_scale": 0.70, "highlight_scale": 0.90, "locus_target": 58.0},
    6: {"smooth_scale": 0.60, "highlight_scale": 0.85, "locus_target": None},
}


def classify_fitzpatrick(bgr_patch: np.ndarray) -> Dict[str, Any]:
    """Auto-classify a skin patch into a Fitzpatrick phototype via ITA.

    Input: skin patch as (H, W, 3) uint8 BGR OR float32 BGR in [0, 255].
    Converts BGR -> RGB -> CIELAB (float32) via cv2, computes mean L*, mean b*,
    the patch ITA (see :func:`ita_value`), and maps it to a Fitzpatrick type
    using the standard Chardon ITA ranges.

    Args:
        bgr_patch: (H, W, 3) uint8 BGR or float32 BGR in [0, 255].

    Returns:
        Dict with keys: ``ita`` (float), ``type_index`` (1-6),
        ``label`` ("I".."VI"), ``L_star`` (float), ``b_star`` (float).
    """
    img = np.asarray(bgr_patch)
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError("bgr_patch must be (H, W, 3) BGR")
    if img.dtype != np.uint8:
        rgb = np.clip(img.astype(np.float32), 0.0, 255.0)[..., ::-1] / 255.0
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    else:
        lab = cv2.cvtColor(img[..., ::-1], cv2.COLOR_RGB2LAB).astype(np.float32)

    L_mean = float(np.mean(lab[..., 0]))
    b_mean = float(np.mean(lab[..., 2]))
    ita = ita_value(lab)

    if ita > 55:
        idx, label = 1, "I"
    elif ita > 41:
        idx, label = 2, "II"
    elif ita > 28:
        idx, label = 3, "III"
    elif ita > 10:
        idx, label = 4, "IV"
    elif ita > -30:
        idx, label = 5, "V"
    else:
        idx, label = 6, "VI"

    return {
        "ita": ita,
        "type_index": idx,
        "label": label,
        "L_star": L_mean,
        "b_star": b_mean,
    }


def tone_adaptation_params(type_index: int) -> Dict[str, float]:
    """Recommended relative scales per Fitzpatrick type for auto-adaptation.

    Seed table for the R14 "ITA/Fitzpatrick auto-adaptation" behavior. Darker
    types receive slightly lower default smoothing and a warmer/neutral locus
    target (``locus_target`` is a hue in degrees, or ``None`` for the darkest
    type where a hue target is unstable). Full engine wiring is a documented
    follow-up, not done here.

    Args:
        type_index: Fitzpatrick type 1-6.

    Returns:
        Dict with keys ``smooth_scale``, ``highlight_scale``, ``locus_target``.
    """
    if type_index not in _FITZ_TONE:
        raise ValueError("type_index must be in 1..6")
    return dict(_FITZ_TONE[type_index])
