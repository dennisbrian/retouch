"""R10 Chromophore suite + R11 Cosplay skin moat.

Operators built on R7 chromophore decomposition (``retouch.chromophore``):
``decompose_chromophores`` and ``hemoglobin_breakdown_phase``. All pixel math
is float32 (no uint8 intermediates). BGR is the input convention and the guide
maps (melanin / hemoglobin) are float32 in [0, ~1]. Every function returns
BGR float32 in [0, 255]. ``strength == 0`` is an exact no-op; an empty / None
mask returns the image unchanged.

owl: this module is owned by the R10/R11 agent. ``skin.py`` will call into
these functions later; the recommended call sites are listed in the module
docstring footer. Do NOT edit ``chromophore.py`` (R7 agent owns it).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .chromophore import (
    decompose_chromophores,
    hemoglobin_breakdown_phase,
)
from .utils import (
    adaptive_ksize,
    blend_masked,
    bgr_f32_to_lab_f32,
    estimate_face_width,
    guided_filter,
    lab_f32_to_bgr_f32,
)

# Recolor directions in RGB, per unit chromophore concentration, copied from
# ``chromophore.py`` extinction vectors (_K_MELANIN / _K_HEMOGLOBIN rows are
# (R, G, B)). Sign convention: +chromophore => the listed RGB change in LOG
# space, so lowering a chromophore moves rgb opposite this vector. Used only
# for the documented-channel direction notes; the operators below prefer the
# physically robust "pull toward local healthy skin" form.
# ponytail: import _M from chromophore.py once it is exposed (no leading _).
_MELANIN_RGB_DIR = np.array([0.10, 0.20, 0.30], dtype=np.float32)
_HEMOGLOBIN_RGB_DIR = np.array([-0.50, 0.25, 0.25], dtype=np.float32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _as_f32(img_bgr: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 [0, 255] copy; uint8 inputs are kept in range."""
    if img_bgr.dtype == np.float32:
        return np.ascontiguousarray(img_bgr)
    return img_bgr.astype(np.float32)


def _prep_mask(mask: Optional[np.ndarray], h: int, w: int) -> Optional[np.ndarray]:
    """Normalize a mask to float32 (H, W) in [0, 1]; return None if empty."""
    if mask is None:
        return None
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m.reshape(h, w)
    if m.max() > 1.0:
        m = m / 255.0
    if m.max() <= 1e-6:
        return None
    return m


def _lowpass(arr: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian low-pass with an odd kernel sized from sigma."""
    if sigma <= 0:
        return arr.copy()
    ksize = max(3, int(round(2.0 * sigma) + 1) | 1)
    return cv2.GaussianBlur(arr, (ksize, ksize), sigma)


def _face_width(img_bgr: np.ndarray, mask: Optional[np.ndarray]) -> int:
    if mask is not None:
        return estimate_face_width(skin_mask=mask, img_shape=img_bgr.shape[:2])
    return estimate_face_width(img_shape=img_bgr.shape[:2])


def _pull_toward_healthy(
    img_bgr: np.ndarray,
    chromo: np.ndarray,
    mask: Optional[np.ndarray],
    strength: float,
    sigma: float,
    mode: str = "excess",
) -> np.ndarray:
    """Pull img toward the local low-pass (healthy) colour, weighted by chromo
    deviation. ``mode='excess'`` only acts where chromo is *above* its low-pass
    (blemishes / veins / ingrown dots); ``mode='even'`` flattens both tails
    (tan lines / self-tanner streaks)."""
    lp_ch = _lowpass(chromo, sigma)
    if mode == "even":
        dev = np.abs(chromo - lp_ch)
    else:
        dev = np.clip(chromo - lp_ch, 0.0, None)
    mx = dev.max()
    weight = (dev / mx) if mx > 1e-6 else dev
    if mask is not None:
        weight = weight * mask
    healthy = _lowpass(img_bgr, sigma)
    out = img_bgr + strength * weight[:, :, np.newaxis] * (healthy - img_bgr)
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def _compact_spots(
    chmap: np.ndarray,
    mask: Optional[np.ndarray],
    area_max: int,
    top_frac: float = 0.85,
) -> np.ndarray:
    """Mark compact (small-area) connected components that are high-percentile
    spikes of ``chmap`` within ``mask``. Returns uint8 [0, 1]."""
    h, w = chmap.shape[:2]
    if mask is None:
        binmask = np.ones((h, w), np.uint8)
        vals = chmap.ravel()
    else:
        binmask = (mask > 0.5).astype(np.uint8)
        vals = chmap[binmask > 0]
    if vals.size == 0:
        return np.zeros((h, w), np.uint8)
    if vals.max() <= 0.0:
        return np.zeros((h, w), np.uint8)
    thresh = float(np.quantile(vals, top_frac))
    candidates = (chmap > thresh) & (binmask == 1)
    if not candidates.any():
        return np.zeros((h, w), np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidates.astype(np.uint8), 8
    )
    out = np.zeros((h, w), np.uint8)
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] <= area_max:
            out[labels == i] = 1
    return out


# ---------------------------------------------------------------------------
# R10 — Chromophore suite
# ---------------------------------------------------------------------------

def blemish_vs_mole(
    hemoglobin: np.ndarray,
    melanin: np.ndarray,
    mask: Optional[np.ndarray] = None,
    area_max: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split compact skin spots into hemoglobin blemishes vs melanin moles.

    Returns (blemish_mask, mole_mask) as uint8 [0, 1] (H, W). Compact = connected
    component area below ``area_max``.
    """
    h, w = hemoglobin.shape[:2]
    m = _prep_mask(mask, h, w)
    if area_max is None:
        area_max = max(30, int((min(h, w) / 6) ** 2))
    blemish = _compact_spots(hemoglobin, m, area_max)
    mole = _compact_spots(melanin, m, area_max)
    return blemish, mole


def bruise_remove(
    img_bgr: np.ndarray,
    hemoglobin: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.5,
) -> np.ndarray:
    """Even healing (green/yellow phase) bruises toward local healthy skin.

    Uses ``hemoglobin_breakdown_phase`` to isolate green/yellow breakdown
    (not fresh red blemishes), then low-passes the hemoglobin map and pulls the
    bruise pixels toward the local low-pass (healthy) colour.
    """
    if strength == 0:
        return img_bgr
    img = _as_f32(img_bgr)
    h, w = img.shape[:2]
    m = _prep_mask(mask, h, w)

    try:
        # hemoglobin_breakdown_phase expects a 0..180 hue map (uint8 convention);
        # compute it explicitly from a uint8 snapshot so the float32 input does
        # not push hue into the 0..360 float range.
        img_u8 = np.clip(img, 0.0, 255.0).astype(np.uint8)
        hue_u8 = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV)[:, :, 0]
        phase = hemoglobin_breakdown_phase(hemoglobin, hue_u8)
    except (cv2.error, ValueError) as exc:
        import logging
        logging.getLogger(__name__).warning("bruise phase failed: %s", exc)
        return img
    if phase.shape != (h, w):
        phase = cv2.resize(phase, (w, h), interpolation=cv2.INTER_NEAREST)
    bruise = (phase == 1) | (phase == 2)
    if m is not None:
        bruise = bruise & (m > 0.5)
    if not bruise.any():
        return img

    fw = _face_width(img, m)
    sigma = max(6.0, fw * 0.15)
    # Healing bruises are largely a HUE anomaly (near-zero hemoglobin in the
    # decomposition), so weight by the detected bruise region itself and pull
    # its colour toward the local low-pass (healthy) skin. Where hemoglobin is
    # also elevated we bias harder toward the low-pass hemoglobin value.
    lp_hb = _lowpass(hemoglobin, sigma)
    excess = np.clip(hemoglobin - lp_hb, 0.0, None)
    mx = excess.max()
    excess_w = (excess / mx) if mx > 1e-6 else excess
    weight = np.clip(bruise.astype(np.float32) + 0.5 * excess_w, 0.0, 1.0)
    healthy = _lowpass(img, sigma)
    out = img + strength * weight[:, :, np.newaxis] * (healthy - img)
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def vein_attenuate(
    img_bgr: np.ndarray,
    hemoglobin: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.5,
) -> np.ndarray:
    """Attenuate subsurface deoxygenated (blue-green) veins.

    Broad low-passes the hemoglobin map and subtracts ``strength * low_freq``
    from the green/blue channels inside ``mask``. Red channel is untouched so
    any sharp feature carried in red is preserved.
    """
    if strength == 0:
        return img_bgr
    img = _as_f32(img_bgr)
    h, w = img.shape[:2]
    m = _prep_mask(mask, h, w)
    if m is None:
        m = np.ones((h, w), np.float32)

    fw = _face_width(img, m)
    sigma = max(10.0, fw * 0.25)
    lf = _lowpass(hemoglobin, sigma)
    gain = 60.0
    sub = strength * lf * gain * m
    out = img.copy()
    out[:, :, 1] = np.clip(out[:, :, 1] - sub, 0.0, 255.0)  # green
    out[:, :, 0] = np.clip(out[:, :, 0] - sub, 0.0, 255.0)  # blue
    return out.astype(np.float32)


def tanline_even(
    img_bgr: np.ndarray,
    melanin: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.5,
) -> np.ndarray:
    """Even pure low-frequency melanin (self-tanner streaks / tan lines)."""
    if strength == 0:
        return img_bgr
    img = _as_f32(img_bgr)
    h, w = img.shape[:2]
    m = _prep_mask(mask, h, w)
    fw = _face_width(img, m)
    sigma = max(12.0, fw * 0.3)
    return _pull_toward_healthy(img, melanin, m, strength, sigma, mode="even")


def hemoglobin_guided_smooth(
    img_bgr: np.ndarray,
    hemoglobin: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.5,
) -> np.ndarray:
    """Luminance smoothing guided by the hemoglobin map.

    The hemoglobin map (low-passed) is the guide, so smoothing never bleeds
    across a freckle / mole boundary where hemoglobin jumps.
    """
    if strength == 0:
        return img_bgr
    img = _as_f32(img_bgr)
    h, w = img.shape[:2]
    m = _prep_mask(mask, h, w)

    fw = _face_width(img, m)
    radius = max(4, adaptive_ksize(fw, 0.08))
    sigma = max(6.0, fw * 0.15)
    guide = _lowpass(hemoglobin, sigma).astype(np.float32)

    lab = bgr_f32_to_lab_f32(img)
    lum = lab[:, :, 0].astype(np.float32)
    try:
        eps = max(1.0, (fw * 0.02) ** 2)
        guided = guided_filter(lum, radius=radius, eps=eps, guide=guide)
    except (cv2.error, ValueError):
        # Flat guide or numerical edge case -> self-guided blur.
        guided = cv2.GaussianBlur(lum, (radius * 2 + 1, radius * 2 + 1), 0)

    blend = guided if m is None else lum + strength * m * (guided - lum)
    lab[:, :, 0] = np.clip(blend, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


# ---------------------------------------------------------------------------
# R11 — Cosplay-specific skin moat (body-skin defects)
# ---------------------------------------------------------------------------

def compression_mark_remove(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.5,
) -> np.ndarray:
    """Lift linear mid-frequency luminance dips (sock/corset/strap lines).

    Band-passes the luminance (medium blur minus broad blur) to isolate the
    line signature; a dip is negative, so lifting = adding ``broad - med``.
    """
    if strength == 0:
        return img_bgr
    img = _as_f32(img_bgr)
    h, w = img.shape[:2]
    m = _prep_mask(mask, h, w)

    fw = _face_width(img, m)
    broad_s = max(12.0, fw * 0.3)
    med_s = max(4.0, fw * 0.08)
    lab = bgr_f32_to_lab_f32(img)
    lum = lab[:, :, 0].astype(np.float32)
    broad = _lowpass(lum, broad_s)
    med = _lowpass(lum, med_s)
    lift = broad - med  # >0 at a dip
    if m is None:
        wmap = lift
    else:
        wmap = lift * m
    lab[:, :, 0] = np.clip(lum + strength * wmap, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


def goosebumps_smooth(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.5,
) -> np.ndarray:
    """Smooth periodic high-frequency bump patterns (cold-studio goosebumps).

    A whole-region FFT magnitude peak at a consistent mid-band radius signals
    periodicity; if periodic, an anisotropic (directional) blur along the bump
    direction is applied. No periodicity -> image returned unchanged (identity).
    """
    if strength == 0:
        return img_bgr
    img = _as_f32(img_bgr)
    h, w = img.shape[:2]
    m = _prep_mask(mask, h, w)

    lab = bgr_f32_to_lab_f32(img)
    lum = lab[:, :, 0].astype(np.float32)
    if m is not None:
        roi = np.where(m > 0.5, lum, lum.mean())
    else:
        roi = lum

    is_periodic, angle = _detect_periodicity(roi)
    if not is_periodic:
        return img

    # Rotate so the bump direction is horizontal, blur along it, rotate back.
    cx, cy = w / 2.0, h / 2.0
    rot = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), 1.0)
    fw = _face_width(img, m)
    k = max(5, adaptive_ksize(fw, 0.1))
    lum_rot = cv2.warpAffine(lum, rot, (w, h), borderMode=cv2.BORDER_REPLICATE)
    lum_rot_blur = cv2.GaussianBlur(lum_rot, (k, 1), 0)  # blur along x (bumps)
    inv = cv2.invertAffineTransform(rot)
    lum_blur = cv2.warpAffine(lum_rot_blur, inv, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if m is None:
        out_lum = lum + strength * (lum_blur - lum)
    else:
        out_lum = lum + strength * m * (lum_blur - lum)
    lab[:, :, 0] = np.clip(out_lum, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


def _detect_periodicity(roi: np.ndarray) -> Tuple[bool, float]:
    """Return (is_periodic, stripe_angle_radians). Whole-region FFT magnitude."""
    h, w = roi.shape[:2]
    if min(h, w) < 8:
        return False, 0.0
    f = np.fft.fft2(roi - roi.mean())
    mag = np.abs(np.fft.fftshift(f))
    cy, cx = h // 2, w // 2
    mag[cy, cx] = 0.0
    pos = np.unravel_index(np.argmax(mag), mag.shape)
    py, px = pos[0] - cy, pos[1] - cx
    dist = float(np.hypot(py, px))
    peak = float(mag[pos])
    mean_val = float(mag[mag > 0].mean()) if mag[mag > 0].size else 0.0
    min_dim = float(min(h, w))
    in_mid_band = dist > 0.04 * min_dim and dist < 0.45 * min_dim
    dominant = peak > 4.0 * mean_val if mean_val > 1e-6 else False
    if not (in_mid_band and dominant):
        return False, 0.0
    angle = np.arctan2(float(py), float(px))
    return True, angle


def beard_shadow_neutralize(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.5,
) -> np.ndarray:
    """Neutralize a blue-green chin cast (crossplay under light makeup).

    Detects B > G and B > R beyond a threshold inside ``mask`` and pulls the
    LAB a/b chroma toward neutral (128) by ``strength``.
    """
    if strength == 0:
        return img_bgr
    img = _as_f32(img_bgr)
    h, w = img.shape[:2]
    m = _prep_mask(mask, h, w)

    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    thr = 8.0
    cast = (b > g + thr) & (b > r + thr)
    if m is not None:
        cast = cast & (m > 0.5)
    if not cast.any():
        return img

    lab = bgr_f32_to_lab_f32(img)
    a = lab[:, :, 1]
    bb = lab[:, :, 2]
    weight = cast.astype(np.float32)
    lab[:, :, 1] = np.clip(a + strength * weight * (128.0 - a), 0.0, 255.0)
    lab[:, :, 2] = np.clip(bb + strength * weight * (128.0 - bb), 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


def facepaint_crack_repair(
    img_bgr: np.ndarray,
    paint_mask: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """Fill dark creases inside a uniform face-paint region.

    ``paint_mask`` is required (uint8 [0, 1]). Cracks = luminance dips vs the
    paint-mask mean; they are inpainted (Telea) and blended by ``strength``.
    """
    if strength == 0:
        return img_bgr
    img = _as_f32(img_bgr)
    h, w = img.shape[:2]
    pm = _prep_mask(paint_mask, h, w)
    if pm is None:
        return img

    lab = bgr_f32_to_lab_f32(img)
    lum = lab[:, :, 0].astype(np.float32)
    mean_l = float(lum[pm > 0.5].mean()) if (pm > 0.5).any() else float(lum.mean())
    crack = (lum < mean_l - 12.0) & (pm > 0.5)
    if not crack.any():
        return img

    crack_u8 = (crack.astype(np.uint8)) * 255
    u8 = np.clip(img, 0, 255).astype(np.uint8)
    try:
        inpainted = cv2.inpaint(u8, crack_u8, 3, cv2.INPAINT_TELEA)
    except cv2.error:
        # Fallback: local mean fill inside the crack mask.
        inpainted = _local_mean_fill(u8, crack_u8)
    delta = inpainted.astype(np.float32) - u8.astype(np.float32)
    out = img + strength * crack_u8.astype(np.float32)[..., np.newaxis] / 255.0 * delta
    return np.clip(out, 0.0, 255.0).astype(np.float32)


def _local_mean_fill(u8: np.ndarray, crack_u8: np.ndarray) -> np.ndarray:
    k = 5
    blur = cv2.GaussianBlur(u8, (k, k), 0)
    out = u8.copy()
    out[crack_u8 > 0] = blur[crack_u8 > 0]
    return out


def paint_coverage_even(
    img_bgr: np.ndarray,
    paint_mask: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """Even patchy foundation / body-paint luminance inside ``paint_mask``."""
    if strength == 0:
        return img_bgr
    img = _as_f32(img_bgr)
    h, w = img.shape[:2]
    pm = _prep_mask(paint_mask, h, w)
    if pm is None:
        return img

    lab = bgr_f32_to_lab_f32(img)
    lum = lab[:, :, 0].astype(np.float32)
    mean_l = float(lum[pm > 0.5].mean()) if (pm > 0.5).any() else float(lum.mean())
    corrected = lum + strength * pm * (mean_l - lum)
    lab[:, :, 0] = np.clip(corrected, 0.0, 255.0)
    return lab_f32_to_bgr_f32(lab)


def ingrown_hair_cleanup(
    img_bgr: np.ndarray,
    hemoglobin: np.ndarray,
    mask: Optional[np.ndarray] = None,
    strength: float = 0.5,
) -> np.ndarray:
    """Gently even small compact hemoglobin dots (ingrown hairs) on body skin."""
    if strength == 0:
        return img_bgr
    img = _as_f32(img_bgr)
    h, w = img.shape[:2]
    m = _prep_mask(mask, h, w)
    if m is None:
        return img

    area_max = max(20, int((min(h, w) / 10) ** 2))
    dots = _compact_spots(hemoglobin, m, area_max)
    if not dots.any():
        return img

    fw = _face_width(img, m)
    sigma = max(6.0, fw * 0.12)
    healthy = _lowpass(img, sigma)
    out = img + strength * dots[:, :, np.newaxis].astype(np.float32) * (healthy - img)
    return np.clip(out, 0.0, 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Recommended skin.py call sites (DO NOT EDIT skin.py — for the caller):
#   - chromophore_decompose() in skin.py preprocessing -> store melanin/hemoglobin
#   - body-skin pipeline: call bruise_remove / vein_attenuate / tanline_even /
#     ingrown_hair_cleanup with the body-skin mask (S1)
#   - cosplay recipe keys: "compression_mark", "goosebumps", "beard_shadow",
#     "facepaint_crack", "paint_coverage" routed to the matching functions above
# ---------------------------------------------------------------------------
