"""Wig shine matte in the wig's own colour (``hair_deglare``, GUI "Wig Shine").

Synthetic cosplay fibre is smooth plastic, so under flash, softboxes or con
hall lighting it throws a glossy band or streaks across the crown that
natural hair doesn't. This op softens that shine back into the wig's own
colour while keeping the strands and the wig's broad lit/shadow shape.

Why it replaced the first version of ``hairwork.deglare_wig``: that one only
acted on pixels 12 L above the median of the *whole* wig that were also
*less saturated* than a quarter of the wig's median chroma. Shine on a
blonde, pink or silver wig keeps most of the fibre's chroma and the lit side
of a wig sits above the whole-wig median, so on Alex's wig photos it moved
at most 0.5 L (blonde bunny wig) and 2.2 L (white wig) at strength 60.

This version follows ``powder_finish`` and measures everything against the
wig itself:

* a local diffuse baseline of L (masked two-pass blur over the hair at
  0.25 x face width, with pixels already standing out dropped from the
  second pass), so the lit side of a wig rises with its baseline and isn't
  treated as shine;
* the excess of L above that baseline, gated by the wig's own fine strand
  spread (robust sigma of L minus a small blur), not a fixed level;
* the gated excess is smoothed at strand scale, so single bright strands
  and the strand texture are left alone and only the band of shine goes;
* the shine is removed as an equal amount of linear light from B, G and R
  (the dichromatic model: gloss off a dielectric is the light's colour
  added on top of the fibre's colour), capped by the pixel's darkest
  channel, so the fibre colour comes back instead of grey being painted in;
* a strand-flow coherence ramp leaves fuzzy, frizzy and parting areas alone;
* pixels far less saturated than the fibre around them (a white bow or lace
  caught in the hair mask) are left alone, since taking light off them
  would only grey them;
* where the wig's edge meets something brighter (a window, a softbox), the
  light on the loose fibres is rim light, so the silhouette is left alone.

No absolute intensity threshold is used, so it behaves the same on pale and
dark wigs, and on lighter and darker skin (``tests/test_wig_shine.py``).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .utils import bgr_f32_to_lab_f32, normalize_mask

# Broad-form baseline scale, fraction of the face width.
_BASE_SIGMA_FW = 0.25
# Fine strand texture scale for the spread measure.
_FINE_SIGMA_FW = 0.008
# Shine estimate smoothing: wide enough to average across a few strands.
_SHINE_SIGMA_FW = 0.012
# Shine gate: ramps in from _GATE_LO to _GATE_HI robust sigmas of the
# strand-averaged excess over the local baseline, measured on this wig.
_GATE_LO = 0.5
_GATE_HI = 1.5
# Floor of the gate unit, in strand sigmas.
_SIG_D_FLOOR = 2.0
# Second baseline pass drops pixels this many strand sigmas above the first.
_KEEP_SIGMAS = 2.0
# At strength 100 at most this share of the shine goes; the rest keeps the
# wig reading as fibre rather than felt.
_MAX_PULL = 1.0
# Backlit-silhouette guard: ramps out from this far inside the hair mask (x face width).
_EDGE_LO_FW = 0.02
_EDGE_HI_FW = 0.06
# Colour guard: pixel saturation / local fibre saturation ramps in over
# this range; below it the pixel isn't gloss on this wig.
_SAT_RATIO_LO = 0.20
_SAT_RATIO_HI = 0.40
# Local fibre saturation below which the wig counts as white/grey (guard off).
_SAT_NEUTRAL_LO = 0.06
_SAT_NEUTRAL_HI = 0.12
# Strand-flow coherence ramp: below _COH_LO the flow is fuzz/parting.
_COH_LO = 0.15
_COH_HI = 0.35


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x / 255.0, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    y = np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (y * 255.0).astype(np.float32)


def _lab_L_to_Y(L: np.ndarray) -> np.ndarray:
    """CIE L* (0-100) to relative luminance Y (0-1)."""
    f = (L + 16.0) / 116.0
    return np.where(f > 6.0 / 29.0, f ** 3, 3.0 * (6.0 / 29.0) ** 2 * (f - 4.0 / 29.0)).astype(np.float32)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; wide sigmas run on a downsampled copy (same result, far faster)."""
    if sigma <= 6.0:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    f = 3.0 / sigma
    h, w = x.shape[:2]
    sw, sh = max(1, int(round(w * f))), max(1, int(round(h * f)))
    small = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 3.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _masked_blur(x: np.ndarray, w: np.ndarray, sigma: float, fallback: float) -> np.ndarray:
    num = _blur(x * w, sigma)
    den = _blur(w, sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), fallback).astype(np.float32)


def matte_wig_shine(
    img_bgr: np.ndarray,
    hair_mask: Optional[np.ndarray],
    strength: float,
    face_width: float,
    coherence: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Soften glossy wig shine into the wig's own colour.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image (face ROI).
        hair_mask: (H, W) hair mask, 0-1 or 0-255. ``None`` or empty
            returns ``img_bgr`` unchanged.
        strength: 0-100. 0 returns ``img_bgr`` unchanged.
        face_width: face width in pixels; every scale derives from it.
        coherence: optional (H, W) strand-flow coherence in [0, 1]; low
            coherence (fuzz, partings) fades the op out.
        exclude_mask: optional (H, W) mask of pixels never to touch
            (eyebrows); it is dilated slightly.

    Returns:
        (H, W, 3) image, same dtype as ``img_bgr``.
    """
    if strength <= 0 or hair_mask is None:
        return img_bgr
    m = normalize_mask(hair_mask)
    if m is None:
        return img_bgr
    m = m.astype(np.float32, copy=False)
    h, w = img_bgr.shape[:2]
    if m.shape != (h, w):
        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    hair = m > 0.5
    if int(hair.sum()) < 200:
        return img_bgr

    fw = max(float(face_width), 20.0)
    s = min(float(strength), 100.0) / 100.0
    img = img_bgr.astype(np.float32, copy=False)
    L = bgr_f32_to_lab_f32(img)[..., 0] * (100.0 / 255.0)

    # Fine strand spread: robust sigma of L minus a small blur, over the hair.
    fine = L - cv2.GaussianBlur(L, (0, 0), max(0.8, _FINE_SIGMA_FW * fw))
    fv = fine[hair]
    sig = float(1.4826 * np.median(np.abs(fv - np.median(fv))))
    sig = max(sig, 0.4)

    # Two-pass local baseline over confident hair.
    wgt = (m > 0.5).astype(np.float32)
    base_sigma = _BASE_SIGMA_FW * fw
    med = float(np.median(L[hair]))
    base1 = _masked_blur(L, wgt, base_sigma, med)
    keep = wgt * ((L - base1) < _KEEP_SIGMAS * sig).astype(np.float32)
    if keep.sum() < 0.2 * wgt.sum():
        keep = wgt
    base = _masked_blur(L, keep, base_sigma, med)

    # Excess over the baseline, averaged across a few strands: single bright
    # strands and strand texture average out, a band of shine doesn't.
    shine_sigma = max(1.0, _SHINE_SIGMA_FW * fw)
    D = _masked_blur(L - base, wgt, shine_sigma, 0.0)
    dv = D[hair]
    d_med = float(np.median(dv))
    D = D - d_med
    sig_d = float(1.4826 * np.median(np.abs(dv - d_med)))
    # Floor at twice the strand spread so an evenly lit wig with no gloss
    # doesn't have its brightest strands read as shine.
    sig_d = max(sig_d, _SIG_D_FLOOR * sig, 0.3)
    gate = np.clip((D - _GATE_LO * sig_d) / ((_GATE_HI - _GATE_LO) * sig_d), 0.0, 1.0)
    gate = gate * gate * (3.0 - 2.0 * gate)
    # Keep a little of the lift so the band fades into the wig rather than
    # leaving a flat stripe.
    shine_L = np.maximum(D - _GATE_LO * sig_d * 0.5, 0.0) * gate * m

    region = m.copy()
    # Leave a backlit silhouette alone: where the wig's edge meets something
    # brighter than the wig (a window, a softbox), the light on the loose
    # fibres there is rim light or backdrop seen through them, and darkening
    # it leaves a dark fringe. Edges against darker surroundings (face,
    # headband, costume) are left in, since crown gloss sits right there.
    dist = cv2.distanceTransform(hair.astype(np.uint8), cv2.DIST_L2, 5)
    near = 1.0 - np.clip((dist - _EDGE_LO_FW * fw) / ((_EDGE_HI_FW - _EDGE_LO_FW) * fw), 0.0, 1.0)
    outside = 1.0 - wgt
    if near.max() > 0 and outside.sum() > 0:
        bg = _masked_blur(L, outside, max(1.0, 0.08 * fw), med)
        brighter = np.clip((bg - base) / (2.0 * sig_d), 0.0, 1.0)
        region *= 1.0 - near * brighter
    if coherence is not None:
        coh = coherence.astype(np.float32, copy=False)
        if coh.shape != (h, w):
            coh = cv2.resize(coh, (w, h), interpolation=cv2.INTER_LINEAR)
        region *= np.clip((coh - _COH_LO) / (_COH_HI - _COH_LO), 0.0, 1.0)
    if exclude_mask is not None:
        eb = normalize_mask(exclude_mask)
        if eb is not None:
            eb = eb.astype(np.float32, copy=False)
            if eb.shape != (h, w):
                eb = cv2.resize(eb, (w, h), interpolation=cv2.INTER_LINEAR)
            eb = cv2.dilate(eb, np.ones((5, 5), np.uint8))
            region *= 1.0 - np.clip(eb, 0.0, 1.0)

    drop_L = _MAX_PULL * s * shine_L * region
    if float(drop_L.max()) < 0.05:
        return img_bgr

    # Remove the shine as neutral linear light, capped by the darkest channel
    # (gloss can't be brighter than what it sits on top of).
    lin = _srgb_to_linear(img)
    Y0 = _lab_L_to_Y(L)
    Y1 = _lab_L_to_Y(np.maximum(L - drop_L, 0.0))
    k = np.maximum(Y0 - Y1, 0.0)
    k = np.minimum(k, 0.97 * lin.min(axis=2))

    # Gloss on a coloured wig keeps part of the fibre's colour. A pixel far
    # less saturated than the fibre around it (a white ribbon, a lace collar
    # or a bow the hair mask caught) isn't gloss: taking neutral light off
    # it would only grey it. Saturation is measured in linear light as
    # 1 - min/max, so it doesn't depend on how light or dark the wig is.
    mx = lin.max(axis=2)
    sat = 1.0 - lin.min(axis=2) / np.maximum(mx, 1e-6)
    sat_loc = _masked_blur(sat, keep, base_sigma, float(np.median(sat[hair])))
    ratio = sat / np.maximum(sat_loc, 1e-6)
    colour_ok = np.clip((ratio - _SAT_RATIO_LO) / (_SAT_RATIO_HI - _SAT_RATIO_LO), 0.0, 1.0)
    # On a white or grey wig there is no fibre colour to compare with.
    neutral_wig = np.clip((_SAT_NEUTRAL_HI - sat_loc) / (_SAT_NEUTRAL_HI - _SAT_NEUTRAL_LO), 0.0, 1.0)
    k = k * np.maximum(colour_ok, neutral_wig)
    out = _linear_to_srgb(lin - k[..., None])
    out = np.where((k > 0)[..., None], out, img)
    if img_bgr.dtype == np.uint8:
        return np.clip(np.round(out), 0, 255).astype(np.uint8)
    return out
