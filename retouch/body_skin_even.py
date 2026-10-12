"""Body Skin Evening: calm blotchy colour on arms, legs, chest and back.

Main's old ``body_equalize`` pulled a*/b* 20% of the way to one whole-body
median and ran CLAHE on L*. CLAHE raises local contrast, so on Alex's two
bunny photos at 100 it made body blotches ~10% stronger and fine texture
~25% stronger instead of evening anything. This module replaces it.

What it does, on body skin only (the caller supplies the mask):

* Colour is evened in chromaticity (a*/L*, b*/L*), which shading and skin
  tone scale out of: a red patch or an uneven tan is a mid-scale deviation of
  chromaticity from the skin around it (between ``FINE_SCALE`` and
  ``BROAD_SCALE`` x face width) and is pulled back toward it.
* Lightness only follows where the colour also deviates (a red patch is
  darker as well as redder; a tan patch is darker and yellower), so plain
  shading, muscle and limb form, which keep the skin's colour, stay as shot.
* Gloss (oiled skin, sheer hosiery sheen) is left alone: a patch that is
  both lighter and paler than the skin around it reads as sheen, not a
  blotch, and lightness lifts above the skin's own blotch spread are kept.
* Pores, body hair, freckles and moles are kept: anything that stands out at
  fine scale is left out of the estimates, and the correction is a smooth
  field added to every pixel, so a mole keeps its contrast to the skin
  around it. Nothing is blurred.
* Tattoos, costume bleed and paint are left out: pixels far from the body
  skin's own typical chromaticity carry no weight, and near-neutral skin
  (white or grey paint, chroma/L* below ``_NEUTRAL_OFF``) is not touched.

No absolute intensity threshold: every gate is a ratio to L* or a multiple of
the skin's own measured spread. The corrections are low frequency, so they
are computed on a copy scaled to ~``_WORK_FACE_WIDTH`` px per face width and
upsampled.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

__all__ = ["even_body_skin", "FINE_SCALE", "BROAD_SCALE"]

#: Gaussian sigma (x face width) below which detail counts as texture
#: (pores, body hair, freckles, small moles) and is never touched.
FINE_SCALE = 0.04
#: Gaussian sigma (x face width) of the "skin around it" reference. Blotches
#: up to about a hand's width are evened; broader colour changes stay.
BROAD_SCALE = 0.6
#: Share of a colour-coupled lightness blotch removed at strength 100.
L_SHARE = 0.7
#: Face width (px) the corrections are computed at.
_WORK_FACE_WIDTH = 120.0
#: Chroma/L* below which skin reads as white or grey paint (off) and above
#: which it counts as skin colour (on); same gate as skin_warmth.
_NEUTRAL_OFF, _NEUTRAL_ON = 0.05, 0.08
#: Fewest confident body-skin pixels (at work size) worth evening.
_MIN_PX = 200
_TAU_C = 3.0


def _masked_blur(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of ``x`` using only weighted pixels (normalized convolution)."""
    den = cv2.GaussianBlur(w, (0, 0), sigmaX=sigma)
    if x.ndim == 3:
        num = cv2.GaussianBlur(x * w[..., None], (0, 0), sigmaX=sigma)
        return num / np.maximum(den, 1e-6)[..., None]
    return cv2.GaussianBlur(x * w, (0, 0), sigmaX=sigma) / np.maximum(den, 1e-6)


def _spread(x: np.ndarray) -> float:
    """Robust standard deviation (1.4826 x MAD)."""
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x)))) * 1.4826


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _corrections(lab: np.ndarray, mask: np.ndarray, strength: float, fw: float):
    """(dL, a*/b* gain, dab) fields at the resolution of ``lab``.

    Unmasked fields (the caller weights them by the full-resolution mask).
    """
    L = lab[..., 0]
    inside = mask > 0.5
    if int(inside.sum()) < _MIN_PX:
        return None
    L_med = float(np.median(L[inside]))
    # Chromaticity; deep shadow (L* under a fifth of the skin's own median)
    # has no reliable colour and carries no weight.
    Ls = np.maximum(L, 1e-3)
    c = lab[..., 1:] / Ls[..., None]
    lit = _ramp(L, 0.15 * L_med, 0.25 * L_med)
    s_fine = max(1.0, FINE_SCALE * fw)
    s_broad = max(3.0, BROAD_SCALE * fw)
    # Paint gate on the fine-scale colour, so per-pixel noise can't speckle
    # the correction field.
    c_soft = _masked_blur(c.astype(np.float32), np.maximum(mask * lit, 1e-4), s_fine)
    paint_on = _ramp(np.hypot(c_soft[..., 0], c_soft[..., 1]), _NEUTRAL_OFF, _NEUTRAL_ON)
    apply = mask * paint_on

    # Texture out: pixels that stand out from their fine neighbourhood
    # (moles, freckles, hairs, fishnet threads) don't vote.
    w = apply * lit
    d_fine = L - _masked_blur(L, np.maximum(w, 1e-4), s_fine * 0.5)
    sig_fine = max(_spread(d_fine[inside]), 1e-3)
    w = w * np.exp(-0.5 * (d_fine / (2.5 * sig_fine)) ** 2)
    # Not skin-coloured (tattoo, costume bleed): far from the body skin's own
    # typical chromaticity, measured against its own spread.
    sel = inside & (w > 0.3)
    if int(sel.sum()) < _MIN_PX:
        return None
    c0 = np.median(c[sel], axis=0)
    dist = np.hypot(c[..., 0] - c0[0], c[..., 1] - c0[1])
    tau_c = max(_TAU_C * _spread(dist[sel]), 0.25 * float(np.hypot(*c0)), 1e-3)
    w = (w * np.exp(-0.5 * (dist / tau_c) ** 2)).astype(np.float32)

    lo = _masked_blur(np.dstack([L, c]).astype(np.float32), w, s_fine)
    # Where few pixels vote (deep shadow, a fishnet's dark threads, the mask
    # edge) the local estimate is noise divided by noise: fade it out.
    conf = _ramp(cv2.GaussianBlur(w, (0, 0), sigmaX=s_fine), 0.05, 0.3)
    hi = _masked_blur(np.dstack([L, c]).astype(np.float32), w, s_broad)
    dev = lo - hi
    dev_L, dev_c = dev[..., 0], dev[..., 1:]
    dev_cm = np.hypot(dev_c[..., 0], dev_c[..., 1])

    sel = inside & (w > 0.2)
    # Floors: half a delta-E of a*/b* at the skin's own lightness, and half
    # an L* unit, so smooth skin's near-zero spread can't turn rounding noise
    # into "colour deviates".
    sp_c = max(_spread(dev_cm[sel]), 0.5 / max(L_med, 1.0))
    sp_L = max(_spread(dev_L[sel]), 0.5)
    # Colour: pull the mid-scale chromaticity deviation back toward the skin
    # around it.
    # Gloss (oiled skin, sheer hosiery sheen) is lighter AND paler than the
    # skin around it; redness and tan are the opposite (more colour, darker).
    # Where a patch is both lighter and paler it is left as shot.
    c_hi = hi[..., 1:]
    c_dir = c_hi / np.maximum(np.hypot(c_hi[..., 0], c_hi[..., 1]), 1e-6)[..., None]
    radial = (dev_c * c_dir).sum(axis=-1)
    gloss = _ramp(dev_L / sp_L, 0.5, 1.5) * _ramp(-radial / sp_c, 0.0, 1.0)
    keep = paint_on * conf * (1.0 - gloss)
    dc = -strength * keep[..., None] * dev_c
    # Lightness: only where colour also deviates, never on bright gloss.
    coupled = _ramp(dev_cm, 0.5 * sp_c, 2.0 * sp_c)
    gloss_keep = np.exp(-0.5 * (np.maximum(dev_L, 0.0) / (1.5 * sp_L)) ** 2)
    dL = -strength * L_SHARE * keep * coupled * gloss_keep * dev_L
    # In Lab terms against the local (fine-scale) lightness, so the shift is
    # one smooth offset per neighbourhood: scaling it by each pixel's own L*
    # would print the texture's lightness pattern into a*/b* as colour grain.
    # ``gain`` keeps chromaticity where L* moves.
    L_loc = np.maximum(lo[..., 0], 1e-3)
    gain = (L_loc + dL) / L_loc
    dab = dc * (L_loc + dL)[..., None]
    return dL.astype(np.float32), gain.astype(np.float32), dab.astype(np.float32)


def even_body_skin(
    img: np.ndarray,
    mask: np.ndarray,
    strength: float,
    face_width: Optional[float] = None,
) -> np.ndarray:
    """Even blotchy colour on body skin.

    Args:
        img: (H, W, 3) BGR, float32 in [0, 1] or uint8.
        mask: (H, W) float [0, 1] body-skin mask (face, hair, lips and
            costume already left out by the caller).
        strength: 0-1 (slider / 100).
        face_width: largest face's width in px, which sets the blotch scales;
            defaults to 12% of the longer image side.

    Returns:
        Same shape and dtype as ``img``; pixels outside ``mask`` unchanged.
    """
    if strength <= 0 or mask is None:
        return img
    m = mask[..., 0] if mask.ndim == 3 else mask
    m = m.astype(np.float32, copy=False)
    # Full-resolution work is kept to the body-skin bounding box: at 26 MP,
    # whole-frame float copies and Lab round trips cost several seconds.
    support = m > 1e-3
    rows = np.flatnonzero(support.any(axis=1))
    if rows.size == 0 or float(m.max()) < 0.01:
        return img
    cols = np.flatnonzero(support.any(axis=0))
    y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    u8 = img.dtype == np.uint8
    norm = 255.0 if u8 else 1.0
    h, w = img.shape[:2]
    fw = float(face_width) if face_width and face_width > 0 else 0.12 * max(h, w)
    fw = max(fw, 40.0)

    scale = min(1.0, _WORK_FACE_WIDTH / fw)
    if scale < 1.0:
        size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
        f_s = cv2.resize(img, size, interpolation=cv2.INTER_AREA).astype(np.float32) / norm
        m_s = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    else:
        f_s, m_s = img.astype(np.float32) / norm, m
    lab_s = cv2.cvtColor(np.clip(f_s, 0.0, 1.0), cv2.COLOR_BGR2Lab)
    corr = _corrections(lab_s, m_s, float(np.clip(strength, 0.0, 1.0)), fw * scale)
    if corr is None:
        return img
    dL, gain, dab = corr
    # Weight the fields by the mask smoothed at the texture scale (the
    # guided-filter mask follows fine detail such as fishnet or hosiery weave
    # and would print it into the shift).
    m_w = cv2.GaussianBlur(m_s, (0, 0), sigmaX=max(1.0, FINE_SCALE * fw * scale * 0.5))
    fields = np.dstack([dL * m_w, (gain - 1.0) * m_w, dab * m_w[..., None]]).astype(np.float32)
    if scale < 1.0:
        # Upsample only the box. Crop pixel (u, v) is full-res (x0 + u,
        # y0 + v); same pixel-centre mapping as cv2.resize's bilinear.
        sy, sx = fields.shape[0] / h, fields.shape[1] / w
        mx = np.float32([[sx, 0.0, (x0 + 0.5) * sx - 0.5],
                         [0.0, sy, (y0 + 0.5) * sy - 0.5]])
        fields = cv2.warpAffine(
            fields, mx, (x1 - x0, y1 - y0),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        fields = fields[y0:y1, x0:x1]
    sup = support[y0:y1, x0:x1]
    crop = img[y0:y1, x0:x1].astype(np.float32) / norm
    np.clip(crop, 0.0, 1.0, out=crop)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    lab[..., 0] += fields[..., 0]
    np.clip(lab[..., 0], 0.0, 100.0, out=lab[..., 0])
    lab[..., 1:] *= 1.0 + fields[..., 1:2]
    lab[..., 1:] += fields[..., 2:4]
    new = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
    np.clip(new, 0.0, 1.0, out=new)
    if u8:
        new = np.clip(new * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        new = new.astype(img.dtype, copy=False)
    out = img.copy()
    out[y0:y1, x0:x1][sup] = new[sup]
    return out
