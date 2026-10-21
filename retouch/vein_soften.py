"""Vein Softening: calm visible blue-green veins on skin.

Main's old ``vein_attenuate`` (``skin_chromophore.vein_attenuate``) never
looked for veins: it blurred the whole hemoglobin map and subtracted 60x it
from green and blue across all face skin. At the 0.12 that
``vascular_refine_v1`` set, that painted the face orange (up to 61 levels on
the golden studio portrait) and left a vein's own colour as it was. This
module replaces it.

What counts as a vein, on the caller's skin mask only:

* **Cool:** bluer (lower b*) and a little less red (lower a*) than the skin
  around it, measured in chromaticity (a*/L*, b*/L*), so shading and skin
  tone scale out, and in units of that skin's own fine colour spread.
* **Line-shaped:** a line of that coolness 0.6-2% of a face width wide,
  found with an oriented matched filter (integrating along the line lifts a
  faint vein out of skin noise), with the side of a step (a fill-light or
  shadow boundary) cut, and kept only as part of a long, thin connected
  run. Broad cool areas (under-eye shadow, blue stage light, eyeshadow
  wash), specks and round spots are left alone.
* **Faint and colour-led:** veins sit under the skin, so their contrast is
  low and mostly colour. A line far stronger than the skin's own variation
  (tattoo, liner, a blue strap or strand the mask let through), much darker
  than the skin beside it, darker more than it is cooler (a crease or fold
  shadow) or lighter (sheen) is kept. Lines are only judged well inside
  skin-coloured skin, since a strap's or costume edge's blend reads cool.
* **Not paint:** near-neutral skin (white or grey paint, chroma/L* below
  ``_NEUTRAL_OFF``) or skin whose hue is outside natural skin's
  (``_HUE_ON``; bluish-white or coloured paint) is left alone, as in Body
  Skin Evening and Highlight Repair.

The correction fills the vein's chromaticity from the skin right beside it
(the veins themselves left out) at the pixel's own lightness and returns its
lightness dip. It is computed from a lightly smoothed copy, and pores live
in lightness, which only gets the smooth dip back, so texture is kept. No absolute
intensity threshold: every gate is a ratio to L* or a multiple of the skin's
own measured spread. Work happens on a copy scaled to ~``_WORK_FACE_WIDTH``
px per face width inside the mask's bounding box.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

__all__ = ["soften_veins", "vein_weight"]

#: Face width (px) the detection runs at; a temple vein (~1-3% of a face
#: width) is then 3-9 px wide.
_WORK_FACE_WIDTH = 300.0
#: Line widths (Gaussian sigma) tried, as a share of the face width.
_SCALES = (0.006, 0.011, 0.02)
#: "Skin around it" reference, as a share of the face width.
_REF_SCALE = 0.07
#: Scale of the colour the vein is filled with, as a share of the face width.
_FILL_SCALE = 0.035
#: Direction of a vein's colour shift in (a*, b*): mostly bluer, a little
#: less red.
_VEIN_DIR = np.array([-0.35, -0.94], np.float32)
#: Line coolness (in units of the skin's own fine spread) where a line starts
#: and is fully taken as a vein; and where a line is too strong to be one.
_Z_ON = (0.8, 2.0)
_Z_CAP = (7.0, 10.0)
#: Length of the matched filter along the line, in line widths, and the
#: number of directions tried.
_ALONG = 4.0
_ORIENTATIONS = 12
#: Lightness dip (share of the local skin L*) above which a line is too dark
#: to be a vein.
_DARK_CAP = (0.07, 0.14)
#: Colour shift (delta-E along the vein direction) per L* unit of dip where a
#: line counts as colour-led.
_RATIO_ON = (0.4, 0.8)
#: Share of the lightness dip given back at strength 1.
_L_SHARE = 1.0
_NEUTRAL_OFF, _NEUTRAL_ON = 0.05, 0.08
#: CIELab hue (degrees) of skin of any tone, with soft edges, wide enough for
#: a cool studio cast (the bunny photos' body skin reads -30..-2); paint
#: outside it (their bluish-white face paint reads -54..-75) is left alone.
_HUE_ON = (-45.0, -35.0, 80.0, 90.0)
_MIN_PX = 200
#: Slope-to-curvature ratio (both scale-normalised) where a ridge starts to
#: read as the side of a step, and where it is wholly one.
_STEP_OFF = (0.3, 0.6)
#: Detections above this weight form the components judged for shape.
_COMPONENT_ON = 0.03
#: A component must reach this score somewhere to count.
_SEED_ON = 0.3
#: Shortest vein kept, as a share of the face width, and the least
#: length-to-mean-width ratio.
_MIN_LENGTH = 0.12
_MIN_ELONGATION = 4.0


def _ramp(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _hue_on(a, b):
    """1 inside natural skin hues, 0 outside, soft edges (``_HUE_ON``)."""
    h = np.degrees(np.arctan2(b, a))
    lo0, lo1, hi0, hi1 = _HUE_ON
    return _ramp(h, lo0, lo1) * (1.0 - _ramp(h, hi0, hi1))


def _spread(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.median(np.abs(x - np.median(x)))) * 1.4826


def _masked_blur(x: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    den = cv2.GaussianBlur(w, (0, 0), sigmaX=sigma)
    if x.ndim == 3:
        num = cv2.GaussianBlur(x * w[..., None], (0, 0), sigmaX=sigma)
        return num / np.maximum(den, 1e-6)[..., None]
    return cv2.GaussianBlur(x * w, (0, 0), sigmaX=sigma) / np.maximum(den, 1e-6)


_KERNELS = {}


def _line_kernels(sigma: float, length: float, n: int):
    """Oriented matched line and step kernels at one scale (cached).

    The line kernel is a negated second derivative of a Gaussian across the
    direction times a Gaussian along it, scaled so a Gaussian line profile
    of width ``sigma`` and amplitude 1 answers 1. The step kernel is the
    first derivative across, scaled so a unit step answers 1.
    """
    key = (round(sigma, 3), round(length, 3), n)
    if key in _KERNELS:
        return _KERNELS[key]
    half = int(np.ceil(3.0 * max(sigma * 2.0, length)))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float64)
    out = []
    for k in range(n):
        th = np.pi * k / n
        u = xx * np.cos(th) + yy * np.sin(th)       # across the line
        v = -xx * np.sin(th) + yy * np.cos(th)      # along it
        along = np.exp(-0.5 * (v / length) ** 2)
        g = np.exp(-0.5 * (u / sigma) ** 2)
        line = (1.0 - (u / sigma) ** 2) * g * along
        line -= along * (line.sum(axis=None) / along.sum())  # zero mean
        # Response to a unit Gaussian line along v (profile exp(-u^2/2s^2)).
        line /= max(float((line * g).sum()), 1e-9)
        step = (u / sigma) * g * along
        step /= max(float((step * (u > 0)).sum()), 1e-9)
        out.append((line.astype(np.float32), step.astype(np.float32)))
    _KERNELS[key] = out
    return out


def _oriented_lines(z: np.ndarray, fw: float):
    """Line amplitude (z units, best scale and direction) with steps cut.

    Integrating along the line lifts a faint vein out of skin noise that a
    point-wise test can't; a ridge that comes with a matching slope is the
    side of a step (a fill-light or shadow boundary), not a line.
    """
    h, w = z.shape
    best = np.zeros_like(z)
    for share in _SCALES:
        sigma = max(0.8, share * fw)
        # Filter on a copy where the line is ~1.5 px wide (fast kernels).
        f = max(1.0, sigma / 1.5)
        if f > 1.0:
            small = cv2.resize(z, (max(8, int(round(w / f))), max(8, int(round(h / f)))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = z
        sig_s = sigma / f
        resp = np.zeros_like(small)
        for line_k, step_k in _line_kernels(sig_s, _ALONG * sig_s, _ORIENTATIONS):
            r = cv2.filter2D(small, cv2.CV_32F, line_k, borderType=cv2.BORDER_REFLECT)
            e = np.abs(cv2.filter2D(small, cv2.CV_32F, step_k, borderType=cv2.BORDER_REFLECT))
            r = r * (1.0 - _ramp(e / np.maximum(r, 1e-6), *_STEP_OFF))
            np.maximum(resp, r, out=resp)
        if f > 1.0:
            resp = cv2.resize(resp, (w, h), interpolation=cv2.INTER_LINEAR)
        np.maximum(best, resp, out=best)
    return best


def _long_lines(weight: np.ndarray, fw: float) -> np.ndarray:
    """1 on connected detections that are long, thin lines; 0 elsewhere.

    Hysteresis: components are traced on weak scores (a vein's faint
    stretches and branches stay joined) and kept only if they hold a strong
    one. Veins run for a good share of a face width; skin texture, hosiery
    weave and pores that pass the per-pixel tests are short specks.
    """
    binary = (weight > _COMPONENT_ON).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(weight)
    peak = np.zeros(count, np.float32)
    np.maximum.at(peak, labels.ravel(), weight.ravel())
    keep = np.zeros(count, np.float32)
    min_len = _MIN_LENGTH * fw
    for i in range(1, count):
        if peak[i] < _SEED_ON:
            continue
        w_box, h_box, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if max(w_box, h_box) < 0.5 * min_len:
            continue
        ys, xs = np.nonzero(labels == i)
        (_, _), (a, b), _ = cv2.minAreaRect(np.column_stack([xs, ys]).astype(np.float32))
        length = max(a, b)
        # Curved veins fill a wider box: judge thinness by area per length.
        mean_width = area / max(length, 1.0)
        if length >= min_len and length / max(mean_width, 1.0) >= _MIN_ELONGATION:
            keep[i] = 1.0
    return keep[labels]


def vein_weight(lab: np.ndarray, mask: np.ndarray, fw: float):
    """Per-pixel vein weight (0-1) and the reference Lab, at ``lab``'s size.

    ``lab`` is float CIELab (L* 0-100), ``mask`` the skin mask (0-1) and
    ``fw`` the face width in ``lab`` pixels. Returns None when there is too
    little skin to measure.
    """
    L = lab[..., 0]
    inside = mask > 0.5
    if int(inside.sum()) < _MIN_PX:
        return None
    L_med = float(np.median(L[inside]))
    lit = _ramp(L, 0.15 * L_med, 0.25 * L_med)
    Ls = np.maximum(L, 1e-3)
    chroma = lab[..., 1:] / Ls[..., None]
    sel = inside & (lit > 0.5)
    if int(sel.sum()) < _MIN_PX:
        return None
    # Skin-coloured: near the skin's own typical chromaticity. Costume, bows
    # and straps the mask let through carry no weight, so their edges
    # can't read as lines.
    c0 = np.median(chroma[sel], axis=0)
    if _hue_on(c0[0], c0[1]) <= 0.0:
        return None
    dist = np.hypot(chroma[..., 0] - c0[0], chroma[..., 1] - c0[1])
    tau = max(3.0 * _spread(dist[sel]), 0.25 * float(np.hypot(*c0)), 1e-3)
    like = np.exp(-0.5 * (dist / tau) ** 2)
    w = (mask * lit * like).astype(np.float32)
    w_floor = np.maximum(w, 1e-4)
    ref = _masked_blur(lab.astype(np.float32), w_floor, max(2.0, _REF_SCALE * fw))
    L_ref = np.maximum(ref[..., 0], 1e-3)
    # Chromaticity deviation from the skin around it, back in a*/b* units at
    # the reference lightness, so shading (which scales a*, b* with L*)
    # cancels.
    dev_ab = (chroma - ref[..., 1:] / L_ref[..., None]) * L_ref[..., None]
    cool = dev_ab[..., 0] * _VEIN_DIR[0] + dev_ab[..., 1] * _VEIN_DIR[1]
    # Judged at the finest vein scale: per-pixel chroma noise averages out.
    s0 = max(0.8, _SCALES[0] * fw)
    cool = _masked_blur(cool.astype(np.float32), w_floor, s0)
    # Lines are only judged well inside skin: next to a strap, a costume
    # edge or the mask border, the edge's own blend (darker, greyer) reads
    # as a cool line.
    reach = int(np.ceil(2.5 * max(0.8, _SCALES[-1] * fw)))
    # Smoothed first, so a mole or a glint inside the skin doesn't punch a
    # hole; only real boundaries erode.
    core = cv2.erode(cv2.GaussianBlur(w, (0, 0), sigmaX=0.5 * reach),
                     cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1)))
    # Band-pass: only structure up to a few vein widths counts.
    cool = cool - _masked_blur(cool, w_floor, 3.0 * max(0.8, _SCALES[-1] * fw))
    good = sel & (core > 0.5)
    if int(good.sum()) < _MIN_PX:
        return None
    # The skin's own colour spread at that scale; a floor of a fifth of a
    # delta-E keeps very smooth skin's rounding noise out.
    sig = max(_spread(cool[good]), 0.2)
    z = np.clip(cool / sig, -_Z_CAP[1], _Z_CAP[1]) * _ramp(w, 0.3, 0.7)
    z = z.astype(np.float32)
    z_line = _oriented_lines(z, fw)
    paint_on = (_ramp(np.hypot(ref[..., 1], ref[..., 2]) / L_ref, _NEUTRAL_OFF, _NEUTRAL_ON)
                * _hue_on(ref[..., 1], ref[..., 2]))
    L_s = cv2.GaussianBlur(L, (0, 0), sigmaX=s0)
    # Lightness dip against the skin a few vein widths around (the same
    # band as the colour), so broad shading doesn't count.
    L_base = np.maximum(_masked_blur(L.astype(np.float32), w_floor,
                                     3.0 * max(0.8, _SCALES[-1] * fw)), 1e-3)
    dip = (L_base - L_s) / L_base
    dark = np.maximum(dip, 0.0)
    # A vein shifts colour more than lightness; a crease or fold shadow is
    # the other way round, and sheen (lighter and greyer) is not a vein.
    ratio = cool / (np.maximum(dip, 0.0) * L_base + 0.3)
    colour_led = _ramp(ratio, *_RATIO_ON)
    not_lighter = 1.0 - _ramp(-dip, 0.01, 0.03)
    weight = (_ramp(z_line, *_Z_ON)
              * (1.0 - _ramp(z_line, *_Z_CAP))
              * (1.0 - _ramp(dark, *_DARK_CAP))
              * colour_led * not_lighter
              * paint_on * _ramp(core, 0.3, 0.7))
    # Inside a component that passed the shape test, a moderate per-pixel
    # score is already a confident vein: saturate it.
    weight = _ramp(weight, _COMPONENT_ON, 0.2) * _long_lines(weight, fw)
    # Grow by the line's own half-width so the vein's soft edges follow.
    weight = cv2.dilate(weight.astype(np.float32), np.ones((3, 3), np.uint8))
    weight = np.clip(cv2.GaussianBlur(weight, (0, 0), sigmaX=s0), 0.0, 1.0)
    # The reference again without the veins themselves, so their colour
    # doesn't tint the target.
    # Close by (two widest-vein scales), so the fill matches the skin right
    # beside the vein rather than the broader neighbourhood.
    ref = _masked_blur(lab.astype(np.float32), np.maximum(w * (1.0 - weight) ** 2, 1e-4),
                       max(2.0, _FILL_SCALE * fw))
    return weight.astype(np.float32), ref


def soften_veins(
    img: np.ndarray,
    mask: np.ndarray,
    strength: float,
    face_width: Optional[float] = None,
    *,
    value_scale: Optional[float] = None,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Soften visible veins on skin.

    Args:
        img: (H, W, 3) BGR; uint8, or float32 on the 0-1 or 0-255 scale.
        mask: (H, W) skin mask, 0-1 or 0-255 (eyes, brows, lips, hair and
            costume already left out by the caller).
        strength: 0-1 (slider / 100).
        face_width: face width in px; sets the vein scales. Defaults to 12%
            of the longer image side.
        value_scale: 1.0 or 255.0 for float input; guessed when None.
        exclude_mask: optional mask (0-1 or 0-255) of features never
            touched (eyes, brows, lips); lines are only judged a few vein
            widths away from them.

    Returns:
        Same shape and dtype as ``img``; pixels outside ``mask`` unchanged.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0 or mask is None:
        return img
    m = mask[..., 0] if mask.ndim == 3 else mask
    m = m.astype(np.float32)
    if float(m.max(initial=0.0)) > 1.5:
        m = m / 255.0
    np.clip(m, 0.0, 1.0, out=m)
    if exclude_mask is not None:
        ex = exclude_mask[..., 0] if exclude_mask.ndim == 3 else exclude_mask
        ex = ex.astype(np.float32)
        if float(ex.max(initial=0.0)) > 1.5:
            ex = ex / 255.0
        m *= 1.0 - np.clip(ex, 0.0, 1.0)
    support = m > 1e-3
    rows = np.flatnonzero(support.any(axis=1))
    if rows.size == 0:
        return img
    cols = np.flatnonzero(support.any(axis=0))
    h, w = img.shape[:2]
    fw = float(face_width) if face_width and face_width > 0 else 0.12 * max(h, w)
    fw = max(fw, 40.0)
    pad = int(np.ceil(0.1 * fw))
    y0, y1 = max(0, int(rows[0]) - pad), min(h, int(rows[-1]) + 1 + pad)
    x0, x1 = max(0, int(cols[0]) - pad), min(w, int(cols[-1]) + 1 + pad)
    if img.dtype == np.uint8:
        norm = 255.0
    elif value_scale is not None:
        norm = float(value_scale)
    else:
        norm = 255.0 if float(img.max(initial=0.0)) > 1.5 else 1.0

    crop = np.clip(img[y0:y1, x0:x1].astype(np.float32) / norm, 0.0, 1.0)
    m_c = m[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    scale = min(1.0, _WORK_FACE_WIDTH / fw)
    if scale < 1.0:
        size = (max(8, int(round(cw * scale))), max(8, int(round(ch * scale))))
        f_s = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
        m_s = cv2.resize(m_c, size, interpolation=cv2.INTER_AREA)
    else:
        f_s, m_s = crop, m_c
    lab_s = cv2.cvtColor(f_s, cv2.COLOR_BGR2Lab)
    found = vein_weight(lab_s, m_s, fw * scale)
    if found is None:
        return img
    weight, ref = found
    if float(weight.max(initial=0.0)) <= 1e-3:
        return img
    # Correction field from the vein's lightly smoothed deviation (pores
    # live in lightness, which is only lifted by the smooth dip): colour back
    # to the skin around it at the pixel's own lightness, part of the
    # lightness dip returned.
    s_fine = max(0.5, 0.35 * _SCALES[0] * fw * scale)
    lo = cv2.GaussianBlur(lab_s, (0, 0), sigmaX=s_fine)
    L_lo = np.maximum(lo[..., 0], 1e-3)
    L_ref = np.maximum(ref[..., 0], 1e-3)
    dL = np.maximum(L_ref - lo[..., 0], 0.0) * _L_SHARE
    target_ab = ref[..., 1:] / L_ref[..., None] * (L_lo + dL)[..., None]
    k = (strength * weight * m_s).astype(np.float32)
    fields = np.dstack([k * dL, k[..., None] * (target_ab - lo[..., 1:])]).astype(np.float32)
    if scale < 1.0:
        fields = cv2.resize(fields, (cw, ch), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2Lab)
    lab += fields
    np.clip(lab[..., 0], 0.0, 100.0, out=lab[..., 0])
    new = np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0.0, 1.0)
    sup = support[y0:y1, x0:x1] & (np.abs(fields).max(axis=2) > 1e-4)
    out = img.copy()
    if img.dtype == np.uint8:
        vals = np.clip(new * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        vals = (new * norm).astype(img.dtype)
    out[y0:y1, x0:x1][sup] = vals[sup]
    return out
