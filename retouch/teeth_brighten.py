"""Teeth Brightening: lift and de-yellow visible teeth, keep their own shade.

Why a new op next to ``teeth_whiten`` (``retouch/teeth.py``): that op takes
"teeth" to be any mouth pixel brighter and less saturated than the mouth
median, so on a mouth that is mostly lips it picks lip highlights; it adds a
fixed +15 L (uint8 Lab scale) whatever the exposure, and then scales the whole
edit down until the mean Whiteness Index change is 2.62, so at 100 it moves
teeth only a few levels. 40 recipes set it (0.05-0.35), so it is left as is.

This op, per face:

* **Where**: the mouth opening is the inner-lip landmark polygon, grown a
  little and joined with the parser's mouth-interior mask. Inside it a pixel
  is tooth when its redness (a*/L*) is well below the lips' own (gums, tongue
  and lips are red, enamel is not; measured against the lips of this face, so
  lipstick and pale or painted lips move the gate with them) and its linear
  luminance is a good share of the opening's brightest (the dark mouth and the
  gaps between teeth stay out). Both gates are ratios, so a darker or brighter
  exposure, or darker skin, selects the same teeth.
* **What**: yellow is pulled down as a share of its excess over a warm-ivory
  floor (b* = 0.06 L*, never toward blue), a little redness the same way, and
  the teeth are lifted by a linear-light gain (up to +0.6 stop at 100) through
  a shoulder that rolls off toward the encoding's white, so bright enamel
  never clips and darker teeth keep their shade relative to each other.
  Texture, edges between teeth and highlights are multiplicative, so they stay.
* Pixels outside the tooth support are returned bit-identical.

Known gaps: a tongue lighter than the teeth with low redness can read as
teeth; braces and metal read as teeth (bright, not red) and get brightened;
very dark-stained teeth whose redness reaches the lips' are kept as they are.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .utils import normalize_mask

# MediaPipe face-mesh contours (same lists as parsing.LIPS_INNER/OUTER).
_INNER = (78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
          308, 324, 318, 402, 317, 14, 87, 178, 88, 95)
_OUTER = (61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
          291, 409, 270, 269, 267, 0, 37, 39, 40, 185)
_IRIS_CENTRES = (468, 473)

# The outer lip line is inset by this many IED before searching.
_INSET = 0.025
# Teeth sit within this many IED of the inner-lip line.
_BAND = 0.06
# Smallest tooth area kept, as a share of IED^2 (lip glints are smaller).
_MIN_TOOTH_AREA = 0.0006
# Share of a tooth area's surrounding ring that must lie inside the lip line.
_MIN_RING_INSIDE = 0.6
# A tooth area's median linear luminance over its ring's.
_MIN_RING_CONTRAST = 1.3
# Teeth redness gate as a share of the lips' own a*/L*.
_RED_LO, _RED_HI = 0.35, 0.60
# ... and within this a*/L* margin of the least red bright pixel on the band.
_RED_SPAN_LO, _RED_SPAN_HI = 0.05, 0.10
# Lips are rarely less red than this (a*/L*); keeps the gate sane on pale or
# white-painted lips.
_LIP_RED_FLOOR = 0.12
# Luminance gate: share of the opening's p97 linear luminance.
_LUM_LO, _LUM_HI = 0.18, 0.40
# Yellow floor (b* per unit L*): a natural ivory, never bleached blue.
_B_FLOOR = 0.04
_A_FLOOR = 0.01
_MAX_YELLOW_CUT = 0.80
_MAX_RED_CUT = 0.50
_MAX_STOPS = 0.60


def _smooth(x, lo, hi):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _srgb_to_linear(v):
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v):
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def _points(landmarks: Any, size: Tuple[int, int]):
    pts = getattr(landmarks, "landmark", landmarks)
    try:
        if len(pts) < 478:
            return None
    except TypeError:
        return None
    h, w = size

    def xy(idx):
        p = pts[idx]
        return (p.x * w, p.y * h) if hasattr(p, "x") else (p[0] * w, p[1] * h)

    inner = np.array([xy(i) for i in _INNER], np.float32)
    outer = np.array([xy(i) for i in _OUTER], np.float32)
    (lx, ly), (rx, ry) = (xy(i) for i in _IRIS_CENTRES)
    return inner, outer, float(np.hypot(rx - lx, ry - ly))


def _poly_mask(poly, origin, shape):
    mask = np.zeros(shape, np.uint8)
    shifted = np.round((poly - np.array(origin, np.float32)) * 16).astype(np.int32)
    cv2.fillPoly(mask, [shifted], 255, lineType=cv2.LINE_AA, shift=4)
    return mask.astype(np.float32) / 255.0


def teeth_support(bgr01: np.ndarray, mouth: np.ndarray, band: np.ndarray,
                  ied: float) -> np.ndarray:
    """Soft (0-1) tooth membership for a BGR [0, 1] window.

    ``mouth`` is the outer-lip polygon (lips + opening), ``band`` the strip
    around the inner-lip line where the teeth must sit.
    """
    zero = np.zeros(mouth.shape, np.float32)
    inside = mouth > 0.5
    if inside.sum() < 50:
        return zero
    lab = cv2.cvtColor(bgr01.astype(np.float32), cv2.COLOR_BGR2LAB)
    redness = lab[..., 1] / np.maximum(lab[..., 0], 1.0)
    # Lips are most of the outer-lip polygon even in a wide smile.
    lip_red = max(float(np.percentile(redness[inside], 75)), _LIP_RED_FLOOR)
    lin = _srgb_to_linear(bgr01)
    lum = 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]
    near = inside & (band > 0.5)
    if near.sum() < 20:
        return zero
    top = float(np.percentile(lum[near], 97))
    if top <= 0:
        return zero
    lum_ok = _smooth(lum / top, _LUM_LO, _LUM_HI)
    # Enamel is the least red thing on the inner-lip band. Measured from this
    # mouth, the gate also holds two-tone lips (an upper lip under pale face
    # paint is far less red than the lower one but still redder than teeth).
    bright = near & (lum_ok > 0.5)
    least = float(np.percentile(redness[bright], 5)) if bright.sum() >= 20 else lip_red
    lo = min(_RED_LO * lip_red, least + _RED_SPAN_LO)
    hi = min(_RED_HI * lip_red, least + _RED_SPAN_HI)
    red_ok = 1.0 - _smooth(redness, lo, hi)
    score = (red_ok * lum_ok * mouth).astype(np.float32)

    # Keep tooth-shaped areas: on the inner-lip band, big enough, and ringed
    # by the mouth (lips, the dark mouth) rather than by the face outside the
    # lip line, which is what pale skin leaking in at a mouth corner has.
    k = max(3, int(round(0.010 * ied)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    core = cv2.morphologyEx((score > 0.5).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    keep = np.zeros(core.shape, np.uint8)
    min_area = _MIN_TOOTH_AREA * ied * ied
    ring_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    for idx in range(1, count):
        if stats[idx, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y, w, h = (stats[idx, c] for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                                              cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        pad = 2 * k + 2
        y0, y1 = max(y - pad, 0), min(y + h + pad, core.shape[0])
        x0, x1 = max(x - pad, 0), min(x + w + pad, core.shape[1])
        comp = (labels[y0:y1, x0:x1] == idx).astype(np.uint8)
        if not (band[y0:y1, x0:x1][comp > 0] > 0.5).any():
            continue
        ring = (cv2.dilate(comp, ring_k) > 0) & (comp == 0)
        if ring.sum() and float((mouth[y0:y1, x0:x1][ring] > 0.5).mean()) < _MIN_RING_INSIDE:
            continue
        # Teeth stand out from the dark mouth and lips around them; a pale
        # patch of lip or paint is about as bright as its own surroundings.
        lum_w = lum[y0:y1, x0:x1]
        if ring.sum() and (np.median(lum_w[comp > 0])
                           < _MIN_RING_CONTRAST * np.median(lum_w[ring])):
            continue
        keep[y0:y1, x0:x1] |= comp
    if not keep.any():
        return zero
    grown = cv2.dilate(keep, kernel).astype(np.float32)
    sigma = max(0.6, 0.004 * ied)
    soft = cv2.GaussianBlur(score * grown, (0, 0), sigma)
    return np.clip(np.minimum(soft, score), 0.0, 1.0).astype(np.float32)


def brighten_teeth_pixels(bgr01: np.ndarray, strength: float) -> np.ndarray:
    """Full-strength edit of BGR [0, 1] pixels (caller blends by support)."""
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    lab = cv2.cvtColor(bgr01.astype(np.float32), cv2.COLOR_BGR2LAB)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    lab[..., 2] = b - _MAX_YELLOW_CUT * s * np.maximum(b - _B_FLOOR * L, 0.0)
    lab[..., 1] = a - _MAX_RED_CUT * s * np.maximum(a - _A_FLOOR * L, 0.0)
    lin = _srgb_to_linear(np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), 0.0, 1.0))
    lum = 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]
    gain = 2.0 ** (_MAX_STOPS * s)
    # Shoulder toward the encoding's white: Y' = gY / (1 + (g - 1) Y).
    lifted = gain * lum / (1.0 + (gain - 1.0) * lum)
    ratio = np.where(lum > 1e-6, lifted / np.maximum(lum, 1e-6), 1.0)[..., None]
    out = lin * ratio
    # A channel pushed past white would shift hue; pull it back by luminance.
    over = out.max(axis=2, keepdims=True)
    out = np.where(over > 1.0, out / np.maximum(over, 1e-6), out)
    return _linear_to_srgb(out).astype(np.float32)


def brighten_teeth(canvas: np.ndarray, landmarks: Any, strength: float,
                   mouth_interior: Optional[np.ndarray] = None,
                   support_out: Optional[np.ndarray] = None) -> np.ndarray:
    """Brighten one face's visible teeth.

    Args:
        canvas: (H, W, 3) BGR float32 [0, 255] (uint8 accepted), the face ROI.
        landmarks: 478-point MediaPipe landmarks normalised to this ROI.
        strength: 0-100; 0 returns ``canvas`` unchanged.
        mouth_interior: optional (H, W) parser mouth-interior mask.
        support_out: optional (H, W) float32 array; the edit's support is
            max-accumulated into it (for compositing the face back).

    Returns:
        float32 BGR (or ``canvas`` itself when nothing changes); pixels
        outside the tooth support are bit-identical.
    """
    if float(strength) <= 0.0 or landmarks is None:
        return canvas
    h, w = canvas.shape[:2]
    geometry = _points(landmarks, (h, w))
    if geometry is None:
        return canvas
    inner, outer, ied = geometry
    if ied < 8:
        return canvas
    margin = int(np.ceil(0.08 * ied)) + 2
    x0 = max(int(np.floor(outer[:, 0].min())) - margin, 0)
    y0 = max(int(np.floor(outer[:, 1].min())) - margin, 0)
    x1 = min(int(np.ceil(outer[:, 0].max())) + margin + 1, w)
    y1 = min(int(np.ceil(outer[:, 1].max())) + margin + 1, h)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return canvas
    shape = (y1 - y0, x1 - x0)
    # Inset from the outer lip line: where pale skin or paint meets the lip
    # it reads as un-red and bright, and teeth never sit right on that line.
    inset = max(1, int(round(_INSET * ied)))
    mouth = cv2.erode(_poly_mask(outer, (x0, y0), shape), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)))
    # The inner-lip landmarks often collapse onto the lower edge of the upper
    # teeth (the mesh reads a smile with lips apart as a closed mouth), so
    # the teeth are searched for in a band around that line, not inside it.
    line = np.zeros(shape, np.uint8)
    shifted = np.round((inner - np.array((x0, y0), np.float32)) * 16).astype(np.int32)
    cv2.fillPoly(line, [shifted], 255, shift=4)
    cv2.polylines(line, [shifted], True, 255, 1, shift=4)
    reach = max(2, int(round(_BAND * ied)))
    band = cv2.dilate(line, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * reach + 1, 2 * reach + 1))).astype(np.float32) / 255.0
    if mouth_interior is not None and mouth_interior.shape[:2] == (h, w):
        band = np.maximum(band, normalize_mask(mouth_interior)[y0:y1, x0:x1])

    window = canvas[y0:y1, x0:x1].astype(np.float32) / 255.0
    support = teeth_support(window, mouth, band, ied)
    if float(support.max()) < 0.02:
        return canvas
    edited = brighten_teeth_pixels(window, strength)
    blended = window + support[..., None] * (edited - window)
    out = canvas.astype(np.float32, copy=True)
    region = out[y0:y1, x0:x1]
    keep = support <= 0.0
    patch = blended * 255.0
    patch[keep] = region[keep]
    out[y0:y1, x0:x1] = patch
    if support_out is not None and support_out.shape[:2] == (h, w):
        np.maximum(support_out[y0:y1, x0:x1], support, out=support_out[y0:y1, x0:x1])
    return out
