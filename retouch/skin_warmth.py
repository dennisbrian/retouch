"""Skin Warmth: move natural skin toward a warm peach, keyed on each face.

The 13 reference edits Alex supplied put skin at a warm peach (CIELab hue
~32 deg, chroma ~16.5); our recipes leave natural skin pinker and flatter
(hue 5-18 deg, chroma 10-14). The grade's own warmth and ``skin_hue_unify``
can't close that: the first warms the whole frame, the second turns hue at
most 8 deg and never adds colour.

How it works, per face (connected piece of the face skin mask):

* Reference colour = median a*/b* of that face's confident skin, measured on
  the graded image, so the result is where the skin ends up.
* Target = the same colour with its hue pulled into the peach band
  (``HUE_BAND``) and chroma raised to at least ``CHROMA_PER_L`` x L*. The
  chroma floor scales with the skin's own lightness, so darker skin, whose
  chroma is naturally lower in Lab, is not pushed orange; chroma is never
  reduced. Skin already inside the band with enough chroma gets no shift.
* The one a*/b* offset is added to every pixel of the face skin and to
  pixels of the same person whose colour matches that face's skin (neck,
  chest, arms, hands), so body and face stay matched and blush, shading and
  texture are kept. The colour key's width is the face's own a*/b* spread.
* Face paint is left alone: a skin colour far outside the natural skin hue
  range (blue, violet, green paint) or near-neutral for its lightness (white
  paint) gets no shift.

No absolute intensity threshold anywhere: every gate is a hue angle, a
chroma-to-lightness ratio, or relative to the face's own colour spread.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

__all__ = ["skin_warmth", "HUE_BAND", "CHROMA_PER_L"]

#: CIELab hue band (deg) skin is pulled into. The 13 reference edits have a
#: median of 32 deg; darker skin in the references and corpus sits at 37-45.
HUE_BAND = (30.0, 42.0)
#: Chroma floor as a fraction of L*: 0.21 x 75 = 15.8 on light skin (the
#: references' 16.5), 7 on L* 34 skin, whose natural chroma is already above.
CHROMA_PER_L = 0.21
#: Fraction of the way to the target at strength 100. The full pull read a
#: little orange on the test photos.
MAX_PULL = 0.75
#: Natural skin hue range (deg) in which the op acts, with soft edges.
_HUE_ON = (-5.0, 75.0)
_HUE_RAMP = 20.0
#: Chroma/L* below which the skin reads as white/grey paint (off), and above
#: which it counts as skin colour (on). Natural light skin sits at 0.12-0.2.
_NEUTRAL_OFF, _NEUTRAL_ON = 0.05, 0.08


def _clip01(x: np.ndarray) -> np.ndarray:
    """In-place clip to [0, 1]; np.clip is several times slower at 26 MP."""
    np.maximum(x, 0.0, out=x)
    np.minimum(x, 1.0, out=x)
    return x


def _norm(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)  # always a copy, so clipping in place is safe
    if m.ndim == 3:
        m = np.ascontiguousarray(m[..., 0])
    if m.size and float(m.max()) > 1.5:
        m *= 1.0 / 255.0
    return _clip01(m)


def _paint_gate(hue_deg: float, chroma: float, light: float) -> float:
    """1 for natural skin colour, 0 for face paint, soft in between."""
    lo, hi = _HUE_ON
    if hue_deg < lo:
        g_hue = 1.0 - (lo - hue_deg) / _HUE_RAMP
    elif hue_deg > hi:
        g_hue = 1.0 - (hue_deg - hi) / _HUE_RAMP
    else:
        g_hue = 1.0
    ratio = chroma / max(light, 1.0)
    g_neutral = (ratio - _NEUTRAL_OFF) / (_NEUTRAL_ON - _NEUTRAL_OFF)
    return float(np.clip(g_hue, 0.0, 1.0) * np.clip(g_neutral, 0.0, 1.0))


def _target_shift(a0: float, b0: float, L0: float) -> tuple:
    """a*/b* offset that moves (a0, b0) to the peach target, before strength."""
    C0 = float(np.hypot(a0, b0))
    h0 = float(np.degrees(np.arctan2(b0, a0)))
    ht = float(np.clip(h0, *HUE_BAND))
    Ct = max(C0, CHROMA_PER_L * L0)
    ta = Ct * np.cos(np.radians(ht))
    tb = Ct * np.sin(np.radians(ht))
    return ta - a0, tb - b0


def skin_warmth(
    img: np.ndarray,
    skin_mask: np.ndarray,
    strength: float,
    person_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Warm natural skin toward peach. See the module docstring.

    Args:
        img: BGR image, uint8 or float32 in [0, 1].
        skin_mask: (H, W) face skin mask (the engine's accumulated skin),
            0-1 or 0-255.
        strength: 0-100; 0 returns ``img`` unchanged.
        person_mask: optional (H, W) person mask; when given, same-coloured
            body skin inside it is warmed with the face.

    Returns:
        Image of the same dtype and range as ``img``.
    """
    s = float(strength) / 100.0
    if s <= 0 or skin_mask is None:
        return img
    m = _norm(skin_mask)
    if m.shape != img.shape[:2] or float(m.max()) < 0.5:
        return img

    is_u8 = img.dtype == np.uint8
    f = img.astype(np.float32) * (1.0 / 255.0) if is_u8 else _clip01(img.astype(np.float32))
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2Lab)  # true CIELab: L 0-100, a/b ~±127
    A, B = lab[..., 1], lab[..., 2]

    pm = None
    if person_mask is not None:
        pm = _norm(person_mask)
        if pm.shape != m.shape:
            pm = None

    n, labels, stats, _ = cv2.connectedComponentsWithStats((m > 0.5).astype(np.uint8), 8)
    min_px = max(200, int(0.00002 * m.size))
    da = np.zeros(m.shape, np.float32)
    db = np.zeros(m.shape, np.float32)
    weight_sum = np.zeros(m.shape, np.float32)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_px:
            continue
        sel = labels == i
        conf = sel & (m > 0.8)
        if int(conf.sum()) < min_px // 2:
            conf = sel
        a_s, b_s = A[conf], B[conf]
        a0, b0 = float(np.median(a_s)), float(np.median(b_s))
        L0 = float(np.median(lab[..., 0][conf]))
        C0 = float(np.hypot(a0, b0))
        h0 = float(np.degrees(np.arctan2(b0, a0)))
        gate = _paint_gate(h0, C0, L0)
        if gate <= 0:
            continue
        sa, sb = _target_shift(a0, b0, L0)
        if abs(sa) + abs(sb) < 1e-3:
            continue
        # This face's reach: its own skin (feathered mask) plus same-person
        # pixels whose a*/b* match its skin. Key width from the face's own
        # colour spread, so blush and shading stay inside the key.
        x, y, w, h = stats[i, :4]
        face_w = int(max(w, h))
        spread = float(np.hypot(np.percentile(a_s, 84) - np.percentile(a_s, 16),
                                np.percentile(b_s, 84) - np.percentile(b_s, 16))) / 2.0
        sigma = max(2.5, 1.5 * spread)
        key = cv2.magnitude(A - a0, B - b0)
        key *= key * np.float32(-1.0 / (2.0 * sigma * sigma))
        cv2.exp(key, key)
        # Stay near this person: inside the person mask when there is one,
        # else within a few face-widths of the face.
        if pm is not None:
            reach = pm
        else:
            reach = np.zeros(m.shape, np.float32)
            y0, y1 = max(0, y - face_w), min(m.shape[0], y + h + 3 * face_w)
            x0, x1 = max(0, x - 2 * face_w), min(m.shape[1], x + w + 2 * face_w)
            reach[y0:y1, x0:x1] = 1.0
            k = max(3, face_w // 4) | 1
            reach = cv2.GaussianBlur(reach, (k, k), 0)
        # Own face skin via the feathered mask; other pixels via the key.
        kd = max(3, face_w // 8) | 1
        own = m * cv2.dilate(sel.astype(np.uint8), np.ones((kd, kd), np.uint8)).astype(np.float32)
        key *= reach
        w_i = np.maximum(own, key, out=key)
        w_i *= np.float32(gate)
        cv2.scaleAdd(w_i, float(sa), da, dst=da)
        cv2.scaleAdd(w_i, float(sb), db, dst=db)
        weight_sum += w_i

    if not weight_sum.any():
        return img
    # Two faces' reaches can overlap (similar skin): average their shifts
    # there instead of adding them.
    over = weight_sum > 1.0
    if over.any():
        da[over] /= weight_sum[over]
        db[over] /= weight_sum[over]

    pull = np.float32(MAX_PULL * min(s, 1.0))
    da *= pull
    db *= pull
    lab[..., 1] += da
    lab[..., 2] += db
    out = _clip01(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR))
    if is_u8:
        return cv2.convertScaleAbs(out, alpha=255.0)
    return out
