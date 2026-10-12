"""Powder finish in the skin's own tone (opt-in ``powder_finish``).

A setting-powder look: shine and sheen on the face go soft and velvety,
the way a light dusting of translucent powder reads, while the skin keeps
its own colour.

Why not the existing ``specular_finish="powder"`` mode: it subtracts the
same amount from B, G and R, sized by the *whole* pixel intensity of
anything bright and low-chroma. On white face paint or pale skin every lit
cheek qualifies, so the finish paints grey patches onto the face (Alex's
bunny photos: face L 159 -> 134 at strength 1.0). And Face Polish's shine
removal only acts where a highlight is *less saturated* than the skin
around it; on white face paint shine keeps the paint's chroma, so it is
left almost untouched there.

This op measures everything against the face's own skin:

* a local diffuse baseline of L, a and b (masked two-pass blur of the skin
  at 0.2 x inter-eye distance, with pixels already standing out removed
  from the second pass), so the lit side of a face rises with its baseline
  and isn't treated as shine;
* the excess of L above that baseline, gated by the face's own fine
  texture spread (robust sigma of L minus a small blur), not a fixed level;
* the excess is pulled most of the way back to the baseline, and a/b move
  toward the local skin colour by the same amount, so a highlight settles
  into the skin tone instead of turning grey;
* the broad part of that change is given back, so a lit cheek keeps its
  brightness and only the local peak of each highlight goes;
* a velvet pass trims the bright side of fine sheen (sparkle on pores and
  paint) by up to a third, leaving the dark side, so texture stays.

No absolute intensity threshold is used anywhere, so it behaves the same on
lighter and darker skin (see ``tests/test_powder_finish.py``).
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .utils import bgr_f32_to_lab_f32, lab_f32_to_bgr_f32, normalize_mask

# Broad-form baseline scale, fraction of the inter-eye distance (IED).
_BASE_SIGMA_IED = 0.20
# Scale of the change given back to keep overall brightness.
_KEEP_SIGMA_IED = 0.35
# Fine texture scale for the spread measure and the velvet pass.
_FINE_SIGMA_IED = 0.02
# Shine gate: ramps in from _GATE_LO to _GATE_HI robust sigmas of the
# face's own fine L texture above the local baseline.
_GATE_LO = 1.5
_GATE_HI = 4.0
# At strength 100 at most this share of the excess goes: a soft residue
# keeps the face from reading as flat paint.
_MAX_PULL = 0.70
# Share of the bright side of fine sheen removed at strength 100.
_VELVET = 0.35


def _masked_blur(x: np.ndarray, w: np.ndarray, sigma: float, fallback: float) -> np.ndarray:
    num = cv2.GaussianBlur(x * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), fallback).astype(np.float32)


def powder_finish(
    canvas: np.ndarray,
    skin_mask: Optional[np.ndarray],
    strength: float,
    ied: float,
    eyes_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Matte the face's shine toward its own skin tone.

    Args:
        canvas: (H, W, 3) float32 BGR [0, 255] face ROI.
        skin_mask: (H, W) skin mask, 0-1 or 0-255. ``None`` or empty
            returns ``canvas`` unchanged.
        strength: 0-100. 0 returns ``canvas`` unchanged.
        ied: inter-eye distance in ROI pixels (sets every spatial scale).
        eyes_mask: optional (H, W) eye mask; dilated and left untouched so
            catchlights and lash-line shine stay.

    Returns:
        (H, W, 3) float32 BGR [0, 255].
    """
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if s <= 0.0 or skin_mask is None:
        return canvas
    sk = np.clip(normalize_mask(skin_mask), 0.0, 1.0)
    core = (sk > 0.3).astype(np.float32)
    if core.sum() < 50:
        return canvas

    ied = float(ied) if ied and ied > 0 else float(max(canvas.shape[:2])) * 0.3
    work = canvas.astype(np.float32, copy=False)
    lab = bgr_f32_to_lab_f32(work)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # The face's own fine texture spread sets the shine margin (tone-relative).
    fine = max(0.8, ied * _FINE_SIGMA_IED)
    hp = L - cv2.GaussianBlur(L, (0, 0), fine)
    hp_skin = hp[core > 0]
    spread = max(0.75, 1.4826 * float(np.median(np.abs(hp_skin - np.median(hp_skin)))))

    sel = core > 0
    med_L = float(np.median(L[sel]))
    sigma = max(2.0, ied * _BASE_SIGMA_IED)
    L_base1 = _masked_blur(L, core, sigma, med_L)
    # Second pass: pixels already standing out stop voting, so a broad
    # highlight doesn't lift its own reference and hide itself.
    w2 = core * (1.0 - np.clip((L - L_base1 - _GATE_LO * spread) / (spread * 2.0), 0.0, 1.0))
    L_base = _masked_blur(L, w2, sigma, med_L)
    a_base = _masked_blur(a, w2, sigma, float(np.median(a[sel])))
    b_base = _masked_blur(b, w2, sigma, float(np.median(b[sel])))

    excess = np.maximum(L - L_base, 0.0)
    gate = np.clip((excess - _GATE_LO * spread) / ((_GATE_HI - _GATE_LO) * spread), 0.0, 1.0)
    gate = cv2.GaussianBlur(gate, (0, 0), max(0.8, fine))

    protect = sk
    if eyes_mask is not None:
        em = normalize_mask(eyes_mask)
        if em is not None and em.shape == sk.shape and em.max() > 0:
            r = max(1, int(round(ied * 0.06)))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            em = cv2.GaussianBlur(cv2.dilate(em, k), (0, 0), r * 0.5 + 0.5)
            protect = sk * (1.0 - np.clip(em, 0.0, 1.0))

    amount = np.clip(gate * s * _MAX_PULL, 0.0, _MAX_PULL) * protect
    L_new = L - excess * amount
    a_new = a + (a_base - a) * amount
    b_new = b + (b_base - b) * amount

    # Keep the face's overall brightness: give back the broad part of the
    # change, so only the local peak of the shine goes (powder evens a
    # highlight into the skin around it rather than darkening a lit cheek).
    dL = L_new - L
    dL_broad = _masked_blur(dL, core, max(2.0, ied * _KEEP_SIGMA_IED), 0.0)
    L_new = L_new - dL_broad * protect

    # Velvet: trim the bright side of fine sheen only.
    hp2 = L_new - cv2.GaussianBlur(L_new, (0, 0), fine)
    L_new = L_new - np.maximum(hp2, 0.0) * (s * _VELVET) * protect

    out_lab = np.stack([np.clip(L_new, 0, 255), a_new, b_new], axis=-1).astype(np.float32)
    out = lab_f32_to_bgr_f32(out_lab)
    # Only skin pixels move; everything else is the input, bit for bit.
    m = protect[..., None]
    res = work * (1.0 - m) + out * m
    return np.clip(res, 0.0, 255.0).astype(np.float32)
