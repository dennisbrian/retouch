"""Costume Shine: matte glare on glossy costume parts into the material's colour.

Vinyl, latex, PVC, patent leather, satin and armour throw bright specular
streaks under flash, softboxes and con-hall lighting. A little reads as
gloss; a lot hides the colour and shape of the piece. Nothing in the
pipeline addressed it: Powder Finish and shine removal work on face skin,
Wig Shine on the wig, and Costume Clarity adds contrast. This op takes a
share of that specular light back off the costume only.

Costume region: :func:`costume_clarity.costume_mask` (the person minus skin,
hair, wigs, the face, painted skin), so skin, face paint and hair are never
touched.

Shine (work resolution, face width scaled to about 400 px):

* the material under the shine is a morphological *opening* of each linear
  RGB channel with a disc of 0.06 face widths (pixels outside the costume are
  pinned high so they never win the minimum), lightly smoothed and capped by
  the pixel itself. Bright structures narrower than the disc (specular streaks
  and glints) stand above it; the material's own lit/shadow shape and anything
  wider than the disc (a broad satin sheen, a white panel) are the base;
* the gate is the L* excess over that base, ramped in from 4 to 10 times the
  costume's own fine-texture spread (robust sigma of L* minus a small blur),
  so a matte fabric's weave never counts and the level follows the photo;
* the added light must be close to neutral (the dichromatic model: gloss off a
  dielectric is the light's colour on top of the material's). Channels within
  15% of clipping are ignored for this test, so a pinkish-white streak on red
  vinyl (red already near clipping) still counts, while a coloured print on a
  darker piece does not;
* thick white objects are kept: bright connected regions (L* above 65% of the
  costume's own 99.5th percentile) that contain a core at least as wide as
  the disc (0.06 face widths), plus everything connected to that core (white bows, collars, cuffs
  and their ribbons);
* dense bright patterns are kept: where gated pixels cover more than 30-50% of
  the costume around them (3 disc widths), the bright part is the material,
  not gloss (skin through fishnet, white print, rhinestone clusters);
* bright texture filling its surroundings is kept: where the mean linear lift
  over the base within 2 disc widths is more than 20-32% of the strongest
  lift there (skin between the threads of a fishnet stocking, a light weave).
  A streak on darker material sits in a window that is mostly material;
* tiny glints (connected gated areas under (0.02 face width)^2) are kept as
  sparkle (studs, rhinestones, sequins).

Removal (full resolution): the neutral part of the added light (the smallest
channel of pixel minus base, in linear light, capped by the pixel's darkest
channel) is taken off, up to 80% at strength 100, so the streak sinks back
into the material's own colour and some gloss stays. Every other pixel is
returned bit-identical.

No absolute intensity threshold is used on the subject; every level is
measured on this costume (``tests/test_costume_shine.py`` checks exposure
invariance). Off by default (strength 0 is a no-op).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .costume_clarity import costume_prepass

_logger = logging.getLogger(__name__)

# Opening disc diameter (x face width): shine is narrower than this.
_DISC_FW = 0.06
# Fine-texture spread scale (x face width) for the gate unit.
_FINE_SIGMA_FW = 0.006
# Gate: excess over the base from _GATE_LO to _GATE_HI fine-texture sigmas.
_GATE_LO, _GATE_HI = 4.0, 10.0
_SIG_FLOOR = 0.4
# Neutrality of the added light: (max - min) / max over unclipped channels.
_NEUTRAL_LO, _NEUTRAL_HI = 0.25, 0.45
# Channels above this linear value are treated as clipped for that test.
_CLIP_LIN = 0.85
# Thick white objects: L* above this share of the costume's p99.5, with a core
# at least as wide as the opening disc (radius _THICK_FW face widths).
_WHITE_REL = 0.65
_THICK_FW = 0.03
# Dense bright patterns: window (x disc) and density ramp.
_DENSE_WIN = 3.0
_DENSE_LO, _DENSE_HI = 0.3, 0.5
# Fill guard: local mean lift / pixel lift within one disc.
_FILL_LO, _FILL_HI = 0.2, 0.32
_FILL_WIN = 2.0
# Tiny glints kept: connected gated area below (_GLINT_FW x face width)^2.
_GLINT_FW = 0.02
# Share of the shine taken off at strength 100.
_MAX_PULL = 0.8


def _ramp(x, lo, hi):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


def shine_gate(
    img: np.ndarray,
    mask: np.ndarray,
    face_width: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """Shine gate and material base at work resolution.

    Args:
        img: (h, w, 3) float32 BGR [0, 1] work-resolution image.
        mask: (h, w) costume mask in [0, 1].
        face_width: face width in ``img`` pixels.

    Returns:
        ``(gate, base_linear, diag)``; gate is None when no shine is found.
    """
    diag: Dict[str, Any] = {}
    cost = mask > 0.5
    if int(cost.sum()) < 64:
        diag["reason"] = "no_costume"
        return None, None, diag
    fw = max(float(face_width), 8.0)
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    lin = _to_linear(img)
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]

    r = max(2, int(round(_DISC_FW * fw / 2.0)))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    pinned = np.where(cost[..., None], lin, 1.0).astype(np.float32)
    base = cv2.dilate(cv2.erode(pinned, disc), disc)
    base = np.minimum(base, lin)
    base = cv2.GaussianBlur(base, (0, 0), max(1.0, r / 3.0))
    base = np.minimum(base, lin)
    add = lin - base

    L_base = cv2.cvtColor(_to_srgb(base), cv2.COLOR_BGR2LAB)[..., 0]
    excess = L - L_base
    fine = L - cv2.GaussianBlur(L, (0, 0), max(0.8, _FINE_SIGMA_FW * fw))
    fv = fine[cost]
    sig = max(float(1.4826 * np.median(np.abs(fv - np.median(fv)))), _SIG_FLOOR)
    gate = _ramp(excess, _GATE_LO * sig, _GATE_HI * sig)

    # Neutral added light, ignoring channels already near clipping.
    unclipped = lin < _CLIP_LIN
    big = np.where(unclipped, add, -np.inf).max(axis=2)
    small = np.where(unclipped, add, np.inf).min(axis=2)
    two = unclipped.sum(axis=2) >= 2
    ratio = np.where(two, (big - small) / np.maximum(big, 1e-4), 0.0)
    g = gate * (1.0 - _ramp(np.nan_to_num(ratio), _NEUTRAL_LO, _NEUTRAL_HI)) * mask

    # Thick white objects and everything connected to their core.
    white_level = _WHITE_REL * float(np.percentile(L[cost], 99.5))
    bright = ((L > white_level) & cost).astype(np.uint8)
    rr = max(1, int(round(_THICK_FW * fw)))
    core = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rr + 1, 2 * rr + 1)))
    if core.any():
        _, labels, _, _ = cv2.connectedComponentsWithStats(bright, connectivity=8)
        hit = np.unique(labels[core > 0])
        white = np.isin(labels, hit[hit > 0]).astype(np.uint8)
        white = cv2.dilate(white, np.ones((5, 5), np.uint8)) > 0
        g = g * ~white
        diag["white_px"] = int(white.sum())

    # Shape measures use the linear lift over the base, which a change of
    # exposure scales but does not reshape.
    Y = lin.mean(axis=2)
    Yb = base.mean(axis=2)
    lift = np.maximum(Y - Yb, 0.0) * cost
    costf = cost.astype(np.float32)

    # Dense bright patterns: where gated pixels cover more than 30-50% of the
    # costume around them (3 disc widths), the bright part is the material,
    # not gloss (skin through fishnet, white print, rhinestone clusters).
    win = max(3, int(round(_DENSE_WIN * 2 * r))) | 1
    gated = (gate > 0.3).astype(np.float32) * cost
    density = cv2.boxFilter(gated, -1, (win, win), normalize=False) / np.maximum(
        cv2.boxFilter(costf, -1, (win, win), normalize=False), 1.0)
    g = g * (1.0 - _ramp(density, _DENSE_LO, _DENSE_HI))

    # Bright texture filling its surroundings (skin between fishnet threads, a
    # weave of light threads): within two disc widths, the mean lift is a
    # large share of the strongest lift there. A streak or glint on darker
    # material sits in a window that is mostly material.
    k = int(round(_FILL_WIN * 2 * r)) | 1
    local = cv2.boxFilter(lift, -1, (k, k)) / np.maximum(cv2.boxFilter(costf, -1, (k, k)), 1e-3)
    peak = cv2.dilate(lift, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    fill = local / np.maximum(peak, 1e-4)
    g = g * (1.0 - _ramp(fill, _FILL_LO, _FILL_HI))

    # Tiny glints stay as sparkle.
    n, labels, stats, _ = cv2.connectedComponentsWithStats((g > 0.2).astype(np.uint8), connectivity=8)
    tiny = np.zeros(n, bool)
    tiny[1:] = stats[1:, 4] < (_GLINT_FW * fw) ** 2
    keep = cv2.dilate(tiny[labels].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    g = (g * ~keep).astype(np.float32)

    diag.update(texture_sigma=round(sig, 3), shine_share=round(float((g > 0.5).sum() / cost.sum()), 4))
    if float(g.max()) < 0.05:
        diag["reason"] = "no_shine"
        return None, None, diag
    return g, base, diag


def matte_shine(img: np.ndarray, gate: np.ndarray, base_lin: np.ndarray, strength: float) -> np.ndarray:
    """Take the neutral specular light under ``gate`` off ``img`` (full resolution).

    ``gate`` and ``base_lin`` may be at work resolution; they are upsampled.
    Pixels with a zero gate are returned bit-identical.
    """
    H, W = img.shape[:2]
    if gate.shape != (H, W):
        gate = cv2.resize(gate, (W, H), interpolation=cv2.INTER_LINEAR)
        base_lin = cv2.resize(base_lin, (W, H), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(gate > 1e-3)
    if not len(ys):
        return img
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = img[y0:y1, x0:x1].astype(np.float32)
    g = gate[y0:y1, x0:x1]
    lin = _to_linear(crop)
    neutral = np.clip((lin - base_lin[y0:y1, x0:x1]).min(axis=2), 0.0, None)
    k = np.minimum(_MAX_PULL * float(strength) * g * neutral, 0.97 * lin.min(axis=2))
    on = k > 0
    if not on.any():
        return img
    out = img.copy()
    region = out[y0:y1, x0:x1]
    region[on] = _to_srgb(lin[on] - k[on][:, None]).astype(out.dtype)
    return out


def apply_costume_shine(
    img: np.ndarray,
    strength: float,
    face_boxes: Sequence[Tuple[int, int, int, int]],
    segment_classes=None,
    hair_full=None,
    person_mask: Optional[np.ndarray] = None,
    protect: Optional[np.ndarray] = None,
    face_skin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Costume Shine on a float32 BGR [0, 1] image; returns ``(image, diag)``.

    Arguments as :func:`costume_clarity.apply_costume_clarity`.
    """
    diag: Dict[str, Any] = {"applied": False}
    if strength <= 0:
        diag["reason"] = "off"
        return img, diag
    H, W = img.shape[:2]
    fw_full = float(np.median([b[2] for b in face_boxes])) if face_boxes else 0.1 * min(H, W)
    pre = costume_prepass(
        img, face_boxes, segment_classes, hair_full,
        person_mask=person_mask, protect=protect, face_skin=face_skin,
    )
    scale = pre["scale"]
    if pre["n_painted"]:
        diag["painted_faces"] = pre["n_painted"]
    diag.update(pre["diag"])
    mask = pre["mask"]
    if mask is None:
        return img, diag
    work = pre["work"]
    gate, base, gdiag = shine_gate(work, mask, fw_full * scale)
    diag.update(gdiag)
    if gate is None:
        return img, diag
    out = matte_shine(img, gate, base, min(float(strength), 1.0))
    diag["applied"] = out is not img
    if diag["applied"]:
        _logger.info("costume shine: %s", diag)
    return out, diag
