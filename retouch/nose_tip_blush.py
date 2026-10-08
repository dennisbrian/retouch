"""Nose-tip blush (opt-in ``nose_tip_blush``).

A soft pink wash on the tip of the nose (and a little onto the wings) is a
staple of the cute cosplay / douyin makeup look. The engine already had a
``nose_blush`` checkbox, but it never really showed:

* it only ran inside the cheek blush, so it did nothing unless Blush
  Strength was also above 0;
* it drew a disc of radius ``0.072 x face width`` at 0.6, then blurred it
  with the cheek blur (kernel ``0.36 x face width``), so the tip got well
  under half of the cheek's colour: with cosplay_clear_v1 at blush 30 the
  tip moved by at most +2 a* (0-255 Lab) on Alex's two bunny photos, an
  invisible change.

This op draws its own tip-shaped mask from the landmarks and tints it like
a pigment instead of adding a fixed a* offset:

* **Mask.** An anisotropic Gaussian centred between landmarks 4 and 1 (the
  tip), its axis along the nose (168 -> 1), sized from the alar width
  (129 <-> 358). It reaches a little up the lower bridge and onto the
  wings but falls off fast below the tip, so the philtrum and lips stay
  clean; the lips mask is also cut out and the result is limited to the
  skin mask, so hair, glasses or a mask strap over the nose are left alone.
* **Nostrils.** Pixels much darker than the tip's own skin (linear
  luminance under ~0.35-0.6 x the masked median) are faded out, so the
  nostril openings don't turn brown-red. The gate is a ratio to the face's
  own skin, not a level, so it behaves the same on every skin tone.
* **Tint.** In linear light, each channel is multiplied by a pink
  reflectance (red kept, green and blue absorbed a little, as a cream
  blush does), then most of the lost luminance is given back. The colour
  change is a *ratio*, so it scales with the skin's own lightness: darker
  skin gets a proportionate, never chalky, shift and nothing is ever pushed
  toward a fixed pink. White face paint takes the tint well, which is the
  intended look on painted bunny faces.

No absolute intensity thresholds anywhere (see CLAUDE.md, Tone-Invariance).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .utils import get_points, normalize_mask

# Landmarks: just above the tip (4), the tip (1), the bridge top (168, sets
# the nose axis) and the outer alar points (129, 358, set the width).
_TIP_UPPER = 4
_TIP = 1
_AXIS_TOP = 168
_ALAR = (129, 358)

# Pink reflectance at the mask peak and strength 100, as (B, G, R) linear
# multipliers. Red is kept; green absorbs more than blue, giving a rosy pink
# (not orange, not magenta). Tuned on Alex's two bunny photos so 100 moves
# the tip about +10 a* on natural skin and 50 is a soft, everyday blush.
_TINT_BGR = np.array([0.84, 0.76, 1.0], np.float32)
# Share of the lost luminance given back: 1.0 is a pure colour change, 0.0
# is a plain multiply (pigment darkens). A blush darkens only a touch.
_LUMA_KEEP = 0.85
# Mask shape, as fractions of the alar width.
_SIGMA_ACROSS = 0.34
_SIGMA_UP = 0.36   # up the lower bridge
_SIGMA_DOWN = 0.17  # toward the philtrum: falls off fast
# Nostril fade: linear-luminance ratio to the tip's masked median skin.
_DARK_LO = 0.35
_DARK_HI = 0.60
_LUMA_W = np.array([0.0722, 0.7152, 0.2126], np.float32)  # B, G, R


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def tip_mask(landmarks: Any, shape: tuple, ied: float) -> np.ndarray:
    """Soft nose-tip mask from the landmarks, (H, W) float32 in [0, 1].

    Zero everywhere if the landmarks can't be read.
    """
    h, w = shape[:2]
    out = np.zeros((h, w), np.float32)
    try:
        pts = get_points(
            landmarks, [_TIP_UPPER, _TIP, _AXIS_TOP, _ALAR[0], _ALAR[1]], w, h,
        ).astype(np.float32)
    except (IndexError, AttributeError, TypeError):
        return out
    upper, tip, top, al_l, al_r = pts
    centre = 0.5 * (upper + tip)
    axis = tip - top
    n = float(np.hypot(axis[0], axis[1]))
    if n < 1e-3:
        return out
    axis = axis / n  # points down the nose
    perp = np.array([-axis[1], axis[0]], np.float32)
    width = float(np.hypot(*(al_r - al_l)))
    if width < 1e-3:
        width = 0.55 * float(ied or 0.0)
    if width < 2.0:
        return out
    s_x = max(1.0, _SIGMA_ACROSS * width)
    s_up = max(1.0, _SIGMA_UP * width)
    s_dn = max(1.0, _SIGMA_DOWN * width)
    pad = 3.0 * max(s_x, s_up)
    x0 = int(max(0, np.floor(centre[0] - pad)))
    x1 = int(min(w, np.ceil(centre[0] + pad) + 1))
    y0 = int(max(0, np.floor(centre[1] - pad)))
    y1 = int(min(h, np.ceil(centre[1] + pad) + 1))
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx = xx - centre[0]
    dy = yy - centre[1]
    along = dx * axis[0] + dy * axis[1]  # > 0 below the centre
    across = dx * perp[0] + dy * perp[1]
    s_al = np.where(along > 0, s_dn, s_up)
    g = np.exp(-0.5 * ((across / s_x) ** 2 + (along / s_al) ** 2))
    g[g < 1e-3] = 0.0
    out[y0:y1, x0:x1] = g.astype(np.float32)
    return out


def add_nose_tip_blush(
    canvas: np.ndarray,
    landmarks: Any,
    strength: float,
    ied: float,
    skin_mask: Optional[np.ndarray] = None,
    lips_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Tint the nose tip a soft pink.

    Args:
        canvas: (H, W, 3) BGR float32 [0, 255] (uint8 accepted), the face ROI.
        landmarks: MediaPipe-style landmark list normalised to this ROI.
            ``None`` returns ``canvas`` unchanged.
        strength: 0-100. 0 returns ``canvas`` unchanged.
        ied: inter-eye distance in ROI pixels (fallback width only).
        skin_mask: optional (H, W) skin mask, 0-1 or 0-255; the tint is
            limited to it.
        lips_mask: optional (H, W) lips mask, cut out of the tint.

    Returns:
        (H, W, 3) float32 BGR [0, 255]; pixels outside the mask are
        bit-identical to the input.
    """
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if s <= 0.0 or landmarks is None:
        return canvas
    h, w = canvas.shape[:2]
    mask = tip_mask(landmarks, (h, w), ied)
    sk = normalize_mask(skin_mask)
    if sk is not None and sk.shape[:2] == (h, w):
        mask = mask * np.clip(sk, 0.0, 1.0)
    lp = normalize_mask(lips_mask)
    if lp is not None and lp.shape[:2] == (h, w):
        mask = mask * (1.0 - np.clip(lp, 0.0, 1.0))
    if not np.any(mask > 1e-3):
        return canvas

    ys, xs = np.nonzero(mask > 1e-3)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out = canvas.astype(np.float32, copy=True)
    m = mask[y0:y1, x0:x1]
    lin = _srgb_to_linear(np.clip(out[y0:y1, x0:x1], 0.0, 255.0) / 255.0)
    y_lin = lin @ _LUMA_W

    # Nostril fade, relative to the tip's own skin.
    core = m > 0.5 * float(m.max())
    med = float(np.median(y_lin[core])) if np.any(core) else float(np.median(y_lin))
    if med > 1e-6:
        m = m * _smoothstep((y_lin / med - _DARK_LO) / (_DARK_HI - _DARK_LO))

    k = (s * m)[:, :, None]
    tinted = lin * (1.0 - k * (1.0 - _TINT_BGR[None, None, :]))
    y_t = tinted @ _LUMA_W
    gain = np.where(y_t > 1e-6, y_lin / np.maximum(y_t, 1e-6), 1.0) ** _LUMA_KEEP
    tinted = tinted * gain[:, :, None]
    patch = _linear_to_srgb(tinted) * 255.0
    keep = (m <= 1e-4)[:, :, None]
    out[y0:y1, x0:x1] = np.where(keep, out[y0:y1, x0:x1], patch)
    return out.astype(np.float32)
