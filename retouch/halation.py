"""Point-light halation (film-style red-orange glow around lamps).

Colour negative film without an anti-halation layer (CineStill 800T is the
well-known one) shows a red-orange glow around bare bulbs, neon tubes and
street lamps that sit against darker surroundings. The glow comes from light
bouncing off the film base back into the red-sensitive layer, so it is always
red-orange, whatever colour the light is, and it is strongest where a light is
overexposed next to something dark.

``grading.ColorGrader._add_halation`` (the original effect, still used by
every recipe that sets a plain ``halation`` number) glows any pixel whose grey
value is above 200 with a fixed 15 px blur. At 26 MP a 15 px glow is close to
invisible, and the grey test also catches lit pale skin, white face paint and
white wigs, which then pick up an orange halo.

This version is opted into with ``"halation": {"mode": "point_light", ...}``:

* Sources are pixels near the encoding ceiling (the brightest channel at
  0.94-0.995 of full scale: a level of the file format, not of anyone's skin)
  that are also several times brighter, in linear light, than their own
  surroundings. A lamp against a dark street qualifies; a lit white wig or a
  pale face, which sit well below the ceiling and close to their surroundings,
  does not. Neither does a large blown window: it is most of its own
  surround, so it never paints an orange band (small lights beside it do).
* Skin never acts as a source (the engine passes its accumulated skin mask),
  so flash on a pale or painted face cannot halo. Glow from a nearby lamp
  still falls on skin, as it would on film.
* The glow radius is a share of the photo's long side, so the effect looks the
  same at 2000 px and at 26 MP. The glow is built on a copy at most 1024 px
  long (it is a low-frequency signal) and resized back.
* It is added as light (linear-light screen), so pixels with no glow are left
  bit-identical.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .white_balance import linear_to_srgb, srgb_to_linear

# Brightest channel (encoded, 0-1) where a pixel starts / fully counts as a
# light source. The only absolute level in here: it is the file's ceiling.
_SOURCE_LEVEL_START = 0.94
_SOURCE_LEVEL_FULL = 0.995
# Source luminance over its surround (linear light) where it starts / fully
# counts. A light has to stand well clear of what is around it.
_RATIO_START = 2.0
_RATIO_FULL = 4.0
# Surround and glow sizes, as shares of the long side.
_SURROUND_SIGMA = 0.03
_GLOW_SIGMA_NEAR = 0.005
_GLOW_SIGMA_FAR = 0.016
_GLOW_FAR_WEIGHT = 0.4
# Glow gain: a small lamp spreads its energy over a wide blur, so the blurred
# field is lifted before it is added. Capped at 1 (no more than the source).
_GLOW_GAIN = 4.0
# Red-sensitive-layer colour of the glow, linear light (B, G, R).
_GLOW_COLOUR_BGR = np.array([0.06, 0.28, 1.0], dtype=np.float32)
_WORK_MAX_DIM = 1024


def _smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def halation_sources(
    img_f: np.ndarray, skin_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Return the source weight map (H, W) in [0, 1] for a float BGR image."""
    lin = srgb_to_linear(img_f)
    y = 0.0722 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.2126 * lin[..., 2]
    long_side = max(img_f.shape[:2])
    sigma = max(1.0, _SURROUND_SIGMA * long_side)
    surround = cv2.GaussianBlur(y, (0, 0), sigma)
    ratio = y / np.maximum(surround, 1e-4)
    level = img_f.max(axis=2)
    w = _smoothstep(level, _SOURCE_LEVEL_START, _SOURCE_LEVEL_FULL)
    w = w * _smoothstep(ratio, _RATIO_START, _RATIO_FULL)
    if skin_mask is not None:
        skin = np.clip(np.asarray(skin_mask, dtype=np.float32), 0.0, 1.0)
        if skin.shape != w.shape:
            skin = cv2.resize(skin, (w.shape[1], w.shape[0]), interpolation=cv2.INTER_LINEAR)
        # Grow skin a little so a face's lit rim never seeds a halo.
        grow = max(1, int(round(0.004 * long_side)))
        skin = cv2.dilate(skin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
        w = w * (1.0 - skin)
    return w.astype(np.float32)


def apply_point_light_halation(
    img: np.ndarray,
    intensity: float,
    skin_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Add point-light halation. Accepts uint8 or float32 [0, 1] BGR.

    Returns the same dtype. ``intensity`` 0 returns the input unchanged.
    """
    if intensity <= 0:
        return img
    is_float = img.dtype == np.float32
    img_f = img if is_float else img.astype(np.float32) / 255.0
    h, w = img_f.shape[:2]
    scale = min(1.0, _WORK_MAX_DIM / float(max(h, w)))
    if scale < 1.0:
        small = cv2.resize(img_f, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
                           interpolation=cv2.INTER_AREA)
    else:
        small = img_f
    skin_small = None
    if skin_mask is not None:
        skin_small = cv2.resize(np.asarray(skin_mask, dtype=np.float32),
                                (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
    src = halation_sources(small, skin_small)
    if not np.any(src > 1e-4):
        return img
    lin_small = srgb_to_linear(small)
    y_small = 0.0722 * lin_small[..., 0] + 0.7152 * lin_small[..., 1] + 0.2126 * lin_small[..., 2]
    energy = src * np.clip(y_small, 0.0, 1.0)
    long_small = max(small.shape[:2])
    near = cv2.GaussianBlur(energy, (0, 0), max(0.8, _GLOW_SIGMA_NEAR * long_small))
    far = cv2.GaussianBlur(energy, (0, 0), max(1.0, _GLOW_SIGMA_FAR * long_small))
    glow = (1.0 - _GLOW_FAR_WEIGHT) * near + _GLOW_FAR_WEIGHT * far
    glow = np.clip(glow * _GLOW_GAIN, 0.0, 1.0)
    if scale < 1.0:
        glow = cv2.resize(glow, (w, h), interpolation=cv2.INTER_LINEAR)
    glow = np.clip(glow * float(intensity), 0.0, 1.0)
    touched = glow > 1e-4
    if not np.any(touched):
        return img
    lin = srgb_to_linear(img_f[touched])
    add = glow[touched][:, None] * _GLOW_COLOUR_BGR[None, :]
    lin = 1.0 - (1.0 - lin) * (1.0 - add)
    out_f = img_f.copy()
    out_f[touched] = linear_to_srgb(lin)
    if is_float:
        return out_f.astype(np.float32)
    out = img.copy()
    out[touched] = np.clip(np.round(out_f[touched] * 255.0), 0, 255).astype(np.uint8)
    return out
