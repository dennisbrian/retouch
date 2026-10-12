"""Nose bridge highlight (opt-in ``nose_highlight``).

A thin, soft highlight down the nose bridge is the one "lift" in the
porcelain cosplay look the engine could not make. The existing ops that
claim to do it never reached the bridge:

* ``dodge_burn`` brightens ``regions.nose_bridge``, but that region is a
  polygon of four midline landmarks (168, 6, 197, 195). The points are
  nearly collinear, so the mask is a sliver: about 50-80 pixels at
  0.2-0.25 peak on Alex's bunny photos, and ``dodge_burn`` at 100 moved the
  bridge by 0 L.
* ``sculpt`` reshapes the broad form band from a light direction; on the
  same photos it darkened the bridge (-4 to -12 L) instead of lifting it.

This op draws its own ridge from the landmarks (between the brows down to
just above the tip), a Gaussian stripe about a tenth of the inter-eye
distance wide, faded in at the top and out toward the tip, limited to the
skin mask so hair, brows and glasses are left alone.

The lift is an exposure gain in linear light with a shoulder::

    lin' = lin + g * mask * lin * (1 - lin)

so every skin tone gets the same kind of lift (about +0.4 to +0.5 EV on
the ridge at 100), hue and saturation stay those of the face's own skin,
and bright paint rolls off instead of clipping. There is no intensity
threshold anywhere, so it acts the same way on darker and lighter skin.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .utils import get_points

# Ridge landmarks, top to bottom: between the brows (168), upper bridge (6),
# mid bridge (197, 195), lower bridge (5) and just above the tip (4).
RIDGE = (168, 6, 197, 195, 5, 4)

# Linear-light gain on the ridge centre at strength 100. Tuned on Alex's
# two bunny photos against 13 finished reference edits (bridge minus bridge
# sides, median +16 L on the 0-255 Lab scale): 100 lands the natural recipe
# near that median; start around 50 for a subtle lift.
_GAIN = 0.55
# Stripe half-width (Gaussian sigma) as a fraction of the inter-eye distance.
_SIGMA_IED_FRAC = 0.045
_MIN_SIGMA_PX = 1.0
# Fraction of the ridge length over which the highlight fades in at the top
# (between the brows) and out toward the tip.
_FADE_TOP = 0.25
_FADE_BOTTOM = 0.30


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ridge_mask(
    points: np.ndarray, shape: tuple, ied: float,
) -> np.ndarray:
    """Soft stripe along a top-to-bottom polyline of ridge points.

    Args:
        points: (N, 2) float pixel coordinates, top first.
        shape: (H, W) of the output.
        ied: inter-eye distance in pixels (sets the stripe width).

    Returns:
        (H, W) float32 mask in [0, 1]: a Gaussian across the ridge, faded in
        over the top ``_FADE_TOP`` and out over the bottom ``_FADE_BOTTOM``
        of the ridge length. Zero outside a few sigma of the ridge.
    """
    h, w = shape[:2]
    out = np.zeros((h, w), np.float32)
    pts = np.asarray(points, np.float32).reshape(-1, 2)
    if len(pts) < 2:
        return out
    seg = np.diff(pts, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    total = float(seg_len.sum())
    if total < 1e-3:
        return out
    sigma = max(_MIN_SIGMA_PX, float(ied) * _SIGMA_IED_FRAC)
    pad = 3.0 * sigma
    x0 = int(max(0, np.floor(pts[:, 0].min() - pad)))
    x1 = int(min(w, np.ceil(pts[:, 0].max() + pad) + 1))
    y0 = int(max(0, np.floor(pts[:, 1].min() - pad)))
    y1 = int(min(h, np.ceil(pts[:, 1].max() + pad) + 1))
    if x1 <= x0 or y1 <= y0:
        return out
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)

    best_d2 = np.full(yy.shape, np.inf, np.float32)
    best_s = np.zeros(yy.shape, np.float32)  # arc length of the nearest point
    start = 0.0
    for p, v, ln in zip(pts[:-1], seg, seg_len):
        if ln < 1e-6:
            continue
        t = np.clip(((xx - p[0]) * v[0] + (yy - p[1]) * v[1]) / (ln * ln), 0.0, 1.0)
        d2 = (xx - p[0] - t * v[0]) ** 2 + (yy - p[1] - t * v[1]) ** 2
        closer = d2 < best_d2
        best_d2 = np.where(closer, d2, best_d2)
        best_s = np.where(closer, start + t * ln, best_s)
        start += float(ln)

    u = best_s / total
    along = _smoothstep(u / _FADE_TOP) * _smoothstep((1.0 - u) / _FADE_BOTTOM)
    across = np.exp(-best_d2 / (2.0 * sigma * sigma))
    out[y0:y1, x0:x1] = (across * along).astype(np.float32)
    return out


def add_nose_highlight(
    canvas: np.ndarray,
    landmarks: Any,
    strength: float,
    ied: float,
    skin_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Brighten a soft stripe down the nose bridge.

    Args:
        canvas: (H, W, 3) BGR float32 [0, 255] (uint8 accepted), the face ROI.
        landmarks: MediaPipe-style landmark list normalised to this ROI.
            ``None`` returns ``canvas`` unchanged.
        strength: 0-100. 0 returns ``canvas`` unchanged.
        ied: inter-eye distance in ROI pixels.
        skin_mask: optional (H, W) skin mask, 0-1 or 0-255; the highlight is
            limited to it so hair, brows or glasses over the bridge stay as
            they are.

    Returns:
        (H, W, 3) float32 BGR [0, 255].
    """
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if s <= 0.0 or landmarks is None or not ied or ied <= 0:
        return canvas
    h, w = canvas.shape[:2]
    try:
        pts = get_points(landmarks, list(RIDGE), w, h).astype(np.float32)
    except (IndexError, AttributeError, TypeError):
        return canvas
    mask = ridge_mask(pts, (h, w), ied)
    if skin_mask is not None:
        sk = skin_mask.astype(np.float32)
        if sk.max() > 1.5:
            sk = sk / 255.0
        mask = mask * np.clip(sk, 0.0, 1.0)
    if not np.any(mask > 1e-3):
        return canvas

    ys, xs = np.nonzero(mask > 1e-3)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    out = canvas.astype(np.float32, copy=True)
    roi = np.clip(out[y0:y1, x0:x1], 0.0, 255.0) / 255.0
    lin = _srgb_to_linear(roi)
    g = (_GAIN * s * mask[y0:y1, x0:x1])[:, :, None]
    lin = lin + g * lin * (1.0 - lin)
    out[y0:y1, x0:x1] = _linear_to_srgb(lin) * 255.0
    return out.astype(np.float32)
