"""Iris Pop: colour and depth back into the iris (classical, no model weights).

Coloured contacts and soft studio light leave the iris flat in a photo: a
dull, greyed disc with the lens pattern washed out. Retouchers fix that with
three small moves, all inside the visible iris only:

1. **Colour.** CIELab chroma is raised with a vibrance curve: pixels far
   below the iris's own 90th-percentile chroma get the most, already-vivid
   ones little, and hue is kept. Chroma is limited to 1.6x the iris's own
   90th percentile so a lens cannot turn neon.
2. **Texture.** Iris and lens pattern detail is raised in log luminance
   (``log Y`` minus a masked local mean, sigma 12% of the iris radius), soft
   limited at 2.5x its own spread. A log-domain detail is the same at any
   exposure, so a darker-lit or darker-skinned face gets the same relative
   lift.
3. **Depth.** A darker limbal ring at the iris edge (linear-light gain down
   to 0.78 at 100) and a soft lift of the lower iris body, where light that
   enters through the cornea lands (gain up to 1.14). Both are gains, never
   offsets, so the eye's own lightness range decides the result.

Where the iris is comes from the image, not only from the landmarks: circle
lenses are wider than the landmark iris (1.2-1.7x on Alex's bunny photos), so
the visible radius is the first ring whose brightest angular sector reaches
70% of the way from the iris to this eye's own white (shared with
``bloodshot_eyes``). The edit is also limited to the eye opening (away from its
rim, where lashes and the waterline sit), and the pupil, lashes and
catchlights are faded out by brightness *relative to the iris body* (darker
than 0.35-0.6x or brighter than 1.8-2.6x its median linear luminance). There
is no absolute intensity threshold.

Strength 0 returns the input unchanged; pixels outside the iris support are
bit-identical. The engine composites a face back through its skin, lips and
eye-sharpen masks, and the eye part of that is scaled down by
``eye_artifact_safety`` (to about 35% on a saturated contact lens), so the
caller passes ``support_out`` and composites the iris support at full weight. An eye whose opening mask is empty (closed or hidden, blanked
by the shared eye-visibility gate) is skipped. This deliberately does not
use ``eye_artifact_safety``'s chroma back-off: that guard backs off on
saturated irises, which is exactly the coloured-contact case this is for;
the chroma cap above bounds the result instead.

Known limits: an iris under a heavy lash shadow reads partly as lash and is
only partly lifted; the visible-radius search needs some white on at least
one side of the iris, otherwise it falls back to 1.2x the landmark radius.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .bloodshot_eyes import _visible_iris_radius
from .utils import bgr_f32_to_lab_f32, lab_f32_to_bgr_f32

__all__ = ["iris_pop", "iris_pop_eye"]

# MediaPipe refine_landmarks iris points: centre, then 4 ring points.
_IRISES: Tuple[Tuple[int, Sequence[int]], ...] = (
    (468, (469, 470, 471, 472)),
    (473, (474, 475, 476, 477)),
)
_MIN_R = 3.0                 # px; smaller irises are left alone
_WINDOW = 4.0                # eye search window half-size, x landmark r
_RIM_LO, _RIM_HI = 0.02, 0.06  # eye-opening rim ramp, x eye width
_WHITE_RING = 1.25           # sclera sample starts this far out, x landmark r
_EDGE_FEATHER = 0.10         # visible-iris edge feather, x visible radius
_DARK_LO, _DARK_HI = 0.35, 0.60     # pupil/lash fade, x body luminance
_BRIGHT_LO, _BRIGHT_HI = 1.8, 2.6   # catchlight fade, x body luminance
_CHROMA_GAIN = 1.0           # max chroma gain at 100 (vibrance-weighted)
_CHROMA_CAP = 1.6            # x the iris's own p90 chroma
_DETAIL_SIGMA = 0.12         # x visible radius
_DETAIL_GAIN = 0.9           # extra detail at 100
_DETAIL_LIMIT = 2.5          # soft limit, x the detail's own spread
_LIMBAL_DARKEN = 0.22        # linear gain loss at the rim, at 100
_LIMBAL_START = 0.78         # ring ramps in from this x visible radius
_LIFT_GAIN = 0.14            # lower-body linear gain, at 100
_LUMA_W = np.array([0.0722, 0.7152, 0.2126], np.float32)  # B, G, R


def _smoothstep(x: np.ndarray, e0: float, e1: float) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _masked_blur(val: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur(val * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)


def _mask_scale(mask: np.ndarray) -> float:
    """255 for a 0-255 mask, 1 for a 0-1 one (same rule as normalize_mask)."""
    if mask.dtype == np.uint8 or float(mask.max()) > 1.5:
        return 255.0
    return 1.0


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def iris_pop_eye(
    canvas: np.ndarray,
    eye_mask: Optional[np.ndarray],
    center: Tuple[float, float],
    radius: float,
    strength: float,
    support_out: Optional[np.ndarray] = None,
    _inplace: bool = False,
) -> np.ndarray:
    """Pop one iris.

    Args:
        canvas: (H, W, 3) BGR float32 [0, 255] (uint8 accepted).
        eye_mask: (H, W) eye-opening mask (sclera + iris), 0-1 or 0-255.
        center: iris centre (x, y) in canvas pixels.
        radius: landmark iris radius in canvas pixels.
        strength: 0-100.
        support_out: optional (H, W) float32 array; the edit's support is
            max-accumulated into it (for compositing the face back).

    Returns:
        float32 BGR; pixels outside this iris's support are unchanged.
    """
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if s <= 0.0 or eye_mask is None or radius < _MIN_R:
        return canvas
    h, w = canvas.shape[:2]
    if eye_mask.shape[:2] != (h, w):
        return canvas
    cx, cy = float(center[0]), float(center[1])
    if not (0 <= cx < w and 0 <= cy < h):
        return canvas

    # Work in a window around the iris (an eye corner sits within about 3.5
    # iris radii of the centre), so a full-frame mask is never normalised.
    win = int(np.ceil(_WINDOW * radius)) + 2
    wx0, wx1 = max(int(cx) - win, 0), min(int(cx) + win + 1, w)
    wy0, wy1 = max(int(cy) - win, 0), min(int(cy) + win + 1, h)
    opening_w = eye_mask[wy0:wy1, wx0:wx1] > 0.5 * _mask_scale(eye_mask)
    if not opening_w.any():
        return canvas
    ys, xs = np.nonzero(opening_w)
    eye_w = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1, 2.0 * radius))
    pad = int(np.ceil(2.2 * radius)) + 2
    x0, x1 = max(int(cx) - pad, 0), min(int(cx) + pad + 1, w)
    y0, y1 = max(int(cy) - pad, 0), min(int(cy) + pad + 1, h)
    crop = canvas[y0:y1, x0:x1].astype(np.float32)
    op = opening_w[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0].astype(np.uint8)
    if op.sum() < 8:
        return canvas

    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.hypot(xx - cx, yy - cy)
    angle = np.arctan2(yy - cy, xx - cx)

    lin = _srgb_to_linear(np.clip(crop, 0.0, 255.0) / 255.0).astype(np.float32)
    lum = lin @ _LUMA_W
    lab = bgr_f32_to_lab_f32(crop)
    L = lab[..., 0]

    dist_rim = cv2.distanceTransform(op, cv2.DIST_L2, 3)
    rim = _smoothstep(dist_rim, _RIM_LO * eye_w, _RIM_HI * eye_w)
    inner = (rim > 0.5) & (op > 0)
    iris_in = inner & (dist < 0.9 * radius)
    if iris_in.sum() < 6:
        return canvas

    # Visible iris radius: where the eye turns to its own white.
    white_zone = inner & (dist >= _WHITE_RING * radius)
    if white_zone.sum() >= 6:
        l_white = float(np.percentile(L[white_zone], 90.0))
        r_vis = _visible_iris_radius(L, dist, angle, iris_in, white_zone, radius, l_white)
    else:
        r_vis = 1.2 * radius
    r_vis = max(r_vis, radius)

    disc = 1.0 - _smoothstep(dist, (1.0 - _EDGE_FEATHER) * r_vis, r_vis)
    geom = disc * rim * (op > 0)

    # Iris body level: linear luminance in the mid annulus.
    body = (geom > 0.5) & (dist > 0.35 * r_vis) & (dist < 0.85 * r_vis)
    if body.sum() < 6:
        return canvas
    y_body = float(np.median(lum[body]))
    if y_body <= 1e-6:
        return canvas
    ratio = lum / y_body
    tone = _smoothstep(ratio, _DARK_LO, _DARK_HI) * (1.0 - _smoothstep(ratio, _BRIGHT_LO, _BRIGHT_HI))
    support = (geom * tone).astype(np.float32)
    core = support > 0.5
    if core.sum() < 6:
        return canvas

    # 2-3. Texture and depth as linear-light gains.
    log_y = np.log(np.maximum(lum, 1e-5))
    sigma = max(_DETAIL_SIGMA * r_vis, 0.8)
    base = _masked_blur(log_y, support, sigma)
    detail = log_y - base
    spread = max(1.4826 * float(np.median(np.abs(detail[core] - np.median(detail[core])))), 0.02)
    lim = _DETAIL_LIMIT * spread
    detail = lim * np.tanh(detail / lim)
    log_gain = _DETAIL_GAIN * s * detail
    ring = _smoothstep(dist / r_vis, _LIMBAL_START, 0.98)
    log_gain += np.log1p(-_LIMBAL_DARKEN * s * ring)
    lower = np.clip((yy - cy) / r_vis, 0.0, 1.0)
    lift = lower * np.exp(-((dist / r_vis - 0.55) ** 2) / (2 * 0.22 ** 2))
    log_gain += np.log1p(_LIFT_GAIN * s * lift)
    gain = np.exp(log_gain).astype(np.float32)
    lin_new = lin * gain[..., None]

    # 1. Colour: vibrance-weighted chroma gain in CIELab, hue kept.
    new_bgr = _linear_to_srgb(lin_new) * 255.0
    lab_new = bgr_f32_to_lab_f32(new_bgr.astype(np.float32))
    a, b = lab_new[..., 1] - 128.0, lab_new[..., 2] - 128.0
    chroma = np.hypot(a, b)
    c_ref = max(float(np.percentile(np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)[core], 90.0)), 2.0)
    vib = 1.0 - 0.7 * np.clip(chroma / c_ref, 0.0, 1.0)
    target = np.minimum(chroma * (1.0 + _CHROMA_GAIN * s * vib), np.maximum(chroma, _CHROMA_CAP * c_ref))
    scale = target / np.maximum(chroma, 1e-4)
    lab_new[..., 1] = 128.0 + a * scale
    lab_new[..., 2] = 128.0 + b * scale
    edited = np.clip(lab_f32_to_bgr_f32(lab_new), 0.0, 255.0)

    out = canvas if _inplace else canvas.astype(np.float32, copy=True)
    m = support[..., None]
    region = out[y0:y1, x0:x1]
    touched = support > 0
    region[touched] = (crop + m * (edited - crop))[touched]
    if support_out is not None and support_out.shape[:2] == (h, w):
        sub_out = support_out[y0:y1, x0:x1]
        np.maximum(sub_out, support, out=sub_out)
    return out


def _pair(landmarks: Any, size: Tuple[int, int]):
    h, w = size
    pts = getattr(landmarks, "landmark", landmarks)
    try:
        if len(pts) < 478:
            return []
    except TypeError:
        return []
    out = []
    for c, ring in _IRISES:
        cx, cy = pts[c].x * w, pts[c].y * h
        r = float(np.mean([np.hypot(pts[i].x * w - cx, pts[i].y * h - cy) for i in ring]))
        out.append(((cx, cy), r))
    return out


def iris_pop(
    canvas: np.ndarray,
    landmarks: Any,
    strength: float,
    eye_masks: Sequence[Optional[np.ndarray]],
    support_out: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Pop both irises.

    Args:
        canvas: (H, W, 3) BGR float32 [0, 255], the face ROI.
        landmarks: 478-point MediaPipe landmarks normalised to this ROI
            (iris points 468-477). ``None`` returns ``canvas`` unchanged.
        strength: 0-100. 0 returns ``canvas`` unchanged.
        eye_masks: eye-opening masks (left, right) in any order; each iris
            uses the mask that covers its centre.
        support_out: optional (H, W) float32 array the edits' support is
            max-accumulated into.

    Returns:
        float32 BGR; pixels outside both irises are unchanged.
    """
    if float(strength) <= 0.0 or landmarks is None:
        return canvas
    h, w = canvas.shape[:2]
    masks = [m for m in eye_masks if m is not None and m.shape[:2] == (h, w)]
    out = None
    for (cx, cy), r in _pair(landmarks, (h, w)):
        xi, yi = int(round(cx)), int(round(cy))
        if not (0 <= xi < w and 0 <= yi < h):
            continue
        cover = [m for m in masks if m[yi, xi] > 0.5 * _mask_scale(m)]
        if not cover:
            continue  # closed, hidden or gated eye
        if out is None:  # one float32 copy for both eyes
            out = canvas.astype(np.float32, copy=True)
        out = iris_pop_eye(out, cover[0], (cx, cy), r, strength, support_out, _inplace=True)
    return canvas if out is None else out
