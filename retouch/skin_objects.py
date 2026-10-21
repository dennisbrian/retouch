"""Cut non-skin objects (clips, frames, props, stray hair) out of a skin mask.

The face skin mask is built from landmark polygons, optionally refined by a
segmenter; neither looks at colour, so a hair clip at the temple, glasses
frames, a prop held to the jaw or a lock of bangs inside the face oval all
count as skin and every skin op (smoothing, whitening, albedo evening,
redness evening, ...) repaints them. Golden case ``hair_clip_temple``: 40-70%
of a pink clip changed under 20 of the 42 skin ops.

An object pixel is one whose chromaticity (a*/L*, b*/L*) sits far from this
face's own skin: more than ``_OUTLIER_Z`` robust spreads from the skin
median. Chromaticity divided by L* is unchanged by shading and exposure, so
shadowed skin, darker skin and lighter skin all stay in; the spread is the
face's own, with a floor proportional to its own chroma. Only outlier
regions that touch the outside of the mask are cut: an object enters the
face from outside it (hair, a clip, frames reaching the temples, a prop),
while blush, eye and lip makeup, a peach nose on white face paint, freckles
and moles sit inside the face outline (or around the eye and lip holes) and
stay part of it. Objects also have a hard edge (ordinary skin within 3-6.5%
of the eye distance of them); makeup that reaches the face outline fades
out over a wider band and is kept.
"""

from __future__ import annotations

import cv2
import numpy as np

# Robust z (in units of the face's own chromaticity spread) above which a
# pixel cannot be this face's skin. Skin blush/redness reaches ~3-4.
_OUTLIER_Z = 6.0
# Spread floor as a share of the skin's median chromaticity magnitude, and an
# absolute floor for near-neutral (painted) skin, in a*/L* units.
_SPREAD_FLOOR_REL = 0.10
_SPREAD_FLOOR_ABS = 0.02
# Hard-edge test: in a ring 3-6.5% of the inter-eye distance outside a
# candidate (past JPEG chroma bleed), the median z must already be below
# _RIM_SKIN_Z. Measured: planted clips 0.8-1.0; under-eye makeup and wig
# wisps on Alex's cosplay frames 1.5-3.5.
_EDGE_RING_IN = 0.03
_EDGE_RING_OUT = 0.065
_RIM_SKIN_Z = 1.2
# Minimum skin core (pixels) for the statistics to mean anything.
_MIN_CORE_PX = 200


def foreign_object_mask(img_bgr: np.ndarray, skin: np.ndarray, ied: float) -> np.ndarray:
    """Soft (H, W) float32 mask of non-skin objects inside ``skin``.

    ``skin`` is a float mask in [0, 1] at the size of ``img_bgr``. Returns
    zeros when the skin is too small to measure.
    """
    h, w = skin.shape[:2]
    out = np.zeros((h, w), np.float32)
    if img_bgr is None or img_bgr.shape[:2] != (h, w):
        return out
    ied = float(ied) if ied and ied > 0 else 0.25 * float(min(h, w))

    src = img_bgr if img_bgr.dtype == np.uint8 else np.clip(img_bgr, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[..., 0] * (100.0 / 255.0)
    # Chromaticity: a*, b* per unit lightness (floor keeps near-black stable).
    denom = np.maximum(L, 8.0)
    x = (lab[..., 1] - 128.0) / denom
    y = (lab[..., 2] - 128.0) / denom

    inside = skin > 0.5
    r = max(1, int(round(0.03 * ied)))
    core = cv2.erode(inside.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0
    if core.sum() < _MIN_CORE_PX:
        return out

    xs, ys = x[core], y[core]
    mx, my = float(np.median(xs)), float(np.median(ys))
    floor = max(_SPREAD_FLOOR_ABS, _SPREAD_FLOOR_REL * float(np.hypot(mx, my)))
    sx = max(floor, 1.4826 * float(np.median(np.abs(xs - mx))))
    sy = max(floor, 1.4826 * float(np.median(np.abs(ys - my))))
    z = np.hypot((x - mx) / sx, (y - my) / sy)

    cand = ((z > _OUTLIER_Z) & (skin > 0.05)).astype(np.uint8)
    if not cand.any():
        return out

    # Keep candidate regions that touch the outside of the face: the part of
    # the non-skin area connected to the crop border. The eye, brow and lip
    # holes are not part of it, so makeup around them stays skin.
    non_skin = (skin < 0.2).astype(np.uint8)
    nl, hole_labels = cv2.connectedComponents(non_skin, connectivity=8)
    border = np.unique(np.concatenate([
        hole_labels[0, :], hole_labels[-1, :], hole_labels[:, 0], hole_labels[:, -1]]))
    exterior = np.isin(hole_labels, border[border > 0]).astype(np.uint8)
    outside = cv2.dilate(exterior, np.ones((3, 3), np.uint8)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    min_area = max(12, int((0.06 * ied) ** 2))
    touching = np.unique(labels[outside & (labels > 0)])
    keep = np.zeros(n, bool)
    r_in = max(1.5, _EDGE_RING_IN * ied)
    r_out = max(r_in + 1.5, _EDGE_RING_OUT * ied)
    pad = int(np.ceil(r_out)) + 1
    for lab_id in touching:
        if stats[lab_id, cv2.CC_STAT_AREA] < min_area:
            continue
        x0, y0, cw, ch = (int(v) for v in stats[lab_id, :4])
        x1, y1 = max(0, x0 - pad), max(0, y0 - pad)
        x2, y2 = min(w, x0 + cw + pad), min(h, y0 + ch + pad)
        comp = (labels[y1:y2, x1:x2] == lab_id).astype(np.uint8)
        dist = cv2.distanceTransform(1 - comp, cv2.DIST_L2, 3)
        ring = (dist > r_in) & (dist <= r_out) & inside[y1:y2, x1:x2]
        if ring.sum() < 4:
            continue
        # An object has a hard edge: just past its rim the skin is already
        # ordinary skin. Makeup and hair wisps fade out over a wider band.
        if float(np.median(z[y1:y2, x1:x2][ring])) < _RIM_SKIN_Z:
            keep[lab_id] = True

    if not keep.any():
        return out
    obj = keep[labels].astype(np.uint8)

    grow = max(1, int(round(0.015 * ied)))
    obj = cv2.dilate(obj, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1)))
    soft = cv2.GaussianBlur(obj.astype(np.float32), (0, 0), max(0.8, 0.5 * grow))
    return np.clip(np.maximum(soft, obj.astype(np.float32)), 0.0, 1.0)


def exclude_foreign_objects(regions, img_bgr: np.ndarray, ied: float) -> None:
    """Remove non-skin objects from ``regions.skin`` in place."""
    skin = getattr(regions, "skin", None)
    if skin is None or img_bgr is None:
        return
    skin = np.asarray(skin, np.float32)
    obj = foreign_object_mask(img_bgr, skin, ied)
    if obj.any():
        regions.skin = (skin * (1.0 - obj)).astype(np.float32)
