"""Flash red-eye removal (classical, no model weights).

On-camera flash light bounces off the retina and comes back through a wide
pupil, so the pupil glows red. The fix is old and well understood: find the
red pupil and take the red out of it. The work here is in *not* touching
things that are red on purpose (red contact lenses, red eyeliner, a red wig
strand across the eye, a red background behind a false face detection).

Per eye, on a small crop around the iris landmarks (MediaPipe 468-477):

1. **Redness.** ``(R - max(G, B)) / (R + k)`` with a hue gate. It is a ratio,
   so scaling a pixel's brightness does not change it (a dark red pupil and a
   bright one score the same), and ``k`` is a fraction of the face's own skin
   level so near-black pupil noise cannot read as red. The hue gate keeps
   orange and brown (``G`` well above ``B``) out.
2. **Is this eye red-eyed?** Only when the pupil core is red *and* clearly
   redder than the skin around the eye (a margin above the face's own
   baseline, never an absolute threshold, see CLAUDE.md Tone-Invariance).
   A red contact lens leaves a dark pupil core, so it fails the first test; a
   red background or wig fills the surround too, so it fails the second.
3. **Region.** The red pixels connected to the pupil core, inside the iris
   disc and the eye opening, with holes (the catchlight) filled.
4. **Fix.** Each pupil pixel goes to a neutral dark grey at the level of its
   smallest channel, feathered at the edge. The catchlight becomes a neutral
   white at its own brightness and the iris (not red) is left as it was, so
   the eye keeps its sparkle and colour.

Known limits: "white eye" (a pupil blown to pure white) and the gold or
green eye-shine animals give are left alone; a pupil smaller than a few
pixels is skipped.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .parsing import LEFT_EYE, LEFT_IRIS, RIGHT_EYE, RIGHT_IRIS

__all__ = ["remove_red_eye", "remove_red_eye_faces", "redness_map"]

_MIN_IRIS_R = 4.0        # px; smaller irises carry too few pixels to judge
_ROI_R = 2.6             # crop half-size in iris radii
_DISC_R = 1.15           # red region may not leave the iris disc (x iris r)
_CORE_R = 0.40           # pupil core radius (x iris r)
_SKIN_IN, _SKIN_OUT = 2.0, 2.6   # skin ring around the eye (x iris r)
_K_FRAC = 0.4           # ratio softener, as a fraction of the skin level
_HUE_MAX = 22.0          # degrees either side of pure red
_RED_LO, _RED_HI = 0.15, 0.35    # pixel gate on (redness - skin baseline)
_CORE_MARGIN = 0.10      # pupil core must beat the skin baseline by this
_SURROUND_RED = 0.35     # share of the skin ring allowed to be pupil-red
_EDGE_RED = 0.50         # share of the iris-disc edge allowed to be pupil-red
_WEAK_LO = 0.06          # hysteresis floor for the pupil rim
_GROW_R = 0.15           # rim band the region may grow into (x iris r)
_DARKEN = 0.15           # extra darkening where the pupil glowed red
_HL_LO, _HL_HI = 0.45, 0.75      # catchlight: min(G, B) vs the skin level


def _smoothstep(x: np.ndarray, e0: float, e1: float) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def redness_map(bgr: np.ndarray, k: float) -> np.ndarray:
    """Hue-gated red ratio in ``[-1, 1]`` for a float32 BGR image.

    ``(R - max(G, B)) / (R + k)`` times a gate that is 1 for pure red and
    falls to 0 at ``_HUE_MAX`` degrees toward orange or magenta.
    """
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    mx = np.maximum(g, b)
    ratio = (r - mx) / (r + float(k))
    # HSV hue for red-dominant pixels: 60 * (G - B) / (R - min).
    span = np.maximum(r - np.minimum(g, b), 1e-3)
    hue = 60.0 * np.abs(g - b) / span
    hue_gate = 1.0 - _smoothstep(hue, 0.6 * _HUE_MAX, _HUE_MAX)
    return np.where(ratio > 0, ratio * hue_gate, ratio).astype(np.float32)


def _pts(landmarks: Any, idx: Sequence[int], w: int, h: int) -> np.ndarray:
    lm = landmarks.landmark
    return np.array([[lm[i].x * w, lm[i].y * h] for i in idx], dtype=np.float32)


def _fix_eye(
    img: np.ndarray, landmarks: Any, iris_idx: Sequence[int], eye_idx: Sequence[int],
    s: float,
) -> Dict[str, float]:
    """Correct one eye of ``img`` (float32 BGR) in place. Returns diagnostics."""
    H, W = img.shape[:2]
    d = {"fixed": 0.0, "core_redness": 0.0, "skin_redness": 0.0, "area_px": 0.0}
    iris = _pts(landmarks, iris_idx, W, H)
    c = iris[0]
    r = float(np.linalg.norm(iris[1:] - c, axis=1).mean())
    if not np.isfinite(r) or r < _MIN_IRIS_R:
        return d

    half = int(np.ceil(_ROI_R * r)) + 2
    x0, y0 = int(max(0, np.floor(c[0]) - half)), int(max(0, np.floor(c[1]) - half))
    x1, y1 = int(min(W, np.floor(c[0]) + half + 1)), int(min(H, np.floor(c[1]) + half + 1))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return d
    roi = img[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    cx, cy = float(c[0] - x0), float(c[1] - y0)
    yy, xx = np.mgrid[0:rh, 0:rw].astype(np.float32)
    dist = np.hypot(xx - cx, yy - cy) / r

    opening = np.zeros((rh, rw), np.uint8)
    poly = _pts(landmarks, eye_idx, W, H) - np.array([x0, y0], np.float32)
    cv2.fillPoly(opening, [np.round(poly).astype(np.int32)], 1)
    # One pixel of slack so the lid edge of a red pupil is caught too.
    opening = cv2.dilate(opening, np.ones((3, 3), np.uint8)).astype(bool)

    skin_ring = (dist >= _SKIN_IN) & (dist <= _SKIN_OUT) & ~opening
    if int(skin_ring.sum()) < 8:
        return d
    skin_level = float(np.median(roi[skin_ring].max(axis=1)))
    red = redness_map(roi, max(4.0, _K_FRAC * skin_level))
    skin_base = float(np.median(red[skin_ring]))
    core = (dist <= _CORE_R) & opening
    if int(core.sum()) < 3:
        return d
    core_red = float(np.median(red[core]))
    d["core_redness"], d["skin_redness"] = core_red, skin_base

    rel = red - skin_base
    # Decision: a red pupil core, clearly above this face's own skin, and a
    # surround that is not itself pupil-red (background / wig / false face).
    if core_red - skin_base < _CORE_MARGIN:
        return d
    if float((rel[skin_ring] > _RED_LO).mean()) > _SURROUND_RED:
        return d

    score = _smoothstep(rel, _RED_LO, _RED_HI)
    allowed = (dist <= _DISC_R) & opening
    cand = ((score > 0.5) & allowed).astype(np.uint8)
    n, labels = cv2.connectedComponents(cand, connectivity=8)
    if n <= 1:
        return d
    keep = np.unique(labels[core & (cand > 0)])
    keep = keep[keep > 0]
    if keep.size == 0:
        return d
    strong = np.isin(labels, keep).astype(np.uint8)
    # A red pupil sits inside the iris. Red that runs out past the iris edge
    # on most of its circumference is not a pupil (a red blob on a false face
    # detection, a red lens over the whole eye).
    edge = (dist >= 0.95) & (dist <= _DISC_R) & opening
    if int(edge.sum()) >= 3 and (
        float(strong[edge].mean()) > _EDGE_RED
        or float(np.median(rel[edge])) > core_red - skin_base - _CORE_MARGIN
    ):
        return d
    # Hysteresis: the pink rim of a red pupil scores lower than its centre.
    # Grow into weaker red, but only a thin band (so a brownish iris next to
    # the pupil cannot be swallowed).
    band = max(1, int(round(_GROW_R * r)))
    kb = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band + 1, 2 * band + 1))
    weak = (rel > _WEAK_LO) & allowed
    region = (cv2.dilate(strong, kb).astype(bool) & weak).astype(np.uint8)
    region = np.maximum(region, strong)
    # Fill holes (catchlight, specks) so the fix has no pin-holes.
    ff = np.pad(1 - region, 1, constant_values=1).astype(np.uint8)
    cv2.floodFill(ff, None, (0, 0), 2)
    region = np.maximum(region, (ff[1:-1, 1:-1] == 1).astype(np.uint8))
    region &= allowed.astype(np.uint8)

    # Feathered weight. Inside the region every pixel is corrected (holes are
    # catchlights, which the fix only neutralises); on the feather a soft
    # redness gate keeps the non-red iris untouched.
    feather = max(0.6, 0.08 * r)
    soft = _smoothstep(rel, _WEAK_LO, _RED_LO)
    w = cv2.GaussianBlur(region.astype(np.float32), (0, 0), feather)
    w = np.where(region > 0, 1.0, w * soft)
    w = (w * allowed.astype(np.float32) * s)[..., None]

    # Neutral dark pupil: every channel to the pixel's smallest one, a little
    # darker where it glowed red. The catchlight (green and blue already
    # bright, measured against this face's own skin level) goes to a neutral
    # white at its green/blue level instead, so the eye keeps its sparkle.
    low = roi.min(axis=2) * (1.0 - _DARKEN * score)
    gb = roi[..., :2]
    hl = _smoothstep(gb.min(axis=2) / max(skin_level, 1.0), _HL_LO, _HL_HI)
    target = (low * (1.0 - hl) + gb.max(axis=2) * hl)[..., None]
    roi += w * (target - roi)
    d["fixed"] = 1.0
    d["area_px"] = float(region.sum())
    return d


def remove_red_eye(
    img_bgr: np.ndarray, landmarks: Any, strength: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Remove flash red-eye from one face.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
        landmarks: MediaPipe-style normalized landmark list with the refined
            iris points (478 landmarks) in the frame of ``img_bgr``.
        strength: 0-100. 0 returns the input unchanged; 100 removes all of
            the red from the detected pupils.

    Returns:
        (image, diagnostics). The image keeps the input's dtype and shape and
        is only changed inside red pupils. Diagnostics: ``eyes_fixed`` (0-2)
        and per-eye ``left_*`` / ``right_*`` core and skin redness.
    """
    diag: Dict[str, float] = {"eyes_fixed": 0.0}
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if s <= 0.0 or landmarks is None or len(landmarks.landmark) < 478:
        return img_bgr, diag

    is_float = img_bgr.dtype == np.float32
    work = img_bgr.astype(np.float32, copy=True)
    for side, iris_idx, eye_idx in (("left", LEFT_IRIS, LEFT_EYE),
                                    ("right", RIGHT_IRIS, RIGHT_EYE)):
        e = _fix_eye(work, landmarks, iris_idx, eye_idx, s)
        diag["eyes_fixed"] += e["fixed"]
        diag[f"{side}_core_redness"] = e["core_redness"]
        diag[f"{side}_skin_redness"] = e["skin_redness"]
        diag[f"{side}_area_px"] = e["area_px"]
    if diag["eyes_fixed"] == 0:
        return img_bgr, diag
    if is_float:
        return work, diag
    return np.clip(np.round(work), 0, 255).astype(img_bgr.dtype), diag


def remove_red_eye_faces(
    img_bgr: np.ndarray, faces: Sequence[Any], strength: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Apply :func:`remove_red_eye` to every detected face.

    ``faces`` are ``FaceData``-like objects with ``.landmarks`` normalized to
    ``img_bgr``.
    """
    diags: List[Dict[str, float]] = []
    if strength <= 0 or not faces:
        return img_bgr, diags
    out = img_bgr
    for f in faces:
        lms = getattr(f, "landmarks", None)
        if lms is None:
            diags.append({"eyes_fixed": 0.0})
            continue
        out, dg = remove_red_eye(out, lms, strength)
        diags.append(dg)
    return out, diags
