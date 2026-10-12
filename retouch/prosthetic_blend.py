"""Prosthetic edge blending: hide the seam where an appliance meets skin.

Cosplayers glue on latex or silicone appliances: elf-ear tips over the real
ear, forehead pieces, and the skin-toned base plates that hold horns. However
well they are painted, the edge usually shows in photos as a thin line where
the colour steps from the appliance to the real skin, often with a small lip
that catches light or casts a hairline shadow. Retouchers heal that line by
hand, photo after photo.

This module is classical (no model weights) and opt-in through the
``prosthetic_blend`` strength (0-100). When it is above zero, per face:

1. **Where to look.** Two ear zones (beside the head, reaching well above the
   ear for elf tips, stopping above the jaw) and a forehead zone (from just
   above the brows to above the hairline, where horn plates and forehead
   pieces sit), placed from the face landmarks so they follow head roll.
   Brows and the face outline beside the cheeks are excluded.
2. **Skin-like pixels.** Inside the zones, pixels whose colour sits near the
   face's own skin, on the person and away from the hair mask. Colour is
   compared as OKLab chromaticity (a/b divided by lightness, so shading does
   not change it), in units of the face's own skin spread; lightness as a
   ratio to the face's median. Tone-relative throughout, so it behaves the
   same on light and deep skin. An appliance painted to match counts as
   skin-like; a horn, wig or costume does not.
3. **Seams.** Edges are found on the frame before skin smoothing (which
   softens them), as sharp steps in chromaticity or lightness that stand
   well above the face's own skin, thinned to one-pixel lines. A line is a
   seam when it is long (at least ``MIN_SEAM_LENGTH`` face widths), has
   close-to-skin colour on both sides (rejects the face outline, horn edges
   and hairlines, including pale hair the hair mask missed), and the colour
   differs from one side to the other along most of it. That last check
   rejects wrinkles, ear folds and shadow edges, which change lightness only.
4. **Blend** by ``strength`` inside a feathered band around each seam, at
   full resolution on the retouched image: the step is spread into a gentle
   ramp (a wide average taken over skin-like pixels only, so hair and horns
   do not bleed in), the lip line goes with it, and fine skin texture is
   kept. Pixels that are not skin-like are never changed.

Limits, measured on simulated appliances (no real prosthetic photo yet):
seams shorter than about a third of the face width are not found, and a long
seam is often found only in part, because it has to stand out from the
face's own skin texture along its whole run; that part is blended and the
rest stays. A perfectly colour-matched edge that shows only as a lip is not
found. Only the seam is blended: an appliance whose whole surface is a
different tone from the skin keeps that tone away from the edge. Seams on
the neck, cheeks, chin or body are outside the zones.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .color_science import bgr_to_oklab, oklab_to_bgr

_logger = logging.getLogger(__name__)

# Detection runs with each face scaled to about this width in pixels.
_WORK_FACE_WIDTH = 400.0

# MediaPipe face-mesh indices.
_EAR_LEFT = 234   # face outline at ear height, camera-viewer left
_EAR_RIGHT = 454  # face outline at ear height, camera-viewer right
_FOREHEAD_TOP = 10
_CHIN = 152
_BROWS = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46,
          300, 293, 334, 296, 336, 285, 295, 282, 283, 276]
_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

# Skin-like: chroma distance from the face's skin median in units of the
# face's own robust skin spread, and lightness as a ratio to its median.
SKIN_LIKE_CHROMA_Z = 5.0
SKIN_LIKE_L_RATIO = (0.55, 1.45)
# Floor for the robust chroma spread (OKLab units) so very even skin does
# not make every small colour change look like a different material.
_MIN_CHROMA_SPREAD = 0.004

# Seam edges: a sharp step in chromaticity or lightness, measured at this
# scale (face widths), at least this many times the median step over the
# face's own skin ...
_STEP_SIGMA = 0.006
SEAM_STEP_Z = 2.5
# ... traced along weaker edges down to this ...
SEAM_TRACE_Z = 1.6
# ... with a colour offset from one side of the line to the other (median
# along the line, in skin spreads).
SEAM_SIDE_OFFSET = 1.0

# Shortest seam line kept, in face widths.
MIN_SEAM_LENGTH = 0.30
# Both sides of a seam, this far away (face widths), must be close to skin:
# chromaticity within this many skin spreads.
SIDE_DISTANCE = 0.025
SIDE_SKIN_Z = 5.0
SIDE_MIN_SATURATION = 0.6

# Blend band half-width and ramp scale, in face widths.
_BAND = 0.07
_RAMP_SIGMA = 0.06
# Fine skin texture below this scale (face widths) is kept.
_TEXTURE_SIGMA = 0.003


def _to_oklab(img: np.ndarray) -> np.ndarray:
    """OKLab of a uint8 BGR image or a float BGR image in [0, 1]."""
    if img.dtype == np.uint8:
        return bgr_to_oklab(img)
    return bgr_to_oklab(np.clip(img.astype(np.float32), 0.0, 1.0) * 255.0)


def _from_oklab(lab: np.ndarray, like: np.ndarray) -> np.ndarray:
    out = oklab_to_bgr(lab.astype(np.float32), float32_out=True)
    if like.dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.clip(out / 255.0, 0.0, 1.0).astype(like.dtype, copy=False)


def _as_2d(mask: Optional[np.ndarray], size: Tuple[int, int]) -> Optional[np.ndarray]:
    """Float [0, 1] single-channel mask resized to ``size`` (w, h)."""
    if mask is None:
        return None
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.max() > 1.5:
        m = m / 255.0
    if (m.shape[1], m.shape[0]) != size:
        m = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    return np.clip(m, 0.0, 1.0)


def _landmark_px(landmarks: Any, w: int, h: int) -> Optional[np.ndarray]:
    """(N, 2) float pixel coords from normalized landmarks, or None."""
    pts = getattr(landmarks, "landmark", landmarks)
    try:
        arr = np.array([[p.x * w, p.y * h] for p in pts], dtype=np.float32)
    except (TypeError, AttributeError):
        return None
    if arr.shape[0] < 468:
        return None
    return arr


def _face_frame(pts: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Face width, face height, unit 'right' and unit 'up' vectors."""
    right = pts[_EAR_RIGHT] - pts[_EAR_LEFT]
    fw = float(np.linalg.norm(right))
    up = pts[_FOREHEAD_TOP] - pts[_CHIN]
    fh = float(np.linalg.norm(up))
    r = right / max(fw, 1e-6)
    u = up / max(fh, 1e-6)
    # Image y points down, so 'up' has negative y. Re-orthogonalize 'up'
    # against 'right' so a slightly skewed mesh still gives a clean frame.
    u = u - r * float(np.dot(u, r))
    u = u / max(float(np.linalg.norm(u)), 1e-6)
    return fw, fh, r, u


def _quad(origin: np.ndarray, a: np.ndarray, b: np.ndarray,
          a_rng: Tuple[float, float], b_rng: Tuple[float, float]) -> np.ndarray:
    corners = [
        origin + a * a_rng[0] + b * b_rng[0],
        origin + a * a_rng[1] + b * b_rng[0],
        origin + a * a_rng[1] + b * b_rng[1],
        origin + a * a_rng[0] + b * b_rng[1],
    ]
    return np.round(np.array(corners)).astype(np.int32)


def seam_zones(pts: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Boolean mask of the two ear zones and the forehead zone for one face."""
    h, w = shape
    fw, fh, r, u = _face_frame(pts)
    zone = np.zeros((h, w), dtype=np.uint8)
    # Ear zones: from the face outline to well outside it, from about the
    # ear lobe to high above the ear (elf tips). They stop above the jaw.
    for idx, out_dir in ((_EAR_LEFT, -r), (_EAR_RIGHT, r)):
        cv2.fillPoly(zone, [_quad(pts[idx], out_dir, u, (-0.03 * fw, 0.45 * fw),
                                  (-0.12 * fh, 0.70 * fh))], 1)
    # Forehead zone: from just above the brows to above the mesh top.
    centre = (pts[_EAR_LEFT] + pts[_EAR_RIGHT]) / 2.0
    brow_top = max(float(np.dot(pts[i] - centre, u)) for i in _BROWS)
    top = float(np.dot(pts[_FOREHEAD_TOP] - centre, u))
    cv2.fillPoly(zone, [_quad(centre, r, u, (-0.62 * fw, 0.62 * fw),
                              (brow_top + 0.03 * fh, top + 0.45 * fh))], 1)
    # The face outline beside the cheeks and jaw is a natural edge (cheek
    # against ear, neck or hair), not a seam.
    side = [i for i in _FACE_OVAL if float(np.dot(pts[i] - centre, u)) < brow_top]
    outline = np.zeros_like(zone)
    for a_i, b_i in zip(side[:-1], side[1:]):
        if abs(_FACE_OVAL.index(a_i) - _FACE_OVAL.index(b_i)) == 1:
            cv2.line(outline, tuple(np.round(pts[a_i]).astype(int)),
                     tuple(np.round(pts[b_i]).astype(int)), 1, max(1, int(round(0.07 * fw))))
    zone[outline > 0] = 0
    brows = np.zeros_like(zone)
    for side in (_BROWS[:10], _BROWS[10:]):
        cv2.fillPoly(brows, [cv2.convexHull(np.round(pts[side]).astype(np.int32))], 1)
    k = max(3, int(round(0.06 * fw)) | 1)
    brows = cv2.dilate(brows, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return (zone > 0) & (brows == 0)


def chromaticity(lab: np.ndarray, l_ref: float) -> np.ndarray:
    """OKLab a/b divided by lightness (scaled back by ``l_ref``).

    Shading darkens skin and lowers a/b together, so this barely changes
    across a shadow, a wrinkle or the folds of an ear, while a change of
    material (paint or latex over skin) still shows.
    """
    l_floor = max(0.3 * l_ref, 1e-3)
    return lab[..., 1:3] * (l_ref / np.maximum(lab[..., 0:1], l_floor))


def _skin_stats(lab: np.ndarray, pts: np.ndarray, skin: np.ndarray) -> Optional[Dict[str, Any]]:
    """Median and robust spread of the face's own skin (lightness, chromaticity)."""
    oval = np.zeros(skin.shape, dtype=np.uint8)
    cv2.fillPoly(oval, [np.round(pts[_FACE_OVAL]).astype(np.int32)], 1)
    inside = oval > 0
    if not inside.any():
        return None
    # Soft skin masks do not always reach 1 (a wig or heavy makeup lowers the
    # segmenter's confidence), so threshold relative to the face's own peak.
    thr = max(0.1, 0.5 * float(skin[inside].max()))
    sel = inside & (skin > thr)
    if sel.sum() < 200:
        return None
    l_med = float(np.median(lab[..., 0][sel]))
    ch = chromaticity(lab, l_med)[sel]
    med = np.median(ch, axis=0)
    mad = np.median(np.abs(ch - med), axis=0) * 1.4826
    return {
        "L": l_med, "a": float(med[0]), "b": float(med[1]),
        "sa": max(float(mad[0]), _MIN_CHROMA_SPREAD),
        "sb": max(float(mad[1]), _MIN_CHROMA_SPREAD),
        "sel": sel,
    }


def _skin_z(ch: np.ndarray, stats: Dict[str, Any]) -> np.ndarray:
    za = (ch[..., 0] - stats["a"]) / stats["sa"]
    zb = (ch[..., 1] - stats["b"]) / stats["sb"]
    return np.sqrt(za * za + zb * zb)


def skin_like(lab: np.ndarray, stats: Dict[str, Any], max_z: float = SKIN_LIKE_CHROMA_Z) -> np.ndarray:
    """Pixels whose colour is close to this face's own skin (tone-relative)."""
    ratio = lab[..., 0] / max(stats["L"], 1e-4)
    return (_skin_z(chromaticity(lab, stats["L"]), stats) <= max_z) & \
        (ratio >= SKIN_LIKE_L_RATIO[0]) & (ratio <= SKIN_LIKE_L_RATIO[1])


def edge_step(ch: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    """Height and direction of the local step in ``ch`` at scale ``sigma``.

    ``ch`` is an (H, W, C) image (chromaticity, or lightness as one channel).
    The Gaussian-derivative magnitude times ``sigma * sqrt(2 pi)`` equals the
    step height for a sharp edge at any scale, while a soft transition much
    wider than ``sigma`` (blush, contour) reads far lower. The direction
    (radians) is the one of greatest change, from the structure tensor.
    """
    jxx = np.zeros(ch.shape[:2], dtype=np.float32)
    jxy = np.zeros_like(jxx)
    jyy = np.zeros_like(jxx)
    for c in range(ch.shape[2]):
        g = cv2.GaussianBlur(np.ascontiguousarray(ch[..., c], dtype=np.float32), (0, 0), sigma)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) * 0.125
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) * 0.125
        jxx += gx * gx
        jxy += gx * gy
        jyy += gy * gy
    height = np.sqrt(jxx + jyy) * float(sigma * np.sqrt(2.0 * np.pi))
    angle = 0.5 * np.arctan2(2.0 * jxy, jxx - jyy)
    return height, angle


def _two_sided(ok: np.ndarray, angle: np.ndarray, dist: float) -> np.ndarray:
    """True where the pixels ``dist`` px away on both sides across the edge
    are skin-like (a seam has skin or skin-toned appliance on both sides)."""
    h, w = ok.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.cos(angle).astype(np.float32) * dist
    dy = np.sin(angle).astype(np.float32) * dist
    src = ok.astype(np.uint8)
    side_a = cv2.remap(src, xx + dx, yy + dy, cv2.INTER_NEAREST, borderValue=0)
    side_b = cv2.remap(src, xx - dx, yy - dy, cv2.INTER_NEAREST, borderValue=0)
    return (side_a > 0) & (side_b > 0)


def _edge_ridge(step: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """Pixels where ``step`` is a local maximum across the edge direction."""
    q = np.round(((angle % np.pi) / (np.pi / 4.0))).astype(np.int32) % 4
    pad = np.pad(step, 1, mode="edge")
    h, w = step.shape
    keep = np.zeros((h, w), dtype=bool)
    for k, (dy, dx) in enumerate(((0, 1), (1, 1), (1, 0), (1, -1))):
        fwd = pad[1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        bwd = pad[1 - dy:1 - dy + h, 1 - dx:1 - dx + w]
        keep |= (q == k) & (step >= fwd) & (step >= bwd)
    return keep


def find_seams(
    lab: np.ndarray,
    pts: np.ndarray,
    skin: np.ndarray,
    person: Optional[np.ndarray] = None,
    hair: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Seam lines for one face at working scale.

    Returns ``(seam_mask, skin_like_mask, info)``; both masks are boolean and
    the seam mask is empty when nothing qualifies.
    """
    h, w = lab.shape[:2]
    info: Dict[str, Any] = {"seams": 0}
    empty = np.zeros((h, w), dtype=bool)
    stats = _skin_stats(lab, pts, skin)
    if stats is None:
        info["reason"] = "too_little_skin"
        return empty, empty, info
    fw = _face_frame(pts)[0]
    zone = seam_zones(pts, (h, w))
    allowed = np.ones((h, w), dtype=bool)
    if person is not None:
        allowed &= person > 0.5
    if hair is not None:
        k = max(3, int(round(0.03 * fw)) | 1)
        allowed &= cv2.dilate((hair > 0.5).astype(np.uint8), np.ones((k, k), np.uint8)) == 0
    like = skin_like(lab, stats) & allowed
    # Close small holes so the seam line itself (a colour outlier) and its
    # lip stay inside the skin-like region.
    k = max(3, int(round(0.015 * fw)) | 1)
    like = cv2.morphologyEx(like.astype(np.uint8), cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0

    sigma = max(1.0, _STEP_SIGMA * fw)
    # Edges: sharp steps in chromaticity or in lightness, each measured in
    # units of this face's own typical skin step. A lightness edge alone is
    # common (wrinkles, ear folds, the lip of an appliance); what makes it a
    # seam is checked per line further down.
    step_c, angle_c = edge_step(chromaticity(lab, stats["L"]), sigma)
    step_l, angle_l = edge_step(lab[..., 0:1], sigma)
    ref_c = max(float(np.median(step_c[stats["sel"]])), 1e-5)
    ref_l = max(float(np.median(step_l[stats["sel"]])), 1e-5)
    zc, zl = step_c / ref_c, step_l / ref_l
    step = np.maximum(zc, zl)
    angle = np.where(zl > zc, angle_l, angle_c)
    # Hysteresis: lines are traced over weaker edges but must contain strong
    # ones, so one seam is found whole instead of in fragments.
    strong = step > SEAM_STEP_Z
    cand = step > SEAM_TRACE_Z
    # Seam candidates must sit inside a skin-like area, away from its border
    # (the face outline against hair or background is not a seam) ...
    kr = max(3, int(round(0.02 * fw)) | 1)
    interior = cv2.erode(like.astype(np.uint8),
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr, kr))) > 0
    cand &= zone & interior
    # ... and have close-to-skin colour on both sides (a hairline has hair on
    # one side, even where the hair mask misses fine or grey strands).
    smooth = cv2.GaussianBlur(lab, (0, 0), sigma)
    ratio = smooth[..., 0] / max(stats["L"], 1e-4)
    ch_s = chromaticity(smooth, stats["L"])
    # Grey, white and blond hair is paler than skin: each side must keep most
    # of the skin's own saturation (relative to this face, not absolute).
    sat = np.hypot(ch_s[..., 0], ch_s[..., 1]) / max(float(np.hypot(stats["a"], stats["b"])), 1e-4)
    tight = (_skin_z(ch_s, stats) <= SIDE_SKIN_Z) & (sat >= SIDE_MIN_SATURATION) & allowed & \
        (ratio >= SKIN_LIKE_L_RATIO[0]) & (ratio <= SKIN_LIKE_L_RATIO[1])
    side_d = max(2.0, SIDE_DISTANCE * fw)
    cand &= _two_sided(tight, angle, side_d)
    info["ref_step"] = round(ref_c, 6)
    if not cand.any():
        return empty, like, info

    # Colour offset across each edge pixel: chromaticity (shading-invariant)
    # a little way out on each side, in units of the skin's own spread.
    ch_w = chromaticity(cv2.GaussianBlur(lab, (0, 0), max(1.0, 0.01 * fw)), stats["L"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (np.cos(angle) * side_d).astype(np.float32)
    dy = (np.sin(angle) * side_d).astype(np.float32)
    off = np.zeros((h, w), dtype=np.float32)
    for c, spread in ((0, stats["sa"]), (1, stats["sb"])):
        plane = np.ascontiguousarray(ch_w[..., c], dtype=np.float32)
        side_a = cv2.remap(plane, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        side_b = cv2.remap(plane, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        off += ((side_a - side_b) / spread) ** 2
    off = np.sqrt(off)

    # Thin the candidates to one-pixel edge lines (non-maximum suppression
    # across the edge, as in Canny), then trace each line. A seam must be
    # long, and along most of its length the colour must differ from one
    # side to the other: a wrinkle, an ear fold or a shadow edge changes
    # lightness only, while an appliance is never exactly the skin's colour.
    cand &= _edge_ridge(step, angle)
    n, labels, cc_stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    has_strong = np.bincount(labels[strong & cand], minlength=n) > 0
    seams = np.zeros((h, w), dtype=bool)
    kept: List[Dict[str, float]] = []
    for i in range(1, n):
        x, y, bw, bh, length = cc_stats[i]
        if not has_strong[i] or length < MIN_SEAM_LENGTH * fw:
            continue
        comp = labels[y:y + bh, x:x + bw] == i
        side_off = float(np.median(off[y:y + bh, x:x + bw][comp]))
        if side_off < SEAM_SIDE_OFFSET:
            continue
        seams[y:y + bh, x:x + bw] |= comp
        kept.append({
            "length_fw": round(float(length) / fw, 3),
            "side_offset": round(side_off, 2),
        })
    info["seams"] = len(kept)
    info["seam_lines"] = kept
    return seams, like, info


def blend_seams(
    img: np.ndarray,
    band: np.ndarray,
    like: np.ndarray,
    face_width: float,
    strength: float,
) -> np.ndarray:
    """Blend the image inside ``band`` (float [0, 1], same size as ``img``).

    The colour step across the seam becomes a smooth ramp: mid and low
    frequencies are replaced by a wide average over skin-like pixels, and
    fine texture is kept with its outliers (the lip line) clipped.
    """
    lab = _to_oklab(img)
    # Soften the skin-like map so the blend has no speckle along its edge.
    likef = cv2.GaussianBlur(like.astype(np.float32), (0, 0), max(1.0, 0.01 * face_width))
    s_tex = max(0.8, _TEXTURE_SIGMA * face_width)
    s_ramp = max(2.0, _RAMP_SIGMA * face_width)
    low = cv2.GaussianBlur(lab, (0, 0), s_tex)
    high = lab - low
    num = cv2.GaussianBlur(lab * likef[..., None], (0, 0), s_ramp)
    den = cv2.GaussianBlur(likef, (0, 0), s_ramp)[..., None]
    wide = np.where(den > 1e-3, num / np.maximum(den, 1e-3), low)
    alpha = np.clip(band * strength, 0.0, 1.0) * likef
    sel = alpha > 0.05
    if sel.any():
        scale = np.median(np.abs(high[sel]), axis=0) * 1.4826 + 1e-5
        clipped = np.clip(high, -3.5 * scale, 3.5 * scale)
    else:
        clipped = high
    a = alpha[..., None]
    out = low + a * (wide - low) + high + a * (clipped - high)
    new = _from_oklab(out, img)
    return np.where(alpha[..., None] > 1e-3, new, img)


def apply_prosthetic_blend(
    img: np.ndarray,
    faces: Sequence[Any],
    acc_skin: Optional[np.ndarray],
    strength: float,
    person_mask: Optional[np.ndarray] = None,
    hair_mask: Optional[np.ndarray] = None,
    ref: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Find and blend prosthetic seams around each face by ``strength`` (0-1).

    ``faces`` are FaceData-like objects with ``landmarks`` (normalized to the
    full image) and ``bbox``. ``ref`` is the same frame before skin edits
    (same size and value range); seams are found on it when given, because
    skin smoothing softens them, and blended on ``img``. Returns
    ``(image, diagnostics)``; the image is unchanged when no seam is found.
    """
    diag: Dict[str, Any] = {"faces": [], "applied": False}
    if strength <= 0 or acc_skin is None or not faces:
        diag["reason"] = "off" if strength <= 0 else "no_face"
        return img, diag
    h, w = img.shape[:2]
    if acc_skin.shape[:2] != (h, w):
        diag["reason"] = "mask_size_mismatch"
        return img, diag

    if ref is None or ref.shape[:2] != (h, w):
        ref = img
    out = img
    for fi, face in enumerate(faces):
        pts_full = _landmark_px(getattr(face, "landmarks", None), w, h)
        if pts_full is None:
            diag["faces"].append({"face": fi, "reason": "no_landmarks"})
            continue
        fw_full = _face_frame(pts_full)[0]
        if fw_full < 40:
            diag["faces"].append({"face": fi, "reason": "face_too_small"})
            continue
        # Work on a crop around the face, scaled so the face is ~400 px wide.
        fw, fh, r, u = _face_frame(pts_full)
        centre = (pts_full[_EAR_LEFT] + pts_full[_EAR_RIGHT]) / 2.0
        reach = 1.3 * max(fw, fh)
        x0, x1 = int(max(0, centre[0] - reach)), int(min(w, centre[0] + reach))
        y0, y1 = int(max(0, centre[1] - reach * 1.2)), int(min(h, centre[1] + reach))
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        scale = min(1.0, _WORK_FACE_WIDTH / fw_full)
        size = (max(8, int(round((x1 - x0) * scale))), max(8, int(round((y1 - y0) * scale))))
        crop = out[y0:y1, x0:x1]
        ref_crop = ref[y0:y1, x0:x1]
        small = cv2.resize(ref_crop, size, interpolation=cv2.INTER_AREA) if scale < 1.0 else ref_crop
        lab_s = _to_oklab(small)
        pts_s = (pts_full - np.array([x0, y0], dtype=np.float32)) * scale
        skin_s = _as_2d(acc_skin[y0:y1, x0:x1], size)
        person_s = _as_2d(person_mask[y0:y1, x0:x1], size) if person_mask is not None else None
        hair_s = _as_2d(hair_mask[y0:y1, x0:x1], size) if hair_mask is not None else None
        seams, like, info = find_seams(lab_s, pts_s, skin_s, person_s, hair_s)
        info["face"] = fi
        diag["faces"].append(info)
        if not seams.any():
            continue
        fw_s = fw_full * scale
        kb = max(3, int(round(2 * _BAND * fw_s)) | 1)
        band = cv2.dilate(seams.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb))).astype(np.float32)
        band = cv2.GaussianBlur(band, (0, 0), max(1.0, 0.5 * _BAND * fw_s))
        full_size = (x1 - x0, y1 - y0)
        if scale < 1.0:
            band = cv2.resize(band, full_size, interpolation=cv2.INTER_LINEAR)
            like_full = cv2.resize(like.astype(np.float32), full_size, interpolation=cv2.INTER_LINEAR) > 0.5
        else:
            like_full = like
        blended = blend_seams(crop, np.clip(band, 0.0, 1.0), like_full, fw_full, strength)
        if out is img:
            out = img.copy()
        out[y0:y1, x0:x1] = blended
        diag["applied"] = True
    if diag["applied"]:
        _logger.info("prosthetic blend: %s", [f.get("seams", 0) for f in diag["faces"]])
    else:
        diag.setdefault("reason", "no_seam_found")
    return out, diag
