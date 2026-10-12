"""Body paint support: find painted skin, keep its colour, even patchy coverage.

Cosplayers paint faces and bodies blue, green, grey, purple and so on. The
skin pipeline is tuned for human skin, so on painted skin it can nudge the
paint's colour (tone unification, redness and body-to-face matching pull
toward a human skin hue; grey paint picks up a pink cast), and it has no tool
for the real problem with paint: patchy coverage where a thin coat lets warm
skin show through, or where sponge marks leave blotches.

This module is classical (no model weights) and is opt-in through the
``body_paint`` strength (0-100). When it is above zero:

1. **Detect** paint per face on the pre-retouch reference. A face counts as
   painted when enough of its skin sits outside the human skin hue band in
   OKLCh, or is near-neutral (grey paint). Near-neutral is measured as
   chroma relative to the face's own lightness, and a whole-image check stops
   black-and-white photos reading as grey paint.
2. **Map** the paint: the painted face's skin plus person pixels of the same
   paint colour that connect to it (painted neck, shoulders, arms).
3. **Lock** the paint's colour: inside the paint map the OKLab a/b channels
   are restored from the reference, so skin-colour edits made for human skin
   do not tint the paint. Lightness edits (smoothing, relight, blemish heal)
   are kept. Colour grading runs later and still applies to the whole photo.
4. **Even** patchy coverage by ``strength``: mid-scale lightness blotches are
   pulled toward the local paint level (large shading and fine texture kept,
   strong features such as brows and nostrils left alone), and patches where
   skin shows through are pulled toward the surrounding paint colour.

Limits: red, orange and tan paint sit inside the human hue band and are not
detected; a costume in exactly the paint's colour that touches painted skin
is included in the map.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .color_science import bgr_to_oklab, oklab_to_bgr

_logger = logging.getLogger(__name__)

# Human skin in OKLCh sits between pinkish-red (~345 deg, wrapping through 0)
# and yellow-orange (~60 deg); green light casts reach ~110 deg. Measured on
# 11 real portraits (fair to deep skin, one cosplay): face-skin hue p2..p98
# spans 0-107 deg with a pink tail near 356 deg. Paint outside this band
# (green ~150, teal ~200, blue ~250, purple ~300) is detectable by hue.
HUMAN_HUE_LO = 345.0
HUMAN_HUE_HI = 115.0
# Below this chroma the hue angle is noise, so hue-based paint needs it.
MIN_PAINT_CHROMA = 0.035
# Grey paint: chroma relative to the pixel's own lightness (C / L). Real skin
# measured 0.04-0.13 at the median (deep skin highest); grey paint is ~0.01.
GREY_CHROMA_RATIO = 0.035
# Share of a face's skin that must look like paint before it counts.
MIN_PAINTED_FRACTION = 0.4
# A photo where fewer than this share of pixels carry real colour is treated
# as black-and-white (or fully desaturated) rather than grey paint.
MONOCHROME_COLOURFUL_FRACTION = 0.01
_COLOURFUL_CHROMA = 0.05
_MIN_FACE_SKIN_PX = 200
# Working sizes (face width in px): detection and the paint map run at
# _DETECT_FACE_WIDTH, the low-frequency evening corrections at
# _WORK_FACE_WIDTH. Full-resolution pixels only take the final corrections.
_DETECT_FACE_WIDTH = 480.0
_WORK_FACE_WIDTH = 320.0


@dataclass
class PaintedFace:
    """Paint found on one face."""

    face_index: int
    kind: str  # "colour" or "grey"
    painted_fraction: float
    paint_a: float
    paint_b: float
    paint_hue_deg: float
    paint_chroma: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "face_index": self.face_index,
            "kind": self.kind,
            "painted_fraction": round(self.painted_fraction, 3),
            "paint_hue_deg": round(self.paint_hue_deg, 1),
            "paint_chroma": round(self.paint_chroma, 4),
        }


def _to_oklab(img: np.ndarray) -> np.ndarray:
    """OKLab of a uint8 BGR image or a float BGR image in [0, 1]."""
    if img.dtype == np.uint8:
        return bgr_to_oklab(img)
    return bgr_to_oklab(np.clip(img.astype(np.float32), 0.0, 1.0) * 255.0)


def _from_oklab(lab: np.ndarray, like: np.ndarray) -> np.ndarray:
    out = oklab_to_bgr(lab.astype(np.float32), float32_out=True)
    if like.dtype == np.uint8:
        return np.clip(out, 0, 255).astype(np.uint8)
    return np.clip(out / 255.0, 0.0, 1.0).astype(np.float32)


def paint_like_pixels(lab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel (colour_paint, grey_paint) boolean maps from OKLab."""
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    C = np.sqrt(a * a + b * b)
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    off_hue = (hue > HUMAN_HUE_HI) & (hue < HUMAN_HUE_LO)
    colour = off_hue & (C >= MIN_PAINT_CHROMA)
    grey = C < GREY_CHROMA_RATIO * np.maximum(L, 0.2)
    return colour, grey


def is_monochrome(lab: np.ndarray) -> bool:
    """True when almost nothing in the photo carries colour (B&W, desaturated)."""
    C = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    return float(np.mean(C >= _COLOURFUL_CHROMA)) < MONOCHROME_COLOURFUL_FRACTION


def detect_painted_face(
    lab: np.ndarray,
    face_skin: np.ndarray,
    face_index: int = 0,
    monochrome: Optional[bool] = None,
) -> Optional[PaintedFace]:
    """Decide whether one face's skin is painted.

    Args:
        lab: OKLab of the reference image (H, W, 3).
        face_skin: boolean face-skin mask for this face (H, W).
        face_index: index recorded on the result.
        monochrome: precomputed :func:`is_monochrome`; computed when None.

    Returns:
        A :class:`PaintedFace`, or None when the face does not look painted.
    """
    n = int(face_skin.sum())
    if n < _MIN_FACE_SKIN_PX:
        return None
    colour, grey = paint_like_pixels(lab)
    colour_frac = float(colour[face_skin].mean())
    grey_frac = float(grey[face_skin].mean())
    if colour_frac >= MIN_PAINTED_FRACTION and colour_frac >= grey_frac:
        kind, sel = "colour", colour & face_skin
    elif grey_frac >= MIN_PAINTED_FRACTION:
        if monochrome is None:
            monochrome = is_monochrome(lab)
        if monochrome:
            return None
        kind, sel = "grey", grey & face_skin
    else:
        return None
    a = float(np.median(lab[..., 1][sel]))
    b = float(np.median(lab[..., 2][sel]))
    return PaintedFace(
        face_index=face_index,
        kind=kind,
        painted_fraction=colour_frac if kind == "colour" else grey_frac,
        paint_a=a,
        paint_b=b,
        paint_hue_deg=float(np.degrees(np.arctan2(b, a)) % 360.0),
        paint_chroma=float(np.hypot(a, b)),
    )


def _paint_colour_match(lab: np.ndarray, face: PaintedFace, face_sel: np.ndarray) -> np.ndarray:
    """Pixels whose a/b sit within the painted face's own colour spread."""
    da = lab[..., 1] - face.paint_a
    db = lab[..., 2] - face.paint_b
    dist = np.sqrt(da * da + db * db)
    spread = float(np.percentile(dist[face_sel], 90)) if face_sel.any() else 0.0
    tol = max(0.025, 1.5 * spread)
    L = lab[..., 0]
    if face_sel.any():
        lo, hi = np.percentile(L[face_sel], [2, 98])
    else:
        lo, hi = 0.0, 1.0
    return (dist <= tol) & (L >= lo - 0.05) & (L <= hi + 0.05)


def paint_region_mask(
    lab: np.ndarray,
    faces: Sequence[PaintedFace],
    face_skins: Sequence[np.ndarray],
    person_mask: Optional[np.ndarray] = None,
    exclude: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Float [0, 1] map of painted skin: painted face skin plus connected body paint.

    Other pixels count when they are on the person, match the painted face's
    colour and lightness range, and connect to that face's skin. The face
    parser often misses part of a painted face (it was trained on human skin),
    so the colour match also fills the face itself; the lightness range keeps
    dark pupils, brows and bright sclera out.
    """
    h, w = lab.shape[:2]
    region = np.zeros((h, w), dtype=bool)
    person = None
    if person_mask is not None:
        pm = person_mask.astype(np.float32)
        if pm.ndim == 3:
            pm = pm[..., 0]
        if pm.max() > 1.5:
            pm = pm / 255.0
        if pm.shape == (h, w):
            person = pm > 0.5
    for face, skin in zip(faces, face_skins):
        match = _paint_colour_match(lab, face, skin)
        if person is not None:
            match &= person
        seed = skin | match
        n_labels, labels = cv2.connectedComponents(seed.astype(np.uint8), connectivity=8)
        if n_labels <= 1:
            continue
        ids = np.unique(labels[skin])
        ids = ids[ids != 0]
        region |= np.isin(labels, ids) & (match | skin)
    if exclude is not None and exclude.shape[:2] == (h, w):
        ex = exclude.astype(np.float32)
        if ex.ndim == 3:
            ex = ex[..., 0]
        if ex.max() > 1.5:
            ex = ex / 255.0
        region &= ex < 0.5
    k = max(3, int(round(min(h, w) * 0.004)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    region_u8 = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    region_u8 = cv2.morphologyEx(region_u8, cv2.MORPH_CLOSE, kernel)
    soft = cv2.GaussianBlur(region_u8.astype(np.float32), (0, 0), sigmaX=k / 2.0)
    return np.clip(soft, 0.0, 1.0)


def lock_paint_colour(
    img: np.ndarray, ref: np.ndarray, mask: np.ndarray, ref_lab: Optional[np.ndarray] = None
) -> np.ndarray:
    """Restore the reference's OKLab a/b inside ``mask``; keep current lightness."""
    lab = _to_oklab(img)
    if ref_lab is None:
        ref_lab = _to_oklab(ref)
    m = mask[..., None].astype(np.float32)
    lab[..., 1:] = lab[..., 1:] + m * (ref_lab[..., 1:] - lab[..., 1:])
    return _from_oklab(lab, img)


def _masked_blur(x: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of ``x`` using only weighted pixels (normalized convolution)."""
    wsum = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma)
    if x.ndim == 3:
        num = cv2.GaussianBlur(x * weight[..., None], (0, 0), sigmaX=sigma)
        return num / np.maximum(wsum, 1e-6)[..., None]
    num = cv2.GaussianBlur(x * weight, (0, 0), sigmaX=sigma)
    return num / np.maximum(wsum, 1e-6)


def _resize(x: np.ndarray, size: Tuple[int, int], shrink: bool) -> np.ndarray:
    """Resize to (w, h): area-average when shrinking, bilinear when growing."""
    interp = cv2.INTER_AREA if shrink else cv2.INTER_LINEAR
    return cv2.resize(x, size, interpolation=interp)


def _even_corrections(
    lab: np.ndarray,
    wmask: np.ndarray,
    strength: float,
    fw: float,
    faces: Sequence[PaintedFace],
) -> Tuple[np.ndarray, np.ndarray]:
    """(dL, dab) evening corrections at the resolution of ``lab``."""
    inside = wmask > 0.5
    s_fine = max(1.0, 0.012 * fw)
    s_broad = max(3.0, 0.06 * fw)

    L = lab[..., 0]
    low = _masked_blur(L, wmask, s_fine)
    broad = _masked_blur(L, wmask, s_broad)
    blotch = low - broad
    spread = float(np.median(np.abs(blotch[inside]))) * 1.4826
    tau = max(spread * 3.0, 0.004)
    keep = np.exp(-((blotch / tau) ** 2) * 0.5)  # 1 for blotches, ->0 for features
    d_l = -strength * wmask * keep * blotch

    # Colour: pull a/b toward the local paint colour estimated from paint-like
    # pixels only, so show-through patches borrow from their paint neighbours.
    ab = lab[..., 1:]
    if faces:
        paint_w = np.zeros_like(wmask)
        for face in faces:
            d = np.hypot(ab[..., 0] - face.paint_a, ab[..., 1] - face.paint_b)
            spread_ab = float(np.percentile(d[inside], 50))
            paint_w = np.maximum(paint_w, (d <= max(0.02, 2.0 * spread_ab)).astype(np.float32))
        paint_w *= wmask
    else:
        paint_w = wmask
    d_ab = np.zeros_like(ab)
    if paint_w.sum() > _MIN_FACE_SKIN_PX * 0.25:
        local = _masked_blur(ab, paint_w, s_broad * 2.0)
        ab_low = _masked_blur(ab, wmask, s_fine)
        d_ab = strength * wmask[..., None] * (local - ab_low)
    return d_l.astype(np.float32), d_ab.astype(np.float32)


def _even_lab(
    lab: np.ndarray,
    wmask: np.ndarray,
    strength: float,
    face_width: float,
    faces: Sequence[PaintedFace] = (),
) -> np.ndarray:
    """Apply patchy-paint evening to an OKLab image in place and return it.

    The corrections are low-frequency (the finest scale is ~1% of the face
    width), so they are computed with the face scaled to about
    ``_WORK_FACE_WIDTH`` px and upsampled; fine texture comes from the
    full-resolution pixels they are added to.
    """
    if strength <= 0 or wmask.max() < 0.01:
        return lab
    if int((wmask > 0.5).sum()) < _MIN_FACE_SKIN_PX:
        return lab
    fw = max(float(face_width), 40.0)
    h, w = wmask.shape
    scale = min(1.0, _WORK_FACE_WIDTH / fw)
    if scale < 1.0:
        size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
        lab_s = _resize(lab, size, shrink=True)
        wm_s = _resize(wmask.astype(np.float32), size, shrink=True)
        if int((wm_s > 0.5).sum()) < 16:
            return lab
        d_l, d_ab = _even_corrections(lab_s, wm_s, strength, fw * scale, faces)
        d_l = _resize(d_l, (w, h), shrink=False)
        d_ab = _resize(d_ab, (w, h), shrink=False)
    else:
        d_l, d_ab = _even_corrections(lab, wmask.astype(np.float32), strength, fw, faces)
    lab[..., 0] += d_l
    lab[..., 1:] += d_ab
    return lab


def even_paint(
    img: np.ndarray,
    mask: np.ndarray,
    strength: float,
    face_width: float,
    faces: Sequence[PaintedFace] = (),
) -> np.ndarray:
    """Even patchy paint inside ``mask``.

    Lightness: a band-pass between ~1% and ~6% of the face width finds
    sponge marks and thin patches; each is pulled toward its neighbourhood
    by ``strength``. Deviations much larger than the paint's own typical
    blotch (brows, nostrils, deep shadow lines) are left alone, measured
    relative to this paint's spread, not an absolute level.

    Colour: patches whose a/b drift from the paint (warm skin showing through
    a thin coat) are pulled toward the surrounding paint colour, estimated
    from pixels that match the paint.

    Args:
        img: BGR image, uint8 or float in [0, 1].
        mask: float [0, 1] paint map.
        strength: 0-1.
        face_width: face width in pixels (sets the blotch scale).
        faces: painted faces (their colour picks the paint-matching weights).
    """
    if strength <= 0 or mask.max() < 0.01 or int((mask > 0.5).sum()) < _MIN_FACE_SKIN_PX:
        return img
    lab = _even_lab(_to_oklab(img), mask.astype(np.float32), strength, face_width, faces)
    return _from_oklab(lab, img)


def _face_skin_masks(
    acc_skin: np.ndarray, boxes: Sequence[Tuple[int, int, int, int]]
) -> List[np.ndarray]:
    """Split the accumulated face-skin mask into one boolean mask per face box."""
    skin = acc_skin.astype(np.float32)
    if skin.ndim == 3:
        skin = skin[..., 0]
    if skin.max() > 1.5:
        skin = skin / 255.0
    h, w = skin.shape
    k = max(3, int(round(min(h, w) * 0.003)) | 1)
    core = cv2.erode((skin > 0.5).astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    out = []
    for (x, y, bw, bh) in boxes:
        px, py = int(bw * 0.2), int(bh * 0.2)
        box = np.zeros((h, w), dtype=bool)
        box[max(0, y - py): min(h, y + bh + py), max(0, x - px): min(w, x + bw + px)] = True
        out.append(core & box)
    return out


def _as_2d(mask: Optional[np.ndarray], size: Tuple[int, int], shrink: bool) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = mask.astype(np.float32)
    if m.ndim == 3:
        m = m[..., 0]
    if m.max() > 1.5:
        m = m / 255.0
    if (m.shape[1], m.shape[0]) != size:
        m = _resize(m, size, shrink)
    return m


def apply_body_paint(
    img: np.ndarray,
    ref: np.ndarray,
    acc_skin: Optional[np.ndarray],
    face_boxes: Sequence[Tuple[int, int, int, int]],
    strength: float,
    person_mask: Optional[np.ndarray] = None,
    exclude: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect paint, lock its colour and even it by ``strength`` (0-1).

    ``img`` is the retouched image and ``ref`` the same frame before face
    edits (same size and dtype convention). Returns ``(image, diagnostics)``;
    the image is unchanged when no painted face is found.

    Detection and the paint map run with faces scaled to about
    ``_DETECT_FACE_WIDTH`` px; the colour lock and evening run at full
    resolution inside the paint map's bounding box only.
    """
    diag: Dict[str, Any] = {"painted_faces": [], "applied": False}
    if strength <= 0 or acc_skin is None or not face_boxes:
        diag["reason"] = "off" if strength <= 0 else "no_face"
        return img, diag
    if ref is None or ref.shape[:2] != img.shape[:2] or acc_skin.shape[:2] != img.shape[:2]:
        diag["reason"] = "reference_unavailable"
        return img, diag

    h, w = img.shape[:2]
    face_w = float(np.median([b[2] for b in face_boxes]))
    scale = min(1.0, _DETECT_FACE_WIDTH / max(face_w, 1.0))
    size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
    shrink = scale < 1.0
    ref_s = _resize(ref, size, shrink) if shrink else ref
    ref_lab = _to_oklab(ref_s)
    skin_s = _as_2d(acc_skin, size, shrink)
    person_s = _as_2d(person_mask, size, shrink)
    exclude_s = _as_2d(exclude, size, shrink)
    boxes_s = [
        (int(x * scale), int(y * scale), max(1, int(bw * scale)), max(1, int(bh * scale)))
        for (x, y, bw, bh) in face_boxes
    ]

    skins = _face_skin_masks(skin_s, boxes_s)
    mono = is_monochrome(ref_lab)
    painted: List[PaintedFace] = []
    painted_skins: List[np.ndarray] = []
    for i, skin in enumerate(skins):
        face = detect_painted_face(ref_lab, skin, face_index=i, monochrome=mono)
        if face is not None:
            painted.append(face)
            painted_skins.append(skin)
    diag["painted_faces"] = [f.as_dict() for f in painted]
    if not painted:
        grey_rejected = False
        if mono:
            grey = paint_like_pixels(ref_lab)[1]
            grey_rejected = any(
                s.sum() >= _MIN_FACE_SKIN_PX and grey[s].mean() >= MIN_PAINTED_FRACTION
                for s in skins
            )
        diag["reason"] = "monochrome_photo" if grey_rejected else "no_paint_found"
        return img, diag

    mask = paint_region_mask(ref_lab, painted, painted_skins, person_s, exclude_s)
    if shrink:
        mask = _resize(mask, (w, h), shrink=False)
    support = mask > 1e-3
    if not support.any():
        diag["reason"] = "empty_paint_map"
        return img, diag
    ys, xs = np.where(support)
    pad = max(8, int(0.02 * max(h, w)))
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    diag["paint_area_px"] = int((mask > 0.5).sum())

    m = mask[y0:y1, x0:x1]
    cur = img[y0:y1, x0:x1]
    lab = _to_oklab(cur)
    ref_lab_c = _to_oklab(ref[y0:y1, x0:x1])
    lab[..., 1:] += m[..., None] * (ref_lab_c[..., 1:] - lab[..., 1:])
    widths = [face_boxes[f.face_index][2] for f in painted]
    lab = _even_lab(lab, m, strength, float(np.median(widths)), painted)
    new = _from_oklab(lab, cur)
    out = img.copy()
    out[y0:y1, x0:x1] = np.where(support[y0:y1, x0:x1, None], new, cur)
    diag["applied"] = True
    _logger.info("body paint: %d painted face(s) %s", len(painted), diag["painted_faces"])
    return out, diag
