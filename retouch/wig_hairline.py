"""Wig hairline blend: soften the hard line where a wig's front meets the forehead.

A wig's front edge photographs as a line that real hair does not have. A
natural hairline thins out over a few millimetres; a hard-front wig stops
dead, and a lace-front wig often leaves a thin band of lace, glue shine or
make-up a shade lighter or greyer than the forehead just outside the hair.
Retouchers fix it by hand: feather the edge and tone the band back to the
skin.

The ``cosplay_wig_lace_blend`` setting (0-100) used to call
``cosplay_moat.WigLaceBlender``, which guided-filtered every hue edge inside
a skin-plus-hair mask with eps 1.0 on a 0-255 image: an identity filter in
practice (at most 1 level of change on real wig photos). This module
replaces it. Classical, no model weights. Per face:

1. **Where.** A forehead and temple zone placed from the face landmarks,
   starting ``ZONE_ABOVE_BROWS`` face heights above the brows and reaching
   well above the mesh top, so it follows head roll. Bangs that come down to
   the brows, and side locks over the cheeks, are outside it and stay as
   they are.
2. **The hairline.** The hair mask is snapped to the photo with a guided
   filter (segmenter masks are coarse) and its edge is kept only where the
   pixels just outside it are skin-like: colour close to the face's own skin
   (OKLab chromaticity in units of the face's own skin spread, lightness as
   a ratio to its median; tone-relative, no absolute level). A headband,
   hat or background against the hair is not a hairline.
3. **Band tone.** Skin pixels within ``STRIP`` face widths outside the
   hairline are compared with the forehead a little further out (a wide
   average over skin-like pixels only). Their smooth colour offset from it
   (lace, glue shine, a pale or grey band) is removed by ``strength``,
   capped at ``MAX_TONE_SHIFT`` of the local skin level, and fine texture
   is kept.
4. **Feather.** Inside a band ``BAND`` face widths either side of the
   hairline, the mid-frequency part of the image between the
   ``_TEXTURE_SIGMA`` and ``_FEATHER_SIGMA`` scales is taken out by
   ``strength``: the hard step becomes a soft ramp of hair colour into skin,
   while strand and pore texture and the broad shading stay.

Nothing outside the band changes. Limits: tested only on a public-domain
studio portrait with a natural hairline and on a hard-front wig and lace
band planted on it (none of the cosplay photos at hand shows a hairline;
every wig has full bangs). Bangs that end high on the forehead, inside the
zone, get their tips feathered too. A lace whose colour matches the skin
and shows only as mesh texture is not removed.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .prosthetic_blend import (
    _BROWS,
    _EAR_LEFT,
    _EAR_RIGHT,
    _FACE_OVAL,
    _FOREHEAD_TOP,
    _as_2d,
    _face_frame,
    _landmark_px,
    _quad,
    _to_oklab,
)

_logger = logging.getLogger(__name__)

# Masks and measurements run with each face scaled to about this width.
_WORK_FACE_WIDTH = 400.0

# Zone: from this many face heights above the brow tops to this many above
# the mesh top, and this many face widths either side of the face centre.
ZONE_ABOVE_BROWS = 0.03
ZONE_ABOVE_TOP = 0.45
ZONE_HALF_WIDTH = 0.62

# Skin-like: chroma distance from the face's skin median in units of its
# robust skin spread, lightness as a ratio to its median (as prosthetic_blend).
SKIN_LIKE_CHROMA_Z = 5.0
SKIN_LIKE_L_RATIO = (0.55, 1.45)
_MIN_CHROMA_SPREAD = 0.004

# A hair-mask edge counts as hairline where this share of the pixels within
# _SIDE (face widths) outside it, past the STRIP band, are skin-like.
_SIDE = 0.05
SIDE_MIN_SKIN = 0.5
# A coarse hair-mask edge is re-labelled by colour within this many face
# widths of it.
_SNAP = 0.03
# The hairline needs a colour/lightness step (OKLab gradient at _STEP_SIGMA
# face widths) within _STEP_REACH of the mask edge, at least STEP_Z times the
# median gradient over the face's own skin.
_STEP_SIGMA = 0.004
_STEP_REACH = 0.008
STEP_Z = 2.0
# Hairlines shorter than this (face widths) are ignored.
MIN_HAIRLINE_LENGTH = 0.08

# Band half-width either side of the hairline, and the skin strip outside it
# whose tone is matched to the forehead (face widths).
BAND = 0.05
STRIP = 0.03
# Forehead reference ring for the strip tone (face widths from the hairline).
_REF_RING = (0.04, 0.12)
MAX_TONE_SHIFT = 0.12

# Large blurs run on a copy scaled so the face is about this wide.
_LOWRES_FACE_WIDTH = 160.0

# Feather scales (face widths): texture finer than _TEXTURE_SIGMA is kept,
# the step is spread over about _FEATHER_SIGMA.
_TEXTURE_SIGMA = 0.004
_FEATHER_SIGMA = 0.025


def hairline_zone(pts: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Boolean mask of the forehead and temple zone for one face."""
    h, w = shape
    fw, fh, r, u = _face_frame(pts)
    centre = (pts[_EAR_LEFT] + pts[_EAR_RIGHT]) / 2.0
    brow_top = max(float(np.dot(pts[i] - centre, u)) for i in _BROWS)
    top = float(np.dot(pts[_FOREHEAD_TOP] - centre, u))
    zone = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(zone, [_quad(centre, r, u, (-ZONE_HALF_WIDTH * fw, ZONE_HALF_WIDTH * fw),
                              (brow_top + ZONE_ABOVE_BROWS * fh, top + ZONE_ABOVE_TOP * fh))], 1)
    return zone > 0


def _skin_like(lab: np.ndarray, pts: np.ndarray, skin: Optional[np.ndarray],
               not_hair: np.ndarray) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Pixels whose colour is close to this face's own skin."""
    h, w = lab.shape[:2]
    oval = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(oval, [cv2.convexHull(np.round(pts[_FACE_OVAL]).astype(np.int32))], 1)
    ref = (oval > 0) & not_hair
    if skin is not None:
        ref &= skin > 0.5
    if int(ref.sum()) < 200:
        return None, {"reason": "no_skin_reference"}
    L = lab[..., 0]
    l_med = float(np.median(L[ref]))
    if l_med <= 1e-4:
        return None, {"reason": "no_skin_reference"}
    # Chromaticity (a/L, b/L) so shading does not change it.
    ca = lab[..., 1] / np.maximum(L, 1e-3)
    cb = lab[..., 2] / np.maximum(L, 1e-3)
    ma, mb = float(np.median(ca[ref])), float(np.median(cb[ref]))
    da, db = ca - ma, cb - mb
    dist = np.sqrt(da * da + db * db)
    spread = max(float(np.median(dist[ref])) * 1.4826, _MIN_CHROMA_SPREAD / max(l_med, 1e-3))
    ratio = L / l_med
    like = ((dist / spread) < SKIN_LIKE_CHROMA_Z) & (ratio > SKIN_LIKE_L_RATIO[0]) & (ratio < SKIN_LIKE_L_RATIO[1])
    return like & not_hair, {"skin_L": round(l_med, 3)}


def _snap_hair(lab: np.ndarray, hb: np.ndarray, fw: float) -> np.ndarray:
    """Move a coarse hair-mask edge onto the photo's own hair/skin boundary.

    Segmenter masks are coarse (the no-BiSeNet class mask is 256 px), so the
    edge can sit a little inside the hair or out on the skin. Pixels within
    ``_SNAP`` face widths of the edge are re-labelled by which side's local
    colour (OKLab, averaged a little further in on each side) they are
    closer to (OKLab, with a/b weighted over lightness).
    """
    reach = max(2.0, _SNAP * fw)
    d_in = cv2.distanceTransform(hb.astype(np.uint8), cv2.DIST_L2, 5)
    d_out = cv2.distanceTransform((~hb).astype(np.uint8), cv2.DIST_L2, 5)
    unsure = (d_in <= reach) & (d_out <= reach)
    if not unsure.any():
        return hb
    sure_h = (hb & (d_in > reach) & (d_in <= 2.5 * reach)).astype(np.float32)
    sure_o = (~hb & (d_out > reach) & (d_out <= 2.5 * reach)).astype(np.float32)
    sig = 2.0 * reach
    wh = cv2.GaussianBlur(sure_h, (0, 0), sig)
    wo = cv2.GaussianBlur(sure_o, (0, 0), sig)
    mh = cv2.GaussianBlur(lab * sure_h[..., None], (0, 0), sig) / np.maximum(wh, 1e-6)[..., None]
    mo = cv2.GaussianBlur(lab * sure_o[..., None], (0, 0), sig) / np.maximum(wo, 1e-6)[..., None]
    fine = cv2.GaussianBlur(lab, (0, 0), max(0.7, 0.003 * fw))
    # Hue tells hair from skin better than lightness (a lace band or glue
    # shine is lighter than the forehead but keeps its hue), so a/b count double.
    wts = np.array([1.0, 4.0, 4.0], dtype=np.float32)
    dh = np.sum(wts * (fine - mh) ** 2, axis=2)
    do = np.sum(wts * (fine - mo) ** 2, axis=2)
    ok = unsure & (wh > 1e-3) & (wo > 1e-3)
    out = hb.copy()
    out[ok] = dh[ok] < do[ok]
    # Tidy specks left by texture.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    u8 = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(u8, cv2.MORPH_CLOSE, k).astype(bool)


def _face_connected(like: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Keep skin-like pixels connected to the skin inside the face outline.

    A beige wall or a pale costume beside the wig can match the skin colour;
    the forehead under a wig front is the skin that joins the face.
    """
    h, w = like.shape
    oval = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(oval, [cv2.convexHull(np.round(pts[_FACE_OVAL]).astype(np.int32))], 1)
    n, lbl = cv2.connectedComponents(like.astype(np.uint8), connectivity=8)
    if n <= 1:
        return like
    seeds = np.unique(lbl[(oval > 0) & like])
    keep = np.zeros(n, dtype=bool)
    keep[seeds[seeds > 0]] = True
    return keep[lbl]


def find_hairline(img_s: np.ndarray, pts: np.ndarray, hair: Optional[np.ndarray],
                  skin: Optional[np.ndarray],
                  person: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (hair_binary, hairline, info) at work scale.

    ``hairline`` is a boolean one-pixel line on the hair side of the
    wig-to-forehead edge, inside the zone, where skin lies just outside.
    """
    h, w = img_s.shape[:2]
    empty = np.zeros((h, w), dtype=bool)
    if hair is None or float(hair.max()) < 0.5:
        return empty, empty, {"reason": "no_hair_mask"}
    fw = _face_frame(pts)[0]
    hb = hair > 0.5
    if person is not None:
        hb &= person > 0.3
    if not hb.any():
        return hb, empty, {"reason": "no_hair_mask"}

    lab = _to_oklab(img_s)
    hb = _snap_hair(lab, hb, fw)
    like, info = _skin_like(lab, pts, skin, ~hb)
    if like is None:
        return hb, empty, info
    like = _face_connected(like, pts)
    zone = hairline_zone(pts, (h, w))
    edge = hb & ~cv2.erode(hb.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    side_k = max(3, int(round(2 * _SIDE * fw)) | 1)
    # Share of non-hair pixels near each edge pixel that are skin-like.
    # Skin-likeness is read past the band a lace or glue line may occupy.
    d_out = cv2.distanceTransform((~hb).astype(np.uint8), cv2.DIST_L2, 5)
    beyond = (d_out > STRIP * fw).astype(np.float32)
    near_like = cv2.blur(like.astype(np.float32) * beyond, (side_k, side_k))
    near_out = cv2.blur(beyond, (side_k, side_k))
    skin_side = near_like > SIDE_MIN_SKIN * np.maximum(near_out, 1e-6)
    # The hair must lie on the far side of the edge from the face centre
    # (a hairline frames the forehead); the hair's outer edge against a
    # skin-coloured wall or costume faces the other way.
    hs = cv2.GaussianBlur(hb.astype(np.float32), (0, 0), max(1.0, 0.01 * fw))
    gx = cv2.Sobel(hs, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(hs, cv2.CV_32F, 0, 1, ksize=3)
    centre = (pts[_EAR_LEFT] + pts[_EAR_RIGHT]) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    outward = gx * (xx - centre[0]) + gy * (yy - centre[1]) > 0
    line = edge & zone & skin_side & outward & (near_out > 0.02)
    # A real colour or lightness step must sit on the line: where the mask
    # edge lies in plain hair or plain skin, feathering would only smear.
    if line.any():
        sig = max(0.8, _STEP_SIGMA * fw)
        sm = cv2.GaussianBlur(lab, (0, 0), sig)
        gmag = np.zeros((h, w), dtype=np.float32)
        for ch in range(3):
            gx_c = cv2.Sobel(sm[..., ch], cv2.CV_32F, 1, 0, ksize=3)
            gy_c = cv2.Sobel(sm[..., ch], cv2.CV_32F, 0, 1, ksize=3)
            gmag += gx_c * gx_c + gy_c * gy_c
        gmag = np.sqrt(gmag)
        oval = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(oval, [cv2.convexHull(np.round(pts[_FACE_OVAL]).astype(np.int32))], 1)
        skin_ref = like & (oval > 0)
        if int(skin_ref.sum()) > 50:
            floor = float(np.median(gmag[skin_ref]))
            reach = max(3, int(round(2 * _STEP_REACH * fw)) | 1)
            near_step = cv2.dilate(gmag, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (reach, reach)))
            line &= near_step > STEP_Z * max(floor, 1e-5)
    # Drop short pieces (stray strands, mask noise).
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(
        cv2.dilate(line.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))), 8)
    keep = np.zeros(n, dtype=bool)
    min_len = MIN_HAIRLINE_LENGTH * fw
    for i in range(1, n):
        if max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= min_len:
            keep[i] = True
    line &= keep[lbl]
    info.update({"hairline_px": int(line.sum()),
                 "hairline_len_fw": round(float(line.sum()) / max(fw, 1.0), 3)})
    return hb, line, info


def _norm_blur(val: np.ndarray, wgt: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian average of ``val`` over pixels weighted by ``wgt``."""
    num = cv2.GaussianBlur(val * wgt[..., None] if val.ndim == 3 else val * wgt, (0, 0), sigma)
    den = cv2.GaussianBlur(wgt, (0, 0), sigma)
    if val.ndim == 3:
        den = den[..., None]
    return num / np.maximum(den, 1e-6)


def _blur_lowres(val: np.ndarray, sigma: float, fw: float) -> np.ndarray:
    """Gaussian blur of a smooth (large-sigma) field, computed small and upsized."""
    ds = min(1.0, _LOWRES_FACE_WIDTH / max(fw, 1.0))
    if ds >= 1.0 or sigma * ds < 2.0:
        return cv2.GaussianBlur(val, (0, 0), sigma)
    h, w = val.shape[:2]
    small = cv2.resize(val, (max(1, int(round(w * ds))), max(1, int(round(h * ds)))),
                       interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma * ds)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def blend_hairline(crop: np.ndarray, hb: np.ndarray, line: np.ndarray, like: np.ndarray,
                   fw: float, strength: float) -> np.ndarray:
    """Tone the skin strip and feather the hairline on a full-res crop.

    ``crop`` is float BGR in [0, 1]; ``hb``, ``line`` and ``like`` are boolean
    maps at the crop's resolution; ``fw`` is the face width in crop pixels.
    """
    img = crop if crop.dtype == np.float32 else crop.astype(np.float32)
    d_line = cv2.distanceTransform((~line).astype(np.uint8), cv2.DIST_L2, 5)
    band = np.exp(-0.5 * (d_line / max(1.0, BAND * fw * 0.6)) ** 2).astype(np.float32)
    band[d_line > 1.6 * BAND * fw] = 0.0
    ys, xs = np.nonzero(band > 1e-3)
    if ys.size == 0:
        return crop
    pad = int(np.ceil(3 * max(_FEATHER_SIGMA, _REF_RING[1]) * fw))
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + pad + 1)
    sub = img[y0:y1, x0:x1]
    hb_s, like_s, band_s = hb[y0:y1, x0:x1], like[y0:y1, x0:x1], band[y0:y1, x0:x1]

    # 1. Band tone: the skin strip just outside the hair is pulled to the
    # forehead a little further out (smooth offset only; texture is kept).
    d_hair = cv2.distanceTransform((~hb_s).astype(np.uint8), cv2.DIST_L2, 5)
    near_line = d_line[y0:y1, x0:x1] < BAND * fw
    strip = (d_hair > 0) & (d_hair <= STRIP * fw) & near_line
    ring = like_s & (d_hair > _REF_RING[0] * fw) & (d_hair <= _REF_RING[1] * fw)
    out = sub.copy()
    if strip.any() and int(ring.sum()) > 20:
        s_ref = max(1.0, 0.06 * fw)
        ring_f = ring.astype(np.float32)
        cover = _blur_lowres(ring_f, s_ref, fw)
        base = _blur_lowres(sub * ring_f[..., None], s_ref, fw) / np.maximum(cover, 1e-6)[..., None]
        s_loc = max(0.8, 0.004 * fw)
        local = _norm_blur(sub, strip.astype(np.float32), s_loc)
        offset = base - local
        lvl = np.maximum(base.mean(axis=2, keepdims=True), 1e-3)
        offset = np.clip(offset, -MAX_TONE_SHIFT * lvl, MAX_TONE_SHIFT * lvl)
        ramp = np.clip((1.0 - d_hair / (STRIP * fw)) / 0.4, 0.0, 1.0)
        w_tone = strip * ramp * np.clip(cover / 0.05, 0.0, 1.0)
        w_tone = cv2.GaussianBlur(w_tone.astype(np.float32), (0, 0), max(0.7, 0.003 * fw))
        out = out + strength * w_tone[..., None] * offset

    # 2. Feather: take out the mid frequencies (between the texture and the
    # feather scales) inside the band, so the step becomes a ramp.
    fine = cv2.GaussianBlur(out, (0, 0), max(0.6, _TEXTURE_SIGMA * fw))
    broad = _blur_lowres(out, max(1.5, _FEATHER_SIGMA * fw), fw)
    out = out - (strength * band_s)[..., None] * (fine - broad)

    res = crop.copy()
    res[y0:y1, x0:x1] = np.clip(out, 0.0, 1.0)
    return res


def _unit_scale(a: np.ndarray) -> float:
    """255 for uint8 or float 0-255 input, 1 for float 0-1 input."""
    if a.dtype == np.uint8:
        return 255.0
    return 1.0 if a.size == 0 or float(np.nanmax(a)) <= 1.0 + 1e-3 else 255.0


def _unit_float(a: np.ndarray) -> np.ndarray:
    """Float32 [0, 1] view or copy of an image crop."""
    scale = _unit_scale(a)
    if scale == 1.0 and a.dtype == np.float32:
        return a
    return a.astype(np.float32) / scale


def _like_input(blended: np.ndarray, img: np.ndarray, crop_scale: float) -> np.ndarray:
    """Return ``blended`` (float [0, 1]) in the input's dtype and range."""
    if img.dtype == np.uint8:
        return np.clip(blended * 255.0 + 0.5, 0, 255).astype(np.uint8)
    if crop_scale != 1.0:
        return (blended * crop_scale).astype(img.dtype, copy=False)
    return blended.astype(img.dtype, copy=False)


def apply_wig_hairline(
    img: np.ndarray,
    faces: Sequence[Any],
    acc_skin: Optional[np.ndarray],
    hair_mask: Optional[np.ndarray],
    strength: float,
    person_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Blend each face's wig hairline into the forehead by ``strength`` (0-1).

    ``img`` is float BGR in [0, 1] (as in the global stages). ``faces`` are
    FaceData-like objects with normalized ``landmarks``. Returns
    ``(image, diagnostics)``; the image is unchanged when no hairline is found.
    """
    diag: Dict[str, Any] = {"faces": [], "applied": False}
    if strength <= 0 or not faces:
        diag["reason"] = "off" if strength <= 0 else "no_face"
        return img, diag
    if hair_mask is None:
        diag["reason"] = "no_hair_mask"
        return img, diag
    h, w = img.shape[:2]
    out = img
    strength = float(np.clip(strength, 0.0, 1.0))
    for fi, face in enumerate(faces):
        pts_full = _landmark_px(getattr(face, "landmarks", None), w, h)
        if pts_full is None:
            diag["faces"].append({"face": fi, "reason": "no_landmarks"})
            continue
        fw_full, fh_full, _, _ = _face_frame(pts_full)
        if fw_full < 40:
            diag["faces"].append({"face": fi, "reason": "face_too_small"})
            continue
        centre = (pts_full[_EAR_LEFT] + pts_full[_EAR_RIGHT]) / 2.0
        reach = 1.2 * max(fw_full, fh_full)
        x0, x1 = int(max(0, centre[0] - reach)), int(min(w, centre[0] + reach))
        y0, y1 = int(max(0, centre[1] - reach * 1.3)), int(min(h, centre[1] + reach * 0.5))
        if x1 - x0 < 16 or y1 - y0 < 16:
            continue
        scale = min(1.0, _WORK_FACE_WIDTH / fw_full)
        size = (max(8, int(round((x1 - x0) * scale))), max(8, int(round((y1 - y0) * scale))))
        crop = _unit_float(out[y0:y1, x0:x1])
        small = cv2.resize(crop, size, interpolation=cv2.INTER_AREA) if scale < 1.0 else crop
        pts_s = (pts_full - np.array([x0, y0], dtype=np.float32)) * scale
        hair_s = _as_2d(hair_mask[y0:y1, x0:x1], size)
        skin_s = _as_2d(acc_skin[y0:y1, x0:x1], size) if acc_skin is not None else None
        person_s = _as_2d(person_mask[y0:y1, x0:x1], size) if person_mask is not None else None
        hb, line, info = find_hairline(small, pts_s, hair_s, skin_s, person_s)
        info["face"] = fi
        diag["faces"].append(info)
        if not line.any():
            continue
        lab_s = _to_oklab(small)
        like, _ = _skin_like(lab_s, pts_s, skin_s, ~hb)
        if like is None:
            continue
        # Only the band around the hairline (plus blur reach) is touched.
        ys, xs = np.nonzero(line)
        reach = int(np.ceil((1.6 * BAND + 3 * max(_FEATHER_SIGMA, _REF_RING[1])) * fw_full * scale)) + 2
        ry0, ry1 = max(0, ys.min() - reach), min(line.shape[0], ys.max() + reach + 1)
        rx0, rx1 = max(0, xs.min() - reach), min(line.shape[1], xs.max() + reach + 1)
        fy0, fy1 = y0 + int(round(ry0 / scale)), min(y1, y0 + int(round(ry1 / scale)))
        fx0, fx1 = x0 + int(round(rx0 / scale)), min(x1, x0 + int(round(rx1 / scale)))
        full = (fx1 - fx0, fy1 - fy0)
        hb_r, like_r, line_r = hb[ry0:ry1, rx0:rx1], like[ry0:ry1, rx0:rx1], line[ry0:ry1, rx0:rx1]
        if scale < 1.0:
            hb_f = cv2.resize(hb_r.astype(np.float32), full, interpolation=cv2.INTER_LINEAR) > 0.5
            like_f = cv2.resize(like_r.astype(np.float32), full, interpolation=cv2.INTER_LINEAR) > 0.5
            # Keep the line one pixel wide at full size: snap it to the
            # upscaled hair edge near the work-scale line.
            near = cv2.resize(cv2.dilate(line_r.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32),
                              full, interpolation=cv2.INTER_LINEAR) > 0.5
            edge = hb_f & ~cv2.erode(hb_f.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
            line_f = edge & near
        else:
            hb_f, like_f, line_f = hb_r, like_r, line_r
        region = img[fy0:fy1, fx0:fx1]
        blended = blend_hairline(_unit_float(region), hb_f, line_f, like_f, fw_full, strength)
        if out is img:
            out = img.copy()
        out[fy0:fy1, fx0:fx1] = _like_input(blended, img, crop_scale=_unit_scale(region))
        diag["applied"] = True
    if not diag["applied"]:
        diag.setdefault("reason", "no_hairline_found")
        return img, diag
    _logger.info("wig hairline: %s", [f.get("hairline_len_fw") for f in diag["faces"]])
    return out, diag
