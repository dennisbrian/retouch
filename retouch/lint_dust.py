"""Lint & Dust Cleanup: remove lint, fluff and dust specks off costume and backdrop.

The old ``backdrop_cleanup`` inpainted every high-frequency outlier outside
the selfie person mask. On real cosplay photos the person mask misses props
(bunny ears, glove tips, a wristband, fishnet), so it smeared the edges of the
costume and never touched lint on the costume itself; it also looped over
every connected component in Python (minutes at 26 MP). This op replaces it.

Where it looks (work resolution, face width scaled to about 400 px): the whole
frame minus what belongs to the person's skin, hair and face:

* skin: the multiclass segmenter's body/face skin classes, kept only where the
  colour could be this person's skin (``costume_clarity``'s chromaticity
  model, measured on this photo), so a black glove labelled arm still counts
  as costume; without the segmenter, person pixels in the skin colour;
* painted skin (``body_paint``), hair (class + whole-frame hair mask), the
  face pipeline's own skin / hair / lip masks and an ellipse over each face;
* without any skin model or segmenter only the backdrop outside the person
  is searched.

What counts as a speck (full resolution, CIELab L*):

* a morphological top-hat (bright: L - opening, dark: closing - L) with a
  structuring element about 2% of the face width finds features narrower
  than that, either polarity, so light lint on black vinyl and dark dust on a
  pale backdrop both show;
* contrast is measured in units of the local texture spread of the surround
  (the image with all small features removed), floored at the frame's own
  sensor noise, so a speck must stand out from *this* fabric's grain;
* the surround must be homogeneous (a speck's contrast well above the
  surround's own local standard deviation), so an object edge, seam or print
  is never a speck;
* the top-hat at twice the element size must not be much larger: a glint
  sitting on a broader sheen (vinyl, satin) is part of a highlight;
* components must be small (compact up to the element size) or thin fibres
  up to 15% of the face width; a cluster of candidates within 20% of the face
  width is texture (fishnet knots, sequins, rhinestones, carpet), not dust;
* bright components at the frame's specular tier (its own p99.5 L*) are
  glints.
* a second, coarser pass looks for soft dark sensor-dust spots on the
  backdrop only (outside the person).

Heal: masked normalised-convolution fill from the surround (so the speck's
own colour goes too) plus fine grain copied from a nearby clean donor, blended
through a 1 px feather. Pixels outside the healed specks are bit-identical.

No absolute intensity threshold is used on the subject; the only fixed level
is a 3 L* contrast floor (about 1.5 just-noticeable differences) below which a
speck cannot be seen anyway. Off by default (strength 0 is a no-op).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .costume_clarity import (
    _BODY_SKIN,
    _CLOTHES,
    _CLS_HI,
    _CLS_LO,
    _FACE_SKIN,
    _HAIR,
    _OTHERS,
    _SKIN_D_IN_WIDE,
    _SKIN_D_OUT_WIDE,
    _WORK_FACE_WIDTH,
    _as_2d,
    _face_ellipses,
    _frame_noise_sigma,
    _ramp,
    _skin_chroma_model,
    _skin_colour_likeness,
    painted_skin_mask,
)

_logger = logging.getLogger(__name__)

# Speck width limit (x face width). Fixed: a wider element at high strength
# joins more specks to their surroundings and finds fewer, not more.
_W_FW_LO, _W_FW_HI = 0.0225, 0.0225
# Fibre length limit (x face width).
_FIBRE_LEN_FW = 0.15
# Contrast threshold in surround-spread units at strength 0 -> 1.
_Z_LO, _Z_HI = 4.5, 3.5
# Contrast must exceed this many times the surround's own range (max - min
# of the speck-free surface within twice the element).
_RANGE_RATIO = 2.0
# Gloss: more than this many bright candidates within this window (x face
# width) mark a glossy surface; its bright specks are glints.
_GLOSS_FW = 0.6
_GLOSS_MAX = 3
# A broad highlight nearby above this share of the speck's contrast = glossy.
_SHEEN_SHARE = 0.35
# Isolation: the candidate's twice-element blob may be at most this many
# times its own area.
_ISO_RATIO = 2.5
# ... measured on pixels above this share of the threshold at twice the element.
_ISO_GROW = 0.7
# Sharpness: steepest slope (L* per px) at least this share of the contrast.
_MIN_SHARP = 0.25
# Texture / gloss counts include every feature above this many spread units
# (fixed, so strength does not change what counts as a busy surface).
_COUNT_Z = 3.75
# Busy surround: features above this share of the speck's own contrast
# within three widths.
_BUSY_SHARE = 0.4
# Ring evenness: 10-90% spread of L* just around it, as a share of contrast.
_RING_SHARE = 0.35
# Top-hat at 2x element over top-hat at 1x: above this, it sits on a broader
# structure (sheen, highlight, shading) rather than a flat surround.
_BROAD_RATIO = 1.6
# Hysteresis: grow seeds into pixels above this share of the threshold.
_GROW = 0.45
# Perceptual contrast floor (L* units).
_FLOOR_L = 3.0
# Texture density: more than this many candidates within the window.
_DENSITY_FW = 0.2
_DENSITY_MAX = 3
# Specular tier: bright specks at or above this percentile of the frame's L*.
_SPECULAR_PCT = 99.5
# Coarse sensor-dust pass (backdrop only, dark spots): element size (x face
# width) and a lower contrast floor (sensor dust is soft and faint).
_DUST_W_FW = 0.06
_DUST_FLOOR_L = 2.0
# Exclusion margin around skin/hair/face (x face width).
_EXCL_MARGIN_FW = 0.012


def _region_mask(
    work: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    probs: Optional[np.ndarray],
    hair: Optional[np.ndarray],
    protect: Optional[np.ndarray],
    face_skin: Optional[np.ndarray],
    person: Optional[np.ndarray],
    fw: float,
) -> Tuple[Any, Any, Any, Optional[Dict[str, float]], Dict[str, Any]]:
    """Search region, backdrop and subject masks at work resolution (0/1 float).

    Returns ``(region, backdrop, subject, skin_model, diag)``; region is None
    when nothing can be searched safely. Light specks are only looked for on
    the subject (on the backdrop they are lights and reflections).
    """
    diag: Dict[str, Any] = {}
    h, w = work.shape[:2]
    lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
    have_classes = probs is not None and probs.shape[:2] == (h, w)
    person_r = _ramp(person, 0.3, 0.7) if person is not None else None

    model = None
    if have_classes:
        body_skin = (probs[..., _BODY_SKIN] > 0.7) & (probs.argmax(axis=2) == _BODY_SKIN)
        model = _skin_chroma_model(lab, body_skin)
    if model is None and face_skin is not None:
        model = _skin_chroma_model(lab, face_skin > 0.5)
    if model is not None and model["neutral"]:
        diag["skin_colour_veto"] = "skipped_neutral_skin"
        model = None

    excl = np.zeros((h, w), np.float32)
    if have_classes:
        skin_cls = _ramp(np.maximum(probs[..., _BODY_SKIN], probs[..., _FACE_SKIN]), _CLS_LO, _CLS_HI)
        if model is not None:
            skin_cls = skin_cls * _skin_colour_likeness(lab, model, _SKIN_D_IN_WIDE, _SKIN_D_OUT_WIDE)
        excl = np.maximum(excl, np.maximum(skin_cls, _ramp(probs[..., _HAIR], _CLS_LO, _CLS_HI)))
        clothes = _ramp(np.clip(probs[..., _CLOTHES] + probs[..., _OTHERS], 0.0, 1.0), _CLS_LO, _CLS_HI)
        diag["source"] = "classes"
    else:
        clothes = np.zeros((h, w), np.float32)
        if person_r is not None and model is not None:
            excl = np.maximum(excl, person_r * _skin_colour_likeness(lab, model, _SKIN_D_IN_WIDE, _SKIN_D_OUT_WIDE))
        diag["source"] = "person_mask"

    paint, n_painted = painted_skin_mask(work, boxes, face_skin, person)
    if paint is not None:
        excl = np.maximum(excl, paint * (1.0 - _ramp(clothes, 0.5, 0.9)))
        diag["painted_faces"] = n_painted
    for m in (hair, protect, face_skin):
        if m is not None:
            excl = np.maximum(excl, m)
    if boxes:
        excl = np.maximum(excl, _face_ellipses((w, h), boxes))

    k = max(3, int(round(2 * _EXCL_MARGIN_FW * fw)) | 1)
    excl_hard = cv2.dilate((excl > 0.3).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    backdrop = None
    subject = None
    if person_r is not None:
        subj = np.maximum(person_r, clothes) > 0.3
        subject = subj.astype(np.float32)
        kb = max(3, int(round(0.04 * fw)) | 1)
        subj = cv2.dilate(subj.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kb, kb)))
        backdrop = ((subj == 0) & (excl_hard == 0)).astype(np.float32)

    if have_classes or model is not None:
        region = (excl_hard == 0).astype(np.float32)
    elif backdrop is not None:
        # No way to tell skin from costume: backdrop only.
        region = backdrop
        diag["scope"] = "backdrop_only"
    else:
        diag["reason"] = "no_segmentation"
        return None, None, None, None, diag
    diag["region_share"] = round(float(region.mean()), 4)
    if region.max() <= 0:
        diag["reason"] = "no_region"
        return None, None, None, model, diag
    return region, backdrop, subject, model, diag


def _rect(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; wide ones run on a reduced copy (seconds, not minutes, at 26 MP)."""
    if sigma <= 8.0:
        return cv2.GaussianBlur(x, (0, 0), sigmaX=sigma)
    f = sigma / 4.0
    h, w = x.shape[:2]
    small = cv2.resize(x, (max(1, int(round(w / f))), max(1, int(round(h / f)))), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigmaX=4.0)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _box_count(points: np.ndarray, win: int) -> np.ndarray:
    """Sum of ``points`` in a win x win box around each pixel (on a reduced grid)."""
    f = max(1, win // 16)
    if f > 1:
        h, w = points.shape
        hs, ws = -(-h // f), -(-w // f)
        pad = np.zeros((hs * f, ws * f), np.float32)
        pad[:h, :w] = points
        small = pad.reshape(hs, f, ws, f).sum(axis=(1, 3))
        k = max(1, int(round(win / f)) | 1)
        small = cv2.boxFilter(small, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
        return np.repeat(np.repeat(small, f, axis=0), f, axis=1)[:h, :w]
    return cv2.boxFilter(points, -1, (win, win), normalize=False, borderType=cv2.BORDER_CONSTANT)


def _morph(img: np.ndarray, op: int, se: np.ndarray) -> np.ndarray:
    # Reflected border: the default constant border makes the frame edge
    # itself read as a speck.
    return cv2.morphologyEx(img, op, se, borderType=cv2.BORDER_REFLECT_101)


def find_specks(
    L: np.ndarray,
    region: np.ndarray,
    width: float,
    z_thr: float,
    floor_l: float,
    noise: float,
    face_width: float,
    polarities: Sequence[str] = ("bright", "dark"),
    fibre_len: Optional[float] = None,
    specular_l: Optional[float] = None,
    min_sharp: Optional[float] = None,
    bright_region: Optional[np.ndarray] = None,
    min_surface_l: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Boolean mask of specks in ``L`` (L* 0-100, float32) inside ``region``.

    Args:
        L: (H, W) L* image.
        region: (H, W) bool/0-1 search region.
        width: largest speck width (px); the structuring element size.
        z_thr: contrast threshold in surround-spread units.
        floor_l: contrast floor (L* units).
        noise: frame sensor-noise sigma of L* (spread floor).
        face_width: face width in px (density window, fibre length).
        polarities: "bright" and/or "dark".
        fibre_len: longest thin fibre (px); None = compact specks only.
        specular_l: bright components peaking at or above this L* are glints.
        min_sharp: steepest edge slope (L* per px) over peak contrast a speck
            needs (in focus); None skips the test.
        bright_region: where light specks are looked for (default: region).
            Light specks on the backdrop are lights and reflections.
        min_surface_l: only look where the speck-free surface is at least
            this light (L*), e.g. sensor dust, which only shows on light,
            even areas.
    """
    counts = {"candidates": 0, "texture": 0, "gloss": 0, "joined": 0, "soft": 0, "shape": 0, "region": 0, "glint": 0, "busy": 0, "kept": 0}
    H, W = L.shape
    k = max(3, int(round(1.5 * width)) | 1)
    k2 = 2 * k + 1
    L8 = np.clip(L * 2.55 + 0.5, 0, 255).astype(np.uint8)
    se, se2 = _rect(k), _rect(k2)
    opened = _morph(L8, cv2.MORPH_OPEN, se)
    closed = _morph(L8, cv2.MORPH_CLOSE, se)
    # Surround: small bright and dark features both removed.
    base8 = _morph(opened, cv2.MORPH_CLOSE, se)
    # Unit: local RMS of both top-hats over a wide window, i.e. how far this
    # surface's own grain, weave or noise pokes out at the speck's scale. A
    # lone speck adds little to it (it covers a small share of the window).
    tw = (L8.astype(np.int16) - opened).astype(np.float32) / 2.55
    tb = (closed.astype(np.int16) - L8).astype(np.float32) / 2.55
    s_tex = max(3.0, 3.0 * k)
    spread = np.sqrt(_blur(tw * tw + tb * tb, s_tex))
    spread = np.maximum(spread, noise + 0.05)
    # Surround range: how much the speck-free surface itself changes within
    # twice the element (an edge, seam or panel boundary nearby).
    rng = (_morph(base8, cv2.MORPH_DILATE, se2).astype(np.int16)
           - _morph(base8, cv2.MORPH_ERODE, se2)).astype(np.float32) / 2.55
    reg = np.asarray(region) > 0.5
    if min_surface_l is not None:
        reg = reg & (base8 >= np.uint8(min(255, int(round(min_surface_l * 2.55)))))

    seeds = np.zeros((H, W), bool)
    seeds_b = np.zeros((H, W), bool)
    seeds_ref = np.zeros((H, W), bool)
    seeds_bref = np.zeros((H, W), bool)
    grown = np.zeros((H, W), bool)
    grown2 = np.zeros((H, W), bool)
    for pol in polarities:
        if pol == "bright":
            top = tw
            top2 = (L8.astype(np.int16) - _morph(L8, cv2.MORPH_OPEN, se2)).astype(np.float32) / 2.55
        else:
            top = tb
            top2 = (_morph(L8, cv2.MORPH_CLOSE, se2).astype(np.int16) - L8).astype(np.float32) / 2.55
        thr = np.maximum(z_thr * spread, floor_l)
        # Growth and the isolation test use the strength-0 threshold, so a
        # higher strength only adds fainter seeds and never merges a speck
        # into its surroundings.
        thr_ref = np.maximum(_Z_LO * spread, floor_l)
        flat = top > _RANGE_RATIO * rng
        narrow = top * _BROAD_RATIO >= top2
        rg = reg if (pol == "dark" or bright_region is None) else (reg & (np.asarray(bright_region) > 0.5))
        s = (top > thr) & flat & narrow & rg
        g = (top > _GROW * thr_ref) & rg
        seeds |= s
        # Texture and gloss counts use a fixed (strength-0) threshold, so a
        # higher strength does not make a surface look busier.
        s_ref = (top > np.maximum(_COUNT_Z * spread, floor_l)) & rg
        seeds_ref |= s_ref
        if pol == "bright":
            seeds_b |= s
            seeds_bref |= s_ref
        grown |= g | s
        grown2 |= (top2 > _ISO_GROW * thr_ref) & reg
    if not seeds.any():
        return np.zeros((H, W), bool), counts

    n, lab_img, stats, cents = cv2.connectedComponentsWithStats(grown.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros((H, W), bool), counts
    # Keep only grown components that contain a seed.
    has_seed = np.zeros(n, bool)
    has_seed[np.unique(lab_img[seeds])] = True
    has_seed[0] = False
    ids = np.nonzero(has_seed)[0]
    counts["candidates"] = int(ids.size)
    if ids.size == 0:
        return np.zeros((H, W), bool), counts

    keep = has_seed.copy()
    bw, bh, area = stats[:, cv2.CC_STAT_WIDTH], stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_AREA]
    longest = np.maximum(bw, bh).astype(np.float32)

    # Texture: candidates (seeded components) clustered within the window.
    win = max(5, int(round(_DENSITY_FW * face_width)) | 1)
    ref = np.zeros(n, bool)
    ref[np.unique(lab_img[seeds_ref])] = True
    ref[0] = False
    rids = np.nonzero(ref)[0]
    cnt = np.zeros((H, W), np.float32)
    np.add.at(cnt, (np.clip(cents[rids, 1].round().astype(int), 0, H - 1),
                    np.clip(cents[rids, 0].round().astype(int), 0, W - 1)), 1.0)
    cx = np.clip(cents[ids, 0].round().astype(int), 0, W - 1)
    cy = np.clip(cents[ids, 1].round().astype(int), 0, H - 1)
    dens = _box_count(cnt, win)
    texture = np.zeros(n, bool)
    texture[ids] = dens[cy, cx] > _DENSITY_MAX
    counts["texture"] = int(texture[ids].sum())
    keep &= ~texture

    # Gloss: several bright candidates across the same stretch of surface
    # are glints on vinyl, satin or metal, not lint.
    bright = np.zeros(n, bool)
    bright[np.unique(lab_img[seeds_b])] = True
    bright[0] = False
    bids = np.nonzero(bright & has_seed)[0]
    if bids.size:
        gwin = max(5, int(round(_GLOSS_FW * face_width)) | 1)
        gc = np.zeros((H, W), np.float32)
        bref = np.zeros(n, bool)
        bref[np.unique(lab_img[seeds_bref])] = True
        bref[0] = False
        brids = np.nonzero(bref)[0]
        np.add.at(gc, (np.clip(cents[brids, 1].round().astype(int), 0, H - 1),
                       np.clip(cents[brids, 0].round().astype(int), 0, W - 1)), 1.0)
        bx = np.clip(cents[bids, 0].round().astype(int), 0, W - 1)
        by = np.clip(cents[bids, 1].round().astype(int), 0, H - 1)
        gd = _box_count(gc, gwin)
        glossy = np.zeros(n, bool)
        glossy[bids] = gd[by, bx] > _GLOSS_MAX
        # Sheen: a broad highlight on the speck-free surface close by means
        # the surface is glossy (vinyl, latex, satin); a light speck there is
        # a glint. Measured on a reduced copy at four times the element.
        f = max(1, k // 6)
        osm = cv2.resize(opened, (max(1, W // f), max(1, H // f)), interpolation=cv2.INTER_AREA)
        k4 = max(3, int(round(4 * k / f)) | 1)
        sheen = (osm.astype(np.int16) - _morph(osm, cv2.MORPH_OPEN, _rect(k4))).astype(np.float32) / 2.55
        kw = max(3, int(round(2 * k / f)) | 1)
        sheen = _morph(sheen, cv2.MORPH_DILATE, _rect(kw))
        sy = np.clip(by // f, 0, sheen.shape[0] - 1)
        sx = np.clip(bx // f, 0, sheen.shape[1] - 1)
        pkb = np.zeros(n, np.float32)
        ysb, xsb = np.nonzero(seeds_b)
        np.maximum.at(pkb, lab_img[ysb, xsb], tw[ysb, xsb])
        glossy[bids] |= sheen[sy, sx] > _SHEEN_SHARE * pkb[bids]
        counts["gloss"] = int((keep & glossy).sum())
        keep &= ~glossy

    # Isolation across scales: at twice the element a speck is still the
    # same small blob; the tip of a gap, strap or panel joins something
    # larger.
    sel0 = keep[lab_img]
    ys0, xs0 = np.nonzero(sel0)
    if ys0.size:
        _, lab2, st2, _ = cv2.connectedComponentsWithStats(grown2.astype(np.uint8), connectivity=8)
        a2 = np.zeros(n, np.float64)
        np.maximum.at(a2, lab_img[ys0, xs0], st2[lab2[ys0, xs0], cv2.CC_STAT_AREA] * (lab2[ys0, xs0] > 0))
        joined = keep & (a2 > _ISO_RATIO * np.maximum(area, 1))
        counts["joined"] = int(joined.sum())
        keep &= ~joined
        if min_sharp is not None:
            # In focus: lint lies on the subject; a soft blob of the same
            # size is a blurred glint or a background light.
            Ls = cv2.GaussianBlur(L, (0, 0), sigmaX=0.7)
            gx = cv2.Sobel(Ls, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(Ls, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            li0 = lab_img[ys0, xs0]
            g = np.zeros(n, np.float32)
            np.maximum.at(g, li0, np.hypot(gx[ys0, xs0], gy[ys0, xs0]))
            pk = np.zeros(n, np.float32)
            np.maximum.at(pk, li0, np.maximum(tw[ys0, xs0], tb[ys0, xs0]))
            # Edge pixels just outside the component carry the steepest slope.
            soft = keep & (g < min_sharp * np.maximum(pk, 1e-3))
            dil = cv2.dilate((keep[lab_img]).astype(np.uint8), _rect(3)) > 0
            yd, xd = np.nonzero(dil & ~sel0)
            if yd.size:
                near = cv2.dilate(np.where(sel0, lab_img, 0).astype(np.float32), _rect(3))[yd, xd].astype(np.int64)
                gd = np.zeros(n, np.float32)
                np.maximum.at(gd, near, np.hypot(gx[yd, xd], gy[yd, xd]))
                soft = keep & (np.maximum(g, gd) < min_sharp * np.maximum(pk, 1e-3))
            counts["soft"] = int(soft.sum())
            keep &= ~soft

    compact = longest <= width * 1.3
    if fibre_len is not None:
        # A diagonal fibre's length is about its box diagonal.
        span = np.hypot(bw, bh).astype(np.float32)
        thin = (span <= fibre_len) & (area / np.maximum(span, 1.0) <= max(3.5, 0.5 * width))
        shape_ok = compact | thin
    else:
        shape_ok = compact
    counts["shape"] = int((keep & ~shape_ok).sum())
    keep &= shape_ok

    # Per-component stats over component pixels only.
    ys, xs = np.nonzero(lab_img)
    li = lab_img[ys, xs]
    sel = keep[li]
    ys, xs, li = ys[sel], xs[sel], li[sel]
    if li.size == 0:
        return np.zeros((H, W), bool), counts
    npx = np.bincount(li, minlength=n).astype(np.float32)
    in_reg = np.bincount(li, weights=reg[ys, xs].astype(np.float32), minlength=n)
    bad_reg = keep & (in_reg < 0.9 * np.maximum(npx, 1))
    counts["region"] = int(bad_reg.sum())
    keep &= ~bad_reg

    if specular_l is not None:
        peak = np.zeros(n, np.float32)
        is_bright = L[ys, xs] > base8[ys, xs] / 2.55
        np.maximum.at(peak, li, np.where(is_bright, L[ys, xs], 0.0).astype(np.float32))
        glint = keep & (peak >= specular_l)
        counts["glint"] = int(glint.sum())
        keep &= ~glint

    # (Skipped for the soft sensor-dust pass, whose halo spreads past the
    # ring; on the backdrop the isolation and range tests carry it.)
    # Busy surround: other features of comparable contrast within three
    # widths (studs, rhinestones, a cluster of glints, print, an edge) mean
    # this is part of the surface's own detail. A lone fleck of lint sits on
    # surface that only has grain around it.
    keep[0] = False
    kids = np.nonzero(keep)[0] if min_sharp is not None else np.zeros(0, int)
    if kids.size:
        r = int(round(3.0 * width)) + 2
        busy = np.zeros(n, bool)
        for i in kids:
            x, y, bw_i, bh_i = (int(v) for v in stats[i, :4])
            y0, y1 = max(0, y - r), min(H, y + bh_i + r)
            x0, x1 = max(0, x - r), min(W, x + bw_i + r)
            t = np.maximum(tw[y0:y1, x0:x1], tb[y0:y1, x0:x1])
            t = cv2.GaussianBlur(t, (0, 0), sigmaX=1.0)
            own = (lab_img[y0:y1, x0:x1] == i).astype(np.uint8)
            peak = float(t[own > 0].max())
            near = cv2.dilate(own, _rect(5)) > 0
            others = (t > _BUSY_SHARE * peak) & ~near
            busy[i] = int(others.sum()) > max(4, int(0.5 * area[i]))
            if not busy[i]:
                # The ring right around it must be one even surface (a
                # glint's ring holds the sheen it sits in).
                kr = max(3, int(round(width)) | 1)
                ring = (cv2.dilate(own, _rect(kr + 4)) > 0) & ~near
                lv = L[y0:y1, x0:x1][ring]
                if lv.size >= 8:
                    q1, q3 = np.percentile(lv, [10, 90])
                    busy[i] = (q3 - q1) > _RING_SHARE * peak
        counts["busy"] = int(busy.sum())
        keep &= ~busy
    keep[0] = False
    counts["kept"] = int(keep.sum())
    if _logger.isEnabledFor(logging.DEBUG):
        for i in np.nonzero(keep)[0]:
            _logger.debug("kept speck at (%d, %d) area %d box %dx%d bright %s",
                          int(cents[i, 0]), int(cents[i, 1]), int(area[i]), int(bw[i]), int(bh[i]),
                          bool(bright[i]))
    return keep[lab_img], counts


def heal_specks(
    img: np.ndarray,
    specks: np.ndarray,
    width: float,
    soft_edge: bool = False,
) -> Tuple[np.ndarray, int]:
    """Replace each speck with its speck-free surface plus nearby grain.

    The surface is the morphological opening-then-closing of each channel at
    the detection element (small bright and dark features removed, the rest
    kept, so no colour bleeds in from beyond the element). It is smoothed
    lightly and the fine grain of a clean donor patch beside the speck is
    added back. Works per speck on a small window; other pixels are untouched.
    """
    if not specks.any():
        return img, 0
    H, W = specks.shape
    k = max(3, int(round(1.5 * width)) | 1)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # A soft sensor-dust spot's halo runs past its detected core: widen and
    # feather more (the surface equals the photo away from the spot, so a
    # wider blend costs nothing).
    sig = max(1.5, k / 8.0) if soft_edge else 1.5
    kd = (max(5, int(round(k / 3.0)) | 1)) if soft_edge else 5
    m_all = cv2.dilate(specks.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
    n, lab_img, stats, _ = cv2.connectedComponentsWithStats(m_all, connectivity=8)
    out = img.copy()
    total = 0
    for i in range(1, n):
        x, y, bw, bh = (int(v) for v in stats[i, :4])
        d = max(bw, bh) + int(4 * sig) + 2
        pad = 2 * k + d + int(4 * sig) + 2
        y0, y1 = max(0, y - pad), min(H, y + bh + pad)
        x0, x1 = max(0, x - pad), min(W, x + bw + pad)
        crop = np.clip(img[y0:y1, x0:x1], 0.0, 1.0).astype(np.float32)
        m = (lab_img[y0:y1, x0:x1] == i).astype(np.float32)
        surf = np.empty_like(crop)
        for c in range(3):
            o = _morph(crop[..., c], cv2.MORPH_OPEN, se)
            surf[..., c] = _morph(o, cv2.MORPH_CLOSE, se)
        surf = cv2.GaussianBlur(surf, (0, 0), sigmaX=max(1.0, k / 6.0))
        grain = crop - cv2.GaussianBlur(crop, (0, 0), sigmaX=1.2)
        soft = np.maximum(cv2.GaussianBlur(m, (0, 0), sigmaX=sig), m)
        hole = soft > 0.004
        gsel = np.zeros_like(grain)
        got = np.zeros(m.shape, bool)
        for dy, dx in ((0, d), (0, -d), (d, 0), (-d, 0)):
            sh = np.roll(np.roll(hole, dy, axis=0), dx, axis=1)
            gg = np.roll(np.roll(grain, dy, axis=0), dx, axis=1)
            take = hole & ~sh & ~got
            gsel[take] = gg[take]
            got |= take
        a = soft[..., None]
        healed = np.clip(surf + gsel, 0.0, 1.0)
        blend = crop * (1.0 - a) + healed * a
        region = out[y0:y1, x0:x1]
        region[hole] = blend[hole]
        total += int(hole.sum())
    return out, total


def apply_lint_dust(
    img: np.ndarray,
    strength: float,
    face_boxes: Sequence[Tuple[int, int, int, int]],
    segment_classes=None,
    hair_full=None,
    person_mask: Optional[np.ndarray] = None,
    protect: Optional[np.ndarray] = None,
    face_skin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Lint & Dust Cleanup on a float32 BGR [0, 1] image; returns ``(image, diag)``.

    Args:
        img: (H, W, 3) float32 BGR [0, 1].
        strength: 0-1 (0 is a no-op).
        face_boxes: (x, y, w, h) per face, full-resolution pixels.
        segment_classes: callable(uint8 BGR) -> (h, w, 6) confidences or None.
        hair_full: callable(uint8 BGR) -> (h, w) hair confidence or None.
        person_mask: (H, W) person mask.
        protect: (H, W) union of the face pipeline's skin, hair and lip masks.
        face_skin: (H, W) face skin mask.
    """
    diag: Dict[str, Any] = {"applied": False}
    if strength <= 0:
        diag["reason"] = "off"
        return img, diag
    s = float(np.clip(strength, 0.0, 1.0))
    H, W = img.shape[:2]
    fw_full = float(np.median([b[2] for b in face_boxes])) if face_boxes else 0.1 * min(H, W)
    fw_full = max(fw_full, 40.0)
    scale = min(1.0, _WORK_FACE_WIDTH / fw_full)
    size = (max(16, int(round(W * scale))), max(16, int(round(H * scale))))
    work = img if scale >= 1.0 else cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    work = np.clip(work, 0.0, 1.0).astype(np.float32)
    work_u8 = (work * 255.0 + 0.5).astype(np.uint8)
    boxes_s = [
        (int(x * scale), int(y * scale), max(1, int(bw * scale)), max(1, int(bh * scale)))
        for (x, y, bw, bh) in face_boxes
    ]
    probs = segment_classes(work_u8) if segment_classes is not None else None
    hair = hair_full(work_u8) if hair_full is not None else None
    region_s, backdrop_s, subject_s, model, rdiag = _region_mask(
        work, boxes_s, probs, _as_2d(hair, size), _as_2d(protect, size),
        _as_2d(face_skin, size), _as_2d(person_mask, size), fw_full * scale,
    )
    diag.update(rdiag)
    if region_s is None:
        return img, diag

    def _up(m: np.ndarray) -> np.ndarray:
        if scale >= 1.0:
            return m > 0.5
        return cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5

    region = _up(region_s)
    src = np.clip(img, 0.0, 1.0).astype(np.float32)
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    noise = _frame_noise_sigma(src)
    backdrop_full = _up(backdrop_s) if backdrop_s is not None else None
    specular_l = float(np.percentile(L[::4, ::4], _SPECULAR_PCT))
    width = (_W_FW_LO + (_W_FW_HI - _W_FW_LO) * s) * fw_full
    z_thr = _Z_LO + (_Z_HI - _Z_LO) * s
    specks, c1 = find_specks(
        L, region, width, z_thr, _FLOOR_L, noise, fw_full,
        fibre_len=_FIBRE_LEN_FW * fw_full, specular_l=specular_l,
        min_sharp=_MIN_SHARP,
        bright_region=None if subject_s is None else _up(subject_s),
    )
    diag["fine"] = c1
    out = src
    out, n1 = heal_specks(out, specks, width)
    diag["pixels"] = n1
    if backdrop_s is not None:
        backdrop = backdrop_full
        if backdrop.any():
            dw = _DUST_W_FW * fw_full
            Lb = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0] if n1 else L
            dust, c2 = find_specks(
                Lb, backdrop, dw, z_thr, _DUST_FLOOR_L, noise, fw_full, polarities=("dark",),
                # Sensor dust shows on the lighter half of the backdrop
                # (relative to this photo's own backdrop).
                min_surface_l=float(np.median(Lb[::4, ::4][backdrop[::4, ::4]])),
            )
            diag["sensor_dust"] = c2
            out, n2 = heal_specks(out, dust, dw, soft_edge=True)
            diag["pixels"] += n2
    if diag["pixels"] == 0:
        diag["reason"] = "no_specks"
        return img, diag
    diag["applied"] = True
    # Untouched pixels keep the caller's exact values (incl. out-of-range).
    changed = np.any(out != src, axis=2)
    res = img.copy()
    res[changed] = out[changed]
    _logger.info("lint & dust: %s", diag)
    return res, diag
