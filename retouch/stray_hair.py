"""Stray hair cleanup over skin (``hair_remove_flyaways``, GUI "Stray Hair Cleanup").

Loose wig fibres and flyaways that land on a cheek, the forehead, the neck
or a shoulder are thin, long, smoothly curving lines of a colour that isn't
the skin's. This op finds them on the skin (face skin and the neck/chest
skin parsed with the face), and heals each one from the skin beside it. The
hair itself, bangs, brows, lashes, eyes and lips are never searched.

Why it replaced ``hairwork.remove_flyaways``: that version searched the
whole hair mask plus a silhouette band and, optionally, the skin with a
morphological top-hat/black-hat at 2-4 px, normalised against the single
largest response in the band, then Telea-inpainted everything above a
fraction of that maximum with a radius of 2% of the face width. At 26 MP a
wig fibre is a few pixels wide while the strongest response in the band
comes from lash lines, wig edges and costume trim, so real flyaways over
skin sat far below the cut-off, and what it did inpaint inside the wig was
strand texture smeared at a 15-20 px radius.

How this version works, measured against the face itself:

* line detection is a multi-scale Hessian ridge measure on L at fibre
  scales (fractions of the face width), in units of the skin's own robust
  spread of that measure at each scale, so the same setting finds the same
  strands on pale, painted and darker skin and doesn't fire on pores;
* candidates must be line-shaped (one strong principal curvature, the other
  small), long (skeleton length) and thin (area / length), so pores, spots,
  freckles and broad shading are left alone;
* dark lines also have to move the colour away from the skin's own hue
  toward the hair's colour (or be clearly darker than any shading of that
  skin), which keeps skin creases (smile lines, neck rings, crow's feet);
* each strand is filled from the skin around it (normalised convolution
  that ignores the strand) plus fine texture taken from the skin a few
  strand widths to the side, along the strand's normal, so the healed line
  keeps the skin's grain rather than turning into a smooth smear.

No absolute intensity threshold is used. Strength 0 is a no-op.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import bgr_f32_to_lab_f32, normalize_mask

# Fibre scales (Gaussian sigma) as fractions of the face width, floored in px.
_SIGMAS_FW = (0.0016, 0.0026, 0.0042)
_SIGMA_FLOOR_PX = 0.9
# Ridge z-score threshold (robust sigmas of the skin's own ridge response).
# Lerped from _Z_AT_MIN (strength 0) to _Z_AT_MAX (strength 100), in units
# of the line-integrated response's own spread on the skin.
_Z_AT_MIN = 30.0
_Z_AT_MAX = 12.0
# A strand's skeleton should run fairly straight (end-to-end extent over
# skeleton length); curlier chains need a much stronger response.
_STRAIGHT_MIN = 0.6
_CURLY_Z_FACTOR = 3.0
# Skin edge band (fraction of face width) next to anything that isn't hair
# (collar, costume, background): the face outline and its shadow are lines
# there, a strand coming off the wig is not.
_OUTLINE_BAND_FW = 0.025
# Side test: a stray on skin has skin on both sides; the wig's outermost
# strand (its outline) has hair on one side. Per strand pixel, the colour
# distance from the strand to its nearer side over that to its farther
# side; a strand whose median ratio is below this is the wig's edge.
_SIDE_RATIO_MIN = 0.35
_SIDE_DIST_SIGMAS = 4.0
# Hysteresis: weak pixels connected to a strong core are kept.
_Z_WEAK_FRAC = 0.4
# Line-shape: |lambda2| / |lambda1| must stay under this.
_ANISO_MAX = 0.45
# Minimum strand length (skeleton px), fraction of face width.
_MIN_LEN_FW = 0.045
# Maximum mean strand width, in multiples of the largest detection sigma.
_MAX_WIDTH_SIGMAS = 3.2
# Line integration: length (fraction of face width) and angle bins.
_INTEGRATE_LEN_FW = 0.05
_N_ANGLES = 12
# Raw (un-integrated) ridge z a pixel needs to belong to a strand.
_RAW_FLOOR = 1.0
# Step-edge ratio above which a ridge response is a step's flank, not a line.
_EDGE_RATIO_MAX = 0.6
# Hair confidence above which a pixel counts as the wig itself (never searched).
_HAIR_CORE = 0.6


def _robust_sigma(v: np.ndarray) -> float:
    if v.size < 16:
        return 0.0
    med = float(np.median(v))
    return float(1.4826 * np.median(np.abs(v - med)))


def _hessian(g: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale-normalised Hessian of an image already blurred at ``sigma``."""
    dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
    dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
    dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
    # ksize=3 Sobel second derivatives have a gain of 4.
    s2 = sigma * sigma / 4.0
    return dxx * s2, dyy * s2, dxy * s2


def _eig(a: np.ndarray, c: np.ndarray, b: np.ndarray):
    """Eigen-decomposition of [[a, b], [b, c]] per pixel.

    Returns (lam1, lam2, nx, ny): lam1 is the eigenvalue of larger magnitude,
    (nx, ny) its unit eigenvector (the strand normal).
    """
    tr = 0.5 * (a + c)
    d = np.sqrt(0.25 * (a - c) ** 2 + b * b)
    e1 = tr + d
    e2 = tr - d
    big = np.abs(e1) >= np.abs(e2)
    lam1 = np.where(big, e1, e2)
    lam2 = np.where(big, e2, e1)
    vx = b
    vy = lam1 - a
    alt = np.abs(vx) + np.abs(vy) < 1e-6
    vx = np.where(alt, lam1 - c, vx)
    vy = np.where(alt, b, vy)
    n = np.sqrt(vx * vx + vy * vy) + 1e-9
    return lam1, lam2, vx / n, vy / n


def _masked_blur(img: np.ndarray, w: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    wb = cv2.GaussianBlur(w, (0, 0), sigma)
    if img.ndim == 3:
        num = cv2.GaussianBlur(img * w[..., None], (0, 0), sigma)
        return num / np.maximum(wb, 1e-6)[..., None], wb
    num = cv2.GaussianBlur(img * w, (0, 0), sigma)
    return num / np.maximum(wb, 1e-6), wb


# Feature exclusions: (region names, margin as a fraction of face width).
# Eyes get a wide margin: lash lines, liner, lid creases and aegyo-sal
# makeup are all long thin lines of a non-skin colour.
_EXCLUDE_GROUPS = (
    (("left_eye", "right_eye", "left_iris", "right_iris",
      "left_eyebrow", "right_eyebrow"), 0.09),
    (("left_under_eye", "right_under_eye"), 0.01),
    (("lips", "mouth_interior"), 0.03),
    # Nostril rims and the alar crease are dark curved lines.
    (("nose",), 0.07),
    # Smile lines are long dark curves of skin colour.
    (("nasolabial_l", "nasolabial_r"), 0.015),
)


def exclusion_from_regions(regions: Any, face_width: float) -> Optional[np.ndarray]:
    """Mask (0/1 float32) of the facial features a strand search keeps clear of.

    Eyes, brows, lashes and eye makeup, the under-eye area, lips and mouth,
    and the nose (nostril rims, alar crease), each grown by a margin scaled
    to the face width.
    """
    out = None
    for names, margin in _EXCLUDE_GROUPS:
        parts = []
        for name in names:
            m = getattr(regions, name, None)
            if m is not None and getattr(m, "size", 0):
                parts.append(normalize_mask(m) > 0.3)
        if not parts:
            continue
        u = np.logical_or.reduce(parts).astype(np.uint8)
        g = max(int(round(margin * face_width)), 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
        u = cv2.dilate(u, k)
        out = u if out is None else np.maximum(out, u)
    return None if out is None else out.astype(np.float32)


def detect_stray_hairs(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    face_width: float,
    strength: float,
    hair_mask: Optional[np.ndarray] = None,
    feature_mask: Optional[np.ndarray] = None,
    _records: Optional[list] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Find stray hairs lying on skin.

    Args:
        img_bgr: (H, W, 3) float32 or uint8 BGR in [0, 255].
        skin_mask: (H, W) skin mask (face skin plus neck/body skin).
        face_width: face width in px (sets the fibre scales).
        strength: 0-100; higher finds fainter strands.
        hair_mask: optional hair mask: the wig itself is never searched, and
            its colour is what a dark strand must lean toward.
        feature_mask: optional mask of areas never searched (see
            :func:`exclusion_from_regions`, which already adds margins).

    Returns:
        (strand_mask, normal_x, normal_y, diags): strand_mask is a binary
        float32 mask of the strands' cores; normals are the strand normal
        direction per pixel (only meaningful on the mask).
    """
    h, w = img_bgr.shape[:2]
    diags: Dict[str, Any] = {"strands": 0}
    zero = np.zeros((h, w), np.float32)
    skin = normalize_mask(skin_mask)
    if skin is None or strength <= 0 or face_width <= 0:
        return zero, zero, zero, diags
    if skin.shape != (h, w):
        skin = cv2.resize(skin, (w, h), interpolation=cv2.INTER_LINEAR)
    support = skin > 0.3

    if feature_mask is not None:
        feat = normalize_mask(feature_mask)
        if feat.shape != (h, w):
            feat = cv2.resize(feat, (w, h), interpolation=cv2.INTER_LINEAR)
        support &= feat < 0.5

    hair = None
    if hair_mask is not None:
        hair = normalize_mask(hair_mask)
        if hair.shape != (h, w):
            hair = cv2.resize(hair, (w, h), interpolation=cv2.INTER_LINEAR)
        support &= hair < _HAIR_CORE

    other = (skin <= 0.3) & ((hair < 0.3) if hair is not None else True)
    if other.any():
        g = max(int(round(_OUTLINE_BAND_FW * face_width)), 1)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))
        support &= cv2.dilate(other.astype(np.uint8), k) == 0

    if support.sum() < 400:
        return zero, zero, zero, diags

    img = img_bgr.astype(np.float32, copy=False)
    lab = bgr_f32_to_lab_f32(img)
    L = lab[:, :, 0]

    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    z_thr = _Z_AT_MIN + (_Z_AT_MAX - _Z_AT_MIN) * s

    # Colour references: the skin's own median and the hair's direction
    # away from it (Lab). A strand lying on skin pulls its pixels toward
    # the hair colour; skin texture, shine and shading mostly don't.
    sk = lab[support]
    skin_med = np.median(sk, axis=0)
    hair_dir = None
    if hair is not None and (hair > 0.8).sum() > 200:
        hd = (np.median(lab[hair > 0.8], axis=0) - skin_med).astype(np.float32)
        if float(np.linalg.norm(hd)) > 2.0:
            hair_dir = hd / float(np.linalg.norm(hd))

    # Channels searched for ridges: L (either sign: blonde strands on darker
    # skin, dark strands on pale skin) and, when the hair colour is known,
    # the projection onto the hair direction (positive ridges only).
    channels = [(L, 0)]
    if hair_dir is not None:
        channels.append(((lab - skin_med) @ hair_dir, 1))

    # Ridge response per scale and polarity, in units of the skin's own
    # spread of that response (so pores, grain and sparkle set the floor).
    sigmas = [max(_SIGMA_FLOOR_PX, f * face_width) for f in _SIGMAS_FW]
    inner = cv2.erode(support.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    pol_z = {}      # polarity key -> (z, nx, ny)
    for ci, (chan, is_hair) in enumerate(channels):
        chan = np.ascontiguousarray(chan, dtype=np.float32)
        for sg in sigmas:
            g = cv2.GaussianBlur(chan, (0, 0), sg)
            a, c, b = _hessian(g, sg)
            lam1, lam2, nx, ny = _eig(a, c, b)
            aniso = np.abs(lam2) / (np.abs(lam1) + 1e-6)
            unit = _robust_sigma(lam1[inner])
            if unit <= 1e-6:
                continue
            line = np.abs(lam1) * np.clip(1.0 - aniso / _ANISO_MAX, 0.0, 1.0)
            # Step-edge ratio: sigma * |gradient| over sigma^2 * |lambda1|.
            # A blurred step peaks in |lambda1| exactly where this ratio is
            # 1; a ridge's core sits near 0. Keeps the wig's own edge,
            # collar lines and the jaw outline from reading as strands.
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8.0
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8.0
            edge = (np.sqrt(gx * gx + gy * gy) * sg) / (np.abs(lam1) + 1e-6)
            line = line * np.clip((_EDGE_RATIO_MAX - edge) / (0.5 * _EDGE_RATIO_MAX), 0.0, 1.0)
            z = (line / unit).astype(np.float32)
            # lam1 < 0: bright ridge in this channel; > 0: dark ridge.
            for pol, sel in ((1, lam1 < 0), (-1, lam1 > 0)):
                if is_hair and pol < 0:
                    continue
                key = (ci, pol)
                zp = np.where(sel, z, 0.0).astype(np.float32)
                if key not in pol_z:
                    pol_z[key] = (zp, nx, ny)
                else:
                    z0, nx0, ny0 = pol_z[key]
                    upd = zp > z0
                    pol_z[key] = (np.where(upd, zp, z0), np.where(upd, nx, nx0),
                                  np.where(upd, ny, ny0))

    # Integrate each polarity's response along its own local direction
    # (a matched line filter): a strand stays strong over its whole length
    # while texture, whose ridge directions are random, averages out. The
    # integrated map is re-normalised by its own spread on the skin.
    half = max(3, int(round(0.5 * _INTEGRATE_LEN_FW * face_width)))
    best_z = np.zeros((h, w), np.float32)
    best_raw = np.zeros((h, w), np.float32)
    best_nx = np.zeros((h, w), np.float32)
    best_ny = np.zeros((h, w), np.float32)
    best_sign = np.zeros((h, w), np.float32)
    best_hairch = np.zeros((h, w), bool)
    for (ci, pol), (zp, nx, ny) in pol_z.items():
        tang = np.arctan2(nx, -ny)  # tangent = normal rotated 90 deg
        acc = np.zeros((h, w), np.float32)
        for k in range(_N_ANGLES):
            th = np.pi * k / _N_ANGLES
            wk = np.cos(tang - th) ** 2
            wk = (wk ** 8).astype(np.float32)
            ker = np.zeros((2 * half + 1, 2 * half + 1), np.float32)
            cv2.line(ker, (int(round(half - half * np.cos(th))), int(round(half - half * np.sin(th)))),
                     (int(round(half + half * np.cos(th))), int(round(half + half * np.sin(th)))), 1.0, 1, cv2.LINE_AA)
            ker /= ker.sum()
            acc = np.maximum(acc, cv2.filter2D(zp * wk, -1, ker, borderType=cv2.BORDER_REFLECT))
        unit = _robust_sigma(acc[inner])
        med = float(np.median(acc[inner]))
        if unit <= 1e-6:
            continue
        zi = ((acc - med) / unit).astype(np.float32)
        upd = zi > best_z
        best_z = np.where(upd, zi, best_z)
        best_raw = np.where(upd, zp, best_raw)
        best_nx = np.where(upd, nx, best_nx)
        best_ny = np.where(upd, ny, best_ny)
        best_sign = np.where(upd, float(pol), best_sign)
        best_hairch = np.where(upd, ci == 1, best_hairch)
    best_z *= support
    # Keep the strand's own pixels: the integrated map also spreads along
    # the line beyond its ends and a little to the sides.
    best_z *= best_raw > _RAW_FLOOR
    strong = (best_z > z_thr).astype(np.uint8)
    weak = (best_z > z_thr * _Z_WEAK_FRAC).astype(np.uint8)
    if not strong.any():
        return zero, best_nx, best_ny, diags
    # Bridge the small gaps where strands cross each other (the Hessian is
    # not line-shaped at a crossing), so a crossed strand stays one piece.
    rc = int(np.ceil(1.5 * max(sigmas)))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rc + 1, 2 * rc + 1))
    weak = cv2.morphologyEx(weak, cv2.MORPH_CLOSE, kc) & support.astype(np.uint8)
    strong &= weak

    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(weak, connectivity=8)
    has_core = np.zeros(n, bool)
    has_core[np.unique(lab_cc[strong > 0])] = True
    has_core[0] = False

    min_len = max(8.0, _MIN_LEN_FW * face_width)
    max_w = _MAX_WIDTH_SIGMAS * max(sigmas)

    # Local skin baseline without the candidates, for the colour test.
    cand = (weak > 0) & has_core[lab_cc]
    base_w = (support & ~cv2.dilate(cand.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)).astype(np.float32)
    base, _ = _masked_blur(lab, base_w, 4.0 * max(sigmas))
    off = lab - base

    skin_ab = skin_med[1:3] - 128.0
    ab_noise = max(_robust_sigma(off[..., 1][support]), _robust_sigma(off[..., 2][support]), 0.3)

    # Colour on each side of every pixel, a few strand widths along the normal.
    sig_max = max(sigmas)
    lab_s = cv2.GaussianBlur(lab, (0, 0), sig_max)
    gy, gx = np.mgrid[0:h, 0:w].astype(np.float32)
    dside = _SIDE_DIST_SIGMAS * sig_max
    sides = [cv2.remap(lab_s, gx + sg_ * dside * best_nx, gy + sg_ * dside * best_ny,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
             for sg_ in (1.0, -1.0)]
    d_a = np.linalg.norm(sides[0] - lab_s, axis=2)
    d_b = np.linalg.norm(sides[1] - lab_s, axis=2)
    side_ratio = np.minimum(d_a, d_b) / (np.maximum(d_a, d_b) + 1e-3)

    keep = np.zeros(n, bool)
    rejected = {"short": 0, "wide": 0, "colour": 0}
    for i in range(1, n):
        if not has_core[i]:
            continue
        x, y, ww, hh, area = stats[i]
        comp = (lab_cc[y:y + hh, x:x + ww] == i).astype(np.uint8)
        skel = cv2.ximgproc.thinning(comp * 255) if hasattr(cv2, "ximgproc") else comp
        length = float(np.count_nonzero(skel))
        pts = cv2.findNonZero(skel)
        diam = max(cv2.minAreaRect(pts)[1]) if pts is not None and len(pts) > 2 else 0.0
        if _records is not None:
            _records.append({"i": i, "bbox": (x, y, ww, hh), "len": length,
                             "straight": diam / max(length, 1.0),
                             "zpk": float(best_z[y:y + hh, x:x + ww][comp > 0].max())})
        if length < min_len:
            rejected["short"] += 1
            continue
        if float(np.median(side_ratio[y:y + hh, x:x + ww][comp > 0])) < _SIDE_RATIO_MIN:
            rejected["wig_edge"] = rejected.get("wig_edge", 0) + 1
            continue
        zpk = float(best_z[y:y + hh, x:x + ww][comp > 0].max())
        if diam / max(length, 1.0) < _STRAIGHT_MIN and zpk < _CURLY_Z_FACTOR * z_thr:
            rejected["curly"] = rejected.get("curly", 0) + 1
            continue
        # Width of the strand's core (response above half its own peak), at
        # the skeleton (2 x distance to the edge), median: independent of
        # contrast, and robust to two strands bridged at a crossing.
        zc = best_z[y:y + hh, x:x + ww]
        peak = float(np.percentile(zc[comp > 0], 90))
        core = ((zc >= 0.5 * peak) & (comp > 0)).astype(np.uint8)
        dt = cv2.distanceTransform(np.pad(core, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        sk = (skel > 0) & (core > 0)
        width = 2.0 * float(np.median(dt[sk])) if sk.any() else 0.0
        if width > max_w:
            rejected["wide"] += 1
            continue
        pix = comp > 0
        o = off[y:y + hh, x:x + ww][pix]
        sign = float(np.median(best_sign[y:y + hh, x:x + ww][pix]))
        via_hair = float(np.mean(best_hairch[y:y + hh, x:x + ww][pix])) > 0.5
        if sign < 0 and not via_hair:
            # Dark line: skin creases are dark too. A strand moves the
            # colour toward the hair; a crease keeps the skin's hue.
            om = np.median(o, axis=0)
            ok = False
            if hair_dir is not None:
                cos = float(np.dot(om, hair_dir) / (np.linalg.norm(om) + 1e-6))
                ok = cos > 0.6
            # Chroma swing off the skin's own hue axis, in noise units.
            sab = skin_ab / (np.linalg.norm(skin_ab) + 1e-6)
            perp = abs(om[1] * sab[1] - om[2] * sab[0]) / ab_noise
            ok = ok or perp > 2.5
            if not ok:
                rejected["colour"] += 1
                continue
        keep[i] = True

    mask = keep[lab_cc].astype(np.float32)
    diags.update({
        "strands": int(keep.sum()), "candidates": int(has_core.sum()),
        "rejected": rejected, "z_thr": round(z_thr, 2),
        "sigmas": [round(v, 2) for v in sigmas],
    })
    return mask, best_nx, best_ny, diags


def remove_stray_hairs(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray],
    face_width: float,
    strength: float,
    hair_mask: Optional[np.ndarray] = None,
    feature_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Heal stray hairs lying on skin. Returns (image, diagnostics).

    The image keeps the input dtype; pixels away from detected strands are
    returned unchanged.
    """
    if strength <= 0 or skin_mask is None:
        return img_bgr, {"strands": 0}
    h, w = img_bgr.shape[:2]
    skin = normalize_mask(skin_mask)
    if skin is None or skin.max() <= 0.3:
        return img_bgr, {"strands": 0}
    if skin.shape != (h, w):
        skin = cv2.resize(skin, (w, h), interpolation=cv2.INTER_LINEAR)

    # Work inside the skin's bounding box plus a margin for the kernels.
    ys, xs = np.where(skin > 0.3)
    pad = int(np.ceil(0.06 * face_width)) + 8
    y0, y1 = max(int(ys.min()) - pad, 0), min(int(ys.max()) + pad + 1, h)
    x0, x1 = max(int(xs.min()) - pad, 0), min(int(xs.max()) + pad + 1, w)
    crop = img_bgr[y0:y1, x0:x1].astype(np.float32)

    def _c(m):
        if m is None:
            return None
        m = normalize_mask(m)
        if m.shape != (h, w):
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return m[y0:y1, x0:x1]

    skin_c = skin[y0:y1, x0:x1]
    strands, nx, ny, diags = detect_stray_hairs(
        crop, skin_c, face_width, strength,
        hair_mask=_c(hair_mask), feature_mask=_c(feature_mask),
    )
    if not strands.any():
        return img_bgr, diags

    sig = max(_SIGMA_FLOOR_PX, max(_SIGMAS_FW) * face_width)
    r = int(np.ceil(1.2 * sig)) + 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    heal = cv2.dilate(strands.astype(np.uint8), k).astype(np.float32)

    # Fill from the skin around each strand (not from the strand itself).
    good = ((skin_c > 0.3) & (heal == 0)).astype(np.float32)
    fill, wsum = _masked_blur(crop, good, 1.5 * r)
    fill2, _ = _masked_blur(crop, good, 4.0 * r)
    t = np.clip(wsum / 0.15, 0.0, 1.0)[..., None]
    fill = fill * t + fill2 * (1.0 - t)

    # Skin grain taken a few strand widths to the side, along the normal.
    tex_sig = max(0.8, 0.6 * r)
    hp = crop - cv2.GaussianBlur(crop, (0, 0), tex_sig)
    nx_s = cv2.GaussianBlur(nx * heal, (0, 0), r)
    ny_s = cv2.GaussianBlur(ny * heal, (0, 0), r)
    nn = np.sqrt(nx_s ** 2 + ny_s ** 2) + 1e-6
    nx_s, ny_s = nx_s / nn, ny_s / nn
    gy, gx = np.mgrid[0:crop.shape[0], 0:crop.shape[1]].astype(np.float32)
    dist = 3.0 * r
    tex = np.zeros_like(crop)
    got = np.zeros(crop.shape[:2], np.float32)
    for sgn in (1.0, -1.0):
        mx = gx + sgn * dist * nx_s
        my = gy + sgn * dist * ny_s
        d_hp = cv2.remap(hp, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        d_ok = cv2.remap(good, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        take = (d_ok > 0.5) & (got == 0)
        tex[take] = d_hp[take]
        got[take] = 1.0
    healed = fill + tex

    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    amount = min(1.0, 0.5 + s)
    alpha = cv2.GaussianBlur(heal, (0, 0), max(0.7, 0.4 * r))
    alpha = np.clip(alpha * amount, 0.0, 1.0)
    alpha *= np.clip(skin_c / 0.5, 0.0, 1.0)
    out_c = crop * (1.0 - alpha[..., None]) + healed * alpha[..., None]

    out = img_bgr.copy()
    if img_bgr.dtype == np.uint8:
        out[y0:y1, x0:x1] = np.clip(np.round(out_c), 0, 255).astype(np.uint8)
    else:
        out[y0:y1, x0:x1] = out_c.astype(img_bgr.dtype)
    diags["healed_px"] = int((alpha > 0.05).sum())
    return out, diags
