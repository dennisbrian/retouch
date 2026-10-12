"""Automatic spot healing: find pimples and small spots, heal each one alone.

Classical, no model weights. Unlike smoothing (which softens the whole skin
region) and the legacy ``blemish`` pass (a fixed grey-level threshold followed
by Telea inpainting, which leaves a flat, textureless patch), this op edits
only the pixels of each spot it finds and rebuilds them the way a retoucher's
healing brush does:

1. **Detect** (on a copy scaled so the face is about ``_WORK_FACE_W`` px wide).
   The local skin level is a median over a window several spot-widths wide,
   taken on an image whose non-skin pixels were first filled from nearby skin
   (so eyes, brows, lips and hair never read as the "local level"). A spot is
   a blob that is *darker* (L) and/or *redder* (a*) than that level. Both
   margins are divided by the face's own skin texture spread (robust MAD of
   the residual), so the test is relative to each face and behaves the same on
   every skin tone (CLAUDE.md, Tone-Invariance). Seeds above the threshold
   grow by hysteresis; blobs are kept only if they are spot-sized (relative to
   face width), compact (not a wrinkle, lash shadow or hair strand) and sit
   inside the skin away from its edge.
2. **Keep identity marks.** A dark blob with no extra redness that is larger
   than a pimple-sized core reads as a mole or beauty mark and is kept. Bright
   blobs (piercings, catchlights, whiteheads under flash) are never
   candidates.
3. **Heal** (at full resolution, per spot). The replacement is the local skin
   level (a normalized convolution that ignores every spot) plus fine texture
   copied from a clean donor patch next to the spot (the healing-brush idea),
   so pores continue across the healed area instead of going flat. The result
   is blended in through a feathered footprint only a little larger than the
   spot; every other pixel is returned untouched.

Strength (0-100) sets how faint and how large a spot may be and still be
healed. 0 = off (the default), so existing recipes render unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .utils import bgr_f32_to_lab_f32

__all__ = ["heal_spots", "detect_spots", "feature_mask_from_regions"]

# Detection runs with the face resized to about this width (px).
_WORK_FACE_W = 640.0

# Spot size limits as fractions of face width (face width = 2.5 x IED).
_D_MIN = 0.004           # smaller than this is a pore
_D_MAX_LO = 0.028        # largest spot healed at strength ~0
_D_MAX_HI = 0.050        # largest spot healed at strength 100
_MOLE_D = 0.016          # dark, non-red blob wider than this is kept (mole)
_MOLE_REL_DARK = 0.16    # ... or darker than its surround by this fraction of L

# Anomaly thresholds in units of the face's own skin-texture spread.
_Z_HI_LO = 7.0           # seed threshold at strength ~0
_Z_HI_HI = 4.5           # seed threshold at strength 100
_Z_GROW = 0.5            # a spot region never extends below this x seed level
_LEVELS = (0.5, 0.65, 0.8)  # spot region = above this fraction of its peak
_RED_Z = 1.5             # a* margin (in spreads) that marks a blob as red
_RED_RATIO = 0.35        # ... when the a* rise is at least this share of the L drop
_SIGMA_FLOOR_L = 1.2     # spread floors (L/a units, 0-255 LAB scale) so a
_SIGMA_FLOOR_A = 0.6     # painted, textureless face does not flag its noise

# A side of the ring around a spot may reach at most this fraction of the
# spot's own peak anomaly.
_RING_FRAC = 0.45

# Detection blur scales, as multiples of the smallest spot diameter.
_SCALES = (0.35, 0.9, 1.6)

# Smallest spot worth healing, CIE delta-E against the skin around it.
_MIN_DELTA_E = 2.5

# Shape filters.
_MAX_ASPECT = 2.6        # bounding ellipse major/minor
_MIN_FILL = 0.45         # area / (bbox area)

# Features (eyes, brows, lips, mouth, hair) are kept this far away
# (fraction of face width): lash lines, liner and lip borders read as spots.
_FEATURE_MARGIN = 0.03

# Healing.
_PAD = 0.35              # footprint grows by this fraction of the spot radius
_EXTENT = 0.2            # heal out to where the anomaly falls to this x peak
_EXTENT_R = 2.2          # ... but no further than this x the core radius
_RING_RESID = 0.6        # skip if the ring around a spot departs from a plane
                         # by more than this x the spot's own contrast
_TEXTURE_SIGMA = 0.0035  # texture band (fraction of face width)


def _norm_conv(values: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Weighted Gaussian average blur(v*w)/blur(w); falls back to v."""
    if values.ndim == 3:
        w3 = weight[:, :, None]
        num = cv2.GaussianBlur(values * w3, (0, 0), sigma)
        den = cv2.GaussianBlur(weight, (0, 0), sigma)[:, :, None]
    else:
        num = cv2.GaussianBlur(values * weight, (0, 0), sigma)
        den = cv2.GaussianBlur(weight, (0, 0), sigma)
    out = num / np.maximum(den, 1e-4)
    return np.where(den > 1e-3, out, values).astype(np.float32)


def _robust_sigma(x: np.ndarray, floor: float) -> float:
    if x.size < 16:
        return floor
    med = float(np.median(x))
    return max(1.4826 * float(np.median(np.abs(x - med))), floor)


def _to_unit_mask(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(np.float32)
    if m.size and float(m.max()) > 1.5:
        m = m / 255.0
    return np.clip(m, 0.0, 1.0)


def feature_mask_from_regions(regions: Any) -> Optional[np.ndarray]:
    """Union (0-1 float) of the eye, brow, lip, mouth and hair masks."""
    parts = []
    for name in ("left_eye", "right_eye", "left_eyebrow", "right_eyebrow",
                 "left_iris", "right_iris", "lips", "mouth_interior", "hair"):
        m = getattr(regions, name, None)
        if m is not None and getattr(m, "size", 0):
            parts.append(_to_unit_mask(m))
    if not parts:
        return None
    return np.maximum.reduce(parts)


def _effective_skin(skin: np.ndarray, feature_mask: Optional[np.ndarray],
                    face_width: float) -> np.ndarray:
    """Skin with a margin around every facial feature removed."""
    if feature_mask is None:
        return skin
    feat = (_to_unit_mask(feature_mask) > 0.3).astype(np.uint8)
    if not feat.any():
        return skin
    g = max(int(round(_FEATURE_MARGIN * face_width)), 1)
    feat = cv2.dilate(feat, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
    return np.where(feat > 0, 0.0, skin).astype(np.float32)


def detect_spots(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    face_width: float,
    strength: float,
    protect_mask: Optional[np.ndarray] = None,
    feature_mask: Optional[np.ndarray] = None,
    _rejects: Optional[list] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Find healable spots on one face.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR face ROI.
        skin_mask: (H, W) skin mask, 0-1 or 0-255 (eyes, brows, lips and hair
            already excluded, as ``FaceRegions.skin`` is).
        face_width: face width in pixels of ``img_bgr``.
        strength: 0-100.
        protect_mask: optional (H, W) 0-1 mask of pixels that must not be
            healed (e.g. a mark policy's preserve set).
        feature_mask: optional (H, W) mask of eyes, brows, lips, mouth and
            hair (see :func:`feature_mask_from_regions`); nothing within
            ``_FEATURE_MARGIN`` of it is healed.
        _rejects: tuning aid; when a list is given, every rejected candidate
            is appended as ``(cx, cy, r, reason)`` in full-resolution pixels.

    Returns:
        (spots, diagnostics). Each spot is a dict with ``cx``, ``cy``, ``r``
        (full-resolution pixels), ``kind`` (``"red"`` or ``"dark"``) and the
        component ``mask`` (uint8, full resolution, ``bbox``-local) with its
        ``bbox`` (x, y, w, h).
    """
    s = float(np.clip(strength / 100.0, 0.0, 1.0))
    h, w = img_bgr.shape[:2]
    skin = _effective_skin(_to_unit_mask(skin_mask), feature_mask, face_width)
    diags: Dict[str, Any] = {"candidates": 0, "found": 0, "kept_moles": 0}
    if s <= 0.0 or face_width <= 0 or float(skin.max(initial=0.0)) < 0.5:
        return [], diags

    scale = min(1.0, _WORK_FACE_W / float(face_width))
    fw = face_width * scale
    if scale < 1.0:
        small = cv2.resize(img_bgr.astype(np.float32), None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        skin_s = cv2.resize(skin, (small.shape[1], small.shape[0]),
                            interpolation=cv2.INTER_AREA)
    else:
        small = img_bgr.astype(np.float32)
        skin_s = skin
    lab = bgr_f32_to_lab_f32(np.clip(small, 0.0, 255.0))
    L = lab[:, :, 0]
    A = lab[:, :, 1]

    d_min = _D_MIN * fw
    d_max = (_D_MAX_LO + (_D_MAX_HI - _D_MAX_LO) * s) * fw
    skin_hard = (skin_s > 0.5).astype(np.float32)
    if skin_hard.sum() < 200:
        return [], diags

    # Fill non-skin from nearby skin so the local median never sees eyes,
    # brows, lips or hair.
    # Window sizes use the largest spot size at any strength, so a higher
    # strength only ever adds spots (never shifts the local level).
    d_geo = _D_MAX_HI * fw
    fill_sigma = max(d_geo, 3.0)
    L_fill = np.where(skin_hard > 0, L, _norm_conv(L, skin_hard, fill_sigma))
    A_fill = np.where(skin_hard > 0, A, _norm_conv(A, skin_hard, fill_sigma))

    k = int(round(2.2 * d_geo)) | 1
    k = max(k, 5)

    def _median(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        span = max(hi - lo, 1e-3)
        u8 = np.clip((x - lo) * (255.0 / span), 0, 255).astype(np.uint8)
        return cv2.medianBlur(u8, k).astype(np.float32) * (span / 255.0) + lo

    bg_L = _median(L_fill)
    bg_A = _median(A_fill)

    core = skin_hard.astype(np.uint8)
    er = max(int(round(0.6 * d_geo)), 1)
    core = cv2.erode(core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * er + 1, 2 * er + 1)))
    if core.sum() < 100:
        return [], diags
    sel = core > 0

    # Matched to spot size: the residual is blurred at a few spot-sized
    # scales and each is divided by its own spread over the face, so a faint
    # but wide mark scores as high as a small, sharp one. Pore-scale noise
    # averages out at every scale.
    rawL = bg_L - L_fill       # + = darker
    rawA = A_fill - bg_A       # + = redder
    zL = np.zeros_like(rawL)
    zA = np.zeros_like(rawA)
    dL = dA = None
    for k_s, frac in enumerate(_SCALES):
        sg = max(frac * d_min, 0.6)
        bl = cv2.GaussianBlur(rawL, (0, 0), sg)
        ba = cv2.GaussianBlur(rawA, (0, 0), sg)
        if k_s == 0:
            dL, dA = bl, ba
            diags.update(sigma_L=round(_robust_sigma(bl[sel], _SIGMA_FLOOR_L), 3),
                         sigma_a=round(_robust_sigma(ba[sel], _SIGMA_FLOOR_A), 3))
        zL = np.maximum(zL, np.maximum(bl, 0.0) / _robust_sigma(bl[sel], _SIGMA_FLOOR_L))
        zA = np.maximum(zA, np.maximum(ba, 0.0) / _robust_sigma(ba[sel], _SIGMA_FLOOR_A))
    z = np.sqrt(zL * zL + zA * zA)

    z_hi = _Z_HI_LO + (_Z_HI_HI - _Z_HI_LO) * s
    seeds = ((z > z_hi) & sel).astype(np.uint8)
    allowed = skin_hard > 0
    if protect_mask is not None:
        prot = _to_unit_mask(protect_mask)
        if scale < 1.0:
            prot = cv2.resize(prot, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA)
        allowed &= prot <= 0.3
        seeds[~allowed] = 0
    if not seeds.any():
        return [], diags

    # Candidates are local maxima of the anomaly above the seed level,
    # strongest first; each claims its spot so a spot is healed once.
    nb = max(int(round(d_min)) | 1, 3)
    peaks_mask = (z >= cv2.dilate(z, np.ones((nb, nb), np.uint8))) & (seeds > 0)
    pys, pxs = np.nonzero(peaks_mask)
    order = np.argsort(-z[pys, pxs])
    diags["candidates"] = int(order.size)

    def _reject(cx: float, cy: float, d: float, why: str) -> None:
        if _rejects is not None:
            _rejects.append((cx * inv, cy * inv, 0.5 * d * inv, why))

    spots: List[Dict[str, Any]] = []
    inv = 1.0 / scale
    Hs, Ws = z.shape
    win = int(np.ceil(1.2 * d_geo)) + 2
    claimed = np.zeros_like(seeds)
    for oi in order:
        py, px = int(pys[oi]), int(pxs[oi])
        if claimed[py, px]:
            continue
        peak = float(z[py, px])
        # The spot is the half-peak region around this seed's peak, found in
        # a window just larger than the biggest allowed spot. A region that
        # reaches the window edge is a broad shadow or a feature, not a spot.
        wx0, wy0 = max(px - win, 0), max(py - win, 0)
        wx1, wy1 = min(px + win + 1, Ws), min(py + win + 1, Hs)
        # A spot on a shaded slope joins the shading at half its peak, so
        # tighter levels are tried before the peak is given up on.
        zw = z[wy0:wy1, wx0:wx1]
        aw = allowed[wy0:wy1, wx0:wx1]
        found = False
        for frac in _LEVELS:
            level = max(frac * peak, _Z_GROW * z_hi)
            _, rl = cv2.connectedComponents(((zw > level) & aw).astype(np.uint8), connectivity=8)
            comp_w = rl == rl[py - wy0, px - wx0]
            ys, xs = np.nonzero(comp_w)
            area = int(xs.size)
            d_eq = 2.0 * np.sqrt(area / np.pi)
            touches = (xs.min() == 0 and wx0 > 0) or (ys.min() == 0 and wy0 > 0) or \
                (xs.max() == wx1 - wx0 - 1 and wx1 < Ws) or (ys.max() == wy1 - wy0 - 1 and wy1 < Hs)
            if not touches and d_eq <= d_max:
                found = True
                break
        cx, cy = float(xs.mean() + wx0), float(ys.mean() + wy0)
        claimed[wy0:wy1, wx0:wx1][comp_w] = 1
        if not found or d_eq < d_min:
            _reject(cx, cy, d_eq, "size")
            continue
        x, y = int(xs.min()) + wx0, int(ys.min()) + wy0
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        comp = comp_w[y - wy0:y - wy0 + bh, x - wx0:x - wx0 + bw]
        if area < _MIN_FILL * bw * bh:
            _reject(cx, cy, d_eq, "fill")
            continue
        if xs.size >= 5:
            ev = np.linalg.eigvalsh(np.cov(np.stack([xs.astype(np.float32), ys.astype(np.float32)])))
            if ev[0] <= 1e-6 or np.sqrt(ev[1] / ev[0]) > _MAX_ASPECT:
                _reject(cx, cy, d_eq, "aspect")
                continue
        # A spot must sit inside skin: a blob whose surroundings are mostly
        # non-skin is a feature edge (nostril, lash line, lip corner).
        pad = max(int(round(d_eq)), 2)
        y0, y1 = max(y - pad, 0), min(y + bh + pad, Hs)
        x0, x1 = max(x - pad, 0), min(x + bw + pad, Ws)
        if skin_hard[y0:y1, x0:x1].mean() < 0.92:
            _reject(cx, cy, d_eq, "edge")
            continue
        dl = float(dL[y:y + bh, x:x + bw][comp].max())
        da = float(dA[y:y + bh, x:x + bw][comp].max())
        # Perceptual floor: below about 2.5 delta-E nobody sees the spot, so
        # it isn't worth rewriting skin for. (A perceptual difference, not an
        # intensity level, so it holds on every skin tone.)
        if np.hypot(max(dl, 0.0) / 2.55, max(da, 0.0)) < _MIN_DELTA_E:
            _reject(cx, cy, d_eq, "faint")
            continue
        # A spot stands out from skin on every side. Shading (the side of the
        # nose, a smile line) is as anomalous on one side of the ring around
        # it as at its centre.
        r_eq = 0.5 * d_eq
        ring_max = _ring_sector_max(z, cx, cy, 1.6 * r_eq + 1.0, 2.6 * r_eq + 2.0)
        if ring_max > _RING_FRAC * peak:
            _reject(cx, cy, d_eq, "ring")
            continue
        za = float(zA[y:y + bh, x:x + bw][comp].max())
        # Red (inflamed) when the a* rise is a sizeable share of the L drop.
        # Raw LAB units, not spreads: the a* spread is usually much smaller
        # than the L spread, which would make every brown mole look "red".
        red = za >= _RED_Z and da >= _RED_RATIO * dl
        kind = "red" if red else "dark"
        # Moles and beauty marks: non-red and either wide or deeply dark
        # relative to the skin around them (a fraction, so tone-invariant).
        rel_dark = dl / max(float(bg_L[int(cy), int(cx)]), 1.0)
        if kind == "dark" and (d_eq > _MOLE_D * fw or rel_dark > _MOLE_REL_DARK):
            diags["kept_moles"] += 1
            _reject(cx, cy, d_eq, "mole")
            continue
        # The core found above is the bright middle of the spot; its redness
        # or shadow fades out further. Heal out to where the anomaly drops to
        # a small share of the peak (never past a disk around the core, so a
        # neighbouring shadow can't be pulled in).
        ext_level = max(_EXTENT * peak, 0.35 * _Z_GROW * z_hi)
        _, rl = cv2.connectedComponents(((zw > ext_level) & aw).astype(np.uint8), connectivity=8)
        ext = rl == rl[py - wy0, px - wx0]
        gy, gx = np.mgrid[wy0:wy1, wx0:wx1]
        ext &= np.hypot(gx - cx, gy - cy) <= _EXTENT_R * r_eq + 1.0
        ext |= comp_w
        ys, xs = np.nonzero(ext)
        x, y = int(xs.min()) + wx0, int(ys.min()) + wy0
        bw, bh = int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
        comp = ext[y - wy0:y - wy0 + bh, x - wx0:x - wx0 + bw]
        d_eq = 2.0 * np.sqrt(xs.size / np.pi)
        # Full-resolution footprint of this component.
        fx0, fy0 = int(np.floor(x * inv)), int(np.floor(y * inv))
        fx1 = min(int(np.ceil((x + bw) * inv)), img_bgr.shape[1])
        fy1 = min(int(np.ceil((y + bh) * inv)), img_bgr.shape[0])
        comp_u8 = comp.astype(np.uint8) * 255
        if scale < 1.0:
            comp_u8 = cv2.resize(comp_u8, (max(fx1 - fx0, 1), max(fy1 - fy0, 1)),
                                 interpolation=cv2.INTER_LINEAR)
            comp_u8 = (comp_u8 > 127).astype(np.uint8) * 255
        spots.append({
            "cx": float(cx * inv), "cy": float(cy * inv), "r": float(0.5 * d_eq * inv),
            "kind": kind, "z": round(peak, 2), "contrast": float(np.hypot(max(dl, 0.0), max(da, 0.0))),
            "bbox": (fx0, fy0, comp_u8.shape[1], comp_u8.shape[0]), "mask": comp_u8,
        })
    diags["found"] = len(spots)
    diags["red"] = sum(1 for sp in spots if sp["kind"] == "red")
    diags["dark"] = diags["found"] - diags["red"]
    return spots, diags


def _ring_sector_max(z: np.ndarray, cx: float, cy: float, r0: float, r1: float) -> float:
    """Largest mean anomaly over 8 sectors of the ring r0..r1 around (cx, cy)."""
    H, W = z.shape
    R = int(np.ceil(r1)) + 1
    x0, x1 = max(int(cx) - R, 0), min(int(cx) + R + 1, W)
    y0, y1 = max(int(cy) - R, 0), min(int(cy) + R + 1, H)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx, dy = xx - cx, yy - cy
    dist = np.hypot(dx, dy)
    ring = (dist >= r0) & (dist <= r1)
    if not ring.any():
        return 0.0
    sector = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * 8).astype(np.int32) % 8
    zz = z[y0:y1, x0:x1]
    best = 0.0
    for k in range(8):
        m = ring & (sector == k)
        if m.any():
            best = max(best, float(zz[m].mean()))
    return best


def _spot_union(spots: List[Dict[str, Any]], shape: Tuple[int, int], grow: float) -> np.ndarray:
    """Binary uint8 map of every spot footprint, each dilated by grow*r."""
    out = np.zeros(shape, np.uint8)
    for sp in spots:
        x, y, bw, bh = sp["bbox"]
        out[y:y + bh, x:x + bw] = np.maximum(out[y:y + bh, x:x + bw], sp["mask"][: shape[0] - y, : shape[1] - x])
    if not spots:
        return out
    r = max(int(round(grow * float(np.median([sp["r"] for sp in spots])))), 1)
    return cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))


def heal_spots(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    face_width: float,
    strength: float,
    protect_mask: Optional[np.ndarray] = None,
    feature_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Detect and heal spots on one face ROI.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR.
        skin_mask: (H, W) skin mask (0-1 or 0-255).
        face_width: face width in pixels of ``img_bgr``.
        strength: 0-100; 0 returns the input unchanged.
        protect_mask: optional 0-1 mask of pixels to leave alone.
        feature_mask: optional mask of eyes, brows, lips, mouth and hair.

    Returns:
        (image, diagnostics). The image has the input's dtype; pixels outside
        the healed footprints are bit-identical to the input.
    """
    if strength <= 0:
        return img_bgr, {"healed": 0}
    spots, diags = detect_spots(img_bgr, skin_mask, face_width, strength,
                                protect_mask, feature_mask)
    diags.update(healed=0, skipped=0)
    if not spots:
        return img_bgr, diags

    src = img_bgr.astype(np.float32)
    out = src.copy()
    H, W = src.shape[:2]
    skin = _effective_skin(_to_unit_mask(skin_mask), feature_mask, face_width)
    skin_hard = (skin > 0.5).astype(np.uint8)
    # Every spot (grown) is excluded from both the local level and donors.
    avoid = _spot_union(spots, (H, W), 0.8)
    tex_sigma = max(_TEXTURE_SIGMA * face_width, 0.8)

    for sp in spots:
        r = max(sp["r"], 1.0)
        cx, cy = sp["cx"], sp["cy"]
        pad_px = _PAD * r + 1.0
        R = int(np.ceil(4.5 * r + 3 * tex_sigma + 4))
        x0, y0 = max(int(cx) - R, 0), max(int(cy) - R, 0)
        x1, y1 = min(int(cx) + R + 1, W), min(int(cy) + R + 1, H)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        patch = src[y0:y1, x0:x1]
        ph, pw = patch.shape[:2]

        # Footprint: the component, grown by pad_px, feathered.
        foot = np.zeros((ph, pw), np.uint8)
        bx, by, bw, bh = sp["bbox"]
        fx0, fy0 = max(bx - x0, 0), max(by - y0, 0)
        fx1, fy1 = min(bx + bw - x0, pw), min(by + bh - y0, ph)
        if fx1 <= fx0 or fy1 <= fy0:
            continue
        foot[fy0:fy1, fx0:fx1] = sp["mask"][fy0 - (by - y0): fy1 - (by - y0), fx0 - (bx - x0): fx1 - (bx - x0)]
        g = max(int(round(pad_px)), 1)
        foot = cv2.dilate(foot, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1)))
        alpha = cv2.GaussianBlur(foot.astype(np.float32) / 255.0, (0, 0), max(0.35 * pad_px, 0.7))
        alpha *= skin[y0:y1, x0:x1]
        if float(alpha.max()) <= 0.01:
            diags["skipped"] += 1
            continue

        # Local skin level with every spot ignored: a plane fitted to the
        # clean skin in a ring around the footprint, so a spot on a lit slope
        # heals to the slope, not to a flat average.
        clean = (skin_hard[y0:y1, x0:x1] > 0) & (avoid[y0:y1, x0:x1] == 0)
        dist = cv2.distanceTransform((foot == 0).astype(np.uint8), cv2.DIST_L2, 5)
        ring = clean & (dist > 1.0) & (dist <= max(1.2 * r, 3.0))
        if int(ring.sum()) < 12:
            diags["skipped"] += 1
            continue
        ry, rx = np.nonzero(ring)
        A = np.stack([np.ones_like(rx), rx, ry], axis=1).astype(np.float32)
        smooth = cv2.GaussianBlur(patch, (0, 0), tex_sigma)
        coef, *_ = np.linalg.lstsq(A, smooth[ry, rx], rcond=None)
        # Skip spots on a step (paint border, nostril rim): a plane can't
        # follow those, and the heal would smear across them.
        resid = smooth[ry, rx] - A @ coef
        rr = float(np.sqrt((resid ** 2).mean(axis=0)).max())
        if rr > _RING_RESID * max(float(sp["contrast"]), 1.0):
            diags["skipped"] += 1
            continue
        gy, gx = np.mgrid[0:ph, 0:pw]
        level = (coef[0][None, None, :] + gx[:, :, None] * coef[1][None, None, :]
                 + gy[:, :, None] * coef[2][None, None, :]).astype(np.float32)

        # Fine texture from the cleanest donor offset.
        base = smooth
        tex = patch - base
        best, best_cost = None, np.inf
        fy, fx = np.nonzero(foot)
        for hop in (2.4, 3.2):
            for ang in np.arange(8) * (np.pi / 4.0):
                dx = int(round(np.cos(ang) * hop * r))
                dy = int(round(np.sin(ang) * hop * r))
                ty, tx = fy + dy, fx + dx
                if ty.min() < 0 or tx.min() < 0 or ty.max() >= ph or tx.max() >= pw:
                    continue
                ok = clean[ty, tx]
                if ok.mean() < 0.98:
                    continue
                cost = float(np.abs(base[ty, tx] - level[fy, fx]).mean())
                if cost < best_cost:
                    best, best_cost = (dx, dy), cost
        donor_tex = np.zeros_like(patch)
        if best is not None:
            M = np.float32([[1, 0, best[0]], [0, 1, best[1]]])
            donor_tex = cv2.warpAffine(tex, M, (pw, ph),
                                       flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
                                       borderMode=cv2.BORDER_REFLECT)
        healed = np.clip(level + donor_tex, 0.0, 255.0)
        a3 = alpha[:, :, None]
        region = out[y0:y1, x0:x1]
        out[y0:y1, x0:x1] = region * (1.0 - a3) + healed * a3
        diags["healed"] += 1

    if img_bgr.dtype == np.uint8:
        return np.round(out).astype(np.uint8), diags
    return out.astype(img_bgr.dtype), diags
