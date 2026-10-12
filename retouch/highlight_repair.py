"""Blown highlight repair on skin (opt-in ``highlight_repair``).

Flash or harsh sun can push the hottest spots of a face (forehead, nose
bridge and tip, cheekbones, chin) past the sensor's ceiling. Those pixels
are pure white, or have one or two channels pinned at 255: they carry no
skin colour and no texture, so shine removal, Face Polish and Powder Finish
can only turn them into a flat grey or beige patch.

This op rebuilds them from the skin around each spot:

* **Where:** pixels whose brightest channel sits at the file's ceiling
  (the clip level is a property of the 8-bit encoding, not of the skin, so
  it is the same on every skin tone), inside the face's skin, away from the
  eyes, brows, lips, mouth and hair. Each blown spot must be ringed mostly
  by clean skin (so a white prop, costume part or background showing
  through is left alone). A painted face is left alone entirely: its
  skin's chroma / lightness is too low, or its hue is outside the band that
  natural skin of every tone sits in (white paint under cool light reads
  blue-magenta); so is a face where most of the skin is blown (nothing
  left to rebuild from).
* **Tone:** the excess over a local diffuse skin baseline (masked blur of
  the clean skin at 0.2 x inter-eye distance) is rolled off by a smooth,
  monotone curve measured against the headroom to white, so the spot and
  its bright shoulder settle below white with no halo and still read as a
  highlight. The blown core gets a soft dome instead of a flat plateau.
* **Colour:** a*/b* inside the blown core are filled in from the
  unclipped skin right around it (so it joins its own shoulder without an
  outline), blending toward the local diffuse skin colour in the middle of
  the spot, a little less saturated the brighter the rebuilt pixel is (a
  highlight is diffuse colour plus white light). The washed-out shoulder
  just under the ceiling is pulled the same way, so no pale ring is left.
* **Texture:** fine skin texture is copied into the core from a clean
  patch of the same face next to the spot (the healing-brush idea), or,
  when no clean patch fits, grain matched to the face's own texture spread.

Every threshold except the encoding ceiling is relative to the face's own
skin, so lighter and darker skin are treated alike (see
``tests/test_highlight_repair.py``). Pixels outside the repaired spots are
returned bit for bit.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .utils import bgr_f32_to_lab_f32, lab_f32_to_bgr_f32, normalize_mask

# A channel at or above _CLIP_HI (0-255) counts as fully blown; the weight
# ramps in from _CLIP_LO so JPEG ringing around a blown spot is included.
_CLIP_LO = 244.0
_CLIP_HI = 252.0
# Blown spots smaller than this (pixels) are left: specks, not patches.
_MIN_AREA = 12
# Local diffuse baseline scale, fraction of the inter-eye distance (IED).
_BASE_SIGMA_IED = 0.20
# Fine texture scale, fraction of IED.
_FINE_SIGMA_IED = 0.02
# Pixels around a blown spot kept out of the baseline (its bright halo).
_HALO_IED = 0.03
# Zone around each blown spot where the tone roll-off acts.
_ZONE_IED = 0.12
# Ring used to judge what surrounds a spot.
_RING_IED = 0.06
# A spot is repaired only if at least this share of its ring is skin.
_MIN_RING_SKIN = 0.60
# White face paint: skin chroma / lightness (CIELab) below this.
_PAINT_CL = 0.06
# A face whose skin hue is outside this band (CIELab degrees) is painted:
# natural skin, pink to olive, lighter or darker, sits inside it; white face
# paint under cool light reads blue-magenta.
_FACE_HUE = (-5.0, 80.0)
# Skip a face whose skin is mostly blown: nothing to rebuild from.
_MAX_BLOWN_SHARE = 0.35
# Tone curve on x = excess / headroom-to-white: identity below the knee
# (_KNEE, lower for a low target), and white (x = 1) lands at most _T_MIN
# of the headroom and at most _MAX_GAIN x the skin's diffuse luminance, at
# strength 100.
_KNEE = 0.30
_T_MIN = 0.55
_MAX_GAIN = 3.0
# Soft dome inside the blown core, share of the remaining headroom.
_DOME = 0.25
# Share of the diffuse skin colour at the middle of a blown spot.
_CENTRE_SKIN = 0.8
# Highlight desaturation: chroma kept falls by this share at x = 1.
_DESAT = 0.25
# Margin around eyes/brows/lips/mouth/hair, fraction of IED.
_FEATURE_MARGIN_IED = 0.05


def _unit(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = normalize_mask(mask)
    if m is None:
        return None
    return np.clip(m.astype(np.float32), 0.0, 1.0)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur; wide ones run on a downsampled copy (same result to
    within a fraction of a level, orders of magnitude faster)."""
    if sigma <= 6.0:
        return cv2.GaussianBlur(x, (0, 0), sigma)
    h, w = x.shape[:2]
    f = sigma / 3.0
    sw, sh = max(1, int(round(w / f))), max(1, int(round(h / f)))
    small = cv2.resize(x, (sw, sh), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma * sw / w)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _masked_blur(x: np.ndarray, w: np.ndarray, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
    num = _blur(x * w, sigma)
    den = _blur(w, sigma)
    return num / np.maximum(den, 1e-6), den


def _baseline(x: np.ndarray, w: np.ndarray, sigma: float, fallback: float) -> np.ndarray:
    """Masked blur that widens where a hole is bigger than ``sigma``."""
    out, den = _masked_blur(x, w, sigma)
    wide, den_w = _masked_blur(x, w, sigma * 3.0)
    out = np.where(den > 0.05, out, np.where(den_w > 0.01, wide, fallback))
    return out.astype(np.float32)


def _ellipse(r: int) -> np.ndarray:
    r = max(int(r), 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def _tone_curve(x: np.ndarray, t) -> np.ndarray:
    """Monotone roll-off: identity to the knee, C1 there, 1 -> t, concave above.

    ``t`` is a scalar or an array broadcastable to ``x`` (per-pixel target).
    """
    t = np.minimum(np.asarray(t, dtype=np.float32), 1.0)
    k = np.minimum(_KNEE, 0.6 * t)
    p = (1.0 - k) / np.maximum(t - k, 1e-3)
    u = np.clip((x - k) / (1.0 - k), 0.0, 1.0)
    y = k + (t - k) * (1.0 - (1.0 - u) ** p)
    y = np.where(x <= k, x, y)
    return np.where(t >= 1.0, x, y).astype(np.float32)


def _lstar_to_y(lstar: np.ndarray) -> np.ndarray:
    f = (lstar + 16.0) / 116.0
    return np.where(lstar > 8.0, f ** 3, lstar / 903.3)


def _y_to_lstar(y: np.ndarray) -> np.ndarray:
    return np.where(y > 0.008856, 116.0 * np.cbrt(y) - 16.0, 903.3 * y)


def repair_blown_highlights(
    canvas: np.ndarray,
    skin_mask: Optional[np.ndarray],
    strength: float,
    ied: float,
    feature_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Rebuild blown (clipped) highlights on one face's skin.

    Args:
        canvas: (H, W, 3) float32 BGR [0, 255] face ROI.
        skin_mask: (H, W) skin mask, 0-1 or 0-255. ``None`` returns the
            input unchanged.
        strength: 0-100. 0 returns the input unchanged.
        ied: inter-eye distance in ROI pixels (sets every spatial scale).
        feature_mask: optional (H, W) union of eye, brow, lip, mouth and
            hair masks; these and a margin around them are left alone.

    Returns:
        (image, diagnostics). The image is float32 BGR [0, 255]; pixels
        outside the repaired zones equal the input.
    """
    diags: Dict[str, Any] = {"spots": 0, "repaired": 0, "skipped_ring": 0,
                             "painted": False, "blown_share": 0.0}
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    sk = _unit(skin_mask)
    if s <= 0.0 or sk is None or sk.shape != canvas.shape[:2]:
        return canvas, diags
    work = canvas.astype(np.float32, copy=False)
    H, W = work.shape[:2]
    ied = float(ied) if ied and ied > 0 else float(max(H, W)) * 0.3

    mx = work.max(axis=2)
    wc = np.clip((mx - _CLIP_LO) / (_CLIP_HI - _CLIP_LO), 0.0, 1.0)
    if not (wc >= 1.0).any():
        return canvas, diags

    # Skin support: the segmenter often drops blown pixels from the skin
    # mask, so close small gaps and fill holes enclosed by skin.
    skin_core = (sk > 0.3).astype(np.uint8)
    if int(skin_core.sum()) < 200:
        return canvas, diags
    close_r = max(1, int(round(0.08 * ied)))
    support = cv2.morphologyEx(skin_core, cv2.MORPH_CLOSE, _ellipse(close_r))
    flood = support.copy()
    ff = np.zeros((H + 2, W + 2), np.uint8)
    for sx, sy in ((0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)):
        if flood[sy, sx] == 0:
            cv2.floodFill(flood, ff, (sx, sy), 2)
    support = np.where(flood == 0, 1, support).astype(np.uint8)

    feat = None
    fm = _unit(feature_mask)
    if fm is not None and fm.shape == sk.shape and fm.max() > 0:
        g = max(1, int(round(_FEATURE_MARGIN_IED * ied)))
        feat = cv2.dilate((fm > 0.3).astype(np.uint8), _ellipse(g))
        support = np.where(feat > 0, 0, support).astype(np.uint8)

    blown = ((wc >= 1.0) & (support > 0)).astype(np.uint8)
    blown_share = float(blown.sum()) / float(max(int(support.sum()), 1))
    diags["blown_share"] = round(blown_share, 4)
    if blown_share > _MAX_BLOWN_SHARE or not blown.any():
        return canvas, diags

    lab = bgr_f32_to_lab_f32(work)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # Clean skin: confident skin, not near any blown pixel, not a feature.
    halo = cv2.dilate((wc > 0).astype(np.uint8), _ellipse(max(1, int(round(_HALO_IED * ied)))))
    clean = (sk > 0.5) & (halo == 0)
    if feat is not None:
        clean &= feat == 0
    if int(clean.sum()) < 100:
        return canvas, diags

    # Is the face painted? Median chroma / lightness (CIELab) per pixel over
    # the darker half of its clean skin (the darker half keeps the light's
    # own white on lit skin, which is a wide shoulder on darker skin, out).
    fL = L[clean] * (100.0 / 255.0)
    fC = np.hypot(a[clean] - 128.0, b[clean] - 128.0)
    flow = fL <= np.median(fL)
    face_cl = float(np.median(fC[flow] / np.maximum(fL[flow], 1.0)))
    face_hue = float(np.degrees(np.arctan2(np.median(b[clean][flow] - 128.0),
                                           np.median(a[clean][flow] - 128.0))))
    painted = face_cl < _PAINT_CL or not (_FACE_HUE[0] <= face_hue <= _FACE_HUE[1])
    diags["face_cl"] = round(face_cl, 3)
    diags["face_hue"] = round(face_hue, 1)
    if painted:
        # Face paint: a blown spot on it has no skin colour to bring back,
        # and blush or eyeshadow next to it would be smeared in. Left alone,
        # unpainted parts (a bare nose) included.
        diags["painted"] = True
        return canvas, diags

    # Local diffuse skin baseline from clean skin. Second pass: pixels that
    # stand out above the first pass (the specular shoulder of a hot spot,
    # which on darker skin is mostly the light's white) stop voting, gated
    # in units of the face's own fine texture spread.
    fine = max(0.8, _FINE_SIGMA_IED * ied)
    tex = L - cv2.GaussianBlur(L, (0, 0), fine)
    tex_clean = tex[clean]
    spread = max(0.75, 1.4826 * float(np.median(np.abs(tex_clean - np.median(tex_clean)))))
    cw = clean.astype(np.float32)
    sigma = max(2.0, _BASE_SIGMA_IED * ied)
    L_b1 = _baseline(L, cw, sigma, float(np.median(L[clean])))
    cw = cw * (1.0 - np.clip((L - L_b1 - 1.5 * spread) / (2.0 * spread), 0.0, 1.0))
    L_b = _baseline(L, cw, sigma, float(np.median(L[clean])))
    a_b = _baseline(a, cw, sigma, float(np.median(a[clean])))
    b_b = _baseline(b, cw, sigma, float(np.median(b[clean])))

    # Accept or reject each blown spot on what surrounds it: the ring is
    # taken around the whole bright region the spot sits in (every pixel
    # near the ceiling connected to it), so a wide shoulder is not mistaken
    # for the surroundings.
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(blown, connectivity=8)
    near = ((wc > 0) & (support > 0)).astype(np.uint8)
    _, near_lbl = cv2.connectedComponents(near, connectivity=8)
    keep = np.zeros((H, W), np.uint8)
    ring_r = max(2, int(round(_RING_IED * ied)))
    halo_r = max(1, int(round(_HALO_IED * ied)))
    lab_ok = np.isfinite(L)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < _MIN_AREA:
            continue
        diags["spots"] += 1
        ys, xs = np.nonzero(lbl == i)
        region = near_lbl == near_lbl[ys[0], xs[0]]
        ry, rx = np.nonzero(region)
        pad = halo_r + ring_r + 2
        xa, ya = max(int(rx.min()) - pad, 0), max(int(ry.min()) - pad, 0)
        xb, yb = min(int(rx.max()) + pad + 1, W), min(int(ry.max()) + pad + 1, H)
        comp = (lbl[ya:yb, xa:xb] == i).astype(np.uint8)
        reg = region[ya:yb, xa:xb].astype(np.uint8)
        inner = cv2.dilate(reg, _ellipse(halo_r))
        outer = cv2.dilate(reg, _ellipse(halo_r + ring_r))
        ring = (outer > 0) & (inner == 0)
        if int(ring.sum()) < 8:
            diags["skipped_ring"] += 1
            continue
        ring_clean = ring & clean[ya:yb, xa:xb] & lab_ok[ya:yb, xa:xb]
        if float(ring_clean.sum()) / float(ring.sum()) < _MIN_RING_SKIN:
            diags["skipped_ring"] += 1
            continue
        keep[ya:yb, xa:xb] |= comp
        diags["repaired"] += 1
    if not keep.any():
        return canvas, diags

    # Zone: the kept spots grown over their bright shoulder, inside skin.
    zone_r = max(2, int(round(_ZONE_IED * ied)))
    zone = cv2.dilate(keep, _ellipse(zone_r)).astype(np.float32)
    zone = _blur(zone, max(1.0, zone_r * 0.35))
    zone = np.clip(zone * 1.5, 0.0, 1.0) * np.clip(support.astype(np.float32) + sk, 0.0, 1.0)
    zone = zone.astype(np.float32)

    # Core weight: blown pixels of kept spots plus their JPEG fringe.
    fringe = cv2.dilate(keep, _ellipse(max(1, halo_r // 2)))
    core = np.maximum(keep.astype(np.float32), wc * (fringe > 0)).astype(np.float32)
    core = cv2.GaussianBlur(core, (0, 0), max(0.7, 0.01 * ied))
    core = np.clip(core, 0.0, 1.0) * zone

    # Tone: roll the excess off against the headroom to white.
    # White lands at most _MAX_GAIN x the diffuse skin's own luminance (a
    # strong highlight relative to that skin, the same on every tone) and
    # never above _T_MIN of the headroom.
    head = np.maximum(255.0 - L_b, 1.0)
    x = np.clip((L - L_b) / head, 0.0, 1.0)
    y_cap = np.minimum(_lstar_to_y(L_b * (100.0 / 255.0)) * _MAX_GAIN, 1.0)
    t_cap = (_y_to_lstar(y_cap) * (255.0 / 100.0) - L_b) / head
    t = (1.0 - s * (1.0 - np.clip(t_cap, 0.05, _T_MIN))).astype(np.float32)
    y = _tone_curve(x, t)
    # Soft dome over each blown core so it is not a flat plateau.
    # Each spot's dome is scaled to its own size (its deepest pixel).
    rad = cv2.distanceTransform(keep, cv2.DIST_L2, 5)
    kn, klbl = cv2.connectedComponents(keep, connectivity=8)
    rmax = np.zeros(kn, np.float32)
    np.maximum.at(rmax, klbl.ravel(), rad.ravel())
    rnorm = rad / (0.5 * rmax[klbl] + 1.0)
    dome = cv2.GaussianBlur(np.clip(rnorm, 0.0, 1.0).astype(np.float32), (0, 0), max(0.7, 0.01 * ied))
    y = y + core * dome * _DOME * (1.0 - t)
    L_tone = L_b + y * head
    L_new = np.where(L > L_b, L + (L_tone - L) * zone, L)

    # Colour: filled in from the unclipped skin right around the blown core
    # (smallest scale that reaches it), so it joins the highlight's own
    # shoulder without an outline; far from any real colour it falls back
    # to the local skin baseline. A little desaturation with height.
    col_w = ((wc <= 0.0) & (support > 0)).astype(np.float32)
    a_f = a_b.copy()
    b_f = b_b.copy()
    filled = np.zeros((H, W), bool)
    for frac in (0.03, 0.06, 0.12, 0.25):
        sg = max(1.0, frac * ied)
        a_s, den = _masked_blur(a, col_w, sg)
        b_s, _ = _masked_blur(b, col_w, sg)
        ok = (den > 0.05) & ~filled
        a_f = np.where(ok, a_s, a_f)
        b_f = np.where(ok, b_s, b_f)
        filled |= ok
    # Toward the middle of the spot, blend in the diffuse skin colour: on
    # darker skin the shoulder is mostly the light's white, and a core
    # filled from it alone would read grey.
    wash = zone * np.clip((x - 0.6) / 0.35, 0.0, 1.0)
    beta = _CENTRE_SKIN * np.maximum(np.maximum(dome, 0.5 * core), 0.5 * wash)
    a_f = a_f + (a_b - a_f) * beta
    b_f = b_f + (b_b - b_f) * beta
    keep_c = 1.0 - _DESAT * np.clip(y - _KNEE, 0.0, 1.0)
    a_t = 128.0 + (a_f - 128.0) * keep_c
    b_t = 128.0 + (b_f - 128.0) * keep_c
    # The washed-out shoulder just outside the blown core (bright but under
    # the ceiling) is pulled toward the same colour, by how close to white
    # it is, so no pale ring is left around the rebuilt core.
    col = np.maximum(core, wash * s)
    a_new = a + (a_t - a) * col
    b_new = b + (b_t - b) * col

    # Texture: the face's own fine texture, copied from a clean patch.
    donor = np.zeros_like(L)
    placed = np.zeros((H, W), bool)
    kn, klbl, kstats, _ = cv2.connectedComponentsWithStats(keep, connectivity=8)
    for i in range(1, kn):
        x0, y0, bw, bh = kstats[i, :4]
        g = max(int(round(fine * 2)), 2)
        xa, ya = max(x0 - g, 0), max(y0 - g, 0)
        xb, yb = min(x0 + bw + g, W), min(y0 + bh + g, H)
        foot = cv2.dilate((klbl[ya:yb, xa:xb] == i).astype(np.uint8), _ellipse(g)) > 0
        fy, fx = np.nonzero(foot)
        best, best_cost = None, np.inf
        size = max(bw, bh)
        for hop in (1.3, 1.8, 2.5):
            for ang in np.arange(8) * (np.pi / 4.0):
                dx = int(round(np.cos(ang) * hop * size))
                dy = int(round(np.sin(ang) * hop * size))
                ty, tx = fy + ya + dy, fx + xa + dx
                if ty.min() < 0 or tx.min() < 0 or ty.max() >= H or tx.max() >= W:
                    continue
                if clean[ty, tx].mean() < 0.95:
                    continue
                cost = float(np.abs(L_b[ty, tx] - L_b[fy + ya, fx + xa]).mean())
                if cost < best_cost:
                    best, best_cost = (dx, dy), cost
        if best is not None:
            donor[fy + ya, fx + xa] = tex[fy + ya + best[1], fx + xa + best[0]]
            placed[fy + ya, fx + xa] = True
    # No clean patch nearby: grain at the face's own texture scale and spread.
    missing = (keep > 0) & ~placed
    if missing.any() and spread > 0:
        rng = np.random.default_rng(1234)
        grain = cv2.GaussianBlur(rng.standard_normal((H, W)).astype(np.float32), (0, 0), fine * 0.6)
        gs = float(grain.std()) or 1.0
        grain *= spread / gs
        miss = cv2.dilate(missing.astype(np.uint8), _ellipse(max(int(round(fine * 2)), 2))) > 0
        donor = np.where(miss & ~placed, grain, donor)
    # Highlights show less texture than diffuse skin; keep most of it.
    # Only fine grain, not a neighbour's edges or lines.
    donor = np.clip(donor, -2.5 * spread, 2.5 * spread)
    L_new = L_new + donor * core * (0.6 + 0.4 * (1.0 - np.clip(y, 0, 1))) * s

    out_lab = np.stack([np.clip(L_new, 0.0, 255.0), a_new, b_new], axis=-1).astype(np.float32)
    out = lab_f32_to_bgr_f32(out_lab)
    # Skin colour at highlight lightness can push one channel (red, mostly)
    # back to the ceiling; scale such pixels just under it, hue kept.
    peak = out.max(axis=2)
    lim = _CLIP_LO - 2.0
    scale = np.where(peak > lim, lim / np.maximum(peak, 1e-3), 1.0)
    scale = 1.0 + (scale - 1.0) * np.clip(core * 4.0, 0.0, 1.0)
    out = out * scale[..., None].astype(np.float32)
    # Every weight is already folded in above; outside the zones the input
    # is returned as is (no Lab round-trip error).
    res = np.where((np.maximum(zone, core) > 0)[..., None], out, work)
    return np.clip(res, 0.0, 255.0).astype(np.float32), diags
