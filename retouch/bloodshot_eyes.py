"""Bloodshot eye whites: calm red, veiny sclera (classical, no model weights).

Coloured contacts, late nights and a long con day leave the whites of the
eyes pink or streaked with vessels. This takes the red back out of the
visible white and lifts it a touch, and leaves everything that is not white
of the eye alone: the iris (and a red contact lens that is wider than the
iris landmarks), the lash line and waterline, the lid skin, catchlights and
the eyeball's own shading.

This replaces the first version of ``eye_sclera_vessel_remove``. That one
flagged pixels redder than the sclera mask's own median and Telea-inpainted
them from uint8. The landmark sclera mask on real faces also holds the
waterline, the lash line and the rim of a big contact lens, so on Alex's
red-contact bunny photos it did nothing at 20 and at 60 painted up to 27
delta-E patches over the contact rim and the waterline ("demon eyes" in the
2026-07-12 auto_clean_v1 QA), while removing about 40% of planted redness.

Per eye, on a crop around the landmark sclera mask:

1. **Where the white is.** The sclera mask, away from its rim (lid margin,
   waterline and lashes sit on the rim; ramp over 3-8% of the eye's width),
   outside the *visible* iris, and only pixels near the eye's own white
   level (``L / L_white`` ramped over 0.62-0.80, where ``L_white`` is the
   90th percentile over the interior). A ratio to the eye's own white, so a
   darker-lit or darker-skinned face gets the same support; there is no
   absolute brightness gate. The visible iris radius comes from the image,
   not the landmarks: the first ring (1.2-2.0 landmark radii; circle lenses
   run 1.2-1.7 on the bunny photos) with an angular sector whose median
   brightness is 70% of the way from the iris to the white, so a circle lens
   wider than the iris stays out.
2. **How red it is.** A local baseline of CIELab a* over that support
   (masked Gaussian, sigma 6% of the eye's width; a second pass leaves out
   pixels that already read as vessels) and its own noise (MAD). Vessels are
   pixels above the baseline by 1-3 of those noise units. A natural white is
   the lower of the white's own 20th-percentile a* and a cap that follows
   the light's colour as the white's b* shows it: blood is red (a+, b* about
   level), a warm light is yellow (b+) and a cool or magenta light is a+ with
   b- (the bunny photos' clean whites read a* +5..+9 at b* -8 under their
   softboxes, and must stay as they are).
3. **Fix.** a* goes toward that natural white (a vessel's and a diffuse
   pink's excess alike), a vessel's b* and darkening go back to the local
   baseline, and the white gets a gentle uniform brightness gain (at most
   3.5% at 100, never past the white's own 99th percentile). Brightness is
   only ever scaled, never flattened, so the eyeball keeps its shading.

Strength 0 returns the input unchanged. Closed or hidden eyes are skipped by
the shared eye-visibility gate when landmarks are available, and an eye with
too little visible white is skipped here.

Known limits: under strongly red or pink stage light the white's red is the
light, not the eye, and gets calmed too (keep the slider low for those); a
thick vessel wider than about 6% of the eye reads as baseline and is only
partly removed.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from .utils import bgr_f32_to_lab_f32, lab_f32_to_bgr_f32, normalize_mask

__all__ = ["calm_bloodshot_eyes", "calm_bloodshot_eye"]

_MIN_IRIS_R = 2.5          # px; smaller irises cannot anchor the exclusion
_MIN_WHITE_PX = 30         # visible white pixels needed to act on an eye
_RIM_LO, _RIM_HI = 0.03, 0.08   # rim ramp, fraction of eye width
_WHITE_LO, _WHITE_HI = 0.62, 0.80  # L / L_white ramp
_IRIS_RING_LO, _IRIS_RING_HI = 1.2, 2.0   # search range, x landmark iris r
_SECTORS = 8               # angular sectors per ring
_IRIS_EDGE_FRAC = 0.70     # ring level this far from iris to white = edge
_IRIS_FEATHER = 0.12       # exclusion feather, x landmark iris r
_BASE_SIGMA = 0.06         # a* baseline sigma, fraction of eye width
_VESSEL_LO, _VESSEL_HI = 1.0, 3.0  # vessel ramp in noise units
_VESSEL_LIFT_MAX = 0.12    # vessel darkening restored, at most x L_white
_SPECULAR = 1.08           # above this x L_white a pixel is a catchlight
_NOISE_FLOOR = 0.8         # a* noise floor (CIELab units)
_WARM_CAP = 3.0            # natural white a* cap at b* = 0
_WARM_SLOPE = 0.25         # cap rises with the white's own b* (warm light)
_COOL_SLOPE = 0.60         # ... and with -b* (cool/magenta light is a+ b-)
_DIFFUSE_FRAC = 1.0        # share of a diffuse pink removed at 100
_LIFT = 0.035              # uniform brightness gain at 100


def _smoothstep(x: np.ndarray, e0: float, e1: float) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _masked_blur(val: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    num = cv2.GaussianBlur(val * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-4)


def _visible_iris_radius(
    L: np.ndarray,
    dist_c: np.ndarray,
    angle: np.ndarray,
    iris_in: np.ndarray,
    interior: np.ndarray,
    r: float,
    l_white: float,
) -> float:
    """Radius where the eye turns from iris (or contact lens) to white.

    Each ring is split into angular sectors and the brightest sector's median
    is used: lids and the shadowed corner cover most of a ring, and a look to
    one side leaves white on one side only.
    """
    vals = L[iris_in]
    if vals.size == 0:
        return _IRIS_RING_LO * r
    l_iris = float(np.median(vals))
    edge = l_iris + _IRIS_EDGE_FRAC * (l_white - l_iris)
    sector = ((angle + np.pi) * (_SECTORS / (2.0 * np.pi))).astype(np.int32) % _SECTORS
    step = 0.05 * r
    for rr in np.arange(_IRIS_RING_LO * r, _IRIS_RING_HI * r, step):
        ring = interior & (dist_c >= rr) & (dist_c < rr + max(step, 1.0))
        if ring.sum() < 4:
            continue
        sec = sector[ring]
        lv = L[ring]
        for q in range(_SECTORS):
            sel = lv[sec == q]
            if sel.size >= 4 and float(np.median(sel)) >= edge:
                return float(rr)
    return _IRIS_RING_HI * r


def calm_bloodshot_eye(
    img_bgr: np.ndarray,
    sclera: Optional[np.ndarray],
    iris: Optional[np.ndarray],
    strength: float,
) -> np.ndarray:
    """Calm redness in one eye's visible white.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 BGR in [0, 255].
        sclera: (H, W) sclera mask (eye opening minus iris), 0-1 or 0-255.
        iris: (H, W) landmark iris mask for the same eye.
        strength: 0-100.

    Returns:
        Image of the same dtype; pixels outside the sclera mask are unchanged.
    """
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    sclera = normalize_mask(sclera)
    iris = normalize_mask(iris)
    if s <= 0.0 or sclera is None or iris is None or sclera.max() < 0.1:
        return img_bgr
    iris_in_full = iris > 0.5
    n_iris = int(iris_in_full.sum())
    r = float(np.sqrt(n_iris / np.pi)) if n_iris else 0.0
    if r < _MIN_IRIS_R:
        return img_bgr

    eye_full = (sclera > 0.1) | iris_in_full
    ys, xs = np.where(eye_full)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    eye_w = float(max(x1 - x0, y1 - y0))
    pad = int(np.ceil(0.1 * eye_w)) + 2
    h, w = img_bgr.shape[:2]
    y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0)
    y1, x1 = min(y1 + pad, h), min(x1 + pad, w)

    crop = img_bgr[y0:y1, x0:x1].astype(np.float32)
    sc = sclera[y0:y1, x0:x1]
    iris_in = iris_in_full[y0:y1, x0:x1]
    eye = eye_full[y0:y1, x0:x1].astype(np.uint8)

    lab = bgr_f32_to_lab_f32(crop)
    L, A, B = lab[..., 0], lab[..., 1] - 128.0, lab[..., 2] - 128.0

    # 1. Support: interior of the eye opening, outside the visible iris,
    #    near this eye's own white level.
    dist_rim = cv2.distanceTransform(eye, cv2.DIST_L2, 3)
    rim = _smoothstep(dist_rim, _RIM_LO * eye_w, _RIM_HI * eye_w)
    interior = (rim > 0.5) & (sc > 0.5)
    if interior.sum() < _MIN_WHITE_PX:
        return img_bgr
    l_white = float(np.percentile(L[interior], 90.0))
    if l_white <= 1.0:
        return img_bgr

    cy, cx = np.argwhere(iris_in).mean(axis=0)
    yy, xx = np.mgrid[0:L.shape[0], 0:L.shape[1]].astype(np.float32)
    dist_c = np.hypot(yy - cy, xx - cx)
    angle = np.arctan2(yy - cy, xx - cx)
    r_vis = _visible_iris_radius(L, dist_c, angle, iris_in, interior, r, l_white)
    iris_keep = _smoothstep(dist_c, r_vis + 0.05 * r, r_vis + (0.05 + _IRIS_FEATHER) * r)

    # Soften the geometric part only: the brightness gate stays per pixel so
    # the blur cannot carry the edit onto a dark lash, contact rim or lid.
    soft = max(0.012 * eye_w, 0.6)
    geom = cv2.GaussianBlur(sc * rim * iris_keep, (0, 0), soft) * (sc > 0.05)
    whiteness = _smoothstep(L / l_white, _WHITE_LO, _WHITE_HI)
    support = geom * whiteness
    core = support > 0.5
    if int(core.sum()) < _MIN_WHITE_PX:
        return img_bgr

    # 2. Redness: local baseline, its noise, and a natural-white target.
    sigma = max(_BASE_SIGMA * eye_w, 1.5)
    wsup = support.astype(np.float32)
    base_a = _masked_blur(A, wsup, sigma)
    resid = (A - base_a)[core]
    noise = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))),
                _NOISE_FLOOR)
    # Catchlights stay out of the baseline: next to one, a reddish contact
    # rim would otherwise read as a dark vessel to be lifted.
    clean = wsup * (A - base_a < _VESSEL_LO * noise) * (L <= _SPECULAR * l_white)
    base_a = _masked_blur(A, clean, sigma)
    base_b = _masked_blur(B, clean, sigma)
    base_l = _masked_blur(L, clean, sigma)
    resid = (A - base_a)[core]
    noise = max(1.4826 * float(np.median(np.abs(resid - np.median(resid)))),
                _NOISE_FLOOR)
    vessel = _smoothstep(A - base_a, _VESSEL_LO * noise, _VESSEL_HI * noise)

    a_p20 = float(np.percentile(A[core], 20.0))
    b_med = float(np.median(B[core]))
    a_nat = min(a_p20, _WARM_CAP + _WARM_SLOPE * max(b_med, 0.0)
                + _COOL_SLOPE * max(-b_med, 0.0))
    target = base_a - _DIFFUSE_FRAC * np.maximum(base_a - a_nat, 0.0)

    # 3. Fix.
    k = s * support
    kv = k * vessel
    A_new = A - k * np.maximum(A - target, 0.0)
    B_new = B - kv * np.maximum(B - base_b, 0.0)
    L_new = L + kv * np.minimum(np.maximum(base_l - L, 0.0),
                                _VESSEL_LIFT_MAX * l_white)
    l_cap = float(np.percentile(L[core], 99.0))
    lifted = L_new * (1.0 + _LIFT * k)
    # The lift never takes a pixel past the white's own 99th percentile, and
    # never lowers one already above it (catchlights keep their level).
    L_new = np.maximum(L_new, np.minimum(lifted, l_cap))
    L_new = np.minimum(L_new, np.maximum(L, l_cap))

    out_lab = np.stack([L_new, A_new + 128.0, B_new + 128.0], axis=-1)
    fixed = lab_f32_to_bgr_f32(out_lab.astype(np.float32))
    # Blend by the support so float round-off never touches unsupported pixels.
    wmix = (support > 1e-3).astype(np.float32)[..., None]
    new_crop = crop * (1.0 - wmix) + fixed * wmix

    out = img_bgr.copy()
    if img_bgr.dtype == np.uint8:
        out[y0:y1, x0:x1] = np.clip(np.rint(new_crop), 0, 255).astype(np.uint8)
    else:
        out[y0:y1, x0:x1] = new_crop.astype(img_bgr.dtype)
    return out


def calm_bloodshot_eyes(
    img_bgr: np.ndarray,
    regions: Any,
    strength: float,
) -> np.ndarray:
    """Calm both eyes' whites using ``regions`` (FaceRegions-like).

    Uses ``left_sclera``/``right_sclera`` when present, else eye minus iris.
    """
    if strength <= 0:
        return img_bgr
    out = img_bgr
    for side in ("left", "right"):
        iris = getattr(regions, f"{side}_iris", None)
        sclera = getattr(regions, f"{side}_sclera", None)
        if sclera is None:
            eye = getattr(regions, f"{side}_eye", None)
            if eye is None or iris is None:
                continue
            sclera = np.clip(normalize_mask(eye) - normalize_mask(iris), 0.0, 1.0)
        out = calm_bloodshot_eye(out, sclera, iris, strength)
    return out
