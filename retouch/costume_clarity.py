"""Costume Clarity: local contrast and texture on the costume only.

The global ``clarity`` control (``grading._F_add_clarity``) lifts detail over
the whole frame, so any amount that makes fabric, armour, vinyl or lace read
crisply also roughens the skin the face pipeline just smoothed. Cosplay
photos want the opposite split: the costume is the craft on show, the skin
should stay soft. This op adds clarity and fine texture to costume and props
only and leaves skin, painted skin, hair, wigs and the face exactly as they
are.

Costume region (work resolution, face width scaled to about 400 px):

* the person (selfie person mask, plus the multiclass segmenter's *clothes*
  and *others* classes for props past the silhouette). The segmenter is
  unsure on black vinyl gloves and bunny ears (on Alex's bunny photos it
  gave them little clothes confidence), so the costume is everything on the
  person that is known not to be something else, rather than only what the
  segmenter calls clothes;
* minus skin: the body-skin and face-skin classes, but only where the pixel's
  chromaticity (a*/L*, b*/L*) and lightness could be this person's skin, so a
  black glove the segmenter calls arm skin still counts as costume. The skin
  colour model is measured on this photo's own confident skin (samples at
  least half as bright as its median), so it follows darker and lighter skin
  alike; when that skin is near-neutral (white paint) only the classes decide;
* minus hair: its hair class and the whole-frame hair mask
  (``FaceParser.parse_hair_full_image``, which grows hair over long pale
  wigs), the face pipeline's own skin / hair / lip masks and an ellipse over
  each face box (eyes, lashes, brows, mouth);
* minus painted skin when a face is painted (``body_paint`` detection: the
  painted face plus same-coloured person pixels connected to it), except
  where the segmenter is sure it is clothes (a white bow under white paint);
* eroded and feathered so the ramp sits inside the costume (no halo on skin);
* at full resolution, minus pixels whose colour sits close to that skin
  model, so skin seen through fishnet or lace keeps its texture.

Enhancement (CIELab L only, so colours keep their hue and chroma):

* a fine *texture* band (L minus a small Gaussian, weave / lace / sequins)
  and a *clarity* band (that Gaussian minus an edge-preserving guided filter,
  seams / panels / armour relief), sizes scaled to the subject's face width;
* each band's boost is soft-limited at a few times that band's own robust
  spread on this costume, so strong edges and specular glints on vinyl do
  not halo or blow out, whatever the costume's tone;
* the texture band is cored below the image's own noise level (2x2 Haar
  estimate on the costume), so high-ISO grain in a black outfit is not
  amplified;
* the change rolls off toward the L* = 0 and 100 encoding limits so black
  and white fabric are not crushed or clipped.

No absolute intensity threshold is used on the subject: every level is the
photo's own (skin colour, band spreads, noise), see
``tests/test_costume_clarity.py``. Off by default (strength 0 is a no-op).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

from .utils import normalize_mask

_logger = logging.getLogger(__name__)

# Multiclass segmenter channels (model card order, see parsing._segment_classes).
_BG, _HAIR, _BODY_SKIN, _FACE_SKIN, _CLOTHES, _OTHERS = range(6)

# Mask work resolution: faces scaled to about this width.
_WORK_FACE_WIDTH = 400.0
# Class confidence ramp (lo -> hi).
_CLS_LO, _CLS_HI = 0.3, 0.6
# Face ellipse padding (x face box).
_FACE_PAD = 1.25
# Feather sigma (x face width); the costume mask is eroded by ~1.5 sigma first.
_FEATHER_FW = 0.03

# Band sizes (x face width): fine texture Gaussian sigma, clarity guided radius.
_FINE_SIGMA_FW = 0.006
_MID_RADIUS_FW = 0.06
# Band gains at strength 100.
_FINE_GAIN = 2.0
_MID_GAIN = 2.5
# Soft limit of each band's boost, in units of that band's robust spread.
_LIMIT_SIGMAS = 4.0
# Guided-filter eps: edges above this many mid-band spreads are kept as base.
_EPS_SIGMAS = 2.0
# Coring: each band's boost ramps in from 2x to 4x that band's own noise
# sigma (the image noise passed through the same band filter).
_CORE_LO, _CORE_HI = 2.0, 4.0
# Roll-off near the L* encoding limits (L* units).
_EDGE_ROLLOFF_L = 8.0

# Skin colour veto: distance in robust spreads of the skin's chromaticity.
_SKIN_D_IN, _SKIN_D_OUT = 1.5, 2.5
# Wider band where the segmenter also says skin.
_SKIN_D_IN_WIDE, _SKIN_D_OUT_WIDE = 3.5, 6.0
# Pixels darker than this share of the skin's own 2nd-percentile L* ramp
# out of the skin test.
_SKIN_DARK_LO, _SKIN_DARK_HI = 0.35, 0.6
# Minimum chromaticity spread (a*/L*, b*/L* units) so a tight skin cluster
# still tolerates shading.
_SKIN_SPREAD_FLOOR = 0.02
# Skin whose median chroma / L* is below this is near-neutral (white paint):
# the colour veto would catch white and grey costume parts, so it is skipped.
_SKIN_NEUTRAL_RATIO = 0.06
_MIN_SKIN_PX = 200
# Skin samples for the colour model: L* at least this share of the median.
_SKIN_SAMPLE_REL_L = 0.5


def _ramp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _as_2d(mask: Optional[np.ndarray], size: Tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = normalize_mask(mask)
    if m.ndim == 3:
        m = m[..., 0]
    if (m.shape[1], m.shape[0]) != size:
        m = cv2.resize(m, size, interpolation=cv2.INTER_AREA)
    return m


def _robust_sigma(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return 1.4826 * float(np.median(np.abs(x - med)))


def _sample(x: np.ndarray, sel: np.ndarray, n: int = 400_000) -> np.ndarray:
    v = x[sel]
    if v.size > n:
        v = v[:: max(1, v.size // n)]
    return v


def _guided_filter(I: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Self-guided filter (He et al.) on one channel.

    Fast variant (He & Sun 2015): the linear coefficients are fitted on a copy
    subsampled by about r / 8 and upsampled, which at the clarity radius
    (tens of px at full resolution) is visually identical and much cheaper.
    """
    s = max(1, r // 8)
    Is = I if s == 1 else cv2.resize(I, None, fx=1.0 / s, fy=1.0 / s, interpolation=cv2.INTER_AREA)
    rs = max(1, int(round(r / s)))
    k = (2 * rs + 1, 2 * rs + 1)
    mean_I = cv2.boxFilter(Is, -1, k)
    var_I = cv2.boxFilter(Is * Is, -1, k) - mean_I * mean_I
    a = var_I / (var_I + eps)
    b = mean_I - a * mean_I
    a = cv2.boxFilter(a, -1, k)
    b = cv2.boxFilter(b, -1, k)
    if s > 1:
        size = (I.shape[1], I.shape[0])
        a = cv2.resize(a, size, interpolation=cv2.INTER_LINEAR)
        b = cv2.resize(b, size, interpolation=cv2.INTER_LINEAR)
    return a * I + b


def _noise_sigma(L: np.ndarray) -> float:
    """Noise sigma of L from 2x2 Haar HH coefficients in the flattest blocks.

    HH is RMS-pooled over 8x8 coefficient blocks (16x16 px) and the 10th
    percentile block is taken: fine weave, fishnet or sequins put real
    structure into HH, so a median over the costume would read the costume's
    own texture as noise and core it away. Somewhere in the crop (backdrop,
    smooth skin, flat fabric) there are blocks holding only sensor noise.
    """
    h2, w2 = (L.shape[0] // 16) * 16, (L.shape[1] // 16) * 16
    if h2 < 16 or w2 < 16:
        return 0.0
    q = L[:h2, :w2]
    hh = (q[0::2, 0::2] - q[0::2, 1::2] - q[1::2, 0::2] + q[1::2, 1::2]) * 0.5
    bh, bw = hh.shape[0] // 8, hh.shape[1] // 8
    rms = np.sqrt((hh.reshape(bh, 8, bw, 8) ** 2).mean(axis=(1, 3)))
    return float(np.percentile(rms, 10))


def _frame_noise_sigma(img: np.ndarray, tile: int = 64, max_tiles: int = 400) -> float:
    """:func:`_noise_sigma` over tiles spread across the whole frame.

    Sensor noise is a property of the frame, and a costume crop can be all
    structure (a fishnet leg), so flat blocks are looked for everywhere.
    """
    H, W = img.shape[:2]
    ny, nx = max(1, H // tile), max(1, W // tile)
    step = max(1, int(np.ceil(np.sqrt(ny * nx / max_tiles))))
    tiles = [
        np.clip(img[y * tile:(y + 1) * tile, x * tile:(x + 1) * tile], 0.0, 1.0)
        for y in range(0, ny, step)
        for x in range(0, nx, step)
    ]
    if not tiles or min(H, W) < 16:
        return 0.0
    strip = np.concatenate([t for t in tiles if t.shape[:2] == tiles[0].shape[:2]], axis=1)
    L = cv2.cvtColor(strip.astype(np.float32), cv2.COLOR_BGR2LAB)[..., 0]
    return _noise_sigma(L)


def _band_noise(noise: float, s_fine: float, r_mid: int) -> Tuple[float, float]:
    """Std of white noise ``noise`` after the fine and clarity band filters.

    Measured on a fixed synthetic field (Gaussian stand-in for the guided
    filter, which on pure noise smooths like a box of the same radius).
    """
    if noise <= 0:
        return 0.0, 0.0
    n = max(64, int(8 * r_mid))
    rng = np.random.default_rng(0)
    z = rng.standard_normal((n, n)).astype(np.float32)
    b = cv2.GaussianBlur(z, (0, 0), sigmaX=s_fine)
    c = cv2.GaussianBlur(z, (0, 0), sigmaX=max(2.0, r_mid / 2.0))
    m = slice(n // 4, 3 * n // 4)
    return noise * float((z - b)[m, m].std()), noise * float((b - c)[m, m].std())


def _skin_chroma_model(lab: np.ndarray, skin: np.ndarray) -> Optional[Dict[str, float]]:
    """Median and robust spread of the skin's chromaticity (a*/L*, b*/L*).

    Only skin at least half as bright as its own median is sampled: in deep
    shadow chromaticity is noise, and the segmenter's skin class also bleeds
    onto dark gloves and stockings.
    """
    if int(skin.sum()) < _MIN_SKIN_PX:
        return None
    L_all = lab[..., 0][skin]
    keep = L_all >= _SKIN_SAMPLE_REL_L * float(np.median(L_all))
    if int(keep.sum()) < _MIN_SKIN_PX:
        return None
    L = L_all[keep]
    qa = lab[..., 1][skin][keep] / L
    qb = lab[..., 2][skin][keep] / L
    ma, mb = float(np.median(qa)), float(np.median(qb))
    return {
        "l_lo": float(np.percentile(L, 5)),
        "ma": ma,
        "mb": mb,
        "sa": max(_robust_sigma(qa), _SKIN_SPREAD_FLOOR),
        "sb": max(_robust_sigma(qb), _SKIN_SPREAD_FLOOR),
        "neutral": float(np.hypot(ma, mb)) < _SKIN_NEUTRAL_RATIO,
    }


def _skin_colour_likeness(
    lab: np.ndarray,
    model: Dict[str, float],
    d_in: Optional[float] = None,
    d_out: Optional[float] = None,
) -> np.ndarray:
    """0-1 likeness of each pixel's chromaticity and lightness to the skin's."""
    d_in = _SKIN_D_IN if d_in is None else d_in
    d_out = _SKIN_D_OUT if d_out is None else d_out
    L = np.maximum(lab[..., 0], 5.0)
    da = (lab[..., 1] / L - model["ma"]) / model["sa"]
    db = (lab[..., 2] / L - model["mb"]) / model["sb"]
    d = np.sqrt(da * da + db * db)
    # Far darker than this person's own darkest skin, chromaticity is noise
    # (black fabric), so it is not skin.
    dark = _ramp(lab[..., 0], _SKIN_DARK_LO * model["l_lo"], _SKIN_DARK_HI * model["l_lo"])
    return (1.0 - _ramp(d, d_in, d_out)) * dark


def _face_ellipses(size: Tuple[int, int], boxes: Sequence[Tuple[int, int, int, int]]) -> np.ndarray:
    w, h = size
    out = np.zeros((h, w), dtype=np.uint8)
    for x, y, bw, bh in boxes:
        cx, cy = x + bw / 2.0, y + bh / 2.0
        ax, ay = max(1, int(bw * _FACE_PAD / 2)), max(1, int(bh * _FACE_PAD / 2))
        cv2.ellipse(out, (int(cx), int(cy)), (ax, ay), 0, 0, 360, 1, -1)
    return out.astype(np.float32)


def costume_mask(
    img: np.ndarray,
    face_boxes: Sequence[Tuple[int, int, int, int]],
    class_probs: Optional[np.ndarray],
    person_mask: Optional[np.ndarray] = None,
    hair_mask: Optional[np.ndarray] = None,
    protect: Optional[np.ndarray] = None,
    face_skin: Optional[np.ndarray] = None,
    paint_mask: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]], Dict[str, Any]]:
    """Feathered costume mask at the size of ``img`` (work resolution).

    Args:
        img: (h, w, 3) float32 BGR [0, 1] work-resolution image.
        face_boxes: (x, y, w, h) per face, in ``img`` pixels.
        class_probs: (h, w, 6) multiclass confidences, or None.
        person_mask, hair_mask, protect, face_skin, paint_mask: optional
            (h, w) masks; ``protect`` is everything the face pipeline owns.

    Returns:
        ``(mask, skin_model, diag)``; mask is None when no costume is found.
    """
    diag: Dict[str, Any] = {}
    h, w = img.shape[:2]
    size = (w, h)
    fw = float(np.median([b[2] for b in face_boxes])) if face_boxes else 0.1 * min(h, w)
    fw = max(fw, 8.0)
    lab = cv2.cvtColor(np.clip(img, 0.0, 1.0).astype(np.float32), cv2.COLOR_BGR2LAB)

    # The costume is the person minus what is known not to be costume, so a
    # black glove or vinyl panel the segmenter is unsure about still counts.
    pm = _as_2d(person_mask, size)
    person = _ramp(pm, 0.3, 0.7) if pm is not None else None
    have_classes = class_probs is not None and class_probs.shape[:2] == (h, w)
    if have_classes:
        p = class_probs
        clothes = _ramp(np.clip(p[..., _CLOTHES] + p[..., _OTHERS], 0.0, 1.0), _CLS_LO, _CLS_HI)
        cost = clothes if person is None else np.maximum(person, clothes)
        skin_cls = _ramp(np.maximum(p[..., _BODY_SKIN], p[..., _FACE_SKIN]), _CLS_LO, _CLS_HI)
        hair_cls = _ramp(p[..., _HAIR], _CLS_LO, _CLS_HI)
        body_skin = (p[..., _BODY_SKIN] > 0.7) & (p.argmax(axis=2) == _BODY_SKIN)
        diag["source"] = "classes"
    elif person is not None:
        cost = person
        clothes = np.zeros((h, w), dtype=np.float32)
        skin_cls = np.zeros((h, w), dtype=np.float32)
        hair_cls = np.zeros((h, w), dtype=np.float32)
        body_skin = np.zeros((h, w), dtype=bool)
        diag["source"] = "person_mask"
    else:
        diag["reason"] = "no_segmentation"
        return None, None, diag

    # Skin colour model: confident body skin, else the face's own skin.
    model = _skin_chroma_model(lab, body_skin)
    if model is None and face_skin is not None:
        fs = _as_2d(face_skin, size)
        model = _skin_chroma_model(lab, fs > 0.5)
    if model is not None:
        diag["skin_chroma"] = round(float(np.hypot(model["ma"], model["mb"])), 4)
        if model["neutral"]:
            diag["skin_colour_veto"] = "skipped_neutral_skin"
            model = None
    if model is None and not have_classes:
        diag["reason"] = "no_skin_model"
        return None, None, diag

    if model is not None:
        # The skin class is trusted where the colour could be this person's
        # skin (a black glove labelled arm skin is not).
        # Skin seen through fishnet or lace, or a stray patch labelled
        # clothes, is caught per pixel at full resolution (enhance_costume).
        skin = skin_cls * _skin_colour_likeness(lab, model, _SKIN_D_IN_WIDE, _SKIN_D_OUT_WIDE)
    else:
        skin = skin_cls

    excl = np.maximum(skin, hair_cls)
    paint = _as_2d(paint_mask, size)
    if paint is not None:
        # Painted skin is excluded, but not a confident costume part in the
        # paint's colour (a white bow under white face paint).
        excl = np.maximum(excl, paint * (1.0 - _ramp(clothes, 0.5, 0.9)))
    for m in (_as_2d(hair_mask, size), _as_2d(protect, size)):
        if m is not None:
            excl = np.maximum(excl, m)
    if face_boxes:
        excl = np.maximum(excl, _face_ellipses(size, face_boxes))
    cost = cost * (1.0 - np.clip(excl, 0.0, 1.0))

    sigma = max(1.0, _FEATHER_FW * fw)
    hard = (cost > 0.5).astype(np.uint8)
    ke = max(3, int(round(3.0 * sigma)) | 1)
    hard = cv2.erode(hard, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ke, ke)))
    soft = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), sigmaX=sigma)
    mask = np.minimum(soft, cost)
    diag["costume_share"] = round(float((mask > 0.5).mean()), 4)
    if mask.max() < 0.05:
        diag["reason"] = "no_costume"
        return None, model, diag
    return mask.astype(np.float32), model, diag


def enhance_costume(
    img: np.ndarray,
    mask: np.ndarray,
    strength: float,
    face_width: float,
    skin_model: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Add costume clarity and texture to ``img`` inside ``mask``.

    Args:
        img: (H, W, 3) float32 BGR in [0, 1], full resolution.
        mask: (H, W) float32 costume mask in [0, 1].
        strength: 0-1.
        face_width: subject face width in ``img`` pixels (band scale).
        skin_model: chromaticity model from :func:`costume_mask`; pixels
            inside it are left alone.
    """
    diag: Dict[str, Any] = {}
    support = mask > 1e-3
    if strength <= 0 or not support.any():
        return img, diag
    H, W = img.shape[:2]
    fw = max(float(face_width), 8.0)
    r_mid = max(4, int(round(_MID_RADIUS_FW * fw)))
    s_fine = max(1.0, _FINE_SIGMA_FW * fw)
    pad = 3 * r_mid + int(4 * s_fine)
    ys, xs = np.where(support)
    y0, y1 = max(0, ys.min() - pad), min(H, ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(W, xs.max() + pad + 1)

    crop = np.clip(img[y0:y1, x0:x1], 0.0, 1.0).astype(np.float32)
    m = mask[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    L = lab[..., 0]
    # Per-pixel work runs on the costume pixels only (flat vectors); the
    # filters need the whole crop.
    ry, rx = np.nonzero(m > 1e-3)
    if skin_model is not None:
        veto = np.zeros(m.shape, np.float32)
        veto[ry, rx] = _skin_colour_likeness(lab[ry, rx][:, None, :], skin_model)[:, 0]
        veto = cv2.GaussianBlur(veto, (0, 0), sigmaX=1.0)
        mv = m[ry, rx] * (1.0 - veto[ry, rx])
    else:
        mv = m[ry, rx]
    sel = np.zeros(m.shape, bool)
    sel[ry, rx] = mv > 0.5
    if int(sel.sum()) < 64:
        diag["reason"] = "costume_all_skin_coloured"
        return img, diag

    blur = cv2.GaussianBlur(L, (0, 0), sigmaX=s_fine)
    # Edge-preserving threshold from the band's spread, measured on a 1/4 copy.
    q = 4 if min(L.shape) >= 64 else 1
    Lq = cv2.resize(blur, None, fx=1.0 / q, fy=1.0 / q, interpolation=cv2.INTER_AREA)
    selq = cv2.resize(sel.astype(np.uint8), (Lq.shape[1], Lq.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    coarse_q = cv2.GaussianBlur(Lq, (0, 0), sigmaX=max(1.0, r_mid / (2.0 * q)))
    sig_mid0 = max(_robust_sigma(_sample(Lq - coarse_q, selq)), 0.05)
    eps = (_EPS_SIGMAS * sig_mid0 / 100.0) ** 2
    base = _guided_filter(blur / 100.0, r_mid, eps) * 100.0

    Lv = L[ry, rx]
    fine = Lv - blur[ry, rx]
    mid = blur[ry, rx] - base[ry, rx]
    sv = sel[ry, rx]
    sig_fine = max(_robust_sigma(_sample(fine, sv)), 0.02)
    sig_mid = max(_robust_sigma(_sample(mid, sv)), 0.05)
    noise = _frame_noise_sigma(img)
    k_fine = _LIMIT_SIGMAS * sig_fine
    k_mid = _LIMIT_SIGMAS * sig_mid
    n_fine, n_mid = _band_noise(noise, s_fine, r_mid)
    core_f = _ramp(np.abs(fine), _CORE_LO * n_fine, _CORE_HI * n_fine) if n_fine > 0 else 1.0
    core_m = _ramp(np.abs(mid), _CORE_LO * n_mid, _CORE_HI * n_mid) if n_mid > 0 else 1.0
    delta = strength * (
        _FINE_GAIN * core_f * k_fine * np.tanh(fine / k_fine)
        + _MID_GAIN * core_m * k_mid * np.tanh(mid / k_mid)
    )
    roll = np.where(delta > 0, 100.0 - Lv, Lv) / _EDGE_ROLLOFF_L
    delta = delta * np.clip(roll, 0.0, 1.0) * mv
    px = lab[ry, rx]
    px[:, 0] = np.clip(Lv + delta, 0.0, 100.0)
    new = np.clip(cv2.cvtColor(px[:, None, :], cv2.COLOR_LAB2BGR)[:, 0], 0.0, 1.0)
    out = img.copy()
    out[y0 + ry, x0 + rx] = new.astype(out.dtype)
    diag.update(
        fine_sigma=round(sig_fine, 3),
        mid_sigma=round(sig_mid, 3),
        noise_sigma=round(noise, 3),
        mean_abs_delta=round(float(np.abs(delta[sv]).mean()), 3),
    )
    return out, diag


def apply_costume_clarity(
    img: np.ndarray,
    strength: float,
    face_boxes: Sequence[Tuple[int, int, int, int]],
    segment_classes=None,
    hair_full=None,
    person_mask: Optional[np.ndarray] = None,
    protect: Optional[np.ndarray] = None,
    face_skin: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Costume Clarity on a float32 BGR [0, 1] image; returns ``(image, diag)``.

    Args:
        img: (H, W, 3) float32 BGR [0, 1].
        strength: 0-1 (0 is a no-op).
        face_boxes: (x, y, w, h) per face, full-resolution pixels.
        segment_classes: callable(uint8 BGR) -> (h, w, 6) confidences or None.
        hair_full: callable(uint8 BGR) -> (h, w) hair confidence or None.
        person_mask: (H, W) person mask.
        protect: (H, W) union of the face pipeline's skin, hair and lip masks.
        face_skin: (H, W) face skin mask (paint detection, skin colour fallback).
    """
    diag: Dict[str, Any] = {"applied": False}
    if strength <= 0:
        diag["reason"] = "off"
        return img, diag
    H, W = img.shape[:2]
    fw_full = float(np.median([b[2] for b in face_boxes])) if face_boxes else 0.1 * min(H, W)
    scale = min(1.0, _WORK_FACE_WIDTH / max(fw_full, 1.0))
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

    paint = None
    skin_s = _as_2d(face_skin, size)
    if skin_s is not None and boxes_s:
        from .body_paint import (
            _face_skin_masks,
            _to_oklab,
            detect_painted_face,
            is_monochrome,
            paint_region_mask,
        )

        ok = _to_oklab(work)
        mono = is_monochrome(ok)
        painted, painted_skins = [], []
        for i, sk in enumerate(_face_skin_masks(skin_s, boxes_s)):
            pf = detect_painted_face(ok, sk, face_index=i, monochrome=mono)
            if pf is not None:
                painted.append(pf)
                painted_skins.append(sk)
        if painted:
            paint = paint_region_mask(ok, painted, painted_skins, _as_2d(person_mask, size))
            diag["painted_faces"] = len(painted)

    mask_s, model, mdiag = costume_mask(
        work, boxes_s, probs, person_mask=person_mask, hair_mask=hair,
        protect=protect, face_skin=face_skin, paint_mask=paint,
    )
    diag.update(mdiag)
    if mask_s is None:
        return img, diag
    mask = mask_s if scale >= 1.0 else cv2.resize(mask_s, (W, H), interpolation=cv2.INTER_LINEAR)
    out, ediag = enhance_costume(img, np.clip(mask, 0.0, 1.0), strength, fw_full, model)
    diag.update(ediag)
    diag["applied"] = out is not img
    if diag["applied"]:
        _logger.info("costume clarity: %s", diag)
    return out, diag
