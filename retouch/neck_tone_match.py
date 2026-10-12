"""Neck Tone Match: keep the neck and chest the same tone as the retouched face.

Whitening, warming, Face Polish's exposure lift or a recipe's face look move
the face skin but not the skin below it, so the face ends up lighter, pinker
or more yellow than the neck it sits on. The engine had a hidden version of
this (``SkinProcessor.harmonize_neck``, run automatically whenever whitening
or tone evening was on): it worked inside the face crop only, matched the
neck's *median* to the face's (so the natural shadow under the chin was
lifted away whenever it acted), and on the bunny test photos it moved the
neck by at most 2-6 levels. This module replaces it when the Neck Tone Match
slider is on; with the slider at 0 the old automatic pass still runs, so no
recipe changes.

How it works, per face:

* **Where.** A neck-and-chest zone laid out in the face's own frame (chin
  point, chin-to-forehead axis, face width), so a tilted or lying face gets a
  zone that follows it. Inside the person mask component that holds the face.
* **Which pixels.** The neck's own skin colour is sampled just below the
  chin from the photo *before* face edits (a hue/chroma skin test that is a
  ratio to lightness, so it holds for every skin tone, then the median a*/b*).
  Only pixels within that colour's own spread are moved: collars, ribbons,
  wigs and costume parts in other colours stay out. Hair masks are excluded
  too.
* **How much.** Measured in linear light and CIELab, on the pre-edit photo
  and on the current image:

  - brightness: the change the retouch made to the face relative to the
    neck (``(face_out/face_in) / (neck_out/neck_in)``), applied as one
    linear-light gain. A gain keeps every shadow the same fraction darker
    than the skin around it, so the shadow under the chin stays. The
    brightness gap the photo already had is left alone: on a real photo it
    is mostly light falloff and the shadow under the jaw.
  - colour: the a*/b* gap the retouch opened between face and neck, plus
    ``BASE_AB_PULL`` of the gap the photo already had (foundation that does
    not match the neck), as one a*/b* offset scaled down in shadows.

* **Left alone.** A face whose pre-edit skin reads as face paint (white,
  grey, blue, green: hue or chroma/lightness outside natural skin, the same
  test Skin Warmth uses) gets no match, so the neck is not dragged toward the
  paint. A neck sample that reads as paint is skipped the same way. A
  pre-existing colour gap many times the neck's own spread is treated as
  paint or costume, not foundation, and is not pulled.

No absolute intensity threshold: every gate is a hue angle, a chroma/L*
ratio, a ratio of the face's own brightness or the neck's own colour spread.
"""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

from .utils import guided_filter

__all__ = ["neck_tone_match", "neck_support", "BASE_AB_PULL", "MAX_GAIN"]

#: Share of the face/neck colour gap the photo already had that is closed at
#: strength 100 (foundation that does not match the neck).
BASE_AB_PULL = 0.6
#: Largest linear-light gain either way (about 1.6 EV).
MAX_GAIN = 3.0
#: Largest a*/b* offset (CIELab units).
MAX_AB = 14.0
#: A pre-existing face/neck gap this far apart in chromaticity (a*/L*,
#: b*/L*) is paint or costume, not foundation: the base pull fades out
#: between these distances. Natural skin across one body sits within ~0.1.
_BASE_GAP_FADE = (0.12, 0.2)

#: Floor for the neck colour key's width, in a*/L*, b*/L* units.
_KEY_MIN_CHROMATICITY = 0.07
#: Neck/chest pixels within this chromaticity distance of the face's own
#: skin are candidates for the colour sample (a red corset sits ~0.3-0.4
#: away); the sample itself is the pixels within ``_MODE_RADIUS`` of their
#: most common colour.
_SEED_RADIUS = 0.15
_MODE_RADIUS = 0.05

# MediaPipe face-mesh indices.
_CHIN = 152
_FOREHEAD = 10

# Natural-skin colour test (true CIELab), shared with Skin Warmth.
_HUE_ON = (-5.0, 75.0)
_HUE_RAMP = 20.0
_NEUTRAL_OFF, _NEUTRAL_ON = 0.05, 0.08
# Neck sample hue band: pale skin under cool light reads down to ~-30 deg.
_NECK_HUE_ON = (-35.0, 75.0)
# Near-neutral cut-off for body skin pixels (see _skin_like_pixels).
_BODY_NEUTRAL = (0.035, 0.06)


def _clip01(x: np.ndarray) -> np.ndarray:
    np.maximum(x, 0.0, out=x)
    np.minimum(x, 1.0, out=x)
    return x


def _norm(mask: Optional[np.ndarray], shape) -> Optional[np.ndarray]:
    if mask is None:
        return None
    m = np.asarray(mask).astype(np.float32)
    if m.ndim == 3:
        m = np.ascontiguousarray(m[..., 0])
    if m.shape != tuple(shape):
        return None
    if m.size and float(m.max()) > 1.5:
        m *= 1.0 / 255.0
    return _clip01(m)


def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img.astype(np.float32) * (1.0 / 255.0)
    return _clip01(img.astype(np.float32))


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x * (1.0 / 12.92),
                    np.power((x + 0.055) * (1.0 / 1.055), 2.4)).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def _luma(lin_bgr: np.ndarray) -> np.ndarray:
    return (0.0722 * lin_bgr[..., 0] + 0.7152 * lin_bgr[..., 1]
            + 0.2126 * lin_bgr[..., 2]).astype(np.float32)


def _skin_colour_gate(lab: np.ndarray, hue_on=_HUE_ON) -> np.ndarray:
    """Per-pixel 0-1 natural-skin colour likelihood (hue band, chroma/L*)."""
    a, b, L = lab[..., 1], lab[..., 2], lab[..., 0]
    hue = np.degrees(np.arctan2(b, a))
    lo, hi = hue_on
    g_hue = np.clip(1.0 - np.maximum(lo - hue, hue - hi) / _HUE_RAMP, 0.0, 1.0)
    ratio = np.hypot(a, b) / np.maximum(L, 1.0)
    g_neu = np.clip((ratio - _NEUTRAL_OFF) / (_NEUTRAL_ON - _NEUTRAL_OFF), 0.0, 1.0)
    return (g_hue * g_neu).astype(np.float32)


def _colour_gate(a: float, b: float, L: float, cool: bool = False) -> float:
    """Scalar :func:`_skin_colour_gate` for a median colour. ``cool`` widens
    the hue band for body skin under cool light (pinkish, hue below 0)."""
    lab = np.array([[[L, a, b]]], np.float32)
    return float(_skin_colour_gate(lab, _NECK_HUE_ON if cool else _HUE_ON)[0, 0])


def _skin_like_pixels(lab: np.ndarray) -> np.ndarray:
    """Loose per-pixel test for body skin: anything not near-neutral.

    Pale skin under cool or mixed light, and its bluish shadows and veins,
    read pinkish-magenta to blue-grey (hue well below 0 deg), which the
    face-paint hue test rejects; leaving those pixels out left dark blotches
    once the skin around them was brightened. The hue side is left to the
    colour key, which is centred on this person's own neck/chest colour.
    Near-neutral pixels (white collars, ribbons, grey fabric) stay out.
    """
    a, b, L = lab[..., 1], lab[..., 2], lab[..., 0]
    ratio = np.hypot(a, b) / np.maximum(L, 1.0)
    return np.clip((ratio - _BODY_NEUTRAL[0]) / (_BODY_NEUTRAL[1] - _BODY_NEUTRAL[0]),
                   0.0, 1.0).astype(np.float32)


def _chroma_mode(ca: np.ndarray, cb: np.ndarray, fca: float, fcb: float):
    """Peak of the 2D chromaticity histogram (bins 0.015) around the face."""
    r = _SEED_RADIUS
    nb = round(2 * r / 0.015)
    hist, ea, eb = np.histogram2d(ca, cb, bins=nb, range=[[fca - r, fca + r], [fcb - r, fcb + r]])
    hist = cv2.GaussianBlur(hist.astype(np.float32), (3, 3), 0)
    i, j = np.unravel_index(int(np.argmax(hist)), hist.shape)
    return 0.5 * (ea[i] + ea[i + 1]), 0.5 * (eb[j] + eb[j + 1])


def _landmarks_px(face, w: int, h: int) -> Optional[np.ndarray]:
    lms = getattr(face, "landmarks", None)
    if lms is None:
        return None
    pts = getattr(lms, "landmark", lms)
    try:
        arr = np.array([[p.x * w, p.y * h] for p in pts], np.float32)
    except (AttributeError, TypeError):
        return None
    if arr.shape[0] <= max(_CHIN, _FOREHEAD):
        return None
    return arr


def _face_frame(pts: np.ndarray):
    """Chin point, unit 'down' axis, unit 'across' axis, face height, width."""
    chin = pts[_CHIN]
    top = pts[_FOREHEAD]
    down = chin - top
    fh = float(np.hypot(*down))
    if fh < 4.0:
        return None
    down = down / fh
    across = np.array([-down[1], down[0]], np.float32)
    rel = pts - chin
    widths = rel @ across
    fw = float(widths.max() - widths.min())
    if fw < 4.0:
        return None
    return chin, down, across, fh, fw


def _zone_weight(xx, yy, chin, down, across, fh, fw) -> np.ndarray:
    """Neck-and-chest zone in the face's frame: full near, fading out."""
    dx = xx - chin[0]
    dy = yy - chin[1]
    v = (dx * down[0] + dy * down[1]) / fh       # below the chin, face heights
    u = np.abs(dx * across[0] + dy * across[1]) / fw  # sideways, face widths
    # Starts a little above the chin point (under the jaw line), full from
    # just below it to 2.2 face heights down, gone by 3.2.
    wv = np.clip((v + 0.25) / 0.2, 0.0, 1.0) * np.clip((3.2 - v) / 1.0, 0.0, 1.0)
    wu = np.clip((2.4 - u) / 0.8, 0.0, 1.0)
    return (wv * wu).astype(np.float32), v, u


def _face_plan(face, H, W, fs, ref, cur, has_ref, pm, labels, ex) -> Optional[dict]:
    """Per-face support and correction, or None when the face is skipped."""
    pts = _landmarks_px(face, W, H)
    if pts is None:
        return None
    frame = _face_frame(pts)
    if frame is None:
        return None
    chin, down, across, fh, fw = frame

    # Bounding box of the zone (and the face) in pixels.
    corners = np.array([chin + down * (vv * fh) + across * (uu * fw)
                        for vv in (-1.3, 3.3) for uu in (-2.5, 2.5)])
    x0 = int(max(0, np.floor(corners[:, 0].min())))
    x1 = int(min(W, np.ceil(corners[:, 0].max()) + 1))
    y0 = int(max(0, np.floor(corners[:, 1].min())))
    y1 = int(min(H, np.ceil(corners[:, 1].max()) + 1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    sl = (slice(y0, y1), slice(x0, x1))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    zone, v, _ = _zone_weight(xx, yy, chin, down, across, fh, fw)

    # The face itself (landmark hull) is never part of the zone.
    hull = cv2.convexHull(pts.astype(np.int32))
    face_poly = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(face_poly, hull, 1)
    face_poly = face_poly[sl]
    face_sel = (face_poly > 0) & (fs[sl] > 0.5)
    if int(face_sel.sum()) < 64:
        return None
    kd = max(3, int(0.06 * fw)) | 1
    face_grown = cv2.dilate(face_poly, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)))
    zone *= 1.0 - cv2.GaussianBlur(face_grown.astype(np.float32), (kd, kd), 0)

    # Same person only: the person-mask component holding the face.
    if labels is not None:
        lab_ids = labels[sl][face_sel]
        lab_ids = lab_ids[lab_ids != 0]
        if lab_ids.size:
            owner = int(np.bincount(lab_ids).argmax())
            zone *= (labels[sl] == owner).astype(np.float32)
        zone *= pm[sl]
    if ex is not None:
        zone *= 1.0 - ex[sl]

    ref_c = np.ascontiguousarray(ref[sl])
    cur_c = np.ascontiguousarray(cur[sl])
    lab_ref = cv2.cvtColor(ref_c, cv2.COLOR_BGR2Lab)
    lab_cur = lab_ref if not has_ref else cv2.cvtColor(cur_c, cv2.COLOR_BGR2Lab)

    # Face paint: no match (judged on the pre-edit face).
    fa = float(np.median(lab_ref[..., 1][face_sel]))
    fb = float(np.median(lab_ref[..., 2][face_sel]))
    fL = float(np.median(lab_ref[..., 0][face_sel]))
    g_face = _colour_gate(fa, fb, fL)
    if g_face <= 0:
        return None

    # Colour works on a*/L*, b*/L* (chromaticity): shading scales a*/b*
    # roughly with L*, so an a*/b* key dropped the chest under a softbox
    # falloff, and hue is noise on pale, low-chroma skin.
    inv_l = 1.0 / np.maximum(lab_ref[..., 0], 1.0)
    ca = lab_ref[..., 1] * inv_l
    cb = lab_ref[..., 2] * inv_l
    fca = float(np.median(ca[face_sel]))
    fcb = float(np.median(cb[face_sel]))

    # Neck/chest colour sample: the most common chromaticity among the
    # zone's skin-coloured pixels near the face's own skin colour (the face
    # is natural skin here; paint was skipped above). A plain sample just
    # under the chin landed on collars and the jaw shadow, and a median over
    # the zone was pulled toward wig strands.
    skin_like = _skin_like_pixels(lab_ref)
    cand = ((zone > 0.5) & (v < 1.8) & (skin_like > 0.5)
            & (np.hypot(ca - fca, cb - fcb) < _SEED_RADIUS))
    min_seed = max(64, int(0.02 * fh * fw))
    if int(cand.sum()) < min_seed:
        return None
    mca, mcb = _chroma_mode(ca[cand], cb[cand], fca, fcb)
    seed = cand & (np.hypot(ca - mca, cb - mcb) < _MODE_RADIUS)
    if int(seed.sum()) < min_seed:
        return None
    sa_ = lab_ref[..., 1][seed]
    sb_ = lab_ref[..., 2][seed]
    g_neck = _colour_gate(float(np.median(sa_)), float(np.median(sb_)),
                          float(np.median(lab_ref[..., 0][seed])), cool=True)
    if g_neck <= 0:
        return None
    # Colour key around the sample, width from its own spread.
    ca_s, cb_s = ca[seed], cb[seed]
    ca0, cb0 = float(np.median(ca_s)), float(np.median(cb_s))
    c_spread = float(np.hypot(np.percentile(ca_s, 84) - np.percentile(ca_s, 16),
                              np.percentile(cb_s, 84) - np.percentile(cb_s, 16))) / 2.0
    sig_c = max(_KEY_MIN_CHROMATICITY, 1.5 * c_spread)
    # Flat-topped: full inside 1 width, gone by 2.5. A Gaussian key gave
    # skin a weight of 0.5-0.9 that followed its texture, so the correction
    # came out blotchy and short.
    dist = np.hypot(ca - ca0, cb - cb0) / sig_c
    key = np.clip((2.5 - dist) / 1.5, 0.0, 1.0).astype(np.float32)
    weight = zone * key * skin_like
    # Fill holes the key leaves in shadowed skin (shadow under a softbox
    # shifts skin colour a little): leaving them out left dark blotches once
    # the skin around them was brightened. Only skin-ish colours fill in.
    kc = max(3, int(0.2 * fw)) | 1
    closed = cv2.morphologyEx(weight, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc)))
    loose = np.clip((4.0 - dist) / 1.5, 0.0, 1.0) * np.clip(skin_like * 2.0, 0.0, 1.0)
    weight = np.maximum(weight, closed * loose * zone)
    # Edge-aware smoothing so the correction is even across the skin and
    # still stops at the edge of a collar, strap or wig strand.
    r = max(4, int(0.04 * fw))
    weight = np.clip(guided_filter(weight, r, 1e-3, guide=lab_ref[..., 0] * (1.0 / 100.0)), 0.0, 1.0)
    # Skin that clears the key gets the full correction: partial weights
    # left the neck short of the face.
    weight = np.clip((weight - 0.1) * 2.0, 0.0, 1.0) * np.minimum(1.0, 2.0 * zone)
    target = weight > 0.5
    if int(target.sum()) < min_seed:
        return None
    # Pixels well above the skin's own bright end are white fabric (bows,
    # collars) or already-bright highlights: roll the correction off from
    # the skin's p95 to twice it (linear light, so relative to this skin).
    lum_ref = _luma(_srgb_to_linear(ref_c))
    hi = float(np.percentile(lum_ref[target], 95)) + 1e-4
    weight *= np.clip((2.0 * hi - lum_ref) / hi, 0.0, 1.0)
    target = weight > 0.5
    if int(target.sum()) < min_seed:
        return None

    # Brightness: the change the face edits made to the face relative to the
    # neck, in linear light (medians; a ratio, so tone-invariant).
    lg = 0.0
    if has_ref:
        lum_cur = _luma(_srgb_to_linear(cur_c))
        eps = 1e-4
        face_in = float(np.median(lum_ref[face_sel])) + eps
        face_out = float(np.median(lum_cur[face_sel])) + eps
        neck_in = float(np.median(lum_ref[target])) + eps
        neck_out = float(np.median(lum_cur[target])) + eps
        lg = float(np.clip(np.log(face_out / face_in) - np.log(neck_out / neck_in),
                           -np.log(MAX_GAIN), np.log(MAX_GAIN)))

    # Colour: the a*/b* gap the edits opened, plus part of the gap the photo
    # already had (foundation), unless that gap is too big to be foundation.
    rna = float(np.median(lab_ref[..., 1][target]))
    rnb = float(np.median(lab_ref[..., 2][target]))
    gap_a, gap_b = fa - rna, fb - rnb
    lo, hi = _BASE_GAP_FADE
    c_gap = float(np.hypot(fca - float(np.median(ca[target])), fcb - float(np.median(cb[target]))))
    base_ok = float(np.clip((hi - c_gap) / (hi - lo), 0.0, 1.0))
    sh_a = sh_b = 0.0
    if has_ref:
        sh_a = (float(np.median(lab_cur[..., 1][face_sel])) - float(np.median(lab_cur[..., 1][target]))) - gap_a
        sh_b = (float(np.median(lab_cur[..., 2][face_sel])) - float(np.median(lab_cur[..., 2][target]))) - gap_b
    sh_a += BASE_AB_PULL * base_ok * gap_a
    sh_b += BASE_AB_PULL * base_ok * gap_b
    mag = float(np.hypot(sh_a, sh_b))
    if mag > MAX_AB:
        sh_a, sh_b = sh_a * MAX_AB / mag, sh_b * MAX_AB / mag

    return {
        "slice": sl,
        "weight": (weight * np.float32(g_face * g_neck)).astype(np.float32),
        "log_gain": lg,
        "shift_ab": (sh_a, sh_b),
        "face_sel": face_sel,
    }


def _plans(img, faces, face_skin, reference, person_mask, exclude_mask):
    H, W = img.shape[:2]
    fs = _norm(face_skin, (H, W))
    if fs is None or float(fs.max()) < 0.5 or not faces:
        return []
    cur = _to_float01(img)
    has_ref = reference is not None and reference.shape == img.shape
    ref = _to_float01(reference) if has_ref else cur
    pm = _norm(person_mask, (H, W))
    ex = _norm(exclude_mask, (H, W))
    labels = None
    if pm is not None:
        _, labels = cv2.connectedComponents((pm > 0.5).astype(np.uint8))
    plans = []
    for face in faces:
        p = _face_plan(face, H, W, fs, ref, cur, has_ref, pm, labels, ex)
        if p is not None:
            plans.append(p)
    return plans


def neck_support(
    img: np.ndarray,
    faces: Sequence,
    face_skin: Optional[np.ndarray],
    reference: Optional[np.ndarray] = None,
    person_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """(H, W) 0-1 map of the neck/chest pixels the op would move (for QA)."""
    out = np.zeros(img.shape[:2], np.float32)
    for p in _plans(img, faces, face_skin, reference, person_mask, exclude_mask):
        np.maximum(out[p["slice"]], p["weight"], out=out[p["slice"]])
    return out


def neck_tone_match(
    img: np.ndarray,
    faces: Sequence,
    face_skin: Optional[np.ndarray],
    strength: float,
    reference: Optional[np.ndarray] = None,
    person_mask: Optional[np.ndarray] = None,
    exclude_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Match neck and chest skin to the retouched face. See the module docstring.

    Args:
        img: current BGR image (uint8, or float32 in [0, 1]).
        faces: detected faces (``FaceData``: normalized landmarks in this
            image's frame).
        face_skin: (H, W) accumulated face skin mask, 0-1 or 0-255.
        strength: 0-100; 0 returns ``img`` unchanged.
        reference: the same frame before face edits (same shape/range as
            ``img``). Without it the edits' own change can't be measured and
            only the foundation colour pull acts.
        person_mask: optional (H, W) person mask; the zone is limited to the
            component holding each face.
        exclude_mask: optional (H, W) mask of pixels never to touch (hair).

    Returns:
        Image of the same dtype and range as ``img``.
    """
    s = float(strength) / 100.0
    if s <= 0 or not faces:
        return img
    plans = _plans(img, faces, face_skin, reference, person_mask, exclude_mask)
    if not plans:
        return img

    H, W = img.shape[:2]
    log_gain = np.zeros((H, W), np.float32)
    da = np.zeros((H, W), np.float32)
    db = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    y0, x0, y1, x1 = H, W, 0, 0
    for p in plans:
        sl, w_i = p["slice"], p["weight"]
        log_gain[sl] += w_i * np.float32(p["log_gain"])
        da[sl] += w_i * np.float32(p["shift_ab"][0])
        db[sl] += w_i * np.float32(p["shift_ab"][1])
        wsum[sl] += w_i
        y0, y1 = min(y0, sl[0].start), max(y1, sl[0].stop)
        x0, x1 = min(x0, sl[1].start), max(x1, sl[1].stop)
    # Two faces' zones can overlap: average their corrections there.
    over = wsum > 1.0
    if over.any():
        log_gain[over] /= wsum[over]
        da[over] /= wsum[over]
        db[over] /= wsum[over]

    sl = (slice(y0, y1), slice(x0, x1))
    cur = _to_float01(img)
    crop = np.ascontiguousarray(cur[sl])
    ws = wsum[sl]
    # One linear-light gain: every shadow stays the same fraction darker
    # than the skin around it, so the shadow under the chin is kept.
    gain = np.exp(log_gain[sl] * np.float32(s))[..., None]
    out = _clip01(_linear_to_srgb(_srgb_to_linear(crop) * gain))
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2Lab)
    # The colour offset fades in shadow, where skin chroma is lower too.
    sel = ws > 0.5
    lref = float(np.median(lab[..., 0][sel])) if sel.any() else 50.0
    shade = np.clip(lab[..., 0] / max(lref, 1.0), 0.0, 1.0)
    lab[..., 1] += da[sl] * np.float32(s) * shade
    lab[..., 2] += db[sl] * np.float32(s) * shade
    out = _clip01(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR))
    # No authority outside the weighted support (also undoes Lab round-trip
    # drift there).
    out = np.where((ws > 1e-3)[..., None], out, crop)

    result = cur.copy()
    result[sl] = out
    if img.dtype == np.uint8:
        return cv2.convertScaleAbs(result, alpha=255.0)
    return result
