"""Glasses, goggle and visor glare removal (classical, no model weights).

A reflection on a lens is *additive*: the camera sees the scene behind the
lens plus a mirror image of the light source (flash, softbox, window). Most
such reflections are smooth and low-frequency, so removing them is a matter of
estimating that additive veil and subtracting it, which leaves the detail under
the glass (lashes, iris, skin texture, frame edges) in place.

Per face, on a crop around the eyes:

1. **Lens support.** One rotated ellipse per eye, sized from the inter-eye
   distance (IED) the way real frames are (lens about 0.85 IED wide). The eye
   opening and the eyebrows are excluded from every *estimate* (the sclera is
   bright and would read as glare; brows are dark and would pull the reference
   down).
2. **Local reference.** The non-glare level is a normalized convolution of L
   over the known, non-flagged pixels (two passes: the first pass flags obvious
   glare, the second re-estimates without it). This is a *margin above the
   face's own local level*, never an absolute brightness threshold, so it
   behaves the same on every skin tone (see CLAUDE.md, Tone-Invariance).
3. **Veil.** ``excess = lowpass(L) - reference``; a smoothstep between two
   margins turns it into a gate. The veil is ``gate * excess`` inside the lens
   support. Inside the eye opening the low-pass is interpolated from the
   surround, so a glare patch that crosses the eye is lifted off the eye too,
   while the eye's own catchlights and sclera (which have no glare around them)
   are left alone.
4. **Colour.** Additive white or coloured (anti-reflective coating) glare
   washes out or tints the skin under it, so a/b are pulled toward the local
   non-glare chroma by the same gate (outside the eye opening only, so iris
   colour is never replaced by skin colour).
5. **Blown cores.** Where the glare clipped the sensor there is nothing to
   recover; those pixels are refilled with the local non-glare level and colour
   plus grain matched to the surrounding skin so they don't read as flat plastic.

Known limits: a reflection that sits entirely inside the eye opening looks the
same as a catchlight and is kept; glare outside the eye area (the lower half of
a full-face visor) is not touched; very large blown areas can only be filled,
not recovered.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .parsing import LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW
from .utils import bgr_f32_to_lab_f32, lab_f32_to_bgr_f32, inter_eye_distance

__all__ = ["remove_lens_glare", "remove_lens_glare_faces", "lens_support"]

# Geometry, as fractions of the inter-eye distance (IED).
_LENS_HALF_W = 0.43        # lens half-width (a 50 mm lens on a 63 mm PD)
_LENS_HALF_H = 0.32        # lens half-height
_LENS_DROP = 0.04          # lens centre sits slightly below the pupil
_LENS_FEATHER = 0.04       # support feather sigma
_ROI_MARGIN = 0.30         # crop margin around the two lenses
_EYE_DILATE = 0.035        # eye-opening exclusion margin (lashes, liner)
_BROW_DILATE = 0.03        # eyebrow exclusion margin
_LOWPASS_SIGMA = 0.012     # veil is estimated on this low-pass of L
_REF_SIGMA = 0.12          # chroma reference sigma
_CLOSE_R = 0.05            # closing radius: fills frames/lashes thinner than this
_OPEN_R = 0.13             # opening radius: glare narrower than ~2x this is removed

# Margins in L units (0-255 LAB scale) above the local reference.
_GATE_LO = 9.0
_GATE_HI = 24.0
_CHROMA_PULL = 0.85        # fraction of the a/b gap closed at full gate
_CLIP_LEVEL = 250.0        # max channel at or above this = blown (no data)


def _smoothstep(x: np.ndarray, e0: float, e1: float) -> np.ndarray:
    t = np.clip((x - e0) / max(e1 - e0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _norm_conv(values: np.ndarray, weight: np.ndarray, sigma: float) -> np.ndarray:
    """Weighted Gaussian average: blur(v*w)/blur(w), falling back to v."""
    num = cv2.GaussianBlur(values * weight, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), values).astype(np.float32)


def _points(landmarks: Any, idx: Sequence[int], w: int, h: int) -> np.ndarray:
    lm = landmarks.landmark
    return np.array([[lm[i].x * w, lm[i].y * h] for i in idx], dtype=np.float32)


def lens_support(
    landmarks: Any, w: int, h: int, ied: Optional[float] = None,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float], float]], float]:
    """Lens ellipses ``((cx, cy), (full_w, full_h), angle_deg)`` for both eyes.

    Returns the two ellipses (image pixel frame) and the IED used.
    """
    if ied is None or ied <= 0:
        ied = inter_eye_distance(landmarks, w, h)
    left = _points(landmarks, LEFT_EYE, w, h).mean(axis=0)
    right = _points(landmarks, RIGHT_EYE, w, h).mean(axis=0)
    dx, dy = right - left
    angle = float(np.degrees(np.arctan2(dy, dx)))
    # Unit "down" vector perpendicular to the eye line.
    norm = float(np.hypot(dx, dy)) or 1.0
    down = np.array([-dy / norm, dx / norm], dtype=np.float32)
    if down[1] < 0:
        down = -down
    ellipses = []
    for c in (left, right):
        cc = c + down * (_LENS_DROP * ied)
        ellipses.append(((float(cc[0]), float(cc[1])),
                         (2.0 * _LENS_HALF_W * ied, 2.0 * _LENS_HALF_H * ied),
                         angle))
    return ellipses, float(ied)


def remove_lens_glare(
    img_bgr: np.ndarray,
    landmarks: Any,
    strength: float,
    ied: Optional[float] = None,
    seed: int = 0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Remove lens glare over one face's eyes.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
        landmarks: MediaPipe-style normalized landmark list (``.landmark[i].x``)
            in the frame of ``img_bgr``.
        strength: 0-100. 0 returns the input unchanged. 100 subtracts the whole
            estimated veil.
        ied: Inter-eye distance in pixels (computed from landmarks if omitted).
        seed: RNG seed for the grain put back into blown cores.

    Returns:
        (image, diagnostics). The image has the input's dtype and shape.
        Diagnostics: ``glare_fraction`` (share of the lens area gated as
        glare), ``blown_fraction`` and ``mean_lift`` (mean L removed over the
        glare pixels, 0-255 scale).
    """
    diag = {"glare_fraction": 0.0, "blown_fraction": 0.0, "mean_lift": 0.0}
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if s <= 0.0 or landmarks is None:
        return img_bgr, diag

    H, W = img_bgr.shape[:2]
    ellipses, ied = lens_support(landmarks, W, H, ied)
    if ied < 8.0:
        return img_bgr, diag

    # Crop around both lenses.
    pts = []
    for (cx, cy), (ew, eh), _ in ellipses:
        r = 0.5 * max(ew, eh)
        pts += [(cx - r, cy - r), (cx + r, cy + r)]
    pts = np.array(pts)
    m = _ROI_MARGIN * ied
    x0 = int(max(0, np.floor(pts[:, 0].min() - m)))
    y0 = int(max(0, np.floor(pts[:, 1].min() - m)))
    x1 = int(min(W, np.ceil(pts[:, 0].max() + m)))
    y1 = int(min(H, np.ceil(pts[:, 1].max() + m)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return img_bgr, diag

    is_float = img_bgr.dtype == np.float32
    roi = img_bgr[y0:y1, x0:x1].astype(np.float32)
    rh, rw = roi.shape[:2]
    lab = bgr_f32_to_lab_f32(roi)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

    # --- Masks (ROI frame) -------------------------------------------------
    support = np.zeros((rh, rw), np.float32)
    for (cx, cy), axes, ang in ellipses:
        cv2.ellipse(support, ((cx - x0, cy - y0), axes, ang), 1.0, -1)
    hard_support = support > 0.5
    support = cv2.GaussianBlur(support, (0, 0), max(1.0, _LENS_FEATHER * ied))

    def _poly_mask(indices: Sequence[int], dilate: float) -> np.ndarray:
        mk = np.zeros((rh, rw), np.uint8)
        p = _points(landmarks, indices, W, H) - np.array([x0, y0], np.float32)
        cv2.fillPoly(mk, [np.round(p).astype(np.int32)], 1)
        r = max(1, int(round(dilate * ied)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        return cv2.dilate(mk, k).astype(np.float32)

    eye = np.maximum(_poly_mask(LEFT_EYE, _EYE_DILATE), _poly_mask(RIGHT_EYE, _EYE_DILATE))
    brow = np.maximum(_poly_mask(LEFT_EYEBROW, _BROW_DILATE),
                      _poly_mask(RIGHT_EYEBROW, _BROW_DILATE))
    known = (1.0 - eye) * (1.0 - brow)

    # --- Veil estimate -----------------------------------------------------
    lp = max(1.0, _LOWPASS_SIGMA * ied)
    ref_sigma = max(2.0, _REF_SIGMA * ied)
    L_low = _norm_conv(L, known, lp)
    # Inside the eye opening the low-pass is the surround's, spread in.
    L_low = np.where(eye > 0.5, _norm_conv(L_low, 1.0 - eye, 2.5 * lp), L_low)

    # Reference = the local non-glare level. Close first (fills thin dark
    # structures such as frames and lashes with the surrounding level), then
    # open (removes bright blobs narrower than the kernel, i.e. the glare).
    # Opening never raises a pixel, so wide bright skin keeps its own level.
    rc = max(1, int(round(_CLOSE_R * ied)))
    ro = max(2, int(round(_OPEN_R * ied)))
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rc + 1, 2 * rc + 1))
    ko = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ro + 1, 2 * ro + 1))
    closed = cv2.morphologyEx(L_low, cv2.MORPH_CLOSE, kc)
    ref = cv2.morphologyEx(closed, cv2.MORPH_OPEN, ko)
    ref = cv2.GaussianBlur(ref, (0, 0), max(1.0, 0.35 * ro))
    # Measured on the closed signal: a reflection is additive, so it lifts a
    # thin dark frame or lash line by the same amount as the skin around it,
    # and the closing carries that surrounding level across them.
    excess = closed - ref
    dark = (L_low < closed - _GATE_LO).astype(np.float32)
    flag = (excess > _GATE_LO).astype(np.float32)
    gate = _smoothstep(excess, _GATE_LO, _GATE_HI) * support
    veil = gate * np.maximum(excess, 0.0)
    if float(veil.max()) < 1.0:
        return img_bgr, diag

    L_new = L - s * veil

    # Chroma back toward the local non-glare chroma (not inside the eye).
    w_ref = known * (1.0 - flag) * (1.0 - dark)
    a_ref = _norm_conv(A, w_ref, ref_sigma)
    b_ref = _norm_conv(B, w_ref, ref_sigma)
    pull = s * _CHROMA_PULL * gate * (1.0 - eye)
    A_new = A + pull * (a_ref - A)
    B_new = B + pull * (b_ref - B)

    out_lab = np.stack([L_new, A_new, B_new], axis=-1)
    out = lab_f32_to_bgr_f32(out_lab)

    # --- Blown cores: refill + matched grain ------------------------------
    blown = ((roi.max(axis=2) >= _CLIP_LEVEL) & (gate > 0.5) & (eye < 0.5)).astype(np.uint8)
    blown_px = int(blown.sum())
    if blown_px > 0:
        r = max(1, int(round(0.02 * ied)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        core = cv2.dilate(blown, k)
        # Grain: robust std of the high-pass of L on nearby non-glare skin.
        hp = L - cv2.GaussianBlur(L, (0, 0), max(1.0, 0.01 * ied))
        ring = (hard_support & (flag < 0.5) & (known > 0.5))
        sigma_n = 0.0
        if int(ring.sum()) > 50:
            med = np.median(hp[ring])
            sigma_n = float(1.4826 * np.median(np.abs(hp[ring] - med)))
        # Fill with the local non-glare level and colour (not an inpaint from
        # the core's edge, which would drag a dark frame into the fill).
        L_fill = ref.copy()
        if sigma_n > 0.0:
            rng = np.random.default_rng(seed)
            L_fill = L_fill + rng.standard_normal((rh, rw)).astype(np.float32) * sigma_n
        filled = lab_f32_to_bgr_f32(np.stack([L_fill, a_ref, b_ref], axis=-1))
        cmask = cv2.GaussianBlur(core.astype(np.float32), (0, 0), max(1.0, 0.5 * r))
        cmask = (cmask * s)[..., None]
        out = out * (1.0 - cmask) + filled * cmask

    # Composite only where the veil acted (keep everything else bit-exact).
    act = np.clip(gate * 4.0, 0.0, 1.0)
    act = cv2.GaussianBlur(act, (0, 0), max(1.0, lp))[..., None]
    out = roi * (1.0 - act) + out * act
    out = np.clip(out, 0.0, 255.0)

    lens_area = float(hard_support.sum()) or 1.0
    gl = gate > 0.5
    diag["glare_fraction"] = float(gl.sum()) / lens_area
    diag["blown_fraction"] = blown_px / lens_area
    diag["mean_lift"] = float((s * veil)[gl].mean()) if gl.any() else 0.0

    result = img_bgr.copy()
    if is_float:
        result[y0:y1, x0:x1] = out.astype(np.float32)
    else:
        result[y0:y1, x0:x1] = np.round(out).astype(img_bgr.dtype)
    return result, diag


def remove_lens_glare_faces(
    img_bgr: np.ndarray,
    faces: Sequence[Any],
    strength: float,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Apply :func:`remove_lens_glare` to every detected face.

    ``faces`` are ``FaceData``-like objects with ``.landmarks`` normalized to
    ``img_bgr`` and an optional ``.ied``.
    """
    diags: List[Dict[str, float]] = []
    if strength <= 0 or not faces:
        return img_bgr, diags
    out = img_bgr
    for i, f in enumerate(faces):
        lms = getattr(f, "landmarks", None)
        if lms is None:
            diags.append({"glare_fraction": 0.0, "blown_fraction": 0.0, "mean_lift": 0.0})
            continue
        out, d = remove_lens_glare(out, lms, strength, ied=None, seed=i)
        diags.append(d)
    return out, diags
