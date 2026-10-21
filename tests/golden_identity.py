"""Per-face identity scorecard for the golden QA harness.

Used by scripts/qa/golden_qa.py (one row per case, recipe and face) and
tests/test_golden_identity.py. It answers "does the person still look like
themselves?" with four measures; everything is relative to each face:

- geometry drift: MediaPipe landmarks re-detected on the output, feature
  distances (eye spacing, nose width, jaw, lip shape, ...) divided by the
  face's own landmark spread and compared with the input. The tolerance per
  feature is the face's own landmark jitter (the same input re-detected
  after a 2 px shift and a JPEG round trip) times ``GEOMETRY_JITTER_K``,
  floored at ``GEOMETRY_FLOOR`` of the feature's length.
- skin lightness shift: face-skin median L* of output vs input, converted
  to relative luminance and reported in stops (a ratio, so the same edit
  reads the same on darker and lighter skin), next to
  ``qa_detectors.detect_color_drift``'s ΔE and chroma-floored hue shift.
- mark survival: moles and freckles planted on the input in linear light
  (a transmittance, so they read the same on every skin tone) and measured
  on the render as their low-pass Lab residual against the unplanted
  render, as a share of the residual they had in the input.
- AA6: ``qa_detectors.perceived_retouching_vector`` (face SSIM contrast-
  structure term and signed high-pass change D) on this face's skin, plus
  the leak checks and the count of skipped faces from golden_qa.py.

No new model: the landmarks come from the engine's own FaceDetector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: (landmark a, landmark b) per feature distance, MediaPipe FaceMesh indices.
FEATURES: Dict[str, Tuple[int, int]] = {
    "eye_spacing": (33, 263),
    "inner_eye_spacing": (133, 362),
    "left_eye_width": (33, 133),
    "right_eye_width": (362, 263),
    "nose_width": (129, 358),
    "nose_length": (168, 2),
    "mouth_width": (61, 291),
    "upper_lip": (0, 13),
    "lower_lip": (14, 17),
    "jaw_width": (172, 397),
    "cheek_width": (234, 454),
    "chin_height": (17, 152),
    "face_height": (10, 152),
}
#: Tolerance per feature = max(floor, K x the face's own landmark jitter).
#: Drift is the change of a feature distance in units of the face's landmark
#: spread (RMS distance of the 468 mesh points from their centroid), so a
#: short feature (lip height) is not judged by its own tiny length.
GEOMETRY_JITTER_K = 3.0
GEOMETRY_FLOOR = 0.04
#: Faces narrower than this many inter-eye distances' worth of pixels give
#: landmarks too coarse to compare (size gate, not a tone gate).
GEOMETRY_MIN_IED_PX = 30.0
#: |lightness shift| above this many stops counts as lightened / darkened.
LIGHTNESS_MAX_STOPS = 0.15
#: A planted mark is "kept" while it keeps at least this share of its residual.
MARK_KEPT_MIN = 0.5

# Planted marks: BGR transmittance at the centre (linear light) and radius
# as a share of face width. Moles are sharp-edged and dark brown; freckles
# soft and light brown.
MOLE_TRANS = np.array([0.30, 0.32, 0.40])
FRECKLE_TRANS = np.array([0.70, 0.74, 0.86])
MOLE_R_FW = 0.016
FRECKLE_R_FW = 0.010
MARKS_PER_KIND = 4


# ---------------------------------------------------------------- helpers

def _srgb_to_lin(x):
    x = np.asarray(x, np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(y):
    y = np.clip(y, 0.0, 1.0)
    return 255.0 * np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055)


def _lab(img_u8: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(np.asarray(img_u8, np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)


def _l_to_y(l_star: float) -> float:
    """CIE L* -> relative luminance Y (0-1)."""
    f = (l_star + 16.0) / 116.0
    return f ** 3 if f ** 3 > 0.008856 else l_star / 903.3


def face_region(ctx, shape, name: str) -> np.ndarray:
    """One face's FaceRegions mask in frame coordinates (float 0-1)."""
    from retouch.engine import RetouchEngine

    out = np.zeros(shape, np.float32)
    m = getattr(ctx.regions, name, None)
    if m is None:
        return out
    x, y, w, h = (int(v) for v in ctx.face_data.bbox)
    pt, pb, pl, pr = RetouchEngine._compute_face_roi_padding(w, h)
    y1, y2 = max(0, y - pt), min(shape[0], y + h + pb)
    x1, x2 = max(0, x - pl), min(shape[1], x + w + pr)
    m = np.asarray(m, np.float32)
    if m.max() > 1.5:
        m = m / 255.0
    if m.shape != (y2 - y1, x2 - x1):
        m = cv2.resize(m, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    out[y1:y2, x1:x2] = m
    return out


def face_skin(ctx, shape) -> np.ndarray:
    """This face's skin (bool), eroded 2% of face width off its edges."""
    m = face_region(ctx, shape, "skin") > 0.5
    k = max(1, int(round(0.02 * float(ctx.face_data.bbox[2]))))
    return cv2.erode(m.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0


# ---------------------------------------------------------------- geometry

def landmarks_px(face, shape) -> np.ndarray:
    h, w = shape[:2]
    lms = face.landmarks.landmark
    return np.array([[lms[i].x * w, lms[i].y * h] for i in range(min(468, len(lms)))], np.float64)


def feature_ratios(pts: np.ndarray) -> Dict[str, float]:
    """Feature distances divided by the landmark cloud's RMS spread."""
    scale = float(np.sqrt(((pts - pts.mean(0)) ** 2).sum(1).mean()))
    return {k: float(np.linalg.norm(pts[a] - pts[b])) / max(scale, 1e-6) for k, (a, b) in FEATURES.items()}


def match_face(faces, bbox) -> Optional[object]:
    """Detected face whose bbox centre is nearest ``bbox``'s (within half its width)."""
    x, y, w, h = (float(v) for v in bbox)
    cx, cy = x + w / 2, y + h / 2
    best, dist = None, 0.5 * w
    for f in faces:
        fx, fy, fw, fh = (float(v) for v in f.bbox)
        d = float(np.hypot(fx + fw / 2 - cx, fy + fh / 2 - cy))
        if d <= dist:
            best, dist = f, d
    return best


def perturbations(img: np.ndarray) -> List[np.ndarray]:
    """The same photo with no geometric change: shifted 2 px + JPEG q92, a
    slight blur and a slight sharpen (the photometric kinds of change a
    retouch makes). Re-detecting on them measures the face's landmark jitter."""
    img = np.asarray(img, np.uint8)
    h, w = img.shape[:2]
    m = np.float32([[1, 0, 2], [0, 1, 2]])
    moved = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    ok, buf = cv2.imencode(".jpg", moved, [cv2.IMWRITE_JPEG_QUALITY, 92])
    blur = cv2.GaussianBlur(img, (0, 0), 0.8)
    sharp = cv2.addWeighted(img, 1.6, blur, -0.6, 0)
    return [cv2.imdecode(buf, cv2.IMREAD_COLOR), blur, sharp]


@dataclass
class GeometryBaseline:
    """Input landmark ratios and per-feature tolerance for one face."""
    ratios: Dict[str, float]
    tolerance: Dict[str, float]


def geometry_baseline(detector, img: np.ndarray, bbox) -> Optional[GeometryBaseline]:
    """Input ratios plus a tolerance from the face's own landmark jitter.
    None when the face has no landmarks or is too small to compare."""
    img = np.asarray(img, np.uint8)
    face = match_face(detector.detect(img), bbox)
    if face is None or float(getattr(face, "ied", 0.0)) < GEOMETRY_MIN_IED_PX:
        return None
    r0 = feature_ratios(landmarks_px(face, img.shape))
    jitter = {k: 0.0 for k in r0}
    for p in perturbations(img):
        f = match_face(detector.detect(p), bbox)
        if f is None:
            continue
        r1 = feature_ratios(landmarks_px(f, img.shape))
        for k in r0:
            jitter[k] = max(jitter[k], abs(r1[k] - r0[k]))
    tol = {k: max(GEOMETRY_FLOOR, GEOMETRY_JITTER_K * jitter[k]) for k in r0}
    return GeometryBaseline(r0, tol)


def geometry_drift(detector, out: np.ndarray, bbox, base: Optional[GeometryBaseline]) -> Dict[str, object]:
    """Worst feature drift as a share of its tolerance (>1 = flagged)."""
    if base is None:
        return {"geo_drift": None, "geo_feature": None, "geo_ratio": None, "geo_flag": False,
                "geo_note": "not measured (no landmarks, or face too small)"}
    face = match_face(detector.detect(np.asarray(out, np.uint8)), bbox)
    if face is None:
        return {"geo_drift": None, "geo_feature": None, "geo_ratio": None, "geo_flag": True,
                "geo_note": "face lost on output"}
    r = feature_ratios(landmarks_px(face, out.shape))
    rel = {k: abs(r[k] - v) for k, v in base.ratios.items()}
    worst = max(rel, key=lambda k: rel[k] / base.tolerance[k])
    ratio = rel[worst] / base.tolerance[worst]
    return {"geo_drift": rel[worst], "geo_feature": worst, "geo_ratio": ratio,
            "geo_flag": bool(ratio > 1.0), "geo_note": ""}


# ---------------------------------------------------------------- tone

def lightness_shift(out: np.ndarray, inp: np.ndarray, skin: np.ndarray) -> Dict[str, object]:
    """Face-skin median L* change in stops, plus color_drift's ΔE and hue."""
    from retouch.qa_detectors import detect_color_drift

    if skin.sum() < 50:
        return {"l_stops": None, "l_flag": False, "dE": None, "dh_deg": None}
    l_in = float(np.median(_lab(inp)[..., 0][skin]))
    l_out = float(np.median(_lab(out)[..., 0][skin]))
    stops = float(np.log2(max(_l_to_y(l_out), 1e-6) / max(_l_to_y(l_in), 1e-6)))
    cd = detect_color_drift(np.asarray(out, np.uint8), skin.astype(np.float32),
                            np.asarray(inp, np.uint8))
    return {"l_stops": stops, "l_flag": bool(abs(stops) > LIGHTNESS_MAX_STOPS),
            "dE": cd.get("deltaE_mean"), "dh_deg": cd.get("deltaH_mean_chroma_floored_deg")}


# ---------------------------------------------------------------- marks

@dataclass
class Mark:
    kind: str  # "mole" | "freckle"
    x: int
    y: int
    r: float


def plant_mark(img: np.ndarray, mark: Mark) -> np.ndarray:
    """Multiply a mark into ``img`` (uint8 BGR) in linear light."""
    trans = MOLE_TRANS if mark.kind == "mole" else FRECKLE_TRANS
    lin = _srgb_to_lin(img)
    h, w = img.shape[:2]
    R = int(np.ceil(mark.r * 3)) + 1
    y1, y2, x1, x2 = max(0, mark.y - R), min(h, mark.y + R + 1), max(0, mark.x - R), min(w, mark.x + R + 1)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    d = np.hypot(xx - mark.x, yy - mark.y)
    if mark.kind == "mole":
        prof = np.clip((mark.r - d) / 1.0 + 0.5, 0, 1)
    else:
        prof = np.exp(-0.5 * (d / (0.6 * mark.r)) ** 2)
    lin[y1:y2, x1:x2] *= trans[None, None, :] ** prof[:, :, None]
    return np.clip(_lin_to_srgb(lin) + 0.5, 0, 255).astype(np.uint8)


def pick_mark_sites(ctx, shape, n: int, min_gap_fw: float = 0.12, seed: int = 0) -> List[Tuple[int, int]]:
    """Up to ``n`` spread-out skin points away from eyes, brows and lips."""
    fw = float(ctx.face_data.bbox[2])
    skin = face_region(ctx, shape, "skin") > 0.5
    k = max(1, int(round(0.06 * fw)))
    skin = cv2.erode(skin.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    feats = np.zeros(shape, np.float32)
    for name in ("left_eye", "right_eye", "left_eyebrow", "right_eyebrow", "lips",
                 "mouth_interior", "left_iris", "right_iris"):
        feats = np.maximum(feats, face_region(ctx, shape, name))
    feats = cv2.dilate((feats > 0.3).astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    ys, xs = np.nonzero(skin & ~feats)
    if len(xs) == 0:
        return []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(xs))
    sites: List[Tuple[int, int]] = []
    gap = min_gap_fw * fw
    for i in order:
        p = (int(xs[i]), int(ys[i]))
        if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= gap for q in sites):
            sites.append(p)
            if len(sites) >= n:
                break
    return sites


def plan_marks(ctxs: Sequence, shape, per_kind: int = MARKS_PER_KIND) -> List[List[Mark]]:
    """Per face: ``per_kind`` moles and ``per_kind`` freckles on its skin."""
    out = []
    for i, c in enumerate(ctxs):
        fw = float(c.face_data.bbox[2])
        sites = pick_mark_sites(c, shape, 2 * per_kind, seed=i)
        marks = []
        for j, (x, y) in enumerate(sites):
            kind = "mole" if j % 2 == 0 else "freckle"
            r = max(1.5, MOLE_R_FW * fw) if kind == "mole" else max(1.2, FRECKLE_R_FW * fw)
            marks.append(Mark(kind, x, y, r))
        out.append(marks)
    return out


def plant_all(img: np.ndarray, marks: Sequence[Sequence[Mark]]) -> np.ndarray:
    out = np.asarray(img, np.uint8).copy()
    for face_marks in marks:
        for m in face_marks:
            out = plant_mark(out, m)
    return out


def mark_residual(a: np.ndarray, b: np.ndarray, m: Mark) -> float:
    """Mean low-pass Lab distance between two images over the mark's disc."""
    h, w = a.shape[:2]
    R = int(np.ceil(m.r * 3)) + 2
    y1, y2, x1, x2 = max(0, m.y - R), min(h, m.y + R + 1), max(0, m.x - R), min(w, m.x + R + 1)
    sg = max(0.35 * m.r, 0.8)
    la = cv2.GaussianBlur(_lab(a[y1:y2, x1:x2]), (0, 0), sg)
    lb = cv2.GaussianBlur(_lab(b[y1:y2, x1:x2]), (0, 0), sg)
    yy, xx = np.mgrid[y1:y2, x1:x2]
    disc = np.hypot(xx - m.x, yy - m.y) <= max(m.r, 1.0)
    return float(np.sqrt(((la - lb) ** 2).sum(-1))[disc].mean())


def mark_survival(out_planted, out_clean, in_planted, in_clean, marks: Sequence[Mark]) -> Dict[str, object]:
    """Share of each planted mark's residual left on the render."""
    if not marks:
        return {"marks_kept": None, "marks_total": 0, "moles_kept": None, "mark_min": None,
                "mark_flag": False}
    surv = []
    for m in marks:
        base = mark_residual(in_planted, in_clean, m)
        surv.append(mark_residual(out_planted, out_clean, m) / max(base, 1e-6))
    kept = [s >= MARK_KEPT_MIN for s in surv]
    moles = [k for k, m in zip(kept, marks) if m.kind == "mole"]
    return {"marks_kept": int(sum(kept)), "marks_total": len(marks),
            "moles_kept": f"{sum(moles)}/{len(moles)}", "mark_min": float(min(surv)),
            "mark_flag": bool(not all(kept))}


# ---------------------------------------------------------------- AA6

def aa6_face(out: np.ndarray, inp: np.ndarray, skin: np.ndarray) -> Dict[str, object]:
    from retouch.qa_detectors import perceived_retouching_vector

    if skin.sum() < 50:
        return {"aa6_ssim": None, "aa6_d": None}
    v = perceived_retouching_vector(np.asarray(out, np.uint8), np.asarray(inp, np.uint8),
                                    face_mask=skin.astype(np.float32))
    return {"aa6_ssim": v["face_ssim_cs_mean"], "aa6_d": v["face_frequency_d_mean"]}


# ---------------------------------------------------------------- one call

IDENTITY_FLAGS = ("geo_flag", "l_flag", "mark_flag")


def face_scorecard(detector, inp, out, ctx, base: Optional[GeometryBaseline],
                   marks: Sequence[Mark] = (), in_planted=None, out_planted=None) -> Dict[str, object]:
    """Every identity column for one face (``out`` rendered from ``inp``)."""
    inp = np.asarray(inp, np.uint8)
    out = np.asarray(out, np.uint8)
    skin = face_skin(ctx, inp.shape[:2])
    row: Dict[str, object] = {}
    row.update(geometry_drift(detector, out, ctx.face_data.bbox, base))
    row.update(lightness_shift(out, inp, skin))
    if marks and in_planted is not None and out_planted is not None:
        row.update(mark_survival(np.asarray(out_planted, np.uint8), out, np.asarray(in_planted, np.uint8),
                                 inp, marks))
    else:
        row.update(mark_survival(None, None, None, None, ()))
    row.update(aa6_face(out, inp, skin))
    row["identity_flags"] = [f[:-5] for f in IDENTITY_FLAGS if row.get(f)]
    return row
