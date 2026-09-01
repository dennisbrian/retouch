"""Per-op face-retouch visual quality audit — isolation renders over the DSCF corpus.

Study doc: docs/plans/RESEARCH_FACEOP_VISUAL_AUDIT_2026_08_31.md

Method: for each image, render a baseline (``natural`` recipe with every
audited face op explicitly zeroed), then one render per op with ONLY that op
enabled at a moderate-strong test value. The absdiff between the two is
attributable to the op alone. Per render we record:

  * mae_full          — mean |diff| over the whole frame
  * mae_in / mae_out  — mean |diff| inside vs outside the op's expected
                        region mask (leak_ratio = mae_out / mae_in);
                        high leak = op paints outside its region
  * eye_l / eye_r     — mean |diff| inside each eye mask (handedness /
                        cross-eye coupling detector for eye-class ops)
  * inert flag        — mae_in ~ 0 where the op should visibly act

and save a contact sheet (face crop + top diff hotspots, source|op|heat)
for human visual audit — metrics rank, eyes judge (delta-blindness rule).

Baseline is rendered twice (without and with cached face_contexts); op
renders reuse the cached contexts and diff against the cached-context
baseline so the comparison is apples-to-apples. The no-cache vs cache
baseline MAE is recorded as a determinism/cache-equivalence check.

    .venv/bin/python scripts/qa/faceop_visual_audit.py
    .venv/bin/python scripts/qa/faceop_visual_audit.py --images DSCF4463 --ops smooth,catchlight
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RETOUCH_GPU", "0")
os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.engine import RetouchEngine  # noqa: E402
from retouch.parsing import (  # noqa: E402
    FACE_OVAL, LEFT_EYE, RIGHT_EYE, LIPS_OUTER, LIPS_INNER,
    LEFT_UNDER_EYE, RIGHT_UNDER_EYE,
)

OUT_DIR = ROOT / "test_output" / "faceop_visual_audit"
SHEET_DIR = OUT_DIR / "sheets"
MAX_DIM = 2048

IMAGES = [
    "DSCF4463", "DSCF4503", "DSCF4550",  # verified real-face anchors
    "DSCF4552",                          # narrow-eye wig subject
    "DSCF4560",                          # heavy squint (partial)
    "DSCF4576",                          # closed eyes — gate must suppress eye ops
    "DSCF6961",                          # open-eye anchor, different shoot
    "DSCF7204",                          # heavy lashes, open
    "DSCF8007",                          # different shoot (visual_qa default asset)
]

# op name -> (engine kwarg value, region class)
OPS = {
    # skin-wide
    "smooth": (70, "skin"),
    "equalize": (60, "skin"),
    "whiten": (50, "skin"),
    "redness_even": (60, "skin"),
    "skin_sss": (60, "skin"),
    "shine_removal": (60, "skin"),
    "wrinkle_soften": (60, "skin"),
    "sculpt": (50, "skin"),
    "face_exposure": (40, "skin"),
    "dodge_burn": (50, "skin"),
    # eyes
    "eye_enhance": (60, "eyes"),
    "catchlight": (60, "eyes"),
    "eye_sclera_brighten": (60, "eyes"),
    "eye_sclera_vessel_remove": (60, "eyes"),
    "eye_iris_saturate": (60, "eyes"),
    "eye_iris_brightness": (50, "eyes"),
    # under-eye
    "dark_circles": (60, "undereye"),
    "undereye_darken_removal": (60, "undereye"),
    "undereye_puffiness_reduction": (60, "undereye"),
    # mouth
    "teeth_whiten": (60, "mouth"),
    "lip_enhance": (60, "mouth"),
    # cheeks
    "blush": (50, "cheeks"),
    # geometry — leak metric is informational only (warping legitimately
    # moves pixels near the face boundary)
    "slimming": (50, "geometry"),
    "reshape_eye_size": (35, "geometry"),
}

BASE_OFF = {name: 0 for name in OPS}


def shrink(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    scale = MAX_DIM / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def to_u8(res) -> np.ndarray:
    arr = np.asarray(res)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0 or not mask.any():
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask, k)


def _hull_mask(landmarks, indices, w, h) -> np.ndarray:
    """Filled convex hull of a landmark index cluster, full-image coords.

    ``face_data.landmarks`` are normalized to the full image (remapped at the
    detection boundary), so this places the mask without needing the per-face
    ROI origin (FaceRegions masks are ROI-local — unusable here directly).
    """
    out = np.zeros((h, w), np.uint8)
    lms = landmarks.landmark if hasattr(landmarks, "landmark") else landmarks
    pts = np.array([[int(lms[i].x * w), int(lms[i].y * h)] for i in indices], np.int32)
    if len(pts) >= 3:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out


def region_masks(contexts, shape):
    """Per-class expected-region masks (union over faces) from landmark
    clusters, plus per-eye masks for the handedness metric.

    Note: the skin class is the dilated FACE_OVAL hull — BiSeNet-driven skin
    ops legitimately also edit the neck, which this mask does not cover, so
    skin-op leak ratios over-report and are ranking heuristics only (the
    sheets are the evidence)."""
    h, w = shape[:2]
    classes = {c: np.zeros((h, w), np.uint8)
               for c in ("skin", "eyes", "undereye", "mouth", "cheeks", "geometry")}
    eye_l = np.zeros((h, w), np.uint8)
    eye_r = np.zeros((h, w), np.uint8)
    for fc in contexts or []:
        lms = fc.face_data.landmarks
        ied = max(int(fc.face_data.ied), 8)
        oval = _hull_mask(lms, FACE_OVAL, w, h)
        classes["skin"] |= _dilate(oval, int(ied * 0.9))
        le = _dilate(_hull_mask(lms, LEFT_EYE, w, h), int(ied * 0.35))
        re = _dilate(_hull_mask(lms, RIGHT_EYE, w, h), int(ied * 0.35))
        eye_l |= le
        eye_r |= re
        classes["eyes"] |= le | re
        ue = _hull_mask(lms, LEFT_UNDER_EYE, w, h) | _hull_mask(lms, RIGHT_UNDER_EYE, w, h)
        classes["undereye"] |= _dilate(ue, int(ied * 0.3))
        mouth = _hull_mask(lms, LIPS_OUTER, w, h) | _hull_mask(lms, LIPS_INNER, w, h)
        classes["mouth"] |= _dilate(mouth, int(ied * 0.3))
        classes["cheeks"] |= _dilate(oval, int(ied * 0.2))
        x, y, bw, bh = fc.face_data.bbox
        pad_w, pad_h = int(bw * 0.4), int(bh * 0.4)
        x0, y0 = max(x - pad_w, 0), max(y - pad_h, 0)
        x1, y1 = min(x + bw + pad_w, w), min(y + bh + pad_h, h)
        classes["geometry"][y0:y1, x0:x1] = 1
    return classes, eye_l, eye_r


def diff_heat(diff_gray: np.ndarray, gain: float = 8.0) -> np.ndarray:
    d = np.clip(diff_gray.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_INFERNO)


def crop_panel(base, op_img, diff_gray, cx, cy, half, height) -> np.ndarray:
    h, w = base.shape[:2]
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    tiles = [base[y0:y1, x0:x1], op_img[y0:y1, x0:x1],
             diff_heat(diff_gray[y0:y1, x0:x1])]
    tiles = [cv2.resize(t, (int(t.shape[1] * height / max(t.shape[0], 1)), height),
                        interpolation=cv2.INTER_AREA) for t in tiles]
    sep = np.full((height, 3, 3), 220, np.uint8)
    out = []
    for i, t in enumerate(tiles):
        if i:
            out.append(sep)
        out.append(t)
    return np.hstack(out)


def build_sheet(base, op_img, diff_gray, contexts, path: Path) -> None:
    rows = []
    # face-crop row per face (largest 2)
    faces = sorted(contexts or [], key=lambda f: -(f.face_data.bbox[2] * f.face_data.bbox[3]))[:2]
    for fc in faces:
        x, y, bw, bh = fc.face_data.bbox
        cx, cy = x + bw // 2, y + bh // 2
        rows.append(crop_panel(base, op_img, diff_gray, cx, cy, int(max(bw, bh) * 0.75), 360))
    # top-2 diff hotspots (suppress a neighborhood after each pick)
    dg = cv2.GaussianBlur(diff_gray.astype(np.float32), (0, 0), 9)
    for _ in range(2):
        if dg.max() <= 0.5:
            break
        cy, cx = np.unravel_index(int(np.argmax(dg)), dg.shape)
        rows.append(crop_panel(base, op_img, diff_gray, int(cx), int(cy), 160, 240))
        y0, y1 = max(cy - 200, 0), min(cy + 200, dg.shape[0])
        x0, x1 = max(cx - 200, 0), min(cx + 200, dg.shape[1])
        dg[y0:y1, x0:x1] = 0
    if not rows:
        return
    width = max(r.shape[1] for r in rows)
    padded = []
    for i, r in enumerate(rows):
        if r.shape[1] < width:
            r = cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=(30, 30, 30))
        if i:
            padded.append(np.full((4, width, 3), 220, np.uint8))
        padded.append(r)
    cv2.imwrite(str(path), np.vstack(padded), [cv2.IMWRITE_JPEG_QUALITY, 88])


def mean_abs(diff_gray: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is None:
        return float(diff_gray.mean())
    n = int(mask.sum())
    if n == 0:
        return 0.0
    return float(diff_gray[mask.astype(bool)].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=None, help="comma-separated stems (default: study set)")
    ap.add_argument("--ops", default=None, help="comma-separated op names (default: all)")
    args = ap.parse_args()

    stems = args.images.split(",") if args.images else IMAGES
    ops = {k: OPS[k] for k in (args.ops.split(",") if args.ops else OPS)}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    engine = RetouchEngine()
    records = []
    t_start = time.time()
    try:
        for stem in stems:
            src = ROOT / "test_output" / f"{stem}.jpg"
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                print(f"SKIP {stem}: unreadable {src}")
                continue
            img = shrink(img)
            t0 = time.time()
            res1 = engine.process(img, recipe="natural", **BASE_OFF)
            contexts = res1.face_contexts
            base1 = to_u8(res1)
            res2 = engine.process(img, recipe="natural", face_contexts=contexts, **BASE_OFF)
            base = to_u8(res2)
            cache_mae = float(np.mean(np.abs(base.astype(np.float32) - base1.astype(np.float32))))
            input_mae = float(np.mean(np.abs(base.astype(np.float32) - img.astype(np.float32))))
            n_faces = len(contexts or [])
            classes, eye_l, eye_r = region_masks(contexts, img.shape)
            print(f"[{stem}] baseline x2 in {time.time()-t0:.1f}s  faces={n_faces} "
                  f"cache_mae={cache_mae:.4f} baseline_vs_input_mae={input_mae:.2f}", flush=True)
            records.append({"image": stem, "op": "_baseline", "faces": n_faces,
                            "cache_mae": cache_mae, "baseline_vs_input_mae": input_mae})

            for op, (val, cls) in ops.items():
                t1 = time.time()
                kw = dict(BASE_OFF)
                kw[op] = val
                out = to_u8(engine.process(img, recipe="natural", face_contexts=contexts, **kw))
                diff = np.abs(out.astype(np.int16) - base.astype(np.int16)).max(axis=2).astype(np.uint8)
                mask = classes[cls]
                rec = {
                    "image": stem, "op": op, "value": val, "region": cls,
                    "faces": n_faces,
                    "mae_full": round(float(diff.mean()), 4),
                    "mae_in": round(mean_abs(diff, mask), 4),
                    "mae_out": round(mean_abs(diff, 1 - mask), 4),
                    "eye_l": round(mean_abs(diff, eye_l), 4),
                    "eye_r": round(mean_abs(diff, eye_r), 4),
                    "secs": round(time.time() - t1, 2),
                }
                rec["leak_ratio"] = round(rec["mae_out"] / rec["mae_in"], 4) if rec["mae_in"] > 1e-6 else None
                records.append(rec)
                build_sheet(base, out, diff, contexts, SHEET_DIR / f"{stem}_{op}.jpg")
                print(f"  {op:30s} mae_in={rec['mae_in']:7.3f} mae_out={rec['mae_out']:7.3f} "
                      f"leak={rec['leak_ratio']} eyeL={rec['eye_l']:6.3f} eyeR={rec['eye_r']:6.3f} "
                      f"({rec['secs']:.1f}s)", flush=True)
    finally:
        close = getattr(engine, "close", None)
        if callable(close):
            close()

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump({"meta": {"max_dim": MAX_DIM, "recipe_base": "natural+all-audited-ops-off",
                            "date": "2026-08-31"},
                   "records": records}, f, indent=1)
    print(f"DONE {len(records)} records in {time.time()-t_start:.0f}s -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
