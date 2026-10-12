"""Poster-FP veto research, parts 2+3: texture robustness sweeps.

A2 — REAL-FACE FLOOR under degradation (veto safety):
    For all 83 RF-confirmed subjects, extract the central-60% crop at the
    fixed 256px analysis geometry (same as the study's texture metric),
    then apply degradation chains that simulate the worst realistic inputs
    (heavy beauty-filter, soft-focus, low-res source, JPEG re-export,
    combinations) and re-measure Laplacian variance. The MINIMUM value
    observed across all real faces x all degradations is the empirical
    real-face floor a texture veto must not cut below.

A3 — POSTER CEILING under adversarial sharpening/noise (separation margin):
    For the 18 poster-FP boxes, apply unsharp masking and additive noise
    at increasing strength; find the cheapest push that lifts a poster
    above a given texture level. Posters are not adversaries in practice,
    but the margin determines how tight a threshold can be.

    .venv/bin/python scripts/qa/posterfp_texture_sweep.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "test_output" / "detection_recall_study"

ANALYSIS = 256  # fixed analysis geometry (matches study metric)


def analysis_crop(img, box, shrink=0.4):
    """Central region of the box at the fixed analysis geometry, grayscale float."""
    x, y, bw, bh = box[:4]
    h, w = img.shape[:2]
    dx, dy = int(bw * shrink / 2), int(bh * shrink / 2)
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(w, x + bw - dx), min(h, y + bh - dy)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    s = ANALYSIS / max(ch, cw)
    crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)


def lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def degradations():
    """Ordered worst-case-realistic chains. Names document intent."""
    def blur(gray, sigma):
        return cv2.GaussianBlur(gray, (0, 0), sigma)

    def median(gray, k):
        return cv2.medianBlur(np.clip(gray, 0, 255).astype(np.uint8), k).astype(np.float32)

    def bilateral(gray):
        u8 = np.clip(gray, 0, 255).astype(np.uint8)
        b = cv2.bilateralFilter(u8, 9, 75, 75)
        b = cv2.bilateralFilter(b, 9, 75, 75)  # 2 passes — heavy makeup smoothing
        return b.astype(np.float32)

    def jpeg(gray, q):
        u8 = np.clip(gray, 0, 255).astype(np.uint8)
        ok, buf = cv2.imencode(".jpg", u8, [cv2.IMWRITE_JPEG_QUALITY, q])
        dec = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return dec.astype(np.float32)

    def downup(gray, f):
        h, w = gray.shape
        small = cv2.resize(gray, (int(w / f), int(h / f)), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

    def chain(gray, *fns):
        for fn in fns:
            gray = fn(gray)
        return gray

    return {
        "baseline": lambda g: g,
        "blur_s0.5": lambda g: blur(g, 0.5),
        "blur_s1": lambda g: blur(g, 1.0),
        "blur_s2": lambda g: blur(g, 2.0),
        "blur_s4": lambda g: blur(g, 4.0),
        "median_k5": lambda g: median(g, 5),
        "median_k9": lambda g: median(g, 9),
        "bilateral_x2(makeup)": bilateral,
        "jpeg_q85": lambda g: jpeg(g, 85),
        "jpeg_q70": lambda g: jpeg(g, 70),
        "jpeg_q50": lambda g: jpeg(g, 50),
        "jpeg_q30": lambda g: jpeg(g, 30),
        "downup_2x": lambda g: downup(g, 2),
        "downup_4x": lambda g: downup(g, 4),
        "blur1+jpeg70": lambda g: chain(g, lambda x: blur(x, 1.0), lambda x: jpeg(x, 70)),
        "blur2+jpeg50": lambda g: chain(g, lambda x: blur(x, 2.0), lambda x: jpeg(x, 50)),
        "makeup+jpeg70": lambda g: chain(g, bilateral, lambda x: jpeg(x, 70)),
    }


def adversarial_pushes():
    def noise(gray, sigma, seed=11):
        rng = np.random.default_rng(seed)
        return np.clip(gray + rng.normal(0, sigma, gray.shape), 0, 255).astype(np.float32)

    def unsharp(gray, amount):
        blur = cv2.GaussianBlur(gray, (0, 0), 1.5)
        return np.clip(gray + amount * (gray - blur), 0, 255).astype(np.float32)

    return {
        "noise_s5": lambda g: noise(g, 5),
        "noise_s10": lambda g: noise(g, 10),
        "noise_s20": lambda g: noise(g, 20),
        "unsharp_0.5": lambda g: unsharp(g, 0.5),
        "unsharp_1.0": lambda g: unsharp(g, 1.0),
        "unsharp_2.0": lambda g: unsharp(g, 2.0),
        "noise5+unsharp1": lambda g: unsharp(noise(g, 5), 1.0),
    }


def main():
    data = json.loads((OUT_DIR / "gt_union.json").read_text())
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())

    # ---- A2: real faces under degradation ----
    degs = degradations()
    floor_rows = []
    for rel, rec in data.items():
        rf_native = rf.get(rel, {}).get("native", [])
        if not rf_native:
            continue
        subj = max(rf_native, key=lambda b: b[4])  # RF subject box (native coords)
        img = cv2.imread(str(ROOT / rel))
        gray = analysis_crop(img, subj[:4])
        if gray is None:
            continue
        row = {"rel": rel.split("/")[-1]}
        for name, fn in degs.items():
            try:
                row[name] = round(lap_var(fn(gray.copy())), 1)
            except Exception:
                row[name] = None
        floor_rows.append(row)

    print("=== A2: REAL-FACE texture floor under degradation (n=%d) ===" % len(floor_rows))
    print(f"{'degradation':22s} {'min':>8s} {'p10':>8s} {'median':>8s}")
    for name in degs:
        vals = np.array([r[name] for r in floor_rows if r[name] is not None])
        if len(vals) == 0:
            continue
        print(f"{name:22s} {vals.min():8.1f} {np.percentile(vals,10):8.1f} {np.median(vals):8.1f}")
    all_min = min(min((r[n] for n in degs if r[n] is not None), default=1e9) for r in floor_rows)
    print(f"\nGLOBAL real-face floor across all faces x all degradations: {all_min:.1f}")

    # ---- A3: posters under adversarial push ----
    push = adversarial_pushes()
    poster_rows = []
    for rel, rec in data.items():
        rf_native = rf.get(rel, {}).get("native", [])
        rf_proxy = rf.get(rel, {}).get("proxy2048", [])
        img = cv2.imread(str(ROOT / rel))
        for b in rec["strategies"]["S1_engine_proxy2048"]:
            conf = any(iou(b[:4], r[:4]) >= 0.4 for r in rf_native) or \
                   any(iou(b[:4], r[:4]) >= 0.4 for r in rf_proxy)
            if conf:
                continue
            gray = analysis_crop(img, b[:4])
            if gray is None:
                continue
            row = {"rel": rel.split("/")[-1], "box": b[:4]}
            for name, fn in push.items():
                try:
                    row[name] = round(lap_var(fn(gray.copy())), 1)
                except Exception:
                    row[name] = None
            poster_rows.append(row)

    print("\n=== A3: POSTER-FP texture ceiling under adversarial push (n=%d) ===" % len(poster_rows))
    print(f"{'push':18s} {'min':>8s} {'median':>8s} {'max':>8s}")
    for name in push:
        vals = np.array([r[name] for r in poster_rows if r[name] is not None])
        if len(vals) == 0:
            continue
        print(f"{name:18s} {vals.min():8.1f} {np.median(vals):8.1f} {vals.max():8.1f}")
    base_vals = [r["baseline"] for r in poster_rows if r.get("baseline") is not None]
    if base_vals:
        print(f"\nPoster baseline range: {min(base_vals):.1f}..{max(base_vals):.1f}")

    (OUT_DIR / "posterfp_texture_sweep.json").write_text(json.dumps({
        "real_faces": floor_rows, "posters": poster_rows,
        "real_face_global_floor": all_min,
    }, indent=1))
    print("-> posterfp_texture_sweep.json")


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


if __name__ == "__main__":
    main()
