"""Pseudo-ground-truth builder for the detection-recall study (2026-09-08).

Run with the SUPPORTED runtime (.venv, mediapipe 0.10.5 legacy backend):

    .venv/bin/python scripts/qa/detection_recall_gt.py

Protocol
--------
For each corpus image, build a UNION of independent detection strategies,
IoU-deduped, as pseudo-ground-truth:

  S1  engine-equivalent: legacy FaceMesh @ 2048 proxy (what the product runs)
  S2  legacy FaceMesh @ full native resolution
  S3  legacy FaceMesh @ 1024 downscale
  S4  legacy FaceMesh, 3x3 tiles @ 75% scale (small-face sweep)
  S5  legacy FaceMesh, 4x4 tiles @ 60% scale (aggressive small-face sweep)

Each strategy's boxes are also unioned with the engine pass (S1) so that any
face the *product* finds is by definition in the pseudo-GT — recall of S1
against the union is then a conservative lower bound of engine recall.

RetinaFace cross-check (optional, separate env — see README note in the
study doc): run on system python; results merged in analysis, not here.

Output: test_output/detection_recall_study/gt_union.json
  {path: {"size": [w, h], "strategies": {S1: [[x,y,w,h,score], ...], ...},
          "gt": [[x,y,w,h,best_score,found_by], ...]}}
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RETOUCH_GPU", "0")
os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.detection import FaceDetector  # noqa: E402

PROXY = 2048
OUT_DIR = ROOT / "test_output" / "detection_recall_study"
CORPUS = OUT_DIR / "corpus.txt"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def dedup_boxes(boxes, iou_thresh=0.5):
    """Greedy dedup, keeping the highest-score box per cluster."""
    boxes = sorted(boxes, key=lambda b: -b[4])
    kept = []
    for b in boxes:
        if all(iou(b, k) < iou_thresh for k in kept):
            kept.append(b)
    return kept


def to_proxy(img, max_dim):
    h, w = img.shape[:2]
    s = min(1.0, max_dim / max(h, w))
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)


def detect_and_rescale(det, img, max_dim):
    """Run detection on a resized copy, rescale boxes back to native coords."""
    h, w = img.shape[:2]
    small = to_proxy(img, max_dim)
    sh, sw = small.shape[:2]
    faces = det.detect(small)
    out = []
    for f in faces:
        x, y, bw, bh = f.bbox
        sx, sy = w / sw, h / sh
        out.append([int(x * sx), int(y * sy), int(bw * sx), int(bh * sy), float(f.confidence)])
    return out


def detect_tiled(det, img, tiles=3, scale=0.75):
    """Tile the image (with 25% overlap), run detection per tile, rescale boxes."""
    h, w = img.shape[:2]
    sh, sw = int(h * scale), int(w * scale)
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    out = []
    ov = 0.25
    for ty in range(tiles):
        for tx in range(tiles):
            y1 = int(ty * sh * (1 - ov))
            y2 = min(sh, int(y1 + sh / tiles * (1 + ov)))
            x1 = int(tx * sw * (1 - ov))
            x2 = min(sw, int(x1 + sw / tiles * (1 + ov)))
            tile = small[y1:y2, x1:x2]
            if tile.shape[0] < 64 or tile.shape[1] < 64:
                continue
            faces = det.detect(tile)
            for f in faces:
                x, y, bw, bh = f.bbox
                # tile coords -> small coords -> native coords
                nx = (x + x1) * (w / sw)
                ny = (y + y1) * (h / sh)
                out.append([int(nx), int(ny), int(bw * (w / sw)), int(bh * (h / sh)), float(f.confidence)])
    return out


def main():
    det = FaceDetector(max_faces=25, min_confidence=0.4)
    paths = [p.strip() for p in CORPUS.read_text().splitlines() if p.strip()]
    results = {}
    t_start = time.time()

    for i, rel in enumerate(paths):
        path = ROOT / rel
        img = cv2.imread(str(path))
        if img is None:
            print(f"[{i+1}/{len(paths)}] {rel}: LOAD FAIL", flush=True)
            continue
        h, w = img.shape[:2]
        strategies = {}

        strategies["S1_engine_proxy2048"] = detect_and_rescale(det, img, PROXY)
        strategies["S2_native"] = detect_and_rescale(det, img, max(h, w))
        strategies["S3_down1024"] = detect_and_rescale(det, img, 1024)
        strategies["S4_tiled_3x3"] = detect_tiled(det, img, tiles=3, scale=0.75)
        strategies["S5_tiled_4x4"] = detect_tiled(det, img, tiles=4, scale=0.60)

        # union across strategies, dedup at IoU 0.5
        union = []
        for name, boxes in strategies.items():
            for b in boxes:
                union.append([*b, name])
        # sort by score desc for greedy dedup, then strip strategy into found_by
        union_sorted = sorted(union, key=lambda b: -b[4])
        gt = []
        for b in union_sorted:
            core = b[:5]
            if all(iou(core, k[:4]) < 0.5 for k in gt):
                gt.append([*core, b[5]])
        results[rel] = {"size": [w, h], "strategies": strategies, "gt": gt}
        print(f"[{i+1}/{len(paths)}] {rel}: gt={len(gt)} "
              f"(S1={len(strategies['S1_engine_proxy2048'])} S2={len(strategies['S2_native'])} "
              f"S3={len(strategies['S3_down1024'])} S4={len(strategies['S4_tiled_3x3'])} S5={len(strategies['S5_tiled_4x4'])})",
              flush=True)

    det.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "gt_union.json"
    out.write_text(json.dumps(results, indent=1))
    n_gt = sum(len(r["gt"]) for r in results.values())
    print(f"\nDone: {len(results)} images, {n_gt} pseudo-GT faces, {time.time()-t_start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
