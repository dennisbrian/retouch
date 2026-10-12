"""Post-change verification: engine-path detection over the DSCF corpus.

Runs the MODIFIED FaceDetector.detect() (dual-scale augment) at the engine's
proxy scale on all 83 corpus images and checks against the study's stored
evidence:

  1. Subject recall (RF-confirmed subject found, IoU >= 0.4) — expect 83/83
     (was 82/83; DSCF4598 is the fix target).
  2. FP accounting: every detection classified RF-confirmed vs unconfirmed;
     unconfirmed additions vs the pre-change S1 set (expect: no NEW
     unconfirmed detections beyond the pre-existing 18 — the person gate
     must not import the 1024-pass posters).
  3. Timing: per-image detect() cost at proxy scale vs pre-change (~10-30ms
     expected overhead).

    .venv/bin/python scripts/qa/detection_recall_verify_change.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RETOUCH_GPU", "0")
os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")

import cv2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.detection import FaceDetector  # noqa: E402

OUT_DIR = ROOT / "test_output" / "detection_recall_study"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def main():
    data = json.loads((OUT_DIR / "gt_union.json").read_text())
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())
    paths = [p.strip() for p in (OUT_DIR / "corpus.txt").read_text().splitlines() if p.strip()]

    det = FaceDetector(max_faces=25, min_confidence=0.4)
    subject_hits = 0
    subjects_total = 0
    new_detections = []      # (rel, box) — post-change detections absent pre-change
    unconfirmed_post = []    # all post-change RF-unconfirmed detections
    timings = []
    dsced = 0

    try:
        for rel in paths:
            rec = data[rel]
            pre = rec["strategies"]["S1_engine_proxy2048"]
            img = cv2.imread(str(ROOT / rel))
            h, w = img.shape[:2]
            s = min(1.0, 2048.0 / max(h, w))
            proxy = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

            t0 = time.time()
            faces = det.detect(proxy)
            dt = (time.time() - t0) * 1000
            timings.append(dt)

            post = [[int(v / s) for v in f.bbox[:4]] + [1.0] for f in faces]
            if len(post) > len(pre):
                dsced += 1

            rf_native = rf[rel]["native"]
            rf_proxy = rf[rel]["proxy2048"]
            if rf_native:
                subj = max(rf_native, key=lambda b: b[4])
                subjects_total += 1
                if any(iou(subj[:4], b[:4]) >= 0.4 for b in post):
                    subject_hits += 1

            for b in post:
                conf = any(iou(b[:4], r[:4]) >= 0.4 for r in rf_native) or \
                       any(iou(b[:4], r[:4]) >= 0.4 for r in rf_proxy)
                if not conf:
                    unconfirmed_post.append((rel, b))
                if not any(iou(b[:4], p[:4]) >= 0.5 for p in pre):
                    new_detections.append((rel, b, conf))
    finally:
        det.close()

    timings.sort()
    med = timings[len(timings) // 2]
    print(f"Subject recall (post-change, engine path @2048): {subject_hits}/{subjects_total} "
          f"= {subject_hits/subjects_total*100:.1f}%")
    print(f"Images whose detection count changed vs pre-change S1: {dsced}")
    print(f"Median detect() time at proxy scale: {med:.0f} ms (p90: {timings[int(len(timings)*0.9)]:.0f} ms)")
    print(f"\nNEW detections vs pre-change S1: {len(new_detections)}")
    for rel, b, conf in new_detections:
        print(f"  {rel.split('/')[-1]}: {b[:4]} rf_confirmed={conf}")
    print(f"\nRF-unconfirmed detections post-change: {len(unconfirmed_post)} (pre-change: 18)")
    for rel, b in unconfirmed_post:
        print(f"  {rel.split('/')[-1]}: {b[:4]}")

    (OUT_DIR / "verify_change_summary.json").write_text(json.dumps({
        "subject_recall": f"{subject_hits}/{subjects_total}",
        "images_changed": dsced,
        "median_detect_ms": round(med),
        "new_detections": [{"rel": r, "box": b[:4], "rf": c} for r, b, c in new_detections],
        "unconfirmed_count": len(unconfirmed_post),
    }, indent=1))
    print("-> verify_change_summary.json")


if __name__ == "__main__":
    main()
