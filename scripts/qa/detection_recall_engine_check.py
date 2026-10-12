"""Engine-level verification: do poster FPs cause real (bad) renders?

Runs the REAL RetouchEngine (natural recipe, default quality path) on the
FP-affected images and measures pixel deltas inside the FP boxes vs the
RF-confirmed subject box:

  - DSCF4588 (10-frame series, poster FP present + subject detected)
  - DSCF4598 (subject missed entirely — poster FP is the ONLY detection)

Metric: mean |Δ| per channel inside each box vs the untouched input, plus
whole-frame delta for context. Uses quality='full' but capped resolution
(--max-dim 2048 pre-shrink, per CLAUDE.md perf guidance) — detection
behavior at the 2048 proxy is identical (S1 IS the 2048 proxy pass).

    .venv/bin/python scripts/qa/detection_recall_engine_check.py
"""

from __future__ import annotations

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

CASES = [
    # (rel, fp_boxes_native, subject_box_native)
    ("test_output/DSCF4588.jpg", [(524, 155, 758, 904)], None),
    ("test_output/DSCF4598.jpg", [(533, 164, 813, 828)], (1226, 1480, 780, 920)),
]


def mean_delta(img, out, box, shrink=0.7):
    x, y, bw, bh = box
    h, w = img.shape[:2]
    dx, dy = int(bw * (1 - shrink) / 2), int(bh * (1 - shrink) / 2)
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(w, x + bw - dx), min(h, y + bh - dy)
    if x2 <= x1 or y2 <= y1:
        return None
    d = np.abs(out[y1:y2, x1:x2].astype(np.float32) - img[y1:y2, x1:x2].astype(np.float32))
    return float(d.mean()), float(np.percentile(d, 99))


def main():
    eng = RetouchEngine()
    try:
        for rel, fp_boxes, subj in CASES:
            img = cv2.imread(str(ROOT / rel))
            h, w = img.shape[:2]
            s = min(1.0, 2048.0 / max(h, w))
            small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            t0 = time.time()
            result = eng.process(small, recipe="natural")
            dt = time.time() - t0
            out = np.asarray(result)
            faces = getattr(result, "face_count", "?")

            scaled_fps = [(int(b[0] * s), int(b[1] * s), int(b[2] * s), int(b[3] * s)) for b in fp_boxes]
            scaled_subj = None
            if subj:
                scaled_subj = (int(subj[0] * s), int(subj[1] * s), int(subj[2] * s), int(subj[3] * s))

            frame_d = np.abs(out.astype(np.float32) - small.astype(np.float32))
            print(f"\n{rel}: {w}x{h} -> {small.shape[1]}x{small.shape[0]}, "
                  f"face_count={faces}, {dt:.1f}s")
            print(f"  whole-frame mean|Δ|: {frame_d.mean():.2f}  p99: {np.percentile(frame_d, 99):.1f}")
            for i, b in enumerate(scaled_fps):
                md = mean_delta(small, out, b)
                if md:
                    print(f"  poster-FP box {i}: mean|Δ|={md[0]:.2f} p99={md[1]:.1f}  (box={b})")
            if scaled_subj:
                md = mean_delta(small, out, scaled_subj)
                if md:
                    print(f"  SUBJECT box (missed): mean|Δ|={md[0]:.2f} p99={md[1]:.1f}  (box={scaled_subj})")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
