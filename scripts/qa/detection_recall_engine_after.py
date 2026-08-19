"""Post-change engine proof: DSCF4598 subject now gets face work.

Repeats the recipe-harm measurement (cosplay strength) on DSCF4598 with the
dual-scale detect change live. Before: subject box mean delta 1.60 (global
grades only, no face pipeline). After: should reach the detected-subject
class (~4+).

    .venv/bin/python scripts/qa/detection_recall_engine_after.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("RETOUCH_GPU", "0")
os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.engine import RetouchEngine  # noqa: E402

SUBJECT_BOX = (1226, 1480, 780, 920)  # RF-confirmed subject, native coords


def main():
    eng = RetouchEngine()
    try:
        img = cv2.imread(str(ROOT / "test_output/DSCF4598.jpg"))
        h, w = img.shape[:2]
        s = min(1.0, 2048.0 / max(h, w))
        small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        result = eng.process(small, recipe="cosplay")
        out = np.asarray(result)
        faces = getattr(result, "face_count", "?")

        x, y, bw, bh = [int(v * s) for v in SUBJECT_BOX]
        dx, dy = int(bw * 0.15), int(bh * 0.15)
        d = np.abs(out[y + dy:y + bh - dy, x + dx:x + bw - dx].astype(np.float32)
                   - small[y + dy:y + bh - dy, x + dx:x + bw - dx].astype(np.float32))
        frame_d = np.abs(out.astype(np.float32) - small.astype(np.float32))
        print(f"DSCF4598 [cosplay] face_count={faces}")
        print(f"  subject box mean|Δ| = {d.mean():.2f}  p99 = {np.percentile(d, 99):.1f}")
        print(f"  (before change: 1.60 / 6.0 — global-only, no face pipeline)")
        print(f"  (detected-subject class reference: 4.31 / 42 — DSCF4588)")
        print(f"  frame mean|Δ| = {frame_d.mean():.2f}")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
