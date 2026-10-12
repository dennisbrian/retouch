"""FP harm at realistic recipe strength.

The engine check at `natural` showed mild poster-FP deltas. But the product's
use-case for convention corpora is strong recipes (cosplay / porcelain).
Measures per-face harm on the poster FP box and the missed-subject box under
recipe="cosplay" (the flagship convention recipe).

    .venv/bin/python scripts/qa/detection_recall_recipe_harm.py
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
    # (rel, boxes_native: [(label, box)])
    ("test_output/DSCF4588.jpg", [
        ("poster-FP", (524, 155, 758, 904)),
        ("subject", None),  # filled at runtime from RF subject box
    ]),
    ("test_output/DSCF4598.jpg", [
        ("poster-FP", (533, 164, 813, 828)),
        ("subject-missed", (1226, 1480, 780, 920)),
    ]),
]

# RF-confirmed subject box for DSCF4588 (from retinaface.json, most confident)
RF_SUBJECT_4588 = (613, 1080, 1081, 1252)


def mean_delta(img, out, box, shrink=0.7):
    x, y, bw, bh = box
    h, w = img.shape[:2]
    dx, dy = int(bw * (1 - shrink) / 2), int(bh * (1 - shrink) / 2)
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(w, x + bw - dx), min(h, y + bh - dy)
    if x2 <= x1 or y2 <= y1:
        return None
    d = np.abs(out[y1:y2, x1:x2].astype(np.float32) - img[y1:y2, x1:x2].astype(np.float32))
    return float(d.mean()), float(d.max()), float(np.percentile(d, 99))


def main():
    import json
    rf = json.loads((ROOT / "test_output" / "detection_recall_study" / "retinaface.json").read_text())

    eng = RetouchEngine()
    try:
        for rel, boxes in CASES:
            img = cv2.imread(str(ROOT / rel))
            h, w = img.shape[:2]
            s = min(1.0, 2048.0 / max(h, w))
            small = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

            # resolve subject box from retinaface (most confident native det)
            rf_native = rf[rel]["native"]
            subj = max(rf_native, key=lambda b: b[4])
            resolved = []
            for label, box in boxes:
                if box is None:
                    box = subj[:4] if label == "subject" else RF_SUBJECT_4588
                resolved.append((label, box))

            for recipe in ("natural", "cosplay"):
                t0 = time.time()
                result = eng.process(small, recipe=recipe)
                dt = time.time() - t0
                out = np.asarray(result)
                faces = getattr(result, "face_count", "?")
                frame_d = np.abs(out.astype(np.float32) - small.astype(np.float32))
                print(f"\n{rel.split('/')[-1]} [{recipe}] face_count={faces} ({dt:.1f}s)  "
                      f"frame mean|Δ|={frame_d.mean():.2f}")
                for label, box in resolved:
                    sb = tuple(int(v * s) for v in box)
                    md = mean_delta(small, out, sb)
                    if md:
                        print(f"    {label:15s}: mean|Δ|={md[0]:6.2f}  max={md[1]:6.1f}  p99={md[2]:6.1f}")
    finally:
        eng.close()


if __name__ == "__main__":
    main()
