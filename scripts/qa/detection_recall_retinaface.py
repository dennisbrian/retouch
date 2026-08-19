"""RetinaFace cross-check pass for the detection-recall study.

Runs on SYSTEM python (retinaface 0.0.18 + tf available there, not in .venv):
    python3 scripts/qa/detection_recall_retinaface.py

Threshold deliberately low (0.2) to characterize near-threshold behavior;
results are merged into the pseudo-GT union during analysis, deduped at
IoU 0.5. BGR input per retinaface 0.0.18's preprocess contract (FIX F2).

Output: test_output/detection_recall_study/retinaface.json
  {path: {"native": [[x,y,w,h,score], ...],
          "proxy2048": [[x,y,w,h,score], ...]}}
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import cv2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "test_output" / "detection_recall_study"
CORPUS = OUT_DIR / "corpus.txt"
THRESH = 0.2


def boxes_from_resp(resp, sx=1.0, sy=1.0):
    out = []
    if not isinstance(resp, dict):
        return out
    for face_id, info in resp.items():
        box = info.get("facial_area")
        score = info.get("score", 0.0)
        if not box or score < THRESH:
            continue
        x1, y1, x2, y2 = box
        out.append([int(x1 * sx), int(y1 * sy), int((x2 - x1) * sx), int((y2 - y1) * sy), float(score)])
    return out


def main():
    from retinaface import RetinaFace

    paths = [p.strip() for p in CORPUS.read_text().splitlines() if p.strip()]
    results = {}
    for i, rel in enumerate(paths):
        img = cv2.imread(str(ROOT / rel))
        if img is None:
            continue
        h, w = img.shape[:2]
        rec = {}

        resp = RetinaFace.detect_faces(img, threshold=THRESH)
        rec["native"] = boxes_from_resp(resp)

        scale = min(1.0, 2048 / max(h, w))
        if scale < 1.0:
            proxy = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            resp = RetinaFace.detect_faces(proxy, threshold=THRESH)
            rec["proxy2048"] = boxes_from_resp(resp, sx=1.0 / scale, sy=1.0 / scale)
        else:
            rec["proxy2048"] = list(rec["native"])

        results[rel] = rec
        print(f"[{i+1}/{len(paths)}] {rel}: native={len(rec['native'])} proxy={len(rec['proxy2048'])}", flush=True)

    (OUT_DIR / "retinaface.json").write_text(json.dumps(results, indent=1))
    total = sum(len(r["native"]) for r in results.values())
    print(f"\nDone: {len(results)} images, {total} RetinaFace native boxes -> retinaface.json")


if __name__ == "__main__":
    main()
