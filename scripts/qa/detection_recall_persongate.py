"""Person-mask gate hypothesis check for dual-scale union.

Hypothesis: an S3 (1024) addition that S1 (2048) missed should be accepted
only when it lies on the segmented PERSON (selfie segmenter at proxy scale).
Poster/banner faces are off-person; the missed subject (DSCF4598) is on-person.

For each of the 6 S3 additions, measures:
  - person-mask coverage of the box's central region (and full box)
  - at the same 2048 proxy scale the engine uses

    .venv/bin/python scripts/qa/detection_recall_persongate.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("RETOUCH_GPU", "0")
os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.detection import FaceDetector  # noqa: E402

OUT_DIR = ROOT / "test_output" / "detection_recall_study"


def coverage(mask, box, shrink=0.4):
    """Fraction of the box's central area where mask > 0.5."""
    h, w = mask.shape[:2]
    x, y, bw, bh = box[:4]
    dx, dy = int(bw * shrink / 2), int(bh * shrink / 2)
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(w, x + bw - dx), min(h, y + bh - dy)
    if x2 <= x1 or y2 <= y1:
        return None
    region = mask[y1:y2, x1:x2]
    return float((region > 0.5).mean())


def main():
    additions = json.loads((OUT_DIR / "s3_additions.json").read_text())

    det = FaceDetector(max_faces=25, min_confidence=0.4)
    try:
        rows = []
        for a in additions:
            rel = "test_output/" + a["rel"]
            img = cv2.imread(str(ROOT / rel))
            h, w = img.shape[:2]
            s = min(1.0, 2048.0 / max(h, w))
            proxy = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            ph, pw = proxy.shape[:2]
            mask = det.segment_person(proxy)
            sb = tuple(int(v * s) for v in a["box"])
            cov = coverage(mask, sb)
            # control: the S1 subject box coverage for the same image
            rows.append({"rel": a["rel"], "box": a["box"], "rf": a["rf"], "tex": a["tex"],
                         "person_cov": None if cov is None else round(cov, 3)})
            print(f"{a['rel']}: rf={a['rf']} tex={a['tex']} person_cov={cov:.3f}" if cov is not None
                  else f"{a['rel']}: coverage NA")
    finally:
        det.close()

    # also run the S1 subject boxes as controls
    data = json.loads((OUT_DIR / "gt_union.json").read_text())
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())
    det = FaceDetector(max_faces=25, min_confidence=0.4)
    try:
        print("\nControls — RF-confirmed subject boxes (should be on-person):")
        # one control per affected image + a few random
        controls = sorted({a["rel"] for a in additions}) + ["DSCF4463.jpg", "DSCF7204.jpg", "DSCF4613.jpg"]
        for name in controls:
            rel = f"test_output/{name}"
            rec = data.get(rel)
            if not rec:
                continue
            rf_native = rf.get(rel, {}).get("native", [])
            if not rf_native:
                continue
            subj = max(rf_native, key=lambda b: b[4])
            img = cv2.imread(str(ROOT / rel))
            h, w = img.shape[:2]
            s = min(1.0, 2048.0 / max(h, w))
            proxy = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            mask = det.segment_person(proxy)
            sb = tuple(int(v * s) for v in subj[:4])
            cov = coverage(mask, sb)
            print(f"  {name}: subject person_cov={cov:.3f}")
    finally:
        det.close()

    (OUT_DIR / "persongate_rows.json").write_text(json.dumps(rows, indent=1))
    print("\n-> persongate_rows.json")


if __name__ == "__main__":
    main()
