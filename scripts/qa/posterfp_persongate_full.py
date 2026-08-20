"""Poster-FP veto research, part 1: person-coverage distributions.

The dual-scale person gate worked at the poles (posters 0.000, subjects
1.000) on the 6 S3 additions. But the 18 PRE-EXISTING S1 poster FPs were
never person-coverage measured — if the segmenter includes posters in-frame
(as the research doc speculated), coverage is useless as a veto for them.

Measures central-40% person-mask coverage for ALL 100 S1 detections,
split RF-confirmed (84: 82 real + the 2 new dual-scale... actually 82
pre-change + RF-unconfirmed list) vs poster-like unconfirmed (18), at the
2048 proxy scale, same as the engine sees.

    .venv/bin/python scripts/qa/posterfp_persongate_full.py
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
    h, w = mask.shape[:2]
    x, y, bw, bh = box[:4]
    dx, dy = int(bw * shrink / 2), int(bh * shrink / 2)
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(w, x + bw - dx), min(h, y + bh - dy)
    if x2 <= x1 or y2 <= y1:
        return None
    return float((mask[y1:y2, x1:x2] > 0.5).mean())


def main():
    data = json.loads((OUT_DIR / "gt_union.json").read_text())
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())

    det = FaceDetector(max_faces=25, min_confidence=0.4)
    rows = []
    try:
        for rel, rec in data.items():
            rf_native = rf.get(rel, {}).get("native", [])
            rf_proxy = rf.get(rel, {}).get("proxy2048", [])
            img = cv2.imread(str(ROOT / rel))
            h, w = img.shape[:2]
            s = min(1.0, 2048.0 / max(h, w))
            proxy = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
            mask = det.segment_person(proxy)
            for b in rec["strategies"]["S1_engine_proxy2048"]:
                conf = any(iou(b[:4], r[:4]) >= 0.4 for r in rf_native) or \
                       any(iou(b[:4], r[:4]) >= 0.4 for r in rf_proxy)
                sb = tuple(int(v * s) for v in b[:4])
                cov = coverage(mask, sb)
                if cov is not None:
                    rows.append({"rel": rel.split("/")[-1], "box": b[:4],
                                 "rf": conf, "cov": round(cov, 3)})
    finally:
        det.close()

    conf_cov = np.array([r["cov"] for r in rows if r["rf"]])
    unc_cov = np.array([r["cov"] for r in rows if not r["rf"]])
    print(f"RF-confirmed detections (real faces): n={len(conf_cov)}")
    print(f"  coverage: min={conf_cov.min():.3f} p10={np.percentile(conf_cov,10):.3f} "
          f"median={np.median(conf_cov):.3f} max={conf_cov.max():.3f}")
    print(f"RF-unconfirmed (poster-like): n={len(unc_cov)}")
    print(f"  coverage: min={unc_cov.min():.3f} p10={np.percentile(unc_cov,10):.3f} "
          f"median={np.median(unc_cov):.3f} max={unc_cov.max():.3f}")
    print("\nUnconfirmed detail (sorted by coverage):")
    for r in sorted((r for r in rows if not r["rf"]), key=lambda r: r["cov"]):
        print(f"  {r['rel']}: cov={r['cov']:.3f} box={r['box']}")
    low_conf = (conf_cov < 0.5).sum()
    print(f"\nReal faces BELOW 0.5 coverage (would be veto-collateral): {low_conf}")
    high_unc = (unc_cov >= 0.5).sum()
    print(f"Poster FPs AT/ABOVE 0.5 coverage (veto-resistant): {high_unc}")

    (OUT_DIR / "posterfp_persongate_full.json").write_text(json.dumps(rows, indent=1))
    print("-> posterfp_persongate_full.json")


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
