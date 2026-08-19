"""Dual-scale detection upside analysis.

The engine detects at a single 2048 proxy. S3 (1024) finds a DIFFERENT set
of faces — including the DSCF4598 subject that S1 misses (and vice versa).
Quantify, against RetinaFace-confirmed subjects:
  - S1 (2048) alone
  - S3 (1024) alone
  - S1 ∪ S3 union (IoU>=0.5 dedup)
  - S1 ∪ S3 ∪ S2 (native)
to size the recall gain of a dual-scale detect pass (+~10-30 ms cost).

Also: FP-suppression accounting — images where S1's only detections are
RF-unconfirmed (poster FPs), i.e. the engine "succeeded" (found >= 1 face)
but the subject is absent, so the tiled fallback never fires.

    .venv/bin/python scripts/qa/detection_recall_dualscale.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "test_output" / "detection_recall_study"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def dedup(boxes, thresh=0.5):
    out = []
    for b in sorted(boxes, key=lambda x: -x[4]):
        if all(iou(b, k) < thresh for k in out):
            out.append(b)
    return out


def main():
    data = json.loads((OUT_DIR / "gt_union.json").read_text())
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())

    sets = {"S1_2048": 0, "S3_1024": 0, "S1+S3": 0, "S1+S3+S2": 0}
    n_subjects = 0
    only_fp_detections = []   # S1 found faces, ALL unconfirmed, subject missed
    fp_suppressed_tiling = []  # subset where S3 or S4/S5 would have found subject

    for rel, rec in data.items():
        rf_native = rf.get(rel, {}).get("native", [])
        if not rf_native:
            continue
        rf_subj = max(rf_native, key=lambda b: b[4])
        n_subjects += 1

        s1 = rec["strategies"]["S1_engine_proxy2048"]
        s2 = rec["strategies"]["S2_native"]
        s3 = rec["strategies"]["S3_down1024"]

        def found(boxes):
            return any(iou(rf_subj[:4], b[:4]) >= 0.4 for b in boxes)

        f1, f3 = found(s1), found(s3)
        u13 = found(dedup(s1 + s3))
        u132 = found(dedup(s1 + s3 + s2))

        sets["S1_2048"] += f1
        sets["S3_1024"] += f3
        sets["S1+S3"] += u13
        sets["S1+S3+S2"] += u132

        if s1 and not f1:
            all_unconf = not any(
                any(iou(b[:4], r[:4]) >= 0.4 for r in rf_native) or
                any(iou(b[:4], r[:4]) >= 0.4 for r in rf.get(rel, {}).get("proxy2048", []))
                for b in s1
            )
            if all_unconf:
                only_fp_detections.append(rel)
                if f3 or found(rec["strategies"]["S4_tiled_3x3"]) or found(rec["strategies"]["S5_tiled_4x4"]):
                    fp_suppressed_tiling.append(rel)

    print(f"RF-confirmed subjects: {n_subjects}")
    print("\nSubject recall by detection set:")
    for name, v in sets.items():
        print(f"  {name:12s}: {v}/{n_subjects} = {v/n_subjects*100:.1f}%")

    print(f"\nImages where S1's ONLY detections are unconfirmed (poster FPs),")
    print(f"subject absent — tiled fallback suppressed by FP presence: {len(only_fp_detections)}")
    for rel in only_fp_detections:
        print(f"  {rel.split('/')[-1]}  (fallback would help: {rel in fp_suppressed_tiling})")

    (OUT_DIR / "dualscale_summary.json").write_text(json.dumps({
        "n_subjects": n_subjects,
        "recall": {k: v / n_subjects for k, v in sets.items()},
        "raw": sets,
        "only_fp_detections": only_fp_detections,
        "fp_suppressed_tiling": fp_suppressed_tiling,
    }, indent=1))
    print("\n-> dualscale_summary.json")


if __name__ == "__main__":
    main()
