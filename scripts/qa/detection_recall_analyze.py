"""Recall analysis + close-up verification crops for the detection study.

1. Computes engine-pass (S1) recall against the pseudo-GT, overall and by
   face-size bucket.
2. Renders zoomed crops around every GT face MISSED by S1 so a human can
   verify each one is a real face (and not a tiled-pass false positive).

    .venv/bin/python scripts/qa/detection_recall_analyze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

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


def main():
    data = json.loads((OUT_DIR / "gt_union.json").read_text())

    total_gt = 0
    total_matched = 0
    by_bucket = {"subject(>=256px)": [0, 0], "mid(128-256px)": [0, 0], "small(64-128px)": [0, 0], "tiny(<64px)": [0, 0]}
    missed = []  # (rel, gt_box, found_by)

    for rel, rec in data.items():
        s1 = rec["strategies"]["S1_engine_proxy2048"]
        for gt in rec["gt"]:
            total_gt += 1
            hit = any(iou(gt[:4], b[:4]) >= 0.5 for b in s1)
            side = max(gt[2], gt[3])
            if side >= 256:
                bucket = "subject(>=256px)"
            elif side >= 128:
                bucket = "mid(128-256px)"
            elif side >= 64:
                bucket = "small(64-128px)"
            else:
                bucket = "tiny(<64px)"
            by_bucket[bucket][0] += 1
            if hit:
                total_matched += 1
                by_bucket[bucket][1] += 1
            else:
                missed.append((rel, gt))

    print(f"Pseudo-GT faces: {total_gt}")
    print(f"Engine pass (S1 @2048 proxy, legacy FaceMesh) matched: {total_matched}")
    print(f"S1 recall: {total_matched/total_gt*100:.1f}%   miss rate: {(1-total_matched/total_gt)*100:.1f}%")
    print("\nBy face size (native px, longest bbox side):")
    for name, (n, m) in by_bucket.items():
        pct = m / n * 100 if n else 0
        print(f"  {name:18s}: {m}/{n} ({pct:.1f}%)")

    # per-strategy recall (which strategies find what)
    print("\nPer-strategy contribution (recall of each strategy vs union GT):")
    for sname in ["S1_engine_proxy2048", "S2_native", "S3_down1024", "S4_tiled_3x3", "S5_tiled_4x4"]:
        m = sum(
            1
            for rec in data.values()
            for gt in rec["gt"]
            if any(iou(gt[:4], b[:4]) >= 0.5 for b in rec["strategies"][sname])
        )
        print(f"  {sname:20s}: {m}/{total_gt} ({m/total_gt*100:.1f}%)")

    # GT faces found ONLY by tiled strategies -> suspicious / tiny distant
    only_tiled = [
        (rel, gt)
        for rel, rec in data.items()
        for gt in rec["gt"]
        if gt[5] in ("S4_tiled_3x3", "S5_tiled_4x4")
        and not any(iou(gt[:4], b[:4]) >= 0.5 for b in rec["strategies"]["S1_engine_proxy2048"])
        and not any(iou(gt[:4], b[:4]) >= 0.5 for b in rec["strategies"]["S2_native"])
        and not any(iou(gt[:4], b[:4]) >= 0.5 for b in rec["strategies"]["S3_down1024"])
    ]
    print(f"\nGT faces found ONLY by tiled passes (need visual verification): {len(only_tiled)}")

    # render verification crops: every missed face, zoomed
    crop_files = []
    for i, (rel, gt) in enumerate(missed):
        img = cv2.imread(str(ROOT / rel))
        if img is None:
            continue
        x, y, bw, bh = gt[:4]
        h, w = img.shape[:2]
        pad = int(max(bw, bh) * 0.6)
        cx1, cy1 = max(0, x - pad), max(0, y - pad)
        cx2, cy2 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = img[cy1:cy2, cx1:cx2].copy()
        ch, cw = crop.shape[:2]
        # draw the GT box + context box scaled
        cv2.rectangle(crop, (x - cx1, y - cy1), (x - cx1 + bw, y - cy1 + bh), (0, 255, 0), 2)
        scale = 320 / max(cw, ch)
        if scale != 1.0:
            crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        hh, ww = crop.shape[:2]
        label = f"{rel.split('/')[-1]} {bw}x{bh} via {gt[5]}"
        cv2.rectangle(crop, (0, 0), (ww, 20), (0, 0, 0), -1)
        cv2.putText(crop, label[:60], (3, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        fn = OUT_DIR / f"miss_{i:03d}.jpg"
        cv2.imwrite(str(fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
        crop_files.append(fn)

    # contact sheet of missed-face crops
    if crop_files:
        TH = 200
        thumbs = []
        for fn in crop_files:
            im = cv2.imread(str(fn))
            r = TH / im.shape[0]
            thumbs.append(cv2.resize(im, (int(im.shape[1] * r), TH)))
        cols = 4
        rows = (len(thumbs) + cols - 1) // cols
        maxw = max(t.shape[1] for t in thumbs)
        sheet = np.zeros((rows * TH, cols * maxw, 3), dtype=np.uint8)
        for i, t in enumerate(thumbs):
            r, c = divmod(i, cols)
            sheet[r * TH:r * TH + TH, c * maxw:c * maxw + t.shape[1]] = t
        out = OUT_DIR / "missed_faces_sheet.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print(f"\nMissed-face crops: {len(crop_files)} -> sheet {out}")
        print("Human-verify each crop is a REAL face; drop false positives from GT.")

    (OUT_DIR / "missed_index.json").write_text(json.dumps(
        [{"rel": rel, "box": gt[:5], "found_by": gt[5]} for rel, gt in missed], indent=1))
    print("Index: missed_index.json")


if __name__ == "__main__":
    main()
