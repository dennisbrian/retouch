"""Render verification montages for the detection-recall pseudo-GT.

Draws all pseudo-GT boxes (green) and the engine-equivalent S1 detections
(yellow) on downscaled copies, in a grid contact sheet.

    .venv/bin/python scripts/qa/detection_recall_render.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "test_output" / "detection_recall_study"

THUMB_W = 360
COLS = 5


def main():
    data = json.loads((OUT_DIR / "gt_union.json").read_text())
    entries = sorted(data.items())
    thumbs = []
    for rel, rec in entries:
        img = cv2.imread(str(ROOT / rel))
        if img is None:
            continue
        h, w = img.shape[:2]
        sx = THUMB_W / w
        thumb = cv2.resize(img, (THUMB_W, int(h * sx)))
        th, tw = thumb.shape[:2]
        # pseudo-GT boxes: green
        for b in rec["gt"]:
            x, y, bw, bh = b[:4]
            p1 = (int(x * sx), int(y * sx))
            p2 = (int((x + bw) * sx), int((y + bh) * sx))
            cv2.rectangle(thumb, p1, p2, (0, 255, 0), 2)
        # engine S1 boxes: yellow, thicker
        for b in rec["strategies"]["S1_engine_proxy2048"]:
            x, y, bw, bh = b[:4]
            p1 = (int(x * sx) + 1, int(y * sx) + 1)
            p2 = (int((x + bw) * sx) - 1, int((y + bh) * sx) - 1)
            cv2.rectangle(thumb, p1, p2, (0, 255, 255), 2)
        # label
        label = rel.split("/")[-1]
        cv2.rectangle(thumb, (0, 0), (tw, 18), (0, 0, 0), -1)
        cv2.putText(thumb, f"{label} gt={len(rec['gt'])} s1={len(rec['strategies']['S1_engine_proxy2048'])}",
                    (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        thumbs.append((label, thumb))

    # pad to grid
    rows = (len(thumbs) + COLS - 1) // COLS
    max_h = max(t.shape[0] for _, t in thumbs)
    sheet = np.zeros((rows * max_h, COLS * THUMB_W, 3), dtype=np.uint8)
    for i, (_, t) in enumerate(thumbs):
        r, c = divmod(i, COLS)
        sheet[r * max_h:r * max_h + t.shape[0], c * THUMB_W:c * THUMB_W + THUMB_W] = t

    out = OUT_DIR / "gt_verification_sheet.jpg"
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(f"{len(thumbs)} thumbs -> {out} ({sheet.shape[1]}x{sheet.shape[0]})")

    # Also: images where S1 missed >= 2 GT faces — bigger individual renders
    missed2 = [(rel, rec) for rel, rec in entries
               if len(rec["gt"]) - len(rec["strategies"]["S1_engine_proxy2048"]) >= 2]
    print(f"\nimages with S1 miss >= 2: {len(missed2)}")
    for rel, rec in missed2:
        print(f"  {rel}: gt={len(rec['gt'])} s1={len(rec['strategies']['S1_engine_proxy2048'])}")


if __name__ == "__main__":
    main()
