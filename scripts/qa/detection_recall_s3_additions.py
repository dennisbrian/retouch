"""Pre-implementation check: what does the 1024 pass ADD to the 2048 union?

The dual-scale design unions S1 (2048) with S3 (1024). Recall upside is
already measured (83/83 subjects). This checks the PRECISION cost: for every
box S3 adds to the union (IoU < 0.5 vs all S1 boxes), is it a real face
(RetinaFace-confirmed, or non-tiled MediaPipe corroboration) or a poster FP
(RF-unconfirmed + poster texture signature)?

Decision rule for implementation: if any S3 addition is RF-unconfirmed AND
has poster-level texture, the union needs the S1-priority IoU-0.5 dedup only
(no extra veto) only if those FPs are few; otherwise reconsider.

    .venv/bin/python scripts/qa/detection_recall_s3_additions.py
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


def texture_score(img, box):
    x, y, bw, bh = box[:4]
    frac = 0.6
    dx, dy = int(bw * (1 - frac) / 2), int(bh * (1 - frac) / 2)
    h, w = img.shape[:2]
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(w, x + bw - dx), min(h, y + bh - dy)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    s = 256.0 / max(ch, cw)
    crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return float(cv2.Laplacian(gray, cv2.CV_32F, ksize=3).var())


def main():
    data = json.loads((OUT_DIR / "gt_union.json").read_text())
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())

    additions = []
    for rel, rec in data.items():
        s1 = rec["strategies"]["S1_engine_proxy2048"]
        s3 = rec["strategies"]["S3_down1024"]
        rf_native = rf.get(rel, {}).get("native", [])
        rf_proxy = rf.get(rel, {}).get("proxy2048", [])
        for b in s3:
            if any(iou(b[:4], s[:4]) >= 0.5 for s in s1):
                continue  # duplicate, absorbed by dedup
            rf_conf = any(iou(b[:4], r[:4]) >= 0.4 for r in rf_native + rf_proxy)
            # corroboration: does another NON-tiled, non-S3 MediaPipe scale see it?
            s2 = rec["strategies"]["S2_native"]
            mp_corr = any(iou(b[:4], s[:4]) >= 0.5 for s in s2)
            img = cv2.imread(str(ROOT / rel))
            tex = texture_score(img, b) if img is not None else None
            additions.append({
                "rel": rel.split("/")[-1], "box": b[:4], "score": round(b[4], 2),
                "rf": rf_conf, "mp_corroborated": mp_corr, "tex": None if tex is None else round(tex, 1),
                "poster_like": (not rf_conf) and tex is not None and tex < 350,
            })

    real = [a for a in additions if a["rf"] or a["mp_corroborated"]]
    poster = [a for a in additions if a["poster_like"]]
    unclear = [a for a in additions if not a["rf"] and not a["mp_corroborated"] and not a["poster_like"]]

    print(f"S3 additions to the S1 union: {len(additions)}")
    print(f"  REAL (RF-confirmed or MP-corroborated): {len(real)}")
    for a in real:
        print(f"    {a['rel']}: {a['box']} tex={a['tex']} rf={a['rf']}")
    print(f"  POSTER-LIKE (RF-unconfirmed, tex<350): {len(poster)}")
    for a in poster:
        print(f"    {a['rel']}: {a['box']} tex={a['tex']}")
    print(f"  UNCLEAR (unconfirmed, uncorroborated, tex>=350): {len(unclear)}")
    for a in unclear:
        print(f"    {a['rel']}: {a['box']} tex={a['tex']}")

    (OUT_DIR / "s3_additions.json").write_text(json.dumps(additions, indent=1))
    print("-> s3_additions.json")


if __name__ == "__main__":
    main()
