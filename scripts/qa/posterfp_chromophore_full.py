"""Corpus-wide chromophore validation (D1/D2 across all detections).

The survivors probe showed skinlike-fraction separation on 4+4 cases.
This validates across the FULL corpus: all 82 RF-confirmed real faces
(min-of-class matters — the veto must not eat the least-skin-like real
face) and all 18 poster FPs (max-of-class).

skinlike = fraction of pixels with Hb in [0.05, 0.6] AND melanin in
[0.02, 0.5] at the fixed 320px analysis geometry (central 60% of bbox).

    .venv/bin/python scripts/qa/posterfp_chromophore_full.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.chromophore import decompose_chromophores  # noqa: E402

OUT_DIR = ROOT / "test_output" / "detection_recall_study"


def central(img, box, frac=0.6):
    x, y, bw, bh = box[:4]
    h, w = img.shape[:2]
    dx, dy = int(bw * (1 - frac) / 2), int(bh * (1 - frac) / 2)
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(w, x + bw - dx), min(h, y + bh - dy)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    ch, cw = crop.shape[:2]
    s = 320.0 / max(ch, cw)
    if s != 1.0:
        crop = cv2.resize(crop, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_AREA)
    return crop


def metrics(crop):
    mel, hb = decompose_chromophores(crop.astype(np.float32) / 255.0)
    hb_std = float(np.std(hb))
    skinlike = float(((hb > 0.05) & (hb < 0.6) & (mel > 0.02) & (mel < 0.5)).mean())
    return hb_std, skinlike


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
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())

    real_rows, fake_rows = [], []
    for rel, rec in data.items():
        rf_native = rf.get(rel, {}).get("native", [])
        rf_proxy = rf.get(rel, {}).get("proxy2048", [])
        img = cv2.imread(str(ROOT / rel))
        if img is None:
            continue
        for b in rec["strategies"]["S1_engine_proxy2048"]:
            conf = any(iou(b[:4], r[:4]) >= 0.4 for r in rf_native) or \
                   any(iou(b[:4], r[:4]) >= 0.4 for r in rf_proxy)
            crop = central(img, b[:4])
            if crop is None:
                continue
            hb_std, skinlike = metrics(crop)
            row = {"rel": rel.split("/")[-1], "hb_std": round(hb_std, 3), "skinlike": round(skinlike, 3)}
            (real_rows if conf else fake_rows).append(row)

    r_hb = np.array([r["hb_std"] for r in real_rows])
    r_sk = np.array([r["skinlike"] for r in real_rows])
    f_hb = np.array([r["hb_std"] for r in fake_rows])
    f_sk = np.array([r["skinlike"] for r in fake_rows])

    print(f"REAL faces n={len(real_rows)}:")
    print(f"  hb_std:    min={r_hb.min():.3f} p05={np.percentile(r_hb,5):.3f} median={np.median(r_hb):.3f}")
    print(f"  skinlike:  min={r_sk.min():.3f} p05={np.percentile(r_sk,5):.3f} median={np.median(r_sk):.3f}")
    print(f"POSTER FPs n={len(fake_rows)}:")
    print(f"  hb_std:    max={f_hb.max():.3f} p95={np.percentile(f_hb,95):.3f} median={np.median(f_hb):.3f}")
    print(f"  skinlike:  max={f_sk.max():.3f} p95={np.percentile(f_sk,95):.3f} median={np.median(f_sk):.3f}")

    # threshold sweep on skinlike for the 4 joint-rule survivors only
    print("\nskinlike threshold sweep — applied ONLY when cov>=0.9 and tex<T survived:")
    survivors = {
        "DSCF4588.jpg": [524, 155, 758, 904],
        "DSCF4612.jpg": [807, 1864, 393, 423],
        "DSCF4617.jpg": [819, 1876, 399, 423],
        "DSCF4618.jpg": [652, 438, 838, 901],
    }
    for T in (0.05, 0.08, 0.10, 0.12):
        killed = sum(1 for r in fake_rows
                     if r["rel"] in survivors and r["skinlike"] < T)
        collateral = sum(1 for r in real_rows if r["skinlike"] < T)
        print(f"  T={T:.2f}: survivors killed {killed}/4, real-face corpus-wide collateral {collateral}/{len(real_rows)}")

    print("\nReal faces with lowest skinlike (closest to poster territory):")
    for r in sorted(real_rows, key=lambda r: r["skinlike"])[:8]:
        print(f"  {r['rel']}: skinlike={r['skinlike']} hb_std={r['hb_std']}")

    (OUT_DIR / "posterfp_chromophore_full.json").write_text(
        json.dumps({"real": real_rows, "fake": fake_rows}, indent=1))
    print("-> posterfp_chromophore_full.json")


if __name__ == "__main__":
    main()
