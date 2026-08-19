"""Forensic disambiguation of unconfirmed S1 detections (vision-free).

Hypothesis: S1 faces RetinaFace does not confirm are MediaPipe FPs on anime
scenery (posters/banners/figures) — an adversarial artifact of convention
cosplay photography. Tests:

  T1 Texture variance: real skin has pores/hair high-frequency energy;
     cel-shaded anime regions are near-uniform. Metric: normalized
     Laplacian variance inside the central 60% of each box, on grayscale,
     at fixed analysis scale (resized to 256px).
  T2 Chromophore realism: real skin decomposes into melanin/hemoglobin
     with plausible spatial variance; flat paint does not. Metric: std of
     hemoglobin map inside box / std outside box.
  T3 Series geometry: the DSCF4588-4601 unconfirmed boxes share near-
     identical normalized geometry (same banner across one shoot) — a real
     second person varies across frames.
  T4 Control group: run the same metrics on RF-CONFIRMED S1 boxes (assumed
     real faces) for reference ranges.

    .venv/bin/python scripts/qa/detection_recall_forensics.py
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


def central_crop(img, box, frac=0.6):
    x, y, bw, bh = box[:4]
    h, w = img.shape[:2]
    dx = int(bw * (1 - frac) / 2)
    dy = int(bh * (1 - frac) / 2)
    cx1, cy1 = max(0, x + dx), max(0, y + dy)
    cx2, cy2 = min(w, x + bw - dx), min(h, y + bh - dy)
    return img[cy1:cy2, cx1:cx2]


def texture_score(img, box):
    """Normalized Laplacian variance at fixed 256px analysis scale."""
    crop = central_crop(img, box)
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    s = 256.0 / max(ch, cw)
    crop = cv2.resize(crop, (max(1, int(cw * s)), max(1, int(ch * s))), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(lap.var())


def chromophore_score(img, box):
    """Hemoglobin-map variance ratio (inside vs outside the box)."""
    try:
        from retouch.chromophore import decompose_chromophores
    except Exception:
        return None
    x, y, bw, bh = box[:4]
    h, w = img.shape[:2]
    # analysis window: box + 50% margin, clamped
    mx, my = int(bw * 0.5), int(bh * 0.5)
    wx1, wy1 = max(0, x - mx), max(0, y - my)
    wx2, wy2 = min(w, x + bw + mx), min(h, y + bh + my)
    # cap window to 1600px longest side for tractability
    ww, wh = wx2 - wx1, wy2 - wy1
    s = min(1.0, 1600.0 / max(ww, wh))
    window = img[wy1:wy2, wx1:wx2]
    if s < 1.0:
        window = cv2.resize(window, (int(ww * s), int(wh * s)), interpolation=cv2.INTER_AREA)
    try:
        melanin, hemoglobin = decompose_chromophores(window)
    except Exception:
        return None
    # inside mask in window coords
    ib_x1 = int((x - wx1) * s)
    ib_y1 = int((y - wy1) * s)
    ib_x2 = int((x + bw - wx1) * s)
    ib_y2 = int((y + bh - wy1) * s)
    ih, iw = window.shape[:2]
    inside = np.zeros((ih, iw), dtype=bool)
    inside[max(0, ib_y1):min(ih, ib_y2), max(0, ib_x1):min(iw, ib_x2)] = True
    if inside.sum() < 100 or (~inside).sum() < 100:
        return None
    std_in = float(np.std(hemoglobin[inside]))
    std_out = float(np.std(hemoglobin[~inside]))
    return std_in / (std_out + 1e-6)


def main():
    data = json.loads((OUT_DIR / "gt_union.json").read_text())
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())

    unconfirmed = []
    confirmed = []
    for rel, rec in data.items():
        s1 = rec["strategies"]["S1_engine_proxy2048"]
        rf_native = rf.get(rel, {}).get("native", [])
        rf_proxy = rf.get(rel, {}).get("proxy2048", [])
        for b in s1:
            ok = any(iou(b[:4], r[:4]) >= 0.4 for r in rf_native) or \
                 any(iou(b[:4], r[:4]) >= 0.4 for r in rf_proxy)
            (confirmed if ok else unconfirmed).append((rel, b))

    print(f"T4 control: {len(confirmed)} RF-confirmed S1 faces")
    print(f"Target: {len(unconfirmed)} unconfirmed S1 faces\n")

    rows = []
    for label, group in (("CONFIRMED", confirmed), ("UNCONFIRMED", unconfirmed)):
        for rel, b in group:
            img = cv2.imread(str(ROOT / rel))
            if img is None:
                continue
            h, w = img.shape[:2]
            t = texture_score(img, b)
            c = chromophore_score(img, b)
            if t is None:
                continue
            rows.append({
                "label": label, "rel": rel.split("/")[-1], "box": b[:4],
                "norm_geom": (round(b[0] / w, 3), round(b[1] / h, 3), round(b[2] / w, 3), round(b[3] / h, 3)),
                "texture_lap_var": t,
                "hb_ratio": None if c is None else round(c, 3),
            })

    for label in ("CONFIRMED", "UNCONFIRMED"):
        sub = [r for r in rows if r["label"] == label]
        tex = np.array([r["texture_lap_var"] for r in sub])
        hb = np.array([r["hb_ratio"] for r in sub if r["hb_ratio"] is not None])
        print(f"--- {label} (n={len(sub)}) ---")
        print(f"  Laplacian var: median={np.median(tex):.1f} p10={np.percentile(tex,10):.1f} p90={np.percentile(tex,90):.1f}")
        if len(hb):
            print(f"  Hb variance ratio: median={np.median(hb):.2f} p10={np.percentile(hb,10):.2f} p90={np.percentile(hb,90):.2f}")

    print("\n--- Unconfirmed boxes, detail (sorted by texture) ---")
    unc = sorted((r for r in rows if r["label"] == "UNCONFIRMED"), key=lambda r: r["texture_lap_var"])
    for r in unc:
        print(f"  {r['rel']}: tex={r['texture_lap_var']:8.1f} hb={r['hb_ratio']} geom={r['norm_geom']} box={r['box']}")

    # T3: geometry clustering of unconfirmed boxes (DSCF4588-4601 series)
    print("\n--- T3: geometry consistency of the DSCF4588-4601 unconfirmed series ---")
    series = [r for r in rows if r["label"] == "UNCONFIRMED" and any(k in r["rel"] for k in
              [f"DSCF{n}.jpg" for n in range(4588, 4602)])]
    if series:
        geoms = np.array([r["norm_geom"] for r in series])
        print(f"  n={len(series)}; normalized-geometry std per component: {geoms.std(axis=0).round(3)}")
        print(f"  (near-zero std => same scenery object across frames, not a person)")

    (OUT_DIR / "forensics_rows.json").write_text(json.dumps(rows, indent=1))
    print("\n-> forensics_rows.json")


if __name__ == "__main__":
    main()
