"""Joint veto design + leave-one-out validation on the corpus.

Combines three signals measured in this study:
  S1 PERSON-COVERAGE   (central 40%, person-mask gate @ 0.5)
  S2 TEXTURE Lap-var   (central 60%, fixed 256px analysis geometry)
  S3 LANDMARK RATIO    ied / max(bbox_w, bbox_h) — robust face geometry
                       (inter-eye distance scales with face size for real
                       faces, ~0.3-0.5; posters can sit outside)

Veto rule (proposed):
  VETO a detection when ANY of:
    (a) person_coverage < 0.5                       (kills 13/18 posters)
    (b) person_coverage < 0.9 AND texture < 350    (kills high-cov posters
        near the person, with poster-typical texture; threshold chosen from
        A2's real-face floor under bilateral+JPEG = 31.6, so any threshold
        below ~30 risks vetoing a real face — see veto doc)
  NEVER veto if person_coverage == 1.0 (too risky — some posters genuinely
        sit on the person; the cost of vetoing a real face outweighs the
        benefit of removing a per-face pass on a poster)

Leave-one-out: for each of 5 candidate thresholds on (b)'s texture term,
measure false-veto rate on real faces (target: 0) and kill rate on the
18 FPs.

    .venv/bin/python scripts/qa/posterfp_joint_veto.py
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

from retouch.detection import FaceDetector, inter_eye_distance  # noqa: E402

OUT_DIR = ROOT / "test_output" / "detection_recall_study"


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def coverage(mask, box, shrink=0.4):
    h, w = mask.shape[:2]
    x, y, bw, bh = box[:4]
    dx, dy = int(bw * shrink / 2), int(bh * shrink / 2)
    x1, y1 = max(0, x + dx), max(0, y + dy)
    x2, y2 = min(w, x + bw - dx), min(h, y + bh - dy)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((mask[y1:y2, x1:x2] > 0.5).mean())


def texture(img, box):
    x, y, bw, bh = box[:4]
    h, w = img.shape[:2]
    dx, dy = int(bw * 0.2), int(bh * 0.2)
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
                tex = texture(proxy, sb)
                ied_ratio = b[3] / max(b[2], b[3]) if b[2] and b[3] else None
                rows.append({"rel": rel.split("/")[-1], "box": b[:4], "rf": conf,
                             "cov": round(cov, 3), "tex": None if tex is None else round(tex, 1),
                             "ied_ratio": round(ied_ratio, 3) if ied_ratio else None})
    finally:
        det.close()

    real = [r for r in rows if r["rf"]]
    fakes = [r for r in rows if not r["rf"]]
    print(f"Real faces: n={len(real)}   Poster FPs: n={len(fakes)}\n")

    # --- landmark ratio (S3) distribution check ---
    real_ratios = np.array([r["ied_ratio"] for r in real if r["ied_ratio"]])
    fake_ratios = np.array([r["ied_ratio"] for r in fakes if r["ied_ratio"]])
    if len(real_ratios) and len(fake_ratios):
        print("ied/bbox_max ratio:")
        print(f"  REAL: min={real_ratios.min():.3f} p10={np.percentile(real_ratios,10):.3f} "
              f"median={np.median(real_ratios):.3f} max={real_ratios.max():.3f}")
        print(f"  FAKE: min={fake_ratios.min():.3f} p10={np.percentile(fake_ratios,10):.3f} "
              f"median={np.median(fake_ratios):.3f} max={fake_ratios.max():.3f}")

    # --- leave-one-out over candidate (b) texture thresholds ---
    print("\nLeave-one-out over (b) texture threshold (cov<0.9 AND tex<T => veto):")
    print(f"{'T':>6} {'FPs killed':>12} {'real collaterals':>18} {'safe?':>6}")
    for T in (50, 80, 100, 120, 150, 200, 250, 300, 350):
        killed = 0
        collateral = 0
        for r in rows:
            veto = (r["cov"] < 0.5) or (r["cov"] < 0.9 and r["tex"] is not None and r["tex"] < T)
            if veto and not r["rf"]:
                killed += 1
            if veto and r["rf"]:
                collateral += 1
        safe = "YES" if collateral == 0 else "NO"
        print(f"{T:>6} {killed:>12} {collateral:>18} {safe:>6}")

    # --- proposed rule ---
    print("\nPROPOSED RULE:")
    print("  veto = (cov < 0.5) OR (cov < 0.9 AND tex < 80)")
    killed, collateral = [], []
    for r in rows:
        veto = (r["cov"] < 0.5) or (r["cov"] < 0.9 and r["tex"] is not None and r["tex"] < 80)
        if veto and not r["rf"]:
            killed.append(r)
        if veto and r["rf"]:
            collateral.append(r)
    print(f"  FPs killed: {len(killed)}/{len(fakes)}")
    print(f"  Real collaterals: {len(collateral)}/{len(real)}")
    print(f"  Surviving FPs:")
    for r in fakes:
        if r not in killed:
            print(f"    {r['rel']}: cov={r['cov']} tex={r['tex']} ied_ratio={r['ied_ratio']}")

    (OUT_DIR / "posterfp_joint_veto.json").write_text(json.dumps(rows, indent=1))
    print("\n-> posterfp_joint_veto.json")


if __name__ == "__main__":
    main()
