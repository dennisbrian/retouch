"""Merged GT analysis: subject vs background recall + RetinaFace agreement.

Refines the raw union-GT numbers:
1. Merges RetinaFace (native + proxy) boxes into the GT pool as additional
   finders, and records which GT faces RetinaFace confirms.
2. Splits recall into:
   - SUBJECT recall: largest GT face per image (the face a portrait
     retoucher must have),
   - BACKGROUND recall: all other faces (crowd/convention context — the
     engine's per-face retouch would ideally catch them, but a miss is a
     much lower-severity defect).
3. Strips GT faces found only by tiled MediaPipe AND unconfirmed by
   RetinaFace AND < 10% of frame width — those are the least trustworthy
   (tiled passes on posters/anime faces are known MediaPipe FPs). Reports
   recall both with and without this filter so nothing is hidden.

    .venv/bin/python scripts/qa/detection_recall_merge.py
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
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())

    stats = {
        "images": len(data),
        "gt_total": 0,
        "subject_total": 0,
        "subject_hit": 0,
        "bg_total": 0,
        "bg_hit": 0,
        "gt_confirmed_by_retinaface": 0,
        "gt_unconfirmed": 0,
        "stripped_untrustworthy": 0,
        "gt_after_strip": 0,
        "subject_hit_after_strip": 0,
        "bg_hit_after_strip": 0,
    }
    subject_misses = []
    bg_misses_confirmed = []  # retinaface-confirmed background misses

    for rel, rec in data.items():
        gt_faces = rec["gt"]
        s1 = rec["strategies"]["S1_engine_proxy2048"]
        rf_native = rf.get(rel, {}).get("native", [])
        rf_proxy = rf.get(rel, {}).get("proxy2048", [])
        w = rec["size"][0]

        if not gt_faces:
            continue

        # annotate each GT face with retinaface confirmation
        annotated = []
        for gt in gt_faces:
            rf_conf = any(iou(gt[:4], b[:4]) >= 0.4 for b in rf_native) or \
                      any(iou(gt[:4], b[:4]) >= 0.4 for b in rf_proxy)
            annotated.append({"box": gt[:5], "finder": gt[5], "rf": rf_conf})

        # largest face = subject
        order = sorted(range(len(annotated)), key=lambda i: -(annotated[i]["box"][2] * annotated[i]["box"][3]))
        subject_idx = order[0]

        stripped = set()
        for i, a in enumerate(annotated):
            side = max(a["box"][2], a["box"][3])
            if a["finder"].startswith("S4") or a["finder"].startswith("S5"):
                if not a["rf"] and side < 0.10 * w:
                    stripped.add(i)
                    stats["stripped_untrustworthy"] += 1

        for i, a in enumerate(annotated):
            stats["gt_total"] += 1
            if a["rf"]:
                stats["gt_confirmed_by_retinaface"] += 1
            else:
                stats["gt_unconfirmed"] += 1
            hit = any(iou(a["box"][:4], b[:4]) >= 0.5 for b in s1)
            is_subject = i == subject_idx
            if is_subject:
                stats["subject_total"] += 1
                if hit:
                    stats["subject_hit"] += 1
                else:
                    subject_misses.append((rel, a))
            else:
                stats["bg_total"] += 1
                if hit:
                    stats["bg_hit"] += 1
                elif a["rf"]:
                    bg_misses_confirmed.append((rel, a))
            if i not in stripped:
                stats["gt_after_strip"] += 1
                if is_subject:
                    if hit:
                        stats["subject_hit_after_strip"] += 1
                else:
                    if hit:
                        stats["bg_hit_after_strip"] += 1

    s = stats
    print(f"Images: {s['images']}")
    print(f"\n--- Raw union GT ({s['gt_total']} faces) ---")
    print(f"Subject recall (largest face/image): {s['subject_hit']}/{s['subject_total']} = {s['subject_hit']/s['subject_total']*100:.1f}%")
    print(f"Background recall:                    {s['bg_hit']}/{s['bg_total']} = {s['bg_hit']/s['bg_total']*100:.1f}%")
    print(f"Overall recall:                       {s['subject_hit']+s['bg_hit']}/{s['gt_total']} = {(s['subject_hit']+s['bg_hit'])/s['gt_total']*100:.1f}%")
    print(f"\nRetinaFace confirmation: {s['gt_confirmed_by_retinaface']}/{s['gt_total']} GT faces confirmed by independent detector")
    print(f"Stripped as untrustworthy (tiled-only, RF-unconfirmed, <10% frame): {s['stripped_untrustworthy']}")
    print(f"\n--- After stripping ({s['gt_after_strip']} faces) ---")
    bg_after = s['gt_after_strip'] - s['subject_total']
    print(f"Subject recall: {s['subject_hit_after_strip']}/{s['subject_total']}")
    print(f"Background recall: {s['bg_hit_after_strip']}/{bg_after} = {s['bg_hit_after_strip']/bg_after*100:.1f}%" if bg_after else "")

    print(f"\nSUBJECT MISSES ({len(subject_misses)}):")
    for rel, a in subject_misses:
        print(f"  {rel}: box={a['box'][:4]} finder={a['finder']} rf={a['rf']}")
    print(f"\nBackground misses CONFIRMED by RetinaFace ({len(bg_misses_confirmed)}) — definitely real faces:")
    for rel, a in bg_misses_confirmed[:40]:
        print(f"  {rel.split('/')[-1]}: {a['box'][2]}x{a['box'][3]} finder={a['finder']}")

    # render subject misses for verification
    for i, (rel, a) in enumerate(subject_misses):
        img = cv2.imread(str(ROOT / rel))
        if img is None:
            continue
        x, y, bw, bh = a["box"][:4]
        h, w = img.shape[:2]
        pad = int(max(bw, bh) * 0.4)
        crop = img[max(0, y - pad):min(h, y + bh + pad), max(0, x - pad):min(w, x + bw + pad)].copy()
        cv2.rectangle(crop, (x - max(0, x - pad), y - max(0, y - pad)),
                      (x - max(0, x - pad) + bw, y - max(0, y - pad) + bh), (0, 0, 255), 4)
        cv2.imwrite(str(OUT_DIR / f"subject_miss_{i:02d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if subject_misses:
        print(f"\nSubject-miss crops written: subject_miss_*.jpg")


if __name__ == "__main__":
    main()
