"""Cross-detector consensus analysis — no human vision required.

The strongest available ground truth signal is RetinaFace-native (independent
CNN detector, full resolution): it fires ~1 face/image, almost certainly the
real subject. MediaPipe tiled-only "subjects" that RetinaFace does NOT
confirm are suspect (convention posters/anime screens produce MediaPipe FPs
that can be LARGER than the real subject's box, which corrupts the naive
"largest box = subject" assumption).

Analysis layers:
  A. S1 (engine pass) vs RetinaFace-native agreement — cross-detector
     subject recall: of images where RetinaFace finds a subject face, how
     often does S1 find a face at the same location (IoU >= 0.4)?
  B. Disagreement triage:
     - S1 found a RetinaFace-confirmed face AND a bigger tiled-only box
       exists -> tiled-only box is likely a poster FP; S1 subject OK.
     - RetinaFace finds a face, S1 finds NOTHING matching -> genuine
       subject miss (two independent detectors vs zero).
     - RetinaFace finds nothing (rare) -> unresolvable without human review.
  C. For every S1 face: is it RetinaFace-confirmed? (precision signal)

    .venv/bin/python scripts/qa/detection_recall_consensus.py
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


def best_iou(box, boxes):
    return max((iou(box, b) for b in boxes), default=0.0)


def main():
    data = json.loads((OUT_DIR / "gt_union.json").read_text())
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())

    rf_subject_images = 0          # images where RF-native found >= 1 face
    s1_matches_rf = 0              # S1 found a face overlapping the RF subject
    s1_no_match_but_found = []     # S1 found faces but none match RF subject
    s1_zero_faces = []             # S1 found nothing at all
    rf_subject_missed_by_s1 = []   # union of the two above

    s1_faces_confirmed = 0
    s1_faces_total = 0
    s1_faces_unconfirmed = []      # S1 faces RF does not see (FP risk or RF miss)

    for rel, rec in data.items():
        s1 = rec["strategies"]["S1_engine_proxy2048"]
        rf_native = rf.get(rel, {}).get("native", [])
        rf_proxy = rf.get(rel, {}).get("proxy2048", [])

        # C. S1 face confirmation (vs either RF pass)
        for b in s1:
            s1_faces_total += 1
            if best_iou(b[:4], rf_native) >= 0.4 or best_iou(b[:4], rf_proxy) >= 0.4:
                s1_faces_confirmed += 1
            else:
                s1_faces_unconfirmed.append((rel, b))

        if not rf_native:
            continue
        rf_subject_images += 1
        # RF's most confident detection = the subject
        rf_subj = max(rf_native, key=lambda b: b[4])
        matched = any(iou(rf_subj[:4], b[:4]) >= 0.4 for b in s1)
        if matched:
            s1_matches_rf += 1
        else:
            rf_subject_missed_by_s1.append(rel)
            if s1:
                s1_no_match_but_found.append(rel)
            else:
                s1_zero_faces.append(rel)

    n = rf_subject_images
    print(f"Images with a RetinaFace-native subject face: {n}/{len(data)}")
    print(f"\n[A] Cross-detector subject recall (S1 vs RF-native, IoU>=0.4):")
    print(f"    {s1_matches_rf}/{n} = {s1_matches_rf/n*100:.1f}%")
    print(f"\n[B] Triage of the {len(rf_subject_missed_by_s1)} disagreements:")
    print(f"    S1 found ZERO faces:            {len(s1_zero_faces)} -> {s1_zero_faces}")
    print(f"    S1 found faces but wrong place: {len(s1_no_match_but_found)} -> {s1_no_match_but_found}")
    print(f"\n[C] S1 face precision signal:")
    print(f"    {s1_faces_confirmed}/{s1_faces_total} S1 faces confirmed by RetinaFace "
          f"({s1_faces_confirmed/s1_faces_total*100:.0f}%)")
    print(f"    Unconfirmed S1 faces ({len(s1_faces_unconfirmed)}) — FP risk OR RF blind spot:")
    for rel, b in s1_faces_unconfirmed:
        print(f"      {rel.split('/')[-1]}: box={b[:4]}")

    (OUT_DIR / "consensus_summary.json").write_text(json.dumps({
        "rf_subject_images": rf_subject_images,
        "s1_matches_rf": s1_matches_rf,
        "cross_detector_subject_recall": round(s1_matches_rf / n, 4) if n else None,
        "s1_zero_faces_when_rf_finds": s1_zero_faces,
        "s1_wrong_place_when_rf_finds": s1_no_match_but_found,
        "s1_faces_confirmed": s1_faces_confirmed,
        "s1_faces_total": s1_faces_total,
    }, indent=1))
    print("\n-> consensus_summary.json")


if __name__ == "__main__":
    main()
