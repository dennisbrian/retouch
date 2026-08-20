"""Third-discriminator probe on the 4 veto-surviving posters.

The joint rule kills 14/18 poster FPs with 0 real-face collateral; 4 survive
(coverage >= 0.978 — physically on/near the person). This probes two
candidate discriminators on exactly those 4 + 8 real-face controls:

  D1 HEMOGLOBIN MAP VARIANCE (chromophore decompose_chromophores):
     real skin has hemoglobin spatial structure (blush, vessels, lips);
     poster paint is uniform. Metric: std of Hb inside the central crop at
     fixed geometry; real faces should have a floor.

  D2 SKIN-PIXEL FRACTION (BiSeNet-parse skin mask... not available here —
     use chromophore-based melanin/Hb plausibility instead): fraction of
     pixels whose (melanin, Hb) fall in plausible-skin ranges.

    .venv/bin/python scripts/qa/posterfp_survivors.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT))

from retouch.chromophore import decompose_chromophores  # noqa: E402

OUT_DIR = ROOT / "test_output" / "detection_recall_study"

SURVIVORS = [
    ("DSCF4588.jpg", [524, 155, 758, 904]),
    ("DSCF4612.jpg", [807, 1864, 393, 423]),
    ("DSCF4617.jpg", [819, 1876, 399, 423]),
    ("DSCF4618.jpg", [652, 438, 838, 901]),
]
# real-face controls: RF subjects on the same 4 images
CONTROLS = [
    ("DSCF4588.jpg", None),
    ("DSCF4612.jpg", None),
    ("DSCF4617.jpg", None),
    ("DSCF4618.jpg", None),
]


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


def main():
    rf = json.loads((OUT_DIR / "retinaface.json").read_text())

    print(f"{'case':24s} {'Hb std':>8s} {'Hb med':>8s} {'mel std':>8s} {'skinlike':>9s}")
    rows = []
    for rel, box in SURVIVORS:
        img = cv2.imread(str(ROOT / "test_output" / rel))
        crop = central(img, box)
        if crop is None:
            continue
        mel, hb = decompose_chromophores(crop.astype(np.float32) / 255.0)
        hb_std = float(np.std(hb))
        hb_med = float(np.median(hb))
        mel_std = float(np.std(mel))
        # plausibility: Hb in [0.05, 0.6], melanin in [0.02, 0.5] fraction
        skinlike = float(((hb > 0.05) & (hb < 0.6) & (mel > 0.02) & (mel < 0.5)).mean())
        rows.append((rel, "SURVIVOR", hb_std, hb_med, mel_std, skinlike))
        print(f"{rel+' [poster]':24s} {hb_std:8.3f} {hb_med:8.3f} {mel_std:8.3f} {skinlike:9.3f}")

    for rel, _ in CONTROLS:
        rf_native = rf["test_output/" + rel]["native"]
        subj = max(rf_native, key=lambda b: b[4])
        img = cv2.imread(str(ROOT / "test_output" / rel))
        crop = central(img, subj[:4])
        if crop is None:
            continue
        mel, hb = decompose_chromophores(crop.astype(np.float32) / 255.0)
        hb_std = float(np.std(hb))
        hb_med = float(np.median(hb))
        mel_std = float(np.std(mel))
        skinlike = float(((hb > 0.05) & (hb < 0.6) & (mel > 0.02) & (mel < 0.5)).mean())
        rows.append((rel, "REAL", hb_std, hb_med, mel_std, skinlike))
        print(f"{rel+' [subject]':24s} {hb_std:8.3f} {hb_med:8.3f} {mel_std:8.3f} {skinlike:9.3f}")


if __name__ == "__main__":
    main()
