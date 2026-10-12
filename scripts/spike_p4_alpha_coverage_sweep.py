#!/usr/bin/env python3
"""P4 coverage-fraction breakdown sweep — evidence for PLAN_P4_MAKEUP_UNMIX.md §19.

Hypothesis: the remediated alpha cues are per-face median/MAD *outlier*
detectors, so detection collapses once makeup covers a non-minority share of
the skin mask (the makeup pixels shift the median/MAD themselves). Real
foundation covers most of the face, so this is the primary-use-case regime.

Sweeps disk coverage fraction for the tone-matched light foundation
(S*1.18 + warm, true alpha 0.5, same construction as the sect-14.5 probe)
across Fitzpatrick I-VI, with and without mottling, both noise levels.

Run: python3 scripts/spike_p4_alpha_coverage_sweep.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retouch.makeup_unmix import unmix_makeup  # noqa: E402

H = W = 128
TONES = {
    "I": (189, 208, 244),
    "II": (143, 180, 231),
    "III": (109, 152, 208),
    "IV": (87, 114, 165),
    "V": (62, 82, 122),
    "VI": (38, 51, 80),
}
# coverage fraction = pi r^2 / (H W)
RADII = [16, 24, 32, 40, 48, 56, 64]


def main() -> int:
    mask = np.ones((H, W), np.float32)
    yy, xx = np.ogrid[:H, :W]
    fr = ["%4.0f%%" % (100 * np.pi * r * r / (H * W)) for r in RADII]
    print("recovered mean alpha inside disk (true 0.5), fnd_light, sigma=4 noise")
    print(f"{'tone':<5}" + "".join(f"{f:>7}" for f in fr))
    for tone, bgr in TONES.items():
        rng = np.random.RandomState(42)
        base = np.zeros((H, W, 3), np.float32) + np.array(bgr, np.float32)
        noise = rng.normal(0, 4, (H, W, 3)).astype(np.float32)
        bare = np.clip(base + noise, 0, 255)
        mcol = np.clip(np.array(bgr, np.float32) * 1.18 + np.array([0, 8, 12]), 0, 255)
        row = f"{tone:<5}"
        for r in RADII:
            disk = ((yy - H // 2) ** 2 + (xx - W // 2) ** 2) <= r * r
            img = bare.copy()
            img[disk] = 0.5 * img[disk] + 0.5 * mcol
            img = np.clip(img, 0, 255).astype(np.uint8)
            _, _, a = unmix_makeup(img, mask)
            row += f"{float(a[disk].mean()):7.3f}"
        print(row)
    return 0


if __name__ == "__main__":
    sys.exit(main())
