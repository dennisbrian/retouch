#!/usr/bin/env python3
"""P4 dark-skin audit probe — evidence for PLAN_P4_MAKEUP_UNMIX.md §14.

Runs the *shipped* solver (retouch/makeup_unmix.py) across a Fitzpatrick
I–VI-representative tone palette in three scenarios:

  1. bare noisy skin       — false-positive check (alpha should be ~0)
  2. tone-matched foundation disk (true alpha=0.5) — detection sensitivity
  3. specular highlight on bare skin + coverage_even 0.7 — damage check

This is the "dark-skin residual regression test" the plan's §13.5 inventory
listed as missing. It is a characterization probe, not a pass/fail gate:
as of 2026-07-15 the shipped solver FAILS the plan's own NO-GO criteria on
V–VI (see plan §14 for the measured table + mechanism attribution).

Run: python3 scripts/spike_p4_darkskin_probe.py
Exit 0 always (reporting tool).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retouch.makeup_unmix import (  # noqa: E402
    apply_makeup_coverage_even,
    estimate_makeup_alpha,
    unmix_makeup,
)

H = W = 96

# Fitzpatrick-representative sRGB swatches (I..VI), stored BGR.
TONES = {
    "I  (244,208,189)": (189, 208, 244),
    "II (231,180,143)": (143, 180, 231),
    "III(208,152,109)": (109, 152, 208),
    "IV (165,114, 87)": (87, 114, 165),
    "V  (122, 82, 62)": (62, 82, 122),
    "VI ( 80, 51, 38)": (38, 51, 80),
}


def main() -> int:
    rng = np.random.RandomState(42)
    mask = np.ones((H, W), np.float32)
    yy, xx = np.ogrid[:H, :W]
    disk = ((yy - H // 2) ** 2 + (xx - W // 2) ** 2) <= 24 ** 2
    spot = ((yy - H // 2) ** 2 + (xx - W // 2) ** 2) <= 8 ** 2

    print(
        f"{'tone':<18} {'bareA':>6} {'bareA_e2e':>9} {'fndA_in':>8}"
        f" {'fndA_out':>8} {'specA':>6} {'dmg_out':>8}"
    )
    for name, bgr in TONES.items():
        base = np.zeros((H, W, 3), np.float32) + np.array(bgr, np.float32)
        noise = rng.normal(0, 4, (H, W, 3)).astype(np.float32)
        bare = np.clip(base + noise, 0, 255).astype(np.uint8)

        # 1. bare skin: multi-cue init alpha + end-to-end IRLS alpha
        a_init = estimate_makeup_alpha(bare, mask)
        _, _, a_e2e = unmix_makeup(bare, mask)

        # 2. tone-matched foundation (~18% lighter + warm shift), alpha=0.5 disk
        mcol = np.clip(
            np.array(bgr, np.float32) * 1.18 + np.array([0, 8, 12]), 0, 255
        )
        fnd = bare.astype(np.float32).copy()
        fnd[disk] = 0.5 * fnd[disk] + 0.5 * mcol
        fnd = np.clip(fnd, 0, 255).astype(np.uint8)
        _, _, a_f = unmix_makeup(fnd, mask)

        # 3. specular highlight on bare skin + coverage-even damage outside spot
        spec = bare.astype(np.float32).copy()
        spec[spot] = 0.25 * spec[spot] + 0.75 * np.array(
            [235, 238, 240], np.float32
        )
        spec = np.clip(spec, 0, 255).astype(np.uint8)
        _, _, a_s = unmix_makeup(spec, mask)
        out = apply_makeup_coverage_even(spec, mask, 0.7)
        dmg = float(
            np.abs(out.astype(np.float32) - spec.astype(np.float32))[~spot].max()
        )

        print(
            f"{name:<18} {a_init.mean():6.3f} {float(a_e2e.mean()):9.3f}"
            f" {float(a_f[disk].mean()):8.3f} {float(a_f[~disk].mean()):8.3f}"
            f" {float(a_s[spot].mean()):6.3f} {dmg:8.1f}"
        )

    print(
        "\ncols: bareA=init alpha on bare skin | bareA_e2e=IRLS alpha bare |"
        " fndA_in/out=recovered alpha inside/outside 0.5-alpha foundation disk |"
        " specA=alpha on specular spot |"
        " dmg_out=max |out-in| OUTSIDE spot after coverage_even 0.7"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
