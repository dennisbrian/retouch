#!/usr/bin/env python3
"""P4 real-photo internal-consistency probe — PLAN_P4_MAKEUP_UNMIX.md §19.

No ground truth on real photos, so checks are internal:
  - does unmix fire at all (alpha stats over the parsed skin mask)?
  - is alpha spatially plausible (makeup regions vs noise speckle)?
  - identity: |recompose(S, M, alpha) - I| (should be ~0 by construction)
  - product damage: |apply_makeup_coverage_even(I, 0.5) - I| on skin

Saves test_output/spike_p4_alpha_real_<name>.png:
  input | alpha heatmap | S_rec | coverage_even 0.5 output | 8x |delta|

Run: python3 scripts/spike_p4_alpha_real.py [image ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retouch.detection import FaceDetector  # noqa: E402
from retouch.parsing import FaceParser  # noqa: E402
from retouch.makeup_unmix import (  # noqa: E402
    apply_makeup_coverage_even,
    recompose,
    unmix_makeup,
)

DEFAULTS = [
    ROOT / "test_output" / "DSCF4454.jpg",
    ROOT / "test_output" / "DSCF4550.jpg",
]


def heat(a: np.ndarray) -> np.ndarray:
    return cv2.applyColorMap(
        np.clip(a * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO
    )


def main() -> int:
    paths = [Path(p) for p in sys.argv[1:]] or DEFAULTS
    det = FaceDetector()
    parser = FaceParser()
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"{path.name}: unreadable, skipped")
            continue
        scale = 1600.0 / max(img.shape[:2])
        if scale < 1.0:
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        faces = det.detect(img)
        if not faces:
            print(f"{path.name}: no face")
            continue
        f = faces[0]
        regions = parser.parse(f.landmarks, img, f.bbox, ied=f.ied)
        skin = regions.skin
        if skin is None or skin.max() < 0.01:
            print(f"{path.name}: no skin mask")
            continue
        # Crop to face bbox with margin for readable panels
        x, y, w, h = f.bbox
        m = int(0.35 * max(w, h))
        x0, y0 = max(0, x - m), max(0, y - m)
        x1, y1 = min(img.shape[1], x + w + m), min(img.shape[0], y + h + m)
        crop = img[y0:y1, x0:x1]
        skin_c = skin[y0:y1, x0:x1]

        I = crop.astype(np.float32)
        S, M, a = unmix_makeup(crop, skin_c)
        recon = recompose(S, M, a)
        recon_err = float(np.abs(recon - I).mean())
        out = apply_makeup_coverage_even(crop, skin_c, 0.5)
        delta = np.abs(out.astype(np.float32) - I)
        sk = skin_c > 0.5
        print(f"{path.name}: skin px={int(sk.sum())} "
              f"alpha mean={a[sk].mean():.4f} p99={np.percentile(a[sk], 99):.3f} "
              f"frac(a>0.3)={(a[sk] > 0.3).mean():.4f} "
              f"recon_err={recon_err:.2f} "
              f"cov_even0.5 delta mean={delta[sk].mean():.2f} max={delta[sk].max():.0f}")
        panel = np.hstack([
            crop,
            heat(a),
            np.clip(S, 0, 255).astype(np.uint8),
            np.clip(out, 0, 255).astype(np.uint8),
            np.clip(delta.mean(axis=-1, keepdims=True) * 8, 0, 255)
            .astype(np.uint8).repeat(3, axis=-1),
        ])
        ph = 300
        panel = cv2.resize(panel, (int(panel.shape[1] * ph / panel.shape[0]), ph))
        outp = ROOT / "test_output" / f"spike_p4_alpha_real_{path.stem}.png"
        cv2.imwrite(str(outp), panel)
        print(f"  panel -> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
