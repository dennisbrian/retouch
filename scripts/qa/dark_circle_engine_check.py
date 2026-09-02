"""Engine-level check for the v2 dark-circle op: a recipe that sets the
under-eye keys (default clear_skin_v1: dark_circles 0.15 / darken_removal 0.30 /
puffiness 0.20; `natural` sets none) vs the same recipe with them forced to 0, on the study anchors, through
the real face-core dispatch (IED / eye-contour / skin wiring). Writes
`test_output/dark_circle_op_study/engine/<stem>.jpg` (off | natural | heat x12)
and prints the under-eye-hull mae / p99 and the eye-hull max delta.

    RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python \\
        scripts/qa/dark_circle_engine_check.py [--images A,B]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("RETOUCH_GPU", "0")
os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "qa"))
from post_epsilon_faceop_sweep import CORPUS, hull, shrink, u8  # noqa: E402
from retouch.engine import RetouchEngine  # noqa: E402
from retouch.parsing import LEFT_EYE, LEFT_UNDER_EYE, RIGHT_EYE, RIGHT_UNDER_EYE  # noqa: E402

OUT = ROOT / "test_output" / "dark_circle_op_study" / "engine"
DEFAULT = "DSCF6693,DSCF4463,DSCF4576,DSCF4454,DSCF4503,DSCF6961"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=DEFAULT)
    ap.add_argument("--recipe", default="clear_skin_v1",
                    help="must set eyes.dark_circles / undereye.* (natural does not)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    keep = set(a.images.split(","))
    paths = [ROOT / l.strip() for l in CORPUS.read_text().splitlines() if l.strip() and Path(l.strip()).stem in keep]
    eng = RetouchEngine()
    try:
        for p in paths:
            img = shrink(cv2.imread(str(p)))
            h, w = img.shape[:2]
            on = eng.process(img, recipe=a.recipe)
            ctx = on.face_contexts or []
            if not ctx:
                print(p.stem, "no face"); continue
            off = eng.process(img, recipe=a.recipe, face_contexts=ctx, dark_circles=0, undereye_darken_removal=0, undereye_puffiness_reduction=0)
            on_u, off_u = u8(on), u8(off)
            fc = max(ctx, key=lambda f: f.face_data.bbox[2] * f.face_data.bbox[3])
            lms = fc.face_data.landmarks.landmark
            ied = int(fc.face_data.ied)
            ue = hull(lms, LEFT_UNDER_EYE, w, h, int(ied * 0.35)) | hull(lms, RIGHT_UNDER_EYE, w, h, int(ied * 0.35))
            eye = hull(lms, LEFT_EYE, w, h) | hull(lms, RIGHT_EYE, w, h)
            d = cv2.absdiff(on_u, off_u).max(-1)
            sel = d[ue.astype(bool)]
            print(f"{p.stem}: {a.recipe} vs undereye-off | ue-hull mae={sel.mean():.3f} p99={np.percentile(sel, 99):.1f} "
                  f"max={sel.max()} | eye-hull max={d[eye.astype(bool)].max()} | outside ue-hull max={d[~ue.astype(bool)].max()}", flush=True)
            x, y, bw, bh = fc.face_data.bbox
            x0, y0 = max(int(x - 0.15 * bw), 0), max(int(y - 0.1 * bh), 0)
            x1, y1 = min(int(x + 1.15 * bw), w), min(int(y + 0.75 * bh), h)
            tiles = [off_u[y0:y1, x0:x1], on_u[y0:y1, x0:x1],
                     cv2.applyColorMap(np.clip(d[y0:y1, x0:x1].astype(np.float32) * 12, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)]
            tiles = [cv2.resize(t, (int(t.shape[1] * 520 / t.shape[0]), 520), interpolation=cv2.INTER_AREA) for t in tiles]
            for t, n in zip(tiles, (f"{a.recipe}, undereye off", f"{a.recipe} (v2)", "heat x12")):
                cv2.putText(t, f"{p.stem} {n}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(OUT / f"{p.stem}_{a.recipe}.jpg"), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 92])
    finally:
        eng.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
