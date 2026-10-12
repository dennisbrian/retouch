"""Pass 2 of the post-epsilon re-audit: under-eye darkness statistics + two
ops the first pass skipped (2026-09-02).

For each corpus face (2048 proxy, cached contexts):
  * under-eye darkness as the shipped detector sees it: surround-median L minus
    under-eye L, the fraction of under-eye pixels that clear the fixed
    ``threshold_offset=15`` (absolute L units) and the largest connected
    component that survives the 100 px area filter — i.e. whether
    ``UndereyeAnalyzer.detect_dark_circles`` can fire at all on this face;
  * ``dark_circles=100`` isolated render: mae / p99 inside the under-eye hull;
  * ``shine_removal=60`` isolated render: mae / p99 inside the skin hull
    (08-31 audit deferred its verdict to a corpus-wide result).

    RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python \
        scripts/qa/post_epsilon_pass2_undereye_shine.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("RETOUCH_GPU", "0")
os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "qa"))

from post_epsilon_faceop_sweep import (  # noqa: E402
    ALL_OFF, CORPUS, FACE_OVAL, LEFT_UNDER_EYE, RIGHT_UNDER_EYE,
    absdiff, hull, lms_of, shrink, stats, u8,
)
from retouch.engine import RetouchEngine  # noqa: E402

OUT = ROOT / "test_output" / "post_epsilon_faceop_sweep"


def ue_darkness(img, lms, w, h):
    """Replicate UndereyeAnalyzer.detect_dark_circles' measurement per eye."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L = lab[:, :, 0]
    out = {}
    for side, idx in (("L", LEFT_UNDER_EYE), ("R", RIGHT_UNDER_EYE)):
        m = hull(lms, idx, w, h).astype(np.float32)
        if m.sum() < 20:
            out[side] = None
            continue
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        surround = np.clip(cv2.dilate((m > 0.3).astype(np.uint8), k).astype(np.float32) - m, 0, 1)
        sp = L[surround > 0.3]
        med = float(np.median(sp)) if len(sp) >= 20 else float(np.median(L[m > 0.3]))
        ue = L[m > 0.3]
        dark = ((L < med - 15.0) * m > 0.5).astype(np.uint8)
        n, labels, st, _ = cv2.connectedComponentsWithStats(dark)
        big = int(max([st[i, cv2.CC_STAT_AREA] for i in range(1, n)], default=0))
        out[side] = {"surround_med_L": round(med, 1), "ue_med_L": round(float(np.median(ue)), 1),
                     "ue_p10_L": round(float(np.percentile(ue, 10)), 1),
                     "darkness_med": round(med - float(np.median(ue)), 1),
                     "frac_below_thr15": round(float((ue < med - 15).mean()), 4),
                     "frac_below_8pct": round(float((ue < med * 0.92).mean()), 4),
                     "largest_cc_px": big, "fires": big >= 100, "ue_px": int(m.sum())}
    return out


def main() -> int:
    paths = [ROOT / l.strip() for l in CORPUS.read_text().splitlines() if l.strip()]
    eng = RetouchEngine()
    rows = []
    t_all = time.time()
    try:
        for p in paths:
            if not p.exists():
                continue
            img = shrink(cv2.imread(str(p)))
            h, w = img.shape[:2]
            t0 = time.time()
            res = eng.process(img, recipe="natural", **ALL_OFF)
            ctx = res.face_contexts or []
            if not ctx:
                rows.append({"image": p.stem, "faces": 0})
                continue
            base = u8(res)
            fc = max(ctx, key=lambda f: f.face_data.bbox[2] * f.face_data.bbox[3])
            lms = lms_of(fc)
            ied = int(fc.face_data.ied)
            skin_hull = hull(lms, FACE_OVAL, w, h, int(ied * 0.9))
            ue_hull = hull(lms, LEFT_UNDER_EYE, w, h, int(ied * 0.3)) | hull(lms, RIGHT_UNDER_EYE, w, h, int(ied * 0.3))
            rec = {"image": p.stem, "faces": len(ctx), "ied": ied, "undereye": ue_darkness(img, lms, w, h), "ops": {}}
            for op, val, mask in (("dark_circles", 100, ue_hull), ("shine_removal", 60, skin_hull), ("wrinkle_soften", 60, skin_hull)):
                kw = dict(ALL_OFF)
                kw[op] = val
                out = u8(eng.process(img, recipe="natural", face_contexts=ctx, **kw))
                d = absdiff(out, base)
                m, p99, fr = stats(d, mask)
                rec["ops"][op] = {"value": val, "mae_region": round(m, 4), "p99_region": round(p99, 2),
                                  "frac_gt6_region": round(fr, 4), "max_full": int(d.max()), "n_gt6_full": int((d > 6).sum())}
            rows.append(rec)
            ue = rec["undereye"]
            print(f"{p.stem:9s} ue_dark L/R={ue.get('L', {}) and ue['L']['darkness_med']}/{ue.get('R', {}) and ue['R']['darkness_med']}"
                  f" fires={ue.get('L', {}) and ue['L']['fires']}/{ue.get('R', {}) and ue['R']['fires']}"
                  f" | " + " ".join(f"{k}={v['mae_region']:.3f}/p99={v['p99_region']:.0f}" for k, v in rec["ops"].items())
                  + f"  ({time.time()-t0:.0f}s)", flush=True)
            (OUT / "pass2.json").write_text(json.dumps(rows, indent=1))
    finally:
        close = getattr(eng, "close", None)
        if callable(close):
            close()
    (OUT / "pass2.json").write_text(json.dumps(rows, indent=1))
    print(f"DONE pass2 {len(rows)} in {time.time()-t_all:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
