"""Dark-circle op study (2026-09-02): v1 (shipped until today) vs v2.

Background: the post-epsilon re-audit (RESEARCH_POST_EPSILON_FACEOP_REAUDIT
§6) found `dark_circles` effectively inert — the detector selected the lower
lash line (the darkest component of a polygon whose top edge *is* the lash
contour) and the lift was capped twice (≤ 9 L at strength 100). This script
measures, per eye, on the 83-face DSCF corpus (2048 proxy, cached contexts,
real parser masks):

  * geometry: under-eye polygon height / IED and area / IED² (to calibrate the
    IED-from-mask fallback and the downward extension);
  * v1 (verbatim replica of the pre-2026-09-02 `UndereyeProcessor.process`):
    where the |ΔL| mass lands — inside the dilated eye contour ("lash zone")
    vs on under-eye skin — and how large it is;
  * v2 (shipped `UndereyeProcessor.process`): same measures, plus the
    relative-darkness signal (ring-median L vs low-pass under-eye L) and the
    mean weight, so the (T0, T1) band can be read off the corpus;
  * tone probe: both detectors on a darkened copy of the image (BGR × 0.55),
    to show the absolute 15-L threshold under-fires while the relative band
    holds.

Sheets (`sheets/<stem>.jpg`): base | v1@100 | v2@25 | v2@60 | v2@100 | heat
v2@60 ×8 | v2 weight·support.

    RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python \
        scripts/qa/dark_circle_op_study.py [--images A,B] [--sheets-only]
"""
from __future__ import annotations

import argparse
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

from post_epsilon_faceop_sweep import ALL_OFF, CORPUS, lms_of, shrink, u8  # noqa: E402
from retouch import undereye as ue_mod  # noqa: E402
from retouch.engine import RetouchEngine  # noqa: E402
from retouch.parsing import LEFT_EYE, LEFT_UNDER_EYE, RIGHT_EYE, RIGHT_UNDER_EYE  # noqa: E402
from retouch.utils import blend_masked, create_polygon_mask, get_points  # noqa: E402

OUT = ROOT / "test_output" / "dark_circle_op_study"
SHEETS = OUT / "sheets"
# Corpus hygiene list from the post-epsilon re-audit §7: largest detection is
# a hand/background person or has degenerate landmarks. Recorded, not scored.
EXCLUDE = {"DSCF4589", "DSCF4618", "DSCF4596", "DSCF4597", "DSCF4599",
           "DSCF4590", "DSCF4592", "DSCF4600", "DSCF4554", "DSCF4555"}
ANCHORS = ["DSCF6693", "DSCF4463", "DSCF4576", "DSCF4454", "DSCF4503", "DSCF4612", "DSCF6961", "DSCF8007"]


# --- v1 replica (pre-2026-09-02 process(), verbatim semantics) ---------------
def v1_process(img_bgr, mask, strength):
    an, rm = ue_mod.UndereyeAnalyzer(), ue_mod.UndereyeRemover()
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    det, med = an.detect_dark_circles(lab, mask, threshold_offset=15.0)
    if np.isnan(med):
        return img_bgr
    detf = rm.feather_edges(det, feather_radius=5)
    L = lab[:, :, 0].copy()
    darkness = np.clip(med - L, 0.0, 30.0)
    lab[:, :, 0] = np.clip(L + darkness * detf * strength * (30.0 / 100.0), 0, 255)  # the double cap
    res = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return blend_masked(img_bgr, res, mask)


def lab_L(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)


def zone_stats(dL, lash, skin_zone):
    tot = float(dL.sum()) + 1e-6
    return {
        "mass_lash_frac": round(float(dL[lash].sum()) / tot, 3),
        "mean_lash": round(float(dL[lash].mean()) if lash.any() else 0.0, 2),
        "mean_skin": round(float(dL[skin_zone].mean()) if skin_zone.any() else 0.0, 2),
        "p95_skin": round(float(np.percentile(dL[skin_zone], 95)) if skin_zone.any() else 0.0, 2),
        "max": round(float(dL.max()), 2),
        "n_gt1": int((dL > 1).sum()),
    }


def eye_metrics(img, regions, lms, ied, w, h, side):
    ue_mask = regions.left_under_eye if side == "L" else regions.right_under_eye
    eye_idx = LEFT_EYE if side == "L" else RIGHT_EYE
    ue_idx = LEFT_UNDER_EYE if side == "L" else RIGHT_UNDER_EYE
    if ue_mask is None or ue_mask.max() < 0.01:
        return None
    eye_hull = create_polygon_mask(get_points(lms, eye_idx, w, h), (h, w), 0)
    lash = cv2.dilate((eye_hull > 0.5).astype(np.uint8), ue_mod._ellipse(max(int(ied * 0.08), 1))) > 0
    poly = (ue_mask > 0.5)
    ys = np.nonzero(poly.any(1))[0]
    geom = {"poly_h_over_ied": round(float(ys.max() - ys.min() + 1) / ied, 3) if ys.size else 0.0,
            "poly_area_over_ied2": round(float(poly.sum()) / ied ** 2, 4),
            "ied_est_from_mask": round(ue_mod.estimate_ied_from_mask(ue_mask), 1)}

    proc = ue_mod.UndereyeProcessor()
    support, ring, valid = ue_mod.build_undereye_support(ue_mask, ied, exclude=eye_hull, skin=regions.skin)
    skin_zone = support > 0.5
    L0 = lab_L(img)

    # v2 signal (same code path as process)
    ys2, xs2 = np.nonzero((support > 0) | ring)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    weight, lab_lp, ref = proc.analyzer.analyze(lab, support, ring, valid, ied)
    if ref is None:
        sig = {"ref_L": None}
    else:
        rel = (ref[0] - lab_lp[:, :, 0]) / max(float(ref[0]), 1.0)
        sig = {"ref_L": round(float(ref[0]), 1), "ref_a": round(float(ref[1]), 1), "ref_b": round(float(ref[2]), 1),
               "ring_px": int(ring.sum()),
               "rel_dark_med": round(float(np.median(rel[skin_zone])), 4) if skin_zone.any() else None,
               "rel_dark_p90": round(float(np.percentile(rel[skin_zone], 90)), 4) if skin_zone.any() else None,
               "weight_mean": round(float((weight * support)[skin_zone].mean()), 3) if skin_zone.any() else 0.0,
               "weight_frac_gt_half": round(float((weight[skin_zone] > 0.5).mean()), 3) if skin_zone.any() else 0.0,
               "shadow_L_med": round(float(ref[0] - np.median(lab_lp[:, :, 0][skin_zone])), 2) if skin_zone.any() else None,
               "shadow_b_med": round(float(ref[2] - np.median(lab_lp[:, :, 2][skin_zone])), 2) if skin_zone.any() else None}

    out = {"geom": geom, "signal": sig, "v1": {}, "v2": {}}
    v1 = v1_process(img, ue_mask, 1.0)
    out["v1"] = zone_stats(np.abs(lab_L(v1) - L0), lash, skin_zone)
    v2 = proc.process(img, ue_mask, darken_removal_strength=1.0, ied=ied, exclude=eye_hull, skin=regions.skin)
    out["v2"] = zone_stats(np.abs(lab_L(v2) - L0), lash, skin_zone)
    out["v2"]["eye_px_changed"] = int(((v2 != img).any(-1) & (eye_hull > 0.5)).sum())
    out["v2"]["outside_support_changed"] = int(((v2 != img).any(-1) & (support == 0)).sum())

    # tone probe: darken the image, re-run both at strength 1
    dark = np.clip(img.astype(np.float32) * 0.55, 0, 255).astype(np.uint8)
    Ld = lab_L(dark)
    v1d = v1_process(dark, ue_mask, 1.0)
    v2d = proc.process(dark, ue_mask, darken_removal_strength=1.0, ied=ied, exclude=eye_hull, skin=regions.skin)
    out["tone_probe"] = {
        "v1_mean_skin_dark": round(float(np.abs(lab_L(v1d) - Ld)[skin_zone].mean()) if skin_zone.any() else 0.0, 2),
        "v1_mean_lash_dark": round(float(np.abs(lab_L(v1d) - Ld)[lash].mean()) if lash.any() else 0.0, 2),
        "v2_mean_skin_dark": round(float(np.abs(lab_L(v2d) - Ld)[skin_zone].mean()) if skin_zone.any() else 0.0, 2),
        "L_scale": round(float(Ld[skin_zone].mean() / max(L0[skin_zone].mean(), 1)), 3) if skin_zone.any() else None,
    }
    return out


def sheet(stem, img, regions, lms, ied, w, h, fc):
    proc = ue_mod.UndereyeProcessor()
    tiles = {"base": img}
    v1 = img.copy()
    for side in ("L", "R"):
        m = regions.left_under_eye if side == "L" else regions.right_under_eye
        if m is not None and m.max() > 0.01:
            v1 = v1_process(v1, m, 1.0)
    tiles["v1@100"] = v1
    wsum = np.zeros((h, w), np.float32)
    for s in (0.25, 0.6, 1.0):
        cur = img.copy()
        for side in ("L", "R"):
            m = regions.left_under_eye if side == "L" else regions.right_under_eye
            idx = LEFT_EYE if side == "L" else RIGHT_EYE
            if m is None or m.max() < 0.01:
                continue
            eye_hull = create_polygon_mask(get_points(lms, idx, w, h), (h, w), 0)
            cur = proc.process(cur, m, darken_removal_strength=s, ied=ied, exclude=eye_hull, skin=regions.skin)
            if s == 1.0:
                sup, ring, valid = ue_mod.build_undereye_support(m, ied, exclude=eye_hull, skin=regions.skin)
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
                wgt, _, ref = proc.analyzer.analyze(lab, sup, ring, valid, ied)
                wsum = np.maximum(wsum, wgt * sup)
        tiles[f"v2@{int(s*100)}"] = cur
    d = cv2.absdiff(tiles["v2@60"], img).max(-1)
    tiles["heat v2@60 x8"] = cv2.applyColorMap(np.clip(d.astype(np.float32) * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    tiles["weight"] = cv2.cvtColor((wsum * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    # crop: eyes band
    pts = get_points(lms, LEFT_UNDER_EYE + RIGHT_UNDER_EYE + LEFT_EYE + RIGHT_EYE, w, h)
    x0, x1 = max(int(pts[:, 0].min() - 0.35 * ied), 0), min(int(pts[:, 0].max() + 0.35 * ied), w)
    y0, y1 = max(int(pts[:, 1].min() - 0.35 * ied), 0), min(int(pts[:, 1].max() + 0.6 * ied), h)
    row = []
    for name, t in tiles.items():
        c = t[y0:y1, x0:x1].copy()
        c = cv2.resize(c, (int(c.shape[1] * 300 / c.shape[0]), 300), interpolation=cv2.INTER_AREA)
        cv2.putText(c, name, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        row.append(c)
        row.append(np.full((300, 3, 3), 220, np.uint8))
    out = np.hstack(row[:-1])
    cv2.putText(out, stem, (6, 292), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(SHEETS / f"{stem}.jpg"), out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    # 2x2 grid at review size: base | v1@100 / v2@60 | v2@100
    big = []
    for name in ("base", "v1@100", "v2@60", "v2@100"):
        c = tiles[name][y0:y1, x0:x1].copy()
        c = cv2.resize(c, (int(c.shape[1] * 520 / c.shape[0]), 520), interpolation=cv2.INTER_CUBIC)
        cv2.putText(c, f"{stem} {name}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)
        big.append(c)
    grid = np.vstack([np.hstack(big[:2]), np.hstack(big[2:])])
    cv2.imwrite(str(SHEETS / f"{stem}_grid.jpg"), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="")
    ap.add_argument("--sheets-only", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    SHEETS.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / l.strip() for l in CORPUS.read_text().splitlines() if l.strip()]
    if a.images:
        keep = set(a.images.split(","))
        paths = [p for p in paths if p.stem in keep]
    # subset / sheets-only runs must not clobber the full-corpus result
    out_json = OUT / ("study.json" if not (a.images or a.sheets_only) else "study_subset.json")
    eng = RetouchEngine()
    rows = []
    t_all = time.time()
    try:
        for p in paths:
            if not p.exists():
                continue
            t0 = time.time()
            img = shrink(cv2.imread(str(p)))
            h, w = img.shape[:2]
            res = eng.process(img, recipe="natural", **ALL_OFF)
            ctx = res.face_contexts or []
            if not ctx:
                rows.append({"image": p.stem, "faces": 0})
                print(f"{p.stem}: no face", flush=True)
                continue
            fc = max(ctx, key=lambda f: f.face_data.bbox[2] * f.face_data.bbox[3])
            lms = fc.face_data.landmarks
            ied = float(fc.face_data.ied)
            regions = eng._parser.parse(lms, img, fc.face_data.bbox, None, ied)
            rec = {"image": p.stem, "faces": len(ctx), "ied": round(ied, 1), "excluded": p.stem in EXCLUDE, "eyes": {}}
            if not a.sheets_only:
                for side in ("L", "R"):
                    rec["eyes"][side] = eye_metrics(img, regions, lms, ied, w, h, side)
            if p.stem in ANCHORS or a.images:
                sheet(p.stem, img, regions, lms, ied, w, h, fc)
            rows.append(rec)
            if not a.sheets_only:
                e = rec["eyes"]
                def f(side, k1, k2):
                    v = e.get(side)
                    return "-" if not v else v[k1].get(k2)
                print(f"{p.stem:9s} ied={ied:5.0f} polyh={f('L','geom','poly_h_over_ied')} "
                      f"v1 lash%={f('L','v1','mass_lash_frac')}/{f('R','v1','mass_lash_frac')} skin={f('L','v1','mean_skin')}/{f('R','v1','mean_skin')} | "
                      f"v2 lash%={f('L','v2','mass_lash_frac')}/{f('R','v2','mass_lash_frac')} skin={f('L','v2','mean_skin')}/{f('R','v2','mean_skin')} "
                      f"w={f('L','signal','weight_mean')}/{f('R','signal','weight_mean')} shadowL={f('L','signal','shadow_L_med')}/{f('R','signal','shadow_L_med')} "
                      f"eyepx={f('L','v2','eye_px_changed')} ({time.time()-t0:.0f}s)", flush=True)
            out_json.write_text(json.dumps(rows, indent=1))
    finally:
        close = getattr(eng, "close", None)
        if callable(close):
            close()
    out_json.write_text(json.dumps(rows, indent=1))
    print(f"DONE {len(rows)} in {time.time()-t_all:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
