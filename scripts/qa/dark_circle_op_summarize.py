"""Summarise `test_output/dark_circle_op_study/study.json` (dark_circle_op_study.py)
into the tables used by RESEARCH_DARK_CIRCLE_OP_2026_09_02.md.

    python3 scripts/qa/dark_circle_op_summarize.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
J = ROOT / "test_output" / "dark_circle_op_study" / "study.json"


def q(a, p):
    a = np.asarray([x for x in a if x is not None], float)
    return float(np.percentile(a, p)) if a.size else float("nan")


def fmt(a):
    return f"med {q(a, 50):.2f}, p10 {q(a, 10):.2f}, p90 {q(a, 90):.2f}, max {max(x for x in a if x is not None):.2f}"


def main() -> int:
    rows = json.loads(J.read_text())
    scored = [r for r in rows if r.get("faces") and not r.get("excluded") and r.get("eyes")]
    excluded = [r["image"] for r in rows if r.get("excluded")]
    nof = [r["image"] for r in rows if not r.get("faces")]
    eyes = [(r["image"], s, e) for r in scored for s, e in r["eyes"].items() if e]
    print(f"images {len(rows)}  scored {len(scored)}  excluded {len(excluded)} {excluded}  no-face {nof}  eyes {len(eyes)}")

    g = [e["geom"] for _, _, e in eyes]
    print("\n## geometry (per eye)")
    print("polygon height / IED:", fmt([x["poly_h_over_ied"] for x in g]))
    print("polygon area / IED^2:", fmt([x["poly_area_over_ied2"] for x in g]))
    ied_err = [abs(x["ied_est_from_mask"] - r["ied"]) / r["ied"] for r in scored for s, e in r["eyes"].items() if e for x in [e["geom"]]]
    print("IED-from-mask relative error:", fmt(ied_err))

    print("\n## v1 vs v2 at strength 1.0 (|dL|, L in 0-255)")
    for k in ("mass_lash_frac", "mean_lash", "mean_skin", "p95_skin", "max"):
        print(f"{k:16s} v1: {fmt([e['v1'][k] for _, _, e in eyes])}")
        print(f"{'':16s} v2: {fmt([e['v2'][k] for _, _, e in eyes])}")
    print("v2 eye px changed:", sum(e["v2"]["eye_px_changed"] for _, _, e in eyes),
          " outside-support px changed:", sum(e["v2"]["outside_support_changed"] for _, _, e in eyes))
    v1_fires = sum(1 for _, _, e in eyes if e["v1"]["max"] > 0.5)
    v2_fires = sum(1 for _, _, e in eyes if e["v2"]["mean_skin"] > 1.0)
    print(f"v1 fires (max|dL|>0.5): {v1_fires}/{len(eyes)}   v2 mean skin lift > 1 L: {v2_fires}/{len(eyes)}")

    print("\n## v2 signal")
    sig = [e["signal"] for _, _, e in eyes if e["signal"].get("ref_L") is not None]
    print(f"ring found: {len(sig)}/{len(eyes)}")
    print("ring px:", fmt([s["ring_px"] for s in sig]))
    print("ref L:", fmt([s["ref_L"] for s in sig]))
    print("rel darkness (median over support):", fmt([s["rel_dark_med"] for s in sig]))
    print("rel darkness (p90 over support):", fmt([s["rel_dark_p90"] for s in sig]))
    print("shadow depth L (ref - median lp L):", fmt([s["shadow_L_med"] for s in sig]))
    print("shadow b (ref_b - median lp b; + = under-eye is bluer):", fmt([s["shadow_b_med"] for s in sig]))
    print("weight mean over support:", fmt([s["weight_mean"] for s in sig]))
    print("fraction of support with weight > 0.5:", fmt([s["weight_frac_gt_half"] for s in sig]))
    wm = np.array([s["weight_mean"] for s in sig])
    print(f"eyes with weight_mean < 0.05 (op ~off): {(wm < 0.05).sum()}   0.05-0.3: {((wm >= 0.05) & (wm < 0.3)).sum()}   >= 0.3: {(wm >= 0.3).sum()}")

    print("\n## tone probe (image x 0.55)")
    tp = [e["tone_probe"] for _, _, e in eyes]
    print("L scale realised:", fmt([t["L_scale"] for t in tp]))
    v1n = np.array([e["v1"]["mean_lash"] + e["v1"]["mean_skin"] for _, _, e in eyes])
    v1d = np.array([t["v1_mean_lash_dark"] + t["v1_mean_skin_dark"] for t in tp])
    v2n = np.array([e["v2"]["mean_skin"] for _, _, e in eyes])
    v2d = np.array([t["v2_mean_skin_dark"] for t in tp])
    ok = v1n > 0.05
    print(f"v1 lift retained on dark copy (dark/normal, eyes where v1 fired): med {np.median(v1d[ok] / v1n[ok]):.2f}")
    ok2 = v2n > 1.0
    print(f"v2 lift retained on dark copy (dark/normal absolute L): med {np.median(v2d[ok2] / v2n[ok2]):.2f}  "
          f"(expected ~L_scale {np.median([t['L_scale'] for t in tp]):.2f} for a tone-invariant relative op)")
    ls = np.array([t["L_scale"] for t in tp])
    print(f"v2 relative retention (dark/normal ÷ L_scale): med {np.median((v2d[ok2] / v2n[ok2]) / ls[ok2]):.2f}")
    v1_fire_dark = int((v1d > 0.05).sum())
    print(f"v1 fires on dark copy: {v1_fire_dark}/{len(eyes)} (normal: {int(ok.sum())}/{len(eyes)})")

    print("\n## per-image v2 (strength 1) ranked by mean skin lift, L/R")
    per = sorted(scored, key=lambda r: -max((r["eyes"][s] or {"v2": {"mean_skin": 0}})["v2"]["mean_skin"] for s in r["eyes"]))
    for r in per[:12] + [{"image": "..."}] + per[-6:]:
        if r["image"] == "...":
            print("..."); continue
        e = r["eyes"]
        def f(s, k1, k2):
            return "-" if not e.get(s) else e[s][k1][k2]
        print(f"{r['image']:9s} ied={r['ied']:5.0f} v2 skin={f('L','v2','mean_skin')}/{f('R','v2','mean_skin')} "
              f"w={f('L','signal','weight_mean')}/{f('R','signal','weight_mean')} shadowL={f('L','signal','shadow_L_med')}/{f('R','signal','shadow_L_med')} "
              f"| v1 lash%={f('L','v1','mass_lash_frac')}/{f('R','v1','mass_lash_frac')} skin={f('L','v1','mean_skin')}/{f('R','v1','mean_skin')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
