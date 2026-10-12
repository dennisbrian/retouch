"""Eye-occlusion gate calibration — ROC, AUC, and operating points.

Consumes ``sweep_full.json`` + ``labels.json`` from
``test_output/eye_occlusion_study/`` (produced by eye_occlusion_sweep.py and
the labeling pass) and reports, per protocol §5/§6:

  - per-signal, per-arm ROC + AUC over ``occluded`` vs ``visible``
    (``partial`` held out; ``no_face`` reported as its own stratum)
  - signal coverage (how often each signal is even available — contrast
    fails open on BiSeNet class-collapse, so coverage matters as much as AUC)
  - confusion of the *shipped* gate decision at current thresholds
  - candidate operating points at the §2 cost-asymmetry target
    (missed-gate rate on ``occluded`` ≤ 5%)

    .venv/bin/python scripts/qa/eye_occlusion_roc.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "test_output" / "eye_occlusion_study"

sweep = json.loads((OUT / "sweep_full.json").read_text())["rows"]
labels = json.loads((OUT / "labels.json").read_text())["labels"]


def rows_for(arm: str):
    """Yield (key, label, signal-dict) per eye for one arm."""
    for rel, row in sweep.items():
        if "error" in row:
            continue
        stem = Path(rel).stem
        for side in ("left", "right"):
            key = f"{stem}:{side}"
            yield key, labels[key]["label"], row[f"arm_{arm}"][side]


def auc_lower_is_positive(pos, neg):
    """AUC for 'lower value => positive class' via rank statistic."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    # score = -value so that higher score = more positive
    from itertools import product
    wins = sum(1.0 if p < n else (0.5 if p == n else 0.0) for p, n in product(pos, neg))
    return wins / (len(pos) * len(neg))


def signal_stats(arm: str, sig: str):
    vals = {"occluded": [], "visible": [], "partial": [], "no_face": []}
    missing = {k: 0 for k in vals}
    for _, lab, s in rows_for(arm):
        v = s[sig]
        if v is None:
            missing[lab] += 1
        else:
            vals[lab].append(v)
    auc = auc_lower_is_positive(vals["occluded"], vals["visible"])
    def rng(a):
        return None if not a else [round(float(min(a)), 3), round(float(max(a)), 3)]
    return {
        "auc_occluded_vs_visible": None if auc is None else round(auc, 4),
        "coverage": {k: f"{len(vals[k])}/{len(vals[k]) + missing[k]}" for k in vals},
        "range": {k: rng(vals[k]) for k in vals},
        "median": {k: (None if not vals[k] else round(float(np.median(vals[k])), 3)) for k in vals},
    }


def op_point(arm: str, sig: str, target_recall: float = 0.95):
    """Lowest threshold t ('gate if value < t') reaching occluded recall >= target."""
    occ, vis = [], []
    for _, lab, s in rows_for(arm):
        v = s[sig]
        if lab == "occluded":
            occ.append(v)  # None = signal unavailable = un-catchable by this signal
        elif lab == "visible" and v is not None:
            vis.append(v)
    n_occ = len(occ)
    have = sorted(v for v in occ if v is not None)
    best = None
    for t in np.arange(0.02, 1.30, 0.005):
        caught = sum(1 for v in have if v < t)
        if caught / n_occ >= target_recall:
            best = t
            break
    if best is None:
        max_recall = len(have) / n_occ if n_occ else 0.0
        return {"threshold": None, "note": f"unreachable: max recall {max_recall:.2f} (signal missing on {n_occ - len(have)}/{n_occ} occluded)"}
    fg = sum(1 for v in vis if v < best)
    return {
        "threshold": round(float(best), 3),
        "occluded_recall": round(sum(1 for v in have if v < best) / n_occ, 3),
        "false_gate_visible": f"{fg}/{len(vis)}",
        "false_gate_rate": round(fg / len(vis), 3) if vis else None,
    }


def shipped_gate_confusion(arm: str):
    out = {}
    for lab in ("occluded", "visible", "partial", "no_face"):
        gated = tot = 0
        missed = []
        for key, l2, s in rows_for(arm):
            if l2 != lab:
                continue
            tot += 1
            if s["gated"]:
                gated += 1
            elif lab in ("occluded", "no_face"):
                missed.append(key)
        out[lab] = {"gated": gated, "total": tot, "missed": missed if lab in ("occluded", "no_face") else None}
    return out


def joint_op(arm: str, target: float = 1.0):
    """Grid-search (EAR < t1) OR (contrast < t2), minimizing visible false gates
    subject to occluded recall >= target (contrast None = cannot fire)."""
    data = [(lab, s["ear"], s["contrast"]) for _, lab, s in rows_for(arm) if lab in ("occluded", "visible")]
    occ = [(e, c) for lab, e, c in data if lab == "occluded"]
    vis = [(e, c) for lab, e, c in data if lab == "visible"]
    best = None
    for t1 in np.arange(0.06, 0.40, 0.005):
        for t2 in np.arange(0.0, 0.9, 0.025):
            def fire(e, c):
                return (e is not None and e < t1) or (c is not None and c < t2)
            rec = sum(1 for e, c in occ if fire(e, c)) / len(occ)
            if rec < target:
                continue
            fg = sum(1 for e, c in vis if fire(e, c))
            if best is None or fg < best[0]:
                best = (fg, round(float(t1), 3), round(float(t2), 3), round(rec, 3))
    if best is None:
        return {"note": "target unreachable"}
    return {"ear_lt": best[1], "contrast_lt": best[2], "occluded_recall": best[3],
            "false_gate_visible": f"{best[0]}/{len(vis)}", "false_gate_rate": round(best[0] / len(vis), 3)}


report = {}
for arm in ("A", "B"):
    report[f"arm_{arm}"] = {
        "signals": {sig: signal_stats(arm, sig) for sig in ("ear", "contrast", "hair_over_iris")},
        "op_points_single_signal@recall>=0.95": {sig: op_point(arm, sig) for sig in ("ear", "contrast")},
        "joint_op@recall=1.00": joint_op(arm, 1.0),
        "joint_op@recall>=0.95": joint_op(arm, 0.95),
        "shipped_gate_confusion": shipped_gate_confusion(arm),
    }

# structural check: arm-B hair signal must be dead (protocol §4)
hair_b = [s["hair_over_iris"] for _, _, s in rows_for("B") if s["hair_over_iris"] is not None]
report["arm_B_hair_structural_zero"] = bool(all(v == 0.0 for v in hair_b)) if hair_b else "all None"

(OUT / "roc_by_arm.json").write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
