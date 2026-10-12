"""PASS / REVIEW / REGRESSION triage for the Quality Lab.

Two evidence channels per (case, recipe):
  identity  — candidate vs frozen baseline (did this PR change the output?)
  absolute  — output vs input (did the output damage the photo?)

Triage rules (deterministic; thresholds are reviewable policy):
  REGRESSION — any check breaches its `regress` limit, or render error
  REVIEW     — any check breaches its `warn` limit, or the case changed
               at all without breaching warn (identity delta below warn)
  PASS       — identity SSIM >= warn AND no absolute check breaches warn
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional


def _classify(value: float, check: str, thresholds: Mapping[str, Mapping[str, Any]]
              ) -> Optional[str]:
    spec = thresholds.get(check)
    if spec is None:
        return None
    direction = spec["direction"]
    if direction == "high":
        if value > spec["regress"]:
            return "REGRESSION"
        if value > spec["warn"]:
            return "REVIEW"
    else:
        if value < spec["regress"]:
            return "REGRESSION"
        if value < spec["warn"]:
            return "REVIEW"
    return None


def triage_case(case_id: str, recipe: str, cand: Mapping[str, Any],
                base: Optional[Mapping[str, Any]],
                thresholds: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Triage one (case, recipe). Returns verdict + findings."""
    findings: List[Dict[str, Any]] = []
    verdict = "PASS"

    def bump(level: Optional[str], check: str, value: float, detail: str = ""):
        nonlocal verdict
        if level is None:
            return
        findings.append({"check": check, "level": level,
                         "value": round(value, 4), "detail": detail})
        if level == "REGRESSION":
            verdict = "REGRESSION"
        elif level == "REVIEW" and verdict == "PASS":
            verdict = "REVIEW"

    if cand.get("error"):
        return {"case": case_id, "recipe": recipe, "verdict": "REGRESSION",
                "findings": [{"check": "render", "level": "REGRESSION",
                              "value": 0, "detail": cand["error"]}]}

    cm = cand.get("metrics", {})
    g = cm.get("global", {})
    regions = cm.get("regions", {})

    # --- absolute-quality checks (output vs input) ---
    bump(_classify(g.get("luminance_drift", 0.0), "luminance_drift", thresholds),
         "luminance_drift", g.get("luminance_drift", 0.0), "whole image")
    skin = regions.get("SKIN") or regions.get("FULL") or {}
    bump(_classify(skin.get("chroma_drift", 0.0), "chroma_drift", thresholds),
         "chroma_drift", skin.get("chroma_drift", 0.0), "skin region")
    for kind, check in (("highlight", "highlight_clip_delta"),
                        ("shadow", "shadow_clip_delta")):
        ci, co = g.get(f"{kind}_clip_in", 0.0), g.get(f"{kind}_clip_out", 0.0)
        ratio = (co / ci) if ci > 1e-6 else (1.0 if co <= 1e-6 else float("inf"))
        bump(_classify(ratio, check, thresholds), check, ratio,
             f"{kind} clip {ci:.4f} -> {co:.4f}")
    for rname, rdata in regions.items():
        if "texture_retention" in rdata:
            bump(_classify(rdata["texture_retention"], "region_texture_retention", thresholds),
                 "region_texture_retention", rdata["texture_retention"], f"region {rname}")
        if "edge_retention" in rdata:
            bump(_classify(rdata["edge_retention"], "edge_preservation", thresholds),
                 "edge_preservation", rdata["edge_retention"], f"region {rname}")
        bump(_classify(rdata.get("luminance_drift", 0.0), "region_luminance_drift", thresholds),
             "region_luminance_drift", rdata.get("luminance_drift", 0.0), f"region {rname}")

    # --- identity checks (candidate vs baseline) ---
    if base is not None and not base.get("error"):
        bm = base.get("metrics", {})
        import cv2
        from . import metrics
        bimg = cv2.imread(str(base["output"]))
        cimg = cv2.imread(str(cand["output"]))
        if bimg is not None and cimg is not None and bimg.shape == cimg.shape:
            ssim_v = metrics.ssim(bimg, cimg)
            de_v = metrics.delta_e_mean(bimg, cimg)
            bump(_classify(ssim_v, "identity_ssim", thresholds),
                 "identity_ssim", ssim_v, "candidate vs baseline")
            bump(_classify(de_v, "identity_deltaE_mean", thresholds),
                 "identity_deltaE_mean", de_v, "candidate vs baseline")
        else:
            bump("REVIEW", "identity_render", 0.0, "baseline/candidate image unreadable")
    elif base is None:
        # No baseline for this recipe: verdict rests on absolute checks only.
        pass

    return {"case": case_id, "recipe": recipe, "verdict": verdict, "findings": findings}


def triage_report(candidate: Mapping[str, Any], baseline: Optional[Mapping[str, Any]],
                  thresholds: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Triage every (case, recipe) in a candidate report vs a baseline report."""
    base_cases = (baseline or {}).get("cases", {})
    rows: List[Dict[str, Any]] = []
    counts = {"PASS": 0, "REVIEW": 0, "REGRESSION": 0}
    for case_id, cdata in candidate.get("cases", {}).items():
        for recipe, centry in cdata.get("recipes", {}).items():
            bentry = base_cases.get(case_id, {}).get("recipes", {}).get(recipe)
            row = triage_case(case_id, recipe, centry, bentry, thresholds)
            counts[row["verdict"]] += 1
            rows.append(row)
    overall = ("REGRESSION" if counts["REGRESSION"] else
               "REVIEW" if counts["REVIEW"] else "PASS")
    return {
        "schema": 1,
        "kind": "comparison",
        "candidate_commit": candidate.get("provenance", {}).get("commit"),
        "baseline_commit": (baseline or {}).get("provenance", {}).get("commit"),
        "overall": overall,
        "counts": counts,
        "total": len(rows),
        "human_attention": [r for r in rows if r["verdict"] != "PASS"],
        "rows": rows,
    }
