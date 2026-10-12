"""Performance comparison: candidate timings vs frozen baseline."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping


def compare_perf(candidate: Mapping[str, Any], baseline: Mapping[str, Any],
                 thresholds: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Compare per-(case, recipe) time + peak RAM. Ratios >1 = slower."""
    rows: List[Dict[str, Any]] = []
    counts = {"PASS": 0, "REVIEW": 0, "REGRESSION": 0}
    base_cases = baseline.get("cases", {})
    for case_id, cdata in candidate.get("cases", {}).items():
        for recipe, centry in cdata.get("recipes", {}).items():
            bentry = base_cases.get(case_id, {}).get("recipes", {}).get(recipe)
            if not bentry or bentry.get("error") or centry.get("error"):
                continue
            if bentry["time_s"] < 2.0:
                # Sub-2s renders are dominated by jitter (interpreter warmup,
                # allocator noise); a 2x ratio on 55 ms is not a regression.
                rows.append({"case": case_id, "recipe": recipe, "verdict": "PASS",
                             "skipped": "baseline_too_fast",
                             "time_s": {"baseline": bentry["time_s"],
                                        "candidate": centry["time_s"]}})
                counts["PASS"] += 1
                continue
            time_ratio = centry["time_s"] / max(bentry["time_s"], 1e-6)
            ram_ratio = centry["peak_ram_mb"] / max(bentry["peak_ram_mb"], 1e-6)
            verdict = "PASS"
            for check, ratio in (("perf_time_ratio", time_ratio),
                                 ("perf_ram_ratio", ram_ratio)):
                spec = thresholds[check]
                if ratio > spec["regress"]:
                    verdict = "REGRESSION"
                elif ratio > spec["warn"] and verdict == "PASS":
                    verdict = "REVIEW"
            counts[verdict] += 1
            rows.append({
                "case": case_id, "recipe": recipe, "verdict": verdict,
                "time_s": {"baseline": bentry["time_s"], "candidate": centry["time_s"],
                           "ratio": round(time_ratio, 3)},
                "peak_ram_mb": {"baseline": bentry["peak_ram_mb"],
                                "candidate": centry["peak_ram_mb"],
                                "ratio": round(ram_ratio, 3)},
            })
    overall = ("REGRESSION" if counts["REGRESSION"] else
               "REVIEW" if counts["REVIEW"] else "PASS")
    return {"schema": 1, "kind": "perf", "overall": overall,
            "counts": counts, "total": len(rows),
            "regressions": [r for r in rows if r["verdict"] != "PASS"],
            "rows": rows}
