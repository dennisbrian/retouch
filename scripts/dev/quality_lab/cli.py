#!/usr/bin/env python3
"""Autonomous Image Quality & Regression Lab — CLI.

    scripts/dev/quality-lab corpus --build-synthetic
    scripts/dev/quality-lab baseline [--recipes ...] [--tags ...] [-o out.json]
    scripts/dev/quality-lab run [--recipes ...] [--tags ...] [--pr-aware BASE HEAD] [-o out.json]
    scripts/dev/quality-lab compare CANDIDATE.json [BASELINE.json] [--json]
    scripts/dev/quality-lab report FILE.json [--json]
    scripts/dev/quality-lab benchmark CANDIDATE.json [BASELINE.json] [--json]
    scripts/dev/quality-lab check-thresholds [BASE_REF]   (CI guard)

Exit codes: 0 ok / no REGRESSION, 1 REGRESSION or loosened thresholds,
2 setup error.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import (CORPUS_DIR, DEFAULT_RECIPES, LAB_DIR, REPORTS_DIR, ROOT,
                   RISK_RECIPES, build_synthetic_corpus, check_threshold_change,
                   load_thresholds, _load_manifest, select_cases)


def _out_json(data: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(data, indent=2))
        return
    kind = data.get("kind", "?")
    if kind in ("comparison", "perf"):
        counts = data["counts"]
        print(f"{kind.upper()} {data['overall']}: "
              f"{counts['PASS']} PASS, {counts['REVIEW']} REVIEW, "
              f"{counts['REGRESSION']} REGRESSION of {data['total']}")
        key = "human_attention" if kind == "comparison" else "regressions"
        for row in data[key][:20]:
            print(f"  [{row['verdict']}] {row['case']} / {row['recipe']}")
            for f in row.get("findings", [])[:4]:
                print(f"      {f['check']}: {f['value']}  {f.get('detail', '')}")
            if "time_s" in row:
                t, r = row["time_s"], row["peak_ram_mb"]
                print(f"      time {t['baseline']}s -> {t['candidate']}s (x{t['ratio']}), "
                      f"RAM {r['baseline']}MB -> {r['candidate']}MB (x{r['ratio']})")
    else:
        print(json.dumps(data, indent=2))


def _resolve_cases(args) -> "tuple[list, list, list]":
    """Return (cases, active_tags, recipes) for baseline/run."""
    manifest = _load_manifest()
    for case in manifest["cases"]:
        case["_root"] = str(ROOT)
    if getattr(args, "pr_aware", None):
        base, head = args.pr_aware
        out = subprocess.check_output(
            ["git", "diff", "--name-only", f"{base}...{head}"], cwd=ROOT, text=True
        )
        changed = [ln.strip() for ln in out.splitlines() if ln.strip()]
        risk = "MEDIUM"
        try:
            r = subprocess.run(
                [sys.executable, str(ROOT / "scripts/dev/pr_governor.py"),
                 "risk", base, head, "--json"],
                cwd=ROOT, capture_output=True, text=True)
            if r.returncode in (0, 1) and r.stdout.strip():
                risk = json.loads(r.stdout).get("risk", "MEDIUM")
        except Exception:
            pass
        cases, tags = select_cases(manifest, changed_files=changed, risk=risk)
        recipes = args.recipes or RISK_RECIPES.get(risk, RISK_RECIPES["MEDIUM"])
        print(f"pr-aware: risk={risk} tags={tags} recipes={recipes} "
              f"cases={len(cases)}", file=sys.stderr)
    else:
        cases, tags = select_cases(manifest, tags=args.tags)
        recipes = args.recipes or DEFAULT_RECIPES
    return cases, tags, list(recipes)


def cmd_corpus(args) -> int:
    if args.build_synthetic:
        manifest = build_synthetic_corpus(overwrite=args.overwrite)
        print(f"corpus: {len(manifest['cases'])} cases -> {CORPUS_DIR}")
        return 0
    manifest = _load_manifest()
    if args.json:
        print(json.dumps(manifest, indent=2))
    else:
        for case in manifest["cases"]:
            print(f"  {case['id']}: {', '.join(case.get('tags', []))}")
        print(f"{len(manifest['cases'])} cases")
    return 0


def cmd_baseline(args) -> int:
    from .runner import build_report, write_report
    cases, tags, recipes = _resolve_cases(args)
    if not cases:
        print("no cases selected", file=sys.stderr)
        return 2
    report = build_report("baseline", cases, recipes)
    path = Path(args.output) if args.output else REPORTS_DIR / "baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"baseline: {len(cases)} cases x {recipes} -> {path}")
    return 0


def cmd_run(args) -> int:
    from .runner import build_report, write_report
    cases, tags, recipes = _resolve_cases(args)
    if not cases:
        print("no cases selected (impact map: nothing visual to run)", file=sys.stderr)
        if args.output:
            Path(args.output).write_text(json.dumps(
                {"schema": 1, "kind": "candidate", "cases": {},
                 "note": "impact map selected no cases"}, indent=2))
        return 0
    report = build_report("candidate", cases, recipes)
    path = write_report(report, args.output)
    print(f"candidate: {len(cases)} cases x {recipes} -> {path}")
    if args.json:
        print(json.dumps(report, indent=2))
    return 0


def _load_report(path_str: Optional[str], default: str) -> Dict[str, Any]:
    p = Path(path_str) if path_str else REPORTS_DIR / default
    if not p.exists():
        raise SystemExit(f"report not found: {p}")
    return json.loads(p.read_text())


def cmd_compare(args) -> int:
    from .triage import triage_report
    candidate = _load_report(args.candidate, "candidate.json")
    baseline = _load_report(args.baseline, "baseline.json") if args.baseline != "-" else None
    thresholds = load_thresholds()
    result = triage_report(candidate, baseline, thresholds)
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    _out_json(result, args.json)
    return 1 if result["overall"] == "REGRESSION" else 0


def cmd_report(args) -> int:
    data = _load_report(args.file, "baseline.json")
    _out_json(data, args.json or data.get("kind") not in ("comparison", "perf"))
    return 1 if data.get("overall") == "REGRESSION" else 0


def cmd_benchmark(args) -> int:
    from .perf import compare_perf
    candidate = _load_report(args.candidate, "candidate.json")
    baseline = _load_report(args.baseline, "baseline.json")
    result = compare_perf(candidate, baseline, load_thresholds())
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    _out_json(result, args.json)
    return 1 if result["overall"] == "REGRESSION" else 0


def cmd_check_thresholds(args) -> int:
    result = check_threshold_change(args.base_ref)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["loosened"]:
            print("THRESHOLDS LOOSENED (must be reviewed as an engineering change):")
            for lo in result["loosened"]:
                print(f"  {lo['check']}.{lo['level']}: {lo['old']} -> {lo['new']}")
        else:
            print(f"thresholds ok (changed={result['changed']})")
    return 1 if result["loosened"] else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quality-lab", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus", help="list or build the benchmark corpus")
    c.add_argument("--build-synthetic", action="store_true")
    c.add_argument("--overwrite", action="store_true")
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_corpus)

    def add_render_args(sp):
        sp.add_argument("--recipes", nargs="+", default=None)
        sp.add_argument("--tags", nargs="+", default=None)
        sp.add_argument("--pr-aware", nargs=2, metavar=("BASE", "HEAD"), default=None,
                        help="select cases/recipes from the PR diff + risk")
        sp.add_argument("-o", "--output", default=None)
        sp.add_argument("--json", action="store_true")

    b = sub.add_parser("baseline", help="freeze stable outputs as reference")
    add_render_args(b)
    b.set_defaults(func=cmd_baseline)

    r = sub.add_parser("run", help="render candidate + metric report")
    add_render_args(r)
    r.set_defaults(func=cmd_run)

    cp = sub.add_parser("compare", help="triage candidate vs baseline")
    cp.add_argument("candidate", help="candidate report JSON")
    cp.add_argument("baseline", nargs="?", default=None,
                    help="baseline report JSON (default: reports/baseline.json; '-' = none)")
    cp.add_argument("-o", "--output", default=None)
    cp.add_argument("--json", action="store_true")
    cp.set_defaults(func=cmd_compare)

    rp = sub.add_parser("report", help="summarize a report/comparison JSON")
    rp.add_argument("file")
    rp.add_argument("--json", action="store_true")
    rp.set_defaults(func=cmd_report)

    bm = sub.add_parser("benchmark", help="performance vs baseline")
    bm.add_argument("candidate")
    bm.add_argument("baseline", nargs="?", default=None)
    bm.add_argument("-o", "--output", default=None)
    bm.add_argument("--json", action="store_true")
    bm.set_defaults(func=cmd_benchmark)

    ct = sub.add_parser("check-thresholds", help="fail if thresholds were loosened vs a ref")
    ct.add_argument("base_ref", nargs="?", default="origin/main")
    ct.add_argument("--json", action="store_true")
    ct.set_defaults(func=cmd_check_thresholds)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
