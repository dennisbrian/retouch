"""Detached candidate worktrees and fixed Quality Lab execution conditions."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from . import core

# Runs the already committed Quality Lab in the candidate worktree. The payload
# chooses cases and recipes only; candidates cannot supply arbitrary commands.
WORKER = r'''
import json, sys
from pathlib import Path
from scripts.dev.quality_lab.runner import build_report
spec = json.loads(Path(sys.argv[1]).read_text())
report = build_report("candidate", spec["cases"], spec["recipes"])
Path(sys.argv[2]).write_text(json.dumps(report, indent=2))
'''


def _run(argv, cwd=None, timeout=60):
    proc = subprocess.run(argv, cwd=cwd or core.ROOT, capture_output=True,
                          text=True, timeout=timeout)
    if proc.returncode:
        raise ValueError(f"{' '.join(map(str, argv[:3]))}: {proc.stderr.strip()[-1000:]}")
    return proc.stdout.strip()


def changed_paths(worktree):
    tracked = _run(["git", "diff", "--name-only", "HEAD"], cwd=worktree).splitlines()
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard"], cwd=worktree).splitlines()
    # models/ is engine infrastructure shared from the repo root by
    # _link_engine_assets(), never candidate code — exclude it from the
    # change/fingerprint view entirely.
    return sorted(p for p in set(tracked + untracked)
                  if not (p == "models" or p.startswith("models/")))


def allowed_path(path):
    normalized = Path(path)
    if normalized.is_absolute() or ".." in normalized.parts or not path:
        return False
    if path.startswith(core.FORBIDDEN):
        return False
    return path in ("cli.py", "gui.py") or path.startswith(("retouch/", "presets/"))


def source_fingerprint(worktree, paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode())
        item = worktree / path
        if item.is_file():
            digest.update(item.read_bytes())
        else:
            digest.update(b"<deleted>")
    return digest.hexdigest()


def candidate_worktree_path(exp, cand):
    expected = (core.WORK_ROOT / exp["experiment_id"] / "worktrees" / cand["id"]).resolve()
    recorded = Path(cand["worktree"]).resolve() if cand.get("worktree") else None
    if recorded != expected:
        raise ValueError("candidate worktree path differs from managed experiment location")
    return expected


def inspect_candidate(exp, cand):
    worktree = candidate_worktree_path(exp, cand)
    if not worktree.is_dir() or not (worktree / ".git").exists():
        raise ValueError("candidate worktree missing; run prepare")
    head = _run(["git", "rev-parse", "HEAD"], cwd=worktree)
    if head != exp["baseline"]["commit"]:
        raise ValueError("candidate HEAD differs from frozen base commit")
    paths = changed_paths(worktree)
    forbidden = [p for p in paths if not allowed_path(p)]
    if forbidden:
        raise ValueError(f"candidate changed forbidden paths: {forbidden[:8]}")
    if not paths:
        raise ValueError("candidate has no implementation changes")
    for path in paths:
        source = worktree / path
        if source.is_file() and path.endswith(".py"):
            try:
                ast.parse(source.read_text(encoding="utf-8"), filename=path)
            except (SyntaxError, UnicodeError) as exc:
                raise ValueError(f"candidate syntax check failed for {path}: {exc}") from exc
    # Explicitly verify the worktree's Quality Lab policy has not changed.
    if core.sha256(worktree / "quality_lab" / "thresholds.json") != exp["baseline"]["thresholds_sha256"]:
        raise ValueError("candidate Quality Lab thresholds differ from design")
    numstat = _run(["git", "diff", "--numstat", "HEAD"], cwd=worktree)
    added = 0
    for line in numstat.splitlines():
        columns = line.split("\t")
        if len(columns) >= 3 and columns[0].isdigit():
            added += int(columns[0])
    tracked = set(_run(["git", "ls-files"], cwd=worktree).splitlines())
    for path in paths:
        if path not in tracked and (worktree / path).is_file():
            added += len((worktree / path).read_text(errors="replace").splitlines())
    complexity = {"changed_files": len(paths), "loc_added": added,
                  "new_dependencies": 0, "paths": paths,
                  "subsystems": sorted({p.split("/", 1)[0] for p in paths})}
    limits = exp["constraints"]
    if added > limits.get("max_loc_added", 500) or len(paths) > limits.get("max_changed_files", 8):
        raise ValueError("candidate complexity budget exceeded")
    return complexity, source_fingerprint(worktree, paths)


def _link_engine_assets(worktree):
    """Share git-ignored engine assets (models/) with the candidate worktree.

    ``models/`` holds large binaries that are deliberately untracked (only
    manifest.json is committed), so a fresh ``git worktree`` has none of them.
    Without this link every candidate benchmark silently runs the engine's
    no-model fallback paths: different renders (landmark-only masks), slower
    detection, and quality/threshold numbers that describe the fallback, not
    the candidate. A symlink keeps a single physical copy on disk.
    """
    target = worktree / "models"
    if target.is_symlink() and target.exists():
        return
    source = core.ROOT / "models"
    if not source.is_dir():
        return
    if target.exists():
        # git checkout materialised models/ with the tracked manifest.json;
        # the physical asset dir in the repo root is the single source of truth.
        shutil.rmtree(target)
    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError:
        shutil.copytree(source, target, dirs_exist_ok=True)


def prepare(exp, cand):
    core.ensure_frozen(exp)
    if cand["worktree"]:
        raise ValueError("candidate already prepared")
    if exp["status"] not in ("DESIGNED", "RUNNING"):
        raise ValueError("experiment not open for candidate preparation")
    if any(r["split"] == "holdout" for c in exp["candidates"] for r in c["runs"]):
        raise ValueError("candidate preparation is sealed after holdout exposure")
    path = core.WORK_ROOT / exp["experiment_id"] / "worktrees" / cand["id"]
    if path.exists():
        raise ValueError(f"worktree path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    _run(["git", "worktree", "add", "--detach", str(path), exp["baseline"]["commit"]], timeout=60)
    try:
        _link_engine_assets(path)
        if cand["patch"]:
            patch = Path(cand["patch"])
            raw = _run(["git", "apply", "--numstat", str(patch)], cwd=path)
            paths = [line.split("\t", 2)[-1] for line in raw.splitlines()]
            if not paths or any(not allowed_path(p) for p in paths):
                raise ValueError("patch touches no candidate code or a forbidden path")
            _run(["git", "apply", "--check", str(patch)], cwd=path)
            _run(["git", "apply", str(patch)], cwd=path)
        cand["worktree"] = str(path)
        cand["status"] = "READY"
        exp["status"] = "RUNNING"
        exp["started_at"] = exp["started_at"] or core.now()
        core.save(exp)
        return {"candidate": cand["id"], "worktree": str(path), "base_commit": exp["baseline"]["commit"]}
    except Exception:
        # Never force-remove a worktree containing prototype edits.
        if path.exists() and not changed_paths(path):
            _run(["git", "worktree", "remove", str(path)], timeout=60)
        raise


def _copy_inputs(exp, split, worktree):
    corpus = core.read_json(exp["corpus"]["manifest"])
    selected = set(exp["corpus"]["cases"][split])
    cases = []
    for case in corpus["cases"]:
        if case["id"] not in selected:
            continue
        path = Path(case["input"])
        if path.is_absolute():
            raise ValueError("absolute corpus input paths are not supported for isolation")
        src = core.ROOT / path
        dst = worktree / path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or core.sha256(dst) != core.sha256(src):
            shutil.copy2(src, dst)
        selected_case = dict(case)
        selected_case["_root"] = str(worktree)
        cases.append(selected_case)
    return cases


def _paired_target(exp, candidate_report, baseline_report, split):
    metric = exp["success_criteria"]["metric"]
    direction = exp["success_criteria"]["direction"]
    deltas = []
    for case_id in exp["corpus"]["cases"][split]:
        for recipe in exp["baseline"]["recipes"]:
            base = core.metric_value(baseline_report.get("cases", {}).get(case_id, {}).get("recipes", {}).get(recipe, {}), metric)
            cand = core.metric_value(candidate_report.get("cases", {}).get(case_id, {}).get("recipes", {}).get(recipe, {}), metric)
            if base is None or cand is None:
                continue
            deltas.append((cand - base) if direction == "higher" else (base - cand))
    if not deltas:
        return {"sample_size": 0, "median_delta": None, "spread": None, "positive_pairs": 0}
    return {"sample_size": len(deltas), "median_delta": round(statistics.median(deltas), 4),
            "spread": [round(min(deltas), 4), round(max(deltas), 4)],
            "positive_pairs": sum(x > 0 for x in deltas)}


def early_stop(exp, result):
    limits = exp["failure_criteria"]
    measured = sum(result["quality"].values())
    severe = result["quality"]["REGRESSION"] / max(1, measured) > limits["max_regression_fraction"]
    too_slow = result["max_time_ratio"] is not None and result["max_time_ratio"] > limits["max_runtime_ratio"]
    too_large = result["max_ram_ratio"] is not None and result["max_ram_ratio"] > limits["max_ram_ratio"]
    return [name for name, active in (("visual regression", severe), ("runtime", too_slow), ("RAM", too_large)) if active]


def run(exp, cand, split):
    from scripts.dev.quality_lab.core import load_thresholds
    from scripts.dev.quality_lab.triage import triage_report
    from scripts.dev.quality_lab.perf import compare_perf

    if split not in ("development", "validation", "holdout"):
        raise ValueError("split must be development, validation, or holdout")
    core.ensure_frozen(exp)
    if cand["status"] == "ELIMINATED":
        raise ValueError("candidate eliminated")
    if split != "holdout" and any(r["split"] == "holdout" for c in exp["candidates"] for r in c["runs"]):
        raise ValueError("refinement is sealed after first holdout exposure")
    if split == "holdout" and cand["id"] not in exp.get("finalists", []):
        raise ValueError("holdout is sealed until compare selects a finalist")
    if split == "holdout" and any(r["split"] == "holdout" for r in cand["runs"]):
        raise ValueError("holdout may be run once per candidate")
    if not exp["corpus"]["cases"][split]:
        raise ValueError("split has no cases")
    complexity, fingerprint = inspect_candidate(exp, cand)
    if split == "development":
        if any(r["fingerprint"] == fingerprint and r["split"] == "development" for r in cand["runs"]):
            raise ValueError("same implementation already ran development; change code to refine")
        if cand["rounds"] >= core.BUDGETS[exp["budget"]]["rounds"]:
            raise ValueError("refinement-round budget exhausted")
        round_number = cand["rounds"] + 1
    else:
        dev = next((r for r in reversed(cand["runs"]) if r["split"] == "development" and r["fingerprint"] == fingerprint), None)
        if dev is None:
            raise ValueError("run development first with this exact implementation")
        round_number = dev["round"]
        if split == "holdout" and not any(r["split"] == "validation" and r["round"] == round_number for r in cand["runs"]):
            raise ValueError("validation required before holdout")
        if any(r["split"] == split and r["round"] == round_number for r in cand["runs"]):
            raise ValueError("split already measured for this round")
    elapsed = sum(r["elapsed_seconds"] for c in exp["candidates"] for r in c["runs"])
    max_seconds = core.BUDGETS[exp["budget"]]["seconds"]
    remaining = max_seconds - elapsed
    if remaining <= 0:
        raise ValueError("experiment compute budget exhausted")
    worktree = candidate_worktree_path(exp, cand)
    cases = _copy_inputs(exp, split, worktree)
    # TINY experiments are deliberately exploratory. They cannot qualify for promotion.
    if exp["budget"] == "TINY":
        cases = cases[:3]
    spec_path = worktree / "test_output" / "experiment_lab" / "worker-spec.json"
    report_path = worktree / "test_output" / "experiment_lab" / f"{exp['experiment_id']}-{cand['id']}-{split}-r{round_number}.json"
    core.write_json(spec_path, {"cases": cases, "recipes": exp["baseline"]["recipes"]})
    report_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(worktree), "RETOUCH_GPU": "0",
           "RETOUCH_MEDIAPIPE_BACKEND": "legacy"}
    interpreter = core.ROOT / ".venv" / "bin" / "python"
    python = str(interpreter) if interpreter.is_file() else sys.executable
    t0 = time.monotonic()
    try:
        proc = subprocess.run([python, "-c", WORKER, str(spec_path), str(report_path)],
                              cwd=worktree, env=env, capture_output=True, text=True,
                              timeout=max(1, int(remaining)))
    except subprocess.TimeoutExpired as exc:
        cand["status"] = "ELIMINATED"
        cand["elimination_reason"] = "compute budget exceeded"
        core.save(exp)
        raise ValueError("candidate timed out at compute budget") from exc
    wall = round(time.monotonic() - t0, 2)
    if proc.returncode or not report_path.is_file():
        cand["status"] = "ELIMINATED"
        cand["elimination_reason"] = "Quality Lab runner failed: " + proc.stderr[-500:]
        core.save(exp)
        raise ValueError(cand["elimination_reason"])
    _, post_run_fingerprint = inspect_candidate(exp, cand)
    if post_run_fingerprint != fingerprint:
        raise ValueError("candidate implementation changed during benchmark")
    report = core.read_json(report_path)
    baseline = core.read_json(exp["baseline"]["report"])
    if report.get("provenance", {}).get("commit") != exp["baseline"]["commit"]:
        raise ValueError("candidate report came from a different base commit")
    base_subset = {**baseline, "cases": {k: baseline["cases"][k] for k in report["cases"]}}
    comparison = triage_report(report, base_subset, load_thresholds())
    perf = compare_perf(report, base_subset, load_thresholds())
    target = _paired_target(exp, report, baseline, split)
    perf_rows = [r for r in perf["rows"] if "ratio" in r.get("time_s", {})]
    time_ratio = max((r["time_s"]["ratio"] for r in perf_rows), default=None)
    ram_ratio = max((r["peak_ram_mb"]["ratio"] for r in perf_rows), default=None)
    result = {"split": split, "round": round_number, "fingerprint": fingerprint,
              "elapsed_seconds": wall, "report": str(report_path), "report_sha256": core.sha256(report_path),
              "quality": comparison["counts"], "performance": perf["counts"],
              "target": target, "max_time_ratio": time_ratio, "max_ram_ratio": ram_ratio,
              "complexity": complexity, "environment": report.get("provenance", {}).get("engine_versions", {}),
              "case_ids": sorted(report["cases"])}
    if result["environment"] != exp["baseline"]["environment"]:
        raise ValueError("candidate environment versions differ from frozen baseline")
    if split == "development":
        cand["rounds"] = round_number
    cand["runs"].append(result)
    cand["complexity"] = complexity
    cand["status"] = "MEASURED"
    failures = early_stop(exp, result)
    if split == "development" and failures:
        cand["status"] = "ELIMINATED"
        cand["elimination_reason"] = "early stop: " + ", ".join(failures)
    exp["status"] = "ANALYZING"
    core.save(exp)
    return result
