"""Experiment registry, frozen design, evidence comparison, and promotion gates.

Prototype code lives in detached worktrees. Durable manifests contain decisions,
metrics, and negative results; they never contain private corpus images.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from scripts.dev.control_plane import core as cp
from scripts.dev.attention_router import core as attention

ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "experiments"
WORK_ROOT = ROOT / "test_output" / "experiment_lab"
CORPUS = ROOT / "quality_lab" / "corpus_manifest.json"
THRESHOLDS = ROOT / "quality_lab" / "thresholds.json"
STATES = ("PROPOSED", "DESIGNED", "RUNNING", "ANALYZING", "COMPLETED", "INCONCLUSIVE", "REJECTED", "PROMOTED", "ARCHIVED")
BUDGETS = {
    "TINY": {"candidates": 2, "rounds": 1, "seconds": 900},
    "SMALL": {"candidates": 3, "rounds": 2, "seconds": 3600},
    "MEDIUM": {"candidates": 4, "rounds": 3, "seconds": 10800},
    "LARGE": {"candidates": 5, "rounds": 3, "seconds": 28800},
}
ALLOWED_CODE = ("retouch/", "presets/", "cli.py", "gui.py")
FORBIDDEN = ("tests/", ".github/", "quality_lab/", "experiments/", "control-plane/",
             "scripts/dev/", "pyproject.toml", "uv.lock", "requirements", "docs/", "models/")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args, cwd=None):
    proc = subprocess.run(["git", *args], cwd=cwd or ROOT, text=True,
                          capture_output=True, timeout=30)
    if proc.returncode:
        raise ValueError(f"git {' '.join(args)}: {proc.stderr.strip()}")
    return proc.stdout.strip()


def path_for(exp_id):
    if not re.fullmatch(r"EXP-\d{4,}", exp_id):
        raise ValueError("invalid experiment id")
    return REGISTRY / exp_id / "manifest.json"


def list_all():
    rows = []
    for path in sorted(REGISTRY.glob("EXP-*/manifest.json")):
        data = read_json(path)
        if isinstance(data, dict) and data.get("experiment_id") == path.parent.name:
            rows.append(data)
    return rows


def load(exp_id):
    data = read_json(path_for(exp_id))
    if not isinstance(data, dict) or data.get("experiment_id") != exp_id:
        raise ValueError(f"unknown experiment {exp_id}")
    return data


def save(exp):
    exp["updated_at"] = now()
    write_json(path_for(exp["experiment_id"]), exp)


def next_id(rows):
    number = max((int(r["experiment_id"].split("-")[1]) for r in rows), default=0) + 1
    return f"EXP-{number:04d}"


def tokens(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) > 3 and w not in {"with", "from", "that", "this", "into", "using", "improve", "reduces"}}


def similarity(question, hypothesis_id="", *, prior=None, rd_state=None, repo_scan=False):
    """Cross-check experiment history, R&D negative memory, and research titles."""
    prior = list_all() if prior is None else prior
    if rd_state is None:
        try:
            from scripts.dev.rd_director import core as rd
            rd_state = rd.load()
        except ImportError:
            rd_state = {"negative_results": []}
    hits = []
    unavailable = []
    words = tokens(question)
    duplicate = False
    for exp in prior:
        other = tokens(exp.get("question", ""))
        overlap = len(words & other) / max(1, len(words | other))
        exact_hyp = bool(hypothesis_id and hypothesis_id == exp.get("hypothesis_id"))
        if exact_hyp or overlap >= 0.75:
            duplicate = True
            hits.append({"source": "experiment", "id": exp["experiment_id"], "overlap": round(overlap, 2)})
        elif overlap >= 0.35:
            hits.append({"source": "experiment", "id": exp["experiment_id"], "overlap": round(overlap, 2)})
    for neg in rd_state.get("negative_results", []):
        other = tokens(f"{neg.get('tried', '')} {neg.get('result', '')}")
        overlap = len(words & other) / max(1, len(words | other))
        if overlap >= 0.25:
            hits.append({"source": "negative-memory", "id": neg.get("id"),
                         "overlap": round(overlap, 2), "retry_unless": neg.get("retry_unless", "")})
    for path in sorted((ROOT / "docs" / "plans").glob("*.md")):
        title_words = tokens(path.stem.replace("_", " "))
        overlap = len(words & title_words) / max(1, len(words | title_words))
        if overlap >= 0.45:
            hits.append({"source": "research", "path": str(path.relative_to(ROOT)), "overlap": round(overlap, 2)})
    if repo_scan:
        decision_state = attention.load()
        decisions = {d["item_id"]: d for d in decision_state["decisions"]}
        for alert in decision_state["alerts"]:
            if alert.get("id") not in decisions:
                continue
            other = tokens(f"{alert.get('title', '')} {decisions[alert['id']].get('reason', '')}")
            overlap = len(words & other) / max(1, len(words | other))
            if overlap >= 0.35:
                hits.append({"source": "decision-memory", "id": alert["id"],
                             "overlap": round(overlap, 2)})
        try:
            proc = subprocess.run(["gh", "pr", "list", "--state", "all", "--limit", "100",
                                   "--json", "number,title"], cwd=ROOT, capture_output=True,
                                  text=True, timeout=8)
            if proc.returncode:
                unavailable.append("GitHub PR history")
            else:
                for pr in json.loads(proc.stdout or "[]"):
                    other = tokens(pr.get("title", ""))
                    overlap = len(words & other) / max(1, len(words | other))
                    if overlap >= 0.35:
                        hits.append({"source": "pr", "number": pr["number"],
                                     "overlap": round(overlap, 2)})
        except (OSError, ValueError, subprocess.TimeoutExpired):
            unavailable.append("GitHub PR history")
    hits.sort(key=lambda h: -h["overlap"])
    verdict = "DUPLICATE" if duplicate else "RETRY_WITH_NEW_EVIDENCE" if any(h["source"] == "negative-memory" for h in hits) else "RELATED" if hits else "NEW"
    return {"verdict": verdict, "matches": hits[:12], "sources_unavailable": unavailable}


def create(hypothesis_id, title, question, *, budget="SMALL", new_evidence="", rd_state=None,
           repo_scan=False):
    if budget not in BUDGETS or not title.strip() or not question.strip():
        raise ValueError("title, question, and valid budget required")
    if rd_state is None:
        from scripts.dev.rd_director import core as rd
        rd_state = rd.load()
    hyp = next((h for h in rd_state.get("hypotheses", []) if h["id"] == hypothesis_id), None)
    if hyp is None:
        raise ValueError(f"unknown R&D hypothesis {hypothesis_id}")
    prior = list_all()
    similar = similarity(question, hypothesis_id, prior=prior, rd_state=rd_state, repo_scan=repo_scan)
    if similar["verdict"] in ("DUPLICATE", "RETRY_WITH_NEW_EVIDENCE") and not new_evidence.strip():
        raise ValueError(f"{similar['verdict']}: provide --new-evidence to justify a retry")
    exp = {"schema": 1, "experiment_id": next_id(prior), "hypothesis_id": hypothesis_id,
           "title": title.strip(), "question": question.strip(), "hypothesis": hyp["text"],
           "hypothesis_success_text": hyp["success_criteria"], "status": "PROPOSED",
           "baseline": {}, "success_criteria": {}, "failure_criteria": {}, "candidates": [],
           "candidate_count": 0, "corpus": {}, "metrics": [], "constraints": {},
           "budget": budget, "started_at": "", "completed_at": "", "winner": None,
           "conclusion": None, "confidence": "NONE", "negative_findings": [],
           "artifacts": [], "related_research": [m.get("path") for m in similar["matches"] if m["source"] == "research"],
           "promotion_task": None, "similarity": similar, "new_evidence": new_evidence,
           "created_at": now(), "updated_at": now(), "history": [{"at": now(), "event": "created"}]}
    save(exp)
    return exp


def split_cases(cases, seed):
    """Group by subject where known. Small corpora stay unqualified for promotion."""
    groups = {}
    for case in cases:
        groups.setdefault(str(case.get("subject") or case["id"]), []).append(case["id"])
    names = sorted(groups, key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest())
    if len(names) < 6:
        return {"development": sorted(c["id"] for c in cases), "validation": [],
                "holdout": [], "qualified": False, "reason": "fewer than six independent subject groups"}
    n_hold = max(2, round(len(names) * 0.2))
    n_val = max(1, round(len(names) * 0.2))
    split = {"development": names[:-(n_hold + n_val)],
             "validation": names[-(n_hold + n_val):-n_hold], "holdout": names[-n_hold:]}
    return {key: sorted(case_id for group in split[key] for case_id in groups[group])
            for key in ("development", "validation", "holdout")} | {"qualified": True, "reason": "subject-separated"}


def validate_split(cases, split):
    names = ("development", "validation", "holdout")
    if not all(isinstance(split.get(name), list) for name in names):
        raise ValueError("split needs development, validation, and holdout arrays")
    ids = [case_id for name in names for case_id in split[name]]
    all_ids = {case["id"] for case in cases}
    if len(ids) != len(set(ids)) or set(ids) != all_ids or any(not split[name] for name in names):
        raise ValueError("split must assign each case once with every partition nonempty")
    by_group = {}
    for case in cases:
        by_group.setdefault(str(case.get("subject") or case["id"]), set()).add(case["id"])
    for members in by_group.values():
        if sum(bool(members & set(split[name])) for name in names) != 1:
            raise ValueError("subject group crosses development/validation/holdout")
    return {name: sorted(split[name]) for name in names} | {
        "qualified": len(by_group) >= 6 and len(split["holdout"]) >= 2,
        "reason": "subject-separated" if len(by_group) >= 6 else "too few independent groups"}


def metric_value(entry, dotted):
    value = entry.get("metrics", {})
    for key in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def validate_design(exp, spec):
    if exp["status"] != "PROPOSED":
        raise ValueError("design is frozen after the first accepted design")
    base = spec.get("base_commit", "")
    if not re.fullmatch(r"[a-f0-9]{40}", base):
        raise ValueError("base_commit must be a full Git SHA")
    git("cat-file", "-e", f"{base}^{{commit}}")
    baseline_path = Path(spec.get("baseline_report", "")).expanduser().resolve()
    baseline = read_json(baseline_path)
    if not isinstance(baseline, dict) or baseline.get("kind") != "baseline":
        raise ValueError("baseline_report must be a Quality Lab baseline JSON")
    provenance = baseline.get("provenance", {})
    if provenance.get("commit") != base or provenance.get("working_tree_dirty"):
        raise ValueError("baseline must come from the exact clean base commit")
    corpus_path = Path(spec.get("corpus_manifest", CORPUS)).expanduser().resolve()
    corpus = read_json(corpus_path)
    cases = corpus.get("cases", []) if isinstance(corpus, dict) else []
    if len(cases) != len({c.get("id") for c in cases}) or not cases:
        raise ValueError("corpus needs unique case IDs")
    recipes = spec.get("recipes") or provenance.get("recipes")
    if not isinstance(recipes, list) or not recipes or sorted(recipes) != sorted(provenance.get("recipes", [])):
        raise ValueError("recipes must match frozen baseline recipes")
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", str(case["id"])):
            raise ValueError("corpus case IDs must be safe file names")
        row = baseline.get("cases", {}).get(case["id"])
        inp = Path(case["input"])
        if inp.is_absolute() or ".." in inp.parts:
            raise ValueError("corpus inputs must be relative paths within this repository")
        inp = ROOT / inp
        if not inp.is_file() or not row or row.get("input_sha256") != sha256(inp):
            raise ValueError(f"case {case['id']} missing or input differs from baseline")
        if not all(recipe in row.get("recipes", {}) for recipe in recipes):
            raise ValueError(f"case {case['id']} lacks baseline recipe")
        for recipe in recipes:
            output = row["recipes"][recipe]
            rendered = Path(output.get("output", ""))
            if not rendered.is_file() or output.get("output_sha256") != sha256(rendered):
                raise ValueError(f"case {case['id']} baseline render missing or changed")
    success = spec.get("success_criteria", {})
    failure = spec.get("failure_criteria", {})
    if not success.get("metric") or success.get("direction") not in ("higher", "lower"):
        raise ValueError("success metric and direction required before candidate code")
    if not isinstance(success.get("min_delta"), (int, float)) or success["min_delta"] <= 0:
        raise ValueError("positive min_delta required")
    success = {**success, "min_positive_fraction": success.get("min_positive_fraction", 0.8)}
    if not isinstance(success["min_positive_fraction"], (int, float)) or not 0 < success["min_positive_fraction"] <= 1:
        raise ValueError("min_positive_fraction must be in (0, 1]")
    for field in ("max_regression_fraction", "max_runtime_ratio", "max_ram_ratio"):
        if not isinstance(failure.get(field), (int, float)):
            raise ValueError(f"failure_criteria.{field} required")
    if not 0 <= failure["max_regression_fraction"] <= 1 or failure["max_runtime_ratio"] < 1 or failure["max_ram_ratio"] < 1:
        raise ValueError("invalid failure thresholds")
    if not isinstance(spec.get("reject_all_if"), str) or not spec["reject_all_if"].strip():
        raise ValueError("reject_all_if must be specified before candidates")
    for case in cases:
        for recipe in recipes:
            if metric_value(baseline["cases"][case["id"]]["recipes"][recipe], success["metric"]) is None:
                raise ValueError(f"baseline target metric missing for {case['id']} / {recipe}")
    base_blob = git("rev-parse", f"{base}:quality_lab/thresholds.json")
    threshold_hash = sha256(THRESHOLDS)
    if git("hash-object", str(THRESHOLDS)) != base_blob:
        raise ValueError("active Quality Lab thresholds differ from base commit")
    split = (validate_split(cases, spec["split"]) if spec.get("split") else
             split_cases(cases, exp["experiment_id"]))
    real_ids = {c["id"] for c in cases if c.get("evidence_class") == "real_photo"}
    exp["baseline"] = {"commit": base, "report": str(baseline_path), "sha256": sha256(baseline_path),
                       "thresholds_sha256": threshold_hash, "environment": provenance.get("engine_versions", {}),
                       "recipes": recipes}
    exp["corpus"] = {"manifest": str(corpus_path), "sha256": sha256(corpus_path),
                     "cases": split, "case_count": len(cases),
                     "real_holdout": len(real_ids & set(split["holdout"])) >= 2}
    exp["success_criteria"] = success
    exp["failure_criteria"] = failure
    exp["constraints"] = spec.get("constraints", {"max_loc_added": 500, "max_changed_files": 8})
    exp["metrics"] = [success["metric"], "Quality Lab regression count", "time ratio", "RAM ratio", "complexity"]
    exp["reject_all_if"] = spec["reject_all_if"]
    exp["design_sha256"] = design_fingerprint(exp)
    exp["status"] = "DESIGNED"
    exp["history"].append({"at": now(), "event": "design frozen", "sha256": exp["design_sha256"]})
    save(exp)
    return exp


def add_candidate(exp, approach, mechanism, rationale, *, patch=""):
    if exp["status"] not in ("DESIGNED", "RUNNING"):
        raise ValueError("experiment must be designed before adding candidates")
    if any(run["split"] == "holdout" for cand in exp["candidates"] for run in cand["runs"]):
        raise ValueError("candidate search is sealed after holdout exposure")
    if len(exp["candidates"]) >= BUDGETS[exp["budget"]]["candidates"]:
        raise ValueError("candidate budget exhausted")
    if not all(x.strip() for x in (approach, mechanism, rationale)):
        raise ValueError("approach, mechanism, and rationale required")
    key = tokens(mechanism)
    for candidate in exp["candidates"]:
        old = tokens(candidate["mechanism"])
        if key == old or len(key & old) / max(1, len(key | old)) >= 0.8:
            raise ValueError("candidate mechanism duplicates an existing approach")
    patch_path = str(Path(patch).expanduser().resolve()) if patch else ""
    if patch_path and not Path(patch_path).is_file():
        raise ValueError("candidate patch missing")
    ident = chr(ord("A") + len(exp["candidates"]))
    row = {"id": ident, "approach": approach.strip(), "mechanism": mechanism.strip(),
           "rationale": rationale.strip(), "patch": patch_path, "status": "DESIGNED",
           "worktree": "", "rounds": 0, "runs": [], "elimination_reason": "", "complexity": {}}
    exp["candidates"].append(row)
    exp["candidate_count"] = len(exp["candidates"])
    save(exp)
    return row


def candidate(exp, ident):
    row = next((c for c in exp["candidates"] if c["id"] == ident.upper()), None)
    if row is None:
        raise ValueError(f"unknown candidate {ident}")
    return row


def ensure_frozen(exp):
    if not exp.get("design_sha256") or exp["status"] == "PROPOSED":
        raise ValueError("freeze experiment design before implementation")
    if design_fingerprint(exp) != exp["design_sha256"]:
        raise ValueError("frozen success criteria, corpus, or limits changed")
    if sha256(exp["baseline"]["report"]) != exp["baseline"]["sha256"]:
        raise ValueError("baseline report changed since design")
    if sha256(exp["corpus"]["manifest"]) != exp["corpus"]["sha256"]:
        raise ValueError("corpus manifest changed since design")
    if sha256(THRESHOLDS) != exp["baseline"]["thresholds_sha256"]:
        raise ValueError("Quality Lab thresholds changed since design")


def design_fingerprint(exp):
    frozen = {key: exp.get(key) for key in ("baseline", "corpus", "success_criteria",
              "failure_criteria", "constraints", "reject_all_if", "budget")}
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()


def latest_run(cand, split):
    return next((r for r in reversed(cand["runs"])
                 if r["split"] == split and r["round"] == cand["rounds"]), None)


def assess(exp, run):
    if run is None:
        return {"verdict": "MISSING", "reasons": ["split not measured"]}
    target = run["target"]
    expected = len(exp["corpus"]["cases"][run["split"]]) * len(exp["baseline"]["recipes"])
    reasons = []
    uncertain = False
    if target["sample_size"] != expected:
        reasons.append(f"target measured on {target['sample_size']}/{expected} case-recipes")
        uncertain = True
    if target["median_delta"] is None:
        reasons.append("target metric unavailable")
        uncertain = True
    elif target["sample_size"] == expected and target["median_delta"] < exp["success_criteria"]["min_delta"]:
        reasons.append("target improvement below frozen success threshold")
    if target["sample_size"] and target["positive_pairs"] / target["sample_size"] < exp["success_criteria"].get("min_positive_fraction", 0.8):
        reasons.append("too few paired cases improved")
    total = sum(run["quality"].values())
    fraction = run["quality"]["REGRESSION"] / max(1, total)
    if fraction > exp["failure_criteria"]["max_regression_fraction"]:
        reasons.append("visual regression limit exceeded")
    if run["max_time_ratio"] is None or run["max_ram_ratio"] is None:
        reasons.append("performance comparison lacks stable >=2s baseline cases")
        uncertain = True
    else:
        if run["max_time_ratio"] > exp["failure_criteria"]["max_runtime_ratio"]:
            reasons.append("runtime limit exceeded")
        if run["max_ram_ratio"] > exp["failure_criteria"]["max_ram_ratio"]:
            reasons.append("RAM limit exceeded")
    if run["complexity"]["loc_added"] > exp["constraints"].get("max_loc_added", 500):
        reasons.append("complexity limit exceeded")
    hard_fail = any(reason not in ("target metric unavailable", "performance comparison lacks stable >=2s baseline cases")
                    and not reason.startswith("target measured on") for reason in reasons)
    return {"verdict": "FAIL" if hard_fail else "INCONCLUSIVE" if uncertain else "PASS", "reasons": reasons,
            "regression_fraction": round(fraction, 3)}


def pareto_frontier(rows):
    """Maximize quality; minimize runtime, code size, and regressions."""
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if row["id"] == other["id"]:
                continue
            better_or_equal = (other["quality"] >= row["quality"] and
                               other["runtime"] <= row["runtime"] and
                               other["loc"] <= row["loc"] and
                               other["regressions"] <= row["regressions"])
            strictly = (other["quality"] > row["quality"] or other["runtime"] < row["runtime"] or
                        other["loc"] < row["loc"] or other["regressions"] < row["regressions"])
            if better_or_equal and strictly:
                dominated = True
                break
        if not dominated:
            frontier.append(row["id"])
    return sorted(frontier)


def _comparison_row(exp, cand):
    splits = {name: assess(exp, latest_run(cand, name)) for name in ("development", "validation", "holdout")}
    hold = latest_run(cand, "holdout")
    validation = latest_run(cand, "validation")
    perf = [r["max_time_ratio"] for r in (validation, hold) if r and r["max_time_ratio"] is not None]
    regression_count = sum(r["quality"]["REGRESSION"] for r in (validation, hold) if r)
    return {"id": cand["id"], "approach": cand["approach"], "status": cand["status"],
            "splits": splits, "quality": hold["target"]["median_delta"] if hold else None,
            "runtime": max(perf) if perf else None, "loc": cand["complexity"].get("loc_added"),
            "regressions": regression_count, "elimination_reason": cand["elimination_reason"]}


def _negative(exp, cand, reason):
    if any(r["candidate"] == cand["id"] for r in exp["negative_findings"]):
        return
    exp["negative_findings"].append({"candidate": cand["id"], "approach": cand["approach"],
                                      "mechanism": cand["mechanism"], "observed": reason,
                                      "retry_unless": "new mechanism or materially new corpus evidence",
                                      "at": now()})


def compare(exp):
    if exp["status"] in ("PROPOSED", "DESIGNED", "PROMOTED", "ARCHIVED"):
        raise ValueError("experiment needs candidate evidence before comparison")
    if exp["conclusion"] in ("WINNER", "NO_WINNER"):
        return report(exp)
    ensure_frozen(exp)
    for cand in exp["candidates"]:
        for run in cand["runs"]:
            if not Path(run["report"]).is_file() or sha256(run["report"]) != run["report_sha256"]:
                raise ValueError(f"candidate {cand['id']} run artifact missing or modified")
    rows = [_comparison_row(exp, cand) for cand in exp["candidates"]]
    if not rows or not any(c["runs"] for c in exp["candidates"]):
        exp["status"] = "RUNNING"
        exp["conclusion"] = "PENDING_CANDIDATES"
    else:
        for cand in exp["candidates"]:
            if cand["status"] == "ELIMINATED":
                _negative(exp, cand, cand["elimination_reason"])
        finalists = [row["id"] for row in rows if row["status"] != "ELIMINATED"
                     and row["splits"]["development"]["verdict"] == "PASS"
                     and row["splits"]["validation"]["verdict"] == "PASS"]
        exp["finalists"] = finalists
        if not finalists:
            pending_candidates = any(row["status"] != "ELIMINATED" and
                                     row["splits"]["development"]["verdict"] == "MISSING" for row in rows)
            pending = any(row["status"] != "ELIMINATED" and
                          row["splits"]["development"]["verdict"] == "PASS" and
                          row["splits"]["validation"]["verdict"] == "MISSING" for row in rows)
            uncertain = any(row["status"] != "ELIMINATED" and
                            (row["splits"]["development"]["verdict"] == "INCONCLUSIVE" or
                             row["splits"]["validation"]["verdict"] == "INCONCLUSIVE") for row in rows)
            if pending_candidates:
                exp["status"] = "RUNNING"
                exp["conclusion"] = "PENDING_CANDIDATES"
            elif pending:
                exp["status"] = "ANALYZING"
                exp["conclusion"] = "PENDING_VALIDATION"
            elif uncertain:
                exp["status"] = "INCONCLUSIVE"
                exp["conclusion"] = "INCONCLUSIVE"
                exp["completed_at"] = now()
            else:
                exp["status"] = "COMPLETED"
                exp["conclusion"] = "NO_WINNER"
                exp["completed_at"] = now()
                for cand in exp["candidates"]:
                    row = next(r for r in rows if r["id"] == cand["id"])
                    reasons = (row["splits"]["validation"]["reasons"]
                               if row["splits"]["validation"]["verdict"] != "MISSING"
                               else row["splits"]["development"]["reasons"])
                    _negative(exp, cand, cand["elimination_reason"] or "; ".join(reasons))
        elif any(row["id"] in finalists and row["splits"]["holdout"]["verdict"] == "MISSING" for row in rows):
            exp["status"] = "ANALYZING"
            exp["conclusion"] = "PENDING_HOLDOUT"
        else:
            passed = [row for row in rows if row["id"] in finalists and row["splits"]["holdout"]["verdict"] == "PASS"]
            if not passed:
                uncertain = any(row["id"] in finalists and
                                row["splits"]["holdout"]["verdict"] == "INCONCLUSIVE" for row in rows)
                exp["status"] = "INCONCLUSIVE" if uncertain else "COMPLETED"
                exp["conclusion"] = "INCONCLUSIVE" if uncertain else "NO_WINNER"
                exp["completed_at"] = now()
                if not uncertain:
                    for cand in exp["candidates"]:
                        _negative(exp, cand, "; ".join(next(r for r in rows if r["id"] == cand["id"])["splits"]["holdout"]["reasons"]))
            else:
                performance = [{"id": r["id"], "quality": r["quality"], "runtime": r["runtime"],
                                "loc": r["loc"], "regressions": r["regressions"]} for r in passed]
                frontier = pareto_frontier(performance)
                exp["pareto_frontier"] = frontier
                winner = frontier[0] if len(frontier) == 1 else None
                if winner is None and len(frontier) > 1:
                    best_quality = max(r["quality"] for r in performance)
                    tolerance = exp["success_criteria"].get("quality_tolerance", 0)
                    near = [r for r in performance if r["id"] in frontier and best_quality - r["quality"] <= tolerance]
                    near.sort(key=lambda r: (r["runtime"], r["loc"], r["id"]))
                    if len(near) >= 2 and near[0]["runtime"] < near[1]["runtime"] and near[0]["loc"] <= near[1]["loc"]:
                        winner = near[0]["id"]
                if winner and exp["corpus"]["cases"]["qualified"] and exp["corpus"].get("real_holdout", False) and exp["budget"] != "TINY":
                    exp["winner"] = winner
                    exp["conclusion"] = "WINNER"
                    exp["confidence"] = "MEDIUM" if len(exp["corpus"]["cases"]["holdout"]) < 5 else "HIGH"
                    exp["status"] = "COMPLETED"
                    exp["completed_at"] = now()
                    for cand in exp["candidates"]:
                        if cand["id"] != winner and latest_run(cand, "holdout"):
                            reason = ("failed frozen holdout criteria" if
                                      next(r for r in rows if r["id"] == cand["id"])["splits"]["holdout"]["verdict"] != "PASS"
                                      else f"inferior to candidate {winner} on the selected production tradeoff")
                            _negative(exp, cand, reason)
                else:
                    exp["winner"] = None
                    exp["conclusion"] = "INCONCLUSIVE"
                    exp["status"] = "INCONCLUSIVE"
                    exp["confidence"] = "LOW"
                    exp["completed_at"] = now()
    exp["history"].append({"at": now(), "event": "compared", "conclusion": exp["conclusion"]})
    save(exp)
    return report(exp)


def report(exp):
    rows = [_comparison_row(exp, cand) for cand in exp["candidates"]]
    return {"schema": 1, "experiment_id": exp["experiment_id"], "question": exp["question"],
            "status": exp["status"], "conclusion": exp["conclusion"], "winner": exp["winner"],
            "confidence": exp["confidence"], "candidates": rows, "pareto_frontier": exp.get("pareto_frontier", []),
            "negative_findings": exp["negative_findings"], "promotion_task": exp["promotion_task"],
            "budget": {"class": exp["budget"], "seconds_used": round(sum(
                r["elapsed_seconds"] for c in exp["candidates"] for r in c["runs"]), 2),
                "seconds_limit": BUDGETS[exp["budget"]]["seconds"]},
            "holdout": {"subject_separated": exp.get("corpus", {}).get("cases", {}).get("qualified", False),
                        "real_photo": exp.get("corpus", {}).get("real_holdout", False)},
            "human_review_count": len(exp.get("human_reviews", []))}


def promote(exp, tasks=None):
    if exp["status"] != "COMPLETED" or exp["conclusion"] != "WINNER" or not exp["winner"]:
        raise ValueError("only a completed, qualified winner can propose production work")
    if exp["promotion_task"]:
        raise ValueError("promotion task already exists")
    winner = candidate(exp, exp["winner"])
    hold = latest_run(winner, "holdout")
    if assess(exp, hold)["verdict"] != "PASS":
        raise ValueError("holdout gate failed")
    tasks = cp.load_all() if tasks is None else tasks
    task, duplicate = cp.intake(f"Production implementation of {exp['title']} ({exp['experiment_id']})",
                                tasks, task_type="feature", subsystem="pipeline", priority="P2",
                                acceptance=[f"Reproduce {exp['experiment_id']} winner behavior without copying prototype code",
                                            f"Meet frozen metric {exp['success_criteria']['metric']} and holdout limits",
                                            "Pass normal PR Governor, Quality Lab, merge queue, and release gates"],
                                scan_repo=False)
    if duplicate["duplicate"]["verdict"] == "DUPLICATE":
        raise ValueError("similar Control Plane task already exists")
    task.files_hint = winner["complexity"].get("paths", [])
    task.required_tests = ["focused regression tests", "Quality Lab corpus and performance comparison"]
    task.risk_hint = "HIGH" if "retouch/engine.py" in task.files_hint else "MEDIUM"
    task.save()
    exp["promotion_task"] = task.id
    exp["status"] = "PROMOTED"
    exp["history"].append({"at": now(), "event": "promotion proposal created", "task": task.id})
    save(exp)
    return {"experiment_id": exp["experiment_id"], "winner": exp["winner"],
            "task_id": task.id, "prototype_code_copied": False}


def mission_status(experiments=None):
    rows = list_all() if experiments is None else experiments
    complete = [r for r in rows if r["status"] in ("COMPLETED", "INCONCLUSIVE", "PROMOTED", "ARCHIVED")]
    return {"schema": 1, "running": sum(r["status"] == "RUNNING" for r in rows),
            "analyzing": sum(r["status"] == "ANALYZING" for r in rows),
            "completed_today": sum(r.get("completed_at", "")[:10] == now()[:10] for r in complete),
            "experiments": len(rows), "candidate_count": sum(len(r["candidates"]) for r in rows),
            "promoted": sum(bool(r["promotion_task"]) for r in rows),
            "winners": sum(r["conclusion"] == "WINNER" for r in rows),
            "no_winner": sum(r["conclusion"] == "NO_WINNER" for r in rows),
            "negative_result_reuse": sum(r["similarity"]["verdict"] == "RETRY_WITH_NEW_EVIDENCE" for r in rows),
            "human_review_count": sum(len(r.get("human_reviews", [])) for r in rows),
            "compute_seconds": round(sum(run["elapsed_seconds"] for r in rows for c in r["candidates"] for run in c["runs"]), 2)}


def sync_attention(exp, state=None, *, save_state=True):
    state = attention.load() if state is None else state
    state["alerts"] = [a for a in state["alerts"] if not (a.get("source") == "experiment-lab" and a.get("subject") == exp["experiment_id"])]
    if exp["conclusion"] in ("WINNER", "INCONCLUSIVE"):
        tradeoff = len(exp.get("pareto_frontier", [])) > 1
        architecture = tradeoff and exp.get("constraints", {}).get("architecture_review_required", False)
        signal = attention.signal("experiment-lab", exp["experiment_id"], "result", f"Review {exp['experiment_id']}: {exp['conclusion']}",
                                  severity="error" if architecture else "warning", blocking=1 if tradeoff else 0,
                                  priority="P2", evidence=[exp["question"], exp["conclusion"]])
        state["alerts"].append(signal)
    if save_state:
        attention.save(state)
    return state["alerts"]


def sync_negatives(exp, rd_state=None, *, persist=True):
    """Reuse R&D Director negative memory; experiment manifest keeps its own copy."""
    from scripts.dev.rd_director import core as rd
    rd_state = rd.load() if rd_state is None else rd_state
    added = []
    for entry in exp["negative_findings"]:
        if entry.get("rd_negative_id"):
            continue
        item = rd.add_negative_result(rd_state,
                                      f"{exp['experiment_id']} {entry['approach']} ({entry['mechanism']})",
                                      entry["observed"], "rejected by experiment",
                                      retry_unless=entry["retry_unless"])
        entry["rd_negative_id"] = item["id"]
        added.append(item["id"])
    if persist and added:
        rd.save(rd_state)
        save(exp)
    return added


def blind_sheet(exp, split="holdout"):
    """Neutral copies hide candidate identity until preferences are recorded."""
    import random
    import shutil
    if split not in ("validation", "holdout"):
        raise ValueError("blinded review supports validation or holdout")
    chosen = [c for c in exp["candidates"] if latest_run(c, split)]
    if not chosen:
        raise ValueError("no measured candidates for this split")
    base = read_json(exp["baseline"]["report"])
    root = WORK_ROOT / exp["experiment_id"] / "blind" / split
    root.mkdir(parents=True, exist_ok=True)
    rng = random.SystemRandom()
    sheet, key = [], {}
    for case_id in exp["corpus"]["cases"][split]:
        for recipe in exp["baseline"]["recipes"]:
            images = [("baseline", Path(base["cases"][case_id]["recipes"][recipe]["output"]))]
            for cand in chosen:
                report = read_json(latest_run(cand, split)["report"])
                item = report.get("cases", {}).get(case_id, {}).get("recipes", {}).get(recipe, {})
                if item.get("output"):
                    images.append((cand["id"], Path(item["output"])))
            if len(images) < 2:
                continue
            rng.shuffle(images)
            labels = {}
            for n, (identity, path) in enumerate(images):
                if not path.is_file():
                    raise ValueError(f"review image missing: {path}")
                label = chr(ord("X") + n) if n < 3 else f"Image-{n+1}"
                dest = root / f"{case_id}-{recipe}-{label}{path.suffix}"
                shutil.copy2(path, dest)
                labels[label] = identity
            key[f"{case_id}:{recipe}"] = labels
            sheet.append({"case": case_id, "recipe": recipe,
                          "images": {label: str(root / f"{case_id}-{recipe}-{label}{images[n][1].suffix}")
                                     for n, label in enumerate(labels)}})
    write_json(root / "blind-key.json", key)
    write_json(root / "review-sheet.json", {"schema": 1, "experiment_id": exp["experiment_id"],
                                            "split": split, "items": sheet,
                                            "instructions": "Record preferred label before revealing identities"})
    return {"schema": 1, "sheet": str(root / "review-sheet.json"), "items": len(sheet),
            "identity_key_sealed": str(root / "blind-key.json")}


def record_review(exp, split, votes):
    path = WORK_ROOT / exp["experiment_id"] / "blind" / split / "blind-key.json"
    key = read_json(path)
    if not isinstance(key, dict) or not isinstance(votes, list) or not votes:
        raise ValueError("blind sheet and nonempty vote array required")
    if any(r.get("split") == split for r in exp.get("human_reviews", [])):
        raise ValueError("review for this split already recorded")
    resolved = []
    for vote in votes:
        reference = f"{vote.get('case')}:{vote.get('recipe')}"
        label = vote.get("preferred")
        if reference not in key or label not in key[reference]:
            raise ValueError(f"unknown blind vote {reference}/{label}")
        resolved.append({"case": vote["case"], "recipe": vote["recipe"],
                         "preferred": key[reference][label], "label": label})
    exp.setdefault("human_reviews", []).append({"split": split, "votes": resolved,
                                                  "at": now(), "count": len(resolved)})
    save(exp)
    return {"recorded": len(resolved), "split": split,
            "preferences": {name: sum(v["preferred"] == name for v in resolved)
                            for name in sorted({v["preferred"] for v in resolved})}}


def archive(exp):
    if exp["status"] not in ("COMPLETED", "INCONCLUSIVE", "REJECTED", "PROMOTED"):
        raise ValueError("only concluded experiments can be archived")
    exp["status"] = "ARCHIVED"
    exp["completed_at"] = exp["completed_at"] or now()
    exp["history"].append({"at": now(), "event": "archived; prototypes retained until explicit cleanup"})
    save(exp)
    return {"experiment_id": exp["experiment_id"], "status": "ARCHIVED",
            "prototype_worktrees": [c["worktree"] for c in exp["candidates"] if c["worktree"]]}


def cleanup(exp, *, retention_days=30, apply=False, discard_prototypes=False):
    from datetime import timedelta
    if exp["status"] != "ARCHIVED":
        raise ValueError("archive before prototype cleanup")
    completed = exp.get("completed_at")
    if not completed:
        raise ValueError("completion time unavailable")
    if datetime.now(timezone.utc) < datetime.fromisoformat(completed) + timedelta(days=retention_days):
        raise ValueError("retention period has not elapsed")
    managed = (WORK_ROOT / exp["experiment_id"] / "worktrees").resolve()
    paths = []
    for cand in exp["candidates"]:
        if not cand["worktree"]:
            continue
        path = Path(cand["worktree"]).resolve()
        if path != managed / cand["id"]:
            raise ValueError("refusing cleanup outside managed experiment worktrees")
        if path.exists():
            paths.append(path)
    if apply and paths and not discard_prototypes:
        raise ValueError("--discard-prototypes required to delete experiment code")
    removed = []
    if apply:
        for path in paths:
            subprocess.run(["git", "worktree", "remove", "--force", str(path)], cwd=ROOT,
                           capture_output=True, text=True, check=True, timeout=60)
            removed.append(str(path))
        exp["history"].append({"at": now(), "event": "prototype worktrees removed", "paths": removed})
        save(exp)
    return {"eligible": [str(p) for p in paths], "removed": removed,
            "durable_evidence": str(path_for(exp["experiment_id"]))}
