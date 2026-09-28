#!/usr/bin/env python3
"""PR triage, risk classification, overlap detection and merge-queue support
for the Retouch engine's high-velocity AI-assisted workflow.

Design objective: spend machine attention so the single human owner only
reviews consequential changes. Deterministic, stdlib-only, reuses
scripts/dev/governance.py.

Commands (see --help text per subcommand):
    risk BASE [HEAD]      classify a PR's risk (LOW/MEDIUM/HIGH) + JSON report
    triage [BASE]         batch-classify open PRs (via `gh`) for the owner
    overlap               detect open PRs touching the same files/stages
    queue                 show the merge queue (validated, in risk order)
    post-merge [SHA]      validate main after a merge (smoke + stats deltas)
    guardrails            velocity health warnings (reverts, churn, flakiness)

State lives in .git/governance-state.json (untracked; never committed).
Risk policy tables are at the top of this file — edit tables, not logic.

Exit codes: 0 ok, 1 blocking finding (for CI), 2 setup error.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_FILE = ROOT / ".git" / "governance-state.json"
GOVERNANCE = ROOT / "scripts" / "dev" / "governance.py"

# ---------------------------------------------------------------------------
# Risk policy tables (edit here, not in logic)
# ---------------------------------------------------------------------------

# File glob -> always HIGH. Semantic impact, not line count.
HIGH_RISK_PATHS = {
    "retouch/engine.py": "core orchestrator / pipeline ordering",
    "retouch/params.py": "ParamSpec registry — every recipe/CLI/GUI default",
    "retouch/detection.py": "face detection — everything downstream depends on it",
    "retouch/perf_optimizations.py": "process pools / concurrency",
    "retouch/precision.py": "dtype/precision contracts",
    "retouch/io.py": "file IO / export formats / C2PA",
    "retouch/content_credentials.py": "signing / security",
    "retouch/recipe_schema.py": "recipe validation gate",
    ".github/workflows/": "CI permissions",
    "scripts/dev/governance.py": "governance itself must not self-approve",
    "scripts/dev/pr_governor.py": "merge infrastructure must not self-approve",
    "pyproject.toml": "dependency pins",
    "uv.lock": "dependency lockfile",
    "models/manifest.json": "model acquisition",
}

# HIGH if a diff hunk in these files removes or weakens validation.
TEST_WEAKENING_PATHS = ("tests/", ".github/workflows/")

# Path prefix -> risk floor.
FLOOR_RULES = [
    ("tests/", "LOW"),
    ("docs/", "LOW"),
    ("scripts/dev/", "MEDIUM"),
    ("scripts/qa/", "LOW"),
    ("presets/", "MEDIUM"),   # recipes change shipped behavior
    ("retouch/gui_", "MEDIUM"),
    ("gui", "MEDIUM"),
    ("cli.py", "MEDIUM"),
    ("retouch/", "MEDIUM"),   # default for engine code
]

RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

# Suspicious-diff patterns in tests/CI: (regex, why). Applied to removed/added lines.
SUSPICIOUS = [
    (re.compile(r"^\-\s*assert\b"), "assertion removed"),
    (re.compile(r"^\+.*\bpytest\.mark\.(skip|xfail)\b"), "skip/xfail added"),
    (re.compile(r"^\-\s*def test_"), "test deleted"),
    (re.compile(r"(threshold|thr|tol|tolerance|atol|rtol|epsilon|eps)\s*=\s*([0-9.eE+-]+)"),
     "threshold literal changed (check direction)"),
]

# Golden/hash files — any change is at least MEDIUM, reported loudly.
GOLDEN_PATHS = ("tests/golden_",)

# Impact map: module -> related tests (config-driven; naming fallback exists).
# Extend as needed; tests/test_<module>.py is tried automatically first.
IMPACT_MAP = {
    "retouch/engine.py": ["tests/test_engine.py", "tests/test_integration.py",
                          "tests/test_golden_pipeline.py"],
    "retouch/params.py": ["tests/test_recipe_validation.py", "tests/test_gui.py",
                          "tests/test_params.py"],
    "retouch/detection.py": ["tests/test_detection.py", "tests/test_detection_person_gate.py"],
    "retouch/parsing.py": ["tests/test_parsing_class_segmenter.py", "tests/test_wig_hair_growth.py"],
    "retouch/perf_optimizations.py": ["tests/test_engine.py", "tests/test_integration.py"],
    "cli.py": ["tests/test_cli.py", "tests/test_cli_helpers.py"],
    "gui.py": ["tests/test_gui.py"],
}

# Merge-queue smoke suite (post-merge validation).
SMOKE_TESTS = ["tests/test_recipe_validation.py", "tests/test_golden_pipeline.py"]


# ---------------------------------------------------------------------------
# Shell helpers
# ---------------------------------------------------------------------------

def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def gh(*args: str) -> str:
    r = subprocess.run(["gh", *args], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def load_state() -> dict:
    if STATE_FILE.is_file():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"pr_reports": {}, "merge_history": [], "test_runs": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=1, sort_keys=True))


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Diff model
# ---------------------------------------------------------------------------

def pr_diff(base: str, head: str) -> dict:
    """Return files changed with per-file diff text (bounded)."""
    names = git("diff", "--name-status", f"{base}...{head}").splitlines()
    numstat = git("diff", "--numstat", f"{base}...{head}").splitlines()
    stats = {}
    for line in numstat:
        parts = line.split("\t")
        if len(parts) == 3:
            a, d, path = parts
            stats[path] = (int(a) if a != "-" else 0, int(d) if d != "-" else 0)
    files = []
    for line in names:
        parts = line.split("\t")
        if len(parts) >= 2:
            status, path = parts[0], parts[-1]
            a, d = stats.get(path, (0, 0))
            files.append({"path": path, "status": status, "added": a, "deleted": d})
    return {"files": files}


def file_diff_text(base: str, head: str, path: str) -> str:
    try:
        return git("diff", f"{base}...{head}", "--", path)
    except RuntimeError:
        return ""


# ---------------------------------------------------------------------------
# 1. Risk classification
# ---------------------------------------------------------------------------

def classify(base: str, head: str) -> dict:
    diff = pr_diff(base, head)
    files = diff["files"]
    total_add = sum(f["added"] for f in files)
    total_del = sum(f["deleted"] for f in files)

    risk = "LOW"
    reasons: list[str] = []
    warnings: list[str] = []

    def bump(level: str, why: str) -> None:
        nonlocal risk
        if RANK[level] > RANK[risk]:
            risk = level
        reasons.append(f"{level}: {why}")

    # --- path-based floors and HIGH zones ---
    for f in files:
        p = f["path"]
        high_hit = False
        for hp, why in HIGH_RISK_PATHS.items():
            if p == hp or (hp.endswith("/") and p.startswith(hp)):
                bump("HIGH", f"{p} — {why}")
                high_hit = True
                break
        if not high_hit:
            for prefix, floor in FLOOR_RULES:
                if p.startswith(prefix) or p == prefix.rstrip("/"):
                    bump(floor, f"{p} under {prefix} (floor {floor})")
                    break
        if p.startswith(GOLDEN_PATHS):
            bump("MEDIUM", f"{p} is a golden/reference artifact — verify regeneration intent")
        if f["status"] == "D" and p.startswith("tests/"):
            bump("MEDIUM", f"test file deleted: {p}")

    # --- suspicious validation weakening (only scanned in test/CI files) ---
    suspicious_hits = []
    for f in files:
        p = f["path"]
        if not any(p.startswith(t) for t in TEST_WEAKENING_PATHS):
            continue
        for line in file_diff_text(base, head, p).splitlines():
            if not line.startswith(("+", "-")) or line.startswith(("+++", "---")):
                continue
            for pat, why in SUSPICIOUS:
                if pat.search(line):
                    suspicious_hits.append(f"{p}: {why} ({line.strip()[:80]})")
    # Deduplicate while preserving order; cap.
    seen = set()
    for h in suspicious_hits:
        if h not in seen:
            seen.add(h)
            warnings.append(h)
    if suspicious_hits:
        bump("MEDIUM", f"{len(seen)} suspicious validation change(s) — a PR is never LOW "
                       "just because its weakened tests pass")
        for w in list(seen)[:10]:
            reasons.append(f"  suspicious: {w}")

    # --- scale signals (weak, semantic impact dominates) ---
    prod_files = [f for f in files if f["path"].startswith("retouch/") or f["path"] in ("gui.py", "cli.py")]
    if len(prod_files) > 15:
        bump("MEDIUM", f"{len(prod_files)} production files touched — check subsystem count")
    if total_add + total_del > 3000:
        bump("MEDIUM", f"large diff (+{total_add}/-{total_del}) — confirm it's one coherent change")
    if not files:
        warnings.append("empty diff vs base — nothing to merge")

    # --- new module? mention placement ---
    new_modules = [f["path"] for f in files if f["status"] == "A" and f["path"].startswith("retouch/")]
    for m in new_modules:
        reasons.append(f"INFO: new module {m} — pipeline placement + ARCH_REQUIRED entry to confirm")

    return {
        "risk": risk,
        "reasons": reasons,
        "warnings": warnings,
        "files_changed": len(files),
        "additions": total_add,
        "deletions": total_del,
        "production_files": len(prod_files),
        "base": base,
        "head": head,
    }


# ---------------------------------------------------------------------------
# 9/10. Test selection via impact map
# ---------------------------------------------------------------------------

def recommended_tests(base: str, head: str) -> list[str]:
    diff = pr_diff(base, head)
    tests: list[str] = []
    for f in diff["files"]:
        p = f["path"]
        if p.startswith("tests/test_"):
            tests.append(p)
        tests += IMPACT_MAP.get(p, [])
        if p.startswith("retouch/") and p.endswith(".py"):
            cand = f"tests/test_{Path(p).stem}.py"
            if (ROOT / cand).is_file():
                tests.append(cand)
    seen, out = set(), []
    for t in tests:
        if t not in seen and (ROOT / t).is_file():
            seen.add(t)
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# 6. Overlap detection across open PRs
# ---------------------------------------------------------------------------

def open_prs() -> list[dict]:
    try:
        raw = gh("pr", "list", "--limit", "100", "--json", "number,title,headRefName,baseRefName,files")
        return json.loads(raw)
    except (RuntimeError, json.JSONDecodeError) as e:
        print(f"warning: cannot list PRs ({e}); overlap detection skipped", file=sys.stderr)
        return []


def detect_overlap(prs: list[dict]) -> list[str]:
    owners: dict[str, list[int]] = {}
    for pr in prs:
        for f in pr.get("files", []):
            owners.setdefault(f["path"], []).append(pr["number"])
    out = []
    for path, nums in sorted(owners.items()):
        nums = sorted(set(nums))
        if len(nums) > 1:
            out.append(f"PRs {', '.join('#' + str(n) for n in nums)} all modify {path} — "
                       "merge ordering may matter")
    return out


# ---------------------------------------------------------------------------
# 7. Duplicate-work hints (weak heuristic; never blocks)
# ---------------------------------------------------------------------------

def duplicate_hints(base: str, head: str, prs: list[dict]) -> list[str]:
    mine = {f["path"] for f in pr_diff(base, head)["files"]}
    hints = []
    for pr in prs:
        theirs = {f["path"] for f in pr.get("files", [])}
        shared = mine & theirs
        if shared:
            hints.append(f"PR #{pr['number']} ({pr['title'][:60]}) touches {len(shared)} "
                         f"same file(s): {', '.join(sorted(shared)[:5])}")
    return hints


# ---------------------------------------------------------------------------
# 11. Post-merge validation
# ---------------------------------------------------------------------------

def post_merge(smoke_only: bool = True) -> dict:
    result = {"timestamp": now(), "checks": {}, "ok": True}

    gov = subprocess.run([sys.executable, str(GOVERNANCE), "--ci"],
                         cwd=ROOT, capture_output=True, text=True)
    result["checks"]["governance"] = {"exit": gov.returncode}
    if gov.returncode != 0:
        result["ok"] = False
        result["checks"]["governance"]["tail"] = gov.stdout.strip().splitlines()[-5:]

    py = str(ROOT / ".venv" / "bin" / "python")
    if not Path(py).is_file():
        py = sys.executable
    env = {"RETOUCH_GPU": "0", "RETOUCH_MEDIAPIPE_BACKEND": "legacy",
           "PYTHONNOUSERSITE": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"}
    suite = SMOKE_TESTS if smoke_only else ["tests/"]
    t = subprocess.run([py, "-m", "pytest", *suite, "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True,
                       env={**env}, timeout=1200)
    result["checks"]["smoke"] = {"exit": t.returncode,
                                 "tail": t.stdout.strip().splitlines()[-3:]}
    if t.returncode != 0:
        result["ok"] = False

    # Stats deltas: test/module count drops signal accidental deletion.
    cur_modules = len(list(ROOT.glob("retouch/*.py")))
    prev = load_state().get("last_main_stats", {})
    result["checks"]["stats"] = {"modules_now": cur_modules,
                                 "modules_prev": prev.get("modules", cur_modules)}
    if cur_modules < prev.get("modules", cur_modules):
        result["ok"] = False
        result["checks"]["stats"]["alarm"] = "module count dropped — accidental deletion?"

    state = load_state()
    state["last_main_stats"] = {"modules": cur_modules, "checked": now()}
    state.setdefault("merge_history", []).append(
        {"sha": git("rev-parse", "--short", "HEAD").strip(), "ok": result["ok"], "ts": now()})
    save_state(state)
    return result


# ---------------------------------------------------------------------------
# 13. Velocity guardrails
# ---------------------------------------------------------------------------

def guardrails() -> list[str]:
    state = load_state()
    warnings = []
    hist = state.get("merge_history", [])[-20:]
    if hist:
        fails = [h for h in hist if not h.get("ok")]
        if len(fails) >= 3:
            warnings.append(f"{len(fails)}/{len(hist)} recent post-merge validations failed — "
                            "merge quality is degrading; slow the queue")
    # Revert detection in git log.
    log = git("log", "--oneline", "-50").lower()
    reverts = log.count("revert")
    if reverts >= 3:
        warnings.append(f"{reverts} reverts in last 50 commits — high revert rate; "
                        "classify more aggressively")
    # Same-subsystem churn: one file touched by many recent commits.
    churn = git("log", "--since=7 days ago", "--name-only", "--format=").split()
    from collections import Counter
    top = Counter(f for f in churn if f.startswith("retouch/")).most_common(3)
    for path, n in top:
        if n >= 8:
            warnings.append(f"{path} changed {n}x in 7 days — subsystem churn; "
                            "agents may be rewriting each other's work")
    return warnings


# ---------------------------------------------------------------------------
# 14. Owner summary
# ---------------------------------------------------------------------------

def owner_summary(report: dict, tests: list[str], overlaps: list[str],
                  dupes: list[str], pr_number: int | None = None) -> str:
    r = report["risk"]
    eligible = (r == "LOW" and not report["warnings"]
                and report["files_changed"] > 0)
    lines = [
        f"PR {('#' + str(pr_number)) if pr_number else report['head']}",
        f"Risk: {r}" + ("  (auto-merge eligible IF all checks pass)" if eligible else ""),
        f"Confidence: deterministic rules, {report['files_changed']} files, "
        f"+{report['additions']}/-{report['deletions']}",
        "",
        "Why:",
    ]
    lines += [f"  - {x}" for x in report["reasons"][:12]] or ["  - (no signals)"]
    if report["warnings"]:
        lines.append("Suspicious changes:")
        lines += [f"  - {w}" for w in report["warnings"][:8]]
    lines.append(f"Tests: {len(tests)} targeted test file(s) recommended")
    lines += [f"  - {t}" for t in tests[:8]]
    if overlaps:
        lines.append("Overlapping PRs:")
        lines += [f"  - {o}" for o in overlaps[:5]]
    if dupes:
        lines.append("Possible duplicate work:")
        lines += [f"  - {d}" for d in dupes[:5]]
    attention = {
        "LOW": "none — verify checks green, merge",
        "MEDIUM": "confirm pipeline placement + backwards-compatible defaults",
        "HIGH": "architecture review required; never auto-merge",
    }[r]
    lines.append(f"Human attention needed: {attention}")
    r = report["risk"]
    if report["files_changed"] == 0:
        eligibility = "no — empty diff"
    elif r == "LOW" and not report["warnings"]:
        eligibility = "yes (LOW + all checks pass)"
    else:
        eligibility = f"no — {r}"
    lines.append(f"Merge eligibility: {eligibility}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_risk(args: argparse.Namespace) -> int:
    report = classify(args.base, args.head)
    tests = recommended_tests(args.base, args.head)
    overlaps, dupes = [], []
    if not args.offline:
        prs = open_prs()
        overlaps = detect_overlap(prs)
        dupes = duplicate_hints(args.base, args.head, prs)

    state = load_state()
    state["pr_reports"][report["head"]] = {"ts": now(), "risk": report["risk"]}
    save_state(state)

    if args.json:
        out = {**report, "tests": tests, "overlaps": overlaps, "duplicates": dupes}
        print(json.dumps(out, indent=1))
    else:
        print(owner_summary(report, tests, overlaps, dupes, args.pr))
    return 1 if report["risk"] == "HIGH" and args.ci else 0


def cmd_triage(args: argparse.Namespace) -> int:
    prs = open_prs()
    if not prs:
        print("no open PRs (or gh unavailable)")
        return 0
    rows = []
    for pr in prs:
        try:
            report = classify(pr["baseRefName"], pr["headRefName"])
        except RuntimeError as e:
            rows.append((f"#{pr['number']}", "?", f"classify failed: {e}"))
            continue
        rows.append((f"#{pr['number']}", report["risk"], pr["title"][:70]))
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "?": 3}
    rows.sort(key=lambda r: order.get(r[1], 3))
    print(f"{len(rows)} open PRs — triage order:")
    for num, risk, title in rows:
        print(f"  [{risk:6}] {num:5} {title}")
    buckets = {}
    for _, risk, _ in rows:
        buckets[risk] = buckets.get(risk, 0) + 1
    print(f"buckets: " + ", ".join(f"{k}={v}" for k, v in sorted(buckets.items())))
    for w in detect_overlap(prs):
        print(f"  OVERLAP: {w}")
    return 0


def cmd_overlap(args: argparse.Namespace) -> int:
    prs = open_prs()
    hits = detect_overlap(prs)
    if not hits:
        print("no overlapping PRs")
        return 0
    for h in hits:
        print(h)
    return 0


def cmd_queue(args: argparse.Namespace) -> int:
    prs = open_prs()
    if not prs:
        print("merge queue empty")
        return 0
    print("Merge queue (validate against latest main before each merge):")
    ranked = []
    for pr in prs:
        try:
            r = classify(pr["baseRefName"], pr["headRefName"])["risk"]
        except RuntimeError:
            r = "?"
        ranked.append((r, pr["number"], pr["title"][:70]))
    order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "?": 3}
    for r, n, t in sorted(ranked, key=lambda x: order.get(x[0], 3)):
        action = {"LOW": "auto-merge candidate (after checks)",
                  "MEDIUM": "needs 1 human approval",
                  "HIGH": "needs architecture review",
                  "?": "needs manual classification"}[r]
        print(f"  #{n:<5} [{r:6}] {t}\n          -> {action}")
    return 0


def cmd_post_merge(args: argparse.Namespace) -> int:
    result = post_merge(smoke_only=not args.full)
    print(json.dumps(result, indent=1))
    return 0 if result["ok"] else 1


def cmd_guardrails(args: argparse.Namespace) -> int:
    warnings = guardrails()
    if not warnings:
        print("velocity guardrails: all quiet")
        return 0
    for w in warnings:
        print(f"[WARNING] {w}")
    return 0  # advisory only; never blocks


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="pr_governor",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("risk", help="classify a PR range")
    p.add_argument("base", nargs="?", default="origin/main")
    p.add_argument("head", nargs="?", default="HEAD")
    p.add_argument("--json", action="store_true")
    p.add_argument("--ci", action="store_true",
                   help="exit 1 when HIGH (blocks accidental HIGH auto-merge)")
    p.add_argument("--pr", type=int, default=None)
    p.add_argument("--offline", action="store_true",
                   help="skip gh overlap/duplicate lookups (fast CI path)")
    p.set_defaults(fn=cmd_risk)

    p = sub.add_parser("triage", help="batch-classify all open PRs")
    p.set_defaults(fn=cmd_triage)

    p = sub.add_parser("overlap", help="detect open PRs touching same files")
    p.set_defaults(fn=cmd_overlap)

    p = sub.add_parser("queue", help="show merge queue in risk order")
    p.set_defaults(fn=cmd_queue)

    p = sub.add_parser("post-merge", help="validate main after a merge")
    p.add_argument("--full", action="store_true", help="run full test suite")
    p.set_defaults(fn=cmd_post_merge)

    p = sub.add_parser("guardrails", help="velocity health warnings")
    p.set_defaults(fn=cmd_guardrails)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except RuntimeError as e:
        print(f"pr_governor: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
