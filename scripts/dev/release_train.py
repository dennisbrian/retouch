#!/usr/bin/env python3
"""Release train + canary + rollback-recommend for the Retouch engine.

Answers the question PR-level CI cannot: "does THIS batch of merged changes
work together?" Groups merge-PRs on main into release candidates, runs one
combined validation pass, manages dev/canary/stable channels via git tags,
generates the release manifest, and recommends (never executes) rollback.

Commands:
    candidates              PRs merged since last stable tag, risk-bucketed
    cut                     cut an RC tag from current main
    validate [SHA]          combined validation pass (used by cut and manually)
    manifest TAG            generate/print the release manifest for a tag
    channels                show dev/canary/stable pointers
    promote TAG canary|stable   move a channel pointer (human-run)
    bisect GOOD BAD [--cmd C]   isolate the merge that broke a check
    health                  main health vs last stable; rollback recommendation

Conventions:
    RC tags:      rc/YYYY.MM.DD-N   (lightweight, on main)
    Channel tags: canary, stable    (moved only by `promote`, human-run)
    State:        .git/governance-state.json (shared with pr_governor.py)

Rollback is RECOMMEND-ONLY. This tool never reverts, resets, or deletes.

Exit codes: 0 ok, 1 validation/health failure, 2 setup error.
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
PR_GOV = ROOT / "scripts" / "dev" / "pr_governor.py"

# Combined validation for an RC. Each: (name, argv, timeout_s).
# Full suite is the core check — PR tests pass individually; this proves the
# batch together. Benchmark + governance flank it.
def validation_steps(py: str) -> list[tuple[str, list[str], int]]:
    env_note = "RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy (set in run())"
    return [
        ("governance", [sys.executable, str(GOVERNANCE), "--ci"], 120),
        ("full test suite", [py, "-m", "pytest", "tests/", "-q", "-x",
                             "-p", "no:cacheprovider"], 3600),
        ("benchmarks", [py, "scripts/bench/benchmark.py", "--quiet",
                        "--no-save"], 1200),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def git(*args: str, check: bool = True) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout


def py_env() -> dict:
    import os
    e = dict(os.environ)
    e.update({"RETOUCH_GPU": "0", "RETOUCH_MEDIAPIPE_BACKEND": "legacy",
              "PYTHONNOUSERSITE": "1"})
    return e


def repo_python() -> str:
    p = ROOT / ".venv" / "bin" / "python"
    return str(p) if p.is_file() else sys.executable


def load_state() -> dict:
    if STATE_FILE.is_file():
        try:
            return json.loads(STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {}


def save_state(s: dict) -> None:
    STATE_FILE.write_text(json.dumps(s, indent=1, sort_keys=True))


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stable_sha() -> str | None:
    r = subprocess.run(["git", "rev-parse", "stable"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def merge_prs_since(ref: str | None) -> list[dict]:
    """Merge-PR commits on main after ref (or all, if ref is None)."""
    rng = f"{ref}..HEAD" if ref else "HEAD"
    out = []
    for line in git("log", "--format=%H%x09%s", rng).splitlines():
        sha, _, subject = line.partition("\t")
        m = re.match(r"Merge pull request #(\d+) from \S+", subject)
        if m:
            out.append({"sha": sha, "pr": int(m.group(1)),
                        "subject": subject})
    out.reverse()  # oldest first
    return out


def pr_risk(merge_sha: str) -> str:
    """Risk of a merged PR via pr_governor rules (parent1..merge)."""
    try:
        r = subprocess.run(
            [sys.executable, str(PR_GOV), "risk", f"{merge_sha}^1", merge_sha,
             "--offline", "--json"],
            cwd=ROOT, capture_output=True, text=True, timeout=60)
        if r.returncode in (0, 1) and r.stdout.strip():
            return json.loads(r.stdout)["risk"]
    except (RuntimeError, json.JSONDecodeError, subprocess.TimeoutExpired):
        pass
    return "?"


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------

def cmd_candidates(args: argparse.Namespace) -> int:
    ref = stable_sha()
    prs = merge_prs_since(ref)
    if not prs:
        print(f"no merged PRs since {'stable' if ref else 'beginning'} "
              f"({ref[:8] if ref else '—'})")
        return 0
    buckets: dict[str, list[dict]] = {"HIGH": [], "MEDIUM": [], "LOW": [], "?": []}
    for p in prs:
        p["risk"] = pr_risk(p["sha"])
        buckets.setdefault(p["risk"], []).append(p)

    print(f"Release train candidates since stable ({(ref or '')[:8] or '—'}): "
          f"{len(prs)} PRs\n")
    print("Included (auto-groupable):")
    for r in ("LOW", "MEDIUM"):
        for p in buckets[r]:
            print(f"  #{p['pr']:<5} [{r:6}] {p['subject'][:70]}")
    print("\nExcluded (need human decision before boarding):")
    for r in ("HIGH", "?"):
        for p in buckets[r]:
            print(f"  #{p['pr']:<5} [{r:6}] {p['subject'][:70]}")
    print(f"\nComposition: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))
    if args.json:
        print(json.dumps(prs, indent=1))
    return 0


# ---------------------------------------------------------------------------
# Validation (the "A+B+C together" check)
# ---------------------------------------------------------------------------

def run_validation(sha: str, full: bool = True) -> dict:
    py = repo_python()
    steps = validation_steps(py)
    if not full:
        steps = [s for s in steps if s[0] == "governance"]
    result = {"sha": sha, "ts": now(), "steps": {}, "ok": True}
    for name, argv, timeout in steps:
        r = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True,
                           env=py_env(), timeout=timeout)
        tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
        result["steps"][name] = {"exit": r.returncode, "tail": tail}
        if r.returncode != 0:
            result["ok"] = False
    return result


def cmd_validate(args: argparse.Namespace) -> int:
    sha = args.sha or git("rev-parse", "HEAD").strip()
    result = run_validation(sha, full=not args.gov_only)
    print(json.dumps(result, indent=1))
    state = load_state()
    state.setdefault("release_validations", []).append(result)
    save_state(state)
    return 0 if result["ok"] else 1


# ---------------------------------------------------------------------------
# Cut an RC
# ---------------------------------------------------------------------------

def next_rc_name() -> str:
    today = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    n = 1
    existing = set(git("tag", "-l", f"rc/{today}-*").split())
    while f"rc/{today}-{n:02d}" in existing:
        n += 1
    return f"rc/{today}-{n:02d}"


def cmd_cut(args: argparse.Namespace) -> int:
    prs = merge_prs_since(stable_sha())
    if not prs and not args.force:
        print("nothing merged since stable — no RC to cut (use --force to override)")
        return 1

    head = git("rev-parse", "HEAD").strip()
    rc = args.name or next_rc_name()

    print(f"cutting {rc} at {head[:8]} — running combined validation…")
    result = run_validation(head, full=not args.gov_only)
    status = "RELEASE READY" if result["ok"] else "BLOCKED"

    risks: dict[str, int] = {}
    for p in prs:
        r = pr_risk(p["sha"])
        risks[r] = risks.get(r, 0) + 1

    lines = [f"RELEASE CANDIDATE: {rc}", ""]
    for name, step in result["steps"].items():
        lines.append(f"{name:<18} {'GREEN' if step['exit'] == 0 else 'RED'}")
    lines += ["", "Risk composition:"]
    lines += [f"  {k:<7} {v} PRs" for k, v in sorted(risks.items())]
    human = risks.get("HIGH", 0) + risks.get("?", 0)
    lines += ["", f"Human review required: {human} item(s)", "", status]
    print("\n".join(lines))

    state = load_state()
    state.setdefault("release_candidates", {})[rc] = {
        "sha": head, "ts": now(), "ok": result["ok"], "risks": risks,
        "full_validation": not args.gov_only,
        "steps": {k: v["exit"] for k, v in result["steps"].items()},
        "prs": [p["pr"] for p in prs]}
    save_state(state)

    if not result["ok"]:
        print(f"\n{rc} NOT tagged — validation failed. "
              f"Run `release_train.py bisect` to isolate the bad merge.")
        return 1

    git("tag", rc, head)
    print(f"\ntagged {rc} -> {head[:8]} (local; push with: git push origin {rc})")
    return 0


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def cmd_manifest(args: argparse.Namespace) -> int:
    tag = args.tag
    sha = git("rev-parse", tag).strip()
    prev = stable_sha() if tag != "stable" else None
    base = prev or git("tag", "--sort=-creatordate").splitlines()[0] or f"{sha}~1"
    prs = merge_prs_since(base if base != sha else None)

    version = git("describe", "--tags", "--abbrev=0", check=False) or "unversioned"
    pyproject = (ROOT / "pyproject.toml").read_text()
    vm = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.M)
    version = vm.group(1) if vm else version

    feats, fixes, other = [], [], []
    for p in prs:
        s = p["subject"]
        body = git("log", "-1", "--format=%b", p["sha"])
        title = body.splitlines()[0] if body.strip() else s
        low = title.lower()
        if low.startswith(("fix", "bug")):
            fixes.append(title)
        elif low.startswith(("feat", "add")):
            feats.append(title)
        else:
            other.append(title)

    tests = "?"
    stats = subprocess.run([repo_python(), "-m", "pytest", "tests/",
                            "--collect-only", "-q"], cwd=ROOT,
                           capture_output=True, text=True, env=py_env(),
                           timeout=600)
    m = re.search(r"(\d+) tests? collected", stats.stdout)
    if m:
        tests = m.group(1)

    rc_info = load_state().get("release_candidates", {}).get(tag, {})
    lines = [
        f"# Release manifest — {tag}",
        "",
        f"Version: {version}",
        f"Commit: {sha[:8]}",
        f"Cut: {rc_info.get('ts', 'unknown')}",
        "",
        f"## Features ({len(feats)})"]
    lines += [f"+ {f}" for f in feats] or ["— none"]
    lines += ["", f"## Fixes ({len(fixes)})"]
    lines += [f"+ {f}" for f in fixes] or ["— none"]
    if other:
        lines += ["", f"## Other ({len(other)})"]
        lines += [f"+ {o}" for o in other]
    lines += ["", f"## Tests", f"{tests} collected"]
    if rc_info:
        lines += ["", "## RC validation"]
        for k, v in rc_info.get("steps", {}).items():
            lines.append(f"- {k}: {'GREEN' if v == 0 else 'RED'}")
        lines.append(f"- risk composition: {rc_info.get('risks', {})}")
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------------------
# Channels
# ---------------------------------------------------------------------------

def cmd_channels(args: argparse.Namespace) -> int:
    main_sha = git("rev-parse", "main").strip()[:8]
    print(f"dev    -> main ({main_sha})")
    for ch in ("canary", "stable"):
        r = subprocess.run(["git", "rev-parse", ch], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode == 0:
            sha = r.stdout.strip()
            subj = git("log", "-1", "--format=%s", sha).strip()
            print(f"{ch:<6} -> {sha[:8]}  {subj[:60]}")
        else:
            print(f"{ch:<6} -> (unset)")
    return 0


def cmd_promote(args: argparse.Namespace) -> int:
    tag, channel = args.tag, args.channel
    if channel not in ("canary", "stable"):
        print("channel must be canary or stable", file=sys.stderr)
        return 2
    sha = git("rev-parse", tag).strip()

    if channel == "stable":
        # Gate: stable only from a GREEN RC validation (state) or canary.
        state = load_state()
        rc_name = None
        for name, rc in state.get("release_candidates", {}).items():
            if rc.get("sha") == sha:
                rc_name = name
                if not rc.get("ok") or not rc.get("full_validation"):
                    print(f"refusing: {name} has no GREEN full validation "
                          f"(gov-only or failed) — re-cut with full validation "
                          f"before promoting to stable", file=sys.stderr)
                    return 1
        canary_sha = stable_sha_or("canary")
        if rc_name is None and canary_sha != sha:
            print(f"refusing: {tag} ({sha[:8]}) is neither a known GREEN RC "
                  f"nor current canary — promote to canary first", file=sys.stderr)
            return 1

    git("tag", "-f", channel, sha)
    print(f"{channel} -> {sha[:8]} ({tag})  [local tag; push: git push -f origin {channel}]")
    state = load_state()
    state.setdefault("channel_history", []).append(
        {"channel": channel, "sha": sha, "tag": tag, "ts": now(),
         "by": "human (promote command)"})
    save_state(state)
    return 0


def stable_sha_or(ref: str) -> str | None:
    r = subprocess.run(["git", "rev-parse", ref], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


# ---------------------------------------------------------------------------
# Bisect a bad batch
# ---------------------------------------------------------------------------

def cmd_bisect(args: argparse.Namespace) -> int:
    good, bad = args.good, args.bad
    merges = [l.split()[0] for l in
              git("log", "--format=%H", "--merges", f"{good}..{bad}").splitlines()]
    if not merges:
        print(f"no merge commits in {good}..{bad}")
        return 1
    merges.reverse()
    print(f"bisecting {len(merges)} merge commits "
          f"({'~' + str(len(merges).bit_length())} checkouts)…")

    check_cmd = args.cmd or " ".join(
        validation_steps(repo_python())[1][1])  # default: full suite
    lo, hi = 0, len(merges) - 1
    first_bad = None
    # Invariant: good is clean, bad is broken. merges ascending.
    # Binary search over merge points: test at mid; failure => first_bad <= mid.
    probe_bad = run_validation(bad, full=not args.gov_only)["ok"] is False
    if not probe_bad:
        print(f"validation passes at bad={bad[:8]} — nothing to bisect")
        return 0
    while lo <= hi:
        mid = (lo + hi) // 2
        sha = merges[mid]
        git("checkout", "-q", sha)
        ok = run_validation(sha, full=not args.gov_only)["ok"]
        print(f"  {sha[:8]} (merge {mid + 1}/{len(merges)}): "
              f"{'clean' if ok else 'BROKEN'}")
        if ok:
            lo = mid + 1
        else:
            first_bad = sha
            hi = mid - 1
    git("checkout", "-q", "main")

    if first_bad:
        subject = git("log", "-1", "--format=%s", first_bad).strip()
        m = re.search(r"#(\d+)", subject)
        print(f"\nRegression introduced by: {subject}"
              + (f" (PR #{m.group(1)})" if m else ""))
        state = load_state()
        state.setdefault("bisect_history", []).append(
            {"good": good, "bad": bad, "first_bad": first_bad,
             "subject": subject, "ts": now()})
        save_state(state)
        return 1
    print("could not isolate (flaky check?)")
    return 1


# ---------------------------------------------------------------------------
# Health monitoring + rollback recommendation (recommend-only)
# ---------------------------------------------------------------------------

def cmd_health(args: argparse.Namespace) -> int:
    stable = stable_sha()
    if not stable:
        print("no stable tag — nothing to compare against")
        return 0
    main = git("rev-parse", "main").strip()
    state = load_state()

    lines = ["MAIN HEALTH CHECK", f"stable: {stable[:8]}", f"main:   {main[:8]}", ""]
    issues = []

    recent = state.get("merge_history", [])[-10:]
    fails = [h for h in recent if not h.get("ok")]
    if fails:
        issues.append(f"post-merge validation failures: {len(fails)}/{len(recent)}")
    lines.append(f"post-merge validations: {len(recent) - len(fails)}/{len(recent)} green")

    merges_since = merge_prs_since(stable)
    lines.append(f"merges since stable: {len(merges_since)}")

    degraded = bool(fails)
    if degraded:
        lines += ["", "MAIN HEALTH DEGRADED", "",
                  f"Previous stable: {stable[:8]}",
                  f"Current main:    {main[:8]}",
                  f"Detected: {', '.join(issues)}", "",
                  f"Recommended action: ROLL BACK TO stable ({stable[:8]})",
                  f"Suspected merge window: {len(merges_since)} PR(s) since stable —",
                  f"  run: scripts/dev/release-train bisect {stable[:8]} {main[:8]}",
                  "",
                  "Rollback is human-approved only. Suggested manual steps:",
                  f"  git revert -m 1 <merge-sha>   # per offending merge, or",
                  f"  git tag -f stable {stable[:8]}  # re-point after fix",
                  "This tool will not revert on its own."]
        print("\n".join(lines))
        return 1
    lines.append("\nmain healthy vs stable")
    print("\n".join(lines))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="release_train",
                                 description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("candidates", help="PRs merged since stable, risk-bucketed")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_candidates)

    p = sub.add_parser("cut", help="validate main and tag an RC")
    p.add_argument("--name", default=None, help="override rc/YYYY.MM.DD-N")
    p.add_argument("--force", action="store_true")
    p.add_argument("--gov-only", action="store_true",
                   help="governance-only validation (fast; not for real RCs)")
    p.set_defaults(fn=cmd_cut)

    p = sub.add_parser("validate", help="combined validation pass on a SHA")
    p.add_argument("sha", nargs="?", default=None)
    p.add_argument("--gov-only", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("manifest", help="generate release manifest for a tag")
    p.add_argument("tag")
    p.set_defaults(fn=cmd_manifest)

    p = sub.add_parser("channels", help="show dev/canary/stable pointers")
    p.set_defaults(fn=cmd_channels)

    p = sub.add_parser("promote", help="move canary/stable pointer (human-run)")
    p.add_argument("tag")
    p.add_argument("channel")
    p.set_defaults(fn=cmd_promote)

    p = sub.add_parser("bisect", help="binary-search merge commits for a regression")
    p.add_argument("good")
    p.add_argument("bad")
    p.add_argument("--cmd", default=None, help="custom check command")
    p.add_argument("--gov-only", action="store_true")
    p.set_defaults(fn=cmd_bisect)

    p = sub.add_parser("health", help="main health vs stable + rollback recommendation")
    p.set_defaults(fn=cmd_health)

    args = ap.parse_args(argv)
    try:
        return args.fn(args)
    except RuntimeError as e:
        print(f"release_train: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
