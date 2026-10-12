#!/usr/bin/env python3
"""Repository governance checks for the Retouch engine.

One command runs: repo stats, doc-link validation, doc-index coverage,
architecture-drift check, stale-metadata check, and repo-pinned command
validation. Stdlib only.

Severity model (per docs/GOVERNANCE.md):
    ERROR   — dangerous/inconsistent state; fails the check (exit 1)
    WARNING — likely stale; report only by default, fail with --strict
    INFO    — housekeeping opportunity; never fails

Modes:
    (default) / --report   human-readable report
    --stats                print generated repo stats only
    --write-stats          regenerate docs/REPO_STATS.md (never fails)
    --ci                   fast PR checks (ERRORs only; skips test collection)
    --strict               WARNINGs also fail

Exit codes: 0 = no failures, 1 = check failure, 2 = setup error.
Run from the repo root; prefers .venv/bin/python for test collection.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "docs" / "INDEX.md"
# Git-ignored local run outputs (see .gitignore); doc links into it are not checked.
ARTIFACT_DIR = ROOT / "test_output"
STATS_FILE = ROOT / "docs" / "REPO_STATS.md"

# ---------------------------------------------------------------------------
# Policy tables (edit here; no logic changes needed)
# ---------------------------------------------------------------------------

# Doc files intentionally excluded from the INDEX coverage audit.
# Basenames or paths relative to the repo root. TODO_WEEK_* are weekly working
# lists, not reference docs; session logs are append-only records.
INDEX_EXCLUDE_BASENAMES = {"INDEX.md"}
INDEX_EXCLUDE_GLOBS = ("docs/plans/TODO_WEEK_*.md", "docs/review/session/**")
INDEX_EXCLUDE_PATHS = set()  # add "docs/foo.md" for one-off exclusions

# Canonical architecture docs. The drift check verifies every retouch/*.py
# module is represented in at least one canonical doc.
CANONICAL_ARCH_DOCS = [
    "docs/architecture/ARCHITECTURE.md",
    "docs/architecture/API.md",
    "docs/architecture/PIPELINE_FLOW.md",
]

# Doc-file basenames allowed to satisfy a module's representation even if the
# canonical docs name the module's family instead of the file itself.
# e.g. params.py is represented by "PROCESSING_PARAMS"/"ParamSpec" text.
ARCH_ALIASES = {
    "params": ("PROCESSING_PARAMS", "ParamSpec"),
    "engine": ("RetouchEngine", "_stage_"),
    "utils": ("restore_outside_support", "yaw_gate_factor", "YAW_GATE"),
}

# "Significant" modules that MUST appear in a canonical doc (ERROR if absent).
# Everything else missing is a WARNING. Keep this list short and current.
ARCH_REQUIRED = [
    "engine", "detection", "parsing", "geometry", "frequency", "skin",
    "grading", "eyes", "lips", "teeth", "blemish", "film", "params",
    "spot_heal_auto", "eye_visibility", "undereye", "duplicates",
    "watermark", "xmp_sidecar", "edit_report", "content_credentials",
]

# Doc commands that must use the repo-pinned environment, not global python3.
PINNED_CMD_PATTERNS = [
    (re.compile(r"(?<!\.venv/bin/)python3 -m pytest"), "use .venv/bin/python -m pytest"),
]
# Only checked in onboarding/governance docs (agents read these first).
COMMAND_CHECK_FILES = ["CLAUDE.md", "docs/CONTRIBUTING.md", "README.md",
                       "docs/guides/GETTING_STARTED.md"]

# CLAUDE.md freshness: ERROR if "Last Updated" is older than this many days.
CLAUDE_STALE_DAYS = 14

MD_LINK_RE = re.compile(r"\]\(([^)#\s]+?)(?:#[^)]*)?\)")
CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")


def _strip_code(txt: str) -> str:
    """Remove fenced blocks and inline code so math/code is not parsed as links."""
    return INLINE_CODE_RE.sub("", CODE_FENCE_RE.sub("", txt))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sh(*args: str) -> str:
    try:
        return subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:
        return ""


def repo_python() -> str:
    p = ROOT / ".venv" / "bin" / "python"
    return str(p) if p.is_file() else "python3"


def iter_md_files() -> list[Path]:
    files = [ROOT / "CLAUDE.md", ROOT / "README.md"]
    files += sorted(ROOT.glob("docs/**/*.md"))
    if (ROOT / "AGENTS.md").is_file():
        files.append(ROOT / "AGENTS.md")
    return [f for f in files if f.is_file()]


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT))


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.infos: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def info(self, msg: str) -> None:
        self.infos.append(msg)


# ---------------------------------------------------------------------------
# 1. Repo stats (generated, never hand-maintained)
# ---------------------------------------------------------------------------

def collect_stats(run_tests: bool = True) -> dict[str, str]:
    modules = sorted(ROOT.glob("retouch/*.py"))
    loc_files = list(ROOT.glob("retouch/*.py"))
    loc_files += sorted(ROOT.glob("gui*.py")) + [ROOT / "cli.py"]
    loc = 0
    for f in loc_files:
        if f.is_file():
            loc += sum(1 for _ in f.open("rb"))

    tests = "not collected"
    if run_tests:
        py = repo_python()
        out = subprocess.run(
            [py, "-m", "pytest", "tests/", "--collect-only", "-q"],
            cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "RETOUCH_GPU": "0",
                 "RETOUCH_MEDIAPIPE_BACKEND": "legacy"},
            timeout=600).stdout
        m = re.search(r"(\d+) tests? collected", out)
        tests = m.group(1) if m else "collection failed"
    return {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "commit": sh("git", "rev-parse", "--short", "HEAD") or "unknown",
        "commit_date": sh("git", "log", "-1", "--format=%ad", "--date=short") or "unknown",
        "modules_retouch": str(len(modules)),
        "loc_source": f"{loc:,}",
        "tests_collected": tests,
    }


def write_stats_file(rep: Report) -> None:
    s = collect_stats(run_tests=True)
    body = f"""# Repository stats (generated)

<!-- GENERATED FILE — do not hand-edit. Run: scripts/dev/governance --write-stats -->

| Metric | Value |
|---|---|
| Generated | {s['generated_utc']} |
| Commit | {s['commit']} ({s['commit_date']}) |
| Modules in `retouch/` | {s['modules_retouch']} |
| Source LOC (`retouch/*.py` + `gui*.py` + `cli.py`) | {s['loc_source']} |
| Tests collected | {s['tests_collected']} |

These numbers are the single source of truth for repo size. Human-edited docs
(CLAUDE.md, README.md) must not restate them; link here instead.
"""
    STATS_FILE.write_text(body)
    rep.info(f"wrote {rel(STATS_FILE)} "
             f"({s['modules_retouch']} modules, {s['tests_collected']} tests, {s['loc_source']} LOC)")


# ---------------------------------------------------------------------------
# 2. Doc-link validation
# ---------------------------------------------------------------------------

def check_links(rep: Report) -> None:
    broken = 0
    for f in iter_md_files():
        try:
            txt = _strip_code(f.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
        for m in MD_LINK_RE.finditer(txt):
            target = m.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "file:")):
                continue
            # Check .md links and relative file links, skip pure anchors.
            if target.startswith("#"):
                continue
            p = (f.parent / target).resolve()
            # Research notes cite run outputs in the git-ignored test_output/
            # folder and source photos elsewhere on the author's machine
            # (absolute paths). Both exist only where the study ran, so a
            # clean checkout (CI) can never resolve them; only links to files
            # the repo itself carries are checked.
            if ROOT not in p.parents or p == ARTIFACT_DIR or ARTIFACT_DIR in p.parents:
                continue
            if not p.exists():
                broken += 1
                rep.error(f"broken link: {rel(f)} -> {target}")
    if broken == 0:
        rep.info("doc links: all internal links resolve")


# ---------------------------------------------------------------------------
# 3. Documentation index coverage
# ---------------------------------------------------------------------------

def _excluded(path: Path) -> bool:
    r = rel(path)
    if path.name in INDEX_EXCLUDE_BASENAMES or r in INDEX_EXCLUDE_PATHS:
        return True
    return any(path.match(g) for g in INDEX_EXCLUDE_GLOBS)


def check_index_coverage(rep: Report) -> None:
    index_txt = INDEX.read_text(encoding="utf-8", errors="replace") if INDEX.is_file() else ""
    missing = []
    for f in iter_md_files():
        if not f.name.startswith(("PLAN_", "RESEARCH_", "TASK_", "VISION_")) \
           and f.parent == ROOT / "docs" / "plans":
            pass  # plans handled by basename rule below
        if _excluded(f) or f.name in ("CLAUDE.md", "README.md", "AGENTS.md"):
            continue
        # Session/history dirs are covered by INDEX wildcard lines; skip if the
        # INDEX mentions their parent directory or a wildcard of their name.
        if f.name in index_txt:
            continue
        stem = f.stem.rsplit("_20", 1)[0]  # dated suffix wildcard: TEST_REPORT_*
        if re.search(rf"{re.escape(stem)}_\*", index_txt):
            continue
        if f.parent.name + "/" in index_txt:  # dir-level mention: "session/"
            continue
        missing.append(rel(f))
    for m in missing:
        rep.warn(f"doc not indexed in docs/INDEX.md: {m}")
    if not missing:
        rep.info("index coverage: every non-excluded doc is indexed")


# ---------------------------------------------------------------------------
# 4. Architecture drift detection
# ---------------------------------------------------------------------------

def check_arch_drift(rep: Report) -> None:
    canon_txt = ""
    for c in CANONICAL_ARCH_DOCS:
        p = ROOT / c
        if not p.is_file():
            rep.error(f"canonical architecture doc missing: {c}")
            continue
        canon_txt += p.read_text(encoding="utf-8", errors="replace") + "\n"

    modules = sorted(ROOT.glob("retouch/*.py"))
    missing_required, missing_other = [], []
    for mod in modules:
        name = mod.stem
        if name.startswith("__"):
            continue
        if name in canon_txt or f"{name}.py" in canon_txt:
            continue
        if any(alias in canon_txt for alias in ARCH_ALIASES.get(name, ())):
            continue
        if name in ARCH_REQUIRED:
            missing_required.append(name)
        else:
            missing_other.append(name)

    for n in missing_required:
        rep.error(f"architecture drift: required module retouch/{n}.py "
                  f"absent from {', '.join(CANONICAL_ARCH_DOCS)}")
    for n in missing_other:
        rep.warn(f"architecture drift: retouch/{n}.py not represented in "
                 f"canonical architecture docs")
    if not missing_required and not missing_other:
        rep.info("architecture drift: every retouch module is represented")


# ---------------------------------------------------------------------------
# 5. Stale metadata
# ---------------------------------------------------------------------------

def check_stale_metadata(rep: Report) -> None:
    claude = ROOT / "CLAUDE.md"
    if not claude.is_file():
        return
    txt = claude.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"\*\*Last Updated:\*\*\s*(\d{4}-\d{2}-\d{2})", txt)
    if m:
        try:
            d = datetime.strptime(m.group(1), "%Y-%m-%d")
            age = (datetime.now(timezone.utc).replace(tzinfo=None) - d).days
            if age > CLAUDE_STALE_DAYS:
                rep.warn(f"CLAUDE.md 'Last Updated' is {age} days old "
                         f"(>{CLAUDE_STALE_DAYS})")
            else:
                rep.info(f"CLAUDE.md freshness ok ({age} days old)")
        except ValueError:
            rep.warn("CLAUDE.md 'Last Updated' date unparsable")


# ---------------------------------------------------------------------------
# 6. Repo-pinned command validation
# ---------------------------------------------------------------------------

def check_commands(rep: Report) -> None:
    bad = 0
    for c in COMMAND_CHECK_FILES:
        p = ROOT / c
        if not p.is_file():
            rep.warn(f"command-checked doc missing: {c}")
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            for pat, fix in PINNED_CMD_PATTERNS:
                if pat.search(line):
                    bad += 1
                    rep.error(f"global-python command in {c}:{i}: "
                              f"{line.strip()!r} — {fix}")
    if bad == 0:
        rep.info("commands: onboarding docs use the repo-pinned environment")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_report(rep: Report, title: str) -> None:
    print(f"=== governance: {title} ===")
    for lvl, items in (("ERROR", rep.errors), ("WARNING", rep.warnings),
                       ("INFO", rep.infos)):
        for it in items:
            print(f"[{lvl}] {it}")
    print(f"--- {len(rep.errors)} errors, {len(rep.warnings)} warnings, "
          f"{len(rep.infos)} info ---")


def main(argv: list[str]) -> int:
    ci = "--ci" in argv
    strict = "--strict" in argv
    stats_only = "--stats" in argv
    write_stats = "--write-stats" in argv

    if not (ROOT / "retouch").is_dir():
        print("governance: run from the repo root (retouch/ not found)", file=sys.stderr)
        return 2

    rep = Report()

    if write_stats:
        write_stats_file(rep)
        print_report(rep, "stats")
        return 0

    if stats_only:
        for k, v in collect_stats(run_tests=True).items():
            print(f"{k}: {v}")
        return 0

    t0 = time.time()

    # Stats: always cheap; test collection only outside --ci.
    s = collect_stats(run_tests=not ci)
    tests_note = "tests not collected (--ci)" if ci else f"{s['tests_collected']} tests"
    rep.info(f"stats: {s['modules_retouch']} modules, {s['loc_source']} LOC, "
             f"{tests_note}, commit {s['commit']}")

    check_links(rep)
    check_index_coverage(rep)
    check_arch_drift(rep)
    check_stale_metadata(rep)
    check_commands(rep)

    print_report(rep, "ci (errors only)" if ci else "full audit")
    print(f"took {time.time() - t0:.1f}s")

    if rep.errors:
        return 1
    if strict and rep.warnings:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
