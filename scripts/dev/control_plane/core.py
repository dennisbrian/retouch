#!/usr/bin/env python3
"""AI Engineering Control Plane core: task queue, dependencies, leases,
duplicate detection, parallel planning, scope drift, circuit breaker.

Stdlib only. State lives in control-plane/ (tasks as one JSON file each,
health in control-plane/health.json). Git + GitHub (`gh`) queried read-only.
Policy tables at the top — edit tables, not logic.

See docs/ENGINEERING_CONTROL_PLANE.md for the full contract.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QUEUE_DIR = ROOT / "control-plane" / "tasks"
HEALTH_FILE = ROOT / "control-plane" / "health.json"

# ---------------------------------------------------------------------------
# Policy tables (edit here, not in logic)
# ---------------------------------------------------------------------------

TASK_TYPES = (
    "feature", "bugfix", "research", "refactor", "performance",
    "docs", "test", "maintenance", "hotfix",
)

STATES = (
    "BACKLOG", "READY", "CLAIMED", "IN_PROGRESS", "VALIDATING",
    "PR_OPEN", "BLOCKED", "MERGED", "FAILED", "CANCELLED",
)

# Allowed forward transitions. MERGED/FAILED/CANCELLED are terminal.
TRANSITIONS = {
    "BACKLOG": {"READY", "CANCELLED"},
    "READY": {"CLAIMED", "BLOCKED", "CANCELLED"},
    "CLAIMED": {"IN_PROGRESS", "VALIDATING", "READY", "CANCELLED"},
    "IN_PROGRESS": {"VALIDATING", "BLOCKED", "FAILED", "READY", "CANCELLED"},
    "VALIDATING": {"PR_OPEN", "FAILED", "IN_PROGRESS", "CANCELLED", "MERGED"},
    "PR_OPEN": {"MERGED", "FAILED", "IN_PROGRESS", "CANCELLED"},
    "BLOCKED": {"READY", "CANCELLED"},
    "MERGED": set(),
    "FAILED": set(),
    "CANCELLED": set(),
}

PRIORITIES = ("P0", "P1", "P2", "P3", "P4")
PRIORITY_LABEL = {
    "P0": "HOTFIX", "P1": "IMPORTANT", "P2": "NORMAL",
    "P3": "MAINTENANCE", "P4": "RESEARCH",
}
_PRIORITY_RANK = {p: i for i, p in enumerate(PRIORITIES)}

BUDGETS = ("SMALL", "MEDIUM", "LARGE", "RESEARCH")
# Code-churn ceiling per budget class (diff lines, staged+unstaged).
BUDGET_CHURN_CEILING = {"SMALL": 150, "MEDIUM": 600, "LARGE": 2500}

LEASE_MODES = ("SHARED", "CAUTION", "EXCLUSIVE")
LEASE_HOURS = 72.0  # claim lease lifetime; refreshed by heartbeat

# Subsystem -> (lease mode, path prefixes). Path-prefix mapping is for
# auto-inferring subsystems from files_hint and for scope-drift checks.
SUBSYSTEMS = {
    "pipeline": ("EXCLUSIVE", ("retouch/engine.py", "retouch/perf_optimizations.py")),
    "params": ("EXCLUSIVE", ("retouch/params.py", "retouch/recipe_schema.py")),
    "face-analysis": ("EXCLUSIVE", ("retouch/detection.py", "retouch/parsing.py",
                                    "retouch/eye_visibility.py", "retouch/geometry.py")),
    "skin-retouch": ("CAUTION", ("retouch/skin.py", "retouch/frequency.py",
                                 "retouch/blemish.py", "retouch/spot_heal_auto.py",
                                 "retouch/undereye.py")),
    "recipes": ("CAUTION", ("retouch/recipes",)),
    "grading": ("CAUTION", ("retouch/grading.py", "retouch/film.py", "retouch/luts")),
    "gui": ("CAUTION", ("gui.py",)),
    "cli": ("CAUTION", ("cli.py",)),
    "export": ("CAUTION", ("retouch/io.py", "retouch/xmp_sidecar.py",
                           "retouch/edit_report.py", "retouch/content_credentials.py")),
    "metadata": ("SHARED", ("retouch/xmp_sidecar.py",)),
    "tests": ("SHARED", ("tests/",)),
    "docs": ("SHARED", ("docs/",)),
    "governance": ("EXCLUSIVE", ("scripts/dev/governance.py", "scripts/dev/pr_governor.py",
                                 "scripts/dev/control_plane/", ".github/")),
}
_DEFAULT_LEASE_MODE = "CAUTION"

# Housekeeping lanes: fraction of READY slots the planner reserves when
# possible (soft; feature work never starves if nothing else exists).
HOUSEKEEPING_TYPES = {"docs", "test", "maintenance", "refactor", "performance"}
HOUSEKEEPING_SHARE = 0.30

# Circuit-breaker thresholds (conservative).
CB_MAX_FAILED_MERGES = 3        # FAILED tasks in window
CB_MAX_OPEN_PRS = 12            # open PRs at once
CB_MAX_ACTIVE_PER_SUBSYSTEM = 2 # CAUTION/EXCLUSIVE active tasks per subsystem
CB_WINDOW_HOURS = 72.0

# Task-id prefix; ids are TASK-<zero-padded counter>.
ID_RE = re.compile(r"^TASK-(\d{4,})$")

# Duplicate-detection stopwords + minimum evidence. Weak textual similarity
# alone never blocks: verdict needs >= 2 token hits or an exact key-phrase
# collision, plus subsystem agreement.
_STOPWORDS = {
    "add", "the", "a", "an", "for", "to", "of", "and", "or", "in", "on",
    "with", "support", "stage", "auto", "automatic", "new", "improve",
    "improved", "fix", "update", "removal", "cleanup", "detect", "detection",
    "implement", "implementation", "via", "use", "using",
}
_DUP_MIN_TOKEN_HITS = 2
# One shared token is enough when it is long/distinctive ("watermark").
_DUP_DISTINCTIVE_LEN = 7
# Synonym groups: tokens in the same group count as one shared token.
_SYNONYMS = [
    {"visibility", "visible", "occlusion", "occluded", "closed", "blink"},
    {"remove", "removal", "cleanup", "delete", "strip"},
    {"slimming", "reshape", "reshaping", "liquify", "warp"},
    {"metric", "score", "scoring", "measure"},
    {"wavelength", "kelvin", "whitebalance", "wb"},
]


def _expand_synonyms(tokens: set) -> set:
    expanded = set(tokens)
    for group in _SYNONYMS:
        if tokens & group:
            expanded |= group
    return expanded


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_ts(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0.0


def _atomic_write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    return proc.stdout.strip()


def _gh_json(*args: str) -> list | dict:
    """Query GitHub read-only. Returns [] on any failure (gh optional)."""
    try:
        proc = subprocess.run(
            ["gh", *args], cwd=ROOT, capture_output=True, text=True, timeout=30,
        )
        if proc.returncode != 0:
            return []
        return json.loads(proc.stdout or "[]")
    except (OSError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return []


def _tokenize(text: str) -> set[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    tokens = re.findall(r"[a-z0-9]+", text)
    return {t for t in tokens if len(t) > 2 and t not in _STOPWORDS}


# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

@dataclass
class Task:
    id: str
    title: str
    type: str = "feature"
    priority: str = "P2"
    status: str = "BACKLOG"
    subsystem: str = ""
    risk_hint: str = "MEDIUM"
    budget: str = "MEDIUM"
    dependencies: list = field(default_factory=list)
    blocked_by: list = field(default_factory=list)
    conflicts_with: list = field(default_factory=list)
    files_hint: list = field(default_factory=list)
    modules_hint: list = field(default_factory=list)
    acceptance_criteria: list = field(default_factory=list)
    required_tests: list = field(default_factory=list)
    required_docs: list = field(default_factory=list)
    assigned_agent: str = ""
    branch: str = ""
    pr: str = ""
    intent_manifest: dict = field(default_factory=dict)
    heartbeat: str = ""
    base_commit: str = ""
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""

    @property
    def path(self) -> Path:
        return QUEUE_DIR / f"{self.id}.json"

    def save(self) -> None:
        QUEUE_DIR.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(self.path, asdict(self))

    @classmethod
    def load(cls, path: Path) -> "Task":
        data = json.loads(path.read_text())
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


def load_all() -> list[Task]:
    if not QUEUE_DIR.is_dir():
        return []
    tasks = []
    for path in sorted(QUEUE_DIR.glob("TASK-*.json")):
        try:
            tasks.append(Task.load(path))
        except (OSError, json.JSONDecodeError, TypeError) as exc:
            print(f"warning: skipping unreadable task {path.name}: {exc}",
                  file=sys.stderr)
    return tasks


def find_task(tasks: list[Task], task_id: str) -> Task:
    task_id = task_id.upper()
    if not task_id.startswith("TASK-"):
        task_id = f"TASK-{task_id}"
    for task in tasks:
        if task.id == task_id:
            return task
    raise KeyError(task_id)


def next_id(tasks: list[Task]) -> str:
    highest = 0
    for task in tasks:
        match = ID_RE.match(task.id)
        if match:
            highest = max(highest, int(match.group(1)))
    return f"TASK-{highest + 1:04d}"


# ---------------------------------------------------------------------------
# Subsystems & leases
# ---------------------------------------------------------------------------

def infer_subsystems(task: Task) -> set[str]:
    """Subsystems from explicit field plus files_hint/modules_hint mapping."""
    found = set()
    if task.subsystem:
        found.add(task.subsystem)
    hints = list(task.files_hint) + list(task.modules_hint)
    for hint in hints:
        hint = hint.strip().rstrip("/")
        if hint in SUBSYSTEMS:
            found.add(hint)
            continue
        for name, (_mode, prefixes) in SUBSYSTEMS.items():
            if any(hint == p or hint.startswith(p) or p.startswith(hint + "/")
                   for p in prefixes):
                found.add(name)
    return found


def lease_mode_for(subsystem: str) -> str:
    entry = SUBSYSTEMS.get(subsystem)
    return entry[0] if entry else _DEFAULT_LEASE_MODE


def active_leases(tasks: list[Task]) -> dict:
    """subsystem -> {task_id, agent, mode, expires} for live claimed work."""
    leases = {}
    now = time.time()
    for task in tasks:
        if task.status not in ("CLAIMED", "IN_PROGRESS", "VALIDATING", "PR_OPEN"):
            continue
        heartbeat = _parse_ts(task.heartbeat) or _parse_ts(task.started_at)
        expires = heartbeat + LEASE_HOURS * 3600
        if expires < now:
            continue  # abandoned; lease lapsed
        for subsystem in infer_subsystems(task):
            current = leases.get(subsystem)
            mode = lease_mode_for(subsystem)
            # strongest claim wins for reporting
            if current is None or LEASE_MODES.index(mode) > LEASE_MODES.index(current["mode"]):
                leases[subsystem] = {
                    "task_id": task.id, "agent": task.assigned_agent,
                    "mode": mode, "expires": datetime.fromtimestamp(
                        expires, timezone.utc).isoformat(timespec="seconds"),
                }
    return leases


def lease_conflicts(task: Task, tasks: list[Task]) -> list[str]:
    """Warnings if this task's subsystems are already actively leased."""
    warnings = []
    leases = active_leases(tasks)
    for subsystem in sorted(infer_subsystems(task)):
        holder = leases.get(subsystem)
        if not holder or holder["task_id"] == task.id:
            continue
        mode = lease_mode_for(subsystem)
        if mode == "SHARED":
            continue
        verb = "blocked by EXCLUSIVE lease" if mode == "EXCLUSIVE" else \
               "caution: CAUTION lease active"
        warnings.append(
            f"{subsystem}: {verb} held by {holder['task_id']} "
            f"({holder['agent']}, expires {holder['expires']})")
    return warnings


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def dependency_check(task: Task, tasks: list[Task]) -> list[str]:
    """Unfinished prerequisite task ids (dependencies + blocked_by)."""
    by_id = {t.id: t for t in tasks}
    unfinished = []
    for dep_id in list(task.dependencies) + list(task.blocked_by):
        dep = by_id.get(dep_id)
        if dep is None:
            unfinished.append(f"{dep_id} (missing)")
        elif dep.status not in ("MERGED",):
            unfinished.append(f"{dep_id} ({dep.status})")
    return unfinished


def find_cycles(tasks: list[Task]) -> list[list[str]]:
    """All dependency cycles (DFS), each as a list of task ids."""
    edges = {t.id: list(t.dependencies) + list(t.blocked_by) for t in tasks}
    cycles, stack, state = [], [], {}

    def visit(node: str) -> None:
        state[node] = 1
        stack.append(node)
        for nxt in edges.get(node, []):
            if nxt not in edges:
                continue
            if state.get(nxt) == 1:
                cycles.append(stack[stack.index(nxt):] + [nxt])
            elif state.get(nxt) != 2:
                visit(nxt)
        stack.pop()
        state[node] = 2

    for task_id in edges:
        if state.get(task_id) != 2:
            visit(task_id)
    return cycles


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

def duplicate_verdict(title: str, subsystem: str, tasks: list[Task],
                      scan_repo: bool = True) -> dict:
    """DUPLICATE / POSSIBLE_OVERLAP / UNIQUE against queue + repo hints."""
    tokens = _expand_synonyms(_tokenize(title))
    matches = []
    live_states = {"BACKLOG", "READY", "CLAIMED", "IN_PROGRESS",
                   "VALIDATING", "PR_OPEN", "BLOCKED"}
    for other in tasks:
        other_tokens = _expand_synonyms(_tokenize(other.title))
        hits = tokens & other_tokens
        distinctive = any(len(h) >= _DUP_DISTINCTIVE_LEN for h in hits)
        if len(hits) < _DUP_MIN_TOKEN_HITS and not distinctive:
            continue
        same_sub = (not subsystem or not other.subsystem
                    or subsystem == other.subsystem)
        if not same_sub:
            continue
        weight = len(hits) / max(1, min(len(tokens), len(other_tokens)))
        matches.append({
            "id": other.id, "title": other.title, "status": other.status,
            "shared": sorted(hits), "score": round(weight, 2),
        })
    # Repo surfaces: branch names + open PR titles (read-only, best effort).
    if scan_repo:
        for branch in _git("branch", "-a", "--format=%(refname:short)").splitlines():
            branch = branch.strip()
            if not branch:
                continue
            hits = tokens & _tokenize(branch.replace("/", " ").replace("-", " "))
            if len(hits) >= _DUP_MIN_TOKEN_HITS:
                matches.append({"branch": branch, "shared": sorted(hits),
                                "score": 0.5})
        for pr in _gh_json("pr", "list", "--state", "open",
                           "--json", "number,title", "--limit", "50"):
            hits = tokens & _tokenize(str(pr.get("title", "")))
            if len(hits) >= _DUP_MIN_TOKEN_HITS:
                matches.append({"pr": pr.get("number"), "title": pr.get("title"),
                                "shared": sorted(hits), "score": 0.6})
    live = [m for m in matches if m.get("status") in live_states or "branch" in m
            or "pr" in m]
    if any(m.get("score", 0) >= 0.8 and m.get("status") in live_states
           for m in live):
        verdict = "DUPLICATE"
    elif live:
        verdict = "POSSIBLE_OVERLAP"
    else:
        verdict = "UNIQUE"
    return {"verdict": verdict, "matches": matches}


# ---------------------------------------------------------------------------
# Intake & decomposition
# ---------------------------------------------------------------------------

_TYPE_KEYWORDS = {
    "hotfix": ("hotfix", "urgent", "production down"),
    "bugfix": ("fix", "bug", "broken", "regression", "crash", "hang", "inert",
               "no-op", "wrong"),
    "research": ("research", "study", "investigate", "calibrat", "evaluate",
                 "survey", "compare"),
    "refactor": ("refactor", "rename", "restructure", "simplif", "extract",
                 "dedup", "dead code", "dead-code"),
    "performance": ("perf", "speed", "faster", "slow", "memory", "oom",
                    "optimize", "optimise"),
    "docs": ("doc", "documentation", "readme", "guide", "changelog"),
    "test": ("test", "coverage", "golden", "fixture", "mutation"),
    "maintenance": ("bump", "dependency", "cleanup", "housekeep", "stale",
                    "deprecat", "remove"),
}

_SUBSYSTEM_KEYWORDS = {
    "face-analysis": ("eye", "eyes", "face", "detection", "landmark", "bisenet",
                      "segment", "occlu", "blink", "yaw", "parsing"),
    "skin-retouch": ("skin", "blemish", "spot", "pimple", "freckle", "undereye",
                     "dark circle", "wrinkle", "pore", "smooth", "texture"),
    "grading": ("color", "colour", "grade", "grading", "lut", "film", "fuji",
                "white balance", "tone"),
    "recipes": ("recipe", "preset"),
    "pipeline": ("pipeline", "stage order", "engine", "orchestrat", "proxy"),
    "gui": ("gui", "gradio", "slider", "ui"),
    "cli": ("cli", "command line", "batch", "flag"),
    "export": ("export", "xmp", "sidecar", "jpeg", "png", "c2pa", "watermark",
               "metadata", "credential"),
    "docs": ("doc", "documentation", "guide", "index"),
    "governance": ("governance", "control plane", "pr governor", "ci",
                   "workflow"),
}


def _guess_type(title: str) -> str:
    lowered = title.lower()
    for task_type, keywords in _TYPE_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            return task_type
    return "feature"


def _guess_subsystem(title: str) -> str:
    lowered = title.lower()
    best, best_hits = "", 0
    for subsystem, keywords in _SUBSYSTEM_KEYWORDS.items():
        hits = sum(1 for k in keywords if k in lowered)
        if hits > best_hits:
            best, best_hits = subsystem, hits
    # "GUI wiring"/"CLI wiring" are surfaces, not the subject: if a content
    # subsystem tied with a bare surface keyword, the content subsystem wins.
    if best in ("gui", "cli"):
        for subsystem, keywords in _SUBSYSTEM_KEYWORDS.items():
            if subsystem in ("gui", "cli"):
                continue
            if sum(1 for k in keywords if k in lowered) >= best_hits:
                return subsystem
    return best


def intake(title: str, tasks: list[Task], task_type: str = "",
           subsystem: str = "", priority: str = "P2",
           acceptance: list | None = None, scan_repo: bool = True) -> tuple:
    """Create a BACKLOG task from a free-text idea. Returns (task, report)."""
    task_type = task_type or _guess_type(title)
    subsystem = subsystem or _guess_subsystem(title)
    if task_type not in TASK_TYPES:
        raise ValueError(f"unknown type {task_type!r}")
    if priority not in PRIORITIES:
        raise ValueError(f"unknown priority {priority!r}")
    criteria = list(acceptance or [])
    if not criteria:
        criteria = ["behavior implemented as specified",
                    "regression tests added under tests/",
                    "docs/INDEX.md updated if a new doc is added"]
    task = Task(
        id=next_id(tasks), title=title, type=task_type, priority=priority,
        status="BACKLOG", subsystem=subsystem,
        risk_hint="HIGH" if subsystem in ("pipeline", "params", "governance")
                  else "MEDIUM",
        budget="RESEARCH" if task_type == "research" else "MEDIUM",
        acceptance_criteria=criteria,
        created_at=_now(),
    )
    dup = duplicate_verdict(title, subsystem, tasks, scan_repo=scan_repo)
    return task, {"duplicate": dup}


def decompose(task: Task) -> list[str]:
    """Suggested subtask titles (heuristic; human/agent edits, not auto-spawned
    unless the work spans multiple subsystems)."""
    spans = infer_subsystems(task)
    title_wide = any(k in task.title.lower()
                     for k in (" and ", "gui", "cli", "export", "sweep"))
    if len(spans) < 2 and not (title_wide and task.type in ("feature", "refactor")):
        return []  # one-file/one-subsystem work stays one task
    steps = []
    if task.type != "research":
        steps.append(f"research / confirm current behavior for: {task.title}")
    steps.append(f"core implementation: {task.title}")
    if "gui" in spans:
        steps.append(f"GUI wiring: {task.title}")
    if "cli" in spans:
        steps.append(f"CLI wiring: {task.title}")
    steps.append(f"tests: {task.title}")
    steps.append(f"docs: {task.title}")
    return steps


# ---------------------------------------------------------------------------
# Readiness, claiming, pre-flight
# ---------------------------------------------------------------------------

def ready_tasks(tasks: list[Task]) -> list[Task]:
    out = []
    for task in tasks:
        if task.status not in ("BACKLOG", "READY", "BLOCKED"):
            continue
        if dependency_check(task, tasks):
            continue
        out.append(task)
    rank = lambda t: (_PRIORITY_RANK.get(t.priority, 9), t.created_at)
    return sorted(out, key=rank)


def preflight(task: Task, tasks: list[Task]) -> dict:
    """Full pre-start check: READY TO START / BLOCKED / CAUTION."""
    blockers, cautions = [], []
    cycles = find_cycles(tasks)
    in_cycle = [c for c in cycles if task.id in c]
    if in_cycle:
        blockers.append(f"circular dependency: {' -> '.join(in_cycle[0])}")
    for dep in dependency_check(task, tasks):
        blockers.append(f"waiting for {dep}")
    for warning in lease_conflicts(task, tasks):
        (blockers if "EXCLUSIVE" in warning else cautions).append(warning)
    dup = duplicate_verdict(task.title, task.subsystem,
                            [t for t in tasks if t.id != task.id])
    if dup["verdict"] == "DUPLICATE":
        blockers.append(f"duplicate of {dup['matches'][0].get('id', dup['matches'][0])}")
    elif dup["verdict"] == "POSSIBLE_OVERLAP":
        cautions.append(f"possible overlap: {dup['matches'][0]}")
    # Open-PR overlap on files_hint (best effort).
    if task.files_hint:
        for pr in _gh_json("pr", "list", "--state", "open",
                           "--json", "number,title,files", "--limit", "50"):
            pr_files = {f.get("path", "") for f in pr.get("files", [])}
            overlap = sorted(pr_files & set(task.files_hint))
            if overlap:
                cautions.append(
                    f"PR #{pr.get('number')} ({pr.get('title')}) touches "
                    f"{', '.join(overlap[:3])}")
    base = _git("rev-parse", "HEAD")
    behind = _git("rev-list", "--count", f"HEAD..origin/main") if _git(
        "rev-parse", "--verify", "origin/main") else ""
    if behind and behind != "0":
        cautions.append(f"base branch is {behind} commits behind origin/main")
    health = load_health()
    if health_paused(health) and task.priority != "P0" and \
            task.type not in ("hotfix", "test", "maintenance"):
        blockers.append("circuit breaker active: new feature work paused")
    verdict = "BLOCKED" if blockers else ("CAUTION" if cautions
                                          else "READY TO START")
    return {"verdict": verdict, "blockers": blockers, "cautions": cautions,
            "base_commit": base,
            "recommended_tests": recommended_tests(task)}


def recommended_tests(task: Task) -> list[str]:
    tests = list(task.required_tests)
    for subsystem in infer_subsystems(task):
        if subsystem == "tests":
            continue
        tests.append(f"tests/test_{subsystem.replace('-', '_')}*.py (or nearest mirror)")
    if not tests:
        tests.append("tests/ mirror of the touched module")
    return tests


def claim(task: Task, tasks: list[Task], agent: str, branch: str = "",
          force: bool = False) -> dict:
    check = preflight(task, tasks)
    if check["verdict"] == "BLOCKED" and not force:
        return check
    if task.status not in ("READY", "BACKLOG", "BLOCKED"):
        raise ValueError(f"cannot claim from {task.status}")
    task.status = "CLAIMED"
    task.assigned_agent = agent
    task.branch = branch or f"agent/{task.id.lower()}-{task.subsystem or 'misc'}"
    task.base_commit = check["base_commit"]
    task.started_at = _now()
    task.heartbeat = task.started_at
    task.intent_manifest = build_intent_manifest(task)
    task.save()
    check["claimed"] = True
    check["branch"] = task.branch
    check["leases"] = sorted(infer_subsystems(task))
    check["intent_manifest"] = task.intent_manifest
    return check


def build_intent_manifest(task: Task) -> dict:
    """Planned-change contract used later for scope-drift detection."""
    must_not = []
    subsystems = infer_subsystems(task)
    for protected in ("pipeline", "params", "recipes"):
        if protected not in subsystems:
            mode, _ = SUBSYSTEMS[protected]
            if mode == "EXCLUSIVE":
                must_not.append(protected)
    return {
        "expected_subsystems": sorted(subsystems),
        "expected_files": list(task.files_hint),
        "must_not_change": must_not,
        "expected_tests": recommended_tests(task),
        "budget": task.budget,
    }


# ---------------------------------------------------------------------------
# Scope drift
# ---------------------------------------------------------------------------

def scope_drift(task: Task, changed_files: list[str] | None = None) -> dict:
    """Compare intent manifest vs actual working-tree/branch changes."""
    if changed_files is None:
        base = task.base_commit or "HEAD"
        diff_args = ["diff", "--name-only", f"{base}...HEAD"] if task.branch \
            else ["diff", "--name-only", "HEAD"]
        changed_files = [f for f in _git(*diff_args).splitlines() if f]
    manifest = task.intent_manifest or build_intent_manifest(task)
    expected = set(manifest.get("expected_subsystems", []))
    actual_subsystems = set()
    for path in changed_files:
        fake = Task(id=task.id, title=task.title, files_hint=[path])
        actual_subsystems |= infer_subsystems(fake)
    must_not = set(manifest.get("must_not_change", []))
    violations = sorted(actual_subsystems & must_not)
    unexpected = sorted(actual_subsystems - expected - {"tests", "docs"})
    churn = sum(1 for _ in changed_files)
    level = "NONE"
    if violations:
        level = "HIGH"
    elif len(unexpected) >= 2 or (unexpected and task.budget == "SMALL"):
        level = "MEDIUM"
    elif unexpected:
        level = "LOW"
    ceiling = BUDGET_CHURN_CEILING.get(task.budget, 0)
    budget_note = ""
    if ceiling and churn > ceiling:
        budget_note = (f"{churn} files changed exceeds {task.budget} budget "
                       f"ceiling ({ceiling}); consider rescoping or decomposition")
        if level in ("NONE", "LOW"):
            level = "MEDIUM"
    return {"level": level, "changed_files": changed_files,
            "expected_subsystems": sorted(expected),
            "actual_subsystems": sorted(actual_subsystems),
            "unexpected": unexpected, "violations": violations,
            "budget_note": budget_note}


# ---------------------------------------------------------------------------
# Parallel planner
# ---------------------------------------------------------------------------

def conflict_risk(a: Task, b: Task) -> str:
    """LOW/MEDIUM/HIGH pair conflict from subsystem overlap + lease modes."""
    if a.id in list(b.conflicts_with) or b.id in list(a.conflicts_with):
        return "HIGH"
    shared = infer_subsystems(a) & infer_subsystems(b)
    if not shared:
        return "LOW"
    modes = {lease_mode_for(s) for s in shared}
    if "EXCLUSIVE" in modes:
        return "HIGH"
    if "CAUTION" in modes:
        return "MEDIUM"
    return "LOW"


def plan(tasks: list[Task], max_batch: int = 5) -> dict:
    """Recommended safe concurrent batch from ready tasks."""
    candidates = ready_tasks(tasks)
    active = [t for t in tasks if t.status in
              ("CLAIMED", "IN_PROGRESS", "VALIDATING", "PR_OPEN")]
    batch, held, housekeeping_slots = [], [], max(1, int(max_batch * HOUSEKEEPING_SHARE))
    for task in candidates:
        if len(batch) >= max_batch:
            held.append({"id": task.id, "reason": "batch full"})
            continue
        worst = "LOW"
        blocker = ""
        for other in batch + active:
            risk = conflict_risk(task, other)
            if risk == "HIGH":
                blocker = other.id
                worst = risk
                break
            if risk == "MEDIUM":
                worst = "MEDIUM"
        if worst == "HIGH":
            held.append({"id": task.id,
                         "reason": f"overlaps {blocker} (EXCLUSIVE subsystem)"})
            continue
        if task.type in HOUSEKEEPING_TYPES and housekeeping_slots <= 0 and \
                len(batch) >= max_batch - 1:
            # soft: don't let housekeeping starve feature slots at the margin
            held.append({"id": task.id, "reason": "housekeeping quota reserved"})
            continue
        if task.type in HOUSEKEEPING_TYPES:
            housekeeping_slots -= 1
        batch.append({"id": task.id, "title": task.title, "type": task.type,
                      "priority": task.priority,
                      "subsystem": task.subsystem, "conflict_risk": worst,
                      "agent_hint": agent_hint(task)})
    blocked = []
    for task in tasks:
        if task.status in ("BACKLOG", "READY", "BLOCKED"):
            deps = dependency_check(task, tasks)
            if deps:
                blocked.append({"id": task.id, "title": task.title,
                                "waiting_for": deps})
    health = load_health()
    return {"batch": batch, "hold": held, "blocked": blocked,
            "paused": health_paused(health),
            "pause_reasons": health.get("pause_reasons", [])}


def agent_hint(task: Task) -> str:
    if task.type == "research":
        return "RESEARCH AGENT (evidence only, no production code)"
    if task.type == "test":
        return "TEST AGENT (adversarial validation; do not rewrite implementation)"
    if task.type == "docs":
        return "DOC AGENT (architecture/index/docs)"
    if task.type in ("bugfix", "hotfix"):
        return "IMPLEMENTATION AGENT (scoped fix) + REVIEW AGENT on merge"
    return "IMPLEMENTATION AGENT (scoped change)"


# ---------------------------------------------------------------------------
# Health / circuit breaker
# ---------------------------------------------------------------------------

def load_health() -> dict:
    if HEALTH_FILE.is_file():
        try:
            return json.loads(HEALTH_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            pass
    return {"manual_pause": False, "pause_reasons": [], "history": []}


def save_health(health: dict) -> None:
    HEALTH_FILE.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(HEALTH_FILE, health)


def health_paused(health: dict) -> bool:
    return bool(health.get("manual_pause") or health.get("pause_reasons"))


def evaluate_health(tasks: list[Task], open_prs: int | None = None) -> dict:
    """Circuit-breaker evaluation. Conservative defaults; P0 always allowed."""
    reasons = []
    now = time.time()
    window = CB_WINDOW_HOURS * 3600
    recent_failed = [t for t in tasks if t.status == "FAILED"
                     and now - _parse_ts(t.completed_at) < window]
    if len(recent_failed) >= CB_MAX_FAILED_MERGES:
        reasons.append(f"{len(recent_failed)} FAILED tasks in "
                       f"{CB_WINDOW_HOURS:.0f}h (>= {CB_MAX_FAILED_MERGES})")
    if open_prs is None:
        prs = _gh_json("pr", "list", "--state", "open", "--json", "number",
                       "--limit", "100")
        open_prs = len(prs) if isinstance(prs, list) else 0
    if open_prs > CB_MAX_OPEN_PRS:
        reasons.append(f"{open_prs} open PRs (> {CB_MAX_OPEN_PRS})")
    pressure = subsystem_pressure(tasks)
    for subsystem, count in pressure.items():
        if count > CB_MAX_ACTIVE_PER_SUBSYSTEM and \
                lease_mode_for(subsystem) != "SHARED":
            reasons.append(f"{subsystem}: {count} active tasks "
                           f"(> {CB_MAX_ACTIVE_PER_SUBSYSTEM})")
    health = load_health()
    health["pause_reasons"] = reasons
    health["evaluated_at"] = _now()
    health["open_prs"] = open_prs
    save_health(health)
    return {"paused": health_paused(health), "reasons": reasons,
            "open_prs": open_prs}


def subsystem_pressure(tasks: list[Task]) -> dict:
    pressure: dict[str, int] = {}
    for task in tasks:
        if task.status not in ("CLAIMED", "IN_PROGRESS", "VALIDATING", "PR_OPEN"):
            continue
        for subsystem in infer_subsystems(task):
            pressure[subsystem] = pressure.get(subsystem, 0) + 1
    return pressure


# ---------------------------------------------------------------------------
# Status & dashboard
# ---------------------------------------------------------------------------

ACTIVE_STATES = ("CLAIMED", "IN_PROGRESS", "VALIDATING", "PR_OPEN")
STALE_HOURS = LEASE_HOURS


def status_report(tasks: list[Task]) -> dict:
    counts = {state: 0 for state in STATES}
    for task in tasks:
        counts[task.status] = counts.get(task.status, 0) + 1
    active = []
    now = time.time()
    stale = []
    for task in tasks:
        if task.status not in ACTIVE_STATES:
            continue
        entry = {
            "id": task.id, "title": task.title, "agent": task.assigned_agent,
            "subsystem": task.subsystem, "branch": task.branch,
            "status": task.status, "priority": task.priority,
            "risk_hint": task.risk_hint, "heartbeat": task.heartbeat,
            "leases": sorted(infer_subsystems(task)),
        }
        last = _parse_ts(task.heartbeat) or _parse_ts(task.started_at)
        if last and now - last > STALE_HOURS * 3600:
            entry["stale"] = True
            stale.append(task.id)
        active.append(entry)
    blocked = [{"id": t.id, "title": t.title, "agent": t.assigned_agent,
                "reason": ", ".join(dependency_check(t, tasks)) or t.status}
               for t in tasks if t.status == "BLOCKED"]
    health = load_health()
    return {
        "counts": counts,
        "active": sorted(active, key=lambda e: _PRIORITY_RANK.get(e["priority"], 9)),
        "blocked": blocked,
        "stale": stale,
        "subsystem_pressure": subsystem_pressure(tasks),
        "leases": active_leases(tasks),
        "health": {"paused": health_paused(health),
                   "pause_reasons": health.get("pause_reasons", [])},
    }


def attention_items(tasks: list[Task]) -> list[str]:
    """What the human owner should look at right now."""
    items = []
    report = status_report(tasks)
    for task_id in report["stale"]:
        items.append(f"{task_id}: no heartbeat for {STALE_HOURS:.0f}h — "
                     f"abandoned or stuck?")
    for cycle in find_cycles(tasks):
        items.append(f"circular dependency: {' -> '.join(cycle)}")
    health = report["health"]
    if health["paused"]:
        items.append(f"circuit breaker: {'; '.join(health['pause_reasons'])}")
    for task in tasks:
        if task.status == "FAILED":
            items.append(f"{task.id} FAILED: {task.title}")
    return items
