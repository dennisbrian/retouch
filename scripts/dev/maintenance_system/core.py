"""Evidence-led repository maintenance. Standard library only; no hidden repairs."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from scripts.dev import governance
from scripts.dev.control_plane import core as cp
from scripts.dev.attention_router import core as attention

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "control-plane" / "maintenance.json"
TEST_RUNS = ROOT / "test_output" / "maintenance" / "test-runs.json"
PERF_REPORTS = ROOT / "test_output" / "quality_lab" / "reports"
SEVERITIES = tuple(f"D{i}" for i in range(6))
STATUSES = ("OPEN", "PLANNED", "IN_PROGRESS", "ACCEPTED", "SUPPRESSED", "RESOLVED", "REGRESSED")
CAMPAIGNS = {"flaky-tests": {"flaky-test", "slow-test"}, "docs": {"doc-link", "doc-index", "doc-freshness"},
             "dependencies": {"dependency"}, "hotspots": {"hotspot", "module-size", "function-size", "architecture"}}
# Conservative defaults. Adjust here, with evidence, rather than enlarging the scanner.
POLICY = {"large_file_lines": 1400, "large_function_lines": 180, "hotspot_changes": 12,
          "hotspot_days": 30, "todo_days": 180, "slow_test_seconds": 60,
          "flaky_failures": 2, "max_tasks_per_scan": 3, "maintenance_share": 0.20,
          "max_maintenance_share": 0.40}


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def load(path=STATE):
    data = read_json(path, {})
    return {"schema": 1, "debt": data.get("debt", []), "scans": data.get("scans", []),
            "policy": {**POLICY, **data.get("policy", {})}}


def save(data, path=STATE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def debt_id(category, subject):
    return "DEBT-" + hashlib.sha256(f"{category}|{subject}".encode()).hexdigest()[:12].upper()


def finding(category, subject, title, *, severity="D1", evidence=None, subsystem="", metric=None,
            action="investigate", effort="SMALL", safe_repair=""):
    return {"id": debt_id(category, subject), "title": title, "category": category,
            "severity": severity, "evidence": evidence or [], "subsystem": subsystem,
            "subject": subject, "metric": metric, "estimated_effort": effort,
            "suggested_action": action, "safe_repair": safe_repair}


def classify(row, occurrences=1, trend="STABLE"):
    """Promote only persistent, consequential evidence; size alone cannot reach D4."""
    level = SEVERITIES.index(row["severity"])
    if row["category"] == "flaky-test" and row.get("metric", 0) >= 5:
        level = max(level, 3)
    if row["category"] == "hotspot" and row.get("regressions", 0) >= 2:
        level = max(level, 3)
    if trend == "WORSENING" and occurrences >= 3 and level >= 2:
        level = min(level + 1, 4)
    return SEVERITIES[level]


def trend_of(values):
    values = [x for x in values if isinstance(x, (int, float))]
    if len(values) < 3:
        return "INSUFFICIENT_DATA"
    tail = values[-3:]
    if tail[0] < tail[1] < tail[2]:
        return "WORSENING"
    if tail[0] > tail[1] > tail[2]:
        return "IMPROVING"
    return "STABLE"


def _run(*argv):
    try:
        p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True, timeout=30)
        return p.stdout if p.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def governance_findings():
    rep = governance.Report()
    governance.check_links(rep)
    governance.check_index_coverage(rep)
    governance.check_arch_drift(rep)
    governance.check_stale_metadata(rep)
    rows = []
    architecture_messages = []
    for message in rep.errors + rep.warnings:
        if message.startswith("broken link: "):
            rows.append(finding("doc-link", message, "Repair internal documentation link", severity="D2",
                                evidence=[message], subsystem="docs", action="verify target and repair link"))
        elif message.startswith("doc not indexed in "):
            path = message.rsplit(": ", 1)[-1]
            rows.append(finding("doc-index", path, f"Index {path}", evidence=[message], subsystem="docs",
                                action="add a link to docs/INDEX.md", safe_repair="index"))
        elif message.startswith("architecture drift: "):
            architecture_messages.append(message)
        elif "Last Updated" in message:
            rows.append(finding("doc-freshness", "CLAUDE.md", "Review development guide freshness",
                                evidence=[message], subsystem="docs", action="verify and refresh guide"))
    if architecture_messages:
        required = sum("required module" in m for m in architecture_messages)
        rows.append(finding("architecture", "canonical-module-coverage", "Review canonical architecture coverage",
                            severity="D3" if required else "D2",
                            evidence=[f"{len(architecture_messages)} modules absent from canonical docs; {required} required"]
                                     + architecture_messages[:10], subsystem="governance",
                            metric=len(architecture_messages), action="review missing modules and update canonical architecture docs",
                            effort="MEDIUM"))
    return rows


def code_findings(policy):
    rows = []
    files = sorted((ROOT / "retouch").glob("*.py")) + sorted((ROOT / "scripts" / "dev").glob("*.py"))
    for path in files:
        rel = str(path.relative_to(ROOT))
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
        except (OSError, SyntaxError, UnicodeError):
            continue
        count = len(source.splitlines())
        if count > policy["large_file_lines"]:
            rows.append(finding("module-size", rel, f"Review responsibilities of {rel}", severity="D2",
                                evidence=[f"{count} lines; threshold {policy['large_file_lines']}"],
                                subsystem=_subsystem(rel), metric=count, action="assess ownership and decomposition",
                                effort="MEDIUM"))
        large_members = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                length = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
                if length > policy["large_function_lines"]:
                    large_members.append((node.name, length))
        if large_members:
            large_members.sort(key=lambda x: -x[1])
            rows.append(finding("function-size", rel, f"Review long members in {rel}",
                                severity="D2" if large_members[0][1] >= 400 or len(large_members) >= 3 else "D1",
                                evidence=[f"{name}: {length} lines" for name, length in large_members[:10]],
                                subsystem=_subsystem(rel), metric=max(x[1] for x in large_members),
                                action="assess responsibility and test seams", effort="MEDIUM"))
        for number, line in enumerate(source.splitlines(), 1):
            match = re.search(r"\b(?:TODO|FIXME|HACK)\b[^\n]*?\b(20\d\d-\d\d-\d\d)\b", line)
            if not match:
                continue
            try:
                age = (date.today() - date.fromisoformat(match.group(1))).days
            except ValueError:
                continue
            if age > policy["todo_days"]:
                rows.append(finding("stale-todo", f"{rel}:{number}", f"Review dated TODO in {rel}",
                                    evidence=[f"{rel}:{number}: age {age} days"], subsystem=_subsystem(rel),
                                    metric=age, action="resolve or renew with an owner"))
    return rows


def _subsystem(path):
    task = cp.Task(id="TASK-0000", title="", files_hint=[path])
    found = cp.infer_subsystems(task)
    return sorted(found)[0] if found else "governance"


def hotspot_findings(policy, history=None):
    """Commit touch count is activity evidence, never a refactor order by itself."""
    if history is None:
        history = _run("git", "log", f"--since={policy['hotspot_days']}.days", "--format=", "--name-only")
    counts = Counter(line.strip() for line in history.splitlines() if line.strip() and not line.startswith("commit "))
    rows = []
    for path, changes in sorted(counts.items()):
        if changes < policy["hotspot_changes"] or not (ROOT / path).is_file():
            continue
        if not path.endswith(".py"):
            continue
        rows.append(finding("hotspot", path, f"Investigate change concentration in {path}", severity="D2",
                            evidence=[f"{changes} commits in {policy['hotspot_days']} days"],
                            subsystem=_subsystem(path), metric=changes,
                            action="measure regression and review cost before decomposition", effort="RESEARCH"))
    return rows


def test_findings(runs=None, policy=None):
    """Consumes repeated per-test runner observations; no inferred flakiness from one failure."""
    policy = policy or POLICY
    runs = read_json(TEST_RUNS, []) if runs is None else runs
    by_test = defaultdict(list)
    for row in runs if isinstance(runs, list) else []:
        if isinstance(row, dict) and row.get("test") and row.get("outcome") in ("passed", "failed", "skipped", "xfailed"):
            by_test[row["test"]].append(row)
    findings = []
    for name, samples in sorted(by_test.items()):
        failures = sum(s["outcome"] == "failed" for s in samples)
        retries = sum(max(0, int(s.get("retries", 0))) for s in samples)
        runtimes = sorted(float(s.get("duration_seconds", 0)) for s in samples if s.get("duration_seconds") is not None)
        if len(samples) >= 3 and failures >= policy["flaky_failures"] and 0 < failures < len(samples):
            findings.append(finding("flaky-test", name, f"Stabilize {name}", severity="D2",
                                    evidence=[f"{failures}/{len(samples)} failed; {retries} retries; last failure " +
                                              next((str(s.get("at", "unknown")) for s in reversed(samples) if s["outcome"] == "failed"), "unknown")],
                                    subsystem="tests", metric=failures, action="reproduce and repair; retain CI failure", effort="MEDIUM"))
        if len(runtimes) >= 3 and runtimes[len(runtimes)//2] > policy["slow_test_seconds"]:
            median = runtimes[len(runtimes)//2]
            findings.append(finding("slow-test", name, f"Investigate slow test {name}", severity="D2",
                                    evidence=[f"median {median:.1f}s across {len(runtimes)} runs"], subsystem="tests",
                                    metric=median, action="profile setup and preserve coverage", effort="SMALL"))
        if len(samples) >= 3 and all(s["outcome"] in ("skipped", "xfailed") for s in samples):
            findings.append(finding("skipped-test", name, f"Revisit skipped test {name}", severity="D1",
                                    evidence=[f"{len(samples)} consecutive skipped/xfailed observations"], subsystem="tests",
                                    metric=len(samples), action="confirm reason and revisit date"))
    return findings


def dependency_findings():
    """Only exact conflicting duplicate declarations; usage/outdated claims require external proof."""
    declarations = defaultdict(list)
    for path in sorted(ROOT.glob("requirements*.txt")):
        for no, line in enumerate(path.read_text().splitlines(), 1):
            clean = line.split("#", 1)[0].strip()
            match = re.match(r"^([A-Za-z][A-Za-z0-9_.-]*)(?:\[[^]]+\])?\s*([<>=!~].*)?$", clean)
            if match:
                declarations[re.sub(r"[-_.]+", "-", match.group(1).lower())].append((str(path.relative_to(ROOT)), no, (match.group(2) or "").strip()))
    rows = []
    for name, entries in sorted(declarations.items()):
        if len({e[2] for e in entries}) > 1:
            rows.append(finding("dependency", name, f"Review conflicting {name} constraints", severity="D2",
                                evidence=[f"{p}:{n} {spec}" for p, n, spec in entries], subsystem="governance",
                                action="compare supported environments before changing pins"))
    return rows


def duplicate_helper_findings():
    """Only exact AST body duplicates across modules; similarity guesses are excluded."""
    groups = defaultdict(list)
    for path in sorted((ROOT / "retouch").glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeError):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            span = getattr(node, "end_lineno", node.lineno) - node.lineno + 1
            if span < 20:
                continue
            body = ast.dump(ast.Module(body=node.body, type_ignores=[]), include_attributes=False)
            digest = hashlib.sha256(body.encode()).hexdigest()[:16]
            groups[digest].append(f"{path.relative_to(ROOT)}:{node.name}")
    rows = []
    for digest, members in sorted(groups.items()):
        if len({m.split(":", 1)[0] for m in members}) < 2:
            continue
        rows.append(finding("duplicate-helper", digest, "Review identical helper bodies across modules",
                            severity="D2", evidence=sorted(members), subsystem="pipeline",
                            metric=len(members), action="compare ownership and semantics before consolidation",
                            effort="RESEARCH"))
    return rows


def quality_perf_findings(reports=None):
    """Read existing Quality Lab benchmark verdicts; do not rerun image workloads."""
    if reports is None:
        reports = sorted(PERF_REPORTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True) if PERF_REPORTS.exists() else []
    latest = next((data for path in reports if (data := read_json(path, {})).get("kind") == "perf"), None)
    if not latest:
        return []
    rows = []
    for entry in latest.get("regressions", []):
        if entry.get("verdict") not in ("REVIEW", "REGRESSION"):
            continue
        subject = f"{entry.get('case', 'unknown')}:{entry.get('recipe', 'unknown')}"
        time_ratio = entry.get("time_s", {}).get("ratio", 1)
        ram_ratio = entry.get("peak_ram_mb", {}).get("ratio", 1)
        rows.append(finding("render-performance", subject, f"Investigate Quality Lab performance for {subject}",
                            severity="D3" if entry["verdict"] == "REGRESSION" else "D2",
                            evidence=[f"Quality Lab {entry['verdict']}: time ratio {time_ratio}, RAM ratio {ram_ratio}"],
                            subsystem="pipeline", metric=max(time_ratio, ram_ratio),
                            action="profile regression against frozen baseline", effort="MEDIUM"))
    return rows


def has_perf_report():
    return PERF_REPORTS.exists() and any(
        read_json(path, {}).get("kind") == "perf" for path in PERF_REPORTS.glob("*.json"))


def collect(policy=None, *, history=None, runs=None):
    policy = policy or POLICY
    groups = (governance_findings(), code_findings(policy), hotspot_findings(policy, history),
              test_findings(runs, policy), dependency_findings(), duplicate_helper_findings(),
              quality_perf_findings())
    return sorted((row for group in groups for row in group), key=lambda r: r["id"])


def reconcile(state, findings, at=None):
    at = at or utc_now()
    today = at[:10]
    prior = {r["id"]: r for r in state["debt"]}
    seen = set()
    for row in findings:
        ident = row["id"]
        seen.add(ident)
        old = prior.get(ident)
        if old is None:
            item = {**row, "first_seen": at, "last_seen": at, "occurrences": 1,
                    "trend": "INSUFFICIENT_DATA", "related_tasks": [], "related_prs": [],
                    "status": "OPEN", "suppress_until": "", "accepted_reason": "",
                    "observations": [{"date": today, "metric": row.get("metric")}], "last_alerted": ""}
            prior[ident] = item
            continue
        old_metric = old.get("metric")
        old_status = old["status"]
        old.update({k: v for k, v in row.items() if k not in ("severity",)})
        old["last_seen"] = at
        observations = old.setdefault("observations", [])
        if not observations or observations[-1]["date"] != today:
            observations.append({"date": today, "metric": row.get("metric")})
            old["occurrences"] += 1
        else:
            observations[-1]["metric"] = row.get("metric")
        old["observations"] = observations[-30:]
        old["trend"] = trend_of([x.get("metric") for x in observations])
        old["severity"] = classify(row, old["occurrences"], old["trend"])
        if old_status == "RESOLVED":
            old["status"] = "REGRESSED"
            old["severity"] = SEVERITIES[min(5, SEVERITIES.index(old["severity"]) + 1)]
        elif old_status in ("ACCEPTED", "SUPPRESSED"):
            expired = not old.get("suppress_until") or old["suppress_until"] <= today
            material_change = (isinstance(old_metric, (int, float)) and old_metric > 0 and
                               isinstance(row.get("metric"), (int, float)) and row["metric"] >= old_metric * 1.5)
            if expired or material_change or SEVERITIES.index(old["severity"]) > SEVERITIES.index(row["severity"]):
                old["status"] = "OPEN"
                old["accepted_reason"] = ""
                old["suppress_until"] = ""
    unavailable = {"flaky-test", "slow-test", "skipped-test"} if not TEST_RUNS.exists() else set()
    if not has_perf_report():
        unavailable.add("render-performance")
    for ident, old in prior.items():
        if old["category"] in unavailable:
            continue
        if ident not in seen and old["status"] not in ("RESOLVED", "ACCEPTED", "SUPPRESSED"):
            old["status"] = "RESOLVED"
    state["debt"] = sorted(prior.values(), key=lambda r: r["id"])
    state["scans"] = (state.get("scans", []) + [{"at": at, "observed": len(seen),
                       "open": sum(r["status"] in ("OPEN", "REGRESSED", "PLANNED", "IN_PROGRESS") for r in prior.values())}])[-90:]
    return state


def task_candidates(state, tasks, limit=None, campaign=None):
    limit = POLICY["max_tasks_per_scan"] if limit is None else limit
    eligible = []
    live = {"BACKLOG", "READY", "CLAIMED", "IN_PROGRESS", "VALIDATING", "PR_OPEN", "BLOCKED"}
    hotspot_paths = {r["subject"] for r in state["debt"] if r["category"] == "hotspot"
                     and r["status"] in ("OPEN", "REGRESSED", "PLANNED", "IN_PROGRESS")}
    for item in state["debt"]:
        if item["status"] not in ("OPEN", "REGRESSED") or item["occurrences"] < 2:
            continue
        if campaign and item["category"] not in CAMPAIGNS[campaign]:
            continue
        if item["severity"] not in ("D2", "D3"):
            continue
        if item["category"] in ("module-size", "function-size") and not (
                item["trend"] == "WORSENING" or item["subject"] in hotspot_paths):
            continue
        if item["category"] == "hotspot" and item["trend"] != "WORSENING" and not item.get("regressions"):
            continue
        if any(t.id in item["related_tasks"] and t.status in live for t in tasks):
            continue
        eligible.append(item)
    eligible.sort(key=lambda r: (-SEVERITIES.index(r["severity"]), -r["occurrences"], r["id"]))
    # A campaign or scan consumes at most one item per (category, subsystem).
    selected, keys = [], set()
    for row in eligible:
        key = (row["category"], row["subsystem"])
        if key not in keys:
            selected.append(row)
            keys.add(key)
        if len(selected) >= limit:
            break
    return selected


def sync_task_status(state, tasks):
    by_id = {task.id: task for task in tasks}
    for item in state["debt"]:
        if item["status"] not in ("PLANNED", "IN_PROGRESS"):
            continue
        linked = [by_id[ident] for ident in item["related_tasks"] if ident in by_id]
        if not linked:
            continue
        item["related_prs"] = sorted(set(item["related_prs"] + [task.pr for task in linked if task.pr]))
        if any(task.status in cp.ACTIVE_STATES for task in linked):
            item["status"] = "IN_PROGRESS"
        elif all(task.status in ("MERGED", "FAILED", "CANCELLED") for task in linked):
            item["status"] = "OPEN"  # the current scan still observes the cause


def create_tasks(state, tasks, *, limit=None, campaign=None):
    created = []
    for item in task_candidates(state, tasks, limit, campaign):
        batch = [r for r in state["debt"] if r["category"] == item["category"]
                 and r["subsystem"] == item["subsystem"] and r["status"] in ("OPEN", "REGRESSED")
                 and r["occurrences"] >= 2 and r["severity"] in ("D2", "D3")]
        title = (item["title"] if len(batch) == 1 else
                 f"Investigate {len(batch)} {item['category']} findings in {item['subsystem']}")
        task, duplicate = cp.intake(title, tasks, task_type="maintenance", subsystem=item["subsystem"],
                                    priority="P2" if item["severity"] == "D3" else "P3",
                                    acceptance=[f"Investigate debt IDs: {', '.join(r['id'] for r in batch)}",
                                                "Repair cause without weakening tests or changing behavior without review",
                                                "Verify finding clears on a later maintenance scan"], scan_repo=False)
        if duplicate["duplicate"]["verdict"] == "DUPLICATE":
            continue
        task.files_hint = sorted({r["subject"].split(":", 1)[0] for r in batch if "/" in r["subject"]})[:20]
        task.budget = item["estimated_effort"] if item["estimated_effort"] in cp.BUDGETS else "SMALL"
        task.save()
        tasks.append(task)
        for row in batch:
            row["related_tasks"].append(task.id)
            row["status"] = "PLANNED"
        created.append({"debt_ids": [r["id"] for r in batch], "task_id": task.id})
    return created


def sync_attention(state, attention_state=None, *, save_state=True):
    attention_state = attention.load() if attention_state is None else attention_state
    kept = [a for a in attention_state["alerts"] if a.get("source") != "maintenance"]
    routed = []
    for item in state["debt"]:
        if item["status"] not in ("OPEN", "REGRESSED", "PLANNED", "IN_PROGRESS"):
            continue
        sev = item["severity"]
        if sev in ("D0", "D1", "D2") or (sev == "D3" and item["occurrences"] < 3):
            continue
        alert = attention.signal("maintenance", item["id"], item["category"], item["title"],
                                 severity="critical" if sev == "D5" else "error" if sev == "D4" else "warning",
                                 priority="P1" if sev in ("D4", "D5") else "P2",
                                 blocking=1 if sev in ("D4", "D5") else 0, minutes=5,
                                 evidence=item["evidence"])
        alert["first_seen"] = item["first_seen"]
        routed.append(alert)
    attention_state["alerts"] = kept + routed
    if save_state:
        attention.save(attention_state)
    return routed


def budget(state):
    active = [r for r in state["debt"] if r["status"] in ("OPEN", "REGRESSED", "PLANNED", "IN_PROGRESS")]
    high = sum(SEVERITIES.index(r["severity"]) >= 3 for r in active)
    worsening = sum(r["trend"] == "WORSENING" for r in active)
    extra = min(0.20, high * 0.03 + worsening * 0.02)
    share = min(state["policy"]["max_maintenance_share"], state["policy"]["maintenance_share"] + extra)
    return {"maintenance_share_suggested": round(share, 2), "base_share": state["policy"]["maintenance_share"],
            "reasons": {"high_debt": high, "worsening": worsening}, "advisory": True}


def health(state):
    active = [r for r in state["debt"] if r["status"] in ("OPEN", "REGRESSED", "PLANNED", "IN_PROGRESS")]
    groups = {"Architecture": {"architecture", "hotspot", "module-size", "function-size"},
              "Tests": {"slow-test", "skipped-test"}, "Flakiness": {"flaky-test"},
              "Documentation": {"doc-link", "doc-index", "doc-freshness"},
              "Dependencies": {"dependency"}, "Performance": {"slow-test", "render-performance"}}
    def label(kinds):
        selected = [r for r in active if r["category"] in kinds]
        return "RISK" if any(r["severity"] in ("D4", "D5") for r in selected) else "WARNING" if selected else "GOOD"
    has_runs = TEST_RUNS.exists()
    has_perf = has_perf_report()
    dimensions = {name: ("UNKNOWN" if name in ("Tests", "Flakiness") and not has_runs
                         or name == "Performance" and not (has_runs or has_perf) else label(kinds))
                  for name, kinds in groups.items()}
    if not state["scans"]:
        dimensions = {name: "UNKNOWN" for name in dimensions}
    return {"schema": 1, "dimensions": dimensions,
            "critical_debt": sum(r["severity"] == "D5" for r in active),
            "high_debt": sum(r["severity"] in ("D3", "D4") for r in active),
            "maintenance_work": sum(r["status"] in ("PLANNED", "IN_PROGRESS") for r in active),
            "active_debt": len(active), "last_scan": state["scans"][-1]["at"] if state["scans"] else None,
            "budget": budget(state), "evidence_gaps": [] if has_runs else ["test-run history unavailable; flakiness and runtime unknown"]}


def safe_repair(item):
    subject = item.get("subject", "")
    return (item.get("safe_repair") == "index" and item["status"] in ("OPEN", "REGRESSED")
            and subject.startswith("docs/") and (ROOT / subject).resolve().is_relative_to((ROOT / "docs").resolve()))


def heal_index(state, *, apply=False):
    candidates = [r for r in state["debt"] if safe_repair(r)]
    index = ROOT / "docs" / "INDEX.md"
    text = index.read_text()
    changes = []
    for item in candidates:
        target = ROOT / item["subject"]
        if not target.is_file() or target.name in text:
            continue
        link = target.relative_to(index.parent).as_posix()
        changes.append(f"- [{target.stem.replace('_', ' ').title()}]({link})")
    if apply and changes:
        header = "\n## Maintenance index additions\n"
        if header not in text:
            text += header
        text += "\n".join(changes) + "\n"
        index.write_text(text)
    return {"eligible": len(changes), "applied": bool(apply and changes), "changes": changes,
            "validation": "Run governance and normal PR review before merge"}
