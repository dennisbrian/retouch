#!/usr/bin/env python3
"""Control plane CLI. Every command supports --json for machine consumers.

    scripts/dev/control-plane intake "title" [--type ...] [--subsystem ...]
    scripts/dev/control-plane list [--state ...]
    scripts/dev/control-plane ready
    scripts/dev/control-plane claim TASK_ID --agent NAME [--branch ...] [--force]
    scripts/dev/control-plane preflight TASK_ID
    scripts/dev/control-plane heartbeat TASK_ID [--note working|validating|waiting|blocked|done]
    scripts/dev/control-plane block TASK_ID TASK_ID...
    scripts/dev/control-plane unblock TASK_ID
    scripts/dev/control-plane release TASK_ID
    scripts/dev/control-plane complete TASK_ID [--pr N]
    scripts/dev/control-plane fail TASK_ID
    scripts/dev/control-plane status
    scripts/dev/control-plane plan [--max N]
    scripts/dev/control-plane conflicts
    scripts/dev/control-plane drift TASK_ID [--files a b c]
    scripts/dev/control-plane health [--pause] [--resume]
    scripts/dev/control-plane attention

Exit codes: 0 ok, 1 blocking finding, 2 setup error.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from . import core


def _emit(data: dict, as_json: bool, human: str = "") -> None:
    if as_json:
        print(json.dumps(data, indent=2, sort_keys=True))
    elif human:
        print(human)
    else:
        print(json.dumps(data, indent=2, sort_keys=True))


def _fmt_task(task) -> str:
    return (f"{task.id}  [{task.status:<11}] {task.priority} {task.type:<12} "
            f"{task.subsystem or '-':<14} {task.title}")


def cmd_intake(args) -> int:
    if args.priority in ("P3", "P4") and (args.type in (None, "feature")):
        from scripts.dev.attention_router import core as attention
        if attention.overload(attention.load()["daily"]):
            print("error: owner attention budget exceeded for 3 days; P3/P4 feature intake paused", file=sys.stderr)
            return 1
    tasks = core.load_all()
    acceptance = [line.strip() for line in (args.accept or "").split(";")
                  if line.strip()]
    task, report = core.intake(
        args.title, tasks, task_type=args.type or "", subsystem=args.subsystem or "",
        priority=args.priority, acceptance=acceptance or None,
        scan_repo=not args.no_repo_scan)
    task.save()
    dup = report["duplicate"]
    human = [f"created {_fmt_task(task)}", f"duplicate check: {dup['verdict']}"]
    for match in dup["matches"][:5]:
        human.append(f"  overlap: {match}")
    if dup["verdict"] == "DUPLICATE":
        human.append("  NOT blocking; review before claiming.")
    steps = core.decompose(task)
    if steps:
        human.append("suggested decomposition (edit/split if useful):")
        human += [f"  - {s}" for s in steps]
    _emit({"task": core.asdict(task), "duplicate": dup, "decompose": steps},
          args.json, "\n".join(human))
    return 0


def cmd_list(args) -> int:
    tasks = core.load_all()
    if args.state:
        wanted = {s.upper() for s in args.state.split(",")}
        tasks = [t for t in tasks if t.status in wanted]
    tasks.sort(key=lambda t: (core._PRIORITY_RANK.get(t.priority, 9), t.created_at))
    _emit({"tasks": [core.asdict(t) for t in tasks]}, args.json,
          "\n".join(_fmt_task(t) for t in tasks) or "no tasks")
    return 0


def cmd_ready(args) -> int:
    tasks = core.load_all()
    ready = core.ready_tasks(tasks)
    _emit({"ready": [core.asdict(t) for t in ready]}, args.json,
          "\n".join(_fmt_task(t) for t in ready) or "nothing ready")
    return 0


def cmd_preflight(args) -> int:
    tasks = core.load_all()
    task = core.find_task(tasks, args.task)
    check = core.preflight(task, tasks)
    lines = [check["verdict"]]
    lines += [f"  blocked: {b}" for b in check["blockers"]]
    lines += [f"  caution: {c}" for c in check["cautions"]]
    lines += [f"  test: {t}" for t in check["recommended_tests"]]
    _emit(check, args.json, "\n".join(lines))
    return 1 if check["verdict"] == "BLOCKED" else 0


def cmd_claim(args) -> int:
    tasks = core.load_all()
    task = core.find_task(tasks, args.task)
    try:
        check = core.claim(task, tasks, agent=args.agent, branch=args.branch or "",
                           force=args.force)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not check.get("claimed"):
        _emit(check, args.json,
              "BLOCKED:\n" + "\n".join(f"  - {b}" for b in check["blockers"]))
        return 1
    lines = [f"claimed {task.id} for {args.agent}",
             f"  branch: {check['branch']}",
             f"  leases: {', '.join(check['leases']) or 'none'}"]
    lines += [f"  caution: {c}" for c in check["cautions"]]
    manifest = check["intent_manifest"]
    lines.append("  intent manifest:")
    lines.append(f"    expected subsystems: {manifest['expected_subsystems']}")
    lines.append(f"    must NOT change: {manifest['must_not_change'] or 'n/a'}")
    lines.append(f"    expected tests: {manifest['expected_tests']}")
    _emit(check, args.json, "\n".join(lines))
    return 0


def _transition(args, target: str) -> int:
    tasks = core.load_all()
    task = core.find_task(tasks, args.task)
    allowed = core.TRANSITIONS.get(task.status, set())
    if target not in allowed:
        print(f"error: {task.status} -> {target} not allowed "
              f"(allowed: {sorted(allowed) or 'none — terminal'})", file=sys.stderr)
        return 2
    task.status = target
    task.heartbeat = core._now()
    if target in ("MERGED", "FAILED", "CANCELLED"):
        task.completed_at = task.heartbeat
    if getattr(args, "pr", None):
        task.pr = str(args.pr)
    task.save()
    _emit(core.asdict(task), args.json, f"{task.id} -> {target}")
    return 0


def cmd_block(args) -> int:
    tasks = core.load_all()
    task = core.find_task(tasks, args.task)
    deps = []
    for other in args.depends_on:
        dep = core.find_task(tasks, other)
        deps.append(dep.id)
    for dep_id in deps:
        if dep_id not in task.blocked_by:
            task.blocked_by.append(dep_id)
    cycles = core.find_cycles(tasks)
    if cycles:
        print("error: would create circular dependency: "
              + "; ".join(" -> ".join(c) for c in cycles), file=sys.stderr)
        return 2
    if task.status in ("READY", "BACKLOG"):
        task.status = "BLOCKED"
    task.heartbeat = core._now()
    task.save()
    _emit(core.asdict(task), args.json,
          f"{task.id} blocked by {', '.join(deps)}")
    return 0


def cmd_unblock(args) -> int:
    tasks = core.load_all()
    task = core.find_task(tasks, args.task)
    task.blocked_by = []
    if task.status == "BLOCKED":
        task.status = "READY"
    task.heartbeat = core._now()
    task.save()
    _emit(core.asdict(task), args.json, f"{task.id} unblocked")
    return 0


def cmd_heartbeat(args) -> int:
    tasks = core.load_all()
    task = core.find_task(tasks, args.task)
    task.heartbeat = core._now()
    note_map = {"validating": "VALIDATING", "blocked": "BLOCKED",
                "waiting": task.status, "working": "IN_PROGRESS",
                "done": task.status}
    if args.note:
        target = note_map[args.note]
        if target != task.status and target in core.TRANSITIONS.get(task.status, set()):
            task.status = target
    task.save()
    _emit({"id": task.id, "heartbeat": task.heartbeat, "status": task.status},
          args.json, f"{task.id} heartbeat {task.heartbeat} ({task.status})")
    return 0


def cmd_status(args) -> int:
    tasks = core.load_all()
    report = core.status_report(tasks)
    if args.json:
        _emit(report, True)
        return 0
    c = report["counts"]
    lines = ["ACTIVE ENGINEERING", ""]
    for entry in report["active"]:
        flag = " [STALE]" if entry.get("stale") else ""
        lines.append(f"{entry['id']}  {entry['agent'] or '-':<10} "
                     f"{entry['subsystem'] or '-':<14} {entry['status']:<11} "
                     f"{entry['priority']} {entry['title']}{flag}")
    if not report["active"]:
        lines.append("(none)")
    if report["blocked"]:
        lines += ["", "BLOCKED"]
        lines += [f"{b['id']}  {b['title']}  reason: {b['reason']}"
                  for b in report["blocked"]]
    lines += ["",
              f"backlog {c.get('BACKLOG', 0)} | ready {c.get('READY', 0)} | "
              f"active {len(report['active'])} | blocked {c.get('BLOCKED', 0)} | "
              f"pr_open {c.get('PR_OPEN', 0)} | merged {c.get('MERGED', 0)} | "
              f"failed {c.get('FAILED', 0)}"]
    if report["subsystem_pressure"]:
        lines.append("subsystem pressure: " + ", ".join(
            f"{k}={v}" for k, v in sorted(report["subsystem_pressure"].items())))
    if report["health"]["paused"]:
        lines.append("HEALTH: PAUSED — "
                     + "; ".join(report["health"]["pause_reasons"]))
    _emit(report, False, "\n".join(lines))
    return 0


def cmd_plan(args) -> int:
    tasks = core.load_all()
    result = core.plan(tasks, max_batch=args.max)
    if args.json:
        _emit(result, True)
        return 0
    lines = ["RECOMMENDED EXECUTION BATCH", ""]
    for i, item in enumerate(result["batch"], 1):
        lines.append(f"{i}. {item['id']}  {item['title']}")
        lines.append(f"   {item['type']} / {item['subsystem'] or '-'} "
                     f"conflict {item['conflict_risk']}  {item['agent_hint']}")
    if result["hold"]:
        lines += ["", "HOLD"]
        lines += [f"{h['id']}  reason: {h['reason']}" for h in result["hold"]]
    if result["blocked"]:
        lines += ["", "BLOCKED"]
        lines += [f"{b['id']}  {b['title']}  waiting for: {', '.join(b['waiting_for'])}"
                  for b in result["blocked"]]
    if result["paused"]:
        lines += ["", "CIRCUIT BREAKER ACTIVE — P0/hotfix/test/maintenance only",
                  "reasons: " + "; ".join(result["pause_reasons"])]
    _emit(result, False, "\n".join(lines))
    return 0


def cmd_conflicts(args) -> int:
    tasks = core.load_all()
    active = [t for t in tasks if t.status in core.ACTIVE_STATES]
    rows = []
    for i, a in enumerate(active):
        for b in active[i + 1:]:
            risk = core.conflict_risk(a, b)
            if risk != "LOW":
                rows.append({"a": a.id, "b": b.id, "risk": risk,
                             "shared": sorted(core.infer_subsystems(a)
                                              & core.infer_subsystems(b))})
    _emit({"conflicts": rows}, args.json,
          "\n".join(f"{r['a']} <-> {r['b']}  {r['risk']}  {r['shared']}"
                    for r in rows) or "no conflicts among active tasks")
    return 1 if any(r["risk"] == "HIGH" for r in rows) else 0


def cmd_drift(args) -> int:
    tasks = core.load_all()
    task = core.find_task(tasks, args.task)
    result = core.scope_drift(task, changed_files=args.files or None)
    lines = [f"SCOPE DRIFT: {result['level']}",
             f"expected: {result['expected_subsystems']}",
             f"actual:   {result['actual_subsystems']}"]
    if result["unexpected"]:
        lines.append(f"unexpected: {result['unexpected']}")
    if result["violations"]:
        lines.append(f"PROTECTED AREA VIOLATIONS: {result['violations']}")
    if result["budget_note"]:
        lines.append(result["budget_note"])
    _emit(result, args.json, "\n".join(lines))
    return 1 if result["level"] == "HIGH" else 0


def cmd_health(args) -> int:
    health = core.load_health()
    if args.pause:
        health["manual_pause"] = True
        core.save_health(health)
    elif args.resume:
        health["manual_pause"] = False
        health["pause_reasons"] = []
        core.save_health(health)
    tasks = core.load_all()
    result = core.evaluate_health(tasks)
    state = "PAUSED" if result["paused"] else "GREEN"
    human = (f"HEALTH: {state}\n" + "\n".join(f"  - {r}" for r in result["reasons"])
             if result["reasons"] else f"HEALTH: {state}")
    _emit(result, args.json, human)
    return 1 if result["paused"] else 0


def cmd_attention(args) -> int:
    tasks = core.load_all()
    items = core.attention_items(tasks)
    _emit({"attention": items, "count": len(items)}, args.json,
          "\n".join(f"- {i}" for i in items) or "no human attention required")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="control-plane",
                                     description=__doc__.splitlines()[0])
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output")
    parser.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("intake", parents=[common], help="create a structured task from an idea")
    p.add_argument("title")
    p.add_argument("--type", choices=core.TASK_TYPES)
    p.add_argument("--subsystem", choices=sorted(core.SUBSYSTEMS))
    p.add_argument("--priority", default="P2", choices=core.PRIORITIES)
    p.add_argument("--accept", help="semicolon-separated acceptance criteria")
    p.add_argument("--no-repo-scan", action="store_true",
                   help="skip branch/PR duplicate scan (offline)")
    p.set_defaults(func=cmd_intake)

    p = sub.add_parser("list", parents=[common], help="list tasks")
    p.add_argument("--state", help="comma-separated states filter")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("ready", parents=[common], help="tasks with all dependencies satisfied")
    p.set_defaults(func=cmd_ready)

    p = sub.add_parser("preflight", parents=[common], help="full pre-start check for a task")
    p.add_argument("task")
    p.set_defaults(func=cmd_preflight)

    p = sub.add_parser("claim", parents=[common], help="claim a task for an agent")
    p.add_argument("task")
    p.add_argument("--agent", required=True)
    p.add_argument("--branch", default="")
    p.add_argument("--force", action="store_true",
                   help="claim despite BLOCKED preflight (logs cautions)")
    p.set_defaults(func=cmd_claim)

    p = sub.add_parser("heartbeat", parents=[common], help="refresh lease / update progress")
    p.add_argument("task")
    p.add_argument("--note", choices=("working", "validating", "waiting",
                                      "blocked", "done"))
    p.set_defaults(func=cmd_heartbeat)

    p = sub.add_parser("block", parents=[common], help="mark task blocked by other task ids")
    p.add_argument("task")
    p.add_argument("depends_on", nargs="+")
    p.set_defaults(func=cmd_block)

    p = sub.add_parser("unblock", parents=[common], help="clear blocked_by list")
    p.add_argument("task")
    p.set_defaults(func=cmd_unblock)

    p = sub.add_parser("release", parents=[common], help="release claim back to READY")
    p.add_argument("task")
    p.set_defaults(func=lambda a: _transition(a, "READY"))

    p = sub.add_parser("complete", parents=[common], help="mark task merged/done")
    p.add_argument("task")
    p.add_argument("--pr", help="merged PR number")
    p.set_defaults(func=lambda a: _transition(a, "MERGED"))

    p = sub.add_parser("fail", parents=[common], help="mark task failed (feeds circuit breaker)")
    p.add_argument("task")
    p.set_defaults(func=lambda a: _transition(a, "FAILED"))

    p = sub.add_parser("status", parents=[common], help="active engineering overview")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("plan", parents=[common], help="recommended safe concurrent batch")
    p.add_argument("--max", type=int, default=5, dest="max")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("conflicts", parents=[common], help="pairwise conflicts among active tasks")
    p.set_defaults(func=cmd_conflicts)

    p = sub.add_parser("drift", parents=[common], help="scope-drift check vs intent manifest")
    p.add_argument("task")
    p.add_argument("--files", nargs="*", help="override changed-file list")
    p.set_defaults(func=cmd_drift)

    p = sub.add_parser("health", parents=[common], help="circuit breaker status")
    p.add_argument("--pause", action="store_true", help="manual pause")
    p.add_argument("--resume", action="store_true", help="clear pause")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("attention", parents=[common], help="items needing the human owner")
    p.set_defaults(func=cmd_attention)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyError as exc:
        print(f"error: unknown task {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
