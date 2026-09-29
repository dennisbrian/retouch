"""CLI for bounded maintenance evidence and Control Plane task creation."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from scripts.dev.control_plane import core as cp
from . import core


def parser():
    p = argparse.ArgumentParser(prog="maintenance")
    p.add_argument("--json", action="store_true", dest="global_json")
    sub = p.add_subparsers(dest="command")
    for name in ("scan", "status", "debt", "hotspots", "plan"):
        q = sub.add_parser(name)
        q.add_argument("--json", action="store_true")
        if name in ("scan", "plan"):
            q.add_argument("--max-tasks", type=int, default=core.POLICY["max_tasks_per_scan"])
        if name == "scan":
            q.add_argument("--no-tasks", action="store_true")
    q = sub.add_parser("campaign")
    q.add_argument("name", choices=sorted(core.CAMPAIGNS))
    q.add_argument("--max-tasks", type=int, default=core.POLICY["max_tasks_per_scan"])
    q.add_argument("--json", action="store_true")
    for name in ("resolve", "accept", "suppress"):
        q = sub.add_parser(name)
        q.add_argument("id")
        q.add_argument("--reason", required=True)
        q.add_argument("--until", help="YYYY-MM-DD; required for accept/suppress")
        q.add_argument("--json", action="store_true")
    q = sub.add_parser("heal")
    q.add_argument("kind", choices=("index",))
    q.add_argument("--apply", action="store_true")
    q.add_argument("--json", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command or "status"
    as_json = args.global_json or getattr(args, "json", False)
    state = core.load()
    try:
        if command == "scan":
            if args.max_tasks < 0:
                raise ValueError("--max-tasks must be nonnegative")
            findings = core.collect(state["policy"])
            core.reconcile(state, findings)
            control_tasks = cp.load_all()
            core.sync_task_status(state, control_tasks)
            tasks = [] if args.no_tasks else core.create_tasks(state, control_tasks, limit=args.max_tasks)
            alerts = core.sync_attention(state)
            core.save(state)
            result = {"schema": 1, "observed": len(findings), "tasks_created": tasks,
                      "attention_alerts": [a["id"] for a in alerts], "health": core.health(state)}
        elif command == "status":
            result = core.health(state)
        elif command == "debt":
            result = {"schema": 1, "debt": state["debt"]}
        elif command == "hotspots":
            result = {"schema": 1, "hotspots": [r for r in state["debt"] if r["category"] == "hotspot"]}
        elif command == "plan":
            if args.max_tasks < 0:
                raise ValueError("--max-tasks must be nonnegative")
            result = {"schema": 1, "budget": core.budget(state),
                      "candidates": core.task_candidates(state, cp.load_all(), args.max_tasks),
                      "note": "scan creates at most the bounded candidate count; Control Plane schedules tasks"}
        elif command == "campaign":
            if args.max_tasks < 0:
                raise ValueError("--max-tasks must be nonnegative")
            tasks = core.create_tasks(state, cp.load_all(), limit=args.max_tasks, campaign=args.name)
            core.save(state)
            result = {"schema": 1, "campaign": args.name, "tasks_created": tasks}
        elif command in ("resolve", "accept", "suppress"):
            item = next((r for r in state["debt"] if r["id"] == args.id), None)
            if item is None:
                raise ValueError("unknown debt ID")
            if command in ("accept", "suppress"):
                if not args.until:
                    raise ValueError("--until is required")
                date.fromisoformat(args.until)
                if args.until <= date.today().isoformat():
                    raise ValueError("--until must be in the future")
            if not args.reason.strip():
                raise ValueError("--reason is required")
            item["status"] = {"resolve": "RESOLVED", "accept": "ACCEPTED", "suppress": "SUPPRESSED"}[command]
            item["accepted_reason"] = args.reason
            item["suppress_until"] = args.until or ""
            item.setdefault("history", []).append({"at": core.utc_now(), "action": command,
                                                    "reason": args.reason, "until": args.until or ""})
            core.save(state)
            core.sync_attention(state)
            result = {"schema": 1, "debt": item}
        else:
            result = {"schema": 1, **core.heal_index(state, apply=args.apply)}
        if as_json:
            print(json.dumps(result, indent=2, sort_keys=True))
        elif command == "status":
            print("REPOSITORY MAINTENANCE HEALTH")
            for key, value in result["dimensions"].items():
                print(f"  {key:<15} {value}")
            print(f"  Critical debt: {result['critical_debt']}  High debt: {result['high_debt']}  Maintenance work: {result['maintenance_work']}")
            if result["evidence_gaps"]:
                print("  Evidence gaps: " + "; ".join(result["evidence_gaps"]))
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, KeyError, OSError) as exc:
        print(f"maintenance: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
