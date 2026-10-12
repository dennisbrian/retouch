"""CLI for owner attention and decision memory."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from . import core


def parser():
    p = argparse.ArgumentParser(prog="attention")
    p.add_argument("--json", action="store_true", dest="global_json")
    sub = p.add_subparsers(dest="command")
    for name in ("brief", "decisions", "metrics", "policies"):
        q = sub.add_parser(name)
        q.add_argument("--json", action="store_true")
        if name == "brief":
            q.add_argument("--budget", type=int, default=core.DAILY_MINUTES)
        if name == "decisions":
            q.add_argument("item_id", nargs="?")
    q = sub.add_parser("record")
    q.add_argument("item_id")
    q.add_argument("action", choices=core.ACTIONS)
    q.add_argument("--reason", required=True)
    q.add_argument("--actor", default="owner")
    q.add_argument("--until", default="")
    q.add_argument("--temporary-until", default="")
    q.add_argument("--policy-key", default="")
    q.add_argument("--false-positive", action="store_true")
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("ingest")
    q.add_argument("file", help="JSON array of signal objects; stable source/subject/kind required")
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("daily")
    q.add_argument("date", help="YYYY-MM-DD")
    q.add_argument("demand_minutes", type=int)
    q.add_argument("--budget", type=int, default=core.DAILY_MINUTES)
    q.add_argument("--json", action="store_true")
    return p


def signals(state):
    return core.collect() + state["alerts"]


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command or "brief"
    as_json = getattr(args, "json", False) or args.global_json
    state = core.load()
    try:
        if command == "record":
            if args.item_id not in {s["id"] for s in signals(state)}:
                raise ValueError("unknown item id; run attention --json first")
            for field in ("until", "temporary_until"):
                value = getattr(args, field)
                if value:
                    datetime.fromisoformat(value)
            result = core.record(state, args.item_id, args.action, args.reason, until=args.until,
                                 actor=args.actor, policy_key=args.policy_key, temporary_until=args.temporary_until)
            result["false_positive"] = args.false_positive
            state["decisions"][-1]["false_positive"] = args.false_positive
            core.save(state)
        elif command == "ingest":
            raw = core.read_json(args.file, None)
            if not isinstance(raw, list):
                raise ValueError("file must contain a JSON array")
            count = 0
            for item in raw:
                for field in ("source", "subject", "kind", "title"):
                    if not item.get(field):
                        raise ValueError(f"missing {field}")
                if item.get("priority", "P2") not in ("P0", "P1", "P2", "P3", "P4"):
                    raise ValueError("invalid priority")
                if item.get("severity", "warning") not in ("critical", "error", "warning", "info", "success"):
                    raise ValueError("invalid severity")
                if item.get("minutes", 3) < 0 or item.get("blocking", 0) < 0:
                    raise ValueError("minutes and blocking must be nonnegative")
                row = core.signal(item["source"], item["subject"], item["kind"], item["title"],
                                  severity=item.get("severity", "warning"), priority=item.get("priority", "P2"),
                                  blocking=item.get("blocking", 0), minutes=item.get("minutes", 3),
                                  evidence=item.get("evidence", []), parent=item.get("parent", ""),
                                  resolved=item.get("resolved", False))
                previous = next((s for s in state["alerts"] if s["id"] == row["id"]), None)
                row["first_seen"] = previous.get("first_seen") if previous else core.now()
                state["alerts"] = [s for s in state["alerts"] if s["id"] != row["id"]]
                state["alerts"].append(row)
                count += 1
            core.save(state)
            result = {"ingested": count, "unique_alerts": len(state["alerts"])}
        elif command == "daily":
            datetime.fromisoformat(args.date)
            if args.demand_minutes < 0 or args.budget < 0:
                raise ValueError("minutes must be nonnegative")
            state["daily"] = [r for r in state["daily"] if r["date"] != args.date]
            state["daily"].append({"date": args.date, "demand_minutes": args.demand_minutes,
                                   "budget_minutes": args.budget})
            core.save(state)
            result = {"overload": core.overload(state["daily"]), "intake": "pause P3/P4 feature intake" if core.overload(state["daily"]) else "normal",
                      "exceptions": ["hotfix", "recovery", "critical bugfix", "near-completion"]}
        elif command == "decisions":
            ds = state["decisions"]
            ov = state["overrides"]
            if args.item_id:
                ds = [d for d in ds if d["item_id"] == args.item_id]
                ov = [d for d in ov if d["item_id"] == args.item_id]
            result = {"schema": 1, "decisions": ds, "overrides": ov}
        elif command == "policies":
            result = {"schema": 1, "policies": state["policies"]}
        else:
            if getattr(args, "budget", core.DAILY_MINUTES) < 0:
                raise ValueError("budget must be nonnegative")
            routed = core.route(signals(state), state, budget=getattr(args, "budget", core.DAILY_MINUTES))
            result = core.metrics(state, routed) if command == "metrics" else routed
            if command == "brief":
                result["batches"] = core.batches(result["decisions"])
                result["circuit_breaker"] = {"active": core.overload(state["daily"]),
                    "action": "pause P3/P4 feature intake" if core.overload(state["daily"]) else "normal",
                    "exceptions": ["hotfix", "recovery", "critical bugfix", "near-completion"]}
        if as_json:
            print(json.dumps(result, indent=2, sort_keys=True))
        elif command == "brief":
            print(f"Owner brief: {len(result['decisions'])} decisions, ~{result['estimated_minutes']} min; debt {result['attention_debt_minutes']} min")
            for row in result["decisions"]:
                print(f"  {row['level']} {row['id']} {row['title']} ({row['source']})")
            if result["deferred"]:
                print(f"  {len(result['deferred'])} deferred by budget")
            if result["circuit_breaker"]["active"]:
                print("  CIRCUIT BREAKER: pause P3/P4 feature intake")
        else:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (ValueError, KeyError, TypeError) as exc:
        print(f"attention: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
