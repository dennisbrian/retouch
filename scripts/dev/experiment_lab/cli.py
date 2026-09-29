"""Experiment Lab CLI. JSON contracts are stable, schema version 1."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import core, runner


def parser():
    p = argparse.ArgumentParser(prog="experiment", description=__doc__)
    p.add_argument("--json", action="store_true", dest="global_json")
    sub = p.add_subparsers(dest="command")
    for name in ("list", "status", "negatives", "history"):
        q = sub.add_parser(name)
        q.add_argument("--json", action="store_true")
    q = sub.add_parser("similar")
    q.add_argument("question")
    q.add_argument("--hypothesis", default="")
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("create")
    q.add_argument("--hypothesis", required=True)
    q.add_argument("--title", required=True)
    q.add_argument("--question", required=True)
    q.add_argument("--budget", choices=sorted(core.BUDGETS), default="SMALL")
    q.add_argument("--new-evidence", default="")
    q.add_argument("--json", action="store_true")
    for name in ("show", "results", "report", "compare", "promote", "archive"):
        q = sub.add_parser(name)
        q.add_argument("id")
        q.add_argument("--json", action="store_true")
    q = sub.add_parser("design")
    q.add_argument("id")
    q.add_argument("file", help="frozen design JSON")
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("candidate")
    q.add_argument("id")
    q.add_argument("--approach", required=True)
    q.add_argument("--mechanism", required=True)
    q.add_argument("--rationale", required=True)
    q.add_argument("--patch", default="")
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("prepare")
    q.add_argument("id")
    q.add_argument("candidate")
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("run")
    q.add_argument("id")
    q.add_argument("candidate")
    q.add_argument("--split", choices=("development", "validation", "holdout"), default="development")
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("eliminate")
    q.add_argument("id")
    q.add_argument("candidate")
    q.add_argument("--reason", required=True)
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("blind")
    q.add_argument("id")
    q.add_argument("--split", choices=("validation", "holdout"), default="holdout")
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("review")
    q.add_argument("id")
    q.add_argument("file", help="JSON array of blinded votes")
    q.add_argument("--split", choices=("validation", "holdout"), default="holdout")
    q.add_argument("--json", action="store_true")
    q = sub.add_parser("cleanup")
    q.add_argument("id")
    q.add_argument("--retention-days", type=int, default=30)
    q.add_argument("--apply", action="store_true")
    q.add_argument("--discard-prototypes", action="store_true")
    q.add_argument("--json", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command or "status"
    as_json = args.global_json or getattr(args, "json", False)
    try:
        if command == "list":
            result = {"schema": 1, "experiments": [{"experiment_id": e["experiment_id"],
                "title": e["title"], "status": e["status"], "conclusion": e["conclusion"]} for e in core.list_all()]}
        elif command == "status":
            result = core.mission_status()
        elif command == "history":
            result = {"schema": 1, "history": [{"experiment_id": e["experiment_id"],
                "events": e["history"]} for e in core.list_all()]}
        elif command == "negatives":
            from scripts.dev.rd_director import core as rd
            result = {"schema": 1, "experiment_negatives": [
                {"experiment_id": e["experiment_id"], **r} for e in core.list_all() for r in e["negative_findings"]],
                "rd_negatives": rd.load()["negative_results"]}
        elif command == "similar":
            result = {"schema": 1, **core.similarity(args.question, args.hypothesis, repo_scan=True)}
        elif command == "create":
            result = core.create(args.hypothesis, args.title, args.question,
                                 budget=args.budget, new_evidence=args.new_evidence, repo_scan=True)
        else:
            exp = core.load(args.id)
            if command == "show":
                result = exp
            elif command in ("results", "report"):
                result = core.report(exp)
            elif command == "design":
                spec = core.read_json(args.file)
                if not isinstance(spec, dict):
                    raise ValueError("design file must be a JSON object")
                result = core.validate_design(exp, spec)
            elif command == "candidate":
                result = core.add_candidate(exp, args.approach, args.mechanism, args.rationale,
                                            patch=args.patch)
            elif command == "prepare":
                result = runner.prepare(exp, core.candidate(exp, args.candidate))
            elif command == "run":
                result = runner.run(exp, core.candidate(exp, args.candidate), args.split)
            elif command == "eliminate":
                cand = core.candidate(exp, args.candidate)
                if cand["status"] == "ELIMINATED":
                    raise ValueError("candidate already eliminated")
                cand["status"] = "ELIMINATED"
                cand["elimination_reason"] = args.reason
                core.save(exp)
                result = {"experiment_id": exp["experiment_id"], "candidate": cand["id"],
                          "status": cand["status"], "reason": args.reason}
            elif command == "compare":
                result = core.compare(exp)
                if exp["conclusion"] in ("NO_WINNER", "WINNER", "INCONCLUSIVE"):
                    core.sync_negatives(exp)
                    core.sync_attention(exp)
            elif command == "promote":
                result = core.promote(exp)
                core.sync_attention(exp)
            elif command == "archive":
                result = core.archive(exp)
                core.sync_attention(exp)
            elif command == "cleanup":
                if args.retention_days < 0:
                    raise ValueError("retention days must be nonnegative")
                result = core.cleanup(exp, retention_days=args.retention_days,
                                      apply=args.apply, discard_prototypes=args.discard_prototypes)
            elif command == "blind":
                result = core.blind_sheet(exp, split=args.split)
            elif command == "review":
                votes = core.read_json(Path(args.file))
                result = core.record_review(exp, args.split, votes)
            else:
                raise ValueError("unknown command")
        if as_json or command not in ("status", "list"):
            print(json.dumps(result, indent=2, sort_keys=True))
        elif command == "status":
            print(f"R&D LAB  running {result['running']}  analyzing {result['analyzing']}  "
                  f"winners {result['winners']}  no winner {result['no_winner']}  promoted {result['promoted']}")
        else:
            for row in result["experiments"]:
                print(f"{row['experiment_id']} [{row['status']}] {row['title']}")
        return 0
    except (ValueError, KeyError, TypeError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"experiment: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
