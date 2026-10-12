"""CLI for the R&D Director: opportunities, hypotheses, negative results, portfolio."""
from __future__ import annotations

import argparse
import json
import sys

from . import core


def parser():
    p = argparse.ArgumentParser(
        prog="rd",
        description="R&D Director — proposes what to build next. Advisory only; "
                    "the human approves every stage transition and creates the tasks.")
    p.add_argument("--json", action="store_true", dest="global_json")
    sub = p.add_subparsers(dest="command")

    for name in ("brief", "evidence", "opportunities", "hypotheses", "negatives", "portfolio"):
        q = sub.add_parser(name)
        q.add_argument("--json", action="store_true")

    q = sub.add_parser("add")
    q.add_argument("title")
    q.add_argument("--kind", required=True, choices=core.KINDS)
    q.add_argument("--problem", required=True)
    q.add_argument("--proposal", default="")
    q.add_argument("--evidence", action="append", default=[])
    q.add_argument("--effort", default="MEDIUM", choices=core.EFFORTS)
    q.add_argument("--benefit", default="")
    q.add_argument("--risk", default="")
    q.add_argument("--source", default="manual")
    q.add_argument("--json", action="store_true")

    q = sub.add_parser("advance")
    q.add_argument("opp_id")
    q.add_argument("stage", choices=core.STAGES)
    q.add_argument("--reason", required=True)
    q.add_argument("--actor", default="owner")
    q.add_argument("--json", action="store_true")

    q = sub.add_parser("check")
    q.add_argument("title", help="idea text; matched against negative-result memory")
    q.add_argument("--json", action="store_true")

    q = sub.add_parser("reject")
    q.add_argument("tried")
    q.add_argument("--result", required=True)
    q.add_argument("--decision", default="rejected")
    q.add_argument("--retry-unless", default="")
    q.add_argument("--json", action="store_true")

    q = sub.add_parser("hyp")
    q.add_argument("text")
    q.add_argument("--success", required=True, help="success criteria")
    q.add_argument("--opp", default="", help="linked opportunity id")
    q.add_argument("--json", action="store_true")

    q = sub.add_parser("hyp-update")
    q.add_argument("hyp_id")
    q.add_argument("--status", choices=core.HYP_STATUSES)
    q.add_argument("--experiment", help="experiment id to attach")
    q.add_argument("--json", action="store_true")

    q = sub.add_parser("log-work")
    q.add_argument("kind", choices=sorted(core.PORTFOLIO))
    q.add_argument("description")
    q.add_argument("--share", type=float)
    q.add_argument("--json", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    command = args.command or "brief"
    as_json = getattr(args, "json", False) or args.global_json
    state = core.load()
    try:
        result = None
        mutate = True
        if command == "add":
            result = core.add_opportunity(state, args.title, args.kind, args.problem,
                                          proposal=args.proposal, evidence=args.evidence,
                                          effort=args.effort, benefit=args.benefit,
                                          risk=args.risk, source=args.source)
        elif command == "advance":
            result = core.advance(state, args.opp_id, args.stage, args.reason, actor=args.actor)
        elif command == "check":
            result = {"matches": core.check_negative(state, args.title)}
            mutate = False
        elif command == "reject":
            result = core.add_negative_result(state, args.tried, args.result,
                                              args.decision, retry_unless=args.retry_unless)
        elif command == "hyp":
            result = core.add_hypothesis(state, args.text, args.success, opp_id=args.opp)
        elif command == "hyp-update":
            result = core.update_hypothesis(state, args.hyp_id,
                                            status=args.status, experiment=args.experiment)
        elif command == "log-work":
            result = core.log_work(state, args.kind, args.description, share=args.share)
        elif command == "evidence":
            result = {"schema": 1, "evidence": core.collect_evidence()}
            mutate = False
        elif command == "opportunities":
            result = {"schema": 1, "opportunities": state["opportunities"]}
            mutate = False
        elif command == "hypotheses":
            result = {"schema": 1, "hypotheses": state["hypotheses"]}
            mutate = False
        elif command == "negatives":
            result = {"schema": 1, "negative_results": state["negative_results"]}
            mutate = False
        elif command == "portfolio":
            result = core.portfolio(state)
            mutate = False
        else:  # brief
            result = core.brief(state, evidence=core.collect_evidence())
            mutate = False
        if mutate:
            core.save(state)
        if as_json or command != "brief":
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"R&D brief: {result['open_opportunities']} open opportunities "
                  f"({result['by_stage']})")
            for opp in result["top"]:
                print(f"  {opp['id']} [{opp['kind']}/{opp['stage']}] {opp['title']} "
                      f"({len(opp['evidence'])} evidence)")
            for w in result["portfolio"]["warnings"]:
                print(f"  STRATEGY WARNING: {w}")
            print(f"  {result['authority']}")
        return 0
    except (ValueError, KeyError, TypeError) as exc:
        print(f"rd: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
