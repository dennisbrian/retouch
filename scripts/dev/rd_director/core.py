"""R&D Director: opportunity registry, hypothesis tracking, portfolio balance.

Read-only w.r.t. other subsystems; never creates Control Plane tasks itself.
Human stays the strategic authority — this only proposes. Stdlib only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from scripts.dev.attention_router import core as ar
from scripts.dev.maintenance_system import core as ms

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "control-plane" / "rd.json"
LAB_REPORTS = ROOT / "test_output" / "quality_lab" / "reports"

KINDS = ("RESEARCH", "FEATURE", "QUALITY", "PERFORMANCE", "MAINTENANCE", "ARCHITECTURE", "EXPERIMENT")
STAGES = ("IDEA", "EVIDENCE", "EXPERIMENT", "RESULT", "DECISION", "IMPLEMENTATION", "REJECTED", "SHIPPED")
EFFORTS = ("SMALL", "MEDIUM", "LARGE")
HYP_STATUSES = ("PROPOSED", "EXPERIMENTING", "CONFIRMED", "REJECTED")

# Target allocation of engineering effort; advisory, not a quota.
PORTFOLIO = {"QUALITY": 0.35, "FEATURE": 0.25, "PERFORMANCE": 0.15,
             "MAINTENANCE": 0.15, "RESEARCH": 0.10}
PORTFOLIO_WARN_GAP = 0.30  # one bucket exceeding target by this => warning


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def load(path=None):
    data = read_json(STATE if path is None else path, {})
    return {"schema": 1,
            "opportunities": data.get("opportunities", []),
            "hypotheses": data.get("hypotheses", []),
            "negative_results": data.get("negative_results", []),
            "work_log": data.get("work_log", [])}


def save(data, path=None):
    path = Path(STATE if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _next_id(rows, prefix):
    n = max((int(r["id"].split("-")[1]) for r in rows), default=0)
    return f"{prefix}-{n + 1:03d}"


def add_opportunity(state, title, kind, problem, *, evidence=None, effort="MEDIUM",
                    benefit="", risk="", source="manual", proposal=""):
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    if effort not in EFFORTS:
        raise ValueError(f"effort must be one of {EFFORTS}")
    if not title.strip() or not problem.strip():
        raise ValueError("title and problem are required")
    dup = next((o for o in state["opportunities"]
                if o["title"] == title and o["stage"] not in ("REJECTED", "SHIPPED")), None)
    if dup:
        raise ValueError(f"open opportunity with same title exists: {dup['id']}")
    opp = {"id": _next_id(state["opportunities"], "OPP"), "title": title, "kind": kind,
           "problem": problem, "proposal": proposal, "evidence": evidence or [],
           "effort": effort, "benefit": benefit, "risk": risk, "source": source,
           "stage": "IDEA", "created_at": now(), "updated_at": now(),
           "history": [{"at": now(), "stage": "IDEA", "reason": "created"}]}
    state["opportunities"].append(opp)
    return opp


def advance(state, opp_id, stage, reason, *, actor="owner"):
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if not reason.strip():
        raise ValueError("reason is required")
    opp = next((o for o in state["opportunities"] if o["id"] == opp_id), None)
    if opp is None:
        raise ValueError(f"unknown opportunity {opp_id}")
    if stage in ("IMPLEMENTATION", "SHIPPED") and not opp["evidence"]:
        raise ValueError("IMPLEMENTATION/SHIPPED requires evidence; idea -> coding directly is blocked")
    opp["stage"] = stage
    opp["updated_at"] = now()
    opp["history"].append({"at": now(), "stage": stage, "reason": reason, "actor": actor})
    return opp


def check_negative(state, title):
    """Return prior negative results whose text matches the title words."""
    words = {w.lower() for w in title.split() if len(w) > 3}
    hits = []
    for nr in state["negative_results"]:
        nr_words = {w.lower() for w in f"{nr['tried']} {nr['result']}".split() if len(w) > 3}
        if words & nr_words:
            hits.append(nr)
    return hits


def add_negative_result(state, tried, result, decision, *, retry_unless=""):
    if not tried.strip() or not decision.strip():
        raise ValueError("tried and decision are required")
    entry = {"id": _next_id(state["negative_results"], "NEG"), "tried": tried,
             "result": result, "decision": decision, "retry_unless": retry_unless,
             "at": now()}
    state["negative_results"].append(entry)
    return entry


def add_hypothesis(state, text, success_criteria, *, opp_id=""):
    if not text.strip() or not success_criteria.strip():
        raise ValueError("text and success_criteria are required")
    if opp_id and not any(o["id"] == opp_id for o in state["opportunities"]):
        raise ValueError(f"unknown opportunity {opp_id}")
    hyp = {"id": _next_id(state["hypotheses"], "HYP"), "text": text,
           "success_criteria": success_criteria, "status": "PROPOSED",
           "opportunity": opp_id, "experiments": [], "created_at": now()}
    state["hypotheses"].append(hyp)
    return hyp


def update_hypothesis(state, hyp_id, *, status=None, experiment=None):
    hyp = next((h for h in state["hypotheses"] if h["id"] == hyp_id), None)
    if hyp is None:
        raise ValueError(f"unknown hypothesis {hyp_id}")
    if status:
        if status not in HYP_STATUSES:
            raise ValueError(f"status must be one of {HYP_STATUSES}")
        hyp["status"] = status
    if experiment:
        hyp["experiments"].append({"id": experiment, "at": now()})
    return hyp


def collect_evidence(lab_reports=None, attention_signals=None, maintenance=None):
    """Pull weakness signals from existing subsystem state. Advisory only."""
    rows = []
    reports = lab_reports
    if reports is None:
        reports = ([max(LAB_REPORTS.glob("*.json"), key=lambda p: p.stat().st_mtime)]
                   if LAB_REPORTS.exists() and list(LAB_REPORTS.glob("*.json")) else [])
    for path in reports:
        report = read_json(path, {}) if isinstance(path, (Path, str)) else path
        if report.get("kind") != "comparison":
            continue
        commit = report.get("candidate_commit") or "unknown"
        for row in report.get("human_attention", []):
            if row.get("verdict") in ("REGRESSION", "REVIEW"):
                rows.append({"source": "quality-lab", "kind": row["verdict"],
                             "detail": f"{row.get('case')} / {row.get('recipe')} @ {commit}",
                             "at": report.get("as_of", "")})
    if attention_signals is None:
        attention_signals = ar.collect()
    for s in attention_signals:
        rows.append({"source": s["source"], "kind": s["kind"], "detail": s["title"], "at": ""})
    if maintenance is None:
        maintenance = ms.load() if hasattr(ms, "load") else {}
    for row in (maintenance or {}).get("debt", []) if isinstance(maintenance, dict) else []:
        rows.append({"source": "maintenance", "kind": "debt",
                     "detail": str(row.get("subject", row)), "at": ""})
    return rows


def log_work(state, kind, description, *, share=None):
    if kind not in PORTFOLIO:
        raise ValueError(f"kind must be one of {sorted(PORTFOLIO)}")
    entry = {"kind": kind, "description": description, "at": now()}
    if share is not None:
        entry["share"] = share
    state["work_log"].append(entry)
    return entry


def portfolio(state):
    """Actual work mix vs targets. Advisory only."""
    log = state["work_log"]
    if not log:
        return {"entries": 0, "actual": {}, "target": PORTFOLIO, "warnings": []}
    if any("share" in e for e in log):
        actual = {k: round(sum(e.get("share", 0) for e in log if e["kind"] == k), 3)
                  for k in PORTFOLIO}
    else:
        total = len(log)
        actual = {k: round(sum(1 for e in log if e["kind"] == k) / total, 3)
                  for k in PORTFOLIO}
    warnings = []
    for kind, target in PORTFOLIO.items():
        if actual.get(kind, 0) >= target + PORTFOLIO_WARN_GAP:
            warnings.append(f"{kind} at {actual[kind]:.0%} vs target {target:.0%}: rebalance")
    return {"entries": len(log), "actual": actual, "target": PORTFOLIO, "warnings": warnings}


def brief(state, *, evidence=None):
    open_opps = [o for o in state["opportunities"] if o["stage"] not in ("REJECTED", "SHIPPED")]
    port = portfolio(state)
    return {"schema": 1, "as_of": now(),
            "open_opportunities": len(open_opps),
            "by_stage": {s: sum(1 for o in open_opps if o["stage"] == s) for s in STAGES},
            "by_kind": {k: sum(1 for o in open_opps if o["kind"] == k) for k in KINDS},
            "top": sorted(open_opps,
                          key=lambda o: (-len(o["evidence"]), o["created_at"]))[:5],
            "hypotheses": state["hypotheses"],
            "negative_results": state["negative_results"],
            "portfolio": port,
            "recent_evidence": (evidence if evidence is not None else [])[-10:],
            "authority": "advisory only: human approves every stage transition"}
