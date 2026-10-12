"""Small, deterministic owner-attention router. Stdlib only."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.dev.control_plane import core as cp

ROOT = Path(__file__).resolve().parents[3]
STATE = ROOT / "control-plane" / "attention.json"
GOV_STATE = ROOT / ".git" / "governance-state.json"
LAB_REPORTS = ROOT / "test_output" / "quality_lab" / "reports"
LEVELS = ("A0", "A1", "A2", "A3", "A4", "A5")
ACTIONS = ("escalate", "defer", "dismiss", "override", "supersede", "temporary", "deeper-evidence", "decide")
DAILY_MINUTES = 30
OVERLOAD_DAYS = 3


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def load(path=None):
    data = read_json(STATE if path is None else path, {})
    return {"schema": 1, "alerts": data.get("alerts", []),
            "decisions": data.get("decisions", []), "policies": data.get("policies", []),
            "overrides": data.get("overrides", []), "daily": data.get("daily", [])}


def save(data, path=None):
    path = Path(STATE if path is None else path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def key(source, subject, kind):
    return hashlib.sha256(f"{source}|{subject}|{kind}".encode()).hexdigest()[:16]


def signal(source, subject, kind, title, *, severity="warning", priority="P2",
           blocking=0, minutes=3, evidence=None, parent="", resolved=False):
    return {"id": key(source, subject, kind), "source": source, "subject": subject,
            "kind": kind, "title": title, "severity": severity, "priority": priority,
            "blocking": blocking, "minutes": minutes, "evidence": evidence or [],
            "parent": parent, "resolved": resolved}


def collect(tasks=None, gov=None, reports=None):
    """Read existing producer state. Missing optional sources are an explicit gap."""
    tasks = cp.load_all() if tasks is None else tasks
    gov = read_json(GOV_STATE, {}) if gov is None else gov
    reports = ([max(LAB_REPORTS.glob("*.json"), key=lambda p: p.stat().st_mtime)]
               if reports is None and LAB_REPORTS.exists() and list(LAB_REPORTS.glob("*.json"))
               else (reports or []))
    out = []
    for t in tasks:
        if t.status == "FAILED":
            out.append(signal("control-plane", t.id, "failed-task", f"{t.id} failed: {t.title}",
                              severity="critical" if t.priority == "P0" else "error",
                              priority=t.priority, blocking=1, evidence=[t.id]))
        if t.status in cp.ACTIVE_STATES:
            last = cp._parse_ts(t.heartbeat) or cp._parse_ts(t.started_at)
            if last and datetime.now(timezone.utc).timestamp() - last > cp.STALE_HOURS * 3600:
                out.append(signal("control-plane", t.id, "stale-task", f"{t.id} needs recovery or reassignment",
                                  priority=t.priority, blocking=1, evidence=[t.id]))
    for cycle in cp.find_cycles(tasks):
        subject = ":".join(sorted(set(cycle)))
        out.append(signal("control-plane", subject, "dependency-cycle", f"Resolve dependency cycle: {subject}",
                          severity="error", blocking=len(set(cycle)), evidence=sorted(set(cycle))))
    health = cp.load_health()
    if cp.health_paused(health):
        out.append(signal("control-plane", "health", "paused", "Review control-plane pause",
                          severity="error", evidence=health.get("pause_reasons", [])))
    for row in gov.get("merge_history", [])[-20:]:
        if not row.get("ok", True):
            sha = row.get("sha", "unknown")
            out.append(signal("pr-governor", sha, "post-merge-failure", f"Investigate post-merge failure {sha}",
                              severity="critical", blocking=1, evidence=[sha]))
    for path in reports:
        report = read_json(path, {}) if isinstance(path, (Path, str)) else path
        if report.get("kind") != "comparison":
            continue
        commit = report.get("candidate_commit") or "unknown"
        for row in report.get("human_attention", []):
            if row.get("verdict") not in ("REGRESSION", "REVIEW"):
                continue
            subject = f"{commit}:{row.get('case')}:{row.get('recipe')}"
            out.append(signal("quality-lab", subject, "visual-review",
                              f"{row['verdict']}: {row.get('case')} / {row.get('recipe')}",
                              severity="error" if row["verdict"] == "REGRESSION" else "warning",
                              evidence=row.get("findings", [])))
    return out


def classify(item):
    if item.get("resolved") or item.get("severity") in ("success", "info"):
        return "A0"
    if item.get("severity") == "critical":
        return "A5"
    if item.get("severity") == "error" and item.get("blocking", 0):
        return "A4"
    if item.get("severity") == "error":
        return "A3"
    if item.get("blocking", 0):
        return "A3"
    return "A2" if item.get("severity") == "warning" else "A1"


def score(item):
    level = item.get("level") or classify(item)
    return (LEVELS.index(level) * 100 + min(int(item.get("blocking", 0)), 10) * 20
            + (4 - int(str(item.get("priority", "P2"))[1])) * 5)


def active_policy(item, policies, at=None):
    at = at or now()
    matches = [p for p in policies if p.get("active", True)
               and p.get("key") == f"{item['source']}:{item['kind']}"
               and (not p.get("expires_at") or p["expires_at"] > at)]
    return max(matches, key=lambda p: p.get("created_at", ""), default=None)


def route(signals, state, *, budget=DAILY_MINUTES, at=None):
    at = at or now()
    latest = {d["item_id"]: d for d in state["decisions"]}
    by_id = {s["id"]: s for s in signals if not s.get("resolved") and s.get("severity") not in ("success", "info")}
    # Parent decision represents child alerts; repeated producer output retains one stable id.
    for s in list(by_id.values()):
        if s.get("parent") and s["parent"] in by_id:
            by_id.pop(s["id"], None)
    rows = []
    auto = 0
    for s in by_id.values():
        d = latest.get(s["id"])
        policy = active_policy(s, state["policies"], at)
        owner_changed_after_policy = d and policy and d["id"] != policy["id"] and d["at"] >= policy["created_at"]
        if policy and not owner_changed_after_policy and policy["action"] in ("dismiss", "defer"):
            auto += 1
            continue
        if d and d["action"] in ("dismiss", "decide", "override", "supersede"):
            continue
        if d and d["action"] == "temporary" and d.get("temporary_until", "") > at:
            continue
        row = dict(s)
        row["level"] = classify(s)
        if d and d["action"] == "escalate":
            row["level"] = LEVELS[min(5, LEVELS.index(row["level"]) + 1)]
        if d and d["action"] == "defer" and d.get("until", "") > at:
            continue
        row["score"] = score(row)
        row["policy_id"] = policy["id"] if policy else None
        rows.append(row)
    rows.sort(key=lambda x: (-x["score"], x["id"]))
    selected, deferred, spent = [], [], 0
    for r in rows:
        if r["level"] in ("A4", "A5") or spent + r["minutes"] <= budget:
            selected.append(r)
            spent += r["minutes"]
        else:
            deferred.append(r)
    return {"schema": 1, "as_of": at, "budget_minutes": budget,
            "estimated_minutes": spent, "decisions": selected, "deferred": deferred,
            "auto_resolved": auto, "attention_debt_minutes": sum(x["minutes"] for x in deferred),
            "distribution": {level: sum(x["level"] == level for x in rows) for level in LEVELS}}


def batches(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((row["source"], row["kind"]), []).append(row)
    return [{"source": k[0], "kind": k[1], "item_ids": [r["id"] for r in values],
             "count": len(values), "max_level": max((r["level"] for r in values), key=LEVELS.index),
             "estimated_minutes": sum(r["minutes"] for r in values)}
            for k, values in sorted(grouped.items())]


def record(state, item_id, action, reason, *, until="", actor="owner", policy_key="", temporary_until=""):
    if action not in ACTIONS:
        raise ValueError("invalid action")
    if not reason.strip():
        raise ValueError("reason is required")
    if action == "defer" and not until:
        raise ValueError("defer requires --until")
    if action == "temporary" and not temporary_until:
        raise ValueError("temporary decision requires --temporary-until")
    entry = {"id": f"D-{len(state['decisions'])+1:05d}", "item_id": item_id, "action": action,
             "reason": reason, "actor": actor, "at": now(), "until": until,
             "temporary_until": temporary_until}
    state["decisions"].append(entry)
    if action in ("escalate", "defer", "dismiss", "override", "supersede", "temporary", "deeper-evidence"):
        state["overrides"].append(dict(entry))
    if policy_key:
        for p in state["policies"]:
            if p["key"] == policy_key and p.get("active", True):
                p["active"] = False
                p["superseded_by"] = entry["id"]
        if action in ("dismiss", "defer"):
            state["policies"].append({"id": entry["id"], "key": policy_key, "action": action,
                                      "reason": reason, "created_at": entry["at"],
                                      "expires_at": temporary_until or until, "active": True})
    return entry


def overload(daily, at=None):
    day = datetime.fromisoformat(at or now()).date()
    by_day = {x["date"]: x for x in daily}
    return all(by_day.get((day - timedelta(days=n)).isoformat(), {}).get("demand_minutes", 0) >
               by_day.get((day - timedelta(days=n)).isoformat(), {}).get("budget_minutes", DAILY_MINUTES)
               for n in range(OVERLOAD_DAYS))


def metrics(state, routed):
    ds = state["decisions"]
    today = routed["as_of"][:10]
    today_ds = [d for d in ds if d["at"][:10] == today]
    first_seen = {a["id"]: a.get("first_seen") for a in state["alerts"]}
    elapsed = []
    for d in ds:
        seen = first_seen.get(d["item_id"])
        if seen:
            try:
                elapsed.append((datetime.fromisoformat(d["at"]) - datetime.fromisoformat(seen)).total_seconds() / 3600)
            except ValueError:
                pass
    return {"human_decisions_today": len(today_ds), "estimated_human_minutes_today": routed["estimated_minutes"],
            "actual_deferred_decisions": sum(d["action"] == "defer" for d in ds),
            "repeat_question_rate": 0 if not ds else round((len(ds) - len({d['item_id'] for d in ds})) / len(ds), 3),
            "policy_reuse_rate": 0 if not routed["decisions"] else round(routed["auto_resolved"] / (routed["auto_resolved"] + len(routed["decisions"])), 3),
            "false_positive_alerts": sum(d.get("false_positive", False) for d in ds),
            "dismissed_alerts": sum(d["action"] == "dismiss" for d in ds),
            "alerts_auto_resolved": routed["auto_resolved"],
            "blocking_tasks_unblocked_per_decision": None,
            "average_time_to_decision_hours": round(sum(elapsed) / len(elapsed), 2) if elapsed else None,
            "time_to_decision_samples": len(elapsed),
            "alert_volume": len(state["alerts"]),
            "attention_debt_minutes": routed["attention_debt_minutes"],
            "distribution": routed["distribution"], "overload": overload(state["daily"], routed["as_of"])}
