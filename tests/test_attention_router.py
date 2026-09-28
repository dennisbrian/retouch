"""Attention router contract tests; no production state is written."""
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.dev.attention_router import core

ROOT = Path(__file__).resolve().parents[1]


def s(subject="x", **kw):
    return core.signal("test", subject, "owner-choice", subject, **kw)


def test_classification_and_priority_order():
    rows = [s("low"), s("high", severity="critical"), s("block", blocking=3)]
    assert [core.classify(r) for r in rows] == ["A2", "A5", "A3"]
    result = core.route(rows, core.load(Path("/nonexistent/attention.json")), budget=30)
    assert [r["subject"] for r in result["decisions"]] == ["high", "block", "low"]


def test_budget_blocking_multiplier_and_debt():
    rows = [s("a", minutes=10), s("b", minutes=10, blocking=2),
            s("c", minutes=10, severity="error", blocking=1)]
    result = core.route(rows, core.load(Path("/nonexistent/attention.json")), budget=10)
    assert [r["subject"] for r in result["decisions"]] == ["c"]
    assert result["attention_debt_minutes"] == 20
    assert core.score(rows[1]) > core.score(rows[0])


def test_batching_duplicate_and_parent_suppression():
    parent = s("parent")
    child = s("child", parent=parent["id"])
    result = core.route([parent, parent, child], core.load(Path("/nonexistent/attention.json")))
    assert len(result["decisions"]) == 1
    assert core.batches(result["decisions"])[0]["count"] == 1


def test_memory_policy_reuse_supersession_and_override():
    state = core.load(Path("/nonexistent/attention.json"))
    row = s()
    core.record(state, row["id"], "dismiss", "known false positive", policy_key="test:owner-choice")
    assert core.route([row], state)["auto_resolved"] == 1
    core.record(state, row["id"], "supersede", "new evidence", policy_key="test:owner-choice")
    assert not state["policies"][0]["active"]
    assert state["policies"][0]["superseded_by"] == state["decisions"][1]["id"]
    assert len(state["policies"]) == 1
    assert len(state["overrides"]) == 2


def test_defer_temporary_and_deeper_evidence():
    state = core.load(Path("/nonexistent/attention.json"))
    row = s()
    core.record(state, row["id"], "defer", "await result", until="2999-01-01T00:00:00+00:00")
    assert core.route([row], state)["decisions"] == []
    core.record(state, row["id"], "deeper-evidence", "need more photos")
    assert core.route([row], state)["decisions"]
    core.record(state, row["id"], "temporary", "trial", temporary_until="2999-01-01T00:00:00+00:00")
    assert core.route([row], state)["decisions"] == []
    assert core.route([row], state, at="3000-01-01T00:00:00+00:00")["decisions"]


def test_no_noise_and_auto_resolution():
    state = core.load(Path("/nonexistent/attention.json"))
    result = core.route([s("ok", severity="success"), s("resolved", resolved=True), s("real")], state)
    assert len(result["decisions"]) == 1


def test_overload_three_consecutive_days():
    day = datetime.now(timezone.utc).date()
    rows = [{"date": (day-timedelta(days=i)).isoformat(), "demand_minutes": 40,
             "budget_minutes": 30} for i in range(3)]
    assert core.overload(rows)
    rows[1]["demand_minutes"] = 20
    assert not core.overload(rows)


def test_json_output_stability(tmp_path, monkeypatch, capsys):
    from scripts.dev.attention_router import cli
    monkeypatch.setattr(core, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(core, "collect", lambda: [])
    assert cli.main(["--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == 1
    assert set(("decisions", "deferred", "distribution", "batches", "circuit_breaker")) <= result.keys()
    assert cli.main(["brief", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == 1
    assert cli.main(["decisions", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == 1


def test_collect_existing_signals(monkeypatch):
    from scripts.dev.control_plane import core as cp
    monkeypatch.setattr(cp, "load_health", lambda: {"manual_pause": False, "pause_reasons": []})
    task = cp.Task(id="TASK-0099", title="broken", status="FAILED", priority="P1")
    report = {"kind": "comparison", "candidate_commit": "abc", "human_attention": [
        {"case": "face", "recipe": "natural", "verdict": "REGRESSION", "findings": []},
        {"case": "other", "recipe": "natural", "verdict": "PASS"}]}
    rows = core.collect(tasks=[task], gov={"merge_history": [{"sha": "abc", "ok": False}]}, reports=[report])
    assert {r["source"] for r in rows} == {"control-plane", "pr-governor", "quality-lab"}
    assert len(rows) == 3


def test_explicit_escalation_overrides_reused_policy():
    state = core.load(Path("/nonexistent/attention.json"))
    row = s()
    core.record(state, row["id"], "dismiss", "old case", policy_key="test:owner-choice")
    core.record(state, row["id"], "escalate", "new serious evidence")
    routed = core.route([row], state)
    assert routed["decisions"][0]["level"] == "A3"


def test_control_plane_intake_circuit_breaker(monkeypatch, tmp_path):
    from scripts.dev.control_plane import cli as cp_cli
    from scripts.dev.control_plane import core as cp
    monkeypatch.setattr(cp, "QUEUE_DIR", tmp_path / "tasks")
    day = datetime.now(timezone.utc).date()
    rows = [{"date": (day-timedelta(days=i)).isoformat(), "demand_minutes": 40,
             "budget_minutes": 30} for i in range(3)]
    monkeypatch.setattr(core, "load", lambda: {"daily": rows})
    assert cp_cli.main(["intake", "New idea", "--type", "feature", "--priority", "P3"]) == 1
    assert not list(tmp_path.rglob("TASK-*.json"))
