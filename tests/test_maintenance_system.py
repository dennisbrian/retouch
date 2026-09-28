"""Focused contracts for maintenance evidence and bounded routing."""
import json
from copy import deepcopy

from scripts.dev.maintenance_system import core, cli


def _row(category="slow-test", subject="tests/test_a.py::test_slow", metric=61, severity="D2"):
    return core.finding(category, subject, "Investigate test", severity=severity,
                        evidence=["observed"], subsystem="tests", metric=metric)


def test_stable_id_and_classification():
    row = _row()
    assert row["id"] == core.debt_id("slow-test", row["subject"])
    assert core.classify(row, 3, "WORSENING") == "D3"
    assert core.classify(core.finding("module-size", "x", "x", severity="D2"), 1) == "D2"
    assert core.classify(_row("flaky-test", "x", 5), 1) == "D3"


def test_recurrence_trend_and_same_day_scan():
    state = core.load("/does/not/exist")
    row = _row(metric=61)
    core.reconcile(state, [row], "2026-01-01T00:00:00+00:00")
    core.reconcile(state, [_row(metric=62)], "2026-01-01T12:00:00+00:00")
    assert state["debt"][0]["occurrences"] == 1
    core.reconcile(state, [_row(metric=63)], "2026-01-02T00:00:00+00:00")
    core.reconcile(state, [_row(metric=64)], "2026-01-03T00:00:00+00:00")
    assert state["debt"][0]["trend"] == "WORSENING"
    assert state["debt"][0]["severity"] == "D3"


def test_hotspot_is_activity_not_refactor(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    (tmp_path / "retouch").mkdir()
    (tmp_path / "retouch" / "engine.py").write_text("pass\n")
    rows = core.hotspot_findings({**core.POLICY, "hotspot_changes": 3},
                                 "retouch/engine.py\nretouch/engine.py\nretouch/engine.py\n")
    assert len(rows) == 1 and rows[0]["severity"] == "D2"
    assert "measure" in rows[0]["suggested_action"]


def test_exact_duplicate_helpers_only(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    package = tmp_path / "retouch"
    package.mkdir()
    body = "\n".join(f"    value += {i}" for i in range(20))
    (package / "a.py").write_text(f"def helper_a(value):\n{body}\n    return value\n")
    (package / "b.py").write_text(f"def helper_b(value):\n{body}\n    return value\n")
    rows = core.duplicate_helper_findings()
    assert len(rows) == 1 and rows[0]["category"] == "duplicate-helper"
    (package / "b.py").write_text("def helper_b(value):\n    return value\n")
    assert core.duplicate_helper_findings() == []


def test_suppression_expiry_material_change_and_regression():
    state = core.load("/does/not/exist")
    def observed(metric):
        return _row(category="module-size", subject="retouch/engine.py", metric=metric)
    core.reconcile(state, [observed(60)], "2026-01-01T00:00:00+00:00")
    item = state["debt"][0]
    item.update(status="ACCEPTED", suppress_until="2026-03-01", accepted_reason="known")
    core.reconcile(state, [observed(61)], "2026-01-02T00:00:00+00:00")
    assert item["status"] == "ACCEPTED"
    core.reconcile(state, [observed(95)], "2026-01-03T00:00:00+00:00")
    assert item["status"] == "OPEN"
    item.update(status="SUPPRESSED", suppress_until="2026-01-04")
    core.reconcile(state, [observed(96)], "2026-01-04T00:00:00+00:00")
    assert item["status"] == "OPEN"
    core.reconcile(state, [], "2026-01-05T00:00:00+00:00")
    assert item["status"] == "RESOLVED"
    core.reconcile(state, [observed(96)], "2026-01-06T00:00:00+00:00")
    assert item["status"] == "REGRESSED"


def test_task_generation_bounded_and_reuses_control_plane(monkeypatch):
    state = core.load("/does/not/exist")
    rows = [_row(subject=f"tests/test_{n}.py::test_slow") for n in range(5)]
    core.reconcile(state, rows, "2026-01-01T00:00:00+00:00")
    core.reconcile(state, rows, "2026-01-02T00:00:00+00:00")
    assert len(core.task_candidates(state, [], limit=3)) == 1  # same category/subsystem batch
    written = []
    monkeypatch.setattr(core.cp.Task, "save", lambda task: written.append(task))
    created = core.create_tasks(state, [], limit=3)
    assert len(created) == 1 and written[0].type == "maintenance"
    assert written[0].priority == "P3" and written[0].status == "BACKLOG"
    assert not core.task_candidates(state, written)


def test_size_and_hotspot_activity_alone_do_not_create_tasks():
    state = core.load("/does/not/exist")
    rows = [_row(category="module-size", subject="retouch/engine.py"),
            _row(category="hotspot", subject="retouch/other.py")]
    core.reconcile(state, rows, "2026-01-01T00:00:00+00:00")
    core.reconcile(state, rows, "2026-01-02T00:00:00+00:00")
    assert core.task_candidates(state, []) == []


def test_task_lifecycle_follows_control_plane_status():
    state = core.load("/does/not/exist")
    core.reconcile(state, [_row()], "2026-01-01T00:00:00+00:00")
    item = state["debt"][0]
    item.update(status="PLANNED", related_tasks=["TASK-0001"])
    task = core.cp.Task(id="TASK-0001", title="repair", status="IN_PROGRESS", pr="47")
    core.sync_task_status(state, [task])
    assert item["status"] == "IN_PROGRESS" and item["related_prs"] == ["47"]
    task.status = "MERGED"
    core.sync_task_status(state, [task])
    assert item["status"] == "OPEN"


def test_safe_repair_only_index(monkeypatch, tmp_path):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "INDEX.md").write_text("# Index\n")
    (tmp_path / "docs" / "NEW.md").write_text("# New\n")
    state = core.load("/does/not/exist")
    item = core.finding("doc-index", "docs/NEW.md", "Index new", safe_repair="index")
    item["status"] = "OPEN"
    state["debt"] = [item]
    assert core.safe_repair(item)
    assert not core.heal_index(state)["applied"]
    assert "NEW.md" not in (tmp_path / "docs" / "INDEX.md").read_text()
    assert core.heal_index(state, apply=True)["applied"]
    assert core.heal_index(state, apply=True)["eligible"] == 0
    assert not core.safe_repair({**item, "category": "architecture", "safe_repair": ""})


def test_flaky_slow_and_skipped_need_repeated_evidence():
    runs = ([{"test": "a", "outcome": x, "duration_seconds": 1, "retries": 1}
             for x in ("failed", "passed", "failed")]
            + [{"test": "b", "outcome": "passed", "duration_seconds": x}
               for x in (65, 70, 80)]
            + [{"test": "c", "outcome": "skipped"} for _ in range(3)])
    rows = core.test_findings(runs)
    assert {r["category"] for r in rows} == {"flaky-test", "slow-test", "skipped-test"}
    assert core.test_findings([runs[0]]) == []


def test_quality_lab_performance_verdict_reused(tmp_path):
    report = tmp_path / "perf.json"
    report.write_text(json.dumps({"kind": "perf", "regressions": [
        {"case": "portrait", "recipe": "natural", "verdict": "REGRESSION",
         "time_s": {"ratio": 1.9}, "peak_ram_mb": {"ratio": 1.2}}]}))
    rows = core.quality_perf_findings([report])
    assert len(rows) == 1 and rows[0]["severity"] == "D3"
    assert rows[0]["metric"] == 1.9


def test_attention_only_consequential_unsuppressed(monkeypatch):
    state = core.load("/does/not/exist")
    core.reconcile(state, [_row(category="architecture", severity="D4")], "2026-01-01T00:00:00+00:00")
    attention_state = {"schema": 1, "alerts": [], "decisions": [], "policies": [], "overrides": [], "daily": []}
    alerts = core.sync_attention(state, attention_state, save_state=False)
    assert len(alerts) == 1 and alerts[0]["source"] == "maintenance"
    state["debt"][0]["status"] = "ACCEPTED"
    assert core.sync_attention(state, attention_state, save_state=False) == []
    assert attention_state["alerts"] == []


def test_budget_health_and_json_stability(monkeypatch, tmp_path, capsys):
    path = tmp_path / "maintenance.json"
    state = core.load(path)
    core.reconcile(state, [_row(severity="D3")], "2026-01-01T00:00:00+00:00")
    core.save(state, path)
    assert json.loads(path.read_text())["schema"] == 1
    assert core.budget(state)["maintenance_share_suggested"] == 0.23
    assert core.health(state)["dimensions"]["Flakiness"] == "UNKNOWN"
    assert core.health(state)["dimensions"]["Performance"] == "UNKNOWN"
    monkeypatch.setattr(cli.core, "load", lambda: deepcopy(state))
    assert cli.main(["status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == 1 and payload["active_debt"] == 1
    assert cli.main(["debt", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["debt"][0]["id"].startswith("DEBT-")
    assert cli.main(["hotspots", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["hotspots"] == []
    assert cli.main(["plan", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == 1
