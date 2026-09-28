"""Tests for scripts/dev/control_plane (core logic + CLI surface).

Queue state is redirected to a tmp dir per test via monkeypatched
core.QUEUE_DIR / core.HEALTH_FILE — the real control-plane/ dir is never
touched. GitHub (`gh`) calls are stubbed out; git calls use the real repo
(read-only).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.dev.control_plane import core  # noqa: E402
from scripts.dev.control_plane import cli  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "QUEUE_DIR", tmp_path / "tasks")
    monkeypatch.setattr(core, "HEALTH_FILE", tmp_path / "health.json")
    monkeypatch.setattr(core, "_gh_json", lambda *a: [])
    monkeypatch.setattr(core, "_git", lambda *a: "deadbeef" if a[0] == "rev-parse" else "")
    yield


def make_task(**kw) -> core.Task:
    defaults = dict(id="TASK-0001", title="do the thing", created_at=core._now())
    defaults.update(kw)
    return core.Task(**defaults)


# ---------------------------------------------------------------------------
# Task parsing / schema round-trip
# ---------------------------------------------------------------------------

class TestTaskSchema:
    def test_round_trip(self):
        task = make_task(subsystem="face-analysis", files_hint=["retouch/detection.py"])
        task.save()
        loaded = core.Task.load(task.path)
        assert loaded == task

    def test_unknown_keys_ignored(self, tmp_path):
        path = core.QUEUE_DIR / "TASK-0099.json"
        core.QUEUE_DIR.mkdir(parents=True)
        path.write_text(json.dumps({"id": "TASK-0099", "title": "x", "future": 1}))
        assert core.Task.load(path).id == "TASK-0099"

    def test_next_id_gap_safe(self):
        tasks = [make_task(id="TASK-0001"), make_task(id="TASK-0007")]
        assert core.next_id(tasks) == "TASK-0008"

    def test_intake_guesses_type_and_subsystem(self):
        task, report = core.intake("Fix hang in batch CLI", [], scan_repo=False)
        assert task.type == "bugfix"
        assert task.subsystem == "cli"
        assert task.status == "BACKLOG"
        assert report["duplicate"]["verdict"] == "UNIQUE"

    def test_intake_xmp_with_gui_wiring_prefers_export(self):
        task, _ = core.intake("Add XMP sidecar support with CLI and GUI wiring",
                              [], scan_repo=False)
        assert task.subsystem == "export"

    def test_intake_rejects_bad_enum(self):
        with pytest.raises(ValueError):
            core.intake("x", [], task_type="wish", scan_repo=False)


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------

class TestTransitions:
    @pytest.mark.parametrize("src,dst,ok", [
        ("BACKLOG", "READY", True),
        ("READY", "CLAIMED", True),
        ("CLAIMED", "IN_PROGRESS", True),
        ("IN_PROGRESS", "VALIDATING", True),
        ("VALIDATING", "PR_OPEN", True),
        ("PR_OPEN", "MERGED", True),
        ("BACKLOG", "MERGED", False),
        ("MERGED", "READY", False),
        ("READY", "PR_OPEN", False),
    ])
    def test_transition_matrix(self, src, dst, ok):
        assert (dst in core.TRANSITIONS[src]) is ok


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

class TestDependencies:
    def test_unfinished_dependency_blocks_ready(self):
        a = make_task(id="TASK-0001", status="IN_PROGRESS")
        b = make_task(id="TASK-0002", blocked_by=["TASK-0001"])
        tasks = [a, b]
        assert core.dependency_check(b, tasks)
        assert [t.id for t in core.ready_tasks(tasks)] == []

    def test_merged_dependency_unblocks(self):
        a = make_task(id="TASK-0001", status="MERGED")
        b = make_task(id="TASK-0002", blocked_by=["TASK-0001"])
        assert core.dependency_check(b, [a, b]) == []
        assert [t.id for t in core.ready_tasks([a, b])] == ["TASK-0002"]

    def test_cycle_detection(self):
        a = make_task(id="TASK-0001", blocked_by=["TASK-0002"])
        b = make_task(id="TASK-0002", blocked_by=["TASK-0001"])
        cycles = core.find_cycles([a, b])
        assert cycles and set(cycles[0]) == {"TASK-0001", "TASK-0002"}

    def test_no_false_cycle_on_chain(self):
        a = make_task(id="TASK-0001")
        b = make_task(id="TASK-0002", blocked_by=["TASK-0001"])
        c = make_task(id="TASK-0003", blocked_by=["TASK-0002"])
        assert core.find_cycles([a, b, c]) == []

    def test_preflight_blocked_by_cycle(self):
        a = make_task(id="TASK-0001", blocked_by=["TASK-0002"])
        b = make_task(id="TASK-0002", blocked_by=["TASK-0001"])
        check = core.preflight(a, [a, b])
        assert check["verdict"] == "BLOCKED"
        assert any("circular" in x for x in check["blockers"])


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection:
    def test_eye_visibility_vs_closed_eye_detection(self):
        existing = make_task(title="closed-eye detection",
                             subsystem="face-analysis", status="READY")
        result = core.duplicate_verdict("automatic eye visibility", "face-analysis",
                                        [existing], scan_repo=False)
        assert result["verdict"] in ("DUPLICATE", "POSSIBLE_OVERLAP")

    def test_watermark_variants(self):
        existing = make_task(title="watermark cleanup stage", status="READY")
        result = core.duplicate_verdict("remove watermark", "", [existing],
                                        scan_repo=False)
        assert result["verdict"] != "UNIQUE"

    def test_weak_similarity_does_not_block(self):
        existing = make_task(title="improve skin smoothing", status="READY")
        result = core.duplicate_verdict("improve export filenames", "",
                                        [existing], scan_repo=False)
        assert result["verdict"] == "UNIQUE"

    def test_merged_task_not_live_duplicate(self):
        existing = make_task(title="closed-eye detection", status="MERGED")
        result = core.duplicate_verdict("closed-eye detection", "", [existing],
                                        scan_repo=False)
        assert result["verdict"] != "DUPLICATE"


# ---------------------------------------------------------------------------
# Leases
# ---------------------------------------------------------------------------

class TestLeases:
    def test_exclusive_lease_blocks(self):
        holder = make_task(id="TASK-0001", subsystem="pipeline",
                           status="IN_PROGRESS", assigned_agent="agent-A",
                           heartbeat=core._now())
        claimant = make_task(id="TASK-0002", subsystem="pipeline")
        warnings = core.lease_conflicts(claimant, [holder, claimant])
        assert any("EXCLUSIVE" in w for w in warnings)
        check = core.preflight(claimant, [holder, claimant])
        assert check["verdict"] == "BLOCKED"

    def test_caution_lease_warns_not_blocks(self):
        holder = make_task(id="TASK-0001", subsystem="recipes",
                           status="IN_PROGRESS", heartbeat=core._now())
        claimant = make_task(id="TASK-0002", subsystem="recipes")
        check = core.preflight(claimant, [holder, claimant])
        assert check["verdict"] == "CAUTION"

    def test_shared_lease_silent(self):
        holder = make_task(id="TASK-0001", subsystem="docs",
                           status="IN_PROGRESS", heartbeat=core._now())
        claimant = make_task(id="TASK-0002", subsystem="docs")
        assert core.lease_conflicts(claimant, [holder, claimant]) == []

    def test_stale_lease_lapses(self):
        old = "2020-01-01T00:00:00+00:00"
        holder = make_task(id="TASK-0001", subsystem="pipeline",
                           status="IN_PROGRESS", heartbeat=old, started_at=old)
        claimant = make_task(id="TASK-0002", subsystem="pipeline")
        assert core.lease_conflicts(claimant, [holder, claimant]) == []

    def test_subsystem_inferred_from_files_hint(self):
        task = make_task(files_hint=["retouch/engine.py", "tests/test_engine.py"])
        assert {"pipeline", "tests"} <= core.infer_subsystems(task)


# ---------------------------------------------------------------------------
# Claiming
# ---------------------------------------------------------------------------

class TestClaim:
    def test_claim_sets_manifest_and_lease(self):
        task = make_task(subsystem="face-analysis")
        task.save()
        result = core.claim(task, [task], agent="agent-A")
        assert result["claimed"]
        assert task.status == "CLAIMED"
        assert "pipeline" in task.intent_manifest["must_not_change"]
        assert task.base_commit == "deadbeef"

    def test_claim_refused_when_blocked(self):
        blocker = make_task(id="TASK-0001", status="IN_PROGRESS")
        task = make_task(id="TASK-0002", blocked_by=["TASK-0001"])
        result = core.claim(task, [blocker, task], agent="agent-A")
        assert not result.get("claimed")
        assert task.status == "BACKLOG"

    def test_claim_force_overrides(self):
        blocker = make_task(id="TASK-0001", status="IN_PROGRESS")
        task = make_task(id="TASK-0002", blocked_by=["TASK-0001"], status="BLOCKED")
        result = core.claim(task, [blocker, task], agent="agent-A", force=True)
        assert result["claimed"]


# ---------------------------------------------------------------------------
# Parallel planner
# ---------------------------------------------------------------------------

class TestPlanner:
    def test_disjoint_subsystems_parallel_safe(self):
        a = make_task(id="TASK-0001", subsystem="docs", type="docs")
        b = make_task(id="TASK-0002", subsystem="export")
        assert core.conflict_risk(a, b) == "LOW"

    def test_pipeline_overlap_high(self):
        a = make_task(id="TASK-0001", subsystem="pipeline")
        b = make_task(id="TASK-0002", files_hint=["retouch/engine.py"])
        assert core.conflict_risk(a, b) == "HIGH"

    def test_plan_holds_exclusive_overlap(self):
        a = make_task(id="TASK-0001", subsystem="pipeline", status="IN_PROGRESS",
                      heartbeat=core._now())
        b = make_task(id="TASK-0002", subsystem="pipeline")
        c = make_task(id="TASK-0003", subsystem="docs", type="docs")
        result = core.plan([a, b, c])
        batch_ids = {item["id"] for item in result["batch"]}
        assert "TASK-0003" in batch_ids
        assert "TASK-0002" not in batch_ids
        assert any(h["id"] == "TASK-0002" for h in result["hold"])

    def test_plan_reports_blocked_chain(self):
        a = make_task(id="TASK-0001", status="IN_PROGRESS", heartbeat=core._now())
        b = make_task(id="TASK-0002", blocked_by=["TASK-0001"], status="BLOCKED")
        result = core.plan([a, b])
        assert result["blocked"] and result["blocked"][0]["id"] == "TASK-0002"

    def test_priority_ordering(self):
        low = make_task(id="TASK-0001", priority="P4", created_at="2026-01-01T00:00:00+00:00")
        high = make_task(id="TASK-0002", priority="P1", created_at="2026-06-01T00:00:00+00:00")
        ready = core.ready_tasks([low, high])
        assert ready[0].id == "TASK-0002"


# ---------------------------------------------------------------------------
# Scope drift & budgets
# ---------------------------------------------------------------------------

class TestScopeDrift:
    def test_in_scope_change(self):
        task = make_task(subsystem="face-analysis",
                         intent_manifest=core.build_intent_manifest(
                             make_task(subsystem="face-analysis")))
        result = core.scope_drift(task, ["retouch/detection.py",
                                         "tests/test_detection.py"])
        assert result["level"] in ("NONE", "LOW")
        assert result["violations"] == []

    def test_protected_pipeline_violation(self):
        base = make_task(subsystem="face-analysis")
        task = make_task(subsystem="face-analysis",
                         intent_manifest=core.build_intent_manifest(base))
        result = core.scope_drift(task, ["retouch/detection.py", "retouch/engine.py"])
        assert result["level"] == "HIGH"
        assert "pipeline" in result["violations"]

    def test_small_budget_churn_flagged(self):
        base = make_task(subsystem="docs", type="docs", budget="SMALL")
        task = make_task(subsystem="docs", type="docs", budget="SMALL",
                         intent_manifest=core.build_intent_manifest(base))
        many_docs = [f"docs/plans/RESEARCH_X_{i}.md" for i in range(160)]
        result = core.scope_drift(task, many_docs)
        assert result["budget_note"]


# ---------------------------------------------------------------------------
# Circuit breaker / health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_failed_task_spike_pauses(self):
        now = core._now()
        tasks = [make_task(id=f"TASK-000{i}", status="FAILED", completed_at=now)
                 for i in range(3)]
        result = core.evaluate_health(tasks, open_prs=0)
        assert result["paused"]
        assert any("FAILED" in r for r in result["reasons"])

    def test_pr_flood_pauses(self):
        result = core.evaluate_health([], open_prs=core.CB_MAX_OPEN_PRS + 1)
        assert result["paused"]

    def test_calm_repo_green(self):
        result = core.evaluate_health([], open_prs=2)
        assert not result["paused"]

    def test_pause_blocks_feature_preflight_not_hotfix(self):
        core.save_health({"manual_pause": True, "pause_reasons": ["manual"],
                          "history": []})
        feature = make_task(id="TASK-0001", type="feature")
        hotfix = make_task(id="TASK-0002", type="hotfix", priority="P0")
        assert core.preflight(feature, [feature])["verdict"] == "BLOCKED"
        assert core.preflight(hotfix, [hotfix])["verdict"] != "BLOCKED"


# ---------------------------------------------------------------------------
# Status / staleness / decomposition
# ---------------------------------------------------------------------------

class TestStatus:
    def test_stale_detection(self):
        old = "2020-01-01T00:00:00+00:00"
        task = make_task(status="IN_PROGRESS", heartbeat=old, started_at=old)
        report = core.status_report([task])
        assert report["stale"] == [task.id]

    def test_attention_flags_cycles_and_failures(self):
        a = make_task(id="TASK-0001", blocked_by=["TASK-0002"])
        b = make_task(id="TASK-0002", blocked_by=["TASK-0001"])
        c = make_task(id="TASK-0003", status="FAILED")
        items = core.attention_items([a, b, c])
        assert any("circular" in i for i in items)
        assert any("FAILED" in i for i in items)


class TestDecomposition:
    def test_bugfix_not_decomposed(self):
        task, _ = core.intake("Fix hang in batch export", [], scan_repo=False)
        assert core.decompose(task) == []

    def test_multi_subsystem_feature_decomposed(self):
        task = make_task(subsystem="export",
                         files_hint=["retouch/io.py", "gui.py", "cli.py"],
                         title="Add XMP sidecar support with CLI and GUI wiring")
        steps = core.decompose(task)
        assert any("GUI" in s for s in steps)
        assert any("tests" in s for s in steps)


# ---------------------------------------------------------------------------
# CLI surface / JSON stability
# ---------------------------------------------------------------------------

class TestCli:
    def run_cli(self, *argv):
        return cli.main(list(argv))

    def test_intake_then_list_json(self, capsys):
        assert self.run_cli("intake", "Fix thing", "--no-repo-scan") == 0
        assert self.run_cli("--json", "list") == 0
        blocks = capsys.readouterr().out
        # second JSON block is the list output
        out = json.loads(blocks[blocks.index("\n{") + 1:])
        assert len(out["tasks"]) == 1
        assert out["tasks"][0]["type"] == "bugfix"

    def test_json_round_trip(self, capsys):
        assert self.run_cli("intake", "Add export zoom", "--no-repo-scan",
                            "--json") == 0
        intake_out = json.loads(capsys.readouterr().out)
        task_id = intake_out["task"]["id"]
        assert self.run_cli("--json", "claim", task_id, "--agent", "agent-X") == 0
        claim_out = json.loads(capsys.readouterr().out)
        assert claim_out["claimed"]
        assert self.run_cli("--json", "status") == 0
        status = json.loads(capsys.readouterr().out)
        assert status["active"][0]["agent"] == "agent-X"
        # JSON keys are part of the machine contract — assert them.
        assert {"counts", "active", "blocked", "stale", "subsystem_pressure",
                "leases", "health"} <= set(status)

    def test_transition_via_cli(self, capsys):
        self.run_cli("intake", "Task A", "--no-repo-scan")
        capsys.readouterr()
        assert self.run_cli("claim", "TASK-0001", "--agent", "a") == 0
        assert self.run_cli("heartbeat", "TASK-0001", "--note", "validating") == 0
        assert self.run_cli("complete", "TASK-0001", "--pr", "42") == 0
        tasks = core.load_all()
        assert tasks[0].status == "MERGED"
        assert tasks[0].pr == "42"

    def test_bad_transition_rejected(self, capsys):
        self.run_cli("intake", "Task A", "--no-repo-scan")
        capsys.readouterr()
        assert self.run_cli("complete", "TASK-0001") == 2

    def test_block_cycle_rejected_via_cli(self, capsys):
        self.run_cli("intake", "Task A", "--no-repo-scan")
        self.run_cli("intake", "Task B", "--no-repo-scan")
        capsys.readouterr()
        assert self.run_cli("block", "TASK-0001", "TASK-0002") == 0
        assert self.run_cli("block", "TASK-0002", "TASK-0001") == 2

    def test_unknown_task_exit_2(self, capsys):
        assert self.run_cli("claim", "TASK-9999", "--agent", "a") == 2
        assert "unknown task" in capsys.readouterr().err

    def test_wrapper_script_runs(self):
        proc = subprocess.run(["scripts/dev/control-plane", "--json", "status"],
                              cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode == 0
        assert json.loads(proc.stdout)["counts"] is not None
