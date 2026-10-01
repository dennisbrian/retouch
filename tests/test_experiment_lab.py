"""Experiment design, isolation, evidence, and promotion contracts."""
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from scripts.dev.experiment_lab import core, runner, cli


@pytest.fixture
def laboratory(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "ROOT", tmp_path)
    monkeypatch.setattr(core, "REGISTRY", tmp_path / "experiments")
    monkeypatch.setattr(core, "WORK_ROOT", tmp_path / "test_output" / "experiment_lab")
    monkeypatch.setattr(core, "CORPUS", tmp_path / "quality_lab" / "corpus_manifest.json")
    monkeypatch.setattr(core, "THRESHOLDS", tmp_path / "quality_lab" / "thresholds.json")
    core.THRESHOLDS.parent.mkdir()
    core.THRESHOLDS.write_text("{}\n")
    cases, baseline_cases = [], {}
    for n in range(6):
        ident = f"case-{n}"
        inp = tmp_path / "test_output" / "quality_lab" / ident / "input.png"
        out = tmp_path / "test_output" / "quality_lab" / ident / "baseline" / "natural.png"
        inp.parent.mkdir(parents=True)
        out.parent.mkdir(parents=True)
        inp.write_bytes(f"input-{n}".encode())
        out.write_bytes(f"output-{n}".encode())
        cases.append({"id": ident, "input": str(inp.relative_to(tmp_path)), "subject": ident,
                      "evidence_class": "real_photo"})
        baseline_cases[ident] = {"input_sha256": core.sha256(inp), "recipes": {
            "natural": {"output": str(out), "output_sha256": core.sha256(out),
                        "metrics": {"global": {"edge_retention": 0.6}}}}}
    core.write_json(core.CORPUS, {"cases": cases})
    baseline_path = tmp_path / "baseline.json"
    base = "a" * 40
    core.write_json(baseline_path, {"kind": "baseline", "provenance": {
        "commit": base, "working_tree_dirty": False, "recipes": ["natural"],
        "engine_versions": {"cv2": "4.11"}}, "cases": baseline_cases})
    monkeypatch.setattr(core, "git", lambda *args, **kwargs: "blob-id" if args[0] in ("rev-parse", "hash-object") else "")
    rd_state = {"hypotheses": [{"id": "HYP-001", "text": "edge confidence helps",
                                "success_criteria": "better edges"}], "negative_results": []}
    exp = core.create("HYP-001", "Edge confidence", "Can local confidence improve wig edges?",
                      rd_state=rd_state)
    spec = {"base_commit": base, "baseline_report": str(baseline_path),
            "corpus_manifest": str(core.CORPUS), "recipes": ["natural"],
            "success_criteria": {"metric": "global.edge_retention", "direction": "higher",
                                 "min_delta": 0.05, "quality_tolerance": 0.01},
            "failure_criteria": {"max_regression_fraction": 0.02, "max_runtime_ratio": 1.10,
                                 "max_ram_ratio": 1.15},
            "constraints": {"max_loc_added": 500, "max_changed_files": 8},
            "reject_all_if": "all candidates harm backgrounds"}
    return exp, spec, rd_state


def _run_record(exp, cand, split, root, *, delta=0.1, runtime=1.05, regressions=0):
    path = root / f"{cand['id']}-{split}.json"
    path.write_text("{}")
    count = len(exp["corpus"]["cases"][split])
    row = {"split": split, "round": 1, "fingerprint": f"{cand['id']}-code",
           "elapsed_seconds": 2.0, "report": str(path), "report_sha256": core.sha256(path),
           "quality": {"PASS": count - regressions, "REVIEW": 0, "REGRESSION": regressions},
           "performance": {"PASS": count, "REVIEW": 0, "REGRESSION": 0},
           "target": {"sample_size": count, "median_delta": delta, "spread": [delta, delta],
                      "positive_pairs": count},
           "max_time_ratio": runtime, "max_ram_ratio": 1.02,
           "complexity": {"loc_added": 40, "changed_files": 1, "paths": ["retouch/engine.py"]},
           "environment": {"cv2": "4.11"}, "case_ids": exp["corpus"]["cases"][split]}
    cand["runs"].append(row)
    cand["rounds"] = 1
    cand["complexity"] = row["complexity"]
    cand["status"] = "MEASURED"
    return row


def test_design_freezes_baseline_and_criteria(laboratory):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    assert exp["status"] == "DESIGNED" and exp["corpus"]["real_holdout"]
    assert exp["corpus"]["cases"]["qualified"]
    core.ensure_frozen(exp)
    with pytest.raises(ValueError, match="frozen"):
        core.validate_design(exp, spec)
    changed = deepcopy(exp)
    changed["success_criteria"]["min_delta"] = 0.001
    with pytest.raises(ValueError, match="frozen"):
        core.ensure_frozen(changed)


def test_dirty_or_different_baseline_rejected(laboratory):
    exp, spec, _ = laboratory
    baseline = core.read_json(spec["baseline_report"])
    baseline["provenance"]["working_tree_dirty"] = True
    core.write_json(spec["baseline_report"], baseline)
    with pytest.raises(ValueError, match="clean base"):
        core.validate_design(exp, spec)


def test_subject_split_and_holdout_separation():
    cases = [{"id": f"a-{n}", "subject": "A"} for n in range(2)] + [
        {"id": f"{n}", "subject": str(n)} for n in range(5)]
    split = core.split_cases(cases, "seed")
    assert split["qualified"]
    assert set(split["development"]).isdisjoint(split["holdout"])
    bad = {"development": ["a-0", "0", "1"], "validation": ["a-1", "2"],
           "holdout": ["3", "4"]}
    with pytest.raises(ValueError, match="subject group"):
        core.validate_split(cases, bad)
    small = core.split_cases(cases[:3], "seed")
    assert not small["qualified"] and small["holdout"] == []


def test_distinct_candidates_and_budget(laboratory):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    with pytest.raises(ValueError, match="duplicates"):
        core.add_candidate(exp, "gradient 2", "local edge gradient", "new name only")
    core.add_candidate(exp, "semantic", "segmentation boundary fusion", "semantic edge")
    core.add_candidate(exp, "propagation", "guided propagation field", "spatial coherence")
    with pytest.raises(ValueError, match="budget"):
        core.add_candidate(exp, "fourth", "spectral decomposition", "different")


def test_candidate_paths_isolated_and_patch_policy():
    assert runner.allowed_path("retouch/engine.py")
    assert runner.allowed_path("presets/cosplay.yaml")
    for path in ("quality_lab/thresholds.json", "tests/test_engine.py", "../main.py",
                 "scripts/dev/release_train.py", "pyproject.toml"):
        assert not runner.allowed_path(path)


def test_cleanup_rejects_unmanaged_worktree(laboratory):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    cand = core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    cand["worktree"] = str(core.ROOT)
    exp["status"] = "ARCHIVED"
    exp["completed_at"] = "2020-01-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="outside managed"):
        core.cleanup(exp, retention_days=0, apply=True, discard_prototypes=True)


def test_budget_and_holdout_gate_before_execution(laboratory, monkeypatch):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    cand = core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    cand["status"] = "READY"
    cand["rounds"] = core.BUDGETS[exp["budget"]]["rounds"]
    monkeypatch.setattr(runner, "inspect_candidate", lambda *_: ({"loc_added": 1}, "fingerprint"))
    with pytest.raises(ValueError, match="round budget"):
        runner.run(exp, cand, "development")
    with pytest.raises(ValueError, match="holdout is sealed"):
        runner.run(exp, cand, "holdout")


def test_holdout_exposure_seals_refinement(laboratory, monkeypatch, tmp_path):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    cand = core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    _run_record(exp, cand, "holdout", tmp_path)
    with pytest.raises(ValueError, match="sealed"):
        core.add_candidate(exp, "fusion", "semantic boundary fusion", "new")
    with pytest.raises(ValueError, match="sealed"):
        runner.run(exp, cand, "development")


def test_real_detached_worktree_isolation(tmp_path, monkeypatch):
    # The real worktree fixture: git-ignored models/ must reach the candidate.
    root = tmp_path / "repo"
    (root / "retouch").mkdir(parents=True)
    (root / "quality_lab").mkdir()
    (root / "retouch" / "example.py").write_text("VALUE = 1\n")
    (root / "quality_lab" / "thresholds.json").write_text("{}\n")
    (root / "corpus.json").write_text("{}\n")
    (root / "baseline.json").write_text("{}\n")
    (root / ".gitignore").write_text("test_output/\nexperiments/\n/models/*\n!/models/manifest.json\n")
    (root / "models").mkdir()
    (root / "models" / "manifest.json").write_text("{}\n")
    (root / "models" / "engine.onnx").write_bytes(b"\x00MODEL")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Experiment Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "base"], cwd=root, check=True)
    base = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    monkeypatch.setattr(core, "ROOT", root)
    monkeypatch.setattr(core, "REGISTRY", root / "experiments")
    monkeypatch.setattr(core, "WORK_ROOT", root / "test_output" / "experiment_lab")
    monkeypatch.setattr(core, "THRESHOLDS", root / "quality_lab" / "thresholds.json")
    exp = {"schema": 1, "experiment_id": "EXP-0001", "status": "DESIGNED", "budget": "SMALL",
           "baseline": {"commit": base, "report": str(root / "baseline.json"),
                        "sha256": core.sha256(root / "baseline.json"),
                        "thresholds_sha256": core.sha256(core.THRESHOLDS)},
           "corpus": {"manifest": str(root / "corpus.json"), "sha256": core.sha256(root / "corpus.json")},
           "success_criteria": {}, "failure_criteria": {}, "constraints": {"max_loc_added": 5,
           "max_changed_files": 1}, "reject_all_if": "failure", "candidates": [],
           "started_at": "", "history": []}
    exp["design_sha256"] = core.design_fingerprint(exp)
    cand = {"id": "A", "worktree": "", "patch": "", "status": "DESIGNED", "runs": [],
            "rounds": 0, "complexity": {}}
    exp["candidates"].append(cand)
    prepared = runner.prepare(exp, cand)
    worktree = Path(prepared["worktree"])
    try:
        assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=worktree, text=True).strip() == base
        # Regression: git worktrees do not carry git-ignored engine assets.
        # prepare() must share the root models/ dir or every candidate
        # silently benchmarks the engine's no-model fallback paths.
        wt_models = worktree / "models" / "engine.onnx"
        assert wt_models.exists(), "candidate worktree is missing git-ignored models/"
        assert (worktree / "models" / "engine.onnx").read_bytes() == b"\x00MODEL"
        assert (root / "retouch" / "example.py").read_text() == "VALUE = 1\n"
        (worktree / "retouch" / "example.py").write_text("VALUE = 2\n")
        complexity, fingerprint = runner.inspect_candidate(exp, cand)
        assert complexity["paths"] == ["retouch/example.py"] and fingerprint
        assert (root / "retouch" / "example.py").read_text() == "VALUE = 1\n"
        (worktree / "quality_lab" / "thresholds.json").write_text("{\"warn\": 999}\n")
        with pytest.raises(ValueError, match="forbidden"):
            runner.inspect_candidate(exp, cand)
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", str(worktree)], cwd=root, check=True)


def test_target_aggregation_and_early_elimination(laboratory):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    split = "validation"
    base = core.read_json(spec["baseline_report"])
    report = deepcopy(base)
    for case in report["cases"].values():
        case["recipes"]["natural"]["metrics"]["global"]["edge_retention"] = 0.72
    result = runner._paired_target(exp, report, base, split)
    assert result["sample_size"] == len(exp["corpus"]["cases"][split])
    assert result["median_delta"] == 0.12
    failures = runner.early_stop(exp, {"quality": {"PASS": 0, "REVIEW": 0, "REGRESSION": 4},
                                       "max_time_ratio": 2.0, "max_ram_ratio": 1.0})
    assert failures == ["visual regression", "runtime"]


def test_missing_performance_is_inconclusive_not_no_winner(laboratory, tmp_path):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    cand = core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    dev = _run_record(exp, cand, "development", tmp_path)
    validation = _run_record(exp, cand, "validation", tmp_path)
    dev["max_time_ratio"] = None
    validation["max_time_ratio"] = None
    exp["status"] = "ANALYZING"
    result = core.compare(exp)
    assert result["conclusion"] == "INCONCLUSIVE"
    assert result["negative_findings"] == []


def test_pareto_and_qualified_winner(laboratory, tmp_path):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    a = core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    b = core.add_candidate(exp, "fusion", "segmentation boundary fusion", "semantic edge")
    for cand, delta, runtime in ((a, 0.12, 1.02), (b, 0.08, 1.08)):
        for split in ("development", "validation", "holdout"):
            _run_record(exp, cand, split, tmp_path, delta=delta, runtime=runtime)
    exp["status"] = "ANALYZING"
    result = core.compare(exp)
    assert result["conclusion"] == "WINNER" and result["winner"] == "A"
    assert result["pareto_frontier"] == ["A"]
    assert result["confidence"] == "MEDIUM"
    assert result["negative_findings"][0]["candidate"] == "B"


def test_pareto_tradeoff_remains_inconclusive(laboratory, tmp_path):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    a = core.add_candidate(exp, "quality", "local edge gradient", "best detail")
    b = core.add_candidate(exp, "speed", "segmentation boundary fusion", "lower cost")
    for cand, delta, runtime in ((a, 0.2, 1.09), (b, 0.1, 1.01)):
        for split in ("development", "validation", "holdout"):
            _run_record(exp, cand, split, tmp_path, delta=delta, runtime=runtime)
    exp["status"] = "ANALYZING"
    result = core.compare(exp)
    assert result["conclusion"] == "INCONCLUSIVE" and result["winner"] is None
    assert result["pareto_frontier"] == ["A", "B"]


def test_synthetic_only_corpus_cannot_promote(laboratory, tmp_path):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    exp["corpus"]["real_holdout"] = False
    exp["design_sha256"] = core.design_fingerprint(exp)
    cand = core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    for split in ("development", "validation", "holdout"):
        _run_record(exp, cand, split, tmp_path)
    exp["status"] = "ANALYZING"
    assert core.compare(exp)["conclusion"] == "INCONCLUSIVE"
    with pytest.raises(ValueError, match="qualified winner"):
        core.promote(exp, tasks=[])


def test_attention_routes_tradeoff_and_architecture_review(laboratory):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    exp.update(conclusion="INCONCLUSIVE", pareto_frontier=["A", "B"])
    state = {"schema": 1, "alerts": [], "decisions": [], "policies": [], "overrides": [], "daily": []}
    assert core.sync_attention(exp, state, save_state=False)[0]["severity"] == "warning"
    exp["constraints"]["architecture_review_required"] = True
    alert = core.sync_attention(exp, state, save_state=False)[0]
    assert alert["severity"] == "error" and alert["blocking"] == 1


def test_no_winner_and_negative_memory(laboratory, tmp_path):
    exp, spec, rd_state = laboratory
    core.validate_design(exp, spec)
    cand = core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    _run_record(exp, cand, "development", tmp_path, regressions=1)
    cand["status"] = "ELIMINATED"
    cand["elimination_reason"] = "background damage"
    exp["status"] = "ANALYZING"
    result = core.compare(exp)
    assert result["conclusion"] == "NO_WINNER" and result["winner"] is None
    ids = core.sync_negatives(exp, rd_state, persist=False)
    assert len(ids) == 1 and rd_state["negative_results"][0]["retry_unless"]
    assert core.sync_negatives(exp, rd_state, persist=False) == []


def test_failed_development_does_not_wait_for_validation(laboratory, tmp_path):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    cand = core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    _run_record(exp, cand, "development", tmp_path, delta=0.0)
    exp["status"] = "ANALYZING"
    result = core.compare(exp)
    assert result["conclusion"] == "NO_WINNER"
    assert "target improvement below" in result["negative_findings"][0]["observed"]


def test_similarity_blocks_repeat_without_new_evidence(laboratory):
    exp, spec, rd_state = laboratory
    with pytest.raises(ValueError, match="DUPLICATE"):
        core.create("HYP-001", "Repeat", exp["question"], rd_state=rd_state)
    retry = core.create("HYP-001", "Retry", exp["question"], new_evidence="new real corpus",
                        rd_state=rd_state)
    assert retry["similarity"]["verdict"] == "DUPLICATE"


def test_promotion_creates_control_plane_task_without_code(laboratory, monkeypatch, tmp_path):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    cand = core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    for split in ("development", "validation", "holdout"):
        _run_record(exp, cand, split, tmp_path)
    exp["status"] = "ANALYZING"
    core.compare(exp)
    saved = []
    monkeypatch.setattr(core.cp.Task, "save", lambda task: saved.append(task))
    result = core.promote(exp, tasks=[])
    assert result["prototype_code_copied"] is False
    assert saved[0].type == "feature" and saved[0].status == "BACKLOG"
    assert exp["promotion_task"] == saved[0].id
    with pytest.raises(ValueError):
        core.promote(exp, tasks=[])


def test_blind_review_records_preference(laboratory, tmp_path):
    exp, spec, _ = laboratory
    core.validate_design(exp, spec)
    cand = core.add_candidate(exp, "gradient", "local edge gradient", "local structure")
    run = _run_record(exp, cand, "validation", tmp_path)
    base = core.read_json(spec["baseline_report"])
    candidate_report = deepcopy(base)
    for case_id in exp["corpus"]["cases"]["validation"]:
        out = tmp_path / f"{case_id}-candidate.png"
        out.write_bytes(b"candidate")
        candidate_report["cases"][case_id]["recipes"]["natural"]["output"] = str(out)
    core.write_json(run["report"], candidate_report)
    run["report_sha256"] = core.sha256(run["report"])
    sheet = core.blind_sheet(exp, "validation")
    item = core.read_json(sheet["sheet"])["items"][0]
    label = next(iter(item["images"]))
    result = core.record_review(exp, "validation", [{"case": item["case"],
        "recipe": item["recipe"], "preferred": label}])
    assert result["recorded"] == 1 and len(exp["human_reviews"]) == 1


def test_json_status_and_report_stable(laboratory, monkeypatch, capsys):
    exp, _, _ = laboratory
    monkeypatch.setattr(cli.core, "list_all", lambda: [exp])
    assert cli.main(["status", "--json"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["schema"] == 1 and status["experiments"] == 1
    assert cli.main(["list", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["experiments"][0]["experiment_id"] == exp["experiment_id"]
    assert core.report(exp)["schema"] == 1
