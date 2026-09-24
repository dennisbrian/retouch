"""Focused tests for deterministic CLI input selection and resume evidence."""

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image
import pytest

from retouch.cli_input import (
    InputPlan,
    apply_resume_plan,
    build_input_plan,
    inspect_plan_headers,
    load_input_list,
    sha256_file,
)


def _write_png(path: Path, size=(8, 6)) -> None:
    Image.new("RGB", size, (80, 100, 120)).save(path)


def test_multiple_roots_are_sorted_and_hidden_files_are_reported(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    (first / "nested").mkdir(parents=True)
    second.mkdir()
    _write_png(first / "nested" / "B.PNG")
    _write_png(first / "a.jpg")
    _write_png(first / ".hidden.jpg")
    _write_png(second / "c.webp")

    plan = build_input_plan([str(first), str(second)], recursive=True)

    assert [path.name for path in plan.selected_paths] == ["a.jpg", "B.PNG", "c.webp"]
    assert any(row.status == "excluded_hidden" for row in plan.rows)


def test_include_and_exclude_filters_are_visible(tmp_path):
    root = tmp_path / "photos"
    root.mkdir()
    _write_png(root / "keep.jpg")
    _write_png(root / "skip.jpg")

    plan = build_input_plan(
        [str(root)],
        include=["*.jpg"],
        exclude=["skip.jpg"],
    )

    assert [path.name for path in plan.selected_paths] == ["keep.jpg"]
    assert any(row.status == "excluded_exclude_filter" for row in plan.rows)


def test_hardlink_duplicate_is_reported_without_double_selection(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    _write_png(first)
    try:
        second.hardlink_to(first)
    except (AttributeError, NotImplementedError, OSError):
        return

    plan = build_input_plan([str(first), str(second)])

    assert len(plan.selected_paths) == 1
    assert any(row.status == "duplicate_reference" for row in plan.rows)


def test_raw_jpeg_suffix_policy_avoids_pair_collision(tmp_path):
    root = tmp_path / "pair"
    root.mkdir()
    (root / "IMG_0001.RAF").write_bytes(b"raw")
    _write_png(root / "IMG_0001.JPG")

    plan = build_input_plan([str(root)], raw_jpeg_policy="suffix")

    assert {row.output_stem for row in plan.selected_records} == {
        "IMG_0001_raw",
        "IMG_0001_jpeg",
    }
    assert not plan.blocking_issues


def test_header_check_records_dimensions_and_format(tmp_path):
    path = tmp_path / "portrait.png"
    _write_png(path, size=(13, 9))
    plan = build_input_plan([str(path)])

    inspect_plan_headers(plan)

    row = plan.selected_records[0]
    assert row.header_status == "passed"
    assert (row.width, row.height, row.detected_format) == (13, 9, "PNG")
    assert row.decoder_requested == "pillow"
    assert row.planned_output_format is None
    assert row.estimated_working_bytes >= 13 * 9 * 3 * 4


def test_header_pixel_cap_is_a_blocking_observation(tmp_path):
    path = tmp_path / "large.png"
    _write_png(path, size=(13, 9))
    plan = build_input_plan([str(path)])

    inspect_plan_headers(plan, max_pixels=10)

    assert plan.selected_records[0].header_status == "failed"
    assert any(issue["code"] == "header_check_failed" for issue in plan.issues)


def test_multi_frame_inputs_require_an_explicit_first_frame_policy(tmp_path):
    path = tmp_path / "stack.tif"
    Image.new("RGB", (4, 4), (10, 10, 10)).save(
        path,
        save_all=True,
        append_images=[Image.new("RGB", (4, 4), (20, 20, 20))],
    )

    rejected = build_input_plan([str(path)])
    inspect_plan_headers(rejected)
    assert rejected.selected_records[0].header_status == "failed"

    allowed = build_input_plan([str(path)])
    inspect_plan_headers(allowed, multi_frame_policy="first")
    assert allowed.selected_records[0].header_status == "passed"
    assert any(issue["code"] == "multi_frame_first" for issue in allowed.issues)


def test_input_list_resolves_relative_paths_without_shell_expansion(tmp_path):
    photos = tmp_path / "photos"
    photos.mkdir()
    _write_png(photos / "A 01.png")
    listing = tmp_path / "selection.json"
    listing.write_text(
        json.dumps({"schema_version": 1, "base_dir": "photos", "paths": ["A 01.png"]}),
        encoding="utf-8",
    )

    assert load_input_list(listing) == [str(photos / "A 01.png")]


def test_resume_requires_matching_source_and_output_hashes(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    previous = InputPlan([str(source)], recursive=False)
    row = build_input_plan([str(source)]).selected_records[0]
    row.planned_output = str(output)
    row.result_status = "done"
    row.planned_artifacts = [str(output)]
    row.source_sha256 = sha256_file(source)
    row.output_sha256 = sha256_file(output)
    row.artifact_sha256 = {str(output): sha256_file(output)}
    previous.rows = [row]
    previous.config_fingerprint = "config"
    previous_path = tmp_path / "previous.json"
    previous.write(previous_path)

    current = build_input_plan([str(source)])
    current.config_fingerprint = "config"
    current_row = current.selected_records[0]
    current_row.planned_output = str(output)
    current_row.planned_artifacts = [str(output)]
    apply_resume_plan(current, previous_path, "config")

    assert current_row.status == "resume_verified"
    assert current.execution_paths == []


def _write_done_plan(tmp_path, source, output, fingerprint="config"):
    previous = InputPlan([str(source)], recursive=False)
    row = build_input_plan([str(source)]).selected_records[0]
    row.planned_output = str(output)
    row.planned_artifacts = [str(output)]
    row.result_status = "done"
    row.source_sha256 = sha256_file(source)
    row.output_sha256 = sha256_file(output)
    row.artifact_sha256 = {str(output): sha256_file(output)}
    previous.rows = [row]
    previous.config_fingerprint = fingerprint
    previous_path = tmp_path / "previous.json"
    previous.write(previous_path)
    return previous_path


def _valid_processing_evidence():
    return {
        "schema": "retouch.cli_processing_evidence",
        "version": 1,
        "capture_status": "captured",
        "execution_mode": "global_only",
        "face_detection": {"mode": "not_run_global_only"},
        "face_count": None,
        "face_aware_execution": False,
        "qa": {"status": "not_applicable_global_only", "detectors": {}},
        "runtime_diagnostics": {},
        "safe_auto_decisions": [],
        "fa02_diagnostics": [],
        "timings_ms": {},
        "precision": {},
        "color_context": {},
    }


def _valid_face_aware_processing_evidence():
    evidence = _valid_processing_evidence()
    evidence.update(
        execution_mode="engine",
        face_detection={"mode": "face_aware", "available": True},
        face_count=1,
        face_aware_execution=True,
        qa={
            "status": "captured",
            "detectors": {
                "plastic_skin": {
                    "status": "checked-pass", "score": 0.1, "flagged": False,
                }
            },
        },
    )
    return evidence


def _nested_value(depth):
    value = "CPU"
    for _ in range(depth):
        value = [value]
    return value


def _contradictory_global_only_evidence(evidence):
    evidence.update(face_count=2, face_aware_execution=True)
    evidence["face_detection"] = {"mode": "face_aware"}
    return evidence


def _current_plan(source, output):
    current = build_input_plan([str(source)])
    current.selected_records[0].planned_output = str(output)
    current.selected_records[0].planned_artifacts = [str(output)]
    return current


def test_resume_source_change_replaces_plan_owned_output_without_force(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source-v1")
    output.write_bytes(b"output-v1")
    previous_path = _write_done_plan(tmp_path, source, output)

    source.write_bytes(b"source-v2")
    current = _current_plan(source, output)
    apply_resume_plan(current, previous_path, "config")

    row = current.selected_records[0]
    assert current.execution_paths == [source]
    assert row.resume_reason == "source changed"
    # The existing output is byte-identical to the plan's own record, so the
    # re-render may replace it without --force.
    assert row.replace_prior_output is True
    assert not current.blocking_issues


def test_resume_carries_processing_evidence_only_for_verified_results(tmp_path):
    from types import SimpleNamespace

    import cli

    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source-v1")
    output.write_bytes(b"output-v1")
    previous_path = _write_done_plan(tmp_path, source, output)
    payload = json.loads(previous_path.read_text(encoding="utf-8"))
    processing_result = SimpleNamespace(
        face_count=1,
        runtime_diagnostics={
            "face_detection": {
                "mode": "face_aware", "available": True, "backend": "unit-test"
            },
            "denoise": {"backend": "bilateral", "providers": ["CPU"]},
            },
            qa_evidence={
            "plastic_skin": {"status": "checked-pass", "score": 0.2, "flagged": False}
        },
        fa02_diagnostics=[None],
        safe_auto_decisions=[],
        timings={"engine_total": 12.5},
        precision_metadata={},
    )
    payload["rows"][0]["processing_evidence"] = cli._processing_evidence(
        processing_result
    )
    previous_path.write_text(json.dumps(payload), encoding="utf-8")

    current = _current_plan(source, output)
    apply_resume_plan(current, previous_path, "config")
    assert current.selected_records[0].processing_evidence == payload["rows"][0]["processing_evidence"]

    source.write_bytes(b"source-v2")
    changed = _current_plan(source, output)
    apply_resume_plan(changed, previous_path, "config")
    assert changed.selected_records[0].resume_reason == "source changed"
    assert changed.selected_records[0].processing_evidence is None


@pytest.mark.parametrize(
    "mutate_evidence",
    [
        lambda evidence: evidence.update(schema="other.schema"),
        lambda evidence: evidence.update(version=2),
        lambda evidence: evidence.update(execution_mode=["unexpected"]),
        lambda evidence: evidence.update(private={"path": "/private/file"}),
        _contradictory_global_only_evidence,
        lambda evidence: evidence["runtime_diagnostics"].update(
            denoise={"providers": _nested_value(128)}
        ),
        lambda evidence: evidence["color_context"].update(
            source_profile_name="/private/user/profile.icc"
        ),
        lambda evidence: evidence["color_context"].update(
            source_profile_name="x" * 20_000
        ),
        lambda evidence: evidence["timings_ms"].update(total=float("nan")),
    ],
    ids=(
        "wrong-schema", "wrong-version", "wrong-type", "unknown-field",
        "contradictory-mode", "nested-list",
        "path-like", "oversized", "non-finite",
    ),
)
def test_verified_resume_drops_invalid_processing_evidence(tmp_path, mutate_evidence):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    previous_path = _write_done_plan(tmp_path, source, output)
    payload = json.loads(previous_path.read_text(encoding="utf-8"))
    evidence = _valid_processing_evidence()
    mutate_evidence(evidence)
    payload["rows"][0]["processing_evidence"] = evidence
    previous_path.write_text(json.dumps(payload), encoding="utf-8")

    current = _current_plan(source, output)
    apply_resume_plan(current, previous_path, "config")

    row = current.selected_records[0]
    assert row.status == "resume_verified"
    assert row.processing_evidence is None


@pytest.mark.parametrize(
    "mutate_evidence",
    [
        lambda evidence: evidence["face_detection"].update(available="false"),
        lambda evidence: evidence["qa"]["detectors"]["plastic_skin"].update(
            score="bad-score"
        ),
        lambda evidence: evidence["qa"]["detectors"]["plastic_skin"].update(
            flagged="no"
        ),
        lambda evidence: evidence["runtime_diagnostics"].update(
            denoise={"providers": ["CPUExecutionProvider", False]}
        ),
        lambda evidence: evidence["precision"].update(downgraded="false"),
        lambda evidence: evidence["qa"]["detectors"]["plastic_skin"].update(
            score=99.0
        ),
        lambda evidence: evidence["qa"]["detectors"]["plastic_skin"].update(
            status="invented-status"
        ),
        lambda evidence: evidence["qa"]["detectors"]["plastic_skin"].update(
            status="checked-flagged", flagged=True, available=False
        ),
        lambda evidence: evidence["qa"]["detectors"]["plastic_skin"].update(
            status="not-run", available=True
        ),
        lambda evidence: evidence["face_detection"].update(available=False),
        lambda evidence: evidence["timings_ms"].update(total=-1.0),
        lambda evidence: evidence.update(
            fa02_diagnostics=[{"face_width_px": -0.25}]
        ),
        lambda evidence: evidence["safe_auto_decisions"].append(
            {
                "stage": "test", "action": "apply", "confidence": 1.2,
                "strength_scale": 1.0, "reason": "out of range",
            }
        ),
        lambda evidence: evidence["safe_auto_decisions"].append(
            {"action": "apply"}
        ),
    ],
    ids=(
        "boolean-string", "numeric-string", "flag-string", "mixed-provider-list",
        "precision-bool", "qa-score-range", "qa-status", "qa-unavailable-contradiction",
        "qa-not-run-contradiction", "face-mode-availability", "negative-timing",
        "negative-face-width",
        "safe-auto-range", "safe-auto-incomplete",
    ),
)
def test_processing_evidence_rejects_field_type_mismatches(mutate_evidence):
    from retouch.cli_input import _normalise_processing_evidence

    evidence = _valid_face_aware_processing_evidence()
    assert _normalise_processing_evidence(evidence) is not None
    mutate_evidence(evidence)
    assert _normalise_processing_evidence(evidence) is None


def test_processing_evidence_preserves_detector_specific_score_scales():
    from retouch.cli_input import _normalise_processing_evidence

    evidence = _valid_face_aware_processing_evidence()
    evidence["qa"] = {
        "status": "captured",
        "detectors": {
            "halo": {"status": "checked-flagged", "score": 15.2, "flagged": True},
            "skin_score": {"status": "checked-pass", "score": 75.0, "flagged": False},
        },
    }

    assert _normalise_processing_evidence(evidence) is not None


def test_resume_plan_symlink_loop_is_reported_as_invalid(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    resume_loop = tmp_path / "resume-loop.json"
    try:
        resume_loop.symlink_to(resume_loop)
    except OSError as exc:
        pytest.skip("symlinks are unavailable in this test environment: %s" % exc)

    current = _current_plan(source, output)
    apply_resume_plan(current, resume_loop, "config")

    assert any(issue["code"] == "resume_plan_invalid" for issue in current.issues)
    assert current.selected_records[0].status == "selected"


def test_legacy_resume_plan_without_processing_evidence_still_verifies(tmp_path):
    source = tmp_path / "legacy-source.jpg"
    output = tmp_path / "legacy-output.jpg"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    previous_path = _write_done_plan(tmp_path, source, output)
    payload = json.loads(previous_path.read_text(encoding="utf-8"))
    payload["rows"][0].pop("processing_evidence", None)
    previous_path.write_text(json.dumps(payload), encoding="utf-8")

    current = _current_plan(source, output)
    apply_resume_plan(current, previous_path, "config")
    row = current.selected_records[0]
    assert row.status == "resume_verified"
    assert row.processing_evidence is None


def test_resume_rejects_non_array_row_ledger_without_crashing(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    previous_path = _write_done_plan(tmp_path, source, output)
    payload = json.loads(previous_path.read_text(encoding="utf-8"))
    payload["rows"] = 42
    previous_path.write_text(json.dumps(payload), encoding="utf-8")

    current = _current_plan(source, output)
    apply_resume_plan(current, previous_path, "config")

    assert current.execution_paths == [source]
    assert any(issue["code"] == "resume_plan_invalid" for issue in current.issues)


def test_resume_malformed_artifact_hashes_fail_closed(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    previous_path = _write_done_plan(tmp_path, source, output)
    payload = json.loads(previous_path.read_text(encoding="utf-8"))
    payload["rows"][0]["artifact_sha256"] = ["not-an-object"]
    previous_path.write_text(json.dumps(payload), encoding="utf-8")

    current = _current_plan(source, output)
    apply_resume_plan(current, previous_path, "config")

    row = current.selected_records[0]
    assert row.status != "resume_verified"
    assert current.execution_paths == [source]
    assert row.replace_prior_output is False
    assert any(issue["code"] == "resume_output_mismatch" for issue in current.issues)


def test_resume_symlink_loop_artifact_path_fails_closed(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source")
    output.write_bytes(b"output")
    loop = tmp_path / "artifact-loop"
    try:
        loop.symlink_to(loop)
    except OSError as exc:
        pytest.skip("symlinks are unavailable in this test environment: %s" % exc)
    previous_path = _write_done_plan(tmp_path, source, output)
    payload = json.loads(previous_path.read_text(encoding="utf-8"))
    payload["rows"][0]["artifact_sha256"][str(loop)] = "a" * 64
    previous_path.write_text(json.dumps(payload), encoding="utf-8")

    current = _current_plan(source, output)
    apply_resume_plan(current, previous_path, "config")

    row = current.selected_records[0]
    assert row.status != "resume_verified"
    assert current.execution_paths == [source]
    assert row.replace_prior_output is False
    assert any(issue["code"] == "resume_output_mismatch" for issue in current.issues)


def test_resume_guard_treats_dangling_output_symlink_as_existing(tmp_path):
    from retouch.cli_input import _guard_foreign_outputs

    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    external_target = tmp_path / "external-target.jpg"
    output_link = tmp_path / "output.jpg"
    try:
        output_link.symlink_to(external_target)
    except OSError as exc:
        pytest.skip("symlinks are unavailable in this test environment: %s" % exc)

    current = _current_plan(source, output_link)
    row = current.selected_records[0]
    row.resume_reason = "source changed"
    _guard_foreign_outputs(current, row, {}, force=False)

    assert not external_target.exists()
    assert not row.replace_prior_output
    assert any(issue["code"] == "resume_output_mismatch" for issue in current.issues)

    forced = _current_plan(source, output_link)
    forced_row = forced.selected_records[0]
    forced_row.resume_reason = "source changed"
    _guard_foreign_outputs(forced, forced_row, {}, force=True)
    assert forced_row.replace_prior_output


def test_cli_preflight_rejects_dangling_output_symlink_without_following_it(tmp_path):
    import cli

    source = tmp_path / "source.png"
    _write_png(source)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    external_target = tmp_path / "outside.png"
    output_link = output_dir / source.name
    try:
        output_link.symlink_to(external_target)
    except OSError as exc:
        pytest.skip("symlinks are unavailable in this test environment: %s" % exc)

    with pytest.raises(ValueError, match="symlink destination"):
        cli._preflight_destinations(
            [source], output_dir, "same", 8,
            recursive_root=None, compare=False, save_session=None,
        )

    assert output_link.is_symlink()
    assert not external_target.exists()


def test_resume_modified_output_blocks_without_force(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source-v1")
    output.write_bytes(b"output-v1")
    previous_path = _write_done_plan(tmp_path, source, output)

    output.write_bytes(b"hand-edited output")
    current = _current_plan(source, output)
    apply_resume_plan(current, previous_path, "config")

    row = current.selected_records[0]
    assert current.execution_paths == [source]
    assert row.resume_reason == "output modified"
    assert row.replace_prior_output is False
    assert any(issue["code"] == "resume_output_mismatch" for issue in current.blocking_issues)

    forced = _current_plan(source, output)
    apply_resume_plan(forced, previous_path, "config", force=True)
    assert not forced.blocking_issues
    assert forced.selected_records[0].replace_prior_output is True


def test_resume_settings_change_rerenders_instead_of_blocking(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source-v1")
    output.write_bytes(b"output-v1")
    previous_path = _write_done_plan(tmp_path, source, output, fingerprint="old")

    current = _current_plan(source, output)
    apply_resume_plan(current, previous_path, "new")

    row = current.selected_records[0]
    assert current.execution_paths == [source]
    assert row.resume_reason == "settings changed"
    assert row.replace_prior_output is True
    assert not current.blocking_issues
    assert any(issue["code"] == "resume_config_mismatch" for issue in current.issues)


def _run_cli(*args):
    cli_path = Path(__file__).resolve().parents[1] / "cli.py"
    return subprocess.run(
        [sys.executable, str(cli_path), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_processing_evidence_is_compact_json_safe_and_distinguishes_face_routes():
    from types import SimpleNamespace

    import cli
    import numpy as np
    from retouch.cli_input import _normalise_processing_evidence

    result = SimpleNamespace(
        face_count=np.int64(1),
        runtime_diagnostics={
            "face_detection": {
                "mode": "face_aware", "available": True,
                "backend": "test-detector", "model_path": "/private/model.tflite",
            },
            "denoise": {
                "backend": "bilateral", "fallback_reason": "model_unavailable",
                "private": "/private/private.png",
            },
            "healing": [{
                "index": 0, "status": "done", "executed": "telea_donor_constrained",
                "reason": "no_full_source_patch", "private": "/private/mask.png",
            }],
        },
        qa_evidence={
            "plastic_skin": {"status": "checked-pass", "score": np.float32(0.12), "flagged": False},
            "perceived_retouching": {"status": "not-run", "score": None, "flagged": False},
        },
        safe_auto_decisions=[{
            "stage": "optional_model.denoise", "action": "review",
            "confidence": np.float64(0.5), "reason": "fallback_requires_review",
            "evidence": {"private": "/private/detail.png"},
        }],
        fa02_diagnostics=[{
            "face_width_px": np.float64(128.75), "mode": "fa02", "eligible": True,
            "reason": "eligible", "ran": "restored", "roi_box": (1, 2, 3, 4),
        }],
        timings={"total": np.float64(2.5), "invalid": float("nan")},
        precision_metadata={"precision_status": "native_float32", "downgraded": False},
    )

    evidence = cli._processing_evidence(result)
    serialized = json.dumps(evidence, allow_nan=False)
    assert evidence["face_detection"]["mode"] == "face_aware"
    assert evidence["face_count"] == 1
    assert evidence["face_aware_execution"] is True
    assert evidence["qa"]["detectors"]["perceived_retouching"]["status"] == "not-run"
    assert evidence["runtime_diagnostics"]["healing"][0]["reason"] == "no_full_source_patch"
    assert evidence["safe_auto_decisions"][0]["action"] == "review"
    assert evidence["fa02_diagnostics"][0]["face_width_px"] == 128.75
    resumed_evidence = _valid_face_aware_processing_evidence()
    resumed_evidence["fa02_diagnostics"] = [{
        "face_width_px": evidence["fa02_diagnostics"][0]["face_width_px"],
        "mode": "fa02", "eligible": True, "reason": "eligible",
        "ran": "restored",
    }]
    resumed_evidence = _normalise_processing_evidence(resumed_evidence)
    assert resumed_evidence is not None
    assert resumed_evidence["fa02_diagnostics"][0]["face_width_px"] == 128.75
    assert evidence["timings_ms"] == {"total": 2.5}
    assert "/private" not in serialized
    assert "roi_box" not in serialized

    no_face = SimpleNamespace(
        face_count=0,
        runtime_diagnostics={"face_detection": {"mode": "face_aware", "available": True}},
        qa_evidence={"plastic_skin": {"status": "not-run", "score": None}},
    )
    evidence_no_face = cli._processing_evidence(no_face)
    global_only = cli._processing_evidence(np.zeros((2, 2, 3), dtype=np.uint8), global_only=True)
    assert evidence_no_face["face_count"] == 0
    assert evidence_no_face["face_aware_execution"] is False
    assert evidence_no_face["qa"]["detectors"]["plastic_skin"]["status"] == "not-run"
    assert global_only["execution_mode"] == "global_only"
    assert global_only["face_detection"]["mode"] == "not_run_global_only"
    assert global_only["face_aware_execution"] is False
    assert global_only["qa"]["status"] == "not_applicable_global_only"


def _resume_fixture(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    Image.new("RGB", (16, 12), (180, 120, 90)).save(src / "a.png")
    Image.new("RGB", (16, 12), (60, 140, 200)).save(src / "b.png")
    return src, tmp_path / "out"


def _cli_args(src, out, recipe, resume, plan, *extra, workers=1):
    return (
        src, "-o", out, "--global-only", "--no-compare", "--workers", workers,
        "--recipe", recipe, "--skip-disk-check", "--no-review",
        *(("--resume-plan", resume) if resume else ()),
        "--input-plan", plan, *extra,
    )


def test_cli_input_plan_writes_default_review_page(tmp_path):
    from retouch.review_page import load_review_records

    src, out = _resume_fixture(tmp_path)
    plan = tmp_path / "plan.json"
    result = _run_cli(
        src, "-o", out, "--global-only", "--no-compare",
        "--skip-disk-check", "--input-plan", plan,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (out / "review.html").is_file()
    assert {record.status for record in load_review_records(out)} == {"done"}
    payload = json.loads(plan.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row["selected"]]
    assert len(rows) == 2
    assert {row["result_status"] for row in rows} == {"done"}


def _hashes(out):
    return {path.name: sha256_file(path) for path in sorted(out.iterdir())}


@pytest.mark.parametrize("workers", [1, 2])
def test_cli_resume_unchanged_skips_every_row(tmp_path, workers):
    src, out = _resume_fixture(tmp_path)
    plan0, plan1 = tmp_path / "plan0.json", tmp_path / "plan1.json"
    first = _run_cli(*_cli_args(src, out, "natural", None, plan0, workers=workers))
    assert first.returncode == 0, first.stdout + first.stderr
    first_payload = json.loads(plan0.read_text(encoding="utf-8"))
    first_rows = [row for row in first_payload["rows"] if row["selected"]]
    assert len(first_rows) == 2
    assert all(
        row["processing_evidence"]["execution_mode"] == "global_only"
        and row["processing_evidence"]["face_detection"]["mode"] == "not_run_global_only"
        and row["processing_evidence"]["face_aware_execution"] is False
        and row["processing_evidence"]["qa"]["status"] == "not_applicable_global_only"
        for row in first_rows
    )
    before = _hashes(out)

    result = _run_cli(*_cli_args(src, out, "natural", plan0, plan1, workers=workers))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "All selected inputs are already verified" in result.stdout
    assert _hashes(out) == before
    resumed_payload = json.loads(plan1.read_text(encoding="utf-8"))
    assert [row["processing_evidence"] for row in resumed_payload["rows"] if row["selected"]] == [
        row["processing_evidence"] for row in first_rows
    ]
    # A ledger written by a verified resume must itself be resumable.
    plan2 = tmp_path / "plan2.json"
    again = _run_cli(*_cli_args(src, out, "natural", plan1, plan2, workers=workers))
    assert again.returncode == 0, again.stdout + again.stderr
    assert "All selected inputs are already verified" in again.stdout


@pytest.mark.parametrize("workers", [1, 2])
def test_cli_session_save_failure_has_same_failed_status_for_all_worker_modes(
    tmp_path, workers
):
    src = tmp_path / "single-input"
    src.mkdir()
    names = ("a", "b") if workers > 1 else ("a",)
    for name in names:
        _write_png(src / f"{name}.png")
    out = tmp_path / "out"
    out.mkdir()
    for name in names:
        (out / f"{name}.session.json").mkdir()
    plan = tmp_path / "plan.json"

    result = _run_cli(
        *_cli_args(
            src,
            out,
            "natural",
            None,
            plan,
            "--save-session",
            workers=workers,
        )
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert f"0 processed, 0 skipped, {len(names)} failed" in result.stdout
    assert all((out / f"{name}.png").is_file() for name in names)
    rows = [
        row
        for row in json.loads(plan.read_text(encoding="utf-8"))["rows"]
        if row["selected"]
    ]
    assert len(rows) == len(names)
    assert all(row["result_status"] == "failed" for row in rows)
    assert all(
        row["result_message"].startswith("saved_image_but_session_failed:")
        for row in rows
    )
    assert all(
        row["processing_evidence"]["execution_mode"] == "global_only"
        for row in rows
    )
    assert all(row["output_sha256"] is None for row in rows)
    assert all(row["artifact_sha256"] == {} for row in rows)


@pytest.mark.parametrize("workers", [1, 2])
def test_cli_export_failure_is_per_image_in_serial_and_pool_modes(tmp_path, workers):
    src = tmp_path / "input"
    src.mkdir()
    _write_png(src / "a.png", size=(16, 12))
    _write_png(src / "b.png", size=(16, 12))
    out = tmp_path / "out"
    out.mkdir()
    # Make delivery fail for one row while keeping the second row writable.
    (out / "a.png").mkdir()
    plan = tmp_path / "plan.json"

    result = _run_cli(
        *_cli_args(
            src, out, "natural", None, plan, "--force", workers=workers,
        )
    )

    assert result.returncode == 1, result.stdout + result.stderr
    assert "1 processed, 0 skipped, 1 failed" in result.stdout
    assert (out / "a.png").is_dir()
    assert (out / "b.png").is_file()
    rows = {
        Path(row["path"]).name: row
        for row in json.loads(plan.read_text(encoding="utf-8"))["rows"]
        if row["selected"]
    }
    assert rows["a.png"]["result_status"] == "failed"
    assert rows["a.png"]["result_message"].startswith("failed:")
    assert rows["b.png"]["result_status"] == "done"


@pytest.mark.parametrize("workers", [1, 2])
def test_cli_resume_settings_change_rerenders_all_rows(tmp_path, workers):
    src, out = _resume_fixture(tmp_path)
    plan0, plan1 = tmp_path / "plan0.json", tmp_path / "plan1.json"
    first = _run_cli(*_cli_args(src, out, "natural", None, plan0))
    assert first.returncode == 0, first.stdout + first.stderr
    before = _hashes(out)

    result = _run_cli(*_cli_args(src, out, "portrait", plan0, plan1, workers=workers))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Output preflight failed" not in result.stdout
    assert "2 processed, 0 skipped" in result.stdout
    after = _hashes(out)
    assert all(after[name] != before[name] for name in before)
    old_payload = json.loads(plan0.read_text(encoding="utf-8"))
    payload = json.loads(plan1.read_text(encoding="utf-8"))
    assert payload["run"]["config_fingerprint"] != old_payload["run"]["config_fingerprint"]
    rows = [row for row in payload["rows"] if row["selected"]]
    assert {row["result_status"] for row in rows} == {"done"}
    assert {row["resume_reason"] for row in rows} == {"settings changed"}
    assert {row["output_sha256"] for row in rows} == set(after.values())


@pytest.mark.parametrize("workers", [1, 2])
def test_cli_resume_source_change_rerenders_only_that_row(tmp_path, workers):
    src, out = _resume_fixture(tmp_path)
    plan0, plan1 = tmp_path / "plan0.json", tmp_path / "plan1.json"
    first = _run_cli(*_cli_args(src, out, "natural", None, plan0))
    assert first.returncode == 0, first.stdout + first.stderr
    before = _hashes(out)
    b_mtime = (out / "b.png").stat().st_mtime_ns

    Image.new("RGB", (16, 12), (20, 200, 40)).save(src / "a.png")
    result = _run_cli(*_cli_args(src, out, "natural", plan0, plan1, workers=workers))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 processed, 0 skipped" in result.stdout
    after = _hashes(out)
    assert after["a.png"] != before["a.png"]
    assert after["b.png"] == before["b.png"]
    assert (out / "b.png").stat().st_mtime_ns == b_mtime
    rows = {
        Path(row["path"]).name: row
        for row in json.loads(plan1.read_text(encoding="utf-8"))["rows"]
        if row["selected"]
    }
    assert rows["a.png"]["result_status"] == "done"
    assert rows["a.png"]["resume_reason"] == "source changed"
    assert rows["b.png"]["result_status"] == "resume_verified"


def test_cli_resume_dry_run_explains_skip_and_rerender(tmp_path):
    src, out = _resume_fixture(tmp_path)
    plan0, plan1 = tmp_path / "plan0.json", tmp_path / "plan1.json"
    first = _run_cli(*_cli_args(src, out, "natural", None, plan0))
    assert first.returncode == 0, first.stdout + first.stderr
    (out / "b.png").unlink()

    result = _run_cli(*_cli_args(src, out, "natural", plan0, plan1, "--dry-run"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "[verified resume: skip]" in result.stdout
    assert "[re-render: output missing]" in result.stdout


def test_cli_dry_run_accepts_multiple_paths_and_writes_plan(tmp_path):
    first = tmp_path / "A 01.png"
    second = tmp_path / "B 02.png"
    _write_png(first)
    _write_png(second)
    plan_path = tmp_path / "input-plan.json"
    output_dir = tmp_path / "retouched"
    cli_path = Path(__file__).resolve().parents[1] / "cli.py"

    result = subprocess.run(
        [
            sys.executable,
            str(cli_path),
            str(first),
            str(second),
            "--dry-run",
            "--input-plan",
            str(plan_path),
            "--output",
            str(output_dir),
            "--no-compare",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    assert payload["summary"]["selected"] == 2
    assert not output_dir.exists()
