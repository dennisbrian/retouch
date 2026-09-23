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


def _resume_fixture(tmp_path):
    src = tmp_path / "in"
    src.mkdir()
    Image.new("RGB", (16, 12), (180, 120, 90)).save(src / "a.png")
    Image.new("RGB", (16, 12), (60, 140, 200)).save(src / "b.png")
    return src, tmp_path / "out"


def _cli_args(src, out, recipe, resume, plan, *extra, workers=1):
    return (
        src, "-o", out, "--global-only", "--no-compare", "--workers", workers,
        "--recipe", recipe, "--skip-disk-check",
        *(("--resume-plan", resume) if resume else ()),
        "--input-plan", plan, *extra,
    )


def _hashes(out):
    return {path.name: sha256_file(path) for path in sorted(out.iterdir())}


def test_cli_resume_unchanged_skips_every_row(tmp_path):
    src, out = _resume_fixture(tmp_path)
    plan0, plan1 = tmp_path / "plan0.json", tmp_path / "plan1.json"
    first = _run_cli(*_cli_args(src, out, "natural", None, plan0))
    assert first.returncode == 0, first.stdout + first.stderr
    before = _hashes(out)

    result = _run_cli(*_cli_args(src, out, "natural", plan0, plan1))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "All selected inputs are already verified" in result.stdout
    assert _hashes(out) == before
    # A ledger written by a verified resume must itself be resumable.
    plan2 = tmp_path / "plan2.json"
    again = _run_cli(*_cli_args(src, out, "natural", plan1, plan2))
    assert again.returncode == 0, again.stdout + again.stderr
    assert "All selected inputs are already verified" in again.stdout


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
