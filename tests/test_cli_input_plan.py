"""Focused tests for deterministic CLI input selection and resume evidence."""

import json
from pathlib import Path
import subprocess
import sys

from PIL import Image

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


def test_resume_mismatch_blocks_stale_output_without_force(tmp_path):
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.jpg"
    source.write_bytes(b"source-v1")
    output.write_bytes(b"output-v1")
    previous = InputPlan([str(source)], recursive=False)
    row = build_input_plan([str(source)]).selected_records[0]
    row.planned_output = str(output)
    row.planned_artifacts = [str(output)]
    row.result_status = "done"
    row.source_sha256 = sha256_file(source)
    row.output_sha256 = sha256_file(output)
    row.artifact_sha256 = {str(output): sha256_file(output)}
    previous.rows = [row]
    previous.config_fingerprint = "config"
    previous_path = tmp_path / "previous.json"
    previous.write(previous_path)

    source.write_bytes(b"source-v2")
    current = build_input_plan([str(source)])
    current.selected_records[0].planned_output = str(output)
    current.selected_records[0].planned_artifacts = [str(output)]
    apply_resume_plan(current, previous_path, "config")

    assert current.execution_paths == [source]
    assert any(issue["code"] == "resume_output_mismatch" for issue in current.issues)


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
