"""Tests for the cli.py <-> retouch.review_page wiring.

Uses a monkeypatched RetouchEngine so the batch loop runs against synthetic
images without loading any real detection/segmentation models (per
CLAUDE.md: no model downloads in tests). ``--global-only``/``--workers 1``
with a single or a couple of tiny synthetic images keeps the serial loop in
``cli.main()`` in-process, so the monkeypatch actually takes effect (a
``ProcessPoolExecutor`` worker would re-import ``cli`` fresh and never see
it — the multi-file/``--workers>1`` pool path is therefore not exercised
here; see the report for that gap).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import cli as cli_module
from retouch.qa_detectors import QAWarning
from retouch.review_page import PAGE_NAME, load_review_records

CLI_PATH = Path(__file__).resolve().parent.parent / "cli.py"
REPO_ROOT = CLI_PATH.parent


def _write_image(path: Path, value: int = 120, size=(24, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    cv2.imwrite(str(path), np.full((h, w, 3), value, dtype=np.uint8))


def _extract_review_data(html: str) -> dict:
    match = re.search(
        r'<script id="review-data" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    assert match, "review-data JSON script block not found"
    return json.loads(match.group(1))


class _FakeResult(np.ndarray):
    """Minimal stand-in for retouch.engine.ProcessingResult: an ndarray
    carrying .qa / .face_contexts, so cli.py's image-writing code (which
    only cares that the result behaves like a BGR ndarray) keeps working.
    """

    def __new__(cls, image, qa=None, face_contexts=None):
        obj = np.asarray(image).view(cls)
        obj.qa = qa or []
        obj.face_contexts = face_contexts or []
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.qa = getattr(obj, "qa", [])
        self.face_contexts = getattr(obj, "face_contexts", [])


class _FakeEngine:
    """Stands in for RetouchEngine() in the serial (non-pool) batch loop."""

    instances = []
    process_calls = 0

    def __init__(self, *args, **kwargs):
        self.closed = False
        _FakeEngine.instances.append(self)

    def process(self, img, **kwargs):
        _FakeEngine.process_calls += 1
        fc = SimpleNamespace(face_data=SimpleNamespace(bbox=(1, 2, 3, 4)))
        qa = QAWarning(
            detector="halo", score=0.1, flagged=False, message="ok",
            threshold=0.5,
        )
        return _FakeResult(img, qa=[qa], face_contexts=[fc])

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def _reset_fake_engine():
    _FakeEngine.instances = []
    _FakeEngine.process_calls = 0
    yield
    _FakeEngine.instances = []
    _FakeEngine.process_calls = 0


def _run_cli_inprocess(monkeypatch, argv):
    monkeypatch.setattr(cli_module, "RetouchEngine", _FakeEngine)
    monkeypatch.setattr(sys, "argv", ["cli.py", *argv])
    cli_module.main()


class TestNoReviewFlagWiring:
    def test_review_page_built_by_default(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _write_image(input_dir / "a.jpg")
        output_dir = tmp_path / "out"

        _run_cli_inprocess(monkeypatch, [
            str(input_dir), "-o", str(output_dir),
            "--no-compare", "--force", "--workers", "1",
        ])

        assert (output_dir / PAGE_NAME).exists()

    def test_no_review_flag_skips_page(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _write_image(input_dir / "a.jpg")
        output_dir = tmp_path / "out"

        _run_cli_inprocess(monkeypatch, [
            str(input_dir), "-o", str(output_dir),
            "--no-compare", "--force", "--workers", "1", "--no-review",
        ])

        assert not (output_dir / PAGE_NAME).exists()

    def test_help_lists_no_review_flag(self):
        result = subprocess.run(
            [sys.executable, str(CLI_PATH), "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert "--no-review" in result.stdout

    def test_prints_review_page_path_on_success(self, tmp_path, monkeypatch, capsys):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _write_image(input_dir / "a.jpg")
        output_dir = tmp_path / "out"

        _run_cli_inprocess(monkeypatch, [
            str(input_dir), "-o", str(output_dir),
            "--no-compare", "--force", "--workers", "1",
        ])

        out = capsys.readouterr().out
        assert "review.html" in out


class TestReviewRecordWiring:
    def test_done_record_written_with_faces_and_qa(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _write_image(input_dir / "a.jpg")
        output_dir = tmp_path / "out"

        _run_cli_inprocess(monkeypatch, [
            str(input_dir), "-o", str(output_dir),
            "--no-compare", "--force", "--workers", "1",
        ])

        records = load_review_records(output_dir)
        assert len(records) == 1
        rec = records[0]
        assert rec.status == "done"
        assert rec.faces == [[1, 2, 3, 4]]
        assert len(rec.qa) == 1
        assert rec.qa[0]["detector"] == "halo"

    def test_multiple_files_each_get_a_record(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _write_image(input_dir / "a.jpg", value=50)
        _write_image(input_dir / "b.jpg", value=150)
        output_dir = tmp_path / "out"

        # --workers 1 keeps this on the serial loop (the monkeypatched
        # engine only takes effect in-process).
        _run_cli_inprocess(monkeypatch, [
            str(input_dir), "-o", str(output_dir),
            "--no-compare", "--force", "--workers", "1",
        ])

        records = load_review_records(output_dir)
        assert {r.relative for r in records} == {"a.jpg", "b.jpg"}
        assert all(r.status == "done" for r in records)

        data = _extract_review_data((output_dir / PAGE_NAME).read_text(encoding="utf-8"))
        assert data["summary"]["total"] == 2

    def test_skip_writes_qa_not_recorded_stub_when_no_record_exists(
        self, tmp_path, monkeypatch,
    ):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _write_image(input_dir / "a.jpg")
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        # Pre-existing output -> file will be skipped (no --force).
        _write_image(output_dir / "a.jpg", value=222)

        _run_cli_inprocess(monkeypatch, [
            str(input_dir), "-o", str(output_dir),
            "--no-compare", "--workers", "1",
        ])

        records = load_review_records(output_dir)
        assert len(records) == 1
        assert records[0].status == "done"
        assert records[0].qa_recorded is False
        assert records[0].qa == []
        # A skip must not actually reprocess the image.
        assert _FakeEngine.process_calls == 0

    def test_skip_does_not_clobber_an_existing_real_record(self, tmp_path, monkeypatch):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        _write_image(input_dir / "a.jpg")
        output_dir = tmp_path / "out"

        # First pass: real (fake-engine) processing writes a full record.
        _run_cli_inprocess(monkeypatch, [
            str(input_dir), "-o", str(output_dir),
            "--no-compare", "--force", "--workers", "1",
        ])
        first_records = load_review_records(output_dir)
        assert len(first_records) == 1
        assert first_records[0].qa_recorded is True
        assert first_records[0].qa

        # Second pass, no --force: output already exists -> skipped. The
        # existing (qa_recorded=True, non-empty qa) record must survive.
        _run_cli_inprocess(monkeypatch, [
            str(input_dir), "-o", str(output_dir),
            "--no-compare", "--workers", "1",
        ])
        second_records = load_review_records(output_dir)
        assert len(second_records) == 1
        assert second_records[0].qa_recorded is True
        assert second_records[0].qa == first_records[0].qa


class TestRunReviewWrapper:
    def test_run_review_build(self, tmp_path):
        run_script = REPO_ROOT / "run"
        if not run_script.exists():
            pytest.skip("./run script not present yet")

        from retouch.review_page import ReviewRecord, write_review_record

        root = tmp_path
        src = root / "a.jpg"
        _write_image(src)
        write_review_record(root, ReviewRecord(
            source=str(src), output=str(src), status="done", relative="a.jpg",
        ))

        result = subprocess.run(
            ["bash", str(run_script), "review", "build", str(root)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        assert (root / PAGE_NAME).exists()
