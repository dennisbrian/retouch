"""Tests for BatchProcessor async producer-consumer queue (BB7)."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

from retouch.batch_processor import (
    BatchProcessor,
    _compute_queue_depth,
    _estimate_image_working_bytes,
    _run_async_batch_queue,
)


def test_estimate_image_working_bytes(tmp_path):
    img_path = tmp_path / "sample.jpg"
    Image.new("RGB", (1920, 1080), color=(128, 128, 128)).save(img_path)
    est = _estimate_image_working_bytes(img_path)
    assert est == 1920 * 1080 * 3 * 4


def test_compute_queue_depth_scales_with_resolution(tmp_path):
    small = tmp_path / "small.jpg"
    Image.new("RGB", (640, 480), color=(0, 0, 0)).save(small)
    depth = _compute_queue_depth(4, [small], ram_budget_bytes=50 * 1024 * 1024)
    assert 1 <= depth <= 8


def test_run_async_batch_queue_processes_all_files(tmp_path):
    paths = []
    for i in range(6):
        p = tmp_path / f"img_{i}.jpg"
        Image.new("RGB", (32, 32), color=(i, i, i)).save(p)
        paths.append(p)

    def worker(fp: Path) -> Path:
        return fp.with_name(fp.stem + "_out.jpg")

    results = _run_async_batch_queue(
        paths,
        worker,
        num_workers=3,
        total_files=len(paths),
    )
    assert len(results) == 6
    assert all(r.name.endswith("_out.jpg") for r in results)


def test_run_async_batch_queue_skips_failed_workers(tmp_path):
    paths = [tmp_path / "ok.jpg", tmp_path / "fail.jpg"]
    Image.new("RGB", (16, 16)).save(paths[0])
    Image.new("RGB", (16, 16)).save(paths[1])

    def worker(fp: Path):
        if fp.name == "fail.jpg":
            return None
        return fp

    results = _run_async_batch_queue(
        paths, worker, num_workers=2, total_files=len(paths),
    )
    assert results == [paths[0]]


def test_batch_processor_uses_async_queue(monkeypatch, tmp_path):
    engine = MagicMock()
    processor = BatchProcessor(engine)

    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    Image.new("RGB", (64, 64)).save(in_dir / "a.jpg")
    Image.new("RGB", (64, 64)).save(in_dir / "b.jpg")

    called = {"async": False}

    def fake_queue(paths, worker_fn, num_workers, total_files, progress_callback=None):
        called["async"] = True
        return [str(out_dir / "a_retouched.jpg"), str(out_dir / "b_retouched.jpg")]

    monkeypatch.setattr(
        "retouch.batch_processor.analyze_and_group",
        lambda paths, cache, eng: {"Group": paths},
    )
    monkeypatch.setattr(
        "retouch.batch_processor._run_async_batch_queue", fake_queue,
    )
    monkeypatch.setattr(
        processor, "_process_single_file",
        lambda *args, **kwargs: out_dir / "stub.jpg",
    )

    processor.process_folder(
        in_dir, out_dir, generate_sheet=False, num_workers=2,
    )
    assert called["async"] is True


class _FakeResult(np.ndarray):
    """Minimal ndarray subclass mirroring ProcessingResult's .qa attribute."""


def _fake_result(shape=(64, 64, 3), qa=None):
    arr = np.zeros(shape, dtype=np.uint8)
    r = arr.view(_FakeResult)
    r.qa = qa if qa is not None else []
    return r


def _make_input_dir(tmp_path, names=("a.jpg", "b.jpg")):
    in_dir = tmp_path / "in"
    out_dir = tmp_path / "out"
    in_dir.mkdir()
    out_dir.mkdir()
    for name in names:
        Image.new("RGB", (64, 64)).save(in_dir / name)
    return in_dir, out_dir


def test_only_files_restricts_processing(monkeypatch, tmp_path):
    in_dir, out_dir = _make_input_dir(tmp_path, names=("a.jpg", "b.jpg", "c.jpg"))
    engine = MagicMock()
    engine.process.return_value = _fake_result()
    processor = BatchProcessor(engine)

    monkeypatch.setattr(
        "retouch.batch_processor.analyze_and_group",
        lambda paths, cache, eng: {"Group": paths},
    )

    seen = []
    monkeypatch.setattr(
        processor, "_process_single_file",
        lambda fp, *args, **kwargs: (seen.append(fp), out_dir / f"{fp.stem}_out.jpg")[1],
    )

    processor.process_folder(
        in_dir, out_dir, generate_sheet=False, num_workers=1,
        only_files=[in_dir / "a.jpg", in_dir / "c.jpg"],
    )

    assert {p.name for p in seen} == {"a.jpg", "c.jpg"}


def test_on_file_result_fires_for_success_and_failure(monkeypatch, tmp_path):
    in_dir, out_dir = _make_input_dir(tmp_path, names=("good.jpg", "bad.jpg"))
    engine = MagicMock()

    def fake_process(img_bgr, recipe=None, **kwargs):
        return _fake_result()

    engine.process.side_effect = fake_process
    processor = BatchProcessor(engine)

    monkeypatch.setattr(
        "retouch.batch_processor.analyze_and_group",
        lambda paths, cache, eng: {"Group": paths},
    )

    def fake_imread(path):
        if "bad" in str(path):
            raise ValueError("corrupt file")
        return np.zeros((64, 64, 3), dtype=np.uint8)

    monkeypatch.setattr("retouch.batch_processor.imread_exif", fake_imread)
    monkeypatch.setattr(
        "retouch.io.write_image_with_icc", lambda *args, **kwargs: None,
    )

    results = {}

    def on_file_result(source_path, output_path, qa_list, error):
        results[source_path.name] = (output_path, qa_list, error)

    processor.process_folder(
        in_dir, out_dir, generate_sheet=False, num_workers=1,
        on_file_result=on_file_result,
    )

    assert results["good.jpg"][0] is not None
    assert results["good.jpg"][2] is None
    assert results["bad.jpg"][0] is None
    assert results["bad.jpg"][2] is not None


def test_on_file_result_receives_qa_before_export_resize(monkeypatch, tmp_path):
    """Regression: cv2.resize on the export path drops the ProcessingResult
    subclass and its .qa attribute. QA must be captured before that resize,
    not after — this test forces the resize branch and asserts QA survives.
    """
    in_dir, out_dir = _make_input_dir(tmp_path, names=("big.jpg",))
    engine = MagicMock()

    fake_qa = [{"detector": "halo", "score": 0.9, "flagged": True, "message": "halo"}]
    big_result = _fake_result(shape=(2000, 2000, 3), qa=fake_qa)
    engine.process.return_value = big_result

    processor = BatchProcessor(engine)

    monkeypatch.setattr(
        "retouch.batch_processor.analyze_and_group",
        lambda paths, cache, eng: {"Group": paths},
    )
    monkeypatch.setattr(
        "retouch.batch_processor.imread_exif",
        lambda path: np.zeros((2000, 2000, 3), dtype=np.uint8),
    )
    monkeypatch.setattr(
        "retouch.io.write_image_with_icc", lambda *args, **kwargs: None,
    )

    captured = {}

    def on_file_result(source_path, output_path, qa_list, error):
        captured["qa_list"] = qa_list

    processor.process_folder(
        in_dir, out_dir, generate_sheet=False, num_workers=1,
        export_res="720px",  # forces the cv2.resize branch (2000px > 720px)
        on_file_result=on_file_result,
    )

    assert captured["qa_list"] == fake_qa
