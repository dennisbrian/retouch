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
