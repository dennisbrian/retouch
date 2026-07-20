from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "review" / "video_qa_report.py"
SPEC = importlib.util.spec_from_file_location("video_qa_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
video_qa_report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = video_qa_report
SPEC.loader.exec_module(video_qa_report)


def _frames(effect_values: list[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    source = [np.full((32, 32, 3), 100, dtype=np.uint8) for _ in effect_values]
    result = [np.full((32, 32, 3), 100 + value, dtype=np.uint8) for value in effect_values]
    return source, result


def _tracks(count: int) -> dict[int, object]:
    return {index: video_qa_report.FaceTrack(index, 4, 4, 20, 20, 0.99) for index in range(count)}


def test_temporal_metrics_marks_constant_effect_as_stable() -> None:
    source, result = _frames([10, 10, 10])
    metrics = video_qa_report.temporal_metrics(source, result, _tracks(3))
    assert metrics["tracked_frames"] == 3
    assert metrics["mean_effect_energy"] == pytest.approx(10.0)
    assert metrics["mean_effect_flicker"] == pytest.approx(0.0)


def test_temporal_metrics_detects_effect_flicker() -> None:
    source, result = _frames([0, 20, 0])
    metrics = video_qa_report.temporal_metrics(source, result, _tracks(3))
    assert metrics["mean_effect_flicker"] == pytest.approx(20.0)
    assert metrics["p95_effect_flicker"] == pytest.approx(20.0)


def test_low_confidence_breaks_temporal_comparison() -> None:
    source, result = _frames([10, 20, 10])
    tracks = _tracks(3)
    tracks[1] = video_qa_report.FaceTrack(1, 4, 4, 20, 20, 0.50)
    metrics = video_qa_report.temporal_metrics(source, result, tracks, min_confidence=0.75)
    assert metrics["tracked_frames"] == 2
    assert metrics["mean_effect_flicker"] == pytest.approx(0.0)


def test_load_tracks_rejects_duplicate_frame(tmp_path: Path) -> None:
    track_path = tmp_path / "tracks.json"
    track_path.write_text(
        '{"frames": [{"frame": 0, "face": [0, 0, 10, 10], "confidence": 1.0}, '
        '{"frame": 0, "face": [0, 0, 10, 10], "confidence": 1.0}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate"):
        video_qa_report.load_tracks(track_path)
