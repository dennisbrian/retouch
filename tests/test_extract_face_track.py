from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "review" / "extract_face_track.py"
SPEC = importlib.util.spec_from_file_location("extract_face_track", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
extract_face_track = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extract_face_track
SPEC.loader.exec_module(extract_face_track)


def test_select_primary_face_picks_largest_area() -> None:
    primary = extract_face_track.select_primary_face(
        [((0, 0, 10, 10), 0.9), ((5, 5, 40, 30), 0.6), ((2, 2, 20, 20), 0.99)]
    )
    assert primary == ((5, 5, 40, 30), 0.6)


def test_select_primary_face_empty_returns_none() -> None:
    assert extract_face_track.select_primary_face([]) is None


def test_scale_bbox_scales_and_clamps_to_frame() -> None:
    scaled = extract_face_track.scale_bbox((100, 50, 200, 100), 3.0, 1920, 1080)
    assert scaled == (300, 150, 600, 300)
    clamped = extract_face_track.scale_bbox((600, 300, 100, 80), 3.0, 1920, 1080)
    assert clamped == (1800, 900, 120, 180)
    assert clamped[0] + clamped[2] <= 1920
    assert clamped[1] + clamped[3] <= 1080


def test_build_payload_matches_harness_track_contract(tmp_path: Path) -> None:
    records = [{"frame": 0, "face": [4, 4, 20, 20], "confidence": 0.98}]
    payload = extract_face_track.build_payload(records, "clip.mov", 29.97, 1280)

    qa_path = Path(__file__).resolve().parents[1] / "scripts" / "review" / "video_qa_report.py"
    qa_spec = importlib.util.spec_from_file_location("video_qa_report_contract", qa_path)
    assert qa_spec is not None and qa_spec.loader is not None
    video_qa_report = importlib.util.module_from_spec(qa_spec)
    sys.modules[qa_spec.name] = video_qa_report
    qa_spec.loader.exec_module(video_qa_report)

    track_file = tmp_path / "tracks.json"
    track_file.write_text(json.dumps(payload), encoding="utf-8")
    loaded = video_qa_report.load_tracks(track_file)
    assert loaded[0].width == 20
    assert loaded[0].confidence == pytest.approx(0.98)
