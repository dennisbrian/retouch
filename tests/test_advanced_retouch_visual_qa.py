"""Tests for the reproducible Advanced Retouch visual-QA harness."""

import json
from types import SimpleNamespace

import cv2
import numpy as np

import scripts.qa.advanced_retouch_visual_qa as visual_qa
from scripts.qa.advanced_retouch_visual_qa import _editor, _face_brush_mask, main


def test_face_brush_mask_is_limited_to_selected_bbox():
    faces = [
        SimpleNamespace(bbox=(4, 4, 16, 16)),
        SimpleNamespace(bbox=(40, 4, 16, 16)),
    ]
    mask = _face_brush_mask((32, 64), faces, selection=1)
    assert mask[12, 48] > 0.5
    assert mask[12, 12] == 0.0


def test_global_only_visual_qa_writes_non_certifying_manifest(tmp_path):
    image = np.full((48, 64, 3), 120, dtype=np.uint8)
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), image)
    output = tmp_path / "evidence"
    args = SimpleNamespace(
        input=str(source), output=str(output), max_dim=64, global_only=True,
    )

    assert main(args) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["global_only"] is True
    assert manifest["face_aware_certification"] is False
    assert manifest["passed"] is True
    assert (output / "contact_sheet.jpg").is_file()


def test_native_worker_abort_becomes_blocked_manifest(tmp_path, monkeypatch):
    source = tmp_path / "source.jpg"
    source.write_bytes(b"placeholder")
    output = tmp_path / "blocked"
    monkeypatch.setattr(
        visual_qa.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=-6),
    )
    args = SimpleNamespace(
        input=str(source), output=str(output), max_dim=64,
        global_only=False, _worker=False,
    )

    assert main(args) == 2
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked"
    assert manifest["face_aware_certification"] is False
    assert "exit code -6" in manifest["error"]
