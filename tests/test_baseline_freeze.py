from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "qa"))

import baseline_freeze  # noqa: E402


@dataclasses.dataclass
class _FakeWarning:
    detector: str
    score: float
    flagged: bool
    message: str
    threshold: float = 0.0
    details: dict = dataclasses.field(default_factory=dict)


def _fake_result(*, face_count=1, p7=None, fa02=None, qa=None):
    result = MagicMock()
    result.face_count = face_count
    result.qa = qa or []
    result.safe_auto_decisions = []
    result.p7_diagnostics = p7
    result.fa02_diagnostics = fa02
    result.qa_evidence = {}
    result.timings = {"total": 1.23}
    return result


def test_json_safe_converts_dataclass_warning():
    warning = _FakeWarning(detector="banding", score=0.9, flagged=True, message="x", threshold=0.15, details={"a": 1})
    safe = baseline_freeze._json_safe(warning)
    assert safe == {
        "detector": "banding",
        "score": 0.9,
        "flagged": True,
        "message": "x",
        "threshold": 0.15,
        "details": {"a": 1},
    }


def test_freeze_one_asset_flags_missing_ground_truth(tmp_path, monkeypatch):
    image_path = tmp_path / "img.jpg"
    import numpy as np
    import cv2

    cv2.imwrite(str(image_path), (np.ones((20, 20, 3)) * 128).astype("uint8"))

    engine = MagicMock()
    engine.process.return_value = _fake_result(p7={}, fa02=[None])

    asset = {"asset_id": "a1", "sha256": "deadbeef", "person_ids": ["p1"], "split": "dev", "path": "img.jpg"}
    row = baseline_freeze.freeze_one_asset(asset, root=tmp_path, recipe="natural", quality="full", engine=engine)

    assert row["status"] == "processed"
    assert row["ground_truth_available"] is False
    assert any("no reviewed per-region ground-truth" in u for u in row["unknowns"])
    assert any("p7_diagnostics empty" in u for u in row["unknowns"])
    assert any("fa02_diagnostics all None" in u for u in row["unknowns"])


def test_freeze_one_asset_handles_unreadable_image(tmp_path):
    engine = MagicMock()
    asset = {"asset_id": "a2", "sha256": "abc", "person_ids": ["p1"], "split": "dev", "path": "missing.jpg"}
    row = baseline_freeze.freeze_one_asset(asset, root=tmp_path, recipe="natural", quality="full", engine=engine)

    assert row["status"] == "image_unreadable"
    engine.process.assert_not_called()


def test_summarize_qa_signals_reports_flag_fraction():
    rows = [
        {"qa_evidence": {"banding": {"status": "checked-flagged", "flagged": True, "score": 0.9}}},
        {"qa_evidence": {"banding": {"status": "checked-flagged", "flagged": True, "score": 0.8}}},
        {"qa_evidence": {"banding": {"status": "checked-pass", "flagged": False, "score": 0.1}}},
    ]
    summary = baseline_freeze.summarize_qa_signals(rows)

    assert summary["banding"]["flagged_count"] == 2
    assert summary["banding"]["total_count"] == 3
    assert summary["banding"]["flagged_fraction"] == round(2 / 3, 3)


def test_summarize_qa_signals_excludes_not_run_and_unavailable():
    rows = [
        {"qa_evidence": {"banding": {"status": "checked-flagged", "flagged": True, "score": 0.9}}},
        {"qa_evidence": {"banding": {"status": "not-run", "flagged": False, "score": None}}},
        {"qa_evidence": {"banding": {"status": "unavailable", "flagged": False, "score": None}}},
    ]
    summary = baseline_freeze.summarize_qa_signals(rows)

    assert summary["banding"]["total_count"] == 1
    assert summary["banding"]["flagged_fraction"] == 1.0


def test_summarize_qa_signals_counts_unflagged_detector_as_not_flagged_not_absent():
    # A detector never flagged on any asset (real "not-run" is a different
    # status) must still show total_count == asset count with flagged_count
    # 0 -- not disappear from the summary entirely, which is the bug this
    # test guards: reading qa_signals (flagged-only) instead of qa_evidence
    # (complete map) would silently drop this detector.
    rows = [
        {"qa_evidence": {"asymmetry": {"status": "checked-pass", "flagged": False, "score": 0.1}}},
        {"qa_evidence": {"asymmetry": {"status": "checked-pass", "flagged": False, "score": 0.2}}},
    ]
    summary = baseline_freeze.summarize_qa_signals(rows)

    assert summary["asymmetry"]["total_count"] == 2
    assert summary["asymmetry"]["flagged_count"] == 0
    assert summary["asymmetry"]["flagged_fraction"] == 0.0


def test_freeze_one_asset_flags_zero_faces(tmp_path):
    import numpy as np
    import cv2

    image_path = tmp_path / "img.jpg"
    cv2.imwrite(str(image_path), (np.ones((20, 20, 3)) * 128).astype("uint8"))

    engine = MagicMock()
    engine.process.return_value = _fake_result(face_count=0, p7={"a": 1}, fa02=[{"b": 2}])

    asset = {"asset_id": "a3", "sha256": "abc", "person_ids": ["p1"], "split": "dev", "path": "img.jpg"}
    row = baseline_freeze.freeze_one_asset(asset, root=tmp_path, recipe="natural", quality="full", engine=engine)

    assert row["face_count"] == 0
    assert any("zero faces detected" in u for u in row["unknowns"])
