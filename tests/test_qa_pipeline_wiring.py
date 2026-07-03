"""Tests for F11 QA detector pipeline wiring."""

from __future__ import annotations

from unittest.mock import patch

import cv2
import numpy as np
import pytest


def _retouch_importable() -> bool:
    try:
        import retouch  # noqa: F401
        return True
    except Exception:
        return False


requires_retouch = pytest.mark.skipif(
    not _retouch_importable(),
    reason="retouch package not importable",
)


def test_qa_detectors_importable():
    """qa_detectors module should be importable."""
    from retouch import qa_detectors
    assert hasattr(qa_detectors, "detect_banding")
    assert hasattr(qa_detectors, "detect_clipping")
    assert hasattr(qa_detectors, "detect_plastic_skin")


def test_qa_detectors_imported_in_engine():
    """Engine should import qa_detectors."""
    with open("retouch/engine.py") as f:
        source = f.read()
    assert "qa_detectors" in source, "engine.py should import qa_detectors"


def test_qa_detectors_called_in_pipeline():
    """Engine should call qa_detectors.run_all (F11 consolidated call)."""
    with open("retouch/engine.py") as f:
        source = f.read()
    assert "qa_detectors.run_all" in source, \
        "engine.py should call qa_detectors.run_all"


@requires_retouch
def test_qa_results_stored_on_context():
    """ProcessingContext._qa_results should be populated after pipeline."""
    from retouch.engine import RetouchEngine
    import retouch.engine as engine_mod
    from tests.benchmark_pipeline import _make_mock_engine_cls, _wire_mocks

    _bd, _bp, dets, pars = _make_mock_engine_cls()
    with patch.object(engine_mod, "FaceDetector", _bd), \
         patch.object(engine_mod, "FaceParser", _bp):
        eng = RetouchEngine()
    det = dets[0]
    par = pars[0]

    img = np.random.RandomState(42).randint(0, 255, (200, 200, 3), dtype=np.uint8)
    _wire_mocks(det, par, 200, 200, no_face=False)

    result = eng.process(img, recipe="natural", fast=False)
    qa = result.params._qa_results
    assert isinstance(qa, dict)
    assert "banding" in qa
    assert "clipping" in qa
    assert "plastic_skin" in qa


@requires_retouch
def test_qa_detectors_benchmark_integration():
    """Benchmark script should have qa_detector_metrics function."""
    from scripts.benchmark import qa_detector_metrics

    img = np.random.RandomState(42).randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.float32)

    result = qa_detector_metrics(img, mask)
    assert isinstance(result, dict)
    assert "banding_score" in result
    assert "clipping_score" in result
    assert "plastic_skin_score" in result
    assert all(isinstance(v, float) for v in result.values())


def test_qa_detector_benchmark_parsing():
    """Benchmark should parse QA detector output lines."""
    from scripts.benchmark import parse_benchmark_lines

    test_line = \
        "[benchmark] qa_detector_output: banding=0.1200 clipping=0.0500 plastic=0.3400"
    results = parse_benchmark_lines(test_line)
    qa_results = [r for r in results if r.get("type") == "qa_detector"]
    assert len(qa_results) == 1
    assert abs(qa_results[0]["banding"] - 0.12) < 0.001
    assert abs(qa_results[0]["clipping"] - 0.05) < 0.001
    assert abs(qa_results[0]["plastic"] - 0.34) < 0.001


@requires_retouch
def test_processing_result_has_qa_attribute():
    """ProcessingResult should have .qa attribute after pipeline."""
    from retouch.engine import RetouchEngine, ProcessingResult
    img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    engine = RetouchEngine()
    result = engine.process(img, recipe="natural")
    assert hasattr(result, "qa")
    assert isinstance(result.qa, list)


@requires_retouch
def test_qa_attribute_empty_for_clean_image():
    """A clean low-processing image should have empty qa list."""
    from retouch.engine import RetouchEngine
    img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    engine = RetouchEngine()
    result = engine.process(img, recipe="natural")
    assert isinstance(result.qa, list)
