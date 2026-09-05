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


def _mocked_engine_for(img_h: int, img_w: int, *, no_face: bool = True):
    """Build an engine without initializing native MediaPipe resources."""
    from retouch.engine import RetouchEngine
    import retouch.engine as engine_mod
    from tests.benchmark_pipeline import _make_mock_engine_cls, _wire_mocks

    build_detector, build_parser, detectors, parsers = _make_mock_engine_cls()
    with patch.object(engine_mod, "FaceDetector", build_detector), \
         patch.object(engine_mod, "FaceParser", build_parser):
        engine = RetouchEngine()
    _wire_mocks(detectors[0], parsers[0], img_h, img_w, no_face=no_face)
    return engine


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
    """Engine should delegate QA to qa_detectors.run_qa (F11, which calls run_all)."""
    with open("retouch/engine.py") as f:
        source = f.read()
    assert "qa_detectors.run_qa" in source, \
        "engine.py should call qa_detectors.run_qa"


def test_qa_pipeline_exception_is_a_warning(monkeypatch):
    from retouch.engine import RetouchEngine
    import retouch.engine as engine_mod

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic QA pipeline failure")

    monkeypatch.setattr(engine_mod.qa_detectors, "run_all", fail)
    warnings = RetouchEngine._run_qa(
        np.full((16, 16, 3), 128, dtype=np.uint8),
        np.ones((16, 16), dtype=np.float32),
    )

    assert len(warnings) == 1
    assert warnings[0].detector == "qa_pipeline"
    assert warnings[0].flagged is True
    assert warnings[0].details["available"] is False


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

    flagged = {
        name: {"score": 1.0, "flagged": True, "probe": name}
        for name in ("banding", "clipping", "plastic_skin")
    }
    with patch.object(engine_mod.qa_detectors, "run_all", return_value=flagged):
        result = eng.process(img, recipe="natural", fast=False)
    qa = result.params._qa_results
    assert isinstance(qa, dict)
    assert "banding" in qa
    assert "clipping" in qa
    assert "plastic_skin" in qa


@requires_retouch
def test_qa_detectors_benchmark_integration():
    """Benchmark script should have qa_detector_metrics function."""
    from scripts.bench.benchmark import qa_detector_metrics

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
    from scripts.bench.benchmark import parse_benchmark_lines

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
    img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    with _mocked_engine_for(100, 100) as engine:
        result = engine.process(img, recipe="natural")
    assert hasattr(result, "qa")
    assert isinstance(result.qa, list)


@requires_retouch
def test_qa_attribute_empty_for_clean_image():
    """A clean low-processing image should have empty qa list."""
    img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    with _mocked_engine_for(100, 100) as engine:
        result = engine.process(img, recipe="natural")
    assert isinstance(result.qa, list)


def test_run_qa_with_evidence_distinguishes_pass_from_flagged():
    """run_qa_with_evidence must report every detector, not just flagged ones."""
    from retouch import qa_detectors

    img = np.random.RandomState(0).randint(50, 200, (64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.float32)
    warnings, evidence = qa_detectors.run_qa_with_evidence(img, mask)

    assert set(qa_detectors.ALL_DETECTOR_NAMES) <= set(evidence.keys())
    # Every warning must correspond to an explicitly-flagged evidence entry;
    # the converse is not required (a detector can self-report "unavailable"
    # without flagging, e.g. evaluate_harmony on unmeasurable input, and
    # that must not silently gate export).
    for w in warnings:
        assert evidence[w.detector]["flagged"] is True
    # perceived_retouching has no reference image here: must be explicitly
    # not-run, not silently absent or indistinguishable from "passed".
    assert evidence["perceived_retouching"]["status"] == qa_detectors.QA_STATUS_NOT_RUN


def test_run_qa_with_evidence_self_reported_unavailable_does_not_warn():
    """A detector that self-reports available=False without flagging (e.g.
    evaluate_harmony given no face/body mask to measure) must be visible in
    evidence but must NOT become a QAWarning — matching run_qa's original
    behavior, which only ever warned on flagged=True.
    """
    from retouch import qa_detectors

    img = np.random.RandomState(0).randint(50, 200, (64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.float32)
    warnings, evidence = qa_detectors.run_qa_with_evidence(img, mask)

    harmony = evidence["harmony"]
    assert harmony["available"] is False
    assert harmony["flagged"] is False
    assert harmony["status"] == qa_detectors.QA_STATUS_UNAVAILABLE
    assert "harmony" not in {w.detector for w in warnings}


def test_run_qa_with_evidence_no_person_mask_is_not_run_for_every_detector():
    """No person mask means QA never ran; every detector must say so explicitly."""
    from retouch import qa_detectors

    img = np.zeros((32, 32, 3), dtype=np.uint8)
    warnings, evidence = qa_detectors.run_qa_with_evidence(img, None)

    assert warnings == []
    assert set(evidence.keys()) == set(qa_detectors.ALL_DETECTOR_NAMES)
    assert all(v["status"] == qa_detectors.QA_STATUS_NOT_RUN for v in evidence.values())


def test_run_qa_with_evidence_pipeline_failure_is_not_run_not_silently_passed(monkeypatch):
    """A QA pipeline exception must not read as 'every detector passed'."""
    from retouch import qa_detectors

    def fail(*_args, **_kwargs):
        raise RuntimeError("synthetic")

    monkeypatch.setattr(qa_detectors, "run_all", fail)
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.ones((32, 32), dtype=np.float32)
    warnings, evidence = qa_detectors.run_qa_with_evidence(img, mask)

    assert len(warnings) == 1 and warnings[0].detector == "qa_pipeline"
    assert all(v["status"] == qa_detectors.QA_STATUS_NOT_RUN for v in evidence.values())


def test_run_qa_unchanged_return_type_for_existing_callers():
    """run_qa (as opposed to run_qa_with_evidence) must keep its List[QAWarning] contract."""
    from retouch import qa_detectors

    img = np.random.RandomState(1).randint(50, 200, (64, 64, 3), dtype=np.uint8)
    mask = np.ones((64, 64), dtype=np.float32)
    warnings = qa_detectors.run_qa(img, mask)
    assert isinstance(warnings, list)
    assert all(hasattr(w, "detector") and w.flagged for w in warnings)


@requires_retouch
def test_processing_result_qa_evidence_covers_every_detector():
    """result.qa_evidence must let a caller tell 'passed' from 'not checked'."""
    from retouch import qa_detectors

    img = np.random.RandomState(2).randint(50, 200, (100, 100, 3), dtype=np.uint8)
    with _mocked_engine_for(100, 100, no_face=True) as engine:
        result = engine.process(img, recipe="natural")
    assert hasattr(result, "qa_evidence")
    assert isinstance(result.qa_evidence, dict)
    # No face detected: every detector must explicitly say not-run, not be
    # silently absent (which would look identical to "everything passed").
    assert set(result.qa_evidence.keys()) == set(qa_detectors.ALL_DETECTOR_NAMES)
    assert all(
        v["status"] == qa_detectors.QA_STATUS_NOT_RUN
        for v in result.qa_evidence.values()
    )
