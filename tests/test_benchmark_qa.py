"""Tests for QA detector benchmark integration (F11)."""

from __future__ import annotations

import numpy as np
import pytest


def _retouch_importable() -> bool:
    """Check if retouch can be imported (needed for metric functions)."""
    try:
        import retouch  # noqa: F401
        return True
    except Exception:
        return False


requires_retouch = pytest.mark.skipif(
    not _retouch_importable(),
    reason="retouch package not importable (pre-existing env issue)",
)


def test_skin_quality_metrics_import():
    """skin_quality_metrics should be importable from benchmark."""
    from scripts.bench.benchmark import skin_quality_metrics
    assert callable(skin_quality_metrics)


def test_qa_detector_metrics_import():
    """qa_detector_metrics should be importable from benchmark."""
    from scripts.bench.benchmark import qa_detector_metrics
    assert callable(qa_detector_metrics)


@requires_retouch
def test_qa_detector_metrics_none_mask():
    from scripts.bench.benchmark import qa_detector_metrics
    img = np.random.RandomState(42).randint(0, 255, (50, 50, 3), dtype=np.uint8)
    result = qa_detector_metrics(img, None)
    assert isinstance(result, dict)
    assert "banding_score" in result


@requires_retouch
def test_qa_detector_metrics_with_mask():
    from scripts.bench.benchmark import qa_detector_metrics
    img = np.random.RandomState(42).randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.float32)
    result = qa_detector_metrics(img, mask)
    assert all(0.0 <= v <= 1.0 for v in result.values())


@requires_retouch
def test_qa_detector_metrics_empty_mask():
    from scripts.bench.benchmark import qa_detector_metrics
    img = np.random.RandomState(42).randint(0, 255, (50, 50, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.float32)
    result = qa_detector_metrics(img, mask)
    # Should not crash


@requires_retouch
def test_skin_quality_metrics_accepts_inputs():
    """Verify skin_quality_metrics works with its signature."""
    from scripts.bench.benchmark import skin_quality_metrics
    img = np.random.RandomState(42).randint(0, 255, (100, 100, 3), dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.float32)
    result = skin_quality_metrics(img, mask, face_width=80.0)
    assert isinstance(result, dict)
    assert "blotch_std" in result
    assert "chroma_std" in result
    assert result["blotch_std"] >= 0.0
    assert result["chroma_std"] >= 0.0


@requires_retouch
def test_skin_quality_metrics_uniform_image():
    """Uniform image should have near-zero blotch_std."""
    from scripts.bench.benchmark import skin_quality_metrics
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.float32)
    result = skin_quality_metrics(img, mask, face_width=80.0)
    assert result["blotch_std"] < 0.5, f"Uniform image should have low blotch_std: {result['blotch_std']}"


def test_parse_skin_quality_benchmark_lines():
    """parse_benchmark_lines should handle skin_quality output."""
    from scripts.bench.benchmark import parse_benchmark_lines

    output = (
        "[benchmark] skin_quality_input: blotch_std=12.345600 chroma_std=0.789000\n"
        "[benchmark] skin_quality_output: blotch_std=8.901200 chroma_std=0.654000"
    )
    results = parse_benchmark_lines(output)
    skin_results = [r for r in results if r.get("type") == "skin_quality"]
    assert len(skin_results) == 2
    assert skin_results[0]["phase"] == "input"
    assert abs(skin_results[0]["blotch_std"] - 12.3456) < 0.001
    assert skin_results[1]["phase"] == "output"
    assert abs(skin_results[1]["blotch_std"] - 8.9012) < 0.001


def test_parse_qa_detector_benchmark_lines():
    """parse_benchmark_lines should handle QA detector output."""
    from scripts.bench.benchmark import parse_benchmark_lines

    output = "[benchmark] qa_detector_output: banding=0.1200 clipping=0.0500 plastic=0.3400"
    results = parse_benchmark_lines(output)
    qa_results = [r for r in results if r.get("type") == "qa_detector"]
    assert len(qa_results) == 1
    assert qa_results[0]["phase"] == "output"
    assert abs(qa_results[0]["banding"] - 0.12) < 0.001
    assert abs(qa_results[0]["clipping"] - 0.05) < 0.001
    assert abs(qa_results[0]["plastic"] - 0.34) < 0.001
