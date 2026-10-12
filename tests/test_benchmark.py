"""Tests for retouch/benchmark.py — Performance Benchmarking & Pre-Flight Checks."""

import numpy as np
import pytest

from retouch.benchmark import (
    run_preflight_checks,
    benchmark_engine_throughput,
)


class TestBenchmarkEngine:
    """Tests for system pre-flight checks and throughput benchmarks."""

    def test_run_preflight_checks(self):
        results = run_preflight_checks()
        assert isinstance(results, dict)
        assert results.get("color_science_oklab") is True
        assert results.get("qa_detectors_run_all") is True
        assert results.get("acceleration_layer") is True

    def test_benchmark_engine_throughput(self):
        def mock_process(img):
            return img

        res = benchmark_engine_throughput(
            mock_process,
            resolutions={"test_crop": (100, 100)},
            iterations=2,
        )
        assert "test_crop" in res
        bench = res["test_crop"]
        assert bench.megapixels > 0.0
        assert bench.mpx_per_sec > 0.0
        assert bench.fps > 0.0
