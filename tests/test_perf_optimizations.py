"""Tests for retouch/perf_optimizations.py — providers, JIT kernels, warm-up.

Covers items with 0% coverage:
- build_ort_providers()  — provider priority selection
- warmup_jit_kernels()   — JIT compile trigger
- _apply_tonal_lut()     — Numba pixel loop
- _blend_highpass()      — Numba pixel loop
"""

from unittest.mock import patch

import numpy as np
import onnxruntime as ort
import pytest


# ---------------------------------------------------------------------------
# build_ort_providers
# ---------------------------------------------------------------------------


class TestBuildOrtProviders:
    def test_returns_non_empty_list(self):
        from retouch.perf_optimizations import build_ort_providers

        providers = build_ort_providers()
        assert isinstance(providers, list)
        assert len(providers) >= 1

    def test_always_includes_cpu_fallback(self):
        from retouch.perf_optimizations import build_ort_providers

        providers = build_ort_providers()
        # CPU is always appended at the end as a safety net
        assert "CPUExecutionProvider" in providers

    def test_cpu_only_when_no_accelerators(self):
        from retouch.perf_optimizations import build_ort_providers

        with patch("retouch.perf_optimizations.ort") as mock_ort:
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
            providers = build_ort_providers()

        assert providers == ["CPUExecutionProvider"]

    def test_cuda_selected_when_available(self):
        from retouch.perf_optimizations import build_ort_providers

        with patch("retouch.perf_optimizations.ort") as mock_ort:
            mock_ort.get_available_providers.return_value = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = build_ort_providers()

        # First entry should be CUDA tuple with config dict
        assert providers[0] == (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 ** 3,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
            },
        )
        # CPU is appended as the final fallback
        assert providers[-1] == "CPUExecutionProvider"

    def test_directml_selected_when_no_cuda(self):
        from retouch.perf_optimizations import build_ort_providers

        with patch("retouch.perf_optimizations.ort") as mock_ort:
            mock_ort.get_available_providers.return_value = [
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = build_ort_providers()

        assert providers[0] == "DmlExecutionProvider"
        assert providers[-1] == "CPUExecutionProvider"

    def test_coreml_takes_precedence_over_cuda(self):
        from retouch.perf_optimizations import build_ort_providers

        with patch("retouch.perf_optimizations.ort") as mock_ort:
            # CoreML wins (Apple Silicon priority) even when CUDA is present
            mock_ort.get_available_providers.return_value = [
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = build_ort_providers()

        first = providers[0]
        assert isinstance(first, tuple)
        assert first[0] == "CoreMLExecutionProvider"
        assert first[1]["MLComputeUnits"] == "ALL"
        assert first[1]["ModelFormat"] == "MLProgram"
        assert first[1]["RequireStaticInputShapes"] == "0"
        # CPU is the last fallback
        assert providers[-1] == "CPUExecutionProvider"

    def test_cuda_preferred_over_directml(self):
        from retouch.perf_optimizations import build_ort_providers

        with patch("retouch.perf_optimizations.ort") as mock_ort:
            mock_ort.get_available_providers.return_value = [
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]
            providers = build_ort_providers()

        # CUDA wins over DirectML
        assert providers[0][0] == "CUDAExecutionProvider"
        # No Dml in the chain
        assert "DmlExecutionProvider" not in providers


