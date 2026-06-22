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


# ---------------------------------------------------------------------------
# warmup_jit_kernels
# ---------------------------------------------------------------------------


class TestWarmupJitKernels:
    def test_runs_without_error(self):
        from retouch.perf_optimizations import warmup_jit_kernels

        # Should not raise — JIT compiles (or no-ops without numba)
        warmup_jit_kernels()

    def test_calls_underlying_jit_kernels(self):
        from retouch import perf_optimizations

        with patch.object(
            perf_optimizations, "_apply_tonal_lut", wraps=perf_optimizations._apply_tonal_lut
        ) as mock_lut, patch.object(
            perf_optimizations, "_blend_highpass", wraps=perf_optimizations._blend_highpass
        ) as mock_blend:
            perf_optimizations.warmup_jit_kernels()

        assert mock_lut.called
        assert mock_blend.called
        assert mock_lut.call_count == 1
        assert mock_blend.call_count == 1

    def test_passes_dummy_data_of_correct_shape(self):
        from retouch import perf_optimizations

        captured = {}

        def fake_lut(img_f, y_lut):
            captured["img_shape"] = img_f.shape
            captured["img_dtype"] = img_f.dtype
            captured["lut_len"] = len(y_lut)
            return img_f.copy()

        def fake_blend(low, original, alpha_mask):
            captured["low_shape"] = low.shape
            captured["mask_shape"] = alpha_mask.shape
            return low.copy()

        with patch.object(perf_optimizations, "_apply_tonal_lut", side_effect=fake_lut), \
             patch.object(perf_optimizations, "_blend_highpass", side_effect=fake_blend):
            perf_optimizations.warmup_jit_kernels()

        # 4x4x3 float32 image, 256-entry LUT
        assert captured["img_shape"] == (4, 4, 3)
        assert captured["img_dtype"] == np.float32
        assert captured["lut_len"] == 256
        # Mask must be 4x4 float32
        assert captured["low_shape"] == (4, 4, 3)
        assert captured["mask_shape"] == (4, 4)


# ---------------------------------------------------------------------------
# _apply_tonal_lut  (Numba kernel, gracefully no-ops without numba)
# ---------------------------------------------------------------------------


class TestApplyTonalLutKernel:
    def test_identity_lut_returns_input(self):
        from retouch.perf_optimizations import _apply_tonal_lut

        img = np.array(
            [[[10, 20, 30], [40, 50, 60]],
             [[70, 80, 90], [100, 110, 120]]],
            dtype=np.float32,
        )
        identity = np.arange(256, dtype=np.float32)
        out = _apply_tonal_lut(img, identity)
        assert out.shape == img.shape
        assert out.dtype == np.float32
        assert np.allclose(out, img)

    def test_invert_lut_produces_complement(self):
        from retouch.perf_optimizations import _apply_tonal_lut

        img = np.full((4, 4, 3), 100.0, dtype=np.float32)
        invert = (255.0 - np.arange(256)).astype(np.float32)
        out = _apply_tonal_lut(img, invert)
        # 100 -> 255 - 100 = 155
        assert np.allclose(out, 155.0)

    def test_zero_lut_returns_zero(self):
        from retouch.perf_optimizations import _apply_tonal_lut

        img = np.random.rand(4, 4, 3).astype(np.float32) * 255.0
        zero_lut = np.zeros(256, dtype=np.float32)
        out = _apply_tonal_lut(img, zero_lut)
        assert np.all(out == 0.0)

    def test_output_shape_matches_input(self):
        from retouch.perf_optimizations import _apply_tonal_lut

        img = np.zeros((6, 10, 3), dtype=np.float32)
        lut = np.arange(256, dtype=np.float32)
        out = _apply_tonal_lut(img, lut)
        assert out.shape == (6, 10, 3)

    def test_per_channel_mapping(self):
        from retouch.perf_optimizations import _apply_tonal_lut

        # Build an image where each channel has a distinct known value
        img = np.zeros((2, 2, 3), dtype=np.float32)
        img[..., 0] = 10.0
        img[..., 1] = 50.0
        img[..., 2] = 200.0

        # LUT that doubles each index, then clamps at 255
        lut = np.clip(np.arange(256, dtype=np.float32) * 2.0, 0, 255)

        out = _apply_tonal_lut(img, lut)
        assert np.allclose(out[..., 0], 20.0)   # 10 * 2
        assert np.allclose(out[..., 1], 100.0)  # 50 * 2
        assert np.allclose(out[..., 2], 255.0)  # 200 * 2 = 400 -> clamped to 255

    def test_clamps_out_of_range_indices(self):
        from retouch.perf_optimizations import _apply_tonal_lut

        # Values that go above 255 or below 0 must be clamped before lookup
        img = np.array(
            [[[300.0, -50.0, 50.0]]],
            dtype=np.float32,
        )
        # 1-to-1 LUT so we can verify clamping
        lut = np.arange(256, dtype=np.float32)
        out = _apply_tonal_lut(img, lut)
        assert out[0, 0, 0] == 255.0   # 300 -> 255
        assert out[0, 0, 1] == 0.0     # -50 -> 0
        assert out[0, 0, 2] == 50.0    # 50 -> 50


# ---------------------------------------------------------------------------
# _blend_highpass  (Numba kernel, gracefully no-ops without numba)
# ---------------------------------------------------------------------------


class TestBlendHighpassKernel:
    def test_alpha_zero_returns_original(self):
        from retouch.perf_optimizations import _blend_highpass

        low = np.full((4, 4, 3), 50.0, dtype=np.float32)
        original = np.full((4, 4, 3), 200.0, dtype=np.float32)
        mask = np.zeros((4, 4), dtype=np.float32)

        out = _blend_highpass(low, original, mask)
        # out = low + (orig - low) * (1 - 0) = orig
        assert np.allclose(out, 200.0)

    def test_alpha_one_returns_low(self):
        from retouch.perf_optimizations import _blend_highpass

        low = np.full((4, 4, 3), 50.0, dtype=np.float32)
        original = np.full((4, 4, 3), 200.0, dtype=np.float32)
        mask = np.ones((4, 4), dtype=np.float32)

        out = _blend_highpass(low, original, mask)
        # out = low + (orig - low) * 0 = low
        assert np.allclose(out, 50.0)

    def test_alpha_half_returns_average(self):
        from retouch.perf_optimizations import _blend_highpass

        low = np.full((4, 4, 3), 100.0, dtype=np.float32)
        original = np.full((4, 4, 3), 200.0, dtype=np.float32)
        mask = np.full((4, 4), 0.5, dtype=np.float32)

        out = _blend_highpass(low, original, mask)
        # 100 + (200-100)*0.5 = 150
        assert np.allclose(out, 150.0)

    def test_output_shape_matches_low(self):
        from retouch.perf_optimizations import _blend_highpass

        low = np.zeros((6, 8, 3), dtype=np.float32)
        original = np.ones((6, 8, 3), dtype=np.float32)
        mask = np.zeros((6, 8), dtype=np.float32)
        out = _blend_highpass(low, original, mask)
        assert out.shape == (6, 8, 3)
        assert out.dtype == np.float32

    def test_spatial_mask_applies_per_pixel(self):
        from retouch.perf_optimizations import _blend_highpass

        low = np.zeros((2, 2, 3), dtype=np.float32)
        original = np.full((2, 2, 3), 100.0, dtype=np.float32)
        mask = np.array([[0.0, 1.0], [0.5, 0.25]], dtype=np.float32)

        out = _blend_highpass(low, original, mask)
        # out[y,x] = 0 + (100 - 0) * (1 - mask[y,x])
        assert np.allclose(out[0, 0], 100.0)   # mask=0 -> 1-mask=1
        assert np.allclose(out[0, 1], 0.0)     # mask=1 -> 1-mask=0
        assert np.allclose(out[1, 0], 50.0)    # mask=0.5 -> 1-mask=0.5
        assert np.allclose(out[1, 1], 75.0)    # mask=0.25 -> 1-mask=0.75

    def test_does_not_modify_inputs(self):
        from retouch.perf_optimizations import _blend_highpass

        low = np.full((4, 4, 3), 50.0, dtype=np.float32)
        original = np.full((4, 4, 3), 200.0, dtype=np.float32)
        mask = np.full((4, 4), 0.5, dtype=np.float32)

        low_copy = low.copy()
        original_copy = original.copy()
        mask_copy = mask.copy()

        _ = _blend_highpass(low, original, mask)
        assert np.array_equal(low, low_copy)
        assert np.array_equal(original, original_copy)
        assert np.array_equal(mask, mask_copy)


# ---------------------------------------------------------------------------
# onnxruntime provider discovery sanity check
# ---------------------------------------------------------------------------


def test_ort_providers_list_is_a_list():
    """Sanity check that the underlying ORT call returns a list of strings."""
    providers = ort.get_available_providers()
    assert isinstance(providers, list)
    for p in providers:
        assert isinstance(p, str)
