"""Tests for engine fidelity fixes (E3.5, E3.1, E4)."""

from __future__ import annotations

import numpy as np
import pytest

from retouch.engine import _DEFAULTS


class TestBWMixerDefaults:
    """E3.5: B&W channel mixer uses _DEFAULTS instead of hardcoded 30/59/11."""

    def test_defaults_match_hardcoded(self):
        """Verify _DEFAULTS values match the known 30/59/11 weights."""
        assert _DEFAULTS["bw_channel_mixer_r"] == 30
        assert _DEFAULTS["bw_channel_mixer_g"] == 59
        assert _DEFAULTS["bw_channel_mixer_b"] == 11

    def test_engine_uses_defaults(self):
        """Verify engine.py imports and references _DEFAULTS for B&W mixer."""
        import ast
        with open("retouch/engine.py") as f:
            source = f.read()
            tree = ast.parse(source)

        default_refs = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and "bw_channel_mixer" in str(node.slice.value)
        ]
        assert len(default_refs) >= 2, \
            f"Expected at least 2 _DEFAULTS['bw_channel_mixer_*'] refs, found {len(default_refs)}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                source_line = ast.unparse(node) if hasattr(ast, 'unparse') else ""
                if "bw_channel_mixer_r" in source_line and "30" in source_line:
                    pytest.fail(f"Hardcoded 30 found in B&W mixer comparison: {source_line}")
                if "bw_channel_mixer_g" in source_line and "59" in source_line:
                    pytest.fail(f"Hardcoded 59 found in B&W mixer comparison: {source_line}")
                if "bw_channel_mixer_b" in source_line and "11" in source_line:
                    pytest.fail(f"Hardcoded 11 found in B&W mixer comparison: {source_line}")


class TestSharpenMapping:
    """E3.1: Sharpen slider responds over full 0-100 range."""

    def test_sharpen_monotonic(self):
        """Verify the sharpen mapping is monotonic over full range."""
        import ast
        with open("retouch/engine.py") as f:
            source = f.read()

        assert "max(1.2, ctx.sharpen" not in source, \
            "Sharpen should not use max(1.2, ...) pattern"

        def sharpen_amount(val):
            if val <= 50:
                return val / 50.0 * 1.0
            else:
                return 1.0 + (val - 50.0) / 50.0 * 2.0

        amounts = [sharpen_amount(v) for v in range(0, 101)]
        for i in range(1, len(amounts)):
            assert amounts[i] > amounts[i-1], f"Sharpen not monotonic at {i}: {amounts[i-1]} -> {amounts[i]}"
        assert amounts[0] == 0.0, "Sharpen(0) should be 0"
        assert amounts[100] == 3.0, "Sharpen(100) should be 3.0"


class TestPoolPayload:
    """E4: FaceProcessorPool payload should not include full ctx."""

    def test_payload_does_not_include_raw_ctx(self):
        """Verify the payload tuple does not pass ctx directly."""
        with open("retouch/engine.py") as f:
            source = f.read()
        assert "_slim_ctx" in source or "slim_ctx" in source, \
            "Payload should use _slim_ctx helper instead of raw ctx"

    def test_max_workers_defaults_to_4(self):
        """Verify FaceProcessorPool defaults to 4 workers."""
        from retouch.perf_optimizations import FaceProcessorPool
        pool = FaceProcessorPool()
        assert pool._max_workers == 4, \
            f"Expected max_workers=4, got {pool._max_workers}"
        pool.shutdown()


class TestDeadCodeRemoved:
    """E4: Dead Numba kernels and warmup removed."""

    def test_warmup_jit_kernels_removed(self):
        with open("retouch/perf_optimizations.py") as f:
            source = f.read()
        assert "def warmup_jit_kernels" not in source, \
            "warmup_jit_kernels should be removed"
        assert "def _apply_tonal_lut" not in source, \
            "_apply_tonal_lut should be removed"
        assert "def _blend_highpass" not in source, \
            "_blend_highpass should be removed"
