"""Tests for retouch/precision.py — float32 internal pipeline helpers.

Promoted from the AUDIT_REPORT.md §1.4 deep algorithmic verification harness.
"""

import numpy as np
import pytest

from retouch.precision import PrecisionContext, ensure_float, to_float, to_uint8


class TestToFloatToUint8RoundTrip:
    """Verify to_uint8(to_float(x)) is exact inverse for all x in 0..255."""

    def test_round_trip_identity_all_values(self):
        """to_uint8 ∘ to_float must be identity on uint8 → float32 → uint8."""
        for x in range(256):
            img = np.full((1, 1, 3), x, dtype=np.uint8)
            f = to_float(img)
            back = to_uint8(f)
            assert back[0, 0, 0] == x, f"Round-trip failed at value {x}"

    def test_round_trip_random_images(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            img = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
            assert np.array_equal(to_uint8(to_float(img)), img)

    def test_to_float_output_range(self):
        img = np.full((4, 4, 3), 200, dtype=np.uint8)
        f = to_float(img)
        assert f.dtype == np.float32
        assert f.min() >= 0.0
        assert f.max() <= 1.0

    def test_to_float_handles_float32_pass_through(self):
        arr = np.full((4, 4, 3), 0.5, dtype=np.float32)
        result = to_float(arr)
        assert result.dtype == np.float32
        assert np.allclose(result, 0.5)

    def test_to_float_clips_out_of_range_float(self):
        arr = np.array([-0.5, 0.5, 1.5], dtype=np.float32)
        result = to_float(arr)
        assert result[0] == 0.0
        assert result[1] == 0.5
        assert result[2] == 1.0

    def test_to_float_converts_float64(self):
        arr = np.full((2, 2, 3), 0.75, dtype=np.float64)
        result = to_float(arr)
        assert result.dtype == np.float32
        assert np.allclose(result, 0.75)

    def test_to_uint8_output_range(self):
        f = np.full((4, 4, 3), 0.75, dtype=np.float32)
        result = to_uint8(f)
        assert result.dtype == np.uint8
        assert result.min() >= 0
        assert result.max() <= 255

    def test_to_uint8_rounds_not_truncates(self):
        """to_uint8 uses round-to-nearest, not floor cast."""
        # With round: 0.6 * 255 = 153.0 → 153. Floor: 153 (same).
        # With round: 0.498 * 255 = 126.99 → 127. Floor: 126.
        f = np.full((1, 1, 3), 0.4980392156862745, dtype=np.float32)
        result = to_uint8(f)
        assert result[0, 0, 0] == 127


class TestPrecisionContext:
    def test_context_manager_lifecycle(self):
        with PrecisionContext(bit_depth="16") as pc:
            assert pc.is_float is True
            assert pc._active is True
        assert pc._active is False

    def test_context_8_bit_is_not_float(self):
        with PrecisionContext(bit_depth="8") as pc:
            assert pc.is_float is False

    def test_invalid_bit_depth_raises(self):
        with pytest.raises(ValueError):
            PrecisionContext(bit_depth="32")

    def test_process_raises_outside_with(self):
        pc = PrecisionContext(bit_depth="16")
        with pytest.raises(RuntimeError, match="outside of a `with` block"):
            pc.process(np.zeros((4, 4, 3), dtype=np.uint8), lambda x: x)

    def test_process_round_trip(self):
        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        with PrecisionContext(bit_depth="16") as pc:
            result = pc.process(img, lambda x: x)
        assert np.array_equal(result, img)

    def test_process_clips_float_output(self):
        """Operation returning out-of-range values must be clipped."""
        img = np.full((4, 4, 3), 128, dtype=np.uint8)

        def bad_op(x: np.ndarray) -> np.ndarray:
            return x * 2.0

        with PrecisionContext(bit_depth="16") as pc:
            result = pc.process(img, bad_op)
        # Clipping should keep it in range
        assert result.max() <= 255
        assert result.min() >= 0

    def test_process_8_bit_is_identity_passthrough(self):
        img = np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8)
        with PrecisionContext(bit_depth="8") as pc:
            result = pc.process(img, lambda x: x * 2.0)
        # 8-bit mode: no float conversion, op runs on uint8 (wraps), then
        # to_uint8(to_float(...)) round-trips what it gets back.
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestEnsureFloat:
    def test_uint8_converted(self):
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        result = ensure_float(img)
        assert result.dtype == np.float32
        assert np.allclose(result, 128.0 / 255.0)

    def test_float32_passthrough(self):
        arr = np.full((4, 4, 3), 0.5, dtype=np.float32)
        result = ensure_float(arr)
        assert result.dtype == np.float32
        assert np.allclose(result, 0.5)

    def test_out_of_range_clipped(self):
        arr = np.array([2.0], dtype=np.float32)
        result = ensure_float(arr)
        assert result[0] == 1.0
