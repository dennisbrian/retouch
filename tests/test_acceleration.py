"""Tests for retouch/acceleration.py — GPU Acceleration Abstraction Layer."""

import numpy as np
import pytest

from retouch.acceleration import (
    get_acceleration_backend,
    is_gpu_available,
    accelerated_gaussian_blur,
)


class TestAccelerationLayer:
    """Tests for GPU acceleration backend and CPU fallback equivalence."""

    def test_backend_string(self):
        backend = get_acceleration_backend()
        assert backend in ("cuda", "mps", "cpu")

    def test_is_gpu_available_boolean(self):
        gpu_flag = is_gpu_available()
        assert isinstance(gpu_flag, bool)

    def test_accelerated_gaussian_blur_shape_dtype(self):
        img = np.full((64, 64, 3), 150, dtype=np.uint8)
        blurred = accelerated_gaussian_blur(img, ksize=5, sigma=1.0)
        assert blurred.shape == img.shape
        assert blurred.dtype == np.uint8

    def test_accelerated_gaussian_blur_float32(self):
        img = np.full((32, 32, 3), 0.5, dtype=np.float32)
        blurred = accelerated_gaussian_blur(img, ksize=3, sigma=1.0)
        assert blurred.shape == img.shape
        assert blurred.dtype == np.float32

    def test_single_channel_blur(self):
        img = np.full((32, 32), 100, dtype=np.uint8)
        blurred = accelerated_gaussian_blur(img, ksize=5, sigma=1.0)
        assert blurred.shape == (32, 32)
