"""Tests for retouch/utils.py — guided_filter function."""

import numpy as np
import cv2
import pytest
import time

from retouch.utils import guided_filter


class TestGuidedFilterBasics:
    def test_constant_image_returns_constant(self):
        """On a constant image, output should remain constant."""
        img = np.full((64, 64), 100.0, dtype=np.float32)
        result = guided_filter(img, radius=5, eps=1.0, guide=None)
        assert result.shape == img.shape
        assert result.dtype == np.float32
        # Output should be very close to the constant input
        assert np.allclose(result, 100.0, atol=0.1)

    def test_self_guided_vs_explicit_guide(self):
        """Self-guided (guide=None) should match explicit guide=src."""
        img = np.random.randn(64, 64).astype(np.float32) * 10 + 50
        result_self = guided_filter(img, radius=5, eps=1.0, guide=None)
        result_explicit = guided_filter(img, radius=5, eps=1.0, guide=img)
        assert np.allclose(result_self, result_explicit, atol=1e-5)

    def test_output_shape_matches_input(self):
        """Output shape should match input shape."""
        for h, w in [(32, 32), (50, 60), (100, 80)]:
            img = np.random.randn(h, w).astype(np.float32)
            result = guided_filter(img, radius=3, eps=0.5)
            assert result.shape == (h, w)
            assert result.dtype == np.float32

    def test_dtype_validation(self):
        """Should raise ValueError for non-float32 or non-2D input."""
        # Test uint8
        img_uint8 = np.full((64, 64), 100, dtype=np.uint8)
        with pytest.raises(ValueError):
            guided_filter(img_uint8, radius=5, eps=1.0)

        # Test 3D
        img_3d = np.random.randn(64, 64, 3).astype(np.float32)
        with pytest.raises(ValueError):
            guided_filter(img_3d, radius=5, eps=1.0)

    def test_guide_shape_mismatch_raises(self):
        """Should raise ValueError if guide shape doesn't match src."""
        src = np.random.randn(64, 64).astype(np.float32)
        guide = np.random.randn(60, 60).astype(np.float32)
        with pytest.raises(ValueError):
            guided_filter(src, radius=5, eps=1.0, guide=guide)


class TestGuidedFilterEdgePreservation:
    def test_step_edge_preservation(self):
        """On a step edge, output at distance > radius from edge should stay close to input."""
        # Create step edge: left half 50, right half 150
        img = np.zeros((64, 64), dtype=np.float32)
        img[:, :32] = 50.0
        img[:, 32:] = 150.0

        result = guided_filter(img, radius=8, eps=0.1)

        # Check pixels far from edge (at least radius away)
        assert np.allclose(result[:, :20], 50.0, atol=1.0), \
            f"Left side away from edge not preserved: mean={result[:, :20].mean()}"
        assert np.allclose(result[:, 44:], 150.0, atol=1.0), \
            f"Right side away from edge not preserved: mean={result[:, 44:].mean()}"

    def test_smooth_gradient_preservation(self):
        """On a smooth gradient, output should follow the gradient."""
        # Linear gradient from 0 to 255
        gradient = np.linspace(0, 255, 64, dtype=np.float32)
        img = np.tile(gradient, (64, 1))

        result = guided_filter(img, radius=5, eps=0.5)

        # Result should still be a smooth gradient (monotonic)
        assert np.all(np.diff(result.mean(axis=0)) > 0), \
            "Result is not monotonically increasing along gradient"


class TestGuidedFilterDownsampling:
    def test_downsample_path_consistency(self):
        """Output with max_dim downsampling should be close to without."""
        # Create a larger image to trigger downsampling
        np.random.seed(42)
        img = (np.random.randn(2000, 2000).astype(np.float32) * 15 + 100)
        img = np.clip(img, 0, 255)

        # With downsampling (max_dim=1200)
        result_downsampled = guided_filter(img, radius=20, eps=1.0, guide=None, max_dim=1200)

        # Without downsampling (max_dim=None)
        result_full = guided_filter(img, radius=20, eps=1.0, guide=None, max_dim=None)

        # Should be very similar (within 2.0 on 0-255 scale)
        mean_diff = np.abs(result_downsampled - result_full).mean()
        assert mean_diff < 2.0, f"Downsample path diverges too much: mean_diff={mean_diff}"

    def test_no_downsample_for_small_images(self):
        """Small images should not trigger downsampling."""
        img = np.random.randn(100, 100).astype(np.float32) * 10 + 50
        # max_dim=1200, image is small, so no downsampling should occur
        result = guided_filter(img, radius=5, eps=1.0, max_dim=1200)
        assert result.shape == (100, 100)
        # Should work without error
        assert result.dtype == np.float32


class TestGuidedFilterPerformance:
    def test_performance_small_image(self):
        """Benchmark on 400x400 float32 image."""
        img = np.random.randn(400, 400).astype(np.float32) * 15 + 100
        img = np.clip(img, 0, 255)

        start = time.perf_counter()
        result = guided_filter(img, radius=8, eps=1.0)
        elapsed = time.perf_counter() - start

        assert result.shape == (400, 400)
        print(f"guided_filter 400x400: {elapsed*1000:.2f}ms")

    def test_performance_large_image_with_downsampling(self):
        """Benchmark on 1500x1500 float32 image with downsampling."""
        img = np.random.randn(1500, 1500).astype(np.float32) * 15 + 100
        img = np.clip(img, 0, 255)

        start = time.perf_counter()
        result = guided_filter(img, radius=20, eps=1.0, max_dim=1200)
        elapsed = time.perf_counter() - start

        assert result.shape == (1500, 1500)
        print(f"guided_filter 1500x1500 (downsampled): {elapsed*1000:.2f}ms")


class TestGuidedFilterVsBilateral:
    def test_guided_vs_bilateral_similarity(self):
        """On a skin-like gradient+noise image, guided and bilateral should be similar."""
        np.random.seed(42)
        # Create synthetic skin-like image: smooth gradient with noise
        base = np.linspace(80, 180, 400, dtype=np.float32)
        base = np.tile(base, (400, 1))
        noise = np.random.randn(400, 400).astype(np.float32) * 10
        img = base + noise
        img = np.clip(img, 0, 255)

        # Guided filter parameters mapped from bilateral sigmas
        sigma_color = 20.0 + 0.5 * 60.0  # SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR
        sigma_space = 20.0 + 0.5 * 60.0
        radius = max(2, int(round(sigma_space)))
        eps = sigma_color ** 2

        result_guided = guided_filter(img, radius=radius, eps=eps, max_dim=None)

        # Bilateral filter reference
        result_bilateral = cv2.bilateralFilter(img.astype(np.float32), -1, sigma_color, sigma_space)

        # Should be somewhat similar (mean abs diff < 6 on 0-255 scale)
        mean_diff = np.abs(result_guided - result_bilateral).mean()
        assert mean_diff < 6.0, f"guided and bilateral differ too much: mean_diff={mean_diff}"
        print(f"guided vs bilateral mean_diff: {mean_diff:.2f}")

    def test_guided_and_bilateral_performance_comparison(self):
        """Time comparison: guided vs bilateral on 400x400 and 1500x1500."""
        for size in [400, 1500]:
            np.random.seed(42)
            img = np.random.randn(size, size).astype(np.float32) * 15 + 100
            img = np.clip(img, 0, 255)

            sigma_color = 20.0 + 0.5 * 60.0
            sigma_space = 20.0 + 0.5 * 60.0
            radius = max(2, int(round(sigma_space)))
            eps = sigma_color ** 2

            # Time guided filter
            start = time.perf_counter()
            result_guided = guided_filter(img, radius=radius, eps=eps, max_dim=None)
            time_guided = time.perf_counter() - start

            # Time the bilateral reference only at the smaller size. Its
            # runtime grows sharply with image area and is hardware-dependent
            # at 1500x1500; the separate similarity test already covers the
            # bilateral output contract, while the large-image check here is
            # specifically for the guided-filter path.
            if size == 400:
                start = time.perf_counter()
                result_bilateral = cv2.bilateralFilter(img, -1, sigma_color, sigma_space)
                time_bilateral = time.perf_counter() - start
                bilateral_text = f"bilateral: {time_bilateral*1000:.2f}ms"
                assert time_bilateral < 10.0, f"Bilateral filter too slow: {time_bilateral*1000:.2f}ms"
            else:
                bilateral_text = "bilateral: skipped"

            print(f"{size}x{size} guided: {time_guided*1000:.2f}ms, {bilateral_text}")
            assert time_guided < 10.0, f"Guided filter too slow: {time_guided*1000:.2f}ms"
