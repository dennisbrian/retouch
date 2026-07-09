"""Tests for retouch/enhance.py — F7 AI denoise + super-resolution."""

import cv2
import numpy as np
import pytest

from retouch.enhance import AIEnhancer


def _make_noisy(h: int = 128, w: int = 128, noise_std: float = 30.0) -> np.ndarray:
    rng = np.random.default_rng(42)
    base = np.full((h, w, 3), 128.0, dtype=np.float32)
    noise = rng.normal(0.0, noise_std, base.shape).astype(np.float32)
    return np.clip(base + noise, 0.0, 255.0).astype(np.float32)


def _std(img: np.ndarray) -> float:
    return float(np.std(img.astype(np.float32)))


class TestDenoise:
    def test_reduces_noise(self) -> None:
        enh = AIEnhancer(tile=64, overlap=16)
        noisy = _make_noisy(128, 128, noise_std=30.0)
        out = enh.denoise(noisy, strength=0.7)
        assert out.dtype == np.float32
        assert out.shape == noisy.shape
        assert _std(out) < _std(noisy), (
            f"denoise should reduce noise: in={_std(noisy):.2f} out={_std(out):.2f}"
        )

    def test_zero_strength_is_noop(self) -> None:
        enh = AIEnhancer()
        img = _make_noisy(64, 64, noise_std=20.0)
        out = enh.denoise(img, strength=0.0)
        assert np.array_equal(out, img), "strength=0 must return the input unchanged"

    def test_dtype_preservation_float32(self) -> None:
        enh = AIEnhancer()
        img = _make_noisy(64, 64).astype(np.float32)
        out = enh.denoise(img, strength=0.5)
        assert out.dtype == np.float32

    def test_dtype_preservation_uint8(self) -> None:
        enh = AIEnhancer()
        img = _make_noisy(64, 64).astype(np.uint8)
        out = enh.denoise(img, strength=0.5)
        assert out.dtype == np.uint8

    def test_uses_fallback_when_no_model(self, monkeypatch) -> None:
        # Explicitly mock model_exists to force fallback, even if a real model is present
        from retouch import model_fetch
        monkeypatch.setattr(model_fetch, "model_exists", lambda name: False if name == "denoise_nafnet" else True)
        enh = AIEnhancer()
        assert enh._denoise_model() is None, (
            "monkeypatched model_exists should force _denoise_model to return None"
        )
        out = enh.denoise(_make_noisy(64, 64), strength=0.5)
        assert out.shape == (64, 64, 3)
        assert out.dtype == np.float32


class TestSuperResolve:
    def test_increases_resolution(self) -> None:
        enh = AIEnhancer()
        img = _make_noisy(64, 64).astype(np.uint8)
        out = enh.super_resolve(img, scale=2)
        assert out.shape[:2] == (128, 128), f"expected 2x upscale, got {out.shape}"

    def test_scale_1_is_noop(self) -> None:
        enh = AIEnhancer()
        img = _make_noisy(48, 48)
        out = enh.super_resolve(img, scale=1)
        assert np.array_equal(out, img), "scale=1 must return the input unchanged"

    def test_dtype_preservation_float32(self) -> None:
        enh = AIEnhancer()
        img = _make_noisy(48, 48).astype(np.float32)
        out = enh.super_resolve(img, scale=2)
        assert out.dtype == np.float32

    def test_dtype_preservation_uint8(self) -> None:
        enh = AIEnhancer()
        img = _make_noisy(48, 48).astype(np.uint8)
        out = enh.super_resolve(img, scale=2)
        assert out.dtype == np.uint8

    def test_uses_fallback_when_no_model(self) -> None:
        enh = AIEnhancer()
        assert enh._sr_model() is None, "no SR model should be present in the test env"
        out = enh.super_resolve(_make_noisy(48, 48).astype(np.uint8), scale=2)
        assert out.shape[:2] == (96, 96)


class TestEnhance:
    def test_combined_denoise_then_sr(self) -> None:
        enh = AIEnhancer()
        img = _make_noisy(64, 64, noise_std=25.0).astype(np.uint8)
        out = enh.enhance(img, denoise_strength=0.6, sr_scale=2)
        assert out.shape[:2] == (128, 128)
        assert out.dtype == np.uint8

    def test_combined_noop(self) -> None:
        enh = AIEnhancer()
        img = _make_noisy(48, 48)
        out = enh.enhance(img, denoise_strength=0.0, sr_scale=1)
        assert np.array_equal(out, img), "strength=0, scale=1 must be a no-op"


class TestTiledInference:
    def test_tile_grid_covers_image(self) -> None:
        from retouch.enhance import _tile_grid
        tiles = _tile_grid(700, 500, 256, 32)
        xs = sorted({x0 for x0, _, _, _ in tiles})
        ys = sorted({y0 for _, _, y0, _ in tiles})
        assert tiles[0][0] == 0 and tiles[0][2] == 0, "first tile must start at origin"
        assert max(x1 for _, x1, _, _ in tiles) == 500, "tiles must reach right edge"
        assert max(y1 for _, _, _, y1 in tiles) == 700, "tiles must reach bottom edge"

    def test_small_image_single_tile(self) -> None:
        from retouch.enhance import _tile_grid
        tiles = _tile_grid(100, 80, 256, 32)
        assert len(tiles) == 1
        assert tiles[0] == [0, 80, 0, 100]

    def test_identity_roundtrip_no_black_borders(self) -> None:
        """Regression: feather weights must never be exactly zero.

        A zero-weight tile edge at the image border has no overlapping
        neighbour, so the weighted merge produced black (0) border rows/cols.
        An identity run through _tiled_inference must return the input
        unchanged, borders included.
        """
        enh = AIEnhancer(tile=64, overlap=16)
        rng = np.random.default_rng(7)
        img = rng.uniform(50.0, 200.0, (150, 200, 3)).astype(np.float32)
        out = enh._tiled_inference(lambda t: t, img, out_scale=1)
        assert np.abs(out - img).max() < 1e-3, (
            f"identity tiled inference should be lossless; max err "
            f"{np.abs(out - img).max():.4f}, border row mean {out[0].mean():.2f}"
        )


class TestRealNAFNet:
    """Tests for the real NAFNet denoise model (skip if not present).

    These tests exercise the real ONNX model path and will skip gracefully
    until the denoise_nafnet model file is dropped into models/.
    """

    pytestmark = pytest.mark.skipif(
        not __import__("retouch.model_fetch", fromlist=["model_exists"]).model_exists("denoise_nafnet"),
        reason="real NAFNet model not present"
    )

    def test_denoise_improves_noisy_image(self) -> None:
        """Synthetic noisy image: denoise reduces noise by 2x vs fallback."""
        enh = AIEnhancer(tile=256, overlap=32)
        # Create a smooth gradient + gaussian noise
        h, w = 400, 600
        grad = np.linspace(100, 150, w, dtype=np.float32)
        clean = np.tile(grad[None, :, None], (h, 1, 3))
        noise_std = 15.0
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, noise_std, clean.shape).astype(np.float32)
        noisy = np.clip(clean + noise, 0.0, 255.0).astype(np.float32)

        # Denoise at full strength
        denoised = enh.denoise(noisy, strength=1.0)

        # Measure residual noise as std of laplacian (high-frequency content)
        def residual_noise_std(img: np.ndarray) -> float:
            if img.ndim == 3:
                gray = np.mean(img, axis=2)
            else:
                gray = img
            laplacian = cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)
            return float(np.std(laplacian))

        residual_noisy = residual_noise_std(noisy)
        residual_denoised = residual_noise_std(denoised)

        # Real model should reduce high-frequency noise by at least 2x
        assert residual_denoised < residual_noisy / 2.0, (
            f"denoise should reduce residual noise: before={residual_noisy:.2f}, "
            f"after={residual_denoised:.2f}, ratio={residual_noisy / residual_denoised:.2f}"
        )

    def test_denoise_preserves_dtype_uint8(self) -> None:
        """uint8 in → uint8 out, values in [0, 255]."""
        enh = AIEnhancer()
        img = _make_noisy(128, 128, noise_std=12.0).astype(np.uint8)
        out = enh.denoise(img, strength=1.0)
        assert out.dtype == np.uint8, f"expected uint8, got {out.dtype}"
        assert np.all(out >= 0) and np.all(out <= 255), "output values out of uint8 range"

    def test_denoise_preserves_dtype_float32(self) -> None:
        """float32 in → float32 out, values in [0, 255]."""
        enh = AIEnhancer()
        img = _make_noisy(128, 128, noise_std=12.0).astype(np.float32)
        out = enh.denoise(img, strength=1.0)
        assert out.dtype == np.float32, f"expected float32, got {out.dtype}"
        assert np.all(out >= 0) and np.all(out <= 255), "output values out of [0, 255] range"

    def test_denoise_dynamic_shapes(self) -> None:
        """Non-square and sub-tile images run without exception."""
        enh = AIEnhancer(tile=256, overlap=32)

        # Non-square
        img_nonsquare = _make_noisy(320, 480, noise_std=10.0)
        out_nonsquare = enh.denoise(img_nonsquare, strength=0.8)
        assert out_nonsquare.shape == img_nonsquare.shape

        # Sub-tile (smaller than tile size)
        img_small = _make_noisy(180, 240, noise_std=10.0)
        out_small = enh.denoise(img_small, strength=0.8)
        assert out_small.shape == img_small.shape

    def test_denoise_rgb_order_regression(self) -> None:
        """Saturated red/blue regions stay dominant after denoise."""
        enh = AIEnhancer(tile=256, overlap=32)

        h, w = 200, 200
        # Create a test pattern: red on left, blue on right, mild noise
        pattern = np.zeros((h, w, 3), dtype=np.float32)
        pattern[:, :w // 2, 2] = 240.0  # Red channel (BGR indexing: [B, G, R])
        pattern[:, w // 2:, 0] = 240.0  # Blue channel
        pattern[:, :, 1] = 50.0  # Low green everywhere

        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 8.0, pattern.shape).astype(np.float32)
        noisy = np.clip(pattern + noise, 0.0, 255.0).astype(np.float32)

        denoised = enh.denoise(noisy, strength=1.0)

        # Check red region (left half): should stay red-dominant
        red_region_noisy = noisy[:, :w // 2, :]
        red_region_denoised = denoised[:, :w // 2, :]
        assert red_region_denoised[:, :, 2].mean() > red_region_denoised[:, :, 0].mean(), (
            "Red region should have R > B after denoise"
        )
        # Per-channel mean drift should be small (< 10 levels)
        red_drift = np.abs(red_region_denoised.mean() - red_region_noisy.mean())
        assert red_drift < 10.0, f"Red region mean drift too large: {red_drift:.2f}"

        # Check blue region (right half): should stay blue-dominant
        blue_region_noisy = noisy[:, w // 2:, :]
        blue_region_denoised = denoised[:, w // 2:, :]
        assert blue_region_denoised[:, :, 0].mean() > blue_region_denoised[:, :, 2].mean(), (
            "Blue region should have B > R after denoise"
        )
        blue_drift = np.abs(blue_region_denoised.mean() - blue_region_noisy.mean())
        assert blue_drift < 10.0, f"Blue region mean drift too large: {blue_drift:.2f}"
