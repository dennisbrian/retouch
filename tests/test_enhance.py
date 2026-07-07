"""Tests for retouch/enhance.py — F7 AI denoise + super-resolution."""

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

    def test_uses_fallback_when_no_model(self) -> None:
        enh = AIEnhancer()
        assert enh._denoise_model() is None, (
            "no denoise model should be present in the test env"
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
