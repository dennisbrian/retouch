"""Tests for retouch/io.py 16-bit export/import support.

Covers:
  * ``write_image_16bit`` + ``read_image_16bit`` round-trip — float32 [0,255]
    in, write, read, assert max diff <= 1.0 (quantization budget for the
    round() used on write and float division on read).
  * uint8 input -> write_16bit -> read_16bit -> values upscaled correctly
    (x257 then /257 = identity within rounding).
  * Path traversal guard on ``write_image_16bit`` rejects escapes.
  * ``read_image_16bit`` raises FileNotFoundError for non-existent files.
  * Bad format / extension handling.

All temp files use the pytest ``tmp_path`` fixture.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest

from retouch.io import read_image_16bit, write_image_16bit


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=123)


@pytest.fixture
def float_img_255(rng: np.random.Generator) -> np.ndarray:
    """(H, W, 3) float32 BGR image in [0, 255]."""
    return rng.uniform(0.0, 255.0, size=(64, 64, 3)).astype(np.float32)


@pytest.fixture
def uint8_img(rng: np.random.Generator) -> np.ndarray:
    """(H, W, 3) uint8 BGR image."""
    return rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)


@pytest.fixture
def gradient_float_255() -> np.ndarray:
    """Smooth gradient exercising many quantization boundaries."""
    h, w = 32, 32
    yy = np.linspace(0.0, 255.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 255.0, w, dtype=np.float32)[None, :]
    base = (yy + xx) * 0.5
    img = np.stack([base, 255.0 - base, base * 0.5], axis=-1)
    return np.clip(img, 0.0, 255.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Round-trip: float32 [0,255] -> write -> read -> float32 [0,255]
# ---------------------------------------------------------------------------


class TestRoundTripFloat:
    def test_roundtrip_png(self, tmp_path, float_img_255):
        out_path = tmp_path / "out.png"
        write_image_16bit(out_path, float_img_255, format="png")
        read_back = read_image_16bit(out_path)
        assert read_back.dtype == np.float32
        assert read_back.shape == float_img_255.shape
        max_diff = np.abs(read_back - float_img_255).max()
        assert max_diff <= 1.0, f"PNG round-trip max diff {max_diff} > 1.0"

    def test_roundtrip_tiff(self, tmp_path, float_img_255):
        out_path = tmp_path / "out.tif"
        write_image_16bit(out_path, float_img_255, format="tiff")
        read_back = read_image_16bit(out_path)
        assert read_back.dtype == np.float32
        assert read_back.shape == float_img_255.shape
        max_diff = np.abs(read_back - float_img_255).max()
        assert max_diff <= 1.0, f"TIFF round-trip max diff {max_diff} > 1.0"

    def test_roundtrip_gradient(self, tmp_path, gradient_float_255):
        out_path = tmp_path / "grad.png"
        write_image_16bit(out_path, gradient_float_255, format="png")
        read_back = read_image_16bit(out_path)
        max_diff = np.abs(read_back - gradient_float_255).max()
        assert max_diff <= 1.0, f"gradient round-trip max diff {max_diff} > 1.0"

    def test_roundtrip_extremes(self, tmp_path):
        img = np.array([[[0.0, 0.0, 0.0], [255.0, 255.0, 255.0]]], dtype=np.float32)
        out_path = tmp_path / "extreme.png"
        write_image_16bit(out_path, img, format="png")
        read_back = read_image_16bit(out_path)
        max_diff = np.abs(read_back - img).max()
        assert max_diff <= 1.0

    def test_written_png_is_16bit(self, tmp_path, float_img_255):
        out_path = tmp_path / "out.png"
        write_image_16bit(out_path, float_img_255, format="png")
        raw = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
        assert raw is not None
        assert raw.dtype == np.uint16, f"expected uint16 on disk, got {raw.dtype}"


# ---------------------------------------------------------------------------
# uint8 input -> write_16bit -> read_16bit (upscaling path)
# ---------------------------------------------------------------------------


class TestUint8Upscale:
    def test_uint8_input_upscales_correctly(self, tmp_path, uint8_img):
        out_path = tmp_path / "u8.png"
        write_image_16bit(out_path, uint8_img, format="png")
        read_back = read_image_16bit(out_path)
        assert read_back.dtype == np.float32
        expected = uint8_img.astype(np.float32)
        max_diff = np.abs(read_back - expected).max()
        assert max_diff <= 1.0, f"uint8 upscale round-trip max diff {max_diff} > 1.0"

    def test_uint8_input_logs_warning(self, tmp_path, uint8_img, caplog):
        out_path = tmp_path / "u8_warn.png"
        with caplog.at_level(logging.WARNING, logger="retouch.io"):
            write_image_16bit(out_path, uint8_img, format="png")
        assert any("upscaled" in r.message for r in caplog.records), \
            "uint8 input should log an upscale warning"

    def test_uint8_disk_values_are_x257(self, tmp_path, uint8_img):
        out_path = tmp_path / "u8_disk.png"
        write_image_16bit(out_path, uint8_img, format="png")
        raw = cv2.imread(str(out_path), cv2.IMREAD_UNCHANGED)
        assert raw.dtype == np.uint16
        expected = uint8_img.astype(np.uint16) * np.uint16(257)
        assert np.array_equal(raw, expected)

    def test_uint8_roundtrip_identity_through_float(self, tmp_path, uint8_img):
        out_path = tmp_path / "u8_id.png"
        write_image_16bit(out_path, uint8_img, format="png")
        read_back = read_image_16bit(out_path)
        back_as_u8 = np.clip(read_back + 0.5, 0, 255).astype(np.uint8)
        assert np.array_equal(back_as_u8, uint8_img), \
            "uint8 -> 16bit -> float32 -> uint8 should be identity"


# ---------------------------------------------------------------------------
# Path traversal guard
#
# Path traversal guard — the guard in io._resolve_safe_path now rejects
# system paths (/etc, /usr, /bin, /System, etc.) when no base_dir is provided.
# ---------------------------------------------------------------------------


class TestPathTraversal:
    def test_write_rejects_traversal(self, tmp_path, float_img_255):
        bad_path = "../../../etc/passwd"
        with pytest.raises(ValueError, match="traversal"):
            write_image_16bit(bad_path, float_img_255)

    def test_write_rejects_absolute_escape(self, tmp_path, float_img_255):
        bad_path = "/etc/passwd"
        with pytest.raises(ValueError, match="traversal"):
            write_image_16bit(bad_path, float_img_255)

    def test_write_accepts_safe_path(self, tmp_path, float_img_255):
        out_path = tmp_path / "safe.png"
        write_image_16bit(out_path, float_img_255)
        assert out_path.exists()

    def test_read_rejects_traversal(self):
        bad_path = "../../../etc/passwd"
        with pytest.raises(ValueError, match="traversal"):
            read_image_16bit(bad_path)

    def test_write_inside_tmp_path_subdir_works(self, tmp_path, float_img_255):
        sub = tmp_path / "sub"
        sub.mkdir()
        out_path = sub / "nested.png"
        write_image_16bit(out_path, float_img_255)
        assert out_path.exists()


# ---------------------------------------------------------------------------
# Bad-input handling
# ---------------------------------------------------------------------------


class TestBadInput:
    def test_read_nonexistent_raises(self, tmp_path):
        missing = tmp_path / "does_not_exist.png"
        with pytest.raises(FileNotFoundError):
            read_image_16bit(missing)

    def test_read_nonexistent_traversal_safe(self):
        with pytest.raises((ValueError, FileNotFoundError)):
            read_image_16bit("../../../nonexistent/file.png")

    def test_write_rejects_bad_format(self, tmp_path, float_img_255):
        out_path = tmp_path / "out.jpg"
        with pytest.raises(ValueError, match="format"):
            write_image_16bit(out_path, float_img_255, format="jpeg")

    def test_write_rejects_bad_extension(self, tmp_path, float_img_255):
        out_path = tmp_path / "out.bmp"
        with pytest.raises(ValueError, match="extension"):
            write_image_16bit(out_path, float_img_255, format="png")

    def test_write_rejects_unsupported_dtype(self, tmp_path):
        img = np.full((8, 8, 3), 0, dtype=np.int32)
        out_path = tmp_path / "bad.png"
        with pytest.raises(TypeError):
            write_image_16bit(out_path, img, format="png")

    def test_write_uint16_passthrough(self, tmp_path, rng):
        img_u16 = rng.integers(0, 65536, size=(16, 16, 3), dtype=np.uint16)
        out_path = tmp_path / "u16.png"
        write_image_16bit(out_path, img_u16, format="png")
        read_back = read_image_16bit(out_path)
        max_diff = np.abs(read_back - img_u16.astype(np.float32) / 257.0).max()
        assert max_diff <= 1.0


# ---------------------------------------------------------------------------
# Grayscale / alpha handling
# ---------------------------------------------------------------------------


class TestChannelHandling:
    def test_grayscale_roundtrip(self, tmp_path, rng):
        gray = rng.uniform(0.0, 255.0, size=(32, 32)).astype(np.float32)
        out_path = tmp_path / "gray.png"
        write_image_16bit(out_path, gray, format="png")
        read_back = read_image_16bit(out_path)
        assert read_back.ndim == 3 and read_back.shape[2] == 3
        max_diff = max(
            np.abs(read_back[:, :, c] - gray).max() for c in range(3)
        )
        assert max_diff <= 1.0

    def test_float64_input_accepted(self, tmp_path, float_img_255):
        img64 = float_img_255.astype(np.float64)
        out_path = tmp_path / "f64.png"
        write_image_16bit(out_path, img64, format="png")
        read_back = read_image_16bit(out_path)
        max_diff = np.abs(read_back - float_img_255).max()
        assert max_diff <= 1.0
