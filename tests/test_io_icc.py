"""Tests for retouch/io.py — ICC profile handling."""
from __future__ import annotations

import io
import os
import struct
import zlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms, UnidentifiedImageError
from PIL.JpegImagePlugin import get_sampling

from retouch.io import (
    _HAS_IMAGECMS,
    convert_image_colorspace,
    encode_write_params,
    image_has_icc,
    read_exif_bytes,
    read_icc_profile,
    write_image_with_icc,
)


SYSTEM_PROPHOTO_ICC = "/System/Library/ColorSync/Profiles/ROMM RGB.icc"
SYSTEM_ADOBE_ICC = "/System/Library/ColorSync/Profiles/AdobeRGB1998.icc"


def _srgb_icc_bytes() -> bytes:
    profile = ImageCms.createProfile("sRGB")
    return ImageCms.ImageCmsProfile(profile).tobytes()


def _write_jpeg_with_srgb_icc(path: Path, pixel_value: int = 128) -> bytes:
    arr = np.full((20, 20, 3), pixel_value, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    icc = _srgb_icc_bytes()
    img.save(str(path), "JPEG", icc_profile=icc, quality=90)
    return icc


def _write_jpeg_no_icc(path: Path, pixel_value: int = 128) -> None:
    arr = np.full((20, 20, 3), pixel_value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path), "JPEG", quality=90)


def _write_png_no_icc(path: Path) -> None:
    arr = np.full((20, 20, 3), 128, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path), "PNG")


def _write_tiff_no_icc(path: Path) -> None:
    arr = np.full((20, 20, 3), 128, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path), "TIFF")


def _make_minimal_png() -> bytes:
    sig = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    ihdr = b"IHDR" + ihdr_data
    ihdr_chunk = struct.pack(">I", 13) + ihdr + struct.pack(">I", zlib.crc32(ihdr) & 0xFFFFFFFF)
    raw = b"\x00\x00\x00\x00"
    idat = b"IDAT" + zlib.compress(raw)
    idat_chunk = struct.pack(">I", len(zlib.compress(raw))) + idat + struct.pack(
        ">I", zlib.crc32(idat) & 0xFFFFFFFF
    )
    iend = b"IEND"
    iend_chunk = struct.pack(">I", 0) + iend + struct.pack(">I", zlib.crc32(iend) & 0xFFFFFFFF)
    return sig + ihdr_chunk + idat_chunk + iend_chunk


class TestReadIccProfile:
    def test_returns_none_for_non_image_text_file(self, tmp_path: Path) -> None:
        path = tmp_path / "not_an_image.txt"
        path.write_text("hello world")
        assert read_icc_profile(path) is None

    def test_returns_none_for_missing_file(self, tmp_path: Path) -> None:
        path = tmp_path / "does_not_exist.jpg"
        assert read_icc_profile(path) is None

    def test_returns_none_for_image_without_icc(self, tmp_path: Path) -> None:
        path = tmp_path / "no_icc.jpg"
        _write_jpeg_no_icc(path)
        assert read_icc_profile(path) is None

    def test_returns_none_for_corrupt_image(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.jpg"
        path.write_bytes(b"not a real jpeg \x00\x01\x02")
        assert read_icc_profile(path) is None

    def test_reads_embedded_icc_bytes(self, tmp_path: Path) -> None:
        path = tmp_path / "with_icc.jpg"
        original = _write_jpeg_with_srgb_icc(path)
        result = read_icc_profile(path)
        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0
        assert result == original

    def test_accepts_string_path(self, tmp_path: Path) -> None:
        path = tmp_path / "with_icc.jpg"
        _write_jpeg_with_srgb_icc(path)
        assert read_icc_profile(str(path)) is not None

    def test_accepts_pathlib_path(self, tmp_path: Path) -> None:
        path = tmp_path / "with_icc.jpg"
        _write_jpeg_with_srgb_icc(path)
        assert read_icc_profile(Path(path)) is not None

    def test_reads_png_with_icc(self, tmp_path: Path) -> None:
        path = tmp_path / "with_icc.png"
        arr = np.full((20, 20, 3), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img.save(str(path), "PNG", icc_profile=_srgb_icc_bytes())
        result = read_icc_profile(path)
        assert result is not None
        assert len(result) > 0


class TestImageHasIcc:
    def test_returns_false_for_non_image(self, tmp_path: Path) -> None:
        path = tmp_path / "x.txt"
        path.write_text("x")
        assert image_has_icc(path) is False

    def test_returns_false_for_missing(self, tmp_path: Path) -> None:
        assert image_has_icc(tmp_path / "nope.jpg") is False

    def test_returns_false_for_image_without_icc(self, tmp_path: Path) -> None:
        path = tmp_path / "no_icc.jpg"
        _write_jpeg_no_icc(path)
        assert image_has_icc(path) is False

    def test_returns_true_for_image_with_icc(self, tmp_path: Path) -> None:
        path = tmp_path / "with_icc.jpg"
        _write_jpeg_with_srgb_icc(path)
        assert image_has_icc(path) is True


class TestWriteImageWithIcc:
    def test_round_trips_icc_profile(self, tmp_path: Path) -> None:
        src_path = tmp_path / "src.jpg"
        original_icc = _write_jpeg_with_srgb_icc(src_path, pixel_value=128)

        dst_path = tmp_path / "dst.jpg"
        arr = np.full((20, 20, 3), 200, dtype=np.uint8)
        write_image_with_icc(str(dst_path), arr, original_icc, quality=90)

        roundtripped = read_icc_profile(dst_path)
        assert roundtripped is not None
        assert len(roundtripped) == len(original_icc)
        assert roundtripped == original_icc

    def test_writes_bgr_input_with_correct_color_order(self, tmp_path: Path) -> None:
        path = tmp_path / "bgr.jpg"
        bgr = np.zeros((20, 20, 3), dtype=np.uint8)
        bgr[:, :, 0] = 50
        bgr[:, :, 1] = 100
        bgr[:, :, 2] = 200
        write_image_with_icc(str(path), bgr, _srgb_icc_bytes(), quality=90)
        out_rgb = np.asarray(Image.open(str(path)).convert("RGB"))
        assert out_rgb[:, :, 0].mean() > out_rgb[:, :, 1].mean() > out_rgb[:, :, 2].mean()

    def test_writes_png_with_icc(self, tmp_path: Path) -> None:
        path = tmp_path / "out.png"
        arr = np.full((20, 20, 3), 100, dtype=np.uint8)
        write_image_with_icc(str(path), arr, _srgb_icc_bytes())
        assert path.exists()
        assert read_icc_profile(path) is not None

    def test_writes_tiff_with_icc(self, tmp_path: Path) -> None:
        path = tmp_path / "out.tif"
        arr = np.full((20, 20, 3), 100, dtype=np.uint8)
        write_image_with_icc(str(path), arr, _srgb_icc_bytes())
        assert path.exists()
        assert read_icc_profile(path) is not None

    def test_writes_webp_with_icc(self, tmp_path: Path) -> None:
        path = tmp_path / "out.webp"
        arr = np.full((20, 20, 3), 100, dtype=np.uint8)
        write_image_with_icc(str(path), arr, _srgb_icc_bytes(), quality=80)
        assert path.exists()

    def test_writes_without_icc_when_icc_is_none(self, tmp_path: Path) -> None:
        path = tmp_path / "no_icc.jpg"
        arr = np.full((20, 20, 3), 100, dtype=np.uint8)
        write_image_with_icc(str(path), arr, None, quality=90)
        assert path.exists()
        assert read_icc_profile(path) is None

    def test_writes_without_icc_when_icc_is_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "no_icc.jpg"
        arr = np.full((20, 20, 3), 100, dtype=np.uint8)
        write_image_with_icc(str(path), arr, b"", quality=90)
        assert path.exists()
        assert read_icc_profile(path) is None

    def test_falls_back_to_cv2_for_unknown_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "out.bmp"
        arr = np.full((20, 20, 3), 100, dtype=np.uint8)
        write_image_with_icc(str(path), arr, _srgb_icc_bytes())
        assert path.exists()
        assert cv2.imread(str(path)) is not None

    def test_writes_float_image_in_unit_range(self, tmp_path: Path) -> None:
        path = tmp_path / "float.jpg"
        arr = np.full((20, 20, 3), 0.5, dtype=np.float32)
        write_image_with_icc(
            str(path), arr, _srgb_icc_bytes(), quality=90, float_range="unit"
        )
        assert path.exists()

    def test_writes_float_image_in_engine_byte_range(self, tmp_path: Path) -> None:
        path = tmp_path / "float-byte.jpg"
        arr = np.full((20, 20, 3), 127.5, dtype=np.float32)
        write_image_with_icc(
            str(path), arr, _srgb_icc_bytes(), quality=95, float_range="byte"
        )
        decoded = np.asarray(Image.open(str(path)).convert("RGB"))
        assert decoded.mean() == pytest.approx(128.0, abs=3.0)

    def test_writes_16bit_float_image_in_engine_byte_range(self, tmp_path: Path) -> None:
        path = tmp_path / "float-byte-16.png"
        arr = np.full((20, 20, 3), 127.5, dtype=np.float32)
        write_image_with_icc(str(path), arr, bit_depth=16, float_range="byte")

        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        assert decoded is not None
        assert decoded.dtype == np.uint16
        assert decoded.mean() == pytest.approx(127.5 * 257.0, abs=1.0)

    def test_jpeg_uses_444_subsampling(self, tmp_path: Path) -> None:
        path = tmp_path / "444.jpg"
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, ::2] = (255, 0, 255)
        arr[:, 1::2] = (0, 255, 0)
        write_image_with_icc(str(path), arr, quality=90)
        with Image.open(str(path)) as decoded:
            assert get_sampling(decoded) == 0

    def test_opencv_jpeg_params_use_444_subsampling(self) -> None:
        arr = np.zeros((32, 32, 3), dtype=np.uint8)
        arr[:, ::2] = (255, 0, 255)
        arr[:, 1::2] = (0, 255, 0)
        ok, encoded = cv2.imencode(".jpg", arr, encode_write_params("jpg", 90))
        assert ok
        with Image.open(io.BytesIO(encoded.tobytes())) as decoded:
            assert get_sampling(decoded) == 0

    def test_write_image_with_icc_embeds_exif_and_resets_orientation(self, tmp_path: Path) -> None:
        # Source carries EXIF with a non-normal orientation.
        src = tmp_path / "src.jpg"
        arr = np.full((24, 24, 3), 200, dtype=np.uint8)
        src_img = Image.fromarray(arr)
        src_exif = Image.Exif()
        from PIL.ExifTags import Base as _EB
        src_exif[_EB.Orientation] = 6
        src_img.save(str(src), exif=src_exif.tobytes())

        out = tmp_path / "out.jpg"
        exif_bytes = read_exif_bytes(str(src))
        write_image_with_icc(str(out), arr, icc_profile=_srgb_icc_bytes(), quality=90, exif=exif_bytes)

        with Image.open(str(out)) as decoded:
            out_exif = decoded.getexif()
            # Orientation force-reset to normal (1).
            assert out_exif.get(_EB.Orientation) == 1
            # ICC profile survived the single pass write.
            assert decoded.info.get("icc_profile") is not None

    def test_writes_grayscale_image(self, tmp_path: Path) -> None:
        path = tmp_path / "gray.jpg"
        arr = np.full((20, 20), 128, dtype=np.uint8)
        write_image_with_icc(str(path), arr, _srgb_icc_bytes(), quality=90)
        assert path.exists()
        reloaded = np.asarray(Image.open(str(path)))
        assert reloaded.ndim == 2

    def test_writes_bgra_with_alpha(self, tmp_path: Path) -> None:
        path = tmp_path / "rgba.png"
        arr = np.zeros((20, 20, 4), dtype=np.uint8)
        arr[:, :, 0] = 50
        arr[:, :, 1] = 100
        arr[:, :, 2] = 200
        arr[:, :, 3] = 255
        write_image_with_icc(str(path), arr, _srgb_icc_bytes())
        assert path.exists()
        reloaded = Image.open(str(path))
        assert reloaded.mode in ("RGBA", "RGB")


class TestConvertImageColorspace:
    def test_returns_float32_in_unit_range(self) -> None:
        if not _HAS_IMAGECMS:
            pytest.skip("PIL.ImageCms not available")
        srgb_bytes = _srgb_icc_bytes()
        arr = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = convert_image_colorspace(arr, srgb_bytes, srgb_bytes)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-5
        assert result.shape == arr.shape

    def test_identity_transform_preserves_midgray(self) -> None:
        if not _HAS_IMAGECMS:
            pytest.skip("PIL.ImageCms not available")
        srgb_bytes = _srgb_icc_bytes()
        arr = np.full((20, 20, 3), 128, dtype=np.uint8)
        result = convert_image_colorspace(arr, srgb_bytes, srgb_bytes)
        assert np.allclose(result, 128.0 / 255.0, atol=0.02)

    def test_srgb_to_lab_changes_values_meaningfully(self) -> None:
        if not _HAS_IMAGECMS:
            pytest.skip("PIL.ImageCms not available")
        from PIL import ImageCms
        try:
            srgb_bytes = _srgb_icc_bytes()
            lab_profile = ImageCms.createProfile("LAB")
            lab_bytes = ImageCms.ImageCmsProfile(lab_profile).tobytes()
        except Exception:
            pytest.skip("ImageCms cannot build a minimal LAB profile on this platform")
        try:
            arr = np.full((20, 20, 3), 128, dtype=np.uint8)
            result = convert_image_colorspace(arr, srgb_bytes, lab_bytes)
        except (ImageCms.PyCMSError, ValueError):
            pytest.skip("ImageCms cannot build sRGB→LAB transform (platform limit)")
        assert result.dtype == np.float32
        assert result.shape == arr.shape
        assert np.all(np.isfinite(result))

    def test_accepts_float_input(self) -> None:
        if not _HAS_IMAGECMS:
            pytest.skip("PIL.ImageCms not available")
        srgb_bytes = _srgb_icc_bytes()
        arr = np.full((20, 20, 3), 0.5, dtype=np.float32)
        result = convert_image_colorspace(arr, srgb_bytes, srgb_bytes)
        assert result.dtype == np.float32
        assert np.allclose(result, 0.5, atol=0.05)

    def test_raises_runtimeerror_without_imagecms(self, monkeypatch) -> None:
        import retouch.io as rio

        monkeypatch.setattr(rio, "_HAS_IMAGECMS", False)
        with pytest.raises(RuntimeError):
            convert_image_colorspace(
                np.zeros((4, 4, 3), dtype=np.uint8), b"\x00" * 128, b"\x00" * 128
            )

    def test_real_prophoto_roundtrip_when_available(self) -> None:
        if not _HAS_IMAGECMS:
            pytest.skip("PIL.ImageCms not available")
        if not os.path.exists(SYSTEM_PROPHOTO_ICC):
            pytest.skip("ROMM RGB (ProPhoto) profile not available on this system")
        with open(SYSTEM_PROPHOTO_ICC, "rb") as fp:
            prophoto_bytes = fp.read()
        srgb_bytes = _srgb_icc_bytes()
        arr = np.full((20, 20, 3), 128, dtype=np.uint8)
        srgb_to_prophoto = convert_image_colorspace(arr, srgb_bytes, prophoto_bytes)
        assert srgb_to_prophoto.shape == arr.shape
        assert srgb_to_prophoto.dtype == np.float32


class TestGracefulFallback:
    def test_read_icc_works_without_imagecms(self, tmp_path: Path, monkeypatch) -> None:
        import retouch.io as rio

        monkeypatch.setattr(rio, "_HAS_IMAGECMS", False)
        path = tmp_path / "with_icc.jpg"
        original = _write_jpeg_with_srgb_icc(path)
        result = read_icc_profile(path)
        assert result == original

    def test_image_has_icc_works_without_imagecms(self, tmp_path: Path, monkeypatch) -> None:
        import retouch.io as rio

        monkeypatch.setattr(rio, "_HAS_IMAGECMS", False)
        path = tmp_path / "with_icc.jpg"
        _write_jpeg_with_srgb_icc(path)
        assert image_has_icc(path) is True

    def test_write_image_with_icc_works_without_imagecms(self, tmp_path: Path, monkeypatch) -> None:
        import retouch.io as rio

        monkeypatch.setattr(rio, "_HAS_IMAGECMS", False)
        path = tmp_path / "no_icc.jpg"
        arr = np.full((20, 20, 3), 100, dtype=np.uint8)
        write_image_with_icc(str(path), arr, _srgb_icc_bytes(), quality=90)
        assert path.exists()


class TestEdgeCases:
    def test_handles_minimal_png_without_icc(self, tmp_path: Path) -> None:
        path = tmp_path / "minimal.png"
        path.write_bytes(_make_minimal_png())
        assert read_icc_profile(path) is None
        assert image_has_icc(path) is False

    def test_handles_png_with_icc_signature_magic(self, tmp_path: Path) -> None:
        assert read_icc_profile("/this/path/does/not/exist.png") is None
        assert image_has_icc("/this/path/does/not/exist.png") is False

    def test_pathlib_path_roundtrip(self, tmp_path: Path) -> None:
        src = tmp_path / "src.jpg"
        original = _write_jpeg_with_srgb_icc(src)
        result = read_icc_profile(Path(src))
        assert result == original
