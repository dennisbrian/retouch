"""Atomic final-image writes in ``retouch.io``.

``cli.py`` resumes an interrupted batch by skipping any destination that
already exists, so a final output must never be observable half-written.
These tests pin that ``write_image_with_icc`` and ``make_comparison`` write
through a hidden same-directory temp and publish it with ``os.replace``, and
that the published bytes are identical to a direct (non-atomic) write.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from retouch import io as rio
from retouch.io import (
    _atomic_output_path,
    _embed_png_metadata,
    _to_uint16_delivery,
    _to_uint8_delivery,
    _bgr_to_pil,
    _resolve_float_range,
    _write_tiff_16bit,
    encode_write_params,
    make_comparison,
    write_image_with_icc,
)


@pytest.fixture
def img_u8() -> np.ndarray:
    rng = np.random.default_rng(1234)
    return rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)


@pytest.fixture
def img_f32() -> np.ndarray:
    rng = np.random.default_rng(99)
    return (rng.random((40, 56, 3), dtype=np.float32) * 255.0).astype(np.float32)


def _dir_names(d: Path) -> set[str]:
    return {p.name for p in d.iterdir()}


def _assert_no_temps(d: Path) -> None:
    leftovers = [p.name for p in d.iterdir() if ".tmp-" in p.name]
    assert leftovers == [], f"temp files left behind: {leftovers}"


# ---------------------------------------------------------------------------
# Byte-identity vs a direct (pre-change) write
# ---------------------------------------------------------------------------


def _direct_pil_write(path: Path, img: np.ndarray, pil_format: str, quality: int = 95) -> None:
    """Reproduce the pre-atomic 8-bit PIL branch writing straight to *path*."""
    img_u8 = _to_uint8_delivery(img, _resolve_float_range(img, "auto"))
    kwargs = {"format": pil_format}
    if pil_format in ("JPEG", "WEBP"):
        kwargs["quality"] = quality
    if pil_format == "JPEG":
        kwargs["subsampling"] = 0
    _bgr_to_pil(img_u8).save(str(path), **kwargs)


@pytest.mark.parametrize(
    "name,pil_format",
    [("out.jpg", "JPEG"), ("out.png", "PNG"), ("out.tif", "TIFF")],
)
def test_8bit_bytes_identical_to_direct_write(tmp_path, img_u8, name, pil_format):
    atomic_dir = tmp_path / "atomic"
    direct_dir = tmp_path / "direct"
    atomic_dir.mkdir()
    direct_dir.mkdir()

    write_image_with_icc(atomic_dir / name, img_u8, bit_depth=8, quality=95)
    _direct_pil_write(direct_dir / name, img_u8, pil_format, quality=95)

    assert (atomic_dir / name).read_bytes() == (direct_dir / name).read_bytes()
    assert _dir_names(atomic_dir) == {name}


def test_16bit_png_bytes_identical_to_direct_write(tmp_path, img_f32):
    atomic_dir = tmp_path / "atomic"
    direct_dir = tmp_path / "direct"
    atomic_dir.mkdir()
    direct_dir.mkdir()
    icc = b"\x00" * 128  # arbitrary bytes: iCCP embedding is a raw passthrough

    write_image_with_icc(atomic_dir / "o.png", img_f32, icc_profile=icc, bit_depth=16)

    # Pre-change sequence: encode directly, then re-open to add metadata.
    img_16 = _to_uint16_delivery(img_f32, _resolve_float_range(img_f32, "auto"))
    direct = direct_dir / "o.png"
    assert cv2.imwrite(str(direct), img_16)
    _embed_png_metadata(direct, icc, None)

    assert (atomic_dir / "o.png").read_bytes() == direct.read_bytes()
    assert _dir_names(atomic_dir) == {"o.png"}
    decoded = cv2.imread(str(atomic_dir / "o.png"), cv2.IMREAD_UNCHANGED)
    assert decoded.dtype == np.uint16


def test_16bit_tiff_bytes_identical_to_direct_write(tmp_path, img_f32):
    atomic_dir = tmp_path / "atomic"
    direct_dir = tmp_path / "direct"
    atomic_dir.mkdir()
    direct_dir.mkdir()

    write_image_with_icc(atomic_dir / "o.tif", img_f32, bit_depth=16)
    img_16 = _to_uint16_delivery(img_f32, _resolve_float_range(img_f32, "auto"))
    _write_tiff_16bit(direct_dir / "o.tif", img_16, None, None)

    assert (atomic_dir / "o.tif").read_bytes() == (direct_dir / "o.tif").read_bytes()
    assert _dir_names(atomic_dir) == {"o.tif"}


def test_make_comparison_bytes_identical_to_direct_write(tmp_path, img_u8):
    other = np.ascontiguousarray(img_u8[::-1])
    out = tmp_path / "a_compare.jpg"
    make_comparison(img_u8, other, out, "jpg", 90)

    sep = np.full((img_u8.shape[0], 4, 3), 200, dtype=np.uint8)
    direct = tmp_path / "direct.jpg"
    assert cv2.imwrite(str(direct), np.hstack([img_u8, sep, other]), encode_write_params("jpg", 90))

    assert out.read_bytes() == direct.read_bytes()
    _assert_no_temps(tmp_path)


# ---------------------------------------------------------------------------
# Temp naming / cleanup on success
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,bit_depth", [
    ("a.jpg", 8), ("a.png", 8), ("a.tif", 8), ("a.webp", 8),
    ("a.png", 16), ("a.tif", 16),
])
def test_no_temp_left_after_success(tmp_path, img_u8, name, bit_depth):
    write_image_with_icc(tmp_path / name, img_u8, bit_depth=bit_depth)
    assert _dir_names(tmp_path) == {name}


def test_temp_is_hidden_same_dir_and_keeps_extension(tmp_path):
    final = tmp_path / "DSCF1234.jpg"
    with _atomic_output_path(final) as tmp:
        assert tmp.parent == final.parent
        assert tmp.name.startswith(".DSCF1234.tmp-")
        assert tmp.suffix == ".jpg"
        assert tmp != final
        tmp.write_bytes(b"ok")
        assert not final.exists()
    assert final.read_bytes() == b"ok"
    assert _dir_names(tmp_path) == {"DSCF1234.jpg"}


# ---------------------------------------------------------------------------
# Failure mid-write: final path absent / untouched, temp removed
# ---------------------------------------------------------------------------


class _EncoderBoom(RuntimeError):
    pass


def _partial_pil_save(self, fp, *args, **kwargs):
    Path(fp).write_bytes(b"\xff\xd8\xff\xe0 truncated")
    raise _EncoderBoom("simulated encoder failure")


def _partial_cv2_imwrite(filename, *args, **kwargs):
    Path(filename).write_bytes(b"partial")
    raise _EncoderBoom("simulated encoder failure")


@pytest.mark.parametrize("exc_type", [_EncoderBoom, KeyboardInterrupt])
def test_pil_failure_leaves_no_final_and_no_temp(tmp_path, img_u8, monkeypatch, exc_type):
    def boom(self, fp, *a, **k):
        Path(fp).write_bytes(b"\xff\xd8 truncated")
        raise exc_type()

    monkeypatch.setattr(Image.Image, "save", boom)
    final = tmp_path / "x.jpg"
    with pytest.raises(exc_type):
        write_image_with_icc(final, img_u8)
    assert not final.exists()
    assert _dir_names(tmp_path) == set()


def test_16bit_png_failure_leaves_no_final_and_no_temp(tmp_path, img_f32, monkeypatch):
    monkeypatch.setattr(rio.cv2, "imwrite", _partial_cv2_imwrite)
    final = tmp_path / "x.png"
    with pytest.raises(_EncoderBoom):
        write_image_with_icc(final, img_f32, bit_depth=16)
    assert not final.exists()
    assert _dir_names(tmp_path) == set()


def test_16bit_png_metadata_failure_leaves_no_final(tmp_path, img_f32, monkeypatch):
    """The metadata pass re-opens the encoded file; a failure there must not
    publish the metadata-less (or truncated) intermediate."""
    def bad_embed(path, icc, exif):
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\n trunc")
        raise _EncoderBoom("metadata pass failed")

    monkeypatch.setattr(rio, "_embed_png_metadata", bad_embed)
    final = tmp_path / "x.png"
    with pytest.raises(_EncoderBoom):
        write_image_with_icc(final, img_f32, icc_profile=b"\x00" * 64, bit_depth=16)
    assert not final.exists()
    assert _dir_names(tmp_path) == set()


def test_c2pa_embed_runs_on_temp_before_publish(tmp_path, img_u8, monkeypatch):
    seen = []
    real = rio._embed_jpeg_c2pa_manifest

    def spy(path, manifest):
        seen.append(Path(path))
        assert not (tmp_path / "x.jpg").exists(), "final published before C2PA pass"
        real(path, manifest)

    monkeypatch.setattr(rio, "_embed_jpeg_c2pa_manifest", spy)
    write_image_with_icc(tmp_path / "x.jpg", img_u8, c2pa_manifest=b"JUMBFc2pa-test")
    assert len(seen) == 1 and seen[0].name.startswith(".x.tmp-")
    assert b"JUMBFc2pa-test" in (tmp_path / "x.jpg").read_bytes()
    assert _dir_names(tmp_path) == {"x.jpg"}


def test_existing_good_output_untouched_when_rewrite_fails(tmp_path, img_u8, monkeypatch):
    final = tmp_path / "keep.jpg"
    write_image_with_icc(final, img_u8)
    good = final.read_bytes()
    good_mtime = final.stat().st_mtime_ns

    monkeypatch.setattr(Image.Image, "save", _partial_pil_save)
    with pytest.raises(_EncoderBoom):
        write_image_with_icc(final, np.zeros_like(img_u8))

    assert final.read_bytes() == good
    assert final.stat().st_mtime_ns == good_mtime
    assert _dir_names(tmp_path) == {"keep.jpg"}


def test_make_comparison_failure_is_atomic(tmp_path, img_u8, monkeypatch):
    """make_comparison logs (does not raise) on failure; the final compare
    path must still be absent and no temp may remain."""
    monkeypatch.setattr(rio.cv2, "imwrite", _partial_cv2_imwrite)
    final = tmp_path / "a_compare.jpg"
    make_comparison(img_u8, img_u8, final, "jpg", 90)
    assert not final.exists()
    assert _dir_names(tmp_path) == set()


def test_make_comparison_imwrite_false_is_atomic(tmp_path, img_u8, monkeypatch):
    def false_imwrite(filename, *a, **k):
        Path(filename).write_bytes(b"partial")
        return False

    monkeypatch.setattr(rio.cv2, "imwrite", false_imwrite)
    final = tmp_path / "a_compare.jpg"
    make_comparison(img_u8, img_u8, final, "jpg", 90)
    assert not final.exists()
    assert _dir_names(tmp_path) == set()


def test_make_comparison_keeps_existing_on_failure(tmp_path, img_u8, monkeypatch):
    final = tmp_path / "a_compare.jpg"
    make_comparison(img_u8, img_u8, final, "jpg", 90)
    good = final.read_bytes()
    monkeypatch.setattr(rio.cv2, "imwrite", _partial_cv2_imwrite)
    make_comparison(img_u8, np.zeros_like(img_u8), final, "jpg", 90)
    assert final.read_bytes() == good
    assert _dir_names(tmp_path) == {"a_compare.jpg"}
