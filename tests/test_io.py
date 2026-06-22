"""Tests for retouch/io.py — I/O helper functions."""
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from PIL.ExifTags import Base as ExifBase
from retouch.io import (
    resize_for_processing,
    output_format,
    encode_write_params,
    make_comparison,
    copy_exif,
)


class TestResizeForProcessing:
    def test_none_max_dim(self):
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        result, scale = resize_for_processing(img, None)
        assert np.all(result == img)
        assert scale == 1.0

    def test_already_small_enough(self):
        img = np.full((100, 200, 3), 128, dtype=np.uint8)
        result, scale = resize_for_processing(img, 500)
        assert np.all(result == img)
        assert scale == 1.0

    def test_downscales_width(self):
        img = np.full((100, 400, 3), 128, dtype=np.uint8)
        result, scale = resize_for_processing(img, 200)
        assert result.shape[1] <= 200
        assert scale < 1.0

    def test_downscales_height(self):
        img = np.full((400, 100, 3), 128, dtype=np.uint8)
        result, scale = resize_for_processing(img, 200)
        assert result.shape[0] <= 200
        assert scale < 1.0

    def test_output_type(self):
        img = np.full((300, 300, 3), 128, dtype=np.uint8)
        result, scale = resize_for_processing(img, 100)
        assert result.dtype == np.uint8


class TestOutputFormat:
    def test_same_preserves_png(self, tmp_path):
        png = tmp_path / "photo.png"
        png.touch()
        assert output_format(png, "same") == "png"

    def test_same_preserves_webp(self, tmp_path):
        webp = tmp_path / "photo.webp"
        webp.touch()
        assert output_format(webp, "same") == "webp"

    def test_same_maps_jpeg_variants_to_jpg(self, tmp_path):
        for ext in (".jpg", ".jpeg", ".JPG", ".JPEG"):
            path = tmp_path / f"photo{ext}"
            path.touch()
            assert output_format(path, "same") == "jpg"

    def test_explicit_format_overrides_extension(self, tmp_path):
        png = tmp_path / "photo.png"
        png.touch()
        assert output_format(png, "webp") == "webp"

    def test_explicit_jpg(self, tmp_path):
        webp = tmp_path / "photo.webp"
        webp.touch()
        assert output_format(webp, "jpg") == "jpg"

    def test_explicit_png(self, tmp_path):
        jpg = tmp_path / "photo.jpg"
        jpg.touch()
        assert output_format(jpg, "png") == "png"


class TestEncodeWriteParams:
    def test_jpg(self):
        assert encode_write_params("jpg", 95) == [cv2.IMWRITE_JPEG_QUALITY, 95]

    def test_jpeg(self):
        assert encode_write_params("jpeg", 95) == [cv2.IMWRITE_JPEG_QUALITY, 95]

    def test_webp(self):
        assert encode_write_params("webp", 95) == [cv2.IMWRITE_WEBP_QUALITY, 95]

    def test_png_returns_empty(self):
        assert encode_write_params("png", 95) == []

    def test_unknown_returns_empty(self):
        assert encode_write_params("tiff", 95) == []


class TestMakeComparison:
    def test_none_original(self, tmp_path):
        import cv2
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        out = tmp_path / "compare.jpg"
        make_comparison(None, img, out, "jpg", 95)
        assert not out.exists()

    def test_none_retouched(self, tmp_path):
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        out = tmp_path / "compare.jpg"
        make_comparison(img, None, out, "jpg", 95)
        assert not out.exists()

    def test_writes_comparison(self, tmp_path):
        import cv2
        orig = np.full((50, 100, 3), 100, dtype=np.uint8)
        ret = np.full((50, 100, 3), 200, dtype=np.uint8)
        out = tmp_path / "compare.jpg"
        make_comparison(orig, ret, out, "jpg", 95)
        assert out.exists()
        img = cv2.imread(str(out))
        assert img.shape[0] == 50
        assert img.shape[1] == 100 + 4 + 100

    def test_resizes_mismatched_shapes(self, tmp_path):
        import cv2
        orig = np.full((50, 100, 3), 100, dtype=np.uint8)
        ret = np.full((60, 100, 3), 200, dtype=np.uint8)
        out = tmp_path / "compare.jpg"
        make_comparison(orig, ret, out, "jpg", 95)
        assert out.exists()

    def test_jpeg_quality(self, tmp_path):
        import cv2
        orig = np.full((30, 60, 3), 100, dtype=np.uint8)
        ret = np.full((30, 60, 3), 200, dtype=np.uint8)
        out = tmp_path / "compare.jpg"
        make_comparison(orig, ret, out, "jpg", 50)
        assert out.exists()

    def test_png_format(self, tmp_path):
        import cv2
        orig = np.full((30, 60, 3), 100, dtype=np.uint8)
        ret = np.full((30, 60, 3), 200, dtype=np.uint8)
        out = tmp_path / "compare.png"
        make_comparison(orig, ret, out, "png", 95)
        assert out.exists()


# ---------------------------------------------------------------------------
# copy_exif
# ---------------------------------------------------------------------------


def _write_jpeg_with_exif(path, orientation=1, make="TestCam", model="T1"):
    """Helper: write a small JPEG at *path* with EXIF metadata."""
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    arr[5:15, 5:15] = 200
    img = Image.fromarray(arr)
    exif = img.getexif()
    exif[ExifBase.Make] = make
    exif[ExifBase.Model] = model
    exif[ExifBase.Orientation] = orientation
    img.save(str(path), "JPEG", exif=exif.tobytes())


def _write_jpeg_without_exif(path):
    """Helper: write a small JPEG without any EXIF block."""
    arr = np.zeros((20, 20, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(str(path), "JPEG")


class TestCopyExif:
    def test_copies_exif_tags(self, tmp_path):
        src = tmp_path / "src.jpg"
        dst = tmp_path / "dst.jpg"
        _write_jpeg_with_exif(src)
        _write_jpeg_without_exif(dst)

        copy_exif(str(src), str(dst))

        # Re-read the destination and check the EXIF tags are present
        out = Image.open(str(dst))
        exif = out.getexif()
        assert exif.get(ExifBase.Make) == "TestCam"
        assert exif.get(ExifBase.Model) == "T1"

    def test_resets_orientation_to_normal(self, tmp_path):
        src = tmp_path / "src.jpg"
        dst = tmp_path / "dst.jpg"
        # Source has orientation=6 (rotated 90° CCW)
        _write_jpeg_with_exif(src, orientation=6)
        _write_jpeg_without_exif(dst)

        copy_exif(str(src), str(dst))

        out = Image.open(str(dst))
        exif = out.getexif()
        # Orientation is rewritten to 1 (normal) — the destination has its
        # own pixel orientation
        assert exif.get(ExifBase.Orientation) == 1

    def test_preserves_other_exif_tags(self, tmp_path):
        src = tmp_path / "src.jpg"
        dst = tmp_path / "dst.jpg"
        _write_jpeg_with_exif(src, make="CamA", model="M-100")
        _write_jpeg_without_exif(dst)

        copy_exif(str(src), str(dst))

        out = Image.open(str(dst))
        exif = out.getexif()
        # The destination is freshly written; tags that were explicitly
        # set on the source are transferred.
        assert exif.get(ExifBase.Make) == "CamA"
        assert exif.get(ExifBase.Model) == "M-100"

    def test_source_without_exif_is_silent_noop(self, tmp_path):
        src = tmp_path / "src_no_exif.jpg"
        dst = tmp_path / "dst.jpg"
        _write_jpeg_without_exif(src)
        _write_jpeg_without_exif(dst)

        # Should not raise
        copy_exif(str(src), str(dst))

        # Destination remains valid
        out = Image.open(str(dst))
        assert out.size == (20, 20)

    def test_handles_different_orientation_values(self, tmp_path):
        for orig_orientation in (1, 3, 6, 8):
            src = tmp_path / f"src_o{orig_orientation}.jpg"
            dst = tmp_path / f"dst_o{orig_orientation}.jpg"
            _write_jpeg_with_exif(src, orientation=orig_orientation)
            _write_jpeg_without_exif(dst)

            copy_exif(str(src), str(dst))

            out = Image.open(str(dst))
            exif = out.getexif()
            # All orientations are normalised to 1
            assert exif.get(ExifBase.Orientation) == 1

    def test_does_not_raise_on_invalid_source(self, tmp_path):
        # Source doesn't exist — should log a warning and return, not raise
        src = tmp_path / "nonexistent.jpg"
        dst = tmp_path / "dst.jpg"
        _write_jpeg_without_exif(dst)

        # Should not raise even when source is missing
        copy_exif(str(src), str(dst))


import cv2
