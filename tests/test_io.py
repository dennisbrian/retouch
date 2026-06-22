"""Tests for retouch/io.py — I/O helper functions."""
from pathlib import Path
import numpy as np
import pytest
from retouch.io import (
    resize_for_processing,
    output_format,
    encode_write_params,
    make_comparison,
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


import cv2
