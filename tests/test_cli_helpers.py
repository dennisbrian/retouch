"""Unit tests for CLI helpers and shared I/O utilities."""

from pathlib import Path

import numpy as np
import pytest

from cli import _finalize_params
from retouch.io import imread_exif, output_format


class TestFinalizeParams:
    def test_color_ref_path_not_consumed_on_copy(self, tmp_path):
        ref_path = tmp_path / "ref.jpg"
        pytest.importorskip("cv2")
        import cv2

        cv2.imwrite(str(ref_path), np.full((8, 8, 3), 128, dtype=np.uint8))

        base = {"recipe": "natural", "color_ref_path": str(ref_path)}
        first = _finalize_params(base)
        second = _finalize_params(base)

        assert "color_ref" in first
        assert "color_ref_path" not in first
        assert base["color_ref_path"] == str(ref_path)
        assert "color_ref" in second
        assert np.array_equal(first["color_ref"], second["color_ref"])

    def test_without_color_ref_is_shallow_copy(self):
        base = {"recipe": "natural", "smooth": 50}
        job = _finalize_params(base)
        assert job == base
        assert job is not base


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
        for ext in (".jpg", ".jpeg", ".JPG"):
            path = tmp_path / f"photo{ext}"
            path.touch()
            assert output_format(path, "same") == "jpg"

    def test_explicit_format_overrides_extension(self, tmp_path):
        png = tmp_path / "photo.png"
        png.touch()
        assert output_format(png, "webp") == "webp"


class TestImreadExif:
    def test_reads_jpeg(self, tmp_path):
        pytest.importorskip("cv2")
        import cv2

        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), np.full((16, 16, 3), 100, dtype=np.uint8))
        img = imread_exif(path)
        assert img.shape == (16, 16, 3)
        assert img.dtype == np.uint8
