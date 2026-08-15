"""Unit tests for CLI helpers and shared I/O utilities."""

from pathlib import Path

import numpy as np
import pytest

from cli import (
    _destination_for_image,
    _finalize_params,
    _preflight_destinations,
)
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


class TestDestinationSafety:
    def test_recursive_destination_preserves_relative_path(self, tmp_path):
        input_root = tmp_path / "input"
        source = input_root / "card-a" / "IMG_0001.jpg"
        output_root = tmp_path / "output"

        destination = _destination_for_image(
            source, output_root, "jpg", input_root=input_root,
        )

        assert destination == output_root / "card-a" / "IMG_0001.jpg"

    def test_preflight_rejects_flattened_duplicate_destinations(self, tmp_path):
        first = tmp_path / "card-a" / "IMG_0001.jpg"
        second = tmp_path / "card-b" / "IMG_0001.jpg"
        first.parent.mkdir()
        second.parent.mkdir()
        first.touch()
        second.touch()

        with pytest.raises(ValueError, match="Duplicate output destination"):
            _preflight_destinations(
                [first, second], tmp_path / "output", "jpg", 8,
                recursive_root=None, compare=False, save_session=None,
            )

    def test_preflight_rejects_same_format_source_overwrite(self, tmp_path):
        source = tmp_path / "photo.jpg"
        source.touch()

        with pytest.raises(ValueError, match="overwrite source"):
            _preflight_destinations(
                [source], None, "same", 8,
                recursive_root=None, compare=False, save_session=None,
            )

    def test_preflight_rejects_existing_hardlink_destination(self, tmp_path):
        source = tmp_path / "source.jpg"
        source.write_bytes(b"source bytes")
        output = tmp_path / "output"
        output.mkdir()
        destination = output / "source.jpg"
        try:
            destination.hardlink_to(source)
        except (AttributeError, NotImplementedError, OSError):
            pytest.skip("hard links are unavailable on this filesystem")

        with pytest.raises(ValueError, match="alias"):
            _preflight_destinations(
                [source], output, "jpg", 8,
                recursive_root=None, compare=False, save_session=None,
            )

    def test_recursive_preflight_rejects_nested_output_tree(self, tmp_path):
        input_root = tmp_path / "input"
        input_root.mkdir()
        source = input_root / "source.jpg"
        source.touch()

        with pytest.raises(ValueError, match="outside input tree"):
            _preflight_destinations(
                [source], input_root / "exports", "jpg", 8,
                recursive_root=input_root, compare=False, save_session=None,
            )


class TestImreadExif:
    def test_reads_jpeg(self, tmp_path):
        pytest.importorskip("cv2")
        import cv2

        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), np.full((16, 16, 3), 100, dtype=np.uint8))
        img = imread_exif(path)
        assert img.shape == (16, 16, 3)
        assert img.dtype == np.uint8
