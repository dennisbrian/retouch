"""Unit tests for CLI helpers and shared I/O utilities."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from cli import (
    _check_disk_space,
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


class TestCheckDiskSpace:
    def _make_file(self, tmp_path, size_bytes):
        path = tmp_path / "input.jpg"
        # The disk check only needs the logical byte count.  Keep large-size
        # cases sparse so this regression suite never allocates gigabytes.
        with path.open("wb") as handle:
            handle.truncate(size_bytes)
        return path

    def test_exits_when_projected_free_space_too_low(self, tmp_path, monkeypatch, capsys):
        source = self._make_file(tmp_path, 1_000_000)

        class FakeUsage:
            free = 5_500_000  # just over 5GB min margin plus estimate below

        monkeypatch.setattr(shutil, "disk_usage", lambda _: FakeUsage())
        with pytest.raises(SystemExit) as exc_info:
            _check_disk_space([source], tmp_path, compare=False)
        assert exc_info.value.code == 1
        assert "Disk space warning" in capsys.readouterr().out

    def test_passes_when_plenty_of_free_space(self, tmp_path, monkeypatch, capsys):
        source = self._make_file(tmp_path, 1_000_000)

        class FakeUsage:
            free = 100 * 1024 ** 3  # 100 GiB

        monkeypatch.setattr(shutil, "disk_usage", lambda _: FakeUsage())
        _check_disk_space([source], tmp_path, compare=False)  # must not raise
        assert "Disk space warning" not in capsys.readouterr().out

    def test_compare_flag_doubles_estimate(self, tmp_path, monkeypatch):
        source = self._make_file(tmp_path, 2 * 1024 ** 3)  # 2 GiB

        # Free space that clears the no-compare estimate but not the
        # compare-doubled one, to prove `compare=True` changes the outcome.
        # no-compare: 2GiB * 2.0 = 4GiB estimate; compare: 8GiB estimate.
        class FakeUsage:
            free = 10 * 1024 ** 3  # 10 GiB

        monkeypatch.setattr(shutil, "disk_usage", lambda _: FakeUsage())
        _check_disk_space([source], tmp_path, compare=False)  # 10 - 4 = 6 >= 5, ok

        with pytest.raises(SystemExit):
            _check_disk_space([source], tmp_path, compare=True)  # 10 - 8 = 2 < 5

    def test_skips_silently_on_stat_failure(self, tmp_path, monkeypatch):
        missing = tmp_path / "does_not_exist.jpg"
        monkeypatch.setattr(
            shutil, "disk_usage",
            lambda _: (_ for _ in ()).throw(OSError("no such volume")),
        )
        _check_disk_space([missing], tmp_path, compare=False)  # must not raise

    def test_checks_volume_of_nearest_existing_ancestor(self, tmp_path, monkeypatch):
        # output_dir is created by the caller *after* this check runs, so a
        # fresh --output path won't exist yet. disk_usage() on a missing path
        # raises FileNotFoundError -- if _check_disk_space passed the
        # not-yet-created path straight through, that would be silently
        # swallowed by the OSError handler and never warn. Regression for
        # exactly that: only tmp_path itself exists on disk.
        source = self._make_file(tmp_path, 2 * 1024 ** 3)  # 2 GiB
        not_yet_created = tmp_path / "brand-new-output-dir"
        assert not not_yet_created.exists()

        seen_paths = []

        class FakeUsage:
            free = 1024  # far below any margin, to force the warning path

        def fake_disk_usage(path):
            seen_paths.append(Path(path))
            return FakeUsage()

        monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)
        with pytest.raises(SystemExit):
            _check_disk_space([source], not_yet_created, compare=False)

        assert seen_paths == [tmp_path]

    def test_in_place_conversion_checks_source_parent_volume(self, tmp_path, monkeypatch):
        source_dir = tmp_path / "photos"
        source_dir.mkdir()
        source = self._make_file(source_dir, 1_000_000)
        seen_paths = []

        class FakeUsage:
            free = 100 * 1024 ** 3

        def fake_disk_usage(path):
            seen_paths.append(Path(path))
            return FakeUsage()

        monkeypatch.setattr(shutil, "disk_usage", fake_disk_usage)
        _check_disk_space([source], None, compare=False)

        assert seen_paths == [source_dir]

    def test_in_place_sources_on_one_volume_share_one_estimate(self, tmp_path, monkeypatch):
        first_dir = tmp_path / "card-a"
        second_dir = tmp_path / "card-b"
        first_dir.mkdir()
        second_dir.mkdir()
        first = self._make_file(first_dir, 1024 ** 3)
        second = self._make_file(second_dir, 1024 ** 3)

        class FakeUsage:
            # Each source alone would leave 6 GiB after its 2 GiB estimate;
            # together they leave 4 GiB, below the 5 GiB safety margin.
            free = 8 * 1024 ** 3

        monkeypatch.setattr(shutil, "disk_usage", lambda _: FakeUsage())
        with pytest.raises(SystemExit):
            _check_disk_space([first, second], None, compare=False)


class TestImreadExif:
    def test_reads_jpeg(self, tmp_path):
        pytest.importorskip("cv2")
        import cv2

        path = tmp_path / "test.jpg"
        cv2.imwrite(str(path), np.full((16, 16, 3), 100, dtype=np.uint8))
        img = imread_exif(path)
        assert img.shape == (16, 16, 3)
        assert img.dtype == np.uint8
