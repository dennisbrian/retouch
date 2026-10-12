"""Tests for retouch/style_library.py — dynamic preset management."""

import json
from pathlib import Path

import pytest

from retouch.style_library import (
    ensure_style_dir,
    normalize_stem,
    save_style_profile,
    load_style_profile,
    list_styles,
    learn_dataset_style,
)


class TestEnsureStyleDir:
    def test_creates_directory(self, tmp_path):
        d = tmp_path / "new_styles"
        assert not d.exists()
        result = ensure_style_dir(d)
        assert d.exists()
        assert d.is_dir()
        assert result == d

    def test_returns_path(self, tmp_path):
        result = ensure_style_dir(tmp_path)
        assert isinstance(result, Path)

    def test_no_error_if_exists(self, tmp_path):
        d = tmp_path / "existing"
        d.mkdir(parents=True)
        result = ensure_style_dir(d)
        assert result == d


class TestNormalizeStem:
    def test_lowercases(self):
        assert normalize_stem("DSC_1234") == "dsc_1234"

    def test_strips_edit_suffix(self):
        assert normalize_stem("portrait_edit") == "portrait"

    def test_strips_edited_suffix(self):
        assert normalize_stem("portrait_edited") == "portrait"

    def test_strips_retouched_suffix(self):
        assert normalize_stem("portrait_retouched") == "portrait"

    def test_strips_version_suffix(self):
        assert normalize_stem("portrait_v3") == "portrait"

    def test_strips_crop_suffix(self):
        assert normalize_stem("landscape_crop") == "landscape"

    def test_strips_copy_suffix(self):
        assert normalize_stem("landscape_copy") == "landscape"

    def test_no_suffix_unchanged(self):
        assert normalize_stem("original_photo") == "original_photo"

    def test_non_alpha_numeric_handling(self):
        assert normalize_stem("My Photo!_v2") == "my photo!"

    def test_empty_string(self):
        assert normalize_stem("") == ""

    def test_only_suffix_stripped(self):
        assert normalize_stem("edit") == "edit"

    def test_edit_inside_word_not_stripped(self):
        assert normalize_stem("editorial") == "editorial"


class TestSaveLoadRoundtrip:
    def test_save_and_load(self, tmp_path):
        path = save_style_profile(
            "test_style",
            {"brightness_delta": 5.0, "contrast_delta": 3.0},
            author="Tester",
            tags=["portrait", "warm"],
            directory=tmp_path,
        )
        assert path.exists()
        loaded = load_style_profile(path)
        assert loaded["name"] == "test_style"
        assert loaded["author"] == "Tester"
        assert loaded["version"] == "1.0"
        assert loaded["tags"] == ["portrait", "warm"]
        assert loaded["profile"]["brightness_delta"] == 5.0
        assert loaded["profile"]["contrast_delta"] == 3.0

    def test_roundtrip_keeps_all_fields(self, tmp_path):
        profile_data = {
            "brightness_delta": 2.5,
            "contrast_delta": -1.0,
            "saturation_delta": 8.0,
            "skin_l_mean_delta": 0.0,
            "skin_a_mean_delta": 0.0,
            "skin_b_mean_delta": 0.0,
            "skin_smooth_strength": 0.0,
            "skin_mid_reduction": 0.0,
            "skin_texture_opacity": 1.0,
        }
        path = save_style_profile("full_profile", profile_data, directory=tmp_path)
        loaded = load_style_profile(path)
        assert loaded["profile"] == profile_data

    def test_auto_increment_version(self, tmp_path):
        save_style_profile("dup_style", {"brightness_delta": 1.0}, directory=tmp_path)
        p2 = save_style_profile("dup_style", {"brightness_delta": 2.0}, directory=tmp_path)
        assert "v2.0" in p2.name or "_v2" in p2.name

    def test_filename_sanitized(self, tmp_path):
        p = save_style_profile("My Cool Style!!!", {"brightness_delta": 0.0}, directory=tmp_path)
        assert "my_cool_style" in p.name

    def test_default_author(self, tmp_path):
        p = save_style_profile("noauthor", {}, directory=tmp_path)
        loaded = load_style_profile(p)
        assert loaded["author"] == "Default User"

    def test_created_at_iso_format(self, tmp_path):
        p = save_style_profile("iso_test", {}, directory=tmp_path)
        loaded = load_style_profile(p)
        assert "created_at" in loaded
        assert "T" in loaded["created_at"]


class TestListStyles:
    def test_returns_empty_list_when_no_files(self, tmp_path):
        assert list_styles(tmp_path) == []

    def test_lists_saved_styles(self, tmp_path):
        save_style_profile("style_a", {"brightness_delta": 1.0}, directory=tmp_path)
        save_style_profile("style_b", {"brightness_delta": 2.0}, directory=tmp_path)
        styles = list_styles(tmp_path)
        assert len(styles) == 2
        names = {s["name"] for s in styles}
        assert names == {"style_a", "style_b"}

    def test_each_entry_has_filepath(self, tmp_path):
        save_style_profile("fp_test", {}, directory=tmp_path)
        styles = list_styles(tmp_path)
        assert "filepath" in styles[0]

    def test_skips_invalid_json(self, tmp_path):
        (tmp_path / "invalid.json").write_text("not json")
        save_style_profile("valid", {"brightness_delta": 0.0}, directory=tmp_path)
        styles = list_styles(tmp_path)
        assert len(styles) == 1


class TestLearnDatasetStyle:
    def test_raises_if_original_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            learn_dataset_style(tmp_path / "nonexistent", tmp_path)

    def test_raises_if_edited_dir_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            learn_dataset_style(tmp_path, tmp_path / "nonexistent")

    def test_empty_directories_return_empty_profile(self, tmp_path):
        orig = tmp_path / "original"
        edit = tmp_path / "edited"
        orig.mkdir(parents=True)
        edit.mkdir(parents=True)
        profile, count = learn_dataset_style(orig, edit)
        assert count == 0
        assert profile.brightness_delta == 0.0
        assert profile.contrast_delta == 0.0

    def test_no_image_files_returns_zero_count(self, tmp_path):
        orig = tmp_path / "original"
        edit = tmp_path / "edited"
        orig.mkdir(parents=True)
        edit.mkdir(parents=True)
        (orig / "readme.txt").write_text("not an image")
        (edit / "readme.txt").write_text("not an image")
        profile, count = learn_dataset_style(orig, edit)
        assert count == 0
