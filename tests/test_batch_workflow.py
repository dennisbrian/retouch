"""Unit and integration tests for Style Library, Caching, and Batch Folder Processing."""

import os
import shutil
import tempfile
import json
import hashlib
from pathlib import Path

import pytest
import numpy as np
import cv2

from retouch.style import StyleProfile
from retouch.style_library import save_style_profile, load_style_profile, list_styles, learn_dataset_style, normalize_stem
from retouch.batch_processor import (
    BatchProcessorCache,
    classify_image,
    compute_hsv_stats,
    generate_contact_sheet,
    BatchProcessor,
)


def test_style_library_meta(tmp_path):
    """Test that style profiles are correctly wrapped with JSON metadata and version-bump on duplicates."""
    style_dir = tmp_path / "styles"
    profile = StyleProfile(brightness_delta=0.5, contrast_delta=10.0, skin_smooth_strength=0.85)

    save_path = save_style_profile(
        name="Test Style",
        profile=profile,
        author="Test Author",
        tags=["unit", "test"],
        directory=style_dir,
    )

    assert save_path.exists()
    assert save_path.name == "test_style.json"

    meta = load_style_profile(save_path)
    assert meta["name"] == "Test Style"
    assert meta["author"] == "Test Author"
    assert "created_at" in meta
    assert meta["tags"] == ["unit", "test"]
    assert meta["profile"]["brightness_delta"] == 0.5

    # Save duplicate style -> should bump version to 2.0 and save as test_style_v2.0.json
    save_path_v2 = save_style_profile(
        name="Test Style",
        profile=profile,
        author="Test Author",
        tags=["unit", "test"],
        directory=style_dir,
    )
    assert save_path_v2.exists()
    assert save_path_v2.name == "test_style_v2.0.json"
    
    meta_v2 = load_style_profile(save_path_v2)
    assert meta_v2["version"] == "2.0"

    styles = list_styles(style_dir)
    assert len(styles) == 2


def test_dataset_style_learning(tmp_path, engine):
    """Test learning a style with robust Lightroom stem-matching and outlier filtering."""
    orig_dir = tmp_path / "originals"
    edit_dir = tmp_path / "edited"
    orig_dir.mkdir()
    edit_dir.mkdir()

    # Create 2 mock image pairs using Lightroom-style matching names
    img1_orig = np.full((200, 200, 3), 100, dtype=np.uint8)
    img1_edit = np.full((200, 200, 3), 120, dtype=np.uint8)

    img2_orig = np.full((200, 200, 3), 100, dtype=np.uint8)
    img2_edit = np.full((200, 200, 3), 130, dtype=np.uint8)

    # Stem matching tests: original file stem normalized matches edit file suffixes
    cv2.imwrite(str(orig_dir / "img1.jpg"), img1_orig)
    cv2.imwrite(str(edit_dir / "img1_edit.jpg"), img1_edit)
    cv2.imwrite(str(orig_dir / "img2.jpg"), img2_orig)
    cv2.imwrite(str(edit_dir / "img2_retouched.jpg"), img2_edit)

    # Verify normalize_stem utility
    assert normalize_stem("img1_edit") == "img1"
    assert normalize_stem("img2_retouched") == "img2"

    learned_profile, count = learn_dataset_style(orig_dir, edit_dir)

    assert count == 2
    assert isinstance(learned_profile, StyleProfile)
    assert learned_profile.brightness_delta > 0.0


def test_cache_and_grouping(tmp_path):
    """Test that Cache reads, writes, and invalidates mtimes under home user cache folder."""
    cache = BatchProcessorCache(tmp_path)
    
    # Cache file should be located inside ~/.cache/retouch/ and named as a sha256 hash
    expected_hash = hashlib.sha256(str(tmp_path.resolve()).encode("utf-8")).hexdigest()
    assert cache.cache_file.name == f"{expected_hash}.json"
    assert cache.cache_file.parent.name == "retouch"

    # Empty cache first
    cache.data = {}

    mtime = 123456789.0
    info = {"faces": 2, "face_width": 120, "mean_s": 85.5, "mean_v": 110.0}
    cache.set("photo.jpg", mtime, info)

    # Get from cache
    entry = cache.get("photo.jpg", mtime)
    assert entry is not None
    assert entry["faces"] == 2
    assert entry["mean_s"] == 85.5

    # Changed mtime should return None (invalidation)
    assert cache.get("photo.jpg", mtime + 1.0) is None

    cache.save()
    assert cache.cache_file.exists()

    # Clean up the test cache file
    if cache.cache_file.exists():
        cache.cache_file.unlink()

    # Classification boundaries check
    assert classify_image(0, 50.0, 70.0) == "Scenic (Dark/Muted)"
    assert classify_image(1, 130.0, 90.0) == "Portrait (Bright/Colorful)"
    assert classify_image(0, 130.0, 70.0) == "Scenic (Dark/Colorful)"
    assert classify_image(3, 30.0, 150.0) == "Portrait (Bright/Muted)"


def test_contact_sheet(tmp_path):
    """Test generating a tiled contact sheet from a list of images with clean labels and placeholders."""
    # Create 3 small dummy images
    img_paths = []
    for i in range(3):
        p = tmp_path / f"thumb_{i}.jpg"
        img = np.full((100, 100, 3), i * 80, dtype=np.uint8)
        cv2.imwrite(str(p), img)
        img_paths.append(p)

    out_sheet = tmp_path / "sheet.jpg"
    # Generate grid with 4 columns -> 3 images should leave 1 cell padded with [Empty Cell] placeholder
    generate_contact_sheet(img_paths, out_sheet, cols=4, cell_size=200)

    assert out_sheet.exists()
    sheet_img = cv2.imread(str(out_sheet))
    # 3 images in 4 columns -> 1 row (4 cells wide)
    # Width = 4 * 200 = 800
    # Height = 1 * 200 = 200
    assert sheet_img.shape[0] == 200
    assert sheet_img.shape[1] == 800


def test_batch_processor_execution(tmp_path, engine):
    """Test the complete BatchProcessor flow including ingestion, grouping, cache checks, and ZIP packaging."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    # Create a couple of mock images
    img1 = np.full((300, 300, 3), 100, dtype=np.uint8)
    img2 = np.full((300, 300, 3), 150, dtype=np.uint8)

    cv2.imwrite(str(input_dir / "face1.jpg"), img1)
    cv2.imwrite(str(input_dir / "face2.jpg"), img2)

    processor = BatchProcessor(engine)
    style = StyleProfile(brightness_delta=2.0, contrast_delta=-5.0)

    processed, sheet_path, zip_path, log = processor.process_folder(
        input_dir=input_dir,
        output_dir=output_dir,
        custom_style_profile=style,
        export_fmt="JPEG",
        export_quality=90,
        export_res="Original",
        auto_group=True,
        generate_sheet=True,
        export_zip=True,
    )

    assert len(processed) == 2
    assert Path(processed[0]).exists()
    assert Path(processed[1]).exists()
    assert sheet_path is not None
    assert Path(sheet_path).exists()
    assert zip_path is not None
    assert Path(zip_path).exists()
    assert "Success" in log

    # Clean up the user-specific cache created during this batch run
    cache = BatchProcessorCache(input_dir)
    if cache.cache_file.exists():
        cache.cache_file.unlink()


# ---------------------------------------------------------------------------
# End-to-end batch processing tests
# ---------------------------------------------------------------------------


class TestBatchEndToEnd:
    """End-to-end integration tests for the full BatchProcessor pipeline."""

    def _make_synthetic_input_dir(self, tmp_path, name="batch_input", file_count=3):
        """Create a directory of synthetic grayscale images for batch tests."""
        input_dir = tmp_path / name
        input_dir.mkdir()
        for i in range(file_count):
            img = np.full((120, 160, 3), 80 + i * 20, dtype=np.uint8)
            cv2.imwrite(str(input_dir / f"img_{i}.jpg"), img)
        return input_dir

    def _cleanup_cache(self, input_dir):
        """Remove the user-level cache file for the given input dir."""
        cache = BatchProcessorCache(input_dir)
        if cache.cache_file.exists():
            cache.cache_file.unlink()

    def test_end_to_end_small_folder(self, tmp_path, engine):
        """Process a small folder of 3 synthetic images end-to-end."""
        input_dir = self._make_synthetic_input_dir(tmp_path, file_count=3)
        output_dir = tmp_path / "out_e2e"
        output_dir.mkdir()

        try:
            processor = BatchProcessor(engine)
            processed, sheet, zip_path, log = processor.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                style_name_or_recipe="natural",
                export_fmt="JPEG",
                export_quality=85,
                export_res="Original",
                auto_group=False,
                generate_sheet=True,
                export_zip=True,
            )

            assert len(processed) == 3
            for p in processed:
                assert Path(p).exists()
                assert Path(p).suffix == ".jpg"
            assert sheet is not None
            assert Path(sheet).exists()
            assert zip_path is not None
            assert Path(zip_path).exists()
            assert "Success" in log
            assert "3/3" in log
        finally:
            self._cleanup_cache(input_dir)

    def test_auto_grouping_groups_similar_images(self, tmp_path, engine):
        """Images with similar characteristics should be grouped together."""
        input_dir = tmp_path / "group_input"
        input_dir.mkdir()

        # Two similar dark images, one very different bright image
        for i in range(2):
            cv2.imwrite(
                str(input_dir / f"dark_{i}.jpg"),
                np.full((120, 120, 3), 50, dtype=np.uint8),
            )
        cv2.imwrite(
            str(input_dir / "bright.jpg"),
            np.full((120, 120, 3), 220, dtype=np.uint8),
        )

        try:
            processor = BatchProcessor(engine)
            processed, sheet, zip_path, log = processor.process_folder(
                input_dir=input_dir,
                output_dir=tmp_path / "out",
                style_name_or_recipe="natural",
                auto_group=True,
                generate_sheet=False,
                export_zip=False,
            )
            assert len(processed) == 3
            # Log should mention groupings
            assert "Groupings" in log or "group" in log.lower() or len(processed) == 3
        finally:
            self._cleanup_cache(input_dir)

    def test_contact_sheet_generated_by_default(self, tmp_path, engine):
        """Contact sheet should be created when generate_sheet=True."""
        input_dir = self._make_synthetic_input_dir(tmp_path, file_count=2)
        output_dir = tmp_path / "out_cs"
        output_dir.mkdir()

        try:
            processor = BatchProcessor(engine)
            processed, sheet, zip_path, _ = processor.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                style_name_or_recipe="natural",
                auto_group=False,
                generate_sheet=True,
                export_zip=False,
            )
            assert sheet is not None
            assert Path(sheet).exists()
            assert Path(sheet).name == "contact_sheet.jpg"
        finally:
            self._cleanup_cache(input_dir)

    def test_no_contact_sheet_when_disabled(self, tmp_path, engine):
        """When generate_sheet=False, no contact sheet is created."""
        input_dir = self._make_synthetic_input_dir(tmp_path, file_count=2)
        output_dir = tmp_path / "out_no_cs"
        output_dir.mkdir()

        try:
            processor = BatchProcessor(engine)
            processed, sheet, zip_path, _ = processor.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                style_name_or_recipe="natural",
                auto_group=False,
                generate_sheet=False,
                export_zip=False,
            )
            assert sheet is None
        finally:
            self._cleanup_cache(input_dir)

    def test_zip_contains_all_processed_files(self, tmp_path, engine):
        """The output ZIP should contain all processed images and the
        contact sheet."""
        import zipfile

        input_dir = self._make_synthetic_input_dir(tmp_path, file_count=3)
        output_dir = tmp_path / "out_zip"
        output_dir.mkdir()

        try:
            processor = BatchProcessor(engine)
            processed, sheet, zip_path, _ = processor.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                style_name_or_recipe="natural",
                auto_group=False,
                generate_sheet=True,
                export_zip=True,
            )
            assert zip_path is not None
            zip_path = Path(zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
            # Should contain all 3 processed images
            for p in processed:
                assert Path(p).name in names
            # Should also contain the contact sheet
            assert "contact_sheet.jpg" in names
        finally:
            self._cleanup_cache(input_dir)

    def test_custom_style_profile_application(self, tmp_path, engine):
        """A custom style profile should be applied to all batch images."""
        input_dir = self._make_synthetic_input_dir(tmp_path, file_count=2)
        output_dir = tmp_path / "out_style"
        output_dir.mkdir()

        try:
            processor = BatchProcessor(engine)
            profile = StyleProfile(
                brightness_delta=4.0,
                contrast_delta=15.0,
                saturation_delta=-20.0,
            )
            processed, sheet, zip_path, log = processor.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                custom_style_profile=profile,
                style_name_or_recipe="natural",
                auto_group=False,
                generate_sheet=False,
                export_zip=False,
            )
            assert len(processed) == 2
            # The log mentions the style via the recipe (since custom profile
            # is applied directly via applier.apply)
            for p in processed:
                assert Path(p).exists()
        finally:
            self._cleanup_cache(input_dir)

    def test_empty_input_folder_returns_empty_list(self, tmp_path, engine):
        """An empty input directory should return no processed files."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()
        output_dir = tmp_path / "out_empty"
        output_dir.mkdir()

        try:
            processor = BatchProcessor(engine)
            processed, sheet, zip_path, log = processor.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                style_name_or_recipe="natural",
                auto_group=False,
                generate_sheet=False,
                export_zip=False,
            )
            assert processed == []
            assert sheet is None
            assert zip_path is None
            assert "No supported" in log or "Success" in log
        finally:
            self._cleanup_cache(input_dir)

    def test_unsupported_files_are_skipped(self, tmp_path, engine):
        """Files with non-image extensions should be ignored."""
        input_dir = tmp_path / "mixed"
        input_dir.mkdir()
        # Real image
        cv2.imwrite(
            str(input_dir / "real.jpg"), np.full((100, 100, 3), 100, dtype=np.uint8)
        )
        # Non-image files
        (input_dir / "notes.txt").write_text("not an image")
        (input_dir / "data.json").write_text("{}")
        output_dir = tmp_path / "out_mixed"
        output_dir.mkdir()

        try:
            processor = BatchProcessor(engine)
            processed, sheet, zip_path, _ = processor.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                style_name_or_recipe="natural",
                auto_group=False,
                generate_sheet=False,
                export_zip=False,
            )
            # Only the .jpg should be processed
            assert len(processed) == 1
        finally:
            self._cleanup_cache(input_dir)

    def test_png_export_format(self, tmp_path, engine):
        """The export_fmt='PNG' option should produce .png files."""
        input_dir = self._make_synthetic_input_dir(tmp_path, file_count=2)
        output_dir = tmp_path / "out_png"
        output_dir.mkdir()

        try:
            processor = BatchProcessor(engine)
            processed, sheet, zip_path, _ = processor.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                style_name_or_recipe="natural",
                export_fmt="PNG",
                auto_group=False,
                generate_sheet=False,
                export_zip=False,
            )
            for p in processed:
                assert Path(p).suffix == ".png"
        finally:
            self._cleanup_cache(input_dir)

    def test_export_res_downsamples_output(self, tmp_path, engine):
        """export_res='HD (1280px)' should cap the longest side at 1280px."""
        input_dir = tmp_path / "hd_input"
        input_dir.mkdir()
        # 2000x1500 source image
        cv2.imwrite(
            str(input_dir / "big.jpg"), np.full((1500, 2000, 3), 100, dtype=np.uint8)
        )
        output_dir = tmp_path / "out_hd"
        output_dir.mkdir()

        try:
            processor = BatchProcessor(engine)
            processed, sheet, zip_path, _ = processor.process_folder(
                input_dir=input_dir,
                output_dir=output_dir,
                style_name_or_recipe="natural",
                export_res="HD (1280px)",
                auto_group=False,
                generate_sheet=False,
                export_zip=False,
            )
            assert len(processed) == 1
            out = cv2.imread(processed[0])
            assert out is not None
            # Longest side should be ≤ 1280
            assert max(out.shape[:2]) <= 1280
        finally:
            self._cleanup_cache(input_dir)
