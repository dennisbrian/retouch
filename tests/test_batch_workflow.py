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
