"""StyleLibrary — dynamic preset management and dataset-wide style learning."""

from __future__ import annotations

import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

from .style import StyleProfile, StyleAnalyzer
from .io import imread_exif, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)

# Default styles directory relative to the package root or workspace root
DEFAULT_STYLE_DIR = Path(__file__).resolve().parent.parent / "styles"


def ensure_style_dir(directory: Path | str = DEFAULT_STYLE_DIR) -> Path:
    """Ensure the styles directory exists."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_style_profile(
    name: str,
    profile: StyleProfile | Dict[str, float],
    author: Optional[str] = None,
    version: str = "1.0",
    tags: Optional[List[str]] = None,
    directory: Path | str = DEFAULT_STYLE_DIR,
) -> Path:
    """Save a style profile with metadata. Auto-increments version and filename if it already exists."""
    style_dir = ensure_style_dir(directory)
    
    # Standardize profile to dictionary
    if isinstance(profile, StyleProfile):
        profile_dict = profile.to_dict()
    else:
        profile_dict = StyleProfile(**profile).to_dict()

    resolved_author = author or os.environ.get("RETOUCH_AUTHOR", "Default User")
    resolved_version = version

    # Generate base file name: lowercase with underscores
    base_filename = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    if not base_filename:
        base_filename = "untitled_style"

    filepath = style_dir / f"{base_filename}.json"

    # Auto-increment version and suffix if duplicate exists to protect history
    if filepath.exists():
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                old_meta = json.load(f)
            old_version = old_meta.get("version", "1.0")
            v_float = float(old_version)
            resolved_version = f"{v_float + 1.0:.1f}"
        except Exception:
            resolved_version = "2.0"
        
        filepath = style_dir / f"{base_filename}_v{resolved_version}.json"
        
        suffix_idx = 2
        while filepath.exists():
            filepath = style_dir / f"{base_filename}_v{resolved_version}_{suffix_idx}.json"
            suffix_idx += 1

    metadata = {
        "name": name,
        "author": resolved_author,
        "version": resolved_version,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "tags": tags or [],
        "profile": profile_dict,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Saved style profile '%s' (v%s) to %s", name, resolved_version, filepath)
    return filepath


def load_style_profile(filepath: Path | str) -> Dict[str, Any]:
    """Load a wrapped style profile with its metadata."""
    filepath = Path(filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def list_styles(directory: Path | str = DEFAULT_STYLE_DIR) -> List[Dict[str, Any]]:
    """List all custom style profiles in the styles directory with their metadata."""
    style_dir = ensure_style_dir(directory)
    styles = []
    
    for filepath in style_dir.glob("*.json"):
        try:
            meta = load_style_profile(filepath)
            meta["filepath"] = str(filepath)
            styles.append(meta)
        except Exception as e:
            logger.warning("Failed to load style file %s: %s", filepath, e)
            
    return styles


def normalize_stem(stem: str) -> str:
    """Normalize filename stems to match original/edited pairs with Lightroom-style suffixes."""
    s = stem.lower()
    # Strip suffixes like _edit, _edited, _retouched, _v1, _v2, _v3, _crop, _copy
    s = re.sub(r'_(edit|edited|retouched|v\d+|crop|copy)$', '', s)
    return s


def learn_dataset_style(
    original_dir: str | Path,
    edited_dir: str | Path,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[StyleProfile, int]:
    """Analyze a folder of original and edited images to extract an average style profile.

    Features robust stem-matching, outlier trimming, and zero-delta fallback exclusion.
    """
    orig_path = Path(original_dir)
    edit_path = Path(edited_dir)

    if not orig_path.exists() or not edit_path.exists():
        raise FileNotFoundError("Both original and edited directories must exist.")

    # 1. Index edited files by their normalized stems
    edited_index: Dict[str, Path] = {}
    for root, _, files in os.walk(edit_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                filepath = Path(root) / file
                norm_stem = normalize_stem(filepath.stem)
                edited_index[norm_stem] = filepath

    # 2. Gather original files
    original_files: List[Path] = []
    for root, _, files in os.walk(orig_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in IMAGE_EXTENSIONS:
                original_files.append(Path(root) / file)

    total_files = len(original_files)
    if total_files == 0:
        return StyleProfile(), 0

    analyzer = StyleAnalyzer()
    profiles: List[StyleProfile] = []
    matched_count = 0

    for idx, orig_file in enumerate(original_files):
        if progress_callback:
            progress_callback(
                0.1 + 0.8 * (idx / total_files),
                f"Analyzing pair {idx + 1}/{total_files}: {orig_file.name}...",
            )

        norm_stem = normalize_stem(orig_file.stem)
        edit_file = edited_index.get(norm_stem)

        if edit_file and edit_file.exists():
            try:
                orig_img = imread_exif(orig_file)
                edit_img = imread_exif(edit_file)

                profile = analyzer.extract(orig_img, edit_img)
                
                # Check for zero-delta profile (indicates face detection failed and returned fallback)
                is_zero = (
                    abs(profile.brightness_delta) < 1e-5 and
                    abs(profile.contrast_delta) < 1e-5 and
                    abs(profile.skin_smooth_strength) < 1e-5
                )

                if is_zero:
                    logger.warning("Skipping pair %s: zero-delta profile extracted (failed face detection)", orig_file.name)
                    continue

                profiles.append(profile)
                matched_count += 1
                logger.info("Successfully analyzed pair: %s <-> %s", orig_file.name, edit_file.name)
            except Exception as e:
                logger.warning("Failed to analyze pair %s: %s", orig_file.name, e)

    if not profiles:
        return StyleProfile(), 0

    if progress_callback:
        progress_callback(0.9, f"Averaging parameters across {len(profiles)} valid pairs...")

    # 3. Average parameters using a trimmed mean if N >= 5
    avg_params = {}
    field_keys = [
        "brightness_delta",
        "contrast_delta",
        "saturation_delta",
        "skin_l_mean_delta",
        "skin_a_mean_delta",
        "skin_b_mean_delta",
        "skin_smooth_strength",
        "skin_mid_reduction",
        "skin_texture_opacity",
    ]

    for key in field_keys:
        vals = [getattr(p, key) for p in profiles]
        if len(vals) >= 5:
            # Trim top and bottom 10% to eliminate outliers
            vals = sorted(vals)
            trim_count = max(1, int(len(vals) * 0.1))
            trimmed_vals = vals[trim_count:-trim_count]
            avg_params[key] = float(np.mean(trimmed_vals))
        else:
            avg_params[key] = float(np.mean(vals))

    if progress_callback:
        progress_callback(1.0, "Dataset learning completed!")

    return StyleProfile(**avg_params), matched_count
