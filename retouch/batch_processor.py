"""BatchProcessor — directory walking, cache orchestration, descriptive grouping, and contact sheets."""

from __future__ import annotations

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .style import StyleProfile
from .io import imread_exif, IMAGE_EXTENSIONS

logger = logging.getLogger(__name__)


class BatchProcessorCache:
    """Persistent JSON cache for face counts, sizes, and color statistics.

    Stored under ~/.cache/retouch/ to prevent write errors on read-only folders.
    Keyed by the SHA-256 hash of the resolved input directory.
    """

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir
        self.cache_dir = Path.home() / ".cache" / "retouch"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        input_hash = hashlib.sha256(str(input_dir.resolve()).encode("utf-8")).hexdigest()
        self.cache_file = self.cache_dir / f"{input_hash}.json"
        
        self.data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception as e:
                logger.warning("Failed to load cache file %s: %s", self.cache_file, e)

    def save(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save cache file %s: %s", self.cache_file, e)

    def get(self, filename: str, mtime: float) -> Optional[Dict[str, Any]]:
        entry = self.data.get(filename)
        if entry and entry.get("mtime") == mtime:
            return entry
        return None

    def set(self, filename: str, mtime: float, info: Dict[str, Any]) -> None:
        info["mtime"] = mtime
        self.data[filename] = info


def compute_hsv_stats(img_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Compute average Saturation (S) and Value (V) channels, restricted to the subject mask if available."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    if mask is not None:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        if mask_f.ndim == 3:
            mask_f = mask_f[:, :, 0]
            
        indices = mask_f > 0.1
        if np.sum(indices) > 20:
            mean_s = float(np.mean(s[indices]))
            mean_v = float(np.mean(v[indices]))
            return mean_s, mean_v

    # Fallback to full-frame averages
    mean_s = float(np.mean(s))
    mean_v = float(np.mean(v))
    return mean_s, mean_v


def classify_image(faces_count: int, mean_s: float, mean_v: float) -> str:
    """Determine descriptive rule-based group name."""
    img_type = "Portrait" if faces_count > 0 else "Scenic"
    exposure = "Dark" if mean_v < 80.0 else "Bright"
    saturation = "Colorful" if mean_s > 120.0 else "Muted"
    return f"{img_type} ({exposure}/{saturation})"


def analyze_and_group(
    image_paths: List[Path],
    cache: BatchProcessorCache,
    engine: Any,
) -> Dict[str, List[Path]]:
    """Scan and group images based on faces and HSV characteristics, using cache."""
    groups = {
        "Portrait (Bright/Colorful)": [],
        "Portrait (Bright/Muted)": [],
        "Portrait (Dark/Colorful)": [],
        "Portrait (Dark/Muted)": [],
        "Scenic (Bright/Colorful)": [],
        "Scenic (Bright/Muted)": [],
        "Scenic (Dark/Colorful)": [],
        "Scenic (Dark/Muted)": [],
    }

    for path in image_paths:
        try:
            mtime = os.path.getmtime(path)
            filename = path.name
            cached_info = cache.get(filename, mtime)

            if cached_info is not None:
                faces = cached_info["faces"]
                mean_s = cached_info["mean_s"]
                mean_v = cached_info["mean_v"]
            else:
                img = imread_exif(path)
                # Use engine detector to find faces
                detected_faces = engine._detector.detect(img)
                faces = len(detected_faces)
                face_width = 0
                if faces > 0:
                    face_width = int(detected_faces[0].ied * 2.5)

                # Segment subject/person if detector supports it
                person_mask = None
                try:
                    person_mask = engine._detector.segment_person(img)
                except Exception:
                    pass

                mean_s, mean_v = compute_hsv_stats(img, person_mask)

                info = {
                    "faces": faces,
                    "face_width": face_width,
                    "mean_s": mean_s,
                    "mean_v": mean_v,
                }
                cache.set(filename, mtime, info)

            cat = classify_image(faces, mean_s, mean_v)
            if cat in groups:
                groups[cat].append(path)
            else:
                groups.setdefault(cat, []).append(path)

        except Exception as e:
            logger.warning("Failed to analyze/classify image %s: %s", path, e)

    cache.save()
    # Filter out empty lists
    return {k: v for k, v in groups.items() if v}


def generate_contact_sheet(
    image_paths: List[str | Path],
    output_path: str | Path,
    cols: int = 4,
    cell_size: int = 320,
) -> Path:
    """Generate a tiled grid contact sheet of processed images with clean PIL labels."""
    if not image_paths:
        return Path(output_path)

    num_images = len(image_paths)
    rows = (num_images + cols - 1) // cols
    sheet_w = cols * cell_size
    sheet_h = rows * cell_size

    # Create canvas
    canvas = np.zeros((sheet_h, sheet_w, 3), dtype=np.uint8)

    # Pre-resolve a nice TTF font or load default
    font = None
    for font_name in ["Arial.ttf", "Helvetica.ttf", "LiberationSans-Regular.ttf", "sans-serif.ttf"]:
        try:
            font = ImageFont.truetype(font_name, 14)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    for idx in range(rows * cols):
        r = idx // cols
        c = idx % cols

        cell = np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

        if idx < num_images:
            img_path = image_paths[idx]
            img = cv2.imread(str(img_path))
            if img is not None:
                text_height = 40
                max_img_h = cell_size - text_height
                max_img_w = cell_size

                h, w = img.shape[:2]
                scale = min(max_img_w / w, max_img_h / h)
                nw, nh = int(w * scale), int(h * scale)
                resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

                # Center in upper zone
                x_offset = (cell_size - nw) // 2
                y_offset = (max_img_h - nh) // 2
                cell[y_offset : y_offset + nh, x_offset : x_offset + nw] = resized

                # Subtle cell border
                cv2.rectangle(cell, (0, 0), (cell_size - 1, cell_size - 1), (50, 50, 50), 1)

                filename = Path(img_path).name
                if len(filename) > 25:
                    filename = filename[:22] + "..."

                # Text background bar
                cv2.rectangle(cell, (0, max_img_h), (cell_size, cell_size), (15, 15, 15), -1)

                # Render with PIL ImageFont for clean typography
                cell_pil = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(cell_pil)
                try:
                    bbox = draw.textbbox((0, 0), filename, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w, text_h = draw.textsize(filename, font=font)

                text_x = (cell_size - text_w) // 2
                text_y = max_img_h + (text_height - text_h) // 2
                draw.text((text_x, text_y), filename, fill=(220, 220, 220), font=font)
                cell = cv2.cvtColor(np.array(cell_pil), cv2.COLOR_RGB2BGR)
        else:
            # Draw placeholder for empty cells to keep visual grid balance
            cv2.rectangle(cell, (0, 0), (cell_size - 1, cell_size - 1), (25, 25, 25), -1)
            cv2.rectangle(cell, (0, 0), (cell_size - 1, cell_size - 1), (50, 50, 50), 1)

            cell_pil = Image.fromarray(cv2.cvtColor(cell, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(cell_pil)
            placeholder_text = "[Empty Cell]"
            try:
                bbox = draw.textbbox((0, 0), placeholder_text, font=font)
                text_w = bbox[2] - bbox[0]
                text_h = bbox[3] - bbox[1]
            except Exception:
                text_w, text_h = draw.textsize(placeholder_text, font=font)

            text_x = (cell_size - text_w) // 2
            text_y = (cell_size - text_h) // 2
            draw.text((text_x, text_y), placeholder_text, fill=(75, 75, 75), font=font)
            cell = cv2.cvtColor(np.array(cell_pil), cv2.COLOR_RGB2BGR)

        canvas[r * cell_size : (r + 1) * cell_size, c * cell_size : (c + 1) * cell_size] = cell

    cv2.imwrite(str(output_path), canvas)
    return Path(output_path)


class BatchProcessor:
    """Ingests a folder of images, classifies/groups them, processes using a style, and exports results."""

    def __init__(self, engine: Any = None):
        from .engine import RetouchEngine
        self.engine = engine or RetouchEngine()

    def process_folder(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        style_name_or_recipe: str = "natural",
        custom_style_profile: Optional[StyleProfile] = None,
        export_fmt: str = "JPEG",
        export_quality: int = 95,
        export_res: str = "Original",
        auto_group: bool = False,
        generate_sheet: bool = True,
        export_zip: bool = False,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[List[str], Optional[str], Optional[str], str]:
        """Ingests, classifies, processes, and packages a folder of photos."""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Ingest files
        all_files = []
        for root, _, files in os.walk(input_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    all_files.append(Path(root) / f)

        if not all_files:
            return [], None, None, "No supported images found in the input folder."

        # 2. Setup cache (stored under user home to prevent read-only directory issues)
        cache = BatchProcessorCache(input_path)

        # 3. Analyze & Group
        if progress_callback:
            progress_callback(0.1, "Analyzing & grouping images...")
        
        groups = analyze_and_group(all_files, cache, self.engine)

        status_msg = f"Ingested {len(all_files)} images.\n"
        status_msg += "Groupings detected:\n"
        for grp_name, paths in groups.items():
            status_msg += f"  - {grp_name}: {len(paths)} files\n"

        # 4. Process images
        from .style import StyleApplier
        applier = StyleApplier(self.engine)

        processed_paths = []
        total_files = len(all_files)

        if progress_callback:
            progress_callback(0.2, "Processing photos...")

        for idx, file_path in enumerate(all_files):
            try:
                img_bgr = imread_exif(file_path)

                if custom_style_profile is not None:
                    result = applier.apply(img_bgr, custom_style_profile)
                else:
                    result = self.engine.process(img_bgr, recipe=style_name_or_recipe)

                # Resolution resizing
                export_max = {
                    "Original": None,
                    "4K (3840px)": 3840,
                    "2K (2048px)": 2048,
                    "Full HD (1920px)": 1920,
                    "HD (1280px)": 1280,
                    "720px": 720,
                }.get(export_res)
                
                if export_max is not None:
                    h, w = result.shape[:2]
                    if max(h, w) > export_max:
                        scale = export_max / max(h, w)
                        result = cv2.resize(result, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                # Output filename
                ext = {"JPEG": ".jpg", "PNG": ".png", "WebP": ".webp"}.get(export_fmt, ".jpg")
                out_name = f"{file_path.stem}_retouched{ext}"
                out_file_path = output_path / out_name

                # Write to disk
                from .io import encode_write_params
                write_params = encode_write_params(export_fmt.lower(), export_quality)
                cv2.imwrite(str(out_file_path), result, write_params)

                # Copy EXIF if supported
                from .io import copy_exif
                copy_exif(file_path, out_file_path)

                processed_paths.append(out_file_path)

                if progress_callback:
                    prog = 0.2 + (0.6 * (idx + 1) / total_files)
                    progress_callback(prog, f"Processed {idx + 1}/{total_files} files...")

            except Exception as e:
                logger.error("Failed to process %s: %s", file_path, e)
                status_msg += f"Error processing {file_path.name}: {e}\n"

        # 5. Generate Contact Sheet
        contact_sheet_path = None
        if generate_sheet and processed_paths:
            if progress_callback:
                progress_callback(0.85, "Generating contact sheet...")
            sheet_out = output_path / "contact_sheet.jpg"
            try:
                generate_contact_sheet(processed_paths, sheet_out)
                contact_sheet_path = str(sheet_out)
                status_msg += "Generated contact sheet: contact_sheet.jpg\n"
            except Exception as e:
                logger.error("Failed to generate contact sheet: %s", e)
                status_msg += f"Failed to generate contact sheet: {e}\n"

        # 6. Package into ZIP
        zip_path = None
        if export_zip and processed_paths:
            if progress_callback:
                progress_callback(0.95, "Packaging ZIP...")
            import zipfile
            z_path = output_path / "batch_export.zip"
            try:
                with zipfile.ZipFile(z_path, "w") as zipf:
                    for exp_path in processed_paths:
                        zipf.write(exp_path, arcname=exp_path.name)
                    if contact_sheet_path:
                        zipf.write(contact_sheet_path, arcname="contact_sheet.jpg")
                zip_path = str(z_path)
                status_msg += "Packaged files into ZIP: batch_export.zip\n"
            except Exception as e:
                logger.error("Failed to package ZIP: %s", e)
                status_msg += f"Failed to package ZIP: {e}\n"

        if progress_callback:
            progress_callback(1.0, "Complete!")

        status_msg += f"\nSuccess: Processed {len(processed_paths)}/{total_files} files successfully ✓"
        return [str(p) for p in processed_paths], contact_sheet_path, zip_path, status_msg
