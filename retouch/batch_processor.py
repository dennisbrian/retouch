"""BatchProcessor — directory walking, cache orchestration, descriptive grouping, and contact sheets."""

from __future__ import annotations

import os
import sys
import json
import logging
import hashlib
import queue
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .style import StyleProfile
from .io import (
    imread_exif,
    color_context_for_path,
    IMAGE_EXTENSIONS,
    EXPORT_RES_MAP,
    EXT_MAP,
)
from .utils import get_cache_dir, normalize_mask

logger = logging.getLogger(__name__)

# Default RAM budget for in-flight batch images (~2 GB working set).
_DEFAULT_RAM_BUDGET_BYTES = 2 * 1024 * 1024 * 1024
_ANALYSIS_CACHE_VERSION = "batch-analysis-v2"


_SessionInput = Optional[Union["Session", str, Path]]


def _path_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(parent.resolve(strict=False))
        return True
    except ValueError:
        return False


def validate_batch_roots(input_dir: Union[str, Path], output_dir: Union[str, Path]) -> Tuple[Path, Path]:
    """Resolve roots and reject self-ingesting output layouts."""

    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if _path_within(output_path, input_path):
        raise ValueError(
            "Batch output directory must be outside the input tree: %s" % output_path
        )
    return input_path, output_path


def _batch_destination(
    file_path: Path,
    input_root: Path,
    output_root: Path,
    extension: str,
    *,
    discriminator: Optional[str] = None,
) -> Path:
    try:
        relative = file_path.resolve().relative_to(input_root.resolve())
    except ValueError:
        relative = Path(file_path.name)
    suffix = "_%s" % discriminator if discriminator else ""
    return output_root / relative.parent / ("%s%s_retouched%s" % (relative.stem, suffix, extension))


def _destination_key(path: Path) -> str:
    value = os.path.normcase(str(path.resolve(strict=False)))
    return value.casefold() if sys.platform == "darwin" else value


def plan_batch_outputs(
    files: List[Path],
    input_root: Path,
    output_root: Path,
    export_fmt: str,
) -> Dict[str, Path]:
    """Preflight unique deterministic destinations for every batch source."""

    extension = EXT_MAP.get(export_fmt, ".jpg")
    grouped: Dict[str, List[Path]] = {}
    for source in files:
        candidate = _batch_destination(source, input_root, output_root, extension)
        grouped.setdefault(_destination_key(candidate), []).append(source)

    plan: Dict[str, Path] = {}
    claimed: Dict[str, Path] = {}
    for sources in grouped.values():
        for source in sources:
            discriminator = None
            if len(sources) > 1:
                source_ext = source.suffix.lower().lstrip(".") or "image"
                discriminator = source_ext
            candidate = _batch_destination(
                source,
                input_root,
                output_root,
                extension,
                discriminator=discriminator,
            )
            key = _destination_key(candidate)
            if key in claimed:
                relative = str(source.resolve()).encode("utf-8")
                short_hash = hashlib.sha256(relative).hexdigest()[:8]
                discriminator = "%s_%s" % (discriminator or "image", short_hash)
                candidate = _batch_destination(
                    source,
                    input_root,
                    output_root,
                    extension,
                    discriminator=discriminator,
                )
                key = _destination_key(candidate)
            if key in claimed:
                raise ValueError(
                    "Batch output collision between %s and %s" % (claimed[key], source)
                )
            claimed[key] = source
            plan[str(source.resolve())] = candidate
    return plan


def _estimate_image_working_bytes(path: Path) -> int:
    """Estimate peak RAM for one image (input + output + pipeline buffers)."""
    try:
        with Image.open(path) as im:
            w, h = im.size
        return h * w * 3 * 4
    except Exception:
        return 24 * 1024 * 1024  # conservative 4K RGB fallback


def _compute_queue_depth(
    num_workers: int,
    image_paths: List[Path],
    ram_budget_bytes: int = _DEFAULT_RAM_BUDGET_BYTES,
) -> int:
    """Bound in-flight work queue depth from sampled image sizes."""
    if not image_paths:
        return max(2, num_workers)
    sample = image_paths[: min(8, len(image_paths))]
    avg_bytes = sum(_estimate_image_working_bytes(p) for p in sample) // len(sample)
    depth = max(1, min(num_workers * 2, ram_budget_bytes // max(avg_bytes, 1)))
    return depth


def _run_async_batch_queue(
    image_paths: List[Path],
    worker_fn: Callable[[Path], Optional[Path]],
    num_workers: int,
    total_files: int,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    ram_budget_bytes: int = _DEFAULT_RAM_BUDGET_BYTES,
) -> List[Path]:
    """Producer-consumer batch queue with bounded backpressure for RAM control."""
    if total_files <= 0:
        return []

    max_queue_depth = _compute_queue_depth(num_workers, image_paths, ram_budget_bytes)
    work_q: queue.Queue = queue.Queue(maxsize=max_queue_depth)
    results: List[Path] = []
    results_lock = threading.Lock()
    completed = [0]
    _POISON = object()

    def producer() -> None:
        for fp in image_paths:
            work_q.put(fp)
        for _ in range(num_workers):
            work_q.put(_POISON)

    def consumer() -> None:
        while True:
            item = work_q.get()
            try:
                if item is _POISON:
                    break
                res_path = worker_fn(item)
                if res_path is not None:
                    with results_lock:
                        results.append(res_path)
            finally:
                with results_lock:
                    completed[0] += 1
                    done = completed[0]
                if progress_callback:
                    prog = 0.2 + (0.6 * done / total_files)
                    progress_callback(prog, f"Processed {done}/{total_files} files...")
                work_q.task_done()

    prod_thread = threading.Thread(target=producer, daemon=True)
    workers = [
        threading.Thread(target=consumer, daemon=True)
        for _ in range(num_workers)
    ]
    prod_thread.start()
    for t in workers:
        t.start()
    prod_thread.join()
    for t in workers:
        t.join()
    return results


def _resolve_session(session: _SessionInput) -> Optional["Session"]:
    """Normalize a session input into a Session object, or None.

    Accepts a Session instance, or a path/str pointing to a session JSON file.
    Returns None when session is None so callers can fall back to the legacy
    recipe/params flow.

    Thread-safety: Session.from_file reads the JSON immutably and returns a
    fresh instance — no shared mutable state is held across calls.
    """
    if session is None:
        return None

    # Local import to avoid a circular dependency at module load time
    # (session.py does not import batch_processor.py, but keeping this local
    # makes the dependency direction explicit and lazy).
    from .session import Session

    if isinstance(session, Session):
        return session

    if isinstance(session, (str, Path)):
        path = Path(session)
        if not path.exists():
            raise FileNotFoundError(f"Session file not found: {path}")
        try:
            return Session.from_file(str(path))
        except (json.JSONDecodeError, OSError, ValueError) as e:
            raise ValueError(f"Failed to load session from {path}: {e}") from e

    raise TypeError(
        f"session must be a Session, str, Path, or None — got {type(session).__name__}"
    )


class BatchProcessorCache:
    """Persistent JSON cache for face counts, sizes, and color statistics.

    Stored under the configured Retouch cache directory to prevent write
    errors on read-only input folders.
    Keyed by the SHA-256 hash of the resolved input directory.
    """

    def __init__(self, input_dir: Path) -> None:
        self.input_dir = input_dir
        self.cache_dir = get_cache_dir()
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

    def relative_key(self, path: Path) -> str:
        """Return a normalized input-relative cache key, never basename-only."""
        try:
            return path.resolve().relative_to(self.input_dir.resolve()).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    def save(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save cache file %s: %s", self.cache_file, e)

    def get(self, filename: str, mtime: float) -> Optional[Dict[str, Any]]:
        entry = self.data.get(filename)
        if entry and entry.get("mtime") == mtime and entry.get("cache_version") == _ANALYSIS_CACHE_VERSION:
            return entry
        return None

    def set(self, filename: str, mtime: float, info: Dict[str, Any]) -> None:
        info["mtime"] = mtime
        info["cache_version"] = _ANALYSIS_CACHE_VERSION
        self.data[filename] = info


def compute_hsv_stats(img_bgr: np.ndarray, mask: Optional[np.ndarray] = None) -> Tuple[float, float]:
    """Compute average Saturation (S) and Value (V) channels, restricted to the subject mask if available."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    if mask is not None:
        mask_f = normalize_mask(mask)
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
            filename = cache.relative_key(path)
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

                # Segment subject/person if detector supports it
                person_mask = None
                try:
                    person_mask = engine._detector.segment_person(img)
                except Exception as seg_err:
                    logger.warning(
                        "Person segmentation failed for %s: %s. Continuing without person mask.",
                        path, seg_err
                    )

                mean_s, mean_v = compute_hsv_stats(img, person_mask)

                info = {
                    "faces": faces,
                    "mean_s": mean_s,
                    "mean_v": mean_v,
                    # New or changed files are content-addressed before the
                    # analysis result is considered reusable. Unchanged
                    # records take the persisted mtime fast path above.
                    "content_id": _file_sha256(path),
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
        except Exception as font_err:
            logger.debug(
                "Tried font %s but unavailable: %s. Continuing to next font candidate.",
                font_name, font_err
            )
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
                except Exception as text_err:
                    logger.debug(
                        "draw.textbbox failed for filename label (font=%s): %s. Falling back to legacy draw.textsize.",
                        getattr(font, 'path', font), text_err
                    )
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
            except Exception as text_err:
                logger.debug(
                    "draw.textbbox failed for placeholder text (font=%s): %s. Falling back to legacy draw.textsize.",
                    getattr(font, 'path', font), text_err
                )
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

    def __init__(self, engine: Optional[Any] = None) -> None:
        from .engine import RetouchEngine
        self.engine = engine or RetouchEngine()
        # Only an engine we created is ours to close; a caller-supplied one
        # may outlive this processor.
        self._owns_engine = engine is None

    def close(self) -> None:
        """Release the engine if this processor created it.

        Idempotent. Leaving the engine unclosed lets MediaPipe's
        FaceLandmarker be finalized by the garbage collector, where its
        __del__ blocks forever on a serial-dispatcher future — the batch
        appears to hang after processing completes.
        """
        if self._owns_engine and self.engine is not None:
            self.engine.close()
            self.engine = None
            self._owns_engine = False

    def __enter__(self) -> "BatchProcessor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

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
        session: _SessionInput = None,
        num_workers: int = 4,
        on_file_result: Optional[
            Callable[[Path, Optional[Path], Optional[List[Any]], Optional[str]], None]
        ] = None,
        only_files: Optional[List[Path]] = None,
        output_path_overrides: Optional[Dict[Union[str, Path], Union[str, Path]]] = None,
    ) -> Tuple[List[str], Optional[str], Optional[str], str]:

        """Ingests, classifies, processes, and packages a folder of photos.

        ``on_file_result``, if given, is called once per file (success or
        failure) as ``(source_path, output_path_or_None, qa_list_or_None,
        error_or_None)``. This is the hook the Job Dashboard uses to record
        per-file QA state without this module needing to know about jobs.py.

        ``only_files``, if given, restricts processing to this subset of the
        walked file list (used for "re-run flagged files only").

        When ``session`` is provided (a Session object or a path to a session
        JSON file), its params dict is used as the base processing parameters
        for every image in the batch (the "tune once, apply to 500 shots"
        workflow). The session's recipe is used unless ``style_name_or_recipe``
        was explicitly set by the caller — note that callers cannot distinguish
        the default ``"natural"`` from an explicit override via this signature,
        so to force a specific recipe alongside a session, load the session
        with that recipe set, or pass ``custom_style_profile`` instead.

        Session params are merged UNDER any explicit per-image logic: when
        ``custom_style_profile`` is provided it takes precedence (the style
        applier path is used). Otherwise session.params become the kwargs
        passed to ``engine.process``.

        Thread-safety: the session is resolved to an immutable params snapshot
        once at the start of the call; no shared mutable state is mutated
        across concurrent invocations.
        """
        input_path, output_path = validate_batch_roots(input_dir, output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Resolve session once (immutable snapshot for the whole batch).
        resolved_session = _resolve_session(session)
        session_params: Dict[str, Any] = {}
        if resolved_session is not None:
            # Copy so we never mutate the caller's session.params — critical
            # for Gradio thread-safety where the same Session object may be
            # referenced by multiple worker threads.
            raw_params = dict(resolved_session.params)

            # Session is forward-compatible (unknown keys warn, don't crash),
            # but engine.process has a fixed signature with no **kwargs. Filter
            # to only valid process() parameter names so an older session
            # loaded against a newer engine (or vice-versa) degrades gracefully
            # rather than raising TypeError.
            import inspect

            try:
                valid_keys = set(
                    inspect.signature(self.engine.process).parameters.keys()
                )
            except (ValueError, TypeError) as inspect_err:
                logger.warning(
                    "Could not introspect engine.process signature for session "
                    "param filtering: %s. Passing all session params through.",
                    inspect_err,
                )
                valid_keys = None

            if valid_keys is not None:
                dropped = {
                    k: v for k, v in raw_params.items() if k not in valid_keys
                }
                if dropped:
                    logger.warning(
                        "Session params not valid for engine.process (ignored): %s",
                        sorted(dropped.keys()),
                    )
                session_params = {
                    k: v for k, v in raw_params.items() if k in valid_keys
                }
            else:
                session_params = raw_params

            # Session's recipe takes precedence over the default "natural",
            # but a caller who passes BOTH a session AND a non-default
            # style_name_or_recipe is treated as explicitly overriding the
            # recipe.
            if (
                resolved_session.recipe
                and style_name_or_recipe == "natural"
            ):
                style_name_or_recipe = resolved_session.recipe
            logger.info(
                "Batch processing with session (recipe=%s, %d params)",
                style_name_or_recipe,
                len(session_params),
            )

        # 1. Ingest files
        all_files = []
        for root, _, files in os.walk(input_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    all_files.append(Path(root) / f)

        if only_files is not None:
            only_set = {Path(p).resolve() for p in only_files}
            all_files = [fp for fp in all_files if fp.resolve() in only_set]

        if not all_files:
            return [], None, None, "No supported images found in the input folder."
        output_plan = plan_batch_outputs(
            all_files, input_path, output_path, export_fmt,
        )
        if output_path_overrides:
            for source_value, destination_value in output_path_overrides.items():
                source_key = str(Path(source_value).expanduser().resolve())
                if source_key not in output_plan:
                    raise ValueError(
                        "Batch output override references a source outside this batch: %s" % source_value
                    )
                destination = Path(destination_value).expanduser().resolve()
                if not _path_within(destination, output_path):
                    raise ValueError(
                        "Batch output override must stay inside the output root: %s" % destination
                    )
                output_plan[source_key] = destination
            destination_keys: Dict[str, str] = {}
            for source_key, destination in output_plan.items():
                key = _destination_key(destination)
                if key in destination_keys:
                    raise ValueError(
                        "Batch output overrides collide for %s and %s" % (
                            destination_keys[key], source_key,
                        )
                    )
                destination_keys[key] = source_key

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

        if num_workers > 1 and total_files > 1:
            if not self._owns_engine:
                # A caller-supplied engine may be a test double or a custom
                # stateful implementation. Never invoke it concurrently;
                # the default GUI path uses worker-local engines below.
                logger.warning(
                    "Batch received a caller-supplied engine; using one worker "
                    "to preserve engine isolation"
                )
                effective_workers = 1
            else:
                effective_workers = min(int(num_workers), total_files)

            if effective_workers == 1:
                # Keep the queue contract (and its bounded backpressure) even
                # when a caller-supplied engine must be serialized. The
                # important invariant is that no engine is invoked by two
                # threads at once.
                def _serial_worker(fp: Path) -> Optional[Path]:
                    return self._process_single_file(
                        fp,
                        custom_style_profile,
                        session_params,
                        style_name_or_recipe,
                        export_res,
                        export_fmt,
                        export_quality,
                        output_path,
                        applier,
                        on_file_result,
                        input_root=input_path,
                        engine=self.engine,
                        output_file=output_plan[str(fp.resolve())],
                    )

                processed_paths = _run_async_batch_queue(
                    all_files,
                    _serial_worker,
                    1,
                    total_files,
                    progress_callback,
                )
            else:
                # RetouchEngine owns detector/parser/mask caches. One engine
                # and one StyleApplier per queue thread keeps request state
                # isolated while retaining bounded parallelism.
                worker_local = threading.local()
                worker_engines: List[Any] = []
                worker_engines_lock = threading.Lock()

                def _worker(fp: Path) -> Optional[Path]:
                    worker_engine = getattr(worker_local, "engine", None)
                    if worker_engine is None:
                        from .engine import RetouchEngine

                        worker_engine = RetouchEngine()
                        worker_local.engine = worker_engine
                        worker_local.applier = StyleApplier(worker_engine)
                        with worker_engines_lock:
                            worker_engines.append(worker_engine)
                    return self._process_single_file(
                        fp,
                        custom_style_profile,
                        session_params,
                        style_name_or_recipe,
                        export_res,
                        export_fmt,
                        export_quality,
                        output_path,
                        worker_local.applier,
                        on_file_result,
                        input_root=input_path,
                        engine=worker_engine,
                        output_file=output_plan[str(fp.resolve())],
                    )

                try:
                    processed_paths = _run_async_batch_queue(
                        all_files,
                        _worker,
                        effective_workers,
                        total_files,
                        progress_callback,
                    )
                finally:
                    for worker_engine in worker_engines:
                        try:
                            worker_engine.close()
                        except Exception:
                            logger.warning(
                                "Failed to close a batch worker engine",
                                exc_info=True,
                            )
        else:
            for idx, file_path in enumerate(all_files):
                res_path = self._process_single_file(
                    file_path,
                    custom_style_profile,
                    session_params,
                    style_name_or_recipe,
                    export_res,
                    export_fmt,
                    export_quality,
                    output_path,
                    applier,
                    on_file_result,
                    input_root=input_path,
                    output_file=output_plan[str(file_path.resolve())],
                )
                if res_path is not None:
                    processed_paths.append(res_path)
                if progress_callback:
                    prog = 0.2 + (0.6 * (idx + 1) / total_files)
                    progress_callback(prog, f"Processed {idx + 1}/{total_files} files...")

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
                from .utils import log_crash
                log_crash(e, {"stage": "generate_contact_sheet"})
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
                        try:
                            arcname = Path(exp_path).resolve().relative_to(output_path.resolve()).as_posix()
                        except ValueError:
                            arcname = Path(exp_path).name
                        zipf.write(exp_path, arcname=arcname)
                    if contact_sheet_path:
                        zipf.write(contact_sheet_path, arcname="contact_sheet.jpg")
                zip_path = str(z_path)
                status_msg += "Packaged files into ZIP: batch_export.zip\n"
            except Exception as e:
                logger.error("Failed to package ZIP: %s", e)
                from .utils import log_crash
                log_crash(e, {"stage": "package_zip"})
                status_msg += f"Failed to package ZIP: {e}\n"

        if progress_callback:
            progress_callback(1.0, "Complete!")

        succeeded_count = len(processed_paths)
        failed_count = total_files - succeeded_count
        if failed_count == 0:
            status_msg += f"\nSuccess: Processed {succeeded_count}/{total_files} files successfully ✓"
        elif succeeded_count:
            status_msg += (
                f"\nPartial: Processed {succeeded_count}/{total_files} files successfully; "
                f"{failed_count} failed and remain retryable."
            )
        else:
            status_msg += f"\nFailed: 0/{total_files} files completed; all files remain retryable."
        return [str(p) for p in processed_paths], contact_sheet_path, zip_path, status_msg

    def _process_single_file(
        self,
        file_path: Path,
        custom_style_profile: Optional[StyleProfile],
        session_params: Dict[str, Any],
        style_name_or_recipe: str,
        export_res: str,
        export_fmt: str,
        export_quality: int,
        output_path: Path,
        applier: Any,
        on_file_result: Optional[
            Callable[[Path, Optional[Path], Optional[List[Any]], Optional[str]], None]
        ] = None,
        input_root: Optional[Path] = None,
        engine: Optional[Any] = None,
        output_file: Optional[Path] = None,
    ) -> Optional[Path]:
        """Process a single file and write output with embedded metadata."""
        try:
            img_bgr = imread_exif(file_path)
            request_engine = engine if engine is not None else self.engine
            color_context = color_context_for_path(file_path)

            if custom_style_profile is not None:
                result = applier.apply(img_bgr, custom_style_profile)
            elif session_params:
                kwargs = {k: v for k, v in session_params.items() if v is not None}
                result = request_engine.process(img_bgr, recipe=style_name_or_recipe, **kwargs)
            else:
                result = request_engine.process(img_bgr, recipe=style_name_or_recipe)

            # Capture QA now, before any cv2 op below reassigns `result` to a
            # plain ndarray (cv2.resize/cvtColor drop the ProcessingResult
            # subclass and its .qa attribute). Must stay the first thing done
            # with `result` after it's produced — see docs/plans for the Job
            # Dashboard on why this ordering is load-bearing.
            qa_list = getattr(result, "qa", None)

            export_max = EXPORT_RES_MAP.get(export_res)
            if export_max is not None:
                h, w = result.shape[:2]
                if max(h, w) > export_max:
                    scale = export_max / max(h, w)
                    result = cv2.resize(result, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            ext = EXT_MAP.get(export_fmt, ".jpg")
            if output_file is not None:
                out_file_path = Path(output_file)
            else:
                source_root = input_root.resolve() if input_root is not None else file_path.parent.resolve()
                out_file_path = _batch_destination(
                    file_path, source_root, output_path, ext,
                )
            out_file_path.parent.mkdir(parents=True, exist_ok=True)

            from .io import (
                write_image_with_color_context,
                read_c2pa_manifest,
                read_exif_bytes,
            )
            exif_bytes = read_exif_bytes(file_path)
            c2pa_manifest = read_c2pa_manifest(file_path)
            fd, temp_name = tempfile.mkstemp(
                prefix=f".{out_file_path.stem}-", suffix=out_file_path.suffix,
                dir=str(out_file_path.parent),
            )
            os.close(fd)
            try:
                write_image_with_color_context(
                    temp_name,
                    result,
                    color_context,
                    bit_depth=8,
                    quality=export_quality,
                    exif=exif_bytes,
                    c2pa_manifest=c2pa_manifest,
                )
                if not os.path.isfile(temp_name) or os.path.getsize(temp_name) <= 0:
                    raise IOError("image writer produced no non-empty output")
                os.replace(temp_name, out_file_path)
            except Exception:
                try:
                    os.remove(temp_name)
                except OSError:
                    pass
                raise
            if on_file_result is not None:
                on_file_result(file_path, out_file_path, qa_list, None)
            return out_file_path
        except Exception as e:
            logger.error("Failed to process %s: %s", file_path, e)
            from .utils import log_crash
            log_crash(e, {
                "stage": "process_file",
                "file_path": str(file_path),
                "style_name_or_recipe": str(style_name_or_recipe),
                "export_fmt": export_fmt
            })
            if on_file_result is not None:
                on_file_result(file_path, None, None, str(e))
            return None
