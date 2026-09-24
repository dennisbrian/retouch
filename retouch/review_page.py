"""Per-batch review page: records, thumbnails, ``review.html`` and decisions.

After a batch run, every source image gets one :class:`ReviewRecord` written
under ``<root>/.retouch-review/records/``.  :func:`build_review_page` turns
those records into a single self-contained ``<root>/review.html`` (rendered
by :mod:`retouch.review_template`) plus JPEG thumbnails and before/after face
crops under ``<root>/.retouch-review/thumbs/``.  Every path inside the page is
relative to ``review.html`` so the folder can be moved or zipped.

The page lets the user pick or reject images and export the decisions as
JSON; :func:`apply_decisions` then copies picks into ``<root>/picks/`` and,
only when asked, moves rejects aside into ``<root>/rejected/``.  Nothing is
ever deleted or overwritten.

Command line::

    python -m retouch.review_page build ROOT [--source DIR] [--recursive]
                                             [--title T] [--workers N]
    python -m retouch.review_page apply ROOT DECISIONS.json
                                             [--picks-dir D] [--move-rejects]
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import multiprocessing
import os
import re
import shlex
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import cv2
import numpy as np

logger = logging.getLogger(__name__)

REVIEW_DIRNAME = ".retouch-review"
PAGE_NAME = "review.html"
GRID_EDGE = 480  # long edge of the small card thumbnails in the grid view
STATUSES = ("done", "qa_fail", "failed", "skipped")
DECISIONS = ("pick", "reject")
PAGE_DATA_VERSION = 1

_RECORDS_SUBDIR = "records"
_THUMBS_SUBDIR = "thumbs"
_DEFAULT_PICKS_DIR = "picks"
_DEFAULT_REJECTS_DIR = "rejected"
# Output-root subfolders that never hold batch outputs: the review cache,
# face-aware social crops, and apply_decisions' default destinations.
_BACKFILL_SKIP_DIRS = (REVIEW_DIRNAME, "social", _DEFAULT_PICKS_DIR, _DEFAULT_REJECTS_DIR)
_FACE_EXPAND = 1.8
_THUMB_JPEG_QUALITY = 85
_SAFE_STEM_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SAFE_STEM_MAX = 60


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class ReviewRecord:
    """Everything the review page needs to know about one source image.

    Attributes:
        source: Absolute source path.
        output: Absolute retouched output path, or ``None`` when none was
            written (QA fail, failure).
        compare: Absolute ``*_compare`` path when one was written.
        status: One of :data:`STATUSES`.
        recipe: Recipe name the batch ran with, if any.
        faces: Face boxes ``[x, y, w, h]`` in native source pixel coords.
        qa: QA entries ``{"detector", "score", "threshold", "flagged",
            "message", "available"}``.
        error: Failure message for ``status == "failed"``.
        elapsed_s: Wall-clock processing time in seconds.
        created_at: ISO-8601 UTC timestamp; filled by
            :func:`write_review_record` when empty.
        relative: Source path relative to the batch input root (display name).
        qa_recorded: ``False`` when the record was reconstructed without QA
            (backfilled past batches, skipped images) so the page can say
            "QA not recorded" instead of implying a clean pass.
    """

    source: str
    output: Optional[str] = None
    compare: Optional[str] = None
    status: str = "done"
    recipe: Optional[str] = None
    faces: List[List[int]] = field(default_factory=list)
    qa: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    elapsed_s: Optional[float] = None
    created_at: str = ""
    relative: str = ""
    qa_recorded: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict of all fields."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReviewRecord":
        """Build a record from a dict, ignoring unknown keys.

        Raises:
            ValueError: When the required ``source`` key is missing.
        """
        if "source" not in d:
            raise ValueError("review record is missing 'source'")
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Coerce numpy/python scalars to a JSON-safe float (NaN/inf -> default)."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _qa_entry(warning: Any) -> Optional[Dict[str, Any]]:
    """Normalise one QA warning (``QAWarning`` dataclass or dict) to a dict."""
    if isinstance(warning, Mapping):
        get = warning.get
        details = warning.get("details") or {}
    else:
        def get(key: str, default: Any = None) -> Any:
            return getattr(warning, key, default)
        details = getattr(warning, "details", None) or {}
    detector = get("detector")
    if detector is None:
        return None
    available = get("available")
    if available is None:
        available = details.get("available", True) if isinstance(details, Mapping) else True
    return {
        "detector": str(detector),
        "score": _to_float(get("score")),
        "threshold": _to_float(get("threshold")),
        "flagged": bool(get("flagged", False)),
        "message": str(get("message", "") or ""),
        "available": bool(available),
    }


def qa_entries(qa: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    """Normalise a QA list (``QAWarning`` objects and/or dicts) for a record.

    Entries without a ``detector`` are dropped. Never raises.
    """
    out: List[Dict[str, Any]] = []
    try:
        for warning in qa or []:
            entry = _qa_entry(warning)
            if entry is not None:
                out.append(entry)
    except Exception as exc:  # defensive: record writing must never break a batch
        logger.warning("Could not normalise QA warnings for review: %s", exc)
    return out


def _proxy_multiplier(result: Any) -> float:
    """Multiplier from the engine's bbox frame to the result's pixel frame.

    The F8.1 ``quality="draft"`` proxy path leaves ``face_contexts`` in
    PROXY_MAX_DIM coordinates while the returned image is native size; the
    default ``quality="full"`` path already rescales boxes to native.
    """
    params = getattr(result, "params", None)
    if getattr(params, "quality", "full") != "draft":
        return 1.0
    shape = getattr(result, "shape", None)
    if not shape or len(shape) < 2:
        return 1.0
    try:
        from .engine import PROXY_MAX_DIM
    except ImportError as exc:
        logger.debug("engine unavailable for proxy bbox correction: %s", exc)
        return 1.0
    long_edge = max(int(shape[0]), int(shape[1]))
    return long_edge / float(PROXY_MAX_DIM) if long_edge > PROXY_MAX_DIM else 1.0


def result_review_meta(result: Any, scale: float = 1.0) -> Dict[str, Any]:
    """Extract ``{'faces': [...], 'qa': [...]}`` from a ``ProcessingResult``.

    Call this BEFORE any ``cv2.resize`` of the result: ``cv2.resize`` returns
    a plain ndarray and drops ``.qa`` / ``.face_contexts``.

    Args:
        result: ``RetouchEngine.process()`` result (or any object; missing
            attributes yield empty lists).
        scale: The ``resize_for_processing`` scale (< 1 when the engine ran
            on a downscaled copy). Face boxes are divided by it so they land
            in native source coordinates.

    Returns:
        ``{"faces": [[x, y, w, h], ...], "qa": [qa entry, ...]}``. Never raises.
    """
    faces: List[List[int]] = []
    try:
        div = float(scale) if scale and float(scale) > 0 else 1.0
        mult = _proxy_multiplier(result) / div
        for fc in getattr(result, "face_contexts", None) or []:
            face_data = getattr(fc, "face_data", None)
            bbox = getattr(face_data, "bbox", None)
            if bbox is None or len(bbox) != 4:
                continue
            x, y, w, h = (float(v) for v in bbox)
            if w <= 0 or h <= 0:
                continue
            faces.append([int(round(x * mult)), int(round(y * mult)),
                          int(round(w * mult)), int(round(h * mult))])
    except Exception as exc:  # defensive: never break a batch for review metadata
        logger.warning("Could not read face boxes for review: %s", exc)
        faces = []
    return {"faces": faces, "qa": qa_entries(getattr(result, "qa", None))}


def review_root_for(output_dir: Optional[Path], input_path: Path) -> Path:
    """Return the review root: the batch output dir, else the input folder."""
    if output_dir is not None:
        return Path(output_dir)
    input_path = Path(input_path)
    return input_path if input_path.is_dir() else input_path.parent


def _abs(path: Union[Path, str]) -> Path:
    """Absolute, normalised path without requiring it to exist."""
    return Path(path).expanduser().resolve(strict=False)


def record_key(source: Union[Path, str]) -> str:
    """Deterministic, filesystem-safe, collision-safe key for a source path.

    ``f"{safe_stem}-{sha1(absolute path)[:8]}"`` — two sources with the same
    stem in different folders (or with different extensions) get distinct
    keys; the same path always yields the same key.
    """
    abs_path = _abs(source)
    stem = _SAFE_STEM_RE.sub("_", abs_path.stem).strip("._") or "image"
    digest = hashlib.sha1(str(abs_path).encode("utf-8")).hexdigest()[:8]
    return f"{stem[:_SAFE_STEM_MAX]}-{digest}"


def _review_dir(root: Path) -> Path:
    return Path(root) / REVIEW_DIRNAME


def _records_dir(root: Path) -> Path:
    return _review_dir(root) / _RECORDS_SUBDIR


def _thumbs_dir(root: Path) -> Path:
    return _review_dir(root) / _THUMBS_SUBDIR


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write *data* to *path* via a same-directory temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError as exc:
            logger.debug("Could not remove temp file %s: %s", tmp, exc)
        raise


def write_review_record(root: Path, record: ReviewRecord) -> Path:
    """Atomically write *record* to ``<root>/.retouch-review/records/<key>.json``.

    Fills ``created_at`` when empty. Returns the record path.
    """
    if not record.created_at:
        record.created_at = _now_iso()
    path = _records_dir(root) / f"{record_key(record.source)}.json"
    payload = json.dumps(record.to_dict(), indent=2, ensure_ascii=False, default=str)
    _atomic_write_bytes(path, payload.encode("utf-8"))
    return path


def _record_sort_key(record: ReviewRecord) -> Tuple[str, str]:
    return ((record.relative or Path(record.source).name).casefold(), record.source)


def load_review_records(root: Path) -> List[ReviewRecord]:
    """Load every record under *root*, sorted by display name then source.

    Corrupt or unreadable record files are skipped with a logged warning.
    """
    records: List[ReviewRecord] = []
    rec_dir = _records_dir(root)
    if not rec_dir.is_dir():
        return records
    for path in sorted(rec_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping):
                raise ValueError("record is not a JSON object")
            records.append(ReviewRecord.from_dict(data))
        except (OSError, ValueError, TypeError) as exc:
            logger.warning("Skipping unreadable review record %s: %s", path, exc)
    records.sort(key=_record_sort_key)
    return records


# ---------------------------------------------------------------------------
# Thumbnails (runs in worker processes when workers > 1)
# ---------------------------------------------------------------------------

class _DropBitDepthNotice(logging.Filter):
    """Silence the ingest reader's per-file 16->8-bit notice for thumbnails.

    Reducing a 16-bit output to an 8-bit preview is the intent here, so the
    reader's warning would only add one line of noise per image.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "8-bit non-RAW working contract" not in record.getMessage()


_BIT_DEPTH_NOTICE_FILTER = _DropBitDepthNotice()


def _read_bgr8(path: str) -> Optional[np.ndarray]:
    """Decode *path* to a uint8 BGR image, or ``None`` if unreadable.

    Uses the engine ingest reader (EXIF orientation, ICC -> sRGB, RAW via
    rawpy) so boxes recorded in engine coordinates line up; falls back to
    ``cv2.imread`` (which also converts 16-bit to 8-bit).
    """
    img: Optional[np.ndarray] = None
    io_logger = logging.getLogger("retouch.io")
    io_logger.addFilter(_BIT_DEPTH_NOTICE_FILTER)
    try:
        from .io import imread_engine_with_context
        img, _ctx = imread_engine_with_context(path, prefer_16bit=False)
    except Exception as exc:  # any decoder failure -> try OpenCV
        logger.debug("engine reader failed for %s (%s); trying cv2.imread", path, exc)
        img = None
    finally:
        io_logger.removeFilter(_BIT_DEPTH_NOTICE_FILTER)
    if img is None:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.dtype == np.uint16:
        img = (img >> 8).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _fit(img: np.ndarray, edge: int) -> np.ndarray:
    """Downscale so the long edge is at most *edge* (never upscales)."""
    h, w = img.shape[:2]
    long_edge = max(h, w)
    if edge <= 0 or long_edge <= edge:
        return img
    s = edge / float(long_edge)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=cv2.INTER_AREA)


def _write_jpeg(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, _THUMB_JPEG_QUALITY])
    if not ok:
        raise OSError(f"JPEG encode failed for {path}")
    _atomic_write_bytes(path, buf.tobytes())


def _face_crop_box(box: Sequence[float], frame: Tuple[int, int],
                   img_size: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
    """Map a face box from *frame* (w, h) coords into an image of *img_size*.

    Expands the box x1.8 around its centre and clamps it to the image.
    Returns ``(x0, y0, x1, y1)`` or ``None`` when the box falls outside.
    """
    fw, fh = frame
    iw, ih = img_size
    sx = iw / float(fw) if fw > 0 else 1.0
    sy = ih / float(fh) if fh > 0 else 1.0
    x, y, w, h = (float(v) for v in box)
    cx, cy = (x + w / 2.0) * sx, (y + h / 2.0) * sy
    hw, hh = w * sx * _FACE_EXPAND / 2.0, h * sy * _FACE_EXPAND / 2.0
    x0, y0 = max(0, int(round(cx - hw))), max(0, int(round(cy - hh)))
    x1, y1 = min(iw, int(round(cx + hw))), min(ih, int(round(cy + hh)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _render_image_thumbs(job: Dict[str, Any]) -> Dict[str, Any]:
    """Decode one image and write its main thumb + face crops.

    Top-level (picklable) so it can run in a ``ProcessPoolExecutor``.

    Args:
        job: ``{"src", "thumb", "faces": [(face_path, box)], "frame": (w, h)
            or None, "thumb_edge", "face_edge"}``. ``frame`` is the pixel
            frame the boxes are expressed in (None = this image's own size).

    Returns:
        ``{"thumb": bool, "faces": [bool, ...], "size": (w, h) or None,
        "error": str or None}``.
    """
    faces: List[Tuple[str, Sequence[float]]] = job.get("faces") or []
    out: Dict[str, Any] = {"thumb": False, "faces": [False] * len(faces),
                           "size": None, "error": None}
    try:
        img = _read_bgr8(job["src"])
        if img is None:
            out["error"] = f"could not decode {job['src']}"
            return out
        ih, iw = img.shape[:2]
        out["size"] = (iw, ih)
        if job.get("thumb"):
            thumb = _fit(img, int(job["thumb_edge"]))
            _write_jpeg(Path(job["thumb"]), thumb)
            if job.get("grid"):
                # Small card image so a grid of hundreds stays light.
                _write_jpeg(Path(job["grid"]), _fit(thumb, GRID_EDGE))
            out["thumb"] = True
        frame = tuple(job.get("frame") or (iw, ih))
        for i, (face_path, box) in enumerate(faces):
            crop_box = _face_crop_box(box, frame, (iw, ih))
            if crop_box is None:
                continue
            x0, y0, x1, y1 = crop_box
            _write_jpeg(Path(face_path), _fit(img[y0:y1, x0:x1], int(job["face_edge"])))
            out["faces"][i] = True
    except Exception as exc:  # report per-image; one bad file must not stop the page
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _mtime(path: Optional[Union[str, Path]]) -> Optional[float]:
    if not path:
        return None
    try:
        return os.stat(path).st_mtime
    except OSError:
        return None


def _is_fresh(target: Path, newer_than: float) -> bool:
    t = _mtime(target)
    return t is not None and t >= newer_than


def _image_size(path: str) -> Optional[Tuple[int, int]]:
    """Cheap (w, h) of an image from its header, EXIF-orientation aware."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
            orientation = im.getexif().get(0x0112, 1)
            if orientation in (5, 6, 7, 8):
                w, h = h, w
            return w, h
    except Exception as exc:  # unreadable header -> caller decodes instead
        logger.debug("Could not read image size of %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def _href(target: Optional[Union[str, Path]], root: Path) -> Optional[str]:
    """Relative, URL-escaped link from ``review.html`` to *target*."""
    if not target:
        return None
    try:
        rel = os.path.relpath(str(_abs(target)), str(_abs(root)))
    except ValueError:  # different drive on Windows
        return None
    return quote(Path(rel).as_posix(), safe="/")


def batch_id_for(root: Path) -> str:
    """localStorage namespace for a review root: sha1(abs root)[:12]."""
    return hashlib.sha1(str(_abs(root)).encode("utf-8")).hexdigest()[:12]


def apply_command_for(root: Path) -> str:
    """The shell command the page tells the user to run after exporting."""
    return (f"./run review apply {shlex.quote(str(_abs(root)))} "
            f"~/Downloads/{batch_id_for(root)}-decisions.json")


def _run_jobs(jobs: List[Dict[str, Any]], workers: int) -> List[Dict[str, Any]]:
    """Run thumbnail jobs serially or in a spawn-context process pool."""
    if not jobs:
        return []
    n = max(1, min(int(workers or 1), len(jobs)))
    if n > 1:
        try:
            ctx = multiprocessing.get_context("spawn")
            with ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
                return list(pool.map(_render_image_thumbs, jobs))
        except Exception as exc:  # pool unavailable (sandbox, pickling) -> serial
            logger.warning("Thumbnail pool failed (%s); rendering serially", exc)
    return [_render_image_thumbs(job) for job in jobs]


def _plan_item(root: Path, record: ReviewRecord, record_mtime: float,
               thumb_edge: int, face_edge: int) -> Tuple[Dict[str, Any], List[Tuple[str, Dict[str, Any]]]]:
    """Build the page item for *record* and the thumb jobs it still needs."""
    key = record_key(record.source)
    tdir = _thumbs_dir(root)
    rel_thumb = f"{REVIEW_DIRNAME}/{_THUMBS_SUBDIR}"
    faces = [list(b) for b in (record.faces or []) if isinstance(b, (list, tuple)) and len(b) == 4]
    source = record.source if _mtime(record.source) is not None else None
    output = record.output if (record.output and _mtime(record.output) is not None) else None

    # Face boxes are native-source coordinates, i.e. the frame of the image
    # the engine saw; the CLI writes outputs at that size, so the output is
    # the reference frame when present (a RAW before thumb may decode to a
    # slightly different size and is scaled into it).
    frame: Optional[Tuple[int, int]] = None
    if output:
        frame = _image_size(output)
    if frame is None and source:
        frame = _image_size(source)

    jobs: List[Tuple[str, Dict[str, Any]]] = []
    for side, src in (("before", source), ("after", output)):
        if not src:
            continue
        src_mtime = _mtime(src) or 0.0
        thumb = tdir / f"{key}_{side}.jpg"
        grid = tdir / f"{key}_{side}_grid.jpg"
        face_paths = [tdir / f"{key}_face{i}_{side}.jpg" for i in range(len(faces))]
        need_thumb = not (_is_fresh(thumb, src_mtime) and _is_fresh(grid, src_mtime))
        face_cutoff = max(src_mtime, record_mtime)
        need_faces = [(str(p), b) for p, b in zip(face_paths, faces)
                      if not _is_fresh(p, face_cutoff)]
        if need_thumb or need_faces:
            jobs.append((side, {
                "src": str(src),
                "thumb": str(thumb) if need_thumb else None,
                "grid": str(grid) if need_thumb else None,
                "faces": need_faces,
                "frame": frame,
                "thumb_edge": thumb_edge,
                "face_edge": face_edge,
            }))
        # Drop crops for faces that no longer exist (re-run found fewer).
        for stale in tdir.glob(f"{_glob_escape(key)}_face*_{side}.jpg"):
            m = re.fullmatch(rf"{re.escape(key)}_face(\d+)_{side}\.jpg", stale.name)
            if m and int(m.group(1)) >= len(faces):
                try:
                    stale.unlink()
                except OSError as exc:
                    logger.debug("Could not remove stale face crop %s: %s", stale, exc)

    qa = qa_entries(record.qa)
    item: Dict[str, Any] = {
        "key": key,
        "name": record.relative or Path(record.source).name,
        "status": record.status if record.status in STATUSES else "failed",
        "recipe": record.recipe,
        "elapsed_s": _to_float(record.elapsed_s),
        "before": f"{rel_thumb}/{key}_before.jpg" if source else None,
        "after": f"{rel_thumb}/{key}_after.jpg" if output else None,
        "grid": (f"{rel_thumb}/{key}_after_grid.jpg" if output
                 else f"{rel_thumb}/{key}_before_grid.jpg" if source else None),
        "output_href": _href(output, root),
        "compare_href": _href(record.compare, root) if _mtime(record.compare) is not None else None,
        "faces": [{"before": f"{rel_thumb}/{key}_face{i}_before.jpg" if source else None,
                   "after": f"{rel_thumb}/{key}_face{i}_after.jpg" if output else None}
                  for i in range(len(faces))],
        "qa_recorded": bool(record.qa_recorded),
        "flagged": any(e["flagged"] for e in qa),
        "qa": qa,
        "error": record.error,
    }
    if record.status not in STATUSES:
        item["error"] = record.error or f"unknown status {record.status!r}"
    return item, jobs


def _glob_escape(text: str) -> str:
    """Escape glob metacharacters in *text* (keys are already safe; belt and braces)."""
    return re.sub(r"([*?\[])", r"[\1]", text)


def _reconcile(item: Dict[str, Any], side: str, result: Dict[str, Any], root: Path) -> None:
    """Null out page paths whose thumb could not be produced."""
    if result.get("error"):
        logger.warning("Review thumbnail (%s) for %s: %s", side, item["name"], result["error"])
    thumb = item.get(side)
    if thumb and not (root / thumb).is_file():
        item[side] = None
    for face in item["faces"]:
        if face.get(side) and not (root / face[side]).is_file():
            face[side] = None


def build_page_data(root: Path, *, title: Optional[str] = None, workers: int = 1,
                    thumb_edge: int = 1600, face_edge: int = 480) -> Dict[str, Any]:
    """Generate missing/stale thumbnails and return the page data dict.

    See the module docstring and :func:`build_review_page`.
    """
    root = _abs(root)
    records = load_review_records(root)
    _thumbs_dir(root).mkdir(parents=True, exist_ok=True)
    rec_dir = _records_dir(root)

    items: List[Dict[str, Any]] = []
    jobs: List[Dict[str, Any]] = []
    job_owner: List[Tuple[int, str]] = []
    for record in records:
        record_mtime = _mtime(rec_dir / f"{record_key(record.source)}.json") or 0.0
        item, item_jobs = _plan_item(root, record, record_mtime, thumb_edge, face_edge)
        for side, job in item_jobs:
            job_owner.append((len(items), side))
            jobs.append(job)
        items.append(item)

    if jobs:
        logger.info("Rendering review thumbnails for %d image(s)", len(jobs))
    for (idx, side), result in zip(job_owner, _run_jobs(jobs, workers)):
        _reconcile(items[idx], side, result, root)
    for item in items:  # fresh thumbs that vanished, or never rendered
        for side in ("before", "after"):
            _reconcile(item, side, {}, root)

    summary: Dict[str, int] = {"total": len(items)}
    for status in STATUSES:
        summary[status] = sum(1 for it in items if it["status"] == status)
    summary["flagged"] = sum(1 for it in items if it["flagged"])

    return {
        "version": PAGE_DATA_VERSION,
        "title": title or f"{root.name or str(root)} review",
        "batch_id": batch_id_for(root),
        "generated_at": _now_iso(),
        "root": str(root),
        "apply_command": apply_command_for(root),
        "summary": summary,
        "items": items,
    }


def build_review_page(root: Path, *, title: Optional[str] = None, workers: int = 1,
                      thumb_edge: int = 1600, face_edge: int = 480) -> Path:
    """(Re)generate thumbnails and write ``<root>/review.html`` atomically.

    Thumbnails are regenerated only when missing or older than their source
    image (face crops also when older than the record). Missing source or
    output files are tolerated: the item's image paths are ``None`` and the
    page shows them as missing.

    Args:
        root: Review root (batch output dir).
        title: Page title; defaults to ``"<root name> review"``.
        workers: Thumbnail processes; > 1 uses a process pool.
        thumb_edge: Long edge of main thumbnails in pixels.
        face_edge: Long edge of face crops in pixels.

    Returns:
        Path of the written ``review.html``.
    """
    from .review_template import render_review_html

    root = _abs(root)
    data = build_page_data(root, title=title, workers=workers,
                           thumb_edge=thumb_edge, face_edge=face_edge)
    html = render_review_html(data)
    page = root / PAGE_NAME
    _atomic_write_bytes(page, html.encode("utf-8"))
    logger.info("Review page written: %s (%d items)", page, len(data["items"]))
    return page


# ---------------------------------------------------------------------------
# Backfill for batches that ran before records existed
# ---------------------------------------------------------------------------

def _is_image(path: Path) -> bool:
    from .io import IMAGE_EXTENSIONS
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _iter_files(base: Path, recursive: bool, skip_dirs: Sequence[str] = ()) -> Iterable[Path]:
    """Yield non-hidden files under *base* in sorted order.

    Hidden files/dirs (leading dot) are always skipped; *skip_dirs* names are
    skipped at the top level of *base* only (e.g. ``social``, ``picks``).
    """
    if not recursive:
        yield from sorted(p for p in base.iterdir()
                          if p.is_file() and not p.name.startswith("."))
        return
    for dirpath, dirnames, filenames in os.walk(base):
        at_top = Path(dirpath) == base
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith(".") and not (at_top and d in skip_dirs))
        for name in sorted(filenames):
            if not name.startswith("."):
                yield Path(dirpath) / name


def backfill_records(root: Path, source_dir: Path, *, recursive: bool = False) -> int:
    """Write ``done`` records for a past batch that has none.

    Pairs each output image in *root* (excluding ``*_compare.*``, hidden
    files/dirs such as ``.retouch-review``, and the top-level ``social``,
    ``picks`` and ``rejected`` folders) with the source in *source_dir*
    that has the same relative path minus extension. Backfilled records
    carry ``qa=[]``, ``faces=[]`` and ``qa_recorded=False``. Existing records
    are never overwritten.

    Returns:
        Number of records written.
    """
    root, source_dir = _abs(root), _abs(source_dir)
    skip = _BACKFILL_SKIP_DIRS

    sources: Dict[str, List[Path]] = {}
    for src in _iter_files(source_dir, recursive, skip):
        if not _is_image(src) or src.stem.endswith("_compare"):
            continue
        rel_stem = src.relative_to(source_dir).with_suffix("").as_posix()
        sources.setdefault(rel_stem, []).append(src)

    written = 0
    for out in _iter_files(root, recursive, skip):
        if not _is_image(out) or out.stem.endswith("_compare"):
            continue
        rel_stem = out.relative_to(root).with_suffix("").as_posix()
        candidates = [s for s in sources.get(rel_stem, []) if s != out]
        if not candidates:
            continue
        if len(candidates) > 1:
            logger.info("Backfill: %d sources match %s; using %s",
                        len(candidates), out.name, candidates[0].name)
        src = candidates[0]
        if (_records_dir(root) / f"{record_key(src)}.json").exists():
            continue
        compare = out.with_name(f"{out.stem}_compare{out.suffix}")
        write_review_record(root, ReviewRecord(
            source=str(src),
            output=str(out),
            compare=str(compare) if compare.is_file() else None,
            status="done",
            relative=src.relative_to(source_dir).as_posix(),
            qa_recorded=False,
        ))
        written += 1
    return written


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------

def _load_decisions(root: Path, decisions: Union[Path, str, Mapping[str, Any]]) -> Dict[str, str]:
    """Accept a mapping, an exported decisions file, or a bare-mapping file."""
    if isinstance(decisions, Mapping):
        data: Any = decisions
    else:
        data = json.loads(Path(decisions).expanduser().read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError(f"{decisions}: expected a JSON object")
    if isinstance(data.get("decisions"), Mapping):
        exported_id = data.get("batch_id")
        if exported_id and exported_id != batch_id_for(root):
            logger.warning("Decisions were exported for batch %s but %s is batch %s; "
                           "keys that do not match are counted as missing",
                           exported_id, root, batch_id_for(root))
        data = data["decisions"]
    out: Dict[str, str] = {}
    for key, value in data.items():
        if value in DECISIONS:
            out[str(key)] = str(value)
        elif value not in (None, "", "clear"):
            logger.warning("Ignoring unknown decision %r for %s", value, key)
    return out


def _dest_rel(root: Path, record: ReviewRecord, path: Path) -> Path:
    """Relative location to preserve under picks/rejects for *path*."""
    try:
        return path.relative_to(root)
    except ValueError:
        parent = Path(record.relative).parent if record.relative else Path()
        return parent / path.name


def apply_decisions(root: Path, decisions: Union[Path, Mapping[str, str]], *,
                    picks_dir: str = _DEFAULT_PICKS_DIR, move_rejects: bool = False,
                    rejects_dir: str = _DEFAULT_REJECTS_DIR) -> Dict[str, int]:
    """Apply exported pick/reject decisions. Never deletes or overwrites.

    Picks: the output (only) is COPIED to ``<root>/<picks_dir>/`` preserving
    its subdirectory relative to *root*. Rejects: only with *move_rejects*,
    the output and its ``_compare`` file are MOVED to ``<root>/<rejects_dir>/``
    and the record is updated to point at the new location. A destination
    file that already exists is skipped and counted as ``exists``; unknown
    keys and missing outputs count as ``missing``.

    Args:
        root: Review root.
        decisions: ``{record_key: "pick"|"reject"}`` or the exported JSON file.
        picks_dir: Picks folder name under *root*.
        move_rejects: Move rejected outputs aside (default: leave in place).
        rejects_dir: Rejects folder name under *root*.

    Returns:
        ``{"picked": n, "rejected": n, "exists": n, "missing": n}``.
    """
    root = _abs(root)
    wanted = _load_decisions(root, decisions)
    by_key = {record_key(r.source): r for r in load_review_records(root)}
    counts = {"picked": 0, "rejected": 0, "exists": 0, "missing": 0}

    for key, decision in sorted(wanted.items()):
        record = by_key.get(key)
        if record is None:
            logger.warning("No review record for decision key %s", key)
            counts["missing"] += 1
            continue
        if decision == "reject" and not move_rejects:
            continue
        output = Path(record.output) if record.output else None
        if output is None or not output.is_file():
            logger.warning("Output missing for %s (%s)", record.relative or record.source, decision)
            counts["missing"] += 1
            continue

        if decision == "pick":
            dest = root / picks_dir / _dest_rel(root, record, output)
            if dest.exists():
                counts["exists"] += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output, dest)
            counts["picked"] += 1
            continue

        # reject + move_rejects
        dest = root / rejects_dir / _dest_rel(root, record, output)
        if dest.exists():
            counts["exists"] += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(output), str(dest))
        record.output = str(dest)
        counts["rejected"] += 1
        if record.compare and Path(record.compare).is_file():
            compare = Path(record.compare)
            cdest = root / rejects_dir / _dest_rel(root, record, compare)
            if cdest.exists():
                counts["exists"] += 1
            else:
                cdest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(compare), str(cdest))
                record.compare = str(cdest)
        write_review_record(root, record)
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retouch.review_page",
        description="Build a batch review page or apply its exported decisions.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="(Re)build <root>/review.html")
    b.add_argument("root", type=Path, help="Review root (the batch output folder)")
    b.add_argument("--source", type=Path, default=None,
                   help="Source folder: backfill records for a batch run without them")
    b.add_argument("--recursive", action="store_true",
                   help="With --source: match sources/outputs in subfolders too")
    b.add_argument("--title", default=None, help="Page title")
    b.add_argument("--workers", type=int, default=1, help="Thumbnail worker processes")

    a = sub.add_parser("apply", help="Copy picks / move rejects from an exported decisions file")
    a.add_argument("root", type=Path, help="Review root (the batch output folder)")
    a.add_argument("decisions", type=Path, help="Exported <batch_id>-decisions.json")
    a.add_argument("--picks-dir", default=_DEFAULT_PICKS_DIR,
                   help="Folder under ROOT for picks (default: picks)")
    a.add_argument("--move-rejects", action="store_true",
                   help="Also move rejected outputs + compares into --rejects-dir")
    a.add_argument("--rejects-dir", default=_DEFAULT_REJECTS_DIR,
                   help="Folder under ROOT for rejects (default: rejected)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m retouch.review_page``. Returns an exit code."""
    args = _build_parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    root: Path = args.root
    if not root.is_dir():
        print(f"error: review root not found: {root}", file=sys.stderr)
        return 1

    if args.command == "build":
        if args.source is not None:
            if not args.source.is_dir():
                print(f"error: source folder not found: {args.source}", file=sys.stderr)
                return 1
            n = backfill_records(root, args.source, recursive=args.recursive)
            print(f"Backfilled {n} record(s)")
        if not load_review_records(root):
            print(f"error: no review records in {root / REVIEW_DIRNAME}; "
                  "run a batch first or pass --source DIR to backfill", file=sys.stderr)
            return 1
        page = build_review_page(root, title=args.title, workers=args.workers)
        print(f"Review page → {page}")
        return 0

    if not args.decisions.is_file():
        print(f"error: decisions file not found: {args.decisions}", file=sys.stderr)
        return 1
    try:
        counts = apply_decisions(root, args.decisions, picks_dir=args.picks_dir,
                                 move_rejects=args.move_rejects, rejects_dir=args.rejects_dir)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Picked {counts['picked']} → {root / args.picks_dir}")
    if args.move_rejects:
        print(f"Rejected {counts['rejected']} → {root / args.rejects_dir}")
    if counts["exists"]:
        print(f"Skipped {counts['exists']} already present at the destination")
    if counts["missing"]:
        print(f"Missing {counts['missing']} (no record or output file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
