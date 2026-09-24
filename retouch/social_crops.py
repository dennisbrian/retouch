"""Face-aware social crops for retouched photos.

Turns each finished photo into the shapes people actually post: 4:5 for the
Instagram feed, 9:16 for Reels / TikTok / Stories, 1:1 square and 3:4. The
crop is the largest rectangle of the target shape that fits the photo, so no
resolution is thrown away beyond what the shape itself removes, and it is
placed around the subject instead of the frame centre:

* the largest face, plus any face at least ``SUBJECT_REL_AREA`` of its area
  (a duo or group), defines the subject; small background faces are ignored;
* horizontally the subject is centred;
* vertically the subject's face sits on the upper third, but the crop never
  cuts closer than ``HEADROOM`` face-heights above the forehead when it can
  avoid it, so wigs, ears and headpieces survive;
* when the faces are spread wider than the crop, it centres on the group;
* with no face it falls back to a centre crop.

Crops are written as sRGB JPEG without EXIF (so GPS is stripped before
posting), downscaled to the platform size with light output sharpening and
never upscaled.

Usage::

    python -m retouch.social_crops out/ --formats 4:5,9:16
    python cli.py shoot/ -o out/ --social-crops        # during a batch
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

_logger = logging.getLogger(__name__)

Box = Tuple[int, int, int, int]  # (x, y, w, h) in image pixels

# A face counts as part of the subject when its area is at least this
# fraction of the largest face (keeps duos together, drops the crowd).
SUBJECT_REL_AREA = 0.35
# Room kept above the forehead, in face heights. Landmark boxes stop at the
# hairline; cosplay wigs, ears and headpieces extend well above it.
HEADROOM = 0.6
# Room kept below the chin, in face heights, when the crop is tight.
CHIN_ROOM = 0.25
# Where the subject's face centre sits vertically within the crop.
FACE_LINE = 1.0 / 3.0

# Output subfolders that never hold retouched originals: earlier crops, the
# review page's cache (dot folder) and its default picks/rejects copies.
_SKIP_DIRS = {"social", "picks", "rejected"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@dataclasses.dataclass(frozen=True)
class SocialFormat:
    key: str
    slug: str
    label: str
    ratio: Tuple[int, int]
    width: int
    height: int

    @property
    def aspect(self) -> float:
        return self.ratio[0] / self.ratio[1]


FORMATS = {
    f.key: f
    for f in (
        SocialFormat("4:5", "4x5", "Instagram feed", (4, 5), 1080, 1350),
        SocialFormat("9:16", "9x16", "Reels / TikTok / Stories", (9, 16), 1080, 1920),
        SocialFormat("1:1", "1x1", "Square", (1, 1), 1080, 1080),
        SocialFormat("3:4", "3x4", "Instagram grid / Threads", (3, 4), 1080, 1440),
    )
}
DEFAULT_FORMATS = ("4:5", "9:16", "1:1")


def parse_formats(spec: Union[str, Iterable[str]]) -> List[SocialFormat]:
    """Parse ``"4:5,9:16"``, ``"4x5"``, ``"all"`` or a list of keys."""
    if isinstance(spec, str):
        items = spec.split(",")
    else:
        items = list(spec)
    by_slug = {f.slug: f for f in FORMATS.values()}
    out: List[SocialFormat] = []
    for raw in items:
        item = str(raw).strip().lower()
        if not item:
            continue
        if item == "all":
            chosen = list(FORMATS.values())
        elif item in FORMATS:
            chosen = [FORMATS[item]]
        elif item in by_slug:
            chosen = [by_slug[item]]
        else:
            raise ValueError(
                f"unknown social format {raw!r}; choose from "
                f"{', '.join(FORMATS)} or 'all'"
            )
        for fmt in chosen:
            if fmt not in out:
                out.append(fmt)
    if not out:
        raise ValueError("no social formats given")
    return out


@dataclasses.dataclass
class CropPlan:
    x: int
    y: int
    w: int
    h: int
    reason: str  # "face" | "group" | "no_face" | "faces_wider_than_crop"
    n_faces: int = 0


@dataclasses.dataclass
class CropResult:
    path: Path
    fmt_key: str
    plan: Optional[CropPlan]
    status: str  # "done" | "skipped" | "failed"
    error: Optional[str] = None


def select_subject_faces(
    boxes: Sequence[Box], min_rel_area: float = SUBJECT_REL_AREA,
) -> List[Box]:
    """Keep the largest face and any face of comparable size."""
    boxes = [tuple(int(v) for v in b) for b in boxes if b[2] > 0 and b[3] > 0]
    if not boxes:
        return []
    largest = max(b[2] * b[3] for b in boxes)
    return [b for b in boxes if b[2] * b[3] >= min_rel_area * largest]


def _max_crop_size(img_w: int, img_h: int, fmt: SocialFormat) -> Tuple[int, int]:
    rw, rh = fmt.ratio
    if img_w * rh >= img_h * rw:  # image wider than the target shape
        ch = img_h
        cw = min(img_w, int(round(img_h * rw / rh)))
    else:
        cw = img_w
        ch = min(img_h, int(round(img_w * rh / rw)))
    return cw, ch


def _clamp(v: float, lo: int, hi: int) -> int:
    return int(round(min(max(v, lo), hi)))


def plan_crop(
    img_w: int,
    img_h: int,
    face_boxes: Sequence[Box],
    fmt: SocialFormat,
    subject_top: Optional[int] = None,
) -> CropPlan:
    """Place the largest ``fmt``-shaped crop around the subject.

    ``subject_top`` is the highest row of the subject above the faces (from
    :func:`find_subject_top`); when given, the crop tries to keep it in.
    """
    cw, ch = _max_crop_size(img_w, img_h, fmt)
    max_x, max_y = img_w - cw, img_h - ch
    faces = select_subject_faces(face_boxes)
    if not faces:
        return CropPlan(max_x // 2, max_y // 2, cw, ch, "no_face", 0)

    ux0 = min(b[0] for b in faces)
    uy0 = min(b[1] for b in faces)
    ux1 = max(b[0] + b[2] for b in faces)
    uy1 = max(b[1] + b[3] for b in faces)
    uw = ux1 - ux0
    face_h = max(b[3] for b in faces)
    reason = "group" if len(faces) > 1 else "face"

    # Horizontal: centre the subject. If it is wider than the crop there is
    # no placement that keeps everyone; centring on the group is the least bad.
    cx = (ux0 + ux1) / 2.0
    x = _clamp(cx - cw / 2.0, 0, max_x)
    if uw > cw:
        reason = "faces_wider_than_crop"

    # Vertical: face centre on the upper third...
    cy = (uy0 + uy1) / 2.0
    y = cy - ch * FACE_LINE
    face_top = uy0 - HEADROOM * face_h
    bottom_wanted = uy1 + CHIN_ROOM * face_h
    # ...but keep the whole head in: the person mask's top (wig, ears,
    # headpiece) when it fits with the chin, else plain face headroom.
    full_top = face_top
    if subject_top is not None:
        full_top = min(face_top, subject_top - 0.15 * face_h)
    if y > full_top and full_top + ch >= bottom_wanted:
        y = full_top
    elif y > face_top:
        if face_top + ch >= bottom_wanted:
            y = face_top
        else:
            # Too tight for both margins: centre on the faces themselves.
            y = cy - ch / 2.0
    if y + ch < bottom_wanted and bottom_wanted - face_top <= ch:
        y = bottom_wanted - ch
    y = _clamp(y, 0, max_y)
    return CropPlan(x, y, cw, ch, reason, len(faces))


def render_crop(
    img: np.ndarray,
    plan: CropPlan,
    fmt: SocialFormat,
    size: str = "platform",
    sharpen: float = 0.3,
) -> np.ndarray:
    """Cut ``plan`` out of ``img`` and size it for posting (uint8 BGR)."""
    if img.dtype != np.uint8:
        if img.dtype == np.uint16:
            img = (img.astype(np.float32) / 257.0)
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    crop = img[plan.y:plan.y + plan.h, plan.x:plan.x + plan.w]
    if size == "full" or plan.w <= fmt.width:
        return np.ascontiguousarray(crop)
    if size != "platform":
        raise ValueError(f"size must be 'platform' or 'full', got {size!r}")
    out = cv2.resize(crop, (fmt.width, fmt.height), interpolation=cv2.INTER_AREA)
    if sharpen > 0:
        # Light "output sharpening for screen": a downscale softens edges.
        blur = cv2.GaussianBlur(out, (0, 0), 0.8)
        out = cv2.addWeighted(out, 1.0 + sharpen, blur, -sharpen, 0)
    return out


def _make_detector():
    from retouch.detection import FaceDetector

    return FaceDetector(max_faces=10, allow_unavailable=True)


def detect_faces(img: np.ndarray, detector=None, max_side: int = 1280) -> List[Box]:
    """Face boxes in full-image pixels, detected on a small proxy."""
    if detector is None:
        return []
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    small = img
    if scale < 1.0:
        small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    if small.dtype != np.uint8:
        small = np.clip(np.rint(small), 0, 255).astype(np.uint8)
    boxes: List[Box] = []
    for face in detector.detect(small) or []:
        bx, by, bw, bh = face.bbox
        boxes.append((
            int(round(bx / scale)), int(round(by / scale)),
            int(round(bw / scale)), int(round(bh / scale)),
        ))
    return boxes


def find_subject_top(
    person_mask: Optional[np.ndarray],
    face_boxes: Sequence[Box],
    threshold: float = 0.4,
    max_face_heights: float = 3.0,
) -> Optional[int]:
    """Top row of the person mask above the subject's faces.

    Looks only in a band one face-width either side of the faces and at most
    ``max_face_heights`` above them, so a tall background or a neighbour does
    not pull the crop up. Returns ``None`` without a usable mask.
    """
    faces = select_subject_faces(face_boxes)
    if person_mask is None or not faces:
        return None
    mh, mw = person_mask.shape[:2]
    ux0 = min(b[0] for b in faces)
    ux1 = max(b[0] + b[2] for b in faces)
    uy0 = min(b[1] for b in faces)
    fw = max(b[2] for b in faces)
    fh = max(b[3] for b in faces)
    x0, x1 = max(0, ux0 - fw), min(mw, ux1 + fw)
    y_lo = max(0, int(uy0 - max_face_heights * fh))
    y_hi = min(mh, max(0, uy0))
    if x1 <= x0 or y_hi <= y_lo:
        return None
    band = person_mask[y_lo:y_hi, x0:x1]
    rows = np.flatnonzero((band > threshold).any(axis=1))
    if rows.size == 0:
        return None
    return int(y_lo + rows[0])


def detect_subject(
    img: np.ndarray, detector=None, max_side: int = 1280,
) -> Tuple[List[Box], Optional[int]]:
    """Face boxes plus the subject's top row, in full-image pixels."""
    boxes = detect_faces(img, detector, max_side=max_side)
    if not boxes or not hasattr(detector, "segment_person"):
        return boxes, None
    h, w = img.shape[:2]
    scale = min(1.0, float(max_side) / max(h, w))
    small = img
    if scale < 1.0:
        small = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                           interpolation=cv2.INTER_AREA)
    if small.dtype != np.uint8:
        small = np.clip(np.rint(small), 0, 255).astype(np.uint8)
    try:
        mask = detector.segment_person(small)
    except Exception as exc:
        _logger.debug("social crops: person segmentation failed: %s", exc)
        return boxes, None
    if mask is None or mask.shape[:2] != small.shape[:2] or float(np.min(mask)) >= 0.5:
        # Missing, mis-sized, or the all-ones "unavailable" fallback.
        return boxes, None
    small_boxes = [tuple(int(round(v * scale)) for v in b) for b in boxes]
    top = find_subject_top(mask, small_boxes)
    return boxes, (None if top is None else int(round(top / scale)))


def crop_path(social_dir: Union[str, Path], image_path: Union[str, Path], fmt: SocialFormat) -> Path:
    stem = Path(image_path).stem
    return Path(social_dir) / fmt.slug / f"{stem}_{fmt.slug}.jpg"


def export_social_crops(
    image_path: Union[str, Path],
    social_dir: Union[str, Path],
    formats: Sequence[SocialFormat],
    *,
    detector=None,
    size: str = "platform",
    quality: int = 92,
    force: bool = False,
) -> List[CropResult]:
    """Write one crop per format for ``image_path``."""
    from retouch.io import imread_exif_with_context, write_image_with_color_context

    image_path = Path(image_path)
    targets = [(fmt, crop_path(social_dir, image_path, fmt)) for fmt in formats]
    results: List[CropResult] = []
    pending = []
    for fmt, path in targets:
        if path.exists() and not force:
            results.append(CropResult(path, fmt.key, None, "skipped"))
        else:
            pending.append((fmt, path))
    if not pending:
        return results

    try:
        img, color_context = imread_exif_with_context(image_path)
        if img is None:
            raise ValueError("could not read image")
        boxes, subject_top = detect_subject(img, detector)
    except Exception as exc:  # unreadable file: report per format, keep going
        _logger.warning("social crops: %s: %s", image_path.name, exc)
        return results + [
            CropResult(path, fmt.key, None, "failed", str(exc)) for fmt, path in pending
        ]

    h, w = img.shape[:2]
    for fmt, path in pending:
        plan = plan_crop(w, h, boxes, fmt, subject_top=subject_top)
        try:
            out = render_crop(img, plan, fmt, size=size)
            path.parent.mkdir(parents=True, exist_ok=True)
            write_image_with_color_context(
                str(path), out, color_context, bit_depth=8, quality=quality,
            )
            results.append(CropResult(path, fmt.key, plan, "done"))
        except Exception as exc:
            _logger.warning("social crops: %s %s: %s", image_path.name, fmt.key, exc)
            results.append(CropResult(path, fmt.key, plan, "failed", str(exc)))
    return results


def find_crop_sources(input_path: Union[str, Path], recursive: bool = False) -> List[Path]:
    """Retouched images under ``input_path``, minus compares and old crops."""
    p = Path(input_path)
    if p.is_file():
        return [p] if p.suffix.lower() in IMAGE_EXTENSIONS else []
    it = p.rglob("*") if recursive else p.glob("*")
    out = []
    for f in sorted(it):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if f.stem.endswith("_compare"):
            continue
        parents = f.relative_to(p).parts[:-1]
        if any(d in _SKIP_DIRS or d.startswith(".") for d in parents):
            continue
        out.append(f)
    return out


def export_folder(
    paths: Sequence[Union[str, Path]],
    social_dir: Union[str, Path],
    formats: Sequence[SocialFormat],
    *,
    size: str = "platform",
    quality: int = 92,
    force: bool = False,
    detector=None,
    progress=None,
) -> dict:
    """Export crops for every path; creates (and closes) a detector if needed."""
    own_detector = detector is None
    if own_detector:
        detector = _make_detector()
        if not getattr(detector, "available", True):
            _logger.warning(
                "social crops: face detection unavailable (%s); using centre crops",
                getattr(detector, "unavailable_reason", "unknown"),
            )
    summary = {"done": 0, "skipped": 0, "failed": 0, "results": []}
    try:
        for i, path in enumerate(paths):
            res = export_social_crops(
                path, social_dir, formats, detector=detector,
                size=size, quality=quality, force=force,
            )
            for r in res:
                summary[r.status] += 1
            summary["results"].extend(res)
            if progress is not None:
                progress(i + 1, len(paths), Path(path).name)
    finally:
        if own_detector and hasattr(detector, "close"):
            detector.close()
    return summary


def format_summary(summary: dict, social_dir: Union[str, Path]) -> str:
    return (
        f"Social crops: {summary['done']} written, {summary['skipped']} skipped, "
        f"{summary['failed']} failed → {social_dir}"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m retouch.social_crops",
        description="Make face-aware 4:5 / 9:16 / 1:1 / 3:4 crops of retouched photos.",
    )
    parser.add_argument("input", help="Retouched image or folder")
    parser.add_argument("-o", "--output", help="Crop folder (default: <input>/social)")
    parser.add_argument("--formats", default=",".join(DEFAULT_FORMATS),
                        help=f"Comma list of {', '.join(FORMATS)} or 'all' "
                             f"(default: {','.join(DEFAULT_FORMATS)})")
    parser.add_argument("--size", choices=["platform", "full"], default="platform",
                        help="platform = 1080 px wide (default); full = native crop resolution")
    parser.add_argument("--quality", type=int, default=92, help="JPEG quality (default: 92)")
    parser.add_argument("-f", "--force", action="store_true", help="Overwrite existing crops")
    parser.add_argument("-r", "--recursive", action="store_true", help="Include subfolders")
    args = parser.parse_args(argv)

    try:
        formats = parse_formats(args.formats)
    except ValueError as exc:
        parser.error(str(exc))
    src = Path(args.input).expanduser()
    if not src.exists():
        print(f"✖ Input not found: {src}")
        return 1
    files = find_crop_sources(src, args.recursive)
    if not files:
        print("✖ No images found")
        return 1
    if args.output:
        social_dir = Path(args.output).expanduser()
    else:
        social_dir = (src if src.is_dir() else src.parent) / "social"

    def _progress(i, n, name):
        print(f"  [{i}/{n}] {name}", flush=True)

    summary = export_folder(
        files, social_dir, formats, size=args.size, quality=args.quality,
        force=args.force, progress=_progress,
    )
    for r in summary["results"]:
        if r.status == "failed":
            print(f"  ✖ {r.path.name}: {r.error}")
    print(format_summary(summary, social_dir))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
