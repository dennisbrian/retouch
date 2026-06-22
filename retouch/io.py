"""Shared image I/O helpers for CLI, GUI, and batch tooling."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp",
    ".raf", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf", ".rw2", ".pef", ".srw", ".x3f",
}

RAW_EXTENSIONS = {
    ".raf", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf", ".rw2", ".pef", ".srw", ".x3f",
}

EXPORT_RES_MAP: Dict[str, Optional[int]] = {
    "Original": None,
    "4K (3840px)": 3840,
    "2K (2048px)": 2048,
    "Full HD (1920px)": 1920,
    "HD (1280px)": 1280,
    "720px": 720,
}

EXT_MAP: Dict[str, str] = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WebP": ".webp",
}


def imread_exif(path: Union[str, Path]) -> np.ndarray:
    """Read image (supports RAW via rawpy), applying EXIF orientation.

    Args:
        path: Filesystem path to the image. RAW formats are decoded via rawpy.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    path = Path(path)
    if path.suffix.lower() in RAW_EXTENSIONS:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, bright=1.5)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    from PIL import ImageOps

    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img) or pil_img
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def resize_for_processing(
    img_bgr: np.ndarray,
    max_dim: Optional[int],
) -> Tuple[np.ndarray, float]:
    """Downscale so longest side ≤ max_dim.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        max_dim: Maximum allowed longer-side in pixels. ``None`` is a no-op.

    Returns:
        Tuple of (resized image, scale factor).
    """
    if max_dim is None:
        return img_bgr, 1.0
    h, w = img_bgr.shape[:2]
    current_max = max(h, w)
    if current_max <= max_dim:
        return img_bgr, 1.0
    scale = max_dim / current_max
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def output_format(img_path: Union[str, Path], format_arg: str) -> str:
    """Resolve CLI --format for a single input file.

    Args:
        img_path: Path to the input image (used to derive the default format).
        format_arg: ``--format`` CLI value, typically ``"same"`` to mirror the input.

    Returns:
        Lowercase format string suitable for OpenCV: ``"jpg"``, ``"png"`` or
        ``"webp"``.
    """
    if format_arg != "same":
        return format_arg
    ext = Path(img_path).suffix.lower()
    if ext == ".png":
        return "png"
    if ext == ".webp":
        return "webp"
    return "jpg"


def encode_write_params(fmt: str, quality: int) -> List[int]:
    """Return OpenCV imwrite params for lossy formats.

    Args:
        fmt: Lowercase format string (``"jpg"``, ``"jpeg"`` or ``"webp"``).
        quality: 0–100 quality value.

    Returns:
        OpenCV imwrite flag list, possibly empty for lossless formats.
    """
    if fmt in ("jpg", "jpeg"):
        return [cv2.IMWRITE_JPEG_QUALITY, quality]
    if fmt == "webp":
        return [cv2.IMWRITE_WEBP_QUALITY, quality]
    return []


def copy_exif(src_path: Union[str, Path], dst_path: Union[str, Path]) -> None:
    """Copy EXIF metadata, resetting orientation to normal."""
    try:
        from PIL.ExifTags import Base as ExifBase

        src_img = Image.open(src_path)
        exif = src_img.getexif()
        if not exif:
            return

        try:
            from PIL.ExifTags import Base
            if_base = Base
        except ImportError:
            if_base = ExifBase

        orientation_tag = getattr(if_base, "Orientation", None)
        if orientation_tag is None:
            orientation_tag = 0x0112
        exif[orientation_tag] = 1
        dst_img = Image.open(dst_path)
        dst_img.save(dst_path, exif=exif.tobytes())
    except Exception as exc:
        logger.warning("Failed to copy EXIF from %s to %s: %s", src_path, dst_path, exc)


def make_comparison(
    original: Optional[np.ndarray],
    retouched: Optional[np.ndarray],
    compare_path: Union[str, Path],
    fmt: str,
    quality: int,
) -> None:
    """Stitch a side-by-side comparison of original and retouched images."""
    try:
        if original is None or retouched is None:
            return

        if original.shape[:2] != retouched.shape[:2]:
            retouched = cv2.resize(retouched, (original.shape[1], original.shape[0]))

        h = original.shape[0]
        separator = np.full((h, 4, 3), 200, dtype=np.uint8)
        combined = np.hstack([original, separator, retouched])
        cv2.imwrite(str(compare_path), combined, encode_write_params(fmt, quality))
    except Exception as exc:
        logger.warning("Failed to write comparison image %s: %s", compare_path, exc)
