"""Shared image I/O helpers for CLI, GUI, and batch tooling."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    from PIL import ImageCms

    _HAS_IMAGECMS = True
except ImportError:
    ImageCms = None  # type: ignore[assignment,misc]
    _HAS_IMAGECMS = False

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


# ---------------------------------------------------------------------------
# ICC profile support
# ---------------------------------------------------------------------------

_ICC_WRITE_FORMAT_MAP: Dict[str, str] = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".jpe": "JPEG",
    ".png": "PNG",
    ".tif": "TIFF",
    ".tiff": "TIFF",
    ".webp": "WEBP",
}


def read_icc_profile(path: Union[str, Path]) -> Optional[bytes]:
    """Return the raw embedded ICC profile bytes from *path*, or ``None``.

    Args:
        path: Filesystem path to a JPEG/PNG/TIFF/WebP image.

    Returns:
        Raw ICC profile bytes when present, otherwise ``None``. Non-image
        files, missing files, and corrupt images all yield ``None``.
    """
    try:
        with Image.open(str(path)) as pil_img:
            icc = pil_img.info.get("icc_profile")
    except (FileNotFoundError, OSError, Image.UnidentifiedImageError, Image.DecompressionBombError):
        return None
    if not icc:
        return None
    return bytes(icc)


def image_has_icc(path: Union[str, Path]) -> bool:
    """Return ``True`` if *path* is an image that embeds an ICC profile."""
    return read_icc_profile(path) is not None


def _bgr_to_pil(img: np.ndarray) -> Image.Image:
    if img.ndim == 2:
        return Image.fromarray(img)
    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxW, HxWx3 or HxWx4 image, got shape {img.shape}")
    if img.dtype != np.uint8:
        if img.dtype == np.float32 or img.dtype == np.float64:
            clipped = np.clip(img, 0.0, 1.0)
            img = (clipped * 255.0 + 0.5).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    if img.shape[2] == 3:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb, mode="RGB")
    rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    return Image.fromarray(rgba, mode="RGBA")


def _icc_to_profile(icc_profile: bytes) -> "ImageCms.core.CmsProfile":  # type: ignore[name-defined]
    return ImageCms.getOpenProfile(io.BytesIO(icc_profile))


def write_image_with_icc(
    path: Union[str, Path],
    img: np.ndarray,
    icc_profile: Optional[bytes] = None,
    **kwargs: Any,
) -> None:
    """Write *img* (BGR ndarray) to *path* with an optional embedded ICC profile.

    Args:
        path: Destination filesystem path. Format is inferred from extension
            (``.jpg``/``.jpeg``/``.png``/``.tif``/``.tiff``/``.webp``).
        img: Source image. Accepts ``HxW`` grayscale, ``HxWx3`` BGR, or
            ``HxWx4`` BGRA uint8 (or float in [0, 1]). Channel order is
            converted to RGB/RGBA for PIL.
        icc_profile: Raw ICC profile bytes to embed. If ``None`` or empty,
            the image is saved without an embedded profile.
        **kwargs: ``quality`` (int, default ``95``) and ``format`` (str) are
            consumed; all remaining kwargs are forwarded to ``PIL.Image.save``.

    Falls back to ``cv2.imwrite`` (which never embeds ICC) when PIL or the
    destination format is unavailable.
    """
    path = Path(path)
    ext = path.suffix.lower()
    pil_format = kwargs.pop("format", _ICC_WRITE_FORMAT_MAP.get(ext))
    quality = int(kwargs.pop("quality", 95))

    if pil_format is None:
        cv2.imwrite(str(path), img, encode_write_params(ext.lstrip("."), quality))
        return

    pil_img = _bgr_to_pil(img)
    save_kwargs: Dict[str, Any] = {"format": pil_format}
    if pil_format in ("JPEG", "WEBP"):
        save_kwargs["quality"] = quality
    if icc_profile:
        save_kwargs["icc_profile"] = icc_profile
    save_kwargs.update(kwargs)
    pil_img.save(str(path), **save_kwargs)


def convert_image_colorspace(
    img: np.ndarray,
    src_icc: bytes,
    dst_icc: bytes,
) -> np.ndarray:
    """Convert *img* (BGR) from *src_icc* to *dst_icc* using LittleCMS2.

    Args:
        img: ``HxWx3`` BGR image (uint8 or float in [0, 1]).
        src_icc: Raw ICC profile bytes describing the source pixels.
        dst_icc: Raw ICC profile bytes describing the destination space.

    Returns:
        BGR ``float32`` image in ``[0, 1]``. Out-of-gamut pixels are clipped
        to the destination gamut by LittleCMS2.

    Raises:
        RuntimeError: If ``PIL.ImageCms`` is not available in this build.
    """
    if not _HAS_IMAGECMS:
        raise RuntimeError(
            "PIL.ImageCms is not available; install Pillow with lcms2 support "
            "to enable ICC colorspace conversion."
        )

    was_float = img.dtype == np.float32 or img.dtype == np.float64
    pil_img = _bgr_to_pil(img)
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")

    src_profile = _icc_to_profile(src_icc)
    dst_profile = _icc_to_profile(dst_icc)
    transformed = ImageCms.profileToProfile(pil_img, src_profile, dst_profile)
    out_rgb = np.asarray(transformed)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    out = out_bgr.astype(np.float32) / 255.0
    if was_float:
        return out
    return out
