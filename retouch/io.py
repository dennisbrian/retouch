"""Shared image I/O helpers for CLI, GUI, and batch tooling."""

from __future__ import annotations

import io
import logging
import os
import shutil
import subprocess
import tempfile
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
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".exr",
    ".raf", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf", ".rw2", ".pef", ".srw", ".x3f",
}

RAW_EXTENSIONS = {
    ".raf", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf", ".rw2", ".pef", ".srw", ".x3f",
}

# ``raf2jpeg`` is maintained as a sibling checkout in this workspace.  Keep
# discovery here (rather than baking the absolute workstation path into the
# CLI) so API callers and worker processes resolve it consistently.
_RAF2JPEG_SIBLING = Path(__file__).resolve().parents[2] / "raf2jpeg" / "bin" / "raf2jpeg"

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
    "PNG-16": ".png",
    "WebP": ".webp",
    "EXR": ".exr",
}


def read_c2pa_manifest(path: Union[str, Path]) -> Optional[bytes]:
    """Read raw C2PA Content Credentials manifest bytes from a JPEG or PNG file.

    Returns the raw JUMBF manifest block if present, or None if absent.
    """
    path = Path(path)
    if not path.exists():
        return None

    try:
        with open(path, "rb") as f:
            data = f.read()

        # JPEG marker check for APP11 (0xFFEB) C2PA/JUMBF block
        if data.startswith(b"\xff\xd8"):
            idx = 2
            while idx < len(data) - 4:
                if data[idx] != 0xFF:
                    break
                marker = data[idx + 1]
                if marker in (0xD9, 0xDA):  # EOI or SOS
                    break
                length = (data[idx + 2] << 8) + data[idx + 3]
                if marker == 0xEB:  # APP11
                    segment = data[idx + 4 : idx + 2 + length]
                    if b"c2pa" in segment or b"jp2c" in segment or b"JUMBF" in segment:
                        return segment
                idx += 2 + length
    except Exception as exc:
        logger.debug("Failed to read C2PA manifest from %s: %s", path, exc)
    return None



def _resolve_safe_path(path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None) -> Path:
    """Resolve *path* and reject directory-traversal escapes.

    The engine must never write outside a caller-designated output tree via
    a crafted relative path (``../../etc/foo``). When *base_dir* is provided,
    the resolved path must stay within it. When *base_dir* is None (single-file
    CLI/GUI case), the path must not resolve to a system directory.
    """
    p = Path(path).expanduser().resolve()
    if base_dir is not None:
        base = Path(base_dir).expanduser().resolve()
        try:
            p.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"Path traversal rejected: {path!r} escapes base {base!s}") from exc
    else:
        # The OS temp dir is a legitimate write target (GUI/CLI export scratch).
        # On Linux TMPDIR often resolves under /var/tmp, which would otherwise be
        # rejected by the /var system-path guard below, so allow it explicitly.
        tmp_root = Path(tempfile.gettempdir()).resolve()
        try:
            p.relative_to(tmp_root)
            return p
        except ValueError:
            pass
        for forbidden in ("/etc", "/usr", "/bin", "/sbin", "/System", "/private/etc", "/var", "/dev"):
            try:
                p.relative_to(forbidden)
            except ValueError:
                continue
            raise ValueError(f"Path traversal rejected: {path!r} resolves to system path {p!s}")
    return p


def imwrite_16bit_png(path: Union[str, Path], img: np.ndarray) -> bool:
    """Write image as 16-bit PNG (uint16).

    .. deprecated:: use :func:`write_image_16bit` for new callers; this
       helper assumes float32 input is in [0, 1] (legacy convention) and
       is retained for backwards compatibility.

    Args:
        path: Output file path.
        img: (H, W, 3) uint8 or float32 [0,1] BGR image.

    Returns:
        True on success.
    """
    if img.dtype == np.float32:
        img_u16 = np.clip(img * 65535.0, 0, 65535).astype(np.uint16)
    elif img.dtype == np.uint8:
        img_u16 = img.astype(np.uint16) * 257
    else:
        img_u16 = img.astype(np.uint16)
    rgb = cv2.cvtColor(img_u16, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb, mode="RGB;16" if hasattr(Image, "fromarray") else "RGB")
    pil_img.save(str(path), format="PNG")
    return True


def write_image_16bit(
    path: Union[str, Path],
    img: np.ndarray,
    format: str = "png",
) -> bool:
    """Write *img* as a 16-bit PNG or TIFF.

    The engine works internally in float32 with values in [0, 255]. This
    function scales that range losslessly to uint16 [0, 65535] by multiplying
    by 257.0 (65535 / 255), so a float value of 255.0 maps exactly to 65535
    and 0.0 maps to 0. Rounding is used (not truncation) to match the
    round-trip inverse used by :func:`read_image_16bit`.

    Args:
        path: Destination filesystem path. Extension must be ``.png``,
            ``.tif`` or ``.tiff``; anything else raises ``ValueError``.
        img: ``(H, W, 3)`` BGR or ``(H, W)`` grayscale image. Accepted
            dtypes: float32/float64 in [0, 255], uint8, or uint16. uint8
            input is upscaled to uint16 (``* 257``) but, because no real
            precision is created, a warning is logged.
        format: ``"png"`` or ``"tiff"``. Selects the OpenCV imwrite
            compression flag (``IMWRITE_PNG_COMPRESSION`` /
            ``IMWRITE_TIFF_COMPRESSION``). Default ``"png"``.

    Returns:
        True on success.

    Raises:
        ValueError: If *format* is unsupported or the path extension is
            incompatible with 16-bit output.
        cv2.error: If the underlying ``cv2.imwrite`` fails.
    """
    fmt = format.lower()
    if fmt not in ("png", "tiff", "tif"):
        raise ValueError(
            f"write_image_16bit: format must be 'png' or 'tiff', got {format!r}"
        )
    safe = _resolve_safe_path(path)
    ext = safe.suffix.lower()
    if ext not in (".png", ".tif", ".tiff"):
        raise ValueError(
            f"write_image_16bit: path extension {ext!r} cannot hold 16-bit data; "
            "use .png or .tif/.tiff"
        )

    if img.dtype == np.float32 or img.dtype == np.float64:
        img_u16 = np.clip(np.round(img * 257.0), 0, 65535).astype(np.uint16)
    elif img.dtype == np.uint8:
        logger.warning(
            "write_image_16bit: uint8 input upscaled to uint16 (x257); "
            "this does not add real precision — supply float32 for true 16-bit export"
        )
        img_u16 = (img.astype(np.uint16) * np.uint16(257)).astype(np.uint16)
    elif img.dtype == np.uint16:
        img_u16 = img
    else:
        raise TypeError(
            f"write_image_16bit: unsupported dtype {img.dtype!r}; "
            "expected float32/float64, uint8 or uint16"
        )

    if img_u16.ndim == 3 and img_u16.shape[2] == 4:
        bgr = img_u16[:, :, :3]
    else:
        bgr = img_u16

    if fmt == "png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 3]
    else:
        params = [cv2.IMWRITE_TIFF_COMPRESSION, 5]

    ok = cv2.imwrite(str(safe), bgr, params)
    if not ok:
        raise cv2.error(f"cv2.imwrite returned False for 16-bit write to {safe}")
    return True


def read_image_16bit(path: Union[str, Path]) -> np.ndarray:
    """Read a 16-bit PNG/TIFF/RAW image and return float32 BGR in [0, 255].

    The inverse of :func:`write_image_16bit`: uint16 [0, 65535] is divided
    by 257.0 to land in float32 [0, 255], preserving the engine's internal
    working range. Files that are actually 8-bit are read as uint8 and
    promoted to float32 [0, 255] (multiplied by 1.0, i.e. identity), so
    callers always receive the same dtype/range regardless of source depth.

    Args:
        path: Filesystem path to a 16-bit (or 8-bit) PNG/TIFF, or a RAW
            camera file (``.raf``/``.cr2``/``.nef``/``.dng``/…). RAW files
            are decoded via rawpy at 16-bit when the driver supports it.

    Returns:
        ``(H, W, 3)`` float32 BGR image with values in [0, 255].

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If *path* cannot be decoded as an image.
        cv2.error: On OpenCV-level decode failures.
    """
    safe = _resolve_safe_path(path)
    if not safe.exists():
        raise FileNotFoundError(f"read_image_16bit: {safe} not found")

    ext = safe.suffix.lower()
    if ext in RAW_EXTENSIONS:
        import rawpy

        with rawpy.imread(str(safe)) as raw:
            rgb16 = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                bright=1.0,
                output_bps=16,
                output_color=rawpy.ColorSpace.sRGB,
                highlight_mode=rawpy.HighlightMode.ReconstructDefault,
            )
        bgr16 = cv2.cvtColor(rgb16, cv2.COLOR_RGB2BGR)
        return (bgr16.astype(np.float32) / 257.0).astype(np.float32, copy=False)

    flag = cv2.IMREAD_UNCHANGED if ext in (".png", ".tif", ".tiff") else cv2.IMREAD_COLOR
    img = cv2.imread(str(safe), flag)
    if img is None:
        raise ValueError(f"read_image_16bit: failed to decode {safe}")

    if img.dtype == np.uint16:
        out = (img.astype(np.float32) / 257.0).astype(np.float32, copy=False)
    elif img.dtype == np.uint8:
        out = img.astype(np.float32, copy=True)
    else:
        out = img.astype(np.float32, copy=False)

    if out.ndim == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
    elif out.ndim == 3 and out.shape[2] == 4:
        out = cv2.cvtColor(out, cv2.COLOR_BGRA2BGR)

    return out


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
            # Faithful, neutral RAW development that matches the camera's own
            # rendition: in-camera white balance, no auto-exposure, and a
            # standard 1.0 brightness (rawpy's sRGB output is the closest
            # neutral analogue to the camera JPEG). The previous bright=1.5
            # over-lit the image ~50% vs what was shot.
            rgb = raw.postprocess(
                use_camera_wb=True,
                no_auto_bright=True,
                bright=1.0,
                output_color=rawpy.ColorSpace.sRGB,
                highlight_mode=rawpy.HighlightMode.ReconstructDefault,
            )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    from PIL import ImageOps

    pil_img = Image.open(path)
    pil_img = ImageOps.exif_transpose(pil_img) or pil_img
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def resolve_raf2jpeg(path: Optional[Union[str, Path]] = None) -> Path:
    """Locate the ``raf2jpeg`` executable used for opt-in RAF development.

    Resolution order is an explicit *path*, ``RETOUCH_RAF2JPEG_PATH``, the
    sibling ``../raf2jpeg/bin/raf2jpeg`` checkout, then ``PATH``.  An explicit
    path may be either a filesystem path or an executable name on ``PATH``.
    """
    configured = path or os.environ.get("RETOUCH_RAF2JPEG_PATH")
    if configured:
        candidate = Path(configured).expanduser()
        if candidate.is_file() and os.access(str(candidate), os.X_OK):
            return candidate.resolve()
        located = shutil.which(str(configured))
        if located:
            return Path(located).resolve()
        raise FileNotFoundError(
            f"raf2jpeg executable not found or not executable: {configured}"
        )

    if _RAF2JPEG_SIBLING.is_file() and os.access(str(_RAF2JPEG_SIBLING), os.X_OK):
        return _RAF2JPEG_SIBLING.resolve()
    located = shutil.which("raf2jpeg")
    if located:
        return Path(located).resolve()
    raise FileNotFoundError(
        "raf2jpeg was not found. Install it on PATH, set RETOUCH_RAF2JPEG_PATH, "
        "or pass --raf2jpeg-path."
    )


def read_raf_with_raf2jpeg(
    path: Union[str, Path],
    raf2jpeg_path: Optional[Union[str, Path]] = None,
    quality: int = 100,
) -> np.ndarray:
    """Develop one RAF with ``raf2jpeg`` and return its uint8 BGR JPEG.

    The conversion is written to a temporary directory and is never left next
    to the user's source RAF.  ``raf2jpeg`` is deliberately opt-in: unlike
    the native RAW path it produces an 8-bit JPEG, so callers choosing it are
    prioritising its camera rendition over the native 16-bit edit headroom.
    """
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() != ".raf":
        raise ValueError(f"raf2jpeg only supports .RAF inputs, got {source.name!r}")
    if not source.is_file():
        raise FileNotFoundError(f"RAF input not found: {source}")
    if not 1 <= int(quality) <= 100:
        raise ValueError(f"raf2jpeg quality must be between 1 and 100, got {quality!r}")

    executable = resolve_raf2jpeg(raf2jpeg_path)
    with tempfile.TemporaryDirectory(prefix="retouch-raf2jpeg-") as output_dir:
        command = [
            str(executable), str(source), "--output", output_dir,
            "--quality", str(int(quality)), "--force", "--flat",
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "no diagnostic output").strip()
            raise RuntimeError(
                f"raf2jpeg failed while converting {source.name}: {detail}"
            ) from exc

        converted = Path(output_dir) / f"{source.stem}.jpg"
        if not converted.is_file():
            raise RuntimeError(
                f"raf2jpeg completed but did not create expected JPEG: {converted.name}"
            )
        try:
            image = imread_exif(converted)
        except Exception as exc:
            raise RuntimeError(
                "raf2jpeg reported success but produced an unreadable JPEG; "
                "check the converter's selected backend"
            ) from exc
        if image is None:
            raise RuntimeError(
                "raf2jpeg reported success but produced an unreadable JPEG"
            )
        return image


def read_raf_with_fuji_match(
    path: Union[str, Path],
    raf2jpeg_path: Optional[Union[str, Path]] = None,
    strength: float = 0.85,
) -> np.ndarray:
    """Load full-resolution RAF data and calibrate it to its Fuji preview.

    This keeps the native 16-bit RAW decode while learning global tone and
    colour characteristics from the RAF's embedded camera JPEG.  It is a
    calibrated approximation of Fuji's proprietary camera processing, not a
    byte-identical replacement for it.
    """
    source = read_image_16bit(path)
    preview = read_raf_with_raf2jpeg(path, raf2jpeg_path, quality=100)
    from .fuji_match import calibrate_to_fuji_preview

    return calibrate_to_fuji_preview(source, preview, strength, inplace=True)


def imread_engine(
    path: Union[str, Path],
    prefer_16bit: bool = True,
    raw_decoder: str = "rawpy",
    raf2jpeg_path: Optional[Union[str, Path]] = None,
    raf2jpeg_quality: int = 100,
    fuji_match_strength: float = 0.85,
    optical_correction: bool = False,
) -> np.ndarray:
    """Load an image for the engine, using the 16-bit path for RAW.

    This is the wired ingest entry: RAW camera files (``.raf``/``.cr2``/…)
    are decoded through the 16-bit pipeline (:func:`read_image_16bit`),
    returning **float32 [0, 255] sRGB BGR** so the extra tonal headroom
    reaches ``RetouchEngine.process()`` (which now accepts float32 and
    threads that precision through global grading). Non-RAW formats fall
    back to :func:`imread_exif` (uint8 BGR with EXIF orientation applied),
    so JPEG/PNG/TIFF loading is byte-identical to before.

    Note: the 16-bit RAW decode stays in the engine's native sRGB
    (display-referred) domain — it is *not* gamma-linearised — because the
    pipeline's tonal/skin-LAB math is sRGB-native. Feeding linear RGB here
    would look dramatically under-toned. The true-linear develop
    (``raw_develop.py``) is a separate, LinearGrader-only path.

    Args:
        path: Filesystem path to any supported image or RAW file.
        prefer_16bit: When True (default), route RAW files through the
            16-bit float32 decode. Set False to force the legacy uint8
            RAW path (:func:`imread_exif`).
        raw_decoder: ``"rawpy"`` (default) for Retouch's 16-bit RAW path,
            ``"raf2jpeg"`` for the embedded camera JPEG, or
            ``"rawpy-fuji-match"`` for full-resolution RAW calibrated to
            that JPEG. Other RAW formats still use rawpy.
        raf2jpeg_path: Optional executable path for ``raw_decoder="raf2jpeg"``.
        raf2jpeg_quality: JPEG quality (1--100) passed to ``raf2jpeg`` for
            any re-encoding fallback. Its embedded-camera-JPEG path keeps the
            original bytes unchanged.
        fuji_match_strength: Calibration blend for ``rawpy-fuji-match`` from
            native RAW (0) to the learned camera-preview look (1).

    Returns:
        ``(H, W, 3)`` BGR image: float32 [0, 255] for native and calibrated
        RAW paths; uint8 for ``raf2jpeg`` and non-RAW formats.
    """
    p = Path(path)
    if raw_decoder not in ("rawpy", "raf2jpeg", "rawpy-fuji-match"):
        raise ValueError(
            "raw_decoder must be 'rawpy', 'raf2jpeg', or 'rawpy-fuji-match', "
            f"got {raw_decoder!r}"
        )
    if raw_decoder == "raf2jpeg" and p.suffix.lower() == ".raf":
        image = read_raf_with_raf2jpeg(p, raf2jpeg_path, raf2jpeg_quality)
    elif raw_decoder == "rawpy-fuji-match" and p.suffix.lower() == ".raf":
        image = read_raf_with_fuji_match(p, raf2jpeg_path, fuji_match_strength)
    elif prefer_16bit and p.suffix.lower() in RAW_EXTENSIONS:
        image = read_image_16bit(p)
    else:
        image = imread_exif(p)
    if optical_correction:
        from .capture_fidelity import apply_optical_corrections, read_capture_metadata
        image, status = apply_optical_corrections(image, read_capture_metadata(p))
        logger.info("Optical correction status for %s: %s", p, status)
    return image


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
        params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        # Delivery JPEGs preserve chroma resolution. This prevents the export
        # boundary from reintroducing 4:2:0 bleed at lip and costume edges.
        if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR") and hasattr(
            cv2, "IMWRITE_JPEG_SAMPLING_FACTOR_444"
        ):
            params.extend([
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR,
                cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
            ])
        return params
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


def read_exif_bytes(path: Union[str, Path]) -> Optional[bytes]:
    """Return the raw EXIF blob from *path*, or ``None`` if absent/unreadable."""
    try:
        with Image.open(str(path)) as pil_img:
            exif = pil_img.getexif()
    except (FileNotFoundError, OSError, Image.UnidentifiedImageError, Image.DecompressionBombError):
        return None
    if not exif:
        return None
    return exif.tobytes()


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


def _resolve_float_range(img: np.ndarray, float_range: str) -> str:
    """Resolve an explicit float export range without changing uint8 callers."""
    if float_range not in ("auto", "unit", "byte"):
        raise ValueError("float_range must be 'auto', 'unit', or 'byte'")
    if float_range != "auto" or img.dtype not in (np.float32, np.float64):
        return float_range
    finite = img[np.isfinite(img)]
    return "unit" if finite.size == 0 or float(finite.max()) <= 1.0 else "byte"


def _to_uint8_delivery(img: np.ndarray, float_range: str) -> np.ndarray:
    """Convert a final image to uint8, applying dither only at delivery."""
    if img.dtype == np.uint8:
        return img
    if img.dtype in (np.float32, np.float64):
        from .precision import to_uint8_dithered

        unit = img if float_range == "unit" else img * (1.0 / 255.0)
        return to_uint8_dithered(unit.astype(np.float32, copy=False))
    if img.dtype == np.uint16:
        return np.clip(np.round(img.astype(np.float32) / 257.0), 0, 255).astype(np.uint8)
    return np.clip(img, 0, 255).astype(np.uint8)


def _icc_to_profile(icc_profile: bytes) -> "ImageCms.core.CmsProfile":  # type: ignore[name-defined]
    return ImageCms.getOpenProfile(io.BytesIO(icc_profile))


def write_image_with_icc(
    path: Union[str, Path],
    img: np.ndarray,
    icc_profile: Optional[bytes] = None,
    bit_depth: int = 8,
    exif: Optional[bytes] = None,
    **kwargs: Any,
) -> None:
    """Write *img* (BGR ndarray) to *path* with an optional embedded ICC profile.

    Args:
        path: Destination filesystem path. Format is inferred from extension
            (``.jpg``/``.jpeg``/``.png``/``.tif``/``.tiff``/``.webp``).
        img: Source image. Accepts ``HxW`` grayscale, ``HxWx3`` BGR, or
            ``HxWx4`` BGRA uint8. Float input must declare ``float_range``
            when the range is ambiguous; the engine's future float contract is
            ``[0, 255]`` (``float_range="byte"``).
        icc_profile: Raw ICC profile bytes to embed. If ``None`` or empty,
            the image is saved without an embedded profile.
        exif: Raw EXIF bytes to embed. Orientation is force-reset to normal
            (1) before embedding so the engine output is not re-oriented by
            viewers. Passing EXIF here (rather than re-saving the destination
            afterward) preserves ICC profile, bit depth, and quality settings.
        bit_depth: Output bit depth. ``8`` (default) for uint8, ``16`` for
            uint16 (PNG/TIFF only). The 16-bit OpenCV path cannot embed an
            ICC profile; JPEG/WebP always output 8-bit.
        **kwargs: ``quality`` (int, default ``95``), ``format`` (str), and
            ``float_range`` (``"auto"``, ``"unit"``, or ``"byte"``) are
            consumed; all remaining kwargs are forwarded to ``PIL.Image.save``.

    Falls back to ``cv2.imwrite`` (which never embeds ICC) when PIL or the
    destination format is unavailable.
    """
    path = Path(path)
    ext = path.suffix.lower()
    pil_format = kwargs.pop("format", _ICC_WRITE_FORMAT_MAP.get(ext))
    quality = int(kwargs.pop("quality", 95))
    float_range = _resolve_float_range(img, str(kwargs.pop("float_range", "auto")))

    # 16-bit export: only PNG and TIFF support it
    if bit_depth == 16:
        if ext not in (".png", ".tif", ".tiff"):
            logger.warning(
                "16-bit export requested for %s format; falling back to 8-bit. "
                "Use PNG or TIFF for 16-bit output.", ext
            )
            bit_depth = 8
        else:
            # Convert to uint16 for 16-bit export
            if img.dtype == np.float32 or img.dtype == np.float64:
                scale = 65535.0 if float_range == "unit" else 257.0
                img_16 = np.clip(img * scale + 0.5, 0, 65535).astype(np.uint16)
            elif img.dtype == np.uint8:
                # Upscale uint8 to uint16: multiply by 257 to fill the range
                img_16 = (img.astype(np.uint16) * 257).astype(np.uint16)
            else:
                img_16 = img.astype(np.uint16)
            
            if img_16.ndim not in (2, 3) or (
                img_16.ndim == 3 and img_16.shape[2] not in (3, 4)
            ):
                raise ValueError(f"16-bit export: unsupported image shape {img_16.shape}")

            # Pillow cannot reliably construct 16-bit RGB/RGBA images across
            # supported versions. OpenCV preserves the actual channel depth;
            # unlike the old path, it cannot attach an ICC profile, so report
            # that limitation instead of writing an invalid or 8-bit file.
            if icc_profile:
                logger.warning("16-bit export does not embed ICC profiles; OpenCV writer is used")
            if not cv2.imwrite(str(path), img_16):
                raise cv2.error(f"cv2.imwrite returned False for 16-bit write to {path}")
            return

    # 8-bit delivery is the only place where blue-noise dither is allowed.
    # Intermediate pipeline conversions must stay deterministic and noiseless.
    img_u8 = _to_uint8_delivery(img, float_range)
    if pil_format is None:
        if icc_profile or exif:
            logger.warning(
                "%s has no PIL writer mapping; falling back to cv2.imwrite, "
                "which embeds neither ICC profile nor EXIF", path
            )
        cv2.imwrite(str(path), img_u8, encode_write_params(ext.lstrip("."), quality))
        return

    pil_img = _bgr_to_pil(img_u8)
    save_kwargs_8: Dict[str, Any] = {"format": pil_format}
    if pil_format in ("JPEG", "WEBP"):
        save_kwargs_8["quality"] = quality
    if pil_format == "JPEG":
        save_kwargs_8["subsampling"] = 0  # 4:4:4
    if icc_profile:
        save_kwargs_8["icc_profile"] = icc_profile
    if exif:
        try:
            from PIL.ExifTags import Base as _ExifBase
            _exif = Image.Exif()
            _exif.load(exif)
            orientation_tag = getattr(_ExifBase, "Orientation", 0x0112)
            _exif[orientation_tag] = 1
            save_kwargs_8["exif"] = _exif.tobytes()
        except Exception as exc:
            logger.warning("Failed to embed EXIF into %s: %s", path, exc)
    save_kwargs_8.update(kwargs)
    pil_img.save(str(path), **save_kwargs_8)


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
