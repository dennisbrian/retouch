"""Capture-fidelity metadata and sensor calibration primitives.

This module deliberately separates evidence collection from correction. EXIF
identification is always available through Pillow. Lensfun correction is an
optional integration and must report unavailable rather than silently applying
the creative RGB displacement or vignette effects used elsewhere in Retouch.
Sensor calibration operates on a linear RAW mosaic before demosaicing when the
RAW caller supplies that buffer.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ExifTags

logger = logging.getLogger(__name__)


def _tag_name(tag: int) -> str:
    return str(ExifTags.TAGS.get(tag, tag))


def _number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return float(value[0]) / float(value[1])
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        try:
            return float(value.numerator) / float(value.denominator)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return None


@dataclass(frozen=True)
class CaptureMetadata:
    """Camera/lens facts read from the source file, with no guessed values."""

    make: Optional[str] = None
    model: Optional[str] = None
    lens_model: Optional[str] = None
    focal_length_mm: Optional[float] = None
    aperture: Optional[float] = None
    iso: Optional[float] = None
    shutter_seconds: Optional[float] = None
    orientation: Optional[int] = None
    source_path: Optional[str] = None

    @property
    def camera_model(self) -> Optional[str]:
        if self.make and self.model and self.make.lower() not in self.model.lower():
            return f"{self.make} {self.model}".strip()
        return self.model or self.make

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self) | {"camera_model": self.camera_model}


def read_capture_metadata(path: Union[str, Path]) -> CaptureMetadata:
    """Read camera/lens EXIF facts without treating missing data as a match."""
    source = Path(path)
    try:
        with Image.open(source) as image:
            exif = image.getexif()
            named = {_tag_name(tag): value for tag, value in exif.items()}
    except (FileNotFoundError, OSError, Image.UnidentifiedImageError):
        return CaptureMetadata(source_path=str(source))

    exposure = named.get("ExposureTime")
    shutter = _number(exposure)
    if shutter is None and exposure:
        try:
            shutter = 1.0 / float(str(exposure).split("/")[-1])
        except (TypeError, ValueError, ZeroDivisionError):
            shutter = None

    return CaptureMetadata(
        make=str(named["Make"]).strip() if named.get("Make") else None,
        model=str(named["Model"]).strip() if named.get("Model") else None,
        lens_model=str(named["LensModel"]).strip() if named.get("LensModel") else None,
        focal_length_mm=_number(named.get("FocalLength")),
        aperture=_number(named.get("FNumber")),
        iso=_number(named.get("ISOSpeedRatings", named.get("PhotographicSensitivity"))),
        shutter_seconds=shutter,
        orientation=int(named["Orientation"]) if named.get("Orientation") is not None else None,
        source_path=str(source),
    )


@dataclass(frozen=True)
class SensorCalibrationProfile:
    """Camera/setting-keyed calibration assets for RAW-domain correction."""

    camera_model: str
    iso: Optional[float] = None
    shutter_seconds: Optional[float] = None
    lens_model: Optional[str] = None
    aperture: Optional[float] = None
    dark_frame: Optional[str] = None
    flat_field: Optional[str] = None
    hot_pixels: Tuple[Tuple[int, int], ...] = field(default_factory=tuple)
    license: Optional[str] = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SensorCalibrationProfile":
        pixels = tuple((int(p[0]), int(p[1])) for p in payload.get("hot_pixels", ()))
        return cls(
            camera_model=str(payload.get("camera_model", "")),
            iso=_number(payload.get("iso")),
            shutter_seconds=_number(payload.get("shutter_seconds")),
            lens_model=payload.get("lens_model"),
            aperture=_number(payload.get("aperture")),
            dark_frame=payload.get("dark_frame"),
            flat_field=payload.get("flat_field"),
            hot_pixels=pixels,
            license=payload.get("license"),
        )


def load_sensor_profiles(path: Union[str, Path]) -> Tuple[SensorCalibrationProfile, ...]:
    """Load a JSON list of calibration profiles; reject malformed entries."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("profiles", [])
    if not isinstance(payload, list):
        raise ValueError("sensor calibration file must contain a profile list")
    return tuple(SensorCalibrationProfile.from_mapping(row) for row in payload)


def select_sensor_profile(
    metadata: CaptureMetadata,
    profiles: Iterable[SensorCalibrationProfile],
) -> Optional[SensorCalibrationProfile]:
    """Select only a camera-matching profile; settings are nearest-neighbor."""
    candidates = [
        profile for profile in profiles
        if profile.camera_model and metadata.camera_model
        and profile.camera_model.lower() in metadata.camera_model.lower()
    ]
    if not candidates:
        return None

    def distance(profile: SensorCalibrationProfile) -> float:
        score = 0.0
        if profile.iso is not None and metadata.iso is not None:
            score += abs(np.log2(max(profile.iso, 1.0) / max(metadata.iso, 1.0)))
        if profile.shutter_seconds is not None and metadata.shutter_seconds is not None:
            score += abs(np.log2(max(profile.shutter_seconds, 1e-6) / max(metadata.shutter_seconds, 1e-6)))
        if profile.aperture is not None and metadata.aperture is not None:
            score += abs(profile.aperture - metadata.aperture)
        if profile.lens_model and metadata.lens_model and profile.lens_model.lower() not in metadata.lens_model.lower():
            score += 10.0
        return score

    return min(candidates, key=distance)


def apply_sensor_calibration(
    raw_mosaic: np.ndarray,
    *,
    dark_frame: Optional[np.ndarray] = None,
    flat_field: Optional[np.ndarray] = None,
    hot_pixels: Sequence[Tuple[int, int]] = (),
) -> np.ndarray:
    """Correct a RAW mosaic before demosaicing, preserving float precision.

    Dark subtraction and flat-field correction are normalized conservatively;
    no operation occurs when its asset is absent. Hot pixels are replaced by
    the median of a 3x3 neighborhood. The function never mutates the input.
    """
    raw = np.asarray(raw_mosaic)
    if raw.ndim != 2:
        raise ValueError("raw_mosaic must be a two-dimensional Bayer buffer")
    work = raw.astype(np.float32, copy=True)
    if dark_frame is not None:
        dark = np.asarray(dark_frame, dtype=np.float32)
        if dark.shape != work.shape:
            raise ValueError("dark_frame must match raw_mosaic shape")
        work -= dark
    if flat_field is not None:
        flat = np.asarray(flat_field, dtype=np.float32)
        if flat.shape != work.shape:
            raise ValueError("flat_field must match raw_mosaic shape")
        valid = flat > 1e-6
        reference = float(np.median(flat[valid])) if np.any(valid) else 1.0
        work = np.where(valid, work * (reference / np.maximum(flat, 1e-6)), work)
    if hot_pixels:
        padded = np.pad(work, 1, mode="reflect")
        for y, x in hot_pixels:
            if 0 <= y < work.shape[0] and 0 <= x < work.shape[1]:
                work[y, x] = float(np.median(padded[y:y + 3, x:x + 3]))
    return np.clip(work, 0.0, None).astype(np.float32)


def lens_correction_status() -> Dict[str, Any]:
    """Return truthful availability of the optional Lensfun integration."""
    try:
        import lensfunpy  # type: ignore
    except ImportError:
        return {
            "available": False,
            "backend": None,
            "reason": "optional lensfunpy dependency or Lensfun database is unavailable",
        }
    return {"available": True, "backend": "lensfunpy", "reason": None}


def apply_optical_corrections(
    img_bgr: np.ndarray,
    metadata: CaptureMetadata,
    *,
    corrections: Sequence[str] = ("distortion", "tca", "vignetting"),
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Apply explicit Lensfun corrections, or return a truthful no-op status.

    ``lensfunpy``'s image modifiers operate on 8-bit display-referred arrays.
    Retouch must therefore never silently pass a 16-bit/float capture through
    an 8-bit buffer and cast it back. Such inputs remain unchanged until a
    precision-preserving Lensfun backend is available; the returned status is
    deliberately actionable for CLI and GUI callers.
    """
    requested = tuple(str(item) for item in corrections)
    valid = {"distortion", "tca", "vignetting"}
    if any(item not in valid for item in requested):
        raise ValueError(f"corrections must be drawn from {sorted(valid)}")
    status = lens_correction_status()
    if not status["available"]:
        return img_bgr.copy(), {**status, "applied": (), "requested": requested}
    if not metadata.make or not metadata.model or not metadata.lens_model:
        return img_bgr.copy(), {
            "available": True, "backend": "lensfunpy", "applied": (),
            "requested": requested, "reason": "camera/lens EXIF is incomplete",
        }
    if metadata.focal_length_mm is None or metadata.aperture is None:
        return img_bgr.copy(), {
            "available": True, "backend": "lensfunpy", "applied": (),
            "requested": requested, "reason": "focal length or aperture EXIF is missing",
        }
    source = np.asarray(img_bgr)
    if source.dtype != np.uint8:
        return img_bgr.copy(), {
            "available": True,
            "backend": "lensfunpy",
            "applied": (),
            "requested": requested,
            "precision_preserved": True,
            "reason": (
                "Lensfun correction skipped: installed backend accepts only uint8; "
                "non-8-bit input was left unchanged to preserve precision"
            ),
        }
    try:
        import lensfunpy  # type: ignore
        database = lensfunpy.Database()
        cameras = database.find_cameras(metadata.make, metadata.model)
        if not cameras:
            raise LookupError("camera is not present in Lensfun database")
        lenses = database.find_lenses(cameras[0], metadata.lens_model)
        if not lenses:
            raise LookupError("lens is not present in Lensfun database")
        rgb = source[..., ::-1].copy()
        modifier = lensfunpy.Modifier(lenses[0], 1.0, rgb.shape[1], rgb.shape[0])
        modifier.initialize(metadata.focal_length_mm, metadata.aperture, 10.0)
        applied = []
        if "distortion" in requested and hasattr(modifier, "apply_geometry_distortion"):
            rgb = modifier.apply_geometry_distortion(rgb)
            applied.append("distortion")
        if "tca" in requested and hasattr(modifier, "apply_subpixel_distortion"):
            rgb = modifier.apply_subpixel_distortion(rgb)
            applied.append("tca")
        if "vignetting" in requested and hasattr(modifier, "apply_color_modification"):
            rgb = modifier.apply_color_modification(rgb)
            applied.append("vignetting")
        corrected = np.asarray(rgb)[..., ::-1]
        return corrected.astype(source.dtype, copy=False), {
            "available": True, "backend": "lensfunpy", "applied": tuple(applied),
            "requested": requested,
            "precision_preserved": True,
            "reason": None if applied else "Lensfun backend exposes no requested correction",
        }
    except Exception as exc:
        logger.warning("Lensfun optical correction unavailable: %s", exc)
        return img_bgr.copy(), {
            "available": True, "backend": "lensfunpy", "applied": (),
            "requested": requested,
            "reason": f"correction failed: {type(exc).__name__}: {exc}",
        }
