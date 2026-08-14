"""Pure native-resolution inspection contracts for the Retouch GUI.

This module is deliberately independent of Gradio, the image engine, and
``gui.py``.  It defines the small contract that a future GUI integration can
use to inspect the *actual* native-resolution render:

* ``fit`` keeps the whole native image and asks the view to fit it;
* ``100%`` keeps the whole native image at a one-to-one display scale;
* ``face`` selects one detector face bounding box; and
* ``roi`` selects an explicit native-pixel region.

The crop is never resized by this module.  A contract records the
half-open native bounds ``[x:x1, y:y1]`` and the render revision that produced
the source pixels.  A caller must provide both the revision it requested and
the current draft revision.  Any mismatch is rejected before a crop can be
accepted as the visible inspection.

Preview state is intentionally non-downloadable.  Inspection is an ephemeral
view of a render; a separate full-quality export contract is responsible for
creating a downloadable artifact.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


SCHEMA_VERSION = 1
CONTRACT_NAME = "retouch.gui.inspection"

MODE_FIT = "fit"
MODE_100_PERCENT = "100%"
MODE_100 = MODE_100_PERCENT
MODE_ONE_TO_ONE = MODE_100_PERCENT
MODE_FACE = "face"
MODE_ROI = "roi"
SUPPORTED_MODES = (MODE_FIT, MODE_100_PERCENT, MODE_FACE, MODE_ROI)

_MODE_ALIASES = {
    "fit": MODE_FIT,
    "fit_to_view": MODE_FIT,
    "fit-to-view": MODE_FIT,
    "100": MODE_100_PERCENT,
    "100%": MODE_100_PERCENT,
    "1:1": MODE_100_PERCENT,
    "one_to_one": MODE_100_PERCENT,
    "one-to-one": MODE_100_PERCENT,
    "native": MODE_100_PERCENT,
    "face": MODE_FACE,
    "roi": MODE_ROI,
}


class InspectionContractError(ValueError):
    """Raised when an inspection request cannot be made safely."""


class RevisionMismatchError(InspectionContractError):
    """The pixels came from a different revision than the request expects."""


class StaleInspectionError(RevisionMismatchError):
    """The render is no longer the current draft revision."""


def _require_revision(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InspectionContractError(
            "%s must be a non-negative integer" % name
        )
    return value


def _coerce_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InspectionContractError("%s must be a positive integer" % name)
    return value


def normalize_mode(mode: Any) -> str:
    """Return the canonical inspection mode."""

    if not isinstance(mode, str):
        raise InspectionContractError("inspection mode must be a string")
    key = mode.strip().lower().replace(" ", "_")
    canonical = _MODE_ALIASES.get(key)
    if canonical is None:
        raise InspectionContractError("unsupported inspection mode: %r" % mode)
    return canonical


@dataclass(frozen=True)
class NativeSize:
    """Width/height of the source render in native pixels."""

    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "width", _coerce_positive_int(self.width, "width"))
        object.__setattr__(self, "height", _coerce_positive_int(self.height, "height"))

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> Dict[str, int]:
        return {"width": self.width, "height": self.height}


def coerce_native_size(value: Any) -> NativeSize:
    """Normalize ``(width, height)`` or a width/height mapping."""

    if isinstance(value, NativeSize):
        return value
    if isinstance(value, Mapping):
        if "width" not in value or "height" not in value:
            raise InspectionContractError(
                "native_size mapping requires width and height"
            )
        return NativeSize(value["width"], value["height"])
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 2:
            raise InspectionContractError(
                "native_size sequence must contain width and height"
            )
        return NativeSize(value[0], value[1])
    raise InspectionContractError(
        "native_size must be NativeSize, a width/height mapping, or a pair"
    )


@dataclass(frozen=True)
class CropRect:
    """A non-empty half-open rectangle in native pixel coordinates.

    ``x`` and ``y`` are inclusive starts.  ``x1`` and ``y1`` are exclusive
    ends, matching NumPy slicing and Pillow's crop box convention.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        for field_name in ("x", "y"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise InspectionContractError("%s must be an integer" % field_name)
        object.__setattr__(self, "width", _coerce_positive_int(self.width, "width"))
        object.__setattr__(self, "height", _coerce_positive_int(self.height, "height"))

    @property
    def x1(self) -> int:
        return self.x + self.width

    @property
    def y1(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.width, self.height

    def to_xyxy(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.x1, self.y1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "x1": self.x1,
            "y1": self.y1,
            "coordinate_space": "native_pixels",
            "end_inclusive": False,
        }


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise InspectionContractError("%s must be numeric" % name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise InspectionContractError("%s must be numeric" % name)
    if not math.isfinite(result):
        raise InspectionContractError("%s must be finite" % name)
    return result


def _rect_from_values(x: Any, y: Any, width: Any, height: Any) -> CropRect:
    """Convert a possibly fractional UI rectangle to safe pixel bounds.

    Starts are floored and ends are ceiled so a selected ROI is not silently
    made smaller by integer conversion.  The resulting rectangle is still
    strictly native-pixel based.
    """

    x_value = _finite_number(x, "roi.x")
    y_value = _finite_number(y, "roi.y")
    width_value = _finite_number(width, "roi.width")
    height_value = _finite_number(height, "roi.height")
    if width_value <= 0 or height_value <= 0:
        raise InspectionContractError("ROI width and height must be positive")
    x0 = math.floor(x_value)
    y0 = math.floor(y_value)
    x1 = math.ceil(x_value + width_value)
    y1 = math.ceil(y_value + height_value)
    if x1 <= x0 or y1 <= y0:
        raise InspectionContractError("ROI does not cover any pixels")
    return CropRect(x0, y0, x1 - x0, y1 - y0)


def coerce_crop_rect(value: Any) -> CropRect:
    """Normalize a ``CropRect``, ``(x, y, width, height)``, or mapping."""

    if isinstance(value, CropRect):
        return value
    if isinstance(value, Mapping):
        if all(key in value for key in ("x", "y", "width", "height")):
            return _rect_from_values(
                value["x"], value["y"], value["width"], value["height"]
            )
        if all(key in value for key in ("left", "top", "right", "bottom")):
            left = _finite_number(value["left"], "roi.left")
            top = _finite_number(value["top"], "roi.top")
            right = _finite_number(value["right"], "roi.right")
            bottom = _finite_number(value["bottom"], "roi.bottom")
            return _rect_from_values(left, top, right - left, bottom - top)
        raise InspectionContractError(
            "ROI mapping requires x/y/width/height or left/top/right/bottom"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) != 4:
            raise InspectionContractError(
                "ROI sequence must contain x, y, width, and height"
            )
        return _rect_from_values(value[0], value[1], value[2], value[3])
    raise InspectionContractError(
        "ROI must be CropRect, a four-value sequence, or a rectangle mapping"
    )


def clamp_native_crop(rect: Any, native_size: Any) -> CropRect:
    """Clamp a native rectangle to image bounds and reject empty overlap."""

    source = coerce_crop_rect(rect)
    size = coerce_native_size(native_size)
    x0 = max(0, min(size.width, source.x))
    y0 = max(0, min(size.height, source.y))
    x1 = max(0, min(size.width, source.x1))
    y1 = max(0, min(size.height, source.y1))
    if x1 <= x0 or y1 <= y0:
        raise InspectionContractError("crop does not intersect native image bounds")
    return CropRect(x0, y0, x1 - x0, y1 - y0)


def _full_image_crop(native_size: NativeSize) -> CropRect:
    return CropRect(0, 0, native_size.width, native_size.height)


def _face_bbox(face: Any) -> CropRect:
    """Read a Retouch FaceData/FaceContext-like ``bbox`` value."""

    candidate = face
    if isinstance(candidate, Mapping) and "face_data" in candidate:
        candidate = candidate["face_data"]
    elif hasattr(candidate, "face_data"):
        candidate = getattr(candidate, "face_data")

    if isinstance(candidate, Mapping):
        if "bbox" not in candidate:
            raise InspectionContractError("face selection has no bbox")
        bbox = candidate["bbox"]
    else:
        bbox = getattr(candidate, "bbox", None)
        if bbox is None:
            raise InspectionContractError("face selection has no bbox")
    return coerce_crop_rect(bbox)


@dataclass(frozen=True)
class FaceCropSelection:
    """Evidence for the face chosen by a native inspection contract."""

    index: int
    source_bbox: CropRect
    crop: CropRect
    padding: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "face",
            "index": self.index,
            "source_bbox": self.source_bbox.to_dict(),
            "crop": self.crop.to_dict(),
            "padding": self.padding,
        }


def select_face_crop(
    faces: Iterable[Any],
    face_index: int,
    native_size: Any,
    *,
    padding: int = 0,
) -> FaceCropSelection:
    """Select and clamp one face's native ``(x, y, w, h)`` bounding box."""

    if isinstance(face_index, bool) or not isinstance(face_index, int) or face_index < 0:
        raise InspectionContractError("face_index must be a non-negative integer")
    if isinstance(faces, (str, bytes, bytearray)):
        raise InspectionContractError("faces must be an iterable of face records")
    try:
        face_values = list(faces)
    except TypeError:
        raise InspectionContractError("faces must be an iterable of face records")
    if face_index >= len(face_values):
        raise InspectionContractError("face_index is outside the detected face list")
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise InspectionContractError("face padding must be a non-negative integer")

    source_bbox = _face_bbox(face_values[face_index])
    padded = CropRect(
        source_bbox.x - padding,
        source_bbox.y - padding,
        source_bbox.width + (2 * padding),
        source_bbox.height + (2 * padding),
    )
    crop = clamp_native_crop(padded, native_size)
    return FaceCropSelection(
        index=face_index,
        source_bbox=source_bbox,
        crop=crop,
        padding=padding,
    )


def select_roi_crop(roi: Any, native_size: Any) -> CropRect:
    """Convert and clamp an explicit ROI in native pixel coordinates."""

    return clamp_native_crop(roi, native_size)


def compare_render_revisions(
    rendered_revision: Any,
    requested_revision: Any,
    current_revision: Any,
) -> Dict[str, Any]:
    """Compare render, request, and current-draft revisions fail-closed."""

    try:
        rendered = _require_revision(rendered_revision, "rendered_revision")
        requested = _require_revision(requested_revision, "requested_revision")
        current = _require_revision(current_revision, "current_revision")
    except InspectionContractError as exc:
        return {
            "valid": False,
            "stale": True,
            "status": "unresolved",
            "reason": str(exc),
            "rendered_revision": None,
            "requested_revision": None,
            "current_revision": None,
        }

    if rendered != requested:
        return {
            "valid": False,
            "stale": True,
            "status": "revision_mismatch",
            "reason": "rendered_revision_does_not_match_request",
            "rendered_revision": rendered,
            "requested_revision": requested,
            "current_revision": current,
        }
    if rendered != current:
        status = "stale" if rendered < current else "invalid_order"
        return {
            "valid": False,
            "stale": True,
            "status": status,
            "reason": "rendered_revision_is_not_current",
            "rendered_revision": rendered,
            "requested_revision": requested,
            "current_revision": current,
        }
    return {
        "valid": True,
        "stale": False,
        "status": "current",
        "reason": None,
        "rendered_revision": rendered,
        "requested_revision": requested,
        "current_revision": current,
    }


def assert_current_render_revision(
    rendered_revision: Any,
    current_revision: Any,
    *,
    requested_revision: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return revision evidence or reject stale/mismatched render pixels."""

    requested = current_revision if requested_revision is None else requested_revision
    evidence = compare_render_revisions(rendered_revision, requested, current_revision)
    if not evidence["valid"]:
        if evidence["status"] == "revision_mismatch":
            raise RevisionMismatchError(evidence["reason"])
        raise StaleInspectionError(evidence["reason"])
    return evidence


def _download_disabled_preview_state(
    render_revision: int,
    mode: str,
) -> Dict[str, Any]:
    return {
        "kind": "native_inspection_preview",
        "preview_only": True,
        "render_revision": render_revision,
        "mode": mode,
        "download_enabled": False,
        "download": {
            "allowed": False,
            "available": False,
            "href": None,
            "reason": "inspection previews are ephemeral; use full-quality export",
        },
    }


def build_download_disabled_preview_state(
    render_revision: Any,
    *,
    mode: Any = MODE_FIT,
) -> Dict[str, Any]:
    """Build standalone preview state with download explicitly disabled."""

    revision = _require_revision(render_revision, "render_revision")
    canonical_mode = normalize_mode(mode)
    return _download_disabled_preview_state(revision, canonical_mode)


def build_inspection_contract(
    *,
    mode: Any,
    native_size: Any,
    render_revision: Any,
    current_revision: Any,
    requested_revision: Optional[Any] = None,
    faces: Optional[Iterable[Any]] = None,
    face_index: int = 0,
    face_padding: int = 0,
    roi: Any = None,
) -> Dict[str, Any]:
    """Build a native-resolution inspection contract.

    ``render_revision`` identifies the pixels supplied by the render.  If
    ``requested_revision`` is omitted, the current draft revision is the
    requested revision.  The function rejects both a render/request mismatch
    and a render that became stale while queued.
    """

    canonical_mode = normalize_mode(mode)
    size = coerce_native_size(native_size)
    requested = current_revision if requested_revision is None else requested_revision
    revision_evidence = assert_current_render_revision(
        render_revision,
        current_revision,
        requested_revision=requested,
    )
    rendered = revision_evidence["rendered_revision"]

    selection: Dict[str, Any]
    if canonical_mode == MODE_FIT:
        crop = _full_image_crop(size)
        selection = {"kind": "full_image"}
        display = {
            "zoom": "fit",
            "scale": "fit",
            "scale_factor": None,
        }
    elif canonical_mode == MODE_100_PERCENT:
        crop = _full_image_crop(size)
        selection = {"kind": "full_image"}
        display = {
            "zoom": "100%",
            "scale": "native_one_to_one",
            "scale_factor": 1.0,
        }
    elif canonical_mode == MODE_FACE:
        if faces is None:
            raise InspectionContractError("face mode requires detected faces")
        selected = select_face_crop(
            faces,
            face_index,
            size,
            padding=face_padding,
        )
        crop = selected.crop
        selection = selected.to_dict()
        display = {
            "zoom": "100%",
            "scale": "native_one_to_one",
            "scale_factor": 1.0,
        }
    else:
        if roi is None:
            raise InspectionContractError("ROI mode requires an ROI")
        crop = select_roi_crop(roi, size)
        selection = {"kind": "roi", "roi": crop.to_dict()}
        display = {
            "zoom": "100%",
            "scale": "native_one_to_one",
            "scale_factor": 1.0,
        }

    contract = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "mode": canonical_mode,
        "render": {
            "revision": rendered,
            "requested_revision": revision_evidence["requested_revision"],
            "current_revision": revision_evidence["current_revision"],
            "status": revision_evidence["status"],
            "stale": False,
        },
        "native": {
            "dimensions": size.to_dict(),
            "resolution": "native",
            "crop_is_resized": False,
            "interpolation": "none",
        },
        "crop": crop.to_dict(),
        "selection": selection,
        "display": display,
        "preview": _download_disabled_preview_state(rendered, canonical_mode),
        "safe_to_commit": True,
    }
    return contract


def _contract_crop(contract: Mapping[str, Any]) -> CropRect:
    native = contract.get("native")
    if not isinstance(native, Mapping):
        raise InspectionContractError("inspection contract has no native dimensions")
    crop = contract.get("crop")
    if crop is None:
        raise InspectionContractError("inspection contract has no crop")
    return clamp_native_crop(crop, native.get("dimensions"))


def validate_inspection_contract(
    contract: Any,
    current_revision: Any,
    *,
    requested_revision: Optional[Any] = None,
) -> Dict[str, Any]:
    """Validate a queued inspection contract before it replaces the view."""

    errors: List[str] = []
    if not isinstance(contract, Mapping):
        return {"valid": False, "stale": True, "errors": ["contract_mapping"]}
    if contract.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version")
    if contract.get("contract") != CONTRACT_NAME:
        errors.append("contract_name")
    try:
        mode = normalize_mode(contract.get("mode"))
    except InspectionContractError:
        errors.append("mode")
        mode = None

    render = contract.get("render")
    if not isinstance(render, Mapping):
        errors.append("render")
        render = {}
    rendered = render.get("revision")
    requested = (
        render.get("requested_revision")
        if requested_revision is None
        else requested_revision
    )
    if requested is None:
        requested = current_revision
    evidence = compare_render_revisions(rendered, requested, current_revision)
    if not evidence["valid"]:
        errors.append("revision")

    native = contract.get("native")
    try:
        size = coerce_native_size(native.get("dimensions")) if isinstance(native, Mapping) else None
    except InspectionContractError:
        size = None
    if size is None:
        errors.append("native_dimensions")
    elif (
        native.get("resolution") != "native"
        or native.get("crop_is_resized") is not False
        or native.get("interpolation") != "none"
    ):
        errors.append("native_resolution")

    if mode is not None and size is not None:
        try:
            _contract_crop(contract)
        except InspectionContractError:
            errors.append("crop")

    preview = contract.get("preview")
    if (
        not isinstance(preview, Mapping)
        or preview.get("preview_only") is not True
        or preview.get("download_enabled") is not False
        or not isinstance(preview.get("download"), Mapping)
        or preview["download"].get("allowed") is not False
        or preview["download"].get("available") is not False
    ):
        errors.append("download_disabled_preview")
    if isinstance(preview, Mapping) and preview.get("render_revision") != rendered:
        errors.append("preview_revision")

    if contract.get("safe_to_commit") is not True:
        errors.append("safe_to_commit")
    return {
        "valid": not errors,
        "stale": bool(evidence["stale"] or errors),
        "errors": errors,
        "revision": evidence,
        "mode": mode,
    }


def require_current_inspection(
    contract: Mapping[str, Any],
    current_revision: Any,
    *,
    requested_revision: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a copy of a valid current contract or reject it fail-closed."""

    report = validate_inspection_contract(
        contract,
        current_revision,
        requested_revision=requested_revision,
    )
    if not report["valid"]:
        if "revision" in report["errors"]:
            evidence = report.get("revision", {})
            if evidence.get("status") == "revision_mismatch":
                raise RevisionMismatchError(evidence.get("reason", "revision mismatch"))
            raise StaleInspectionError(evidence.get("reason", "stale inspection"))
        raise InspectionContractError(
            "invalid inspection contract: %s" % ", ".join(report["errors"])
        )
    return copy.deepcopy(dict(contract))


def native_image_size(image: Any) -> NativeSize:
    """Infer native dimensions from a PIL-like or array-like image."""

    image_size = getattr(image, "size", None)
    if isinstance(image_size, Sequence) and not isinstance(image_size, (str, bytes)):
        if len(image_size) == 2 and not hasattr(image_size, "__call__"):
            try:
                return NativeSize(image_size[0], image_size[1])
            except InspectionContractError:
                pass
    shape = getattr(image, "shape", None)
    if isinstance(shape, Sequence) and len(shape) >= 2:
        return NativeSize(shape[1], shape[0])
    raise InspectionContractError(
        "image must expose PIL size or an array-like shape"
    )


def crop_native_image(image: Any, crop: Any, native_size: Any = None) -> Any:
    """Return a native crop without resizing, conversion, or interpolation."""

    size = native_image_size(image) if native_size is None else coerce_native_size(native_size)
    bounds = clamp_native_crop(crop, size)
    if hasattr(image, "crop") and not hasattr(image, "shape"):
        return image.crop(bounds.to_xyxy())
    try:
        result = image[bounds.y : bounds.y1, bounds.x : bounds.x1]
    except (IndexError, KeyError, TypeError):
        try:
            rows = image[bounds.y : bounds.y1]
            result = [row[bounds.x : bounds.x1] for row in rows]
        except (IndexError, KeyError, TypeError) as exc:
            raise InspectionContractError("image does not support native cropping") from exc
    copy_method = getattr(result, "copy", None)
    return copy_method() if callable(copy_method) else result


def crop_from_inspection_contract(
    image: Any,
    contract: Mapping[str, Any],
    current_revision: Any,
    *,
    requested_revision: Optional[Any] = None,
) -> Any:
    """Validate revision and return the contract's native crop."""

    accepted = require_current_inspection(
        contract,
        current_revision,
        requested_revision=requested_revision,
    )
    return crop_native_image(image, accepted["crop"], accepted["native"]["dimensions"])


__all__ = [
    "CONTRACT_NAME",
    "SCHEMA_VERSION",
    "MODE_FIT",
    "MODE_100_PERCENT",
    "MODE_100",
    "MODE_ONE_TO_ONE",
    "MODE_FACE",
    "MODE_ROI",
    "SUPPORTED_MODES",
    "InspectionContractError",
    "RevisionMismatchError",
    "StaleInspectionError",
    "NativeSize",
    "CropRect",
    "FaceCropSelection",
    "normalize_mode",
    "coerce_native_size",
    "coerce_crop_rect",
    "clamp_native_crop",
    "select_face_crop",
    "select_roi_crop",
    "compare_render_revisions",
    "assert_current_render_revision",
    "build_download_disabled_preview_state",
    "build_inspection_contract",
    "validate_inspection_contract",
    "require_current_inspection",
    "native_image_size",
    "crop_native_image",
    "crop_from_inspection_contract",
]
