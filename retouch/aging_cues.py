"""P8 R1 observable appearance-cue measurements.

This module is deliberately measurement-only.  It describes what is visible
in the supplied photograph; it does not infer chronological/apparent age,
estimate physiological tissue properties, or edit pixels.

The first P8 tranche qualifies two descriptors that already have bounded
implementations in Retouch:

* signed LAB contrast for eyes, lips, and brows versus nearby skin; and
* face-scaled skin chroma variation (with the existing luminance texture
  fields retained as diagnostics).

Support is part of every result.  Missing or invalid support is represented by
an unavailable value rather than a numerical zero, because zero contrast and
no observable region are different research outcomes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .perceptual_metrics import (
    FeatureContrast,
    facial_feature_contrast,
    skin_homogeneity_state,
)


P8_DESCRIPTOR_REVISION = "p8-r1-appearance-v1"
P8_COLOR_CONVENTION = (
    "input BGR uint8 or float image in [0,255]; LAB L/a/b on [0,255] "
    "with a/b centered at 128"
)
P8_MEASUREMENT_SCOPE = "observable appearance only; no age or tissue inference"

_DEFAULT_MIN_PIXELS = 32
_FEATURE_REGION_ATTRS: Mapping[str, Sequence[str]] = {
    "eyes": ("left_eye", "right_eye"),
    "lips": ("lips",),
    "brows": ("left_eyebrow", "right_eyebrow"),
}


@dataclass(frozen=True)
class CueSupport:
    """Support and quality state for one observable measurement."""

    status: str
    reason: str
    pixel_count: int
    confidence: float
    details: Mapping[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "pixel_count": int(self.pixel_count),
            "confidence": float(self.confidence),
            "details": _json_safe(self.details),
        }


@dataclass(frozen=True)
class CueObservation:
    """One descriptor result with explicit unavailable-state semantics."""

    descriptor: str
    support: CueSupport
    values: Optional[Mapping[str, Any]]
    units: Mapping[str, str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "descriptor": self.descriptor,
            "support": self.support.to_dict(),
            "values": None if self.values is None else _json_safe(self.values),
            "units": _json_safe(self.units),
        }


@dataclass(frozen=True)
class P8CueReadout:
    """R1 readout for one face support on one image."""

    descriptor_revision: str
    measurement_scope: str
    color_convention: str
    image_shape: Tuple[int, int]
    feature_contrast: Mapping[str, CueObservation]
    chroma_variation: CueObservation

    @property
    def available_descriptors(self) -> Tuple[str, ...]:
        """Names of descriptors with usable support, in deterministic order."""
        names = [
            *(f"feature_contrast.{name}" for name, item in sorted(self.feature_contrast.items())
              if item.support.status == "available"),
            *(["skin_chroma_variation"]
              if self.chroma_variation.support.status == "available" else []),
        ]
        return tuple(names)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "descriptor_revision": self.descriptor_revision,
            "measurement_scope": self.measurement_scope,
            "color_convention": self.color_convention,
            "image_shape": [int(v) for v in self.image_shape],
            "feature_contrast": {
                name: item.to_dict()
                for name, item in sorted(self.feature_contrast.items())
            },
            "chroma_variation": self.chroma_variation.to_dict(),
            "available_descriptors": list(self.available_descriptors),
        }

    def to_json(self, *, indent: Optional[int] = None) -> str:
        """Return a deterministic JSON representation for QA evidence."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)


def measure_p8_cues(
    img_bgr: np.ndarray,
    regions: Any,
    *,
    skin_mask: Optional[np.ndarray] = None,
    min_pixels: int = _DEFAULT_MIN_PIXELS,
) -> P8CueReadout:
    """Measure the current P8 R1 appearance descriptors for one face.

    Args:
        img_bgr: BGR image in the engine's uint8 or float ``[0,255]``
            convention.
        regions: FaceRegions-like object.  The required attributes are
            ``skin``, ``left_eye``, ``right_eye``, ``lips``,
            ``left_eyebrow`` and ``right_eyebrow``.  Missing attributes are
            reported as missing support.
        skin_mask: Optional explicit skin support.  When omitted,
            ``regions.skin`` is used.
        min_pixels: Minimum support for a measurement to be considered
            available; must be at least 32 to match the underlying metric.
            This is a support rule, not a confidence interval.

    Returns:
        :class:`P8CueReadout` containing signed feature components and the
        chroma variation descriptor.  No scalar age score is produced.

    Raises:
        ValueError: if the image is not a finite ``HxWx3`` array or
            ``min_pixels`` is not positive.
    """
    _validate_image(img_bgr)
    if int(min_pixels) < _DEFAULT_MIN_PIXELS:
        raise ValueError(f"min_pixels must be at least {_DEFAULT_MIN_PIXELS}")
    min_pixels = int(min_pixels)
    shape = tuple(int(v) for v in img_bgr.shape[:2])

    source_skin = skin_mask if skin_mask is not None else _get_attr(regions, "skin")
    normalized_skin, skin_state = _normalise_support_mask(source_skin, shape)
    skin_pixels = _count_pixels(normalized_skin)
    if skin_state.status == "valid" and skin_pixels < min_pixels:
        skin_support = CueSupport(
            status="insufficient_support",
            reason=f"skin_fewer_than_{min_pixels}_pixels",
            pixel_count=skin_pixels,
            confidence=0.0,
            details={"minimum_pixels": min_pixels},
        )
    elif skin_state.status == "valid":
        skin_support = CueSupport(
            status="available",
            reason="available",
            pixel_count=skin_pixels,
            confidence=float(np.clip(skin_pixels / 800.0, 0.0, 1.0)),
            details={"minimum_pixels": min_pixels},
        )
    else:
        skin_support = CueSupport(
            status=skin_state.status,
            reason=skin_state.reason,
            pixel_count=skin_pixels,
            confidence=0.0,
            details={"minimum_pixels": min_pixels},
        )

    # Normalize the masks at the P8 boundary.  The existing metrics accept
    # both 0..1 and 0..255 masks, but a tiny float overshoot above 1.0 must be
    # clipped rather than mistaken for an 8-bit mask and divided by 255.
    normalized_regions, region_states = _normalised_feature_regions(regions, shape, normalized_skin)
    raw_features = facial_feature_contrast(
        img_bgr,
        normalized_regions,
        min_pixels=min_pixels,
    )

    feature_observations: Dict[str, CueObservation] = {}
    for name, attrs in _FEATURE_REGION_ATTRS.items():
        raw = raw_features[name]
        feature_observations[name] = _feature_observation(
            name,
            raw,
            skin_support,
            [region_states[attr] for attr in attrs],
            min_pixels,
        )

    raw_skin = skin_homogeneity_state(img_bgr, normalized_skin)
    chroma_values: Optional[Mapping[str, Any]] = None
    if skin_support.status == "available":
        chroma_values = {
            # This is a bandpass image-color statistic, not pigment or age.
            "chroma_blotch_std": float(raw_skin.chroma_blotch_std),
            # Retain the existing luminance diagnostic without presenting it
            # as part of the chroma descriptor.
            "pore_energy": float(raw_skin.pore_energy),
            "pore_retention": (
                None if raw_skin.pore_retention is None
                else float(raw_skin.pore_retention)
            ),
            "hemoglobin_variance": (
                None if raw_skin.hemoglobin_variance is None
                else float(raw_skin.hemoglobin_variance)
            ),
        }
    chroma_observation = CueObservation(
        descriptor="skin_chroma_variation",
        support=CueSupport(
            status=skin_support.status,
            reason=skin_support.reason,
            pixel_count=skin_support.pixel_count,
            confidence=float(raw_skin.confidence) if chroma_values is not None else 0.0,
            details={
                **dict(skin_support.details),
                "metric": "skin_homogeneity_state",
            },
        ),
        values=chroma_values,
        units={
            "chroma_blotch_std": "LAB chroma bandpass magnitude on 0-255 units",
            "pore_energy": "LAB L high-band standard deviation on 0-255 units",
            "pore_retention": "unitless ratio to supplied reference when present",
            "hemoglobin_variance": "variance of supplied hemoglobin map units",
        },
    )

    return P8CueReadout(
        descriptor_revision=P8_DESCRIPTOR_REVISION,
        measurement_scope=P8_MEASUREMENT_SCOPE,
        color_convention=P8_COLOR_CONVENTION,
        image_shape=shape,
        feature_contrast=feature_observations,
        chroma_variation=chroma_observation,
    )


# Descriptive alias used by the research protocol; keep the P8-specific name
# as the canonical API so callers do not mistake this for a generic age model.
measure_observable_cues = measure_p8_cues


@dataclass(frozen=True)
class _MaskState:
    status: str
    reason: str


def _feature_observation(
    name: str,
    raw: FeatureContrast,
    skin_support: CueSupport,
    region_states: Sequence[_MaskState],
    min_pixels: int,
) -> CueObservation:
    values: Optional[Mapping[str, Any]] = None
    support_status = skin_support.status
    reason = skin_support.reason
    details: Dict[str, Any] = {"minimum_pixels": min_pixels}

    if support_status == "available":
        valid_regions = [state for state in region_states if state.status == "valid"]
        invalid = next((state for state in region_states if state.status == "invalid_support"), None)
        missing = next((state for state in region_states if state.status == "missing_support"), None)
        # ``eyes`` and ``brows`` are composite descriptors.  One eye/brow can
        # remain measurable when its counterpart is absent; only an entirely
        # absent composite support should be reported as missing.
        if not valid_regions and invalid is not None:
            support_status, reason = invalid.status, invalid.reason
        elif not valid_regions and missing is not None:
            support_status, reason = missing.status, missing.reason
        elif raw.feature_pixels < min_pixels:
            support_status, reason = "insufficient_support", f"{name}_feature_fewer_than_{min_pixels}_pixels"
        elif raw.surround_pixels < min_pixels:
            support_status, reason = "insufficient_support", f"{name}_surround_fewer_than_{min_pixels}_pixels"
        else:
            values = {
                # Keep signed components.  A magnitude-only scalar can hide
                # whether a feature became lighter, darker, warmer, or cooler.
                "luminance_contrast": float(raw.luminance_contrast),
                "chroma_contrast": float(raw.chroma_contrast),
                "lab_delta": [float(v) for v in raw.lab_delta],
            }

    details.update({
        "feature_pixels": int(raw.feature_pixels),
        "surround_pixels": int(raw.surround_pixels),
        "region_support": [state.status for state in region_states],
    })
    return CueObservation(
        descriptor=f"feature_contrast.{name}",
        support=CueSupport(
            status=support_status,
            reason=reason,
            pixel_count=min(int(raw.feature_pixels), int(raw.surround_pixels)),
            confidence=float(raw.confidence) if values is not None else 0.0,
            details=details,
        ),
        values=values,
        units={
            "luminance_contrast": "relative signed LAB L difference, unitless",
            "chroma_contrast": "relative LAB chroma difference magnitude, unitless",
            "lab_delta": "signed LAB delta on 0-255 units (L/a/b; a/b centered at 128)",
        },
    )


def _normalised_feature_regions(
    regions: Any,
    shape: Tuple[int, int],
    skin: Optional[np.ndarray],
) -> Tuple[Any, Dict[str, _MaskState]]:
    values: Dict[str, Any] = {}
    states: Dict[str, _MaskState] = {}
    for attr in (
        "left_eye", "right_eye", "lips", "left_eyebrow", "right_eyebrow",
    ):
        raw = _get_attr(regions, attr)
        normalized, state = _normalise_support_mask(raw, shape)
        values[attr] = normalized
        states[attr] = state

    # Parser skin is intentionally an edit support and normally excludes the
    # eyes, lips, and brows.  The contrast descriptor needs the photographed
    # feature pixels while still using nearby skin for its annulus, so create
    # a private measurement-only union.  The original skin mask is retained
    # for chroma variation and is never mutated.
    if skin is not None:
        contrast_skin = skin.copy()
        for attr in values:
            feature = values[attr]
            if feature is not None:
                contrast_skin = np.maximum(contrast_skin, feature)
        values["skin"] = contrast_skin
    else:
        values["skin"] = None
    return SimpleNamespace(**values), states


def _normalise_support_mask(
    mask: Optional[np.ndarray],
    shape: Tuple[int, int],
) -> Tuple[Optional[np.ndarray], _MaskState]:
    if mask is None:
        return None, _MaskState("missing_support", "region_not_supplied")
    try:
        arr = np.asarray(mask)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[..., 0]
        if arr.shape != shape:
            return None, _MaskState("invalid_support", "mask_shape_mismatch")
        arr = arr.astype(np.float32, copy=False)
    except (TypeError, ValueError):
        return None, _MaskState("invalid_support", "mask_not_numeric")
    if not np.isfinite(arr).all():
        return None, _MaskState("invalid_support", "mask_contains_non_finite_values")
    if arr.size and float(arr.max()) > 1.0:
        # Masks in the engine convention are either [0,1] or [0,255].  Keep a
        # small float overshoot in the normalized branch; otherwise use the
        # explicit 8-bit branch.
        if float(arr.max()) <= 1.0 + 1e-3:
            arr = np.clip(arr, 0.0, 1.0)
        else:
            arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0), _MaskState("valid", "available")


def _count_pixels(mask: Optional[np.ndarray]) -> int:
    return 0 if mask is None else int(np.count_nonzero(mask > 0.5))


def _get_attr(obj: Any, name: str) -> Any:
    return getattr(obj, name, None) if obj is not None else None


def _validate_image(img_bgr: np.ndarray) -> None:
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be an HxWx3 BGR image")
    if img_bgr.size == 0 or not np.isfinite(img_bgr).all():
        raise ValueError("img_bgr must be non-empty and finite")


def _json_safe(value: Any) -> Any:
    """Convert nested NumPy/dataclass values to strict JSON primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    return str(value)


__all__ = [
    "CueObservation",
    "CueSupport",
    "P8CueReadout",
    "P8_COLOR_CONVENTION",
    "P8_DESCRIPTOR_REVISION",
    "P8_MEASUREMENT_SCOPE",
    "measure_observable_cues",
    "measure_p8_cues",
]
