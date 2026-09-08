"""Class-aware mark records and opt-in policy mask compilation.

This module deliberately does not alter any retouch stage.  It converts the
existing detector outputs into a shared taxonomy and, when a caller supplies
an explicit policy, compiles the requested preserve/heal/attenuate regions.
Keeping the no-policy result empty is the compatibility boundary: legacy
freckle and mole controls continue to own their current pixels unchanged.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Optional, Sequence

import cv2
import numpy as np

from .freckle import FreckleClassification, FreckleRemover
from .utils import normalize_mask


MARK_CLASSES = frozenset({
    "mole", "freckle", "acne_blemish", "scar", "drawn_makeup_mark",
    "stray_hair", "sensor_dust", "vellus_sheen", "unknown",
})
_ACTIONS = frozenset({"preserve", "remove", "attenuate", "enhance"})
_FRECKLE_CLASS_MAP = {
    "beauty_mark": "mole",
    "freckle": "freckle",
    "blemish": "acne_blemish",
    "noise": "unknown",
    # FreckleRemover routes an unresolved tie/near-tie between competing
    # class scores here (retouch/freckle.py's "ambiguous" outcome) rather
    # than guessing; "unknown" is the existing MARK_CLASSES member for
    # "detected, but its type could not be determined" and every named
    # policy already has a defined action for it (protect_identity and
    # preserve_all both preserve unknown).
    "ambiguous": "unknown",
}

# Named policies are deliberately conservative. They make the class-aware
# preserve mask available to the existing freckle stage; removal/attenuation
# masks are compiled for later consumers but do not add a new destructive path.
MARK_POLICY_PRESETS: dict[str, Optional[dict[str, Any]]] = {
    "legacy": None,
    "protect_identity": {
        "min_confidence": 0.6,
        "mole": {"action": "preserve"},
        "drawn_makeup_mark": {"action": "preserve"},
        "scar": {"action": "preserve"},
        "unknown": {"action": "preserve"},
        "freckle": {"action": "attenuate", "strength": 40},
        "acne_blemish": {"action": "remove"},
    },
    "preserve_all": {
        "min_confidence": 0.0,
        "mole": {"action": "preserve"},
        "freckle": {"action": "preserve"},
        "acne_blemish": {"action": "preserve"},
        "scar": {"action": "preserve"},
        "drawn_makeup_mark": {"action": "preserve"},
        "stray_hair": {"action": "preserve"},
        "sensor_dust": {"action": "preserve"},
        "vellus_sheen": {"action": "preserve"},
        "unknown": {"action": "preserve"},
    },
}
MARK_POLICY_PRESET_NAMES = tuple(MARK_POLICY_PRESETS)


def resolve_mark_policy(policy: Optional[Mapping[str, Any] | str]) -> Optional[dict[str, Any]]:
    """Resolve a named UI/CLI policy or copy an explicit policy mapping.

    ``legacy`` and ``None`` retain the pre-policy execution path exactly.
    """
    if policy is None or policy == "legacy":
        return None
    if isinstance(policy, str):
        if policy not in MARK_POLICY_PRESETS:
            raise ValueError(
                f"Unknown mark policy {policy!r}; choose from {list(MARK_POLICY_PRESET_NAMES)}"
            )
        return copy.deepcopy(MARK_POLICY_PRESETS[policy])
    if isinstance(policy, Mapping):
        return copy.deepcopy(dict(policy))
    raise TypeError("mark policy must be a preset name, mapping, or None")


@dataclass(frozen=True)
class MarkRecord:
    """One detector observation in the common, scale-free mark taxonomy."""

    mark_id: int
    mark_class: str
    confidence: float
    centroid: tuple[float, float]
    area_norm: float
    bbox: tuple[int, int, int, int]
    features: dict[str, float]
    on_body: bool

    def __post_init__(self) -> None:
        if self.mark_class not in MARK_CLASSES:
            raise ValueError(f"Unknown mark class {self.mark_class!r}")
        if self.mark_id < 0:
            raise ValueError("mark_id must be non-negative")
        if self.area_norm < 0:
            raise ValueError("area_norm must be non-negative")
        if len(self.bbox) != 4 or self.bbox[2] < 0 or self.bbox[3] < 0:
            raise ValueError("bbox must be (x, y, width, height)")
        object.__setattr__(self, "confidence", float(np.clip(self.confidence, 0.0, 1.0)))
        object.__setattr__(
            self,
            "features",
            {str(name): float(value) for name, value in self.features.items()},
        )


@dataclass(frozen=True)
class MarkPolicyMasks:
    """Masks produced by an explicit mark policy.

    ``attenuate`` and ``enhance`` are strength-weighted float32 masks in
    ``[0, 1]``.  The heal and preserve masks are binary uint8 masks suitable
    for the existing inpaint and protect paths.
    """

    preserve: np.ndarray
    heal: np.ndarray
    attenuate: np.ndarray
    enhance: np.ndarray


def _shape2d(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) < 2:
        raise ValueError(f"image shape must have at least two dimensions, got {shape}")
    return int(shape[0]), int(shape[1])


def _component_records(
    mask: np.ndarray,
    *,
    mark_class: str,
    face_width: float,
    start_id: int,
    confidence: float,
    on_body: bool,
) -> list[MarkRecord]:
    """Adapt an existing detector's binary mask without re-detecting it."""
    binary = (normalize_mask(mask) > 0.3).astype(np.uint8)
    labels_count, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    records: list[MarkRecord] = []
    denom = max(float(face_width) ** 2, 1.0)
    for label in range(1, labels_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = float(stats[label, cv2.CC_STAT_AREA])
        records.append(MarkRecord(
            mark_id=start_id + len(records),
            mark_class=mark_class,
            confidence=confidence,
            centroid=(float(centroids[label, 0]), float(centroids[label, 1])),
            area_norm=area / denom,
            bbox=(x, y, width, height),
            features=_geometry_features(width, height),
            on_body=on_body,
        ))
    return records


def _geometry_features(width: int, height: int) -> dict[str, float]:
    major = float(max(width, height, 1))
    minor = float(max(min(width, height), 1))
    return {
        "eccentricity": float(np.sqrt(max(0.0, 1.0 - (minor / major) ** 2))),
        "edge_sharpness": 0.0,
        "symmetry_score": 0.0,
        "cluster_density": 0.0,
    }


def adapt_detector_mask(
    mask: np.ndarray,
    *,
    mark_class: str,
    face_width: float,
    confidence: float = 1.0,
    on_body: bool = False,
    start_id: int = 0,
) -> list[MarkRecord]:
    """Adapt a binary mole/blemish/flyaway detector output to mark records."""
    if mark_class not in MARK_CLASSES:
        raise ValueError(f"Unknown mark class {mark_class!r}")
    return _component_records(
        mask,
        mark_class=mark_class,
        face_width=face_width,
        start_id=start_id,
        confidence=confidence,
        on_body=on_body,
    )


def _relative_features(
    img_bgr: np.ndarray,
    classification: FreckleClassification,
    reference_mask: np.ndarray,
    mel_map: Optional[np.ndarray],
    hb_map: Optional[np.ndarray],
    *,
    lab: Optional[np.ndarray] = None,
    reference_pixels: Optional[np.ndarray] = None,
    reference_stats: Optional[dict[int, tuple[float, float]]] = None,
    map_context: Optional[dict[str, tuple[np.ndarray, float, float]]] = None,
) -> dict[str, float]:
    """Extract only subject-relative appearance features for one component.

    ``lab``/``reference_pixels``/``*_stats`` are optional so this private
    helper remains compatible with callers that used the original signature.
    ``detect_marks`` supplies a precomputed context: those values are
    invariant for every component in one detection pass and used to be
    recomputed in the per-component loop.
    """
    if lab is None:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    reference = reference_mask > 0.3
    if reference_pixels is None:
        reference_pixels = lab[reference] if np.any(reference) else lab.reshape(-1, 3)

    if reference_stats is None:
        reference_stats = {}
        for channel in (0, 1):
            values = reference_pixels[:, channel]
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            reference_stats[channel] = (median, mad)

    x, y, width, height = classification.bbox
    h, w = img_bgr.shape[:2]
    x0, x1 = max(x, 0), min(x + width, w)
    y0, y1 = max(y, 0), min(y + height, h)
    component = lab[y0:y1, x0:x1]
    component_pixels = component.reshape(-1, 3) if component.size else reference_pixels

    def relative(channel: int) -> float:
        median, mad = reference_stats[channel]
        return (float(np.median(component_pixels[:, channel])) - median) / max(1.4826 * mad, 1.0)

    features = _geometry_features(width, height)
    features.update({"l_margin": relative(0), "a_margin": relative(1)})
    for name, source in (("mel_rel", mel_map), ("hb_rel", hb_map)):
        if source is None:
            features[name] = 0.0
            continue
        if map_context is not None and name in map_context:
            values, median, mad = map_context[name]
        else:
            values = source.astype(np.float32)
            baseline = values[reference] if np.any(reference) else values.reshape(-1)
            median = float(np.median(baseline))
            mad = float(np.median(np.abs(baseline - median)))
        comp_values = values[y0:y1, x0:x1]
        features[name] = (float(np.median(comp_values)) - median) / max(1.4826 * mad, 1e-4)
    return features


def _with_cluster_density(records: Sequence[MarkRecord], face_width: float) -> list[MarkRecord]:
    radius = max(float(face_width) * 0.12, 1.0)
    result: list[MarkRecord] = []
    for record in records:
        neighbours = sum(
            np.hypot(record.centroid[0] - other.centroid[0], record.centroid[1] - other.centroid[1]) <= radius
            for other in records if other.mark_id != record.mark_id
        )
        features = dict(record.features)
        features["cluster_density"] = float(neighbours)
        result.append(replace(record, features=features))
    return result


def detect_marks(
    img_bgr: np.ndarray,
    *,
    face_mask: Optional[np.ndarray] = None,
    face_width: Optional[float] = None,
    confidence_threshold: float = 0.6,
    mel_map: Optional[np.ndarray] = None,
    hb_map: Optional[np.ndarray] = None,
    mole_mask: Optional[np.ndarray] = None,
    blemish_mask: Optional[np.ndarray] = None,
    stray_hair_mask: Optional[np.ndarray] = None,
    on_body: bool = False,
) -> list[MarkRecord]:
    """Unify the existing freckle and optional detector masks into records.

    Optional masks are adapters for R10 mole/blemish and hairwork flyaway
    outputs. They are inputs rather than new detector calls, so this helper
    remains lightweight and does not change any production execution path.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3 or img_bgr.dtype != np.uint8:
        raise ValueError("detect_marks expects a uint8 BGR image")
    h, w = img_bgr.shape[:2]
    normalized_face = normalize_mask(face_mask)
    if normalized_face is None:
        normalized_face = np.ones((h, w), dtype=np.float32)
    if normalized_face.shape != (h, w):
        normalized_face = cv2.resize(normalized_face, (w, h), interpolation=cv2.INTER_LINEAR)
    width = float(face_width) if face_width is not None else float(max(np.count_nonzero(normalized_face > 0.3) ** 0.5, 1.0))

    records: list[MarkRecord] = []
    classifications = FreckleRemover().classify_anomalies(
        img_bgr, face_mask=normalized_face, confidence_threshold=confidence_threshold)

    if classifications:
        # Build the appearance context once per detection pass.  These values
        # are identical for every component; keeping them outside the
        # classification loop removes repeated LAB conversion, boolean
        # extraction, median and MAD work without changing per-component
        # arithmetic.  A no-component pass skips this extra conversion.
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        reference = normalized_face > 0.3
        has_reference = bool(np.any(reference))
        reference_pixels = lab[reference] if has_reference else lab.reshape(-1, 3)
        reference_stats: dict[int, tuple[float, float]] = {}
        for channel in (0, 1):
            values = reference_pixels[:, channel]
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            reference_stats[channel] = (median, mad)
        map_context: dict[str, tuple[np.ndarray, float, float]] = {}
        for name, source in (("mel_rel", mel_map), ("hb_rel", hb_map)):
            if source is None:
                continue
            values = source.astype(np.float32)
            baseline = values[reference] if has_reference else values.reshape(-1)
            median = float(np.median(baseline))
            mad = float(np.median(np.abs(baseline - median)))
            map_context[name] = (values, median, mad)

    else:
        # Keep these names defined for type checkers; the loop below is empty.
        lab = None
        reference_pixels = None
        reference_stats = None
        map_context = None

    for classification in classifications:
        records.append(MarkRecord(
            mark_id=len(records),
            mark_class=_FRECKLE_CLASS_MAP[classification.classification],
            confidence=classification.confidence,
            centroid=classification.centroid,
            area_norm=classification.area / max(width ** 2, 1.0),
            bbox=classification.bbox,
            features=_relative_features(
                img_bgr,
                classification,
                normalized_face,
                mel_map,
                hb_map,
                lab=lab,
                reference_pixels=reference_pixels,
                reference_stats=reference_stats,
                map_context=map_context,
            ),
            on_body=on_body,
        ))
    for detector_mask, mark_class in (
        (mole_mask, "mole"),
        (blemish_mask, "acne_blemish"),
        (stray_hair_mask, "stray_hair"),
    ):
        if detector_mask is not None:
            records.extend(adapt_detector_mask(
                detector_mask, mark_class=mark_class, face_width=width,
                start_id=len(records), on_body=on_body))
    return _with_cluster_density(records, width)


def resolve_mark_action(
    record: MarkRecord,
    mark_policy: Optional[Mapping[str, Any]],
) -> Optional[tuple[str, float]]:
    """Resolve one record's explicit policy action and normalized strength.

    ``None`` means no policy was supplied.  Consumers that need legacy
    behavior must retain their existing path in that case rather than infer a
    new default from this module.
    """
    if mark_policy is None:
        return None
    policy = mark_policy
    min_confidence = float(policy.get("min_confidence", 0.0))
    mark_class = record.mark_class if record.confidence >= min_confidence else "unknown"
    if record.on_body and isinstance(policy.get("body"), Mapping):
        body_policy = policy["body"]
        config = body_policy.get(mark_class, policy.get(mark_class, policy.get("unknown", {"action": "preserve"})))
    else:
        config = policy.get(mark_class, policy.get("unknown", {"action": "preserve"}))
    if not isinstance(config, Mapping):
        raise ValueError(f"Policy for {mark_class!r} must be a mapping")
    action = str(config.get("action", "preserve"))
    if action not in _ACTIONS:
        raise ValueError(f"Unknown mark-policy action {action!r}")
    strength = float(np.clip(config.get("strength", 100.0), 0.0, 100.0))
    return action, strength / 100.0


def _record_footprint(record: MarkRecord, shape: tuple[int, int]) -> np.ndarray:
    footprint = np.zeros(shape, dtype=np.uint8)
    x, y, width, height = record.bbox
    cx, cy = int(round(record.centroid[0])), int(round(record.centroid[1]))
    rx = max(int(round(width / 2.0)), 1)
    ry = max(int(round(height / 2.0)), 1)
    cv2.ellipse(footprint, (cx, cy), (rx, ry), 0.0, 0.0, 360.0, 255, -1)
    return footprint


def compile_mark_policy(
    records: Iterable[MarkRecord],
    image_shape: tuple[int, ...],
    mark_policy: Optional[Mapping[str, Any]],
) -> MarkPolicyMasks:
    """Compile explicit policy actions into masks without applying an edit.

    ``mark_policy=None`` is intentionally an exact empty result. A caller
    must opt in before this module can affect legacy output.
    """
    h, w = _shape2d(image_shape)
    preserve = np.zeros((h, w), dtype=np.uint8)
    heal = np.zeros((h, w), dtype=np.uint8)
    attenuate = np.zeros((h, w), dtype=np.float32)
    enhance = np.zeros((h, w), dtype=np.float32)
    if mark_policy is None:
        return MarkPolicyMasks(preserve, heal, attenuate, enhance)
    for record in records:
        action_strength = resolve_mark_action(record, mark_policy)
        assert action_strength is not None
        action, strength = action_strength
        footprint = _record_footprint(record, (h, w))
        if action == "preserve":
            preserve = np.maximum(preserve, footprint)
        elif action == "remove":
            heal = np.maximum(heal, footprint)
        elif action == "attenuate":
            attenuate = np.maximum(attenuate, (footprint > 0).astype(np.float32) * strength)
        else:
            enhance = np.maximum(enhance, (footprint > 0).astype(np.float32) * strength)
    heal[preserve > 0] = 0
    return MarkPolicyMasks(preserve, heal, attenuate, enhance)
