"""Metamorphic robustness checks for image-processing stages.

The lab does not prescribe one exact output for every transformed input. It
undoes the controlled input transform where possible, then measures whether a
logically equivalent input produces a materially different result or stage
decision. This is intentionally processor-agnostic so it can cover the engine,
Safe Auto, and Advanced Retouch handlers without importing the GUI.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

import cv2
import numpy as np


ImageProcessor = Callable[[np.ndarray], np.ndarray]
DecisionProvider = Callable[[np.ndarray], Mapping[str, object]]


@dataclass(frozen=True)
class MetamorphicVariant:
    name: str
    image_bgr: np.ndarray
    restore: Callable[[np.ndarray], np.ndarray]
    metadata: Mapping[str, object]


@dataclass(frozen=True)
class MetamorphicObservation:
    name: str
    passed: bool
    mean_abs_error: float
    p95_abs_error: float
    max_abs_error: float
    decision_equal: Optional[bool]
    reason: str


def _uint8_bgr(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected BGR image with shape (H,W,3), got {arr.shape}.")
    if arr.dtype != np.uint8:
        scale = 255.0 if arr.size and float(np.nanmax(arr)) <= 1.0 + 1e-6 else 1.0
        arr = np.clip(arr.astype(np.float32) * scale, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _resize(image: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA if h < image.shape[0] else cv2.INTER_LINEAR)


def _restore_shape(image: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    return _resize(image, shape) if image.shape[:2] != tuple(shape[:2]) else image


def generate_variants(
    image_bgr: np.ndarray,
    *,
    jpeg_quality: int = 75,
    resize_scale: float = 0.5,
    exposure_factor: float = 1.10,
) -> List[MetamorphicVariant]:
    """Create controlled variants and inverse transforms for one BGR image."""
    base = _uint8_bgr(image_bgr)
    shape = base.shape[:2]
    variants: List[MetamorphicVariant] = []

    rotated = np.rot90(base, 1).copy()
    variants.append(MetamorphicVariant(
        "exif_rotation", rotated,
        lambda output: np.rot90(output, -1).copy(),
        {"rotation_degrees": 90},
    ))

    scale = float(np.clip(resize_scale, 0.1, 0.95))
    small_shape = (max(1, int(shape[0] * scale)), max(1, int(shape[1] * scale)))
    small = _resize(base, small_shape)
    variants.append(MetamorphicVariant(
        "resize_proxy", small,
        lambda output: _restore_shape(output, shape),
        {"scale": scale},
    ))

    ok, encoded = cv2.imencode(".jpg", base, [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(jpeg_quality, 10, 100))])
    if not ok:
        raise ValueError("Could not create JPEG metamorphic variant.")
    jpeg = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if jpeg is None:
        raise ValueError("Could not decode JPEG metamorphic variant.")
    variants.append(MetamorphicVariant(
        "jpeg_quality", jpeg,
        lambda output: _restore_shape(output, shape),
        {"quality": int(np.clip(jpeg_quality, 10, 100))},
    ))

    factor = max(0.1, float(exposure_factor))
    exposed = np.clip(base.astype(np.float32) * factor, 0.0, 255.0).astype(np.uint8)
    variants.append(MetamorphicVariant(
        "exposure_plus", exposed,
        lambda output: np.clip(output.astype(np.float32) / factor, 0.0, 255.0).astype(np.uint8),
        {"factor": factor},
    ))

    # Deliberately use channel gains instead of a named color-space conversion:
    # this tests white-balance sensitivity without claiming that OpenCV's BGR
    # arithmetic is a Display-P3 transform.
    gains = np.array([1.04, 1.0, 0.96], dtype=np.float32)
    white_balance = np.clip(base.astype(np.float32) * gains[None, None, :], 0.0, 255.0).astype(np.uint8)
    variants.append(MetamorphicVariant(
        "white_balance_shift", white_balance,
        lambda output: np.clip(output.astype(np.float32) / gains[None, None, :], 0.0, 255.0).astype(np.uint8),
        {"bgr_gains": gains.tolist()},
    ))

    # Quantize through 16-bit storage and back. For uint8 inputs this should be
    # byte-identical; the case still catches code that mishandles bit-depth
    # metadata or converts through a different channel order.
    sixteen = (base.astype(np.uint16) * 257).astype(np.uint16)
    eight_roundtrip = np.rint(sixteen.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    variants.append(MetamorphicVariant(
        "bit_depth_roundtrip", eight_roundtrip,
        lambda output: _restore_shape(output, shape),
        {"source_bits": 8, "roundtrip_bits": 16},
    ))
    return variants


def _decision_equal(baseline: Optional[Mapping[str, object]], candidate: Optional[Mapping[str, object]]) -> Optional[bool]:
    if baseline is None or candidate is None:
        return None
    return dict(baseline) == dict(candidate)


def run_lab(
    image_bgr: np.ndarray,
    processor: ImageProcessor,
    *,
    variants: Optional[Iterable[MetamorphicVariant]] = None,
    decision_provider: Optional[DecisionProvider] = None,
    mean_threshold: float = 8.0,
    p95_threshold: float = 24.0,
) -> List[MetamorphicObservation]:
    """Run a processor against variants and return explicit gate observations."""
    source = _uint8_bgr(image_bgr)
    baseline = _uint8_bgr(processor(source.copy()))
    baseline_decision = decision_provider(source.copy()) if decision_provider else None
    observations: List[MetamorphicObservation] = []
    for variant in variants if variants is not None else generate_variants(source):
        candidate = _uint8_bgr(processor(variant.image_bgr.copy()))
        restored = _uint8_bgr(variant.restore(candidate))
        restored = _restore_shape(restored, baseline.shape[:2])
        delta = np.abs(restored.astype(np.float32) - baseline.astype(np.float32))
        mean_error = float(delta.mean())
        p95_error = float(np.percentile(delta, 95))
        max_error = float(delta.max())
        candidate_decision = decision_provider(variant.image_bgr.copy()) if decision_provider else None
        decision_equal = _decision_equal(baseline_decision, candidate_decision)
        passed = mean_error <= mean_threshold and p95_error <= p95_threshold and decision_equal is not False
        reasons = []
        if mean_error > mean_threshold:
            reasons.append(f"mean error {mean_error:.2f} > {mean_threshold:.2f}")
        if p95_error > p95_threshold:
            reasons.append(f"p95 error {p95_error:.2f} > {p95_threshold:.2f}")
        if decision_equal is False:
            reasons.append("stage decision changed")
        observations.append(MetamorphicObservation(
            variant.name, passed, mean_error, p95_error, max_error,
            decision_equal, "; ".join(reasons) if reasons else "within configured tolerance",
        ))
    return observations


def observations_to_dict(observations: Iterable[MetamorphicObservation]) -> List[Dict[str, object]]:
    return [
        {
            "name": observation.name,
            "passed": observation.passed,
            "mean_abs_error": observation.mean_abs_error,
            "p95_abs_error": observation.p95_abs_error,
            "max_abs_error": observation.max_abs_error,
            "decision_equal": observation.decision_equal,
            "reason": observation.reason,
        }
        for observation in observations
    ]
