"""Transparent, non-decisional face-quality measurements.

This module measures geometry and local focus evidence for faces already
detected by :mod:`retouch.detection`. It deliberately does not identify a
person, infer attractiveness/expression, decide whether eyes are open, or
select/reject an image. Consumers must keep the resulting evidence reviewable.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .shoot_review import FaceQualityEvidence


FACE_QUALITY_VERSION = "face-quality-v1"
SHARPNESS_METHOD = "contrast-normalized-tenengrad-v1"

# Canonical MediaPipe face-mesh eye contours. These indices exist in the
# 468-point mesh as well as the refined 478-point result, so iris refinement is
# not required merely to locate an eye crop.
LEFT_EYE_INDICES = (263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466)
RIGHT_EYE_INDICES = (33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246)


def _gray_float(image: np.ndarray) -> np.ndarray:
    """Return a finite grayscale float image in the range 0..1."""
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[2] == 4:
            array = cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        elif array.shape[2] == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("face-quality input must have 1, 3, or 4 channels")
    elif array.ndim != 2:
        raise ValueError("face-quality input must be a 2D or 3D image")
    if array.size == 0:
        raise ValueError("face-quality input must not be empty")

    gray = array.astype(np.float32, copy=False)
    if np.issubdtype(array.dtype, np.integer):
        maximum = float(np.iinfo(array.dtype).max)
        if maximum > 0.0:
            gray = gray / maximum
    else:
        finite = gray[np.isfinite(gray)]
        if finite.size == 0:
            raise ValueError("face-quality input contains no finite pixels")
        observed_max = float(np.max(finite))
        if observed_max > 1.0:
            gray = gray / (65535.0 if observed_max > 255.0 else 255.0)
    return np.nan_to_num(gray, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)


def _normalized_tenengrad(crop: np.ndarray, target_size: Tuple[int, int]) -> Tuple[Optional[float], float]:
    """Measure first-derivative energy after local contrast normalization.

    The fixed analysis size removes the most direct crop-resolution dependency.
    Percentile normalization reduces brightness/skin-tone dependence. The
    returned value remains raw evidence, not a calibrated pass/fail score.
    """
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None, 0.0
    low, high = np.percentile(crop, (5.0, 95.0))
    contrast_span = float(max(0.0, high - low))
    if contrast_span <= 1e-6:
        return 0.0, contrast_span
    normalized = np.clip((crop - low) / contrast_span, 0.0, 1.0)
    interpolation = cv2.INTER_AREA if (
        normalized.shape[1] > target_size[0] or normalized.shape[0] > target_size[1]
    ) else cv2.INTER_LINEAR
    normalized = cv2.resize(normalized, target_size, interpolation=interpolation)
    # A small, declared prefilter reduces the tendency of derivative measures
    # to reward isolated sensor/JPEG noise as useful detail.
    normalized = cv2.GaussianBlur(normalized, (0, 0), sigmaX=0.8, sigmaY=0.8)
    grad_x = cv2.Sobel(normalized, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(normalized, cv2.CV_32F, 0, 1, ksize=3)
    return float(np.mean(grad_x * grad_x + grad_y * grad_y)), contrast_span


def _clip_bbox(
    bbox: Sequence[float], image_width: int, image_height: int,
) -> Tuple[int, int, int, int, bool]:
    if len(bbox) != 4:
        raise ValueError("face bbox must contain x, y, width, height")
    x, y, width, height = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("face bbox values must be finite")
    if width <= 0.0 or height <= 0.0:
        raise ValueError("face bbox width and height must be positive")
    x1 = max(0, min(image_width, int(math.floor(x))))
    y1 = max(0, min(image_height, int(math.floor(y))))
    x2 = max(0, min(image_width, int(math.ceil(x + width))))
    y2 = max(0, min(image_height, int(math.ceil(y + height))))
    clipped = x1 != int(math.floor(x)) or y1 != int(math.floor(y)) or x2 != int(math.ceil(x + width)) or y2 != int(math.ceil(y + height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("face bbox lies outside the image")
    return x1, y1, x2, y2, clipped


def _landmark_points(landmarks: Any, indices: Iterable[int], width: int, height: int) -> List[Tuple[float, float]]:
    values = getattr(landmarks, "landmark", None)
    if values is None:
        return []
    points: List[Tuple[float, float]] = []
    for index in indices:
        if index >= len(values):
            return []
        landmark = values[index]
        x = float(getattr(landmark, "x", float("nan"))) * width
        y = float(getattr(landmark, "y", float("nan"))) * height
        if not math.isfinite(x) or not math.isfinite(y):
            return []
        points.append((x, y))
    return points


def _eye_crop(
    gray: np.ndarray,
    landmarks: Any,
    indices: Sequence[int],
    *,
    padding: float = 0.35,
) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    height, width = gray.shape[:2]
    points = _landmark_points(landmarks, indices, width, height)
    if not points:
        return None, None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    raw_width = max(xs) - min(xs)
    raw_height = max(ys) - min(ys)
    if raw_width <= 0.0 or raw_height <= 0.0:
        return None, None
    pad_x = raw_width * padding
    pad_y = max(raw_height * padding, raw_width * 0.12)
    x1 = max(0, int(math.floor(min(xs) - pad_x)))
    y1 = max(0, int(math.floor(min(ys) - pad_y)))
    x2 = min(width, int(math.ceil(max(xs) + pad_x)))
    y2 = min(height, int(math.ceil(max(ys) + pad_y)))
    if x2 <= x1 or y2 <= y1:
        return None, None
    return gray[y1:y2, x1:x2], (x1, y1, x2 - x1, y2 - y1)


def _observation_id(bbox: Sequence[float]) -> str:
    geometry = ",".join(f"{float(value):.6f}" for value in bbox)
    return "observation-" + hashlib.sha256(geometry.encode("ascii")).hexdigest()[:16]


class FaceQualityAnalyzer:
    """Compute review evidence for already-detected faces.

    Geometry thresholds only decide whether a measurement is too small to be
    trustworthy. They never select or reject an asset.
    """

    def __init__(
        self,
        *,
        min_face_pixels: int = 64,
        min_eye_pixels: int = 10,
        detector_details: Optional[Mapping[str, Any]] = None,
    ):
        if min_face_pixels < 1 or min_eye_pixels < 1:
            raise ValueError("face/eye pixel thresholds must be positive")
        self.min_face_pixels = int(min_face_pixels)
        self.min_eye_pixels = int(min_eye_pixels)
        self.detector_details = dict(detector_details or {})

    def analyze(self, image_bgr: np.ndarray, faces: Sequence[Any]) -> List[FaceQualityEvidence]:
        gray = _gray_float(image_bgr)
        image_height, image_width = gray.shape[:2]
        observations: List[FaceQualityEvidence] = []

        for face in faces:
            uncertainty: List[str] = ["blink_analysis_deferred"]
            x1, y1, x2, y2, clipped = _clip_bbox(face.bbox, image_width, image_height)
            pixel_width = x2 - x1
            pixel_height = y2 - y1
            if clipped:
                uncertainty.append("face_bbox_clipped")
            if min(pixel_width, pixel_height) < self.min_face_pixels:
                uncertainty.append("face_crop_too_small")

            face_crop = gray[y1:y2, x1:x2]
            face_sharpness, face_contrast = _normalized_tenengrad(face_crop, (256, 256))
            if face_contrast <= 1e-6:
                uncertainty.append("face_local_contrast_unavailable")

            left_crop, left_box = _eye_crop(gray, face.landmarks, LEFT_EYE_INDICES)
            right_crop, right_box = _eye_crop(gray, face.landmarks, RIGHT_EYE_INDICES)
            left_sharpness: Optional[float] = None
            right_sharpness: Optional[float] = None
            left_contrast: Optional[float] = None
            right_contrast: Optional[float] = None
            if left_crop is None or left_box is None:
                uncertainty.append("left_eye_landmarks_unavailable")
            elif min(left_crop.shape[:2]) < self.min_eye_pixels:
                uncertainty.append("left_eye_crop_too_small")
            else:
                left_sharpness, left_contrast = _normalized_tenengrad(left_crop, (128, 64))
                if left_contrast <= 1e-6:
                    uncertainty.append("left_eye_local_contrast_unavailable")
            if right_crop is None or right_box is None:
                uncertainty.append("right_eye_landmarks_unavailable")
            elif min(right_crop.shape[:2]) < self.min_eye_pixels:
                uncertainty.append("right_eye_crop_too_small")
            else:
                right_sharpness, right_contrast = _normalized_tenengrad(right_crop, (128, 64))
                if right_contrast <= 1e-6:
                    uncertainty.append("right_eye_local_contrast_unavailable")

            confidence_source = str(getattr(face, "confidence_source", "unknown") or "unknown")
            confidence: Optional[float] = None
            raw_confidence = getattr(face, "confidence", None)
            if confidence_source not in {"unknown", "unavailable", "mediapipe_presence_unavailable"}:
                try:
                    candidate = float(raw_confidence)
                except (TypeError, ValueError):
                    candidate = float("nan")
                if math.isfinite(candidate) and 0.0 <= candidate <= 1.0:
                    confidence = candidate
            if confidence is None:
                uncertainty.append("detector_confidence_unavailable")

            normalized_bbox = (
                x1 / float(image_width),
                y1 / float(image_height),
                pixel_width / float(image_width),
                pixel_height / float(image_height),
            )
            observations.append(FaceQualityEvidence(
                face_id=_observation_id(normalized_bbox),
                bbox=normalized_bbox,
                coverage=(pixel_width * pixel_height) / float(image_width * image_height),
                detector_confidence=confidence,
                detector_confidence_source=confidence_source,
                face_sharpness=face_sharpness,
                left_eye_sharpness=left_sharpness,
                right_eye_sharpness=right_sharpness,
                eyes_open="uncertain",
                measurement_version=FACE_QUALITY_VERSION,
                sharpness_method=SHARPNESS_METHOD,
                measurement_details={
                    "detector": dict(self.detector_details),
                    "minimum_face_pixels": self.min_face_pixels,
                    "minimum_eye_pixels": self.min_eye_pixels,
                    "face_bbox_pixels": [x1, y1, pixel_width, pixel_height],
                    "face_contrast_span": face_contrast,
                    "inter_eye_distance_pixels": float(getattr(face, "ied", 0.0)),
                    "left_eye_bbox_pixels": list(left_box) if left_box else None,
                    "left_eye_contrast_span": left_contrast,
                    "right_eye_bbox_pixels": list(right_box) if right_box else None,
                    "right_eye_contrast_span": right_contrast,
                },
                uncertainty=sorted(set(uncertainty)),
            ))

        return sorted(
            observations,
            key=lambda item: (
                item.bbox[1] if item.bbox else 0.0,
                item.bbox[0] if item.bbox else 0.0,
                item.face_id,
            ),
        )


__all__ = [
    "FACE_QUALITY_VERSION",
    "SHARPNESS_METHOD",
    "LEFT_EYE_INDICES",
    "RIGHT_EYE_INDICES",
    "FaceQualityAnalyzer",
]
