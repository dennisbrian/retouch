"""Shared, confidence-rated key-light estimation for face-local operations.

The vector uses image coordinates: ``(1, 0)`` points right and ``(0, -1)``
points up.  It describes the image-plane direction toward the brighter/key-lit
side of the face, not a 3D light position.  Consumers must respect
``confidence``; an ``unknown`` result is intentionally safer than a guessed
direction for catchlight or lip-highlight synthesis.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import cv2
import numpy as np

__all__ = ["LightDirection", "estimate_light_direction"]


@dataclass(frozen=True)
class LightDirection:
    """One face's estimated key-light direction in image coordinates.

    ``direction`` is a unit vector for known estimates and ``(0, 0)`` for an
    unknown estimate.  ``source`` is useful for downstream conservative gates:
    paired catchlights are the strongest evidence, while broad shading is only
    a fallback.
    """

    direction: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    source: str = "unknown"

    @property
    def is_known(self) -> bool:
        return self.confidence > 0.0 and self.source != "unknown"


def _gray_f32(img_bgr: np.ndarray) -> np.ndarray:
    """Return BGR input luminance as float32 in [0, 1]."""
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("estimate_light_direction expects a 3-channel BGR image")
    img = img_bgr.astype(np.float32, copy=False)
    if img.size and float(np.nanmax(img)) > 1.5:
        img = img * (1.0 / 255.0)
    return cv2.cvtColor(np.clip(img, 0.0, 1.0), cv2.COLOR_BGR2GRAY)


def _unit(vector: np.ndarray) -> Optional[Tuple[float, float]]:
    length = float(np.linalg.norm(vector))
    if not np.isfinite(length) or length < 1e-6:
        return None
    return float(vector[0] / length), float(vector[1] / length)


def _iris_catchlight_direction(
    gray: np.ndarray, iris_mask: Optional[np.ndarray]
) -> Optional[Tuple[np.ndarray, float]]:
    """Return a normalized iris-to-catchlight vector and evidence strength."""
    if iris_mask is None or iris_mask.shape != gray.shape:
        return None
    mask = iris_mask.astype(np.float32, copy=False) > 0.35
    count = int(mask.sum())
    if count < 20:
        return None

    values = gray[mask]
    base = float(np.percentile(values, 55.0))
    peak = float(np.percentile(values, 99.0))
    contrast = peak - base
    # A flat bright iris is not a catchlight; it carries no light direction.
    if contrast < 0.06:
        return None

    # The bright patch can cover more than the top 4% of a small iris.  Use a
    # contrast-relative floor as well as the percentile so that case remains a
    # compact candidate instead of the whole iris.
    threshold = max(float(np.percentile(values, 96.0)), base + contrast * 0.55)
    candidate = mask & (gray >= threshold)
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate.astype(np.uint8), connectivity=8
    )
    best = None
    for label in range(1, labels_count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 1 or area > max(3, int(count * 0.12)):
            continue
        component = labels == label
        excess = np.maximum(gray[component] - base, 0.0)
        score = float(excess.sum())
        if best is None or score > best[0]:
            best = score, component, excess
    if best is None:
        return None

    _, component, excess = best
    yy, xx = np.nonzero(component)
    weights = excess + 1e-6
    catch_x = float(np.average(xx, weights=weights))
    catch_y = float(np.average(yy, weights=weights))

    iy, ix = np.nonzero(mask)
    iris_x = float(ix.mean())
    iris_y = float(iy.mean())
    radius = max(float(np.sqrt(count / np.pi)), 1.0)
    offset = np.array([(catch_x - iris_x) / radius, (catch_y - iris_y) / radius])
    direction = _unit(offset)
    if direction is None:
        return None

    # Contrast and a compact highlight both increase confidence.  A highlight
    # that fills a large section of the iris is deliberately down-weighted.
    compactness = 1.0 - min(1.0, float(component.sum()) / max(count * 0.08, 1.0))
    evidence = min(1.0, contrast / 0.30) * (0.55 + 0.45 * compactness)
    return np.array(direction, dtype=np.float32), float(evidence)


def _catchlight_estimate(gray: np.ndarray, regions: Any) -> Optional[LightDirection]:
    observations = []
    for name in ("left_iris", "right_iris"):
        detected = _iris_catchlight_direction(gray, getattr(regions, name, None))
        if detected is not None:
            observations.append(detected)
    if not observations:
        return None

    vectors = np.stack([item[0] for item in observations])
    weights = np.array([item[1] for item in observations], dtype=np.float32)
    combined = _unit(np.average(vectors, axis=0, weights=weights))
    if combined is None:
        return None

    if len(observations) == 1:
        return LightDirection(combined, float(weights[0] * 0.45), "catchlight_single")

    agreement = max(0.0, float(np.dot(vectors[0], vectors[1])))
    # Disagreeing eyes must not produce a seemingly reliable shared direction.
    if agreement < 0.35:
        return None
    confidence = float(np.mean(weights) * (0.40 + 0.60 * agreement))
    return LightDirection(combined, confidence, "catchlights")


def _shading_estimate(
    gray: np.ndarray, regions: Any, face_width: Optional[float]
) -> LightDirection:
    """Fit a low-frequency illumination plane within the skin mask."""
    h, w = gray.shape
    skin = getattr(regions, "skin", None) if regions is not None else None
    if skin is not None and skin.shape == gray.shape:
        mask = skin.astype(np.float32, copy=False) > 0.20
    else:
        # Avoid using arbitrary background when the parser did not make skin.
        yy, xx = np.ogrid[:h, :w]
        mask = ((xx - (w - 1) / 2.0) / max(w * 0.35, 1.0)) ** 2 + (
            (yy - (h - 1) / 2.0) / max(h * 0.42, 1.0)
        ) ** 2 <= 1.0
    if int(mask.sum()) < 100:
        return LightDirection()

    sigma = max(2.0, min(h, w) * 0.06)
    if face_width is not None:
        sigma = max(sigma, float(face_width) * 0.10)
    low = cv2.GaussianBlur(gray, (0, 0), sigma)
    yy, xx = np.nonzero(mask)
    x = (xx.astype(np.float32) - (w - 1) / 2.0) / max(w / 2.0, 1.0)
    y = (yy.astype(np.float32) - (h - 1) / 2.0) / max(h / 2.0, 1.0)
    design = np.column_stack((x, y, np.ones_like(x)))
    coeff, _, _, _ = np.linalg.lstsq(design, low[mask], rcond=None)
    direction = _unit(coeff[:2])
    if direction is None:
        return LightDirection()

    fitted = design @ coeff
    residual = float(np.std(low[mask] - fitted))
    amplitude = float(np.linalg.norm(coeff[:2]))
    confidence = min(0.45, amplitude / max(residual * 5.0, 0.035)) * 0.45
    if confidence < 0.05:
        return LightDirection()
    return LightDirection(direction, float(confidence), "shading")


def estimate_light_direction(
    img_bgr: np.ndarray,
    regions: Any = None,
    face_width: Optional[float] = None,
) -> LightDirection:
    """Estimate a face-local key-light direction without changing the image.

    Paired corneal highlights are preferred because they provide direct optical
    evidence.  When they are absent or disagree, a broad shading-plane fit in
    the skin region supplies a deliberately lower-confidence fallback.
    """
    gray = _gray_f32(img_bgr)
    if regions is not None:
        catchlights = _catchlight_estimate(gray, regions)
        if catchlights is not None:
            return catchlights
    return _shading_estimate(gray, regions, face_width)
