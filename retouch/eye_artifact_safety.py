"""Source-adaptive strength guard for eye retouching.

Eye masks can be geometrically valid while still being too small to support
iris sculpting, whitening, catchlight synthesis, and sharpening at the
requested strength.  Highly saturated source irises (including coloured
contacts) also have little safe chroma headroom.  Applying the legacy and v0
eye stacks at full strength in those cases can produce a painted or synthetic
eye even though the eye is genuinely visible.

This module does not classify contact lenses and does not change eye colour.
It derives a conservative per-eye scale from two source facts only:

* the equivalent native-pixel radius of the parsed iris mask; and
* the existing 75th-percentile HSV saturation inside that mask.

Large, ordinary-chroma irises receive a neutral scale of 1.0.  Small or
already-saturated irises smoothly back off toward the untouched source.  Bad
or missing evidence fails open at 1.0; the separate visibility gate remains
responsible for fully suppressing closed or occluded eyes.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import cv2
import numpy as np


# Calibrated against native-resolution parser masks from the FF47 convention
# set.  The near-profile failure (DSCF0892) has r~=8.5 px, while dependable
# close portraits are generally r>=20 px.  Smooth interpolation avoids a hard
# resolution cliff.
_MIN_IRIS_RADIUS_PX = 5.0
_FULL_IRIS_RADIUS_PX = 20.0

# Saturation is measured in OpenCV's uint8 HSV convention [0, 255].  Below 96
# there is normal headroom.  Above 168 the source already carries strong iris
# colour, so added sculpt/saturation/whitening/sharpening is capped at 35%.
_CHROMA_HEADROOM_START = 96.0
_CHROMA_HEADROOM_END = 168.0
_MIN_CHROMA_SCALE = 0.35


def _smoothstep01(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _geometry_scale(radius: float) -> float:
    span = _FULL_IRIS_RADIUS_PX - _MIN_IRIS_RADIUS_PX
    if span <= 0.0:
        return 1.0
    position = (float(radius) - _MIN_IRIS_RADIUS_PX) / span
    return _smoothstep01(position)


def _chroma_scale(saturation_p75: float) -> float:
    span = _CHROMA_HEADROOM_END - _CHROMA_HEADROOM_START
    if span <= 0.0:
        return 1.0
    position = (float(saturation_p75) - _CHROMA_HEADROOM_START) / span
    risk = _smoothstep01(position)
    return 1.0 - (1.0 - _MIN_CHROMA_SCALE) * risk


def _as_mask(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        mask = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if mask.ndim != 2 or mask.size == 0:
        return None
    return np.clip(
        np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0),
        0.0,
        1.0,
    )


def _neutral_evidence(reason: str) -> Dict[str, Any]:
    return {
        "scale": 1.0,
        "geometry_scale": 1.0,
        "chroma_scale": 1.0,
        "iris_radius_px": None,
        "saturation_p75": None,
        "reason": reason,
    }


def assess_eye_artifact_scales(
    img_bgr: Optional[np.ndarray],
    regions: Any,
) -> Dict[str, Dict[str, Any]]:
    """Return source-adaptive strength evidence for ``left`` and ``right``.

    The returned ``scale`` is safe to multiply into every cosmetic eye stage
    and the selective eye-sharpen mask.  The function is analysis-only and
    never mutates ``img_bgr`` or ``regions``.
    """
    image_u8: Optional[np.ndarray] = None
    if isinstance(img_bgr, np.ndarray) and img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        try:
            image_u8 = np.clip(img_bgr, 0.0, 255.0).astype(np.uint8)
        except (TypeError, ValueError):
            image_u8 = None

    hsv: Optional[np.ndarray] = None
    decisions: Dict[str, Dict[str, Any]] = {}

    for side in ("left", "right"):
        iris = _as_mask(getattr(regions, f"{side}_iris", None))
        if iris is None:
            decisions[side] = _neutral_evidence("iris_mask_unavailable")
            continue

        support = iris > 0.25
        support_pixels = int(np.count_nonzero(support))
        if support_pixels < 5:
            decisions[side] = _neutral_evidence("iris_support_unavailable")
            continue

        radius = float(np.sqrt(support_pixels / np.pi))
        geometry = _geometry_scale(radius)
        chroma = 1.0
        saturation_p75: Optional[float] = None

        if image_u8 is not None and image_u8.shape[:2] == iris.shape:
            if hsv is None:
                hsv = cv2.cvtColor(image_u8, cv2.COLOR_BGR2HSV)
            saturation = hsv[:, :, 1][support]
            if saturation.size:
                saturation_p75 = float(np.percentile(saturation, 75.0))
                chroma = _chroma_scale(saturation_p75)

        scale = float(np.clip(min(geometry, chroma), 0.0, 1.0))
        reasons = []
        if geometry < 0.999:
            reasons.append("small_iris")
        if chroma < 0.999:
            reasons.append("limited_chroma_headroom")

        decisions[side] = {
            "scale": scale,
            "geometry_scale": float(geometry),
            "chroma_scale": float(chroma),
            "iris_radius_px": radius,
            "saturation_p75": saturation_p75,
            "reason": "+".join(reasons) if reasons else "full_headroom",
        }

    return decisions


def resolve_eye_scale(
    decisions: Optional[Mapping[str, Mapping[str, Any]]],
    side: str,
) -> float:
    """Read one clamped scale from optional evidence, defaulting to 1.0."""
    if decisions is None:
        return 1.0
    try:
        value = float(decisions.get(side, {}).get("scale", 1.0))
    except (AttributeError, TypeError, ValueError):
        return 1.0
    if not np.isfinite(value):
        return 1.0
    return float(np.clip(value, 0.0, 1.0))


__all__ = ["assess_eye_artifact_scales", "resolve_eye_scale"]
