"""Eye Enhancement v0 — sclera brightening and iris saturation/hue adjustment.

Three independent sub-modules:
    1. Sclera brightening: detect sclera (eye whites) via iris-mask subtraction,
       whiten via L-boost in LAB space while preserving chroma.
    2. Iris enhancement: boost saturation and optionally shift hue (for contact lens effects),
       with optional brightness adjustment.
    3. EyeEnhancer: Orchestrator combining both operations per-eye (left/right detected separately).
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple

import cv2
import numpy as np

from .utils import (
    feather_mask,
)
from .eye_visibility import gate_occluded_eye_regions
from .eye_artifact_safety import resolve_eye_scale


def bgr_to_lab_f32(img: np.ndarray) -> np.ndarray:
    """Convert BGR to LAB, handling both uint8 and float32 inputs."""
    if img.dtype == np.float32:
        # Convert float32 [0, 255] to uint8 for cvtColor
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        return lab
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def lab_f32_to_bgr(lab: np.ndarray, is_float: bool) -> np.ndarray:
    """Convert LAB back to BGR, preserving output dtype."""
    clipped = np.clip(lab, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(clipped, cv2.COLOR_LAB2BGR)
    if is_float:
        return bgr.astype(np.float32)
    return bgr


def bgr_to_lch_f32(img: np.ndarray) -> np.ndarray:
    """Convert BGR to LCH via LAB intermediate."""
    lab = bgr_to_lab_f32(img)

    # Extract a, b channels
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0

    # Convert to LCH
    C = np.sqrt(a * a + b * b)
    H = np.arctan2(b, a) * 180.0 / np.pi  # Convert to degrees
    H = np.where(H < 0, H + 360.0, H)  # Normalize to [0, 360)

    lch = np.stack([lab[:, :, 0], C, H], axis=2)
    return lch


def lch_f32_to_bgr(lch: np.ndarray, is_float: bool) -> np.ndarray:
    """Convert LCH back to BGR."""
    L = lch[:, :, 0]
    C = lch[:, :, 1]
    H = lch[:, :, 2]

    # Convert H from degrees to radians
    H_rad = H * np.pi / 180.0

    # Convert LCH back to LAB
    a = C * np.cos(H_rad) + 128.0
    b = C * np.sin(H_rad) + 128.0

    lab = np.stack([L, a, b], axis=2)
    return lab_f32_to_bgr(lab, is_float)


def _detect_iris_circle(
    iris_mask: np.ndarray,
) -> Tuple[float, float, float]:
    """Detect iris as a circle: center and radius.

    Args:
        iris_mask: (H, W) float32 mask where iris pixels are > 0.5

    Returns:
        (cy, cx, radius): iris center y, center x, and estimated radius
    """
    y_indices, x_indices = np.where(iris_mask > 0.5)
    if len(x_indices) < 5:
        return 0.0, 0.0, 0.0

    cy = y_indices.mean()
    cx = x_indices.mean()
    radius = np.sqrt(len(x_indices) / np.pi)

    return cy, cx, radius


class SclerbBrightener:
    """Detect and whiten eye sclera (whites)."""

    def brighten(
        self,
        img_bgr: np.ndarray,
        eye_mask: np.ndarray,
        iris_mask: np.ndarray,
        strength: float = 0.0,
    ) -> np.ndarray:
        """Brighten sclera by boosting L in LAB space.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image in [0, 255].
            eye_mask: (H, W) float32 eye region mask where eye pixels are > 0.
            iris_mask: (H, W) float32 iris mask (used for subtraction).
            strength: 0–1 normalized strength (0 = no-op, 1 = maximum).

        Returns:
            (H, W, 3) result matching input dtype.
        """
        if strength <= 0.0 or eye_mask.max() < 0.01:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        # Create sclera mask: eye region minus iris
        sclera_mask = np.clip(eye_mask - iris_mask, 0, 1).astype(np.float32)
        if sclera_mask.max() < 0.01:
            return img_bgr

        # Feather mask edges to avoid hard transitions
        sclera_mask_feathered = feather_mask(sclera_mask, radius=5)

        # Convert to LAB
        lab = bgr_to_lab_f32(img_bgr)
        L = lab[:, :, 0]

        # Boost L by strength * 30 (capped at 255)
        delta_L = strength * 30.0
        L_new = np.clip(L + delta_L * sclera_mask_feathered, 0, 255)
        lab[:, :, 0] = L_new

        # Convert back to BGR
        result = lab_f32_to_bgr(lab, is_float)
        return result


class IrisEnhancer:
    """Boost iris saturation and optionally shift hue."""

    def enhance(
        self,
        img_bgr: np.ndarray,
        iris_mask: np.ndarray,
        saturate_strength: float = 0.0,
        hue_shift: float = 0.0,
        brightness_strength: float = 0.0,
    ) -> np.ndarray:
        """Enhance iris via saturation/hue/brightness adjustments.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image in [0, 255].
            iris_mask: (H, W) float32 iris mask where iris pixels are > 0.5.
            saturate_strength: 0–1 saturation boost (0 = no change, 1 = max).
            hue_shift: -360 to +360 degrees hue shift (for contact lens effects).
            brightness_strength: 0–1 brightness boost (0 = no change, 1 = max).

        Returns:
            (H, W, 3) result matching input dtype.
        """
        if (saturate_strength <= 0.0 and hue_shift == 0.0 and brightness_strength <= 0.0
            or iris_mask.max() < 0.01):
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        # Feather mask for smooth blending
        iris_mask_feathered = feather_mask(iris_mask, radius=3)

        # Convert to LCH
        lch = bgr_to_lch_f32(img_bgr)
        L = lch[:, :, 0].copy()
        C = lch[:, :, 1].copy()
        H = lch[:, :, 2].copy()

        m = iris_mask_feathered

        # Saturate: C' = C × (1 + strength × 0.5), capped at 128
        if saturate_strength > 0.0:
            saturation_factor = 1.0 + saturate_strength * 0.5
            C_new = np.clip(C * saturation_factor, 0, 128)
            C = C * (1.0 - m) + C_new * m

        # Hue shift: H' = H + hue_shift, wrapped to [0, 360)
        if hue_shift != 0.0:
            H_new = (H + hue_shift) % 360.0
            H = H * (1.0 - m) + H_new * m

        # Brightness: L' = L × (1 + brightness * 0.2), capped at 255
        if brightness_strength > 0.0:
            brightness_factor = 1.0 + brightness_strength * 0.2
            L_new = np.clip(L * brightness_factor, 0, 255)
            L = L * (1.0 - m) + L_new * m

        # Restack LCH
        lch[:, :, 0] = L
        lch[:, :, 1] = C
        lch[:, :, 2] = H

        # Convert back to BGR
        result = lch_f32_to_bgr(lch, is_float)
        return result


class EyeEnhancer:
    """Orchestrator for per-eye sclera and iris enhancement."""

    def __init__(self):
        self.sclera_brightener = SclerbBrightener()
        self.iris_enhancer = IrisEnhancer()

    def enhance(
        self,
        img_bgr: np.ndarray,
        regions: Any,
        sclera_brighten: float = 0.0,
        iris_saturate: float = 0.0,
        iris_hue_shift: float = 0.0,
        iris_brightness: float = 0.0,
        eye_scales: Optional[Mapping[str, Mapping[str, Any]]] = None,
    ) -> np.ndarray:
        """Run full eye enhancement pipeline.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image in [0, 255].
            regions: FaceRegions with left_eye, right_eye, left_iris, right_iris masks.
            sclera_brighten: 0–100 sclera whitening strength.
            iris_saturate: 0–100 iris saturation boost.
            iris_hue_shift: -30 to +30 degrees hue shift (for contact lens effects).
            iris_brightness: 0–100 iris brightness boost.
            eye_scales: Optional source-adaptive per-eye strength evidence from
                the full face pipeline. Direct callers may omit it to retain
                the standalone enhancer's historical behaviour.

        Returns:
            (H, W, 3) result matching input dtype.
        """
        # Normalize strengths to [0, 1]
        sclera_s = sclera_brighten / 100.0
        iris_sat_s = iris_saturate / 100.0
        iris_bright_s = iris_brightness / 100.0

        # Keep the v0 eye controls aligned with the legacy eye stage: a
        # landmark-only iris under hair/wig is not safe to retouch.  Idempotent
        # re-application of the _process_face_core gate; protects direct callers.
        regions = gate_occluded_eye_regions(regions, img_bgr=img_bgr)

        left_scale = resolve_eye_scale(eye_scales, "left")
        right_scale = resolve_eye_scale(eye_scales, "right")
        result = img_bgr.copy()

        # Sclera brightening per eye
        if sclera_s > 0:
            if (regions.left_eye is not None and regions.left_eye.max() > 0.01
                and regions.left_iris is not None and regions.left_iris.max() > 0.01):
                result = self.sclera_brightener.brighten(
                    result,
                    regions.left_eye,
                    regions.left_iris,
                    sclera_s * left_scale,
                )

            if (regions.right_eye is not None and regions.right_eye.max() > 0.01
                and regions.right_iris is not None and regions.right_iris.max() > 0.01):
                result = self.sclera_brightener.brighten(
                    result,
                    regions.right_eye,
                    regions.right_iris,
                    sclera_s * right_scale,
                )

        # Iris enhancement per eye
        if iris_sat_s > 0 or iris_hue_shift != 0 or iris_bright_s > 0:
            if regions.left_iris is not None and regions.left_iris.max() > 0.01:
                result = self.iris_enhancer.enhance(
                    result, regions.left_iris,
                    saturate_strength=iris_sat_s * left_scale,
                    hue_shift=iris_hue_shift * left_scale,
                    brightness_strength=iris_bright_s * left_scale,
                )

            if regions.right_iris is not None and regions.right_iris.max() > 0.01:
                result = self.iris_enhancer.enhance(
                    result, regions.right_iris,
                    saturate_strength=iris_sat_s * right_scale,
                    hue_shift=iris_hue_shift * right_scale,
                    brightness_strength=iris_bright_s * right_scale,
                )

        return result
