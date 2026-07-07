"""Teeth whitening via LAB correction.

Uses MediaPipe mouth landmarks to isolate teeth.  Detects teeth pixels by
brightness + low-saturation within the mouth-interior region.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .utils import blend_masked, feather_mask, bgr_f32_to_lab_f32, lab_f32_to_bgr_f32


def _to_lab(img_bgr: np.ndarray, is_float: bool) -> np.ndarray:
    if is_float:
        return bgr_f32_to_lab_f32(img_bgr)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab: np.ndarray, is_float: bool) -> np.ndarray:
    clipped = np.clip(lab, 0, 255)
    if is_float:
        return lab_f32_to_bgr_f32(clipped)
    return cv2.cvtColor(clipped.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _bgr_to_hsv_uint8_scale(img_bgr: np.ndarray, is_float: bool) -> np.ndarray:
    if is_float:
        hsv = cv2.cvtColor(
            np.clip(img_bgr, 0.0, 255.0) * (1.0 / 255.0), cv2.COLOR_BGR2HSV
        ).astype(np.float32)
        hsv[:, :, 1] *= 255.0
        hsv[:, :, 2] *= 255.0
        return hsv
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)


class TeethWhitener:
    """Detect and whiten visible teeth."""

    def whiten(
        self,
        img_bgr: np.ndarray,
        mouth_interior_mask: Optional[np.ndarray],
        strength: int = 30,
    ) -> np.ndarray:
        """Whiten teeth within the mouth interior.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR.
            mouth_interior_mask: (H, W) float mask 0–1 (inner lip region).
            strength: 0–100.

        Returns:
            (H, W, 3) same dtype as input.
        """
        if strength <= 0:
            return img_bgr
        if mouth_interior_mask is None or mouth_interior_mask.max() < 0.01:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        s = strength / 100.0
        teeth_mask = self._detect_teeth(img_bgr, mouth_interior_mask)

        if teeth_mask.max() < 0.01:
            return img_bgr

        lab = _to_lab(img_bgr, is_float)

        m = teeth_mask * s

        # Lighten teeth
        lab[:, :, 0] = np.clip(lab[:, :, 0] + m * 15, 0, 255)

        # Reduce yellowness (b channel > 128 = yellow)
        yellow_excess = np.clip(lab[:, :, 2] - 128, 0, 40)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - m * yellow_excess * 0.5, 0, 255)

        # Slight reduction in redness
        red_excess = np.clip(lab[:, :, 1] - 128, 0, 20)
        lab[:, :, 1] = np.clip(lab[:, :, 1] - m * red_excess * 0.3, 0, 255)

        return _from_lab(lab, is_float)

    def _detect_teeth(self, img_bgr: np.ndarray, mouth_mask: np.ndarray) -> np.ndarray:
        """Detect teeth pixels within mouth interior.

        Teeth characteristics in LAB/HSV:
            • High luminance (L > median of mouth region).
            • Low saturation (not pink/red like gums or tongue).
            • Not very dark (not mouth void).

        Returns:
            (H, W) float mask 0–1.
        """
        is_float = img_bgr.dtype == np.float32
        lab = _to_lab(img_bgr, is_float)
        hsv = _bgr_to_hsv_uint8_scale(img_bgr, is_float)

        l_channel = lab[:, :, 0]
        s_channel = hsv[:, :, 1]

        # Get stats within mouth region
        mouth_pixels_l = l_channel[mouth_mask > 0.3]
        mouth_pixels_s = s_channel[mouth_mask > 0.3]

        if len(mouth_pixels_l) < 20:
            return np.zeros_like(mouth_mask)

        l_median = np.median(mouth_pixels_l)
        s_median = np.median(mouth_pixels_s)

        # Teeth: brighter than median, less saturated than median
        teeth_candidate = (
            (l_channel > l_median * 0.9) &  # reasonably bright
            (s_channel < s_median * 1.2) &   # less saturated
            (l_channel > 80) &               # not dark void
            (mouth_mask > 0.3)
        ).astype(np.float32)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        teeth_candidate = cv2.morphologyEx(
            (teeth_candidate * 255).astype(np.uint8),
            cv2.MORPH_OPEN, kernel
        ).astype(np.float32) / 255.0

        # Soft feather
        teeth_candidate = feather_mask(teeth_candidate, radius=3)

        return teeth_candidate * mouth_mask
