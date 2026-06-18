"""Teeth whitening via LAB correction.

Uses MediaPipe mouth landmarks to isolate teeth.  Detects teeth pixels by
brightness + low-saturation within the mouth-interior region.
"""

import cv2
import numpy as np

from .utils import blend_masked, feather_mask


class TeethWhitener:
    """Detect and whiten visible teeth."""

    def whiten(self, img_bgr, mouth_interior_mask, strength=30):
        """Whiten teeth within the mouth interior.

        Args:
            img_bgr: (H, W, 3) uint8.
            mouth_interior_mask: (H, W) float mask 0–1 (inner lip region).
            strength: 0–100.

        Returns:
            (H, W, 3) uint8 result.
        """
        if strength <= 0:
            return img_bgr
        if mouth_interior_mask is None or mouth_interior_mask.max() < 0.01:
            return img_bgr

        s = strength / 100.0
        teeth_mask = self._detect_teeth(img_bgr, mouth_interior_mask)

        if teeth_mask.max() < 0.01:
            return img_bgr

        # LAB correction: increase L, decrease b (yellow)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        m = teeth_mask * s

        # Lighten teeth
        lab[:, :, 0] = np.clip(lab[:, :, 0] + m * 15, 0, 255)

        # Reduce yellowness (b channel > 128 = yellow)
        yellow_excess = np.clip(lab[:, :, 2] - 128, 0, 40)
        lab[:, :, 2] = np.clip(lab[:, :, 2] - m * yellow_excess * 0.5, 0, 255)

        # Slight reduction in redness
        red_excess = np.clip(lab[:, :, 1] - 128, 0, 20)
        lab[:, :, 1] = np.clip(lab[:, :, 1] - m * red_excess * 0.3, 0, 255)

        whitened = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return whitened

    def _detect_teeth(self, img_bgr, mouth_mask):
        """Detect teeth pixels within mouth interior.

        Teeth characteristics in LAB/HSV:
            • High luminance (L > median of mouth region).
            • Low saturation (not pink/red like gums or tongue).
            • Not very dark (not mouth void).

        Returns:
            (H, W) float mask 0–1.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        l_channel = lab[:, :, 0].astype(np.float32)
        s_channel = hsv[:, :, 1].astype(np.float32)

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
