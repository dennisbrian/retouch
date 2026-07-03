"""Under-eye dark circle repair module.

Goes beyond simple brightening by detecting the *colour* of dark circles
(purple, blue, or brown) and applying targeted colour + brightness correction.
"""

from __future__ import annotations

import cv2
import numpy as np

from .parsing import FaceRegions
from .utils import apply_u8_op_float, blend_masked


class UnderEyeRepairer:
    """Detect and repair under-eye dark circles."""

    def repair(
        self,
        img_bgr: np.ndarray,
        regions: FaceRegions,
        strength: int = 40,
    ) -> np.ndarray:
        """Repair dark circles under both eyes.

        Args:
            img_bgr: (H, W, 3) uint8.
            regions: FaceRegions from parser.
            strength: 0–100.

        Returns:
            (H, W, 3) uint8 result.
        """
        if strength <= 0:
            return img_bgr

        if img_bgr.dtype == np.float32:
            # E1 delta adapter (see apply_u8_op_float).
            return apply_u8_op_float(img_bgr, self.repair, regions, strength)

        s = strength / 100.0
        result = img_bgr.copy()

        for mask in (regions.left_under_eye, regions.right_under_eye):
            if mask is not None and mask.max() > 0.01:
                result = self._repair_region(result, mask, s)

        return result

    def _repair_region(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Repair dark circles in one under-eye region.

        Strategy:
            1. Analyse LAB + HSV to detect dark circle colour cast.
            2. Apply *colour correction* to neutralise the cast.
            3. Apply *brightness correction* to lift darkness.
            These are independent — both are needed for natural results.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

        # ---- Detect dark circle intensity ----
        # Dark circles: low L (dark), possibly shifted a or b
        l_channel = lab[:, :, 0]
        a_channel = lab[:, :, 1]  # green-red axis
        b_channel = lab[:, :, 2]  # blue-yellow axis

        # Get reference skin brightness from surrounding skin
        # (use the mask periphery as reference)
        dilated = cv2.dilate(
            (mask > 0.3).astype(np.uint8), 
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)),
            iterations=2
        ).astype(np.float32)
        surround = np.clip(dilated - mask, 0, 1)

        surround_pixels = l_channel[surround > 0.3]
        if len(surround_pixels) < 10:
            return img_bgr

        ref_l = np.median(surround_pixels)
        ref_a = np.median(a_channel[surround > 0.3])
        ref_b = np.median(b_channel[surround > 0.3])

        # ---- Colour correction ----
        # Neutralise colour cast: push a and b towards surrounding skin values
        a_correction = (ref_a - a_channel) * mask * strength * 0.5
        b_correction = (ref_b - b_channel) * mask * strength * 0.5
        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_correction, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_correction, 0, 255)

        # ---- Brightness correction ----
        # Lift dark pixels towards reference brightness
        darkness = np.clip(ref_l - l_channel, 0, 60)  # cap the lift
        l_lift = darkness * mask * strength * 0.6
        lab[:, :, 0] = np.clip(lab[:, :, 0] + l_lift, 0, 255)

        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result
