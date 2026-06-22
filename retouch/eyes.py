"""Eye enhancement module — whites, iris, and catchlight processing.

Three independent sub-modules:
    1. Eye whites: remove redness/yellowness via LAB correction.
    2. Iris: boost clarity, micro-contrast, saturation, and sharpness.
    3. Catchlight: detect and amplify existing highlights (never create fakes).
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from .utils import blend_masked, feather_mask


class EyeEnhancer:
    """Professional eye enhancement pipeline."""

    def enhance(
        self,
        img_bgr: np.ndarray,
        regions: Any,
        strength: int = 40,
        catchlight_strength: Optional[int] = None,
    ) -> np.ndarray:
        """Run the full eye enhancement pipeline.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            regions: FaceRegions from the parser.
            strength: 0–100 overall intensity.
            catchlight_strength: 0–100 catchlight-specific intensity. If None,
                falls back to ``strength`` (backward compatible).

        Returns:
            (H, W, 3) uint8 result.
        """
        if strength <= 0:
            return img_bgr

        s = strength / 100.0
        result = img_bgr.copy()

        # Eye whites — combine both eyes
        whites_mask_l = np.clip(regions.left_eye - regions.left_iris, 0, 1)
        whites_mask_r = np.clip(regions.right_eye - regions.right_iris, 0, 1)
        whites_mask = np.clip(whites_mask_l + whites_mask_r, 0, 1)
        result = self._enhance_whites(result, whites_mask, s)

        # Iris — sculpt each eye separately
        if regions.left_iris is not None and regions.left_iris.max() > 0.01:
            result = self._sculpt_iris(result, regions.left_iris, s)
        if regions.right_iris is not None and regions.right_iris.max() > 0.01:
            result = self._sculpt_iris(result, regions.right_iris, s)

        # Catchlights — detect and amplify existing highlights
        iris_mask = np.clip(regions.left_iris + regions.right_iris, 0, 1)
        cl_s = catchlight_strength if catchlight_strength is not None else strength
        result = self._enhance_catchlights(result, iris_mask, cl_s / 100.0)

        return result

    def _enhance_whites(
        self,
        img_bgr: np.ndarray,
        whites_mask: Optional[np.ndarray],
        strength: float,
    ) -> np.ndarray:
        """Remove redness and yellowness from eye whites.

        Works in LAB space:
            • Reduce 'a' channel (green-red) to remove redness.
            • Reduce 'b' channel (blue-yellow) to remove yellowness.
            • Slightly lift 'L' for brighter whites.
        """
        if whites_mask is None or whites_mask.max() < 0.01:
            return img_bgr

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Only apply whitening to pixels that are relatively bright (L > 100)
        # to prevent eyelashes/eyeliner and dark shadow areas from turning gray/dusty.
        bright_sclera = (lab[:, :, 0] > 100.0).astype(np.float32)
        m = whites_mask * strength * bright_sclera

        # Reduce redness (a channel > 128 means more red)
        lab[:, :, 1] = lab[:, :, 1] - m * np.clip(lab[:, :, 1] - 128, 0, 30) * 0.4

        # Reduce yellowness (b channel > 128 means more yellow)
        lab[:, :, 2] = lab[:, :, 2] - m * np.clip(lab[:, :, 2] - 128, 0, 30) * 0.4

        # Subtle brightness lift
        lab[:, :, 0] = np.clip(lab[:, :, 0] + m * 6, 0, 255)

        result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result

    def _sculpt_iris(
        self,
        img_bgr: np.ndarray,
        iris_mask: Optional[np.ndarray],
        strength: float,
    ) -> np.ndarray:
        """Perform 3D iris sculpting: darken pupil & limbal ring, brighten iris body,
        and boost micro-contrast and saturation.
        """
        if iris_mask is None or iris_mask.max() < 0.01:
            return img_bgr

        # Find center and radius of this iris
        y_indices, x_indices = np.where(iris_mask > 0.5)
        if len(x_indices) < 5:
            return img_bgr

        cy, cx = y_indices.mean(), x_indices.mean()
        r = max(np.sqrt(len(x_indices) / np.pi), 3.0)

        # We will work in a local crop around the iris for high performance
        h, w = img_bgr.shape[:2]
        pad = int(r * 1.5)
        x1 = max(int(cx - pad), 0)
        y1 = max(int(cy - pad), 0)
        x2 = min(int(cx + pad), w)
        y2 = min(int(cy + pad), h)

        if (x2 - x1) < 4 or (y2 - y1) < 4:
            return img_bgr

        crop = img_bgr[y1:y2, x1:x2].copy()
        crop_mask = iris_mask[y1:y2, x1:x2]

        # Coordinates relative to center
        yy, xx = np.mgrid[y1:y2, x1:x2]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r

        # 3D Sculpting masks
        # 1. Pupil mask (center dark area)
        pupil_mask = np.clip(1.0 - dist / 0.35, 0, 1) * crop_mask
        
        # 2. Iris body mask (glowing middle ring)
        # Peak at d=0.55
        body_mask = np.clip(1.0 - np.abs(dist - 0.55) / 0.25, 0, 1) * crop_mask

        # 3. Limbal ring mask (dark outer edge)
        # Peak at d=0.9
        limbal_mask = np.clip(1.0 - np.abs(dist - 0.9) / 0.2, 0, 1) * crop_mask

        # Apply edits in LAB and HSV spaces on the crop
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Pupil: darken
        lab[:, :, 0] = np.clip(lab[:, :, 0] - pupil_mask * 45.0 * strength, 0, 255)

        # Iris body: brighten
        lab[:, :, 0] = np.clip(lab[:, :, 0] + body_mask * 25.0 * strength, 0, 255)

        # Limbal ring: darken
        lab[:, :, 0] = np.clip(lab[:, :, 0] - limbal_mask * 30.0 * strength, 0, 255)

        crop_edited = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # HSV saturation boost on the iris body
        hsv = cv2.cvtColor(crop_edited, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] + body_mask * hsv[:, :, 1] * 0.45 * strength, 0, 255)
        crop_edited = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        # Sharpness via unsharp mask on the crop
        blurred = cv2.GaussianBlur(crop_edited, (3, 3), 1.0)
        crop_sharp = cv2.addWeighted(crop_edited, 1.0 + strength * 0.5, blurred, -strength * 0.5, 0)

        # Blend the edited crop back into the image
        img_bgr_out = img_bgr.copy()
        m = crop_mask[:, :, np.newaxis]
        img_bgr_out[y1:y2, x1:x2] = np.clip(
            crop.astype(np.float32) * (1.0 - m) + crop_sharp.astype(np.float32) * m,
            0, 255
        ).astype(np.uint8)

        return img_bgr_out

    def _enhance_catchlights(
        self,
        img_bgr: np.ndarray,
        iris_mask: Optional[np.ndarray],
        strength: float,
    ) -> np.ndarray:
        """Detect and amplify existing catchlights. Never create fakes.

        Catchlights are small, bright specular highlights in the iris.
        We find them via thresholding in the iris region and amplify only
        existing ones.
        """
        if iris_mask is None or iris_mask.max() < 0.01:
            return img_bgr

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Only look within iris
        iris_gray = gray * iris_mask

        # Detect existing bright spots (catchlights)
        # Use adaptive threshold relative to iris brightness
        iris_pixels = gray[iris_mask > 0.5]
        if len(iris_pixels) == 0:
            return img_bgr

        iris_mean = iris_pixels.mean()
        iris_std = max(iris_pixels.std(), 1.0)
        threshold = iris_mean + iris_std * 1.2

        catchlight_mask = ((iris_gray > threshold) & (iris_mask > 0.3)).astype(np.float32)

        if catchlight_mask.sum() < 2:
            return img_bgr

        # Feather the catchlight mask
        catchlight_mask = feather_mask(catchlight_mask, radius=2)

        # Amplify: brighten catchlight areas using a safe soft-clipping formula
        # to subtly enhance the catchlights without blowing them out (max 0.25 boost coefficient)
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]
        
        l_boost = (255.0 - l_chan) * catchlight_mask * 0.25 * strength
        lab[:, :, 0] = np.clip(l_chan + l_boost, 0, 255)
        
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result
