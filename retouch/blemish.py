"""AI blemish removal via local contrast anomaly detection + inpainting.

Detects acne, small spots, and temporary blemishes by finding local contrast
anomalies in the skin region, then removes them with OpenCV inpainting.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import estimate_face_width


def inpaint_and_blend(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    inpaint_radius: int,
    flags: int = cv2.INPAINT_TELEA,
    blend_ksize: int = 7,
) -> np.ndarray:
    """Inpaint masked regions and softly blend the result.

    Shared helper used by BlemishRemover and heal_region.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        mask: (H, W) uint8 binary mask (255 = region to inpaint).
        inpaint_radius: Radius for cv2.inpaint.
        flags: cv2.INPAINT_TELEA or cv2.INPAINT_NS.
        blend_ksize: Gaussian blur kernel size for soft-edge blending (must be odd).

    Returns:
        (H, W, 3) uint8 result with inpainted regions softly blended.
    """
    if mask.sum() == 0:
        return img_bgr

    blend_ksize = max(blend_ksize, 3) | 1

    inpainted = cv2.inpaint(img_bgr, mask, inpaintRadius=inpaint_radius, flags=flags)

    blend_mask = cv2.GaussianBlur(
        mask.astype(np.float32) / 255.0, (blend_ksize, blend_ksize), 0
    )
    blend_mask = blend_mask[:, :, np.newaxis]
    result = (
        img_bgr.astype(np.float32) * (1.0 - blend_mask)
        + inpainted.astype(np.float32) * blend_mask
    )
    return np.clip(result, 0, 255).astype(np.uint8)


class BlemishRemover:
    """Detect and remove skin blemishes via inpainting."""

    def remove(
        self,
        img_bgr: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: int = 50,
    ) -> np.ndarray:
        """Detect blemishes and inpaint them.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            skin_mask: (H, W) float mask 0–1.
            strength: 0–100. Higher = more aggressive detection.

        Returns:
            (H, W, 3) uint8 result.
        """
        if strength <= 0:
            return img_bgr

        # Estimate face width from skin mask
        face_width = estimate_face_width(skin_mask=skin_mask, img_shape=img_bgr.shape[:2])

        s = strength / 100.0
        blemish_mask = self._detect(img_bgr, skin_mask, s, face_width)

        if blemish_mask.sum() == 0:
            return img_bgr

        scale = face_width / 500.0
        inpaint_r = max(int(3 * scale), 2)
        blend_k = max(int(7 * scale), 3) | 1
        return inpaint_and_blend(img_bgr, blemish_mask, inpaint_r, cv2.INPAINT_TELEA, blend_k)

    def _detect(
        self,
        img_bgr: np.ndarray,
        skin_mask: np.ndarray,
        sensitivity: float,
        face_width: float,
    ) -> np.ndarray:
        """Detect blemishes via local contrast anomalies.

        Strategy:
            1. Convert to grayscale.
            2. Compare each pixel to its local neighbourhood mean.
            3. Pixels significantly darker than surroundings (within skin)
               are potential blemishes.
            4. Filter by size: only small spots (not shadows or features).

        Returns:
            uint8 binary mask of blemish pixels (255 = blemish).
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Scale parameters based on face width
        scale = face_width / 500.0

        # Local mean (adaptive neighbourhood)
        ksize = max(int(31 * scale), 9) | 1
        local_mean = cv2.GaussianBlur(gray, (ksize, ksize), 0)

        # Deviation from local mean
        deviation = local_mean - gray  # positive = darker than surroundings

        # Threshold: pixels that are notably darker
        threshold = 8.0 + (1.0 - sensitivity) * 15.0  # lower sensitivity = higher threshold
        blemish_candidates = (deviation > threshold).astype(np.uint8) * 255

        # Restrict to skin region
        skin_binary = (skin_mask > 0.3).astype(np.uint8) * 255
        blemish_candidates = cv2.bitwise_and(blemish_candidates, skin_binary)

        # Also check for reddish spots (common for acne)
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        redness = np.zeros_like(gray, dtype=np.uint8)
        # Red hue range in HSV
        red_low1 = cv2.inRange(hsv, (0, 80, 50), (10, 255, 255))
        red_low2 = cv2.inRange(hsv, (170, 80, 50), (180, 255, 255))
        red_combined = cv2.bitwise_or(red_low1, red_low2)
        # Only count red that's also a contrast anomaly (dark spot)
        red_blemish = cv2.bitwise_and(red_combined, blemish_candidates)
        blemish_candidates = cv2.bitwise_or(blemish_candidates, red_blemish)

        # Size filter: remove large regions (those are shadows, not blemishes)
        # and very tiny noise
        k_open = max(int(2 * scale), 1)
        k_close = max(int(3 * scale), 2)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        blemish_candidates = cv2.morphologyEx(blemish_candidates, cv2.MORPH_OPEN, kernel_open)
        blemish_candidates = cv2.morphologyEx(blemish_candidates, cv2.MORPH_CLOSE, kernel_close)

        # Connected components: reject regions that are too large
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            blemish_candidates, connectivity=8
        )
        
        # Adaptive area thresholds
        max_area = int((500 * sensitivity + 100) * (scale ** 2))
        min_area = max(int(4 * (scale ** 2)), 2)
        
        result = np.zeros_like(blemish_candidates)
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                result[labels == i] = 255

        # Dilate slightly to cover edges of blemishes
        k_dilate = max(int(3 * scale), 1)
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))
        result = cv2.dilate(result, dilate_kernel, iterations=1)

        return result


def compute_skin_quality_map(
    image_bgr: np.ndarray,
    skin_mask: np.ndarray,
    patch_size: int = 15,
) -> np.ndarray:
    """Compute a skin quality map from local variance.

    Returns a float32 map (same HxW as image, single channel) where:
    - High values = rough skin (acne, pores, blemishes) -> needs more smoothing
    - Low values = smooth skin -> needs less smoothing

    Uses local variance of luminance in a sliding window.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    ksize = max(3, patch_size) | 1

    local_mean = cv2.boxFilter(gray, ddepth=-1, ksize=(ksize, ksize),
                               borderType=cv2.BORDER_REFLECT)
    local_sq_mean = cv2.boxFilter(gray * gray, ddepth=-1, ksize=(ksize, ksize),
                                  borderType=cv2.BORDER_REFLECT)

    local_var = np.clip(local_sq_mean - local_mean * local_mean, 0, None)

    m = skin_mask.astype(np.float32)
    skin_pixels = local_var[m > 0.05]
    if skin_pixels.size == 0:
        return np.zeros_like(gray, dtype=np.float32)

    vmin = float(skin_pixels.min())
    vmax = float(skin_pixels.max())
    if vmax - vmin < 1e-6:
        quality = np.zeros_like(local_var, dtype=np.float32)
    else:
        quality = ((local_var - vmin) / (vmax - vmin)).astype(np.float32)

    k = max(5, patch_size // 3) | 1
    m_blur = cv2.GaussianBlur(m, (k, k), 0)

    return quality * m_blur
