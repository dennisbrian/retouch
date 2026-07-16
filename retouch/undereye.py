"""Under-eye dark circle repair module.

Targeted darkness/shadow reduction under the eyes via LAB L-channel selective
brightening, with optional chroma desaturation for puffiness reduction.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .parsing import FaceRegions
from .chromophore import decompose_chromophores, reconstruct_from_chromophores
from .utils import blend_masked, normalize_mask, feather_mask as _feather_mask, bgr_f32_to_lab_f32, lab_f32_to_bgr_f32


def _to_lab(img_bgr: np.ndarray, is_float: bool) -> np.ndarray:
    """Convert BGR to LAB, handling both uint8 and float32."""
    if is_float:
        return bgr_f32_to_lab_f32(img_bgr)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab: np.ndarray, is_float: bool) -> np.ndarray:
    """Convert LAB back to BGR, handling both uint8 and float32."""
    clipped = np.clip(lab, 0, 255)
    if is_float:
        return lab_f32_to_bgr_f32(clipped)
    return cv2.cvtColor(clipped.astype(np.uint8), cv2.COLOR_LAB2BGR)


class UndereyeAnalyzer:
    """Detect dark-circle regions via LAB L-channel analysis."""

    def detect_dark_circles(
        self,
        lab: np.ndarray,
        mask: np.ndarray,
        threshold_offset: float = 15.0,
    ) -> Tuple[np.ndarray, float]:
        """Detect dark circles by comparing L-channel to surrounding skin.

        Args:
            lab: (H, W, 3) LAB image in float32.
            mask: (H, W) under-eye region mask [0, 1].
            threshold_offset: L-darkening threshold (default 15 units).

        Returns:
            (detection_mask, local_median_l): Binary detection mask and reference L value.
        """
        if mask.max() < 0.01:
            return np.zeros_like(mask), np.nan

        L = lab[:, :, 0]

        # Get local skin reference: dilate mask to get surrounding cheek area
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        dilated = cv2.dilate((mask > 0.3).astype(np.uint8), kernel, iterations=1).astype(np.float32)
        surround = np.clip(dilated - mask, 0.0, 1.0)

        surround_pixels = L[surround > 0.3]
        # If no surround found, fall back to median of the masked region itself
        if len(surround_pixels) < 20:
            mask_pixels = L[mask > 0.3]
            if len(mask_pixels) < 20:
                return np.zeros_like(mask), np.nan
            local_median_l = np.median(mask_pixels)
        else:
            local_median_l = np.median(surround_pixels)

        # Flag regions darker than median - threshold_offset
        dark_threshold = local_median_l - threshold_offset
        dark_mask = ((L < dark_threshold) * mask).astype(np.float32)

        # Filter noise via connected components (min area ~100 pixels)
        dark_u8 = (dark_mask > 0.5).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dark_u8)

        filtered_dark = np.zeros_like(dark_u8)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= 100:
                filtered_dark[labels == i] = 1

        return filtered_dark.astype(np.float32), float(local_median_l)


class UndereyeRemover:
    """Apply selective L-brightening to dark areas with edge preservation."""

    def brighten_dark_circles(
        self,
        lab: np.ndarray,
        dark_circle_mask: np.ndarray,
        local_median_l: float,
        strength: float = 0.5,
        max_lift: float = 30.0,
    ) -> np.ndarray:
        """Brighten dark circle regions via selective L-channel lifting.

        Args:
            lab: (H, W, 3) LAB image in float32.
            dark_circle_mask: (H, W) detection mask from UndereyeAnalyzer.
            local_median_l: Reference L value from surrounding skin.
            strength: 0–1 strength factor.
            max_lift: Maximum L increase (default 30, conservative).

        Returns:
            Modified LAB image.
        """
        if dark_circle_mask.max() < 0.01 or strength <= 0:
            return lab

        L = lab[:, :, 0].copy()
        darkness = np.clip(local_median_l - L, 0.0, max_lift)

        # Selective brightening: only brighten pixels darker than median
        l_lift = darkness * dark_circle_mask * strength * (max_lift / 100.0)
        lab[:, :, 0] = np.clip(L + l_lift, 0.0, 255.0)

        return lab

    def reduce_puffiness_chroma(
        self,
        lab: np.ndarray,
        dark_circle_mask: np.ndarray,
        strength: float = 0.5,
        chroma_reduction: float = 0.7,
    ) -> np.ndarray:
        """Desaturate under-eye via chroma reduction for puffiness minimization.

        Args:
            lab: (H, W, 3) LAB image in float32.
            dark_circle_mask: (H, W) under-eye region mask.
            strength: 0–1 strength factor.
            chroma_reduction: Target chroma ratio (0.7 = 30% desaturation).

        Returns:
            Modified LAB image with reduced chroma in dark-circle regions.
        """
        if dark_circle_mask.max() < 0.01 or strength <= 0:
            return lab

        a = lab[:, :, 1].copy()
        b = lab[:, :, 2].copy()

        # Compute chroma (offset-adjusted per cv2 LAB convention)
        chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)
        hue = np.arctan2(b - 128.0, a - 128.0)

        # Reduce chroma selectively
        new_chroma = chroma * (1.0 - (1.0 - chroma_reduction) * strength * dark_circle_mask)

        # Reconstruct a/b
        a_new = 128.0 + new_chroma * np.cos(hue)
        b_new = 128.0 + new_chroma * np.sin(hue)

        lab[:, :, 1] = np.clip(a_new, 0.0, 255.0)
        lab[:, :, 2] = np.clip(b_new, 0.0, 255.0)

        return lab

    def feather_edges(
        self,
        mask: np.ndarray,
        feather_radius: int = 5,
    ) -> np.ndarray:
        """Apply morphological opening (dilate then erode) for soft edge blending.

        Args:
            mask: (H, W) binary mask.
            feather_radius: Feather kernel size in pixels.

        Returns:
            Feathered mask.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feather_radius, feather_radius))
        # Opening: erode then dilate (shrink then expand, softens edges)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # Gaussian blur for further softening
        feathered = cv2.GaussianBlur(opened.astype(np.float32), (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
        return np.clip(feathered, 0.0, 1.0).astype(np.float32)


class UndereyeProcessor:
    """Orchestrate under-eye dark circle analysis + removal."""

    def __init__(self):
        """Initialize analyzer and remover."""
        self.analyzer = UndereyeAnalyzer()
        self.remover = UndereyeRemover()

    def process(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        darken_removal_strength: float = 0.0,
        puffiness_reduction_strength: float = 0.0,
    ) -> np.ndarray:
        """Process under-eye region with selective brightening and optional chroma reduction.

        Supports both uint8 and float32 input. Output dtype matches input dtype.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            mask: (H, W) under-eye region mask [0, 1].
            darken_removal_strength: 0–1 darkness removal strength.
            puffiness_reduction_strength: 0–1 chroma desaturation strength.

        Returns:
            (H, W, 3) BGR image, same dtype as input.
        """
        if (darken_removal_strength <= 0 and puffiness_reduction_strength <= 0) or mask.max() < 0.01:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        # Convert to LAB
        lab = _to_lab(img_bgr, is_float)

        # Detect dark circles
        dark_circle_mask, local_median_l = self.analyzer.detect_dark_circles(lab, mask, threshold_offset=15.0)

        if np.isnan(local_median_l):
            return img_bgr

        # Feather detection mask for smooth edges
        dark_circle_mask_feathered = self.remover.feather_edges(dark_circle_mask, feather_radius=5)

        # Apply selective L-brightening
        if darken_removal_strength > 0:
            lab = self.remover.brighten_dark_circles(
                lab, dark_circle_mask_feathered, local_median_l,
                strength=darken_removal_strength, max_lift=30.0
            )

        # Apply chroma reduction (puffiness)
        if puffiness_reduction_strength > 0:
            lab = self.remover.reduce_puffiness_chroma(
                lab, dark_circle_mask_feathered,
                strength=puffiness_reduction_strength, chroma_reduction=0.7
            )

        # Convert back to BGR
        result = _from_lab(lab, is_float)

        # Blend within the mask to avoid hard edges
        return blend_masked(img_bgr, result, mask)

    def attenuate_hemoglobin(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        strength: float = 0.0,
    ) -> np.ndarray:
        """Experimental E-EYE-4 spike: reduce vascular color, not luminance.

        The existing under-eye path can brighten and desaturate a dark region,
        but neither operation distinguishes vascular color from a true shadow.
        This leaf operator attenuates only hemoglobin that exceeds the nearby
        cheek baseline, then restores the source L channel before compositing.
        It is intentionally not wired to engine parameters or recipes until
        real-image visual QA establishes a safe product control.
        """
        if strength <= 0.0 or mask is None or mask.max() < 0.01:
            return img_bgr

        s = float(np.clip(strength, 0.0, 1.0))
        is_float = img_bgr.dtype == np.float32
        region = normalize_mask(mask)
        assert region is not None
        if region.max() < 0.01:
            return img_bgr

        # Estimate a robust cheek baseline from a ring surrounding the
        # landmark-defined under-eye region. A minimum scale rejects normal
        # sensor variation on otherwise uniform skin.
        ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
        ring = np.clip(
            cv2.dilate((region > 0.3).astype(np.uint8), ring_kernel).astype(np.float32)
            - region,
            0.0,
            1.0,
        )
        work = img_bgr.astype(np.float32)
        melanin, hemoglobin = decompose_chromophores(work)
        samples = hemoglobin[ring > 0.3]
        if samples.size < 20:
            samples = hemoglobin[region > 0.3]
        if samples.size < 20:
            return img_bgr

        baseline = float(np.median(samples))
        mad = float(np.median(np.abs(samples - baseline)))
        sigma = max(1.4826 * mad, 0.03)
        excess = np.clip(hemoglobin - baseline, 0.0, None)
        confidence = np.clip((excess - 2.0 * sigma) / (3.0 * sigma), 0.0, 1.0)
        effective_mask = region * confidence
        if effective_mask.max() < 1e-4:
            return img_bgr

        corrected_hb = hemoglobin - s * excess
        reconstructed = reconstruct_from_chromophores(
            melanin,
            corrected_hb,
            img_bgr=work,
        )

        # Chromophore reconstruction changes both color and brightness. Keep
        # source lightness so this spike cannot flatten an anatomical shadow.
        # Use the float-native LAB convention for both images. Mixing OpenCV's
        # uint8 LAB scale with float LAB would reintroduce a luminance shift.
        source_lab = _to_lab(work, True)
        corrected_lab = _to_lab(reconstructed, True)
        corrected_lab[:, :, 0] = source_lab[:, :, 0]
        corrected = _from_lab(corrected_lab, True)
        if not is_float:
            corrected = np.clip(corrected, 0.0, 255.0).astype(np.uint8)
        return blend_masked(img_bgr, corrected, effective_mask)


class UnderEyeRepairer:
    """Legacy interface for backward compatibility with existing engine wiring."""

    def __init__(self):
        """Initialize the processor."""
        self._processor = UndereyeProcessor()

    def repair(
        self,
        img_bgr: np.ndarray,
        regions: FaceRegions,
        strength: int = 40,
    ) -> np.ndarray:
        """Repair dark circles under both eyes (legacy interface).

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            regions: FaceRegions from parser.
            strength: 0–100 darkness removal intensity.

        Returns:
            (H, W, 3) BGR image, same dtype as input.
        """
        if strength <= 0:
            return img_bgr

        s = strength / 100.0
        result = img_bgr.copy()

        for mask in (regions.left_under_eye, regions.right_under_eye):
            if mask is not None and mask.max() > 0.01:
                result = self._processor.process(result, mask, darken_removal_strength=s)

        return result
