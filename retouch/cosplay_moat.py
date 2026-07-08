"""A3: Cosplay skin moat — makeup-agnostic enhancements for cosplay retouching.

Stages:
1. Wig-lace blend: seamless transition where wig meets skin at hairline.
   - Detect wig edges via color/saturation clustering
   - Feather blend zone (±20px)
   - Preserve underlying skin detail

2. Stockings: detect hosiery via chroma+saturation patterns.
   - Uniform color, low variance detection
   - Targeted smoothing without over-blurring
   - Apply frequency smoothing (guided filter)

3. Shoot-consistency lock: maintain continuity across multiple shots.
   - Compare lighting/white-balance drift between frames
   - Detect skin tone (LAB L/a/b) in reference frames
   - Suggest white-balance_kelvin delta to maintain continuity
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np

from .utils import (
    guided_filter,
    feather_mask,
    normalize_mask,
    squeeze_mask,
    blend_masked,
    bgr_f32_to_lab_f32,
    lab_f32_to_bgr_f32,
)
from .color_space import bgr_to_lch

logger = logging.getLogger(__name__)


class WigLaceBlender:
    """Detects and blends wig edges seamlessly into skin at the hairline.

    Strategy:
    1. Detect wig edges via hue+saturation clustering on hair-like colors
    2. Identify the transition zone (high hue/saturation variance edge)
    3. Feather blend zone ±20px around detected edge
    4. Use guided filtering to smooth the blend while preserving skin detail
    """

    def __init__(self):
        self.edge_threshold = 0.30  # normalized gradient threshold for edge detection
        self.feather_width = 20  # pixels on each side of detected edge
        self.blend_strength_base = 0.6  # base blend strength [0,1]

    def blend(
        self,
        img_bgr: np.ndarray,
        hair_mask: Optional[np.ndarray],
        skin_mask: Optional[np.ndarray],
        strength: float = 0.5,
    ) -> np.ndarray:
        """Blend wig-lace transition with underlying skin.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            hair_mask: (H, W) float mask of hair regions [0, 1].
            skin_mask: (H, W) float mask of skin regions [0, 1].
            strength: 0–1 blending intensity (0=no effect, 1=full blend).

        Returns:
            (H, W, 3) uint8 or float32 BGR image, matching input dtype.
        """
        if strength <= 0 or hair_mask is None or skin_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
        else:
            img_u8 = img_bgr

        h, w = img_u8.shape[:2]

        # Convert to LCH for wig-like color detection
        lch = bgr_to_lch(img_u8)
        hue = lch[:, :, 0]  # [0, 360)
        chroma = lch[:, :, 1]  # [0, 100]

        # Wig-lace edges are typically high hue variance + moderate chroma
        # (synthetic fibers, not natural skin). Detect via a morphological
        # approach: find high-gradient regions in hue/chroma inside hair_mask.

        hair_indices = hair_mask > 0.3
        skin_indices = skin_mask > 0.3

        # Compute hue gradient (circular, so use sin/cos to handle wraparound)
        hue_rad = np.radians(hue)
        hue_grad_x = cv2.Sobel(np.sin(hue_rad), cv2.CV_32F, 1, 0, ksize=3)
        hue_grad_y = cv2.Sobel(np.sin(hue_rad), cv2.CV_32F, 0, 1, ksize=3)
        hue_grad_mag = np.sqrt(hue_grad_x**2 + hue_grad_y**2)

        # Chroma gradient (saturation variance)
        chroma_grad_x = cv2.Sobel(chroma, cv2.CV_32F, 1, 0, ksize=3)
        chroma_grad_y = cv2.Sobel(chroma, cv2.CV_32F, 0, 1, ksize=3)
        chroma_grad_mag = np.sqrt(chroma_grad_x**2 + chroma_grad_y**2)

        # Combined edge signal: high in both hue and chroma variance
        edge_signal = np.clip(
            (hue_grad_mag / 0.2 + chroma_grad_mag / 10.0) / 2.0, 0.0, 1.0
        )

        # Locate wig-lace edges: gradient peaks inside hair_mask, near skin_mask boundary
        edge_mask = edge_signal > self.edge_threshold
        edge_mask = edge_mask & hair_indices

        # Dilate edge_mask to create blend zone
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blend_zone = cv2.dilate(
            edge_mask.astype(np.uint8),
            kernel,
            iterations=self.feather_width // 5,
        ).astype(np.float32)

        # Feather the blend zone for smooth transition (ensure odd kernel size)
        feather_ksize = self.feather_width if self.feather_width % 2 == 1 else self.feather_width + 1
        blend_zone = cv2.GaussianBlur(blend_zone, (feather_ksize, feather_ksize), 0)

        # Apply guided filter in the blend zone to smooth hair while preserving skin detail
        # Split into channels for guided filtering
        result = img_u8.astype(np.float32).copy()
        for c in range(3):
            channel = img_u8[:, :, c].astype(np.float32)
            # Guided filter with radius based on feather_width
            filtered = guided_filter(channel, radius=self.feather_width // 2, eps=1.0, guide=channel)
            # Blend between original and filtered based on blend_zone
            result[:, :, c] = (
                channel * (1.0 - blend_zone * strength * self.blend_strength_base) +
                filtered * blend_zone * strength * self.blend_strength_base
            )

        result = np.clip(result, 0, 255).astype(np.uint8)

        # Return in original dtype
        if is_float:
            return result.astype(np.float32) / 255.0

        return result

    def detect_wig_edges(
        self, hair_mask: np.ndarray, lch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Detect wig-lace edges via hue/chroma clustering.

        Returns:
            (edge_mask, edge_confidence): both (H, W) float32 [0, 1]
        """
        if hair_mask is None or hair_mask.max() < 0.01:
            return np.zeros_like(hair_mask), np.zeros_like(hair_mask)

        hue = lch[:, :, 0]
        chroma = lch[:, :, 1]

        hair_indices = hair_mask > 0.3

        # Hue variance inside hair region
        if np.sum(hair_indices) < 100:
            return np.zeros_like(hair_mask), np.zeros_like(hair_mask)

        hue_mean = np.mean(hue[hair_indices])
        hue_std = np.std(hue[hair_indices])

        # Chroma variance inside hair region
        chroma_mean = np.mean(chroma[hair_indices])
        chroma_std = np.std(chroma[hair_indices])

        # High hue/chroma variance signals a synthetic wig with distinct fibers
        # (natural hair has lower variance within the same general hue band).
        # Detect edges where variance is high.
        hue_rad = np.radians(hue)
        hue_grad_x = cv2.Sobel(np.sin(hue_rad), cv2.CV_32F, 1, 0, ksize=3)
        hue_grad_y = cv2.Sobel(np.sin(hue_rad), cv2.CV_32F, 0, 1, ksize=3)
        hue_grad_mag = np.sqrt(hue_grad_x**2 + hue_grad_y**2)

        chroma_grad_x = cv2.Sobel(chroma, cv2.CV_32F, 1, 0, ksize=3)
        chroma_grad_y = cv2.Sobel(chroma, cv2.CV_32F, 0, 1, ksize=3)
        chroma_grad_mag = np.sqrt(chroma_grad_x**2 + chroma_grad_y**2)

        # Normalize by the hair region's own variance for scale-invariance
        hue_edge_signal = hue_grad_mag / (hue_std + 1.0)
        chroma_edge_signal = chroma_grad_mag / (chroma_std + 1.0)

        combined_signal = np.clip(
            (hue_edge_signal + chroma_edge_signal) / 2.0, 0.0, 1.0
        )

        edge_confidence = combined_signal
        edge_mask = (combined_signal > self.edge_threshold).astype(np.float32)

        return edge_mask, edge_confidence


class HosierySmoother:
    """Detects and smooths hosiery (stockings, tights) via chroma+saturation patterns.

    Strategy:
    1. Detect hosiery via LAB chroma thresholding (uniform color, low variance)
    2. Identify connected regions matching hosiery color patterns
    3. Apply frequency smoothing (guided filter) without over-blurring texture
    """

    def __init__(self):
        self.chroma_uniformity_threshold = 8.0  # chroma variance threshold for hosiery
        self.min_hosiery_area = 500  # minimum pixels for a valid hosiery region
        self.smoothing_strength_base = 0.7

    def smooth(
        self,
        img_bgr: np.ndarray,
        person_mask: Optional[np.ndarray],
        strength: float = 0.5,
    ) -> np.ndarray:
        """Detect and smooth hosiery regions.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            person_mask: (H, W) float mask of person [0, 1].
            strength: 0–1 smoothing intensity.

        Returns:
            (H, W, 3) uint8 or float32 BGR image, matching input dtype.
        """
        if strength <= 0 or person_mask is None:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
        else:
            img_u8 = img_bgr

        h, w = img_u8.shape[:2]

        # Detect hosiery via LAB chroma uniformity
        hosiery_mask = self._detect_hosiery(img_u8, person_mask)

        if hosiery_mask.max() < 0.01:
            return img_bgr

        # Apply guided filtering to smooth hosiery while preserving texture
        result = img_u8.astype(np.float32).copy()
        for c in range(3):
            channel = img_u8[:, :, c].astype(np.float32)
            # Guided filter with small radius (preserve local texture)
            filtered = guided_filter(channel, radius=3, eps=1.0, guide=channel)
            # Blend based on hosiery mask and strength
            result[:, :, c] = (
                channel * (1.0 - hosiery_mask * strength * self.smoothing_strength_base) +
                filtered * hosiery_mask * strength * self.smoothing_strength_base
            )

        result = np.clip(result, 0, 255).astype(np.uint8)

        if is_float:
            return result.astype(np.float32) / 255.0

        return result

    def _detect_hosiery(
        self, img_u8: np.ndarray, person_mask: Optional[np.ndarray]
    ) -> np.ndarray:
        """Detect hosiery regions via chroma uniformity.

        Hosiery typically has:
        - Uniform color (low chroma variance within a region)
        - Moderate saturation (not as muted as skin, not as vivid as makeup)
        - Smooth surface (low texture variance)

        Returns:
            (H, W) float32 mask [0, 1] of detected hosiery regions.
        """
        if img_u8.dtype != np.uint8:
            img_u8 = np.clip(img_u8, 0, 255).astype(np.uint8)

        h, w = img_u8.shape[:2]

        # Convert to LAB for chroma detection
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
        a_chan = lab[:, :, 1]
        b_chan = lab[:, :, 2]

        # Compute chroma: sqrt((a-128)^2 + (b-128)^2)
        # Subtract 128 to center around neutral
        a_centered = a_chan - 128.0
        b_centered = b_chan - 128.0
        chroma = np.sqrt(a_centered**2 + b_centered**2)

        # Apply local chroma uniformity check:
        # Compute chroma variance in small local windows (5x5)
        kernel_size = 5
        chroma_mean = cv2.blur(chroma, (kernel_size, kernel_size))
        chroma_sq_mean = cv2.blur(chroma**2, (kernel_size, kernel_size))
        chroma_var = np.clip(chroma_sq_mean - chroma_mean**2, 0, None)

        # Hosiery has LOW chroma variance (uniform color)
        # Skin has HIGHER chroma variance (natural mottling)
        hosiery_uniformity = 1.0 - np.clip(chroma_var / (self.chroma_uniformity_threshold**2), 0, 1)

        # Additional gate: moderate chroma (not skin-pale, not makeup-vivid)
        # Skin chroma ~10-30, makeup >40, stockings ~15-35
        chroma_gate = np.clip(
            1.0 - np.abs(chroma - 22.0) / 15.0, 0.0, 1.0
        )

        # Combine signals
        hosiery_candidate = hosiery_uniformity * chroma_gate

        # Apply person_mask to avoid false positives on background
        if person_mask is not None:
            person_mask_normalized = normalize_mask(person_mask)
            person_mask_normalized = squeeze_mask(person_mask_normalized)
            hosiery_candidate = hosiery_candidate * person_mask_normalized

        # Morphological cleanup: remove small spurious detections
        hosiery_bin = (hosiery_candidate > 0.5).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        hosiery_bin = cv2.morphologyEx(hosiery_bin, cv2.MORPH_CLOSE, kernel)
        hosiery_bin = cv2.morphologyEx(hosiery_bin, cv2.MORPH_OPEN, kernel)

        # Connected components: keep only sufficiently large regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(hosiery_bin)
        hosiery_mask = np.zeros((h, w), dtype=np.float32)

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_hosiery_area:
                hosiery_mask[labels == i] = hosiery_candidate[labels == i]

        # Smooth mask edges for clean blending (5x5 is odd, so safe)
        hosiery_mask = cv2.GaussianBlur(hosiery_mask, (5, 5), 0)

        return hosiery_mask


class ShootConsistencyLock:
    """Maintains white-balance and lighting continuity across multiple shots of the same cosplay.

    Strategy:
    1. Analyze reference shot(s) to establish baseline lighting/white-balance
    2. Detect drift in skin tone (LAB L/a/b) across subsequent shots
    3. Suggest corrective white-balance_kelvin delta to maintain visual continuity
    4. Optional: apply corrective shift automatically if user sets shoot_consistency=high
    """

    def __init__(self):
        self.skin_roi_fraction = 0.3  # use center 30% of detected faces for analysis
        self.kelvin_per_ab_delta = 50.0  # heuristic: ~50K per 1 unit a/b drift

    def analyze_shot(
        self, img_bgr: np.ndarray, face_data: Optional[Any] = None
    ) -> Dict[str, float]:
        """Analyze white-balance and lighting state of a shot.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            face_data: Optional face detection data with landmarks/ROI.

        Returns:
            {
                "lab_l": float (mean L in detected skin),
                "lab_a": float (mean a in detected skin),
                "lab_b": float (mean b in detected skin),
                "chroma": float (mean chroma in detected skin),
                "valid": bool (True if sufficient skin data found),
            }
        """
        is_float = img_bgr.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
        else:
            img_u8 = img_bgr

        h, w = img_u8.shape[:2]

        # Simple fallback: if no face data, analyze center region (most likely to be skin)
        if face_data is None:
            roi_h_start = int(h * 0.2)
            roi_h_end = int(h * 0.5)
            roi_w_start = int(w * 0.2)
            roi_w_end = int(w * 0.8)
            roi = img_u8[roi_h_start:roi_h_end, roi_w_start:roi_w_end]
        else:
            # Use face ROI if available (e.g., a FaceData object with bbox)
            # For now, assume simple center-crop from face_data dict if provided
            if isinstance(face_data, dict) and "bbox" in face_data:
                x, y, w_face, h_face = face_data["bbox"]
                # Crop to center of face
                margin = int(self.skin_roi_fraction * min(w_face, h_face))
                roi = img_u8[
                    max(0, y + margin) : min(h, y + h_face - margin),
                    max(0, x + margin) : min(w, x + w_face - margin),
                ]
            else:
                # Fallback to center region
                roi_h_start = int(h * 0.2)
                roi_h_end = int(h * 0.5)
                roi_w_start = int(w * 0.2)
                roi_w_end = int(w * 0.8)
                roi = img_u8[roi_h_start:roi_h_end, roi_w_start:roi_w_end]

        if roi.size < 100:
            return {
                "lab_l": 128.0,
                "lab_a": 128.0,
                "lab_b": 128.0,
                "chroma": 0.0,
                "valid": False,
            }

        # Convert ROI to LAB
        lab_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_mean = np.mean(lab_roi[:, :, 0])
        a_mean = np.mean(lab_roi[:, :, 1])
        b_mean = np.mean(lab_roi[:, :, 2])

        # Compute chroma
        a_centered = a_mean - 128.0
        b_centered = b_mean - 128.0
        chroma = np.sqrt(a_centered**2 + b_centered**2)

        return {
            "lab_l": float(l_mean),
            "lab_a": float(a_mean),
            "lab_b": float(b_mean),
            "chroma": float(chroma),
            "valid": True,
        }

    def suggest_white_balance_delta(
        self,
        shot1_analysis: Dict[str, float],
        shot2_analysis: Dict[str, float],
    ) -> Dict[str, float]:
        """Suggest white-balance correction to match shot1 from shot2.

        Args:
            shot1_analysis: Analysis of reference shot (target lighting).
            shot2_analysis: Analysis of current shot (to be corrected).

        Returns:
            {
                "white_balance_kelvin_delta": float (suggested kelvin shift),
                "confidence": float (0–1, how confident the suggestion is),
                "drift_magnitude": float (measured LAB a/b drift),
            }
        """
        if not shot1_analysis.get("valid") or not shot2_analysis.get("valid"):
            return {
                "white_balance_kelvin_delta": 0.0,
                "confidence": 0.0,
                "drift_magnitude": 0.0,
            }

        # Measure a/b drift (white-balance is primarily a/b tint)
        # Positive a = more red, negative a = more green
        # Positive b = more yellow, negative b = more blue
        a_drift = shot2_analysis["lab_a"] - shot1_analysis["lab_a"]
        b_drift = shot2_analysis["lab_b"] - shot1_analysis["lab_b"]

        drift_magnitude = np.sqrt(a_drift**2 + b_drift**2)

        # Heuristic: a/b drift maps to Kelvin shift
        # Warm → high kelvin (more red/yellow, positive a/b)
        # Cool → low kelvin (more blue/green, negative b, positive a)
        # This is a simplified model; real color temperature is complex.

        # For now: positive b drift (toward yellow) suggests warmer light
        # (higher kelvin), so positive delta.
        kelvin_delta = b_drift * self.kelvin_per_ab_delta

        # Confidence: drift < 3 units is noise, >15 is a serious mismatch
        confidence = np.clip(drift_magnitude / 15.0, 0.0, 1.0)

        return {
            "white_balance_kelvin_delta": float(kelvin_delta),
            "confidence": float(confidence),
            "drift_magnitude": float(drift_magnitude),
        }

    def apply_consistency_lock(
        self,
        img_bgr: np.ndarray,
        reference_analysis: Dict[str, float],
        current_analysis: Dict[str, float],
        strength: float = 0.5,
    ) -> np.ndarray:
        """Apply white-balance correction to match reference shot.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            reference_analysis: Analysis of reference shot.
            current_analysis: Analysis of current shot (will be corrected).
            strength: 0–1 correction intensity.

        Returns:
            (H, W, 3) uint8 or float32 BGR image, matching input dtype.
        """
        if strength <= 0:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        if is_float:
            img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8)
        else:
            img_u8 = img_bgr

        # Get suggested correction
        suggestion = self.suggest_white_balance_delta(reference_analysis, current_analysis)
        delta_kelvin = suggestion["white_balance_kelvin_delta"] * strength

        if abs(delta_kelvin) < 50:
            # Too small to matter
            return img_bgr

        # Apply Kelvin shift via LAB a/b adjustment
        lab = cv2.cvtColor(img_u8, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Map Kelvin to a/b:
        # Positive kelvin (warmer) → increase a (red) and increase b (yellow)
        # Negative kelvin (cooler) → decrease a and decrease b

        # Simplified model: ~50 Kelvin = 1 unit a/b change
        a_shift = delta_kelvin / self.kelvin_per_ab_delta
        b_shift = delta_kelvin / self.kelvin_per_ab_delta

        lab[:, :, 1] = np.clip(lab[:, :, 1] + a_shift, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + b_shift, 0, 255)

        result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        if is_float:
            return result.astype(np.float32) / 255.0

        return result
