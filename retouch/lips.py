"""Lip enhancement module — vibrance, tinting, and texture preservation.

Key design: uses *vibrance* (smart saturation) instead of raw saturation.
Vibrance boosts under-saturated pixels more, producing natural-looking
enhancement without over-saturating already-vivid lips.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .utils import blend_masked, estimate_face_width, vibrance as vibrance_fn, apply_u8_op_float


# Predefined lip tint colours (BGR)
LIP_TINTS = {
    "nude":    (155, 175, 195),
    "pink":    (160, 140, 210),
    "rose":    (140, 120, 200),
    "coral":   (120, 140, 210),
    "berry":   (140, 80, 170),
    "red":     (80, 80, 210),
    "cosplay": (180, 130, 220),
}
LIP_TINT_NAMES = list(LIP_TINTS.keys())


class LipEnhancer:
    """Lip enhancement: vibrance, colour tint, and texture preservation."""

    def enhance(
        self,
        img_bgr: np.ndarray,
        lip_mask: Optional[np.ndarray],
        strength: int = 30,
        tint: Optional[Union[str, Tuple[int, int, int]]] = None,
        finish: str = "gloss",
    ) -> np.ndarray:
        """Full lip enhancement pipeline.

        Args:
            img_bgr: (H, W, 3) uint8.
            lip_mask: (H, W) float mask 0–1.
            strength: 0–100 overall intensity.
            tint: Optional tint name ("nude", "pink", "rose", "coral",
                  "berry", "red", "cosplay") or BGR tuple.
            finish: Lip finish type: "matte", "gloss", or "velvet" (default: "gloss").

        Returns:
            (H, W, 3) uint8 result.
        """
        if strength <= 0 or lip_mask is None or lip_mask.max() < 0.01:
            return img_bgr

        if img_bgr.dtype == np.float32:
            # E1 delta adapter: lip pipeline is uint8-contract. Delta is
            # confined to the lip region.
            return apply_u8_op_float(img_bgr, self.enhance, lip_mask, strength,
                                     tint=tint, finish=finish)

        # Estimate face width from lip mask (lip width is typically ~30% of face width)
        face_width = estimate_face_width(lip_mask=lip_mask, img_shape=img_bgr.shape[:2])

        s = strength / 100.0
        result = img_bgr.copy()

        # ---- Texture preservation: extract high-frequency detail ----
        lip_texture = self._extract_texture(result, lip_mask, face_width)

        # ---- Vibrance (smart saturation) ----
        result = vibrance_fn(result, lip_mask, s * 0.25)

        # ---- Smoothing based on finish ----
        if finish == "velvet":
            # Stronger bilateral smoothing for velvet look
            result = self._smooth(result, lip_mask, s * 0.65)
        else:
            # Subtle smoothing for matte/gloss
            result = self._smooth(result, lip_mask, s * 0.3)

        # ---- Colour tint (optional) ----
        if tint is not None:
            # For cosplay, apply a much stronger tint wash (0.85 opacity max) to achieve vibrant anime look
            tint_opacity = s * 0.85 if (isinstance(tint, str) and tint == "cosplay") else s * 0.25
            result = self._apply_tint(result, lip_mask, tint, tint_opacity)

        # ---- Re-apply texture with dynamic opacity based on finish ----
        if finish == "velvet":
            # Re-apply texture very subtly (20% strength) to keep the soft velvet sheen
            result = self._reapply_texture(result, lip_texture, lip_mask, face_width, opacity=0.20)
        else:
            # Standard texture re-application (75% strength)
            result = self._reapply_texture(result, lip_texture, lip_mask, face_width, opacity=0.75)

        # ---- Lip Gloss Specular Highlights (gloss only) ----
        if finish == "gloss":
            result = self._add_lip_gloss(result, lip_mask, s)

        return result

    def _add_lip_gloss(
        self,
        img_bgr: np.ndarray,
        lip_mask: Optional[np.ndarray],
        strength: float,
    ) -> np.ndarray:
        """Detect bright specular spots inside lips and apply a subtle specular gloss boost.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            lip_mask: (H, W) float mask 0–1.
            strength: 0.0–1.0 gloss intensity.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if strength <= 0 or lip_mask is None or lip_mask.max() < 0.01:
            return img_bgr

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]

        # Find median/std of lip luminance
        lip_pixels = l_chan[lip_mask > 0.4]
        if len(lip_pixels) == 0:
            return img_bgr

        lip_median = np.median(lip_pixels)
        lip_std = max(lip_pixels.std(), 1.0)

        # Specular mask: bright pixels inside lips
        specular_thresh = lip_median + lip_std * 1.5
        specular_mask = ((l_chan > specular_thresh) & (l_chan > 130.0) & (lip_mask > 0.4)).astype(np.float32)

        if specular_mask.sum() == 0:
            return img_bgr

        # Feather slightly
        specular_mask = cv2.GaussianBlur(specular_mask, (3, 3), 0)

        # Boost highlights using the safe soft-clipping lift
        # Increased lift factor from 0.25 to 0.45 to make specular highlights pop strongly
        l_boost = (255.0 - l_chan) * specular_mask * 0.45 * strength
        lab[:, :, 0] = np.clip(l_chan + l_boost, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _extract_texture(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        face_width: float,
    ) -> np.ndarray:
        """Extract high-frequency lip texture before processing.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            mask: (H, W) float mask restricting the texture extraction.
            face_width: Approximate face width in pixels (scales the kernel).

        Returns:
            (H, W) float32 high-frequency texture map.
        """
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ksize = max(int(5 * (face_width / 500.0)), 3) | 1
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
        texture = gray - blurred  # high-frequency detail
        return texture * mask

    def _reapply_texture(
        self,
        img_bgr: np.ndarray,
        texture: np.ndarray,
        mask: np.ndarray,
        face_width: float,
        opacity: float = 0.75,
    ) -> np.ndarray:
        """Re-apply preserved lip texture onto processed image.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            texture: (H, W) float32 texture map (unused for sizing but required).
            mask: (H, W) float mask.
            face_width: Approximate face width in pixels (unused but kept for
                signature parity).
            opacity: 0.0–1.0 strength of texture re-application.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + texture * opacity, 0, 255)
        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    def _smooth(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        """Very gentle bilateral filter on lips.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            mask: (H, W) float mask 0–1.
            strength: 0.0–1.0 smoothing strength.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        if strength <= 0:
            return img_bgr
        smoothed = cv2.bilateralFilter(img_bgr, 5, 30, 30)
        return blend_masked(img_bgr, smoothed, mask * strength)

    def _apply_tint(
        self,
        img_bgr: np.ndarray,
        mask: np.ndarray,
        tint: Union[str, Tuple[int, int, int]],
        strength: float,
    ) -> np.ndarray:
        """Apply a colour tint to lips.

        Args:
            tint: Either a string key from LIP_TINTS or a (B, G, R) tuple.
        """
        if isinstance(tint, str):
            tint_bgr = LIP_TINTS.get(tint, LIP_TINTS["nude"])
        else:
            tint_bgr = tint

        # Create solid tint layer
        tint_layer = np.full_like(img_bgr, tint_bgr, dtype=np.uint8)

        # Soft light blend mode (preserves luminance structure)
        img_f = img_bgr.astype(np.float32) / 255.0
        tint_f = tint_layer.astype(np.float32) / 255.0

        # Soft light formula
        blended = np.where(
            tint_f <= 0.5,
            img_f * (2 * tint_f + img_f * (1 - 2 * tint_f)),
            img_f * (1 - (1 - img_f) * (2 * tint_f - 1)) + np.sqrt(img_f) * (2 * tint_f - 1)
        )
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

        return blend_masked(img_bgr, blended, mask * strength)
