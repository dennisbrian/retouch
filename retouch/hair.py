"""Hair & Outfit shine enhancer with Wig/Head Region Isolation.

Detects shiny highlights on wigs/hair in the near-head region and boosts
their specular brightness and local contrast to create a silky, premium look.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import (
    normalize_mask,
    squeeze_mask,
    bgr_f32_to_lab_f32,
    lab_f32_to_bgr_f32,
)

from .hairwork import hair_flow, unify_hair_color


def _to_lab(img_bgr: np.ndarray, is_float: bool) -> np.ndarray:
    if is_float:
        return bgr_f32_to_lab_f32(img_bgr)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def _from_lab(lab: np.ndarray, is_float: bool) -> np.ndarray:
    clipped = np.clip(lab, 0, 255)
    if is_float:
        return lab_f32_to_bgr_f32(clipped)
    return cv2.cvtColor(clipped.astype(np.uint8), cv2.COLOR_LAB2BGR)


class HairEnhancer:
    """Enhance hair/wig highlights and local contrast."""

    def enhance(
        self,
        img_bgr: np.ndarray,
        person_mask: Optional[np.ndarray],
        face_oval_mask: Optional[np.ndarray],
        bbox: Tuple[int, int, int, int],
        strength: int = 40,
        hair_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Enhance hair shine within the hair region.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.
            person_mask: (H, W) float person mask.
            face_oval_mask: (H, W) float face oval mask.
            bbox: (x, y, w, h) face bounding box.
            strength: 0-100 overall intensity.
            hair_mask: Optional BiSeNet hair mask.

        Returns:
            (H, W, 3) same dtype as input.
        """
        if strength <= 0:
            return img_bgr

        is_float = img_bgr.dtype == np.float32

        if hair_mask is not None and hair_mask.max() > 0.01:
            h_mask = hair_mask.copy()
            x_face, y_face, w_face, h_face = bbox
        else:
            if person_mask is None or face_oval_mask is None:
                return img_bgr
            H, W = img_bgr.shape[:2]
            x_face, y_face, w_face, h_face = bbox
            cx = x_face + w_face / 2.0
            cy = y_face + h_face / 2.0

            # ---- 1. Isolate Near-Head Region (Wig/Hair area) ----
            # Expand bounding box around face center to cover hair
            y1 = max(int(cy - h_face * 1.3), 0)
            y2 = min(int(cy + h_face * 1.0), H)
            x1 = max(int(cx - w_face * 1.2), 0)
            x2 = min(int(cx + w_face * 1.2), W)

            if (y2 - y1) < 4 or (x2 - x1) < 4:
                return img_bgr

            near_head = np.zeros((H, W), dtype=np.float32)
            near_head[y1:y2, x1:x2] = 1.0
            
            # Soft feather the bounding region to avoid boundary artifacts
            k_blur = max(int(min(w_face, h_face) * 0.2), 15) | 1
            near_head = cv2.GaussianBlur(near_head, (k_blur, k_blur), 0)

            # Squeeze person mask if it is 3D
            pm = normalize_mask(person_mask)
            pm = squeeze_mask(pm)

            # Hair mask = (person_mask - face_oval) * near_head
            h_mask = np.clip(pm - face_oval_mask, 0, 1) * near_head

        if h_mask.max() < 0.01:
            return img_bgr

        # ---- 2. Detect bright hair/wig highlights ----
        lab = _to_lab(img_bgr, is_float)
        l_chan = lab[:, :, 0]

        # Apply global hair region lifts (exposure, midtones, highlights) to the L channel
        s = strength / 100.0
        l_lifted = l_chan.copy()
        
        # Exposure lift: +0.08 * s
        l_lifted += (255.0 - l_lifted) * 0.08 * s
        
        # Midtones lift: +0.12 * s
        l_lifted += (255.0 - l_lifted) * 0.12 * s
        
        # Highlights lift: +0.10 * s on pixels where L > 128
        high_idx = l_chan > 128
        l_lifted[high_idx] += (255.0 - l_lifted[high_idx]) * 0.10 * s
        
        # Blend the globally lifted L channel back using h_mask
        l_chan_new = l_chan * (1.0 - h_mask) + l_lifted * h_mask
        l_chan_new = np.clip(l_chan_new, 0, 255)

        # ---- 3. Detect bright hair/wig highlights for specular shine ----
        hair_pixels = l_chan_new[h_mask > 0.3]
        if len(hair_pixels) > 0:
            mean_val = hair_pixels.mean()
            std_val = max(hair_pixels.std(), 1.0)
            
            # Specular highlight threshold (must be reasonably bright)
            thresh = max(mean_val + std_val * 0.8, 120.0)

            # Hair highlight mask
            specular_mask = ((l_chan_new > thresh) & (h_mask > 0.3)).astype(np.float32)
            if specular_mask.sum() > 0:
                scale = w_face / 500.0
                k_dilate = max(int(3 * scale), 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dilate, k_dilate))
                specular_mask = cv2.dilate(specular_mask, kernel, iterations=1)

                k_feather = max(int(7 * scale), 3) | 1
                specular_mask = cv2.GaussianBlur(specular_mask, (k_feather, k_feather), 0)

                # ---- 4. Specular & local contrast boost ----
                # Specular boost: +10% lift to highlights
                l_boosted = np.clip(l_chan_new * (1.0 + 0.10 * s), 0, 255)

                # Local contrast via unsharp mask
                k_sharp = max(int(5 * scale), 3) | 1
                l_blurred = cv2.GaussianBlur(l_chan_new, (k_sharp, k_sharp), 0)
                l_sharp = np.clip(l_chan_new + (l_chan_new - l_blurred) * 0.5 * s, 0, 255)

                # Recombine: 40% specular boost + 60% local contrast sharpening
                l_enhanced = l_boosted * 0.4 + l_sharp * 0.6

                # Blend back using the specular mask
                l_chan_new = l_chan_new * (1.0 - specular_mask) + l_enhanced * specular_mask
                l_chan_new = np.clip(l_chan_new, 0, 255)

        lab[:, :, 0] = l_chan_new

        return _from_lab(lab, is_float)


def cleanup_flyaway_strands(
    img_bgr: np.ndarray,
    hair_mask: Optional[np.ndarray],
    face_mask: Optional[np.ndarray] = None,
    strength: float = 1.0,
    max_strand_width: int = 3,
) -> np.ndarray:
    """Detect and clean up stray flyaway hair strands across skin and background.

    Uses multi-scale curvilinear morphological top-hat ridge filters. Scope-guarded
    to operate strictly outside the hair mass silhouette so main hair body and hairline
    are preserved 100%.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 BGR image.
        hair_mask: (H, W) float hair mask. If None, returns original.
        face_mask: Optional (H, W) float face skin mask.
        strength: Cleanup strength [0, 1].
        max_strand_width: Maximum strand width in pixels (default 3).

    Returns:
        (H, W, 3) BGR image with stray flyaway hair strands cleaned up.
    """
    if strength <= 0.0 or hair_mask is None or hair_mask.max() < 0.01:
        return img_bgr

    is_float = img_bgr.dtype == np.float32
    img_u8 = np.clip(img_bgr, 0, 255).astype(np.uint8) if is_float else img_bgr

    H, W = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_u8, cv2.COLOR_BGR2GRAY)

    # 1. Scope Guard: Region strictly OUTSIDE main hair mass
    hair_bin = (hair_mask > 0.3).astype(np.uint8)
    kernel_size = max(5, int(min(H, W) * 0.02)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Hair silhouette region (hair body + 3-pixel border)
    hair_body_dilated = cv2.dilate(hair_bin, kernel, iterations=1)
    
    # Allowed zone: skin or background outside hair body
    allowed_zone = (hair_body_dilated == 0).astype(np.float32)
    if face_mask is not None:
        allowed_zone = np.maximum(allowed_zone, (face_mask > 0.1).astype(np.float32) * (hair_bin == 0).astype(np.float32))

    # 2. Curvilinear Ridge Detection via Black Top-Hat & White Top-Hat
    tophat_ksize = max_strand_width * 2 + 1
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_ksize, tophat_ksize))

    black_tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, tophat_kernel)
    white_tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel)

    # Tone-adaptive margin over local baseline
    local_blur = cv2.GaussianBlur(gray, (15, 15), 0).astype(np.float32) + 1.0
    strand_contrast = np.maximum(black_tophat.astype(np.float32), white_tophat.astype(np.float32)) / local_blur

    # Strand candidate mask
    strand_mask = (strand_contrast > 0.08).astype(np.float32) * allowed_zone * min(strength, 1.0)

    if strand_mask.max() < 0.01:
        return img_bgr

    # 3. Inpaint detected flyaway strands
    inpaint_mask = (strand_mask > 0.2).astype(np.uint8)
    if not np.any(inpaint_mask):
        return img_bgr

    inpainted_u8 = cv2.inpaint(img_u8, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    m = strand_mask[:, :, np.newaxis]
    blended = np.clip(img_u8.astype(np.float32) * (1.0 - m) + inpainted_u8.astype(np.float32) * m, 0, 255)

    if is_float:
        return blended.astype(np.float32)
    return blended.astype(np.uint8)


