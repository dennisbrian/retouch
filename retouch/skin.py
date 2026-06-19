"""Skin processing — smoothing, whitening, and CLAHE tone equalization.

All operations work within the skin mask to never affect hair, eyes, or background.
"""

import warnings

import cv2
import numpy as np

from .utils import blend_masked


class SkinProcessor:
    """Skin smoothing, whitening, and tone equalization."""

    MEDIAPIPE_CHIN_IDX = 152

    def smooth(self, img_bgr, skin_mask, strength=50):
        """Deprecated: use FrequencySeparator.combine() instead."""
        warnings.warn(
            "SkinProcessor.smooth() is a no-op. Use frequency.combine() directly.",
            DeprecationWarning, stacklevel=2,
        )
        return img_bgr

    def whiten(self, img_bgr, skin_mask, strength=30, tone="rosy"):
        """Adaptive Rosy Foundation: LAB-based skin whitening and rosy/porcelain cosmetic shift.

        Uses a soft-clipping luminance lift to brighten the skin without clipping highlights,
        and applies a rosy-porcelain color correction (nudge 'a' positive for pink/rosy tones,
        nudge 'b' negative for cool porcelain tones). Supports negative strength for skin darkening.

        Args:
            img_bgr: (H, W, 3) uint8.
            skin_mask: (H, W) float mask 0–1.
            strength: -100–100.
            tone: "rosy", "porcelain", or "neutral".

        Returns:
            (H, W, 3) uint8 result.
        """
        if strength == 0:
            return img_bgr

        s = strength / 100.0
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_original = lab.copy()  # Snapshot lab before any channel modifications (Bug 1 & 2)

        protection = self._get_highlight_protection(lab)

        # Compute skin_color_mask incorporating shadow protection to preserve 3D contours (Issue 8)
        l_val_orig = lab_original[:, :, 0]
        shadow_protection = np.clip((l_val_orig - 80.0) / 40.0, 0.0, 1.0)
        skin_color_mask = skin_mask * shadow_protection

        # Soft-clipping luminance lift/reduction using skin_color_mask
        if s >= 0:
            lift_factor = 0.16 * s
            lab[:, :, 0] = lab[:, :, 0] + (255.0 - lab[:, :, 0]) * skin_color_mask * lift_factor * protection
        else:
            # Darkening (tanning/moody look): protect deep shadows from clipping
            shadow_decay = np.clip((lab[:, :, 0] - 10.0) / 20.0, 0.0, 1.0)
            lift_factor = 0.16 * s
            lab[:, :, 0] = lab[:, :, 0] + lab[:, :, 0] * skin_color_mask * lift_factor * shadow_decay

        abs_s = abs(s)
        if tone == "porcelain":
            # cool/porcelain only: no rosy positive shift on a channel, only negative on b channel
            lab[:, :, 2] = lab[:, :, 2] - 5.0 * abs_s * skin_color_mask
        elif tone == "neutral":
            # neutral: no color shift at all, just luminance lift/reduction
            pass
        else:  # rosy
            lab[:, :, 1] = lab[:, :, 1] + 3.0 * abs_s * skin_color_mask
            lab[:, :, 2] = lab[:, :, 2] - 4.0 * abs_s * skin_color_mask

        # Nudge AB toward the person's own median to avoid imposing a fixed skin tone
        skin_indices = skin_mask > 0.3
        if np.any(skin_indices):
            # Use lab_original to ensure reference is from original channels (Bug 1 & 2)
            median_a = np.median(lab_original[:, :, 1][skin_indices])
            median_b = np.median(lab_original[:, :, 2][skin_indices])
            a_target = 128.0 + (median_a - 128.0) * 0.85
            b_target = 128.0 + (median_b - 128.0) * 0.85
            blend_factor = 0.12 * abs_s
            lab[:, :, 1] = lab[:, :, 1] + (a_target - lab[:, :, 1]) * skin_mask * blend_factor
            lab[:, :, 2] = lab[:, :, 2] + (b_target - lab[:, :, 2]) * skin_mask * blend_factor

        whitened = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, whitened, skin_mask)

    def equalize(self, img_bgr, skin_mask, strength=40, ref_lab=None):
        """CLAHE + local average color equalization to unify skin tone.

        Pulls yellow foreheads or red cheeks toward the local skin median color.
        Non-skin regions are zeroed before CLAHE to avoid histogram contamination.
        """
        if strength <= 0:
            return img_bgr

        s = strength / 100.0
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 1. Local skin color harmonization using original ref_lab if provided (Issue 3)
        skin_indices = skin_mask > 0.3
        if np.any(skin_indices):
            ref_source = ref_lab.astype(np.float32) if ref_lab is not None else lab
            median_a = np.median(ref_source[:, :, 1][skin_indices])
            median_b = np.median(ref_source[:, :, 2][skin_indices])
            # Pull towards median by 20% scaled by strength, applied only inside the skin mask
            # This avoids shifting non-skin pixels and matches the strength parameter.
            pull = 0.20 * s * skin_mask
            lab[:, :, 1] = lab[:, :, 1] + (median_a - lab[:, :, 1]) * pull
            lab[:, :, 2] = lab[:, :, 2] + (median_b - lab[:, :, 2]) * pull

        # 2. CLAHE on luminance channel — write result back only over skin
        lab_u = np.clip(lab, 0, 255).astype(np.uint8)
        clip_limit = 1.0 + s * 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_channel = lab_u[:, :, 0].copy()
        l_clahe = clahe.apply(l_channel)

        # Apply highlight protection inside equalize to prevent clipping (Design issue)
        protection = self._get_highlight_protection(lab)
        l_channel_f = l_channel.astype(np.float32)
        l_clahe_f = l_clahe.astype(np.float32)
        l_final_f = l_clahe_f * protection + l_channel_f * (1.0 - protection)
        lab_u[:, :, 0] = np.clip(l_final_f, 0, 255).astype(np.uint8)

        # Blend back using the soft skin mask to avoid hard edge seams (Bug 3 & 4)
        equalized = cv2.cvtColor(lab_u, cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, equalized, skin_mask)

    def dodge_burn(self, img_bgr, regions, strength=40):
        """Subtle 3-5% sculpting (brighten nose bridge, forehead center, cheeks;
        darken jawline/edges), scaled by highlight protection.
        """
        if strength <= 0:
            return img_bgr

        s = strength / 100.0
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        protection = self._get_highlight_protection(lab)

        # Brighten masks
        brighten_mask = np.zeros_like(regions.skin)
        for mask in (regions.nose_bridge, regions.forehead_center,
                     regions.cheek_highlights_l, regions.cheek_highlights_r):
            if mask is not None:
                brighten_mask = np.clip(brighten_mask + mask, 0, 1)

        # Darken mask
        darken_mask = regions.jawline_contour if regions.jawline_contour is not None else np.zeros_like(regions.skin)
        # Ensure mutually exclusive masks to prevent negative/asymmetric shifts on overlap (Bug 5)
        darken_mask = np.clip(darken_mask - brighten_mask, 0.0, 1.0)

        # Brighten highlights: +5% max
        l_val = lab[:, :, 0]
        lab[:, :, 0] = l_val + (255.0 - l_val) * brighten_mask * 0.05 * s * protection

        # Darken contours: -4% max (jawline doesn't need highlight protection)
        lab[:, :, 0] = np.clip(lab[:, :, 0] - lab[:, :, 0] * darken_mask * 0.04 * s, 0, 255)

        result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, result, regions.skin)

    def harmonize_neck(self, img_bgr, face_landmarks, person_mask, face_skin_mask, neck_mask=None, strength=40):
        """Unify the neck and chest skin color and brightness with the face skin to prevent the 'white face + yellow neck' discrepancy."""
        if strength <= 0 or person_mask is None or face_skin_mask is None:
            return img_bgr

        s = strength / 100.0
        h_img, w_img = img_bgr.shape[:2]
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Get reference face skin colors
        face_skin_indices = face_skin_mask > 0.3
        if not np.any(face_skin_indices):
            return img_bgr

        face_median_l = np.median(lab[:, :, 0][face_skin_indices])
        face_median_a = np.median(lab[:, :, 1][face_skin_indices])
        face_median_b = np.median(lab[:, :, 2][face_skin_indices])

        # Estimate face size from landmarks unconditionally to make k_blur adaptive (Bug 7 & 8)
        coords = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in face_landmarks.landmark]
        if not coords:
            return img_bgr
        xs, ys = zip(*coords)
        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)

        # ---- 1. Segment Neck/Chest skin ----
        if neck_mask is not None and neck_mask.max() > 0.01:
            neck_mask_final = neck_mask.copy()
        else:
            # MediaPipe FaceMesh lower chin landmark index
            chin_y = int(face_landmarks.landmark[self.MEDIAPIPE_CHIN_IDX].y * h_img)
            chin_x = int(face_landmarks.landmark[self.MEDIAPIPE_CHIN_IDX].x * w_img)

            # Neck region box directly below the chin
            neck_y1 = chin_y
            neck_y2 = min(int(chin_y + face_h * 1.5), h_img)
            neck_x1 = max(int(chin_x - face_w * 0.8), 0)
            neck_x2 = min(int(chin_x + face_w * 0.8), w_img)

            if neck_y2 <= neck_y1 or neck_x2 <= neck_x1:
                return img_bgr

            # Squeeze person mask if it is 3D
            pm = person_mask.astype(np.float32)
            if pm.max() > 1.0:
                pm /= 255.0
            if pm.ndim == 3:
                pm = pm[..., 0]

            # Fill the neck region using person mask
            neck_mask_est = np.zeros((h_img, w_img), dtype=np.float32)
            neck_mask_est[neck_y1:neck_y2, neck_x1:neck_x2] = pm[neck_y1:neck_y2, neck_x1:neck_x2]

            # Restrict to skin color (pixels close to the face skin color in AB space)
            dist_ab = np.sqrt((lab[:, :, 1] - face_median_a) ** 2 + (lab[:, :, 2] - face_median_b) ** 2)
            face_std_a = np.std(lab[:, :, 1][face_skin_indices])
            face_std_b = np.std(lab[:, :, 2][face_skin_indices])
            ab_threshold = max(12.0, (face_std_a + face_std_b) * 2.0)
            skin_match = (dist_ab < ab_threshold).astype(np.float32)
            
            # Exclude face_skin_mask to avoid double-processing the face
            neck_mask_final = np.clip(neck_mask_est * skin_match - face_skin_mask, 0, 1)

        if neck_mask_final.max() < 0.01:
            return img_bgr

        # Soft feather
        # Adaptive blur kernel size proportional to face width
        k_blur = max(15, int(face_w * 0.03)) | 1
        neck_mask_final = cv2.GaussianBlur(neck_mask_final, (k_blur, k_blur), 0)

        # ---- 2. Face-to-Neck Harmonization ----
        neck_indices = neck_mask_final > 0.3
        if not np.any(neck_indices):
            return img_bgr

        neck_median_l = np.median(lab[:, :, 0][neck_indices])
        neck_median_a = np.median(lab[:, :, 1][neck_indices])
        neck_median_b = np.median(lab[:, :, 2][neck_indices])

        # Shift neck luminance (brightness adjustment, can be positive or negative)
        # Allow up to 75% matching to keep shadows natural
        l_diff = (face_median_l - neck_median_l) * 0.75 * s
        lab[:, :, 0] = np.clip(lab[:, :, 0] + neck_mask_final * l_diff, 0, 255)

        # Shift neck color channels (AB) towards face color to align color cast
        a_diff = (face_median_a - neck_median_a) * 0.7 * s
        b_diff = (face_median_b - neck_median_b) * 0.7 * s
        lab[:, :, 1] = np.clip(lab[:, :, 1] + neck_mask_final * a_diff, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + neck_mask_final * b_diff, 0, 255)

        # Document: OpenCV LAB channels for float32 are encoded in [0, 255]
        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def apply_specular_bloom(self, img_bgr, skin_mask, strength=40, tone="rosy"):
        """Generates a soft pink/lavender or neutral halo around skin highlights where L > 220."""
        if strength <= 0 or skin_mask is None:
            return img_bgr

        # Convert to LAB to find highlights in L channel
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]

        # Soft threshold ramp from L=220 to L=240 inside skin region (Issue 7)
        highlight_mask = np.clip((l_chan - 220.0) / 20.0, 0.0, 1.0) * (skin_mask > 0.1).astype(np.float32)

        if highlight_mask.max() < 0.01:
            return img_bgr

        # Calculate kernel size proportional to image size (approx 0.5% for dilation, 3% for blur)
        h, w = img_bgr.shape[:2]
        min_dim = min(h, w)
        dilate_k = max(3, int(min_dim * 0.005)) | 1
        blur_k = max(15, int(min_dim * 0.03)) | 1

        # Dilate and Gaussian blur the highlight mask to create the soft halo/glow bleed
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        highlight_mask = cv2.dilate(highlight_mask, kernel_dilate)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (blur_k, blur_k), 0)
        # Re-clamp mask to skin boundary to prevent glow from bleeding into hair/eyes/background
        highlight_mask *= (skin_mask > 0.1).astype(np.float32)
        highlight_mask = np.clip(highlight_mask, 0.0, 1.0)

        # Apply color shift to highlights
        lab_shifted = lab.copy()
        lab_shifted[:, :, 0] = np.clip(lab_shifted[:, :, 0] + 15.0, 0, 255)
        if tone == "neutral":
            # neutral white highlight glow
            pass
        elif tone == "porcelain":
            # cool porcelain color shift (LAB L+15, b-10)
            lab_shifted[:, :, 2] = np.clip(lab_shifted[:, :, 2] - 10.0, 0, 255)
        else: # rosy
            # rosy-lavender color shift (LAB L+15, a+10, b-5)
            lab_shifted[:, :, 1] = np.clip(lab_shifted[:, :, 1] + 10.0, 0, 255)
            lab_shifted[:, :, 2] = np.clip(lab_shifted[:, :, 2] - 5.0, 0, 255)

        # Convert shifted image back to BGR
        img_shifted_bgr = cv2.cvtColor(np.clip(lab_shifted, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # Blend using the blurred highlight mask * strength
        s = strength / 100.0
        return blend_masked(img_bgr, img_shifted_bgr, highlight_mask * s)

    @staticmethod
    def _get_highlight_protection(lab):
        """Linearly decays adjustments for bright pixels (L > 220) to prevent specular clipping."""
        l_val = lab[:, :, 0]
        # protection factor: 1.0 when L <= 220, decays to 0.0 at L=250
        protection = np.clip(1.0 - (l_val - 220.0) / 30.0, 0.0, 1.0)
        return protection
