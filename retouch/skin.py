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

    def whiten(self, img_bgr, skin_mask, strength=30):
        """Adaptive Rosy Foundation: LAB-based skin whitening and rosy/porcelain cosmetic shift.

        Uses a soft-clipping luminance lift to brighten the skin without clipping highlights,
        and applies a rosy-porcelain color correction (nudge 'a' positive for pink/rosy tones,
        nudge 'b' negative for cool porcelain tones).

        Args:
            img_bgr: (H, W, 3) uint8.
            skin_mask: (H, W) float mask 0–1.
            strength: 0–100.

        Returns:
            (H, W, 3) uint8 result.
        """
        if strength <= 0:
            return img_bgr

        s = strength / 100.0
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        protection = self._get_highlight_protection(lab)

        # Soft-clipping luminance lift
        lift_factor = 0.16 * s
        lab[:, :, 0] = lab[:, :, 0] + (255.0 - lab[:, :, 0]) * skin_mask * lift_factor * protection

        # Rosy color shift: nudge a channel (green-red) positive for pink rosy glow
        # Porcelain color shift: nudge b channel (blue-yellow) negative for cool porcelain tones
        lab[:, :, 1] = lab[:, :, 1] + 3.0 * s * skin_mask
        lab[:, :, 2] = lab[:, :, 2] - 4.0 * s * skin_mask

        # Nudge AB toward the person's own median to avoid imposing a fixed skin tone
        skin_indices = skin_mask > 0.3
        if np.any(skin_indices):
            median_a = np.median(lab[:, :, 1][skin_indices])
            median_b = np.median(lab[:, :, 2][skin_indices])
            a_target = 128.0 + (median_a - 128.0) * 0.85
            b_target = 128.0 + (median_b - 128.0) * 0.85
            blend_factor = 0.12 * s
            lab[:, :, 1] = lab[:, :, 1] + (a_target - lab[:, :, 1]) * skin_mask * blend_factor
            lab[:, :, 2] = lab[:, :, 2] + (b_target - lab[:, :, 2]) * skin_mask * blend_factor

        whitened = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return whitened

    def equalize(self, img_bgr, skin_mask, strength=40):
        """CLAHE + local average color equalization to unify skin tone.

        Pulls yellow foreheads or red cheeks toward the local skin median color.
        Non-skin regions are zeroed before CLAHE to avoid histogram contamination.
        """
        if strength <= 0:
            return img_bgr

        s = strength / 100.0
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # 1. Local skin color harmonization
        skin_indices = skin_mask > 0.3
        if np.any(skin_indices):
            median_a = np.median(lab[:, :, 1][skin_indices])
            median_b = np.median(lab[:, :, 2][skin_indices])
            pull = 0.20 * s
            lab[:, :, 1] = lab[:, :, 1] + (median_a - lab[:, :, 1]) * skin_mask * pull
            lab[:, :, 2] = lab[:, :, 2] + (median_b - lab[:, :, 2]) * skin_mask * pull

        # 2. CLAHE on luminance channel — write result back only over skin
        lab_u = np.clip(lab, 0, 255).astype(np.uint8)
        clip_limit = 1.0 + s * 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_channel = lab_u[:, :, 0].copy()
        l_clahe = clahe.apply(l_channel)
        skin_bin = (skin_mask > 0.3).astype(np.uint8)
        lab_u[:, :, 0] = l_clahe * skin_bin + l_channel * (1 - skin_bin)

        equalized = cv2.cvtColor(lab_u, cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, equalized, skin_mask * s)

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

        # Brighten highlights: +5% max
        l_val = lab[:, :, 0]
        lab[:, :, 0] = l_val + (255.0 - l_val) * brighten_mask * 0.05 * s * protection

        # Darken contours: -4% max (jawline doesn't need highlight protection)
        lab[:, :, 0] = np.clip(lab[:, :, 0] - lab[:, :, 0] * darken_mask * 0.04 * s, 0, 255)

        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

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

        # ---- 1. Segment Neck/Chest skin ----
        if neck_mask is not None and neck_mask.max() > 0.01:
            neck_mask_final = neck_mask.copy()
        else:
            # MediaPipe FaceMesh lower chin landmark index
            chin_y = int(face_landmarks.landmark[self.MEDIAPIPE_CHIN_IDX].y * h_img)
            chin_x = int(face_landmarks.landmark[self.MEDIAPIPE_CHIN_IDX].x * w_img)

            # Estimate face size from landmarks
            coords = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in face_landmarks.landmark]
            xs, ys = zip(*coords)
            face_w = max(xs) - min(xs)
            face_h = max(ys) - min(ys)

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
        # | 1 forces odd kernel size
        k_blur = 15
        neck_mask_final = cv2.GaussianBlur(neck_mask_final, (k_blur, k_blur), 0)

        # ---- 2. Face-to-Neck Harmonization ----
        neck_indices = neck_mask_final > 0.3
        if not np.any(neck_indices):
            return img_bgr

        neck_median_l = np.median(lab[:, :, 0][neck_indices])
        neck_median_a = np.median(lab[:, :, 1][neck_indices])
        neck_median_b = np.median(lab[:, :, 2][neck_indices])

        # Shift neck luminance (brightness lift) if neck is darker
        # Allow up to 75% matching to keep shadows natural
        if neck_median_l < face_median_l:
            l_diff = (face_median_l - neck_median_l) * 0.75 * s
            lab[:, :, 0] = np.clip(lab[:, :, 0] + neck_mask_final * l_diff, 0, 255)

        # Shift neck color channels (AB) towards face color to align color cast
        a_diff = (face_median_a - neck_median_a) * 0.7 * s
        b_diff = (face_median_b - neck_median_b) * 0.7 * s
        lab[:, :, 1] = np.clip(lab[:, :, 1] + neck_mask_final * a_diff, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + neck_mask_final * b_diff, 0, 255)

        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    @staticmethod
    def _get_highlight_protection(lab):
        """Linearly decays adjustments for bright pixels (L > 220) to prevent specular clipping."""
        l_val = lab[:, :, 0]
        # protection factor: 1.0 when L <= 220, decays to 0.0 at L=250
        protection = np.clip(1.0 - (l_val - 220.0) / 30.0, 0.0, 1.0)
        return protection
