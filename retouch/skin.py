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
        """Adaptive Rosy Foundation: LAB-based skin whitening and rosy/porcelain cosmetic shift."""
        if strength == 0 or skin_mask is None:
            return img_bgr

        s = strength / 100.0
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        protection = self._get_highlight_protection(lab)

        # Fix #2: Copy l_val to prevent stale reads on subsequent passes
        l_val = lab[:, :, 0].copy()
        shadow_protection = np.clip((l_val - 80.0) / 40.0, 0.0, 1.0)
        skin_color_mask = skin_mask * shadow_protection

        # Fix #7: Calculate has_skin once to avoid redundant np.any calls
        skin_indices = skin_mask > 0.3
        has_skin = np.any(skin_indices)
        
        median_a = 128.0
        median_b = 128.0
        if has_skin:
            median_a = np.median(lab[:, :, 1][skin_indices])
            median_b = np.median(lab[:, :, 2][skin_indices])

        # Soft-clipping luminance lift/reduction using skin_color_mask
        if s >= 0:
            lift_factor = 0.16 * s
            lab[:, :, 0] = l_val + (255.0 - l_val) * skin_color_mask * lift_factor * protection
        else:
            shadow_decay = np.clip((l_val - 10.0) / 20.0, 0.0, 1.0)
            lift_factor = 0.16 * s
            lab[:, :, 0] = l_val + l_val * skin_color_mask * lift_factor * shadow_decay

        abs_s = abs(s)
        if tone == "porcelain":
            lab[:, :, 2] = lab[:, :, 2] - 5.0 * abs_s * skin_color_mask
        elif tone == "neutral":
            pass
        else:  # rosy
            lab[:, :, 1] = lab[:, :, 1] + 3.0 * abs_s * skin_color_mask
            lab[:, :, 2] = lab[:, :, 2] - 4.0 * abs_s * skin_color_mask

        if has_skin:
            a_target = 128.0 + (median_a - 128.0) * 0.85
            b_target = 128.0 + (median_b - 128.0) * 0.85
            blend_factor = 0.12 * abs_s
            lab[:, :, 1] = lab[:, :, 1] + (a_target - lab[:, :, 1]) * skin_mask * blend_factor
            lab[:, :, 2] = lab[:, :, 2] + (b_target - lab[:, :, 2]) * skin_mask * blend_factor

        whitened = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, whitened, skin_mask)

    def equalize(self, img_bgr, skin_mask, strength=40, ref_lab=None):
        """CLAHE + local average color equalization to unify skin tone."""
        if strength <= 0:
            return img_bgr

        s = strength / 100.0
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        skin_indices = skin_mask > 0.3
        if np.any(skin_indices):
            # Fix #1: Safely snapshot the reference before mutations, keeping channel indexing correct
            if ref_lab is not None:
                ref_ab = ref_lab.astype(np.float32)[:, :, 1:3].copy()
            else:
                ref_ab = lab[:, :, 1:3].copy()
            median_a = np.median(ref_ab[:, :, 0][skin_indices])
            median_b = np.median(ref_ab[:, :, 1][skin_indices])
            pull = (0.20 * s * skin_mask)
            lab[:, :, 1] = lab[:, :, 1] + (median_a - lab[:, :, 1]) * pull
            lab[:, :, 2] = lab[:, :, 2] + (median_b - lab[:, :, 2]) * pull

        # PERF: Run CLAHE only on the L channel to save memory
        l_chan_u8 = np.clip(lab[:, :, 0], 0, 255).astype(np.uint8)
        clip_limit = 1.0 + s * 2.0
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_chan_u8)

        protection = self._get_highlight_protection(lab)
        l_clahe_f = l_clahe.astype(np.float32)
        l_final_f = l_clahe_f * protection + lab[:, :, 0] * (1.0 - protection)
        lab[:, :, 0] = np.clip(l_final_f, 0, 255)

        # Fix #6: Explicitly clip all channels to prevent AB wrap-around artifacts
        equalized = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
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

        brighten_mask = np.zeros(regions.skin.shape, dtype=np.float32)
        for mask in (regions.nose_bridge, regions.forehead_center,
                     regions.cheek_highlights_l, regions.cheek_highlights_r):
            if mask is not None:
                brighten_mask = np.clip(brighten_mask + mask, 0, 1)

        darken_mask = (regions.jawline_contour.astype(np.float32, copy=False)
                       if regions.jawline_contour is not None
                       else np.zeros(regions.skin.shape, dtype=np.float32))
        darken_mask = np.clip(darken_mask - brighten_mask, 0.0, 1.0)

        l_val = lab[:, :, 0]
        lab[:, :, 0] = l_val + (255.0 - l_val) * brighten_mask * 0.05 * s * protection
        lab[:, :, 0] = np.clip(lab[:, :, 0] - lab[:, :, 0] * darken_mask * 0.04 * s, 0, 255)

        result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, result, regions.skin)

    def harmonize_neck(self, img_bgr, face_landmarks, person_mask, face_skin_mask, neck_mask=None, strength=40):
        """Unify the neck and chest skin color and brightness with the face skin."""
        if strength <= 0 or person_mask is None or face_skin_mask is None:
            return img_bgr

        s = strength / 100.0
        h_img, w_img = img_bgr.shape[:2]
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        face_skin_indices = face_skin_mask > 0.3
        if not np.any(face_skin_indices):
            return img_bgr

        face_median_l = np.median(lab[:, :, 0][face_skin_indices])
        face_median_a = np.median(lab[:, :, 1][face_skin_indices])
        face_median_b = np.median(lab[:, :, 2][face_skin_indices])

        coords = [(int(lm.x * w_img), int(lm.y * h_img)) for lm in face_landmarks.landmark]
        if not coords:
            return img_bgr
        xs, ys = zip(*coords)
        face_w = max(xs) - min(xs)
        face_h = max(ys) - min(ys)

        if neck_mask is not None and neck_mask.max() > 0.01:
            neck_mask_final = neck_mask.copy()
        else:
            chin_y = int(face_landmarks.landmark[self.MEDIAPIPE_CHIN_IDX].y * h_img)
            chin_x = int(face_landmarks.landmark[self.MEDIAPIPE_CHIN_IDX].x * w_img)

            neck_y1 = chin_y
            neck_y2 = min(int(chin_y + face_h * 1.5), h_img)
            neck_x1 = max(int(chin_x - face_w * 0.8), 0)
            neck_x2 = min(int(chin_x + face_w * 0.8), w_img)

            if neck_y2 <= neck_y1 or neck_x2 <= neck_x1:
                return img_bgr

            pm = person_mask.astype(np.float32)
            if pm.max() > 1.0:
                pm /= 255.0
            if pm.ndim == 3:
                pm = pm[..., 0]

            neck_mask_est = np.zeros((h_img, w_img), dtype=np.float32)
            neck_mask_est[neck_y1:neck_y2, neck_x1:neck_x2] = pm[neck_y1:neck_y2, neck_x1:neck_x2]

            dist_ab = np.sqrt((lab[:, :, 1] - face_median_a) ** 2 + (lab[:, :, 2] - face_median_b) ** 2)
            face_std_a = np.std(lab[:, :, 1][face_skin_indices])
            face_std_b = np.std(lab[:, :, 2][face_skin_indices])
            ab_threshold = max(12.0, (face_std_a + face_std_b) * 2.0)
            skin_match = (dist_ab < ab_threshold).astype(np.float32)
            
            # Fix #5: Erode face skin mask to create a smoother boundary exclusion
            face_skin_eroded = cv2.erode(face_skin_mask, np.ones((5, 5), np.uint8))
            neck_mask_final = np.clip(neck_mask_est * skin_match - face_skin_eroded, 0, 1)

        lm = face_landmarks.landmark
        d_left = abs(lm[6].x - lm[234].x)
        d_right = abs(lm[454].x - lm[6].x)
        yaw_ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-5)

        if yaw_ratio <= 1.3:
            p1 = np.array([lm[33].x * w_img, lm[33].y * h_img, lm[33].z * face_w])
            p2 = np.array([lm[263].x * w_img, lm[263].y * h_img, lm[263].z * face_w])
            p3 = np.array([lm[6].x * w_img, lm[6].y * h_img, lm[6].z * face_w])
            
            n = np.cross(p2 - p1, p3 - p1)
            n = n / (np.linalg.norm(n) + 1e-5)
            
            z_neck = lm[152].z * face_w
            Y, X = np.ogrid[:h_img, :w_img]
            dist_to_plane = np.abs(n[0] * (X - p1[0]) + n[1] * (Y - p1[1]) + n[2] * (z_neck - p1[2]))
            
            threshold = face_w * 0.15
            depth_gate = (dist_to_plane < threshold).astype(np.float32)
            neck_mask_final *= depth_gate

        if neck_mask_final.max() < 0.01:
            return img_bgr

        k_blur = max(15, int(face_w * 0.03)) | 1
        neck_mask_final = cv2.GaussianBlur(neck_mask_final, (k_blur, k_blur), 0)

        neck_indices = neck_mask_final > 0.3
        if not np.any(neck_indices):
            return img_bgr

        neck_median_l = np.median(lab[:, :, 0][neck_indices])
        neck_median_a = np.median(lab[:, :, 1][neck_indices])
        neck_median_b = np.median(lab[:, :, 2][neck_indices])

        l_diff = (face_median_l - neck_median_l) * 0.75 * s
        lab[:, :, 0] = np.clip(lab[:, :, 0] + neck_mask_final * l_diff, 0, 255)

        a_diff = (face_median_a - neck_median_a) * 0.7 * s
        b_diff = (face_median_b - neck_median_b) * 0.7 * s
        lab[:, :, 1] = np.clip(lab[:, :, 1] + neck_mask_final * a_diff, 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2] + neck_mask_final * b_diff, 0, 255)

        return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    def apply_specular_bloom(self, img_bgr, skin_mask, strength=40, tone="rosy"):
        """Generates a soft pink/lavender or neutral halo around skin highlights where L > 220."""
        if strength <= 0 or skin_mask is None:
            return img_bgr

        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_chan = lab[:, :, 0]

        highlight_mask = np.clip((l_chan - 220.0) / 20.0, 0.0, 1.0) * (skin_mask > 0.1).astype(np.float32)

        if highlight_mask.max() < 0.01:
            return img_bgr

        h, w = img_bgr.shape[:2]
        min_dim = min(h, w)
        dilate_k = max(3, int(min_dim * 0.005)) | 1
        blur_k = max(15, int(min_dim * 0.03)) | 1

        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k))
        highlight_mask = cv2.dilate(highlight_mask, kernel_dilate)
        highlight_mask = cv2.GaussianBlur(highlight_mask, (blur_k, blur_k), 0)
        highlight_mask *= (skin_mask > 0.1).astype(np.float32)
        highlight_mask = np.clip(highlight_mask, 0.0, 1.0)

        # Fix #1: Explicitly copy channels to avoid passing live views into cv2.merge, clipping to prevent wraps
        l_shifted = np.clip(l_chan + 15.0, 0, 255)
        a_shifted = np.clip(lab[:, :, 1], 0, 255)
        b_shifted = np.clip(lab[:, :, 2], 0, 255)
        
        if tone == "porcelain":
            b_shifted = np.clip(b_shifted - 10.0, 0, 255)
        elif tone == "rosy":
            a_shifted = np.clip(a_shifted + 10.0, 0, 255)
            b_shifted = np.clip(b_shifted - 5.0, 0, 255)
        
        lab_shifted = cv2.merge([l_shifted, a_shifted, b_shifted])
        img_shifted_bgr = cv2.cvtColor(lab_shifted.astype(np.uint8), cv2.COLOR_LAB2BGR)

        s = strength / 100.0
        return blend_masked(img_bgr, img_shifted_bgr, highlight_mask * s)

    @staticmethod
    def _get_highlight_protection(lab):
        """Linearly decays adjustments for bright pixels (L > 220) to prevent specular clipping."""
        l_val = lab[:, :, 0]
        protection = np.clip(1.0 - (l_val - 220.0) / 30.0, 0.0, 1.0)
        return protection


def adaptive_smooth(image_bgr, skin_mask, base_smooth=0.5, base_texture=0.25, smooth_boost=0.3):
    """Skin smoothing with quality-adaptive strength."""
    from .blemish import compute_skin_quality_map
    from .frequency import combine_adaptive
    quality_map = compute_skin_quality_map(image_bgr, skin_mask)
    return combine_adaptive(image_bgr, skin_mask, quality_map,
                            base_smooth=base_smooth, base_texture=base_texture,
                            smooth_boost=smooth_boost)
