"""Makeup engine layer — blush washes, foundation bases, and face color enhancements.

Performs makeup enhancements in LAB space for optimal, natural blending.
"""

import cv2
import numpy as np


class MakeupEngine:
    """Professional portrait makeup engine."""

    def apply_blush(self, img_bgr, face_landmarks, face_width, strength=30, regions=None, nose_blush=False, under_eye_blush=False):
        """Apply a natural rosy/pink blush wash to the cheeks, nose tip, and under-eye area.

        Works by generating soft radial feathered masks on cheek highlights (landmarks
        117 and 346), optionally the nose tip (landmark 4), and blending under-eye regions.
        Increases the maximum shift strength for more vibrant cosplay styles.

        Args:
            img_bgr: (H, W, 3) uint8 BGR input image.
            face_landmarks: Normalized landmarks list.
            face_width: Estimated width of the face in pixels.
            strength: 0–100 blush intensity.
            regions: FaceRegions object from parser.
            nose_blush: Apply cute circular nose tip blush wash.
            under_eye_blush: Blend under-eye regions for anime eyeshadow effect.

        Returns:
            (H, W, 3) uint8 image with blush applied.
        """
        if strength <= 0:
            return img_bgr

        h, w = img_bgr.shape[:2]
        s = strength / 100.0

        # Cheek landmarks: 117 (right cheek), 346 (left cheek)
        landmarks = face_landmarks.landmark
        c_r = (int(landmarks[117].x * w), int(landmarks[117].y * h))
        c_l = (int(landmarks[346].x * w), int(landmarks[346].y * h))

        # Blush radius scaled to face size
        blush_radius = int(face_width * 0.18)
        if blush_radius < 5:
            return img_bgr

        # Create radial blush mask
        blush_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(blush_mask, c_r, blush_radius, 1.0, -1)
        cv2.circle(blush_mask, c_l, blush_radius, 1.0, -1)

        # Optional nose tip blush (landmark 4 is the nose tip)
        if nose_blush:
            c_n = (int(landmarks[4].x * w), int(landmarks[4].y * h))
            cv2.circle(blush_mask, c_n, int(blush_radius * 0.4), 0.6, -1)

        # Heavily feather the circle mask first
        k_blur = (2 * blush_radius) | 1
        blush_mask = cv2.GaussianBlur(blush_mask, (k_blur, k_blur), 0)

        # Optional under-eye blush wash (anime-style eyeshadow blend)
        if under_eye_blush and regions is not None:
            # Heavily feather the under-eye masks before blending to prevent a dirty/bruised look
            if regions.left_under_eye is not None:
                left_ue = cv2.GaussianBlur(regions.left_under_eye, (15, 15), 0)
                blush_mask = np.maximum(blush_mask, left_ue * 0.35)
            if regions.right_under_eye is not None:
                right_ue = cv2.GaussianBlur(regions.right_under_eye, (15, 15), 0)
                blush_mask = np.maximum(blush_mask, right_ue * 0.35)

        # Exclude lips from blush mask to prevent blush from altering lip color
        if regions is not None and regions.lips is not None:
            lips_f = regions.lips.astype(np.float32)
            if lips_f.max() > 1.0:
                lips_f = lips_f / 255.0
            blush_mask = np.clip(blush_mask - lips_f, 0.0, 1.0)

        # Apply in LAB space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Nudge the a-channel (green -> red/pink)
        # Shift factor set to 16.0 for a vibrant but soft and clean cosplay look
        lab[:, :, 1] = np.clip(lab[:, :, 1] + blush_mask * 16.0 * s, 0, 255)

        # Optional: very subtle brightness lift on cheeks/nose for a fresh look (+2.0 max)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + blush_mask * 2.0 * s, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
