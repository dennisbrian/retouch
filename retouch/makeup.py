"""Makeup engine layer — blush washes, foundation bases, and face color enhancements.

Performs makeup enhancements in LAB space for optimal, natural blending.
"""

import cv2
import numpy as np


class MakeupEngine:
    """Professional portrait makeup engine."""

    def apply_blush(self, img_bgr, face_landmarks, face_width, strength=30):
        """Apply a natural rosy/pink blush wash to the cheeks.

        Works by generating soft radial feathered masks on cheek highlights (landmarks
        117 and 346) and nudging the LAB a-channel (green-red) positively.

        Args:
            img_bgr: (H, W, 3) uint8 BGR input image.
            face_landmarks: Normalized landmarks list.
            face_width: Estimated width of the face in pixels.
            strength: 0–100 blush intensity.

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

        # Heavily feather the mask
        k_blur = (2 * blush_radius) | 1
        blush_mask = cv2.GaussianBlur(blush_mask, (k_blur, k_blur), 0)

        # Apply in LAB space
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Nudge the a-channel (green -> red/pink)
        # 12.0 is a safe maximum shift for highly natural redness
        lab[:, :, 1] = np.clip(lab[:, :, 1] + blush_mask * 12.0 * s, 0, 255)

        # Optional: very subtle brightness lift on cheeks for a fresh look (+2.0 max)
        lab[:, :, 0] = np.clip(lab[:, :, 0] + blush_mask * 2.0 * s, 0, 255)

        return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
