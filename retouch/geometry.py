"""Geometry manipulation layer — photographer-grade face slimming and reshaping.

Applies local translation warping (liquid warp) on specific jaw, cheek, and chin
landmarks to perform non-destructive, subtle reshaping.
"""

from __future__ import annotations

from typing import Any, List

import cv2
import numpy as np


class FaceReshaper:
    """Photographer-grade face slimming and reshaping using local translation warping."""

    def reshape(
        self,
        img_bgr: np.ndarray,
        faces: List[Any],
        strength: int = 30,
    ) -> np.ndarray:
        """Perform face slimming (jaw, cheeks, chin) for all detected faces.

        Args:
            img_bgr: (H, W, 3) uint8 BGR input image.
            faces: List of FaceData objects.
            strength: 0–100 overall reshaping intensity.

        Returns:
            (H, W, 3) reshaped image.
        """
        if strength <= 0 or not faces:
            return img_bgr

        h, w = img_bgr.shape[:2]
        s = strength / 100.0

        # Initialize global displacement maps
        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        has_deformations = False

        for face in faces:
            landmarks = face.landmarks.landmark
            # Calculate dynamic face size parameters
            x_234 = landmarks[234].x * w
            x_454 = landmarks[454].x * w
            face_width = abs(x_454 - x_234)
            if face_width < 10.0:
                continue

            # Slimming warp definitions: (ControlPoint, TargetPoint, Radius)
            warps = []

            # 1. Jaw Slimming (Landmarks 234 right, 454 left)
            # Right jaw 234: shift inward (to the right, positive X)
            c_rjaw = (int(landmarks[234].x * w), int(landmarks[234].y * h))
            t_rjaw = (int(c_rjaw[0] + face_width * 0.04 * s), c_rjaw[1])
            warps.append((c_rjaw, t_rjaw, int(face_width * 0.5)))

            # Left jaw 454: shift inward (to the left, negative X)
            c_ljaw = (int(landmarks[454].x * w), int(landmarks[454].y * h))
            t_ljaw = (int(c_ljaw[0] - face_width * 0.04 * s), c_ljaw[1])
            warps.append((c_ljaw, t_ljaw, int(face_width * 0.5)))

            # 2. Cheek Compression (Landmarks 117 right, 346 left)
            # Right cheek 117: shift inward (to the right, positive X)
            c_rchk = (int(landmarks[117].x * w), int(landmarks[117].y * h))
            t_rchk = (int(c_rchk[0] + face_width * 0.02 * s), c_rchk[1])
            warps.append((c_rchk, t_rchk, int(face_width * 0.4)))

            # Left cheek 346: shift inward (to the left, negative X)
            c_lchk = (int(landmarks[346].x * w), int(landmarks[346].y * h))
            t_lchk = (int(c_lchk[0] - face_width * 0.02 * s), c_lchk[1])
            warps.append((c_lchk, t_lchk, int(face_width * 0.4)))

            # 3. Chin Refinement (Landmark 152 center bottom)
            # Chin 152: shift upward (negative Y)
            c_chin = (int(landmarks[152].x * w), int(landmarks[152].y * h))
            t_chin = (c_chin[0], int(c_chin[1] - face_width * 0.015 * s))
            warps.append((c_chin, t_chin, int(face_width * 0.35)))

            # Accumulate displacements on the global map
            for C, T, R in warps:
                cx, cy = C
                tx, ty = T
                ux = tx - cx
                uy = ty - cy

                # Crop deformation bounding box for high efficiency
                x1 = max(cx - R, 0)
                y1 = max(cy - R, 0)
                x2 = min(cx + R, w)
                y2 = min(cy + R, h)

                if (x2 - x1) < 4 or (y2 - y1) < 4:
                    continue

                # Local mesh coordinates relative to C
                local_dx = map_x[y1:y2, x1:x2] - cx
                local_dy = map_y[y1:y2, x1:x2] - cy
                dist_sq = local_dx * local_dx + local_dy * local_dy
                R_sq = R * R

                mask = dist_sq < R_sq
                if not np.any(mask):
                    continue

                # Smooth weight falloff: (1 - d^2 / R^2)^2
                t = 1.0 - dist_sq / R_sq
                weight = t * t
                weight[~mask] = 0.0

                # Displace the source lookup mapping
                map_x[y1:y2, x1:x2] -= weight * ux
                map_y[y1:y2, x1:x2] -= weight * uy
                has_deformations = True

        if not has_deformations:
            return img_bgr

        # Apply global deformation in a single pass
        reshaped = cv2.remap(
            img_bgr,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        return reshaped
