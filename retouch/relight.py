"""Virtual studio relighting — directional 3D shading based on FaceMesh depth map."""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np


class Relighter:
    """Estimates a 3D depth map and normal map from face landmarks to apply directional relighting."""

    def __init__(self, alpha: float = 32.0):
        """Initialize the relighter.

        Args:
            alpha: Specular roughness exponent for Blinn-Phong shading.
        """
        self.alpha = alpha

    def relight(
        self,
        canvas: np.ndarray,
        landmarks: Any,
        skin_mask: Optional[np.ndarray],
        face_width: float,
        strength: float = 0.0,
        azimuth: float = 0.0,
        elevation: float = 30.0,
    ) -> np.ndarray:
        """Apply directional lighting to the skin region of the portrait canvas.

        Args:
            canvas: Crop ROI image (BGR, uint8).
            landmarks: Shifted face landmarks relative to crop ROI canvas.
            skin_mask: Soft float mask (0-1) for face skin area.
            face_width: Approximate face width in pixels (ied * 2.5).
            strength: Slider strength (0-100).
            azimuth: Light angle in degrees (-180 to 180).
            elevation: Light elevation angle in degrees (-90 to 90).

        Returns:
            Relit canvas (BGR, uint8).
        """
        if strength <= 0.0 or skin_mask is None or skin_mask.max() < 0.01:
            return canvas

        h, w = canvas.shape[:2]
        lm = landmarks.landmark

        # 1. Temple-Calibrated Yaw Guard
        # lm[6] is nose bridge midpoint, lm[234] is left temple, lm[454] is right temple
        d_left = abs(lm[6].x - lm[234].x)
        d_right = abs(lm[454].x - lm[6].x)
        ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-5)
        # Attenuate strength linearly as ratio ranges from 1.5 to 1.7
        yaw_factor = 1.0 - np.clip((ratio - 1.5) / 0.2, 0.0, 1.0)

        effective_strength = strength * yaw_factor
        if effective_strength <= 0.0:
            return canvas

        # 2. Build Delaunay Triangulation & Render Z Depth Map
        # Extract 2D coords of the first 468 landmarks (MediaPipe topology)
        pts_2d = []
        for i in range(468):
            px = max(0.01, min(w - 0.01, lm[i].x * w))
            py = max(0.01, min(h - 0.01, lm[i].y * h))
            pts_2d.append((px, py))

        subdiv = cv2.Subdiv2D((0, 0, w, h))
        for pt in pts_2d:
            subdiv.insert(pt)

        triangle_list = subdiv.getTriangleList()
        triangle_arr = np.array(triangle_list, dtype=np.float32)
        triangles = triangle_arr.reshape(-1, 3, 2)

        # Filter out-of-bounds helper triangles
        valid_mask = (
            (triangles[:, :, 0] >= 0) & (triangles[:, :, 0] <= w) &
            (triangles[:, :, 1] >= 0) & (triangles[:, :, 1] <= h)
        ).all(axis=1)
        valid_triangles = triangles[valid_mask]

        if len(valid_triangles) == 0:
            return canvas

        # Vectorised vertex matching to find corresponding landmark indices
        pts_arr = np.array(pts_2d, dtype=np.float32)
        tri_vertices = valid_triangles.reshape(-1, 2)
        diff = pts_arr[np.newaxis, :, :] - tri_vertices[:, np.newaxis, :]
        dist_sq = np.sum(diff**2, axis=-1)
        matched_indices = np.argmin(dist_sq, axis=1).reshape(-1, 3)

        # Average Z-coordinates per triangle
        z_coords = np.array([lm[i].z for i in range(468)], dtype=np.float32)
        tri_z = z_coords[matched_indices].mean(axis=1)

        # Draw mesh onto float32 depth map
        depth_map = np.zeros((h, w), dtype=np.float32)
        for i in range(len(valid_triangles)):
            pts = valid_triangles[i].astype(np.int32)
            cv2.fillConvexPoly(depth_map, pts, float(tri_z[i]))

        # Aspect-Ratio Scaling: convert normalised Z coordinates to pixels
        Z_pixels = depth_map * face_width

        # Gaussian blur to smooth the mesh depth triangles
        blur_k = int(face_width / 15.0) | 1
        blur_k = max(3, blur_k)
        Z_blurred = cv2.GaussianBlur(Z_pixels, (blur_k, blur_k), 0).astype(np.float32)

        # 3. Compute Surface Normals
        gx = cv2.Sobel(Z_blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Z_blurred, cv2.CV_32F, 0, 1, ksize=3)
        norm = np.sqrt(gx**2 + gy**2 + 1.0)
        # Surface normal vector pointing outward
        N_x = -gx / norm
        N_y = -gy / norm
        N_z = 1.0 / norm

        # 4. Blinn-Phong Lighting Shader
        theta = azimuth * np.pi / 180.0
        phi = elevation * np.pi / 180.0
        L_x = np.cos(phi) * np.sin(theta)
        L_y = -np.sin(phi)
        L_z = np.cos(phi) * np.cos(theta)

        # View vector V = (0, 0, 1) -> Half-vector H = normalize(L + V)
        H_x = L_x
        H_y = L_y
        H_z = L_z + 1.0
        H_norm = np.sqrt(H_x**2 + H_y**2 + H_z**2) + 1e-5
        H_x /= H_norm
        H_y /= H_norm
        H_z /= H_norm

        # Diffuse and Specular components
        I_diffuse = np.clip(N_x * L_x + N_y * L_y + N_z * L_z, 0.0, 1.0)
        I_specular = np.clip(N_x * H_x + N_y * H_y + N_z * H_z, 0.0, 1.0) ** self.alpha

        # 5. Composite onto LAB's Luminance (L) Channel
        lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB).astype(np.float32)
        L = lab[:, :, 0]

        # Highlight protection factor: decays from 1.0 (L <= 220) to 0.0 (L = 250)
        prot = np.clip(1.0 - (L - 220.0) / 30.0, 0.0, 1.0)

        # Ambient-clamped diffuse relighting
        s_diff = effective_strength * 0.008
        diffuse_term = 1.0 + s_diff * (I_diffuse - 0.6)
        diffuse_term = np.clip(diffuse_term, 0.5, 1.5)

        # Specular component (additive)
        s_spec = effective_strength * 0.005
        specular_term = s_spec * I_specular * 255.0 * prot

        L_new = np.clip(L * diffuse_term + specular_term, 0.0, 255.0)
        lab[:, :, 0] = L_new

        relit_canvas = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

        # 6. Blend using the soft skin mask
        mask_3d = skin_mask[:, :, np.newaxis] if skin_mask.ndim == 2 else skin_mask
        output = np.clip(canvas * (1.0 - mask_3d) + relit_canvas * mask_3d, 0, 255).astype(np.uint8)
        return output
