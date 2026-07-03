"""Virtual studio relighting — directional 3D shading based on FaceMesh depth map."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .utils import apply_u8_op_float


class Relighter:
    """Estimates a 3D depth map and normal map from face landmarks to apply directional relighting."""

    def __init__(self, alpha: float = 32.0) -> None:
        """Initialize the relighter.

        Args:
            alpha: Specular roughness exponent for Blinn-Phong shading.
        """
        self.alpha = alpha

    def _shading_geometry(
        self,
        canvas_shape: Tuple[int, int],
        landmarks: Any,
        face_width: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, float]:
        """Extract shading geometry: normal map and mesh data.

        Builds Delaunay triangulation from landmarks, computes depth map,
        and derives surface normals via Sobel gradients. Downsamples to
        ~160px face width for coarse computation.

        Args:
            canvas_shape: (height, width) of the canvas.
            landmarks: MediaPipe landmarks object with .landmark list.
            face_width: Approximate face width in pixels.

        Returns:
            Tuple of (N_x, N_y, N_z, small_w, small_h, scale) where:
            - N_x, N_y, N_z: Normal field at coarse scale (float32).
            - small_w, small_h: Coarse resolution dimensions.
            - scale: Downsampling factor (≤1.0).
        """
        h, w = canvas_shape
        lm = landmarks.landmark

        # Build Delaunay Triangulation & Render Z Depth Map
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
            # Fallback: return zero normals with input scale
            target = 160.0
            scale = min(1.0, target / max(face_width, 1.0))
            h_small = int(h * scale)
            w_small = int(w * scale)
            return np.zeros((h_small, w_small), dtype=np.float32), \
                   np.zeros((h_small, w_small), dtype=np.float32), \
                   np.ones((h_small, w_small), dtype=np.float32), \
                   w_small, h_small, scale

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

        # Downsample-compute/upsample-apply pattern to reduce faceting artifacts
        # Target face width ~160px for coarse computation
        target = 160.0
        scale = min(1.0, target / max(face_width, 1.0))

        if scale < 1.0:
            # Downsample depth map
            h_small = int(h * scale)
            w_small = int(w * scale)
            Z_small = cv2.resize(Z_pixels, (w_small, h_small), interpolation=cv2.INTER_AREA)
            # Scale Z values to match resized x/y grid gradients
            Z_small = Z_small * scale
        else:
            Z_small = Z_pixels
            h_small = h
            w_small = w

        # Gaussian blur to smooth the mesh depth triangles (at small scale)
        blur_k = max(3, int(face_width * scale / 10.0) | 1)
        Z_blurred = cv2.GaussianBlur(Z_small, (blur_k, blur_k), 0).astype(np.float32)

        # Compute Surface Normals (at small scale)
        gx = cv2.Sobel(Z_blurred, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(Z_blurred, cv2.CV_32F, 0, 1, ksize=3)
        norm = np.sqrt(gx**2 + gy**2 + 1.0)
        # Surface normal vector pointing outward
        N_x = -gx / norm
        N_y = -gy / norm
        N_z = 1.0 / norm

        return N_x, N_y, N_z, w_small, h_small, scale

    def _relight_v1(
        self,
        canvas: np.ndarray,
        N_x: np.ndarray,
        N_y: np.ndarray,
        N_z: np.ndarray,
        skin_mask: np.ndarray,
        effective_strength: float,
        azimuth: float,
        elevation: float,
        scale: float,
    ) -> np.ndarray:
        """Legacy v1 relighting: Blinn-Phong on LAB L channel with multiplicative diffuse.

        Args:
            canvas: BGR uint8 image.
            N_x, N_y, N_z: Normal fields at coarse scale.
            skin_mask: Soft float mask (0-1).
            effective_strength: Yaw-attenuated strength value.
            azimuth, elevation: Light direction in degrees.
            scale: Downsampling factor used in geometry computation.

        Returns:
            Relit canvas (BGR, uint8).
        """
        h, w = canvas.shape[:2]

        # Blinn-Phong Lighting Shader
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

        # Upsample shading fields back to original canvas resolution
        if scale < 1.0:
            I_diffuse = cv2.resize(I_diffuse, (w, h), interpolation=cv2.INTER_LINEAR)
            I_specular = cv2.resize(I_specular, (w, h), interpolation=cv2.INTER_LINEAR)

        # Composite onto LAB's Luminance (L) Channel
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

        # Blend using the soft skin mask
        mask_3d = skin_mask[:, :, np.newaxis] if skin_mask.ndim == 2 else skin_mask
        output = np.clip(canvas * (1.0 - mask_3d) + relit_canvas * mask_3d, 0, 255).astype(np.uint8)
        return output

    def _relight_v2(
        self,
        canvas: np.ndarray,
        N_x: np.ndarray,
        N_y: np.ndarray,
        N_z: np.ndarray,
        skin_mask: np.ndarray,
        effective_strength: float,
        azimuth: float,
        elevation: float,
        scale: float,
        face_width: float,
    ) -> np.ndarray:
        """v2 relighting: Decompose and recompose shading in linear RGB.

        Fits existing shading via least-squares, computes target shading from
        light direction, blends multiplicatively, and applies in linear RGB
        with highlight protection.

        Args:
            canvas: BGR uint8 image.
            N_x, N_y, N_z: Normal fields at coarse scale.
            skin_mask: Soft float mask (0-1).
            effective_strength: Yaw-attenuated strength value.
            azimuth, elevation: Light direction in degrees.
            scale: Downsampling factor used in geometry computation.
            face_width: Approximate face width in pixels.

        Returns:
            Relit canvas (BGR, uint8).
        """
        h, w = canvas.shape[:2]
        h_small = N_x.shape[0]
        w_small = N_x.shape[1]

        # 3a. Downsample canvas and compute linear luminance
        canvas_small = cv2.resize(canvas, (w_small, h_small), interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(canvas_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        Y = gray_small ** 2.2

        # 3b. Low-band via Gaussian blur
        sigma = max(2.0, face_width * scale / 8.0)
        Y_low = cv2.GaussianBlur(Y, (0, 0), sigmaX=sigma)

        # 3c. Downsample skin mask
        mask_small = cv2.resize(skin_mask, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        valid = (mask_small > 0.3) & (N_z < 0.999)

        # 3d. Fit existing shading via least squares
        if np.sum(valid) >= 50:
            # Design matrix: [1, N_x, N_y, N_z]
            Y_valid = Y_low[valid].reshape(-1, 1)
            design = np.column_stack([
                np.ones(np.sum(valid)),
                N_x[valid].flatten(),
                N_y[valid].flatten(),
                N_z[valid].flatten(),
            ])
            try:
                coeffs = np.linalg.lstsq(design, Y_valid, rcond=None)[0].flatten()
                c0, cx, cy, cz = coeffs
                norm_dir = np.sqrt(cx**2 + cy**2 + cz**2)
                if norm_dir < 1e-6:
                    # Fallback: uniform shading
                    S_old = np.ones((h_small, w_small), dtype=np.float32)
                else:
                    S_old = c0 + cx * N_x + cy * N_y + cz * N_z
                    S_old = np.clip(S_old, 0.15, 3.0)
                    S_old = S_old / np.mean(S_old[valid])
            except np.linalg.LinAlgError:
                S_old = np.ones((h_small, w_small), dtype=np.float32)
        else:
            S_old = np.ones((h_small, w_small), dtype=np.float32)

        # 3e. Target shading from light direction
        theta = azimuth * np.pi / 180.0
        phi = elevation * np.pi / 180.0
        L_x = np.cos(phi) * np.sin(theta)
        L_y = -np.sin(phi)
        L_z = np.cos(phi) * np.cos(theta)

        lambert = np.clip(N_x * L_x + N_y * L_y + N_z * L_z, 0.0, 1.0)
        S_new = 0.55 + 0.45 * lambert
        S_new = S_new / np.mean(S_new[valid])

        # 3f. Blend: multiplicative blend with strength cap
        t = np.clip((effective_strength / 100.0) * 0.6, 0.0, 0.6)
        S_final = (S_old ** (1.0 - t)) * (S_new ** t)
        gain = np.clip(S_final / np.maximum(S_old, 0.15), 0.55, 1.65)

        # 3g. Specular at coarse scale
        H_x = L_x
        H_y = L_y
        H_z = L_z + 1.0
        H_norm = np.sqrt(H_x**2 + H_y**2 + H_z**2) + 1e-5
        H_x /= H_norm
        H_y /= H_norm
        H_z /= H_norm
        I_specular = np.clip(N_x * H_x + N_y * H_y + N_z * H_z, 0.0, 1.0) ** self.alpha

        # 3h. Upsample gain and specular
        gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
        I_specular = cv2.resize(I_specular, (w, h), interpolation=cv2.INTER_LINEAR)

        # 3i. Apply in linear RGB
        canvas_lin = (canvas.astype(np.float32) / 255.0) ** 2.2
        gray_full = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
        prot = np.clip(1.0 - (gray_full - 220.0) / 30.0, 0.0, 1.0)

        # Reshape gain and prot for broadcasting: (h, w) -> (h, w, 1)
        gain_3d = gain[..., np.newaxis]
        prot_3d = prot[..., np.newaxis]

        out_lin = canvas_lin * (1.0 + (gain_3d - 1.0) * prot_3d)
        out_lin = out_lin + (effective_strength / 100.0) * 0.35 * I_specular[..., np.newaxis] * prot_3d
        out_lin = np.clip(out_lin, 0.0, 1.0)
        out = (out_lin ** (1.0 / 2.2) * 255.0).astype(np.uint8)

        # 3j. Blend with skin mask
        mask_3d = skin_mask[:, :, np.newaxis] if skin_mask.ndim == 2 else skin_mask
        output = np.clip(canvas * (1.0 - mask_3d) + out * mask_3d, 0, 255).astype(np.uint8)
        return output

    def relight(
        self,
        canvas: np.ndarray,
        landmarks: Any,
        skin_mask: Optional[np.ndarray],
        face_width: float,
        strength: float = 0.0,
        azimuth: float = 0.0,
        elevation: float = 30.0,
        engine: str = "v2",
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
            engine: Relighting algorithm ("v1" or "v2", default "v2").

        Returns:
            Relit canvas (BGR, uint8).
        """
        if strength <= 0.0 or skin_mask is None or skin_mask.max() < 0.01:
            return canvas

        if canvas.dtype == np.float32:
            # E1 delta adapter (see apply_u8_op_float).
            return apply_u8_op_float(
                canvas,
                self.relight,
                landmarks,
                skin_mask,
                face_width,
                strength=strength,
                azimuth=azimuth,
                elevation=elevation,
                engine=engine,
            )

        h, w = canvas.shape[:2]
        lm = landmarks.landmark

        # Temple-Calibrated Yaw Guard (applies to both engines)
        # lm[6] is nose bridge midpoint, lm[234] is left temple, lm[454] is right temple
        d_left = abs(lm[6].x - lm[234].x)
        d_right = abs(lm[454].x - lm[6].x)
        ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-5)
        # Attenuate strength linearly as ratio ranges from 1.5 to 1.7
        yaw_factor = 1.0 - np.clip((ratio - 1.5) / 0.2, 0.0, 1.0)

        effective_strength = strength * yaw_factor
        if effective_strength <= 0.0:
            return canvas

        # Extract shading geometry (common to both engines)
        N_x, N_y, N_z, w_small, h_small, scale = self._shading_geometry(
            (h, w), landmarks, face_width
        )

        if engine == "v1":
            return self._relight_v1(
                canvas, N_x, N_y, N_z, skin_mask,
                effective_strength, azimuth, elevation, scale
            )
        else:  # v2 (default)
            return self._relight_v2(
                canvas, N_x, N_y, N_z, skin_mask,
                effective_strength, azimuth, elevation, scale, face_width
            )

    def sculpt(
        self,
        canvas: np.ndarray,
        landmarks: Any,
        skin_mask: Optional[np.ndarray],
        face_width: float,
        strength: float = 0.0,
        light_azimuth: Optional[float] = None,
        light_elevation: Optional[float] = None,
    ) -> np.ndarray:
        """Sculpt facial structure by shaping reflectance (low-band luminance modulation).

        This stage complements relight() by sculpting the form frequency (cheekbone depth,
        nose ridge, jawline contour) without changing illumination direction. It applies
        a shading correction to the low-band (form) frequency only, preserving micro-texture
        (pores, blotches).

        Algorithm:
        1. Extract surface normals from landmarks via Delaunay triangulation.
        2. Compute target shading from a light direction (auto-estimated from L gradient or explicit).
        3. Compute low-band correction: (target_shading - 1.0) * strength * skin_mask.
        4. Apply correction to the Gaussian-blurred (low-band) L channel.
        5. Re-add the untouched high-frequency residual to preserve texture.
        6. Fade correction via yaw guard for profile faces (reuses relight() logic).

        Args:
            canvas: Crop ROI image (BGR, uint8).
            landmarks: Shifted face landmarks relative to crop ROI canvas.
            skin_mask: Soft float mask (0-1) for face skin area.
            face_width: Approximate face width in pixels (ied * 2.5).
            strength: Slider strength (0-100), default 0 (no effect).
            light_azimuth: Light angle in degrees (-180 to 180), or None for auto-estimate.
            light_elevation: Light elevation angle in degrees (-90 to 90), or None for auto-estimate.

        Returns:
            Sculpted canvas (BGR, uint8).

        Notes:
            - Early return if strength <= 0, skin_mask is None, or skin_mask is nearly empty.
            - Yaw guard attenuates strength for profile faces (temple ratio 1.5→1.7 fade).
            - Strength scaling: effective correction is (strength / 100) * 0.35 to keep sculpting subtle.
            - Target shading uses Lambertian model: S_target = 0.55 + 0.45 * max(N·L, 0).
            - Auto light direction estimated from blurred L-channel gradient direction.
        """
        if strength <= 0.0 or skin_mask is None or skin_mask.max() < 0.01:
            return canvas

        if canvas.dtype == np.float32:
            # E1 delta adapter (see apply_u8_op_float).
            return apply_u8_op_float(
                canvas,
                self.sculpt,
                landmarks,
                skin_mask,
                face_width,
                strength=strength,
                light_azimuth=light_azimuth,
                light_elevation=light_elevation,
            )

        h, w = canvas.shape[:2]
        lm = landmarks.landmark

        # Temple-Calibrated Yaw Guard (identical to relight)
        # lm[6] is nose bridge midpoint, lm[234] is left temple, lm[454] is right temple
        d_left = abs(lm[6].x - lm[234].x)
        d_right = abs(lm[454].x - lm[6].x)
        ratio = max(d_left, d_right) / (min(d_left, d_right) + 1e-5)
        # Attenuate strength linearly as ratio ranges from 1.5 to 1.7
        yaw_factor = 1.0 - np.clip((ratio - 1.5) / 0.2, 0.0, 1.0)

        effective_strength = strength * yaw_factor
        if effective_strength <= 0.0:
            return canvas

        # Extract shading geometry
        N_x, N_y, N_z, w_small, h_small, scale = self._shading_geometry(
            (h, w), landmarks, face_width
        )

        # Downsample canvas for coarse computation and extract L channel
        canvas_small = cv2.resize(canvas, (w_small, h_small), interpolation=cv2.INTER_AREA)
        gray_small = cv2.cvtColor(canvas_small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        Y = gray_small ** 2.2  # Linear luminance

        # Low-band via Gaussian blur at form-frequency scale
        sigma = max(2.0, face_width * scale / 8.0)  # Same sigma as relight v2
        Y_low = cv2.GaussianBlur(Y, (0, 0), sigmaX=sigma)

        # Downsample skin mask
        mask_small = cv2.resize(skin_mask, (w_small, h_small), interpolation=cv2.INTER_LINEAR)
        valid = (mask_small > 0.3) & (N_z < 0.999)

        # Auto-estimate light direction if not provided
        if light_azimuth is None or light_elevation is None:
            # Fit existing shading via least squares (reuses relight v2's approach)
            if np.sum(valid) >= 50:
                Y_valid = Y_low[valid].reshape(-1, 1)
                design = np.column_stack([
                    np.ones(np.sum(valid)),
                    N_x[valid].flatten(),
                    N_y[valid].flatten(),
                    N_z[valid].flatten(),
                ])
                try:
                    coeffs = np.linalg.lstsq(design, Y_valid, rcond=None)[0].flatten()
                    c0, cx, cy, cz = coeffs
                    norm_dir = np.sqrt(cx**2 + cy**2 + cz**2)
                    if norm_dir >= 1e-6:
                        # Fitted (cx, cy, cz) is the light direction: convert to angles
                        light_azimuth = np.arctan2(cx, cz) * 180.0 / np.pi
                        light_elevation = -np.arcsin(np.clip(cy, -1.0, 1.0)) * 180.0 / np.pi
                    else:
                        # Fallback: frontal light
                        light_azimuth = 0.0
                        light_elevation = 30.0
                except np.linalg.LinAlgError:
                    light_azimuth = 0.0
                    light_elevation = 30.0
            else:
                light_azimuth = 0.0
                light_elevation = 30.0

        # Compute light direction vector from azimuth/elevation
        theta = light_azimuth * np.pi / 180.0
        phi = light_elevation * np.pi / 180.0
        L_x = np.cos(phi) * np.sin(theta)
        L_y = -np.sin(phi)
        L_z = np.cos(phi) * np.cos(theta)

        # Target shading using Lambertian model (matching relight v2)
        lambert = np.clip(N_x * L_x + N_y * L_y + N_z * L_z, 0.0, 1.0)
        S_target = 0.55 + 0.45 * lambert
        S_target = S_target / np.mean(S_target[valid])

        # Compute sculpting gain at coarse scale
        # Sculpt uses a conservative multiplicative gain to preserve flat areas
        strength_factor = (effective_strength / 100.0) * 0.35
        # Blend toward target shading, but bounded to avoid over-correction
        gain = np.clip(S_target, 0.7, 1.3)
        gain_final = gain ** strength_factor

        # Upsample gain back to full resolution
        gain_full = cv2.resize(gain_final, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_LINEAR)

        # Apply gain correction to full-res image in LAB space
        lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_full = lab[:, :, 0]

        # Decompose into low-band and high-band at full resolution
        # Form-frequency scale per spec: face_width * 0.15
        L_full_norm = L_full / 255.0
        blur_sigma = face_width * 0.15
        L_low_full_norm = cv2.GaussianBlur(L_full_norm, (0, 0), sigmaX=blur_sigma)
        L_high_norm = L_full_norm - L_low_full_norm

        # Apply multiplicative gain only to low-band
        L_low_corrected_norm = np.clip(L_low_full_norm * gain_full, 0.0, 1.0)

        # Reconstruct: corrected low-band + original high-band
        L_result_norm = np.clip(L_low_corrected_norm + L_high_norm, 0.0, 1.0)
        L_result = L_result_norm * 255.0

        # Blend with skin mask
        L_final = L_full * (1.0 - mask_full) + L_result * mask_full
        lab[:, :, 0] = np.clip(L_final, 0.0, 255.0)

        result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return result
