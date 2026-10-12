"""Body-skin relighting and dodge/burn — landmark-free sculpting for exposed
arms/legs/chest/décolletage.

Separate module by design: face relighting (relight.py) depends on FaceMesh
468-point Delaunay triangulation, which has no body equivalent. This module
instead derives a coarse local-luminance normal approximation directly from
the image, so it works on any body-skin mask without pose/body landmarks.
Kept isolated from engine.py's existing _stage_body_skin call sites — it is
opt-in via new ParamSpecs and does not alter current body_skin behavior
(smooth/equalize/whiten/match_face) unless explicitly enabled in a recipe.
"""

from __future__ import annotations

import cv2
import numpy as np

from .utils import (
    bgr_f32_to_lab_f32,
    blend_masked,
    lab_f32_to_bgr_f32,
)


class BodyRelighter:
    """Directional shading and micro dodge/burn scoped to a body-skin mask."""

    def relight(
        self,
        img_bgr: np.ndarray,
        body_skin_mask: np.ndarray,
        strength: float = 0,
        azimuth: float = 45.0,
        elevation: float = 45.0,
    ) -> np.ndarray:
        """Apply directional relighting to body skin using a luminance-derived
        normal approximation (no landmarks required).

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            body_skin_mask: (H, W) float32 mask in [0, 1] — where to apply.
            strength: 0-100 relight intensity.
            azimuth: Light direction in degrees (0 = right, 90 = up).
            elevation: Light elevation in degrees (0 = grazing, 90 = overhead).

        Returns:
            Image of the same dtype/shape as input.
        """
        if strength <= 0 or body_skin_mask is None or body_skin_mask.max() < 0.01:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        s = strength / 100.0
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # Coarse luminance surface used as a pseudo-height-field: blur heavily
        # so gradients reflect large-scale body form (limb curvature, torso
        # roundness) rather than skin texture/pores.
        blur_radius = max(9, int(min(img_bgr.shape[:2]) * 0.03) | 1)
        height_field = cv2.GaussianBlur(gray, (blur_radius, blur_radius), 0)

        gx = cv2.Sobel(height_field, cv2.CV_32F, 1, 0, ksize=5)
        gy = cv2.Sobel(height_field, cv2.CV_32F, 0, 1, ksize=5)

        az = np.radians(azimuth)
        el = np.radians(elevation)
        light_x = np.cos(el) * np.cos(az)
        light_y = np.cos(el) * np.sin(az)
        light_z = np.sin(el)

        # Approximate surface normal from gradients (unit z-dominant field),
        # then take dot product with light direction as a shading term.
        norm_len = np.sqrt(gx * gx + gy * gy + 1.0)
        n_x = -gx / norm_len
        n_y = -gy / norm_len
        n_z = 1.0 / norm_len

        shading = n_x * light_x + n_y * light_y + n_z * light_z
        shading = np.clip(shading, -1.0, 1.0)
        # Center so mean shading is neutral (avoid a global brightness shift).
        shading -= shading.mean()

        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_val = lab[:, :, 0]

        protection = np.clip(1.0 - (l_val - 220.0) / 30.0, 0.0, 1.0)
        # Max shading delta raised 18.0 -> 32.0 — the original magnitude
        # was too subtle to visually approach the face pipeline's combined
        # relight+dodge_burn+frequency-separation intensity, leaving body
        # skin looking comparatively untouched even at strength=100.
        delta = shading * 32.0 * s * protection
        lab[:, :, 0] = np.clip(l_val + delta, 0, 255)

        if is_float:
            result = lab_f32_to_bgr_f32(lab)
        else:
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        # Blend weight raised (was body_skin_mask * s, capping the visible
        # opacity at s itself) — full mask opacity now, strength only
        # controls the shading magnitude above, not a second multiplier.
        return blend_masked(img_bgr, result, body_skin_mask)

    def dodge_burn(
        self,
        img_bgr: np.ndarray,
        body_skin_mask: np.ndarray,
        strength: float = 0,
    ) -> np.ndarray:
        """Subtle local-contrast sculpting on body skin: brighten local
        highlights, deepen local shadows, scaled by highlight protection.

        Mirrors skin.py's face dodge_burn in spirit (same protection curve,
        similar magnitude) but derives its brighten/darken zones from local
        luminance contrast (CLAHE-vs-original delta) instead of face
        landmark sub-masks, since body has none.

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            body_skin_mask: (H, W) float32 mask in [0, 1].
            strength: 0-100 dodge & burn intensity.

        Returns:
            Image of the same dtype/shape as input.
        """
        if strength <= 0 or body_skin_mask is None or body_skin_mask.max() < 0.01:
            return img_bgr

        is_float = img_bgr.dtype == np.float32
        s = strength / 100.0
        if is_float:
            lab = bgr_f32_to_lab_f32(img_bgr)
        else:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_val = lab[:, :, 0]

        l_u8 = np.clip(l_val, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        l_local = clahe.apply(l_u8).astype(np.float32)

        # Local-contrast delta: positive where CLAHE brightens (local
        # highlight), negative where it darkens (local shadow).
        local_delta = l_local - l_val

        protection = np.clip(1.0 - (l_val - 220.0) / 30.0, 0.0, 1.0)
        # Local-contrast multiplier raised 0.5 -> 1.0 to match face-side
        # dodge_burn's visible sculpting strength (see skin.py dodge_burn).
        lab[:, :, 0] = np.clip(l_val + local_delta * 1.0 * s * protection, 0, 255)

        if is_float:
            result = lab_f32_to_bgr_f32(lab)
        else:
            result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, result, body_skin_mask)
