"""3-level frequency separation — the core of professional retouching.

Splits an image into three frequency bands:

    Low  — colour / overall tone  (large-scale)
    Mid  — blemishes, wrinkles    (medium-scale)
    High — pores, fine hairs      (small-scale)

Processing strategy:
    • Smooth the LOW layer to even out skin tone.
    • Reduce the MID layer to remove blemishes.
    • Keep the HIGH layer mostly untouched → natural texture preserved.
    • Recombine: Result = SmoothedLow + ReducedMid + High.

Key design choice: compositing is done on the **final reconstructed pixel**
values, not on individual layers. This avoids tonal discontinuities at
feathered mask boundaries where different layers have been modified
by different amounts.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

from .utils import adaptive_ksize, blend_masked, estimate_face_width, guided_filter

# Constants for adaptive sizing and parameters
DEFAULT_FEATHER_FACTOR = 0.015
FALLBACK_FEATHER_FACTOR = 0.0075
DEFAULT_FEATHER_MIN = 5
APPROX_FACE_WIDTH_RATIO = 0.4
SMOOTH_K_FACTOR = 0.22
SMOOTH_K_MIN = 9
SIGMA_BASE = 20.0
SIGMA_STRENGTH_FACTOR = 60.0
GAUSSIAN_BLEND_FACTOR = 0.25


class FrequencyLayers:
    """Container for the three frequency bands."""
    __slots__ = ["low", "mid", "high"]

    def __init__(self, low: np.ndarray, mid: np.ndarray, high: np.ndarray) -> None:
        self.low = low    # float32, [0, 255]
        self.mid = mid    # float32, can be negative
        self.high = high  # float32, can be negative

    def reconstruct(self) -> np.ndarray:
        """Return low + mid + high as uint8.

        Returns:
            (H, W, 3) uint8 BGR image.
        """
        return np.clip(self.low + self.mid + self.high, 0, 255).astype(np.uint8)


class FrequencySeparator:
    """3-level frequency separation stage.

    Splits an image into three frequency bands:

        Low  — colour / overall tone  (large-scale)
        Mid  — blemishes, wrinkles    (medium-scale)
        High — pores, fine hairs      (small-scale)

    Processing strategy:
        • Smooth the LOW layer to even out skin tone.
        • Reduce the MID layer to remove blemishes.
        • Keep the HIGH layer mostly untouched → natural texture preserved.
        • Recombine: Result = SmoothedLow + ReducedMid + High.

    Key design choice: compositing is done on the **final reconstructed pixel**
    values, not on individual layers. This avoids tonal discontinuities at
    feathered mask boundaries where different layers have been modified
    by different amounts.

    The class is stateless; instances are cheap to create and may be shared
    across threads safely.
    """

    def separate(self, img_bgr: np.ndarray, face_width: float) -> "FrequencyLayers":
        """Split image into low / mid / high frequency layers.

        Args:
            img_bgr: (H, W, 3) uint8 BGR image.
            face_width: Approximate face width in pixels (used to size kernels).
                        Typically inter_eye_distance * 2.5.

        Returns:
            FrequencyLayers with .low, .mid, .high (all float32).
        """
        img_f = img_bgr.astype(np.float32)

        k_low = adaptive_ksize(face_width, factor=0.12, minimum=5)
        k_mid = adaptive_ksize(face_width, factor=0.04, minimum=3)

        low = cv2.GaussianBlur(img_f, (k_low, k_low), 0)
        med_blur = cv2.GaussianBlur(img_f, (k_mid, k_mid), 0)

        mid = med_blur - low
        high = img_f - med_blur

        if os.getenv("FREQ_DEBUG"):
            reconstruction = low + mid + high
            err = np.abs(reconstruction - img_f).max()
            assert err < 1e-3, f"Separation lossy: max_err={err:.4f}"

        return FrequencyLayers(low, mid, high)

    def combine(
        self,
        layers: "FrequencyLayers",
        skin_mask: Optional[np.ndarray] = None,
        smooth_strength: float = 0.5,
        mid_reduction: float = 0.4,
        texture_opacity: float = 1.0,
        face_width: Optional[float] = None,
        pore_synthesis: float = 0.0,
        roi_coords: Optional[Tuple[int, int]] = None,
        smooth_engine: str = "guided",
        float32_out: bool = False,
    ) -> np.ndarray:
        """Re-combine layers after selective processing.

        Compositing is done on final pixel values to avoid tonal discontinuities
        at feathered mask boundaries.

        Args:
            layers: FrequencyLayers from separate().
            skin_mask: (H, W) float mask. Processing applied only here.
            smooth_strength: How much additional smoothing on the low layer (0–1).
            mid_reduction: How much to reduce mid-frequency (0 = keep, 1 = remove).
            texture_opacity: High-frequency reprojection opacity (0–1).
                             Recommended range 0.4–1.0. Lower = smoother/waxier.
            face_width: Face width in pixels (for adaptive mask feathering).
            pore_synthesis: Strength of micro-pore texture synthesis (0-1).
            roi_coords: Tuple of (roi_x1, roi_y1) coordinates for deterministic spatial seeding.
            smooth_engine: Smoothing method: "guided" (guided filter, default) or "bilateral" (legacy).
            float32_out: If True, return float32 [0, 255] instead of uint8. Default False.

        Returns:
            (H, W, 3) uint8 BGR result, or float32 if float32_out=True.
        """
        if skin_mask is None:
            result = layers.reconstruct()
            if float32_out:
                return result.astype(np.float32)
            return result

        m_raw = skin_mask.astype(np.float32)

        # --- Bounding Box Optimization ---
        # Crop to the mask region to avoid running heavy ops (like bilateral filter)
        # on the entire image when the skin only covers a small fraction.
        ys, xs = np.where(m_raw > 0.01)
        if len(xs) == 0:
            result = layers.reconstruct()
            if float32_out:
                return result.astype(np.float32)
            return result

        fw_approx = face_width if face_width else estimate_face_width(img_shape=layers.low.shape[:2], fallback_ratio=APPROX_FACE_WIDTH_RATIO)
        pad = max(10, int(fw_approx * 0.1))
        x1, x2 = max(0, xs.min() - pad), min(m_raw.shape[1], xs.max() + pad + 1)
        y1, y2 = max(0, ys.min() - pad), min(m_raw.shape[0], ys.max() + pad + 1)

        low = layers.low[y1:y2, x1:x2].copy()
        mid_original = layers.mid[y1:y2, x1:x2]
        mid = mid_original.copy()
        high = layers.high[y1:y2, x1:x2].copy()
        m_raw = m_raw[y1:y2, x1:x2]

        texture_opacity = max(0.0, min(1.0, texture_opacity))

        if face_width:
            feather_r = max(DEFAULT_FEATHER_MIN, int(face_width * DEFAULT_FEATHER_FACTOR) | 1)
        else:
            h_full, w_full = layers.low.shape[:2]
            feather_r = max(DEFAULT_FEATHER_MIN, int(min(h_full, w_full) * FALLBACK_FEATHER_FACTOR) | 1)

        m_2d = cv2.GaussianBlur(m_raw, (feather_r, feather_r), 0)
        m_3d = m_2d[:, :, np.newaxis]

        # Texture opacity — attenuate high band inside the mask
        if texture_opacity < 1.0:
            high = high * (1.0 - m_3d * (1.0 - texture_opacity))

        # Pore synthesis — inject synthetic high-frequency noise
        if face_width and roi_coords and pore_synthesis > 0:
            roi_x1, roi_y1 = roi_coords
            seed = abs(hash((int(face_width * 100), roi_x1, roi_y1))) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)
            h, w = low.shape[:2]

            # Generate 2D (1-channel) noise to avoid chroma noise.
            noise = rng.standard_normal((h, w)).astype(np.float32) * 15.0

            sigma = max(0.5, face_width / 120.0)
            noise_blur = cv2.GaussianBlur(noise, (0, 0), sigma)

            P = (noise - noise_blur)[:, :, np.newaxis]
            high = high + (pore_synthesis * P * m_3d)

        # Reduce mid layer
        if mid_reduction > 0:
            mid = mid * (1.0 - m_3d * mid_reduction)

        # Early exit for no-smoothing case
        if smooth_strength <= 0:
            processed_crop = low + mid + high
            orig_crop = layers.low[y1:y2, x1:x2] + layers.mid[y1:y2, x1:x2] + layers.high[y1:y2, x1:x2]
            result_crop = blend_masked(orig_crop, processed_crop, m_2d)

            if float32_out:
                # E1 float path: keep the full result float32 — no uint8 paste target.
                full_result = np.clip(layers.low + layers.mid + layers.high, 0, 255).astype(np.float32)
                full_result[y1:y2, x1:x2] = np.clip(result_crop, 0, 255)
                return full_result
            full_result = layers.reconstruct()
            full_result[y1:y2, x1:x2] = result_crop
            return full_result

        # Smooth low + mid layers
        f_width = face_width if face_width else fw_approx
        k_smooth = adaptive_ksize(f_width, factor=SMOOTH_K_FACTOR, minimum=SMOOTH_K_MIN)
        smoothed_low_gaussian = cv2.GaussianBlur(low, (k_smooth, k_smooth), 0)

        # Smoothing filter (guided or bilateral)
        # Keep in float32 to avoid quantization banding on gradients (uint8 artifacts).
        low_mid_f32 = np.clip(low + mid_original, 0, 255)
        sigma_color = SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR
        sigma_space = SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR

        if smooth_engine == "guided":
            # Guided filter: radius from sigma_space, eps from sigma_color^2
            radius = max(2, int(round(sigma_space)))
            eps = sigma_color ** 2

            # Apply guided filter to each channel separately (self-guided)
            smoothed_f32 = np.zeros_like(low_mid_f32)
            for c in range(low_mid_f32.shape[2]):
                smoothed_f32[:, :, c] = guided_filter(
                    low_mid_f32[:, :, c],
                    radius=radius,
                    eps=eps,
                    guide=None,  # self-guided
                    max_dim=None  # crops are already bounded
                )
            smoothed_low_bilateral = smoothed_f32 - mid_original
        else:
            # Bilateral filter (legacy path)
            # Use d=-1 to let OpenCV compute an optimal, efficient kernel size.
            smoothed_f32 = cv2.bilateralFilter(low_mid_f32, -1, sigma_color, sigma_space)
            smoothed_low_bilateral = smoothed_f32 - mid_original

        # Hybrid blend (cv2.addWeighted is slightly faster and purely SIMD optimized)
        blend_gaussian = min(1.0, smooth_strength * GAUSSIAN_BLEND_FACTOR)
        smoothed_low_final = cv2.addWeighted(
            smoothed_low_bilateral, 1.0 - blend_gaussian,
            smoothed_low_gaussian, blend_gaussian,
            0.0
        )

        low = low * (1.0 - m_3d) + smoothed_low_final * m_3d

        # Composite on final pixel values
        processed_crop = low + mid + high
        orig_crop = layers.low[y1:y2, x1:x2] + layers.mid[y1:y2, x1:x2] + layers.high[y1:y2, x1:x2]
        result_crop = blend_masked(orig_crop, processed_crop, m_2d)

        if float32_out:
            # E1 float path: keep the full result float32 — no uint8 paste target.
            full_result = np.clip(layers.low + layers.mid + layers.high, 0, 255).astype(np.float32)
            full_result[y1:y2, x1:x2] = np.clip(result_crop, 0, 255)
            return full_result

        # Paste back into the full image
        full_result = layers.reconstruct()
        full_result[y1:y2, x1:x2] = result_crop
        return full_result


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (DEPRECATED — use FrequencySeparator)
# ---------------------------------------------------------------------------

def separate(img_bgr: np.ndarray, face_width: float) -> "FrequencyLayers":
    """Deprecated. Use ``FrequencySeparator().separate()`` instead."""
    return FrequencySeparator().separate(img_bgr, face_width)


def combine(
    layers: "FrequencyLayers",
    skin_mask: Optional[np.ndarray] = None,
    smooth_strength: float = 0.5,
    mid_reduction: float = 0.4,
    texture_opacity: float = 1.0,
    face_width: Optional[float] = None,
    pore_synthesis: float = 0.0,
    roi_coords: Optional[Tuple[int, int]] = None,
    smooth_engine: str = "guided",
) -> np.ndarray:
    """Deprecated. Use ``FrequencySeparator().combine()`` instead."""
    return FrequencySeparator().combine(
        layers,
        skin_mask=skin_mask,
        smooth_strength=smooth_strength,
        mid_reduction=mid_reduction,
        texture_opacity=texture_opacity,
        face_width=face_width,
        pore_synthesis=pore_synthesis,
        roi_coords=roi_coords,
        smooth_engine=smooth_engine,
    )
