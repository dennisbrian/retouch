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

import os
import cv2
import numpy as np

from .utils import adaptive_ksize


class FrequencyLayers:
    """Container for the three frequency bands."""
    __slots__ = ["low", "mid", "high"]

    def __init__(self, low, mid, high):
        self.low = low    # float32, [0, 255]
        self.mid = mid    # float32, can be negative
        self.high = high  # float32, can be negative

    def reconstruct(self):
        """Return low + mid + high as uint8."""
        return np.clip(self.low + self.mid + self.high, 0, 255).astype(np.uint8)


def separate(img_bgr, face_width):
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


def combine(layers, skin_mask=None, smooth_strength=0.5,
            mid_reduction=0.4, texture_opacity=1.0, face_width=None):
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

    Returns:
        (H, W, 3) uint8 BGR result.
    """
    if skin_mask is None:
        return layers.reconstruct()

    low = layers.low.copy()
    mid_original = layers.mid
    mid = mid_original.copy()
    high = layers.high.copy()

    texture_opacity = max(0.0, min(1.0, texture_opacity))

    m_raw = skin_mask.astype(np.float32)

    if face_width:
        # | 1 forces odd kernel size (GaussianBlur requirement) (Issue 12)
        feather_r = max(5, int(face_width * 0.015) | 1)
    else:
        h, w = layers.low.shape[:2]
        feather_r = max(5, int(min(h, w) * 0.0075) | 1)

    m = cv2.GaussianBlur(m_raw, (feather_r, feather_r), 0)
    m = m[:, :, np.newaxis]

    # ---- Build the processed result inside the mask ----
    # Reduce mid layer (remove blemishes/wrinkles)
    if mid_reduction > 0:
        mid = mid * (1.0 - m * mid_reduction)

    # Smooth low + mid layers (even out colour/tone transitions)
    if smooth_strength > 0:
        # 1. Soft Gaussian blur on Low layer for perfectly clean gradients (no bilateral blotches)
        if face_width:
            k_smooth = adaptive_ksize(face_width, factor=0.22, minimum=9)
        else:
            h, w = layers.low.shape[:2]
            approx_face_width = min(h, w) * 0.4
            k_smooth = adaptive_ksize(approx_face_width, factor=0.22, minimum=9)
        smoothed_low_gaussian = cv2.GaussianBlur(low, (k_smooth, k_smooth), 0)

        # Scale d proportionally to face size (Issue 4)
        f_width = face_width if face_width else (min(layers.low.shape[0], layers.low.shape[1]) * 0.4)
        d = max(9, adaptive_ksize(f_width, factor=0.006, minimum=9))
        low_mid_u8 = np.clip(low + mid_original, 0, 255).astype(np.uint8)
        sigma_color = 20.0 + smooth_strength * 60.0
        sigma_space = 20.0 + smooth_strength * 60.0
        smoothed_u8 = cv2.bilateralFilter(low_mid_u8, d, sigma_color, sigma_space)
        smoothed_low_mid = smoothed_u8.astype(np.float32)
        smoothed_low_bilateral = smoothed_low_mid - mid_original

        # 3. Hybrid blend: higher smooth_strength uses slightly more Gaussian blur, but bilateral remains dominant
        blend_gaussian = min(1.0, smooth_strength * 0.25)
        smoothed_low_final = smoothed_low_bilateral * (1.0 - blend_gaussian) + smoothed_low_gaussian * blend_gaussian

        low = low * (1.0 - m) + smoothed_low_final * m

    # Texture opacity
    if texture_opacity < 1.0:
        high = high * (1.0 - m * (1.0 - texture_opacity))

    processed = low + mid + high
    original = layers.low + layers.mid + layers.high

    # Composite on final pixel values to avoid tonal edge artifacts (Issue 1)
    result = original * (1.0 - m) + processed * m
    return np.clip(result, 0, 255).astype(np.uint8)
