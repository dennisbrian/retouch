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
    low = layers.low.copy()
    mid = layers.mid.copy()
    high = layers.high.copy()

    texture_opacity = max(0.0, min(1.0, texture_opacity))

    if skin_mask is None:
        return layers.reconstruct()

    m_raw = skin_mask.astype(np.float32)

    if face_width:
        # | 1 forces odd kernel size (GaussianBlur requirement)
        feather_r = max(3, int(face_width * 0.01) | 1)
    else:
        h, w = layers.low.shape[:2]
        feather_r = max(3, int(min(h, w) * 0.005) | 1)

    m = cv2.GaussianBlur(m_raw, (feather_r, feather_r), 0)
    m = m[:, :, np.newaxis]

    # ---- Build the processed result inside the mask ----
    # Reduce mid layer (remove blemishes/wrinkles)
    if mid_reduction > 0:
        mid = mid * (1.0 - m * mid_reduction)

    # Smooth low + mid layers (even out colour/tone transitions)
    if smooth_strength > 0:
        low_mid = np.clip(low + mid, 0, 255).astype(np.uint8)
        d = 9  # fixed diameter; sigma values below control strength
        sigma_color = int(20 + smooth_strength * 60)
        sigma_space = int(20 + smooth_strength * 60)
        smoothed_low_mid = cv2.bilateralFilter(low_mid, d, sigma_color, sigma_space).astype(np.float32)
        low = low * (1.0 - m) + (smoothed_low_mid - mid) * m

    # Texture opacity
    if texture_opacity < 1.0:
        high = high * (1.0 - m * (1.0 - texture_opacity))

    processed = low + mid + high
    original = layers.reconstruct().astype(np.float32)

    # Composite on final pixel values to avoid tonal edge artifacts
    result = original * (1.0 - m) + processed * m
    return np.clip(result, 0, 255).astype(np.uint8)
