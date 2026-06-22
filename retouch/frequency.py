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

from .utils import adaptive_ksize, blend_masked

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
            mid_reduction=0.4, texture_opacity=1.0, face_width=None,
            pore_synthesis=0.0, roi_coords=None):
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

    Returns:
        (H, W, 3) uint8 BGR result.
    """
    if skin_mask is None:
        return layers.reconstruct()

    m_raw = skin_mask.astype(np.float32)

    # --- Bounding Box Optimization ---
    # Crop to the mask region to avoid running heavy ops (like bilateral filter)
    # on the entire image when the skin only covers a small fraction.
    ys, xs = np.where(m_raw > 0.01)
    if len(xs) == 0:
        return layers.reconstruct()

    fw_approx = face_width if face_width else (min(layers.low.shape[:2]) * APPROX_FACE_WIDTH_RATIO)
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

        full_result = layers.reconstruct()
        full_result[y1:y2, x1:x2] = result_crop
        return full_result

    # Smooth low + mid layers
    f_width = face_width if face_width else fw_approx
    k_smooth = adaptive_ksize(f_width, factor=SMOOTH_K_FACTOR, minimum=SMOOTH_K_MIN)
    smoothed_low_gaussian = cv2.GaussianBlur(low, (k_smooth, k_smooth), 0)

    # Bilateral Filter
    # Keep in float32 to avoid quantization banding on gradients (uint8 artifacts).
    low_mid_f32 = np.clip(low + mid_original, 0, 255)
    sigma_color = SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR
    sigma_space = SIGMA_BASE + smooth_strength * SIGMA_STRENGTH_FACTOR

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

    # Paste back into the full image
    full_result = layers.reconstruct()
    full_result[y1:y2, x1:x2] = result_crop

    return full_result


class FrequencySeparator:
    """Class-based wrapper around ``separate()`` / ``combine()`` for engine use.

    Provides the same behavior as the module-level functions; the class
    shape exists so it matches the rest of the stage modules in the engine
    (e.g. ``SkinProcessor``, ``BlemishRemover``).
    """

    def separate(self, img_bgr, face_width):
        return separate(img_bgr, face_width)

    def combine(self, layers, skin_mask=None, smooth_strength=0.5,
                mid_reduction=0.4, texture_opacity=1.0, face_width=None,
                pore_synthesis=0.0, roi_coords=None):
        return combine(
            layers,
            skin_mask=skin_mask,
            smooth_strength=smooth_strength,
            mid_reduction=mid_reduction,
            texture_opacity=texture_opacity,
            face_width=face_width,
            pore_synthesis=pore_synthesis,
            roi_coords=roi_coords,
        )
