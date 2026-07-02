"""Quality Assurance detectors for output self-QA.

Provides pure-function artifact detectors that analyze output images for:
  - Banding (posterization on smooth gradients)
  - Clipping (blown highlights and crushed blacks)
  - Plastic skin (over-smoothed texture loss in skin regions)
  - Halo (edge overshoot from aggressive sharpening)

Each detector returns a dict with:
  - "score": float in [0, 1] (higher = more problematic)
  - "flagged": bool (True if score exceeds threshold)
  - Additional detail keys specific to each detector
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


# Threshold constants — adjust based on tuning
BANDING_THRESHOLD = 0.15  # Flag if >15% of smooth pixels on quantization step edges
CLIPPING_THRESHOLD = 0.08  # Flag if clipped blob fraction >8% of image
PLASTIC_SKIN_THRESHOLD = 0.60  # Flag if high-freq energy ratio falls below 60%
HALO_THRESHOLD = 15.0  # Flag if mean overshoot amplitude >15 levels


def _bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8 image to CIELab float32.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.

    Returns:
        (H, W, 3) float32 Lab image, L in [0, 100], a/b in [-128, 128].
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab)
    return img_lab


def detect_banding(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Detect quantization banding on smooth gradients.

    Identifies regions of low local variance (smooth gradients) and measures
    the fraction of pixels that lie on abrupt 1-level step contours in the
    L channel, indicating posterization artifacts.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        mask: Optional (H, W) float mask [0, 1] to restrict analysis.

    Returns:
        dict with keys:
            - "score": float, fraction of smooth-region pixels on step edges
            - "flagged": bool, True if score > BANDING_THRESHOLD
            - "smooth_pixels": int, count of low-variance pixels
            - "step_pixels": int, count of pixels on quantization steps
    """
    if img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
        # Tiny image; cannot measure banding reliably
        return {"score": 0.0, "flagged": False, "smooth_pixels": 0, "step_pixels": 0}

    # Convert to Lab and extract L channel
    img_lab = _bgr_to_lab(img_bgr)
    L = img_lab[:, :, 0]  # float32, [0, 100]

    # Identify smooth (low-variance) regions via 3x3 local std
    # Compute local variance in sliding window
    h, w = L.shape
    local_var = np.zeros((h, w), dtype=np.float32)

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            patch = L[y - 1:y + 2, x - 1:x + 2]
            local_var[y, x] = np.var(patch)

    # Smooth regions: local variance < 0.5 (threshold empirical)
    smooth_mask = local_var < 0.5

    # Restrict to provided mask if given
    if mask is not None:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        smooth_mask = smooth_mask & (mask_f > 0.5)

    if smooth_mask.sum() == 0:
        # No smooth regions found
        return {"score": 0.0, "flagged": False, "smooth_pixels": 0, "step_pixels": 0}

    # Measure gradient magnitude in L channel (Sobel)
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # In smooth regions, high gradient magnitude is suspicious (quantization step).
    # Threshold: gradient > 0.8 (on 0-100 L scale)
    step_mask = (grad_mag > 0.8) & smooth_mask

    smooth_count = int(smooth_mask.sum())
    step_count = int(step_mask.sum())
    score = step_count / max(smooth_count, 1)
    flagged = score > BANDING_THRESHOLD

    return {
        "score": min(1.0, float(score)),
        "flagged": bool(flagged),
        "smooth_pixels": smooth_count,
        "step_pixels": step_count,
    }


def detect_clipping(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Detect blown highlights and crushed blacks.

    Measures the fraction of pixels at 0 or 255 per channel, and identifies
    large connected blobs of clipped pixels (blown-white clusters).

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        mask: Optional (H, W) float mask [0, 1] to restrict analysis.

    Returns:
        dict with keys:
            - "score": float, max of (% pixels at 0/255) or (blob fraction)
            - "flagged": bool, True if blob fraction > CLIPPING_THRESHOLD
            - "clipped_pixel_fraction": float, % of 0/255 pixels
            - "clipped_blob_fraction": float, % of image in clipped blobs
    """
    if img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
        # Tiny image
        return {
            "score": 0.0,
            "flagged": False,
            "clipped_pixel_fraction": 0.0,
            "clipped_blob_fraction": 0.0,
        }

    h, w = img_bgr.shape[:2]
    total_pixels = h * w

    # Find pixels at 0 or 255 per channel
    clipped_any = np.any(
        (img_bgr == 0) | (img_bgr == 255),
        axis=2
    )  # (H, W) bool

    # Apply mask if given
    if mask is not None:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        clipped_any = clipped_any & (mask_f > 0.5)

    clipped_pixel_count = int(clipped_any.sum())
    clipped_pixel_fraction = clipped_pixel_count / total_pixels

    # Find large blobs of clipped pixels
    clipped_uint8 = clipped_any.astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(clipped_uint8)

    # Find the size of the largest blob
    blob_sizes = np.bincount(labels.ravel())
    largest_blob_size = blob_sizes[1:].max() if len(blob_sizes) > 1 else 0
    clipped_blob_fraction = largest_blob_size / total_pixels

    # Score is the max of the two metrics
    score = max(clipped_pixel_fraction, clipped_blob_fraction)
    flagged = clipped_blob_fraction > CLIPPING_THRESHOLD

    return {
        "score": float(score),
        "flagged": bool(flagged),
        "clipped_pixel_fraction": float(clipped_pixel_fraction),
        "clipped_blob_fraction": float(clipped_blob_fraction),
    }


def detect_plastic_skin(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
) -> dict:
    """Detect over-smoothed skin (plastic/waxen appearance).

    Within the mask, measures high-frequency energy as std of (L - gaussian_blur(L))
    normalized by mid-frequency band. If energy is too low, skin texture is erased.

    Optionally compares before/after energy ratio if reference image is provided.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image (output).
        mask: Optional (H, W) float mask [0, 1], typically skin mask.
        reference_img_bgr: Optional (H, W, 3) uint8 BGR image (input) for comparison.

    Returns:
        dict with keys:
            - "score": float, energy ratio (0 = all texture erased, 1 = fully preserved)
            - "flagged": bool, True if score < PLASTIC_SKIN_THRESHOLD
            - "hf_energy_ratio": float, high-freq std / mid-freq std
            - "energy_loss_vs_reference": float or None (if reference provided)
    """
    if img_bgr.shape[0] < 8 or img_bgr.shape[1] < 8:
        # Tiny image
        return {
            "score": 1.0,
            "flagged": False,
            "hf_energy_ratio": 1.0,
            "energy_loss_vs_reference": None,
        }

    # Convert to Lab and extract L channel
    img_lab = _bgr_to_lab(img_bgr)
    L = img_lab[:, :, 0]  # float32, [0, 100]

    # Gaussian blur at sigma=2 (mid-frequency reference)
    L_blur = cv2.GaussianBlur(L, (0, 0), 2.0)

    # High-frequency = original - blurred
    L_hf = L - L_blur
    hf_energy = float(np.std(L_hf))

    # Mid-frequency = some reference level (std of the blurred version)
    mf_energy = float(np.std(L_blur))

    if mf_energy < 1e-6:
        # Flat image, no mid-freq content
        energy_ratio = 1.0
    else:
        energy_ratio = hf_energy / mf_energy

    # Apply mask if given
    if mask is not None:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0

        # Restrict calculation to masked region
        L_masked = L[mask_f > 0.5]
        L_blur_masked = L_blur[mask_f > 0.5]

        if len(L_masked) > 0:
            L_hf_masked = L_masked - L_blur_masked
            hf_energy = float(np.std(L_hf_masked))
            mf_energy = float(np.std(L_blur_masked))

            if mf_energy > 1e-6:
                energy_ratio = hf_energy / mf_energy

    # Compare with reference if provided
    energy_loss = None
    if reference_img_bgr is not None and reference_img_bgr.shape == img_bgr.shape:
        ref_lab = _bgr_to_lab(reference_img_bgr)
        L_ref = ref_lab[:, :, 0]
        L_ref_blur = cv2.GaussianBlur(L_ref, (0, 0), 2.0)
        L_ref_hf = L_ref - L_ref_blur

        ref_mf_energy = float(np.std(L_ref_blur))
        if ref_mf_energy > 1e-6:
            ref_hf_energy = float(np.std(L_ref_hf))
            ref_energy_ratio = ref_hf_energy / ref_mf_energy

            # Energy loss is how much the ratio decreased
            energy_loss = max(0.0, ref_energy_ratio - energy_ratio) / max(ref_energy_ratio, 1e-6)

        if mask is not None:
            mask_f = mask.astype(np.float32)
            if mask_f.max() > 1.0:
                mask_f /= 255.0

            L_ref_masked = L_ref[mask_f > 0.5]
            L_ref_blur_masked = L_ref_blur[mask_f > 0.5]

            if len(L_ref_masked) > 0:
                L_ref_hf_masked = L_ref_masked - L_ref_blur_masked
                ref_mf_energy = float(np.std(L_ref_blur_masked))
                if ref_mf_energy > 1e-6:
                    ref_hf_energy = float(np.std(L_ref_hf_masked))
                    ref_energy_ratio = ref_hf_energy / ref_mf_energy
                    energy_loss = max(0.0, ref_energy_ratio - energy_ratio) / max(ref_energy_ratio, 1e-6)

    # Score is the energy ratio (lower = more plastic)
    # Normalize to 0-1: score = 1 means full texture, 0 means erased
    # For flagging: we flag when score < threshold (i.e., energy is too low)
    score = min(1.0, max(0.0, energy_ratio / 1.5))  # Normalize assuming normal ~1.5
    flagged = energy_ratio < PLASTIC_SKIN_THRESHOLD

    return {
        "score": float(score),
        "flagged": bool(flagged),
        "hf_energy_ratio": float(energy_ratio),
        "energy_loss_vs_reference": energy_loss,
    }


def detect_halo(
    img_bgr: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> dict:
    """Detect edge overshoot from aggressive sharpening (halos/ringing).

    Identifies strong edges via Canny, then measures mean overshoot of L channel
    beyond the pre-edge plateau within a 5-10px band. High overshoot indicates
    sharpening halos.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image.
        mask: Optional (H, W) float mask [0, 1] to restrict analysis.

    Returns:
        dict with keys:
            - "score": float, mean overshoot amplitude in levels
            - "flagged": bool, True if score > HALO_THRESHOLD
            - "mean_overshoot": float, mean L overshoot at edges
            - "edge_count": int, number of edge pixels analyzed
    """
    if img_bgr.shape[0] < 16 or img_bgr.shape[1] < 16:
        # Too small to measure edge halos
        return {
            "score": 0.0,
            "flagged": False,
            "mean_overshoot": 0.0,
            "edge_count": 0,
        }

    # Convert to Lab and extract L channel
    img_lab = _bgr_to_lab(img_bgr)
    L = img_lab[:, :, 0]  # float32, [0, 100]

    # Detect strong edges via Canny
    L_uint8 = np.clip(L, 0, 100).astype(np.uint8) * 255 // 100
    edges = cv2.Canny(L_uint8, 50, 150)  # Returns binary edge map

    if edges.sum() == 0:
        # No edges detected
        return {
            "score": 0.0,
            "flagged": False,
            "mean_overshoot": 0.0,
            "edge_count": 0,
        }

    # For each edge pixel, look at a band perpendicular to the edge
    # Measure overshoot: max within band minus local baseline
    h, w = L.shape
    overshoots = []

    # Find edge coordinates
    edge_coords = np.argwhere(edges > 0)

    for y, x in edge_coords:
        # Local baseline: average a few pixels back from edge (perpendicular)
        baseline = L[max(0, y - 3):y, max(0, x - 1):min(w, x + 2)].mean()

        # Overshoot: max in forward band
        forward_max = L[min(h - 1, y + 3):min(h - 1, y + 8), max(0, x - 1):min(w, x + 2)].max()

        overshoot = forward_max - baseline
        if overshoot > 0:
            overshoots.append(overshoot)

    if not overshoots:
        mean_overshoot = 0.0
    else:
        mean_overshoot = float(np.mean(overshoots))

    # Apply mask if given
    if mask is not None:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0

        # Filter edge coordinates to those in mask
        edge_coords_masked = edge_coords[mask_f[edge_coords[:, 0], edge_coords[:, 1]] > 0.5]
        overshoots_masked = []

        for y, x in edge_coords_masked:
            baseline = L[max(0, y - 3):y, max(0, x - 1):min(w, x + 2)].mean()
            forward_max = L[min(h - 1, y + 3):min(h - 1, y + 8), max(0, x - 1):min(w, x + 2)].max()
            overshoot = forward_max - baseline
            if overshoot > 0:
                overshoots_masked.append(overshoot)

        if overshoots_masked:
            mean_overshoot = float(np.mean(overshoots_masked))
            overshoots = overshoots_masked

    # Score is the mean overshoot
    score = mean_overshoot
    flagged = mean_overshoot > HALO_THRESHOLD

    return {
        "score": float(score),
        "flagged": bool(flagged),
        "mean_overshoot": float(mean_overshoot),
        "edge_count": len(overshoots),
    }


def run_all(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
) -> dict:
    """Run all four QA detectors and aggregate results.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image (output to check).
        skin_mask: Optional (H, W) float mask [0, 1] for skin region.
        reference_img_bgr: Optional (H, W, 3) uint8 BGR image (input) for comparison.

    Returns:
        dict with keys for each detector:
            - "banding": dict from detect_banding()
            - "clipping": dict from detect_clipping()
            - "plastic_skin": dict from detect_plastic_skin()
            - "halo": dict from detect_halo()
    """
    return {
        "banding": detect_banding(img_bgr, mask=skin_mask),
        "clipping": detect_clipping(img_bgr, mask=skin_mask),
        "plastic_skin": detect_plastic_skin(img_bgr, mask=skin_mask, reference_img_bgr=reference_img_bgr),
        "halo": detect_halo(img_bgr, mask=skin_mask),
    }
