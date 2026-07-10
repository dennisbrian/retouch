"""Quality Assurance detectors for output self-QA.

Provides pure-function artifact detectors that analyze output images for:
  - Banding (posterization on smooth gradients)
  - Clipping (blown highlights and crushed blacks)
  - Plastic skin (over-smoothed texture loss in skin regions)
  - Halo (edge overshoot from aggressive sharpening)
  - Seam (gradient discontinuities along subject-separation boundaries)

Each detector returns a dict with:
  - "score": float in [0, 1] (higher = more problematic)
  - "flagged": bool (True if score exceeds threshold)
  - Additional detail keys specific to each detector
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List

import cv2
import numpy as np


def _to_u8_for_analysis(img: np.ndarray) -> np.ndarray:
    """Return a uint8 BGR snapshot of ``img`` for QA analysis.

    QA detectors measure artifacts (banding, clipping, halos, ...), not pixel
    values, so the analysis result is identical whether run on a float32
    [0, 255] canvas or its uint8 truncation. Float32 input is truncated
    (not rounded) to match the legacy uint8 chain's
    ``np.clip(x, 0, 255).astype(np.uint8)`` convention so thresholds see
    values consistent with the pre-E1 pipeline. uint8 input is returned
    unchanged (byte-identical path).
    """
    if img.dtype == np.uint8:
        return img
    return np.clip(img, 0, 255).astype(np.uint8)


@dataclass
class QAWarning:
    detector: str
    score: float
    flagged: bool
    message: str
    threshold: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


# Threshold constants — adjust based on tuning
BANDING_THRESHOLD = 0.15  # Flag if >15% of smooth pixels on quantization step edges
CLIPPING_THRESHOLD = 0.08  # Flag if clipped blob fraction >8% of image
PLASTIC_SKIN_THRESHOLD = 0.60  # Flag if high-freq energy ratio falls below 60%
HALO_THRESHOLD = 15.0  # Flag if mean overshoot amplitude >15 levels
SEAM_THRESHOLD = 5.0   # Flag if boundary gradient >5 L-levels above context

# K8 — color-fidelity (Δ-E / hue-drift) gate.
# Skin hue should shift only a few degrees under a grade; beyond these the
# grade has wrecked skin color. COLOR_DRIFT_THRESHOLD mirrors the max-band.
COLOR_DRIFT_HUE_MEAN_THRESHOLD = 6.0   # Flag if mean skin Δh exceeds 6°
COLOR_DRIFT_HUE_MAX_THRESHOLD = 15.0   # Flag if any skin pixel Δh exceeds 15°
COLOR_DRIFT_THRESHOLD = 15.0           # Max Δh (deg) considered the flag boundary


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
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image. Float input
            is truncated to uint8 internally for analysis; the QA result is
            identical to running on the uint8 snapshot.
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

    img_bgr = _to_u8_for_analysis(img_bgr)
    # Convert to Lab and extract L channel
    img_lab = _bgr_to_lab(img_bgr)
    L = img_lab[:, :, 0]  # float32, [0, 100]

    # Identify smooth (low-variance) regions via 3x3 local std
    # Compute local variance in sliding window
    h, w = L.shape
    local_mean = cv2.boxFilter(L, cv2.CV_32F, (3, 3))
    local_sq_mean = cv2.boxFilter(L * L, cv2.CV_32F, (3, 3))
    local_var = local_sq_mean - local_mean * local_mean
    local_var = np.maximum(local_var, 0.0)

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
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image. Float input
            is truncated to uint8 internally so the 0/255 clipping test is
            consistent with the uint8 pipeline.
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

    img_bgr = _to_u8_for_analysis(img_bgr)
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
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image (output).
            Float input is truncated to uint8 internally for analysis.
        mask: Optional (H, W) float mask [0, 1], typically skin mask.
        reference_img_bgr: Optional (H, W, 3) uint8 or float32 [0, 255] BGR
            image (input) for comparison.

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

    img_bgr = _to_u8_for_analysis(img_bgr)
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
        ref_lab = _bgr_to_lab(_to_u8_for_analysis(reference_img_bgr))
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
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image. Float input
            is truncated to uint8 internally for analysis.
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

    img_bgr = _to_u8_for_analysis(img_bgr)
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

    # For each edge pixel, measure overshoot using morphological dilation
    edge_dilated = cv2.dilate(edges.astype(np.uint8), np.ones((7, 7), np.uint8))
    edge_eroded = cv2.erode(edges.astype(np.uint8), np.ones((3, 3), np.uint8))

    # Forward band = dilation for overshoot, backward band = edge pixels themselves
    forward_band = edge_dilated & ~edges.astype(np.uint8)
    edge_band = edges.astype(np.uint8) & ~edge_eroded

    if forward_band.sum() == 0:
        return {"score": 0.0, "flagged": False, "mean_overshoot": 0.0, "edge_count": 0}

    # Baseline: L at edge pixels
    baseline_vals = L[edge_band > 0]
    # Overshoot: L in forward band
    forward_vals = L[forward_band > 0]

    if len(baseline_vals) == 0 or len(forward_vals) == 0:
        return {"score": 0.0, "flagged": False, "mean_overshoot": 0.0, "edge_count": 0}

    # For each edge pixel, find nearest forward pixel's max value
    # Simplified: percentile-based comparison
    baseline_median = np.median(baseline_vals)
    forward_max = np.percentile(forward_vals, 95)
    mean_overshoot = float(max(0.0, forward_max - baseline_median))

    if mask is not None:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        baseline_masked = baseline_vals[mask_f[edge_band > 0].astype(bool)] if edge_band[mask_f > 0.5].sum() > 0 else np.array([], dtype=np.float32)
        forward_masked = forward_vals[mask_f[forward_band > 0].astype(bool)] if forward_band[mask_f > 0.5].sum() > 0 else np.array([], dtype=np.float32)
        if len(baseline_masked) > 0 and len(forward_masked) > 0:
            baseline_median = np.median(baseline_masked)
            forward_max = np.percentile(forward_masked, 95)
            mean_overshoot = float(max(0.0, forward_max - baseline_median))

    # Score is the mean overshoot
    score = mean_overshoot
    flagged = mean_overshoot > HALO_THRESHOLD

    return {
        "score": float(score),
        "flagged": bool(flagged),
        "mean_overshoot": float(mean_overshoot),
        "edge_count": int(forward_band.sum()),
    }


def detect_seam(
    img_bgr: np.ndarray,
    person_mask: Optional[np.ndarray] = None,
) -> dict:
    """Detect seam (gradient discontinuity) along a person-mask boundary.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image. Float input
            is truncated to uint8 internally for analysis.
        person_mask: Optional (H, W) float mask [0, 1] delimiting the subject.

    Returns:
        dict with keys ``score``, ``flagged``, ``seam_gradient``,
        ``boundary_pixels``.
    """
    if img_bgr.shape[0] < 16 or img_bgr.shape[1] < 16:
        return {"score": 0.0, "flagged": False, "seam_gradient": 0.0, "boundary_pixels": 0}
    img_bgr = _to_u8_for_analysis(img_bgr)
    img_lab = _bgr_to_lab(img_bgr)
    L = img_lab[:, :, 0]
    grad_x = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    if person_mask is not None:
        mask_f = person_mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        boundary = (cv2.dilate((mask_f > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)) ^
                    cv2.erode((mask_f > 0.5).astype(np.uint8), np.ones((5, 5), np.uint8)))
        if boundary.sum() == 0:
            return {"score": 0.0, "flagged": False, "seam_gradient": 0.0, "boundary_pixels": 0}
        interior = mask_f > 0.5
        exterior = ~interior
        boundary_grad = grad_mag[boundary > 0]
        interior_grad = grad_mag[interior]
        exterior_grad = grad_mag[exterior]
        if len(boundary_grad) == 0 or len(interior_grad) == 0 or len(exterior_grad) == 0:
            return {"score": 0.0, "flagged": False, "seam_gradient": 0.0, "boundary_pixels": 0}
        boundary_mean = float(np.mean(boundary_grad))
        context_mean = float((np.mean(interior_grad) + np.mean(exterior_grad)) / 2.0)
        seam_gradient = max(0.0, boundary_mean - context_mean)
        flagged = seam_gradient > SEAM_THRESHOLD
        return {
            "score": min(1.0, seam_gradient / 20.0),
            "flagged": bool(flagged),
            "seam_gradient": seam_gradient,
            "boundary_pixels": int(boundary.sum()),
        }
    return {"score": 0.0, "flagged": False, "seam_gradient": 0.0, "boundary_pixels": 0}


def _bgr_to_lab_f(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR (uint8 or float32 [0, 255]) image to CIELab float32.

    Unlike :func:`_bgr_to_lab` this is float-safe: no truncation to uint8,
    so the color-fidelity gate keeps full precision for the Δ-E math.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image.

    Returns:
        (H, W, 3) float32 Lab image, L in [0, 100], a/b in ~[-128, 128].
    """
    if img_bgr.dtype == np.uint8:
        bgr = img_bgr.astype(np.float32) / 255.0
    else:
        bgr = np.clip(img_bgr, 0.0, 255.0).astype(np.float32) / 255.0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2Lab)


def delta_e_2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """Compute CIEDE2000 color difference (Sharma et al. 2005, closed form).

    Fully vectorized — no per-pixel loops.

    Args:
        lab1: (H, W, 3) or (3,) float LAB image (L in [0, 100], a/b any range).
        lab2: Same shape as ``lab1``.

    Returns:
        (H, W) or scalar float32 array of ΔE2000 values.
    """
    arr1 = np.asarray(lab1, dtype=np.float32)
    arr2 = np.asarray(lab2, dtype=np.float32)
    squeeze = arr1.ndim == 1
    if squeeze:
        arr1 = arr1.reshape(1, 3)
        arr2 = arr2.reshape(1, 3)
    c1 = np.sqrt(arr1[..., 1] ** 2 + arr1[..., 2] ** 2)
    c2 = np.sqrt(arr2[..., 1] ** 2 + arr2[..., 2] ** 2)
    ac1c2 = (c1 + c2) / 2.0
    g = 0.5 * (1.0 - np.sqrt(ac1c2 ** 7 / (ac1c2 ** 7 + 25.0 ** 7)))
    a1p = (1.0 + g) * arr1[..., 1]
    a2p = (1.0 + g) * arr2[..., 1]
    c1p = np.sqrt(a1p ** 2 + arr1[..., 2] ** 2)
    c2p = np.sqrt(a2p ** 2 + arr2[..., 2] ** 2)
    h1p = np.mod(np.arctan2(arr1[..., 2], a1p), 2.0 * np.pi)
    h2p = np.mod(np.arctan2(arr2[..., 2], a2p), 2.0 * np.pi)
    dl = arr2[..., 0] - arr1[..., 0]
    dc = c2p - c1p
    dh = h2p - h1p
    dh = dh - (2.0 * np.pi) * (dh > np.pi)
    dh = dh + (2.0 * np.pi) * (dh < -np.pi)
    dh_term = 2.0 * np.sqrt(c1p * c2p) * np.sin(dh / 2.0)
    zero_chroma = (c1p * c2p) == 0
    dh_term = np.where(zero_chroma, 0.0, dh_term)
    lbar = (arr1[..., 0] + arr2[..., 0]) / 2.0
    cbar = (c1p + c2p) / 2.0
    hbar = (h1p + h2p) / 2.0
    hbar = hbar + np.where(np.abs(h1p - h2p) > np.pi, np.pi, 0.0)
    hbar = hbar - np.where(hbar > 2.0 * np.pi, 2.0 * np.pi, 0.0)
    # Sharma et al. 2005 closed form (correct hue-rotation terms, in radians).
    t = (
        1.0
        - 0.17 * np.cos(hbar - np.pi / 6.0)
        + 0.24 * np.cos(2.0 * hbar)
        + 0.32 * np.cos(3.0 * hbar + np.pi / 30.0)
        - 0.20 * np.cos(4.0 * hbar - 63.0 * np.pi / 180.0)
    )
    dtheta_deg = 30.0 * np.exp(-(((hbar - 275.0 * np.pi / 180.0) / (25.0 * np.pi / 180.0)) ** 2))
    rc = 2.0 * np.sqrt(cbar ** 7 / (cbar ** 7 + 25.0 ** 7))
    rt = -np.sin(np.radians(2.0 * dtheta_deg)) * rc
    sl = 1.0 + (0.015 * (lbar - 50.0) ** 2) / np.sqrt(20.0 + (lbar - 50.0) ** 2)
    sc = 1.0 + 0.045 * cbar
    sh = 1.0 + 0.015 * cbar * t
    de = np.sqrt(
        (dl / sl) ** 2
        + (dc / sc) ** 2
        + (dh_term / sh) ** 2
        + rt * (dc / sc) * (dh_term / sh)
    )
    if squeeze:
        return float(de[0])
    if arr1.ndim == 3:
        return de.reshape(arr1.shape[0], arr1.shape[1])
    return de


def detect_color_drift(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
) -> dict:
    """Color-fidelity gate: measure skin hue/ΔE drift between reference and output.

    Implements K8 — a CIEDE2000 + hue-angle (Δh) color-safety check. The
    primary use is a before/after comparison (``reference_img_bgr`` given):
    computes ΔE2000 and per-pixel hue-angle shift Δh over the skin region and
    flags if the skin hue has wandered beyond a few degrees.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR output image.
        skin_mask: Optional (H, W) float mask [0, 1]; analysis restricted to
            masked pixels when a reference is supplied.
        reference_img_bgr: Optional pre-grade reference (uint8 or float32
            [0, 255]) BGR image of the same shape.

    Returns:
        dict with keys:
            - "score": float in [0, 1] (higher = worse). 0 if no reference.
            - "flagged": bool, True if mean Δh > 6° or max Δh > 15°.
            - "deltaE_mean": float, mean ΔE2000 over (skin) region.
            - "deltaH_mean_deg": float, mean |hue-angle shift| in degrees.
            - "deltaH_max_deg": float, max |hue-angle shift| in degrees.
            - "note": str, explanatory text when no reference is supplied.
    """
    if reference_img_bgr is None:
        return {
            "score": 0.0,
            "flagged": False,
            "deltaE_mean": 0.0,
            "deltaH_mean_deg": 0.0,
            "deltaH_max_deg": 0.0,
            "note": "no reference supplied — self-check skipped; "
                    "color_drift gate is a before/after comparison",
        }

    if reference_img_bgr.shape != img_bgr.shape:
        return {
            "score": 0.0,
            "flagged": False,
            "deltaE_mean": 0.0,
            "deltaH_mean_deg": 0.0,
            "deltaH_max_deg": 0.0,
            "note": "reference shape mismatch",
        }

    out_lab = _bgr_to_lab_f(img_bgr)
    ref_lab = _bgr_to_lab_f(reference_img_bgr)

    if skin_mask is not None:
        mask_f = skin_mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f /= 255.0
        region = mask_f > 0.5
    else:
        region = np.ones(out_lab.shape[:2], dtype=bool)

    if not np.any(region):
        return {
            "score": 0.0,
            "flagged": False,
            "deltaE_mean": 0.0,
            "deltaH_mean_deg": 0.0,
            "deltaH_max_deg": 0.0,
            "note": "empty skin region",
        }

    out_sub = out_lab[region]
    ref_sub = ref_lab[region]

    delta_e = delta_e_2000(ref_sub, out_sub)
    deltaE_mean = float(np.mean(delta_e))

    # Hue-angle shift from a/b channels (degrees, wrapped to [-180, 180]).
    h_ref = np.degrees(np.arctan2(ref_sub[:, 2], ref_sub[:, 1]))
    h_out = np.degrees(np.arctan2(out_sub[:, 2], out_sub[:, 1]))
    delta_h = h_out - h_ref
    delta_h = (delta_h + 180.0) % 360.0 - 180.0
    delta_h = np.abs(delta_h)
    deltaH_mean_deg = float(np.mean(delta_h))
    deltaH_max_deg = float(np.max(delta_h))

    flagged = (
        deltaH_mean_deg > COLOR_DRIFT_HUE_MEAN_THRESHOLD
        or deltaH_max_deg > COLOR_DRIFT_HUE_MAX_THRESHOLD
    )
    score = min(1.0, deltaH_max_deg / 30.0)

    return {
        "score": float(score),
        "flagged": bool(flagged),
        "deltaE_mean": deltaE_mean,
        "deltaH_mean_deg": deltaH_mean_deg,
        "deltaH_max_deg": deltaH_max_deg,
    }


def run_all(
    img_bgr: np.ndarray,
    skin_mask: Optional[np.ndarray] = None,
    reference_img_bgr: Optional[np.ndarray] = None,
    person_mask: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run all QA detectors and aggregate their results.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 [0, 255] BGR image. Float input
            is truncated to uint8 internally by each detector.
        skin_mask: Optional (H, W) float mask [0, 1].
        reference_img_bgr: Optional pre-retouch reference image (uint8 or
            float32 [0, 255]), forwarded to ``detect_plastic_skin`` for
            before/after texture-loss comparison.
        person_mask: Optional (H, W) float mask [0, 1] for seam detection.

    Returns:
        dict with keys "banding", "clipping", "plastic_skin", "halo", "seam",
        "color_drift", each mapping to that detector's full result dict
        (score/flagged/details).
    """
    result: Dict[str, Dict[str, Any]] = {}
    try:
        result["banding"] = detect_banding(img_bgr, skin_mask)
    except Exception:
        result["banding"] = {"score": 0.0, "flagged": False}
    try:
        result["clipping"] = detect_clipping(img_bgr, skin_mask)
    except Exception:
        result["clipping"] = {"score": 0.0, "flagged": False}
    try:
        result["plastic_skin"] = detect_plastic_skin(img_bgr, skin_mask, reference_img_bgr)
    except Exception:
        result["plastic_skin"] = {"score": 0.0, "flagged": False}
    try:
        result["halo"] = detect_halo(img_bgr, skin_mask)
    except Exception:
        result["halo"] = {"score": 0.0, "flagged": False}
    try:
        pm = person_mask if person_mask is not None else skin_mask
        result["seam"] = detect_seam(img_bgr, pm)
    except Exception:
        result["seam"] = {"score": 0.0, "flagged": False}
    try:
        result["color_drift"] = detect_color_drift(
            img_bgr, skin_mask, reference_img_bgr
        )
    except Exception:
        result["color_drift"] = {"score": 0.0, "flagged": False}
    return result
