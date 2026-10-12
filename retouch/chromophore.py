"""Melanin / hemoglobin skin chromophore decomposition (log-linear unmixing).

Implements the well-cited log-RGB chromophore unmixing used by Tsumura
(2003, "Separation of skin texture from shading") and Stamatas et al. for
skin diffuse reflectance. No training, no neural net.

In log-RGB space skin diffuse reflectance is approximately linear in the
chromophore concentrations:

    L = -log(rgb) = M @ [melanin, hemoglobin] - c

where ``L`` is the 3-vector of per-channel optical density, ``M`` is a 3x2
(extinction) coefficient matrix with channels (R, G, B) on its rows and the
two chromophores (melanin, hemoglobin) on its columns, and ``c`` is a scalar
camera/scene constant that is identical across the three channels for a given
pixel. Because ``c`` is shared across channels it cancels when we difference
the channels, so we solve the 2x2 subsystem (R-B, G-B) via the pseudo-inverse
of ``M`` with the blue column removed.

Both maps are returned float32 in [0, ~1+] (clamped at 0). The exported names
``decompose_chromophores`` and ``hemoglobin_breakdown_phase`` are stable: they
are imported by the R10 (chromophore suite) and R11 (cosplay moat) agents.

BGR is the input convention (image is converted to RGB [0, 1] internally).
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

# Extinction coefficients in log-RGB space. Rows = (R, G, B) channels,
# columns = (melanin, hemoglobin).
#
# Melanin (eumelanin) is a near-neutral brown absorber: it darkens every
# channel, slightly more at blue — the standard Tsumura melanin direction.
# Hemoglobin, in skin, reads as *redness* (more blood => redder skin), so its
# basis vector brightens red and darkens green/blue. This orientation makes
# the decomposition map intuitively: a red pixel => high hemoglobin, a
# brown/dark pixel => high melanin. Deviation from the literal Stamatas
# oxy/deoxy hemoglobin RGB extinction vector is intentional so that the
# resulting hemoglobin map correlates with perceived skin redness (required
# by R10 bruise/redness operators). ponytail: swap K_HEMOGLOBIN for the
# merged oxy+deoxy extinction triple when a strictly physical model is needed.
_K_MELANIN = np.array([0.1, 0.2, 0.3], dtype=np.float32)
_K_HEMOGLOBIN = np.array([-0.5, 0.25, 0.25], dtype=np.float32)

# 3x2 coefficient matrix M (rows R,G,B; cols melanin, hemoglobin).
_M = np.stack([_K_MELANIN, _K_HEMOGLOBIN], axis=1).astype(np.float32)  # (3, 2)

# Pseudo-inverse of M with the blue channel subtracted (removes the shared
# constant c). M' = M - M[blue]; pinv is (2, 3) and applied per pixel.
_M_PRIME = _M - _M[2:3, :]  # broadcast subtract blue row
_M_PINV = np.linalg.pinv(_M_PRIME)  # (2, 3)

# Canonical OpenCV-HSV hue centroids (0..179 scale) for bruise phases.
_PHASE_HUE = np.array([150.0, 60.0, 30.0], dtype=np.float32)  # purple, green, yellow


def _bgr_to_log_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR uint8/float in [0,255] to log-RGB optical density.

    Returns float32 HxWx3 with L = -log(rgb) where rgb is in [0, 1].
    """
    rgb = img_bgr[..., ::-1].astype(np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 1e-4, 1.0)
    return -np.log(rgb)


def decompose_chromophores(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose skin into melanin and hemoglobin concentration maps.

    Args:
        img_bgr: HxWx3 image, BGR convention, uint8 [0,255] or float [0,1].

    Returns:
        Tuple of two float32 HxW maps:
            melanin    — eumelanin concentration (higher = darker/brown/tan).
            hemoglobin — blood concentration (higher = red/rosacea/vein).
        Both are clamped to >= 0.

    Raises:
        ValueError: if input is not a 3-channel HxWx3 array.
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be an HxWx3 array")

    log_rgb = _bgr_to_log_rgb(img_bgr)  # (H, W, 3), L = -log(rgb)

    # Remove the shared camera/scene constant c (identical across channels)
    # by differencing against the blue channel. M_PINV was built on this
    # blue-subtracted matrix, so apply it to the same differenced vectors.
    log_diff = log_rgb - log_rgb[..., 2:3]  # (H, W, 3), blue column = 0

    # Apply the (2, 3) pseudo-inverse to each pixel's 3-vector.
    conc = log_diff @ _M_PINV.T  # (H, W, 2) -> [melanin, hemoglobin]
    melanin = conc[..., 0]
    hemoglobin = conc[..., 1]

    melanin = np.clip(melanin, 0.0, None).astype(np.float32)
    hemoglobin = np.clip(hemoglobin, 0.0, None).astype(np.float32)
    return melanin, hemoglobin


def reconstruct_from_chromophores(
    melanin: np.ndarray,
    hemoglobin: np.ndarray,
    c: Optional[np.ndarray] = None,
    *,
    img_bgr: Optional[np.ndarray] = None,
    skin_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Forward map mel/hb → BGR float32 [0,255].

    If ``c`` is None and ``img_bgr`` is given, fit shared ambient optical density
    as median residual on skin (or whole image). Else ``c=0``.
    """
    mel = melanin.astype(np.float32)
    hb = hemoglobin.astype(np.float32)
    conc = np.stack([mel, hb], axis=-1)  # H,W,2
    log_hat = conc @ _M.T  # H,W,3 RGB optical density (c=0)

    if c is None:
        if img_bgr is not None:
            log_rgb = _bgr_to_log_rgb(img_bgr)
            resid = log_rgb - log_hat  # H,W,3
            # Shared scalar ambient: mean across channels, median over skin
            per_px = resid.mean(axis=-1)
            if skin_mask is not None:
                m = skin_mask.astype(np.float32)
                if m.max() > 1.5:
                    m = m / 255.0
                sel = per_px[m > 0.5]
                c_val = float(np.median(sel)) if sel.size else 0.0
            else:
                c_val = float(np.median(per_px))
            c = np.full(mel.shape, c_val, dtype=np.float32)
        else:
            c = 0.0

    if np.isscalar(c):
        log_rgb_hat = log_hat + float(c)
    else:
        c_arr = np.asarray(c, dtype=np.float32)
        if c_arr.ndim == 2:
            c_arr = c_arr[..., None]
        log_rgb_hat = log_hat + c_arr
    rgb = np.exp(-np.clip(log_rgb_hat, -20.0, 20.0))
    rgb = np.clip(rgb, 0.0, 1.0)
    bgr = (rgb[..., ::-1] * 255.0).astype(np.float32)
    return bgr


def hemoglobin_breakdown_phase(
    hemoglobin_map: np.ndarray,
    hue_map_or_bgr: np.ndarray,
) -> np.ndarray:
    """Label each pixel's hemoglobin signal with a bruise breakdown phase.

    Bruises progress purple -> green -> yellow as hemoglobin breaks down.
    The phase is derived from the local hue of the source (the BGR image or a
    pre-computed hue map): each pixel is assigned the nearest of the three
    canonical hues.

    Phase codes:
        0 = purple
        1 = green
        2 = yellow

    Args:
        hemoglobin_map: HxW float32 hemoglobin map (unused for the label
            itself but accepted so callers can pass it through; kept in the
            signature for R10 compositing).
        hue_map_or_bgr: Either an HxW uint8/int hue map (e.g. from
            ``cv2.cvtColor(img, COLOR_BGR2HSV)[:, :, 0]``) or an HxWx3 BGR
            image from which hue is computed internally.

    Returns:
        HxW uint8 map with values in {0, 1, 2}.
    """
    if hue_map_or_bgr.ndim == 3:
        hsv = cv2.cvtColor(hue_map_or_bgr, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float32)
    else:
        hue = hue_map_or_bgr.astype(np.float32)

    # Circular distance on the 0..180 OpenCV hue wheel.
    diff = np.abs(hue[..., np.newaxis] - _PHASE_HUE)  # (H, W, 3)
    dist = np.minimum(diff, 180.0 - diff)
    phase = np.argmin(dist, axis=2).astype(np.uint8)
    return phase
