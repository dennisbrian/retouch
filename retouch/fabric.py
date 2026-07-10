"""Auto fabric / clothing wrinkle smoothing (backlog #3).

Lightens mid-frequency fold ridges on cloth while preserving the fabric's
high-frequency weave. Mirrors :func:`retouch.skin.wrinkle_soften` (ridge-aware
attenuation) but operates on the whole image's L channel, restricted to a
cloth mask.

Design notes
------------
* Operates on float32 BGR. The engine wires it in ``_run_global_phases`` where
  ``result`` is float32 in [0, 1]; a [0, 255] float input is also accepted and
  auto-detected so the helper is reusable from tests.
* Fold detection is a luminance difference-of-Gaussians on the LAB L channel:
  dark folds are negative DoG values. Only that ridge signal is attenuated
  (capped at 60% depth reduction), so the weave texture (high frequency) and
  the gentle low-frequency shading of a real fold are left intact.
* Explicit ``cv2.cvtColor`` at the BGR->LAB boundary; no uint8 intermediate in
  pixel math.
"""

from typing import Optional

import cv2
import numpy as np

from .utils import normalize_mask, feather_mask, blend_masked


def smooth_fabric_wrinkles(
    img_bgr: np.ndarray,
    cloth_mask: Optional[np.ndarray],
    strength: float,
) -> np.ndarray:
    """Attenuate mid-frequency fold ridges inside the cloth region.

    Args:
        img_bgr: (H, W, 3) float32 BGR image in [0, 1] (or [0, 255]).
        cloth_mask: (H, W) float32 cloth mask [0, 1]; subject's clothing.
        strength: 0–100 wrinkle smoothing intensity.

    Returns:
        (H, W, 3) float32 BGR image, same scale as input; unchanged if
        ``strength <= 0`` or ``cloth_mask`` is None/empty.
    """
    if strength <= 0 or cloth_mask is None:
        return img_bgr

    cloth = normalize_mask(cloth_mask)
    if cloth.max() < 0.01:
        return img_bgr

    is_float = img_bgr.dtype == np.float32
    # Detect scale: float input may be [0,1] or [0,255].
    in_max = float(np.max(img_bgr)) if img_bgr.size else 0.0
    scale = 255.0 if (is_float and in_max > 1.5) else 1.0

    img = img_bgr.astype(np.float32) / scale  # -> [0, 1]

    # Explicit colorspace boundary: BGR -> LAB (cv2 float expects [0,1]).
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0] * 2.55  # -> [0, 255] to mirror tuned ridge math

    h, w = L.shape
    fold = max(6, int(min(h, w) * 0.015))
    sigma_small = max(1.0, fold / 6.0)
    sigma_large = max(1.0, fold / 2.0)

    g_small = cv2.GaussianBlur(L, (0, 0), sigma_small)
    g_large = cv2.GaussianBlur(L, (0, 0), sigma_large)
    dog = g_small - g_large

    # Dark folds appear as negative DoG; isolate their magnitude.
    ridge = np.clip(-dog, 0.0, 255.0)
    ridge *= cloth  # restrict to cloth

    s = strength / 100.0
    max_reduction_frac = 0.6  # cap depth reduction to preserve geometry
    attenuation = ridge * s * max_reduction_frac

    # Lighten the dark folds on the L channel, then back to BGR.
    lab[:, :, 0] = np.clip(L + attenuation, 0.0, 255.0) / 2.55
    processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Blend using the (feathered) cloth mask so edges stay clean.
    cloth_blend = feather_mask(cloth, radius=2)
    result = blend_masked(img, processed, cloth_blend)

    if not is_float:
        return np.clip(result * 255.0, 0.0, 255.0).astype(np.uint8)
    if scale == 255.0:
        return np.clip(result * 255.0, 0.0, 255.0).astype(np.float32)
    return result.astype(np.float32)
