"""Spot heal / object removal via OpenCV inpainting.

Removes stray hairs, dust, wig lace, background clutter without Photoshop.
Uses Telea or Navier-Stokes inpainting with automatic radius scaling.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .blemish import inpaint_and_blend


def heal_region(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    method: str = "telea",
    radius: Optional[int] = None,
) -> np.ndarray:
    """Heal a masked region using inpainting.

    Args:
        img_bgr: (H, W, 3) uint8 BGR image or float32 [0, 255] BGR image.
        mask: (H, W) uint8 binary mask (255 = region to heal) or float32 [0,1].
        method: "telea" or "ns" (Navier-Stokes).
        radius: Inpaint radius. If None, auto-scales from mask bounding box.

    Returns:
        (H, W, 3) image matching input dtype. The uint8 path is byte-identical
        to the legacy implementation; the float32 path returns float32 [0, 255].
    """
    if mask.sum() == 0:
        return img_bgr

    if mask.dtype == np.float32 or mask.max() <= 1.0:
        mask_uint8 = (mask * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)

    if radius is None:
        radius = _auto_radius(mask_uint8)

    flags = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS

    blend_ksize = max(radius * 2 + 1, 3) | 1

    return inpaint_and_blend(img_bgr, mask_uint8, radius, flags, blend_ksize)


def _auto_radius(mask: np.ndarray) -> int:
    """Compute inpaint radius from mask bounding box size.

    Scales with the mask's bounding box diagonal, not face width.
    """
    coords = cv2.findNonZero(mask)
    if coords is None:
        return 3

    x, y, w, h = cv2.boundingRect(coords)
    diag = (w * w + h * h) ** 0.5

    radius = max(int(diag * 0.1), 2)
    radius = min(radius, 50)
    return radius


def mask_to_b64(mask: np.ndarray) -> str:
    """Encode a mask (uint8 or float32) to base64 PNG string."""
    import base64

    if mask.dtype == np.float32:
        mask_uint8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    else:
        mask_uint8 = mask.astype(np.uint8)

    _, buf = cv2.imencode(".png", mask_uint8)
    return base64.b64encode(buf).decode("utf-8")


def b64_to_mask(b64_str: str, target_shape: Optional[tuple] = None) -> np.ndarray:
    """Decode a base64 PNG string back to a uint8 mask.

    If target_shape is provided, resize to match.
    """
    import base64

    buf = base64.b64decode(b64_str)
    arr = np.frombuffer(buf, dtype=np.uint8)
    mask = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

    if mask is None:
        raise ValueError("Failed to decode mask from base64")

    if target_shape is not None and mask.shape[:2] != target_shape[:2]:
        mask = cv2.resize(mask, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

    return mask
