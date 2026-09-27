"""Keep the nose's own shading through skin smoothing (opt-in ``nose_shape``).

Frequency-separation smoothing treats the nose like cheek skin. On a face
the nose is the most three-dimensional area, and its lit tip, bridge
highlight and shaded sides sit in the same mid band that smoothing evens
out. On Alex's white-face-paint bunny photos the default ``natural`` recipe
kept only 77-85% of the nose's broad shading: the nose turned into a flat,
pale shape with its crease left standing like a drawn outline. Whitening
barely touches it (about +1 L level), so the fix belongs here, straight
after smoothing, rather than in the whitening step.

The op puts back exactly the broad shading smoothing removed inside the
landmark nose region::

    canvas += s * nose_mask * (blur(before) - blur(after))

``blur`` is a Gaussian at a fraction of the inter-eye distance, so fine
texture (pores, shine speckle, spots) stays smoothed and every op that runs
later (whitening, shine removal, relight, colour grading) still acts on the
nose. The restore is a difference between two versions of the same pixels,
so it scales with the face's own tones and needs no intensity threshold.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

# Blur sigma as a fraction of the inter-eye distance (IED). 0.04 x IED
# separates the nose's broad form (tip, bridge, alae shading) from pore-
# and spot-scale texture on faces from 150 px to 1500 px IED.
_SIGMA_IED_FRAC = 0.04
_MIN_SIGMA_PX = 1.5
# The landmark nose polygon stops just inside the alar crease; grow it a
# little so the crease is restored with the nose instead of being left as a
# sharp outline against flattened skin.
_GROW_IED_FRAC = 0.04


def keep_nose_shape(
    canvas: np.ndarray,
    before: np.ndarray,
    nose_mask: Optional[np.ndarray],
    strength: float,
    ied: float,
    skin_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Restore the nose's broad shading that smoothing flattened.

    Args:
        canvas: (H, W, 3) float32 BGR [0, 255], the smoothed face ROI.
        before: (H, W, 3) BGR, the same ROI just before smoothing.
        nose_mask: (H, W) landmark nose mask, 0-1 or 0-255. ``None`` or
            empty returns ``canvas`` unchanged.
        strength: 0-100. 0 returns ``canvas`` unchanged; 100 puts back all
            of the broad nose shading smoothing removed.
        ied: inter-eye distance in ROI pixels (sets the blur scale).
        skin_mask: optional (H, W) skin mask, 0-1 or 0-255; the restore is
            limited to it so hair or glasses over the nose stay as they are.

    Returns:
        (H, W, 3) float32 BGR [0, 255].
    """
    s = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if s <= 0.0 or nose_mask is None:
        return canvas
    mask = nose_mask.astype(np.float32)
    if mask.max() > 1.5:
        mask = mask / 255.0
    mask = np.clip(mask, 0.0, 1.0)
    if not np.any(mask > 0.01):
        return canvas

    ied = float(ied) if ied and ied > 0 else float(max(canvas.shape[:2])) * 0.3
    grow = max(1, int(round(ied * _GROW_IED_FRAC)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    mask = cv2.dilate(mask, k)
    mask = cv2.GaussianBlur(mask, (0, 0), grow * 0.5 + 0.5)
    if skin_mask is not None:
        sk = skin_mask.astype(np.float32)
        if sk.max() > 1.5:
            sk = sk / 255.0
        mask = mask * np.clip(sk, 0.0, 1.0)

    sigma = max(_MIN_SIGMA_PX, ied * _SIGMA_IED_FRAC)
    canvas_f = canvas.astype(np.float32)
    lost = (
        cv2.GaussianBlur(before.astype(np.float32), (0, 0), sigma)
        - cv2.GaussianBlur(canvas_f, (0, 0), sigma)
    )
    out = canvas_f + (s * mask)[:, :, None] * lost
    return np.clip(out, 0.0, 255.0).astype(np.float32)
