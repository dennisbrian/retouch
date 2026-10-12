"""Local shadow lift — targeted fill-light for real photographic shadows
(e.g. a chin/jaw or chest area sitting in shade next to a bright prop/light).

Separate, opt-in module: unlike global shadows/highlights (tonal.py) or
per-face relight (relight.py), this targets *local* luminance dips relative
to their immediate surroundings, so it can brighten a shaded jaw or
décolletage without lifting the whole image's shadows uniformly. Off by
default — must be explicitly enabled via a recipe's `shadow_lift` key. Does
not alter any existing stage; called as an optional extra pass.
"""

from __future__ import annotations

import cv2
import numpy as np

from .utils import apply_u8_op_float, blend_masked


class ShadowLifter:
    """Detects and brightens local (not global) shadow regions."""

    def lift(
        self,
        img_bgr: np.ndarray,
        region_mask: np.ndarray,
        strength: float = 0,
        radius_fraction: float = 0.08,
    ) -> np.ndarray:
        """Brighten pixels that sit notably darker than their local neighborhood
        average, within region_mask (e.g. skin_mask or body_skin_mask).

        Args:
            img_bgr: (H, W, 3) uint8 or float32 BGR image.
            region_mask: (H, W) float32 mask in [0, 1] — where to consider lifting.
            strength: 0-100 lift intensity.
            radius_fraction: Local-average blur radius as a fraction of the
                shorter image dimension. Larger values treat coarser regions
                as "local"; smaller values react to finer shadow shapes.

        Returns:
            Image of the same dtype/shape as input.
        """
        if strength <= 0 or region_mask is None or region_mask.max() < 0.01:
            return img_bgr

        if img_bgr.dtype == np.float32:
            return apply_u8_op_float(
                img_bgr, self.lift, region_mask, strength, radius_fraction
            )

        s = strength / 100.0
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_val = lab[:, :, 0]

        radius = max(9, int(min(img_bgr.shape[:2]) * radius_fraction) | 1)
        local_avg = cv2.GaussianBlur(l_val, (radius, radius), 0)

        # Shadow deficit: how far below the local average this pixel sits.
        # Only lift where the pixel is darker than its surroundings (clip
        # positive deficits to 0 so already-bright pixels are untouched).
        deficit = np.clip(local_avg - l_val, 0, None)

        # Normalize deficit so a ~40 L-point local shadow reaches full lift
        # strength; deeper shadows don't get disproportionately blown out.
        deficit_norm = np.clip(deficit / 40.0, 0.0, 1.0)

        lift_amount = deficit_norm * 25.0 * s
        lab[:, :, 0] = np.clip(l_val + lift_amount, 0, 255)

        result = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        return blend_masked(img_bgr, result, region_mask)
