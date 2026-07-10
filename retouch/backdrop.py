"""Auto backdrop cleanup (backlog #1).

Removes dust, specks and hard fold-crease lines from a (typically studio)
background by detecting them as high-frequency outliers relative to a
smoothed backdrop estimate, then inpainting them. Runs only outside the
feathered person mask, so the subject is never touched.

Design notes
------------
* Operates on float32 BGR in [0, 1] (the internal convention used by
  ``engine._run_global_phases``). uint8 inputs are converted at the
  boundary; dtype is preserved on output.
* Defect detection is a luminance high-pass: ``|gray - GaussianBlur(gray)|``.
  Dust/specks and fold crease *lines* are sharp high-frequency edges, so
  they light up; the smooth low-frequency shading of a gentle fold does
  not and is intentionally left (flattening real backdrop shading would
  look worse than the crease).
* Threshold is adaptive to the local backdrop (median residual), scaled by
  strength. Connected components larger than 5% of the frame are dropped so
  legitimate large background texture is never wiped.
* The person mask is eroded + feathered before use so the subject silhouette
  is never touched by the inpaint.
"""

from __future__ import annotations

import cv2
import numpy as np

from .utils import normalize_mask, feather_mask

_MAX_COMPONENT_FRAC = 0.05


def clean_backdrop(
    img_f01: np.ndarray,
    person_mask: np.ndarray,
    strength: float,
) -> np.ndarray:
    """Clean dust/specks/creases from the background of ``img_f01``.

    Args:
        img_f01: (H, W, 3) float32 BGR image in [0, 1].
        person_mask: (H, W) person mask (any range); subject region.
        strength: 0–100 cleanup intensity.

    Returns:
        (H, W, 3) float32 BGR in [0, 1]; unchanged if ``strength <= 0``.
    """
    if strength <= 0 or person_mask is None:
        return img_f01
    s = strength / 100.0

    pm = normalize_mask(person_mask)
    h, w = pm.shape[:2]
    # Erode by a margin LARGER than the feather radius so the backdrop region
    # (and the defect feather) stays strictly outside the subject. Without
    # this, the sharp subject/background boundary creates a high-frequency
    # residual in the background just outside the subject; detecting it there
    # and feathering would bleed the inpaint ~radius px INTO the subject edge.
    erode = max(8, int(min(h, w) * 0.015))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode, erode))
    backdrop = cv2.erode(1.0 - pm, kernel)
    backdrop = feather_mask(backdrop, radius=4)
    if backdrop.max() < 0.01:
        return img_f01

    gray = (
        0.299 * img_f01[:, :, 2]
        + 0.587 * img_f01[:, :, 1]
        + 0.114 * img_f01[:, :, 0]
    )
    k = max(3, int(min(h, w) * 0.02) | 1)
    low = cv2.GaussianBlur(gray, (k, k), 0)
    residual = np.abs(gray - low)

    rback = residual[backdrop > 0.2]
    if rback.size == 0:
        return img_f01
    med = float(np.median(rback))
    factor = 8.0 - 6.0 * s  # higher strength -> lower threshold -> more removed
    thr = med * factor + 0.01  # absolute floor avoids removing gentle texture
    defect = (residual > thr) & (backdrop > 0.2)

    # Drop large connected components (legit background texture, not defects)
    nb, labels = cv2.connectedComponents(defect.astype(np.uint8))
    keep = np.zeros_like(defect)
    max_area = int(h * w * _MAX_COMPONENT_FRAC)
    for c in range(1, nb):
        if int((labels == c).sum()) < max_area:
            keep |= labels == c
    defect = keep.astype(np.uint8)
    if defect.sum() == 0:
        return img_f01

    img_u8 = np.clip(img_f01 * 255.0, 0, 255).astype(np.uint8)
    inp = cv2.inpaint(img_u8, defect * 255, 3, cv2.INPAINT_TELEA)
    inp_f = inp.astype(np.float32) / 255.0

    # Hard composite: the inpaint output is already smooth/continuous with
    # its surroundings, so a feathered blend mask only weakens removal of
    # tiny specks (a 2px defect blurred by a wide feather never reaches
    # full strength). The subject is already protected by the eroded backdrop
    # above, so a hard mask is safe here.
    m = defect.astype(np.float32) * s
    m3 = np.clip(m[:, :, None], 0, 1)
    return img_f01 * (1.0 - m3) + inp_f * m3
