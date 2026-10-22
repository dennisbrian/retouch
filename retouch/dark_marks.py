"""Keep small dark marks when an op mattes the skin around them.

Beauty marks, drawn tear-moles and freckles are part of a cosplay's
makeup. Shine removal and powder finish pull lit skin down toward its local
baseline; a mark sitting on that lit skin then loses contrast even though
the op never touches the mark itself (rednote shrine photo: a tear-mole's
depth below the cheek went 28.7 -> 20.7 L at Powder Finish 50, and further
under Face Polish's shine removal).

``find_dark_marks`` records each mark's depth below a morphological closing
of L (the closing fills blobs narrower than the element, so it is the skin
around the mark). ``restore_dark_marks`` puts that depth back below the
closing of the edited L, only on the mark pixels, never lighter than the
edit left them. The threshold is in units of the face's own fine-texture
spread, so it is the same on lighter and darker skin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# A mark is at most this radius (fraction of the inter-eye distance).
MARK_MAX_RADIUS_IED = 0.06
# ... and sits more than this many fine-texture sigmas below the closing.
MARK_MIN_SIGMAS = 5.0
# Marks are not looked for within this distance of the eye mask (lashes
# crossing a loose skin mask would read as marks).
MARK_EYE_GAP_IED = 0.25
# Isolation: the median of the ring around a blob (1-2 radii out) must
# still be lighter than the blob by the threshold. Not a low percentile:
# a tear-mole sits next to the lower lash line and eye makeup.
MARK_RING_PERCENTILE = 50.0
# Lash band: the eye mask grown by this much is left out of that ring.
MARK_LASH_IED = 0.15


@dataclass
class DarkMarks:
    weight: np.ndarray   # (H, W) 0-1, the mark pixels (grown by 1 px)
    depth: np.ndarray    # (H, W) depth of L below its closing, before the edit
    kernel: np.ndarray   # closing element


def find_dark_marks(
    L: np.ndarray,
    core: np.ndarray,
    ied: float,
    spread: float,
    eyes_mask: Optional[np.ndarray] = None,
) -> Optional[DarkMarks]:
    """Find small dark marks on skin. ``None`` when there are none.

    Args:
        L: (H, W) float lightness, any scale (``spread`` must match it).
        core: (H, W) bool/0-1 confident skin.
        ied: inter-eye distance in pixels.
        spread: robust sigma of the skin's fine L texture, same scale as L.
        eyes_mask: optional (H, W) eye mask, 0-1.
    """
    r_m = max(1, int(round(float(ied) * MARK_MAX_RADIUS_IED)))
    k_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_m + 1, 2 * r_m + 1))
    L = L.astype(np.float32, copy=False)
    depth = cv2.morphologyEx(L, cv2.MORPH_CLOSE, k_m, borderType=cv2.BORDER_REFLECT) - L
    core_b = np.asarray(core) > 0
    thr = MARK_MIN_SIGMAS * max(float(spread), 1e-3)
    dark = (depth > thr) & core_b
    lashes = None
    if eyes_mask is not None and eyes_mask.shape == L.shape and eyes_mask.max() > 0:
        eye_u8 = (eyes_mask > 0.3).astype(np.uint8)
        r_e = max(1, int(round(float(ied) * MARK_EYE_GAP_IED)))
        near = cv2.dilate(
            eye_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_e + 1, 2 * r_e + 1)),
        ) > 0
        dark &= ~near
        # The lash line itself does not count against a mark's isolation.
        r_l = max(1, int(round(float(ied) * MARK_LASH_IED)))
        lashes = cv2.dilate(
            eye_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_l + 1, 2 * r_l + 1)),
        ) > 0
    n, lab_cc, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    # Only small blobs: brows, nostrils, lash lines and cast shadows are
    # bigger and stay with the op.
    small = np.zeros(n, dtype=bool)
    small[1:] = stats[1:, cv2.CC_STAT_AREA] <= np.pi * r_m * r_m
    # ... and isolated: at least half the skin ringing the blob is lighter than it
    # by the same margin. A notch on the rim of a shadow or a lash line has
    # its own dark area in the ring and is left to the op.
    h, w = L.shape
    k_in = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_m + 1, 2 * r_m + 1))
    k_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4 * r_m + 1, 4 * r_m + 1))
    for i in np.flatnonzero(small):
        x, y, bw, bh = stats[i, :4]
        x1, y1 = max(0, x - 2 * r_m), max(0, y - 2 * r_m)
        x2, y2 = min(w, x + bw + 2 * r_m), min(h, y + bh + 2 * r_m)
        blob = (lab_cc[y1:y2, x1:x2] == i).astype(np.uint8)
        ring = (cv2.dilate(blob, k_out) > 0) & ~(cv2.dilate(blob, k_in) > 0)
        ring &= np.asarray(core)[y1:y2, x1:x2] > 0
        if lashes is not None:
            ring &= ~lashes[y1:y2, x1:x2]
        if ring.sum() < 4:
            small[i] = False
            continue
        lit = np.percentile(L[y1:y2, x1:x2][ring], MARK_RING_PERCENTILE)
        if lit - float(L[y1:y2, x1:x2][blob > 0].min()) <= thr:
            small[i] = False
    dark = small[lab_cc]
    if not dark.any():
        return None
    weight = cv2.dilate(dark.astype(np.float32), np.ones((3, 3), np.uint8))
    return DarkMarks(weight=weight, depth=depth, kernel=k_m)


def restore_dark_marks(L_new: np.ndarray, marks: Optional[DarkMarks], protect: Optional[np.ndarray] = None) -> np.ndarray:
    """Return ``L_new`` with each mark back at its own depth below the skin."""
    if marks is None:
        return L_new
    w = marks.weight if protect is None else marks.weight * protect
    target = cv2.morphologyEx(L_new.astype(np.float32, copy=False), cv2.MORPH_CLOSE, marks.kernel, borderType=cv2.BORDER_REFLECT) - marks.depth
    return L_new + np.minimum(target - L_new, 0.0) * w


def fine_spread(L: np.ndarray, core: np.ndarray, ied: float, floor: float = 0.75) -> float:
    """Robust sigma of the skin's fine L texture (L minus a 0.02 IED blur)."""
    fine = max(0.8, float(ied) * 0.02)
    hp = (L - cv2.GaussianBlur(L, (0, 0), fine))[np.asarray(core) > 0]
    if hp.size == 0:
        return floor
    return max(floor, 1.4826 * float(np.median(np.abs(hp - np.median(hp)))))
