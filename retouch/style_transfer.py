"""Subject-aware color transfer helpers.

Stateless helpers that perform LAB-space color matching from a reference image
to a target image, restricted to soft masks for subject regions (background,
skin, hair).  These functions live in their own module so the engine can
import them without pulling in ``retouch.style`` (which keeps the engine
module free of the style-extraction machinery and avoids a circular import
between the engine and the style profile dataclass).

The functions accept an ``engine`` argument but only use the duck-typed
``engine._detector`` and ``engine._parser`` attributes, so any object that
exposes those attributes is a valid caller.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from .utils import normalize_mask


def weighted_mean_std(data: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the weighted mean and standard deviation of a LAB/BGR image under a soft mask."""
    h, w = data.shape[:2]
    # Downsample if image is > 1 MP to save massive memory and CPU time
    if h * w > 1024 * 1024:
        scale = 1024.0 / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        data = cv2.resize(data, (nw, nh), interpolation=cv2.INTER_AREA)
        weights = cv2.resize(weights, (nw, nh), interpolation=cv2.INTER_AREA)

    w = weights[:, :, np.newaxis] if weights.ndim == 2 else weights
    sum_w = np.sum(w)
    if sum_w < 1e-3:
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)

    mean = np.sum(data * w, axis=(0, 1)) / sum_w
    variance = np.sum(((data - mean) ** 2) * w, axis=(0, 1)) / sum_w
    std = np.sqrt(variance) + 1e-5
    return mean, std


def reinhard_transfer_masked(
    src_img: np.ndarray,
    ref_img: np.ndarray,
    src_mask: np.ndarray,
    ref_mask: np.ndarray,
) -> np.ndarray:
    """Perform Reinhard color transfer from ref_img to src_img, restricted to the masks."""
    src_lab = cv2.cvtColor(src_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref_img, cv2.COLOR_BGR2LAB).astype(np.float32)

    # Compute soft mask weighted statistics to avoid hard thresholds
    mean_src, std_src = weighted_mean_std(src_lab, src_mask)
    mean_ref, std_ref = weighted_mean_std(ref_lab, ref_mask)

    if np.sum(src_mask) < 0.1 or np.sum(ref_mask) < 0.1:
        return src_img.copy()

    trans_lab = src_lab.copy()

    # Match channel-wise: (val - mean_src) * (std_ref / std_src) + mean_ref
    # Then blend back using the soft mask
    for c in range(3):
        val = src_lab[:, :, c]
        # Guard against ratio explosion if src std is extremely low
        if std_src[c] < 1e-3:
            continue
        ratio = np.clip(std_ref[c] / std_src[c], 0.3, 3.0)
        trans_val = (val - mean_src[c]) * ratio + mean_ref[c]
        trans_lab[:, :, c] = val * (1.0 - src_mask) + trans_val * src_mask

    trans_lab = np.clip(trans_lab, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(trans_lab, cv2.COLOR_LAB2BGR)


def subject_aware_transfer(
    engine: Any,
    target_img: np.ndarray,
    ref_img: np.ndarray,
    target_faces: Optional[List[Any]] = None,
    target_person: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Segment and match skin, hair, and background color statistics independently from the source image.

    This avoids compounding transfer and color drift in overlapping regions.
    """
    # 1. Target masks
    target_faces = target_faces if target_faces is not None else engine._detector.detect(target_img)
    if target_person is None:
        target_person = engine._detector.segment_person(target_img)

    target_person_f = normalize_mask(target_person)
    if target_person_f.ndim == 3:
        target_person_f = target_person_f[:, :, 0]

    target_bg_mask = 1.0 - target_person_f

    # 2. Reference masks
    ref_faces = engine._detector.detect(ref_img)
    ref_person = engine._detector.segment_person(ref_img)

    ref_person_f = normalize_mask(ref_person)
    if ref_person_f.ndim == 3:
        ref_person_f = ref_person_f[:, :, 0]

    ref_bg_mask = 1.0 - ref_person_f

    # Base transfer for background
    result = reinhard_transfer_masked(target_img, ref_img, target_bg_mask, ref_bg_mask)

    # --- Skin & Hair Transfer (if faces detected in both) ---
    if target_faces and ref_faces:
        # Sort both by area descending for deterministic largest-face matching
        target_faces = sorted(target_faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        ref_faces = sorted(ref_faces, key=lambda f: f.bbox[2] * f.bbox[3], reverse=True)
        t_face = target_faces[0]
        r_face = ref_faces[0]

        t_regions = engine._parser.parse(
            t_face.landmarks, target_img, t_face.bbox, target_person_f, t_face.ied
        )
        r_regions = engine._parser.parse(
            r_face.landmarks, ref_img, r_face.bbox, ref_person_f, r_face.ied
        )

        # Skin transfer (matched from original target_img and blended back)
        if t_regions.skin is not None and r_regions.skin is not None:
            t_skin = normalize_mask(t_regions.skin)
            r_skin = normalize_mask(r_regions.skin)

            skin_trans = reinhard_transfer_masked(target_img, ref_img, t_skin, r_skin)
            t_skin_3d = t_skin[:, :, np.newaxis]
            result = (result.astype(np.float32) * (1.0 - t_skin_3d) + skin_trans.astype(np.float32) * t_skin_3d).astype(np.uint8)

        # Hair transfer (matched from original target_img and blended back)
        if t_regions.hair is not None and r_regions.hair is not None:
            t_hair = normalize_mask(t_regions.hair)
            r_hair = normalize_mask(r_regions.hair)

            hair_trans = reinhard_transfer_masked(target_img, ref_img, t_hair, r_hair)
            t_hair_3d = t_hair[:, :, np.newaxis]
            result = (result.astype(np.float32) * (1.0 - t_hair_3d) + hair_trans.astype(np.float32) * t_hair_3d).astype(np.uint8)

    return result
