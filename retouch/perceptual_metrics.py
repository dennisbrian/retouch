"""Subject-relative perceptual calibration metrics.

These metrics are observational.  They describe feature contrast and the
skin's chroma/texture state without inferring age, ethnicity, health, or a
universal target appearance.  Recipe targets must be calibrated from a
reviewed corpus before they are used to drive an edit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import cv2
import numpy as np

from .utils import bgr_f32_to_lab_f32


_EPS = 1e-6
_FEATURES: Mapping[str, Sequence[str]] = {
    "eyes": ("left_eye", "right_eye"),
    "lips": ("lips",),
    "brows": ("left_eyebrow", "right_eyebrow"),
}


@dataclass(frozen=True)
class FeatureContrast:
    """Feature-versus-own-surrounding-skin contrast for one facial feature."""

    luminance_contrast: float
    chroma_contrast: float
    lab_delta: tuple[float, float, float]
    feature_pixels: int
    surround_pixels: int
    confidence: float


@dataclass(frozen=True)
class SkinHomogeneityState:
    """Blotch-scale chroma and pore-scale luminance state inside skin."""

    chroma_blotch_std: float
    pore_energy: float
    pore_retention: Optional[float]
    hemoglobin_variance: Optional[float]
    pixel_count: int
    confidence: float


def facial_feature_contrast(
    img_bgr: np.ndarray,
    regions: object,
    *,
    min_pixels: int = 32,
) -> Dict[str, FeatureContrast]:
    """Measure eyes, lips, and brows relative to each feature's own skin annulus.

    The metric is intentionally within-face.  It is robust to global exposure
    and skin-tone changes because every feature is compared only with nearby
    surrounding skin after all other feature masks have been excluded.
    """
    img = _as_bgr255(img_bgr)
    skin = _mask(getattr(regions, "skin", None), img.shape[:2])
    if skin is None:
        return {name: _empty_feature() for name in _FEATURES}

    masks = {
        name: _combine_masks([getattr(regions, attr, None) for attr in attrs], img.shape[:2])
        for name, attrs in _FEATURES.items()
    }
    all_features = np.maximum.reduce(list(masks.values()))
    face_width = _mask_width(skin)
    radius = max(2, int(round(face_width * 0.055)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    lab = bgr_f32_to_lab_f32(img)
    result: Dict[str, FeatureContrast] = {}

    for name, feature in masks.items():
        feature_region = (feature > 0.5) & (skin > 0.05)
        dilated = cv2.dilate((feature > 0.5).astype(np.uint8), kernel) > 0
        surround = dilated & (skin > 0.5) & ~(all_features > 0.1)
        result[name] = _feature_measurement(lab, feature_region, surround, min_pixels)
    return result


def skin_homogeneity_state(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    *,
    reference_img_bgr: Optional[np.ndarray] = None,
    hemoglobin: Optional[np.ndarray] = None,
) -> SkinHomogeneityState:
    """Measure chroma blotchiness while keeping pore texture separate.

    Chroma is measured in a face-width-scaled mid band.  Pore energy is a
    luminance high-band measure, so a smoother chroma field cannot score as an
    improvement if it removed real luminance texture.
    """
    img = _as_bgr255(img_bgr)
    skin = _mask(skin_mask, img.shape[:2])
    if skin is None:
        return SkinHomogeneityState(0.0, 0.0, None, None, 0, 0.0)
    selected = skin > 0.5
    count = int(np.count_nonzero(selected))
    if count < 32:
        return SkinHomogeneityState(0.0, 0.0, None, None, count, 0.0)

    lab = bgr_f32_to_lab_f32(img)
    face_width = _mask_width(skin)
    fine_sigma = max(0.8, face_width * 0.012)
    broad_sigma = max(2.0, face_width * 0.060)
    chroma = lab[..., 1:3] - 128.0
    fine = cv2.GaussianBlur(chroma, (0, 0), fine_sigma)
    broad = cv2.GaussianBlur(chroma, (0, 0), broad_sigma)
    blotch = fine - broad
    chroma_blotch_std = float(np.std(np.linalg.norm(blotch[selected], axis=1)))

    luminance = lab[..., 0]
    pore_sigma = max(0.6, face_width * 0.010)
    pore_energy = float(np.std((luminance - cv2.GaussianBlur(luminance, (0, 0), pore_sigma))[selected]))
    pore_retention = None
    if reference_img_bgr is not None and reference_img_bgr.shape == img_bgr.shape:
        ref = _as_bgr255(reference_img_bgr)
        ref_l = bgr_f32_to_lab_f32(ref)[..., 0]
        ref_energy = float(np.std((ref_l - cv2.GaussianBlur(ref_l, (0, 0), pore_sigma))[selected]))
        pore_retention = float(np.clip(pore_energy / max(ref_energy, _EPS), 0.0, 2.0))

    hb_variance = None
    if hemoglobin is not None and hemoglobin.shape == skin.shape:
        hb_variance = float(np.var(np.asarray(hemoglobin, dtype=np.float32)[selected]))

    confidence = float(np.clip(count / 800.0, 0.0, 1.0))
    return SkinHomogeneityState(
        chroma_blotch_std=chroma_blotch_std,
        pore_energy=pore_energy,
        pore_retention=pore_retention,
        hemoglobin_variance=hb_variance,
        pixel_count=count,
        confidence=confidence,
    )


def _feature_measurement(
    lab: np.ndarray,
    feature: np.ndarray,
    surround: np.ndarray,
    min_pixels: int,
) -> FeatureContrast:
    n_feature = int(np.count_nonzero(feature))
    n_surround = int(np.count_nonzero(surround))
    if n_feature < min_pixels or n_surround < min_pixels:
        return _empty_feature(n_feature, n_surround)
    f = np.mean(lab[feature], axis=0)
    s = np.mean(lab[surround], axis=0)
    delta = f - s
    l_contrast = float(delta[0] / max(float(f[0] + s[0]), _EPS))
    f_chroma = float(np.linalg.norm(f[1:] - 128.0))
    s_chroma = float(np.linalg.norm(s[1:] - 128.0))
    chroma_contrast = float(np.linalg.norm(delta[1:]) / max(f_chroma + s_chroma, 1.0))
    confidence = float(np.clip(min(n_feature, n_surround) / 500.0, 0.0, 1.0))
    return FeatureContrast(
        luminance_contrast=l_contrast,
        chroma_contrast=chroma_contrast,
        lab_delta=(float(delta[0]), float(delta[1]), float(delta[2])),
        feature_pixels=n_feature,
        surround_pixels=n_surround,
        confidence=confidence,
    )


def _empty_feature(feature_pixels: int = 0, surround_pixels: int = 0) -> FeatureContrast:
    return FeatureContrast(0.0, 0.0, (0.0, 0.0, 0.0), feature_pixels, surround_pixels, 0.0)


def _as_bgr255(img_bgr: np.ndarray) -> np.ndarray:
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be an HxWx3 BGR image")
    if not np.isfinite(img_bgr).all():
        raise ValueError("img_bgr must contain finite values")
    img = img_bgr.astype(np.float32, copy=False)
    if np.issubdtype(img_bgr.dtype, np.floating) and img.size and float(img.max()) <= 1.0:
        img = img * 255.0
    return np.clip(img, 0.0, 255.0).astype(np.float32, copy=False)


def _mask(mask: Optional[np.ndarray], shape: tuple[int, int]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    arr = np.asarray(mask)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[..., 0]
    if arr.shape != shape:
        return None
    arr = arr.astype(np.float32)
    if arr.size and float(arr.max()) > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _combine_masks(masks: Sequence[Optional[np.ndarray]], shape: tuple[int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    for item in masks:
        normalized = _mask(item, shape)
        if normalized is not None:
            out = np.maximum(out, normalized)
    return out


def _mask_width(mask: np.ndarray) -> int:
    ys, xs = np.nonzero(mask > 0.5)
    if len(xs) < 2:
        return max(mask.shape)
    return max(16, int(xs.max() - xs.min() + 1))
