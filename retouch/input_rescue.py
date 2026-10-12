"""Non-mutating input-quality preflight for conservative rescue decisions."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .utils import bgr_f32_to_lab_f32


@dataclass(frozen=True)
class InputQualityPreflight:
    """Evidence only; callers choose whether an opt-in repair is appropriate."""

    chroma_noise: float
    jpeg_grid_evidence: float
    flat_fraction: float
    confidence: float
    suggested_rescue: bool


def analyze_input_quality(img_bgr: np.ndarray) -> InputQualityPreflight:
    """Estimate chroma-noise and 8-pixel JPEG-grid evidence without mutation."""
    img = _as_bgr255(img_bgr)
    if min(img.shape[:2]) < 16:
        return InputQualityPreflight(0.0, 0.0, 0.0, 0.0, False)
    lab = bgr_f32_to_lab_f32(img)
    luminance = lab[..., 0]
    gx = cv2.Sobel(luminance, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(luminance, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gx, gy)
    threshold = float(np.quantile(gradient, 0.35))
    flat = gradient <= max(threshold, 0.5)
    flat_fraction = float(np.mean(flat))

    chroma = lab[..., 1:3] - 128.0
    residual = chroma - cv2.GaussianBlur(chroma, (0, 0), 1.0)
    chroma_noise = float(np.median(np.linalg.norm(residual[flat], axis=1))) if np.any(flat) else 0.0

    grid = _jpeg_grid_evidence(luminance)
    confidence = float(np.clip(flat_fraction / 0.35, 0.0, 1.0))
    suggested = bool(confidence > 0.5 and (chroma_noise > 1.75 or grid > 0.22))
    return InputQualityPreflight(chroma_noise, grid, flat_fraction, confidence, suggested)


def _jpeg_grid_evidence(luminance: np.ndarray) -> float:
    """Compare gradients on 8-pixel boundaries against neighbouring columns/rows."""
    dx = np.abs(np.diff(luminance, axis=1))
    dy = np.abs(np.diff(luminance, axis=0))
    values = []
    for grad, axis in ((dx, 1), (dy, 0)):
        axis_length = grad.shape[axis]
        indices = np.arange(axis_length)
        boundary = (indices + 1) % 8 == 0
        if np.any(boundary) and np.any(~boundary):
            on = float(np.mean(grad[:, boundary])) if axis == 1 else float(np.mean(grad[boundary, :]))
            off = float(np.mean(grad[:, ~boundary])) if axis == 1 else float(np.mean(grad[~boundary, :]))
            values.append(max(0.0, (on - off) / max(on + off, 1e-6)))
    return float(max(values, default=0.0))


def _as_bgr255(img_bgr: np.ndarray) -> np.ndarray:
    if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError("img_bgr must be an HxWx3 BGR image")
    if not np.isfinite(img_bgr).all():
        raise ValueError("img_bgr must contain finite values")
    img = img_bgr.astype(np.float32, copy=False)
    if np.issubdtype(img_bgr.dtype, np.floating) and img.size and float(img.max()) <= 1.0:
        img = img * 255.0
    return np.clip(img, 0.0, 255.0).astype(np.float32, copy=False)
