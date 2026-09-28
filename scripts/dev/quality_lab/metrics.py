"""Image metric primitives for the Quality Lab. cv2/numpy only.

Rules baked in:
  * Never use absolute intensity thresholds on signals that scale with skin
    tone — every drift measure is a delta between two images, not an
    absolute gate (repo tone-invariance convention).
  * Pixel similarity is identity/regression evidence, NOT a quality score.
"""
from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np


def to_f32(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img.astype(np.float32)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Structural similarity on luma; images may be uint8 or float32."""
    fa, fb = to_f32(a), to_f32(b)
    ga = cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY)
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_a = cv2.GaussianBlur(ga, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(gb, (11, 11), 1.5)
    va = cv2.GaussianBlur(ga * ga, (11, 11), 1.5) - mu_a * mu_a
    vb = cv2.GaussianBlur(gb * gb, (11, 11), 1.5) - mu_b * mu_b
    vab = cv2.GaussianBlur(ga * gb, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * vab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (va + vb + c2))
    return float(np.mean(m))


def _lab(img: np.ndarray) -> np.ndarray:
    """Lab in OpenCV 0-255 scaling, float32."""
    f = to_f32(img)
    return cv2.cvtColor((f * 255).astype(np.uint8), cv2.COLOR_BGR2Lab).astype(np.float32)


def delta_e_mean(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Mean CIE76 ΔE between two images (optionally masked)."""
    la, lb = _lab(a), _lab(b)
    de = np.sqrt(((la - lb) ** 2).sum(axis=2))
    if mask is not None:
        m = mask > 0.5
        return float(de[m].mean()) if m.any() else 0.0
    return float(de.mean())


def channel_stats(img: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Mean L, a, b (OpenCV 0-255 Lab) over mask."""
    lab = _lab(img)
    m = mask > 0.5 if mask is not None else np.ones(lab.shape[:2], bool)
    if not m.any():
        return {"L": 0.0, "a": 0.0, "b": 0.0}
    return {k: float(lab[:, :, i][m].mean()) for i, k in enumerate(("L", "a", "b"))}


def clip_fraction(img: np.ndarray, kind: str = "highlight") -> float:
    """Fraction of pixels clipped (L>=250 highlight, L<=5 shadow)."""
    g = cv2.cvtColor(to_f32(img), cv2.COLOR_BGR2GRAY) * 255.0
    if kind == "highlight":
        return float((g >= 250).mean())
    return float((g <= 5).mean())


def texture_sigma(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """High-frequency energy: std of Laplacian (texture/detail presence)."""
    g = cv2.cvtColor(to_f32(img), cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    if mask is not None:
        m = mask > 0.5
        return float(lap[m].std()) if m.any() else 0.0
    return float(lap.std())


def edge_density(img: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Canny edge pixel density (lash/hair/edge presence)."""
    u = to_f32(img)
    g = (cv2.cvtColor(u, cv2.COLOR_BGR2GRAY) * 255).astype(np.uint8)
    med = float(np.median(g)) + 1e-6
    lo, hi = int(0.66 * med), int(min(255, 1.33 * med))
    edges = cv2.Canny(g, lo, hi) > 0
    if mask is not None:
        m = mask > 0.5
        return float(edges[m].mean()) if m.any() else 0.0
    return float(edges.mean())


def texture_retention(before: np.ndarray, after: np.ndarray,
                      mask: Optional[np.ndarray] = None) -> float:
    """after/before high-frequency energy ratio. 1.0 = fully retained."""
    b, a = texture_sigma(before, mask), texture_sigma(after, mask)
    return float(a / b) if b > 1e-6 else 1.0


def edge_retention(before: np.ndarray, after: np.ndarray,
                   mask: Optional[np.ndarray] = None) -> float:
    b, a = edge_density(before, mask), edge_density(after, mask)
    return float(a / b) if b > 1e-6 else 1.0
