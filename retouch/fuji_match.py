"""Full-resolution RAW calibration against a Fujifilm RAF preview JPEG.

The preview contains the camera's Film Simulation and rendering decisions but
is normally smaller than the developed RAW.  This module learns only global,
resolution-independent properties from that preview: a monotonic L* tone curve
and guarded Lab chroma statistics.  It can therefore apply the rendition to a
full-resolution RAW decode without upscaling the preview or copying its JPEG
artifacts.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .color_space import bgr_f32_to_lab_f32, lab_f32_to_bgr_f32

_EPS = 1e-6


def _proxy(img_bgr: np.ndarray, max_dim: int = 1024) -> np.ndarray:
    """Return a float32 proxy small enough for inexpensive calibration."""
    img = np.clip(img_bgr, 0.0, 255.0).astype(np.float32, copy=False)
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img
    scale = max_dim / float(longest)
    return cv2.resize(
        img,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32, copy=False)


def _quantile_lut(source_l: np.ndarray, reference_l: np.ndarray) -> np.ndarray:
    """Create a monotonic 256-entry L* mapping from two image distributions."""
    quantiles = np.linspace(0.0, 100.0, 256, dtype=np.float32)
    source_q = np.percentile(source_l, quantiles).astype(np.float32)
    reference_q = np.percentile(reference_l, quantiles).astype(np.float32)

    # Flat portions of a histogram create duplicate x values.  Collapse them
    # by averaging their destination values before interpolation.
    unique_source, inverse = np.unique(source_q, return_inverse=True)
    totals = np.bincount(inverse, weights=reference_q)
    counts = np.bincount(inverse)
    unique_reference = (totals / np.maximum(counts, 1)).astype(np.float32)
    samples = np.arange(256, dtype=np.float32) * (100.0 / 255.0)
    return np.interp(samples, unique_source, unique_reference).astype(np.float32)


def calibrate_to_fuji_preview(
    raw_bgr: np.ndarray,
    preview_bgr: np.ndarray,
    strength: float = 0.85,
    inplace: bool = False,
) -> np.ndarray:
    """Calibrate a full-resolution RAW decode to its camera JPEG preview.

    The calibration is intentionally global: it changes the tonal response and
    overall colour character while retaining RAW resolution and avoiding JPEG
    texture, sharpening, and noise-reduction artifacts.  *strength* blends
    from the native RAW rendition (0) to the learned Fuji match (1).

    Args:
        raw_bgr: Full-resolution float32 or uint8 BGR image in [0, 255].
        preview_bgr: Embedded camera JPEG in BGR [0, 255].
        strength: Blend amount in [0, 1].  ``0.85`` is a restrained default
            that preserves some RAW highlight latitude.
        inplace: Replace ``raw_bgr`` when it is already float32. Retouch uses
            this for a newly decoded RAW buffer to avoid a second ~300 MiB
            full-resolution allocation.
    """
    if raw_bgr.ndim != 3 or raw_bgr.shape[2] != 3:
        raise ValueError(f"raw_bgr must be HxWx3, got {raw_bgr.shape}")
    if preview_bgr.ndim != 3 or preview_bgr.shape[2] != 3:
        raise ValueError(f"preview_bgr must be HxWx3, got {preview_bgr.shape}")
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError(f"strength must be between 0 and 1, got {strength!r}")

    source = raw_bgr.astype(np.float32, copy=False)
    # Normal RAW ingest already guarantees this range.  Avoid a full-frame
    # clip/copy in the normal path: a 26 MP float image is ~300 MiB.
    if float(source.min()) < 0.0 or float(source.max()) > 255.0:
        source = np.clip(source, 0.0, 255.0).astype(np.float32, copy=False)
    if strength == 0.0:
        return source.copy()

    source_proxy = _proxy(source)
    preview_proxy = _proxy(preview_bgr)
    source_proxy_lab = bgr_f32_to_lab_f32(source_proxy)
    preview_proxy_lab = bgr_f32_to_lab_f32(preview_proxy)
    l_lut = _quantile_lut(source_proxy_lab[:, :, 0], preview_proxy_lab[:, :, 0])

    src_a = source_proxy_lab[:, :, 1]
    src_b = source_proxy_lab[:, :, 2]
    ref_a = preview_proxy_lab[:, :, 1]
    ref_b = preview_proxy_lab[:, :, 2]
    a_ratio = float(np.clip(ref_a.std() / (src_a.std() + _EPS), 0.3, 3.0))
    b_ratio = float(np.clip(ref_b.std() / (src_b.std() + _EPS), 0.3, 3.0))

    src_a_mean = float(src_a.mean())
    src_b_mean = float(src_b.mean())
    ref_a_mean = float(ref_a.mean())
    ref_b_mean = float(ref_b.mean())

    # Work in strips instead of allocating several full-frame LAB/BGR working
    # buffers.  This retains the full RAW image plus one output image, while
    # limiting temporary calibration memory to roughly 10--20 MiB at 26 MP.
    result = source if inplace else np.empty_like(source, dtype=np.float32)
    tile_rows = 128
    for y0 in range(0, source.shape[0], tile_rows):
        y1 = min(source.shape[0], y0 + tile_rows)
        source_tile = source[y0:y1]
        lab_tile = bgr_f32_to_lab_f32(source_tile)
        l_index = np.clip(
            np.rint(lab_tile[:, :, 0] * 2.55), 0, 255
        ).astype(np.uint8)
        lab_tile[:, :, 0] = l_lut[l_index]
        lab_tile[:, :, 1] = (lab_tile[:, :, 1] - src_a_mean) * a_ratio + ref_a_mean
        lab_tile[:, :, 2] = (lab_tile[:, :, 2] - src_b_mean) * b_ratio + ref_b_mean
        calibrated_tile = lab_f32_to_bgr_f32(lab_tile)
        result[y0:y1] = source_tile * (1.0 - strength) + calibrated_tile * strength

    return result
