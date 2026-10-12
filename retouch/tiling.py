"""Memory-Efficient Gigapixel Tiled Image Processing Engine.

Splits gigantic (16K/32K) high-resolution images into overlapping spatial tiles,
processes each tile independently through the engine pipeline, and reconstructs
the output using smooth cosine-feathered blending to prevent tile boundary seams.
"""

from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

# Auto-tile when longest side exceeds this (16K-class gigapixel inputs).
TILE_THRESHOLD_DIM = 8192
DEFAULT_TILE_SIZE = 2048
DEFAULT_TILE_OVERLAP = 128


def process_in_tiles(
    img_bgr: np.ndarray,
    process_fn: Callable[[np.ndarray], np.ndarray],
    tile_size: int = 2048,
    overlap: int = 128,
) -> np.ndarray:
    """Process an image in spatial tiles with smooth seam-free overlap blending.

    Args:
        img_bgr: (H, W, 3) uint8 or float32 BGR image.
        process_fn: Pure function mapping tile (h_tile, w_tile, 3) -> (h_tile, w_tile, 3).
        tile_size: Square tile dimension in pixels.
        overlap: Overlap margin in pixels along tile borders.

    Returns:
        (H, W, 3) BGR image reconstructed with smooth seam-free tile blending.
    """
    H, W = img_bgr.shape[:2]
    if H <= tile_size and W <= tile_size:
        return process_fn(img_bgr)

    is_float = img_bgr.dtype == np.float32
    out_accum = np.zeros((H, W, 3), dtype=np.float32)
    weight_accum = np.zeros((H, W, 1), dtype=np.float32)

    step = tile_size - overlap

    for y in range(0, H, step):
        for x in range(0, W, step):
            y1 = y
            y2 = min(y + tile_size, H)
            x1 = x
            x2 = min(x + tile_size, W)

            tile = img_bgr[y1:y2, x1:x2]
            processed_tile = process_fn(tile).astype(np.float32)

            th = y2 - y1
            tw = x2 - x1

            # Cosine window for seamless tile blending
            wy = np.sin(np.pi * (np.arange(th) + 0.5) / float(th))
            wx = np.sin(np.pi * (np.arange(tw) + 0.5) / float(tw))
            tile_weight = np.outer(wy, wx)[:, :, np.newaxis].astype(np.float32)

            out_accum[y1:y2, x1:x2] += processed_tile * tile_weight
            weight_accum[y1:y2, x1:x2] += tile_weight

    weight_accum = np.maximum(weight_accum, 1e-6)
    reconstructed = out_accum / weight_accum

    if is_float:
        return reconstructed.astype(np.float32)
    return np.clip(reconstructed + 0.5, 0, 255).astype(np.uint8)
