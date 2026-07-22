"""Tests for retouch/tiling.py — Gigapixel Tiled Processing Engine."""

import numpy as np
import pytest

from retouch.tiling import process_in_tiles


class TestTiledProcessing:
    """Tests for BB8 gigapixel tiled processing and seam-free blending."""

    def test_identity_tile_reconstruction(self):
        """Identity transform through process_in_tiles must reconstruct the original image."""
        img = np.random.randint(20, 235, (512, 512, 3), dtype=np.uint8)
        reconstructed = process_in_tiles(img, lambda t: t, tile_size=128, overlap=32)
        diff = np.abs(img.astype(int) - reconstructed.astype(int)).max()
        assert diff <= 2, f"Tile reconstruction mismatch: max diff {diff} > 2"

    def test_small_image_bypass(self):
        """Image smaller than tile size bypasses tiling."""
        img = np.full((64, 64, 3), 150, dtype=np.uint8)
        out = process_in_tiles(img, lambda t: t * 0 + 200, tile_size=128, overlap=32)
        assert np.all(out == 200)
