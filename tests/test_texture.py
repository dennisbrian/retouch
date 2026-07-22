"""Tests for retouch/texture.py — Micro-texture grain synthesis."""

import numpy as np
import pytest

from retouch.texture import synthesize_micro_grain


class TestMicroTextureGrain:
    """Tests for AA8 skin micro-texture grain synthesis."""

    def test_zero_intensity_noop(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        out = synthesize_micro_grain(img, intensity=0.0)
        assert np.all(out == img)

    def test_injects_micro_grain(self):
        img = np.full((64, 64, 3), 150, dtype=np.uint8)
        out = synthesize_micro_grain(img, intensity=0.1)
        assert not np.allclose(out, img)
        assert out.dtype == np.uint8

    def test_mask_contained(self):
        img = np.full((64, 64, 3), 150, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0

        out = synthesize_micro_grain(img, skin_mask=mask, intensity=0.2)
        # Outside mask (top-left 10x10) must be 100% unchanged
        assert np.all(out[0:10, 0:10] == img[0:10, 0:10])
