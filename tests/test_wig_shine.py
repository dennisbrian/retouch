"""Tests for H2 wig shine shaping — deglare_wig + add_angel_ring.

Covers:
  - deglare reduces L on bright hair pixels
  - angel_ring adds a bright band
  - zero-strength is a no-op
  - dtype preservation (uint8 + float32)
  - flow field coherence (deglare follows flow direction)
  - mask discipline (outside-mask pixels byte-identical)
  - coherence floor suppresses the ring on low-coherence input
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from retouch.hairwork import deglare_wig, add_angel_ring, hair_flow


def _vertical_strand_image(size: int = 256) -> np.ndarray:
    """Synthetic image with vertical strands (vertical stripes)."""
    img = np.full((size, size, 3), 80, dtype=np.uint8)
    stripe_w = 6
    for x in range(0, size, 2 * stripe_w):
        img[:, x : x + stripe_w] = 160
    return img


def _add_glare_band(img: np.ndarray, y0: int, y1: int) -> np.ndarray:
    """Add a bright low-chroma horizontal glare band (synthetic specular)."""
    out = img.copy()
    band = out[y0:y1, :]
    band_f = band.astype(np.float32)
    band_f[:] = np.clip(band_f + 90.0, 0, 255)
    out[y0:y1, :] = band_f.astype(np.uint8)
    return out


def _full_hair_mask(size: int = 256) -> np.ndarray:
    return np.ones((size, size), dtype=np.float32)


@pytest.fixture
def face_width() -> float:
    return 120.0


class TestDeglare:
    def test_reduces_L_on_bright_pixels(self, face_width):
        size = 256
        img = _add_glare_band(_vertical_strand_image(size), 100, 140)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)

        result = deglare_wig(img, mask, orient, cohere,
                             strength=80, face_width=face_width)

        lab_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_out = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_in = lab_in[105:135, :, 0].mean()
        L_out = lab_out[105:135, :, 0].mean()
        assert L_out < L_in, f"deglare should reduce L: {L_in} -> {L_out}"
        assert (L_in - L_out) > 5.0, "expected a meaningful L reduction"

    def test_zero_strength_is_noop(self, face_width):
        size = 256
        img = _vertical_strand_image(size)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)
        result = deglare_wig(img, mask, orient, cohere,
                             strength=0, face_width=face_width)
        assert np.array_equal(result, img)

    def test_none_mask_is_noop(self, face_width):
        size = 128
        img = _vertical_strand_image(size)
        orient, cohere = hair_flow(img)
        result = deglare_wig(img, None, orient, cohere,
                             strength=80, face_width=face_width)
        assert np.array_equal(result, img)

    def test_dtype_uint8_preserved(self, face_width):
        size = 128
        img = _add_glare_band(_vertical_strand_image(size), 40, 80)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)
        result = deglare_wig(img, mask, orient, cohere,
                             strength=60, face_width=face_width)
        assert result.dtype == np.uint8

    def test_dtype_float32_preserved(self, face_width):
        size = 128
        img = _add_glare_band(_vertical_strand_image(size), 40, 80).astype(np.float32)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img.astype(np.uint8), hair_mask=mask)
        result = deglare_wig(img, mask, orient, cohere,
                             strength=60, face_width=face_width)
        assert result.dtype == np.float32

    def test_mask_discipline_outside_unchanged(self, face_width):
        size = 128
        img = _add_glare_band(_vertical_strand_image(size), 40, 90)
        mask = np.zeros((size, size), dtype=np.float32)
        mask[40:90, 40:90] = 1.0  # isolated blob
        orient, cohere = hair_flow(img, hair_mask=mask)
        result = deglare_wig(img, mask, orient, cohere,
                             strength=80, face_width=face_width)
        outside = np.zeros((size, size), dtype=bool)
        outside[:40, :] = True
        outside[90:, :] = True
        outside[:, :40] = True
        outside[:, 90:] = True
        assert np.array_equal(result[outside], img[outside])

    def test_flow_coherence_steers_deglare(self, face_width):
        # Vertical strands → strong vertical coherence. Deglare with flow
        # should still reduce L, but a zero-coherence input should not deglare
        # (the coherence floor gates it).
        size = 256
        img = _add_glare_band(_vertical_strand_image(size), 100, 140)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)
        result = deglare_wig(img, mask, orient, cohere,
                             strength=90, face_width=face_width)

        zero_cohere = np.zeros_like(cohere)
        result_zero = deglare_wig(img, mask, orient, zero_cohere,
                                  strength=90, face_width=face_width)

        lab_flow = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_zero = cv2.cvtColor(result_zero, cv2.COLOR_BGR2LAB).astype(np.float32)
        assert lab_flow[105:135, :, 0].mean() < lab_zero[105:135, :, 0].mean(), (
            "deglare with real coherence should reduce L more than with zero coherence"
        )


class TestAngelRing:
    def test_adds_bright_band(self, face_width):
        size = 256
        img = _vertical_strand_image(size)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)

        result = add_angel_ring(img, mask, orient, cohere,
                                strength=80, position=30, tint=40,
                                face_width=face_width)

        lab_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_out = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        assert lab_out[..., 0].mean() > lab_in[..., 0].mean(), (
            "angel ring should lift mean L"
        )
        assert lab_out[..., 0].max() > lab_in[..., 0].max(), (
            "angel ring should produce at least one brighter pixel"
        )

    def test_zero_strength_is_noop(self, face_width):
        size = 128
        img = _vertical_strand_image(size)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)
        result = add_angel_ring(img, mask, orient, cohere,
                                strength=0, face_width=face_width)
        assert np.array_equal(result, img)

    def test_none_mask_is_noop(self, face_width):
        size = 128
        img = _vertical_strand_image(size)
        orient, cohere = hair_flow(img)
        result = add_angel_ring(img, None, orient, cohere,
                                strength=80, face_width=face_width)
        assert np.array_equal(result, img)

    def test_dtype_uint8_preserved(self, face_width):
        size = 128
        img = _vertical_strand_image(size)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)
        result = add_angel_ring(img, mask, orient, cohere,
                                strength=60, face_width=face_width)
        assert result.dtype == np.uint8

    def test_dtype_float32_preserved(self, face_width):
        size = 128
        img = _vertical_strand_image(size).astype(np.float32)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img.astype(np.uint8), hair_mask=mask)
        result = add_angel_ring(img, mask, orient, cohere,
                                strength=60, face_width=face_width)
        assert result.dtype == np.float32

    def test_coherence_floor_suppresses_ring(self, face_width):
        # Zero coherence (fuzzy wig floor) → ring must not appear.
        size = 128
        img = _vertical_strand_image(size)
        mask = _full_hair_mask(size)
        orient, _ = hair_flow(img, hair_mask=mask)
        zero_cohere = np.zeros((size, size), dtype=np.float32)
        result = add_angel_ring(img, mask, orient, zero_cohere,
                                strength=100, face_width=face_width)
        assert np.allclose(result, img, atol=1), (
            "zero-coherence input should produce no ring (coherence floor)"
        )

    def test_no_highlight_clipping(self, face_width):
        size = 128
        img = _vertical_strand_image(size)
        img[40:80, 40:80] = 250  # already-near-clipping region
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)
        result = add_angel_ring(img, mask, orient, cohere,
                                strength=100, position=50, tint=0,
                                face_width=face_width)
        lab_out = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        clipped = np.sum(lab_out[..., 0] >= 254.5)
        total = lab_out[..., 0].size
        assert clipped / total < 0.005, (
            f"ring must not push >0.5% of pixels to clipping, got {clipped/total:.4%}"
        )


class TestAnisotropy:
    def test_ring_is_band_not_blob(self, face_width):
        # The angel ring is a band along the crown — its spatial extent must
        # be anisotropic (elongated along the crown arc vs across it). Verify
        # the longer axis is >= 2x the shorter axis (a blob would be ~1:1).
        size = 256
        img = _vertical_strand_image(size)
        mask = _full_hair_mask(size)
        orient, cohere = hair_flow(img, hair_mask=mask)
        result = add_angel_ring(img, mask, orient, cohere,
                                strength=80, position=30, tint=0,
                                face_width=face_width)

        lab_in = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_out = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        delta_L = lab_out[..., 0] - lab_in[..., 0]

        threshold = delta_L.max() * 0.3
        if threshold <= 0.1:
            pytest.skip("ring did not fire strongly enough to measure anisotropy")
        active = delta_L > threshold
        if active.sum() < 5:
            pytest.skip("too few active ring pixels")

        rows = np.where(active.any(axis=1))[0]
        cols = np.where(active.any(axis=0))[0]
        vertical_extent = rows[-1] - rows[0] if len(rows) > 1 else 0
        horizontal_extent = cols[-1] - cols[0] if len(cols) > 1 else 0
        if vertical_extent == 0 or horizontal_extent == 0:
            pytest.skip("degenerate extent")
        longer = max(vertical_extent, horizontal_extent)
        shorter = min(vertical_extent, horizontal_extent)
        assert longer >= 2 * shorter, (
            f"ring should be anisotropic (band, not blob): "
            f"v={vertical_extent}, h={horizontal_extent}, ratio={longer/shorter:.2f}"
        )
