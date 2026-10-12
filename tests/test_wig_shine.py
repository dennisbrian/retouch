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


# ---------------------------------------------------------------------------
# Wig Shine (retouch/wig_shine.py): the deglare rewrite
# ---------------------------------------------------------------------------

from retouch.utils import bgr_f32_to_lab_f32  # noqa: E402
from retouch.wig_shine import (  # noqa: E402
    _linear_to_srgb,
    _srgb_to_linear,
    matte_wig_shine,
)

FW = 200.0


def _wig_scene(scale: float = 1.0, band: bool = True, colour=(0.45, 0.62, 0.80)):
    """A blonde wig patch: vertical strands, a broad lit side, a gloss band.

    ``scale`` multiplies linear light (0.12 is a dark brown wig under the
    same light). The gloss is neutral light added on top of the fibre
    colour, as off real synthetic fibre, so it keeps some of the fibre's
    chroma: the case the first deglare missed.
    """
    rng = np.random.default_rng(7)
    h, w = 300, 300
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    strands = 1.0 + 0.18 * cv2.GaussianBlur(
        rng.standard_normal((1, w)).astype(np.float32), (0, 0), 1.0
    ).repeat(h, 0) + 0.04 * rng.standard_normal((h, w)).astype(np.float32)
    lit = 0.7 + 0.5 * (xx / w)  # broad form, not shine
    lin = np.array(colour, np.float32)[None, None] * (lit * strands)[..., None] * 0.5 * scale
    gloss = np.exp(-((yy - 110) ** 2) / (2 * 8.0**2)).astype(np.float32)
    if band:
        lin = lin + (0.9 * float(np.median(lin.mean(2))) * gloss * strands)[..., None]
    img = _linear_to_srgb(lin)
    mask = np.zeros((h, w), np.float32)
    mask[20:280, 20:280] = 1.0
    return img, mask, gloss > 0.5


def _L(img):
    return bgr_f32_to_lab_f32(img.astype(np.float32))[..., 0] * 100.0 / 255.0


def _C(img):
    lab = bgr_f32_to_lab_f32(img.astype(np.float32))
    return np.hypot(lab[..., 1] - 128.0, lab[..., 2] - 128.0)


def _band_lift(img, band):
    L = _L(img)
    ref = np.zeros_like(band)
    ref[150:200, 40:260] = True
    ref2 = np.zeros_like(band)
    ref2[40:70, 40:260] = True
    return float(L[band][:, None].mean() - 0.5 * (L[ref].mean() + L[ref2].mean()))


class TestWigShineMatte:
    def test_zero_strength_and_missing_mask_are_identity(self):
        img, mask, _ = _wig_scene()
        assert matte_wig_shine(img, mask, 0, FW) is img
        assert matte_wig_shine(img, None, 60, FW) is img
        assert matte_wig_shine(img, np.zeros_like(mask), 60, FW) is img

    def test_softens_colour_keeping_gloss_band(self):
        img, mask, band = _wig_scene()
        out = matte_wig_shine(img, mask, 100, FW)
        before, after = _band_lift(img, band), _band_lift(out, band)
        assert before > 8.0
        assert after < 0.6 * before, (before, after)

    def test_strength_is_monotone(self):
        img, mask, band = _wig_scene()
        lifts = [_band_lift(matte_wig_shine(img, mask, s, FW), band) for s in (0, 50, 100)]
        assert lifts[0] > lifts[1] > lifts[2]

    def test_fibre_colour_comes_back_not_grey(self):
        img, mask, band = _wig_scene()
        out = matte_wig_shine(img, mask, 100, FW)
        assert _C(out)[band].mean() > _C(img)[band].mean() + 1.0

    def test_strand_texture_kept(self):
        img, mask, band = _wig_scene()
        out = matte_wig_shine(img, mask, 100, FW)

        def fine_sd(x):
            L = _L(x)
            return float((L - cv2.GaussianBlur(L, (0, 0), 3))[band].std())

        assert fine_sd(out) > 0.8 * fine_sd(img)

    def test_lit_side_without_gloss_is_left_alone(self):
        img, mask, _ = _wig_scene(band=False)
        out = matte_wig_shine(img, mask, 100, FW)
        assert float(np.abs(_L(out) - _L(img))[mask > 0].mean()) < 0.5

    def test_same_relative_reduction_on_dark_wig(self):
        light, mask, band = _wig_scene()
        dark, _, _ = _wig_scene(scale=0.12)
        r_light = _band_lift(matte_wig_shine(light, mask, 100, FW), band) / _band_lift(light, band)
        r_dark = _band_lift(matte_wig_shine(dark, mask, 100, FW), band) / _band_lift(dark, band)
        assert r_dark < 0.65
        assert abs(r_light - r_dark) < 0.2, (r_light, r_dark)

    def test_outside_mask_and_eyebrows_untouched(self):
        img, mask, _ = _wig_scene()
        brow = np.zeros_like(mask)
        brow[100:120, 100:160] = 1.0
        out = matte_wig_shine(img, mask, 100, FW, exclude_mask=brow)
        assert np.array_equal(out[mask == 0], img[mask == 0])
        assert np.array_equal(out[104:116, 104:156], img[104:116, 104:156])

    def test_low_coherence_is_left_alone(self):
        img, mask, _ = _wig_scene()
        out = matte_wig_shine(img, mask, 100, FW, coherence=np.zeros_like(mask))
        assert out is img or np.array_equal(out, img)

    def test_uint8_in_uint8_out(self):
        img, mask, band = _wig_scene()
        u8 = np.clip(np.round(img), 0, 255).astype(np.uint8)
        out = matte_wig_shine(u8, mask, 100, FW)
        assert out.dtype == np.uint8
        assert _band_lift(out, band) < _band_lift(u8, band)

    def test_deglare_wig_dispatch_reaches_it(self):
        img, mask, band = _wig_scene()
        coh = np.ones_like(mask)
        out = deglare_wig(img, mask, np.zeros_like(mask), coh, strength=100, face_width=FW)
        assert _band_lift(out, band) < 0.6 * _band_lift(img, band)
