"""Dtype-awareness tests for retouch/heal.py, retouch/blemish.py, retouch/qa_detectors.py.

Verifies the float32 [0, 255] migration contract for the three modules:

  * heal.heal_region / blemish.inpaint_and_blend:
      - uint8 input  -> uint8 output
      - float32 input -> float32 output
      - float32 output stays within [0.0, 255.0]
      - float32 output is parity-close to the uint8 path (delta-applied)

  * blemish.BlemishRemover.remove:
      - uint8 input  -> uint8 output
      - float32 input -> float32 output (via apply_u8_op_float delta adapter)
      - float32 output stays within [0.0, 255.0]

  * qa_detectors.{detect_banding, detect_clipping, detect_plastic_skin,
                  detect_halo, detect_seam}:
      - float32 input does not crash
      - float32 input produces identical result to its uint8 truncation
        (analysis is run on a uint8 snapshot internally, per _to_u8_for_analysis)

Synthetic images only — no real photos. Python 3.9+ type hints.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import pytest

from retouch.blemish import BlemishRemover, inpaint_and_blend
from retouch.heal import heal_region
from retouch.qa_detectors import (
    detect_banding,
    detect_clipping,
    detect_halo,
    detect_plastic_skin,
    detect_seam,
    run_all,
)


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

H, W = 96, 96


def _assert_dtype_pair(out_u8: np.ndarray, out_f: np.ndarray) -> None:
    assert out_u8.dtype == np.uint8, "uint8 input must yield uint8 output"
    assert out_f.dtype == np.float32, "float32 input must yield float32 output"


def _assert_float_in_range(out_f: np.ndarray) -> None:
    assert out_f.min() >= 0.0, f"float32 output below 0: {out_f.min()}"
    assert out_f.max() <= 255.0, f"float32 output above 255: {out_f.max()}"


def _parity(u8_out: np.ndarray, f32_out: np.ndarray, max_diff: float = 2.0) -> None:
    """Float path should match the uint8 path within quantization tolerance."""
    d = np.abs(u8_out.astype(np.float32) - np.clip(f32_out, 0.0, 255.0))
    assert d.max() <= max_diff, f"max abs diff {d.max()} > {max_diff}"


# ---------------------------------------------------------------------------
# Synthetic images
# ---------------------------------------------------------------------------


def _skin_img_with_spot() -> np.ndarray:
    """Synthetic skin image with a dark blemish spot for inpainting.

    BGR warm skin tone (R > G > B) with a darker patch in the centre that
    inpainting/blemish removal has something to heal.
    """
    img = np.empty((H, W, 3), dtype=np.uint8)
    img[:, :, 0] = 120  # B
    img[:, :, 1] = 150  # G
    img[:, :, 2] = 180  # R  (warm skin, R > G > B)
    # Dark spot in the centre for heal/blemish to remove
    img[40:56, 40:56] = [30, 30, 30]
    # Slight texture so detectors have something to measure
    rng = np.random.RandomState(7)
    noise = rng.randint(-4, 5, (H, W, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _gradient_img() -> np.ndarray:
    """Smooth horizontal gradient — exercises banding/clipping detectors."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(W):
        img[:, i] = int(i * 255 / max(W - 1, 1))
    return img


def _clipping_img() -> np.ndarray:
    """Image with blown highlights and crushed blacks for clipping detector."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[: H // 2, :] = 255  # blown highlights
    img[H // 2 :, : W // 2] = 0  # crushed blacks
    img[H // 2 :, W // 2 :] = 128  # mid-tone
    return img


def _edge_img() -> np.ndarray:
    """Image with a sharp edge for halo/seam detectors."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:, : W // 2] = 30
    img[:, W // 2 :] = 220
    return img


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def u8_spot() -> np.ndarray:
    return _skin_img_with_spot()


@pytest.fixture
def f32_spot(u8_spot: np.ndarray) -> np.ndarray:
    return u8_spot.astype(np.float32)


@pytest.fixture
def u8_grad() -> np.ndarray:
    return _gradient_img()


@pytest.fixture
def f32_grad(u8_grad: np.ndarray) -> np.ndarray:
    return u8_grad.astype(np.float32)


@pytest.fixture
def u8_clip() -> np.ndarray:
    return _clipping_img()


@pytest.fixture
def f32_clip(u8_clip: np.ndarray) -> np.ndarray:
    return u8_clip.astype(np.float32)


@pytest.fixture
def u8_edge() -> np.ndarray:
    return _edge_img()


@pytest.fixture
def f32_edge(u8_edge: np.ndarray) -> np.ndarray:
    return u8_edge.astype(np.float32)


@pytest.fixture
def heal_mask() -> np.ndarray:
    """uint8 mask covering the dark spot in _skin_img_with_spot."""
    m = np.zeros((H, W), dtype=np.uint8)
    m[40:56, 40:56] = 255
    return m


@pytest.fixture
def heal_mask_float(heal_mask: np.ndarray) -> np.ndarray:
    """float32 [0, 1] version of heal_mask."""
    return (heal_mask.astype(np.float32) / 255.0)


@pytest.fixture
def skin_mask_full() -> np.ndarray:
    """Full-coverage float32 skin mask for BlemishRemover."""
    return np.ones((H, W), dtype=np.float32)


@pytest.fixture
def remover() -> BlemishRemover:
    return BlemishRemover()


@pytest.fixture
def person_mask_half() -> np.ndarray:
    """Person mask covering the left half — gives detect_seam a boundary."""
    m = np.zeros((H, W), dtype=np.float32)
    m[:, : W // 2] = 1.0
    return m


# ===========================================================================
# blemish.inpaint_and_blend
# ===========================================================================


class TestInpaintAndBlendFloat32:
    def test_preserves_dtype(
        self,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        heal_mask: np.ndarray,
    ) -> None:
        out_u8 = inpaint_and_blend(u8_spot, heal_mask, inpaint_radius=3)
        out_f = inpaint_and_blend(f32_spot, heal_mask, inpaint_radius=3)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self,
        f32_spot: np.ndarray,
        heal_mask: np.ndarray,
    ) -> None:
        out = inpaint_and_blend(f32_spot, heal_mask, inpaint_radius=3)
        _assert_float_in_range(out)

    def test_empty_mask_preserves_dtype(
        self,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
    ) -> None:
        empty = np.zeros((H, W), dtype=np.uint8)
        # mask.sum() == 0 short-circuits and returns input untouched
        out_u8 = inpaint_and_blend(u8_spot, empty, inpaint_radius=3)
        out_f = inpaint_and_blend(f32_spot, empty, inpaint_radius=3)
        _assert_dtype_pair(out_u8, out_f)
        assert np.array_equal(out_u8, u8_spot)
        assert np.array_equal(out_f, f32_spot)

    def test_ns_flags_preserves_dtype(
        self,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        heal_mask: np.ndarray,
    ) -> None:
        out_u8 = inpaint_and_blend(u8_spot, heal_mask, 3, flags=cv2.INPAINT_NS)
        out_f = inpaint_and_blend(f32_spot, heal_mask, 3, flags=cv2.INPAINT_NS)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_parity_with_uint8(
        self,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        heal_mask: np.ndarray,
    ) -> None:
        out_u8 = inpaint_and_blend(u8_spot, heal_mask, inpaint_radius=3)
        out_f = inpaint_and_blend(f32_spot, heal_mask, inpaint_radius=3)
        _parity(out_u8, out_f, max_diff=2.0)


# ===========================================================================
# heal.heal_region
# ===========================================================================


class TestHealRegionFloat32:
    def test_preserves_dtype(
        self,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        heal_mask: np.ndarray,
    ) -> None:
        out_u8 = heal_region(u8_spot, heal_mask, method="telea")
        out_f = heal_region(f32_spot, heal_mask, method="telea")
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self,
        f32_spot: np.ndarray,
        heal_mask: np.ndarray,
    ) -> None:
        out = heal_region(f32_spot, heal_mask, method="telea")
        _assert_float_in_range(out)

    def test_ns_method_preserves_dtype(
        self,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        heal_mask: np.ndarray,
    ) -> None:
        out_u8 = heal_region(u8_spot, heal_mask, method="ns")
        out_f = heal_region(f32_spot, heal_mask, method="ns")
        _assert_dtype_pair(out_u8, out_f)

    def test_empty_mask_preserves_dtype(
        self,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
    ) -> None:
        empty = np.zeros((H, W), dtype=np.uint8)
        out_u8 = heal_region(u8_spot, empty, method="telea")
        out_f = heal_region(f32_spot, empty, method="telea")
        _assert_dtype_pair(out_u8, out_f)
        assert np.array_equal(out_u8, u8_spot)
        assert np.array_equal(out_f, f32_spot)

    def test_float_mask_preserves_dtype(
        self,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        heal_mask_float: np.ndarray,
    ) -> None:
        # float32 [0,1] mask should be accepted for both input dtypes
        out_u8 = heal_region(u8_spot, heal_mask_float, method="telea")
        out_f = heal_region(f32_spot, heal_mask_float, method="telea")
        _assert_dtype_pair(out_u8, out_f)

    def test_explicit_radius_preserves_dtype(
        self,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        heal_mask: np.ndarray,
    ) -> None:
        out_u8 = heal_region(u8_spot, heal_mask, method="telea", radius=8)
        out_f = heal_region(f32_spot, heal_mask, method="telea", radius=8)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_parity_with_uint8(
        self,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        heal_mask: np.ndarray,
    ) -> None:
        out_u8 = heal_region(u8_spot, heal_mask, method="telea", radius=4)
        out_f = heal_region(f32_spot, heal_mask, method="telea", radius=4)
        _parity(out_u8, out_f, max_diff=2.0)


# ===========================================================================
# blemish.BlemishRemover.remove
# ===========================================================================


class TestBlemishRemoveFloat32:
    def test_preserves_dtype(
        self,
        remover: BlemishRemover,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        skin_mask_full: np.ndarray,
    ) -> None:
        out_u8 = remover.remove(u8_spot, skin_mask_full, strength=70)
        out_f = remover.remove(f32_spot, skin_mask_full, strength=70)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self,
        remover: BlemishRemover,
        f32_spot: np.ndarray,
        skin_mask_full: np.ndarray,
    ) -> None:
        out = remover.remove(f32_spot, skin_mask_full, strength=80)
        _assert_float_in_range(out)

    def test_zero_strength_preserves_dtype(
        self,
        remover: BlemishRemover,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        skin_mask_full: np.ndarray,
    ) -> None:
        # strength <= 0 short-circuits and returns the input untouched.
        out_u8 = remover.remove(u8_spot, skin_mask_full, strength=0)
        out_f = remover.remove(f32_spot, skin_mask_full, strength=0)
        _assert_dtype_pair(out_u8, out_f)
        assert np.array_equal(out_u8, u8_spot)
        assert np.array_equal(out_f, f32_spot)

    def test_float_parity_with_uint8(
        self,
        remover: BlemishRemover,
        u8_spot: np.ndarray,
        f32_spot: np.ndarray,
        skin_mask_full: np.ndarray,
    ) -> None:
        out_u8 = remover.remove(u8_spot, skin_mask_full, strength=70)
        out_f = remover.remove(f32_spot, skin_mask_full, strength=70)
        _parity(out_u8, out_f, max_diff=2.0)


# ===========================================================================
# qa_detectors — float32 input must not crash and must match uint8 result
# ===========================================================================


class TestQADetectorsFloat32:
    """All QA detectors accept float32 [0, 255] input. Internally they truncate
    to uint8 for analysis (_to_u8_for_analysis), so float32 input produces the
    identical result to running on the uint8 truncation of the same image."""

    def test_detect_banding_float32_no_crash(self, f32_grad: np.ndarray) -> None:
        result = detect_banding(f32_grad)
        assert isinstance(result, dict)
        assert "score" in result and "flagged" in result

    def test_detect_banding_float32_matches_uint8(
        self, u8_grad: np.ndarray, f32_grad: np.ndarray
    ) -> None:
        r_u8 = detect_banding(u8_grad)
        r_f = detect_banding(f32_grad)
        assert r_f["score"] == r_u8["score"]
        assert r_f["flagged"] == r_u8["flagged"]
        assert r_f["smooth_pixels"] == r_u8["smooth_pixels"]
        assert r_f["step_pixels"] == r_u8["step_pixels"]

    def test_detect_clipping_float32_no_crash(self, f32_clip: np.ndarray) -> None:
        result = detect_clipping(f32_clip)
        assert isinstance(result, dict)
        assert "score" in result and "flagged" in result

    def test_detect_clipping_float32_matches_uint8(
        self, u8_clip: np.ndarray, f32_clip: np.ndarray
    ) -> None:
        r_u8 = detect_clipping(u8_clip)
        r_f = detect_clipping(f32_clip)
        assert r_f["score"] == r_u8["score"]
        assert r_f["flagged"] == r_u8["flagged"]
        assert r_f["clipped_pixel_fraction"] == r_u8["clipped_pixel_fraction"]
        assert r_f["clipped_blob_fraction"] == r_u8["clipped_blob_fraction"]

    def test_detect_plastic_skin_float32_no_crash(self, f32_spot: np.ndarray) -> None:
        result = detect_plastic_skin(f32_spot)
        assert isinstance(result, dict)
        assert "score" in result and "flagged" in result

    def test_detect_plastic_skin_float32_matches_uint8(
        self, u8_spot: np.ndarray, f32_spot: np.ndarray
    ) -> None:
        r_u8 = detect_plastic_skin(u8_spot)
        r_f = detect_plastic_skin(f32_spot)
        assert r_f["score"] == r_u8["score"]
        assert r_f["flagged"] == r_u8["flagged"]
        assert r_f["hf_energy_ratio"] == pytest.approx(r_u8["hf_energy_ratio"])

    def test_detect_plastic_skin_with_reference_float32(
        self, u8_spot: np.ndarray, f32_spot: np.ndarray
    ) -> None:
        # Mixed dtype reference (float32 image, uint8 reference) must not crash
        # and must match the all-uint8 result.
        r_u8 = detect_plastic_skin(u8_spot, reference_img_bgr=u8_spot)
        r_f = detect_plastic_skin(f32_spot, reference_img_bgr=u8_spot)
        assert r_f["score"] == r_u8["score"]
        assert r_f["flagged"] == r_u8["flagged"]
        assert r_f["energy_loss_vs_reference"] == pytest.approx(
            r_u8["energy_loss_vs_reference"] or 0.0
        )

    def test_detect_halo_float32_no_crash(self, f32_edge: np.ndarray) -> None:
        result = detect_halo(f32_edge)
        assert isinstance(result, dict)
        assert "score" in result and "flagged" in result

    def test_detect_halo_float32_matches_uint8(
        self, u8_edge: np.ndarray, f32_edge: np.ndarray
    ) -> None:
        r_u8 = detect_halo(u8_edge)
        r_f = detect_halo(f32_edge)
        assert r_f["score"] == r_u8["score"]
        assert r_f["flagged"] == r_u8["flagged"]
        assert r_f["mean_overshoot"] == pytest.approx(r_u8["mean_overshoot"])
        assert r_f["edge_count"] == r_u8["edge_count"]

    def test_detect_seam_float32_no_crash(
        self, f32_edge: np.ndarray, person_mask_half: np.ndarray
    ) -> None:
        result = detect_seam(f32_edge, person_mask_half)
        assert isinstance(result, dict)
        assert "score" in result and "flagged" in result

    def test_detect_seam_float32_matches_uint8(
        self,
        u8_edge: np.ndarray,
        f32_edge: np.ndarray,
        person_mask_half: np.ndarray,
    ) -> None:
        r_u8 = detect_seam(u8_edge, person_mask_half)
        r_f = detect_seam(f32_edge, person_mask_half)
        assert r_f["score"] == r_u8["score"]
        assert r_f["flagged"] == r_u8["flagged"]
        assert r_f["seam_gradient"] == pytest.approx(r_u8["seam_gradient"])
        assert r_f["boundary_pixels"] == r_u8["boundary_pixels"]

    def test_run_all_float32_no_crash(self, f32_spot: np.ndarray) -> None:
        # run_all aggregates every detector — float32 input must not raise.
        results = run_all(f32_spot)
        assert isinstance(results, dict)
        for key in ("banding", "clipping", "plastic_skin", "halo", "seam"):
            assert key in results, f"run_all missing key: {key}"
            assert "score" in results[key] and "flagged" in results[key]

    def test_run_all_float32_matches_uint8(
        self, u8_spot: np.ndarray, f32_spot: np.ndarray
    ) -> None:
        r_u8 = run_all(u8_spot)
        r_f = run_all(f32_spot)
        for key in ("banding", "clipping", "plastic_skin", "halo", "seam"):
            assert r_f[key]["score"] == pytest.approx(r_u8[key]["score"]), (
                f"run_all[{key}].score mismatch: f32={r_f[key]['score']} u8={r_u8[key]['score']}"
            )
            assert r_f[key]["flagged"] == r_u8[key]["flagged"], (
                f"run_all[{key}].flagged mismatch: f32={r_f[key]['flagged']} u8={r_u8[key]['flagged']}"
            )
