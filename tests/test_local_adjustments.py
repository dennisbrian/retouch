"""Tests for F3 local mask infrastructure.

Covers ``retouch.regions.radial_mask``, ``linear_mask``,
``LOCAL_ADJUSTMENT_OPS``, ``apply_local_adjustment``, and the
``RetouchEngine._stage_local_adjustments`` stage.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
import pytest

from retouch.engine import ProcessingContext, RetouchEngine
from retouch.regions import (
    LOCAL_ADJUSTMENT_OPS,
    apply_local_adjustment,
    linear_mask,
    radial_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def img_u8() -> np.ndarray:
    """64x64 uint8 BGR synthetic image with a discernible gradient."""
    h, w = 64, 64
    yy = np.linspace(20, 230, w, dtype=np.float32)
    xx = np.linspace(20, 230, h, dtype=np.float32)[:, np.newaxis]
    base = np.clip((yy[np.newaxis, :] + xx) / 2.0, 0, 255).astype(np.uint8)
    return np.stack([base, base, base], axis=-1).astype(np.uint8)


@pytest.fixture
def img_f32(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32)


@pytest.fixture
def full_mask() -> np.ndarray:
    return np.ones((64, 64), dtype=np.float32)


@pytest.fixture
def half_mask() -> np.ndarray:
    m = np.zeros((64, 64), dtype=np.float32)
    m[:, :32] = 1.0
    return m


@pytest.fixture
def engine() -> RetouchEngine:
    """RetouchEngine without heavy __init__ (detectors/parsers not needed
    for _stage_local_adjustments, which only imports apply_local_adjustment).
    """
    return RetouchEngine.__new__(RetouchEngine)


@pytest.fixture
def ctx() -> ProcessingContext:
    return ProcessingContext()


# ---------------------------------------------------------------------------
# radial_mask
# ---------------------------------------------------------------------------

class TestRadialMask:
    def test_shape(self) -> None:
        m = radial_mask(40, 60, 0.5, 0.5, 0.4)
        assert m.shape == (40, 60)

    def test_dtype_is_float32(self) -> None:
        m = radial_mask(40, 60, 0.5, 0.5, 0.4)
        assert m.dtype == np.float32

    def test_values_in_unit_range(self) -> None:
        m = radial_mask(80, 80, 0.5, 0.5, 0.4)
        assert m.min() >= 0.0
        assert m.max() <= 1.0

    def test_center_is_brightest(self) -> None:
        h, w = 80, 80
        m = radial_mask(h, w, 0.5, 0.5, 0.4, feather=0.3)
        cy, cx = h // 2, w // 2
        assert m[cy, cx] == m.max()
        assert m[cy, cx] >= 1.0 - 1e-6  # center is fully on

    def test_edges_are_dark(self) -> None:
        h, w = 80, 80
        m = radial_mask(h, w, 0.5, 0.5, 0.2, feather=0.3)
        # Corners should be 0 (outside radius + feather)
        assert m[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert m[-1, -1] == pytest.approx(0.0, abs=1e-6)
        assert m[0, -1] == pytest.approx(0.0, abs=1e-6)
        assert m[-1, 0] == pytest.approx(0.0, abs=1e-6)

    def test_center_value_greater_than_corner(self) -> None:
        m = radial_mask(80, 80, 0.5, 0.5, 0.4)
        assert m[40, 40] > m[0, 0]


# ---------------------------------------------------------------------------
# linear_mask
# ---------------------------------------------------------------------------

class TestLinearMask:
    def test_shape(self) -> None:
        m = linear_mask(40, 60, 0.0, 0.0, 1.0, 1.0)
        assert m.shape == (40, 60)

    def test_dtype_is_float32(self) -> None:
        m = linear_mask(40, 60, 0.0, 0.0, 1.0, 1.0)
        assert m.dtype == np.float32

    def test_values_in_unit_range(self) -> None:
        m = linear_mask(80, 80, 0.0, 0.0, 1.0, 1.0)
        assert m.min() >= 0.0
        assert m.max() <= 1.0

    def test_horizontal_gradient_direction(self) -> None:
        # start (left, x=0) -> end (right, x=1): mask grows left to right.
        m = linear_mask(20, 80, 0.5, 0.0, 0.5, 1.0, feather=0.0)
        assert m[10, 0] < m[10, 40] < m[10, 79]

    def test_vertical_gradient_direction(self) -> None:
        # start (top, y=0) -> end (bottom, y=1): mask grows top to bottom.
        m = linear_mask(80, 20, 0.0, 0.5, 1.0, 0.5, feather=0.0)
        assert m[0, 10] < m[40, 10] < m[79, 10]

    def test_end_brighter_than_start(self) -> None:
        m = linear_mask(40, 40, 0.0, 0.0, 1.0, 1.0)
        assert m[-1, -1] >= m[0, 0]


# ---------------------------------------------------------------------------
# apply_local_adjustment — dtype, ops, no-op, semantic
# ---------------------------------------------------------------------------

EXPECTED_OPS = {"exposure", "warmth", "saturation", "clarity", "smooth", "dodge", "burn"}


class TestLocalAdjustmentOps:
    def test_local_adjustment_ops_registry_has_seven(self) -> None:
        assert set(LOCAL_ADJUSTMENT_OPS.keys()) == EXPECTED_OPS

    def test_unknown_op_raises_value_error(
        self, img_u8: np.ndarray, full_mask: np.ndarray
    ) -> None:
        with pytest.raises(ValueError):
            apply_local_adjustment(img_u8, full_mask, "not_a_real_op", 0.5)

    def test_dtype_preserved_uint8(
        self, img_u8: np.ndarray, full_mask: np.ndarray
    ) -> None:
        out = apply_local_adjustment(img_u8, full_mask, "exposure", 0.5)
        assert out.dtype == np.uint8
        assert out.shape == img_u8.shape

    def test_dtype_preserved_float32(
        self, img_f32: np.ndarray, full_mask: np.ndarray
    ) -> None:
        out = apply_local_adjustment(img_f32, full_mask, "exposure", 0.5)
        assert out.dtype == np.float32
        assert out.shape == img_f32.shape

    @pytest.mark.parametrize("op_name", sorted(EXPECTED_OPS))
    def test_each_op_runs_and_preserves_dtype(
        self,
        img_u8: np.ndarray,
        full_mask: np.ndarray,
        op_name: str,
    ) -> None:
        # Use a moderate positive strength; ops clamp internally.
        out = apply_local_adjustment(img_u8, full_mask, op_name, 0.5)
        assert out.dtype == np.uint8
        assert out.shape == img_u8.shape
        # Output must be in uint8 range.
        assert out.min() >= 0
        assert out.max() <= 255

    @pytest.mark.parametrize("op_name", sorted(EXPECTED_OPS))
    def test_each_op_float32(
        self,
        img_f32: np.ndarray,
        full_mask: np.ndarray,
        op_name: str,
    ) -> None:
        out = apply_local_adjustment(img_f32, full_mask, op_name, 0.5)
        assert out.dtype == np.float32
        assert out.shape == img_f32.shape

    @pytest.mark.parametrize("op_name", sorted(EXPECTED_OPS))
    def test_zero_strength_is_noop(
        self,
        img_u8: np.ndarray,
        full_mask: np.ndarray,
        op_name: str,
    ) -> None:
        out = apply_local_adjustment(img_u8, full_mask, op_name, 0.0)
        assert np.array_equal(out, img_u8)

    @pytest.mark.parametrize("op_name", sorted(EXPECTED_OPS))
    def test_zero_strength_is_noop_float32(
        self,
        img_f32: np.ndarray,
        full_mask: np.ndarray,
        op_name: str,
    ) -> None:
        out = apply_local_adjustment(img_f32, full_mask, op_name, 0.0)
        assert np.array_equal(out, img_f32)

    def test_dodge_brightens_masked_region(
        self, img_u8: np.ndarray, half_mask: np.ndarray
    ) -> None:
        out = apply_local_adjustment(img_u8, half_mask, "dodge", 1.0)
        # Left half (mask=1) should be brighter than original; right half untouched.
        left_in = img_u8[:, :32].astype(np.float32).mean()
        left_out = out[:, :32].astype(np.float32).mean()
        right_in = img_u8[:, 32:].astype(np.float32).mean()
        right_out = out[:, 32:].astype(np.float32).mean()
        assert left_out > left_in + 1.0
        assert right_out == pytest.approx(right_in, abs=1.0)

    def test_burn_darkens_masked_region(
        self, img_u8: np.ndarray, half_mask: np.ndarray
    ) -> None:
        out = apply_local_adjustment(img_u8, half_mask, "burn", 1.0)
        left_in = img_u8[:, :32].astype(np.float32).mean()
        left_out = out[:, :32].astype(np.float32).mean()
        assert left_out < left_in - 1.0


class TestSemanticMaskIntersection:
    def test_semantic_mask_zeros_outside_region(
        self, img_u8: np.ndarray, full_mask: np.ndarray
    ) -> None:
        # semantic mask only covers top-left quadrant.
        sem = np.zeros((64, 64), dtype=np.float32)
        sem[:32, :32] = 1.0

        out = apply_local_adjustment(
            img_u8, full_mask, "dodge", 1.0, semantic_mask=sem
        )
        # Top-left should change; bottom-right should be untouched.
        tl_in = img_u8[:32, :32].astype(np.float32).mean()
        tl_out = out[:32, :32].astype(np.float32).mean()
        br_in = img_u8[32:, 32:].astype(np.float32).mean()
        br_out = out[32:, 32:].astype(np.float32).mean()
        assert tl_out > tl_in + 1.0
        assert br_out == pytest.approx(br_in, abs=1.0)

    def test_semantic_mask_none_means_full(
        self, img_u8: np.ndarray, full_mask: np.ndarray
    ) -> None:
        # No semantic mask: full_mask applies everywhere.
        out_no_sem = apply_local_adjustment(
            img_u8, full_mask, "dodge", 0.5, semantic_mask=None
        )
        out_default = apply_local_adjustment(img_u8, full_mask, "dodge", 0.5)
        assert np.array_equal(out_no_sem, out_default)

    def test_zero_semantic_mask_makes_noop(
        self, img_u8: np.ndarray, full_mask: np.ndarray
    ) -> None:
        sem = np.zeros((64, 64), dtype=np.float32)
        out = apply_local_adjustment(
            img_u8, full_mask, "dodge", 1.0, semantic_mask=sem
        )
        assert np.array_equal(out, img_u8)


class TestSoftMaskLinearCoverage:
    """Pin HARD-1: soft-mask coverage must be linear (m), not squared (m^2).

    Each op already applies the mask internally; compositing again in
    ``apply_local_adjustment`` would scale the effective coverage by m^2,
    steepening falloff for every soft brush. These tests assert the linear
    relationship, which fails on the pre-fix (m^2) behavior.
    """

    @staticmethod
    def _colorful_img() -> np.ndarray:
        # Random colorful image: every op (exposure, warmth, saturation,
        # clarity, smooth, dodge, burn) has a real, mask-proportional effect.
        rng = np.random.default_rng(7)
        return rng.integers(0, 255, (64, 64, 3)).astype(np.uint8)

    @staticmethod
    def _const_mask(level: float) -> np.ndarray:
        return np.full((64, 64), level, dtype=np.float32)

    def test_exposure_coverage_is_linear(self) -> None:
        img = self._colorful_img()
        full = apply_local_adjustment(img, self._const_mask(1.0), "exposure", 0.5)
        half = apply_local_adjustment(img, self._const_mask(0.5), "exposure", 0.5)
        delta_full = float(full.astype(np.float32).mean() - img.astype(np.float32).mean())
        delta_half = float(half.astype(np.float32).mean() - img.astype(np.float32).mean())
        assert abs(delta_full) > 1.0  # exposure actually changed the image
        ratio = delta_half / delta_full
        # Linear coverage => half mask gives ~half the shift (not a quarter).
        assert 0.4 < ratio < 0.6

    @pytest.mark.parametrize("op_name", sorted(EXPECTED_OPS))
    def test_each_op_soft_mask_is_linear(self, op_name: str) -> None:
        if op_name == "saturation":
            # Chroma scaling is linear in LCH, not in BGR mean; measured in
            # test_saturation_soft_mask_is_linear below.
            pytest.skip("saturation linearity measured in LCH space")
        img = self._colorful_img()
        full = apply_local_adjustment(img, self._const_mask(1.0), op_name, 0.5)
        half = apply_local_adjustment(img, self._const_mask(0.5), op_name, 0.5)
        delta_full = float(full.astype(np.float32).mean() - img.astype(np.float32).mean())
        delta_half = float(half.astype(np.float32).mean() - img.astype(np.float32).mean())
        if abs(delta_full) < 1.0:
            pytest.skip("op produced no measurable change at full mask")
        ratio = delta_half / delta_full
        assert 0.35 < ratio < 0.65

    @staticmethod
    def _lab_chroma_mean(img: np.ndarray) -> float:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        a = lab[:, :, 1] - 128.0
        b = lab[:, :, 2] - 128.0
        return float(np.sqrt(a * a + b * b).mean())

    def test_saturation_soft_mask_is_linear(self) -> None:
        img = self._colorful_img()
        full = apply_local_adjustment(img, self._const_mask(1.0), "saturation", 0.5)
        half = apply_local_adjustment(img, self._const_mask(0.5), "saturation", 0.5)
        c0 = self._lab_chroma_mean(img)
        d_full = self._lab_chroma_mean(full) - c0
        d_half = self._lab_chroma_mean(half) - c0
        assert abs(d_full) > 1.0
        ratio = d_half / d_full
        assert 0.35 < ratio < 0.65


# ---------------------------------------------------------------------------
# Engine _stage_local_adjustments
# ---------------------------------------------------------------------------

class TestStageLocalAdjustments:
    def test_empty_list_is_noop(
        self,
        engine: RetouchEngine,
        ctx: ProcessingContext,
        img_u8: np.ndarray,
    ) -> None:
        out = engine._stage_local_adjustments(img_u8, ctx, local_adjustments=[])
        assert np.array_equal(out, img_u8)

    def test_none_is_noop(
        self,
        engine: RetouchEngine,
        ctx: ProcessingContext,
        img_u8: np.ndarray,
    ) -> None:
        out = engine._stage_local_adjustments(img_u8, ctx, local_adjustments=None)
        assert np.array_equal(out, img_u8)

    def test_single_adjustment_modifies_image(
        self,
        engine: RetouchEngine,
        ctx: ProcessingContext,
        img_u8: np.ndarray,
        half_mask: np.ndarray,
    ) -> None:
        adj: Dict[str, Any] = {
            "mask": half_mask,
            "op": "dodge",
            "strength": 1.0,
        }
        out = engine._stage_local_adjustments(
            img_u8, ctx, local_adjustments=[adj]
        )
        # Image must change somewhere.
        assert not np.array_equal(out, img_u8)
        # Left half (mask=1) brighter; right half unchanged.
        left_in = img_u8[:, :32].astype(np.float32).mean()
        left_out = out[:, :32].astype(np.float32).mean()
        right_in = img_u8[:, 32:].astype(np.float32).mean()
        right_out = out[:, 32:].astype(np.float32).mean()
        assert left_out > left_in + 1.0
        assert right_out == pytest.approx(right_in, abs=1.0)

    def test_multiple_adjustments_compose(
        self,
        engine: RetouchEngine,
        ctx: ProcessingContext,
        img_u8: np.ndarray,
        half_mask: np.ndarray,
    ) -> None:
        adjs: List[Dict[str, Any]] = [
            {"mask": half_mask, "op": "dodge", "strength": 1.0},
            {"mask": half_mask, "op": "burn", "strength": 1.0},
        ]
        out = engine._stage_local_adjustments(
            img_u8, ctx, local_adjustments=adjs
        )
        # Composed result differs from input and from a single dodge pass.
        single_dodge = engine._stage_local_adjustments(
            img_u8, ctx, local_adjustments=[adjs[0]]
        )
        assert not np.array_equal(out, img_u8)
        assert not np.array_equal(out, single_dodge)

    def test_zero_strength_skipped(
        self,
        engine: RetouchEngine,
        ctx: ProcessingContext,
        img_u8: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        adjs: List[Dict[str, Any]] = [
            {"mask": full_mask, "op": "dodge", "strength": 0.0},
        ]
        out = engine._stage_local_adjustments(
            img_u8, ctx, local_adjustments=adjs
        )
        assert np.array_equal(out, img_u8)

    def test_missing_mask_skipped(
        self,
        engine: RetouchEngine,
        ctx: ProcessingContext,
        img_u8: np.ndarray,
    ) -> None:
        adjs: List[Dict[str, Any]] = [
            {"op": "dodge", "strength": 1.0},  # no mask
        ]
        out = engine._stage_local_adjustments(
            img_u8, ctx, local_adjustments=adjs
        )
        assert np.array_equal(out, img_u8)

    def test_semantic_lookup_from_masks_dict(
        self,
        engine: RetouchEngine,
        ctx: ProcessingContext,
        img_u8: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        sem = np.zeros((64, 64), dtype=np.float32)
        sem[:32, :32] = 1.0
        sem_masks: Dict[str, np.ndarray] = {"skin": sem}

        adj: Dict[str, Any] = {
            "mask": full_mask,
            "op": "dodge",
            "strength": 1.0,
            "semantic": "skin",
        }
        out = engine._stage_local_adjustments(
            img_u8, ctx, local_adjustments=[adj], semantic_masks=sem_masks
        )
        # Top-left changed; bottom-right untouched.
        tl_in = img_u8[:32, :32].astype(np.float32).mean()
        tl_out = out[:32, :32].astype(np.float32).mean()
        br_in = img_u8[32:, 32:].astype(np.float32).mean()
        br_out = out[32:, 32:].astype(np.float32).mean()
        assert tl_out > tl_in + 1.0
        assert br_out == pytest.approx(br_in, abs=1.0)
