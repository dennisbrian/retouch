"""Dtype-awareness tests for relight.py, regions.py, and tonal.py.

Verifies that the recently migrated dtype-aware functions preserve the
input dtype (uint8 -> uint8, float32 -> float32) and keep values within
the [0, 255] BGR working range.

Covers:
- retouch/relight.py: Relighter.relight (v1 + v2), Relighter.sculpt
- retouch/regions.py: _blend_multiply, _blend_screen, _blend_soft_light,
                      _blend_overlay, apply_to_region
- retouch/tonal.py:   apply_hd_curve (luma_only=True and False)
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import pytest

from retouch.relight import Relighter
from retouch.regions import (
    _blend_multiply,
    _blend_overlay,
    _blend_screen,
    _blend_soft_light,
    apply_to_region,
)
from retouch.tonal import apply_hd_curve


# ---------------------------------------------------------------------------
# Mock landmarks (matches the pattern in test_relight.py / test_sculpt.py)
# ---------------------------------------------------------------------------


class MockLandmark:
    """Mock single MediaPipe landmark with x, y, z attributes."""

    def __init__(self, x: float = 0.5, y: float = 0.5, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class MockLandmarksList:
    """Mock MediaPipe landmark list with 468 points and a frontal yaw ratio."""

    def __init__(self, yaw_ratio: float = 1.0) -> None:
        self.landmark = []
        for i in range(468):
            angle = i * 2.0 * math.pi / 468.0
            x = 0.5 + 0.3 * math.cos(angle)
            y = 0.5 + 0.3 * math.sin(angle)
            z = 0.1 * math.sin(angle * 3.0)
            self.landmark.append(MockLandmark(x, y, z))

        # Symmetric temple-to-nose distances (frontal face).
        self.landmark[6].x = 0.5
        self.landmark[234].x = 0.3
        self.landmark[454].x = 0.7

        if yaw_ratio != 1.0:
            shift = 0.2 * (yaw_ratio - 1.0) / (yaw_ratio + 1.0)
            self.landmark[6].x = 0.5 + shift


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


H, W = 96, 96


def _make_gradient_uint8() -> np.ndarray:
    """Synthetic BGR uint8 gradient image in [0, 255]."""
    img = np.empty((H, W, 3), dtype=np.uint8)
    for y in range(H):
        for c in range(3):
            img[:, :, c] = 0
        img[y, :, 0] = int(y * 255.0 / H)
        img[y, :, 1] = int((H - 1 - y) * 255.0 / H)
        img[y, :, 2] = 128
    return img


def _make_gradient_f32() -> np.ndarray:
    """Synthetic BGR float32 gradient image in [0, 255]."""
    img = np.empty((H, W, 3), dtype=np.float32)
    for y in range(H):
        img[y, :, 0] = float(y) * 255.0 / H
        img[y, :, 1] = float(H - 1 - y) * 255.0 / H
        img[y, :, 2] = 128.0
    return img


@pytest.fixture
def img_u8() -> np.ndarray:
    return _make_gradient_uint8()


@pytest.fixture
def img_f32() -> np.ndarray:
    return _make_gradient_f32()


@pytest.fixture
def skin_mask() -> np.ndarray:
    """Soft float32 mask covering the full frame in [0, 1]."""
    m = np.ones((H, W), dtype=np.float32) * 0.8
    return m


@pytest.fixture
def landmarks() -> MockLandmarksList:
    return MockLandmarksList()


@pytest.fixture
def relighter() -> Relighter:
    return Relighter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_dtype_preserved_and_ranged(
    out: np.ndarray, expected_dtype: np.dtype
) -> None:
    assert out.dtype == expected_dtype, (
        f"dtype not preserved: expected {expected_dtype}, got {out.dtype}"
    )
    if expected_dtype == np.float32:
        assert out.min() >= 0.0, f"float output min {out.min()} < 0.0"
        assert out.max() <= 255.0, f"float output max {out.max()} > 255.0"
    else:
        assert out.min() >= 0, f"uint8 output min {out.min()} < 0"
        assert out.max() <= 255, f"uint8 output max {out.max()} > 255"


# ---------------------------------------------------------------------------
# relight.py — Relighter.relight (v1 + v2) and .sculpt
# ---------------------------------------------------------------------------


class TestRelightDtype:
    """dtype preservation for Relighter.relight."""

    @pytest.mark.parametrize("engine", ["v1", "v2"])
    def test_relight_uint8_preserves_dtype(
        self,
        relighter: Relighter,
        img_u8: np.ndarray,
        skin_mask: np.ndarray,
        landmarks: MockLandmarksList,
        engine: str,
    ) -> None:
        out = relighter.relight(
            img_u8, landmarks, skin_mask,
            face_width=float(W), strength=80.0,
            azimuth=30.0, elevation=30.0, engine=engine,
        )
        _assert_dtype_preserved_and_ranged(out, np.uint8)
        assert out.shape == img_u8.shape

    @pytest.mark.parametrize("engine", ["v1", "v2"])
    def test_relight_float32_preserves_dtype(
        self,
        relighter: Relighter,
        img_f32: np.ndarray,
        skin_mask: np.ndarray,
        landmarks: MockLandmarksList,
        engine: str,
    ) -> None:
        out = relighter.relight(
            img_f32, landmarks, skin_mask,
            face_width=float(W), strength=80.0,
            azimuth=30.0, elevation=30.0, engine=engine,
        )
        _assert_dtype_preserved_and_ranged(out, np.float32)
        assert out.shape == img_f32.shape

    def test_relight_float32_changes_image(
        self,
        relighter: Relighter,
        img_f32: np.ndarray,
        skin_mask: np.ndarray,
        landmarks: MockLandmarksList,
    ) -> None:
        out = relighter.relight(
            img_f32, landmarks, skin_mask,
            face_width=float(W), strength=100.0,
            azimuth=45.0, elevation=30.0, engine="v2",
        )
        assert not np.array_equal(out, img_f32), "relight should change the float32 image"


class TestSculptDtype:
    """dtype preservation for Relighter.sculpt."""

    def test_sculpt_uint8_preserves_dtype(
        self,
        relighter: Relighter,
        img_u8: np.ndarray,
        skin_mask: np.ndarray,
        landmarks: MockLandmarksList,
    ) -> None:
        out = relighter.sculpt(
            img_u8, landmarks, skin_mask,
            face_width=float(W), strength=80.0,
        )
        _assert_dtype_preserved_and_ranged(out, np.uint8)
        assert out.shape == img_u8.shape

    def test_sculpt_float32_preserves_dtype(
        self,
        relighter: Relighter,
        img_f32: np.ndarray,
        skin_mask: np.ndarray,
        landmarks: MockLandmarksList,
    ) -> None:
        out = relighter.sculpt(
            img_f32, landmarks, skin_mask,
            face_width=float(W), strength=80.0,
        )
        _assert_dtype_preserved_and_ranged(out, np.float32)
        assert out.shape == img_f32.shape

    def test_sculpt_float32_changes_image(
        self,
        relighter: Relighter,
        img_f32: np.ndarray,
        skin_mask: np.ndarray,
        landmarks: MockLandmarksList,
    ) -> None:
        out = relighter.sculpt(
            img_f32, landmarks, skin_mask,
            face_width=float(W), strength=100.0,
        )
        assert not np.array_equal(out, img_f32), "sculpt should change the float32 image"


# ---------------------------------------------------------------------------
# regions.py — blend primitives + apply_to_region
# ---------------------------------------------------------------------------


class TestBlendPrimitivesDtype:
    """dtype preservation for the four blend primitives."""

    @pytest.mark.parametrize(
        "blend_fn",
        [_blend_multiply, _blend_screen, _blend_soft_light, _blend_overlay],
        ids=["multiply", "screen", "soft_light", "overlay"],
    )
    def test_blend_uint8_preserves_dtype(
        self,
        blend_fn,
        img_u8: np.ndarray,
    ) -> None:
        out = blend_fn(img_u8, img_u8)
        _assert_dtype_preserved_and_ranged(out, np.uint8)
        assert out.shape == img_u8.shape

    @pytest.mark.parametrize(
        "blend_fn",
        [_blend_multiply, _blend_screen, _blend_soft_light, _blend_overlay],
        ids=["multiply", "screen", "soft_light", "overlay"],
    )
    def test_blend_float32_preserves_dtype(
        self,
        blend_fn,
        img_f32: np.ndarray,
    ) -> None:
        out = blend_fn(img_f32, img_f32)
        _assert_dtype_preserved_and_ranged(out, np.float32)
        assert out.shape == img_f32.shape


class TestApplyToRegionDtype:
    """dtype preservation for apply_to_region across blend modes."""

    @staticmethod
    def _op_brighten(img: np.ndarray) -> np.ndarray:
        return np.clip(img.astype(np.float32) + 30.0, 0.0, 255.0).astype(img.dtype)

    @pytest.mark.parametrize(
        "blend_mode",
        ["normal", "multiply", "screen", "soft_light", "overlay"],
    )
    def test_apply_to_region_uint8_preserves_dtype(
        self,
        img_u8: np.ndarray,
        skin_mask: np.ndarray,
        blend_mode: str,
    ) -> None:
        out = apply_to_region(
            img_u8, skin_mask, self._op_brighten,
            blend_mode=blend_mode, strength=1.0,
        )
        _assert_dtype_preserved_and_ranged(out, np.uint8)
        assert out.shape == img_u8.shape

    @pytest.mark.parametrize(
        "blend_mode",
        ["normal", "multiply", "screen", "soft_light", "overlay"],
    )
    def test_apply_to_region_float32_preserves_dtype(
        self,
        img_f32: np.ndarray,
        skin_mask: np.ndarray,
        blend_mode: str,
    ) -> None:
        out = apply_to_region(
            img_f32, skin_mask, self._op_brighten,
            blend_mode=blend_mode, strength=1.0,
        )
        _assert_dtype_preserved_and_ranged(out, np.float32)
        assert out.shape == img_f32.shape

    def test_apply_to_region_float32_changes_image(
        self,
        img_f32: np.ndarray,
        skin_mask: np.ndarray,
    ) -> None:
        out = apply_to_region(
            img_f32, skin_mask, self._op_brighten,
            blend_mode="screen", strength=1.0,
        )
        assert not np.array_equal(out, img_f32), "apply_to_region should change the float32 image"


# ---------------------------------------------------------------------------
# tonal.py — apply_hd_curve
# ---------------------------------------------------------------------------


class TestApplyHdCurveDtype:
    """dtype preservation for apply_hd_curve (luma_only True/False)."""

    @pytest.mark.parametrize("luma_only", [True, False])
    def test_hd_curve_uint8_preserves_dtype(
        self,
        img_u8: np.ndarray,
        luma_only: bool,
    ) -> None:
        out = apply_hd_curve(
            img_u8, strength=0.8, toe=0.1, shoulder=0.1,
            midpoint=0.5, gamma=1.0, luma_only=luma_only,
        )
        _assert_dtype_preserved_and_ranged(out, np.uint8)
        assert out.shape == img_u8.shape

    @pytest.mark.parametrize("luma_only", [True, False])
    def test_hd_curve_float32_preserves_dtype(
        self,
        img_f32: np.ndarray,
        luma_only: bool,
    ) -> None:
        out = apply_hd_curve(
            img_f32, strength=0.8, toe=0.1, shoulder=0.1,
            midpoint=0.5, gamma=1.0, luma_only=luma_only,
        )
        _assert_dtype_preserved_and_ranged(out, np.float32)
        assert out.shape == img_f32.shape

    def test_hd_curve_float32_luma_changes_image(
        self,
        img_f32: np.ndarray,
    ) -> None:
        out = apply_hd_curve(
            img_f32, strength=1.0, luma_only=True,
        )
        assert not np.array_equal(out, img_f32), "hd_curve (luma) should change the float32 image"

    def test_hd_curve_float32_rgb_changes_image(
        self,
        img_f32: np.ndarray,
    ) -> None:
        out = apply_hd_curve(
            img_f32, strength=1.0, luma_only=False,
        )
        assert not np.array_equal(out, img_f32), "hd_curve (rgb) should change the float32 image"


# ---------------------------------------------------------------------------
# Early-return identity path (strength=0) — must also preserve dtype
# ---------------------------------------------------------------------------


class TestNoOpPreservesDtype:
    """strength=0 / None-mask early returns must preserve dtype exactly."""

    def test_relight_zero_strength_float32(
        self,
        relighter: Relighter,
        img_f32: np.ndarray,
        skin_mask: np.ndarray,
        landmarks: MockLandmarksList,
    ) -> None:
        out = relighter.relight(
            img_f32, landmarks, skin_mask,
            face_width=float(W), strength=0.0, engine="v2",
        )
        assert out is img_f32 or np.array_equal(out, img_f32)
        assert out.dtype == np.float32

    def test_sculpt_zero_strength_float32(
        self,
        relighter: Relighter,
        img_f32: np.ndarray,
        skin_mask: np.ndarray,
        landmarks: MockLandmarksList,
    ) -> None:
        out = relighter.sculpt(
            img_f32, landmarks, skin_mask,
            face_width=float(W), strength=0.0,
        )
        assert out is img_f32 or np.array_equal(out, img_f32)
        assert out.dtype == np.float32

    def test_hd_curve_zero_strength_float32(
        self,
        img_f32: np.ndarray,
    ) -> None:
        out = apply_hd_curve(img_f32, strength=0.0)
        assert out is img_f32 or np.array_equal(out, img_f32)
        assert out.dtype == np.float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
