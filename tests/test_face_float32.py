"""Dtype-awareness tests for retouch/eyes.py, retouch/teeth.py, retouch/undereye.py.

Verifies the float32 [0, 255] migration contract for the three face-detail
modules:
  * uint8 input  -> uint8 output
  * float32 input -> float32 output (no uint8 quantization in the path)
  * float32 output stays within [0.0, 255.0]

Synthetic images only — no real photos. Python 3.9+ type hints.
"""

from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np
import pytest

from retouch.eyes import EyeEnhancer
from retouch.teeth import TeethWhitener
from retouch.undereye import UnderEyeRepairer


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


# ---------------------------------------------------------------------------
# Synthetic images
# ---------------------------------------------------------------------------


def _eye_img() -> np.ndarray:
    """Synthetic face image tuned for the eye/teeth/undereye codepaths.

    BGR warm skin tone with: bright eye-whites, bright iris with catchlight,
    bright teeth patch, and a darker under-eye patch.
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    base_skin = np.array([130, 170, 200], dtype=np.float32)  # warm beige BGR
    img[:, :, :] = base_skin.astype(np.uint8)

    # Eye whites: bright desaturated patch (L > 100 in LAB)
    img[20:35, 20:40] = [235, 235, 235]
    img[20:35, 56:76] = [235, 235, 235]

    # Iris: mid-tone brown with a bright catchlight
    img[24:31, 26:34] = [40, 60, 90]
    img[26:29, 28:32] = [10, 10, 10]  # pupil
    img[26:28, 29:31] = [240, 240, 240]  # catchlight
    img[24:31, 62:70] = [40, 60, 90]
    img[26:29, 64:68] = [10, 10, 10]
    img[26:28, 65:67] = [240, 240, 240]

    # Teeth: bright low-saturation patch
    img[60:80, 30:66] = [215, 220, 225]

    # Under-eye: darker patch below each eye
    img[36:50, 22:42] = [90, 110, 140]
    img[36:50, 54:74] = [90, 110, 140]

    rng = np.random.RandomState(11)
    noise = rng.randint(-4, 5, (H, W, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def u8_img() -> np.ndarray:
    return _eye_img()


@pytest.fixture
def f255_img(u8_img: np.ndarray) -> np.ndarray:
    return u8_img.astype(np.float32)


@pytest.fixture
def enhancer() -> EyeEnhancer:
    return EyeEnhancer()


@pytest.fixture
def whitener() -> TeethWhitener:
    return TeethWhitener()


@pytest.fixture
def repairer() -> UnderEyeRepairer:
    return UnderEyeRepairer()


class _EyeRegions:
    """Minimal stub matching FaceRegions attrs read by EyeEnhancer.enhance.

    Provides left_eye / right_eye / left_iris / right_iris as float32 masks
    positioned to overlap the synthetic eye features in _eye_img().
    """

    def __init__(self) -> None:
        z = np.zeros((H, W), dtype=np.float32)
        self.left_eye = z.copy()
        self.right_eye = z.copy()
        self.left_iris = z.copy()
        self.right_iris = z.copy()
        # Eye whites boxes overlap [20:35, 20:40] and [20:35, 56:76]
        self.left_eye[20:35, 20:40] = 1.0
        self.right_eye[20:35, 56:76] = 1.0
        # Iris boxes overlap [24:31, 26:34] and [24:31, 62:70]
        self.left_iris[24:31, 26:34] = 1.0
        self.right_iris[24:31, 62:70] = 1.0


class _UnderEyeRegions:
    """Minimal stub matching FaceRegions attrs read by UnderEyeRepairer.repair."""

    def __init__(self) -> None:
        z = np.zeros((H, W), dtype=np.float32)
        # Under-eye boxes overlap [36:50, 22:42] and [36:50, 54:74]
        self.left_under_eye = z.copy()
        self.right_under_eye = z.copy()
        self.left_under_eye[36:50, 22:42] = 1.0
        self.right_under_eye[36:50, 54:74] = 1.0


@pytest.fixture
def eye_regions() -> _EyeRegions:
    return _EyeRegions()


@pytest.fixture
def undereye_regions() -> _UnderEyeRegions:
    return _UnderEyeRegions()


@pytest.fixture
def mouth_mask() -> np.ndarray:
    """Float32 mouth-interior mask overlapping the synthetic teeth patch."""
    m = np.zeros((H, W), dtype=np.float32)
    m[60:80, 30:66] = 1.0
    return m


# ===========================================================================
# eyes.EyeEnhancer.enhance
# ===========================================================================


class TestEyesEnhanceFloat32:
    def test_preserves_dtype(
        self,
        enhancer: EyeEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        eye_regions: _EyeRegions,
    ) -> None:
        out_u8 = enhancer.enhance(u8_img, eye_regions, strength=40)
        out_f = enhancer.enhance(f255_img, eye_regions, strength=40)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self,
        enhancer: EyeEnhancer,
        f255_img: np.ndarray,
        eye_regions: _EyeRegions,
    ) -> None:
        out = enhancer.enhance(f255_img, eye_regions, strength=80)
        _assert_float_in_range(out)

    def test_catchlight_strength_preserves_dtype(
        self,
        enhancer: EyeEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        eye_regions: _EyeRegions,
    ) -> None:
        out_u8 = enhancer.enhance(u8_img, eye_regions, strength=40, catchlight_strength=50)
        out_f = enhancer.enhance(f255_img, eye_regions, strength=40, catchlight_strength=50)
        _assert_dtype_pair(out_u8, out_f)

    def test_zero_strength_preserves_dtype(
        self,
        enhancer: EyeEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        eye_regions: _EyeRegions,
    ) -> None:
        # strength <= 0 short-circuits and returns the input untouched.
        out_u8 = enhancer.enhance(u8_img, eye_regions, strength=0)
        out_f = enhancer.enhance(f255_img, eye_regions, strength=0)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32


# ===========================================================================
# teeth.TeethWhitener.whiten
# ===========================================================================


class TestTeethWhitenFloat32:
    def test_preserves_dtype(
        self,
        whitener: TeethWhitener,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        mouth_mask: np.ndarray,
    ) -> None:
        out_u8 = whitener.whiten(u8_img, mouth_mask, strength=40)
        out_f = whitener.whiten(f255_img, mouth_mask, strength=40)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self,
        whitener: TeethWhitener,
        f255_img: np.ndarray,
        mouth_mask: np.ndarray,
    ) -> None:
        out = whitener.whiten(f255_img, mouth_mask, strength=80)
        _assert_float_in_range(out)

    def test_none_mask_preserves_dtype(
        self,
        whitener: TeethWhitener,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
    ) -> None:
        # None mask short-circuits and returns the input untouched.
        out_u8 = whitener.whiten(u8_img, None, strength=50)
        out_f = whitener.whiten(f255_img, None, strength=50)
        _assert_dtype_pair(out_u8, out_f)

    def test_zero_strength_preserves_dtype(
        self,
        whitener: TeethWhitener,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        mouth_mask: np.ndarray,
    ) -> None:
        out_u8 = whitener.whiten(u8_img, mouth_mask, strength=0)
        out_f = whitener.whiten(f255_img, mouth_mask, strength=0)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32


# ===========================================================================
# undereye.UnderEyeRepairer.repair
# ===========================================================================


class TestUndereyeRepairFloat32:
    def test_preserves_dtype(
        self,
        repairer: UnderEyeRepairer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        undereye_regions: _UnderEyeRegions,
    ) -> None:
        out_u8 = repairer.repair(u8_img, undereye_regions, strength=40)
        out_f = repairer.repair(f255_img, undereye_regions, strength=40)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self,
        repairer: UnderEyeRepairer,
        f255_img: np.ndarray,
        undereye_regions: _UnderEyeRegions,
    ) -> None:
        out = repairer.repair(f255_img, undereye_regions, strength=80)
        _assert_float_in_range(out)

    def test_zero_strength_preserves_dtype(
        self,
        repairer: UnderEyeRepairer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        undereye_regions: _UnderEyeRegions,
    ) -> None:
        # strength <= 0 short-circuits and returns the input untouched.
        out_u8 = repairer.repair(u8_img, undereye_regions, strength=0)
        out_f = repairer.repair(f255_img, undereye_regions, strength=0)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_none_masks_preserves_dtype(
        self,
        repairer: UnderEyeRepairer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
    ) -> None:
        # Both masks None -> short-circuit returns input untouched.
        regions = _UnderEyeRegions()
        regions.left_under_eye = None
        regions.right_under_eye = None
        out_u8 = repairer.repair(u8_img, regions, strength=50)
        out_f = repairer.repair(f255_img, regions, strength=50)
        _assert_dtype_pair(out_u8, out_f)
