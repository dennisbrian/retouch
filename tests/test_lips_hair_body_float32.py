"""Dtype-awareness tests for retouch/lips.py, retouch/hair.py, retouch/body_relight.py.

Verifies the float32 [0, 255] migration contract for three modules:
  * uint8 input  -> uint8 output
  * float32 input -> float32 output (no uint8 quantization in the path)
  * float32 output stays within [0.0, 255.0]

Synthetic images only — no real photos. Python 3.9+ type hints.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest

from retouch.body_relight import BodyRelighter
from retouch.hair import HairEnhancer
from retouch.lips import LipEnhancer


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


def _face_img() -> np.ndarray:
    """Synthetic portrait-ish BGR image exercising all three modules.

    Layout:
      * base warm skin tone across the canvas
      * a brighter lip patch in the lower-middle (for LipEnhancer)
      * a darker hair band at the top (for HairEnhancer)
      * a body-skin patch with mild luminance gradient (for BodyRelighter)
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    base_skin = np.array([130, 170, 200], dtype=np.uint8)  # warm beige BGR
    img[:, :, :] = base_skin

    # Hair band: darker brown across the top
    img[0:25, :] = [40, 50, 70]

    # Lips: rosy patch in the lower-middle
    img[55:72, 30:66] = [60, 70, 180]

    # Body-skin gradient on the lower half (slight horizontal shading) for relight
    grad = np.linspace(160, 210, W, dtype=np.uint8)
    img[72:, :, :] = img[72:, :, :].astype(np.float32) * 0.5 + grad[None, :, None].astype(np.float32) * 0.5
    img = np.clip(img, 0, 255).astype(np.uint8)

    rng = np.random.RandomState(7)
    noise = rng.randint(-3, 4, (H, W, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def u8_img() -> np.ndarray:
    return _face_img()


@pytest.fixture
def f255_img(u8_img: np.ndarray) -> np.ndarray:
    return u8_img.astype(np.float32)


@pytest.fixture
def lip_mask() -> np.ndarray:
    """Float32 mask overlapping the synthetic lip patch [55:72, 30:66]."""
    m = np.zeros((H, W), dtype=np.float32)
    m[55:72, 30:66] = 1.0
    return m


@pytest.fixture
def face_bbox() -> Tuple[int, int, int, int]:
    """(x, y, w, h) face bounding box centred over the synthetic features."""
    return (20, 20, 56, 56)


@pytest.fixture
def person_mask() -> np.ndarray:
    """Float32 person mask covering the full canvas."""
    return np.ones((H, W), dtype=np.float32)


@pytest.fixture
def face_oval_mask() -> np.ndarray:
    """Float32 face oval covering the upper-middle; excludes hair band."""
    m = np.zeros((H, W), dtype=np.float32)
    m[20:72, 20:76] = 1.0
    return m


@pytest.fixture
def hair_mask() -> np.ndarray:
    """Float32 hair mask covering the top band [0:25, :]."""
    m = np.zeros((H, W), dtype=np.float32)
    m[0:25, :] = 1.0
    return m


@pytest.fixture
def body_skin_mask() -> np.ndarray:
    """Float32 body-skin mask covering the lower half [72:, :]."""
    m = np.zeros((H, W), dtype=np.float32)
    m[72:, :] = 1.0
    return m


@pytest.fixture
def lip_enhancer() -> LipEnhancer:
    return LipEnhancer()


@pytest.fixture
def hair_enhancer() -> HairEnhancer:
    return HairEnhancer()


@pytest.fixture
def relighter() -> BodyRelighter:
    return BodyRelighter()


# ===========================================================================
# lips.LipEnhancer.enhance
# ===========================================================================


class TestLipsEnhanceFloat32:
    def test_preserves_dtype(
        self,
        lip_enhancer: LipEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        lip_mask: np.ndarray,
    ) -> None:
        out_u8 = lip_enhancer.enhance(u8_img, lip_mask, strength=40)
        out_f = lip_enhancer.enhance(f255_img, lip_mask, strength=40)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self,
        lip_enhancer: LipEnhancer,
        f255_img: np.ndarray,
        lip_mask: np.ndarray,
    ) -> None:
        out = lip_enhancer.enhance(f255_img, lip_mask, strength=80)
        _assert_float_in_range(out)

    def test_tint_preserves_dtype(
        self,
        lip_enhancer: LipEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        lip_mask: np.ndarray,
    ) -> None:
        out_u8 = lip_enhancer.enhance(u8_img, lip_mask, strength=50, tint="rose")
        out_f = lip_enhancer.enhance(f255_img, lip_mask, strength=50, tint="rose")
        _assert_dtype_pair(out_u8, out_f)

    def test_finish_matte_preserves_dtype(
        self,
        lip_enhancer: LipEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        lip_mask: np.ndarray,
    ) -> None:
        out_u8 = lip_enhancer.enhance(u8_img, lip_mask, strength=50, finish="matte")
        out_f = lip_enhancer.enhance(f255_img, lip_mask, strength=50, finish="matte")
        _assert_dtype_pair(out_u8, out_f)

    def test_finish_velvet_preserves_dtype(
        self,
        lip_enhancer: LipEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        lip_mask: np.ndarray,
    ) -> None:
        out_u8 = lip_enhancer.enhance(u8_img, lip_mask, strength=50, finish="velvet")
        out_f = lip_enhancer.enhance(f255_img, lip_mask, strength=50, finish="velvet")
        _assert_dtype_pair(out_u8, out_f)

    def test_zero_strength_preserves_dtype(
        self,
        lip_enhancer: LipEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        lip_mask: np.ndarray,
    ) -> None:
        # strength <= 0 short-circuits and returns the input untouched.
        out_u8 = lip_enhancer.enhance(u8_img, lip_mask, strength=0)
        out_f = lip_enhancer.enhance(f255_img, lip_mask, strength=0)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_none_mask_preserves_dtype(
        self,
        lip_enhancer: LipEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
    ) -> None:
        # None mask short-circuits and returns the input untouched.
        out_u8 = lip_enhancer.enhance(u8_img, None, strength=50)
        out_f = lip_enhancer.enhance(f255_img, None, strength=50)
        _assert_dtype_pair(out_u8, out_f)


# ===========================================================================
# hair.HairEnhancer.enhance
# ===========================================================================


class TestHairEnhanceFloat32:
    def test_preserves_dtype_with_derived_mask(
        self,
        hair_enhancer: HairEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        person_mask: np.ndarray,
        face_oval_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> None:
        # No explicit hair_mask: module derives one from person - face_oval.
        out_u8 = hair_enhancer.enhance(u8_img, person_mask, face_oval_mask, face_bbox, strength=50)
        out_f = hair_enhancer.enhance(f255_img, person_mask, face_oval_mask, face_bbox, strength=50)
        _assert_dtype_pair(out_u8, out_f)

    def test_preserves_dtype_with_explicit_hair_mask(
        self,
        hair_enhancer: HairEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        hair_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> None:
        out_u8 = hair_enhancer.enhance(u8_img, None, None, face_bbox, strength=50, hair_mask=hair_mask)
        out_f = hair_enhancer.enhance(f255_img, None, None, face_bbox, strength=50, hair_mask=hair_mask)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self,
        hair_enhancer: HairEnhancer,
        f255_img: np.ndarray,
        hair_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> None:
        out = hair_enhancer.enhance(f255_img, None, None, face_bbox, strength=80, hair_mask=hair_mask)
        _assert_float_in_range(out)

    def test_zero_strength_preserves_dtype(
        self,
        hair_enhancer: HairEnhancer,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        hair_mask: np.ndarray,
        face_bbox: Tuple[int, int, int, int],
    ) -> None:
        out_u8 = hair_enhancer.enhance(u8_img, None, None, face_bbox, strength=0, hair_mask=hair_mask)
        out_f = hair_enhancer.enhance(f255_img, None, None, face_bbox, strength=0, hair_mask=hair_mask)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32


# ===========================================================================
# body_relight.BodyRelighter.relight
# ===========================================================================


class TestBodyRelightFloat32:
    def test_relight_preserves_dtype(
        self,
        relighter: BodyRelighter,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        body_skin_mask: np.ndarray,
    ) -> None:
        out_u8 = relighter.relight(u8_img, body_skin_mask, strength=50)
        out_f = relighter.relight(f255_img, body_skin_mask, strength=50)
        _assert_dtype_pair(out_u8, out_f)

    def test_relight_float_in_range(
        self,
        relighter: BodyRelighter,
        f255_img: np.ndarray,
        body_skin_mask: np.ndarray,
    ) -> None:
        out = relighter.relight(f255_img, body_skin_mask, strength=80)
        _assert_float_in_range(out)

    def test_relight_zero_strength_preserves_dtype(
        self,
        relighter: BodyRelighter,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        body_skin_mask: np.ndarray,
    ) -> None:
        out_u8 = relighter.relight(u8_img, body_skin_mask, strength=0)
        out_f = relighter.relight(f255_img, body_skin_mask, strength=0)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_dodge_burn_preserves_dtype(
        self,
        relighter: BodyRelighter,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        body_skin_mask: np.ndarray,
    ) -> None:
        out_u8 = relighter.dodge_burn(u8_img, body_skin_mask, strength=50)
        out_f = relighter.dodge_burn(f255_img, body_skin_mask, strength=50)
        _assert_dtype_pair(out_u8, out_f)

    def test_dodge_burn_float_in_range(
        self,
        relighter: BodyRelighter,
        f255_img: np.ndarray,
        body_skin_mask: np.ndarray,
    ) -> None:
        out = relighter.dodge_burn(f255_img, body_skin_mask, strength=80)
        _assert_float_in_range(out)

    def test_dodge_burn_zero_strength_preserves_dtype(
        self,
        relighter: BodyRelighter,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        body_skin_mask: np.ndarray,
    ) -> None:
        out_u8 = relighter.dodge_burn(u8_img, body_skin_mask, strength=0)
        out_f = relighter.dodge_burn(f255_img, body_skin_mask, strength=0)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32
