"""Dtype-awareness tests for retouch/skin.py — SkinProcessor.

Verifies the float32 [0, 255] migration contract:
  * uint8 input  -> uint8 output
  * float32 input -> float32 output (no uint8 quantization in the path)
  * float32 output stays within [0.0, 255.0]
  * uint8 path and float32 path agree to uint8 rounding where the op is
    deterministic (CLAHE-quantizing ops are exempt — they round L to 8 bits
    internally by design, documented in skin.py:283).

Synthetic images only — no real photos. Python 3.9+ type hints.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import cv2
import pytest

from retouch.skin import SkinProcessor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H, W = 96, 96  # square so face_width scaling is symmetric


def _skin_tone_bgr() -> np.ndarray:
    """A synthetic skin-tone BGR image with mild gradient + pore-scale noise.

    Built so LAB a/b land inside the skin locus (needed by unify_hue_line's
    eligibility gating) and so L has both mid and bright pixels (needed by
    shine_removal's adaptive threshold).
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    # Base skin tone in BGR (warm beige)
    base = np.array([130, 170, 200], dtype=np.float32)
    # Vertical luminance gradient: brighter at top, darker at bottom
    for y in range(H):
        lerp = 1.0 - (y / H) * 0.4
        img[y, :, :] = np.clip(base * lerp, 0, 255).astype(np.uint8)
    # Pore-scale noise so high-band energy is non-zero (texture_transplant donor)
    rng = np.random.RandomState(7)
    noise = rng.randint(-8, 9, (H, W, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # A bright desaturated patch (specular shine) for shine_removal
    img[10:20, 10:20] = [245, 245, 245]
    return img


@pytest.fixture
def u8_img() -> np.ndarray:
    return _skin_tone_bgr()


@pytest.fixture
def f255_img(u8_img: np.ndarray) -> np.ndarray:
    return u8_img.astype(np.float32)


@pytest.fixture
def skin_mask() -> np.ndarray:
    """Circular float32 skin mask covering most of the image."""
    mask = np.zeros((H, W), dtype=np.float32)
    cy, cx, r = H // 2, W // 2, min(H, W) // 2 - 4
    yy, xx = np.ogrid[:H, :W]
    mask[(xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2] = 1.0
    return cv2.GaussianBlur(mask, (15, 15), 0)


@pytest.fixture
def proc() -> SkinProcessor:
    return SkinProcessor()


class _FakeRegions:
    """Minimal stub matching FaceRegions attrs read by dodge_burn /
    wrinkle_soften / local_clarity / restore_micro_texture.

    Provides every attribute the SkinProcessor reads; pass overrides via
    kwargs to focus a mask on a specific zone.
    """

    def __init__(self, shape: tuple, **overrides: Any) -> None:
        z = np.zeros(shape, dtype=np.float32)
        full = np.ones(shape, dtype=np.float32)
        # All attrs default to a full skin mask so ops always have something
        # to act on; override individual attrs to None/zero to isolate paths.
        self.skin = full
        self.hair = None
        self.left_eyebrow = None
        self.right_eyebrow = None
        self.left_eye = None
        self.right_eye = None
        self.nose = full
        self.lips = full
        self.nose_bridge = full
        self.forehead_center = full
        self.cheek_highlights_l = full
        self.cheek_highlights_r = full
        self.forehead = full
        self.nasolabial_l = full
        self.nasolabial_r = full
        self.crows_feet_l = full
        self.crows_feet_r = full
        self.neck = z
        self.left_under_eye = full
        self.right_under_eye = full
        self.jawline_contour = z
        for k, v in overrides.items():
            setattr(self, k, v)


class _MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0) -> None:
        self.x = x
        self.y = y
        self.z = z


class _MockLandmarksList:
    """468-point landmark list with a low-yaw frontal arrangement."""

    def __init__(self) -> None:
        self.landmark = [_MockLandmark(0.5, 0.5, 0.0) for _ in range(468)]
        self.landmark[6].x = 0.5
        self.landmark[234].x = 0.3
        self.landmark[454].x = 0.7
        self.landmark[33].x = 0.4
        self.landmark[263].x = 0.6
        self.landmark[152].y = 0.7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_dtype_pair(out_u8: np.ndarray, out_f: np.ndarray) -> None:
    assert out_u8.dtype == np.uint8, "uint8 input must yield uint8 output"
    assert out_f.dtype == np.float32, "float32 input must yield float32 output"


def _assert_float_in_range(out_f: np.ndarray) -> None:
    assert out_f.min() >= 0.0, f"float32 output below 0: {out_f.min()}"
    assert out_f.max() <= 255.0, f"float32 output above 255: {out_f.max()}"


# ---------------------------------------------------------------------------
# whiten
# ---------------------------------------------------------------------------


class TestWhitenFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out_u8 = proc.whiten(u8_img, skin_mask, strength=40)
        out_f = proc.whiten(f255_img, skin_mask, strength=40)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out = proc.whiten(f255_img, skin_mask, strength=60)
        _assert_float_in_range(out)

    def test_hue_stable_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out_u8 = proc.whiten(u8_img, skin_mask, strength=40, hue_stable=True)
        out_f = proc.whiten(f255_img, skin_mask, strength=40, hue_stable=True)
        _assert_dtype_pair(out_u8, out_f)

    def test_negative_strength_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out_u8 = proc.whiten(u8_img, skin_mask, strength=-40)
        out_f = proc.whiten(f255_img, skin_mask, strength=-40)
        _assert_dtype_pair(out_u8, out_f)
        _assert_float_in_range(out_f)


# ---------------------------------------------------------------------------
# equalize
# ---------------------------------------------------------------------------


class TestEqualizeFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out_u8 = proc.equalize(u8_img, skin_mask, strength=50)
        out_f = proc.equalize(f255_img, skin_mask, strength=50)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out = proc.equalize(f255_img, skin_mask, strength=80)
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# dodge_burn
# ---------------------------------------------------------------------------


class TestDodgeBurnFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray
    ) -> None:
        regions = _FakeRegions((H, W))
        out_u8 = proc.dodge_burn(u8_img, regions, strength=50)
        out_f = proc.dodge_burn(f255_img, regions, strength=50)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray
    ) -> None:
        regions = _FakeRegions((H, W))
        out = proc.dodge_burn(f255_img, regions, strength=80)
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# wrinkle_soften
# ---------------------------------------------------------------------------


class TestWrinkleSoftenFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray
    ) -> None:
        regions = _FakeRegions((H, W))
        out_u8 = proc.wrinkle_soften(u8_img, regions, strength=50)
        out_f = proc.wrinkle_soften(f255_img, regions, strength=50)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray
    ) -> None:
        regions = _FakeRegions((H, W))
        out = proc.wrinkle_soften(f255_img, regions, strength=80)
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# harmonize_neck
# ---------------------------------------------------------------------------


class TestHarmonizeNeckFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        # Use a distinct neck mask so the op actually runs the blend path.
        neck = np.zeros((H, W), dtype=np.float32)
        neck[H // 2 + 10:, :] = 1.0
        face_skin = skin_mask.copy()
        person = np.ones((H, W), dtype=np.uint8) * 255
        lms = _MockLandmarksList()
        out_u8 = proc.harmonize_neck(u8_img, lms, person, face_skin, neck, strength=80)
        out_f = proc.harmonize_neck(f255_img, lms, person, face_skin, neck, strength=80)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        neck = np.zeros((H, W), dtype=np.float32)
        neck[H // 2 + 10:, :] = 1.0
        face_skin = skin_mask.copy()
        person = np.ones((H, W), dtype=np.uint8) * 255
        lms = _MockLandmarksList()
        out = proc.harmonize_neck(f255_img, lms, person, face_skin, neck, strength=80)
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# shine_removal
# ---------------------------------------------------------------------------


class TestShineRemovalFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        # u8_img has a bright desaturated patch at [10:20,10:20] -> shine detected
        out_u8 = proc.shine_removal(u8_img, skin_mask, strength=80)
        out_f = proc.shine_removal(f255_img, skin_mask, strength=80)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out = proc.shine_removal(f255_img, skin_mask, strength=100)
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# redness_even
# ---------------------------------------------------------------------------


class TestRednessEvenFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out_u8 = proc.redness_even(u8_img, skin_mask, strength=60, face_width=float(W))
        out_f = proc.redness_even(f255_img, skin_mask, strength=60, face_width=float(W))
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out = proc.redness_even(f255_img, skin_mask, strength=80, face_width=float(W))
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


class TestFlattenFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out_u8 = proc.flatten(u8_img, skin_mask, strength=60)
        out_f = proc.flatten(f255_img, skin_mask, strength=60)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out = proc.flatten(f255_img, skin_mask, strength=80)
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# micro_dodge_burn
# ---------------------------------------------------------------------------


class TestMicroDodgeBurnFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out_u8 = proc.micro_dodge_burn(u8_img, skin_mask, strength=60, face_width=float(W))
        out_f = proc.micro_dodge_burn(f255_img, skin_mask, strength=60, face_width=float(W))
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out = proc.micro_dodge_burn(f255_img, skin_mask, strength=80, face_width=float(W))
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# texture_transplant
# ---------------------------------------------------------------------------


class TestTextureTransplantFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out_u8 = proc.texture_transplant(u8_img, skin_mask, strength=60, face_width=float(W))
        out_f = proc.texture_transplant(f255_img, skin_mask, strength=60, face_width=float(W))
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out = proc.texture_transplant(f255_img, skin_mask, strength=80, face_width=float(W))
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# unify_hue_line
# ---------------------------------------------------------------------------


class TestUnifyHueLineFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out_u8 = proc.unify_hue_line(u8_img, skin_mask, hue_strength=50, chroma_strength=50)
        out_f = proc.unify_hue_line(f255_img, skin_mask, hue_strength=50, chroma_strength=50)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray, skin_mask: np.ndarray
    ) -> None:
        out = proc.unify_hue_line(f255_img, skin_mask, hue_strength=80, chroma_strength=80)
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# local_clarity
# ---------------------------------------------------------------------------


class TestLocalClarityFloat32:
    def test_preserves_dtype(
        self, proc: SkinProcessor, u8_img: np.ndarray, f255_img: np.ndarray
    ) -> None:
        regions = _FakeRegions((H, W))
        out_u8 = proc.local_clarity(u8_img, regions, strength=0.20, radius=8)
        out_f = proc.local_clarity(f255_img, regions, strength=0.20, radius=8)
        _assert_dtype_pair(out_u8, out_f)

    def test_float_in_range(
        self, proc: SkinProcessor, f255_img: np.ndarray
    ) -> None:
        regions = _FakeRegions((H, W))
        out = proc.local_clarity(f255_img, regions, strength=0.30, radius=8)
        _assert_float_in_range(out)


# ---------------------------------------------------------------------------
# Cross-cutting: uint8 / float32 byte-identical agreement for ops that do NOT
# route through the CLAHE uint8 histogram (which quantizes L by design).
# ---------------------------------------------------------------------------


class TestU8FloatAgreement:
    """The float32 [0,255] path must agree with the uint8 path to within
    cv2's LAB<->BGR conversion noise.

    cv2 uses fixed-point LUTs for uint8 LAB conversions and float arithmetic
    for float32 conversions (see skin.py:165 vs skin.py:164 / utils
    bgr_f32_to_lab_f32). These two codepaths legitimately differ by a few LSB
    at LAB boundaries — the migration contract is "byte-identical intent",
    not bit-exact, given cv2's two distinct conversion implementations.

    Contract enforced:
      * max divergence <= 6 LSB (covers observed cv2 rounding outliers)
      * 99th percentile <= 4 LSB (catches systematic divergence — a real bug
        would inflate the whole distribution, not just boundary pixels)
    """

    _MAX_LSB = 6.0
    _P99_LSB = 4.0

    @pytest.mark.parametrize(
        "func,kwargs,needs_regions",
        [
            ("whiten", {"strength": 40}, False),
            ("whiten", {"strength": 40, "hue_stable": True}, False),
            ("equalize", {"strength": 50}, False),
            ("dodge_burn", {"strength": 50}, True),
            ("wrinkle_soften", {"strength": 50}, True),
            ("redness_even", {"strength": 60, "face_width": float(W)}, False),
            ("flatten", {"strength": 60}, False),
            ("micro_dodge_burn", {"strength": 60, "face_width": float(W)}, False),
            ("shine_removal", {"strength": 80}, False),
            ("local_clarity", {"strength": 0.20, "radius": 8}, True),
        ],
    )
    def test_u8_float_agreement_within_cv2_noise(
        self,
        proc: SkinProcessor,
        u8_img: np.ndarray,
        f255_img: np.ndarray,
        skin_mask: np.ndarray,
        func: str,
        kwargs: dict,
        needs_regions: bool,
    ) -> None:
        method = getattr(proc, func)
        if needs_regions:
            regions = _FakeRegions((H, W))
            out_u8 = method(u8_img, regions, **kwargs)
            out_f = method(f255_img, regions, **kwargs)
        else:
            out_u8 = method(u8_img, skin_mask, **kwargs)
            out_f = method(f255_img, skin_mask, **kwargs)

        diff = np.abs(out_f.astype(np.float32) - out_u8.astype(np.float32))
        assert diff.max() <= self._MAX_LSB, (
            f"{func}: float path max divergence {diff.max():.3f} > {self._MAX_LSB} LSB"
        )
        p99 = float(np.percentile(diff, 99))
        assert p99 <= self._P99_LSB, (
            f"{func}: float path p99 divergence {p99:.3f} > {self._P99_LSB} LSB "
            f"(mean {diff.mean():.3f}) — systematic divergence suspected"
        )
