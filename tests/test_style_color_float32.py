"""Dtype-awareness tests for retouch/style.py, retouch/style_transfer.py,
and retouch/color_science.py.

Verifies the float32 [0, 255] migration contract:
  * uint8 input  -> uint8 output
  * float32 input -> float32 output (no uint8 quantization in the path)
  * float32 output stays within [0.0, 255.0]
  * color-science round-trips are stable in both dtypes

Synthetic images only — no real photos. Python 3.9+ type hints.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional

import cv2
import numpy as np
import pytest

from retouch.color_science import (
    bgr_to_oklab,
    oklab_to_bgr,
    oklab_to_oklch,
    oklch_to_oklab,
)
from retouch.style import StyleApplier, StyleProfile
from retouch.style_transfer import (
    reinhard_transfer_masked,
    subject_aware_transfer,
    weighted_mean_std,
)


# ---------------------------------------------------------------------------
# Fixtures / synthetic images
# ---------------------------------------------------------------------------

H, W = 64, 64


def _gradient_bgr_u8() -> np.ndarray:
    """Synthetic uint8 BGR image with a smooth L/R gradient + warm tint."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    base = np.array([90, 140, 180], dtype=np.float32)  # warm beige BGR
    for x in range(W):
        lerp = 0.5 + 0.5 * (x / max(W - 1, 1))
        img[:, x, :] = np.clip(base * lerp, 0, 255).astype(np.uint8)
    rng = np.random.RandomState(11)
    noise = rng.randint(-4, 5, (H, W, 3), dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _to_f32(img_u8: np.ndarray) -> np.ndarray:
    return img_u8.astype(np.float32)


@pytest.fixture
def u8_img() -> np.ndarray:
    return _gradient_bgr_u8()


@pytest.fixture
def f32_img() -> np.ndarray:
    return _to_f32(_gradient_bgr_u8())


@pytest.fixture
def u8_ref() -> np.ndarray:
    """A second synthetic image with different color stats for transfer."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    base = np.array([60, 100, 220], dtype=np.float32)  # cooler/bluer
    for y in range(H):
        lerp = 0.6 + 0.4 * (y / max(H - 1, 1))
        img[y, :, :] = np.clip(base * lerp, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def f32_ref(u8_ref: np.ndarray) -> np.ndarray:
    return u8_ref.astype(np.float32)


@pytest.fixture
def full_mask() -> np.ndarray:
    """A uniform soft mask (everywhere active) for transfer ops."""
    return np.ones((H, W), dtype=np.float32)


@pytest.fixture
def half_mask() -> np.ndarray:
    """A left-half soft mask for transfer ops."""
    m = np.zeros((H, W), dtype=np.float32)
    m[:, : W // 2] = 1.0
    return m


# ===========================================================================
# color_science.py — bgr_to_oklab / oklab_to_bgr
# ===========================================================================


class TestOklabDtype:
    """Dtype-awareness of the Oklab converters."""

    def test_bgr_to_oklab_preserves_float32(self, f32_img: np.ndarray) -> None:
        out = bgr_to_oklab(f32_img)
        assert out.dtype == np.float32, "float32 input must yield float32 Oklab"
        assert out.shape == (H, W, 3)

    def test_bgr_to_oklab_uint8_returns_float32(self, u8_img: np.ndarray) -> None:
        # Oklab is intrinsically float; verify the uint8 path also returns float32
        out = bgr_to_oklab(u8_img)
        assert out.dtype == np.float32
        # L must be in [0, 1]
        L = out[..., 0]
        assert L.min() >= -1e-4 and L.max() <= 1.0 + 1e-4

    def test_oklab_to_bgr_uint8_default(self, u8_img: np.ndarray) -> None:
        ok = bgr_to_oklab(u8_img)
        out = oklab_to_bgr(ok)
        assert out.dtype == np.uint8, "default oklab_to_bgr must return uint8"
        assert out.shape == (H, W, 3)
        assert out.min() >= 0 and out.max() <= 255

    def test_oklab_to_bgr_float32_explicit(self, u8_img: np.ndarray) -> None:
        ok = bgr_to_oklab(u8_img)
        out = oklab_to_bgr(ok, float32_out=True)
        assert out.dtype == np.float32, "float32_out=True must return float32"
        assert out.shape == (H, W, 3)
        assert out.min() >= 0.0 and out.max() <= 255.0

    def test_oklab_roundtrip_uint8_close(self, u8_img: np.ndarray) -> None:
        # uint8 -> oklab -> uint8 should be near-identity (within 1 step)
        out = oklab_to_bgr(bgr_to_oklab(u8_img))
        diff = np.abs(out.astype(np.int16) - u8_img.astype(np.int16))
        assert diff.max() <= 2, f"uint8 round-trip diff too high: {diff.max()}"

    def test_oklab_roundtrip_float32_close(self, f32_img: np.ndarray) -> None:
        # float32 -> oklab -> float32 should be near-identity (no quantization)
        out = oklab_to_bgr(bgr_to_oklab(f32_img), float32_out=True)
        assert out.dtype == np.float32
        diff = np.abs(out - f32_img)
        assert diff.max() <= 2.0, f"float32 round-trip diff too high: {diff.max()}"

    def test_oklab_float_path_avoids_uint8_quantization(
        self, u8_img: np.ndarray, f32_img: np.ndarray
    ) -> None:
        """The float32 path must not quantize to 8 bits mid-pipeline.

        For a smooth gradient, the float32 round-trip residual should be at
        least as accurate as the uint8 round-trip residual.
        """
        u8_rt = oklab_to_bgr(bgr_to_oklab(u8_img)).astype(np.float32)
        f32_rt = oklab_to_bgr(bgr_to_oklab(f32_img), float32_out=True)
        u8_err = np.abs(u8_rt - f32_img).mean()
        f32_err = np.abs(f32_rt - f32_img).mean()
        assert f32_err <= u8_err + 0.5, (
            f"float32 path should not be worse than uint8: "
            f"f32_err={f32_err:.4f} u8_err={u8_err:.4f}"
        )

    def test_oklab_to_oklch_roundtrip_dtype(self, f32_img: np.ndarray) -> None:
        ok = bgr_to_oklab(f32_img)
        lch = oklab_to_oklch(ok)
        assert lch.dtype == np.float32
        back = oklch_to_oklab(lch)
        assert back.dtype == np.float32
        assert np.allclose(back, ok, atol=1e-5), "oklch round-trip must be stable"


# ===========================================================================
# style_transfer.py — reinhard_transfer_masked
# ===========================================================================


class TestReinhardDtype:
    """Dtype-awareness of reinhard_transfer_masked."""

    def test_uint8_in_uint8_out(
        self, u8_img: np.ndarray, u8_ref: np.ndarray, full_mask: np.ndarray
    ) -> None:
        out = reinhard_transfer_masked(u8_img, u8_ref, full_mask, full_mask)
        assert out.dtype == np.uint8, "uint8 inputs must yield uint8 output"
        assert out.shape == u8_img.shape
        assert out.min() >= 0 and out.max() <= 255

    def test_float32_in_float32_out(
        self,
        f32_img: np.ndarray,
        f32_ref: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        out = reinhard_transfer_masked(f32_img, f32_ref, full_mask, full_mask)
        assert out.dtype == np.float32, "float32 inputs must yield float32 output"
        assert out.shape == f32_img.shape
        assert out.min() >= 0.0 and out.max() <= 255.0

    def test_float32_no_uint8_quantization(
        self,
        f32_img: np.ndarray,
        f32_ref: np.ndarray,
        full_mask: np.ndarray,
    ) -> None:
        """float32 path output should differ from a uint8-quantized path.

        With a non-trivial mask the float32 result preserves sub-byte precision
        and should not equal the uint8-rounded result at every pixel.
        """
        out_f32 = reinhard_transfer_masked(f32_img, f32_ref, full_mask, full_mask)
        # Run the same op via the uint8 path on the rounded input
        out_u8 = reinhard_transfer_masked(
            f32_img.astype(np.uint8), f32_ref.astype(np.uint8), full_mask, full_mask
        ).astype(np.float32)
        # They should be close but not bit-identical (float path keeps precision)
        diff = np.abs(out_f32 - out_u8)
        assert diff.mean() < 5.0, "float and uint8 paths should be close"
        # And the float path has finer granularity (not all differences are 0)
        nonzero = np.count_nonzero(diff > 0.01)
        assert nonzero > 0, "float path must preserve sub-byte precision somewhere"

    def test_half_mask_preserves_unmasked_region(
        self,
        u8_img: np.ndarray,
        u8_ref: np.ndarray,
        half_mask: np.ndarray,
    ) -> None:
        """Outside the mask, the source is blended as-is; only LAB round-trip
        quantization (~1 LSB) is tolerated."""
        out = reinhard_transfer_masked(u8_img, u8_ref, half_mask, half_mask)
        # Right half (mask=0) should equal source up to LAB round-trip quantization
        diff = np.abs(
            out[:, W // 2:].astype(np.int16) - u8_img[:, W // 2:].astype(np.int16)
        )
        assert diff.max() <= 2, f"unmasked region changed by {diff.max()} (>2 LSB)"

    def test_half_mask_preserves_unmasked_region_f32(
        self,
        f32_img: np.ndarray,
        f32_ref: np.ndarray,
        half_mask: np.ndarray,
    ) -> None:
        out = reinhard_transfer_masked(f32_img, f32_ref, half_mask, half_mask)
        assert out.dtype == np.float32
        # float path has no uint8 quantization, so unmasked region is closer
        diff = np.abs(out[:, W // 2:] - f32_img[:, W // 2:])
        assert diff.max() <= 1.0, f"unmasked region changed by {diff.max()}"

    def test_zero_mask_returns_copy(
        self, u8_img: np.ndarray, u8_ref: np.ndarray
    ) -> None:
        zero = np.zeros((H, W), dtype=np.float32)
        out = reinhard_transfer_masked(u8_img, u8_ref, zero, zero)
        assert np.array_equal(out, u8_img)
        # And it is a copy, not the same buffer
        assert out is not u8_img


# ===========================================================================
# style_transfer.py — weighted_mean_std
# ===========================================================================


class TestWeightedMeanStd:
    """Sanity + dtype checks for weighted_mean_std."""

    def test_returns_float32(self, u8_img: np.ndarray, full_mask: np.ndarray) -> None:
        lab = cv2.cvtColor(u8_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        mean, std = weighted_mean_std(lab, full_mask)
        assert mean.dtype == np.float32
        assert std.dtype == np.float32
        assert mean.shape == (3,)
        assert std.shape == (3,)

    def test_zero_mask_returns_defaults(self, u8_img: np.ndarray) -> None:
        lab = cv2.cvtColor(u8_img, cv2.COLOR_BGR2LAB).astype(np.float32)
        zero = np.zeros((H, W), dtype=np.float32)
        mean, std = weighted_mean_std(lab, zero)
        assert np.allclose(mean, 0.0)
        assert np.allclose(std, 1.0)


# ===========================================================================
# style_transfer.py — subject_aware_transfer
# ===========================================================================


class _FakeRegions:
    """Minimal stand-in for FaceParser.Regions."""

    def __init__(self, skin: Optional[np.ndarray], hair: Optional[np.ndarray]) -> None:
        self.skin = skin
        self.hair = hair


class _FakeFace:
    """Minimal stand-in for a detected face."""

    def __init__(self) -> None:
        self.bbox = (0, 0, W, H)
        self.landmarks = np.zeros((68, 2), dtype=np.float32)
        self.ied = 1.0


class _FakeDetector:
    """Detector stub returning one face + full person mask."""

    def detect(self, img: np.ndarray) -> list:
        return [_FakeFace()]

    def segment_person(self, img: np.ndarray) -> np.ndarray:
        return np.ones((img.shape[0], img.shape[1]), dtype=np.float32)


class _FakeParser:
    """Parser stub returning full-frame skin/hair masks."""

    def __init__(self) -> None:
        self.skin_mask: Optional[np.ndarray] = None
        self.hair_mask: Optional[np.ndarray] = None

    def parse(
        self,
        landmarks: Any,
        img: np.ndarray,
        bbox: Any,
        person_f: np.ndarray,
        ied: float,
    ) -> _FakeRegions:
        h, w = img.shape[:2]
        skin = np.ones((h, w), dtype=np.float32) if self.skin_mask is None else self.skin_mask
        hair = None if self.hair_mask is None else self.hair_mask
        return _FakeRegions(skin=skin, hair=hair)


class TestSubjectAwareDtype:
    """Dtype-awareness of subject_aware_transfer (using stubbed detector/parser)."""

    def _engine(self) -> SimpleNamespace:
        return SimpleNamespace(_detector=_FakeDetector(), _parser=_FakeParser())

    def test_uint8_in_uint8_out(
        self, u8_img: np.ndarray, u8_ref: np.ndarray
    ) -> None:
        out = subject_aware_transfer(self._engine(), u8_img, u8_ref)
        assert out.dtype == np.uint8, "uint8 subject_aware_transfer must yield uint8"
        assert out.shape == u8_img.shape
        assert out.min() >= 0 and out.max() <= 255

    def test_float32_in_float32_out(
        self, f32_img: np.ndarray, f32_ref: np.ndarray
    ) -> None:
        out = subject_aware_transfer(self._engine(), f32_img, f32_ref)
        assert out.dtype == np.float32, "float32 subject_aware_transfer must yield float32"
        assert out.shape == f32_img.shape
        assert out.min() >= 0.0 and out.max() <= 255.0

    def test_float32_stays_in_range(
        self, f32_img: np.ndarray, f32_ref: np.ndarray
    ) -> None:
        out = subject_aware_transfer(self._engine(), f32_img, f32_ref)
        assert out.dtype == np.float32
        assert np.isfinite(out).all(), "output must be finite (no NaN/Inf)"
        assert out.min() >= 0.0 and out.max() <= 255.0


# ===========================================================================
# style.py — StyleApplier.apply (LAB skin-tone shift branch, dtype-aware)
# ===========================================================================


class _NoOpEngine:
    """Engine stub whose process() is identity — isolates the LAB-shift branch.

    StyleApplier.apply calls engine.process() then optionally a LAB skin-tone
    shift gated on skin_a/b_mean_delta. We return the input unchanged so the
    only observable mutation is the LAB shift, whose dtype-aware path is what
    we are verifying.
    """

    def __init__(self, detector: Any, parser: Any) -> None:
        self._detector = detector
        self._parser = parser

    def process(self, img: np.ndarray, **kwargs: Any) -> np.ndarray:
        return img.copy()


class _SkinParser(_FakeParser):
    """Parser stub that always returns a full skin mask."""

    def __init__(self) -> None:
        super().__init__()
        self.skin_mask = np.ones((H, W), dtype=np.float32)


class TestStyleApplierDtype:
    """Dtype-awareness of the LAB skin-tone shift in StyleApplier.apply."""

    def _applier(self) -> StyleApplier:
        det = _FakeDetector()
        par = _SkinParser()
        engine = _NoOpEngine(det, par)
        return StyleApplier(engine=engine, detector=det, parser=par)

    def _profile(self) -> StyleProfile:
        # a/b deltas > 0.5 trigger the LAB-shift branch (style.py:562)
        return StyleProfile(
            skin_a_mean_delta=4.0,
            skin_b_mean_delta=-4.0,
            skin_smooth_strength=0.0,
            skin_l_mean_delta=0.0,
            skin_mid_reduction=0.0,
            skin_texture_opacity=1.0,
            brightness_delta=0.0,
            contrast_delta=0.0,
            saturation_delta=0.0,
        )

    def test_uint8_in_uint8_out(self, u8_img: np.ndarray) -> None:
        out = self._applier().apply(u8_img, self._profile())
        assert out.dtype == np.uint8, "uint8 StyleApplier.apply must yield uint8"
        assert out.shape == u8_img.shape
        assert out.min() >= 0 and out.max() <= 255

    def test_float32_in_float32_out(self, f32_img: np.ndarray) -> None:
        out = self._applier().apply(f32_img, self._profile())
        assert out.dtype == np.float32, "float32 StyleApplier.apply must yield float32"
        assert out.shape == f32_img.shape
        assert out.min() >= 0.0 and out.max() <= 255.0
        assert np.isfinite(out).all()

    def test_lab_shift_actually_shifts_uint8(self, u8_img: np.ndarray) -> None:
        """The LAB a/b shift must measurably change the image (sanity)."""
        out = self._applier().apply(u8_img, self._profile())
        diff = np.abs(out.astype(np.int16) - u8_img.astype(np.int16))
        assert diff.mean() > 0.5, "LAB shift profile should alter the image"

    def test_lab_shift_actually_shifts_f32(self, f32_img: np.ndarray) -> None:
        out = self._applier().apply(f32_img, self._profile())
        assert out.dtype == np.float32
        diff = np.abs(out - f32_img)
        assert diff.mean() > 0.5, "LAB shift profile should alter the image"

    def test_float_path_more_precise_than_uint8(
        self, u8_img: np.ndarray, f32_img: np.ndarray
    ) -> None:
        """The float32 path output is not just a uint8 round-trip.

        Both paths start from the same source; the float path must keep more
        precision than the uint8 path (whose LAB conversion quantizes to 8
        bits). We assert the two outputs are close but not bit-identical.
        """
        applier = self._applier()
        profile = self._profile()
        out_u8 = applier.apply(u8_img, profile).astype(np.float32)
        out_f32 = applier.apply(f32_img, profile)
        diff = np.abs(out_f32 - out_u8)
        # Close (same op, same source) but not identical everywhere
        assert diff.mean() < 5.0
        assert np.count_nonzero(diff > 0.05) > 0, (
            "float path must differ from uint8-quantized path somewhere"
        )

    def test_zero_delta_no_shift_uint8(self, u8_img: np.ndarray) -> None:
        """When a/b deltas are zero the LAB-shift branch is skipped (identity)."""
        profile = StyleProfile()  # all zeros
        out = self._applier().apply(u8_img, profile)
        assert np.array_equal(out, u8_img)

    def test_zero_delta_no_shift_f32(self, f32_img: np.ndarray) -> None:
        profile = StyleProfile()
        out = self._applier().apply(f32_img, profile)
        assert out.dtype == np.float32
        assert np.allclose(out, f32_img)
