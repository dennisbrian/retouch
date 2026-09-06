"""Tests for the float-native pipeline.

Covers:
  * ``ColorGrader._F_*`` methods in ``retouch/grading.py`` — verify
    float32 [0,1] in -> float32 [0,1] out, no NaN/inf, output in valid
    range, and rough parity vs the uint8 path within a quantization delta.
  * ``utils.vibrance()`` float32 [0,255] path — float in -> float out,
    uint8 path byte-identical to a known snapshot, strength=0 is a no-op.
  * ``engine._F_adjust_vibrance`` / ``_F_apply_uniform_saturation`` —
    float32 [0,1] in/out and monotonic in strength.

All images are synthetic (random or gradients); no real photos used.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
import pytest

from retouch.grading import (
    ColorGrader,
    _bgr_f_to_hsv_u8_conv,
    _bgr_f_to_lab_u8_conv,
    _hsv_u8_conv_to_bgr_f,
    _lab_u8_conv_to_bgr_f,
)
from retouch.utils import vibrance


# ---------------------------------------------------------------------------
# Shared synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture
def float_img_01(rng: np.random.Generator) -> np.ndarray:
    """(H, W, 3) float32 BGR image in [0, 1] with varied colour content."""
    img = rng.random((64, 64, 3), dtype=np.float32)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@pytest.fixture
def gradient_float_01() -> np.ndarray:
    """Smooth gradient float32 image — exercises curves/clarity without noise."""
    h, w = 64, 64
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    base = (yy + xx) * 0.5
    img = np.stack([base, 1.0 - base, base * 0.5 + 0.25], axis=-1)
    return np.clip(img, 0.0, 1.0).astype(np.float32)


@pytest.fixture
def uint8_img(float_img_01: np.ndarray) -> np.ndarray:
    """uint8 BGR image matching float_img_01 (used for uint8-vs-float parity)."""
    return np.clip(float_img_01 * 255.0 + 0.5, 0, 255).astype(np.uint8)


@pytest.fixture
def uint8_vibrance_img(rng: np.random.Generator) -> np.ndarray:
    """uint8 BGR image with varied hue/saturation for vibrance tests."""
    img = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    return img


@pytest.fixture
def float_vibrance_img_255(uint8_vibrance_img: np.ndarray) -> np.ndarray:
    """float32 BGR [0,255] image derived from the uint8 fixture."""
    return uint8_vibrance_img.astype(np.float32)


@pytest.fixture
def grader() -> ColorGrader:
    return ColorGrader()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assert_float01(out: np.ndarray, ref_shape: Tuple[int, ...]) -> None:
    assert out.dtype == np.float32, f"expected float32, got {out.dtype}"
    assert out.shape == ref_shape
    assert np.isfinite(out).all(), "output contains NaN or inf"
    assert out.min() >= 0.0 - 1e-6, f"output below 0: {out.min()}"
    assert out.max() <= 1.0 + 1e-6, f"output above 1: {out.max()}"


def _uint8_path(grader: ColorGrader, name: str, img_u8: np.ndarray, *args, **kwargs) -> np.ndarray:
    """Run the matching uint8 method (the non-``_F_`` counterpart)."""
    method = getattr(grader, name)
    return method(img_u8, *args, **kwargs)


# ---------------------------------------------------------------------------
# grading._F_* float variants
# ---------------------------------------------------------------------------


class TestFApplyLuminanceCurve:
    def test_float_in_float_out_in_range(self, grader, float_img_01):
        curve = [(0, 0), (128, 200), (255, 255)]
        out = grader._F_apply_luminance_curve(float_img_01, curve)
        _assert_float01(out, float_img_01.shape)

    def test_identity_curve_is_noop(self, grader, float_img_01):
        curve = [(0, 0), (128, 128), (255, 255)]
        out = grader._F_apply_luminance_curve(float_img_01, curve)
        assert np.allclose(out, float_img_01, atol=1e-6)

    def test_parity_with_uint8(self, grader, float_img_01, uint8_img):
        curve = [(0, 0), (128, 200), (255, 255)]
        f_out = grader._F_apply_luminance_curve(float_img_01, curve)
        u_out = grader._apply_luminance_curve(uint8_img, curve)
        f_as_u8 = np.clip(f_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        max_diff = np.abs(f_as_u8.astype(np.int32) - u_out.astype(np.int32)).max()
        assert max_diff <= 3, f"luminance curve float-vs-uint8 delta too high: {max_diff}"


class TestFLiftShadows:
    def test_float_in_float_out_in_range(self, grader, float_img_01):
        out = grader._F_lift_shadows(float_img_01, 30.0)
        _assert_float01(out, float_img_01.shape)

    def test_lift_brightens_shadows(self, grader, float_img_01):
        out = grader._F_lift_shadows(float_img_01, 40.0)
        lab_in = _bgr_f_to_lab_u8_conv(float_img_01)
        lab_out = _bgr_f_to_lab_u8_conv(out)
        dark = lab_in[:, :, 0] < 80.0
        assert (lab_out[:, :, 0][dark] >= lab_in[:, :, 0][dark] - 1.0).all()

    def test_zero_lift_returns_near_identical(self, grader, float_img_01):
        out = grader._F_lift_shadows(float_img_01, 0.0)
        assert np.allclose(out, float_img_01, atol=2e-2)


class TestFAdjustWarmth:
    def test_float_in_float_out_in_range(self, grader, float_img_01):
        out = grader._F_adjust_warmth(float_img_01, 0.5)
        _assert_float01(out, float_img_01.shape)

    def test_warmth_shifts_b_channel(self, grader, float_img_01):
        out = grader._F_adjust_warmth(float_img_01, 0.5)
        lab_in = _bgr_f_to_lab_u8_conv(float_img_01)
        lab_out = _bgr_f_to_lab_u8_conv(out)
        assert lab_out[:, :, 2].mean() >= lab_in[:, :, 2].mean() - 1.0

    def test_parity_with_uint8(self, grader, float_img_01, uint8_img):
        f_out = grader._F_adjust_warmth(float_img_01, 0.4)
        u_out = grader._adjust_warmth(uint8_img, 0.4)
        f_as_u8 = np.clip(f_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        max_diff = np.abs(f_as_u8.astype(np.int32) - u_out.astype(np.int32)).max()
        assert max_diff <= 12, f"warmth float-vs-uint8 delta too high: {max_diff}"


class TestFAdjustSaturation:
    def test_float_in_float_out_in_range(self, grader, float_img_01):
        out = grader._F_adjust_saturation(float_img_01, 0.5)
        _assert_float01(out, float_img_01.shape)

    def test_boost_increases_saturation(self, grader, float_img_01):
        out = grader._F_adjust_saturation(float_img_01, 0.5)
        hsv_in = _bgr_f_to_hsv_u8_conv(float_img_01)
        hsv_out = _bgr_f_to_hsv_u8_conv(out)
        assert hsv_out[:, :, 1].mean() >= hsv_in[:, :, 1].mean() - 1.0

    def test_parity_with_uint8(self, grader, float_img_01, uint8_img):
        f_out = grader._F_adjust_saturation(float_img_01, 0.3)
        u_out = grader._adjust_saturation(uint8_img, 0.3)
        f_as_u8 = np.clip(f_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        max_diff = np.abs(f_as_u8.astype(np.int32) - u_out.astype(np.int32)).max()
        assert max_diff <= 5, f"saturation float-vs-uint8 delta too high: {max_diff}"


class TestFSplitTone:
    TONES = {"shadows": (20, 220), "highlights": (220, 20)}

    def test_float_in_float_out_in_range(self, grader, float_img_01):
        out = grader._F_split_tone(float_img_01, self.TONES)
        _assert_float01(out, float_img_01.shape)

    def test_mask_restricts_change(self, grader, float_img_01):
        mask = np.zeros((float_img_01.shape[0], float_img_01.shape[1]), dtype=np.float32)
        mask[:32, :] = 1.0
        masked = grader._F_split_tone(float_img_01, self.TONES, mask=mask)
        unmasked_orig = float_img_01[32:, :, :]
        unmasked_out = masked[32:, :, :]
        assert np.allclose(unmasked_orig, unmasked_out, atol=1e-5)
        masked_orig = float_img_01[:32, :, :]
        masked_out = masked[:32, :, :]
        assert not np.allclose(masked_orig, masked_out, atol=1e-3), \
            "masked region should be modified by split_tone"

    def test_parity_with_uint8(self, grader, float_img_01, uint8_img):
        f_out = grader._F_split_tone(float_img_01, self.TONES)
        u_out = grader._split_tone(uint8_img, self.TONES)
        f_as_u8 = np.clip(f_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        max_diff = np.abs(f_as_u8.astype(np.int32) - u_out.astype(np.int32)).max()
        assert max_diff <= 35, f"split_tone float-vs-uint8 delta too high: {max_diff}"


class TestFAddClarity:
    def test_float_in_float_out_in_range(self, grader, gradient_float_01):
        out = grader._F_add_clarity(gradient_float_01, 0.3)
        _assert_float01(out, gradient_float_01.shape)

    def test_zero_strength_is_noop(self, grader, gradient_float_01):
        out = grader._F_add_clarity(gradient_float_01, 0.0)
        assert np.allclose(out, gradient_float_01, atol=1e-6)

    def test_parity_with_uint8(self, grader, gradient_float_01):
        grad_u8 = np.clip(gradient_float_01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
        f_out = grader._F_add_clarity(gradient_float_01, 0.4)
        u_out = grader._add_clarity(grad_u8, 0.4)
        f_as_u8 = np.clip(f_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        max_diff = np.abs(f_as_u8.astype(np.int32) - u_out.astype(np.int32)).max()
        assert max_diff <= 15, f"clarity float-vs-uint8 delta too high: {max_diff}"

    @staticmethod
    def _hf_energy(img_bgr_u8: np.ndarray) -> float:
        """Laplacian-variance proxy for high-frequency energy on the L channel."""
        lab = cv2.cvtColor(img_bgr_u8, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        return float(cv2.Laplacian(l_chan, cv2.CV_32F, ksize=3).var())

    def test_low_texture_region_not_disproportionately_amplified(self, grader, gradient_float_01):
        """A flat/smooth region must not gain HF energy out of proportion to
        a region with genuine edge detail, at the same clarity strength.

        Regression for the 2026-09-07 clarity-quantization bug: the old
        uint8 BGR->LAB->BGR roundtrip inside _add_clarity/_F_add_clarity
        injected high-frequency energy on flat regions (measured 31.7->65.6
        Laplacian-variance on a real photo's flat sky patch, effectively at
        clarity strength 0) that had nothing to do with the recipe's
        intended amplification. gradient_float_01 is a smooth, noise-free
        gradient — the flat-region proxy. A checkerboard-edge image is the
        genuine-detail proxy; clarity is expected to amplify its real edges
        far more than it amplifies the gradient's near-zero residual.
        """
        strength = 0.04  # matches cosplay_kitsune_daylight_v1's recipe value / 100

        flat_u8 = np.clip(gradient_float_01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
        flat_src_hf = self._hf_energy(flat_u8)
        flat_out = grader._F_add_clarity(gradient_float_01, strength)
        flat_out_u8 = np.clip(flat_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        flat_out_hf = self._hf_energy(flat_out_u8)

        # Edge fixture: a coarse checkerboard in the *mid-range* (0.3-0.7),
        # giving real step-edges with headroom on both sides for clarity to
        # amplify into. A 0/1 (pure black/white) checkerboard saturates at
        # the op's own np.clip(..., 0, 255) boundary — amplifying a residual
        # that's already fully saturated is a no-op by construction, which
        # would make this fixture indistinguishable from "clarity does
        # nothing" rather than "clarity has nothing to amplify here".
        h, w = gradient_float_01.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        checker = (((yy // 8) + (xx // 8)) % 2).astype(np.float32)
        checker = 0.3 + checker * 0.4  # values in {0.3, 0.7}
        edge_img = np.stack([checker, checker, checker], axis=-1)
        edge_u8 = np.clip(edge_img * 255.0 + 0.5, 0, 255).astype(np.uint8)
        edge_src_hf = self._hf_energy(edge_u8)
        edge_out = grader._F_add_clarity(edge_img, strength)
        edge_out_u8 = np.clip(edge_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        edge_out_hf = self._hf_energy(edge_out_u8)

        # The flat region's HF energy must stay close to its own source
        # (no artifact injection from nothing) ...
        assert flat_out_hf <= flat_src_hf * 1.5 + 1.0, (
            f"flat region HF energy grew disproportionately: "
            f"source={flat_src_hf:.3f} -> out={flat_out_hf:.3f}"
        )
        # ... while genuine edge detail is still free to be amplified by
        # clarity (the whole point of the op). Compare ABSOLUTE HF-energy
        # deltas, not ratios: the two fixtures start from wildly different
        # baselines (a near-zero-HF gradient vs. a real-edge checkerboard),
        # so a ratio makes a tiny absolute change on the gradient look like
        # a large relative jump and can invert the comparison. The
        # quantization bug this test guards against injected HF energy from
        # nothing on a flat region — an absolute-delta comparison is the
        # metric that actually distinguishes "amplifying real detail" from
        # "injecting energy where there was none".
        flat_delta = flat_out_hf - flat_src_hf
        edge_delta = edge_out_hf - edge_src_hf
        assert edge_delta > flat_delta, (
            f"expected genuine edge detail to gain more HF energy than a "
            f"flat region: flat_delta={flat_delta:.3f} edge_delta={edge_delta:.3f}"
        )


class TestFApplyRgbCurves:
    CURVES = {"R": [(0, 0), (128, 200), (255, 255)], "G": [(0, 0), (255, 255)]}

    def test_float_in_float_out_in_range(self, grader, float_img_01):
        out = grader._F_apply_rgb_curves(float_img_01, self.CURVES)
        _assert_float01(out, float_img_01.shape)

    def test_identity_curves_noop(self, grader, float_img_01):
        identity = {"R": [(0, 0), (255, 255)], "G": [(0, 0), (255, 255)], "B": [(0, 0), (255, 255)]}
        out = grader._F_apply_rgb_curves(float_img_01, identity)
        assert np.allclose(out, float_img_01, atol=1e-6)

    def test_parity_with_uint8(self, grader, float_img_01, uint8_img):
        f_out = grader._F_apply_rgb_curves(float_img_01, self.CURVES)
        u_out = grader._apply_rgb_curves(uint8_img, self.CURVES)
        f_as_u8 = np.clip(f_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        max_diff = np.abs(f_as_u8.astype(np.int32) - u_out.astype(np.int32)).max()
        assert max_diff <= 5, f"rgb_curves float-vs-uint8 delta too high: {max_diff}"


class TestFHslHueShift:
    SHIFTS = {"yellow": 15.0, "blue": -10.0}

    def test_float_in_float_out_in_range(self, grader, float_img_01):
        out = grader._F_hsl_hue_shift(float_img_01, self.SHIFTS)
        _assert_float01(out, float_img_01.shape)

    def test_zero_shifts_noop(self, grader, float_img_01):
        out = grader._F_hsl_hue_shift(float_img_01, {"red": 0.0, "green": 0.0})
        assert np.allclose(out, float_img_01, atol=1e-6)

    def test_parity_with_uint8(self, grader, float_img_01, uint8_img):
        f_out = grader._F_hsl_hue_shift(float_img_01, self.SHIFTS)
        u_out = grader._hsl_hue_shift(uint8_img, self.SHIFTS)
        f_as_u8 = np.clip(f_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        max_diff = np.abs(f_as_u8.astype(np.int32) - u_out.astype(np.int32)).max()
        assert max_diff <= 50, f"hsl_hue_shift float-vs-uint8 delta too high: {max_diff}"


class TestFAdjustWhiteBalance:
    WB = {"R": 1.2, "G": 1.0, "B": 0.8}

    def test_float_in_float_out_in_range(self, grader, float_img_01):
        out = grader._F_adjust_white_balance(float_img_01, self.WB)
        _assert_float01(out, float_img_01.shape)

    def test_neutral_multipliers_noop(self, grader, float_img_01):
        out = grader._F_adjust_white_balance(float_img_01, {"R": 1.0, "G": 1.0, "B": 1.0})
        assert np.allclose(out, float_img_01, atol=1e-6)

    def test_parity_with_uint8(self, grader, float_img_01, uint8_img):
        f_out = grader._F_adjust_white_balance(float_img_01, self.WB)
        u_out = grader._adjust_white_balance(uint8_img, self.WB)
        f_as_u8 = np.clip(f_out * 255.0 + 0.5, 0, 255).astype(np.uint8)
        max_diff = np.abs(f_as_u8.astype(np.int32) - u_out.astype(np.int32)).max()
        assert max_diff <= 10, f"white_balance float-vs-uint8 delta too high: {max_diff}"


class TestFApplyCalibration:
    CALIB = {"red": {"hue": 5, "sat": 10}, "green": {"hue": -3, "sat": 0}, "blue": {"hue": 0, "sat": -5}}

    def test_float_in_float_out_in_range(self, grader, float_img_01):
        out = grader._F_apply_calibration(float_img_01, self.CALIB)
        _assert_float01(out, float_img_01.shape)

    def test_empty_calibration_noop(self, grader, float_img_01):
        out = grader._F_apply_calibration(float_img_01, {})
        assert np.allclose(out, float_img_01, atol=1e-6)

    def test_zero_shifts_noop(self, grader, float_img_01):
        zero = {"red": {"hue": 0, "sat": 0}, "green": {"hue": 0, "sat": 0}, "blue": {"hue": 0, "sat": 0}}
        out = grader._F_apply_calibration(float_img_01, zero)
        assert np.allclose(out, float_img_01, atol=1e-6)


class TestFSplitToneThreeWay:
    TONES = {
        "shadows": {"hue": 30.0, "sat": 60.0},
        "midtones": {"hue": 0.0, "sat": 0.0},
        "highlights": {"hue": 210.0, "sat": 50.0},
        "balance": 0.0,
    }

    def test_float_in_float_out_in_range(self, grader, float_img_01):
        out = grader._F_split_tone_three_way(float_img_01, self.TONES)
        _assert_float01(out, float_img_01.shape)

    def test_empty_tones_noop(self, grader, float_img_01):
        out = grader._F_split_tone_three_way(float_img_01, {})
        assert np.allclose(out, float_img_01, atol=1e-6)

    def test_mask_restricts_change(self, grader, float_img_01):
        mask = np.zeros((float_img_01.shape[0], float_img_01.shape[1]), dtype=np.float32)
        mask[:, :32] = 1.0
        out = grader._F_split_tone_three_way(float_img_01, self.TONES, mask=mask)
        assert np.allclose(float_img_01[:, 32:, :], out[:, 32:, :], atol=1e-5)
        assert not np.allclose(float_img_01[:, :32, :], out[:, :32, :])


class TestFApplyHslAdjustments:
    ADJ = {
        "hue": {"yellow": 10.0, "blue": -8.0},
        "saturation": {"red": 15, "green": -10},
        "luminance": {"orange": 5},
    }

    def test_float_in_float_out_in_range(self, grader, float_img_01):
        out = grader._F_apply_hsl_adjustments(float_img_01, self.ADJ)
        _assert_float01(out, float_img_01.shape)

    def test_zero_adjustments_noop(self, grader, float_img_01):
        zero = {"hue": {}, "saturation": {}, "luminance": {}}
        out = grader._F_apply_hsl_adjustments(float_img_01, zero)
        assert np.allclose(out, float_img_01, atol=1e-6)


class TestFAddHaze:
    def test_float_in_float_out_in_range(self, grader, gradient_float_01):
        out = grader._F_add_haze(gradient_float_01, 0.3)
        _assert_float01(out, gradient_float_01.shape)

    def test_zero_strength_noop(self, grader, gradient_float_01):
        out = grader._F_add_haze(gradient_float_01, 0.0)
        assert np.allclose(out, gradient_float_01, atol=1e-6)

    def test_mask_restricts_change(self, grader, gradient_float_01):
        h, w = gradient_float_01.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        mask[:h // 2, :] = 1.0
        out = grader._F_add_haze(gradient_float_01, 0.4, mask=mask)
        assert np.allclose(gradient_float_01[h // 2:, :], out[h // 2:, :], atol=1e-2)


# ---------------------------------------------------------------------------
# Lab / HSV float-conversion helpers (round-trip sanity)
# ---------------------------------------------------------------------------


class TestColorspaceHelpers:
    def test_lab_roundtrip_preserves_values(self, float_img_01):
        lab = _bgr_f_to_lab_u8_conv(float_img_01)
        assert lab.dtype == np.float32
        assert np.isfinite(lab).all()
        bgr = _lab_u8_conv_to_bgr_f(lab)
        assert bgr.dtype == np.float32
        assert np.allclose(bgr, float_img_01, atol=2e-2), \
            "LAB round-trip drifted too far"

    def test_hsv_roundtrip_preserves_values(self, float_img_01):
        hsv = _bgr_f_to_hsv_u8_conv(float_img_01)
        assert hsv.dtype == np.float32
        assert np.isfinite(hsv).all()
        bgr = _hsv_u8_conv_to_bgr_f(hsv)
        assert bgr.dtype == np.float32
        assert np.allclose(bgr, float_img_01, atol=2e-2), \
            "HSV round-trip drifted too far"

    def test_lab_neutral_pixel(self):
        gray = np.full((1, 1, 3), 0.5, dtype=np.float32)
        lab = _bgr_f_to_lab_u8_conv(gray)
        assert abs(lab[0, 0, 1] - 128.0) < 1.0
        assert abs(lab[0, 0, 2] - 128.0) < 1.0


# ---------------------------------------------------------------------------
# utils.vibrance() float path
# ---------------------------------------------------------------------------


class TestVibranceFloat:
    def test_float_in_float_out(self, float_vibrance_img_255):
        out = vibrance(float_vibrance_img_255, None, 0.5)
        assert out.dtype == np.float32
        assert out.shape == float_vibrance_img_255.shape

    def test_float_output_in_range(self, float_vibrance_img_255):
        out = vibrance(float_vibrance_img_255, None, 0.8)
        assert np.isfinite(out).all()
        assert out.min() >= 0.0 - 1e-4
        assert out.max() <= 255.0 + 1e-4

    def test_strength_zero_is_noop(self, float_vibrance_img_255):
        out = vibrance(float_vibrance_img_255, None, 0.0)
        assert out is float_vibrance_img_255 or np.array_equal(out, float_vibrance_img_255)

    def test_uint8_path_byte_identical_snapshot(self, uint8_vibrance_img):
        out = vibrance(uint8_vibrance_img, None, 0.5)
        assert out.dtype == np.uint8

        expected = _vibrance_uint8_reference(uint8_vibrance_img, 0.5)
        assert np.array_equal(out, expected), "uint8 vibrance path drifted from reference"

    def test_uint8_no_quantization_for_zero_strength(self, uint8_vibrance_img):
        out = vibrance(uint8_vibrance_img, None, 0.0)
        assert np.array_equal(out, uint8_vibrance_img)

    def test_float_path_no_uint8_quantization(self, float_vibrance_img_255):
        out = vibrance(float_vibrance_img_255, None, 0.3)
        u8_roundtrip = np.clip(out, 0, 255).astype(np.uint8).astype(np.float32)
        diff = np.abs(out - u8_roundtrip)
        n_quantized = int((diff > 0.01).sum())
        assert n_quantized < out.size, \
            f"float output matches uint8 grid too closely — quantization suspected ({n_quantized} pixels)"


def _vibrance_uint8_reference(img_u8: np.ndarray, strength: float) -> np.ndarray:
    """Reference implementation of the uint8 vibrance path (matches utils.vibrance)."""
    from retouch.utils import blend_masked
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s = hsv[:, :, 0], hsv[:, :, 1]
    factor = 1.0 + strength * (1.0 - s / 255.0)
    skin_hue = ((h > 0) & (h < 25)) | (h > 160)
    skin_factor = np.clip(1.0 - strength * 0.5, 0.5, 1.0)
    factor = np.where(skin_hue, np.minimum(factor, skin_factor), factor)
    hsv[:, :, 1] = np.clip(s * factor, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return blend_masked(img_u8, result, None)


# ---------------------------------------------------------------------------
# engine._F_adjust_vibrance / _F_apply_uniform_saturation
# ---------------------------------------------------------------------------


class TestEngineFAdjustVibrance:
    @pytest.fixture
    def f_func(self):
        from retouch.engine import _F_adjust_vibrance
        return _F_adjust_vibrance

    def test_float_in_float_out_in_range(self, f_func, float_img_01):
        out = f_func(float_img_01, 40.0)
        _assert_float01(out, float_img_01.shape)

    def test_zero_vibrance_is_noop(self, f_func, float_img_01):
        out = f_func(float_img_01, 0.0)
        assert np.allclose(out, float_img_01, atol=1e-6)

    def test_monotonic_in_strength(self, f_func, float_img_01):
        sats = []
        for v in (0.0, 25.0, 50.0, 75.0, 100.0):
            out = f_func(float_img_01, v)
            hsv = _bgr_f_to_hsv_u8_conv(out)
            sats.append(float(hsv[:, :, 1].mean()))
        for i in range(1, len(sats)):
            assert sats[i] >= sats[i - 1] - 1.0, \
                f"vibrance not monotonic at step {i}: {sats[i - 1]} -> {sats[i]}"


class TestEngineFApplyUniformSaturation:
    @pytest.fixture
    def f_func(self):
        from retouch.engine import _F_apply_uniform_saturation
        return _F_apply_uniform_saturation

    def test_float_in_float_out_in_range(self, f_func, float_img_01):
        out = f_func(float_img_01, 30.0)
        _assert_float01(out, float_img_01.shape)

    def test_zero_saturation_is_noop(self, f_func, float_img_01):
        out = f_func(float_img_01, 0.0)
        assert np.array_equal(out, float_img_01)

    def test_monotonic_in_strength(self, f_func, float_img_01):
        sats = []
        for s in (-50.0, 0.0, 25.0, 50.0, 100.0):
            out = f_func(float_img_01, s)
            hsv = _bgr_f_to_hsv_u8_conv(out)
            sats.append(float(hsv[:, :, 1].mean()))
        for i in range(1, len(sats)):
            assert sats[i] >= sats[i - 1] - 1.0, \
                f"uniform saturation not monotonic at step {i}: {sats[i - 1]} -> {sats[i]}"


# ---------------------------------------------------------------------------
# color_space.py float-native helpers (F1/E2 Wave 4)
# ---------------------------------------------------------------------------


class TestColorSpaceFloatHelpers:
    """Verify the float-native LAB/LCH helpers in color_space.py.

    These helpers accept float32 BGR [0, 255] and return float32 LAB/LCH in
    the same convention as the uint8 counterparts (L in [0, 100],
    a/b in [-128, 127] for LAB; L in [0, 100], C in [0, ~180],
    H in [0, 360) for LCH).
    """

    @pytest.fixture
    def u8_img(self, rng):
        return rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)

    @pytest.fixture
    def f255_img(self, u8_img):
        return u8_img.astype(np.float32)

    def test_bgr_f32_to_lab_f32_dtype_and_shape(self, f255_img):
        from retouch.color_space import bgr_f32_to_lab_f32
        lab = bgr_f32_to_lab_f32(f255_img)
        assert lab.dtype == np.float32
        assert lab.shape == f255_img.shape

    def test_bgr_f32_to_lab_f32_range(self, f255_img):
        from retouch.color_space import bgr_f32_to_lab_f32
        lab = bgr_f32_to_lab_f32(f255_img)
        assert lab[:, :, 0].min() >= 0.0 and lab[:, :, 0].max() <= 100.0
        assert lab[:, :, 1].min() >= -128.0 and lab[:, :, 1].max() <= 127.0
        assert lab[:, :, 2].min() >= -128.0 and lab[:, :, 2].max() <= 127.0

    def test_lab_f32_to_bgr_f32_dtype_and_range(self, f255_img):
        from retouch.color_space import bgr_f32_to_lab_f32, lab_f32_to_bgr_f32
        lab = bgr_f32_to_lab_f32(f255_img)
        bgr = lab_f32_to_bgr_f32(lab)
        assert bgr.dtype == np.float32
        assert bgr.min() >= 0.0 and bgr.max() <= 255.0

    def test_lab_round_trip_float_better_than_uint8(self, u8_img, f255_img):
        """Float LAB round-trip should be at least as accurate as uint8."""
        from retouch.color_space import bgr_to_lab, lab_to_bgr, bgr_f32_to_lab_f32, lab_f32_to_bgr_f32
        # uint8 path
        lab_u8 = bgr_to_lab(u8_img)
        bgr_u8_back = lab_to_bgr(lab_u8)
        u8_err = np.abs(bgr_u8_back.astype(np.float32) - f255_img).max()
        # float path
        lab_f = bgr_f32_to_lab_f32(f255_img)
        bgr_f_back = lab_f32_to_bgr_f32(lab_f)
        f_err = np.abs(bgr_f_back - f255_img).max()
        assert f_err <= u8_err + 2.0, \
            f"float round-trip err {f_err} should be <= uint8 err {u8_err} + 2"

    def test_bgr_f32_to_lch_f32_dtype_and_range(self, f255_img):
        from retouch.color_space import bgr_f32_to_lch_f32
        lch = bgr_f32_to_lch_f32(f255_img)
        assert lch.dtype == np.float32
        assert lch.shape == f255_img.shape
        assert lch[:, :, 0].min() >= 0.0 and lch[:, :, 0].max() <= 100.0
        assert lch[:, :, 1].min() >= 0.0
        assert lch[:, :, 2].min() >= 0.0 and lch[:, :, 2].max() < 360.0

    def test_lch_f32_to_bgr_f32_round_trip(self, f255_img):
        from retouch.color_space import bgr_f32_to_lch_f32, lch_f32_to_bgr_f32
        lch = bgr_f32_to_lch_f32(f255_img)
        bgr_back = lch_f32_to_bgr_f32(lch)
        assert bgr_back.dtype == np.float32
        assert bgr_back.min() >= 0.0 and bgr_back.max() <= 255.0
        err = np.abs(bgr_back - f255_img).max()
        assert err < 3.0, f"LCH round-trip error {err} too high"

    def test_float_lch_matches_uint8_lch_within_quantization(self, u8_img, f255_img):
        """Float LCH values should match uint8 LCH within uint8 quantization noise.

        Note: H channel is unstable at low chroma (atan2 of near-zero a/b),
        so we only compare H where chroma is meaningful (C > 10).
        """
        from retouch.color_space import bgr_to_lch, bgr_f32_to_lch_f32
        lch_u8 = bgr_to_lch(u8_img)
        lch_f = bgr_f32_to_lch_f32(f255_img)
        # L channel: uint8 path has L in [0, 100] after rescale, float path same
        assert np.abs(lch_u8[:, :, 0] - lch_f[:, :, 0]).max() < 3.0
        # C channel
        assert np.abs(lch_u8[:, :, 1] - lch_f[:, :, 1]).max() < 3.0
        # H channel: only compare where chroma is meaningful (C > 10)
        # At low chroma, tiny a/b perturbations cause large hue swings (atan2 instability)
        high_chroma = lch_f[:, :, 1] > 10.0
        if high_chroma.any():
            h_diff = np.minimum(
                np.abs(lch_u8[high_chroma, 2] - lch_f[high_chroma, 2]),
                360.0 - np.abs(lch_u8[high_chroma, 2] - lch_f[high_chroma, 2]),
            )
            assert h_diff.max() < 5.0, f"H diff at high chroma: {h_diff.max()}"


# ---------------------------------------------------------------------------
# dtype-aware grading methods (F1/E2 Wave 4)
# ---------------------------------------------------------------------------


class TestDtypeAwareGradingMethods:
    """Verify that grading methods now accept both uint8 and float32 [0, 255]
    BGR input, returning the same dtype, with the uint8 path byte-identical
    to the pre-change behavior.
    """

    @pytest.fixture
    def grader(self):
        return ColorGrader()

    @pytest.fixture
    def u8_img(self, rng):
        return rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)

    @pytest.fixture
    def f255_img(self, u8_img):
        return u8_img.astype(np.float32)

    def test_white_balance_lch_preserves_dtype(self, grader, u8_img, f255_img):
        out_u8 = grader.white_balance_lch(u8_img, temperature=5500.0, tint=10.0)
        out_f = grader.white_balance_lch(f255_img, temperature=5500.0, tint=10.0)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_white_balance_lch_float_better_than_uint8(self, grader, u8_img, f255_img):
        """Float path should not introduce additional quantization (mean diff small).

        Max diff can be high at hue boundaries where small LAB differences
        cause large BGR shifts, but mean diff should be within quantization noise.
        """
        out_u8 = grader.white_balance_lch(u8_img, temperature=5500.0, tint=10.0)
        out_f = grader.white_balance_lch(f255_img, temperature=5500.0, tint=10.0)
        mean_diff = np.abs(out_f - out_u8.astype(np.float32)).mean()
        assert mean_diff < 5.0, f"float vs uint8 mean diff {mean_diff} too high"

    def test_adjust_hsl_lch_preserves_dtype(self, grader, u8_img, f255_img):
        out_u8 = grader.adjust_hsl_lch(u8_img, hue_shift=5.0, sat_scale=1.2, lum_shift=3.0)
        out_f = grader.adjust_hsl_lch(f255_img, hue_shift=5.0, sat_scale=1.2, lum_shift=3.0)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_highlight_drift_preserves_dtype(self, grader, u8_img, f255_img):
        out_u8 = grader.highlight_drift(u8_img, strength=0.5)
        out_f = grader.highlight_drift(f255_img, strength=0.5)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_negative_split_tone_preserves_dtype(self, grader, u8_img, f255_img):
        out_u8 = grader.negative_split_tone(u8_img, shadow_desat=0.5, highlight_desat=0.3)
        out_f = grader.negative_split_tone(f255_img, shadow_desat=0.5, highlight_desat=0.3)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_color_transfer_preserves_dtype(self, grader, u8_img, f255_img):
        ref = u8_img
        out_u8 = grader.color_transfer(u8_img, ref, intensity=0.8)
        out_f = grader.color_transfer(f255_img, ref, intensity=0.8)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_add_glow_preserves_dtype(self, grader, u8_img, f255_img):
        out_u8 = grader._add_glow(u8_img, opacity=0.5)
        out_f = grader._add_glow(f255_img, opacity=0.5)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_fade_toe_preserves_dtype(self, grader, u8_img, f255_img):
        out_u8 = grader.fade_toe(u8_img, strength=0.5)
        out_f = grader.fade_toe(f255_img, strength=0.5)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_airy_haze_preserves_dtype(self, grader, u8_img, f255_img):
        out_u8 = grader.airy_haze(u8_img, strength=0.5)
        out_f = grader.airy_haze(f255_img, strength=0.5)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_clarity_split_preserves_dtype(self, grader, u8_img, f255_img):
        out_u8 = grader.clarity_split(u8_img, negative_strength=0.5, positive_strength=0.3)
        out_f = grader.clarity_split(f255_img, negative_strength=0.5, positive_strength=0.3)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_channel_mixer_bw_preserves_dtype(self, grader, u8_img, f255_img):
        out_u8 = grader.channel_mixer_bw(u8_img, r_weight=0.4, g_weight=0.5, b_weight=0.1)
        out_f = grader.channel_mixer_bw(f255_img, r_weight=0.4, g_weight=0.5, b_weight=0.1)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_channel_mixer_bw_float_matches_uint8_rounded(self, grader, u8_img, f255_img):
        """Float output rounded should match uint8 output (pure weighted sum, no LAB)."""
        out_u8 = grader.channel_mixer_bw(u8_img, r_weight=0.4, g_weight=0.5, b_weight=0.1)
        out_f = grader.channel_mixer_bw(f255_img, r_weight=0.4, g_weight=0.5, b_weight=0.1)
        diff = np.abs(out_f - out_u8.astype(np.float32)).max()
        assert diff < 1.5, f"channel_mixer_bw float vs uint8 diff {diff} too high"


# ---------------------------------------------------------------------------
# dtype-aware grain.py, utils.py, highlight.py, engine.py (F1/E2 Wave 4)
# ---------------------------------------------------------------------------


class TestDtypeAwareModules:
    """Verify dtype-awareness in grain.py, utils.py, highlight.py, engine.py."""

    @pytest.fixture
    def u8_img(self, rng):
        return rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)

    @pytest.fixture
    def f255_img(self, u8_img):
        return u8_img.astype(np.float32)

    @pytest.fixture
    def skin_mask(self, rng):
        return rng.random((48, 48), dtype=np.float32)

    def test_apply_film_grain_preserves_dtype(self, u8_img, f255_img):
        from retouch.grain import apply_film_grain
        out_u8 = apply_film_grain(u8_img, strength=0.3, seed=42)
        out_f = apply_film_grain(f255_img, strength=0.3, seed=42)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_apply_film_grain_float_in_range(self, f255_img):
        from retouch.grain import apply_film_grain
        out = apply_film_grain(f255_img, strength=0.3, seed=42)
        assert out.min() >= 0.0 and out.max() <= 255.0

    def test_apply_skin_diffusion_preserves_dtype(self, u8_img, f255_img, skin_mask):
        from retouch.utils import apply_skin_diffusion
        out_u8 = apply_skin_diffusion(u8_img, skin_mask, strength=50.0)
        out_f = apply_skin_diffusion(f255_img, skin_mask, strength=50.0)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_apply_highlight_rolloff_preserves_dtype(self, u8_img, f255_img):
        from retouch.highlight import apply_highlight_rolloff
        out_u8 = apply_highlight_rolloff(u8_img, strength=0.7)
        out_f = apply_highlight_rolloff(f255_img, strength=0.7)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_soft_clip_highlights_preserves_dtype(self, u8_img, f255_img):
        from retouch.highlight import soft_clip_highlights
        out_u8 = soft_clip_highlights(u8_img)
        out_f = soft_clip_highlights(f255_img)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_soft_clip_highlights_float_respects_threshold(self, f255_img):
        from retouch.highlight import soft_clip_highlights
        out = soft_clip_highlights(f255_img, threshold=230.0, rolloff_start=200.0)
        assert out.max() <= 230.0

    def test_apply_white_costume_lift_preserves_dtype(self, u8_img, f255_img, skin_mask):
        """Test the engine._apply_white_costume_lift static method."""
        from retouch.engine import RetouchEngine
        acc_lips = np.zeros_like(skin_mask)
        out_u8 = RetouchEngine._apply_white_costume_lift(u8_img, skin_mask, acc_lips, 1.0)
        out_f = RetouchEngine._apply_white_costume_lift(f255_img, skin_mask, acc_lips, 1.0)
        assert out_u8.dtype == np.uint8
        assert out_f.dtype == np.float32

    def test_apply_white_costume_lift_no_white_returns_unchanged(self, f255_img, skin_mask):
        """If no white pixels, should return input unchanged (same dtype)."""
        from retouch.engine import RetouchEngine
        # Dark image with no white pixels
        dark = np.zeros_like(f255_img)
        acc_lips = np.zeros_like(skin_mask)
        out = RetouchEngine._apply_white_costume_lift(dark, skin_mask, acc_lips, 1.0)
        assert out.dtype == np.float32
        assert np.array_equal(out, dark)
