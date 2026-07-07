"""Tests for retouch/makeup_v2.py — MakeupEngineV2.

Tests cover: dtype handling (uint8 + float32), no-op on zero strength,
shape/dtype preservation, and that the five operations actually change the
image when given valid landmarks.
"""

from __future__ import annotations

import numpy as np
import pytest

from retouch.makeup_v2 import MakeupEngineV2


# ---------------------------------------------------------------------------
# Mock landmarks — synthetic but anatomically plausible on a 100x100 image.
# ---------------------------------------------------------------------------


class MockLandmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class MockLandmarks:
    """MediaPipe-compatible landmark list with 478 points.

    Key landmarks are positioned to form two eyes, two eyebrows, and lips
    on a 100x100 canvas so every MakeupEngineV2 operation has valid geometry.
    """

    def __init__(self):
        self.landmark = [MockLandmark(0.5, 0.5) for _ in range(478)]

        # Left eye contour (indices in LEFT_EYE) — centered at (35, 45).
        left_eye_pts = [
            (0.30, 0.45), (0.30, 0.43), (0.32, 0.42), (0.34, 0.41),
            (0.36, 0.42), (0.38, 0.43), (0.40, 0.44), (0.40, 0.47),
            (0.38, 0.48), (0.36, 0.49), (0.34, 0.49), (0.32, 0.48),
            (0.33, 0.46), (0.34, 0.46), (0.36, 0.46), (0.35, 0.45),
        ]
        from retouch.parsing import LEFT_EYE
        for i, idx in enumerate(LEFT_EYE):
            if i < len(left_eye_pts):
                self.landmark[idx] = MockLandmark(*left_eye_pts[i])

        # Right eye contour (RIGHT_EYE) — centered at (65, 45).
        right_eye_pts = [
            (0.60, 0.44), (0.60, 0.42), (0.62, 0.41), (0.64, 0.41),
            (0.66, 0.42), (0.68, 0.43), (0.70, 0.45), (0.70, 0.47),
            (0.68, 0.48), (0.66, 0.49), (0.64, 0.49), (0.62, 0.48),
            (0.63, 0.46), (0.64, 0.46), (0.66, 0.46), (0.65, 0.45),
        ]
        from retouch.parsing import RIGHT_EYE
        for i, idx in enumerate(RIGHT_EYE):
            if i < len(right_eye_pts):
                self.landmark[idx] = MockLandmark(*right_eye_pts[i])

        # Left eyebrow (LEFT_EYEBROW) — above left eye, y ~ 0.35.
        from retouch.parsing import LEFT_EYEBROW
        left_brow_pts = [
            (0.28, 0.38), (0.30, 0.36), (0.33, 0.35), (0.35, 0.35),
            (0.38, 0.36), (0.28, 0.40), (0.30, 0.39), (0.33, 0.38),
            (0.36, 0.38), (0.40, 0.40),
        ]
        for i, idx in enumerate(LEFT_EYEBROW):
            if i < len(left_brow_pts):
                self.landmark[idx] = MockLandmark(*left_brow_pts[i])

        # Right eyebrow (RIGHT_EYEBROW) — above right eye.
        from retouch.parsing import RIGHT_EYEBROW
        right_brow_pts = [
            (0.72, 0.38), (0.70, 0.36), (0.67, 0.35), (0.65, 0.35),
            (0.62, 0.36), (0.72, 0.40), (0.70, 0.39), (0.67, 0.38),
            (0.64, 0.38), (0.60, 0.40),
        ]
        for i, idx in enumerate(RIGHT_EYEBROW):
            if i < len(right_brow_pts):
                self.landmark[idx] = MockLandmark(*right_brow_pts[i])

        # Lips outer contour — centered at (50, 75).
        from retouch.parsing import LIPS_OUTER
        lip_pts = [
            (0.40, 0.73), (0.42, 0.71), (0.46, 0.70), (0.50, 0.70),
            (0.54, 0.70), (0.58, 0.71), (0.60, 0.73), (0.60, 0.76),
            (0.55, 0.80), (0.50, 0.82), (0.45, 0.80), (0.40, 0.76),
            (0.42, 0.74), (0.46, 0.73), (0.50, 0.73), (0.54, 0.73),
            (0.58, 0.74), (0.46, 0.76), (0.50, 0.77), (0.54, 0.76),
        ]
        for i, idx in enumerate(LIPS_OUTER):
            if i < len(lip_pts):
                self.landmark[idx] = MockLandmark(*lip_pts[i])

        # Key single landmarks used by contour pass.
        # 116/345 = under-cheek points, 168/2 = nose bridge top/bottom.
        self.landmark[116] = MockLandmark(0.30, 0.62)
        self.landmark[345] = MockLandmark(0.70, 0.62)
        self.landmark[168] = MockLandmark(0.50, 0.40)
        self.landmark[2] = MockLandmark(0.50, 0.65)
        # Jaw sides
        for idx, (x, y) in zip(
            [172, 136, 150, 149, 176],
            [(0.20, 0.55), (0.25, 0.65), (0.30, 0.75), (0.35, 0.82), (0.22, 0.60)],
        ):
            self.landmark[idx] = MockLandmark(x, y)
        for idx, (x, y) in zip(
            [397, 365, 379, 378, 400],
            [(0.80, 0.55), (0.75, 0.65), (0.70, 0.75), (0.65, 0.82), (0.78, 0.60)],
        ):
            self.landmark[idx] = MockLandmark(x, y)
        # Nose sides
        for idx, (x, y) in zip(
            [220, 115, 48, 64],
            [(0.45, 0.50), (0.43, 0.55), (0.43, 0.60), (0.45, 0.62)],
        ):
            self.landmark[idx] = MockLandmark(x, y)
        for idx, (x, y) in zip(
            [439, 344, 278, 294],
            [(0.55, 0.50), (0.57, 0.55), (0.57, 0.60), (0.55, 0.62)],
        ):
            self.landmark[idx] = MockLandmark(x, y)
        # Cheekbone hollow polygon points
        for idx, (x, y) in zip(
            [117, 118, 50, 205, 425, 346, 347, 280],
            [(0.32, 0.60), (0.33, 0.62), (0.25, 0.65), (0.28, 0.58),
             (0.72, 0.60), (0.68, 0.60), (0.67, 0.62), (0.75, 0.65)],
        ):
            self.landmark[idx] = MockLandmark(x, y)
        # Outer eye corners (for eyeliner wing)
        self.landmark[33] = MockLandmark(0.30, 0.45)
        self.landmark[263] = MockLandmark(0.70, 0.45)
        self.landmark[133] = MockLandmark(0.40, 0.47)
        self.landmark[362] = MockLandmark(0.60, 0.47)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    return MakeupEngineV2()


@pytest.fixture
def img_u8():
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def img_f32():
    return np.full((100, 100, 3), 128.0, dtype=np.float32)


@pytest.fixture
def landmarks():
    return MockLandmarks()


# ---------------------------------------------------------------------------
# Eyeshadow
# ---------------------------------------------------------------------------


class TestApplyEyeshadow:
    def test_zero_strength_noop(self, engine, img_u8, landmarks):
        result = engine.apply_eyeshadow(img_u8, landmarks, strength=0)
        assert np.all(result == img_u8)

    def test_output_shape_dtype_u8(self, engine, img_u8, landmarks):
        result = engine.apply_eyeshadow(img_u8, landmarks, strength=50)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_output_shape_dtype_f32(self, engine, img_f32, landmarks):
        result = engine.apply_eyeshadow(img_f32, landmarks, strength=50)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.float32

    def test_changes_image(self, engine, img_u8, landmarks):
        result = engine.apply_eyeshadow(img_u8, landmarks, strength=80)
        assert not np.allclose(result, img_u8)

    def test_smoky_style(self, engine, img_u8, landmarks):
        result = engine.apply_eyeshadow(img_u8, landmarks, strength=70, style="smoky")
        assert result.shape == (100, 100, 3)
        assert not np.allclose(result, img_u8)

    def test_color_tuple(self, engine, img_u8, landmarks):
        result = engine.apply_eyeshadow(img_u8, landmarks, color=(120, 80, 200), strength=60)
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_u8)

    def test_color_palette_key(self, engine, img_u8, landmarks):
        result = engine.apply_eyeshadow(img_u8, landmarks, color="smoky", strength=60)
        assert not np.allclose(result, img_u8)


# ---------------------------------------------------------------------------
# Eyeliner
# ---------------------------------------------------------------------------


class TestApplyEyeliner:
    def test_zero_thickness_noop(self, engine, img_u8, landmarks):
        result = engine.apply_eyeliner(img_u8, landmarks, thickness=0)
        assert np.all(result == img_u8)

    def test_output_shape_dtype_u8(self, engine, img_u8, landmarks):
        result = engine.apply_eyeliner(img_u8, landmarks, thickness=2)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_output_shape_dtype_f32(self, engine, img_f32, landmarks):
        result = engine.apply_eyeliner(img_f32, landmarks, thickness=2)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.float32

    def test_changes_image(self, engine, img_u8, landmarks):
        result = engine.apply_eyeliner(img_u8, landmarks, thickness=3)
        assert not np.allclose(result, img_u8)

    def test_wing_style(self, engine, img_u8, landmarks):
        result = engine.apply_eyeliner(img_u8, landmarks, thickness=2, style="wing")
        assert result.shape == (100, 100, 3)
        assert not np.allclose(result, img_u8)

    def test_color_tuple(self, engine, img_u8, landmarks):
        result = engine.apply_eyeliner(img_u8, landmarks, color=(40, 50, 70), thickness=2)
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_u8)


# ---------------------------------------------------------------------------
# Contour
# ---------------------------------------------------------------------------


class TestApplyContour:
    def test_zero_strength_noop(self, engine, img_u8, landmarks):
        result = engine.apply_contour(img_u8, landmarks, strength=0)
        assert np.all(result == img_u8)

    def test_output_shape_dtype_u8(self, engine, img_u8, landmarks):
        result = engine.apply_contour(img_u8, landmarks, strength=40)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_output_shape_dtype_f32(self, engine, img_f32, landmarks):
        result = engine.apply_contour(img_f32, landmarks, strength=40)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.float32

    def test_changes_image(self, engine, img_u8, landmarks):
        result = engine.apply_contour(img_u8, landmarks, strength=60)
        assert not np.allclose(result, img_u8)

    def test_full_strength_safe(self, engine, img_u8, landmarks):
        """Full strength must not produce NaN or out-of-range values."""
        result = engine.apply_contour(img_u8, landmarks, strength=100)
        assert result.dtype == np.uint8
        assert np.isfinite(result.astype(np.float32)).all()
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# Brows
# ---------------------------------------------------------------------------


class TestApplyBrows:
    def test_zero_thickness_noop(self, engine, img_u8, landmarks):
        result = engine.apply_brows(img_u8, landmarks, thickness=0)
        assert np.all(result == img_u8)

    def test_output_shape_dtype_u8(self, engine, img_u8, landmarks):
        result = engine.apply_brows(img_u8, landmarks, thickness=2)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_output_shape_dtype_f32(self, engine, img_f32, landmarks):
        result = engine.apply_brows(img_f32, landmarks, thickness=2)
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.float32

    def test_changes_image(self, engine, img_u8, landmarks):
        result = engine.apply_brows(img_u8, landmarks, thickness=3, color="black")
        assert not np.allclose(result, img_u8)

    def test_color_tuple(self, engine, img_u8, landmarks):
        result = engine.apply_brows(img_u8, landmarks, color=(50, 70, 100), thickness=2)
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_u8)


# ---------------------------------------------------------------------------
# Ombre lips
# ---------------------------------------------------------------------------


class TestApplyOmbreLips:
    def test_output_shape_dtype_u8(self, engine, img_u8, landmarks):
        result = engine.apply_ombre_lips(img_u8, landmarks, "red", "pink")
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

    def test_output_shape_dtype_f32(self, engine, img_f32, landmarks):
        result = engine.apply_ombre_lips(img_f32, landmarks, "red", "pink")
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.float32

    def test_changes_image(self, engine, img_u8, landmarks):
        result = engine.apply_ombre_lips(img_u8, landmarks, "red", "pink")
        assert not np.allclose(result, img_u8)

    def test_color_tuples(self, engine, img_u8, landmarks):
        result = engine.apply_ombre_lips(
            img_u8, landmarks, (80, 80, 210), (160, 140, 210)
        )
        assert result.dtype == np.uint8
        assert not np.allclose(result, img_u8)

    def test_safe_range(self, engine, img_u8, landmarks):
        result = engine.apply_ombre_lips(img_u8, landmarks, "plum", "gold")
        assert result.min() >= 0 and result.max() <= 255


# ---------------------------------------------------------------------------
# ParamSpec registration
# ---------------------------------------------------------------------------


class TestParamSpecs:
    def test_makeup_v2_params_registered(self):
        from retouch.params import PROCESSING_PARAMS, get_param

        names = {p.name for p in PROCESSING_PARAMS}
        expected = {
            "mv2_eyeshadow", "mv2_eyeshadow_color", "mv2_eyeshadow_style",
            "mv2_eyeliner", "mv2_eyeliner_color", "mv2_eyeliner_style",
            "mv2_contour", "mv2_brows", "mv2_brows_color",
            "mv2_ombre", "mv2_ombre_color1", "mv2_ombre_color2",
        }
        assert expected.issubset(names)
        # Spot-check a couple via get_param.
        assert get_param("mv2_eyeshadow").default == 0
        assert get_param("mv2_eyeshadow_style").default == "natural"
        assert get_param("mv2_ombre").default is False
