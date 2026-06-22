"""Tests for retouch/grading.py — internal/private methods of ColorGrader."""
import cv2
import numpy as np
import pytest
from retouch.grading import ColorGrader, register_presets_dir, list_available_presets


@pytest.fixture
def grader():
    return ColorGrader()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def varied_img():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32, :, :] = [50, 50, 50]
    img[32:, :, :] = [200, 200, 200]
    return img


@pytest.fixture
def colorful_img():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:21, :, 2] = 200
    img[21:43, :, 1] = 180
    img[43:, :, 0] = 220
    return img


class TestRegisterAndListPresets:
    def test_list_available(self):
        presets = list_available_presets()
        assert len(presets) > 0
        assert "natural" in presets


class TestAddGlow:
    def test_zero_opacity(self, grader, img):
        result = grader._add_glow(img, 0)
        assert np.all(result == img)

    def test_changes_bright_image(self, grader):
        img = np.full((64, 64, 3), 235, dtype=np.uint8)
        result = grader._add_glow(img, 0.5)
        assert not np.allclose(result, img)

    def test_with_tint(self, grader):
        img = np.full((64, 64, 3), 235, dtype=np.uint8)
        result = grader._add_glow(img, 0.5, tint=(200, 100, 150))
        assert not np.allclose(result, img)

    def test_with_mask(self, grader):
        img = np.full((64, 64, 3), 235, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader._add_glow(img, 0.5, mask=mask)
        assert not np.allclose(result, img)

    def test_output_type(self, grader, img):
        result = grader._add_glow(img, 0.5)
        assert result.dtype == np.uint8


class TestSplitTone:
    def test_returns_image(self, grader, varied_img):
        tones = {"shadows": [118, 130], "highlights": [138, 120]}
        result = grader._split_tone(varied_img, tones)
        assert result.shape == varied_img.shape

    def test_with_mask(self, grader, varied_img):
        tones = {"shadows": [118, 130], "highlights": [138, 120]}
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader._split_tone(varied_img, tones, mask=mask)
        assert result.shape == varied_img.shape


class TestSplitToneThreeWay:
    def test_no_tones_returns_original(self, grader, img):
        result = grader._split_tone_three_way(img, {})
        assert np.all(result == img)

    def test_full_tones_changes_image(self, grader, varied_img):
        tones = {
            "shadows": {"hue": 30, "sat": 20},
            "midtones": {"hue": 120, "sat": 15},
            "highlights": {"hue": 300, "sat": 25},
        }
        result = grader._split_tone_three_way(varied_img, tones)
        assert not np.allclose(result, varied_img)

    def test_no_offset_returns_original(self, grader, img):
        tones = {
            "shadows": {"hue": 0, "sat": 0},
            "midtones": {"hue": 0, "sat": 0},
            "highlights": {"hue": 0, "sat": 0},
        }
        result = grader._split_tone_three_way(img, tones)
        assert np.all(result == img)

    def test_with_mask(self, grader, varied_img):
        tones = {
            "shadows": {"hue": 30, "sat": 20},
            "midtones": {"hue": 120, "sat": 15},
            "highlights": {"hue": 300, "sat": 25},
        }
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader._split_tone_three_way(varied_img, tones, mask=mask)
        assert result.shape == varied_img.shape


class TestAddOrtonGlow:
    def test_zero_opacity(self, grader, img):
        result = grader._add_orton_glow(img, 0)
        assert np.all(result == img)

    def test_changes_varied_image(self, grader, varied_img):
        result = grader._add_orton_glow(varied_img, 0.5)
        assert not np.allclose(result, varied_img)

    def test_with_mask(self, grader, varied_img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader._add_orton_glow(varied_img, 0.5, mask=mask)
        assert result.shape == varied_img.shape

    def test_output_type(self, grader, img):
        result = grader._add_orton_glow(img, 0.5)
        assert result.dtype == np.uint8


class TestAddChromaticAberration:
    def test_zero_disp(self, grader, img):
        result = grader._add_chromatic_aberration(img, 0)
        assert np.all(result == img)

    def test_changes_varied_image(self, grader, varied_img):
        result = grader._add_chromatic_aberration(varied_img, 5)
        assert not np.allclose(result, varied_img)

    def test_output_shape(self, grader, img):
        result = grader._add_chromatic_aberration(img, 5)
        assert result.shape == img.shape


class TestAddHalation:
    def test_zero_intensity(self, grader, img):
        result = grader._add_halation(img, intensity=0)
        assert np.all(result == img)

    def test_changes_varied_image(self, grader, varied_img):
        result = grader._add_halation(varied_img, threshold=150, radius=15, intensity=0.5)
        assert not np.allclose(result, varied_img)

    def test_output_shape(self, grader, img):
        result = grader._add_halation(img, threshold=200, radius=15, intensity=0.3)
        assert result.shape == img.shape


class TestAddGrain:
    def test_zero_strength(self, grader, img):
        result = grader._add_grain(img, 0)
        assert np.all(result == img)

    def test_changes_image(self, grader, img):
        result = grader._add_grain(img, 0.1)
        assert not np.allclose(result, img)

    def test_output_shape(self, grader, img):
        result = grader._add_grain(img, 0.05)
        assert result.shape == img.shape


class TestAdjustWhiteBalance:
    def test_identity_multipliers(self, grader, img):
        result = grader._adjust_white_balance(img, {"R": 1.0, "G": 1.0, "B": 1.0})
        assert np.all(result == img)

    def test_changes_image(self, grader, img):
        result = grader._adjust_white_balance(img, {"R": 1.2, "G": 1.0, "B": 0.8})
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        assert not np.allclose(lab, orig_lab)

    def test_empty_multipliers(self, grader, img):
        result = grader._adjust_white_balance(img, {})
        assert np.all(result == img)


class TestAddHaze:
    def test_zero_strength(self, grader, img):
        result = grader._add_haze(img, 0)
        assert np.all(result == img)

    def test_changes_image(self, grader, varied_img):
        result = grader._add_haze(varied_img, 0.3)
        assert not np.allclose(result, varied_img)

    def test_with_mask(self, grader, varied_img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader._add_haze(varied_img, 0.3, mask=mask)
        assert result.shape == varied_img.shape


class TestAddVignette:
    def test_zero_strength(self, grader, img):
        result = grader._add_vignette(img, 0)
        assert np.all(result == img)

    def test_darkens_corners(self, grader):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = grader._add_vignette(img, 0.9)
        assert result[50, 50, 0] > result[5, 5, 0]


class TestApplyRGBCurves:
    def test_identity_returns_original(self, grader, img):
        result = grader._apply_rgb_curves(img, {})
        assert np.all(result == img)

    def test_r_curve_changes(self, grader, varied_img):
        result = grader._apply_rgb_curves(varied_img, {"R": [(0, 0), (128, 200), (255, 255)]})
        assert not np.allclose(result, varied_img)


class TestHSlHueShift:
    def test_no_shifts_returns_original(self, grader, img):
        result = grader._hsl_hue_shift(img, {})
        assert np.all(result == img)

    def test_hue_shift_changes_colorful(self, grader, colorful_img):
        result = grader._hsl_hue_shift(colorful_img, {"red": 30})
        assert not np.allclose(result, colorful_img)


class TestApplyCalibration:
    def test_empty_returns_original(self, grader, img):
        result = grader._apply_calibration(img, {})
        assert np.all(result == img)

    def test_calibration_changes_colorful(self, grader, colorful_img):
        cal = {"red": {"hue": 5, "sat": 3}, "green": {"hue": -2, "sat": 1}}
        result = grader._apply_calibration(colorful_img, cal)
        assert not np.allclose(result, colorful_img)


class TestGuidedFilter:
    def test_output_shape(self, grader, img):
        guide = np.full((64, 64), 128, dtype=np.float32)
        src = np.full((64, 64), 128, dtype=np.float32)
        result = grader._guided_filter(guide, src, r=5, eps=0.02)
        assert result.shape == (64, 64)

    def test_preserves_flat_input(self, grader):
        guide = np.full((32, 32), 100, dtype=np.float32)
        src = np.full((32, 32), 100, dtype=np.float32)
        result = grader._guided_filter(guide, src, r=5, eps=0.02)
        assert np.allclose(result, src, atol=1)


class TestAddClarity:
    def test_zero_strength(self, grader, img):
        result = grader._add_clarity(img, 0)
        assert np.all(result == img)

    def test_changes_varied_image(self, grader, varied_img):
        result = grader._add_clarity(varied_img, 0.5)
        assert not np.allclose(result, varied_img)

    def test_output_type(self, grader, img):
        result = grader._add_clarity(img, 0.3)
        assert result.dtype == np.uint8


class TestApplyHSLAdjustments:
    def test_empty_returns_original(self, grader, img):
        result = grader._apply_hsl_adjustments(img, {})
        assert np.all(result == img)

    def test_hue_adjustment_changes_colorful(self, grader, colorful_img):
        adj = {"hue": {"red": 10, "blue": -5}}
        result = grader._apply_hsl_adjustments(colorful_img, adj)
        assert not np.allclose(result, colorful_img)

    def test_sat_adjustment_changes(self, grader, varied_img):
        adj = {"saturation": {"red": 20, "green": -10}}
        result = grader._apply_hsl_adjustments(varied_img, adj)
        assert not np.allclose(result, varied_img)

    def test_lum_adjustment_changes(self, grader, varied_img):
        adj = {"luminance": {"red": 15, "blue": -10}}
        result = grader._apply_hsl_adjustments(varied_img, adj)
        assert not np.allclose(result, varied_img)


class TestAddFilmEmulation:
    def test_unknown_preset_returns_original(self, grader, img):
        result = grader._add_film_emulation(img, "nonexistent")
        assert np.all(result == img)


class TestAddSparkles:
    def test_zero_strength(self, grader, img):
        result = grader._add_sparkles(img, 0)
        assert np.all(result == img)

    def test_output_shape(self, grader, img):
        result = grader._add_sparkles(img, 0.5)
        assert result.shape == img.shape
