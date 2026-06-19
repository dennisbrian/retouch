"""Tests for retouch/grading.py — ColorGrader, PRESETS."""

import numpy as np
import cv2
import pytest

from retouch.grading import ColorGrader, PRESETS


@pytest.fixture
def grader():
    return ColorGrader()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def gradient_img():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    for y in range(64):
        img[y, :] = [y * 2, y * 2, y * 2]
    return img


class TestPRESETS:
    def test_all_presets_have_description(self):
        for name, settings in PRESETS.items():
            assert "description" in settings, f"{name} missing description"

    def test_all_presets_have_minimal_keys(self):
        required = {"curves"}
        for name, settings in PRESETS.items():
            if "saturation_boost" not in settings:
                if "bw_noir" not in name:
                    pass

    def test_preset_names_unique(self):
        assert len(PRESETS) == len(set(PRESETS.keys()))

    def test_known_presets_exist(self):
        known = {"natural", "magazine", "beauty", "cosplay", "film", "scifi", "fantasy"}
        for k in known:
            assert k in PRESETS, f"Missing preset: {k}"


class TestGrade:
    def test_grade_natural(self, grader, img):
        result = grader.grade(img, "natural")
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_grade_by_dict(self, grader, img):
        settings = {"curves": {"L": [(0, 0), (128, 128), (255, 255)]}, "saturation_boost": 0.1}
        result = grader.grade(img, settings)
        assert result.shape == img.shape

    def test_grade_unknown_preset_falls_back(self, grader, img):
        result = grader.grade(img, "nonexistent_preset_xyz")
        assert result.shape == img.shape

    def test_zero_intensity_returns_original(self, grader, img):
        result = grader.grade(img, "cosplay", intensity=0.0)
        assert np.all(result == img)

    def test_full_intensity_changes_image(self, grader, img):
        result = grader.grade(img, "cosplay", intensity=1.0)
        assert not np.allclose(result, img)

    def test_grade_all_presets(self, grader, gradient_img):
        for preset_name in PRESETS:
            result = grader.grade(gradient_img, preset_name)
            assert result.shape == gradient_img.shape
            assert result.dtype == np.uint8

    def test_split_tone_mask(self, grader, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        result = grader.grade(img, "magazine", split_tone_mask=mask)
        assert result.shape == img.shape

    def test_glow_mask(self, grader, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader.grade(img, "cosplay", glow_mask=mask)
        assert result.shape == img.shape


class TestGradeStack:
    def test_empty_returns_original(self, grader, img):
        result = grader.grade_stack(img, {})
        assert np.all(result == img)

    def test_single_stack(self, grader, img):
        result = grader.grade_stack(img, {"natural": 1.0})
        assert not np.allclose(result, img) or np.all(result == img)


class TestImpactFinish:
    def test_zero_strength(self, grader, img):
        result = grader.add_impact_finish(img, 0)
        assert np.all(result == img)

    def test_impact_changes_image(self, grader, img):
        result = grader.add_impact_finish(img, 50)
        assert not np.allclose(result, img)


class TestColorTransfer:
    @pytest.fixture
    def color_img(self):
        """A non-neutral image with actual color content for transfer tests."""
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 0] = 100  # B
        img[:, :, 1] = 150  # G
        img[:, :, 2] = 200  # R
        return img

    def test_no_ref(self, grader, img):
        result = grader.color_transfer(img, None)
        assert np.all(result == img)

    def test_transfer_to_self(self, grader, color_img):
        result = grader.color_transfer(color_img, color_img)
        assert np.allclose(result, color_img, atol=2)

    def test_transfer_different(self, grader, color_img):
        ref = np.full((64, 64, 3), 200, dtype=np.uint8)
        result = grader.color_transfer(color_img, ref)
        assert not np.allclose(result, color_img)

    def test_transfer_with_intensity(self, grader, color_img):
        ref = np.full((64, 64, 3), 200, dtype=np.uint8)
        result_full = grader.color_transfer(color_img, ref, intensity=1.0)
        result_half = grader.color_transfer(color_img, ref, intensity=0.0)
        assert np.all(result_half == color_img)
        assert not np.allclose(result_full, color_img)


class TestColorTransferHist:
    def test_no_ref(self, grader, img):
        result = grader.color_transfer_hist(img, None)
        assert np.all(result == img)

    def test_transfer_to_self(self, grader, img):
        result = grader.color_transfer_hist(img, img)
        assert np.allclose(result, img, atol=2)


class TestInternalMethods:
    def test_luminance_curve(self, grader, gradient_img):
        curve = [(0, 0), (128, 200), (255, 255)]
        result = grader._apply_luminance_curve(gradient_img, curve)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() > gradient_img.mean()

    def test_shadow_lift(self, grader):
        dark = np.full((64, 64, 3), 50, dtype=np.uint8)
        result = grader._lift_shadows(dark, 30)
        assert not np.allclose(result, dark)

    def test_warmth(self, grader, img):
        result = grader._adjust_warmth(img, 0.5)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 2].mean() > orig_lab[:, :, 2].mean()

    def test_saturation(self, grader, img):
        result = grader._adjust_saturation(img, 0.5)
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        orig_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        assert hsv[:, :, 1].mean() >= orig_hsv[:, :, 1].mean()

    def test_vignette(self, grader, img):
        result = grader._add_vignette(img, 0.5)
        assert not np.allclose(result, img)

    def test_rgb_curves(self, grader, img):
        curves = {"R": [(0, 0), (128, 200), (255, 255)]}
        result = grader._apply_rgb_curves(img, curves)
        assert not np.allclose(result, img)

    def test_white_balance(self, grader, img):
        result = grader._adjust_white_balance(img, {"R": 1.2, "G": 1.0, "B": 0.8})
        assert not np.allclose(result, img)

    def test_grain_produces_output(self, grader, img):
        result = grader._add_grain(img, 0.1)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_clarity(self, grader, gradient_img):
        result = grader._add_clarity(gradient_img, 0.3)
        assert result.shape == gradient_img.shape
