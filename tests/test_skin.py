"""Tests for retouch/skin.py — SkinProcessor."""

import numpy as np
import cv2
import pytest

from retouch.skin import SkinProcessor


@pytest.fixture
def proc():
    return SkinProcessor()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def face_mask():
    mask = np.zeros((64, 64), dtype=np.float32)
    mask[16:48, 16:48] = 1.0
    return cv2.GaussianBlur(mask, (15, 15), 0)


class TestWhiten:
    def test_zero_strength(self, proc, img, face_mask):
        result = proc.whiten(img, face_mask, strength=0)
        assert np.all(result == img)

    def test_whiten_brightens(self, proc, img, face_mask):
        result = proc.whiten(img, face_mask, strength=50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        skin_pixels = lab[:, :, 0][face_mask > 0.3]
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        orig_skin = orig_lab[:, :, 0][face_mask > 0.3]
        assert skin_pixels.mean() > orig_skin.mean()

    def test_negative_strength_darkens(self, proc, img, face_mask):
        result = proc.whiten(img, face_mask, strength=-50)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        skin_pixels = lab[:, :, 0][face_mask > 0.3]
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        orig_skin = orig_lab[:, :, 0][face_mask > 0.3]
        assert skin_pixels.mean() < orig_skin.mean()

    def test_rosy_tone_shifts_a(self, proc, img, face_mask):
        result = proc.whiten(img, face_mask, strength=50, tone="rosy")
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        skin_a = lab[:, :, 1][face_mask > 0.3].mean()
        assert skin_a > 128

    def test_porcelain_tone_shifts_b(self, proc, img, face_mask):
        result = proc.whiten(img, face_mask, strength=50, tone="porcelain")
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        skin_b = lab[:, :, 2][face_mask > 0.3].mean()
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        orig_b = orig_lab[:, :, 2][face_mask > 0.3].mean()
        assert skin_b < orig_b

    def test_neutral_tone_no_color_shift(self, proc, img, face_mask):
        result_neutral = proc.whiten(img, face_mask, strength=50, tone="neutral")
        result_rosy = proc.whiten(img, face_mask, strength=50, tone="rosy")
        lab_neutral = cv2.cvtColor(result_neutral, cv2.COLOR_BGR2LAB)
        lab_rosy = cv2.cvtColor(result_rosy, cv2.COLOR_BGR2LAB)
        assert not np.allclose(lab_neutral[:, :, 1], lab_rosy[:, :, 1])

    def test_output_type_and_shape(self, proc, img, face_mask):
        result = proc.whiten(img, face_mask, strength=30)
        assert result.dtype == np.uint8
        assert result.shape == (64, 64, 3)


class TestEqualize:
    def test_zero_strength(self, proc, img, face_mask):
        result = proc.equalize(img, face_mask, strength=0)
        assert np.all(result == img)

    def test_equalize_changes_image(self, proc, img, face_mask):
        result = proc.equalize(img, face_mask, strength=50)
        assert not np.allclose(result, img)

    def test_with_ref_lab(self, proc, img, face_mask):
        ref = np.full((64, 64, 3), 100, dtype=np.uint8)
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
        result = proc.equalize(img, face_mask, strength=50, ref_lab=ref_lab)
        assert not np.allclose(result, img)

    def test_non_skin_unchanged(self, proc, img, face_mask):
        img_copy = img.copy()
        img_copy[32, 32] = [200, 200, 200]
        result = proc.equalize(img_copy, face_mask, strength=50)
        # Non-skin corner should be unchanged
        assert np.all(result[0:4, 0:4] == img_copy[0:4, 0:4])


class TestDodgeBurn:
    def test_zero_strength(self, proc, img):
        class FakeRegions:
            skin = np.ones((64, 64), dtype=np.float32)
            nose_bridge = None
            forehead_center = None
            cheek_highlights_l = None
            cheek_highlights_r = None
            jawline_contour = None
        result = proc.dodge_burn(img, FakeRegions(), strength=0)
        assert np.all(result == img)

    def test_dodge_burn_changes_image(self, proc, img):
        class FakeRegions:
            skin = np.ones((64, 64), dtype=np.float32)
            nose_bridge = np.ones((64, 64), dtype=np.float32) * 0.5
            forehead_center = np.zeros((64, 64), dtype=np.float32)
            cheek_highlights_l = np.zeros((64, 64), dtype=np.float32)
            cheek_highlights_r = np.zeros((64, 64), dtype=np.float32)
            jawline_contour = np.zeros((64, 64), dtype=np.float32)
        result = proc.dodge_burn(img, FakeRegions(), strength=50)
        assert not np.allclose(result, img)


class TestHarmonizeNeck:
    def test_no_person_mask_returns_original(self, proc, img):
        class FakeLandmarks:
            landmark = []
        result = proc.harmonize_neck(img, FakeLandmarks(), None, np.ones((64, 64)))
        assert np.all(result == img)

    def test_no_face_skin_returns_original(self, proc, img):
        class FakeLandmarks:
            landmark = []
        pm = np.ones((64, 64), dtype=np.uint8) * 255
        result = proc.harmonize_neck(img, FakeLandmarks(), pm, np.zeros((64, 64)))
        assert np.all(result == img)

    def test_depth_gating_applied_when_low_yaw(self, proc):
        class MockLandmark:
            def __init__(self, x, y, z=0.0):
                self.x = x
                self.y = y
                self.z = z
                
        class MockLandmarksList:
            def __init__(self, yaw_ratio=1.0):
                self.landmark = [MockLandmark(0.5, 0.5, 0.0) for _ in range(468)]
                self.landmark[6].x = 0.5
                self.landmark[234].x = 0.3
                self.landmark[454].x = 0.7
                self.landmark[33].x = 0.4
                self.landmark[263].x = 0.6
                self.landmark[152].y = 0.7
                
                if yaw_ratio != 1.0:
                    shift = 0.2 * (yaw_ratio - 1.0) / (yaw_ratio + 1.0)
                    self.landmark[6].x = 0.5 + shift

        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        face_skin = np.zeros((100, 100), dtype=np.float32)
        face_skin[40:60, 40:60] = 1.0
        
        neck_mask = np.zeros((100, 100), dtype=np.float32)
        neck_mask[75:90, 30:70] = 1.0
        
        person_mask = np.ones((100, 100), dtype=np.uint8) * 255
        
        lms_low = MockLandmarksList(yaw_ratio=1.0)
        lms_low.landmark[152].z = 10.0
        
        result_gated = proc.harmonize_neck(img, lms_low, person_mask, face_skin, neck_mask, strength=100)
        assert np.allclose(result_gated, img)


class TestSpecularBloom:
    def test_zero_strength(self, proc, img, face_mask):
        result = proc.apply_specular_bloom(img, face_mask, strength=0)
        assert np.all(result == img)

    def test_no_skin_mask(self, proc, img):
        result = proc.apply_specular_bloom(img, None, strength=50)
        assert np.all(result == img)

    def test_no_highlights_returns_original(self, proc, face_mask):
        dark = np.full((64, 64, 3), 30, dtype=np.uint8)
        result = proc.apply_specular_bloom(dark, face_mask, strength=50)
        assert np.all(result == dark)


class TestHighlightProtection:
    def test_dark_pixels_protected(self, proc):
        lab = np.zeros((10, 10, 3), dtype=np.float32)
        lab[:, :, 0] = 100
        prot = proc._get_highlight_protection(lab)
        assert prot.max() == 1.0

    def test_bright_pixels_decay(self, proc):
        lab = np.zeros((10, 10, 3), dtype=np.float32)
        lab[:, :, 0] = 240
        prot = proc._get_highlight_protection(lab)
        assert prot.max() < 1.0

    def test_very_bright_zero(self, proc):
        lab = np.zeros((10, 10, 3), dtype=np.float32)
        lab[:, :, 0] = 255
        prot = proc._get_highlight_protection(lab)
        assert prot.max() == 0.0
