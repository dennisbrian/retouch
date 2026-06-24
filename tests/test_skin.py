"""Tests for retouch/skin.py — SkinProcessor."""

import numpy as np
import cv2
import pytest

from retouch.perf_optimizations import _build_smooth_mask
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

    def test_dodge_burn_excludes_hair(self, proc):
        # Build a 64x64 image with a non-trivial gradient so dodge/burn
        # produces a measurable change. Mark the top half as hair and the
        # bottom half as skin. A strong nose_bridge mask forces a real edit
        # inside skin; the hair half must remain pixel-identical to input.
        h, w = 64, 64
        img = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            img[y, :, 0] = y * 4
            img[y, :, 1] = 128
            img[y, :, 2] = 64

        skin = np.zeros((h, w), dtype=np.float32)
        skin[:] = 1.0
        hair = np.zeros((h, w), dtype=np.float32)
        hair[:h // 2, :] = 1.0
        nose_bridge = np.zeros((h, w), dtype=np.float32)
        nose_bridge[h // 2:, :] = 1.0  # only below hair, inside skin

        class FakeRegions:
            pass

        regions = FakeRegions()
        regions.skin = skin
        regions.hair = hair
        regions.left_eyebrow = None
        regions.right_eyebrow = None
        regions.nose_bridge = nose_bridge
        regions.forehead_center = np.zeros((h, w), dtype=np.float32)
        regions.cheek_highlights_l = np.zeros((h, w), dtype=np.float32)
        regions.cheek_highlights_r = np.zeros((h, w), dtype=np.float32)
        regions.jawline_contour = np.zeros((h, w), dtype=np.float32)

        result = proc.dodge_burn(img, regions, strength=80)

        # Hair half must be byte-identical to the input — no smoothing, no sculpt.
        assert np.array_equal(result[:h // 2, :], img[:h // 2, :])
        # Sanity: skin half (below hair) actually changed.
        assert not np.array_equal(result[h // 2:, :], img[h // 2:, :])


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


class TestBuildSmoothMask:
    def test_skin_alone(self):
        skin = np.ones((4, 4), dtype=np.float32)
        out = _build_smooth_mask(skin, exclusions=())
        assert np.array_equal(out, skin)

    def test_skin_none_requires_shape(self):
        out = _build_smooth_mask(None, exclusions=(), out_shape=(3, 5))
        assert out.shape == (3, 5)
        assert out.dtype == np.float32
        assert np.all(out == 0.0)

    def test_skin_none_without_shape_raises(self):
        with pytest.raises(ValueError):
            _build_smooth_mask(None, exclusions=())

    def test_excludes_hair(self):
        h, w = 64, 64
        skin = np.ones((h, w), dtype=np.float32)
        hair = np.zeros((h, w), dtype=np.float32)
        hair[: h // 2, :] = 1.0  # top half is hair

        out = _build_smooth_mask(skin, exclusions=(hair,))
        # Hair region must be fully excluded.
        assert np.all(out[: h // 2, :] == 0.0)
        # Skin-only region must be untouched.
        assert np.all(out[h // 2:, :] == 1.0)

    def test_excludes_multiple_regions(self):
        h, w = 32, 32
        skin = np.ones((h, w), dtype=np.float32)
        hair = np.zeros((h, w), dtype=np.float32)
        hair[0:8, :] = 1.0
        left_eye = np.zeros((h, w), dtype=np.float32)
        left_eye[10:18, 4:12] = 1.0
        lips = np.zeros((h, w), dtype=np.float32)
        lips[24:30, 10:22] = 1.0

        out = _build_smooth_mask(
            skin, exclusions=(hair, left_eye, lips)
        )
        assert np.all(out[0:8, :] == 0.0)
        assert np.all(out[10:18, 4:12] == 0.0)
        assert np.all(out[24:30, 10:22] == 0.0)
        # Pixels not in any exclusion remain at 1.0.
        assert np.all(out[20:23, 0:4] == 1.0)

    def test_none_exclusion_is_skipped(self):
        skin = np.ones((4, 4), dtype=np.float32)
        hair = np.zeros((4, 4), dtype=np.float32)
        hair[:2, :] = 1.0
        out = _build_smooth_mask(
            skin, exclusions=(None, hair, None)
        )
        assert np.all(out[:2, :] == 0.0)
        assert np.all(out[2:, :] == 1.0)

    def test_output_is_clipped_to_unit_range(self):
        skin = np.ones((4, 4), dtype=np.float32)
        # Exclusion larger than skin would underflow without clipping.
        excl = np.full((4, 4), 0.5, dtype=np.float32)
        out = _build_smooth_mask(skin, exclusions=(excl,))
        assert out.min() >= 0.0
        assert out.max() <= 1.0
        assert np.allclose(out, 0.5)


class _FakeLocalClarityRegions:
    """Minimal stub matching the FaceRegions attrs read by local_clarity."""

    def __init__(self, nose=None, lips=None, left_eye=None, right_eye=None, nose_bridge=None):
        self.nose = nose
        self.lips = lips
        self.left_eye = left_eye
        self.right_eye = right_eye
        self.nose_bridge = nose_bridge


class TestLocalClarity:
    def test_local_clarity_no_op_at_zero(self, proc):
        # Constant mid-gray image with a bright bar in the centre would
        # normally get sharpened, so any change here is a true signal.
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[24:40, 24:40] = 220
        regions = _FakeLocalClarityRegions(
            nose=np.ones((64, 64), dtype=np.float32)
        )
        result = proc.local_clarity(img, regions, strength=0.0, radius=5)
        assert np.array_equal(result, img)

    def test_local_clarity_boosts_in_mask_regions(self, proc):
        # Build a 64x64 image with a sharp vertical bar (edge) inside the
        # nose region. After local_clarity, the gradient magnitude of the
        # edge inside the mask must increase (high-frequency content is
        # amplified).
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        img[8:56, 28:36] = 220

        nose_mask = np.zeros((64, 64), dtype=np.float32)
        nose_mask[8:56, 24:40] = 1.0
        regions = _FakeLocalClarityRegions(nose=nose_mask)

        def edge_gradient(canvas: np.ndarray) -> float:
            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            return float(mag[nose_mask > 0.5].mean())

        before = edge_gradient(img)
        result = proc.local_clarity(img, regions, strength=0.30, radius=5)
        after = edge_gradient(result)

        assert after > before * 1.05, (
            f"expected gradient inside mask to grow; before={before:.3f} after={after:.3f}"
        )

    def test_local_clarity_does_not_touch_outside_mask(self, proc):
        # Small inner mask, all other region attrs None. Pixels far from
        # the mask (e.g. the four corners) must be returned exactly — the
        # feathering only spreads ~3px from the mask boundary.
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Drop distinctive markers into the corners so any leak shows up.
        img[0:4, 0:4] = 10
        img[0:4, 60:64] = 20
        img[60:64, 0:4] = 30
        img[60:64, 60:64] = 40

        nose_mask = np.zeros((64, 64), dtype=np.float32)
        nose_mask[28:36, 28:36] = 1.0
        regions = _FakeLocalClarityRegions(nose=nose_mask)

        result = proc.local_clarity(img, regions, strength=0.30, radius=5)

        # All four corner blocks must be byte-identical to the input.
        assert np.array_equal(result[0:4, 0:4], img[0:4, 0:4])
        assert np.array_equal(result[0:4, 60:64], img[0:4, 60:64])
        assert np.array_equal(result[60:64, 0:4], img[60:64, 0:4])
        assert np.array_equal(result[60:64, 60:64], img[60:64, 60:64])

    def test_local_clarity_all_regions_none_is_noop(self, proc):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        img[20:30, 20:30] = 240
        regions = _FakeLocalClarityRegions()
        result = proc.local_clarity(img, regions, strength=0.30, radius=5)
        assert np.array_equal(result, img)


class TestBuildDimensionalMask:
    def test_all_none_returns_zeros(self):
        class FakeRegions:
            nose_bridge = None
            cheek_highlights_l = None
            cheek_highlights_r = None
        mask = SkinProcessor._build_dimensional_mask(FakeRegions())
        assert mask.shape == (200, 200)
        assert mask.dtype == np.float32
        assert mask.max() == 0.0

    def test_single_attr_uses_its_shape(self):
        class FakeRegions:
            nose_bridge = np.ones((50, 70), dtype=np.float32) * 0.4
            cheek_highlights_l = None
            cheek_highlights_r = None
        mask = SkinProcessor._build_dimensional_mask(FakeRegions())
        assert mask.shape == (50, 70)
        assert mask.dtype == np.float32
        assert mask.max() == pytest.approx(0.4)

    def test_union_clips_to_one(self):
        class FakeRegions:
            nose_bridge = np.ones((10, 10), dtype=np.float32) * 0.7
            cheek_highlights_l = np.ones((10, 10), dtype=np.float32) * 0.7
            cheek_highlights_r = None
        mask = SkinProcessor._build_dimensional_mask(FakeRegions())
        assert mask.shape == (10, 10)
        assert mask.max() == pytest.approx(1.0)

    def test_custom_attrs(self):
        class FakeRegions:
            nose_bridge = None
            cheek_highlights_l = None
            cheek_highlights_r = None
            forehead_center = np.ones((8, 8), dtype=np.float32) * 0.3
        mask = SkinProcessor._build_dimensional_mask(
            FakeRegions(),
            attrs=("forehead_center",),
        )
        assert mask.shape == (8, 8)
        assert mask.max() == pytest.approx(0.3)

    def test_feather_zero_is_identity(self):
        class FakeRegions:
            nose_bridge = np.ones((20, 20), dtype=np.float32) * 0.5
            cheek_highlights_l = None
            cheek_highlights_r = None
        mask = SkinProcessor._build_dimensional_mask(FakeRegions(), feather=0)
        assert mask.max() == pytest.approx(0.5)

    def test_feather_smooths_peak(self):
        class FakeRegions:
            nose_bridge = np.zeros((40, 40), dtype=np.float32)
            nose_bridge[18:22, 18:22] = 1.0
            cheek_highlights_l = None
            cheek_highlights_r = None
        mask_plain = SkinProcessor._build_dimensional_mask(FakeRegions(), feather=0)
        mask_feather = SkinProcessor._build_dimensional_mask(FakeRegions(), feather=5)
        assert mask_plain.max() == pytest.approx(1.0)
        assert mask_feather.max() < 1.0
        assert mask_feather.max() > 0.0

    def test_uint8_attr_is_promoted(self):
        class FakeRegions:
            nose_bridge = (np.ones((5, 5), dtype=np.uint8) * 200)
            cheek_highlights_l = None
            cheek_highlights_r = None
        mask = SkinProcessor._build_dimensional_mask(FakeRegions())
        assert mask.dtype == np.float32
        assert mask.max() > 0.0

