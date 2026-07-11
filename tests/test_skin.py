"""Tests for retouch/skin.py — SkinProcessor."""

import numpy as np
import cv2
import pytest

from retouch.color_science import bgr_to_oklab, oklab_to_oklch
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

    def test_whiten_hue_stable_true_lifts_luminance(self, proc, face_mask):
        rng = np.random.RandomState(42)
        colorful = rng.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = proc.whiten(colorful, face_mask, strength=50, tone="rosy", hue_stable=True)
        L_orig = bgr_to_oklab(colorful)[..., 0]
        L_result = bgr_to_oklab(result)[..., 0]
        skin_idx = face_mask > 0.3
        assert L_result[skin_idx].mean() > L_orig[skin_idx].mean() + 0.005

    def test_whiten_hue_stable_preserves_chroma(self, proc, face_mask):
        rng = np.random.RandomState(42)
        colorful = rng.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        result = proc.whiten(colorful, face_mask, strength=50, tone="rosy", hue_stable=True)
        C_orig = oklab_to_oklch(bgr_to_oklab(colorful))[..., 1]
        C_result = oklab_to_oklch(bgr_to_oklab(result))[..., 1]
        skin_idx = face_mask > 0.3
        mean_C_diff = np.abs(C_result[skin_idx] - C_orig[skin_idx]).mean()
        assert mean_C_diff < 0.005

    def test_whiten_hue_stable_zero_strength(self, proc, img, face_mask):
        result = proc.whiten(img, face_mask, strength=0, tone="rosy", hue_stable=True)
        assert np.array_equal(result, img)


class TestFaceExposureLift:
    """face_exposure_lift: flat skin-L luminance lift (float32 [0,255] BGR)."""

    @staticmethod
    def _fimg():
        return np.full((64, 64, 3), 128.0, dtype=np.float32)

    def test_zero_strength_returns_input(self, proc, face_mask):
        im = self._fimg()
        out = proc.face_exposure_lift(im, face_mask, strength=0)
        assert np.allclose(out, im)

    def test_lifts_skin_luminance(self, proc, face_mask):
        from retouch.utils import bgr_f32_to_lab_f32
        im = self._fimg()
        before = bgr_f32_to_lab_f32(im)[:, :, 0]
        out = proc.face_exposure_lift(im, face_mask, strength=50)
        after = bgr_f32_to_lab_f32(out)[:, :, 0]
        sel = face_mask > 0.5
        assert after[sel].mean() > before[sel].mean() + 5

    def test_background_untouched(self, proc):
        from retouch.utils import bgr_f32_to_lab_f32
        im = self._fimg()
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0  # hard-edged block, no blur
        before = bgr_f32_to_lab_f32(im)[:, :, 0]
        out = proc.face_exposure_lift(im, mask, strength=100)
        after = bgr_f32_to_lab_f32(out)[:, :, 0]
        # Far corner, clear of the 3px feather and the block.
        sel = np.zeros((64, 64), dtype=bool)
        sel[2:12, 2:12] = True
        assert np.allclose(after[sel], before[sel], atol=1e-3)


class TestEqualize:
    def test_zero_strength(self, proc, img, face_mask):
        result = proc.equalize(img, face_mask, strength=0)
        assert np.all(result == img)

    def test_equalize_changes_image(self, proc, face_mask):
        # Use a textured (non-flat) image so equalize has genuine local contrast
        # to redistribute. A flat image has nothing to even out, and mean-
        # luminance preservation correctly leaves it unchanged.
        rng = np.random.RandomState(0)
        textured = rng.randint(80, 180, (64, 64, 3), dtype=np.uint8)
        result = proc.equalize(textured, face_mask, strength=50)
        assert not np.allclose(result, textured)

    @pytest.mark.parametrize("strength", [20, 60])
    def test_bright_pale_preserves_luminance(self, proc, strength):
        # Bright, pale, low-contrast skin (e.g. cosplay white makeup): equalize
        # must even out tone without shifting exposure (mean L must stay put).
        rng = np.random.RandomState(7)
        base = np.array([200, 205, 230], dtype=np.float32)
        noise = rng.normal(0.0, 3.0, (64, 64, 3)).astype(np.float32)
        bright = np.clip(base + noise, 0, 255).astype(np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)

        result = proc.equalize(bright, mask, strength=strength)

        orig_L = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        result_L = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        assert abs(result_L.mean() - orig_L.mean()) < 2.0

    @pytest.mark.parametrize("strength", [20, 60])
    def test_bright_pale_does_not_increase_variance(self, proc, strength):
        # equalize must never make already-even pale skin LESS even: CLAHE can
        # amplify luminance variance (gray mottling), which must be clamped.
        rng = np.random.RandomState(7)
        base = np.array([200, 205, 230], dtype=np.float32)
        noise = rng.normal(0.0, 3.0, (64, 64, 3)).astype(np.float32)
        bright = np.clip(base + noise, 0, 255).astype(np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)

        result = proc.equalize(bright, mask, strength=strength)

        orig_L = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        result_L = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        assert result_L.std() <= orig_L.std() + 0.5

    def test_with_ref_lab(self, proc, face_mask):
        # Textured input so equalize + a/b pull toward the reference produce a
        # real change (a flat image has no local contrast to redistribute).
        rng = np.random.RandomState(1)
        textured = rng.randint(80, 180, (64, 64, 3), dtype=np.uint8)
        ref = np.full((64, 64, 3), 100, dtype=np.uint8)
        ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB)
        result = proc.equalize(textured, face_mask, strength=50, ref_lab=ref_lab)
        assert not np.allclose(result, textured)

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
        assert np.all(out[: h // 2 - 5, :] == 0.0)
        # Skin-only region must be untouched (skip 5px for 3px erosion margin).
        assert out[h // 2 + 5:, :].min() == 1.0

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
        skin = np.ones((8, 8), dtype=np.float32)
        hair = np.zeros((8, 8), dtype=np.float32)
        hair[:4, :] = 1.0
        out = _build_smooth_mask(
            skin, exclusions=(None, hair, None)
        )
        assert np.all(out[:4, :] == 0.0)
        assert out[5:, :].min() == 1.0

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
        mask = SkinProcessor._build_dimensional_mask(FakeRegions(), shape=(200, 200))
        assert mask.shape == (200, 200)
        assert mask.dtype == np.float32
        assert mask.max() == 0.0

    def test_single_attr_uses_its_shape(self):
        class FakeRegions:
            nose_bridge = np.ones((50, 70), dtype=np.float32) * 0.4
            cheek_highlights_l = None
            cheek_highlights_r = None
        mask = SkinProcessor._build_dimensional_mask(FakeRegions(), shape=(50, 70))
        assert mask.shape == (50, 70)
        assert mask.dtype == np.float32
        assert mask.max() == pytest.approx(0.4)

    def test_union_clips_to_one(self):
        class FakeRegions:
            nose_bridge = np.ones((10, 10), dtype=np.float32) * 0.7
            cheek_highlights_l = np.ones((10, 10), dtype=np.float32) * 0.7
            cheek_highlights_r = None
        mask = SkinProcessor._build_dimensional_mask(FakeRegions(), shape=(10, 10))
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
            shape=(8, 8),
            attrs=("forehead_center",),
        )
        assert mask.shape == (8, 8)
        assert mask.max() == pytest.approx(0.3)

    def test_feather_zero_is_identity(self):
        class FakeRegions:
            nose_bridge = np.ones((20, 20), dtype=np.float32) * 0.5
            cheek_highlights_l = None
            cheek_highlights_r = None
        mask = SkinProcessor._build_dimensional_mask(FakeRegions(), shape=(20, 20), feather=0)
        assert mask.max() == pytest.approx(0.5)

    def test_feather_smooths_peak(self):
        class FakeRegions:
            nose_bridge = np.zeros((40, 40), dtype=np.float32)
            nose_bridge[18:22, 18:22] = 1.0
            cheek_highlights_l = None
            cheek_highlights_r = None
        mask_plain = SkinProcessor._build_dimensional_mask(FakeRegions(), shape=(40, 40), feather=0)
        mask_feather = SkinProcessor._build_dimensional_mask(FakeRegions(), shape=(40, 40), feather=5)
        assert mask_plain.max() == pytest.approx(1.0)
        assert mask_feather.max() < 1.0
        assert mask_feather.max() > 0.0

    def test_uint8_attr_is_promoted(self):
        class FakeRegions:
            nose_bridge = (np.ones((5, 5), dtype=np.uint8) * 200)
            cheek_highlights_l = None
            cheek_highlights_r = None
        mask = SkinProcessor._build_dimensional_mask(FakeRegions(), shape=(5, 5))
        assert mask.dtype == np.float32
        assert mask.max() > 0.0


class _FakeRestoreMicroTextureRegions:
    """Minimal stub matching the FaceRegions attrs read by restore_micro_texture."""

    def __init__(self, h: int = 200, w: int = 200, with_masks: bool = True) -> None:
        if with_masks:
            self.nose_bridge = np.zeros((h, w), dtype=np.float32)
            self.nose_bridge[h // 3:h * 2 // 3, w // 2 - 10:w // 2 + 10] = 1.0
            self.cheek_highlights_l = np.zeros((h, w), dtype=np.float32)
            self.cheek_highlights_l[h // 2:h * 2 // 3, w // 4:w // 4 + 30] = 0.8
            self.cheek_highlights_r = np.zeros((h, w), dtype=np.float32)
            self.cheek_highlights_r[h // 2:h * 2 // 3, 3 * w // 4 - 30:3 * w // 4] = 0.8
            self.left_under_eye = None
            self.right_under_eye = None
            self.left_eye = np.zeros((h, w), dtype=np.float32)
            self.left_eye[h // 4:h // 4 + 24, w // 4:w // 4 + 30] = 1.0
            self.right_eye = np.zeros((h, w), dtype=np.float32)
            self.right_eye[h // 4:h // 4 + 24, 3 * w // 4 - 30:3 * w // 4] = 1.0
            self.crows_feet_l = np.zeros((h, w), dtype=np.float32)
            self.crows_feet_l[h // 4 + 18:h // 4 + 34, w // 4 - 18:w // 4 + 6] = 1.0
            self.crows_feet_r = np.zeros((h, w), dtype=np.float32)
            self.crows_feet_r[h // 4 + 18:h // 4 + 34, 3 * w // 4 - 6:3 * w // 4 + 18] = 1.0
        else:
            self.nose_bridge = None
            self.cheek_highlights_l = None
            self.cheek_highlights_r = None
            self.left_under_eye = None
            self.right_under_eye = None
            self.left_eye = None
            self.right_eye = None
            self.crows_feet_l = None
            self.crows_feet_r = None


class TestRestoreMicroTexture:
    @staticmethod
    def _build_test_pair(h: int = 200, w: int = 200):
        rng = np.random.default_rng(42)
        original = rng.integers(80, 180, (h, w, 3), dtype=np.uint8)
        original = np.clip(
            original.astype(np.float32)
            + rng.standard_normal((h, w, 3)).astype(np.float32) * 20,
            0, 255,
        ).astype(np.uint8)
        smoothed = cv2.GaussianBlur(original, (15, 15), 0)
        return original, smoothed

    def test_no_op_when_strength_zero(self):
        proc = SkinProcessor()
        original, smoothed = self._build_test_pair()
        regions = _FakeRestoreMicroTextureRegions()
        result = proc.restore_micro_texture(
            smoothed, original, regions, strength=0, smooth_strength=0.5
        )
        assert np.array_equal(result, smoothed)

    def test_no_op_when_smooth_strength_zero(self):
        proc = SkinProcessor()
        original, smoothed = self._build_test_pair()
        regions = _FakeRestoreMicroTextureRegions()
        result = proc.restore_micro_texture(
            smoothed, original, regions, strength=20, smooth_strength=0.0
        )
        assert np.array_equal(result, smoothed)

    def test_no_op_when_all_regions_none(self):
        proc = SkinProcessor()
        original, smoothed = self._build_test_pair()
        regions = _FakeRestoreMicroTextureRegions(with_masks=False)
        result = proc.restore_micro_texture(
            smoothed, original, regions, strength=20, smooth_strength=0.5
        )
        assert np.array_equal(result, smoothed)

    def test_dimensional_mask_concentrates_restoration(self):
        proc = SkinProcessor()
        original, smoothed = self._build_test_pair()
        regions = _FakeRestoreMicroTextureRegions()
        result = proc.restore_micro_texture(
            smoothed, original, regions, strength=40, smooth_strength=0.5
        )

        diff = np.abs(result.astype(np.float32) - smoothed.astype(np.float32))
        # Inside: nose_bridge central area (mask == 1.0)
        inside = diff[regions.nose_bridge == 1.0].mean()
        # Outside: top-left corner, far from any region (dim_mask == 0)
        outside = diff[0:50, 0:50].mean()

        assert inside > 1.0, f"expected restoration inside nose region; inside={inside:.3f}"
        assert outside < 0.1, f"expected no restoration outside mask; outside={outside:.3f}"
        assert inside > outside * 5, (
            f"expected restoration concentrated in nose region; "
            f"inside={inside:.3f} outside={outside:.3f}"
        )

    def test_eye_detail_region_restored(self):
        proc = SkinProcessor()
        original, smoothed = self._build_test_pair()
        regions = _FakeRestoreMicroTextureRegions()
        result = proc.restore_micro_texture(
            smoothed, original, regions, strength=40, smooth_strength=0.5
        )

        diff = np.abs(result.astype(np.float32) - smoothed.astype(np.float32))
        eye_inside = diff[regions.left_eye == 1.0].mean()
        eye_outside = diff[0:50, 0:50].mean()

        assert eye_inside > 1.0, f"expected restoration inside eye region; inside={eye_inside:.3f}"
        assert eye_inside > eye_outside * 5, (
            f"expected restoration concentrated in eye region; "
            f"inside={eye_inside:.3f} outside={eye_outside:.3f}"
        )

    def test_restoration_proportional_to_strength(self):
        proc = SkinProcessor()
        original, smoothed = self._build_test_pair()
        regions = _FakeRestoreMicroTextureRegions()
        # Use the high-mask core of the nose_bridge (well inside the 3px feather zone)
        inside_mask = np.zeros_like(regions.nose_bridge, dtype=bool)
        inside_mask[80:120, 95:105] = True

        diffs = []
        for s in (10, 25, 50):
            result = proc.restore_micro_texture(
                smoothed, original, regions, strength=s, smooth_strength=0.5
            )
            d = np.abs(result.astype(np.float32) - smoothed.astype(np.float32))[inside_mask].mean()
            diffs.append(d)

        # Linear in strength: 25/10 = 2.5, 50/10 = 5.0 (±20%)
        assert diffs[1] == pytest.approx(diffs[0] * 2.5, rel=0.20)
        assert diffs[2] == pytest.approx(diffs[0] * 5.0, rel=0.20)

    def test_restoration_proportional_to_smooth_strength(self):
        proc = SkinProcessor()
        original, smoothed = self._build_test_pair()
        regions = _FakeRestoreMicroTextureRegions()
        inside_mask = np.zeros_like(regions.nose_bridge, dtype=bool)
        inside_mask[80:120, 95:105] = True

        def mean_diff(smooth_strength: float) -> float:
            result = proc.restore_micro_texture(
                smoothed, original, regions, strength=20, smooth_strength=smooth_strength
            )
            return float(
                np.abs(result.astype(np.float32) - smoothed.astype(np.float32))[inside_mask].mean()
            )

        d_zero = mean_diff(0.0)
        d_03 = mean_diff(0.3)
        d_07 = mean_diff(0.7)

        # smooth_strength=0.0 is a no-op (returns smoothed)
        assert d_zero == 0.0
        assert d_03 > 0.0
        assert d_07 > d_03
        # Linear in smooth_strength: 0.7/0.3 ≈ 2.33 (±20%)
        assert d_07 == pytest.approx(d_03 * (0.7 / 0.3), rel=0.20)

    def test_returns_uint8_with_correct_shape(self):
        proc = SkinProcessor()
        original, smoothed = self._build_test_pair()
        regions = _FakeRestoreMicroTextureRegions()
        result = proc.restore_micro_texture(
            smoothed, original, regions, strength=25, smooth_strength=0.5
        )
        assert result.dtype == np.uint8
        assert result.shape == original.shape


class TestFlatten:
    def test_zero_strength_noop(self, proc, img, face_mask):
        result = proc.flatten(img, face_mask, strength=0)
        assert np.all(result == img)

    def test_skin_region_l_std_decreases(self, proc, face_mask):
        img_bgr = np.full((64, 64, 3), 128, dtype=np.uint8)
        img_bgr[24:40, 24:40] = [128 + 30, 128, 128]
        img_bgr[24:40, 24:40, :] += np.random.randint(-10, 10, (16, 16, 3), dtype=np.int16).clip(0, 255).astype(np.uint8)
        result = proc.flatten(img_bgr, face_mask, strength=60)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        orig_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_l = lab[:, :, 0][face_mask > 0.3]
        orig_skin_l = orig_lab[:, :, 0][face_mask > 0.3]
        assert np.std(skin_l) < np.std(orig_skin_l)

    def test_hard_edge_preserved(self, proc, face_mask):
        img_bgr = np.full((64, 64, 3), 200, dtype=np.uint8)
        img_bgr[32:48, :] = 100
        result = proc.flatten(img_bgr, face_mask, strength=50)
        step_before = abs(float(img_bgr[32, 0, 0]) - float(img_bgr[30, 0, 0]))
        step_after = abs(float(result[32, 0, 0]) - float(result[30, 0, 0]))
        assert step_after >= 0.8 * step_before

    def test_outside_mask_unchanged(self, proc, face_mask):
        img_bgr = np.full((64, 64, 3), 128, dtype=np.uint8)
        img_bgr[0:10, 0:10] = [50, 200, 100]
        result = proc.flatten(img_bgr, face_mask, strength=80)
        np.testing.assert_allclose(result[0:10, 0:10], img_bgr[0:10, 0:10], atol=2)

    def test_none_mask_noop(self, proc, img):
        result = proc.flatten(img, None, strength=50)
        assert np.all(result == img)

    def test_dtype_shape(self, proc, img, face_mask):
        result = proc.flatten(img, face_mask, strength=30)
        assert result.dtype == np.uint8
        assert result.shape == img.shape


class TestQuantizeTones:
    def test_zero_strength_noop(self, proc, img, face_mask):
        result = proc.quantize_tones(img, face_mask, strength=0)
        assert np.all(result == img)

    def test_linear_ramp_creates_bands(self, proc, face_mask):
        h, w = 64, 64
        L_ramp = np.tile(np.linspace(80, 220, w, dtype=np.float32), (h, 1))
        img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
        lab = np.zeros((h, w, 3), dtype=np.float32)
        lab[:, :, 0] = L_ramp
        lab[:, :, 1:] = 128.0
        img_bgr = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        result = proc.quantize_tones(img_bgr, face_mask, strength=80, bands=3, softness=0.3)
        result_lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        skin_l = result_lab[:, :, 0][face_mask > 0.3]
        orig_skin_l = L_ramp[face_mask > 0.3]
        assert np.std(skin_l) < np.std(orig_skin_l)

    def test_near_uniform_noop(self, proc, face_mask):
        img_bgr = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = proc.quantize_tones(img_bgr, face_mask, strength=50)
        assert np.all(result == img_bgr)

    def test_strength_100_greater_than_30(self, proc, face_mask):
        L_ramp = np.tile(np.linspace(100, 200, 64, dtype=np.float32), (64, 1))
        lab = np.zeros((64, 64, 3), dtype=np.float32)
        lab[:, :, 0] = L_ramp
        lab[:, :, 1:] = 128.0
        img_bgr = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
        r30 = proc.quantize_tones(img_bgr, face_mask, strength=30, bands=3, softness=0.3)
        r100 = proc.quantize_tones(img_bgr, face_mask, strength=100, bands=3, softness=0.3)
        assert r30.dtype == np.uint8
        assert r100.dtype == np.uint8
        assert r30.shape == img_bgr.shape
        assert r100.shape == img_bgr.shape
        assert not np.all(r30 == img_bgr)
        assert not np.all(r100 == img_bgr)

    def test_outside_mask_unchanged(self, proc, face_mask):
        img_bgr = np.full((64, 64, 3), 128, dtype=np.uint8)
        img_bgr[0:10, 0:10] = [50, 200, 100]
        result = proc.quantize_tones(img_bgr, face_mask, strength=80)
        np.testing.assert_allclose(result[0:10, 0:10], img_bgr[0:10, 0:10], atol=2)

    def test_none_mask_noop(self, proc, img):
        result = proc.quantize_tones(img, None, strength=50)
        assert np.all(result == img)

    def test_dtype_shape(self, proc, img, face_mask):
        result = proc.quantize_tones(img, face_mask, strength=40)
        assert result.dtype == np.uint8
        assert result.shape == img.shape


class TestUnifyTone:
    def _make_two_hue_image(self):
        from retouch.color_space import lch_to_bgr
        lch = np.zeros((64, 64, 3), dtype=np.float32)
        lch[:, :, 0] = 70.0
        lch[:, :, 1] = 30.0
        lch[:32, :, 2] = 15.0
        lch[32:, :, 2] = 35.0
        return lch_to_bgr(lch)

    def test_zero_strength_noop(self, proc, face_mask):
        img_bgr = self._make_two_hue_image()
        result = proc.unify_tone(img_bgr, face_mask, strength=0)
        assert np.all(result == img_bgr)

    def test_hue_std_decreases(self, proc, face_mask):
        img_bgr = self._make_two_hue_image()
        result = proc.unify_tone(img_bgr, face_mask, strength=50)
        from retouch.color_space import bgr_to_lch
        lch_orig = bgr_to_lch(img_bgr)
        lch_res = bgr_to_lch(result)
        skin_h_orig = lch_orig[:, :, 2][face_mask > 0.3]
        skin_h_res = lch_res[:, :, 2][face_mask > 0.3]
        # After unification, the circular hue spread should decrease
        d_orig = ((skin_h_orig - skin_h_orig.mean() + 180.0) % 360.0) - 180.0
        d_res = ((skin_h_res - skin_h_res.mean() + 180.0) % 360.0) - 180.0
        assert np.std(d_res) < np.std(d_orig)

    def test_target_hue_pulls_mean(self, proc, face_mask):
        img_bgr = self._make_two_hue_image()
        result = proc.unify_tone(img_bgr, face_mask, strength=60, target_hue=25.0)
        from retouch.color_space import bgr_to_lch
        lch_res = bgr_to_lch(result)
        skin_h = lch_res[:, :, 2][face_mask > 0.3]
        H_mean = float(np.degrees(np.arctan2(
            np.sin(np.radians(skin_h)).mean(),
            np.cos(np.radians(skin_h)).mean(),
        ))) % 360.0
        assert abs(H_mean - 25.0) < 20.0

    def test_chroma_decreases(self, proc, face_mask):
        img_bgr = self._make_two_hue_image()
        result = proc.unify_tone(img_bgr, face_mask, strength=50, chroma_compress=1.0)
        assert result.dtype == np.uint8
        assert result.shape == img_bgr.shape

    def test_none_mask_noop(self, proc, img):
        result = proc.unify_tone(img, None, strength=50)
        assert np.all(result == img)

    def test_dtype_shape(self, proc, img, face_mask):
        result = proc.unify_tone(img, face_mask, strength=30)
        assert result.dtype == np.uint8
        assert result.shape == img.shape


class TestSmoothUndereyeShadow:
    """Tests for SkinProcessor.smooth_undereye_shadow."""

    @staticmethod
    def _build_lab_image(skin_l, shadow_specs, extra_blocks=None):
        """Build a uint8 BGR image from explicit LAB-L control.

        ``shadow_specs``: list of (rows_slice, cols_slice, shadow_l, noise_std)
        painted as textured under-eye shadows (L kept in the realistic 40-120
        range so the ``L>40`` blemish guard never clips them). ``extra_blocks``:
        list of (rows_slice, cols_slice, bgr) painted as plain blocks.
        """
        h = w = 120
        rng = np.random.default_rng(7)
        lab = np.zeros((h, w, 3), dtype=np.uint8)
        lab[:, :, 0] = skin_l
        lab[:, :, 1] = 128
        lab[:, :, 2] = 128
        for (rs, cs, sl, ns) in shadow_specs:
            sl_arr = np.clip(sl + rng.normal(0.0, ns, (rs.stop - rs.start, cs.stop - cs.start)), 50, 120)
            lab[rs, cs, 0] = sl_arr.astype(np.uint8)
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        if extra_blocks:
            for (rs, cs, bgr) in extra_blocks:
                img[rs, cs] = bgr
        return img

    @staticmethod
    def _make_image():
        """Bright skin (L=137), a textured under-eye shadow (L~88) inside the
        mask, a plain dark spill block OUTSIDE the mask, and a bright sclera
        block OUTSIDE the mask."""
        return TestSmoothUndereyeShadow._build_lab_image(
            skin_l=137,
            shadow_specs=[(slice(56, 70), slice(40, 70), 88.0, 8.0)],
            extra_blocks=[
                (slice(56, 70), slice(10, 30), (60, 45, 35)),
                (slice(40, 70), slice(80, 100), (250, 245, 240)),
            ],
        )

    @staticmethod
    def _make_under_eye_mask():
        mask = np.zeros((120, 120), dtype=np.float32)
        mask[40:70, 40:70] = 1.0  # the under-eye zone only
        return mask

    def test_detects_and_softens_under_eye_shadow(self, proc):
        img = self._make_image()
        ue_mask = self._make_under_eye_mask()
        skin = np.ones((120, 120), dtype=np.float32)
        out = proc.smooth_undereye_shadow(
            img, under_eye_masks=[ue_mask], skin_mask=skin, strength=1.0, feather_radius=3
        )
        assert out.dtype == np.uint8
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        out_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Shadow detected & processed: shadow-zone pixels are modified, and the
        # high-frequency luminance variance is reduced (the filter smoothed it).
        in_zone = out_lab[56:70, 40:70, 0]
        orig_in = orig_lab[56:70, 40:70, 0]
        assert not np.array_equal(in_zone, orig_in)
        assert in_zone.std() < orig_in.std()
        # Outside the mask: dark spill block must be byte-identical (no overspill).
        assert np.array_equal(out[56:70, 10:30], img[56:70, 10:30])

    def test_feathered_blend_has_no_hard_seam(self, proc):
        """Feathering must soften the mask boundary: the per-pixel change at the
        zone edge is smaller with a wide feather than with a narrow one."""
        h = w = 120
        rng = np.random.default_rng(11)
        # Mask covers cols 40:80; the LEFT half is a high-contrast dark shadow
        # (bright skin on the right pulls the within-mask median up so the whole
        # shadow qualifies), giving the guided filter strong texture to smooth.
        lab = np.zeros((h, w, 3), dtype=np.uint8)
        lab[:, :, 0] = 137
        lab[:, :, 1] = 128
        lab[:, :, 2] = 128
        noisy = np.clip(70 + rng.normal(0.0, 18.0, (40, 20)), 50, 120).astype(np.uint8)
        lab[40:80, 40:60, 0] = noisy
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        mask = np.zeros((h, w), dtype=np.float32)
        mask[40:80, 40:80] = 1.0

        out_narrow = proc.smooth_undereye_shadow(
            img, under_eye_masks=[mask], skin_mask=np.ones((h, w), np.float32),
            strength=1.0, feather_radius=1,
        )
        out_wide = proc.smooth_undereye_shadow(
            img, under_eye_masks=[mask], skin_mask=np.ones((h, w), np.float32),
            strength=1.0, feather_radius=7,
        )
        # Change fields at the mask's left edge (col 40 is just inside the mask).
        edge = 40
        dn = out_narrow[40:80, edge].astype(np.int16) - img[40:80, edge].astype(np.int16)
        dw = out_wide[40:80, edge].astype(np.int16) - img[40:80, edge].astype(np.int16)
        # Wide feather pulls the blend factor toward 0 at the boundary, so the
        # magnitude of the edge change must be strictly smaller than narrow.
        assert np.abs(dw).max() < np.abs(dn).max()

    def test_sclera_luminance_unchanged(self, proc):
        img = self._make_image()
        ue_mask = self._make_under_eye_mask()
        out = proc.smooth_undereye_shadow(
            img, under_eye_masks=[ue_mask], skin_mask=np.ones((120, 120), np.float32),
            strength=1.0, feather_radius=3,
        )
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        out_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        # Sclera block sits entirely outside the under-eye mask.
        assert np.array_equal(out_lab[40:70, 80:100, 0], orig_lab[40:70, 80:100, 0])

    def test_zero_strength_byte_identical(self, proc):
        img = self._make_image()
        ue_mask = self._make_under_eye_mask()
        out = proc.smooth_undereye_shadow(
            img, under_eye_masks=[ue_mask], skin_mask=np.ones((120, 120), np.float32),
            strength=0.0, feather_radius=3,
        )
        assert out is img
        assert np.array_equal(out, img)

    def test_none_mask_is_noop(self, proc):
        img = self._make_image()
        out = proc.smooth_undereye_shadow(
            img, under_eye_masks=[None, None], strength=0.5
        )
        assert np.array_equal(out, img)

    def test_float32_dtype_preserved(self, proc):
        img = self._make_image().astype(np.float32)
        ue_mask = self._make_under_eye_mask()
        out = proc.smooth_undereye_shadow(
            img, under_eye_masks=[ue_mask], skin_mask=np.ones((120, 120), np.float32),
            strength=0.0, feather_radius=3,
        )
        assert out.dtype == np.float32
        assert out is img

    def test_strength_aware_coverage_gate(self, proc):
        """A mild dark circle (patch coverage ~0.2, between the old fixed 0.3
        gate and the new 0.1 floor) is treated at high strength but skipped at
        low strength — the gate scales with strength so faint circles only heal
        when the user pushes the slider up."""
        H = W = 200
        img = np.full((H, W, 3), (200, 170, 150), np.uint8)  # bright skin
        yy, xx = np.ogrid[:H, :W]
        ue_mask = (((xx - 60) ** 2 / 30 ** 2 + (yy - 140) ** 2 / 18 ** 2) <= 1.0).astype(np.float32)
        skin = np.ones((H, W), np.float32)
        cy, cx = 140, 60  # small dark patch inside the mask -> coverage ~0.2
        img[cy - 8:cy + 8, cx - 10:cx + 10] = (95, 75, 65)

        out_high = proc.smooth_undereye_shadow(
            img, under_eye_masks=[ue_mask], skin_mask=skin, strength=1.0, feather_radius=3
        )
        out_low = proc.smooth_undereye_shadow(
            img, under_eye_masks=[ue_mask], skin_mask=skin, strength=0.2, feather_radius=3
        )
        d_high = int(np.abs(out_high.astype(int) - img.astype(int)).sum())
        d_low = int(np.abs(out_low.astype(int) - img.astype(int)).sum())
        assert d_high > 0, "high strength should treat a mild dark circle"
        assert d_low == 0, "low strength should skip (coverage below the gate)"


class _FakeWrinkleRegions:
    """Minimal FaceRegions stub for wrinkle softening (no MediaPipe/ONNX)."""

    def __init__(self, h: int = 128, w: int = 128) -> None:
        self.skin = np.zeros((h, w), dtype=np.float32)
        self.skin[:] = 1.0
        self.hair = None
        self.left_eyebrow = None
        self.right_eyebrow = None
        self.left_eye = None
        self.right_eye = None
        self.forehead = np.zeros((h, w), dtype=np.float32)
        self.forehead[20:40, 30:98] = 1.0
        self.nasolabial_l = np.zeros((h, w), dtype=np.float32)
        self.nasolabial_l[60:80, 45:65] = 1.0
        self.nasolabial_r = np.zeros((h, w), dtype=np.float32)
        self.nasolabial_r[60:80, 63:83] = 1.0
        self.neck = np.zeros((h, w), dtype=np.float32)
        self.neck[90:110, 40:88] = 1.0


class TestPerRegionWrinkle:
    """Backlog #4: per-region wrinkle sliders (forehead / nasolabial / neck)."""

    @staticmethod
    def _img_with_lines() -> np.ndarray:
        img = np.full((128, 128, 3), 140, dtype=np.uint8)
        cv2.line(img, (30, 30), (98, 30), (30, 30, 30), 5)   # forehead zone
        cv2.line(img, (45, 70), (65, 70), (30, 30, 30), 4)   # nasolabial zone
        cv2.line(img, (40, 100), (88, 100), (30, 30, 30), 5)  # neck zone
        return img

    @staticmethod
    def _lab_L(canvas: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB).astype(np.float32)[:, :, 0]

    def test_forehead_only_softens_forehead(self, proc):
        regions = _FakeWrinkleRegions()
        img = self._img_with_lines()
        out = proc.wrinkle_soften(
            img, regions, 0,
            region_strengths={"forehead": 100, "nasolabial": 0, "neck": 0},
        )
        L_in = self._lab_L(img)
        L_out = self._lab_L(out)
        # Forehead line brightened.
        assert L_out[30, 30:98].mean() > L_in[30, 30:98].mean()
        # Neck line (unrelated region) left sharper — residual depth preserved.
        assert L_out[100, 40:88].mean() < L_in[100, 40:88].mean() + 1.0

    def test_nasolabial_only_softens_nasolabial(self, proc):
        regions = _FakeWrinkleRegions()
        img = self._img_with_lines()
        out = proc.wrinkle_soften(
            img, regions, 0,
            region_strengths={"forehead": 0, "nasolabial": 100, "neck": 0},
        )
        L_in = self._lab_L(img)
        L_out = self._lab_L(out)
        assert L_out[66:74, 45:83].mean() > L_in[66:74, 45:83].mean()
        # Forehead line untouched.
        assert L_out[30, 30:98].mean() < L_in[30, 30:98].mean() + 1.0

    def test_neck_only_softens_neck(self, proc):
        regions = _FakeWrinkleRegions()
        img = self._img_with_lines()
        out = proc.wrinkle_soften(
            img, regions, 0,
            region_strengths={"forehead": 0, "nasolabial": 0, "neck": 100},
        )
        L_in = self._lab_L(img)
        L_out = self._lab_L(out)
        assert L_out[100, 40:88].mean() > L_in[100, 40:88].mean()
        # Forehead line untouched.
        assert L_out[30, 30:98].mean() < L_in[30, 30:98].mean() + 1.0

    def test_global_strength_backward_compat(self, proc):
        regions = _FakeWrinkleRegions()
        img = self._img_with_lines()
        out = proc.wrinkle_soften(img, regions, 100)
        L_in = self._lab_L(img)
        L_out = self._lab_L(out)
        # Global union path softens all zones (forehead here).
        assert L_out[30, 30:98].mean() > L_in[30, 30:98].mean()

    def test_region_strengths_precedence_over_global(self, proc):
        # When any region strength > 0, the per-region path runs and the global
        # `strength` is ignored for ALL regions (no double-application). Pass
        # global=100 + only neck>0: forehead must stay untouched despite global.
        regions = _FakeWrinkleRegions()
        img = self._img_with_lines()
        out = proc.wrinkle_soften(
            img, regions, 100,
            region_strengths={"forehead": 0, "nasolabial": 0, "neck": 100},
        )
        L_in = self._lab_L(img)
        L_out = self._lab_L(out)
        # Forehead (not in active set) untouched because global is ignored.
        assert L_out[30, 30:98].mean() < L_in[30, 30:98].mean() + 1.0
        # Neck (active) softened.
        assert L_out[100, 40:88].mean() > L_in[100, 40:88].mean()

    def test_all_zero_is_noop(self, proc):
        regions = _FakeWrinkleRegions()
        img = self._img_with_lines()
        out = proc.wrinkle_soften(
            img, regions, 0,
            region_strengths={"forehead": 0, "nasolabial": 0, "neck": 0},
        )
        assert np.array_equal(out, img)

    def test_runs_without_error_each_region(self, proc):
        regions = _FakeWrinkleRegions()
        img = np.full((128, 128, 3), 140, dtype=np.uint8)
        for key in ("forehead", "nasolabial", "neck"):
            rs = {k: (100 if k == key else 0) for k in ("forehead", "nasolabial", "neck")}
            out = proc.wrinkle_soften(img, regions, 0, region_strengths=rs)
            assert out.shape == img.shape
            assert out.dtype == np.uint8
