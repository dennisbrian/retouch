"""Tests for retouch/eyes.py — EyeEnhancer."""
import cv2
import numpy as np
import pytest
from retouch.eyes import EyeEnhancer


class MockFaceRegions:
    pass


@pytest.fixture
def enhancer():
    return EyeEnhancer()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def regions():
    r = MockFaceRegions()
    r.left_eye = np.zeros((64, 64), dtype=np.float32)
    r.right_eye = np.zeros((64, 64), dtype=np.float32)
    r.left_iris = np.zeros((64, 64), dtype=np.float32)
    r.right_iris = np.zeros((64, 64), dtype=np.float32)
    r.left_eye[20:30, 20:30] = 1.0
    r.right_eye[20:30, 34:44] = 1.0
    r.left_iris[24:27, 24:27] = 1.0
    r.right_iris[24:27, 37:40] = 1.0
    return r


class TestEnhance:
    def test_zero_strength(self, enhancer, img, regions):
        result = enhancer.enhance(img, regions, strength=0)
        assert np.all(result == img)

    def test_output_shape(self, enhancer, img, regions):
        result = enhancer.enhance(img, regions, strength=50)
        assert result.shape == (64, 64, 3)
        assert result.dtype == np.uint8

    def test_changes_image(self, enhancer, img, regions):
        result = enhancer.enhance(img, regions, strength=80)
        assert not np.allclose(result, img)


class TestEnhanceWhites:
    def test_none_mask(self, enhancer, img):
        result = enhancer._enhance_whites(img, None, 0.5)
        assert np.all(result == img)

    def test_empty_mask(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer._enhance_whites(img, mask, 0.5)
        assert np.all(result == img)

    def test_bright_sclera_only(self, enhancer, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._enhance_whites(img, mask, 0.5)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        # The a channel should be reduced (less red) in bright areas
        assert lab[:, :, 1].mean() <= 128

    def test_output_type(self, enhancer, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._enhance_whites(img, mask, 1.0)
        assert result.dtype == np.uint8

    def test_dark_and_bright_sclera_receive_equal_chroma_correction(self, enhancer):
        """Whitening must not depend on an absolute image-lightness gate."""
        def make_sclera(l_value):
            lab = np.full((48, 48, 3), (l_value, 148, 152), dtype=np.uint8)
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        mask = np.ones((48, 48), dtype=np.float32)
        dark = make_sclera(70)
        bright = make_sclera(180)
        dark_out = enhancer._enhance_whites(dark, mask, 0.5)
        bright_out = enhancer._enhance_whites(bright, mask, 0.5)

        def chroma_delta(before, after):
            lab_before = cv2.cvtColor(before, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_after = cv2.cvtColor(after, cv2.COLOR_BGR2LAB).astype(np.float32)
            return float(np.mean(np.abs(lab_before[:, :, 1:] - lab_after[:, :, 1:])))

        dark_delta = chroma_delta(dark, dark_out)
        bright_delta = chroma_delta(bright, bright_out)
        assert dark_delta > 0.0
        assert dark_delta == pytest.approx(bright_delta, abs=0.5)

    def test_small_dark_sclera_ignores_bright_background(self, enhancer):
        """A tiny valid eye mask must not inherit its reference from the frame."""
        lab = np.full((48, 48, 3), (230, 128, 128), dtype=np.uint8)
        lab[20:23, 20:23] = (70, 148, 152)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        mask = np.zeros((48, 48), dtype=np.float32)
        mask[20:23, 20:23] = 1.0

        out = enhancer._enhance_whites(image, mask, 0.5)
        before_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        after_lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        assert np.mean(np.abs(before_lab[20:23, 20:23, 1:] - after_lab[20:23, 20:23, 1:])) > 0.0
        assert np.all(out[0, 0] == image[0, 0])

    def test_sclera_targets_own_bright_and_chroma_reference(self, enhancer):
        """AA5: improve yellow/dark pixels without manufacturing display white."""
        lab = np.full((48, 48, 3), (150, 132, 142), dtype=np.uint8)
        lab[12:24, 12:24] = (105, 145, 170)  # dark, red/yellow sclera patch
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        mask = np.ones((48, 48), dtype=np.float32)

        out = enhancer._enhance_whites(image, mask, 1.0)
        before = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        after = cv2.cvtColor(out, cv2.COLOR_BGR2LAB).astype(np.float32)
        patch = np.s_[12:24, 12:24]

        assert after[patch][:, :, 0].mean() > before[patch][:, :, 0].mean()
        assert after[patch][:, :, 2].mean() < before[patch][:, :, 2].mean()
        assert after[:, :, 0].max() <= np.percentile(before[:, :, 0], 90) + 1.0


class TestSculptIris:
    def test_none_mask(self, enhancer, img):
        result = enhancer._sculpt_iris(img, None, 0.5)
        assert np.all(result == img)

    def test_empty_mask(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer._sculpt_iris(img, mask, 0.5)
        assert np.all(result == img)

    def test_small_iris_returns_original(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[32, 32] = 1.0
        result = enhancer._sculpt_iris(img, mask, 0.5)
        assert np.all(result == img)

    def test_changes_image(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0
        result = enhancer._sculpt_iris(img, mask, 0.5)
        assert not np.allclose(result, img)

    def test_output_shape(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[20:44, 20:44] = 1.0
        result = enhancer._sculpt_iris(img, mask, 0.5)
        assert result.shape == (64, 64, 3)


class TestEnhanceCatchlights:
    def test_none_mask(self, enhancer, img):
        result = enhancer._enhance_catchlights(img, None, 0.5)
        assert np.all(result == img)

    def test_empty_mask(self, enhancer, img):
        mask = np.zeros((64, 64), dtype=np.float32)
        result = enhancer._enhance_catchlights(img, mask, 0.5)
        assert np.all(result == img)

    def test_no_catchlights_returns_original(self, enhancer, img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._enhance_catchlights(img, mask, 0.5)
        assert np.all(result == img)

    def test_with_bright_catchlight(self, enhancer):
        img = np.full((64, 64, 3), 50, dtype=np.uint8)
        img[30:33, 30:33] = 240
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[25:40, 25:40] = 1.0
        result = enhancer._enhance_catchlights(img, mask, 1.0)
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        assert lab[:, :, 0].mean() > orig_lab[:, :, 0].mean()

    def test_output_type(self, enhancer):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        img[30:33, 30:33] = 240
        mask = np.ones((64, 64), dtype=np.float32)
        result = enhancer._enhance_catchlights(img, mask, 0.5)
        assert result.dtype == np.uint8


class TestSpecularBoost:
    """Tests for the +10% specular catchlight boost inside the iris region.

    Uses a control diff between catchlight_strength=0 and catchlight_strength=50
    to isolate the new code's effect from iris sculpting and other processing.
    """

    def _build_regions(self):
        r = MockFaceRegions()
        r.left_eye = np.zeros((64, 64), dtype=np.float32)
        r.right_eye = np.zeros((64, 64), dtype=np.float32)
        r.left_iris = np.zeros((64, 64), dtype=np.float32)
        r.right_iris = np.zeros((64, 64), dtype=np.float32)
        # Eye mask slightly larger than iris so eye-whites don't swallow the iris
        r.left_eye[15:45, 15:45] = 1.0
        r.left_iris[20:40, 20:40] = 1.0
        return r

    def _diff_L(self, enhancer, img, regions, cl_a, cl_b):
        """Return (b - a) LAB L-channel diff map for two catchlight strengths."""
        res_a = enhancer.enhance(img, regions, strength=40, catchlight_strength=cl_a)
        res_b = enhancer.enhance(img, regions, strength=40, catchlight_strength=cl_b)
        la = cv2.cvtColor(res_a, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        lb = cv2.cvtColor(res_b, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
        return lb - la

    def test_specular_boost_only_on_bright_pixels(self, enhancer):
        # Dim background so iris sculpting effects are visible but small
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        # Bright spot (L=240) inside the iris
        img[24:28, 24:28] = 240
        # Dim spot (L=100) inside the same iris
        img[32:36, 32:36] = 100

        regions = self._build_regions()

        # Diff isolates the catchlight-50 → catchlight-0 change (i.e. the new boost)
        diff = self._diff_L(enhancer, img, regions, cl_a=0, cl_b=50)

        bright_diff = float(diff[24:28, 24:28].mean())
        dim_diff = float(diff[32:36, 32:36].mean())

        # Bright spot (L=240) crosses the L>220 threshold → boosted
        assert bright_diff > 1.0, (
            f"Expected bright spot L to increase, got diff={bright_diff:.2f}"
        )
        # Dim spot (L=100) is far below L>220 → untouched by the new code
        assert abs(dim_diff) < 0.5, (
            f"Expected dim spot L unchanged by new boost, got diff={dim_diff:.2f}"
        )

    def test_specular_boost_contained_to_iris(self, enhancer):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        # Bright spot INSIDE the iris
        img[24:28, 24:28] = 240
        # Bright spot OUTSIDE the iris (and outside the eye)
        img[50:54, 50:54] = 240

        regions = self._build_regions()

        diff = self._diff_L(enhancer, img, regions, cl_a=0, cl_b=50)

        inside_diff = float(diff[24:28, 24:28].mean())
        outside_diff = float(diff[50:54, 50:54].mean())

        # Inside iris: bright pixel boosted by the iris-bounded code
        assert inside_diff > 1.0, (
            f"Expected inside-iris bright spot to be boosted, got diff={inside_diff:.2f}"
        )
        # Outside iris: specular boost is masked to iris, so no change
        assert abs(outside_diff) < 0.5, (
            f"Expected outside-iris bright spot unchanged, got diff={outside_diff:.2f}"
        )


def _random_eye_test_image():
    np.random.seed(42)
    img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    regions = MockFaceRegions()
    regions.left_eye = np.zeros((64, 64), dtype=np.float32)
    regions.right_eye = np.zeros((64, 64), dtype=np.float32)
    regions.left_iris = np.zeros((64, 64), dtype=np.float32)
    regions.right_iris = np.zeros((64, 64), dtype=np.float32)
    regions.left_eye[20:30, 20:30] = 1.0
    regions.right_eye[20:30, 34:44] = 1.0
    regions.left_iris[24:27, 24:27] = 1.0
    regions.right_iris[24:27, 37:40] = 1.0
    return img, regions


def test_enhance_strength_zero():
    img, regions = _random_eye_test_image()
    enhancer = EyeEnhancer()
    result = enhancer.enhance(img, regions, strength=0)
    assert np.all(result == img)


def test_enhance_output_dtype():
    img, regions = _random_eye_test_image()
    enhancer = EyeEnhancer()
    result = enhancer.enhance(img, regions, strength=50)
    assert result.dtype == np.uint8


def test_enhance_output_shape():
    img, regions = _random_eye_test_image()
    enhancer = EyeEnhancer()
    result = enhancer.enhance(img, regions, strength=50)
    assert result.shape == (64, 64, 3)


class TestScleraVesselRemoval:
    """Sclera vessel removal: detect red vessels inside the iris-excluded
    eye-white mask, inpaint them, leave iris/pupil/skin untouched."""

    def _build(self):
        img = np.full((64, 64, 3), 225, dtype=np.uint8)  # off-white sclera
        r = MockFaceRegions()
        r.left_eye = np.zeros((64, 64), dtype=np.float32)
        r.right_eye = np.zeros((64, 64), dtype=np.float32)
        r.left_iris = np.zeros((64, 64), dtype=np.float32)
        r.right_iris = np.zeros((64, 64), dtype=np.float32)
        r.left_eye[15:45, 10:45] = 1.0
        r.left_iris[28:33, 18:23] = 1.0  # excluded from sclera mask
        # Red vessel line in sclera (outside iris), column x=37-38
        img[15:45, 37:39] = (60, 60, 210)
        # Red mark INSIDE the iris (must survive)
        img[29:32, 19:22] = (60, 60, 210)
        return img, r

    def _whites(self, r):
        return np.clip(r.left_eye - r.left_iris, 0, 1)

    def test_zero_strength_noop(self, enhancer):
        img, r = self._build()
        out = enhancer._remove_sclera_vessels(img, self._whites(r), 0)
        assert np.all(out == img)

    def test_removes_vessel_redness(self, enhancer):
        img, r = self._build()
        out = enhancer._remove_sclera_vessels(img, self._whites(r), 100)
        orig_a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1]
        new_a = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 1]
        before = float(orig_a[15:45, 37:39].mean())
        after = float(new_a[15:45, 37:39].mean())
        assert after < before, f"vessel redness not reduced: {before:.1f} -> {after:.1f}"

    def test_iris_untouched(self, enhancer):
        img, r = self._build()
        out = enhancer._remove_sclera_vessels(img, self._whites(r), 100)
        assert np.all(out[29:32, 19:22] == img[29:32, 19:22])

    def test_float32_preserved(self, enhancer):
        img, r = self._build()
        out = enhancer._remove_sclera_vessels(img.astype(np.float32), self._whites(r), 100)
        assert out.dtype == np.float32

    def test_enhance_wiring_vessel_only(self, enhancer):
        # eye_enhance=0 but vessel_strength=100 must still run removal
        img, r = self._build()
        out = enhancer.enhance(img, r, strength=0, vessel_strength=100)
        orig_a = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 1]
        new_a = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[:, :, 1]
        assert float(new_a[15:45, 37:39].mean()) < float(orig_a[15:45, 37:39].mean())


class TestLimbalRingCapAndGate:
    """BB1: Limbal-ring target-seeking contrast cap and dark-iris confidence gate."""

    def _build_iris(self, body_l_value: float, limbal_l_value: float):
        # Build 64x64 LAB image with center iris
        lab = np.zeros((64, 64, 3), dtype=np.uint8)
        lab[:, :, 0] = 128
        lab[:, :, 1] = 128
        lab[:, :, 2] = 128

        cy, cx, r = 32.0, 32.0, 16.0
        yy, xx = np.mgrid[0:64, 0:64]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r

        iris_mask = (dist <= 1.0).astype(np.float32)
        body_mask = ((dist >= 0.3) & (dist <= 0.8)).astype(bool)
        limbal_mask = ((dist > 0.8) & (dist <= 1.0)).astype(bool)

        lab[body_mask, 0] = int(round(body_l_value))
        lab[limbal_mask, 0] = int(round(limbal_l_value))

        img_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return img_bgr, iris_mask

    def test_zero_strength_byte_identity(self, enhancer):
        img_bgr, iris_mask = self._build_iris(body_l_value=70, limbal_l_value=60)
        out = enhancer._sculpt_iris(img_bgr, iris_mask, strength=0.0)
        assert np.all(out == img_bgr)

    def test_dark_iris_gated_to_noop(self, enhancer):
        """Very dark irises (body L* < 30) must gate limbal darkening to zero."""
        img_bgr, iris_mask = self._build_iris(body_l_value=22, limbal_l_value=20)
        out = enhancer._sculpt_iris(img_bgr, iris_mask, strength=1.0)
        # Convert back to LAB to inspect limbal ring L*
        lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        
        # Limbal ring region d ~ 0.9 (outer edge of iris mask)
        cy, cx, r = 32.0, 32.0, 16.0
        yy, xx = np.mgrid[0:64, 0:64]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        limbal_annulus = (dist >= 0.85) & (dist <= 0.95)

        # Without gate, subtracting 30 * strength (30) from L*=20 would black out the ring to ~0.
        # With gate, limbal_darken = 0, so L* remains around 20 (allowing minor USM/body edge spill).
        limbal_l_after = float(np.mean(lab_out[limbal_annulus, 0]))
        assert limbal_l_after > 15.0, f"Dark iris limbal ring was blacked out: L*={limbal_l_after:.1f} (expected ~20)"

    def test_limbal_darkening_monotonic_and_saturates_at_cap(self, enhancer):
        """Normal iris limbal darkening increases monotonically with strength up to the cap then saturates."""
        img_bgr, iris_mask = self._build_iris(body_l_value=80, limbal_l_value=70)
        
        cy, cx, r = 32.0, 32.0, 16.0
        yy, xx = np.mgrid[0:64, 0:64]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / r
        limbal_annulus = (dist >= 0.85) & (dist <= 0.95)

        def get_limbal_l(strength):
            out = enhancer._sculpt_iris(img_bgr, iris_mask, strength=strength)
            lab_out = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
            return float(np.mean(lab_out[limbal_annulus, 0]))

        l_0 = get_limbal_l(0.0)
        l_low = get_limbal_l(0.2)
        l_mid = get_limbal_l(0.5)
        l_high = get_limbal_l(1.0)

        # Monotonic darkening for strength up to headroom cap (initial contrast = 80-70=10, cap=25, headroom=15)
        # At strength 0.2: limbal_darken = 6.0 -> lightness drops
        # At strength 0.5: limbal_darken = 15.0 (hits cap 25 - 10 = 15) -> lightness drops further
        assert l_low < l_0, f"Expected l_low ({l_low:.1f}) < l_0 ({l_0:.1f})"
        assert l_mid < l_low, f"Expected l_mid ({l_mid:.1f}) < l_low ({l_low:.1f})"

        # Verify clear saturation inflection:
        # Initial 0.0 -> 0.5 strength drop is large (~14.8 L* units),
        # while doubling strength from 0.5 -> 1.0 yields only a small residual change (< 3.5 L* units) from USM micro-contrast.
        drop_to_cap = l_0 - l_mid
        drop_past_cap = l_mid - l_high
        assert drop_to_cap > 10.0, f"Expected strong initial darkening up to cap, got drop={drop_to_cap:.2f}"
        assert drop_past_cap < 3.5, f"Expected limbal darkening to saturate past cap, got additional drop={drop_past_cap:.2f}"



