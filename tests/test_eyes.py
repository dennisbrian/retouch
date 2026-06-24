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
