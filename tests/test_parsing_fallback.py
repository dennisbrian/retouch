"""Tests for retouch/parsing.py — FaceParser landmark fallback and helpers."""
import numpy as np
import pytest
from retouch.parsing import FaceParser, FaceRegions
from retouch.detection import _Landmark, _LandmarkCompat
from retouch.utils import get_points


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def _make_landmarks(num=478):
    lm_list = [_Landmark(0.5, 0.5, 0.0) for _ in range(num)]
    np.random.seed(42)
    for i in range(num):
        lm_list[i] = _Landmark(
            float(np.random.uniform(0.1, 0.9)),
            float(np.random.uniform(0.1, 0.9)),
            0.0
        )
    return _LandmarkCompat(lm_list)


@pytest.fixture
def parser():
    return FaceParser()


@pytest.fixture
def img():
    return np.full((200, 200, 3), 128, dtype=np.uint8)


class TestLandmarkFallbackOnly:
    def test_returns_face_regions(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._landmark_fallback_only(landmarks, img, None, 50.0)
        assert isinstance(result, FaceRegions)

    def test_skin_mask_populated(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._landmark_fallback_only(landmarks, img, None, 50.0)
        assert result.skin is not None
        assert result.skin.shape == img.shape[:2]
        assert result.skin.dtype == np.float32
        assert result.skin.max() <= 1.0

    def test_face_oval_populated(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._landmark_fallback_only(landmarks, img, None, 50.0)
        assert result.face_oval is not None
        assert result.face_oval.shape == img.shape[:2]

    def test_with_person_mask(self, parser, img):
        landmarks = _make_landmarks()
        person_mask = np.ones((200, 200), dtype=np.float32)
        person_mask[100:, :] = 0.0
        result = parser._landmark_fallback_only(landmarks, img, person_mask, 50.0)
        assert result.skin is not None

    def test_person_mask_uint8(self, parser, img):
        landmarks = _make_landmarks()
        person_mask = np.ones((200, 200), dtype=np.uint8) * 255
        result = parser._landmark_fallback_only(landmarks, img, person_mask, 50.0)
        assert result.skin is not None

    def test_person_mask_3d(self, parser, img):
        landmarks = _make_landmarks()
        person_mask = np.ones((200, 200, 1), dtype=np.float32)
        result = parser._landmark_fallback_only(landmarks, img, person_mask, 50.0)
        assert result.skin is not None

    def test_all_subregions_populated(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._landmark_fallback_only(landmarks, img, np.ones((200, 200), dtype=np.float32), 50.0)
        assert result.left_eye is not None
        assert result.right_eye is not None
        assert result.lips is not None
        assert result.left_iris is not None
        assert result.right_iris is not None
        assert result.nose is not None
        assert result.left_under_eye is not None
        assert result.right_under_eye is not None
        assert result.forehead is not None

    def test_skin_excludes_eyes_and_lips(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._landmark_fallback_only(landmarks, img, None, 50.0)
        assert result.skin is not None
        if result.left_eye is not None:
            overlap = (result.skin * result.left_eye).max()
            assert overlap < 0.5


class TestAddLandmarkSubregions:
    def test_iris_fallback_on_index_error(self, parser, img):
        regions = FaceRegions()
        regions.skin = np.ones((200, 200), dtype=np.float32)
        bad_landmarks = _LandmarkCompat([_Landmark(0.5, 0.5, 0.0) for _ in range(466)])
        parser._add_landmark_subregions(regions, bad_landmarks, 200, 200, 50.0, 5)
        assert regions.left_iris is not None
        assert regions.right_iris is not None
        assert regions.left_iris.shape == (200, 200)

    def test_nose_bridge_populated(self, parser, img):
        regions = FaceRegions()
        regions.skin = np.ones((200, 200), dtype=np.float32)
        landmarks = _make_landmarks()
        parser._add_landmark_subregions(regions, landmarks, 200, 200, 50.0, 5)
        assert regions.nose_bridge is not None
        assert regions.forehead_center is not None
        assert regions.cheek_highlights_l is not None
        assert regions.cheek_highlights_r is not None
        assert regions.jawline_contour is not None


class TestInternalHelpers:
    def test_mask_returns_float32(self, parser, img):
        landmarks = _make_landmarks()
        from retouch.parsing import FACE_OVAL
        result = parser._mask(landmarks, FACE_OVAL, 200, 200, 5)
        assert result.dtype == np.float32
        assert result.shape == (200, 200)

    def test_circle_mask(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._circle_mask(landmarks, 151, 20, 200, 200, 5)
        assert result.dtype == np.float32
        assert result.shape == (200, 200)
        assert result.max() <= 1.0

    def test_iris_mask(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._iris_mask(landmarks, [468, 469, 470, 471, 472], 200, 200, 50.0)
        assert result.dtype == np.float32
        assert result.shape == (200, 200)

    def test_forehead_mask(self, parser, img):
        landmarks = _make_landmarks()
        result = parser._forehead_mask(landmarks, 200, 200, 5)
        assert result.dtype == np.float32
        assert result.shape == (200, 200)
