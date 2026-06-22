"""Tests for retouch/geometry.py — FaceReshaper."""
import numpy as np
import pytest
from retouch.geometry import FaceReshaper
from retouch.detection import FaceData, _Landmark, _LandmarkCompat


def _make_mock_landmark(x, y):
    return _Landmark(x, y, z=0.0)


def _make_face_data(cx, cy, face_w, face_h):
    lm_list = [_make_mock_landmark(0.5, 0.5) for _ in range(500)]
    lm_list[234] = _make_mock_landmark((cx - face_w // 2) / 400.0, cy / 400.0)
    lm_list[454] = _make_mock_landmark((cx + face_w // 2) / 400.0, cy / 400.0)
    lm_list[117] = _make_mock_landmark(0.6, 0.55)
    lm_list[346] = _make_mock_landmark(0.4, 0.55)
    lm_list[152] = _make_mock_landmark(0.5, 0.7)
    compat = _LandmarkCompat(lm_list)
    return FaceData(landmarks=compat, bbox=(cx - face_w // 2, cy - face_h // 2, face_w, face_h), ied=50.0)


@pytest.fixture
def reshaper():
    return FaceReshaper()


@pytest.fixture
def img():
    return np.full((400, 400, 3), 128, dtype=np.uint8)


class TestReshape:
    def test_zero_strength(self, reshaper, img):
        face = _make_face_data(200, 200, 80, 100)
        result = reshaper.reshape(img, [face], strength=0)
        assert np.all(result == img)

    def test_no_faces(self, reshaper, img):
        result = reshaper.reshape(img, [], strength=50)
        assert np.all(result == img)

    def test_output_shape(self, reshaper, img):
        face = _make_face_data(200, 200, 80, 100)
        result = reshaper.reshape(img, [face], strength=50)
        assert result.shape == (400, 400, 3)
        assert result.dtype == np.uint8

    def test_changes_image(self, reshaper):
        grad = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (400, 1))
        grad_img = np.stack([grad] * 3, axis=-1)
        face = _make_face_data(200, 200, 80, 100)
        result = reshaper.reshape(grad_img, [face], strength=80)
        assert not np.allclose(result, grad_img)

    def test_multiple_faces(self, reshaper, img):
        face1 = _make_face_data(150, 150, 60, 75)
        face2 = _make_face_data(300, 300, 60, 75)
        result = reshaper.reshape(img, [face1, face2], strength=50)
        assert result.shape == (400, 400, 3)

    def test_tiny_face_skipped(self, reshaper, img):
        face = _make_face_data(200, 200, 8, 10)
        result = reshaper.reshape(img, [face], strength=50)
        assert np.all(result == img)

    def test_face_at_edge(self, reshaper, img):
        face = _make_face_data(20, 20, 60, 75)
        result = reshaper.reshape(img, [face], strength=50)
        assert result.shape == (400, 400, 3)
