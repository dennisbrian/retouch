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


# ---------------------------------------------------------------------------
# Backlog #6 — Reshape completeness (L/R side variants + neck)
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _nose_compat():
    lm_list = [_make_mock_landmark(0.5, 0.5) for _ in range(500)]
    lm_list[234] = _make_mock_landmark(0.3, 0.5)
    lm_list[454] = _make_mock_landmark(0.7, 0.5)
    lm_list[168] = _make_mock_landmark(0.5, 0.45)
    lm_list[48] = _make_mock_landmark(0.45, 0.5)
    lm_list[278] = _make_mock_landmark(0.55, 0.5)
    return _LandmarkCompat(lm_list)


def _nose_landmarks():
    return _nose_compat().landmark


class TestReshapeParamWiring:
    def test_context_carries_new_fields(self):
        from retouch.engine import build_context
        rec = {"reshape": {
            "jaw_width_l": 40, "jaw_width_r": 10,
            "nose_width_l": 20, "nose_width_r": 5,
            "eye_size_l": 15, "eye_size_r": 8,
            "neck_width": 25, "neck_length": 12,
        }}
        ctx = build_context("natural", rec, {})
        assert ctx.reshape_jaw_width_l == 40
        assert ctx.reshape_jaw_width_r == 10
        assert ctx.reshape_nose_width_l == 20
        assert ctx.reshape_nose_width_r == 5
        assert ctx.reshape_eye_size_l == 15
        assert ctx.reshape_eye_size_r == 8
        assert ctx.reshape_neck_width == 25
        assert ctx.reshape_neck_length == 12

    def test_defaults_zero(self):
        from retouch.engine import build_context
        ctx = build_context("natural", {}, {})
        for attr in (
            "reshape_jaw_width_l", "reshape_jaw_width_r",
            "reshape_nose_width_l", "reshape_nose_width_r",
            "reshape_eye_size_l", "reshape_eye_size_r",
            "reshape_neck_width", "reshape_neck_length",
        ):
            assert getattr(ctx, attr) == 0


class TestSideWarps:
    def test_jaw_global_two_sides(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._jaw_width_warps(lm, 160.0, 30.0, 400, 400)
        assert len(warps) == 2

    def test_jaw_left_only(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._jaw_width_warps(lm, 160.0, 0.0, 400, 400, slider_l=30.0, slider_r=0.0)
        assert len(warps) == 1

    def test_jaw_both_sides_independent(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._jaw_width_warps(lm, 160.0, 0.0, 400, 400, slider_l=30.0, slider_r=10.0)
        assert len(warps) == 2

    def test_jaw_global_byte_identical(self, reshaper):
        lm = _nose_landmarks()
        legacy = reshaper._jaw_width_warps(lm, 160.0, 30.0, 400, 400)
        via_side = reshaper._jaw_width_warps(lm, 160.0, 0.0, 400, 400, slider_l=30.0, slider_r=30.0)
        assert legacy == via_side

    def test_nose_global_radial(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._nose_width_warps(lm, 160.0, 30.0, 400, 400)
        assert len(warps) == 8  # _SCALE_RING_N

    def test_nose_left_only(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._nose_width_warps(lm, 160.0, 0.0, 400, 400, slider_l=30.0, slider_r=0.0)
        assert len(warps) == 1

    def test_eye_side_no_crash(self, reshaper):
        compat = _nose_compat()
        warps = reshaper._eye_size_warps(
            compat, compat.landmark, 160.0, 0.0, 400, 400, slider_l=30.0, slider_r=0.0
        )
        assert isinstance(warps, list)


class TestNeckWarps:
    def test_neck_width_zero(self, reshaper):
        lm = _nose_landmarks()
        assert reshaper._neck_width_warps(lm, 160.0, 0.0, 400, 400) == []

    def test_neck_width_active(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._neck_width_warps(lm, 160.0, 30.0, 400, 400)
        assert len(warps) == 6

    def test_neck_length_zero(self, reshaper):
        lm = _nose_landmarks()
        assert reshaper._neck_length_warps(lm, 160.0, 0.0, 400, 400) == []

    def test_neck_length_active(self, reshaper):
        lm = _nose_landmarks()
        warps = reshaper._neck_length_warps(lm, 160.0, 30.0, 400, 400)
        assert len(warps) == 3


class TestBackwardCompat:
    def _grad(self):
        grad = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (400, 1))
        return np.stack([grad] * 3, axis=-1)

    def test_side_zero_matches_global(self, reshaper):
        grad = self._grad()
        face = _make_face_data(200, 200, 80, 100)
        ctx_global = SimpleNamespace(reshape_jaw_width=30.0)
        ctx_side_zero = SimpleNamespace(
            reshape_jaw_width=30.0,
            reshape_jaw_width_l=0.0, reshape_jaw_width_r=0.0,
            reshape_nose_width_l=0.0, reshape_nose_width_r=0.0,
            reshape_eye_size_l=0.0, reshape_eye_size_r=0.0,
            reshape_neck_width=0.0, reshape_neck_length=0.0,
        )
        out_g = reshaper.reshape(grad.copy(), [face], ctx_global)
        out_s = reshaper.reshape(grad.copy(), [face], ctx_side_zero)
        assert np.array_equal(out_g, out_s)

    def test_side_value_changes_output(self, reshaper):
        grad = self._grad()
        face = _make_face_data(200, 200, 80, 100)
        ctx = SimpleNamespace(reshape_jaw_width_l=40.0)
        out = reshaper.reshape(grad.copy(), [face], ctx)
        assert not np.array_equal(out, grad)

    def test_neck_value_changes_output(self, reshaper):
        grad = self._grad()
        face = _make_face_data(200, 200, 80, 100)
        ctx = SimpleNamespace(reshape_neck_width=40.0)
        out = reshaper.reshape(grad.copy(), [face], ctx)
        assert not np.array_equal(out, grad)
