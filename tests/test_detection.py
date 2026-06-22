"""Tests for retouch/detection.py — data classes and helpers only (no model)."""

import numpy as np
import pytest

from retouch.detection import (
    _Landmark,
    _LandmarkCompat,
    FaceData,
    FaceContext,
    FaceDetector,
)
from retouch.parsing import FaceRegions


class TestLandmark:
    def test_creates_with_xy(self):
        lm = _Landmark(x=0.5, y=0.3)
        assert lm.x == 0.5
        assert lm.y == 0.3
        assert lm.z == 0.0

    def test_creates_with_xyz(self):
        lm = _Landmark(x=0.1, y=0.2, z=0.5)
        assert lm.x == 0.1
        assert lm.y == 0.2
        assert lm.z == 0.5

    def test_default_z_is_zero(self):
        lm = _Landmark(x=0.0, y=0.0)
        assert lm.z == 0.0


class TestFaceData:
    def test_creates_with_minimal_args(self):
        fd = FaceData(landmarks=None, bbox=(10, 20, 100, 150), ied=60.0)
        assert fd.landmarks is None
        assert fd.bbox == (10, 20, 100, 150)
        assert fd.ied == 60.0
        assert fd.confidence == 1.0

    def test_creates_with_confidence(self):
        fd = FaceData(landmarks=None, bbox=(0, 0, 50, 60), ied=30.0, confidence=0.95)
        assert fd.confidence == 0.95

    def test_with_mock_landmark_compat(self):
        compat = _LandmarkCompat([_Landmark(0.1, 0.2), _Landmark(0.3, 0.4)])
        fd = FaceData(landmarks=compat, bbox=(5, 5, 90, 110), ied=45.0)
        assert fd.landmarks is compat
        assert len(fd.landmarks.landmark) == 2

    def test_bbox_as_tuple(self):
        fd = FaceData(landmarks=None, bbox=(1, 2, 3, 4), ied=10.0)
        x, y, w, h = fd.bbox
        assert (x, y, w, h) == (1, 2, 3, 4)


class TestFaceContext:
    def test_creates_with_minimal_args(self):
        fd = FaceData(landmarks=None, bbox=(0, 0, 10, 10), ied=5.0)
        regions = FaceRegions()
        ctx = FaceContext(face_data=fd, regions=regions)
        assert ctx.face_data is fd
        assert ctx.regions is regions
        assert ctx.index == 0
        assert ctx.face_image is None

    def test_creates_with_all_args(self):
        fd = FaceData(landmarks=None, bbox=(0, 0, 10, 10), ied=5.0)
        regions = FaceRegions()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = FaceContext(face_data=fd, regions=regions, index=2, face_image=img)
        assert ctx.index == 2
        assert ctx.face_image is img

    def test_regions_is_face_regions_instance(self):
        fd = FaceData(landmarks=None, bbox=(0, 0, 10, 10), ied=5.0)
        regions = FaceRegions()
        ctx = FaceContext(face_data=fd, regions=regions)
        assert isinstance(ctx.regions, FaceRegions)
        assert ctx.regions.skin is None  # default state


class TestLandmarkCompat:
    def test_wraps_list(self):
        inner = [_Landmark(0.1, 0.2), _Landmark(0.3, 0.4)]
        compat = _LandmarkCompat(inner)
        assert compat.landmark is inner
        assert len(compat.landmark) == 2
        assert compat.landmark[0].x == 0.1

    def test_wraps_empty_list(self):
        compat = _LandmarkCompat([])
        assert compat.landmark == []

    def test_landmark_attribute_is_exposed(self):
        inner = [_Landmark(0.5, 0.6, 0.7)]
        compat = _LandmarkCompat(inner)
        assert compat.landmark[0].x == 0.5
        assert compat.landmark[0].y == 0.6
        assert compat.landmark[0].z == 0.7


# ---------------------------------------------------------------------------
# Pure static helpers on FaceDetector
# ---------------------------------------------------------------------------


class TestRemapLandmarks:
    """Tests for FaceDetector._remap_landmarks — pure coordinate transform."""

    def test_identity_remap_when_crop_is_full_image(self):
        # When the crop covers the full image, remapping must return the
        # normalised coordinates unchanged (x, y) — only z is preserved.
        full_w, full_h = 100, 100
        ox, oy = 0, 0
        cw, ch = full_w, full_h
        landmarks = [_Landmark(0.1, 0.2, 0.3), _Landmark(0.5, 0.7, -0.1)]

        out = FaceDetector._remap_landmarks(landmarks, ox, oy, cw, ch, full_w, full_h)

        assert len(out) == len(landmarks)
        for src, dst in zip(landmarks, out):
            assert dst.x == pytest.approx(src.x)
            assert dst.y == pytest.approx(src.y)
            assert dst.z == src.z

    def test_translation_to_crop_origin(self):
        # Landmarks at (0, 0) inside a crop offset by (50, 25) should be
        # remapped to (50/full_w, 25/full_h).
        full_w, full_h = 200, 200
        ox, oy = 50, 25
        cw, ch = 100, 100
        landmarks = [_Landmark(0.0, 0.0, 0.0)]

        out = FaceDetector._remap_landmarks(landmarks, ox, oy, cw, ch, full_w, full_h)

        assert out[0].x == pytest.approx(50.0 / 200.0)
        assert out[0].y == pytest.approx(25.0 / 200.0)
        assert out[0].z == 0.0

    def test_crop_at_max_corner(self):
        # Crop at the far edge: (ox + cw, oy + ch) == (full_w, full_h)
        full_w, full_h = 200, 100
        ox, oy = 100, 50
        cw, ch = 100, 50
        landmarks = [_Landmark(1.0, 1.0, 0.0)]

        out = FaceDetector._remap_landmarks(landmarks, ox, oy, cw, ch, full_w, full_h)

        assert out[0].x == pytest.approx(1.0)
        assert out[0].y == pytest.approx(1.0)

    def test_z_is_passed_through_unchanged(self):
        full_w, full_h = 50, 50
        ox, oy = 5, 5
        cw, ch = 40, 40
        landmarks = [_Landmark(0.5, 0.5, 0.123)]

        out = FaceDetector._remap_landmarks(landmarks, ox, oy, cw, ch, full_w, full_h)

        assert out[0].z == pytest.approx(0.123)

    def test_handles_empty_landmark_list(self):
        out = FaceDetector._remap_landmarks([], 0, 0, 10, 10, 100, 100)
        assert out == []

    def test_output_type_is_list_of_landmarks(self):
        landmarks = [_Landmark(0.1, 0.2, 0.0)]
        out = FaceDetector._remap_landmarks(landmarks, 0, 0, 10, 10, 100, 100)
        assert isinstance(out, list)
        assert all(isinstance(lm, _Landmark) for lm in out)

    def test_remap_uses_full_image_dimensions_for_scaling(self):
        # x in crop -> (x * cw + ox) / full_w
        # If full_w is 100 and ox is 25, then 1.0 in crop-relative normalised
        # coords maps to (100 + 25) / 100 = 1.25 of the full image width.
        out = FaceDetector._remap_landmarks(
            [_Landmark(1.0, 1.0, 0.0)],
            ox=25, oy=25, cw=100, ch=100, full_w=100, full_h=100,
        )
        # The landmark was at the far edge of the crop, but the crop is
        # already 100px wide while full image is also 100px wide. The
        # crop is *overhanging* the image. We just check the math
        # matches the formula exactly.
        assert out[0].x == pytest.approx(125.0 / 100.0)
        assert out[0].y == pytest.approx(125.0 / 100.0)


class TestBboxFromLandmarks:
    """Tests for FaceDetector._bbox_from_landmarks — pure bbox math."""

    def _compat(self, pts):
        return _LandmarkCompat([_Landmark(x, y, 0.0) for (x, y) in pts])

    def test_single_landmark_bbox(self):
        compat = self._compat([(0.5, 0.5)])
        x, y, w, h = FaceDetector._bbox_from_landmarks(compat, 100, 100)
        assert (x, y, w, h) == (50, 50, 0, 0)

    def test_two_landmarks_bbox(self):
        compat = self._compat([(0.1, 0.2), (0.3, 0.4)])
        x, y, w, h = FaceDetector._bbox_from_landmarks(compat, 100, 100)
        # x range: 10..30 -> (10, 20, 20)
        # y range: 20..40 -> (10, 20, 20)
        assert x == 10
        assert y == 20
        assert w == 20
        assert h == 20

    def test_clamps_to_image_bounds(self):
        # Landmarks outside [0, w] / [0, h] must be clamped
        compat = self._compat([(-0.1, -0.1), (1.1, 1.1)])
        x, y, w, h = FaceDetector._bbox_from_landmarks(compat, 100, 50)
        # min xs=0, max xs capped at 100
        assert x == 0
        assert y == 0
        # widths reflect clamped range
        assert w == 100
        assert h == 50

    def test_returns_xywh_tuple(self):
        compat = self._compat([(0.0, 0.0), (1.0, 1.0)])
        result = FaceDetector._bbox_from_landmarks(compat, 200, 100)
        assert len(result) == 4
        x, y, w, h = result
        assert (x, y, w, h) == (0, 0, 200, 100)

    def test_pixel_coordinates(self):
        # Landmarks at (0.25, 0.5) on a 200x100 image
        # 0.25 * 200 = 50 (x); 0.5 * 100 = 50 (y)
        compat = self._compat([(0.25, 0.5)])
        x, y, w, h = FaceDetector._bbox_from_landmarks(compat, 200, 100)
        assert (x, y) == (50, 50)
        assert w == 0
        assert h == 0

    def test_landmark_order_does_not_matter(self):
        # Same set of points in different order should produce the same bbox
        compat_a = self._compat([(0.1, 0.2), (0.3, 0.4), (0.5, 0.5)])
        compat_b = self._compat([(0.5, 0.5), (0.1, 0.2), (0.3, 0.4)])
        assert FaceDetector._bbox_from_landmarks(compat_a, 100, 100) == \
               FaceDetector._bbox_from_landmarks(compat_b, 100, 100)


class TestResolveDelegate:
    """Tests for FaceDetector._resolve_delegate — pure provider selection."""

    def test_returns_cpu_delegate(self):
        # Build a fake base object with the expected Delegate enum members.
        class _Base:
            class Delegate:
                CPU = "CPU"
                GPU = "GPU"

        result = FaceDetector._resolve_delegate(_Base)
        # Current implementation always selects CPU regardless of platform
        assert result == "CPU"

    def test_does_not_pick_gpu(self):
        # Even if a GPU option exists, _resolve_delegate should return CPU
        # (the function explicitly ignores platform-specific GPU).
        class _Base:
            class Delegate:
                CPU = "CPU"
                GPU = "GPU"
                NPU = "NPU"

        result = FaceDetector._resolve_delegate(_Base)
        assert result != "GPU"
        assert result != "NPU"

    def test_static_method_works_without_instance(self):
        class _Base:
            class Delegate:
                CPU = "cpu"
                GPU = "gpu"

        # Should be callable as a class method
        result = FaceDetector._resolve_delegate(_Base)
        assert result == "cpu"
