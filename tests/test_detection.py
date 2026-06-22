"""Tests for retouch/detection.py — data classes and helpers only (no model)."""

import numpy as np
import pytest

from retouch.detection import _Landmark, _LandmarkCompat, FaceData, FaceContext
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
