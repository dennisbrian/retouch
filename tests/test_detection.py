"""Tests for retouch/detection.py — data classes and helpers only (no model)."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retouch.detection import (
    _Landmark,
    _LandmarkCompat,
    _detach_landmarks,
    FaceData,
    FaceContext,
    FaceDetector,
)
from retouch.lighting import LightDirection
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
        assert fd.confidence_source == "unknown"

    def test_creates_with_confidence(self):
        fd = FaceData(
            landmarks=None,
            bbox=(0, 0, 50, 60),
            ied=30.0,
            confidence=0.95,
            confidence_source="retinaface",
        )
        assert fd.confidence == 0.95
        assert fd.confidence_source == "retinaface"

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
        assert ctx.light_direction is None

    def test_creates_with_all_args(self):
        fd = FaceData(landmarks=None, bbox=(0, 0, 10, 10), ied=5.0)
        regions = FaceRegions()
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        ctx = FaceContext(face_data=fd, regions=regions, index=2, face_image=img)
        assert ctx.index == 2
        assert ctx.face_image is img

    def test_carries_cached_light_direction(self):
        fd = FaceData(landmarks=None, bbox=(0, 0, 10, 10), ied=5.0)
        estimate = LightDirection((1.0, 0.0), 0.8, "catchlights")
        ctx = FaceContext(face_data=fd, regions=FaceRegions(), light_direction=estimate)
        assert ctx.light_direction is estimate

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

    def test_detach_landmarks_copies_foreign_objects(self):
        foreign = [MagicMock(x=0.2, y=0.4, z=-0.1)]

        detached = _detach_landmarks(foreign)

        assert detached == [_Landmark(0.2, 0.4, -0.1)]
        assert detached[0] is not foreign[0]


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


# ---------------------------------------------------------------------------
# FaceDetector.__init__  (mocked to avoid loading MediaPipe)
# ---------------------------------------------------------------------------


def _mock_create_tasks_default():
    """Build a (mock_landmarker, mock_segmenter) pair for injection."""
    return MagicMock(), MagicMock()


class TestFaceDetectorInit:
    """Tests for FaceDetector.__init__ — covered via _create_tasks mock."""

    def test_allow_unavailable_is_explicit_global_only_fallback(self):
        """The application may start, but the detector must remain non-certifying."""
        with patch.object(
            FaceDetector, "_create_tasks", side_effect=RuntimeError("native probe failed"),
        ):
            detector = FaceDetector(allow_unavailable=True)
        try:
            assert detector.available is False
            assert "native probe failed" in detector.unavailable_reason
            image = np.zeros((12, 16, 3), dtype=np.uint8)
            assert detector.detect(image) == []
            assert np.all(detector.segment_person(image) == 1.0)
        finally:
            detector.close()

    def test_default_params(self):
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=_mock_create_tasks_default(),
        ):
            detector = FaceDetector()
            try:
                assert detector.max_faces == 10
                assert detector.min_confidence == 0.4
                assert detector._landmarker is not None
                assert detector._segmenter is not None
            finally:
                detector.close()

    def test_custom_params(self):
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=_mock_create_tasks_default(),
        ):
            detector = FaceDetector(max_faces=5, min_confidence=0.7)
            try:
                assert detector.max_faces == 5
                assert detector.min_confidence == 0.7
            finally:
                detector.close()

    def test_custom_params_with_segmenter_none(self):
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=(MagicMock(), None),
        ):
            detector = FaceDetector(max_faces=2, min_confidence=0.9)
            try:
                assert detector.max_faces == 2
                assert detector.min_confidence == 0.9
                assert detector._landmarker is not None
                assert detector._segmenter is None
            finally:
                detector.close()

    def test_refine_landmarks_param_accepted(self):
        # `refine_landmarks` is currently a constructor arg that is accepted
        # but not stored. We just need to confirm it does not break init.
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=_mock_create_tasks_default(),
        ):
            detector = FaceDetector(refine_landmarks=False)
            try:
                assert detector.max_faces == 10
            finally:
                detector.close()

    def test_calls_create_tasks_exactly_once(self):
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=_mock_create_tasks_default(),
        ) as mock_create:
            detector = FaceDetector()
            try:
                assert mock_create.call_count == 1
                # _create_tasks signature: (base, vision, delegate, max_faces,
                # min_confidence). Inspect positional args.
                args, kwargs = mock_create.call_args
                # delegate is positional arg index 2
                assert args[3] == 10
                assert args[4] == 0.4
            finally:
                detector.close()

    def test_create_tasks_called_with_resolved_delegate(self):
        # _resolve_delegate is called internally; whatever it returns is
        # forwarded as the delegate to _create_tasks.
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=_mock_create_tasks_default(),
        ) as mock_create, \
             patch.object(
                 FaceDetector, "_resolve_delegate", return_value="FAKE-DELEGATE",
             ) as mock_resolve:
            detector = FaceDetector()
            try:
                assert mock_resolve.called
                args, _ = mock_create.call_args
                assert args[2] == "FAKE-DELEGATE"
            finally:
                detector.close()

    def test_gpu_failure_falls_back_to_cpu(self):
        # When _create_tasks raises on GPU and _resolve_delegate returned
        # base.Delegate.GPU, the constructor must fall back to CPU.
        import mediapipe as mp

        base = mp.tasks.BaseOptions
        gpu = base.Delegate.GPU
        cpu = base.Delegate.CPU

        call_count = {"n": 0}

        def fake_create(base_arg, vision_arg, delegate, max_faces, min_confidence):
            call_count["n"] += 1
            if delegate == gpu:
                raise RuntimeError("GPU init failed")
            return MagicMock(), None

        with patch.object(
            FaceDetector, "_create_tasks", side_effect=fake_create,
        ), patch.object(FaceDetector, "_resolve_delegate", return_value=gpu):
            detector = FaceDetector()
            try:
                # Two calls: one with GPU (failed), one with CPU (succeeded)
                assert call_count["n"] == 2
                assert detector._landmarker is not None
            finally:
                detector.close()

    def test_cpu_failure_re_raises(self):
        # When _create_tasks fails for a non-GPU delegate, the exception
        # must propagate to the caller.
        with patch.object(
            FaceDetector, "_create_tasks",
            side_effect=RuntimeError("CPU init failed"),
        ):
            with pytest.raises(RuntimeError, match="CPU init failed"):
                FaceDetector()


# ---------------------------------------------------------------------------
# FaceDetector.close
# ---------------------------------------------------------------------------


class TestFaceDetectorClose:
    """Tests for FaceDetector.close() — resource cleanup."""

    def test_close_calls_close_on_landmarker(self):
        mock_landmarker = MagicMock()
        mock_segmenter = MagicMock()
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=(mock_landmarker, mock_segmenter),
        ):
            detector = FaceDetector()
            detector.close()
        assert mock_landmarker.close.called

    def test_close_calls_close_on_segmenter(self):
        mock_landmarker = MagicMock()
        mock_segmenter = MagicMock()
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=(mock_landmarker, mock_segmenter),
        ):
            detector = FaceDetector()
            detector.close()
        assert mock_segmenter.close.called

    def test_close_with_none_segmenter(self):
        # When segmenter was not loaded (model missing), close() must not
        # try to call .close() on None.
        mock_landmarker = MagicMock()
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=(mock_landmarker, None),
        ):
            detector = FaceDetector()
            detector.close()
        assert mock_landmarker.close.called

    def test_close_with_no_attributes(self):
        # Build a bare instance without invoking __init__ and call close().
        # The method must guard with hasattr and not raise.
        detector = FaceDetector.__new__(FaceDetector)
        detector.close()

    def test_close_with_none_attributes(self):
        detector = FaceDetector.__new__(FaceDetector)
        detector._landmarker = None
        detector._segmenter = None
        detector.close()


# ---------------------------------------------------------------------------
# FaceDetector context manager protocol
# ---------------------------------------------------------------------------


class TestFaceDetectorContextManager:
    def test_context_manager_returns_self(self):
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=_mock_create_tasks_default(),
        ):
            with FaceDetector() as detector:
                assert isinstance(detector, FaceDetector)

    def test_context_manager_calls_close_on_exit(self):
        mock_landmarker = MagicMock()
        mock_segmenter = MagicMock()
        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=(mock_landmarker, mock_segmenter),
        ):
            with FaceDetector():
                pass
        assert mock_landmarker.close.called
        assert mock_segmenter.close.called


# ---------------------------------------------------------------------------
# FaceDetector._create_tasks  (MediaPipe task creation)
# ---------------------------------------------------------------------------


class TestCreateTasks:
    """Tests for FaceDetector._create_tasks — MediaPipe task wiring."""

    def test_returns_tuple_of_two(self):
        mock_base = MagicMock()
        mock_vision = MagicMock()
        mock_landmarker = MagicMock()
        mock_segmenter = MagicMock()
        mock_vision.FaceLandmarker.create_from_options.return_value = mock_landmarker
        mock_vision.ImageSegmenter.create_from_options.return_value = mock_segmenter

        with patch("retouch.detection.os.path.exists", return_value=True):
            out = FaceDetector._create_tasks(
                mock_base, mock_vision, "CPU", 5, 0.7,
            )
        assert isinstance(out, tuple)
        assert len(out) == 2

    def test_creates_face_landmarker(self):
        mock_base = MagicMock()
        mock_vision = MagicMock()
        mock_vision.FaceLandmarker.create_from_options.return_value = MagicMock()
        mock_vision.ImageSegmenter.create_from_options.return_value = MagicMock()

        with patch("retouch.detection.os.path.exists", return_value=True):
            FaceDetector._create_tasks(mock_base, mock_vision, "CPU", 5, 0.7)
        assert mock_vision.FaceLandmarker.create_from_options.called

    def test_creates_image_segmenter_when_model_exists(self):
        mock_base = MagicMock()
        mock_vision = MagicMock()
        mock_vision.FaceLandmarker.create_from_options.return_value = MagicMock()
        mock_vision.ImageSegmenter.create_from_options.return_value = MagicMock()

        with patch("retouch.detection.os.path.exists", return_value=True):
            landmarker, segmenter = FaceDetector._create_tasks(
                mock_base, mock_vision, "CPU", 5, 0.7,
            )
        assert mock_vision.ImageSegmenter.create_from_options.called
        assert segmenter is not None
        assert landmarker is not None

    def test_skips_image_segmenter_when_model_missing(self):
        mock_base = MagicMock()
        mock_vision = MagicMock()
        mock_vision.FaceLandmarker.create_from_options.return_value = MagicMock()
        mock_vision.ImageSegmenter.create_from_options.return_value = MagicMock()

        def fake_exists(path):
            return "face_landmarker" in path  # segmenter model missing

        with patch("retouch.detection.os.path.exists", side_effect=fake_exists):
            landmarker, segmenter = FaceDetector._create_tasks(
                mock_base, mock_vision, "CPU", 5, 0.7,
            )
        assert not mock_vision.ImageSegmenter.create_from_options.called
        assert segmenter is None
        assert landmarker is not None

    def test_forwards_max_faces_to_landmarker_options(self):
        mock_base = MagicMock()
        mock_vision = MagicMock()
        mock_vision.FaceLandmarker.create_from_options.return_value = MagicMock()
        mock_vision.ImageSegmenter.create_from_options.return_value = MagicMock()

        with patch("retouch.detection.os.path.exists", return_value=True):
            FaceDetector._create_tasks(mock_base, mock_vision, "CPU", 7, 0.6)
        # The function constructs vision.FaceLandmarkerOptions(...) with
        # num_faces=<max_faces>; the mock records that constructor call.
        options_kwargs = mock_vision.FaceLandmarkerOptions.call_args.kwargs
        assert options_kwargs.get("num_faces") == 7

    def test_forwards_min_confidence_to_landmarker_options(self):
        mock_base = MagicMock()
        mock_vision = MagicMock()
        mock_vision.FaceLandmarker.create_from_options.return_value = MagicMock()
        mock_vision.ImageSegmenter.create_from_options.return_value = MagicMock()

        with patch("retouch.detection.os.path.exists", return_value=True):
            FaceDetector._create_tasks(mock_base, mock_vision, "CPU", 5, 0.85)
        options_kwargs = mock_vision.FaceLandmarkerOptions.call_args.kwargs
        assert options_kwargs.get("min_face_detection_confidence") == pytest.approx(0.85)
        assert options_kwargs.get("min_face_presence_confidence") == pytest.approx(0.85)

    def test_forwards_delegate_to_base_options(self):
        mock_base = MagicMock()
        mock_vision = MagicMock()
        mock_vision.FaceLandmarker.create_from_options.return_value = MagicMock()
        mock_vision.ImageSegmenter.create_from_options.return_value = MagicMock()

        with patch("retouch.detection.os.path.exists", return_value=True):
            FaceDetector._create_tasks(mock_base, mock_vision, "MY-DELEGATE", 5, 0.5)
        # The first positional arg to `mock_base` (BaseOptions ctor) is the
        # model path; the keyword arg is the delegate.
        kwargs = mock_base.call_args_list[0].kwargs
        assert kwargs.get("delegate") == "MY-DELEGATE"

    def test_uses_landmarker_model_path(self):
        mock_base = MagicMock()
        mock_vision = MagicMock()
        mock_vision.FaceLandmarker.create_from_options.return_value = MagicMock()
        mock_vision.ImageSegmenter.create_from_options.return_value = MagicMock()

        with patch("retouch.detection.os.path.exists", return_value=True):
            FaceDetector._create_tasks(mock_base, mock_vision, "CPU", 5, 0.5)
        # The model_asset_path kwarg should reference the landmarker model
        # path constant from detection.py
        from retouch import detection as det_mod
        kwargs = mock_base.call_args_list[0].kwargs
        assert kwargs.get("model_asset_path") == det_mod._FACE_LANDMARKER_MODEL


# ---------------------------------------------------------------------------
# FaceDetector.detect  (end-to-end with MediaPipe mocked)
# ---------------------------------------------------------------------------


def _build_mock_landmarks(count: int = 478, eye_sep: float = 0.2) -> list:
    """Return a list of ``count`` MagicMock landmarks shaped like MediaPipe's
    ``NormalizedLandmark`` (``.x``, ``.y``, ``.z`` attributes).

    The iris centres are seeded at indices 468 and 473 with a known
    horizontal separation so ``inter_eye_distance`` has reproducible data
    to read. All other landmarks are placed at the image centre.
    """
    landmarks = []
    for i in range(count):
        if i == 468:  # left iris centre
            lm = MagicMock(x=0.5 - eye_sep / 2.0, y=0.5, z=0.0)
        elif i == 473:  # right iris centre
            lm = MagicMock(x=0.5 + eye_sep / 2.0, y=0.5, z=0.0)
        else:
            lm = MagicMock(x=0.5, y=0.5, z=0.0)
        landmarks.append(lm)
    return landmarks


class TestFaceDetectorDetect:
    """Direct tests for ``FaceDetector.detect()`` with the underlying
    MediaPipe task objects replaced by ``MagicMock`` instances.

    ``retinaface`` is forced to ``None`` in ``sys.modules`` so the
    RetinaFace import raises ``ImportError`` and ``detect()`` reliably
    exercises its full-image MediaPipe fallback — making these tests
    independent of whether the optional ``retinaface`` package is
    installed in the CI environment.
    """

    def _make_detector(self, face_landmarks, segmenter=None):
        """Build a FaceDetector with a mocked landmarker/segmenter pair.

        Returns:
            tuple: ``(detector, mock_landmarker, mock_segmenter)``.
        """
        mock_landmarker = MagicMock()
        mock_result = MagicMock()
        mock_result.face_landmarks = face_landmarks
        mock_landmarker.detect.return_value = mock_result
        mock_segmenter = segmenter if segmenter is not None else MagicMock()

        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=(mock_landmarker, mock_segmenter),
        ), patch.dict(sys.modules, {"retinaface": None}):
            detector = FaceDetector()
        return detector, mock_landmarker, mock_segmenter

    def test_detect_with_face_returns_face_data_list(self):
        """A successful detection must yield a single ``FaceData`` with
        well-formed ``bbox``, ``ied``, and a wrapped landmark list."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        landmarks = _build_mock_landmarks(eye_sep=0.2)

        detector, mock_landmarker, _ = self._make_detector([landmarks])
        try:
            faces = detector.detect(img)
        finally:
            detector.close()

        assert isinstance(faces, list)
        assert len(faces) == 1
        fd = faces[0]
        assert isinstance(fd, FaceData)
        assert isinstance(fd.bbox, tuple) and len(fd.bbox) == 4
        x, y, w, h = fd.bbox
        assert 0 <= x < 200 and 0 <= y < 200
        assert w >= 0 and h >= 0
        # inter_eye_distance in pixels = eye_sep * image_width = 0.2 * 200 = 40
        assert fd.ied == pytest.approx(40.0)
        # landmarks are wrapped via _LandmarkCompat
        assert hasattr(fd.landmarks, "landmark")
        assert len(fd.landmarks.landmark) == 478
        # landmarker must have been called at least once (full-image path)
        assert mock_landmarker.detect.called

    def test_detect_no_faces_returns_empty_list(self):
        """When the landmarker reports no faces, ``detect()`` returns ``[]``
        without raising."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        detector, _, _ = self._make_detector(face_landmarks=[])
        try:
            faces = detector.detect(img)
        finally:
            detector.close()

        assert faces == []

    def test_detect_invokes_underlying_landmarker(self):
        """``detect()`` must delegate to the underlying MediaPipe landmarker
        at least once per call (full-image or downscale-then-full)."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        detector, mock_landmarker, _ = self._make_detector(face_landmarks=[])
        try:
            detector.detect(img)
        finally:
            detector.close()

        # The function may invoke .detect() 1x (small image) or 2x
        # (downscale + full-resolution fallback) — only >= 1 is guaranteed.
        assert mock_landmarker.detect.call_count >= 1

    def test_detect_multiple_faces(self):
        """A result with two face landmark sets must produce two ``FaceData``."""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        lm_left = _build_mock_landmarks(eye_sep=0.1)
        lm_right = _build_mock_landmarks(eye_sep=0.1)
        # shift the right face so bbox dedup doesn't collapse them
        for lm in lm_right:
            lm.x = min(0.99, lm.x + 0.4)

        detector, _, _ = self._make_detector([lm_left, lm_right])
        try:
            faces = detector.detect(img)
        finally:
            detector.close()

        assert len(faces) == 2
        assert all(isinstance(f, FaceData) for f in faces)


# ---------------------------------------------------------------------------
# FaceDetector.segment_person  (MediaPipe segmenter mocked)
# ---------------------------------------------------------------------------


class TestFaceDetectorSegmentPerson:
    """Direct tests for ``FaceDetector.segment_person()``.

    The MediaPipe segmenter is replaced with a ``MagicMock`` so we can
    exercise both the success path (confidence mask returned) and the
    fallback paths (no segmenter loaded / empty confidence list).
    """

    def test_segment_person_with_confidence_mask_returns_float32(self):
        """When the segmenter returns a confidence mask, the result must
        be a 2D float32 array matching the input image's spatial shape."""
        h, w = 100, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)

        mask_array = np.full((h, w), 0.7, dtype=np.float32)
        mock_mask = MagicMock()
        mock_mask.numpy_view.return_value = mask_array
        mock_result = MagicMock()
        mock_result.confidence_masks = [mock_mask]

        mock_landmarker = MagicMock()
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = mock_result

        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=(mock_landmarker, mock_segmenter),
        ):
            detector = FaceDetector()
            try:
                mask = detector.segment_person(img)
            finally:
                detector.close()

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (h, w)
        assert mask.dtype == np.float32
        # The mask was passed through np.squeeze + astype(float32), so
        # all values should still equal 0.7.
        assert mask.min() == pytest.approx(0.7)
        assert mask.max() == pytest.approx(0.7)
        assert mock_segmenter.segment.called

    def test_segment_person_with_no_segmenter_returns_ones(self):
        """When the segmenter was not loaded (model file missing),
        ``segment_person`` must short-circuit and return a full-ones
        float32 mask of the right shape."""
        h, w = 100, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)
        mock_landmarker = MagicMock()

        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=(mock_landmarker, None),
        ):
            detector = FaceDetector()
            try:
                mask = detector.segment_person(img)
            finally:
                detector.close()

        assert isinstance(mask, np.ndarray)
        assert mask.shape == (h, w)
        assert mask.dtype == np.float32
        assert np.all(mask == 1.0)

    def test_segment_person_with_empty_confidence_masks_returns_ones(self):
        """If the segmenter ran but produced no confidence masks, the
        function must fall back to a full-ones mask rather than crash."""
        h, w = 100, 200
        img = np.zeros((h, w, 3), dtype=np.uint8)

        mock_result = MagicMock()
        mock_result.confidence_masks = []  # empty

        mock_landmarker = MagicMock()
        mock_segmenter = MagicMock()
        mock_segmenter.segment.return_value = mock_result

        with patch.object(
            FaceDetector, "_create_tasks",
            return_value=(mock_landmarker, mock_segmenter),
        ):
            detector = FaceDetector()
            try:
                mask = detector.segment_person(img)
            finally:
                detector.close()

        assert mask.shape == (h, w)
        assert np.all(mask == 1.0)


class TestDelegateSelection:
    class _Base:
        class Delegate:
            CPU = object()
            GPU = object()

    @pytest.mark.parametrize("value", [None, "", "0", "false", "False", "no", "off"])
    def test_cpu_is_default_and_zero_is_not_gpu_opt_in(self, monkeypatch, value):
        if value is None:
            monkeypatch.delenv("RETUCH_GPU", raising=False)
        else:
            monkeypatch.setenv("RETUCH_GPU", value)

        assert FaceDetector._resolve_delegate(self._Base) is self._Base.Delegate.CPU

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_explicit_gpu_values_opt_in(self, monkeypatch, value):
        monkeypatch.setenv("RETUCH_GPU", value)

        assert FaceDetector._resolve_delegate(self._Base) is self._Base.Delegate.GPU
