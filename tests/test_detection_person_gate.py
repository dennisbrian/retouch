"""Main-pass person gate (background false-positive veto) tests.

Covers `FaceDetector._person_gate` and its wiring inside `detect()`:

- a main-pass detection with zero person-mask coverage (bokeh, night sky,
  wall, costume) is dropped
- a detection on the person mask is kept
- the gate fails OPEN: if the segmenter raises, every detection is kept
  (the candidates include the subject, so dropping on error is worse)
- when the gate drops every main-pass detection, the tiled fallback still
  runs to look for the real subject
- the tasks/RetinaFace path is gated too
- a real face the full-frame mask loses (near-white wig against a blown
  window, RESEARCH_PERSON_GATE_WIG_FALSENEG_2026_09_25) is kept when a
  face-scale crop reads as face skin; a background FP is still dropped
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np

from retouch.detection import FaceDetector, FaceData, _LandmarkCompat
from tests.test_detection_dual_scale import (
    _legacy_detector,
    _legacy_result,
    _make_img,
    _spread_landmarks,
)

W, H = 2048, 1365


def _face_at(cx, cy):
    return _spread_landmarks(eye_sep=0.1, cx=cx, cy=cy)


def _detect(det, img):
    try:
        return det.detect(img)
    finally:
        det.close()


class TestMainPassPersonGate:
    def test_background_fp_is_dropped(self):
        """Subject at the centre on the person mask + an FP at top-right on
        background: only the subject survives."""
        subject, fp = _face_at(0.5, 0.5), _face_at(0.85, 0.15)
        mask = np.zeros((H, W), dtype=np.float32)
        mask[400:1000, 700:1350] = 1.0  # person covers the subject only
        det = _legacy_detector(
            [_legacy_result([subject, fp]), _legacy_result([])], person_mask=mask
        )
        faces = _detect(det, _make_img())
        assert len(faces) == 1
        x, y, bw, bh = faces[0].bbox
        assert abs((x + bw / 2) - W * 0.5) < 50

    def test_face_on_person_is_kept(self):
        mask = np.ones((H, W), dtype=np.float32)
        det = _legacy_detector(
            [_legacy_result([_face_at(0.3, 0.4), _face_at(0.7, 0.6)]),
             _legacy_result([])],
            person_mask=mask,
        )
        assert len(_detect(det, _make_img())) == 2

    def test_segmenter_failure_keeps_all_faces(self):
        """Fail open: a broken segmenter must not drop the subject."""
        det = _legacy_detector(
            [_legacy_result([_face_at(0.5, 0.5), _face_at(0.85, 0.15)]),
             _legacy_result([])],
            segmenter_raises=True,
        )
        assert len(_detect(det, _make_img())) == 2

    def test_all_dropped_falls_back_to_tiling(self):
        """The only main-pass hit is a background FP → it is dropped and the
        tiled fallback runs; a tiled find on the person is returned."""
        mask = np.zeros((H, W), dtype=np.float32)
        mask[400:1000, 700:1350] = 1.0
        det = _legacy_detector(
            [_legacy_result([_face_at(0.85, 0.15)])], person_mask=mask
        )
        tiled_face = FaceData(
            landmarks=_LandmarkCompat([]), bbox=(900, 550, 250, 250), ied=60.0
        )
        det._detect_tiled_legacy = MagicMock(return_value=[tiled_face])
        faces = _detect(det, _make_img())
        assert det._detect_tiled_legacy.call_count == 1
        assert faces == [tiled_face]

    def test_tiled_fp_is_gated_too(self):
        mask = np.zeros((H, W), dtype=np.float32)
        det = _legacy_detector([_legacy_result([])], person_mask=mask)
        det._detect_tiled_legacy = MagicMock(return_value=[FaceData(
            landmarks=_LandmarkCompat([]), bbox=(100, 100, 250, 250), ied=60.0
        )])
        assert _detect(det, _make_img()) == []

    def test_degenerate_bbox_is_kept(self):
        """No central region to measure → no evidence → keep (fail open)."""
        det = FaceDetector.__new__(FaceDetector)
        det._face_skin_segmenter, det._face_skin_segmenter_failed = None, True
        det.segment_person = MagicMock(return_value=np.zeros((H, W), np.float32))
        face = FaceData(landmarks=_LandmarkCompat([]), bbox=(10, 10, 0, 0), ied=1.0)
        kept, _ = det._person_gate(_make_img(), [face])
        assert kept == [face]

    def test_mask_resized_to_image(self):
        """A lower-resolution mask is resized, not indexed off-scale."""
        det = FaceDetector.__new__(FaceDetector)
        det._face_skin_segmenter, det._face_skin_segmenter_failed = None, True
        small = np.zeros((H // 4, W // 4), np.float32)
        small[: H // 8, : W // 8] = 1.0  # person in the top-left quadrant only
        det.segment_person = MagicMock(return_value=small)
        on = FaceData(landmarks=_LandmarkCompat([]), bbox=(100, 100, 300, 300), ied=60.0)
        off = FaceData(landmarks=_LandmarkCompat([]), bbox=(1500, 900, 300, 300), ied=60.0)
        kept, mask = det._person_gate(_make_img(), [on, off])
        assert kept == [on]
        assert mask.shape == (H, W)


class TestTasksPathPersonGate:
    def test_tasks_path_drops_background_fp(self):
        """RetinaFace unavailable → MediaPipe tasks landmarker fallback; its
        background FP is gated at the end of detect()."""
        det = FaceDetector.__new__(FaceDetector)
        det._face_skin_segmenter, det._face_skin_segmenter_failed = None, True
        det.max_faces, det.min_confidence = 25, 0.4
        det.available, det.backend_name = True, "tasks"
        det._legacy_mesh = None
        det._legacy_segmenter = None
        img = _make_img(w=1000, h=800)  # <= 1024: single tasks call
        subject, fp = _face_at(0.5, 0.5), _face_at(0.85, 0.15)
        det._landmarker = MagicMock()
        det._landmarker.detect.return_value = types.SimpleNamespace(
            face_landmarks=[subject, fp]
        )
        mask = np.zeros((800, 1000), np.float32)
        mask[250:550, 330:670] = 1.0
        det._segmenter = MagicMock()
        det.segment_person = MagicMock(return_value=mask)
        import sys
        saved = sys.modules.get("retinaface")
        sys.modules["retinaface"] = None  # force ImportError → fallback
        try:
            faces = det.detect(img)
        finally:
            if saved is None:
                sys.modules.pop("retinaface", None)
            else:
                sys.modules["retinaface"] = saved
        assert len(faces) == 1
        x, y, bw, bh = faces[0].bbox
        assert abs((x + bw / 2) - 500) < 40


def _skin_segmenter(skin_fn):
    """Fake multiclass segmenter: channel 3 (face skin) = skin_fn(crop_h, crop_w)."""
    def segment(image):
        ch, cw = image.numpy_view().shape[:2]
        masks = [np.zeros((ch, cw), np.float32) for _ in range(6)]
        masks[3] = skin_fn(ch, cw).astype(np.float32)
        return types.SimpleNamespace(confidence_masks=[
            types.SimpleNamespace(numpy_view=(lambda m=m: m)) for m in masks
        ])
    seg = MagicMock()
    seg.segment.side_effect = segment
    return seg


def _centre_skin(ch, cw):
    """Face skin over the middle of the crop, as on a real face crop."""
    m = np.zeros((ch, cw), np.float32)
    m[ch // 4: 3 * ch // 4, cw // 4: 3 * cw // 4] = 1.0
    return m


class TestFaceSkinRescue:
    def _det(self, faces, mask, seg):
        det = _legacy_detector([_legacy_result(faces), _legacy_result([])], person_mask=mask)
        det._face_skin_segmenter, det._face_skin_segmenter_failed = seg, False
        return det

    def test_lost_head_face_is_kept(self):
        """DSCF3773 pattern: torso on the mask, head at 0.000 coverage. The
        face crop reads as face skin, so the subject survives."""
        mask = np.zeros((H, W), np.float32)
        mask[900:, 600:1450] = 1.0  # torso only; the head region is empty
        seg = _skin_segmenter(_centre_skin)
        det = self._det([_face_at(0.5, 0.5)], mask, seg)
        faces = _detect(det, _make_img())
        assert len(faces) == 1
        assert seg.segment.call_count == 1

    def test_background_fp_without_face_skin_is_dropped(self):
        """Bokeh/wall/sky FP: no person and no face skin → still dropped."""
        mask = np.zeros((H, W), np.float32)
        mask[400:1000, 700:1350] = 1.0
        seg = _skin_segmenter(lambda ch, cw: np.zeros((ch, cw)))
        det = self._det([_face_at(0.5, 0.5), _face_at(0.85, 0.15)], mask, seg)
        faces = _detect(det, _make_img())
        assert len(faces) == 1
        x, y, bw, bh = faces[0].bbox
        assert abs((x + bw / 2) - W * 0.5) < 50

    def test_rescue_threshold_is_the_boundary(self):
        """Face skin just under the threshold drops; at it, keeps."""
        det = FaceDetector.__new__(FaceDetector)
        det.segment_person = MagicMock(return_value=np.zeros((H, W), np.float32))
        face = FaceData(landmarks=_LandmarkCompat([]), bbox=(900, 500, 300, 300), ied=60.0)
        for value, expect in ((0.449, []), (0.45, [face])):
            det._face_skin_coverage = MagicMock(return_value=value)
            kept, _ = det._person_gate(_make_img(), [face])
            assert kept == expect, value

    def test_faces_on_the_mask_skip_the_second_opinion(self):
        """The rescue only runs for gate rejects, so ordinary frames pay
        nothing for it."""
        seg = _skin_segmenter(_centre_skin)
        det = self._det([_face_at(0.5, 0.5)], np.ones((H, W), np.float32), seg)
        assert len(_detect(det, _make_img())) == 1
        assert seg.segment.call_count == 0

    def test_unavailable_segmenter_keeps_old_behaviour(self, monkeypatch):
        """Model missing/offline: the reject is dropped as before, and the
        failure is remembered rather than retried per face."""
        import retouch.model_fetch as model_fetch

        calls = []

        def boom(name):
            calls.append(name)
            raise RuntimeError("offline")

        monkeypatch.setattr(model_fetch, "get_model_path", boom)
        det = FaceDetector.__new__(FaceDetector)
        det._face_skin_segmenter, det._face_skin_segmenter_failed = None, False
        det.segment_person = MagicMock(return_value=np.zeros((H, W), np.float32))
        faces = [FaceData(landmarks=_LandmarkCompat([]), bbox=(b, 500, 300, 300), ied=60.0)
                 for b in (300, 1200)]
        kept, _ = det._person_gate(_make_img(), faces)
        assert kept == []
        assert calls == ["selfie_multiclass"]

    def test_dual_scale_addition_uses_the_rescue(self):
        """A 1024-pass-only face the mask lost is added when it is face skin."""
        mask = np.zeros((H, W), np.float32)
        mask[400:1000, 300:800] = 1.0  # main subject at left only
        main, extra = _face_at(0.27, 0.5), _face_at(0.75, 0.5)
        det = _legacy_detector(
            [_legacy_result([main]), _legacy_result([main, extra])], person_mask=mask
        )
        det._face_skin_segmenter, det._face_skin_segmenter_failed = (
            _skin_segmenter(_centre_skin), False
        )
        assert len(_detect(det, _make_img())) == 2
