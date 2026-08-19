"""Dual-scale augmentation tests (legacy FaceMesh path).

Covers `_dual_scale_augment_legacy` and its wiring inside `detect()`:

- gate: only runs when the main (full-frame) pass found >= 1 face and the
  image is larger than the 1024 dual-scale target
- addition: a 1024-pass face absent from the 2048 result is appended, with
  landmarks rescaled to full-image geometry
- dedup: a 1024-pass face overlapping an existing face (IoU > 0.5) is NOT
  added (the main pass's higher-res landmarks win)
- person gate: additions require >= 0.5 person-mask coverage of the box's
  central region (poster/banner FP guard — measured poles on the DSCF
  corpus: posters 0.000 vs subjects 1.000)
- no-gate-fallback: if the person segmenter is unavailable (raises), the
  augmentation declines to add anything rather than risk ungated FPs
"""

from __future__ import annotations

import math
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from retouch.detection import FaceDetector, FaceData, _LandmarkCompat, inter_eye_distance

from tests.test_detection import _build_mock_landmarks


def _spread_landmarks(eye_sep=0.1, cx=0.5, cy=0.5, spread=0.08):
    """Landmarks with real spatial spread (an ellipse), so bbox height/width
    are non-degenerate — required for IoU and person-gate crop logic.
    Iris indices 468/473 keep the known horizontal separation."""
    lm = []
    for i in range(478):
        theta = 2.0 * math.pi * (i / 478.0)
        if i == 468:
            lm.append(MagicMock(x=cx - eye_sep / 2.0, y=cy, z=0.0))
        elif i == 473:
            lm.append(MagicMock(x=cx + eye_sep / 2.0, y=cy, z=0.0))
        else:
            lm.append(MagicMock(
                x=min(0.99, max(0.01, cx + spread * math.cos(theta))),
                y=min(0.99, max(0.01, cy + spread * 0.6 * math.sin(theta))),
                z=0.0,
            ))
    return lm


def _legacy_detector(mesh_results, person_mask=None, segmenter_raises=False):
    """FaceDetector with a mocked legacy FaceMesh + segmenter.

    mesh_results: list of results returned by successive _legacy_mesh.process
    calls (call 1 = full-frame pass; call 2 = dual-scale 1024 pass).
    person_mask: array returned by segment_person (full-image size), or None
    for "unavailable" behavior.
    """
    detector = FaceDetector.__new__(FaceDetector)
    detector.max_faces = 25
    detector.min_confidence = 0.4
    detector.available = True
    detector.backend_name = "mediapipe_legacy"
    detector.unavailable_reason = None
    detector._landmarker = None
    detector._segmenter = None

    mesh = MagicMock()
    mesh.process.side_effect = mesh_results
    detector._legacy_mesh = mesh

    if segmenter_raises:
        seg = MagicMock()
        seg.process.side_effect = RuntimeError("segmenter broken")
        detector._legacy_segmenter = seg
    else:
        def seg_process(img_rgb):
            res = MagicMock()
            res.segmentation_mask = person_mask
            return res
        seg = MagicMock()
        seg.process.side_effect = seg_process
        detector._legacy_segmenter = seg
    return detector


def _legacy_result(landmark_lists):
    """Wrap each list of mock landmarks in a `.landmark` container, matching
    the legacy Solutions API's `multi_face_landmarks[i].landmark` shape that
    `_detach_landmarks` reads."""
    wrapped = []
    for lst in landmark_lists or []:
        holder = types.SimpleNamespace(landmark=lst)
        wrapped.append(holder)
    res = MagicMock()
    res.multi_face_landmarks = wrapped
    return res


def _make_img(w=2048, h=1365, seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


class TestDualScaleAugment:
    def test_addition_of_missed_face(self):
        """Main pass finds a face at the image centre; the 1024 pass finds
        the same face AND one at top-left. The top-left face must be added."""
        img = _make_img()
        main_face = _spread_landmarks(eye_sep=0.1)
        extra = _spread_landmarks(eye_sep=0.1, cx=0.18, cy=0.14)
        mask = np.zeros((1365, 2048), dtype=np.float32)
        mask[:, :] = 1.0  # everything "person" — gate passes trivially

        det = _legacy_detector(
            [_legacy_result([main_face]), _legacy_result([main_face, extra])],
            person_mask=mask,
        )
        try:
            faces = det.detect(img)
        finally:
            det.close()

        assert len(faces) == 2
        centers = sorted(
            (f.bbox[0] + f.bbox[2] / 2.0, f.bbox[1] + f.bbox[3] / 2.0)
            for f in faces
        )
        assert centers[0][0] < 500  # the shifted face made it in
        assert all(f.confidence_source == "mediapipe_presence_unavailable" for f in faces)

    def test_duplicate_not_added(self):
        """1024-pass face overlapping the main-pass face (IoU > 0.5) is a
        re-detection of the same face — only the main result survives."""
        img = _make_img()
        face = _build_mock_landmarks(eye_sep=0.1)
        mask = np.ones((1365, 2048), dtype=np.float32)

        det = _legacy_detector(
            [_legacy_result([face]), _legacy_result([face])],
            person_mask=mask,
        )
        try:
            faces = det.detect(img)
        finally:
            det.close()
        assert len(faces) == 1

    def test_person_gate_blocks_poster_fp(self):
        """A 1024-pass-only face on a region with zero person coverage (an
        anime poster) must NOT be added."""
        img = _make_img()
        main_face = _build_mock_landmarks(eye_sep=0.1)
        poster = _build_mock_landmarks(eye_sep=0.1)
        for lm in poster:
            lm.x = lm.x * 0.3 + 0.6
            lm.y = lm.y * 0.3 + 0.05
        mask = np.zeros((1365, 2048), dtype=np.float32)
        # person covers only the centre (main face) region
        mask[400:900, 700:1300] = 1.0

        det = _legacy_detector(
            [_legacy_result([main_face]), _legacy_result([main_face, poster])],
            person_mask=mask,
        )
        try:
            faces = det.detect(img)
        finally:
            det.close()
        assert len(faces) == 1  # poster addition gated out

    def test_person_gate_accepts_on_person_face(self):
        """A 1024-pass-only face fully on the person mask is added."""
        img = _make_img()
        main_face = _spread_landmarks(eye_sep=0.1)
        extra = _spread_landmarks(eye_sep=0.1, cx=0.62, cy=0.62)
        mask = np.zeros((1365, 2048), dtype=np.float32)
        mask[:, :] = 1.0

        det = _legacy_detector(
            [_legacy_result([main_face]), _legacy_result([main_face, extra])],
            person_mask=mask,
        )
        try:
            faces = det.detect(img)
        finally:
            det.close()
        assert len(faces) == 2

    def test_segmenter_failure_adds_nothing(self):
        """If the person segmenter errors, the augmentation must return []
        (declining to add ungated candidates), and detect() still returns
        the main-pass faces."""
        img = _make_img()
        main_face = _build_mock_landmarks(eye_sep=0.1)
        extra = _build_mock_landmarks(eye_sep=0.1)
        for lm in extra:
            lm.x = lm.x * 0.3 + 0.02

        det = _legacy_detector(
            [_legacy_result([main_face]), _legacy_result([main_face, extra])],
            segmenter_raises=True,
        )
        try:
            faces = det.detect(img)
        finally:
            det.close()
        assert len(faces) == 1

    def test_small_image_skips_dual_scale(self):
        """max(w, h) <= 1024: the dual-scale pass cannot see anything the
        main pass cannot — it must not run (single mesh.process call)."""
        img = _make_img(w=800, h=600)
        face = _build_mock_landmarks(eye_sep=0.1)
        mask = np.ones((600, 800), dtype=np.float32)

        det = _legacy_detector([_legacy_result([face])], person_mask=mask)
        try:
            faces = det.detect(img)
            process_calls = det._legacy_mesh.process.call_count
        finally:
            det.close()
        assert len(faces) == 1
        assert process_calls == 1

    def test_zero_face_main_pass_uses_tiling_not_dual_scale(self):
        """Main pass finds nothing → the existing tiled fallback runs; the
        dual-scale augment does NOT run (it augments, not rescues)."""
        img = _make_img()
        det = _legacy_detector([_legacy_result([]), _legacy_result([])])
        det._detect_tiled_legacy = MagicMock(return_value=[])
        try:
            faces = det.detect(img)
            tiling_calls = det._detect_tiled_legacy.call_count
            process_calls = det._legacy_mesh.process.call_count
        finally:
            det.close()
        assert faces == []
        assert tiling_calls == 1
        # only the main full-frame call ran
        assert process_calls == 1

    def test_addition_landmarks_are_full_image_normalized(self):
        """1024-pass landmarks are normalized [0,1] and geometry-identical
        to full-image landmarks — the addition's bbox must land in the same
        image-relative place (within the resize's rounding)."""
        img = _make_img()
        main_face = _build_mock_landmarks(eye_sep=0.1)
        extra = _build_mock_landmarks(eye_sep=0.1)
        for lm in extra:
            lm.x = lm.x * 0.4 + 0.5
            lm.y = lm.y * 0.4 + 0.4
        mask = np.ones((1365, 2048), dtype=np.float32)

        det = _legacy_detector(
            [_legacy_result([main_face]), _legacy_result([main_face, extra])],
            person_mask=mask,
        )
        try:
            faces = det.detect(img)
        finally:
            det.close()

        added = max(faces, key=lambda f: f.bbox[0])
        # landmark x range 0.5..0.9 → bbox near [1024..1843]
        assert 900 < added.bbox[0] < 1150
        # ied scales with image width: eye_sep 0.1 * 2048 = 204.8
        assert added.ied == pytest.approx(0.1 * 2048, abs=25)
