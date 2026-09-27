"""Landmark-fallback hair/neck/skin masks, with and without the multiclass segmenter.

These use synthetic class probabilities, so no model is downloaded.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from retouch.detection import _Landmark, _LandmarkCompat
from retouch.parsing import (
    _MC_BODY_SKIN,
    _MC_FACE_SKIN,
    _MC_HAIR,
    _MC_NUM_CLASSES,
    FACE_OVAL,
    FaceParser,
)

H = W = 300
CX, CY, RX, RY = 0.5, 0.4, 0.15, 0.2  # normalized face-oval ellipse


def _oval_landmarks():
    pts = [_Landmark(CX, CY, 0.0) for _ in range(478)]
    n = len(FACE_OVAL)
    for k, idx in enumerate(FACE_OVAL):
        t = 2 * np.pi * k / n - np.pi / 2
        pts[idx] = _Landmark(CX + RX * np.cos(t), CY + RY * np.sin(t), 0.0)
    return _LandmarkCompat(pts)


@pytest.fixture
def parser(monkeypatch):
    monkeypatch.setenv("RETOUCH_CLASS_SEGMENTER", "0")
    return FaceParser()


@pytest.fixture
def img():
    return np.full((H, W, 3), 128, dtype=np.uint8)


def _chin_row():
    return int((CY + RY) * H)


def _probs():
    p = np.zeros((H, W, _MC_NUM_CLASSES), dtype=np.float32)
    p[..., 0] = 1.0
    return p


def _set(p, cls, ys, xs):
    p[ys, xs, :] = 0.0
    p[ys, xs, cls] = 1.0


class TestNoModelHairBand:
    def test_hair_is_not_the_whole_person(self, parser, img):
        person = np.ones((H, W), dtype=np.float32)
        r = parser._landmark_fallback_only(_oval_landmarks(), img, person, 60.0)
        old_hair = np.clip(person - r.face_oval, 0, 1)
        assert (r.hair > 0.5).sum() < 0.35 * (old_hair > 0.5).sum()

    def test_hair_band_stays_above_chin(self, parser, img):
        person = np.ones((H, W), dtype=np.float32)
        r = parser._landmark_fallback_only(_oval_landmarks(), img, person, 60.0)
        assert r.hair[_chin_row() + 5:, :].max() < 0.05
        assert r.neck.max() == 0.0

    def test_no_person_mask_means_no_hair(self, parser, img):
        r = parser._landmark_fallback_only(_oval_landmarks(), img, None, 60.0)
        assert r.hair.max() == 0.0


class TestClassProbRefinement:
    def test_hair_comes_from_segmenter(self, parser, img):
        p = _probs()
        _set(p, _MC_HAIR, slice(10, 60), slice(100, 200))
        r = parser._landmark_fallback_only(_oval_landmarks(), img, None, 60.0, class_probs=p)
        assert r.hair[30, 150] > 0.9
        assert r.hair[250, 150] < 0.05

    def test_bangs_are_cut_out_of_skin(self, parser, img):
        lm = _oval_landmarks()
        base = parser._landmark_fallback_only(lm, img, None, 60.0)
        top = int((CY - RY) * H)
        p = _probs()
        _set(p, _MC_HAIR, slice(top, top + 25), slice(0, W))
        r = parser._landmark_fallback_only(lm, img, None, 60.0, class_probs=p)
        row = top + 15
        assert base.skin[row, 150] > 0.5
        assert r.skin[row, 150] < 0.1
        assert r.skin[int(CY * H), 150] == pytest.approx(base.skin[int(CY * H), 150])

    def test_skin_kept_when_segmenter_disagrees_too_much(self, parser, img):
        lm = _oval_landmarks()
        base = parser._landmark_fallback_only(lm, img, None, 60.0)
        p = _probs()
        _set(p, _MC_HAIR, slice(0, H), slice(0, W))  # "whole face is hair"
        r = parser._landmark_fallback_only(lm, img, None, 60.0, class_probs=p)
        np.testing.assert_allclose(r.skin, base.skin)

    def test_neck_is_jaw_connected_body_skin(self, parser, img):
        chin = _chin_row()
        p = _probs()
        # neck column under the jaw, plus a detached hand lower-left
        _set(p, _MC_BODY_SKIN, slice(chin - 20, chin + 40), slice(125, 175))
        _set(p, _MC_BODY_SKIN, slice(chin + 45, chin + 55), slice(110, 130))
        r = parser._landmark_fallback_only(_oval_landmarks(), img, None, 60.0, class_probs=p)
        assert r.neck[chin + 20, 150] > 0.5
        assert r.neck[chin + 50, 115] < 0.1
        # never inside the face oval
        assert r.neck[int(CY * H), 150] < 0.05

    def test_mismatched_probs_are_ignored(self, parser, img):
        r = parser._landmark_fallback_only(
            _oval_landmarks(), img, None, 60.0, class_probs=np.zeros((10, 10, 6), np.float32)
        )
        assert r.neck.max() == 0.0


class TestSegmenterLifecycle:
    def test_env_switch_disables_segmenter(self, parser, img):
        assert parser._get_class_segmenter() is None
        assert parser._segment_classes(img) is None

    def test_pickle_drops_segmenter(self, parser):
        parser._class_segmenter = object()
        clone = pickle.loads(pickle.dumps(parser))
        assert clone._class_segmenter is None
        assert clone._class_segmenter_lock is not None

    def test_close_is_idempotent(self, parser):
        class _Seg:
            closed = 0

            def close(self):
                _Seg.closed += 1

        parser._class_segmenter = _Seg()
        parser.close()
        parser.close()
        assert _Seg.closed == 1
        assert parser._class_segmenter is None


def test_manifest_pins_multiclass_segmenter():
    entry = json.loads(Path("models/manifest.json").read_text(encoding="utf-8"))["models"]["selfie_multiclass"]
    assert entry["availability"] == "downloadable"
    assert entry["license"] == "Apache-2.0"
    assert "generation=" in entry["url"]
    assert len(entry["sha256"]) == 64 and entry["size_bytes"] > 0


class TestSkinGateUnderPaleWig:
    def test_face_skin_class_vouches_for_a_lost_head(self, parser, img):
        # Person segmenter lost the head (pale wig against a window): the
        # old gate zeroed all fallback skin.
        person = np.zeros((H, W), dtype=np.float32)
        lm = _oval_landmarks()
        no_classes = parser._landmark_fallback_only(lm, img, person, 60.0)
        assert no_classes.skin.max() == 0.0
        p = _probs()
        _set(p, _MC_FACE_SKIN, slice(0, H), slice(0, W))
        r = parser._landmark_fallback_only(lm, img, person, 60.0, class_probs=p)
        ungated = parser._landmark_fallback_only(lm, img, None, 60.0, class_probs=p)
        assert r.skin.max() > 0.9
        np.testing.assert_allclose(r.skin, ungated.skin, atol=1e-6)

    def test_background_inside_oval_still_gated(self, parser, img):
        person = np.zeros((H, W), dtype=np.float32)
        r = parser._landmark_fallback_only(
            _oval_landmarks(), img, person, 60.0, class_probs=_probs()
        )
        assert r.skin.max() == 0.0


class TestSkinMaskV2:
    """Soft confidences, full bangs and blocky segmenter edges (2026-09-26)."""

    def test_soft_confidences_reach_full_strength(self, parser, img):
        # Real segmenter output: ~0.9 face-skin with a couple of percent of
        # every other class leaking everywhere; used to cap skin at ~0.87.
        lm = _oval_landmarks()
        base = parser._landmark_fallback_only(lm, img, None, 60.0)
        p = np.full((H, W, _MC_NUM_CLASSES), 0.02, dtype=np.float32)
        p[..., _MC_FACE_SKIN] = 0.9
        person = np.full((H, W), 0.9, dtype=np.float32)
        r = parser._landmark_fallback_only(lm, img, person, 60.0, class_probs=p)
        c = (int((CY + RY / 2) * H), int(CX * W))
        assert base.skin[c] > 0.99
        assert r.skin[c] > 0.99

    def test_full_bangs_above_eye_line_are_cut_not_ignored(self, parser, img):
        # Hair over the whole upper half of the face is over the old 40% loss
        # guard, which used to keep the bangs as skin.
        lm = _oval_landmarks()
        base = parser._landmark_fallback_only(lm, img, None, 60.0)
        p = _probs()
        _set(p, _MC_HAIR, slice(0, int(CY * H) - 2), slice(0, W))
        r = parser._landmark_fallback_only(lm, img, None, 60.0, class_probs=p)
        upper = int((CY - RY / 2) * H)
        lower = int((CY + RY / 2) * H)
        assert base.skin[upper, 150] > 0.5
        assert r.skin[upper, 150] < 0.1
        assert r.skin[lower, 150] == pytest.approx(base.skin[lower, 150], abs=1e-3)

    def test_clothes_over_the_face_keep_landmark_skin(self, parser, img):
        # Face paint or a mask read as clothes: the segmenter disagrees with
        # the landmarks, so the landmark skin is kept.
        lm = _oval_landmarks()
        base = parser._landmark_fallback_only(lm, img, None, 60.0)
        p = _probs()
        _set(p, 4, slice(0, H), slice(0, W))  # clothes channel
        r = parser._landmark_fallback_only(lm, img, None, 60.0, class_probs=p)
        np.testing.assert_allclose(r.skin, base.skin, atol=1e-5)

    def test_blocky_segmenter_edge_is_smoothed(self, parser, img):
        # 256 px segmenter cells upscale to hard 8 px blocks; the mask must
        # not step by a whole block from one pixel to the next.
        lm = _oval_landmarks()
        p = _probs()
        yy, xx = np.mgrid[0:H, 0:W]
        blocks = ((yy // 8 + xx // 8) % 2 == 0) & (yy > int(CY * H))
        p[blocks, :] = 0.0
        p[blocks, 4] = 1.0  # clothes blocks on the lower face
        r = parser._landmark_fallback_only(lm, img, None, 60.0, class_probs=p)
        step = np.abs(np.diff(r.skin, axis=1))
        assert step.max() < 0.5
