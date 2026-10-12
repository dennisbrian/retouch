"""Tests for per-face recipe auto-classification and heuristics."""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from retouch.face_params import coerce_face_params, suggest_face_recipe
from retouch.engine import ProcessingContext


class LandmarkMock:
    def __init__(self, x=0.5, y=0.5):
        self.x = x
        self.y = y
        self.z = 0.0


class LandmarksMockList:
    def __init__(self, landmark_dict=None):
        self.landmark = {}
        for idx in range(478):
            self.landmark[idx] = LandmarkMock()
        if landmark_dict:
            for idx, lm in landmark_dict.items():
                self.landmark[idx] = lm


def test_coerce_face_params_auto():
    assert coerce_face_params("auto") == "auto"
    assert coerce_face_params("AUTO") == "auto"
    assert coerce_face_params(None) is None


def test_suggest_face_recipe_is_neutral_even_for_child_like_geometry():
    # Geometry must not select a demographic treatment.
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    
    # Setup landmarks: we want bottom_ratio low and aspect_ratio low, or ied / w_face high
    # ied = 40, w_face = 80 => ratio = 0.5 > 0.43
    lm = LandmarksMockList()
    bbox = (10, 10, 80, 80)
    ied = 40.0
    
    recipe = suggest_face_recipe(img, lm, bbox, ied)
    assert recipe == "natural"


def test_suggest_face_recipe_is_neutral_when_appearance_is_ambiguous():
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    lm = LandmarksMockList()
    bbox = (10, 10, 100, 120)
    ied = 30.0
    
    # Mock landmarks to avoid child detection
    # top_dist = y_mid - y_top = 0.2
    # bottom_dist = y_bottom - y_mid = 0.3
    # bottom_ratio = 1.5 > 0.96
    lm.landmark[10] = LandmarkMock(x=0.5, y=0.1) # forehead top
    lm.landmark[168] = LandmarkMock(x=0.5, y=0.3) # glabella
    lm.landmark[152] = LandmarkMock(x=0.5, y=0.6) # chin
    lm.landmark[151] = LandmarkMock(x=0.5, y=0.2) # forehead center
    lm.landmark[13] = LandmarkMock(x=0.5, y=0.45) # lips
    
    recipe = suggest_face_recipe(img, lm, bbox, ied)
    assert recipe == "natural"


def test_lens_blur_param():
    ctx = ProcessingContext()
    assert hasattr(ctx, "lens_blur")
    assert ctx.lens_blur == 0.0


def test_custom_hsl_calibration_params():
    ctx = ProcessingContext()
    assert hasattr(ctx, "hsl_hue_green")
    assert ctx.hsl_hue_green == 0.0
    assert hasattr(ctx, "calibration_red_hue")
    assert ctx.calibration_red_hue == 0.0


# --- Anchored per-face targets (T5, 2026-09-23) -------------------------------
# DSCF4599: GUI detection at native res lists faces in the opposite order to
# the engine's 2048/800px detection, so an index-keyed override edited the
# other person. Anchored entries bind by position instead.

from retouch.face_params import ANCHOR_KEY, bind_face_params, filter_face_local, make_anchor

_FRAME = (4160, 6240)
_SUBJECT = (1249, 1517, 740, 874)
_BACKGROUND = (500, 146, 832, 893)


def _anchored(bbox, **extra):
    return {"recipe": "natural", ANCHOR_KEY: make_anchor(bbox, _FRAME), **extra}


def test_anchor_binds_to_same_person_when_engine_order_is_swapped():
    # GUI order: 0 = background, 1 = subject. Engine order: 0 = subject.
    fp = {0: _anchored(_BACKGROUND, smooth=100)}
    bound, report = bind_face_params(fp, [_SUBJECT, _BACKGROUND], _FRAME)
    assert set(bound) == {1}
    assert bound[1]["smooth"] == 100
    assert report == [{"selected_index": 0, "bound_index": 1, "iou": 1.0}]


def test_anchor_matches_across_scales():
    # Same faces detected on an 800px-wide proxy of the 4160px source.
    s = 800 / 4160
    proxy = [tuple(int(round(v * s)) for v in b) for b in (_SUBJECT, _BACKGROUND)]
    bound, _ = bind_face_params({1: _anchored(_SUBJECT)}, proxy, (800, int(round(6240 * s))))
    assert set(bound) == {0}


def test_unmatched_anchor_is_dropped_not_applied_by_index():
    fp = {0: _anchored(_BACKGROUND, smooth=100)}
    bound, report = bind_face_params(fp, [_SUBJECT], _FRAME)
    assert bound is None
    assert report[0]["bound_index"] is None


def test_unanchored_entries_keep_index_semantics():
    fp = {0: {"recipe": "natural", "smooth": 50}}
    bound, report = bind_face_params(fp, [_SUBJECT, _BACKGROUND], _FRAME)
    assert bound == fp and report == []


def test_one_to_one_matching():
    fp = {0: _anchored(_SUBJECT, smooth=10), 1: _anchored(_SUBJECT, smooth=20)}
    bound, report = bind_face_params(fp, [_SUBJECT], _FRAME)
    assert list(bound) == [0]
    assert sum(r["bound_index"] is None for r in report) == 1


def test_anchor_key_is_not_a_face_local_param(caplog):
    with caplog.at_level("WARNING"):
        out = filter_face_local(_anchored(_SUBJECT, smooth=10))
    assert out == {"smooth": 10}
    assert ANCHOR_KEY not in caplog.text


def test_anchor_from_other_source_falls_back_to_index():
    from retouch.face_params import ANCHOR_SOURCE_KEY, face_params_for_source

    fp = {0: {**_anchored(_BACKGROUND, smooth=100), ANCHOR_SOURCE_KEY: "/a.jpg"}}
    same = face_params_for_source(fp, "/a.jpg")
    other = face_params_for_source(fp, "/b.jpg")
    assert ANCHOR_KEY in same[0]
    assert ANCHOR_KEY not in other[0] and ANCHOR_SOURCE_KEY not in other[0]
    assert other[0]["smooth"] == 100
