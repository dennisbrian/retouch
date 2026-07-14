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


def test_suggest_face_recipe_child():
    # Child: eye_width_ratio > 0.43 (ied / w_face)
    img = np.full((128, 128, 3), 128, dtype=np.uint8)
    
    # Setup landmarks: we want bottom_ratio low and aspect_ratio low, or ied / w_face high
    # ied = 40, w_face = 80 => ratio = 0.5 > 0.43
    lm = LandmarksMockList()
    bbox = (10, 10, 80, 80)
    ied = 40.0
    
    recipe = suggest_face_recipe(img, lm, bbox, ied)
    assert recipe == "child"


def test_suggest_face_recipe_female_default():
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
    # Default is female
    assert recipe == "female"


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
