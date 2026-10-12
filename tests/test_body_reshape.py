"""Tests for one-click auto body reshape (backlog #7).

Covers ``suggest_body_reshape`` heuristics and the ``auto_body_reshape``
ParamSpec. No MediaPipe model loading — only hand-built PoseContext.
"""

import pytest

from retouch.body_reshape import PoseContext, suggest_body_reshape


def _standing_figure(shoulder_w=0.26, hip_w=0.20):
    """Build 33 plausible normalized landmarks for a standing figure.

    Arms hang at the sides, legs straight down. Shoulder/hip widths are
    configurable so tests can exercise the proportion heuristics.
    """
    lm = [(0.5, 0.5)] * 33
    # Shoulders (11 left, 12 right)
    lm[11] = (0.5 - shoulder_w / 2.0, 0.30)
    lm[12] = (0.5 + shoulder_w / 2.0, 0.30)
    # Elbows (13, 14)
    lm[13] = (0.5 - shoulder_w / 2.0 - 0.02, 0.45)
    lm[14] = (0.5 + shoulder_w / 2.0 + 0.02, 0.45)
    # Wrists (15, 16)
    lm[15] = (0.5 - shoulder_w / 2.0 - 0.04, 0.60)
    lm[16] = (0.5 + shoulder_w / 2.0 + 0.04, 0.60)
    # Hips (23, 24)
    lm[23] = (0.5 - hip_w / 2.0, 0.62)
    lm[24] = (0.5 + hip_w / 2.0, 0.62)
    # Knees (25, 26)
    lm[25] = (0.5 - hip_w / 2.0, 0.78)
    lm[26] = (0.5 + hip_w / 2.0, 0.78)
    # Ankles (27, 28)
    lm[27] = (0.5 - hip_w / 2.0, 0.95)
    lm[28] = (0.5 + hip_w / 2.0, 0.95)
    return lm


def _ctx(shoulder_w=0.26, hip_w=0.20):
    return PoseContext(
        landmarks=_standing_figure(shoulder_w, hip_w),
        visibility=[1.0] * 33,
        detected=True,
        feature_flags={
            "arm_length": True,
            "leg_length": True,
            "torso_width": True,
            "shoulder_width": True,
            "hip_width": True,
        },
    )


class TestSuggestBodyReshape:
    def test_suggest_returns_dict(self):
        out = suggest_body_reshape(_ctx())
        keys = {"arm_length", "leg_length", "torso_width", "shoulder_width", "hip_width"}
        assert set(out.keys()) == keys
        for v in out.values():
            assert 20.0 <= v <= 80.0

    def test_narrow_shoulders_widen(self):
        # Narrow shoulders relative to hips -> shoulder_width should increase.
        out = suggest_body_reshape(_ctx(shoulder_w=0.18, hip_w=0.20))
        assert out["shoulder_width"] > 50.0
        assert out["hip_width"] < 50.0

    def test_wide_shoulders_narrow(self):
        out = suggest_body_reshape(_ctx(shoulder_w=0.34, hip_w=0.20))
        assert out["shoulder_width"] < 50.0
        assert out["hip_width"] > 50.0

    def test_short_legs_lengthen(self):
        # Short legs relative to torso -> leg_length should increase.
        lm = _standing_figure(shoulder_w=0.26, hip_w=0.20)
        # Move ankles up -> short legs.
        lm[27] = (lm[27][0], 0.80)
        lm[28] = (lm[28][0], 0.80)
        ctx = PoseContext(landmarks=lm, visibility=[1.0] * 33, detected=True)
        out = suggest_body_reshape(ctx)
        assert out["leg_length"] > 50.0

    def test_suggest_no_pose_is_neutral(self):
        out = suggest_body_reshape(PoseContext())
        assert out == {
            "arm_length": 50.0,
            "leg_length": 50.0,
            "torso_width": 50.0,
            "shoulder_width": 50.0,
            "hip_width": 50.0,
        }

    def test_low_visibility_neutral(self):
        ctx = _ctx()
        ctx.visibility = [0.1] * 33  # all below threshold
        out = suggest_body_reshape(ctx)
        assert all(v == 50.0 for v in out.values())

    def test_feature_flag_gates_shoulder(self):
        ctx = _ctx(shoulder_w=0.18, hip_w=0.20)
        ctx.feature_flags["shoulder_width"] = False
        ctx.feature_flags["hip_width"] = False
        out = suggest_body_reshape(ctx)
        assert out["shoulder_width"] == 50.0
        assert out["hip_width"] == 50.0


class TestAutoParamSpec:
    def test_param_exists_with_bounds(self):
        from retouch.params import PROCESSING_PARAMS

        spec = next(
            (s for s in PROCESSING_PARAMS if s.name == "auto_body_reshape"),
            None,
        )
        assert spec is not None
        assert spec.recipe_key == "body_reshape.auto"
        assert spec.cli_flag == "auto-body-reshape"
        assert spec.min_val == 0
        assert spec.max_val == 100
        assert spec.default == 0.0
