"""Per-face recipe assignment (Slice 1)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from retouch.face_params import (
    FACE_LOCAL_PARAM_NAMES,
    coerce_face_params,
    filter_face_local,
    load_face_params_json,
    resolve_face_context,
)
from retouch.engine import ProcessingContext, build_context
from retouch.params import resolve_recipe
from retouch.geometry import FaceReshaper


def test_filter_face_local_keeps_smooth_drops_brightness():
    out = filter_face_local({"smooth": 70, "brightness": 10, "recipe": "x"})
    assert out == {"smooth": 70}
    assert "brightness" not in out


def test_coerce_face_params_string_keys():
    c = coerce_face_params({"0": {"smooth": 50}, "2": {"whiten": 10}})
    assert c == {0: {"smooth": 50}, 2: {"whiten": 10}}


def test_resolve_identity_same_object():
    base = ProcessingContext()
    assert resolve_face_context(base, None) is base
    assert resolve_face_context(base, {}) is base


def test_resolve_overlay_wins():
    base = ProcessingContext(smooth=30.0)
    out = resolve_face_context(base, {"smooth": 80.0})
    assert out.smooth == 80.0
    assert base.smooth == 30.0  # base unchanged


def test_resolve_recipe_expand_face_local_only():
    base = ProcessingContext(smooth=10.0, color_grade="natural", brightness=5.0)
    out = resolve_face_context(base, {"recipe": "cosplay", "smooth": 50.0})
    assert out.smooth == 50.0
    # Global grade/brightness must stay from base (not cosplay recipe)
    assert out.color_grade == base.color_grade
    assert out.brightness == base.brightness


def test_resolve_global_key_stripped(caplog):
    base = ProcessingContext(smooth=20.0)
    out = resolve_face_context(base, {"brightness": 99.0})
    assert out is base or out.smooth == 20.0
    assert out.brightness == base.brightness


def test_load_face_params_json(tmp_path):
    p = tmp_path / "faces.json"
    p.write_text(json.dumps({"0": {"recipe": "natural", "smooth": 40}}), encoding="utf-8")
    d = load_face_params_json(p)
    assert d[0]["smooth"] == 40
    assert d[0]["recipe"] == "natural"


def test_face_local_includes_reshape_and_cosplay():
    assert "reshape_eye_size" in FACE_LOCAL_PARAM_NAMES
    assert "cosplay_wig_lace_blend" in FACE_LOCAL_PARAM_NAMES
    assert "smooth" in FACE_LOCAL_PARAM_NAMES


def test_build_context_still_works():
    rec = resolve_recipe("natural")
    ctx = build_context("natural", rec, {"smooth": 55.0})
    assert ctx.smooth == 55.0


def _reshape_ctx(slimming: float) -> SimpleNamespace:
    keys = (
        "eye_size", "eye_distance", "nose_width", "nose_length", "jaw_width",
        "chin_length", "mouth_size", "smile", "forehead", "jaw_width_l",
        "jaw_width_r", "nose_width_l", "nose_width_r", "eye_size_l",
        "eye_size_r", "neck_width", "neck_length",
    )
    return SimpleNamespace(
        slimming=slimming,
        **{f"reshape_{k}": 0.0 for k in keys},
    )


def test_reshape_b_lite_different_slimming_per_face():
    """Two faces: face0 slim, face1 zero via face_ctxs."""
    from tests.test_geometry import _make_face_data

    reshaper = FaceReshaper()
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    # Gradient so warps produce visible change
    grad = np.tile(np.linspace(0, 255, 400, dtype=np.uint8), (400, 1))
    img = np.stack([grad] * 3, axis=-1)
    face0 = _make_face_data(150, 150, 80, 100)
    face1 = _make_face_data(300, 300, 80, 100)
    out_split = reshaper.reshape(
        img, [face0, face1], _reshape_ctx(0.0),
        face_ctxs=[_reshape_ctx(80.0), _reshape_ctx(0.0)],
    )
    out_none = reshaper.reshape(img, [face0, face1], _reshape_ctx(0.0))
    assert out_split.shape == img.shape
    # With only face0 slim, image should change vs zero-slimming global
    assert not np.array_equal(out_split, out_none)


def test_ctx_for_face_identity_without_params(engine):
    from retouch.engine import ProcessingContext
    ctx = ProcessingContext()
    assert engine._ctx_for_face(ctx, 0) is ctx


def test_ctx_for_face_overlay(engine):
    ctx = ProcessingContext(smooth=20.0)
    ctx.face_params = {0: {"smooth": 90.0}}
    c0 = engine._ctx_for_face(ctx, 0)
    c1 = engine._ctx_for_face(ctx, 1)
    assert c0.smooth == 90.0
    assert c1 is ctx
