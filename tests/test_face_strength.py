"""One strength per face (Impact-style), retouch/face_strength.py.

The scaled set is derived from what the per-face core actually reads, so a
new face op that is in neither table fails here. Engine tests run on the
public golden photos (tests/golden/inputs).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from retouch.face_strength import (
    MAX_FACTOR,
    NOT_SCALED,
    SCALED,
    SWITCHES,
    clamp_factor,
    core_param_reads,
    scale_context,
)

GOLDEN = Path(__file__).parent / "golden" / "inputs"


def test_scaled_set_matches_what_the_core_reads():
    reads = core_param_reads()
    tables = set(SCALED) | set(NOT_SCALED)
    missing = sorted(reads - tables)
    assert not missing, (
        f"per-face core reads {missing}: add each to SCALED (an amount) or "
        f"NOT_SCALED (with the reason) in retouch/face_strength.py"
    )
    stale = sorted(tables - reads)
    assert not stale, f"no longer read by the per-face core: {stale}"
    assert not set(SCALED) & set(NOT_SCALED) - SWITCHES


def test_every_scaled_name_is_a_context_field():
    from retouch.engine import ProcessingContext

    ctx = ProcessingContext()
    assert [n for n in SCALED if not hasattr(ctx, n)] == []


def test_clamp_factor():
    assert clamp_factor(100) == 1.0
    assert clamp_factor(0) == 0.0
    assert clamp_factor(999) == MAX_FACTOR
    assert clamp_factor(-5) == 0.0
    assert clamp_factor("bad") == 1.0
    assert clamp_factor(float("nan")) == 1.0


def test_scale_context_identity_zero_and_boost():
    from retouch.engine import ProcessingContext

    ctx = ProcessingContext()
    ctx.smooth, ctx.whiten, ctx.reshape_eye_size, ctx.mv2_ombre = 40.0, 20.0, -10.0, True
    assert scale_context(ctx, 1.0) is ctx
    zero = scale_context(ctx, 0.0)
    assert zero is not ctx and ctx.smooth == 40.0
    # nose_smooth None means "follow smooth", which is now 0.
    assert all(getattr(zero, n) in (neutral, None) for n, neutral in SCALED.items()
               if n not in ("specular_finish_strength", "specular_recolor"))
    assert zero.mv2_ombre is False
    boost = scale_context(ctx, 1.5)
    assert boost.smooth == 60.0 and boost.whiten == 30.0 and boost.reshape_eye_size == -15.0
    # protections and choices never move
    assert boost.mole_protect == ctx.mole_protect and boost.lip_tint == ctx.lip_tint


def test_specular_default_is_not_an_amount():
    from retouch.engine import ProcessingContext

    ctx = ProcessingContext()
    ctx.smooth = 30.0
    assert scale_context(ctx, 0.0).specular_finish_strength == ctx.specular_finish_strength


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine
    return RetouchEngine()


def _load(name, scale=1.0):
    img = cv2.imread(str(GOLDEN / f"{name}.jpg"))
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return img


_KW = dict(recipe="cosplay_clear_v1", spot_heal=60)


def test_100_percent_is_bit_identical(engine):
    img = _load("portrait_studio")
    base = np.asarray(engine.process(img.copy(), **_KW))
    a = np.asarray(engine.process(img.copy(), face_strength=100, **_KW))
    b = np.asarray(engine.process(img.copy(), face_params={0: {"strength": 100}}, **_KW))
    assert np.array_equal(base, a) and np.array_equal(base, b)


def test_zero_percent_equals_face_ops_off(engine):
    import inspect

    img = _load("portrait_studio")
    kw = dict(recipe="natural", spot_heal=60, smooth=40, whiten=20, eye_enhance=40, lip_enhance=30)
    off = engine.process(img.copy(), face_strength=0, **kw)
    accepted = set(inspect.signature(engine.process).parameters)
    # Amounts process() has no keyword for must already be 0 in this recipe.
    assert all(getattr(off.params, n) == 0 for n in SCALED if n not in accepted)
    # clarity is read by the face core *and* by the whole-frame clarity stage;
    # face strength scales only the face's copy, so keep the global one.
    zeros = {n: 0.0 for n in SCALED if n in accepted and n != "clarity"}
    explicit = np.asarray(engine.process(img.copy(), **dict(kw, **zeros)))
    off = np.asarray(off)
    assert np.abs(off.astype(int) - explicit.astype(int)).max() <= 1
    full = np.asarray(engine.process(img.copy(), **kw))
    assert not np.array_equal(off, full)

def test_strength_on_one_face_leaves_the_other_alone(engine):
    img = _load("two_people", 3.0)
    base = engine.process(img.copy(), **_KW)
    faces = sorted(base.runtime_diagnostics["face_notes"], key=lambda e: e["anchor_bbox_norm"][0])
    assert len(faces) >= 2
    left, right = faces[0], faces[1]
    h, w = img.shape[:2]
    out = engine.process(img.copy(), face_params={
        left["face"]: {"strength": 0, "anchor_bbox_norm": left["anchor_bbox_norm"]},
    }, **_KW)
    x, y, bw, bh = right["anchor_bbox_norm"]
    sl = np.s_[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)]
    diff = np.abs(np.asarray(out).astype(int) - np.asarray(base).astype(int))
    assert diff[sl].max() <= 1, "the other face must keep its retouch"
    x, y, bw, bh = left["anchor_bbox_norm"]
    sl = np.s_[int(y * h):int((y + bh) * h), int(x * w):int((x + bw) * w)]
    assert diff[sl].max() > 3, "the 0% face must lose its retouch"


def test_anchor_keeps_strength_on_the_same_person_when_order_changes(engine):
    """The entry is keyed by the *other* face's index; its anchor wins."""
    img = _load("two_people", 3.0)
    base = engine.process(img.copy(), **_KW)
    entries = base.runtime_diagnostics["face_notes"]
    target, other = entries[1], entries[0]
    out = engine.process(img.copy(), face_params={
        other["face"]: {"strength": 40, "anchor_bbox_norm": target["anchor_bbox_norm"]},
    }, **_KW)
    damped = [e for e in out.runtime_diagnostics["face_notes"]
              if any(n["op"] == "face_strength" for n in e["notes"])]
    assert len(damped) == 1
    assert damped[0]["anchor_bbox_norm"] == pytest.approx(target["anchor_bbox_norm"], abs=0.02)
    assert damped[0]["notes"][0]["factor"] == pytest.approx(0.4)


def test_all_faces_and_per_face_multiply(engine):
    from retouch.engine import ProcessingContext, RetouchEngine

    ctx = ProcessingContext()
    ctx.face_strength = 50
    ctx.face_params = {0: {"strength": 150}, 1: {"strength": 40}}
    assert RetouchEngine._face_strength_factor(ctx, 0) == pytest.approx(0.75)
    assert RetouchEngine._face_strength_factor(ctx, 1) == pytest.approx(0.2)
    assert RetouchEngine._face_strength_factor(ctx, 2) == pytest.approx(0.5)


def test_gui_apply_face_strength_keeps_anchor():
    import gui

    fp = {0: {"recipe": "natural", "anchor_bbox_norm": [0.1, 0.1, 0.2, 0.2]}}
    out, msg = gui.on_apply_face_recipe("0", None, fp, 60)
    assert out[0]["strength"] == 60 and out[0]["anchor_bbox_norm"] == [0.1, 0.1, 0.2, 0.2]
    assert "60%" in msg
    out, _ = gui.on_apply_face_recipe("0", None, out, 100)
    assert "strength" not in out[0]
