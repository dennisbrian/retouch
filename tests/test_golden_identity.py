"""Identity scorecard (tests/golden_identity.py): each column must fire on a
planted failure and stay quiet on a no-op render.

The planted failures are made directly on the golden inputs, not by the
engine: a horizontal stretch of the face (warp), a 1.3x linear-light gain on
the face skin (lightening) and an output that lost its planted moles
(removed mark). The darker_exposure case (0.45x linear, simulated) checks
that the same edit reads the same on a darker capture.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from tests import golden_identity as gi
from tests import golden_qa as gq

CASES = gq.load_cases()


@pytest.fixture(scope="module")
def engine():
    from retouch.engine import RetouchEngine

    return RetouchEngine()


@pytest.fixture(scope="module")
def faces(engine):
    """Per case: (input, face context, geometry baseline)."""
    out = {}
    for name in ("portrait_studio", "darker_exposure"):
        img = CASES[name].image
        ctx = engine.process(img, recipe="natural").face_contexts[0]
        out[name] = (img, ctx, gi.geometry_baseline(engine._detector, img, ctx.face_data.bbox))
    return out


def _stretch_face(img, bbox, factor):
    """Widen the face horizontally about its centre (a planted warp)."""
    h, w = img.shape[:2]
    x, y, fw, fh = bbox
    cx = x + fw / 2.0
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    weight = np.clip(1.0 - np.abs(xs - cx) / fw, 0, 1) * np.clip(1.0 - np.abs(ys - (y + fh / 2.0)) / fh, 0, 1)
    map_x = xs - (xs - cx) * (1.0 - 1.0 / factor) * weight
    return cv2.remap(img, map_x, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _lighten_skin(img, skin, gain):
    lin = gi._srgb_to_lin(img)
    lin[skin] *= gain
    return np.clip(gi._lin_to_srgb(lin) + 0.5, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("name", ["portrait_studio", "darker_exposure"])
def test_no_op_render_is_quiet(engine, faces, name):
    img, ctx, base = faces[name]
    plan = gi.plan_marks([ctx], img.shape[:2])
    planted = gi.plant_all(img, plan)
    row = gi.face_scorecard(engine._detector, img, img.copy(), ctx, base, plan[0], planted, planted.copy())
    assert base is not None
    assert row["geo_ratio"] <= 1.0
    assert row["l_stops"] == pytest.approx(0.0, abs=1e-6)
    assert row["marks_kept"] == row["marks_total"] == 2 * gi.MARKS_PER_KIND
    assert row["aa6_ssim"] == pytest.approx(1.0, abs=1e-6)
    assert row["identity_flags"] == []


@pytest.mark.parametrize("name", ["portrait_studio", "darker_exposure"])
def test_geometry_column_catches_a_warp(engine, faces, name):
    img, ctx, base = faces[name]
    warped = _stretch_face(img, ctx.face_data.bbox, 1.08)
    row = gi.face_scorecard(engine._detector, img, warped, ctx, base)
    assert row["geo_flag"], row
    assert "geo" in row["identity_flags"]
    assert not row["l_flag"]


@pytest.mark.parametrize("name", ["portrait_studio", "darker_exposure"])
def test_lightness_column_catches_lightening(engine, faces, name):
    img, ctx, base = faces[name]
    skin = cv2.dilate(gi.face_skin(ctx, img.shape[:2]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    light = _lighten_skin(img, skin, 1.3)
    row = gi.face_scorecard(engine._detector, img, light, ctx, base)
    # 1.3x linear = +0.38 stop on any skin tone (less where the gain clips).
    assert row["l_stops"] == pytest.approx(np.log2(1.3), abs=0.06)
    assert row["l_flag"] and "l" in row["identity_flags"]
    assert not row["geo_flag"]


def test_lightness_reads_the_same_on_darker_capture(engine, faces):
    stops = []
    for name in ("portrait_studio", "darker_exposure"):
        img, ctx, base = faces[name]
        skin = cv2.dilate(gi.face_skin(ctx, img.shape[:2]).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        stops.append(gi.lightness_shift(_lighten_skin(img, skin, 1.2), img, gi.face_skin(ctx, img.shape[:2]))["l_stops"])
    assert stops[0] == pytest.approx(stops[1], abs=0.03)


@pytest.mark.parametrize("name", ["portrait_studio", "darker_exposure"])
def test_mark_column_catches_a_removed_mole(engine, faces, name):
    img, ctx, base = faces[name]
    plan = gi.plan_marks([ctx], img.shape[:2])
    planted = gi.plant_all(img, plan)
    # Output lost every mark: the planted render equals the clean one.
    row = gi.face_scorecard(engine._detector, img, img.copy(), ctx, base, plan[0], planted, img.copy())
    assert row["marks_kept"] == 0 and row["mark_flag"]
    assert row["moles_kept"] == f"0/{gi.MARKS_PER_KIND}"
    # One mole healed, the rest kept: exactly that one is counted lost.
    one = [m for m in plan[0] if m.kind == "mole"][0]
    partly = gi.plant_all(img, [[m for m in plan[0] if m is not one]])
    row = gi.face_scorecard(engine._detector, img, img.copy(), ctx, base, plan[0], planted, partly)
    assert row["marks_kept"] == len(plan[0]) - 1 and row["mark_flag"]


def test_planted_marks_sit_on_skin_away_from_features(faces):
    img, ctx, _ = faces["portrait_studio"]
    plan = gi.plan_marks([ctx], img.shape[:2])[0]
    assert len(plan) == 2 * gi.MARKS_PER_KIND
    skin = gi.face_region(ctx, img.shape[:2], "skin") > 0.5
    eyes = (gi.face_region(ctx, img.shape[:2], "left_eye") + gi.face_region(ctx, img.shape[:2], "right_eye")) > 0.3
    for m in plan:
        assert skin[m.y, m.x] and not eyes[m.y, m.x]
