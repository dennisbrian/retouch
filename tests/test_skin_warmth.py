"""Skin Warmth (retouch/skin_warmth.py) and its engine wiring."""

from __future__ import annotations

import dataclasses

import cv2
import numpy as np
import pytest

from retouch.engine import RetouchEngine, build_context
from retouch.grading import ColorGrader
from retouch.params import PROCESSING_PARAMS, resolve_recipe
from retouch.skin_warmth import HUE_BAND, skin_warmth

H = W = 160


def _lab_to_bgr(L, a, b):
    lab = np.zeros((H, W, 3), np.float32)
    lab[..., 0], lab[..., 1], lab[..., 2] = L, a, b
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2BGR), 0, 1).astype(np.float32)


def _lab(img):
    return cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2Lab)


def _scene(L, hue_deg, chroma, seed=0):
    """Face-skin block with blush/texture, a same-colour 'arm', a costume."""
    rng = np.random.default_rng(seed)
    a = chroma * np.cos(np.radians(hue_deg))
    b = chroma * np.sin(np.radians(hue_deg))
    Lm = np.full((H, W), L, np.float32) + rng.normal(0, 0.8, (H, W)).astype(np.float32)
    am = np.full((H, W), a, np.float32) + rng.normal(0, 0.6, (H, W)).astype(np.float32)
    bm = np.full((H, W), b, np.float32) + rng.normal(0, 0.6, (H, W)).astype(np.float32)
    am[40:60, 40:60] += 4.0  # blush
    # costume: saturated blue on the right
    am[:, 120:], bm[:, 120:] = 5.0, -40.0
    img = _lab_to_bgr(Lm, am, bm)
    face = np.zeros((H, W), np.float32)
    face[20:90, 20:100] = 1.0
    person = np.zeros((H, W), np.float32)
    person[:, :] = 1.0
    return img, face, person


def _median_hue(img, sel):
    lab = _lab(img)
    a, b = np.median(lab[..., 1][sel]), np.median(lab[..., 2][sel])
    return float(np.degrees(np.arctan2(b, a))), float(np.hypot(a, b))


FACE = (slice(22, 88), slice(62, 98))  # face skin away from the blush
ARM = (slice(110, 150), slice(20, 100))  # outside face mask, same colour
COSTUME = (slice(0, H), slice(125, W))


class TestSkinWarmth:
    def test_zero_is_identity(self):
        img, face, person = _scene(75, 10, 11)
        assert skin_warmth(img, face, 0, person) is img

    @pytest.mark.parametrize("L", [75.0, 35.0], ids=["lighter", "darker"])
    def test_pink_skin_moves_toward_peach(self, L):
        img, face, person = _scene(L, 8, 11)
        out = skin_warmth(img, face, 100, person)
        h0, _ = _median_hue(img, np.s_[FACE])
        h1, _ = _median_hue(out, np.s_[FACE])
        assert h1 - h0 > 12.0
        assert h1 < HUE_BAND[1]

    def test_same_hue_move_on_lighter_and_darker(self):
        # Both above their chroma floor, so the move is a pure hue turn.
        moves = []
        for L, C in ((75.0, 17.0), (35.0, 11.0)):
            img, face, person = _scene(L, 8, C)
            out = skin_warmth(img, face, 60, person)
            moves.append(_median_hue(out, np.s_[FACE])[0] - _median_hue(img, np.s_[FACE])[0])
        assert moves[0] == pytest.approx(moves[1], abs=3.0)

    def test_darker_skin_gets_no_chroma_push(self):
        # Chroma floor scales with L*: 11 on L 35 is already above it.
        img, face, person = _scene(35.0, 8, 11)
        out = skin_warmth(img, face, 100, person)
        c0 = _median_hue(img, np.s_[FACE])[1]
        c1 = _median_hue(out, np.s_[FACE])[1]
        assert abs(c1 - c0) < 1.0

    def test_texture_and_blush_kept(self):
        img, face, person = _scene(75, 8, 11)
        out = skin_warmth(img, face, 100, person)
        a0, a1 = _lab(img)[..., 1], _lab(out)[..., 1]
        blush0 = a0[40:60, 40:60].mean() - a0[FACE].mean()
        blush1 = a1[40:60, 40:60].mean() - a1[FACE].mean()
        assert blush1 == pytest.approx(blush0, abs=0.8)
        assert np.std(a1[FACE]) == pytest.approx(np.std(a0[FACE]), rel=0.15)

    def test_body_skin_follows_face_costume_does_not(self):
        img, face, person = _scene(75, 8, 11)
        out = skin_warmth(img, face, 100, person)
        arm0, arm1 = _median_hue(img, np.s_[ARM])[0], _median_hue(out, np.s_[ARM])[0]
        assert arm1 - arm0 > 10.0
        assert float(np.abs(out[COSTUME] - img[COSTUME]).max()) < 1.0 / 255

    @pytest.mark.parametrize(
        "hue,chroma,L",
        [(20.0, 2.5, 80.0), (-45.0, 8.0, 65.0), (32.0, 17.0, 75.0), (38.0, 24.0, 36.0)],
        ids=["white_paint", "blue_violet_paint", "already_peach", "darker_warm"],
    )
    def test_left_alone(self, hue, chroma, L):
        img, face, person = _scene(L, hue, chroma)
        out = skin_warmth(img, face, 100, person)
        assert float(np.abs(out[FACE] - img[FACE]).mean()) < 0.5 / 255

    def test_uint8_in_uint8_out(self):
        img, face, person = _scene(75, 8, 11)
        u8 = (img * 255).round().astype(np.uint8)
        out = skin_warmth(u8, face, 60, person)
        assert out.dtype == np.uint8 and not np.array_equal(out, u8)

    def test_registered_param(self):
        spec = next(p for p in PROCESSING_PARAMS if p.name == "skin_warmth")
        assert spec.default == 0
        assert spec.cli_flag == "skin-warmth"
        assert spec.recipe_key == "skin.warmth"


class TestEngineWiring:
    def test_stage_grade_applies_skin_warmth(self):
        eng = RetouchEngine.__new__(RetouchEngine)
        eng._grader = ColorGrader()
        img, face, person = _scene(75, 8, 11)
        ctx = build_context("natural", resolve_recipe("natural"), {})
        lips = np.zeros_like(face)
        off = eng._stage_grade(img.copy(), ctx, face, lips, person)
        on = eng._stage_grade(img.copy(), dataclasses.replace(ctx, skin_warmth=80), face, lips, person)
        assert _median_hue(on, np.s_[FACE])[0] - _median_hue(off, np.s_[FACE])[0] > 8.0
