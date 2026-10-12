"""A recipe's fade and split tone cover the whole frame, not just skin.

``_stage_grade`` used to pass the accumulated skin mask as the *apply* mask
of the recipe-level three-way split tone, the preset split tone and the
fade toe, so blacks in hair, costume and background were never lifted and
the tint never reached them. Skin must still get exactly what it got
before; everything else now gets it too.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from retouch.engine import RetouchEngine, build_context
from retouch.grading import ColorGrader
from retouch.params import resolve_recipe


@pytest.fixture(scope="module")
def engine():
    # _stage_grade only touches self._grader; skip __init__ (MediaPipe, pool).
    eng = RetouchEngine.__new__(RetouchEngine)
    eng._grader = ColorGrader()
    return eng


def _ctx(**kw):
    ctx = build_context("natural", resolve_recipe("natural"), {})
    return dataclasses.replace(ctx, **kw)


def _image_and_skin():
    """Dark-to-mid ramp, same on both halves; skin mask on the left half."""
    h, w = 64, 128
    ramp = np.linspace(0.02, 0.5, h, dtype=np.float32)[:, None]
    img = np.repeat(np.repeat(ramp, w, axis=1)[..., None], 3, axis=2)
    img *= np.array([0.9, 1.0, 1.1], np.float32)  # a little warm
    skin = np.zeros((h, w), np.float32)
    skin[:, : w // 2] = 1.0
    return np.clip(img, 0, 1).astype(np.float32), skin


def _run(engine, ctx):
    img, skin = _image_and_skin()
    lips = np.zeros_like(skin)
    out = engine._stage_grade(img.copy(), ctx, skin, lips, None)
    return img, out


class TestFadeToeWholeFrame:
    def test_blacks_off_skin_are_lifted(self, engine):
        img, out = _run(engine, _ctx(fade_toe=50))
        right = (slice(0, 8), slice(80, 128))  # darkest rows, not skin
        assert float((out[right] - img[right]).mean()) > 0.01

    def test_skin_and_non_skin_match(self, engine):
        _, out = _run(engine, _ctx(fade_toe=50))
        np.testing.assert_allclose(out[:, :64], out[:, 64:], atol=1.5 / 255)


class TestSplitToneWholeFrame:
    def test_recipe_three_way_tints_non_skin(self, engine):
        img, out = _run(engine, _ctx(shadow_hue=220.0, shadow_sat=25.0))
        right = (slice(0, 16), slice(80, 128))
        assert float(np.abs(out[right] - img[right]).mean()) > 0.003

    def test_recipe_three_way_skin_and_non_skin_match(self, engine):
        _, out = _run(engine, _ctx(shadow_hue=220.0, shadow_sat=25.0))
        np.testing.assert_allclose(out[:, :64], out[:, 64:], atol=1.5 / 255)

    def test_preset_split_tone_skin_and_non_skin_match(self, engine):
        # "cyberpunk" carries the strongest two-way split_tone of the presets.
        _, base = _run(engine, _ctx(color_grade="cyberpunk", grade_intensity=1.0))
        np.testing.assert_allclose(base[:, :64], base[:, 64:], atol=1.5 / 255)
