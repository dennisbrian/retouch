"""Contract tests for the 2026-09-25 cosplay scene set.

cosplay_neon_night_v1, cosplay_dark_villain_v1 and cosplay_feed_pop_v1 are
curated scene/creative looks. These tests pin the design decisions written in
their recipe comments so a later tweak can't silently undo them.
"""

import pytest

from retouch.engine import build_context, resolve_recipe
from retouch.recipe_cookbook import get_recipe_info
from retouch.recipes import (
    CONDITIONAL_RECIPE_NAMES,
    CURATED_RECIPE_NAMES,
    RECIPES,
    RECOMMENDED_RECIPE_NAMES,
)

SCENE_SET = (
    "cosplay_neon_night_v1",
    "cosplay_dark_villain_v1",
    "cosplay_feed_pop_v1",
)


def _ctx(name):
    return build_context(name, resolve_recipe(name), {})


@pytest.mark.parametrize("name", SCENE_SET)
def test_curated_as_scene_recipe(name):
    assert name in CURATED_RECIPE_NAMES
    assert name in CONDITIONAL_RECIPE_NAMES
    assert name not in RECOMMENDED_RECIPE_NAMES


@pytest.mark.parametrize("name", SCENE_SET)
def test_keeps_real_shape_and_drops_pink_blush_patches(name):
    ctx = _ctx(name)
    assert ctx.slimming == 0.0
    assert ctx.nose_blush is False
    assert ctx.under_eye_blush is False


@pytest.mark.parametrize("name", SCENE_SET)
def test_has_cookbook_description_and_cosplay_category(name):
    info = get_recipe_info(name)
    assert info.category == "cosplay"
    assert not info.description.startswith("Extends ")


@pytest.mark.parametrize("name", ["cosplay_neon_night_v1", "cosplay_dark_villain_v1"])
def test_no_global_harmony_grade_on_the_face(name):
    # color_harmony grades the face too (see studio_gel_color_v1).
    assert resolve_recipe(name)["color_harmony"]["amount"] == 0.0


def test_neon_night_pulls_skin_back_but_keeps_scene_cast():
    ctx = _ctx("cosplay_neon_night_v1")
    assert ctx.skin_hue_unify > 0
    # No white-balance override: the neon cast is the point of the photo.
    assert not any(k.startswith("white_balance") for k in RECIPES["cosplay_neon_night_v1"])
    assert ctx.bloom_threshold >= 210.0


def test_dark_villain_protects_marks_and_dark_makeup():
    rec = resolve_recipe("cosplay_dark_villain_v1")
    ctx = _ctx("cosplay_dark_villain_v1")
    assert rec["mark_policy"] == "protect_identity"
    assert rec["lips"]["tint"] is None
    assert ctx.blush == 0.0
    assert ctx.blacks < 0
    assert ctx.vibrance < 0
    # Under-eye contour must stay mostly intact.
    assert rec["undereye"]["darken_removal"] <= 0.10
    assert rec["eyes"]["dark_circles"] <= 0.10


def test_feed_pop_uses_vibrance_not_saturation():
    ctx = _ctx("cosplay_feed_pop_v1")
    assert ctx.vibrance > 0
    assert not ctx.saturation
    assert ctx.subject_sharpen > 0
    assert ctx.highlight_rolloff_strength >= 0.40
