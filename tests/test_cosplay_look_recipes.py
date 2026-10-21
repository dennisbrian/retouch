"""Contract tests for the 2026-10-21 cosplay look set.

cosplay_pastel_soft_v1, cosplay_ink_wash_v1, cosplay_anime_sky_v1 and
cosplay_night_film_v1 are curated creative looks. These tests pin the design
decisions written in their recipe comments, mostly the ones that keep skin of
every tone (and white face paint) out of the look's colour moves.
"""

import pytest

from retouch.engine import RetouchEngine, build_context, resolve_recipe
from retouch.recipe_cookbook import get_recipe_info
from retouch.recipes import (
    CONDITIONAL_RECIPE_NAMES,
    CURATED_RECIPE_NAMES,
    RECOMMENDED_RECIPE_NAMES,
)

LOOK_SET = (
    "cosplay_pastel_soft_v1",
    "cosplay_ink_wash_v1",
    "cosplay_anime_sky_v1",
    "cosplay_night_film_v1",
)
# Hue bands where skin of every tone sits. The HSV band op adds a fixed
# amount, so any cut here greys darker skin first.
SKIN_BANDS = ("red", "orange")


def _ctx(name):
    return build_context(name, resolve_recipe(name), {})


@pytest.mark.parametrize("name", LOOK_SET)
def test_curated_as_creative_recipe(name):
    assert name in CURATED_RECIPE_NAMES
    assert name in CONDITIONAL_RECIPE_NAMES
    assert name not in RECOMMENDED_RECIPE_NAMES


@pytest.mark.parametrize("name", LOOK_SET)
def test_keeps_real_shape_and_drops_pink_blush_patches(name):
    ctx = _ctx(name)
    assert ctx.slimming == 0.0
    assert ctx.nose_blush is False
    assert ctx.under_eye_blush is False


@pytest.mark.parametrize("name", LOOK_SET)
def test_has_cookbook_description_and_cosplay_category(name):
    info = get_recipe_info(name)
    assert info.category == "cosplay"
    assert not info.description.startswith("Extends ")


@pytest.mark.parametrize("name", LOOK_SET)
def test_no_face_grade_and_no_global_desaturation(name):
    # color_harmony grades the face; global saturation and negative vibrance
    # run on the whole frame with no skin mask (the first pastel / ink wash
    # drafts took 22-28% of the chroma out of natural skin that way).
    rec = resolve_recipe(name)
    ctx = _ctx(name)
    assert rec["color_harmony"]["amount"] == 0.0
    assert ctx.saturation >= 0
    assert ctx.vibrance >= -5.0


@pytest.mark.parametrize("name", LOOK_SET)
def test_hsl_never_touches_skin_bands(name):
    ctx = _ctx(name)
    for band in SKIN_BANDS:
        assert getattr(ctx, f"hsl_hue_{band}") == 0.0
        assert getattr(ctx, f"hsl_sat_{band}") == 0.0
        assert getattr(ctx, f"hsl_lum_{band}") == 0.0


def test_pastel_soft_lifts_blacks_and_spares_painted_lavender():
    ctx = _ctx("cosplay_pastel_soft_v1")
    assert ctx.fade_toe > 0
    assert ctx.highlights < 0
    # Blue / purple / magenta cuts greyed white face paint's faint lavender.
    for band in ("blue", "purple", "magenta"):
        assert getattr(ctx, f"hsl_sat_{band}") == 0.0


def test_ink_wash_mutes_greens_and_keeps_reds():
    ctx = _ctx("cosplay_ink_wash_v1")
    assert ctx.hsl_sat_green < 0
    assert ctx.hsl_hue_green > 0
    assert ctx.highlights < 0 and ctx.highlight_rolloff_strength >= 0.45
    assert resolve_recipe("cosplay_ink_wash_v1")["lips"]["tint"] is None


def test_anime_sky_pushes_blue_without_dimming_it():
    ctx = _ctx("cosplay_anime_sky_v1")
    assert ctx.hsl_sat_blue > 0
    # A blue luminance cut dimmed bluish-white face paint by ~4 L.
    assert ctx.hsl_lum_blue == 0.0
    assert ctx.gamut_compress is True
    assert ctx.contrast > 0 and ctx.highlights <= -8.0


def test_night_film_uses_point_light_halation_and_warm_skin():
    ctx = _ctx("cosplay_night_film_v1")
    assert ctx.halation_mode == "point_light"
    assert ctx.halation > 0
    assert ctx.skin_warmth > 0
    post = RetouchEngine._assemble_post_effects(ctx)
    assert post["halation"] == {"mode": "point_light", "intensity": ctx.halation}


def test_classic_halation_recipes_are_unchanged():
    # Every older recipe that sets a plain halation number keeps the classic
    # effect; the point-light glow is opt-in per recipe.
    for name in ("matsuri_glow_v1", "apex_cinema_v1"):
        ctx = _ctx(name)
        assert ctx.halation_mode == "classic"
        assert isinstance(RetouchEngine._assemble_post_effects(ctx)["halation"], float)
