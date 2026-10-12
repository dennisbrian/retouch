"""Regression coverage for the research-calibrated Meitu candidate recipe."""

from retouch.params import recipe_to_params, resolve_recipe
from retouch.recipe_cookbook import get_recipe_info
from retouch.recipes import (
    CONDITIONAL_RECIPE_NAMES,
    CURATED_RECIPE_NAMES,
    EXPERIMENTAL_RECIPE_NAMES,
    RECIPE_UI_CHOICES,
    RECIPES,
    RECOMMENDED_RECIPE_NAMES,
)


def test_meitu_candidate_is_curated_but_not_a_default_recommendation():
    name = "meitu_porcelain_v1"
    assert name in RECIPES
    assert name in CURATED_RECIPE_NAMES
    assert name in CONDITIONAL_RECIPE_NAMES
    assert name in EXPERIMENTAL_RECIPE_NAMES
    assert name not in RECOMMENDED_RECIPE_NAMES


def test_meitu_candidate_keeps_research_calibrated_controls():
    recipe = resolve_recipe("meitu_porcelain_v1")
    source_recipe = RECIPES["meitu_porcelain_v1"]

    assert source_recipe["extends"] == "convention_clear_v1"
    assert recipe["brightness"] == 8.0
    assert recipe["hsl_sat_global"] == -10
    assert recipe["saturation_mode"] == "subtractive"
    assert recipe["skin"]["porcelain"] == 0.55
    assert recipe["skin"]["rosy"] == 0.0
    assert recipe["skin"]["face_exposure"] == 0.0

    gui_values = recipe_to_params("meitu_porcelain_v1")
    assert gui_values["brightness"] == 8
    assert gui_values["hsl_sat_global"] == -10
    assert gui_values["saturation_mode"] == "subtractive"
    assert gui_values["whiten"] == 55
    assert gui_values["whiten_tone"] == "porcelain"


def test_meitu_candidate_is_explicitly_labelled_for_comparison():
    choices = {value: label for label, value in RECIPE_UI_CHOICES}
    assert (
        choices["meitu_porcelain_v1"]
        == "meitu_porcelain_v1  [Experimental / compare first]"
    )

    info = get_recipe_info("meitu_porcelain_v1")
    assert info is not None
    assert info.extends == "convention_clear_v1"
    assert "experimental" in info.description.lower()
    assert "desaturated" in info.description.lower()
