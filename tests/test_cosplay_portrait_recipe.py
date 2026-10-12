"""GUI-visible recipe coverage for the portrait-polish showcase preset."""

from retouch.engine import resolve_recipe
from retouch.recipes import RECIPES


def test_cosplay_portrait_polish_recipe_resolves_without_p4_unmix():
    recipe = resolve_recipe("cosplay_portrait_polish_v1")
    assert recipe["body_skin"]["match_face"] > 0
    assert recipe["skin"]["texture_transplant"] > 0
    assert recipe.get("skin", {}).get("makeup_coverage_even", 0) == 0
    assert recipe.get("skin", {}).get("makeup_cake_reduce", 0) == 0


def test_cosplay_portrait_polish_recipe_is_registered_for_gui_choices():
    assert "cosplay_portrait_polish_v1" in RECIPES
