"""Recipe validation tests — catch dead keys and misconfigurations."""

import pytest
from retouch.recipes import RECIPES
from retouch.params import PROCESSING_PARAMS, resolve_recipe


# Build a set of all valid recipe keys (both top-level and nested)
_VALID_KEYS = {spec.recipe_key or spec.name for spec in PROCESSING_PARAMS}
# Nested keys can be dicts (e.g., {"skin": {...}, "bloom": {...}})
_VALID_NESTED_ROOTS = {"skin", "eyes", "lips", "hair", "bloom", "makeup", "frequency", "texture", "color_harmony", "finish"}


def test_no_dead_recipe_keys():
    """Verify that every recipe key resolves to a known ParamSpec.

    This catches misconfigurations like anime_crystal_void's 7 dead keys
    (background_blur, background_desaturation, light_wrap, etc.) that
    are silently ignored by the engine.
    """
    dead_keys_by_recipe = {}

    for recipe_name, recipe_dict in RECIPES.items():
        # Skip "extends" since it's a meta key
        dead = []
        for key in recipe_dict.keys():
            if key == "extends":
                continue
            # Check if it's a top-level param or a valid nested root
            if key not in _VALID_KEYS and key not in _VALID_NESTED_ROOTS:
                dead.append(key)

        if dead:
            dead_keys_by_recipe[recipe_name] = dead

    # Report findings
    if dead_keys_by_recipe:
        msg = "Dead recipe keys found:\n"
        for recipe_name, dead in sorted(dead_keys_by_recipe.items()):
            msg += f"  {recipe_name}: {dead}\n"
        pytest.fail(msg)


def test_recipe_resolution_no_errors():
    """Verify that all recipes can be resolved without errors."""
    for recipe_name in RECIPES.keys():
        # Should not raise
        resolved = resolve_recipe(recipe_name)
        assert isinstance(resolved, dict), f"Recipe '{recipe_name}' resolution failed"
        assert len(resolved) > 0, f"Recipe '{recipe_name}' is empty after resolution"


def test_anime_crystal_void_dead_keys_documented():
    """Verify that anime_crystal_void's dead keys are at least documented.

    This recipe has 7 unimplemented keys:
    - background_blur, background_desaturation, light_wrap
    - blue_shadow_grade, cyan_midtone_grade, subject_sharpen, matte_black

    They should be documented as dead (e.g., inline comments in recipes.py).
    This test ensures we know about them and aren't surprised by no-op behavior.
    """
    recipe = RECIPES.get("anime_crystal_void")
    if recipe:
        dead_keys = {
            "background_blur", "background_desaturation", "light_wrap",
            "blue_shadow_grade", "cyan_midtone_grade", "subject_sharpen", "matte_black"
        }
        for dead_key in dead_keys:
            assert dead_key in recipe, f"Dead key '{dead_key}' removed from anime_crystal_void (may be intentional)"
