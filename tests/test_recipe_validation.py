"""Recipe validation tests — catch dead keys and misconfigurations."""

import pytest
from retouch.recipes import RECIPES
from retouch.params import PROCESSING_PARAMS, resolve_recipe


# Build a set of all valid recipe keys (both top-level and nested).
# Include engine_recipe_key as well: when a param's engine-side lookup path
# differs from its recipe_key (e.g. relight_azimuth has recipe_key
# "skin.relight_azimuth" but engine_recipe_key "relight_azimuth"), recipes
# must place the value where the engine reads it — at the engine key — or it
# is silently ignored. Without this, the test would falsely flag the
# correctly-placed top-level key as "dead".
_VALID_KEYS = {spec.recipe_key or spec.name for spec in PROCESSING_PARAMS}
_VALID_KEYS.update(
    spec.engine_recipe_key for spec in PROCESSING_PARAMS
    if spec.engine_recipe_key is not None
)
# Nested keys can be dicts (e.g., {"skin": {...}, "bloom": {...}})
_VALID_NESTED_ROOTS = {"skin", "eyes", "lips", "hair", "bloom", "makeup", "frequency", "texture", "color_harmony", "finish", "body_skin", "background", "harmony"}


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


def test_anime_crystal_void_keys_wired():
    """Guard: anime_crystal_void's 7 previously-dead keys are now wired.

    These keys (background_blur, background_desaturation, light_wrap,
    blue_shadow_grade, cyan_midtone_grade, subject_sharpen, matte_black)
    were aspirational no-ops silently ignored by the engine for years.
    T1 wires them as real ParamSpecs (recipe path ``background.<key>``)
    backed by ``BackgroundReplacer``. This test ensures:
      (a) all 7 keys are present in the recipe under ``background.*``
      (b) each key resolves to a non-zero value (the look is functional)
      (c) each key maps to a registered ParamSpec (no longer dead)
    """
    recipe = RECIPES.get("anime_crystal_void")
    assert recipe is not None, "anime_crystal_void recipe missing"
    bg = recipe.get("background")
    assert isinstance(bg, dict), "anime_crystal_void missing 'background' dict"

    wired_keys = {
        "background_blur", "background_desaturation", "light_wrap",
        "blue_shadow_grade", "cyan_midtone_grade", "subject_sharpen", "matte_black"
    }
    present = wired_keys & set(bg.keys())
    missing = wired_keys - present
    assert not missing, f"anime_crystal_void background.* keys missing: {sorted(missing)}"

    # Each must be a non-zero number — the look must actually do something.
    for k in wired_keys:
        v = bg[k]
        assert isinstance(v, (int, float)) and v > 0, (
            f"anime_crystal_void background.{k} must be a positive number, got {v!r}"
        )

    # Each must map to a registered ParamSpec (the dead-key fix).
    spec_names = {spec.name for spec in PROCESSING_PARAMS}
    for k in wired_keys:
        assert k in spec_names, (
            f"anime_crystal_void background.{k} has no matching ParamSpec — dead key"
        )
