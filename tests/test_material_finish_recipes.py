"""Regression coverage for the opt-in R12 material-finish recipe pack."""

from retouch.engine import build_context
from retouch.params import PROCESSING_PARAMS, resolve_recipe
from retouch.recipes import RECIPES


EXPECTED_FINISHES = {
    "cosplay_powder_v1": ("cosplay_clear_v1", "powder", 0.22),
    "kbeauty_glass_finish_v1": ("korean_glass_clear_v1", "glass_skin", 0.18),
    "game_character_v3": ("game_character_v2", "dewy", 0.18),
    "editorial_matte_v1": ("portrait", "matte", 0.20),
}

REFERENCE_RECIPES = {
    "cosplay_heroic_amber_v1": ("game_character_v3", "dewy"),
    "cosplay_pastel_dream_v1": ("kbeauty_glass_finish_v1", "glass_skin"),
    "cosplay_ice_cathedral_v1": ("kbeauty_glass_finish_v1", "glass_skin"),
}

SHOWCASE_RECIPES = {
    "cosplay_heroic_amber_showcase_v1": ("cosplay_heroic_amber_v1", "dewy"),
    "cosplay_pastel_dream_showcase_v1": ("cosplay_pastel_dream_v1", "glass_skin"),
    "cosplay_ice_cathedral_showcase_v1": ("cosplay_ice_cathedral_v1", "glass_skin"),
}


def test_material_finish_recipe_keys_are_registered() -> None:
    recipe_keys = {spec.recipe_key for spec in PROCESSING_PARAMS}
    assert {
        "skin.specular_finish",
        "skin.specular_finish_strength",
        "skin.specular_recolor",
    } <= recipe_keys


def test_material_finish_recipes_resolve_to_their_intended_modes() -> None:
    for name, (parent, mode, strength) in EXPECTED_FINISHES.items():
        assert RECIPES[name]["extends"] == parent
        resolved = resolve_recipe(name)
        assert resolved["skin"]["specular_finish"] == mode
        assert resolved["skin"]["specular_finish_strength"] == strength
        assert resolved["skin"]["specular_recolor"] == 0.0


def test_game_character_v3_reduces_legacy_specular_bloom() -> None:
    assert resolve_recipe("game_character_v3")["specular_bloom"] == 20


def test_vascular_refine_recipe_is_low_strength_and_texture_preserving() -> None:
    resolved = resolve_recipe("vascular_refine_v1")
    assert RECIPES["vascular_refine_v1"]["extends"] == "natural_polish_v1"
    assert resolved["skin"]["hemoglobin_smooth"] == 0.18
    assert resolved["skin"]["vein_attenuate"] == 0.12
    assert resolved["texture"]["opacity"] == 1.0


def test_mole_safe_portrait_preserves_the_inherited_cleanup_recipe() -> None:
    resolved = resolve_recipe("mole_safe_portrait_v1")
    assert RECIPES["mole_safe_portrait_v1"]["extends"] == "portrait"
    assert resolved["skin"]["mole_protect"] == 0.80
    assert resolved["frequency"]["smooth"] == 0.45


def test_reference_recipes_preserve_their_material_finish_contract() -> None:
    for name, (parent, mode) in REFERENCE_RECIPES.items():
        assert RECIPES[name]["extends"] == parent
        resolved = resolve_recipe(name)
        assert resolved["skin"]["specular_finish"] == mode
        assert resolved["slimming"] == 0.0


def test_showcase_recipes_keep_identity_and_texture_guards() -> None:
    for name, (parent, mode) in SHOWCASE_RECIPES.items():
        assert RECIPES[name]["extends"] == parent
        resolved = resolve_recipe(name)
        assert resolved["skin"]["specular_finish"] == mode
        assert resolved["slimming"] == 0.0
        assert resolved["texture"]["opacity"] >= 0.82


def test_character_showcase_reaches_wig_and_material_finish_stages() -> None:
    resolved = resolve_recipe("cosplay_character_showcase_v1")
    context = build_context("cosplay_character_showcase_v1", resolved, {})

    assert resolved["skin"]["specular_finish"] == "dewy"
    assert resolved["skin"]["specular_finish_strength"] == 0.26
    assert resolved["skin"]["sss"] == 0.42
    assert resolved["cosplay"]["wig_lace_blend"] == 0.42
    assert resolved["slimming"] == 0.0
    assert context.specular_finish == "dewy"
    assert context.specular_finish_strength == 0.26
    assert context.skin_sss == 42.0
    assert context.cosplay_wig_lace_blend == 42
    assert context.slimming == 0.0


def test_guided_feathering_is_opt_in_for_the_two_cosplay_showcase_recipes() -> None:
    for name in (
        "cosplay_heroic_amber_showcase_v1",
        "cosplay_character_showcase_v1",
    ):
        resolved = resolve_recipe(name)
        context = build_context(name, resolved, {})

        assert resolved["mask"]["feather_mode"] == "guided"
        assert context.mask_feather_mode == "guided"

    assert resolve_recipe("cosplay_pastel_dream_showcase_v1").get("mask") is None
