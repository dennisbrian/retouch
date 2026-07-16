"""Tests for retouch/recipes.py — dict structure, extends resolution, ranges."""

import copy
import pytest

from retouch.recipes import RECIPES
from retouch.engine import resolve_recipe

CORE_KEYS = {"natural", "portrait", "cosplay"}
STD_SECTIONS = {"frequency", "skin", "eyes", "lips", "hair", "dodge_burn",
                "color_harmony", "bloom", "texture"}

FLOAT_SECS = {"frequency", "skin", "eyes", "lips", "hair"}

# Nested keys that are legitimately NOT [0, 1] fractions:
#   - relight_azimuth / relight_elevation: light angles in degrees.
#   - unify_hue: -1.0 is the engine sentinel default ("auto/off", engine.py).
NON_FRACTION_SUBKEYS = {"relight_azimuth", "relight_elevation", "unify_hue"}

# Conversion codes (retouch.params) whose *recipe-stored* value is a 0-1 ratio
# that gets ×100'd on the way to the GUI/engine.  For these, the recipe value
# must stay in [0, 1] regardless of the spec's min_val/max_val (those bounds
# describe the post-×100 slider range, e.g. 0-100, not the stored fraction).
_RATIO_CONVERSIONS = frozenset({
    "recipe_pct", "recipe_int_pct", "engine_pct", "relight_pct",
})


def _spec_by_recipe_key():
    """Map every recipe path (recipe_key / gui_recipe_key / engine_recipe_key)
    to its ParamSpec, so a nested recipe value can be validated against the
    spec that actually consumes it (its own bounds + conversion semantics)."""
    from retouch.params import PROCESSING_PARAMS
    index = {}
    for spec in PROCESSING_PARAMS:
        for key in (spec.recipe_key, spec.gui_recipe_key, spec.engine_recipe_key):
            if key:
                index.setdefault(key, spec)
    return index


def _allowed_range_for(dotted_key):
    """Return the inclusive (lo, hi) a nested recipe float may occupy.

    Spec-aware: a ``recipe_direct`` (or other direct) param that consumes a
    raw 0-100 value (e.g. ``frequency.freckle_removal``, ``hair.deglare``)
    must accept its full ``[min_val, max_val]``.  Ratio params (``recipe_pct``
    etc.) store a 0-1 fraction, so they stay strict [0, 1].  Keys with no
    matching spec fall back to the strict [0, 1] ratio guarantee.
    """
    spec = _spec_by_recipe_key().get(dotted_key)
    if spec is None:
        return (0.0, 1.0)
    if spec.conversion in _RATIO_CONVERSIONS:
        return (0.0, 1.0)
    lo = spec.min_val if spec.min_val is not None else 0.0
    hi = spec.max_val if spec.max_val is not None else 1.0
    return (float(lo), float(hi))


def _is_bool(val):
    return isinstance(val, bool)


def _is_float_in_range(val, lo=0.0, hi=1.0):
    return isinstance(val, (int, float)) and lo <= val <= hi


def _walk_values(d, path=""):
    """Yield (path, value) for every leaf in a nested dict."""
    for k, v in d.items():
        full = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            yield from _walk_values(v, full)
        else:
            yield (full, v)


class TestRecipeKeys:
    def test_core_keys_present(self):
        for key in CORE_KEYS:
            assert key in RECIPES, f"Missing core recipe: {key}"

    def test_all_recipes_have_expected_sections(self):
        for name in RECIPES:
            resolved = resolve_recipe(name)
            for sec in STD_SECTIONS:
                msg = f"{name} missing section: {sec}"
                assert sec in resolved, msg

    def test_natural_has_modular_fields(self):
        nat = RECIPES["natural"]
        assert nat.get("slimming") == 0.0
        assert nat.get("blush") == 0.0
        assert nat.get("lip_finish") == "gloss"
        assert nat.get("nose_blush") is False
        assert nat.get("under_eye_blush") is False
        assert nat.get("white_costume_lift") is False

    def test_natural_lips_tint_is_none(self):
        assert RECIPES["natural"]["lips"]["tint"] is None


class TestExtendsMechanism:
    def test_portrait_extends_natural(self):
        assert RECIPES["portrait"].get("extends") == "natural"

    def test_cosplay_extends_natural(self):
        assert RECIPES["cosplay"].get("extends") == "natural"

    def test_resolved_portrait_contains_all_natural_keys(self):
        nat = resolve_recipe("natural")
        por = resolve_recipe("portrait")
        for path, _ in _walk_values(nat):
            parts = path.split(".")
            d = por
            for p in parts:
                assert p in d, f"portrait missing {path}"
                d = d[p]

    def test_resolved_portrait_overrides_specific_values(self):
        por = resolve_recipe("portrait")
        assert por["frequency"]["smooth"] == 0.45
        assert por["skin"]["equalize"] == 0.35
        assert por["bloom"]["opacity"] == 0.03

    def test_extends_chains_multiple_levels(self):
        for name in RECIPES:
            if "extends" in RECIPES.get(name, {}):
                resolved = resolve_recipe(name)
                assert "frequency" in resolved
                assert "skin" in resolved

    def test_anime_cinematic_soft_extends_anime_cinematic_v1(self):
        resolved = resolve_recipe("anime_cinematic_soft")
        assert resolved["contrast"] == 8.0
        assert resolved["bloom"]["opacity"] == 0.22
        assert resolved["clarity"] == 8.0
        assert resolved["hair"]["shine"] == 0.85

    def test_aliases_resolve_correctly(self):
        assert RECIPES["soft"] is RECIPES["anime_cinematic_soft"]
        assert RECIPES["action"] is RECIPES["anime_cinematic_action"]
        assert RECIPES["fantasy"] is RECIPES["anime_cinematic_fantasy"]
        for alias in ("soft", "action", "fantasy"):
            resolved = resolve_recipe(alias)
            assert "frequency" in resolved
            assert "contrast" in resolved


class TestValueRanges:
    def test_nested_float_fields_in_range(self):
        for name, recipe in RECIPES.items():
            for sec in FLOAT_SECS:
                if sec not in recipe:
                    continue
                for subk, subv in recipe[sec].items():
                    path = f"{name}.{sec}.{subk}"
                    if subv is None or isinstance(subv, str) or _is_bool(subv):
                        continue
                    if subk in NON_FRACTION_SUBKEYS:
                        continue
                    # Validate against the range the ParamSpec that consumes
                    # this key actually permits — a direct 0-100 param (e.g.
                    # freckle_removal, hair.deglare) may exceed 1; ratio params
                    # (recipe_pct etc.) and unmapped keys stay strict [0, 1].
                    lo, hi = _allowed_range_for(f"{sec}.{subk}")
                    msg = f"{path}={subv!r} not in [{lo}, {hi}]"
                    assert _is_float_in_range(subv, lo, hi), msg

    def test_bloom_opacity_in_range(self):
        for name, recipe in RECIPES.items():
            bloom = recipe.get("bloom", {})
            if isinstance(bloom, dict) and "opacity" in bloom:
                v = bloom["opacity"]
                assert _is_float_in_range(v), f"{name}.bloom.opacity={v}"

    def test_texture_opacity_in_range(self):
        for name, recipe in RECIPES.items():
            tex = recipe.get("texture", {})
            if isinstance(tex, dict) and "opacity" in tex:
                v = tex["opacity"]
                assert _is_float_in_range(v), f"{name}.texture.opacity={v}"

    def test_dodge_burn_amount_in_range(self):
        for name, recipe in RECIPES.items():
            db = recipe.get("dodge_burn", {})
            if isinstance(db, dict) and "amount" in db:
                v = db["amount"]
                assert _is_float_in_range(v), f"{name}.dodge_burn.amount={v}"

    def test_color_harmony_amount_in_range(self):
        for name, recipe in RECIPES.items():
            ch = recipe.get("color_harmony", {})
            if isinstance(ch, dict) and "amount" in ch:
                v = ch["amount"]
                assert _is_float_in_range(v), f"{name}.color_harmony.amount={v}"

    def test_modular_bools(self):
        for name, recipe in RECIPES.items():
            for key in ("nose_blush", "under_eye_blush", "white_costume_lift"):
                if key in recipe:
                    assert _is_bool(recipe[key]), f"{name}.{key} should be bool"

    def test_modular_floats_are_numeric(self):
        for name, recipe in RECIPES.items():
            for key in ("slimming", "blush"):
                if key in recipe:
                    v = recipe[key]
                    assert isinstance(v, (int, float)), f"{name}.{key} should be numeric"

    def test_lips_tint_is_string_or_none(self):
        for name, recipe in RECIPES.items():
            if "lips" in recipe and "tint" in recipe["lips"]:
                v = recipe["lips"]["tint"]
                assert v is None or isinstance(v, str), f"{name}.lips.tint should be str or None"


class TestUnknownRecipeFallback:
    def test_resolve_unknown_returns_natural(self):
        resolved = resolve_recipe("nonexistent_xyz")
        nat = resolve_recipe("natural")
        assert resolved == nat

    def test_resolve_natural_is_idempotent(self):
        first = resolve_recipe("natural")
        second = resolve_recipe("natural")
        assert first == second


class TestModuleExports:
    def test_recipes_is_dict(self):
        assert isinstance(RECIPES, dict)

    def test_recipes_not_empty(self):
        assert len(RECIPES) > 0
