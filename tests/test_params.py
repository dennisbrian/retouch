"""Tests for retouch/params.py — ParamSpec registry and conversion helpers.

This module is the canonical single source of truth for every processing
parameter; these tests pin down the structural invariants and the recipe
→ GUI conversion behaviour.
"""

import pytest

from retouch.params import (
    PROCESSING_PARAMS,
    ParamSpec,
    _BY_NAME,
    _gui_for_recipe_value,
    _lookup_recipe,
    _resolve_dodge_burn,
    _resolve_recipe_value,
    param_names,
    recipe_to_params,
)
from retouch.recipes import RECIPES


# ---------------------------------------------------------------------------
# ParamSpec structural invariants
# ---------------------------------------------------------------------------


class TestProcessingParamsList:
    def test_processing_params_is_list_of_param_specs(self):
        assert isinstance(PROCESSING_PARAMS, list)
        assert len(PROCESSING_PARAMS) > 0
        for spec in PROCESSING_PARAMS:
            assert isinstance(spec, ParamSpec), (
                f"expected ParamSpec, got {type(spec).__name__}"
            )

    def test_param_spec_required_fields(self):
        for spec in PROCESSING_PARAMS:
            assert isinstance(spec.name, str) and spec.name, (
                f"ParamSpec.name must be a non-empty string (got {spec.name!r})"
            )
            assert isinstance(spec.conversion, str) and spec.conversion, (
                f"ParamSpec.conversion must be a non-empty string "
                f"for {spec.name!r} (got {spec.conversion!r})"
            )

    def test_param_spec_names_unique(self):
        names = [spec.name for spec in PROCESSING_PARAMS]
        assert len(names) == len(set(names)), (
            f"duplicate ParamSpec names: "
            f"{[n for n in names if names.count(n) > 1]}"
        )

    def test_by_name_dict_matches_list(self):
        assert set(_BY_NAME.keys()) == {spec.name for spec in PROCESSING_PARAMS}
        for spec in PROCESSING_PARAMS:
            assert _BY_NAME[spec.name] is spec

    def test_param_names_returns_list_of_strings(self):
        names = param_names()
        assert isinstance(names, list)
        assert len(names) == len(PROCESSING_PARAMS)
        for n in names:
            assert isinstance(n, str) and n
        assert names == [spec.name for spec in PROCESSING_PARAMS]

    def test_param_spec_equality(self):
        a = ParamSpec(
            name="demo",
            cli_flag="demo",
            cli_type=int,
            default=10,
            recipe_key="demo",
            conversion="recipe_pct",
            min_val=0,
            max_val=100,
        )
        b = ParamSpec(
            name="demo",
            cli_flag="demo",
            cli_type=int,
            default=10,
            recipe_key="demo",
            conversion="recipe_pct",
            min_val=0,
            max_val=100,
        )
        c = ParamSpec(
            name="demo2",
            cli_flag=None,
            cli_type=None,
            default=0,
            recipe_key=None,
            conversion="static",
        )
        assert a == b
        assert a != c
        # frozen dataclass — mutation must raise
        with pytest.raises(Exception):
            a.name = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _lookup_recipe — dot-path walker
# ---------------------------------------------------------------------------


class TestLookupRecipe:
    def test_lookup_recipe_simple_path(self):
        assert _lookup_recipe({"a": 1}, "a") == 1
        assert _lookup_recipe({"a": "hello"}, "a") == "hello"

    def test_lookup_recipe_nested_path(self):
        rec = {"eyes": {"catchlight": 0.25, "whites": 0.18}}
        assert _lookup_recipe(rec, "eyes.catchlight") == 0.25
        assert _lookup_recipe(rec, "eyes.whites") == 0.18

    def test_lookup_recipe_deeply_nested(self):
        rec = {"a": {"b": {"c": {"d": 42}}}}
        assert _lookup_recipe(rec, "a.b.c.d") == 42

    def test_lookup_recipe_missing_key_returns_none(self):
        assert _lookup_recipe({"a": 1}, "b") is None
        assert _lookup_recipe({"a": {"b": 1}}, "a.c") is None
        assert _lookup_recipe({"a": {"b": 1}}, "a.b.c") is None

    def test_lookup_recipe_walks_into_none_returns_none(self):
        # Mid-walk a non-dict value — must return None, not raise
        assert _lookup_recipe({"a": 5}, "a.b") is None
        assert _lookup_recipe({"a": "string"}, "a.b") is None

    def test_lookup_recipe_empty_path(self):
        # An empty path splits to [""] which is not a key of any non-empty
        # dict — must return None and not raise.
        assert _lookup_recipe({}, "") is None
        assert _lookup_recipe({"a": 1}, "") is None


# ---------------------------------------------------------------------------
# _gui_for_recipe_value — recipe value → GUI value translation
# ---------------------------------------------------------------------------


class TestGuiForRecipeValue:
    def test_recipe_pct(self):
        # 0.30 recipe → 30 GUI
        assert _gui_for_recipe_value("recipe_pct", 0.30) == 30
        assert _gui_for_recipe_value("recipe_pct", 0.05) == 5
        assert _gui_for_recipe_value("recipe_pct", 1.0) == 100

    def test_recipe_direct(self):
        # 0.4 recipe → 0.4 GUI (no scaling, just float coercion)
        assert _gui_for_recipe_value("recipe_direct", 0.4) == 0.4
        assert _gui_for_recipe_value("recipe_direct", 1.0) == 1.0
        assert _gui_for_recipe_value("recipe_direct", 0.0) == 0.0

    def test_gui_mul500(self):
        # 0.05 recipe → 25 GUI  (×500)
        assert _gui_for_recipe_value("gui_mul500", 0.05) == 25
        assert _gui_for_recipe_value("gui_mul500", 0.1) == 50
        assert _gui_for_recipe_value("gui_mul500", 0.0) == 0

    def test_gui_div100(self):
        # 0.10 recipe → 10 GUI  (×100)
        assert _gui_for_recipe_value("gui_div100", 0.10) == 10
        assert _gui_for_recipe_value("gui_div100", 1.0) == 100

    def test_engine_pct(self):
        # 0.05 recipe → 5 GUI (×100, integer)
        assert _gui_for_recipe_value("engine_pct", 0.05) == 5
        assert _gui_for_recipe_value("engine_pct", 0.5) == 50

    def test_recipe_int_pct(self):
        # 0.30 recipe → 30 GUI (truncated to int)
        assert _gui_for_recipe_value("recipe_int_pct", 0.30) == 30
        assert isinstance(_gui_for_recipe_value("recipe_int_pct", 0.5), int)

    def test_dropdown(self):
        assert _gui_for_recipe_value("dropdown", "rosy") == "rosy"
        # falsy/empty values collapse to the "none" sentinel
        assert _gui_for_recipe_value("dropdown", "") == "none"
        assert _gui_for_recipe_value("dropdown", None) is None

    def test_bool_flag(self):
        assert _gui_for_recipe_value("bool_flag", True) is True
        assert _gui_for_recipe_value("bool_flag", False) is False
        assert _gui_for_recipe_value("bool_flag", 1) is True
        assert _gui_for_recipe_value("bool_flag", 0) is False

    def test_passthrough_conversions(self):
        # "gui_direct"/"static"/"alias" return the recipe value untouched
        for conv in ("gui_direct", "static", "alias", "dodge_burn_pct",
                     "lip_tint_direct"):
            assert _gui_for_recipe_value(conv, 0.42) == 0.42
            assert _gui_for_recipe_value(conv, "hello") == "hello"
            assert _gui_for_recipe_value(conv, None) is None

    def test_relight_pct(self):
        # recipe_pct-style scaling but routed through the relight branch
        assert _gui_for_recipe_value("relight_pct", 0.35) == 35
        assert _gui_for_recipe_value("relight_pct", 0.0) == 0

    def test_unknown_conversion_raises(self):
        with pytest.raises(ValueError):
            _gui_for_recipe_value("nonsense_code_xyz", 0.5)

    def test_none_recipe_value_returns_none(self):
        # All conversions short-circuit on None
        for conv in ("recipe_pct", "recipe_direct", "gui_mul500",
                     "gui_div100", "engine_pct", "dropdown", "bool_flag",
                     "static", "gui_direct"):
            assert _gui_for_recipe_value(conv, None) is None


# ---------------------------------------------------------------------------
# _resolve_recipe_value — engine-side extraction
# ---------------------------------------------------------------------------


class TestResolveRecipeValue:
    def test_resolve_recipe_value_simple(self):
        # smooth spec (recipe_pct, frequency.smooth)
        spec = _BY_NAME["smooth"]
        rec = {"frequency": {"smooth": 0.30}}
        # engine side of recipe_pct is the percentage (×100)
        assert _resolve_recipe_value(spec, rec) == 30.0

    def test_resolve_recipe_value_recipe_direct(self):
        spec = _BY_NAME["mid_reduction"]
        rec = {"frequency": {"mid_reduction": 0.5}}
        assert _resolve_recipe_value(spec, rec) == 0.5

    def test_resolve_recipe_value_missing_returns_default(self):
        spec = _BY_NAME["smooth"]
        # recipe has no frequency.smooth — falls back to spec.default (30)
        assert _resolve_recipe_value(spec, {}) == spec.default

    def test_resolve_recipe_value_static(self):
        # static conversion always returns the default
        from retouch.params import ParamSpec
        spec = ParamSpec(
            name="__test_static__",
            cli_flag=None,
            cli_type=None,
            default=42,
            recipe_key="does.not.matter",
            conversion="static",
        )
        assert _resolve_recipe_value(spec, {"does": {"not": {"matter": 99}}}) == 42
        assert _resolve_recipe_value(spec, {}) == 42

    def test_resolve_recipe_value_gui_direct(self):
        # gui_direct passes the recipe value through unchanged
        spec = _BY_NAME["brightness"]
        rec = {"brightness": 7}
        assert _resolve_recipe_value(spec, rec) == 7
        # missing → spec.default (whatever the spec declares)
        assert _resolve_recipe_value(spec, {}) == spec.default


# ---------------------------------------------------------------------------
# _resolve_dodge_burn — dict and scalar forms
# ---------------------------------------------------------------------------


class TestResolveDodgeBurn:
    def test_resolve_dodge_burn_dict(self):
        # amount 0.10 in dict form is treated as a fraction → 10.0
        rec = {"dodge_burn": {"amount": 0.10}}
        assert _resolve_dodge_burn(rec) == 10.0

    def test_resolve_dodge_burn_scalar_above_one(self):
        # raw 18.0 is > 1.0, so it is treated as already a percent → 18.0
        rec = {"dodge_burn": 18.0}
        assert _resolve_dodge_burn(rec) == 18.0

    def test_resolve_dodge_burn_scalar_below_one(self):
        # raw 0.05 is ≤ 1.0, so it is treated as a fraction → 5.0
        rec = {"dodge_burn": 0.05}
        assert _resolve_dodge_burn(rec) == 5.0

    def test_resolve_dodge_burn_missing(self):
        # missing dodge_burn key → default 0.0
        assert _resolve_dodge_burn({}) == 0.0

    def test_resolve_dodge_burn_zero_dict(self):
        rec = {"dodge_burn": {"amount": 0.0}}
        assert _resolve_dodge_burn(rec) == 0.0


# ---------------------------------------------------------------------------
# recipe_to_params — top-level GUI population
# ---------------------------------------------------------------------------


# The CLI / engine module is heavy; the first call pays an import cost.
# We pre-warm it here so individual tests are fast.
@pytest.fixture(scope="module", autouse=True)
def _warm_recipe_to_params():
    recipe_to_params("natural")
    yield


class TestRecipeToParams:
    def test_recipe_to_params_natural(self):
        result = recipe_to_params("natural")
        assert isinstance(result, dict)
        # Known keys populated for the natural recipe
        for key in (
            "smooth",
            "mid_reduction",
            "whiten",
            "equalize",
            "catchlight",
            "lip_enhance",
            "hair_enhance",
            "dodge_burn",
        ):
            assert key in result, f"recipe_to_params('natural') missing {key!r}"
        # smooth defaults to 30 (0.30 × 100) for natural
        assert result["smooth"] == 30
        # mid_reduction is recipe_direct → 0.35
        assert result["mid_reduction"] == 0.35

    @pytest.mark.parametrize("recipe_name", sorted(RECIPES.keys()))
    def test_recipe_to_params_all_recipes(self, recipe_name):
        # Every recipe should produce a result without raising
        result = recipe_to_params(recipe_name)
        assert isinstance(result, dict)
        assert len(result) == len(PROCESSING_PARAMS)

    def test_recipe_to_params_unknown_falls_back_to_natural(self):
        unknown = recipe_to_params("__this_recipe_does_not_exist__")
        natural = recipe_to_params("natural")
        assert unknown == natural

    def test_recipe_to_params_keys_match_processing_params(self):
        result = recipe_to_params("natural")
        expected = set(param_names())
        actual = set(result.keys())
        # The returned dict must include every processing param
        assert expected.issubset(actual), (
            f"missing keys: {expected - actual}"
        )
        # And it must not carry extra keys (one GUI value per spec)
        assert actual.issubset(expected), (
            f"extra keys: {actual - expected}"
        )

    def test_recipe_to_params_values_are_in_range(self):
        # For every spec with min/max set, the natural-recipe value
        # must fall inside the inclusive bounds.  Skip specs whose GUI
        # value is None (no source data) or non-numeric (dropdowns).
        natural_result = recipe_to_params("natural")
        for spec in PROCESSING_PARAMS:
            if spec.min_val is None or spec.max_val is None:
                continue
            val = natural_result.get(spec.name)
            if val is None or isinstance(val, bool) or isinstance(val, str):
                continue
            lo, hi = spec.min_val, spec.max_val
            assert lo <= val <= hi, (
                f"natural[{spec.name!r}] = {val!r} not in [{lo}, {hi}]"
            )

    def test_recipe_to_params_returns_int_for_int_specs(self):
        # The GUI should see ints for cli_type=int specs (so the Gradio
        # sliders render correctly).  Skip specs where the recipe gives
        # us a None or a string.
        result = recipe_to_params("natural")
        for spec in PROCESSING_PARAMS:
            if spec.cli_type is not int:
                continue
            val = result.get(spec.name)
            if val is None or isinstance(val, bool):
                continue
            assert isinstance(val, int) and not isinstance(val, bool), (
                f"{spec.name!r} should be int, got {type(val).__name__}: {val!r}"
            )


class TestAnimeParams:
    def test_skin_flatten_in_param_names(self):
        assert "skin_flatten" in param_names()

    def test_skin_quantize_in_param_names(self):
        assert "skin_quantize" in param_names()

    def test_skin_flatten_recipe_key(self):
        from retouch.params import _BY_NAME
        spec = _BY_NAME["skin_flatten"]
        assert spec.recipe_key == "skin.flatten"
        assert spec.conversion == "recipe_pct"
        assert spec.min_val == 0
        assert spec.max_val == 100
        assert spec.default == 0

    def test_skin_quantize_recipe_key(self):
        from retouch.params import _BY_NAME
        spec = _BY_NAME["skin_quantize"]
        assert spec.recipe_key == "skin.quantize"
        assert spec.conversion == "recipe_pct"
        assert spec.min_val == 0
        assert spec.max_val == 100
        assert spec.default == 0

    def test_skin_unify_in_param_names(self):
        assert "skin_unify" in param_names()
        assert "skin_unify_hue" in param_names()
        assert "skin_glow" in param_names()

    def test_skin_unify_recipe_key(self):
        from retouch.params import _BY_NAME
        spec = _BY_NAME["skin_unify"]
        assert spec.recipe_key == "skin.unify"
        assert spec.conversion == "recipe_pct"

    def test_skin_unify_hue_recipe_key(self):
        from retouch.params import _BY_NAME
        spec = _BY_NAME["skin_unify_hue"]
        assert spec.recipe_key == "skin.unify_hue"
        assert spec.conversion == "recipe_direct"
        assert spec.default == -1.0

    def test_skin_glow_recipe_key(self):
        from retouch.params import _BY_NAME
        spec = _BY_NAME["skin_glow"]
        assert spec.recipe_key == "skin.glow"
        assert spec.conversion == "recipe_pct"
        assert spec.min_val == 0
        assert spec.max_val == 100
