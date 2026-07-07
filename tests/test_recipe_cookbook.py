"""Tests for retouch.recipe_cookbook (T4 recipe cookbook UI data layer)."""
from __future__ import annotations

import pytest

from retouch.recipe_cookbook import (
    CATEGORIES,
    RecipeInfo,
    get_recipe_info,
    list_categories,
    list_recipes,
    search_recipes,
)


# ---------------------------------------------------------------------------
# list_categories
# ---------------------------------------------------------------------------

class TestListCategories:
    def test_returns_all_expected_categories(self):
        cats = list_categories()
        assert cats == ["portrait", "cosplay", "outdoor", "studio",
                        "convention", "creative"]

    def test_returns_a_copy(self):
        a = list_categories()
        b = list_categories()
        assert a == b
        a.append("x")
        assert "x" not in list_categories()


# ---------------------------------------------------------------------------
# list_recipes
# ---------------------------------------------------------------------------

class TestListRecipes:
    def test_returns_all_recipes_when_no_category(self):
        from retouch.recipes import RECIPES
        infos = list_recipes()
        names = [i.name for i in infos]
        assert set(names) == set(RECIPES.keys())
        assert names == sorted(names)

    def test_filter_by_category(self):
        infos = list_recipes("cosplay")
        assert len(infos) > 0
        assert all(i.category == "cosplay" for i in infos)
        assert "cosplay" in [i.name for i in infos]

    def test_filter_category_case_insensitive(self):
        a = list_recipes("Cosplay")
        b = list_recipes("COSPLAY")
        assert a == b

    def test_unknown_category_returns_empty(self):
        assert list_recipes("nonexistent") == []

    def test_every_recipe_has_valid_category(self):
        infos = list_recipes()
        assert len(infos) > 20
        for info in infos:
            assert info.category in CATEGORIES, f"{info.name}: {info.category}"

    def test_recipe_info_fields_populated(self):
        infos = list_recipes()
        sample = next(i for i in infos if i.name == "natural")
        assert isinstance(sample, RecipeInfo)
        assert sample.name == "natural"
        assert sample.category == "portrait"
        assert isinstance(sample.description, str)
        assert len(sample.description) > 0
        assert sample.extends is None
        assert isinstance(sample.params, dict)

    def test_to_dict_roundtrip(self):
        info = get_recipe_info("natural")
        d = info.to_dict()
        assert d["name"] == "natural"
        assert d["category"] in CATEGORIES
        assert "description" in d
        assert "params" in d


# ---------------------------------------------------------------------------
# Category assignments (spot checks for each category)
# ---------------------------------------------------------------------------

class TestCategoryAssignments:
    def test_portrait_category(self):
        names = {i.name for i in list_recipes("portrait")}
        assert "natural" in names
        assert "beauty" in names
        assert "porcelain_unified_v1" in names

    def test_cosplay_category(self):
        names = {i.name for i in list_recipes("cosplay")}
        assert "cosplay" in names
        assert "anime_cinematic_v1" in names
        assert "scifi_cosplay" in names

    def test_outdoor_category(self):
        names = {i.name for i in list_recipes("outdoor")}
        assert "outdoor_harsh_sun_v1" in names
        assert "outdoor_golden_hour_v1" in names
        assert len(names) == 4

    def test_studio_category(self):
        names = {i.name for i in list_recipes("studio")}
        assert "studio_hard_flash_v1" in names
        assert "studio_softbox_v1" in names
        assert len(names) == 4

    def test_convention_category(self):
        names = {i.name for i in list_recipes("convention")}
        assert "convention_repair_v1" in names
        assert "con_fluorescent_v1" in names

    def test_creative_category(self):
        names = {i.name for i in list_recipes("creative")}
        assert "pink_dream" in names
        assert "provia" in names  # Fuji sim -> creative
        assert "astia" in names

    def test_fuji_sims_are_creative(self):
        for name in ("provia", "astia", "classic_chrome"):
            assert get_recipe_info(name).category == "creative"

    def test_child_inherits_parent_category(self):
        info = get_recipe_info("cosplay_3d")
        assert info.category == "cosplay"
        info2 = get_recipe_info("anime_cinematic_soft")
        assert info2.category == "cosplay"


# ---------------------------------------------------------------------------
# get_recipe_info
# ---------------------------------------------------------------------------

class TestGetRecipeInfo:
    def test_known_recipe(self):
        info = get_recipe_info("natural")
        assert info is not None
        assert info.name == "natural"
        assert info.extends is None

    def test_case_insensitive(self):
        assert get_recipe_info("NATURAL").name == "natural"
        assert get_recipe_info("Cosplay").name == "cosplay"

    def test_unknown_returns_none(self):
        assert get_recipe_info("does_not_exist") is None

    def test_none_returns_none(self):
        assert get_recipe_info(None) is None

    def test_extends_field_populated(self):
        info = get_recipe_info("porcelain_unified_v1")
        assert info.extends == "korean_beauty"

    def test_params_contains_smooth_when_present(self):
        info = get_recipe_info("natural")
        assert "smooth" in info.params


# ---------------------------------------------------------------------------
# search_recipes
# ---------------------------------------------------------------------------

class TestSearchRecipes:
    def test_empty_query_returns_all(self):
        from retouch.recipes import RECIPES
        assert len(search_recipes("")) == len(RECIPES)
        assert len(search_recipes("   ")) == len(RECIPES)

    def test_search_by_name(self):
        results = search_recipes("outdoor")
        names = [r.name for r in results]
        assert "outdoor_harsh_sun_v1" in names
        assert "outdoor_golden_hour_v1" in names
        assert all("outdoor" in r.name.lower() for r in results)

    def test_search_by_description(self):
        results = search_recipes("fuji")
        names = [r.name for r in results]
        assert "provia" in names
        assert "astia" in names
        assert "classic_chrome" in names

    def test_search_case_insensitive(self):
        a = search_recipes("COSPLAY")
        b = search_recipes("cosplay")
        assert a == b

    def test_search_no_matches(self):
        assert search_recipes("zzz_nonexistent_token_xyz") == []

    def test_search_results_sorted_by_name(self):
        results = search_recipes("outdoor")
        names = [r.name for r in results]
        assert names == sorted(names)

    def test_search_returns_recipe_info_objects(self):
        results = search_recipes("porcelain")
        assert all(isinstance(r, RecipeInfo) for r in results)
        assert len(results) >= 2  # porcelain_unified_v1 + fuji_porcelain


# ---------------------------------------------------------------------------
# Description fallback
# ---------------------------------------------------------------------------

class TestDescriptionFallback:
    def test_curated_description_used(self):
        info = get_recipe_info("natural")
        assert "baseline" in info.description.lower() or "natural" in info.description.lower()

    def test_extends_description_for_uncurated(self):
        info = get_recipe_info("cosplay_3d")
        assert "cosplay" in info.description.lower()


# ---------------------------------------------------------------------------
# Integration: every recipe resolves
# ---------------------------------------------------------------------------

class TestEveryRecipeResolvable:
    def test_all_recipes_have_info(self):
        from retouch.recipes import RECIPES
        for name in RECIPES.keys():
            info = get_recipe_info(name)
            assert info is not None, f"missing info for {name}"
            assert info.name == name
            assert info.category in CATEGORIES
            assert isinstance(info.description, str)
            assert isinstance(info.params, dict)
