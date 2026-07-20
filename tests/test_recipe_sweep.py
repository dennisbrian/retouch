"""Unit coverage for recipe-sweep catalog selection."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from retouch.recipes import (
    CONDITIONAL_RECIPE_NAMES,
    CURATED_RECIPE_NAMES,
    RECOMMENDED_RECIPE_NAMES,
)


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "recipes" / "recipe_sweep.py"
_SPEC = importlib.util.spec_from_file_location("recipe_sweep_under_test", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
recipe_sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(recipe_sweep)


def test_default_sweep_is_correction_first():
    assert recipe_sweep._parse_recipe_list(None) == RECOMMENDED_RECIPE_NAMES
    assert recipe_sweep._parse_recipe_list("recommended") == RECOMMENDED_RECIPE_NAMES


def test_explicit_catalogs_are_selectable():
    assert recipe_sweep._parse_recipe_list("conditional") == CONDITIONAL_RECIPE_NAMES
    assert recipe_sweep._parse_recipe_list("curated") == CURATED_RECIPE_NAMES
