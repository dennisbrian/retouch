"""natural_v2 candidate recipe and the ``skin.blemish`` recipe path.

``blemish`` (legacy blemish removal) used to have no recipe key: it always
mirrored ``frequency.smooth``. A recipe may now set ``skin.blemish`` (0-1);
every recipe that doesn't must resolve exactly as before, on the engine side
and the GUI side.
"""

from __future__ import annotations

import copy

import pytest

from retouch.engine import build_context, resolve_recipe
from retouch.params import recipe_to_params
from retouch.recipes import (
    CONDITIONAL_RECIPE_NAMES,
    CURATED_RECIPE_NAMES,
    EXPERIMENTAL_RECIPE_NAMES,
    RECIPES,
    RECOMMENDED_RECIPE_NAMES,
)


def _engine(name, rec=None):
    return build_context(name, rec if rec is not None else resolve_recipe(name), {})


def _mirror_engine(rec):
    return float(rec.get("frequency", {}).get("smooth", 0.5)) * 100.0


def _mirror_gui(rec):
    return int(round(float(rec.get("frequency", {}).get("smooth", 0.30)) * 100.0))


@pytest.mark.parametrize("name", sorted(n for n in RECIPES if n != "natural_v2"))
def test_existing_recipes_still_mirror_smooth(name):
    rec = resolve_recipe(name)
    assert "blemish" not in rec.get("skin", {}), name
    assert _engine(name).blemish == pytest.approx(_mirror_engine(rec))
    assert recipe_to_params(name)["blemish"] == _mirror_gui(rec)


def test_recipe_blemish_key_wins_on_both_sides():
    # resolve_recipe may hand back a cached dict: never edit it in place.
    rec = copy.deepcopy(resolve_recipe("natural"))
    rec["skin"]["blemish"] = 0.4
    assert _engine("natural", rec).blemish == pytest.approx(40.0)
    zero = copy.deepcopy(resolve_recipe("natural"))
    zero["skin"]["blemish"] = 0.0
    assert _engine("natural", zero).blemish == 0.0


def test_caller_override_still_wins():
    ctx = build_context("natural_v2", resolve_recipe("natural_v2"), {"blemish": 25})
    assert ctx.blemish == 25


def test_natural_v2_settings():
    ctx, nat = _engine("natural_v2"), _engine("natural")
    assert ctx.blemish == 0.0 and nat.blemish == pytest.approx(30.0)
    assert ctx.spot_heal == 30 and nat.spot_heal == 0
    assert ctx.mole_protect == pytest.approx(0.80)
    gui = recipe_to_params("natural_v2")
    assert gui["blemish"] == 0 and gui["spot_heal"] == 30 and gui["mark_policy"] == "protect_identity"
    # Skin lightening is an owner decision: natural_v2 keeps natural's rosy 0.10.
    assert ctx.whiten == nat.whiten == pytest.approx(10.0)
    assert ctx.smooth == nat.smooth


def test_natural_v2_is_experimental_not_recommended():
    assert "natural_v2" in CURATED_RECIPE_NAMES
    assert "natural_v2" in EXPERIMENTAL_RECIPE_NAMES
    assert "natural_v2" in CONDITIONAL_RECIPE_NAMES
    assert "natural_v2" not in RECOMMENDED_RECIPE_NAMES
    assert RECOMMENDED_RECIPE_NAMES[0] == "natural"
