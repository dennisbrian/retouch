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
    assert ctx.mole_protect == pytest.approx(1.0)
    assert ctx.face_shadow_fill == nat.face_shadow_fill == 0
    assert ctx.nose_shape == 0 and ctx.powder_finish == 0
    gui = recipe_to_params("natural_v2")
    assert gui["blemish"] == 0 and gui["spot_heal"] == 30 and gui["mark_policy"] == "protect_identity"
    # Preserve the reviewed candidate's skin tone: added lightening is off.
    assert ctx.whiten == 0 and nat.whiten == pytest.approx(10.0)
    assert ctx.smooth == nat.smooth


def test_natural_v2_is_experimental_not_recommended():
    assert "natural_v2" in CURATED_RECIPE_NAMES
    assert "natural_v2" in EXPERIMENTAL_RECIPE_NAMES
    assert "natural_v2" in CONDITIONAL_RECIPE_NAMES
    assert "natural_v2" not in RECOMMENDED_RECIPE_NAMES
    assert RECOMMENDED_RECIPE_NAMES[0] == "natural"


@pytest.mark.parametrize("strength", [0.0, 0.25, 0.55])
def test_independent_blemish_strength_reaches_gui_and_engine(monkeypatch, strength):
    monkeypatch.setitem(RECIPES, "test_independent_blemish", {
        "extends": "natural", "frequency": {"smooth": 0.65},
        "skin": {"blemish": strength},
    })
    name = "test_independent_blemish"
    gui = recipe_to_params(name)
    ctx = build_context(name, resolve_recipe(name), {})
    assert gui["smooth"] == ctx.smooth == 65
    assert gui["blemish"] == round(strength * 100)
    assert ctx.blemish == pytest.approx(strength * 100)
    overridden = build_context(name, resolve_recipe(name), {"blemish": 17})
    assert overridden.blemish == 17


@pytest.mark.parametrize("name", ["natural", "portrait", "cosplay_clear_v1"])
def test_existing_recipes_keep_smoothing_derived_blemish(name):
    rec = resolve_recipe(name)
    expected = rec["frequency"]["smooth"] * 100
    assert recipe_to_params(name)["blemish"] == round(expected)
    assert build_context(name, rec, {}).blemish == expected


def test_candidate_protects_identity_without_brightening_or_legacy_heal():
    gui = recipe_to_params("natural_v2")
    ctx = build_context("natural_v2", resolve_recipe("natural_v2"), {})
    for name, value in {"blemish": 0, "spot_heal": 30, "whiten": 0,
                        "mole_protect": 1.0, "slimming": 0, "reshape_chin_length": 0}.items():
        assert gui[name] == value
        assert getattr(ctx, name) == value
    assert gui["mark_policy"] == "protect_identity"
    assert ctx.mark_policy is not None
    assert "natural_v2" in CURATED_RECIPE_NAMES
    assert "natural_v2" in EXPERIMENTAL_RECIPE_NAMES
    assert "natural_v2" not in RECOMMENDED_RECIPE_NAMES
