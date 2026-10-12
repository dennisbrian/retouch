"""Tests for the AI recipe generator system: schema, loader, CLI."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from retouch.recipe_schema import build_recipe_schema, validate_recipe
from retouch.recipe_loader import (
    _engine_to_flat_recipe,
    _flat_to_engine_recipe,
    _convert_param_to_engine,
    _convert_param_from_engine,
    export_recipe,
    import_recipe,
    list_user_recipes,
    write_recipe_json,
)
from retouch.recipes import RECIPES
from retouch.engine import resolve_recipe


# -------------------------------------------------------------------
# Schema tests
# -------------------------------------------------------------------

class TestRecipeSchema:
    def test_schema_is_dict(self):
        schema = build_recipe_schema()
        assert isinstance(schema, dict)
        assert schema["type"] == "object"
        assert "name" in schema["required"]
        assert "params" in schema["required"]

    def test_schema_includes_all_params(self):
        from retouch.params import PROCESSING_PARAMS
        schema = build_recipe_schema()
        param_props = schema["properties"]["params"]["properties"]
        spec_names = {p.name for p in PROCESSING_PARAMS}
        schema_names = set(param_props.keys())
        assert spec_names == schema_names, f"Missing: {spec_names - schema_names}, Extra: {schema_names - spec_names}"

    def test_validate_minimal_recipe(self):
        is_valid, errors = validate_recipe({
            "name": "test_recipe",
            "params": {"smooth": 0.5}
        })
        assert is_valid, errors

    def test_validate_full_recipe(self):
        is_valid, errors = validate_recipe({
            "name": "full_recipe",
            "version": "1.0",
            "description": "A test recipe",
            "extends": None,
            "author": "Test",
            "tags": ["test", "sample"],
            "params": {
                "smooth": 0.5,
                "whiten": 0.3,
                "whiten_tone": "porcelain",
                "color_grade": "natural",
                "grade_intensity": 0.5,
            }
        })
        assert is_valid, errors

    def test_validate_rejects_bad_name(self):
        is_valid, errors = validate_recipe({
            "name": "Bad-Name!",
            "params": {}
        })
        assert not is_valid
        assert any("name" in e for e in errors)

    def test_validate_rejects_out_of_range(self):
        is_valid, errors = validate_recipe({
            "name": "test",
            "params": {"smooth": 5.0}  # max should be 1.0
        })
        assert not is_valid
        assert any("smooth" in e or "maximum" in e for e in errors)

    def test_validate_rejects_unknown_param(self):
        is_valid, errors = validate_recipe({
            "name": "test",
            "params": {"completely_made_up_param": 0.5}
        })
        assert not is_valid
        assert any("completely_made_up_param" in e for e in errors)

    def test_validate_rejects_missing_name(self):
        is_valid, errors = validate_recipe({
            "params": {"smooth": 0.5}
        })
        assert not is_valid


# -------------------------------------------------------------------
# Conversion tests
# -------------------------------------------------------------------

class TestParamConversion:
    def test_convert_recipe_pct(self):
        # 0-1 → 0-100
        assert _convert_param_to_engine("smooth", 0.5) == 50.0
        assert _convert_param_to_engine("whiten", 0.3) == 30.0

    def test_convert_engine_pct(self):
        # 0-1 → 0-100
        assert _convert_param_to_engine("pore_synthesis", 0.5) == 50.0

    def test_convert_direct(self):
        # Already 0-1, no change
        assert _convert_param_to_engine("mid_reduction", 0.45) == 0.45
        assert _convert_param_to_engine("texture_opacity", 1.0) == 1.0

    def test_convert_from_engine(self):
        assert _convert_param_from_engine("smooth", 50.0) == 0.5
        assert _convert_param_from_engine("mid_reduction", 0.45) == 0.45

    def test_convert_string_passthrough(self):
        assert _convert_param_to_engine("whiten_tone", "porcelain") == "porcelain"
        assert _convert_param_to_engine("lip_tint", "rose") == "rose"

    def test_convert_bool_passthrough(self):
        assert _convert_param_to_engine("nose_blush", True) is True
        assert _convert_param_to_engine("white_costume_lift", False) is False


# -------------------------------------------------------------------
# Round-trip tests (JSON → engine → JSON)
# -------------------------------------------------------------------

class TestRoundTrip:
    def test_flat_to_engine_to_flat_skin(self):
        flat = {"smooth": 0.5, "whiten": 0.3, "whiten_tone": "porcelain"}
        engine = _flat_to_engine_recipe(flat)
        back = _engine_to_flat_recipe(engine)
        assert abs(back["smooth"] - 0.5) < 1e-6
        assert abs(back["whiten"] - 0.3) < 1e-6
        assert back["whiten_tone"] == "porcelain"

    def test_flat_to_engine_to_flat_color_grade(self):
        flat = {"color_grade": "natural", "grade_intensity": 0.7}
        engine = _flat_to_engine_recipe(flat)
        assert engine["color_harmony"]["preset"] == "natural"
        assert engine["color_harmony"]["amount"] == 0.7
        back = _engine_to_flat_recipe(engine)
        assert back["color_grade"] == "natural"
        assert back["grade_intensity"] == 0.7

    def test_flat_to_engine_split_toning(self):
        flat = {"shadow_hue": 200, "shadow_sat": 0.15, "highlight_hue": 30, "highlight_sat": 0.10}
        engine = _flat_to_engine_recipe(flat)
        back = _engine_to_flat_recipe(engine)
        assert back["shadow_hue"] == 200
        assert back["shadow_sat"] == 0.15
        assert back["highlight_hue"] == 30
        assert back["highlight_sat"] == 0.10

    def test_lip_tint_none(self):
        flat = {"lip_tint": "none"}
        engine = _flat_to_engine_recipe(flat)
        assert engine["lips"]["tint"] is None
        back = _engine_to_flat_recipe(engine)
        assert back["lip_tint"] == "none"

    def test_lip_tint_color(self):
        flat = {"lip_tint": "rose"}
        engine = _flat_to_engine_recipe(flat)
        assert engine["lips"]["tint"] == "rose"
        back = _engine_to_flat_recipe(engine)
        assert back["lip_tint"] == "rose"

    def test_bloom(self):
        flat = {"bloom": 0.20, "bloom_threshold": 200, "bloom_softness": 40}
        engine = _flat_to_engine_recipe(flat)
        assert engine["bloom"]["opacity"] == 0.20
        assert engine["bloom"]["threshold"] == 200
        assert engine["bloom"]["softness"] == 40
        back = _engine_to_flat_recipe(engine)
        assert back["bloom"] == 0.20
        assert back["bloom_threshold"] == 200
        assert back["bloom_softness"] == 40


# -------------------------------------------------------------------
# Import/Export integration tests
# -------------------------------------------------------------------

@pytest.fixture
def sample_recipe_json(tmp_path):
    data = {
        "name": "test_cosplay",
        "version": "1.0",
        "description": "Test cosplay recipe",
        "extends": None,
        "author": "Test",
        "tags": ["test", "cosplay"],
        "params": {
            "smooth": 0.6,
            "whiten": 0.5,
            "whiten_tone": "porcelain",
            "lip_tint": "rose",
            "lip_finish": "velvet",
            "color_grade": "cosplay",
            "grade_intensity": 0.4,
            "glow": 0.3,
            "vignette": 0.1,
        }
    }
    p = tmp_path / "test_cosplay.json"
    p.write_text(json.dumps(data, indent=2))
    return p, data


@pytest.fixture
def isolated_recipe_state(tmp_path, monkeypatch):
    """Snapshot RECIPES dict and redirect user recipes dir to tmp_path."""
    from retouch import recipe_loader
    from retouch.recipes import RECIPES

    # Snapshot
    saved_recipes = dict(RECIPES)
    saved_dir = recipe_loader._user_recipes_dir
    saved_env = dict(os.environ) if hasattr(__import__("os"), "environ") else {}

    # Redirect
    recipe_loader._user_recipes_dir = lambda: tmp_path
    monkeypatch.setenv("RETOUCH_USER_RECIPES", str(tmp_path))

    yield tmp_path

    # Restore
    RECIPES.clear()
    RECIPES.update(saved_recipes)
    recipe_loader._user_recipes_dir = saved_dir


class TestImportExport:
    def test_import_recipe(self, sample_recipe_json, isolated_recipe_state):
        json_path, data = sample_recipe_json
        name = import_recipe(json_path, save=True)
        assert name == "test_cosplay"
        assert "test_cosplay" in RECIPES
        assert (isolated_recipe_state / "test_cosplay.json").exists()

    def test_import_duplicate_raises(self, sample_recipe_json, isolated_recipe_state):
        import_recipe(sample_recipe_json[0], save=True)
        with pytest.raises(ValueError, match="already exists"):
            import_recipe(sample_recipe_json[0], save=True)

    def test_import_invalid_raises(self, isolated_recipe_state, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"name": "Bad-Name!", "params": {}}))
        with pytest.raises(ValueError, match="Invalid recipe"):
            import_recipe(bad)

    def test_export_recipe(self, sample_recipe_json, isolated_recipe_state):
        import_recipe(sample_recipe_json[0], save=True)
        data = export_recipe("test_cosplay")
        assert data["name"] == "test_cosplay"
        assert data["params"]["whiten_tone"] == "porcelain"
        assert data["params"]["lip_tint"] == "rose"

    def test_export_unknown_raises(self):
        with pytest.raises(ValueError, match="not found"):
            export_recipe("definitely_not_a_recipe_xyz")

    def test_round_trip_via_file(self, sample_recipe_json, isolated_recipe_state, tmp_path):
        json_path, data = sample_recipe_json
        import_recipe(json_path, save=True)
        # Export
        out = tmp_path / "exported.json"
        write_recipe_json("test_cosplay", out)
        # Validate
        is_valid, errors = validate_recipe(json.loads(out.read_text()))
        assert is_valid, errors
        # Re-import (different name to avoid duplicate)
        exported = json.loads(out.read_text())
        exported["name"] = "test_cosplay_v2"
        v2 = tmp_path / "v2.json"
        v2.write_text(json.dumps(exported))
        name = import_recipe(v2, save=True)
        assert name == "test_cosplay_v2"


# -------------------------------------------------------------------
# Built-in recipe export test
# -------------------------------------------------------------------

class TestBuiltinExport:
    def test_export_natural(self):
        data = export_recipe("natural")
        assert data["name"] == "natural"
        # Natural recipe should have a smooth value
        assert "smooth" in data["params"]

    def test_export_cosplay_has_slimming(self):
        data = export_recipe("cosplay")
        flat = data["params"]
        assert flat.get("slimming", 0) > 0
        assert flat.get("nose_blush", False) is True
