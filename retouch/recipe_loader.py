"""Load, save, and convert user recipes between JSON and engine format.

User-imported recipes live in a per-user directory (default: ``~/.cache/retouch/user_recipes/``)
and are loaded into the global ``RECIPES`` dict at engine import time.

JSON recipes use a **flat 0-1 scale** for all fractional values. The loader
converts them to the engine's internal scale (which uses mixed 0-1 / 0-100
values for different params) by looking up the ``PROCESSING_PARAMS`` spec
list.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .params import PROCESSING_PARAMS, ParamSpec, resolve_recipe
from .recipe_schema import validate_recipe
from .recipes import RECIPES


def _user_recipes_dir() -> Path:
    """Return the user recipes directory, creating it if needed."""
    if "RETOUCH_USER_RECIPES" in os.environ:
        d = Path(os.environ["RETOUCH_USER_RECIPES"])
    else:
        d = Path.home() / ".cache" / "retouch" / "user_recipes"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _spec_lookup() -> Dict[str, ParamSpec]:
    """Return a name→ParamSpec dict for fast lookup."""
    return {p.name: p for p in PROCESSING_PARAMS}


def _convert_param_to_engine(name: str, gui_value: Any) -> Any:
    """Convert a value from JSON recipe scale (0-1) to engine scale.

    For most params the JSON 0-1 maps directly to the engine 0-1.
    Some engine params use 0-100 (percent) — those are converted.
    The mapping is driven by the ``conversion`` field in ``ParamSpec``.
    """
    if not isinstance(gui_value, (int, float)) or isinstance(gui_value, bool):
        return gui_value

    spec = _spec_lookup().get(name)
    if spec is None:
        return gui_value
    conv = spec.conversion

    if conv in ("recipe_pct", "engine_pct"):
        return float(gui_value) * 100.0
    return float(gui_value)


def _convert_param_from_engine(name: str, engine_value: Any) -> Any:
    """Convert a value from engine scale back to JSON recipe scale (0-1)."""
    if not isinstance(engine_value, (int, float)) or isinstance(engine_value, bool):
        return engine_value

    spec = _spec_lookup().get(name)
    if spec is None:
        return engine_value
    conv = spec.conversion

    if conv in ("recipe_pct", "engine_pct"):
        return float(engine_value) / 100.0
    return float(engine_value)


def _flat_to_engine_recipe(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a flat JSON recipe dict to the engine's nested format.

    The engine uses a nested dict like ``{"frequency": {"smooth": 0.5}}``.
    JSON recipes use a flat dict like ``{"smooth": 0.5}``. This function
    maps flat names to nested paths using each param's semantic group.
    """
    engine: Dict[str, Any] = {}

    if "smooth" in flat:
        engine.setdefault("frequency", {})["smooth"] = _convert_param_to_engine("smooth", flat["smooth"])

    if "pore_synthesis" in flat:
        engine.setdefault("texture", {})["pore_synthesis"] = _convert_param_to_engine("pore_synthesis", flat["pore_synthesis"]) / 100.0

    for key in ("whiten", "whiten_tone", "equalize", "relight", "relight_azimuth", "relight_elevation"):
        if key in flat:
            engine.setdefault("skin", {})[key] = _convert_param_to_engine(key, flat[key])

    for key in ("eye_enhance", "catchlight", "dark_circles", "teeth_whiten"):
        if key in flat:
            engine.setdefault("eyes", {})[key] = _convert_param_to_engine(key, flat[key])

    if "lip_enhance" in flat:
        engine.setdefault("lips", {})["lip_enhance"] = _convert_param_to_engine("lip_enhance", flat["lip_enhance"])
    if "lip_tint" in flat:
        tint = flat["lip_tint"]
        engine.setdefault("lips", {})["tint"] = None if tint == "none" else tint
    if "lip_finish" in flat:
        engine.setdefault("lips", {})["gloss"] = _convert_param_to_engine("lip_finish", flat["lip_finish"])
        engine["lip_finish"] = flat["lip_finish"]

    if "blush" in flat:
        engine["blush"] = _convert_param_to_engine("blush", flat["blush"])

    if "hair_enhance" in flat:
        engine.setdefault("hair", {})["shine"] = _convert_param_to_engine("hair_enhance", flat["hair_enhance"])

    if "dodge_burn" in flat:
        engine["dodge_burn"] = {"amount": _convert_param_to_engine("dodge_burn", flat["dodge_burn"]) / 100.0}

    if "specular_bloom" in flat:
        engine["specular_bloom"] = _convert_param_to_engine("specular_bloom", flat["specular_bloom"])
    if "specular_bloom_tone" in flat:
        engine["specular_bloom_tone"] = flat["specular_bloom_tone"]

    if "bloom" in flat or "bloom_threshold" in flat or "bloom_softness" in flat:
        bloom_dict = engine.setdefault("bloom", {})
        if "bloom" in flat:
            bloom_dict["opacity"] = flat["bloom"]
        if "bloom_threshold" in flat:
            bloom_dict["threshold"] = flat["bloom_threshold"]
        if "bloom_softness" in flat:
            bloom_dict["softness"] = flat["bloom_softness"]

    if "impact" in flat:
        engine.setdefault("finish", {})["impact"] = _convert_param_to_engine("impact", flat["impact"]) / 100.0

    if "color_grade" in flat and flat["color_grade"] not in (None, "none"):
        engine.setdefault("color_harmony", {})["preset"] = flat["color_grade"]
    if "grade_intensity" in flat:
        engine.setdefault("color_harmony", {})["amount"] = _convert_param_to_engine("grade_intensity", flat["grade_intensity"])

    for modular_key in ("slimming", "nose_blush", "under_eye_blush", "white_costume_lift"):
        if modular_key in flat:
            engine[modular_key] = flat[modular_key]

    top_level_keys = (
        "clarity", "vibrance", "saturation", "glow", "vignette",
        "sharpen", "sharpen_radius", "subject_separation",
        "contrast", "brightness", "highlights", "shadows", "whites", "blacks",
        "grain", "halation", "chromatic_aberration", "lut", "auto_exposure",
        "shadow_hue", "shadow_sat", "midtone_hue", "midtone_sat", "highlight_hue", "highlight_sat",
        "white_balance_kelvin", "white_balance_tint",
        "bw_channel_mixer_r", "bw_channel_mixer_g", "bw_channel_mixer_b",
        "negative_split_tone_shadow", "negative_split_tone_highlight",
        "hsl_hue_global", "hsl_sat_global", "hsl_lum_global",
    )
    for key in top_level_keys:
        if key in flat:
            engine[key] = _convert_param_to_engine(key, flat[key])

    return engine


def _engine_to_flat_recipe(engine: Dict[str, Any]) -> Dict[str, Any]:
    """Convert the engine's nested recipe dict to a flat JSON recipe."""
    flat: Dict[str, Any] = {}

    freq = engine.get("frequency", {})
    if "smooth" in freq:
        flat["smooth"] = _convert_param_from_engine("smooth", freq["smooth"])

    texture = engine.get("texture", {})
    if "pore_synthesis" in texture:
        flat["pore_synthesis"] = _convert_param_from_engine("pore_synthesis", texture["pore_synthesis"])

    skin = engine.get("skin", {})
    for key in ("whiten", "whiten_tone", "equalize", "relight", "relight_azimuth", "relight_elevation"):
        if key in skin:
            flat[key] = _convert_param_from_engine(key, skin[key])

    eyes = engine.get("eyes", {})
    for key in ("eye_enhance", "catchlight", "dark_circles", "teeth_whiten"):
        if key in eyes:
            flat[key] = _convert_param_from_engine(key, eyes[key])

    lips = engine.get("lips", {})
    if "lip_enhance" in lips:
        flat["lip_enhance"] = _convert_param_from_engine("lip_enhance", lips["lip_enhance"])
    if "tint" in lips:
        flat["lip_tint"] = "none" if lips["tint"] is None else lips["tint"]
    if "gloss" in lips:
        flat["lip_finish"] = _convert_param_from_engine("lip_finish", lips["gloss"])

    if "blush" in engine:
        flat["blush"] = _convert_param_from_engine("blush", engine["blush"])

    hair = engine.get("hair", {})
    if "shine" in hair:
        flat["hair_enhance"] = _convert_param_from_engine("hair_enhance", hair["shine"])

    db = engine.get("dodge_burn", {})
    if isinstance(db, dict) and "amount" in db:
        flat["dodge_burn"] = db["amount"] * 100.0
    elif isinstance(db, (int, float)):
        flat["dodge_burn"] = float(db)

    if "specular_bloom" in engine:
        flat["specular_bloom"] = engine["specular_bloom"]
    if "specular_bloom_tone" in engine:
        flat["specular_bloom_tone"] = engine["specular_bloom_tone"]

    bloom = engine.get("bloom", {})
    if isinstance(bloom, dict):
        if "opacity" in bloom:
            flat["bloom"] = bloom["opacity"]
        if "threshold" in bloom:
            flat["bloom_threshold"] = bloom["threshold"]
        if "softness" in bloom:
            flat["bloom_softness"] = bloom["softness"]

    finish = engine.get("finish", {})
    if isinstance(finish, dict) and "impact" in finish:
        flat["impact"] = finish["impact"] * 100.0

    harmony = engine.get("color_harmony", {})
    if "preset" in harmony and harmony["preset"] not in (None, "none"):
        flat["color_grade"] = harmony["preset"]
    if "amount" in harmony:
        flat["grade_intensity"] = harmony["amount"]

    for key in (
        "slimming", "lip_finish", "nose_blush", "under_eye_blush", "white_costume_lift",
        "clarity", "vibrance", "saturation", "glow", "vignette",
        "sharpen", "sharpen_radius", "subject_separation",
        "contrast", "brightness", "highlights", "shadows", "whites", "blacks",
        "grain", "halation", "chromatic_aberration", "lut", "auto_exposure",
        "shadow_hue", "shadow_sat", "midtone_hue", "midtone_sat", "highlight_hue", "highlight_sat",
        "white_balance_kelvin", "white_balance_tint",
        "bw_channel_mixer_r", "bw_channel_mixer_g", "bw_channel_mixer_b",
        "negative_split_tone_shadow", "negative_split_tone_highlight",
        "hsl_hue_global", "hsl_sat_global", "hsl_lum_global",
    ):
        if key in engine and engine[key] is not None:
            flat[key] = _convert_param_from_engine(key, engine[key])

    return flat


def import_recipe(json_path: Path, *, save: bool = True) -> str:
    """Import a JSON recipe file.

    Validates the JSON, converts to engine format, adds to ``RECIPES``,
    and optionally saves a copy to the user recipes directory.

    Returns:
        The recipe name.
    """
    json_path = Path(json_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    is_valid, errors = validate_recipe(data)
    if not is_valid:
        raise ValueError(f"Invalid recipe '{json_path}':\n  " + "\n  ".join(errors))

    name = data["name"]
    if name in RECIPES:
        raise ValueError(f"Recipe '{name}' already exists. Choose a different name or remove the existing one.")

    flat_params = data.get("params", {})
    engine_recipe = _flat_to_engine_recipe(flat_params)

    parent = data.get("extends")
    if parent:
        if parent not in RECIPES:
            raise ValueError(f"Parent recipe '{parent}' not found in built-in recipes.")
        engine_recipe["extends"] = parent

    RECIPES[name] = engine_recipe

    if save:
        dest = _user_recipes_dir() / f"{name}.json"
        dest.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return name


def export_recipe(recipe_name: str) -> Dict[str, Any]:
    """Export an existing recipe as a JSON-serializable dict."""
    if recipe_name not in RECIPES:
        raise ValueError(f"Recipe '{recipe_name}' not found.")

    engine_recipe = resolve_recipe(recipe_name)
    flat = _engine_to_flat_recipe(engine_recipe)

    return {
        "$schema": "https://retouch.dennisbrian.com/schemas/recipe-v1.json",
        "name": recipe_name,
        "version": "1.0",
        "description": f"Exported from {recipe_name}",
        "extends": engine_recipe.get("extends"),
        "author": "Pro Max Retouch",
        "tags": [],
        "params": flat,
    }


def list_user_recipes() -> List[str]:
    """List all user-imported recipe names."""
    d = _user_recipes_dir()
    return sorted(p.stem for p in d.glob("*.json"))


def remove_user_recipe(name: str) -> bool:
    """Remove a user-imported recipe from disk and the RECIPES dict."""
    name = re.sub(r"[^a-z0-9_]", "", name.lower())
    if name not in RECIPES:
        return False
    d = _user_recipes_dir()
    f = d / f"{name}.json"
    if f.exists():
        f.unlink()
    del RECIPES[name]
    return True


def load_user_recipes() -> List[str]:
    """Load all user recipes from disk into the global RECIPES dict.

    Called once at engine init time. Returns the names of the recipes
    that were loaded.
    """
    loaded = []
    d = _user_recipes_dir()
    for json_path in d.glob("*.json"):
        try:
            name = import_recipe(json_path, save=False)
            loaded.append(name)
        except (ValueError, json.JSONDecodeError) as e:
            print(f"Warning: failed to load user recipe {json_path}: {e}")
    return loaded


def write_recipe_json(recipe_name: str, dest: Optional[Path] = None) -> Path:
    """Write an existing recipe as a JSON file. Returns the output path."""
    data = export_recipe(recipe_name)
    if dest is None:
        dest = _user_recipes_dir() / f"{recipe_name}.json"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return dest
