# AI Recipe Generator

**Date:** 2026-06-22
**Status:** Design document
**Owner:** Dennis

---

## Vision

Enable anyone to **describe a retouching style in natural language** and have an AI agent (Claude, GPT, Gemini, etc.) generate a valid, drop-in recipe that integrates seamlessly into Pro Max Retouch.

**User flow:**
1. User describes a style: *"Fujifilm cosplay with porcelain skin, pink lips, dreamy glow"*
2. AI agent generates a recipe JSON file using the standard schema
3. User runs `python3 -m retouch.recipe_loader import cosplay_dream.json`
4. The recipe appears in the GUI's recipe dropdown and CLI's `--recipe` choices immediately

No Python required. No PRs. No build steps. Just JSON.

---

## Architecture

```
┌─────────────────┐
│  User Prompt    │  "Fujifilm cosplay with porcelain skin..."
└────────┬────────┘
         │ natural language
         ▼
┌─────────────────┐
│   AI Agent      │  Claude / GPT / Gemini
│  (with prompt)  │  reads PROMPT.md + SCHEMA.md
└────────┬────────┘
         │ generates
         ▼
┌─────────────────┐
│  recipe.json    │  validates against JSON Schema
│  (standard      │  errors → AI agent retries
│   format)       │
└────────┬────────┘
         │ imports via
         ▼
┌─────────────────────────────────────┐
│  retouch/recipe_loader.py            │
│  - Validates JSON                    │
│  - Converts to RECIPES dict format   │
│  - Adds to in-memory RECIPES         │
│  - Saves to user recipe dir          │
└────────┬────────────────────────────┘
         │ available in
         ▼
┌─────────────────────────────────────┐
│  GUI dropdown | CLI --recipe | API   │
└─────────────────────────────────────┘
```

---

## Recipe JSON Schema

Recipes are **self-documenting JSON files** with three sections:

```json
{
  "$schema": "https://retouch.dennisbrian.com/schemas/recipe-v1.json",
  "name": "cosplay_dream",
  "version": "1.0",
  "description": "Fujifilm cosplay with porcelain skin, pink lips, dreamy glow",
  "extends": null,
  "author": "AI Agent (Claude 4)",
  "tags": ["cosplay", "fujifilm", "dreamy", "soft"],
  
  "params": {
    "smooth": 0.60,
    "mid_reduction": 0.45,
    "texture_opacity": 0.85,
    "pore_synthesis": 0.30,
    "whiten": 0.55,
    "whiten_tone": "porcelain",
    "equalize": 0.50,
    "blemish": 0.40,
    "blush": 0.35,
    "nose_blush": true,
    "under_eye_blush": true,
    "white_costume_lift": true,
    "eye_enhance": 0.50,
    "catchlight": 0.30,
    "dark_circles": 0.40,
    "teeth_whiten": 0.30,
    "lip_enhance": 0.40,
    "lip_tint": "rose",
    "lip_finish": "velvet",
    "hair_enhance": 0.40,
    "dodge_burn": 0.20,
    "specular_bloom": 0.50,
    "specular_bloom_tone": "rosy",
    "slimming": 0.30,
    "relight": 0.30,
    "relight_azimuth": 45,
    "relight_elevation": 30,
    "bloom": 0.20,
    "bloom_threshold": 200,
    "bloom_softness": 40,
    "sharpen": 0.30,
    "sharpen_radius": 1.2,
    "subject_separation": 0.0,
    "glow": 0.40,
    "vignette": 0.15,
    "color_grade": "fantasy",
    "grade_intensity": 0.60,
    "saturation": 0.15,
    "vibrance": 0.20,
    "clarity": 0.10,
    "contrast": 10,
    "brightness": 5,
    "highlights": -5,
    "shadows": 5,
    "whites": 5,
    "blacks": -5,
    "color_grade_stack": null,
    "grain": 0.05,
    "halation": 0.10,
    "chromatic_aberration": 0.0,
    "lut": null,
    "auto_exposure": false,
    "shadow_hue": 220,
    "shadow_sat": 15,
    "midtone_hue": 0,
    "midtone_sat": 0,
    "highlight_hue": 330,
    "highlight_sat": 12
  }
}
```

### Key Design Decisions

1. **Flat structure** — AI agents reason better about flat key-value than nested dicts
2. **0-1 scale** — All values use 0-1 (or 0-100 where the field is naturally a percentage)
3. **`extends`** — Optional parent recipe name for inheritance
4. **`description` and `tags`** — Self-documenting for the GUI's recipe picker
5. **No nested `frequency`/`skin`/`eyes` dicts** — Those were an artifact of the old `build_context()` code path; the new `params.py` registry uses flat names

---

## File Structure

```
retouch/
├── recipes.py              # Built-in recipes (Python dict, unchanged)
├── recipe_schema.py        # NEW: JSON schema definition + validation
├── recipe_loader.py        # NEW: Import/export/validate JSON recipes
└── params.py               # Param registry (uses spec, unchanged)

recipes_user/              # NEW: User-imported recipes live here
├── cosplay_dream.json
├── fuji_porcelain.json
└── wedding_classic.json

docs/
└── RECIPE_AUTHORING.md     # NEW: How to write recipes by hand

prompts/
└── recipe_generator.md     # NEW: AI agent prompt template
```

---

## Components

### 1. `retouch/recipe_schema.py` — Schema Definition

Defines the JSON Schema for recipes. Auto-generates from `PROCESSING_PARAMS` so the schema stays in sync with the engine.

```python
"""Recipe JSON schema definition and validation."""

import jsonschema
from .params import PROCESSING_PARAMS

RECIPE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "Retouch Recipe",
    "type": "object",
    "required": ["name", "params"],
    "properties": {
        "name": {
            "type": "string",
            "pattern": "^[a-z][a-z0-9_]*$",
            "description": "Lowercase recipe name (used as CLI --recipe value)"
        },
        "version": {"type": "string", "default": "1.0"},
        "description": {"type": "string", "maxLength": 500},
        "extends": {"type": ["string", "null"], "default": None},
        "author": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "params": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                # Auto-generated from PROCESSING_PARAMS
                # Each field has type, range, default, description
            }
        }
    }
}

def build_recipe_schema() -> dict:
    """Build the full recipe JSON schema from PROCESSING_PARAMS."""
    param_props = {}
    for spec in PROCESSING_PARAMS:
        param_props[spec.name] = _spec_to_json_schema_property(spec)
    RECIPE_SCHEMA["properties"]["params"]["properties"] = param_props
    return RECIPE_SCHEMA

def validate_recipe(recipe_dict: dict) -> tuple[bool, list[str]]:
    """Validate a recipe dict against the schema. Returns (is_valid, errors)."""
    try:
        jsonschema.validate(recipe_dict, build_recipe_schema())
        return True, []
    except jsonschema.ValidationError as e:
        return False, [str(e)]
```

### 2. `retouch/recipe_loader.py` — Import/Export Logic

```python
"""Load, save, and convert user recipes between JSON and engine format."""

import json
import copy
from pathlib import Path
from .recipe_schema import validate_recipe
from .recipes import RECIPES, resolve_recipe

USER_RECIPES_DIR = Path.home() / ".cache" / "retouch" / "user_recipes"

def import_recipe(json_path: Path) -> str:
    """Import a JSON recipe file. Returns the recipe name."""
    data = json.loads(json_path.read_text())
    is_valid, errors = validate_recipe(data)
    if not is_valid:
        raise ValueError(f"Invalid recipe: {errors}")
    
    recipe_name = data["name"]
    params = data["params"]
    
    # Convert 0-1 scale to engine scale (e.g., smooth 0.6 → frequency.smooth 0.6)
    engine_recipe = _convert_to_engine_format(params)
    
    # Add to in-memory RECIPES
    RECIPES[recipe_name] = engine_recipe
    
    # Save to user recipes dir
    USER_RECIPES_DIR.mkdir(parents=True, exist_ok=True)
    dest = USER_RECIPES_DIR / f"{recipe_name}.json"
    dest.write_text(json.dumps(data, indent=2))
    
    return recipe_name

def export_recipe(recipe_name: str) -> dict:
    """Export an existing recipe as JSON. Returns the JSON dict."""
    engine_recipe = resolve_recipe(recipe_name)
    params = _convert_from_engine_format(engine_recipe)
    return {
        "$schema": "https://retouch.dennisbrian.com/schemas/recipe-v1.json",
        "name": recipe_name,
        "version": "1.0",
        "description": f"Exported from {recipe_name}",
        "author": "Pro Max Retouch",
        "tags": [],
        "params": params,
    }

def list_user_recipes() -> list[str]:
    """List all user-imported recipe names."""
    if not USER_RECIPES_DIR.exists():
        return []
    return [p.stem for p in USER_RECIPES_DIR.glob("*.json")]

def load_user_recipes() -> None:
    """Load all user recipes from disk into the global RECIPES dict."""
    # Called at engine init
    ...
```

### 3. `prompts/recipe_generator.md` — AI Agent Prompt Template

The prompt that users copy-paste into Claude/GPT/Gemini to generate recipes. Designed to be self-contained — the AI agent only needs this file (plus the schema reference) to generate valid recipes.

```markdown
# Recipe Generator

You are an expert photo retoucher. Generate a JSON recipe for the
Pro Max Retouch engine that achieves the described style.

## Schema Reference

The recipe JSON has this structure:
{
  "name": "lowercase_name",
  "version": "1.0",
  "description": "Human-readable description",
  "author": "AI Agent (Claude 4)",
  "tags": ["tag1", "tag2"],
  "params": {
    // See SCHEMA.md for full param list with ranges and descriptions
  }
}

All param values use 0-1 scale unless noted otherwise (e.g., bloom_threshold 0-255).

## Output Format

Output ONLY the JSON object. No markdown code fences. No explanation.
The JSON must validate against the schema (see SCHEMA.md).

## Examples

User: "Natural outdoor portrait with soft skin"
Output: {"name": "natural_outdoor", "params": {"smooth": 0.3, "whiten": 0.15, ...}}

User: "Heavy cosplay with porcelain skin, pink lips, dreamy glow"
Output: {"name": "cosplay_dream", "params": {"smooth": 0.6, "whiten": 0.5, "whiten_tone": "porcelain", ...}}

[... more examples ...]
```

### 4. `scripts/import_recipe.py` — CLI Tool

```python
"""CLI to import a JSON recipe into Pro Max Retouch."""
import argparse
from pathlib import Path
from retouch.recipe_loader import import_recipe, list_user_recipes

def main():
    parser = argparse.ArgumentParser(description="Import a recipe JSON file")
    parser.add_argument("json_path", type=Path, help="Path to recipe JSON file")
    parser.add_argument("--list", action="store_true", help="List user recipes")
    args = parser.parse_args()
    
    if args.list:
        print("User recipes:")
        for name in list_user_recipes():
            print(f"  - {name}")
        return
    
    name = import_recipe(args.json_path)
    print(f"✓ Imported '{name}' — now available in GUI and CLI")

if __name__ == "__main__":
    main()
```

---

## How the AI Agent Generates Recipes

### Example 1: User prompt → JSON

**User prompt to AI:**
> "Make a recipe for a moody cinematic portrait — dark shadows, lifted blacks, teal-orange color grade, heavy grain, vignette"

**AI agent output:**
```json
{
  "name": "moody_cinematic",
  "description": "Dark cinematic portrait with teal-orange grading",
  "tags": ["moody", "cinematic", "teal-orange"],
  "params": {
    "smooth": 0.25,
    "whiten": 0.0,
    "equalize": 0.30,
    "contrast": 25,
    "brightness": -10,
    "highlights": -15,
    "shadows": 15,
    "whites": -10,
    "blacks": 10,
    "color_grade": "natural",
    "grade_intensity": 0.80,
    "color_grade_stack": [
      {"preset": "cyberpunk", "intensity": 0.40, "mask": "skin"}
    ],
    "split_tone_shadow_hue": 200,
    "split_tone_shadow_saturation": 25,
    "split_tone_highlight_hue": 25,
    "split_tone_highlight_saturation": 20,
    "grain": 0.08,
    "vignette": 0.30,
    "bloom": 0.10,
    "impact": 0.50,
    "saturation": -5,
    "vibrance": 10
  }
}
```

**User action:**
```bash
python3 -m scripts.import_recipe moody_cinematic.json
# ✓ Imported 'moody_cinematic' — now available in GUI and CLI
```

### Example 2: Extending an existing recipe

```json
{
  "name": "fuji_porcelain_soft",
  "extends": "fuji_porcelain",
  "description": "Softer version of fuji_porcelain with less smoothing",
  "params": {
    "smooth": 0.55
  }
}
```

Inherits everything from `fuji_porcelain` but overrides `smooth` to 0.55 (vs the base's 0.72).

---

## Schema Auto-Generation

The `build_recipe_schema()` function reads `PROCESSING_PARAMS` from `retouch/params.py` and auto-generates the JSON schema. This means:

- **Adding a new engine param** → automatically available in recipes (no schema update needed)
- **Changing a param's range** → automatically reflected in the schema
- **No drift** between engine capabilities and recipe schema

Each `ParamSpec` has:
- `name` → JSON property name
- `cli_type` → JSON Schema type (int, float, str, bool)
- `min_val` / `max_val` → JSON Schema minimum/maximum
- `default` → JSON Schema default
- `conversion` → handled by `_convert_to_engine_format()`

---

## Validation Strategy

The recipe JSON is validated in 3 layers:

1. **JSON Schema validation** (structural) — types, ranges, required fields
2. **Semantic validation** — e.g., `extends` references a real recipe, `lip_tint` is in valid choices
3. **Engine smoke test** — try `resolve_recipe(name)` to confirm the engine can build a `ProcessingContext` from it

If any layer fails, the import raises a clear error with the field and reason.

---

## CLI Commands (after implementation)

```bash
# Import a recipe from JSON
python3 -m retouch.recipe_loader import path/to/recipe.json

# List user-imported recipes
python3 -m retouch.recipe_loader list

# Export an existing recipe as JSON
python3 -m retouch.recipe_loader export --recipe natural --output natural.json

# Validate a JSON recipe without importing
python3 -m retouch.recipe_loader validate path/to/recipe.json

# Show the JSON schema (for AI agent context)
python3 -m retouch.recipe_loader schema
```

---

## GUI Integration

The GUI's recipe dropdown should auto-populate with both built-in recipes (from `retouch/recipes.py`) and user-imported recipes (from `~/.cache/retouch/user_recipes/`). User recipes show a 🤖 icon or "(user)" tag to distinguish them.

---

## Effort Estimate

| Component | Effort | Risk |
|---|---|---|
| `recipe_schema.py` (auto-gen from PROCESSING_PARAMS) | 2 days | Low |
| `recipe_loader.py` (import/export/validate) | 3 days | Medium (need to handle extends, version conflicts) |
| Scale conversion (0-1 ↔ engine scale) | 1 day | Low |
| `prompts/recipe_generator.md` (AI prompt) | 1 day | Low |
| CLI (`scripts/import_recipe.py`) | 1 day | Low |
| Tests (`test_recipe_schema.py`, `test_recipe_loader.py`) | 2 days | Low |
| GUI integration (dropdown shows user recipes) | 1 day | Low |
| Documentation (`RECIPE_AUTHORING.md`) | 1 day | Low |
| **Total** | **~2 weeks** | **Low** |

---

## Success Metrics

- A user can describe a style in 1 sentence and get a working recipe in 1 minute (prompt → JSON → import)
- Recipe JSON files are 100% schema-validated (no broken recipes in user dir)
- The schema auto-regenerates when `PROCESSING_PARAMS` changes (no manual sync)
- Zero Python knowledge required to author a new recipe

---

## Future Extensions (Phase 2)

- **Recipe marketplace** — share recipes via URL or GitHub gist
- **Visual recipe editor** — drag-and-drop sliders in the GUI, save as JSON
- **Recipe diff tool** — show what changed between two recipe versions
- **Recipe A/B testing** — apply two recipes to the same image, compare results
- **Recipe inheritance chains** — multi-level extends for theme families
- **Per-photo recipe override** — CLI flag like `--override smooth=0.5`

---

## Why This Is Achievable

1. **`PROCESSING_PARAMS` is the single source of truth** (added in this session's refactor)
2. **`recipe_to_params()` already exists** — we can reverse it for import
3. **JSON Schema validation is mature** — just use `jsonschema` package
4. **AI agents are excellent at structured JSON output** with clear schemas
5. **The existing recipe format is already close** — just needs flattening and standardization

This is the killer feature that turns the engine from a "photo editor" into a **"retouching platform"** — anyone can extend it without coding.
