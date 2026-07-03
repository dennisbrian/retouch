"""JSON Schema definition and validation for user-importable recipes.

This module auto-generates the recipe JSON Schema from ``PROCESSING_PARAMS``,
so the schema is always in sync with the engine's parameter registry.
"""

from __future__ import annotations

from typing import Any, Dict

import jsonschema

from .params import PROCESSING_PARAMS, ParamSpec


RECIPE_SCHEMA_URL = "https://retouch.dennisbrian.com/schemas/recipe-v1.json"


def _spec_to_json_property(spec: ParamSpec) -> Dict[str, Any]:
    """Convert a ParamSpec into a JSON Schema property definition.

    JSON recipes always use:
    - ``number`` type (0-1 scale) for fractional params even if the engine/CLI
      uses int. This makes AI generation easier and matches the documented
      "all values 0-1" convention.
    - 0-1 range for numeric params (the loader converts to engine scale).
    - For string params, optional enum values are detected from a sibling
      ``choices`` attribute (e.g. ``lip_tint`` → ``["none", "cosplay", ...]``).
    """
    prop: Dict[str, Any] = {
        "description": f"{spec.name} (cli: {spec.cli_flag or 'n/a'})",
    }

    if spec.cli_type is bool:
        prop["type"] = "boolean"
    elif spec.cli_type is str:
        prop["type"] = "string"
        choices = getattr(spec, "choices", None)
        if choices:
            prop["enum"] = list(choices)
    elif spec.cli_type in (int, float):
        prop["type"] = "number"
        # JSON recipes always use 0-1 scale; the loader converts internally.
        if spec.conversion in ("recipe_pct", "engine_pct"):
            prop["minimum"] = 0.0
            prop["maximum"] = 1.0
        elif spec.min_val is not None:
            prop["minimum"] = spec.min_val
        if spec.max_val is not None and spec.conversion not in ("recipe_pct", "engine_pct"):
            prop["maximum"] = spec.max_val
    else:
        prop["type"] = ["string", "number", "boolean", "integer", "null"]

    return prop


def build_recipe_schema() -> dict[str, Any]:
    """Build the full recipe JSON Schema from the PROCESSING_PARAMS registry.

    The schema is dynamic — it picks up any new parameters added to
    ``PROCESSING_PARAMS`` automatically.
    """
    param_props = {spec.name: _spec_to_json_property(spec) for spec in PROCESSING_PARAMS}

    schema: Dict[str, Any] = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": RECIPE_SCHEMA_URL,
        "title": "Retouch Recipe",
        "description": "A user-importable recipe for the Pro Max Retouch engine.",
        "type": "object",
        "required": ["name", "params"],
        "properties": {
            "$schema": {"type": "string"},
            "name": {
                "type": "string",
                "pattern": r"^[a-z][a-z0-9_]*$",
                "description": "Lowercase recipe name (used as CLI --recipe value)",
                "maxLength": 64,
            },
            "version": {"type": "string", "default": "1.0"},
            "description": {"type": "string", "maxLength": 500},
            "extends": {
                "type": ["string", "null"],
                "default": None,
                "description": "Name of parent recipe to inherit from (null = no inheritance)",
            },
            "author": {"type": "string", "maxLength": 200},
            "tags": {
                "type": "array",
                "items": {"type": "string", "maxLength": 32},
                "maxItems": 20,
                "description": "Keywords for searchability",
            },
            "params": {
                "type": "object",
                "description": "Recipe parameters. Only include params you want to set.",
                "additionalProperties": False,
                "properties": param_props,
            },
        },
        "additionalProperties": False,
    }
    return schema


def validate_recipe(recipe_dict: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate a recipe dict against the schema.

    Returns:
        (is_valid, errors) — ``is_valid`` is True if no errors, ``errors`` is a
        list of human-readable error messages.
    """
    schema = build_recipe_schema()
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(recipe_dict), key=lambda e: e.path)
    if not errors:
        return True, []
    return False, [f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors]
