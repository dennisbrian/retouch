"""CLI for importing/exporting/validating recipe JSON files.

Usage:
    python3 -m retouch.recipe_loader_cli import path/to/recipe.json
    python3 -m retouch.recipe_loader_cli list
    python3 -m retouch.recipe_loader_cli export --recipe natural --output natural.json
    python3 -m retouch.recipe_loader_cli validate path/to/recipe.json
    python3 -m retouch.recipe_loader_cli schema
    python3 -m retouch.recipe_loader_cli remove my_recipe
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .recipe_loader import (
    export_recipe,
    import_recipe,
    list_user_recipes,
    remove_user_recipe,
    write_recipe_json,
)
from .recipe_schema import RECIPE_SCHEMA_URL, build_recipe_schema, validate_recipe


def _cmd_import(args: argparse.Namespace) -> int:
    try:
        name = import_recipe(Path(args.json_path))
        print(f"✓ Imported '{name}' — now available in GUI and CLI")
        print(f"  File: {Path(args.json_path).resolve()}")
        return 0
    except (ValueError, FileNotFoundError) as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1


def _cmd_list(args: argparse.Namespace) -> int:
    recipes = list_user_recipes()
    if not recipes:
        print("No user recipes found.")
        print("Import one with: python3 -m retouch.recipe_loader_cli import <path.json>")
        return 0
    print(f"User recipes ({len(recipes)}):")
    for name in recipes:
        print(f"  - {name}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    try:
        if args.output:
            dest = write_recipe_json(args.recipe, Path(args.output))
        else:
            data = export_recipe(args.recipe)
            print(json.dumps(data, indent=2))
            return 0
        print(f"✓ Exported '{args.recipe}' to {dest}")
        return 0
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        data = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"✗ Failed to read JSON: {e}", file=sys.stderr)
        return 1
    is_valid, errors = validate_recipe(data)
    if is_valid:
        print(f"✓ Valid recipe: {data.get('name', '<unnamed>')}")
        return 0
    print(f"✗ Invalid recipe:")
    for err in errors:
        print(f"  - {err}")
    return 1


def _cmd_schema(args: argparse.Namespace) -> int:
    schema = build_recipe_schema()
    if args.url:
        print(RECIPE_SCHEMA_URL)
    else:
        print(json.dumps(schema, indent=2))
    return 0


def _cmd_remove(args: argparse.Namespace) -> int:
    if remove_user_recipe(args.name):
        print(f"✓ Removed '{args.name}'")
        return 0
    print(f"✗ Recipe '{args.name}' not found in user recipes", file=sys.stderr)
    return 1


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pro Max Retouch recipe manager",
        epilog="See prompts/recipe_generator.md for the AI recipe generator prompt.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_import = sub.add_parser("import", help="Import a recipe from a JSON file")
    p_import.add_argument("json_path", help="Path to the recipe JSON file")
    p_import.set_defaults(func=_cmd_import)

    p_list = sub.add_parser("list", help="List all user-imported recipes")
    p_list.set_defaults(func=_cmd_list)

    p_export = sub.add_parser("export", help="Export an existing recipe as JSON")
    p_export.add_argument("--recipe", required=True, help="Recipe name to export")
    p_export.add_argument("--output", help="Output file path (default: stdout)")
    p_export.set_defaults(func=_cmd_export)

    p_validate = sub.add_parser("validate", help="Validate a recipe JSON file")
    p_validate.add_argument("json_path", help="Path to the recipe JSON file")
    p_validate.set_defaults(func=_cmd_validate)

    p_schema = sub.add_parser("schema", help="Print the recipe JSON Schema")
    p_schema.add_argument("--url", action="store_true", help="Just print the schema URL")
    p_schema.set_defaults(func=_cmd_schema)

    p_remove = sub.add_parser("remove", help="Remove a user-imported recipe")
    p_remove.add_argument("name", help="Recipe name to remove")
    p_remove.set_defaults(func=_cmd_remove)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
