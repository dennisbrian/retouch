#!/usr/bin/env python3
"""Run the release Core-recipe matrix and create a human-review worksheet.

Usage intentionally requires named corpus cases, for example::

    python scripts/recipes/core_recipe_certification.py \
      --case skin_dark=/path/dark.jpg \
      --case glasses=/path/glasses.jpg \
      --output /tmp/core-cert

The default is face-aware. ``--global-only`` is available only for a
diagnostic smoke run and is written into every manifest; it can never be
treated as face-retouch certification.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retouch.core_recipes import CORE_RECIPE_NAMES, CORE_RECIPE_REVIEW_DIMENSIONS
from retouch.certification import (
    certification_fields,
    core_case_automatic_pass,
    core_matrix_final_certified,
)
from retouch.recipes import RECIPES


def _parse_cases(raw_cases: Sequence[str]) -> List[Tuple[str, Path]]:
    cases: List[Tuple[str, Path]] = []
    for raw in raw_cases:
        if "=" not in raw:
            raise ValueError(f"case must be NAME=PATH, got {raw!r}")
        name, raw_path = raw.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser().resolve()
        if not name or not path.is_file():
            raise ValueError(f"case {raw!r} must name an existing image file")
        cases.append((name, path))
    if not cases:
        raise ValueError("provide at least one --case NAME=PATH")
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, help="Corpus case as NAME=IMAGE_PATH; repeatable")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max-dim", type=int, default=None)
    parser.add_argument("--global-only", action="store_true", help="Diagnostic only; never face-aware certification")
    parser.add_argument("--stop-on-fail", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    unknown = [name for name in CORE_RECIPE_NAMES if name not in RECIPES]
    if unknown:
        raise RuntimeError(f"Core recipe selection contains unknown recipes: {unknown}")
    cases = _parse_cases(args.case)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    recipes = ",".join(CORE_RECIPE_NAMES)
    rows: List[Dict[str, object]] = []
    for case_name, image_path in cases:
        case_output = output / case_name
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/recipes/recipe_sweep.py"),
            str(image_path), "-o", str(case_output),
            "--recipes", recipes, "--compare",
        ]
        if args.max_dim is not None:
            command.extend(["--max-dim", str(args.max_dim)])
        if args.global_only:
            command.append("--global-only")
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), check=False)
        manifest_path = case_output / "manifest.json"
        case_manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        face_aware_run = bool(
            completed.returncode == 0
            and case_manifest.get("global_only") is False
        )
        automatic_pass = core_case_automatic_pass(
            case_manifest,
            case_output,
            len(CORE_RECIPE_NAMES),
            returncode=completed.returncode,
        )
        rows.append({
            "case": case_name,
            "input": str(image_path),
            "returncode": completed.returncode,
            "manifest": str(manifest_path),
            "global_only": bool(case_manifest.get("global_only", args.global_only)),
            "status": "done" if completed.returncode == 0 else "failed",
            "recipes": list(CORE_RECIPE_NAMES),
            "recipe_count": len(case_manifest.get("outputs", [])) if isinstance(case_manifest.get("outputs"), list) else 0,
            **certification_fields(
                face_aware_run=face_aware_run,
                automatic_pass=automatic_pass,
                human_review="pending",
            ),
        })
        if completed.returncode != 0 and args.stop_on_fail:
            break

    matrix = {
        "version": 1,
        "recipes": list(CORE_RECIPE_NAMES),
        "recipe_count": len(CORE_RECIPE_NAMES),
        "full_catalog_count_at_run": len(RECIPES),
        "global_only": bool(args.global_only),
        "cases": rows,
    }
    review_rows = []
    for case_name, _ in cases:
        for recipe in CORE_RECIPE_NAMES:
            review_rows.append({
                "case": case_name,
                "recipe": recipe,
                "dimension": CORE_RECIPE_REVIEW_DIMENSIONS[recipe],
                "automatic_status": "pending",
                "human_natural_output": "pending",
                "reviewer": "",
                "notes": "",
            })
    matrix.update({
        **certification_fields(
            face_aware_run=bool(rows) and all(row["face_aware_run"] for row in rows),
            automatic_pass=bool(rows) and all(row["automatic_pass"] for row in rows),
            human_review="pending",
        ),
        "final_certified": core_matrix_final_certified(rows, review_rows, len(CORE_RECIPE_NAMES)),
    })
    (output / "matrix_manifest.json").write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    (output / "human_review.json").write_text(json.dumps({"version": 1, "rows": review_rows}, indent=2), encoding="utf-8")
    return 0 if matrix["automatic_pass"] else 1


if __name__ == "__main__":
    parser = build_parser()
    try:
        raise SystemExit(run(parser.parse_args()))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
