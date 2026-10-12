#!/usr/bin/env python3
"""Render selected recipes across a folder and export visual-QA artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cli import _apply_global_finish  # noqa: E402
from retouch import RetouchEngine  # noqa: E402
from retouch.batch_processor import generate_contact_sheet  # noqa: E402
from retouch.io import (  # noqa: E402
    IMAGE_EXTENSIONS,
    encode_write_params,
    imread_exif,
    make_comparison,
    resize_for_processing,
    write_image_with_icc,
)
from retouch.recipes import RECIPES  # noqa: E402


def _parse_recipes(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return sorted(RECIPES)
    names = [name.strip() for name in raw.split(",") if name.strip()]
    unknown = [name for name in names if name not in RECIPES]
    if unknown:
        raise ValueError(f"unknown recipe(s): {', '.join(unknown)}")
    if not names:
        raise ValueError("at least one recipe is required")
    return names


def _safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def _write_image(path: Path, img_bgr: np.ndarray, image_format: str, quality: int) -> None:
    if image_format == "png":
        write_image_with_icc(str(path), img_bgr, bit_depth=8, quality=quality)
    else:
        ok = cv2.imwrite(str(path), img_bgr, encode_write_params(image_format, quality))
        if not ok:
            raise IOError(f"failed to write output image: {path}")
    if not path.exists() or path.stat().st_size <= 0:
        raise IOError(f"output image was not written: {path}")


def _image_paths(input_dir: Path, recursive: bool, limit: Optional[int]) -> List[Path]:
    candidates = input_dir.rglob("*") if recursive else input_dir.iterdir()
    paths = sorted(path for path in candidates if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit] if limit is not None else paths


def run_batch(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input directory not found: {input_dir}")

    recipes = _parse_recipes(args.recipes)
    images = _image_paths(input_dir, args.recursive, args.limit)
    if not images:
        raise ValueError(f"no supported images found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    recipe_paths: Dict[str, List[Path]] = {recipe: [] for recipe in recipes}
    manifest: Dict[str, Any] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "recipes": recipes,
        "global_only": bool(args.global_only),
        "max_dim": args.max_dim,
        "compare": bool(args.compare),
        "images": [],
        "contact_sheets": {},
    }

    engine: Optional[RetouchEngine] = None
    if not args.global_only:
        engine = RetouchEngine()

    try:
        for image_index, input_path in enumerate(images, start=1):
            row: Dict[str, Any] = {"input": str(input_path), "outputs": []}
            try:
                source = imread_exif(input_path)
                original_shape = list(source.shape[:2])
                source, scale = resize_for_processing(source, args.max_dim)
                row.update({"original_shape": original_shape, "processed_shape": list(source.shape[:2]), "scale": scale})
            except Exception as exc:  # noqa: BLE001 - one corrupt file must not lose a batch
                row["error"] = f"input: {exc}"
                manifest["images"].append(row)
                print(f"[{image_index:03d}/{len(images):03d}] {input_path.name}: FAILED to read: {exc}")
                continue

            for recipe in recipes:
                started = time.perf_counter()
                recipe_dir = output_dir / _safe_name(recipe)
                recipe_dir.mkdir(exist_ok=True)
                output_path = recipe_dir / f"{input_path.stem}.{args.format}"
                output_row: Dict[str, Any] = {"recipe": recipe, "output": str(output_path.relative_to(output_dir))}
                try:
                    if args.global_only:
                        result = np.asarray(_apply_global_finish(source, {"recipe": recipe}))
                    else:
                        if engine is None:
                            raise RuntimeError("retouch engine is unavailable")
                        result = np.asarray(engine.process(source, recipe=recipe))
                    _write_image(output_path, result, args.format, args.quality)
                    recipe_paths[recipe].append(output_path)
                    if args.compare:
                        compare_path = recipe_dir / f"{input_path.stem}_compare.{args.format}"
                        make_comparison(source, result, compare_path, args.format, args.quality)
                        output_row["compare"] = str(compare_path.relative_to(output_dir))
                    output_row.update({"status": "done", "seconds": round(time.perf_counter() - started, 3)})
                except Exception as exc:  # noqa: BLE001 - retain the other recipes and images
                    output_row.update({"status": "failed", "error": str(exc)})
                row["outputs"].append(output_row)

            done = sum(output["status"] == "done" for output in row["outputs"])
            print(f"[{image_index:03d}/{len(images):03d}] {input_path.name}: {done}/{len(recipes)} recipe(s)")
            manifest["images"].append(row)
    finally:
        if engine is not None:
            engine.close()

    for recipe, paths in recipe_paths.items():
        if not paths:
            continue
        sheet_path = output_dir / _safe_name(recipe) / "contact_sheet.jpg"
        generate_contact_sheet(paths, sheet_path, cols=args.sheet_cols, cell_size=args.cell_size)
        manifest["contact_sheets"][recipe] = str(sheet_path.relative_to(output_dir))

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    failed = sum(
        output["status"] != "done"
        for image in manifest["images"]
        for output in image.get("outputs", [])
    )
    print(f"manifest: {manifest_path}")
    print(f"completed {len(images)} image(s) x {len(recipes)} recipe(s), failures: {failed}")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render selected retouch recipes across an image folder with contact sheets and a manifest."
    )
    parser.add_argument("input_dir", help="Folder containing input images")
    parser.add_argument("-o", "--output", required=True, help="Output folder")
    parser.add_argument(
        "--recipes",
        default="cosplay_character_showcase_v1",
        help="Comma-separated recipe names, or 'all' (default: cosplay_character_showcase_v1)",
    )
    parser.add_argument("--global-only", action="store_true", help="Skip face detection and run global finishing only.")
    parser.add_argument("--compare", action="store_true", help="Write original/result comparison images.")
    parser.add_argument("--max-dim", type=int, default=None, help="Resize the longest side before processing.")
    parser.add_argument("--format", choices=["jpg", "png", "webp"], default="jpg")
    parser.add_argument("--quality", type=int, default=94)
    parser.add_argument("--sheet-cols", type=int, default=5)
    parser.add_argument("--cell-size", type=int, default=420)
    parser.add_argument("--recursive", action="store_true", help="Include images in subdirectories.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most this many images.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run_batch(build_parser().parse_args(argv))
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
