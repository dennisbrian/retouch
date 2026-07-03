#!/usr/bin/env python3
"""Render one input image through every registered retouch recipe.

This is a visual-validation utility: it exports one output per recipe plus a
contact sheet and manifest so recipe drift can be reviewed quickly.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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


def _parse_recipe_list(raw: Optional[str]) -> List[str]:
    if raw is None or raw.strip().lower() == "all":
        return sorted(RECIPES.keys())
    names = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [name for name in names if name not in RECIPES]
    if unknown:
        raise ValueError(f"unknown recipe(s): {', '.join(unknown)}")
    return names


def _safe_stem(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _write_image(path: Path, img_bgr: np.ndarray, fmt: str, quality: int) -> None:
    if fmt == "png":
        write_image_with_icc(str(path), img_bgr, bit_depth=8, quality=quality)
    else:
        ok = cv2.imwrite(str(path), img_bgr, encode_write_params(fmt, quality))
        if not ok:
            raise IOError(f"failed to write output image: {path}")
    if not path.exists() or path.stat().st_size <= 0:
        raise IOError(f"output image was not written: {path}")


def _process_recipe(
    img_bgr: np.ndarray,
    recipe: str,
    engine: Optional[RetouchEngine],
    global_only: bool,
    fail_on_qa: bool,
) -> tuple[np.ndarray, List[Dict[str, Any]]]:
    if global_only:
        result = _apply_global_finish(img_bgr, {"recipe": recipe})
        return np.asarray(result), []

    if engine is None:
        raise RuntimeError("engine is required unless global_only=True")

    result = engine.process(img_bgr, recipe=recipe)
    qa_rows: List[Dict[str, Any]] = []
    for warning in getattr(result, "qa", []) or []:
        row = {
            "detector": warning.detector,
            "score": float(warning.score),
            "threshold": float(warning.threshold),
            "flagged": bool(warning.flagged),
            "message": warning.message,
        }
        qa_rows.append(row)

    if fail_on_qa and any(row["flagged"] for row in qa_rows):
        flagged = ", ".join(row["detector"] for row in qa_rows if row["flagged"])
        raise RuntimeError(f"QA_FAIL: {flagged}")

    return np.asarray(result), qa_rows


def run_sweep(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported input image extension: {input_path.suffix}")

    recipes = _parse_recipe_list(args.recipes)
    skip = set(_parse_recipe_list(args.skip)) if args.skip else set()
    recipes = [name for name in recipes if name not in skip]
    if not recipes:
        raise ValueError("no recipes selected")

    img_bgr = imread_exif(input_path)
    original_shape = img_bgr.shape[:2]
    img_bgr, scale = resize_for_processing(img_bgr, args.max_dim)

    source_path = output_dir / f"00_source.{args.format}"
    _write_image(source_path, img_bgr, args.format, args.quality)

    manifest: Dict[str, Any] = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "recipes_requested": recipes,
        "global_only": bool(args.global_only),
        "max_dim": args.max_dim,
        "original_shape": list(original_shape),
        "processed_shape": list(img_bgr.shape[:2]),
        "scale": scale,
        "outputs": [],
    }

    output_paths: List[Path] = []
    engine: Optional[RetouchEngine] = None
    if not args.global_only:
        engine = RetouchEngine()

    try:
        for idx, recipe in enumerate(recipes, start=1):
            started = time.perf_counter()
            out_name = f"{idx:02d}_{_safe_stem(recipe)}.{args.format}"
            out_path = output_dir / out_name
            compare_path = output_dir / f"{idx:02d}_{_safe_stem(recipe)}_compare.{args.format}"

            row: Dict[str, Any] = {
                "recipe": recipe,
                "output": out_name,
                "status": "pending",
            }
            try:
                result, qa_rows = _process_recipe(
                    img_bgr=img_bgr,
                    recipe=recipe,
                    engine=engine,
                    global_only=bool(args.global_only),
                    fail_on_qa=bool(args.fail_on_qa),
                )
                _write_image(out_path, result, args.format, args.quality)
                output_paths.append(out_path)

                if args.compare:
                    make_comparison(img_bgr, result, compare_path, args.format, args.quality)
                    row["compare"] = compare_path.name

                row.update(
                    {
                        "status": "done",
                        "seconds": round(time.perf_counter() - started, 3),
                        "qa": qa_rows,
                    }
                )
                print(f"[{idx:02d}/{len(recipes):02d}] {recipe}: wrote {out_name}")
            except cv2.error as exc:
                row.update({"status": "failed", "error": f"cv2.error: {exc}"})
                print(f"[{idx:02d}/{len(recipes):02d}] {recipe}: FAILED cv2.error: {exc}")
                if not args.keep_going:
                    raise
            except Exception as exc:
                row.update({"status": "failed", "error": str(exc)})
                print(f"[{idx:02d}/{len(recipes):02d}] {recipe}: FAILED {exc}")
                if not args.keep_going:
                    raise
            finally:
                manifest["outputs"].append(row)
    finally:
        if engine is not None:
            engine.close()

    if args.contact_sheet and output_paths:
        sheet_path = output_dir / "contact_sheet.jpg"
        generate_contact_sheet(output_paths, sheet_path, cols=args.sheet_cols, cell_size=args.cell_size)
        manifest["contact_sheet"] = sheet_path.name
        print(f"contact sheet: {sheet_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"manifest: {manifest_path}")

    failed = [row for row in manifest["outputs"] if row["status"] != "done"]
    if failed:
        print(f"completed with {len(failed)} failed recipe(s)")
        return 1
    print(f"completed {len(output_paths)} recipe render(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one image through all registered recipes and export validation outputs."
    )
    parser.add_argument("input", help="Input image path")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument(
        "--recipes",
        default="all",
        help="Comma-separated recipe names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Comma-separated recipe names to skip",
    )
    parser.add_argument("--format", choices=["jpg", "png", "webp"], default="jpg")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--max-dim", type=int, default=None)
    parser.add_argument(
        "--global-only",
        action="store_true",
        help="Skip face detection and render only global recipe finishing.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Also export side-by-side original/result comparison images.",
    )
    parser.add_argument(
        "--no-contact-sheet",
        action="store_false",
        dest="contact_sheet",
        help="Do not generate contact_sheet.jpg.",
    )
    parser.add_argument("--sheet-cols", type=int, default=4)
    parser.add_argument("--cell-size", type=int, default=320)
    parser.add_argument(
        "--fail-on-qa",
        action="store_true",
        help="Mark recipes failed if QA detectors flag artifacts.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        default=True,
        help="Continue after a recipe fails (default: enabled).",
    )
    parser.add_argument(
        "--stop-on-fail",
        action="store_false",
        dest="keep_going",
        help="Stop at the first recipe failure.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_sweep(args)
    except cv2.error as exc:
        print(f"cv2.error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
