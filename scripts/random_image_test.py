#!/usr/bin/env python3
"""Pick JPEGs from a folder and run retouch recipe sweeps on them."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.recipes.recipe_sweep import build_parser as build_sweep_parser  # noqa: E402
from scripts.recipes.recipe_sweep import run_sweep  # noqa: E402


def find_jpegs(folder: Path, recursive: bool) -> List[Path]:
    paths = folder.rglob("*") if recursive else folder.iterdir()
    return sorted(
        path for path in paths
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )


def safe_name(name: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Randomly select JPEGs and test them through the retouch recipe pipeline."
    )
    parser.add_argument("image_folder", help="Folder containing JPEG images")
    parser.add_argument(
        "-o", "--output",
        help="Output folder (default: test_output/random_image_tests/<image>_<timestamp>)",
    )
    parser.add_argument("--seed", type=int, help="Seed the random choice for a repeatable test")
    parser.add_argument(
        "--count", type=int, default=1,
        help="Number of different JPEGs to test (default: 1)",
    )
    parser.add_argument(
        "--no-recursive", action="store_true",
        help="Only inspect JPEGs directly inside IMAGE_FOLDER",
    )
    parser.add_argument(
        "--recipes", default="recommended",
        help="Recipe list passed to recipe_sweep.py (default: recommended; use curated or all)",
    )
    parser.add_argument("--global-only", action="store_true", help="Skip face detection and test global finishing only")
    parser.add_argument("--compare", action="store_true", help="Write original/result comparison images")
    parser.add_argument("--max-dim", type=int, default=None, help="Resize the longest side before processing")
    parser.add_argument("--format", choices=["jpg", "png", "webp"], default="jpg")
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--fail-on-qa", action="store_true", help="Fail if the full engine reports QA warnings")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    folder = Path(args.image_folder).expanduser().resolve()
    if not folder.is_dir():
        raise NotADirectoryError(f"image folder not found: {folder}")

    images = find_jpegs(folder, recursive=not args.no_recursive)
    if not images:
        raise ValueError(f"no JPEG images found in: {folder}")
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.count > len(images):
        raise ValueError(f"--count {args.count} exceeds available JPEGs: {len(images)}")

    rng = random.Random(args.seed)
    selected_images = rng.sample(images, args.count)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        output_root = Path(args.output).expanduser().resolve()
    elif args.count == 1:
        output_root = PROJECT_ROOT / "test_output" / "random_image_tests" / f"{selected_images[0].stem}_{stamp}"
    else:
        output_root = PROJECT_ROOT / "test_output" / "random_image_tests" / f"batch_{stamp}"

    print(f"JPEG candidates: {len(images)}")
    print(f"Selected images: {len(selected_images)}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print(f"Output folder: {output_root}")

    sweep_parser = build_sweep_parser()
    batch_rows = []
    exit_code = 0
    for index, selected in enumerate(selected_images, start=1):
        output = output_root if args.count == 1 else output_root / f"{index:02d}_{safe_name(selected.stem)}"
        sweep_args = sweep_parser.parse_args([
            str(selected), "--output", str(output),
            "--recipes", args.recipes,
            "--format", args.format,
            "--quality", str(args.quality),
        ] + (["--global-only"] if args.global_only else [])
          + (["--compare"] if args.compare else [])
          + (["--max-dim", str(args.max_dim)] if args.max_dim is not None else [])
          + (["--fail-on-qa"] if args.fail_on_qa else []))
        print(f"\n[{index:02d}/{len(selected_images):02d}] {selected}")
        result = run_sweep(sweep_args)
        exit_code = max(exit_code, result)
        batch_rows.append({"input": str(selected), "output": str(output), "status": "passed" if result == 0 else "failed"})

    if args.count > 1:
        output_root.mkdir(parents=True, exist_ok=True)
        batch_manifest = output_root / "batch_manifest.json"
        batch_manifest.write_text(json.dumps({
            "input_folder": str(folder),
            "seed": args.seed,
            "count": args.count,
            "images": batch_rows,
        }, indent=2), encoding="utf-8")
        print(f"batch manifest: {batch_manifest}")
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
