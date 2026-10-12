#!/usr/bin/env python3
"""Run the metamorphic robustness lab for one recipe processor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retouch.io import imread_exif, resize_for_processing
from retouch.metamorphic import observations_to_dict, run_lab
from retouch.certification import certification_fields


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Input image")
    parser.add_argument("-o", "--output", required=True, help="JSON report path")
    parser.add_argument("--recipe", default="natural")
    parser.add_argument("--max-dim", type=int, default=None)
    parser.add_argument("--global-only", action="store_true", help="Diagnostic mode; report is not face-aware certification")
    parser.add_argument("--mean-threshold", type=float, default=8.0)
    parser.add_argument("--p95-threshold", type=float, default=24.0)
    return parser


def main(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    image = imread_exif(input_path)
    image, scale = resize_for_processing(image, args.max_dim)
    if args.global_only:
        from cli import _apply_global_finish
        processor = lambda source: _apply_global_finish(source, {"recipe": args.recipe})
    else:
        from retouch import RetouchEngine
        engine = RetouchEngine()
        processor = lambda source: engine.process(source, recipe=args.recipe)
    try:
        observations = run_lab(
            image, processor,
            mean_threshold=args.mean_threshold,
            p95_threshold=args.p95_threshold,
        )
    finally:
        if not args.global_only:
            engine.close()
    automatic_pass = bool(observations) and all(item.passed for item in observations)
    report = {
        "version": 1,
        "input": str(input_path),
        "recipe": args.recipe,
        "global_only": bool(args.global_only),
        "scale": scale,
        "observations": observations_to_dict(observations),
        "passed": automatic_pass,
    }
    report.update(certification_fields(
        face_aware_run=not bool(args.global_only),
        automatic_pass=automatic_pass,
        human_review="not_applicable",
    ))
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"], "global_only": report["global_only"]}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(build_parser().parse_args()))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
