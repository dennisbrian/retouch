#!/usr/bin/env python3

import argparse
import sys
import os
import time
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retouch import RetouchEngine
from retouch.engine import _adjust_contrast
from retouch.grading import PRESETS, ColorGrader
from retouch.io import (
    IMAGE_EXTENSIONS,
    copy_exif,
    encode_write_params,
    imread_exif,
    make_comparison,
    output_format,
    resize_for_processing,
)
from retouch.recipes import RECIPES
from retouch.params import PROCESSING_PARAMS, recipe_to_params


class _DeprecatedAliasAction(argparse.Action):
    """Argparse action for a deprecated flag aliased to a newer flag.

    Stores the value into ``dest`` (a separate "legacy" attribute) and emits a
    ``DeprecationWarning`` pointing the user at the replacement option. The
    build_params() step is responsible for picking the new flag's value over
    the legacy one when both are set.
    """

    def __init__(self, option_strings, dest, deprecated_to, **kwargs):
        kwargs.setdefault("default", None)
        self.deprecated_to = deprecated_to
        super().__init__(option_strings=option_strings, dest=dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        warnings.warn(
            f"'{option_string}' is deprecated and will be removed in a future"
            f" release; please use '{self.deprecated_to}' instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        setattr(namespace, self.dest, values)

RECIPE_CHOICES = sorted(RECIPES.keys())
PRESET_CHOICES = RECIPE_CHOICES


def _finalize_params(params):
    """Load color reference once; safe to share read-only across batch jobs."""
    finalized = dict(params)
    ref_path = finalized.pop("color_ref_path", None)
    if ref_path is not None:
        finalized["color_ref"] = imread_exif(Path(ref_path))
    return finalized


_worker_engine = None


def _init_worker():
    global _worker_engine
    _worker_engine = RetouchEngine()


def _process_single(args):
    img_path, output_dir, params, format_arg, quality, force, copy_exif_flag, max_dim, compare_flag, global_only = args
    try:
        fmt = output_format(img_path, format_arg)
        stem = img_path.stem
        out_path = (output_dir / f"{stem}.{fmt}") if output_dir else \
            img_path.with_suffix(f".{fmt}")

        if out_path.exists() and not force:
            return (img_path.name, "skipped")

        img_bgr = imread_exif(img_path)
        orig_shape = img_bgr.shape[:2]
        original_full = img_bgr.copy() if compare_flag else None
        img_bgr, _scale = resize_for_processing(img_bgr, max_dim)

        if global_only:
            result = _apply_global_finish(img_bgr, dict(params))
        else:
            global _worker_engine
            if _worker_engine is not None:
                engine = _worker_engine
                should_close = False
            else:
                engine = RetouchEngine()
                should_close = True

            try:
                result = engine.process(img_bgr, **dict(params))
            finally:
                if should_close:
                    engine.close()

        # Upscale back to original dimensions
        if _scale < 1.0:
            result = cv2.resize(result, (orig_shape[1], orig_shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        cv2.imwrite(str(out_path), result, encode_write_params(fmt, quality))

        if copy_exif_flag:
            copy_exif(img_path, out_path)

        if compare_flag:
            compare_path = (output_dir / f"{stem}_compare.{fmt}") if output_dir else \
                img_path.parent / f"{stem}_compare.{fmt}"
            make_comparison(original_full, result, compare_path, fmt, quality)

        return (img_path.name, "done")
    except Exception as e:
        from retouch.utils import log_crash
        log_crash(e, {
            "image_path": str(img_path),
            "output_dir": str(output_dir) if output_dir else None,
            "params": str(params),
            "format_arg": format_arg,
            "quality": quality,
            "max_dim": max_dim,
            "compare_flag": compare_flag,
            "global_only": global_only
        })
        return (img_path.name, f"failed: {e}")


def _recipe_defaults(recipe_name):
    """Return the recipe-derived defaults for the global-finish path.

    Delegates to ``retouch.params.recipe_to_params`` (the canonical spec
    shared with the engine and the GUI) and converts the GUI-scale
    values it returns into the engine-scale values that
    ``_apply_global_finish`` consumes.  ``grade_intensity`` is stored
    0-100 in the GUI scale but ``grader.grade`` expects 0.0-1.0, and
    ``color_grade`` uses ``None`` (not ``"none"``) to mean "no grade".
    """
    d = recipe_to_params(recipe_name or "natural")
    if d.get("grade_intensity") is not None:
        d["grade_intensity"] = float(d["grade_intensity"]) / 100.0
    if d.get("color_grade") == "none":
        d["color_grade"] = None
    return d


def _apply_global_finish(img_bgr, params):
    """Fast retouch path that avoids face detection and local facial edits."""
    result = img_bgr.copy()
    defaults = _recipe_defaults(params.get("recipe"))
    grader = ColorGrader()

    if params.get("contrast"):
        result = _adjust_contrast(result, params["contrast"])

    color_grade = params.get("color_grade", defaults["color_grade"])
    grade_intensity = params.get(
        "grade_intensity",
        1.0 if params.get("color_grade") else defaults["grade_intensity"],
    )

    settings = PRESETS.get(color_grade, PRESETS["natural"]).copy() if color_grade else {}
    if params.get("chromatic_aberration") is not None:
        settings["chromatic_aberration"] = params["chromatic_aberration"]
    if params.get("halation") is not None:
        settings["halation"] = params["halation"]
    if params.get("grain") is not None:
        settings["grain"] = params["grain"]
    if params.get("lut") is not None:
        settings["lut"] = params["lut"]

    if settings:
        result = grader.grade(result, settings, grade_intensity)

    impact = params.get("impact", defaults["impact"])
    if impact > 0:
        result = grader.add_impact_finish(result, impact)

    return result


def find_images(input_path, recursive):
    path = Path(input_path)
    if path.is_file():
        return [path]
    pattern = "**/*" if recursive else "*"
    files = []
    for f in path.glob(pattern):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            files.append(f)
    return sorted(files)


def _add_processing_arg(parser, spec):
    """Add an argparse argument for a single ``PROCESSING_PARAMS`` entry.

    Booleans get both ``--{flag}`` and ``--no-{flag}`` variants so the
    user can explicitly enable or disable a toggle.  Non-boolean args
    use ``type=cli_type, default=None``; argparse converts hyphens in
    the flag to underscores in the dest, which matches ``spec.name``.
    """
    if spec.cli_flag is None:
        return
    if spec.cli_type is bool:
        parser.add_argument(
            f"--{spec.cli_flag}",
            action="store_true",
            default=None,
            help=f"Enable {spec.name}",
        )
        parser.add_argument(
            f"--no-{spec.cli_flag}",
            action="store_false",
            dest=spec.name,
            default=None,
            help=f"Disable {spec.name}",
        )
        return
    parser.add_argument(
        f"--{spec.cli_flag}",
        type=spec.cli_type,
        default=None,
        help=f"{spec.name} parameter",
    )


def build_params(args):
    params = {}
    if getattr(args, "recipe", None):
        params["recipe"] = args.recipe
    elif getattr(args, "preset", None):
        params["recipe"] = args.preset

    # Process the simple scalar/int/float parameters from the spec list.
    # Each spec maps a CLI flag (e.g. "--smooth") to the engine kwarg name
    # ("smooth").  When a flag is not provided on the command line we leave
    # the engine kwarg unset (None) so the recipe default takes over.
    for spec in PROCESSING_PARAMS:
        if spec.cli_flag is None:
            continue
        # Translate the CLI flag to the argparse attribute name.
        attr = spec.cli_flag.replace("-", "_")
        if not hasattr(args, attr):
            continue
        val = getattr(args, attr)
        if val is None:
            continue
        # auto_exposure uses ``store_true`` semantics: only ``True`` is
        # forwarded (the explicit ``--no-auto-exposure`` sets ``False``,
        # which is the default and should be dropped).
        if spec.name == "auto_exposure":
            if val:
                params[spec.name] = True
            continue
        # For other boolean flags (nose_blush, under_eye_blush,
        # white_costume_lift) the caller can explicitly set ``False`` to
        # override the recipe.  Forward both ``True`` and ``False``.
        if spec.cli_type is bool:
            params[spec.name] = bool(val)
            continue
        params[spec.name] = val

    if args.color_grade:
        params["color_grade"] = args.color_grade
    if args.lip_tint:
        params["lip_tint"] = args.lip_tint
    if getattr(args, "whiten_tone", None):
        params["whiten_tone"] = args.whiten_tone
    if getattr(args, "specular_bloom_tone", None):
        params["specular_bloom_tone"] = args.specular_bloom_tone
    if getattr(args, "lip_finish", None):
        params["lip_finish"] = args.lip_finish
    if getattr(args, "auto_exposure", False):
        params["auto_exposure"] = True

    if getattr(args, "chromatic_aberration", None) is not None:
        params["chromatic_aberration"] = args.chromatic_aberration
    if getattr(args, "grain", None) is not None:
        params["grain"] = args.grain
    if getattr(args, "lut", None) is not None:
        params["lut"] = args.lut

    halation_intensity = getattr(args, "halation_intensity", None)
    if halation_intensity is None:
        halation_intensity = getattr(args, "halation", None)
    if halation_intensity is not None:
        threshold = getattr(args, "halation_threshold", None)
        radius = getattr(args, "halation_radius", None)
        params["halation"] = {
            "intensity": float(halation_intensity),
            "threshold": int(threshold) if threshold is not None else 210,
            "radius": int(radius) if radius is not None else 21,
        }

    if args.color_ref:
        params["color_ref_path"] = args.color_ref

    color_transfer_intensity = getattr(args, "color_transfer_intensity", None)
    if color_transfer_intensity is None:
        color_transfer_intensity = getattr(args, "color_ref_strength", None)
    if color_transfer_intensity is not None:
        params["color_transfer_intensity"] = float(color_transfer_intensity)

    return params


def main():
    warnings.filterwarnings(
        "always",
        category=DeprecationWarning,
        module=r"^(cli|__main__)(\.|$)",
    )

    parser = argparse.ArgumentParser(
        description="Professional batch face retouching tool"
    )
    parser.add_argument("input", help="Image file or directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-q", "--quality", type=int, default=95,
                        help="Output quality 1-100 (default: 95)")
    parser.add_argument("--format", choices=["jpg", "png", "webp", "same"],
                        default="same", help="Output format (default: same as input)")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="Downscale so longest side ≤ N px before processing (faster)")
    parser.add_argument("-r", "--recursive", action="store_true",
                        help="Search subdirectories")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Overwrite existing files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without processing")
    parser.add_argument("--global-only", action="store_true",
                        help="Skip face detection and apply only global color/impact retouch")

    # Processing controls
    parser.add_argument("--recipe", choices=RECIPE_CHOICES,
                        help="Retouch recipe name")
    parser.add_argument("--preset", choices=PRESET_CHOICES,
                        help="Quick parameter preset (legacy alias)")
    parser.add_argument("--color-ref", type=str, default=None,
                        help="Reference image path for colour transfer")
    parser.add_argument("--color-transfer-intensity", type=float, default=None,
                        help="Colour transfer blend intensity 0-1 (default: 1.0)")
    parser.add_argument("--color-ref-strength", type=float, default=None,
                        action=_DeprecatedAliasAction,
                        deprecated_to="--color-transfer-intensity",
                        dest="color_ref_strength",
                        help="[DEPRECATED, use --color-transfer-intensity] "
                             "Colour transfer blend intensity 0-1")
    parser.add_argument("--halation-intensity", type=float, default=None,
                        help="Film halation bleed intensity (0.0 - 1.0)")
    parser.add_argument("--halation-threshold", type=int, default=None,
                        help="Halation highlight threshold 0-255 (default: 210)")
    parser.add_argument("--halation-radius", type=int, default=None,
                        help="Halation blur radius in pixels (default: 21)")
    parser.add_argument("--halation", type=float, default=None,
                        action=_DeprecatedAliasAction,
                        deprecated_to="--halation-intensity",
                        dest="halation",
                        help="[DEPRECATED, use --halation-intensity] "
                             "Film halation bleed intensity (0.0 - 1.0)")

    # Auto-generate processing parameter arguments from PROCESSING_PARAMS
    for spec in PROCESSING_PARAMS:
        _add_processing_arg(parser, spec)

    # Batch
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() // 2),
                        help="Parallel workers (default: CPU count / 2)")
    parser.add_argument("--no-compare", action="store_false", dest="compare",
                        help="Skip side-by-side comparison output")
    parser.add_argument("--no-exif", action="store_true",
                        help="Skip EXIF metadata copying")

    args = parser.parse_args()
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"✖ Input not found: {input_path}")
        sys.exit(1)

    files = find_images(args.input, args.recursive)
    if not files:
        print("✖ No image files found")
        sys.exit(1)

    params = build_params(args)

    if args.dry_run:
        print(f"Dry run — {len(files)} image(s) found:\n")
        for f in files:
            print(f"  {f}")
        print(f"\nSettings: {params}")
        print(f"Workers: {args.workers}")
        return

    params = _finalize_params(params)

    output_dir = Path(args.output) if args.output else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    done = skipped = failed = 0

    if args.workers > 1 and len(files) > 1:
        pool_args = [
            (f, output_dir, params, args.format, args.quality, args.force, not args.no_exif, args.max_dim, args.compare, args.global_only)
            for f in files
        ]
        with ProcessPoolExecutor(
            max_workers=args.workers,
            initializer=_init_worker,
        ) as pool:
            futures = {pool.submit(_process_single, a): a[0] for a in pool_args}
            for future in tqdm(as_completed(futures), total=len(files),
                               desc="Retouching", unit="img"):
                name, status = future.result()
                if status == "done":
                    done += 1
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    tqdm.write(f"  ✖ {name}: {status}")
    else:
        engine = None if args.global_only else RetouchEngine()
        try:
            for f in tqdm(files, desc="Retouching", unit="img"):
                img_bgr = imread_exif(f)
                if img_bgr is None:
                    failed += 1
                    tqdm.write(f"  ✖ {f.name}: failed to read")
                    continue

                fmt = output_format(f, args.format)
                out_path = output_dir / f"{f.stem}.{fmt}" if output_dir else \
                    f.with_suffix(f".{fmt}")

                if out_path.exists() and not args.force:
                    skipped += 1
                    continue

                orig_shape = img_bgr.shape[:2]
                original_full = img_bgr.copy() if args.compare else None
                img_bgr, _scale = resize_for_processing(img_bgr, args.max_dim)

                if args.global_only:
                    result = _apply_global_finish(img_bgr, dict(params))
                else:
                    result = engine.process(img_bgr, **dict(params))

                if _scale < 1.0:
                    result = cv2.resize(result, (orig_shape[1], orig_shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(out_path), result, encode_write_params(fmt, args.quality))
                if not args.no_exif:
                    copy_exif(f, out_path)

                if args.compare:
                    compare_path = (output_dir / f"{f.stem}_compare.{fmt}") if output_dir else \
                        f.parent / f"{f.stem}_compare.{fmt}"
                    make_comparison(original_full, result, compare_path, fmt, args.quality)
                done += 1
        finally:
            if engine is not None:
                engine.close()

    elapsed = time.time() - t0
    print(f"\nDone — {done} processed, {skipped} skipped, {failed} failed"
          f"  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
