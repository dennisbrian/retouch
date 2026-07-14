#!/usr/bin/env python3

import argparse
import sys
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retouch import RetouchEngine
from retouch.engine import _adjust_contrast
from retouch.grading import PRESETS, ColorGrader
from retouch.io import (
    IMAGE_EXTENSIONS,
    RAW_EXTENSIONS,
    _resolve_safe_path,
    copy_exif,
    encode_write_params,
    imread_engine,
    imread_exif,
    make_comparison,
    output_format,
    read_icc_profile,
    resize_for_processing,
)
from retouch.recipes import RECIPES
from retouch.params import PROCESSING_PARAMS, recipe_to_params
from retouch.session import Session, create_session_from_params
from retouch.recipe_cookbook import list_recipes, search_recipes
from retouch.look_extractor import LookExtractor


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


def _smart_params_for_image(img_bgr, base_params):
    """F10 — run smart analysis on one image, return merged engine params.

    Uses :class:`retouch.smart_default.SmartProcessor` to pick a recipe +
    parameter overrides for ``img_bgr``, then layers any explicit CLI
    params from ``base_params`` on top (CLI flags win over smart
    suggestions). Returns a new params dict suitable for
    ``engine.process(**params)``.

    The smart suggestion emits GUI-scale values; we convert them to
    engine kwargs via ``gui_values_to_engine_kwargs``.
    """
    from retouch.smart_default import SmartProcessor
    from retouch.params import gui_values_to_engine_kwargs

    sp = SmartProcessor()
    suggestion = sp.analyze_and_suggest(img_bgr)
    smart_engine_kwargs = gui_values_to_engine_kwargs(suggestion.params)
    smart_engine_kwargs["recipe"] = suggestion.recipe

    # CLI explicit flags (base_params) override smart suggestions.
    merged = dict(smart_engine_kwargs)
    merged.update(base_params)
    # If the CLI explicitly set a recipe, it wins; otherwise keep smart's.
    if "recipe" in base_params:
        merged["recipe"] = base_params["recipe"]
    return merged, suggestion


def _resolve_session_path(path: str) -> Path:
    """Resolve a session file path with a path-traversal guard.

    Session files are user-supplied JSON; reject any path that resolves
    to a system directory (mirrors the guard used for image I/O).
    """
    return _resolve_safe_path(path, base_dir=None)


def _load_session_params(path: str) -> Dict[str, Any]:
    """Load a session JSON file and return its params dict + recipe.

    Returns a dict with two keys:
        ``recipe`` — the session's recipe name (or None)
        ``params`` — the session's params dict (engine-scale kwargs for
                     ``engine.process``), with ``color_ref_path`` stripped
                     (the file referenced at save time may not exist now)

    Raises ValueError on path-traversal rejection; raises FileNotFoundError
    if the session file does not exist; raises json.JSONDecodeError on
    malformed JSON (surfaced by Session.from_file).
    """
    resolved = _resolve_session_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Session file not found: {path}")
    session = Session.from_file(str(resolved))
    params = dict(session.params)
    # ``color_ref_path`` references an image at save time; drop it so
    # _finalize_params doesn't try to load a possibly-stale path. The
    # caller can re-supply --color-ref on the CLI.
    params.pop("color_ref_path", None)
    return {"recipe": session.recipe, "params": params}


def _save_session(
    params: Dict[str, Any],
    image_path: Path,
    out_path: Path,
    save_path: Optional[str],
) -> str:
    """Save the effective params as a session JSON next to the output image.

    If ``save_path`` is provided, write there (after path-traversal guard);
    otherwise auto-name it ``<output_stem>.session.json`` in the same
    directory as ``out_path``. Returns the absolute path written.
    """
    recipe = params.get("recipe")
    # Strip non-engine kwargs before saving so the session round-trips
    # cleanly through engine.process(**session.params).
    save_params = {
        k: v for k, v in params.items()
        if k not in ("color_ref", "color_ref_path")
    }
    if save_path:
        resolved = _resolve_safe_path(save_path, base_dir=None)
        target = str(resolved)
    else:
        target = str(out_path.with_suffix(".session.json"))
    session = create_session_from_params(
        save_params,
        recipe=recipe,
        image_path=str(image_path),
    )
    return session.to_file(target)


_worker_engine = None


def _init_worker():
    global _worker_engine
    _worker_engine = RetouchEngine()


def _linear_raw_to_engine_bgr(path, exposure: float = 0.0, contrast: float = 1.0):
    """T5 path: linear decode → develop → gamma-encode → float32 BGR [0,255]."""
    from retouch.raw_develop import RAWDeveloper

    dev = RAWDeveloper()
    linear_rgb, _meta = dev.load_raw(path)
    linear_rgb = dev.develop(
        linear_rgb, exposure=exposure, contrast=contrast,
    )
    # sRGB-ish encode for engine (display-referred); keep float headroom
    srgb = np.clip(linear_rgb, 0.0, 1.0) ** (1.0 / 2.2)
    bgr = (srgb[..., ::-1] * 255.0).astype(np.float32)
    return bgr


def _process_single(args):
    (img_path, output_dir, params, format_arg, quality, force, copy_exif_flag,
     max_dim, compare_flag, global_only, bit_depth, fail_on_qa, save_session,
     smart, linear_raw, raw_exposure, raw_contrast) = args
    try:
        fmt = output_format(img_path, format_arg)
        stem = img_path.stem
        
        # For 16-bit, force PNG or TIFF
        if bit_depth == 16 and fmt not in ("png", "tif", "tiff"):
            fmt = "png"
        
        out_path = (output_dir / f"{stem}.{fmt}") if output_dir else \
            img_path.with_suffix(f".{fmt}")

        if out_path.exists() and not force:
            return (img_path.name, "skipped")

        if linear_raw and Path(img_path).suffix.lower() in RAW_EXTENSIONS:
            img_bgr = _linear_raw_to_engine_bgr(
                img_path, exposure=raw_exposure, contrast=raw_contrast,
            )
        else:
            img_bgr = imread_engine(img_path)
        orig_shape = img_bgr.shape[:2]
        # 16-bit RAF ingest returns float32 [0,255]; comparison stitching is
        # uint8-only, so snapshot a uint8 original for the compare image.
        original_full = (
            (np.clip(img_bgr, 0, 255).astype(np.uint8) if img_bgr.dtype != np.uint8 else img_bgr.copy())
            if compare_flag else None
        )
        img_bgr, _scale = resize_for_processing(img_bgr, max_dim)

        # F10: --smart — per-image analysis overrides recipe/params.
        # CLI explicit flags (in `params`) win over the smart suggestion.
        effective_params = params
        if smart:
            try:
                effective_params, suggestion = _smart_params_for_image(
                    img_bgr, params
                )
            except (ValueError, RuntimeError) as e:
                # Analysis failed — fall back to the base params and log.
                tqdm.write(
                    f"  ⚠ {img_path.name}: smart analysis failed ({e}), "
                    f"using base params"
                )

        if global_only:
            result = _apply_global_finish(img_bgr, dict(effective_params))
        else:
            global _worker_engine
            if _worker_engine is not None:
                engine = _worker_engine
                should_close = False
            else:
                engine = RetouchEngine()
                should_close = True

            try:
                result = engine.process(img_bgr, **dict(effective_params))
                if fail_on_qa:
                    qa = getattr(result, 'qa', [])
                    if qa:
                        flagged = [w for w in qa if w.flagged]
                        if flagged:
                            reasons = "; ".join(f"{w.detector}={w.score:.2f}" for w in flagged)
                            return (img_path.name, f"QA_FAIL: {reasons}")
            finally:
                if should_close:
                    engine.close()

        # Upscale back to original dimensions
        if _scale < 1.0:
            result = cv2.resize(result, (orig_shape[1], orig_shape[0]),
                                interpolation=cv2.INTER_LINEAR)

        # Use write_image_with_icc for 16-bit support
        from retouch.io import write_image_with_icc
        icc_profile = read_icc_profile(img_path) if copy_exif_flag else None
        write_image_with_icc(str(out_path), result, icc_profile=icc_profile, bit_depth=bit_depth, quality=quality)

        if copy_exif_flag and bit_depth == 8:
            copy_exif(img_path, out_path)

        if compare_flag:
            compare_path = (output_dir / f"{stem}_compare.{fmt}") if output_dir else \
                img_path.parent / f"{stem}_compare.{fmt}"
            make_comparison(original_full, result, compare_path, fmt, quality)

        if save_session is not None:
            save_path = None if save_session is True else save_session
            try:
                _save_session(effective_params, img_path, out_path, save_path)
            except (OSError, ValueError) as e:
                return (img_path.name, f"saved_image_but_session_failed: {e}")

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
            "global_only": global_only,
            "bit_depth": bit_depth,
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
    if params.get("gamut_compress") is not None:
        settings["gamut_compress"] = params["gamut_compress"]
    if params.get("saturation_mode") is not None:
        settings["saturation_mode"] = params["saturation_mode"]

    if settings:
        result = grader.grade(result, settings, grade_intensity)

    impact = params.get("impact", defaults["impact"])
    if impact > 0:
        result = grader.add_impact_finish(result, impact)

    return result


def find_images(input_path: str, recursive: bool) -> list[Path]:
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


def build_params(args: argparse.Namespace) -> dict:
    params = {}
    if getattr(args, "recipe", None):
        params["recipe"] = args.recipe
    elif getattr(args, "preset", None):
        params["recipe"] = args.preset

    if getattr(args, "face_params", None):
        if args.face_params.lower() == "auto":
            params["face_params"] = "auto"
        else:
            from retouch.face_params import load_face_params_json
            params["face_params"] = load_face_params_json(args.face_params)

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


def main() -> None:
    warnings.filterwarnings(
        "always",
        category=DeprecationWarning,
        module=r"^(cli|__main__)(\.|$)",
    )

    parser = argparse.ArgumentParser(
        description="Professional batch face retouching tool"
    )
    parser.add_argument("input", nargs="?", help="Image file or directory")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-q", "--quality", type=int, default=95,
                        help="Output quality 1-100 (default: 95)")
    parser.add_argument("--format", choices=["jpg", "png", "webp", "same"],
                        default="same", help="Output format (default: same as input)")
    parser.add_argument("--bit-depth", type=int, choices=[8, 16], default=8,
                        help="Output bit depth (default: 8). 16-bit requires PNG or TIFF format.")
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
    parser.add_argument("--fail-on-qa", action="store_true",
                        help="Exit with code 1 if any QA detector flags an artifact")

    # Session save/load (F2)
    parser.add_argument("--session", type=str, default=None,
                        metavar="SESSION.json",
                        help="Load params from a session JSON file. Session "
                             "params override --recipe but are overridden by "
                             "explicit CLI flags.")
    parser.add_argument("--save-session", nargs="?", const=True, default=None,
                        metavar="PATH",
                        help="After processing, save the effective params to a "
                             "session JSON file. With no value, saves to "
                             "<output_stem>.session.json next to each output "
                             "image. Path traversal is guarded.")

    # F10: Smart Default — per-image auto-analysis + param suggestion.
    parser.add_argument("--smart", action="store_true",
                        help="Analyze each image with F9 (ImageAnalyzer) and "
                             "auto-select the recipe + params. Overrides "
                             "--recipe per-image; explicit CLI flags still win.")

    # T4: Recipe cookbook browsing (no processing).
    parser.add_argument("--list-recipes", nargs="?", const="", default=None,
                        metavar="CATEGORY",
                        help="List all recipes (optionally filtered by CATEGORY) and exit")
    parser.add_argument("--search-recipes", type=str, default=None,
                        metavar="QUERY",
                        help="Search recipes by name/description and exit")

    # F6: Look extraction from a reference image.
    parser.add_argument("--extract-look", type=str, default=None,
                        metavar="REF_IMG",
                        help="Extract a look from REF_IMG and apply it to the processed images")
    parser.add_argument("--look-base", type=str, default=None,
                        metavar="BASE_IMG",
                        help="Optional original image for paired look extraction (delta vs REF_IMG)")

    # Per-face recipe / param overrides (JSON keyed by detection-order index).
    parser.add_argument("--face-params", type=str, default=None,
                        metavar="PATH",
                        help="JSON file of per-face overrides, e.g. "
                             '{"0": {"recipe": "cosplay", "smooth": 70}, '
                             '"1": {"recipe": "natural"}}. Engine units (0-100).')

    # T5: optional true-linear RAW develop (gamma=1) before engine sRGB path.
    parser.add_argument("--linear-raw", action="store_true",
                        help="For RAW inputs: decode linear (gamma=1,1), optional "
                             "exposure/contrast develop, then gamma-encode into engine.")
    parser.add_argument("--raw-exposure", type=float, default=0.0,
                        help="With --linear-raw: exposure stops (default 0)")
    parser.add_argument("--raw-contrast", type=float, default=1.0,
                        help="With --linear-raw: linear contrast factor (default 1)")

    args = parser.parse_args()

    # T4: recipe cookbook browse mode — exit before requiring an input image.
    if args.list_recipes is not None:
        infos = list_recipes(args.list_recipes or None)
        if not infos:
            print("No recipes found"
                  + (f" in category '{args.list_recipes}'" if args.list_recipes else ""))
            return
        print(f"Recipes ({len(infos)}):")
        for info in infos:
            print(f"  [{info.category}] {info.name}: {info.description}")
        return
    if args.search_recipes:
        infos = search_recipes(args.search_recipes)
        print(f"Search '{args.search_recipes}' -> {len(infos)} match(es):")
        for info in infos:
            print(f"  [{info.category}] {info.name}: {info.description}")
        return

    input_path = Path(args.input)

    if not input_path.exists():
        print(f"✖ Input not found: {input_path}")
        sys.exit(1)

    files = find_images(args.input, args.recursive)
    if not files:
        print("✖ No image files found")
        sys.exit(1)

    params = build_params(args)

    # Session loading: session params override recipe defaults but are
    # overridden by explicit CLI flags. Precedence: CLI > session > recipe.
    if args.session:
        try:
            loaded = _load_session_params(args.session)
        except FileNotFoundError as e:
            print(f"✖ Could not load session: {e}")
            sys.exit(1)
        except ValueError as e:
            print(f"✖ Could not load session: {e}")
            sys.exit(1)
        except OSError as e:
            print(f"✖ Could not load session: {e}")
            sys.exit(1)
        # Start from session params, then layer CLI-provided keys on top.
        # CLI params keys take precedence (explicit flag wins).
        merged: Dict[str, Any] = dict(loaded["params"])
        for k, v in params.items():
            merged[k] = v
        # If the CLI didn't set a recipe but the session has one, use it.
        if "recipe" not in params and loaded["recipe"]:
            merged["recipe"] = loaded["recipe"]
        params = merged

    # F6: extract a look from a reference image and apply it over the current
    # params (look wins). Drops internal keys (e.g. _split_tone_three_way) that
    # are not valid engine kwargs. Failure exits loudly so the user knows.
    if getattr(args, "extract_look", None):
        try:
            ref = imread_exif(Path(args.extract_look))
            base = imread_exif(Path(args.look_base)) if args.look_base else None
            look = LookExtractor().extract(ref, base)
            look_params = {
                k: v for k, v in look["engine_params"].items()
                if not k.startswith("_")
            }
            params.update(look_params)
            print(
                f"Applied extracted look ({look['mode']}): "
                f"{ {k: (round(v, 3) if isinstance(v, float) else v) for k, v in look_params.items()} }"
            )
        except Exception as e:
            print(f"✖ Look extraction failed: {e}")
            sys.exit(1)

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
            (f, output_dir, params, args.format, args.quality, args.force,
             not args.no_exif, args.max_dim, args.compare, args.global_only,
             args.bit_depth, args.fail_on_qa, args.save_session, args.smart,
             args.linear_raw, args.raw_exposure, args.raw_contrast)
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
                if args.linear_raw and f.suffix.lower() in RAW_EXTENSIONS:
                    try:
                        img_bgr = _linear_raw_to_engine_bgr(
                            f, exposure=args.raw_exposure, contrast=args.raw_contrast,
                        )
                    except Exception as e:
                        failed += 1
                        tqdm.write(f"  ✖ {f.name}: linear-raw {e}")
                        continue
                else:
                    img_bgr = imread_engine(f)
                if img_bgr is None:
                    failed += 1
                    tqdm.write(f"  ✖ {f.name}: failed to read")
                    continue

                fmt = output_format(f, args.format)
                # For 16-bit, force PNG or TIFF
                if args.bit_depth == 16 and fmt not in ("png", "tif", "tiff"):
                    fmt = "png"
                out_path = output_dir / f"{f.stem}.{fmt}" if output_dir else \
                    f.with_suffix(f".{fmt}")

                if out_path.exists() and not args.force:
                    skipped += 1
                    continue

                orig_shape = img_bgr.shape[:2]
                original_full = img_bgr.copy() if args.compare else None
                img_bgr, _scale = resize_for_processing(img_bgr, args.max_dim)

                # F10: --smart — per-image analysis overrides recipe/params.
                effective_params = params
                if args.smart:
                    try:
                        effective_params, suggestion = _smart_params_for_image(
                            img_bgr, params
                        )
                        tqdm.write(
                            f"  🧠 {f.name}: {suggestion.recipe} "
                            f"({len(suggestion.params)} overrides)"
                        )
                    except (ValueError, RuntimeError) as e:
                        tqdm.write(
                            f"  ⚠ {f.name}: smart analysis failed ({e}), "
                            f"using base params"
                        )

                if args.global_only:
                    result = _apply_global_finish(img_bgr, dict(effective_params))
                else:
                    result = engine.process(img_bgr, **dict(effective_params))
                    if args.fail_on_qa:
                        qa = getattr(result, 'qa', [])
                        if qa:
                            flagged = [w for w in qa if w.flagged]
                            if flagged:
                                reasons = "; ".join(f"{w.detector}={w.score:.2f}" for w in flagged)
                                print(f"  ✖ {f.name}: QA_FAIL: {reasons}")
                                failed += 1
                                continue

                if _scale < 1.0:
                    result = cv2.resize(result, (orig_shape[1], orig_shape[0]),
                                        interpolation=cv2.INTER_LINEAR)
                
                # Use write_image_with_icc for 16-bit support
                from retouch.io import write_image_with_icc, read_icc_profile
                icc_profile = read_icc_profile(f) if not args.no_exif else None
                write_image_with_icc(str(out_path), result, icc_profile=icc_profile, bit_depth=args.bit_depth, quality=args.quality)
                
                if not args.no_exif and args.bit_depth == 8:
                    copy_exif(f, out_path)

                if args.compare:
                    compare_path = (output_dir / f"{f.stem}_compare.{fmt}") if output_dir else \
                        f.parent / f"{f.stem}_compare.{fmt}"
                    make_comparison(original_full, result, compare_path, fmt, args.quality)

                if args.save_session is not None:
                    save_path = None if args.save_session is True else args.save_session
                    try:
                        written = _save_session(effective_params, f, out_path, save_path)
                    except (OSError, ValueError) as e:
                        tqdm.write(f"  ⚠ {f.name}: could not save session: {e}")
                    else:
                        tqdm.write(f"  💾 session → {written}")
                done += 1
        finally:
            if engine is not None:
                engine.close()

    elapsed = time.time() - t0
    print(f"\nDone — {done} processed, {skipped} skipped, {failed} failed"
          f"  ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
