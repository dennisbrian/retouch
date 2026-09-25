"""Export a recipe's colour look as an Adobe ``.cube`` 3D LUT.

A 3D LUT maps each input colour to one output colour, so it can only carry
the per-pixel part of a recipe: global tone (contrast, brightness,
highlights/shadows/whites/blacks, tonal curve, film density), white balance,
master and per-band HSL, the colour-grade preset and its curves, split
toning, fade, highlight drift, negative split tone and the B&W mixer.

Everything that looks at neighbouring pixels or at a mask is left out,
because a LUT cannot express it: face and skin retouching, reshaping,
background and subject work, clarity, glow/bloom/haze, vignette, grain,
halation, chromatic aberration, sharpening and the impact finish. Steps the
engine only applies inside the skin mask (recipe-level three-way split
toning, fade toe, skin glow, multi-illuminant skin adaptation) are left out
too, so the LUT reproduces exactly what the recipe does to a pixel outside
the face and skin. Reference-image colour transfer depends on the photo's
own statistics and is skipped as well.

The LUT is built by running the engine's own global and grade stages on an
identity lattice of ``size**3`` colours, so it stays in step with the engine
instead of re-implementing its maths.

Usage::

    python -m retouch.lut_export cosplay_clear_v1 -o cosplay_clear.cube
    python -m retouch.lut_export --all -o luts/
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .grading import ColorGrader
from .lut import CubeLUT, _build_identity_lut, load_cube

__all__ = [
    "DEFAULT_LUT_SIZE",
    "SKIPPED_CONTEXT_FIELDS",
    "recipe_lut",
    "write_cube",
    "export_recipe_lut",
    "export_all_recipe_luts",
    "skipped_steps",
    "changes_colour",
    "main",
]

#: 33 points per axis is the size Photoshop, Premiere and Resolve all
#: ship their own LUTs at, and the one they load fastest.
DEFAULT_LUT_SIZE = 33

#: ProcessingContext fields zeroed before rendering the lattice, with the
#: plain-language name reported back to the user when a recipe sets them.
SKIPPED_CONTEXT_FIELDS: Dict[str, str] = {
    "clarity": "clarity",
    "bloom": "bloom",
    "glow": "glow",
    "skin_glow": "skin glow",
    "airy_haze": "airy haze",
    "clarity_split_neg": "clarity split",
    "clarity_split_pos": "clarity split",
    "vignette": "vignette",
    "grain_strength": "film grain",
    "multi_illuminant_mix": "multi-illuminant skin adaptation",
}

# Context fields holding optional per-effect settings (None = off).
_SKIPPED_OPTIONAL_FIELDS: Dict[str, str] = {
    "halation": "halation",
    "grain": "film grain",
    "chromatic_aberration": "chromatic aberration",
    "color_ref": "reference colour transfer",
}


# Colour-grade preset keys that the engine's grade stage renders spatially.
# A preset's own "vignette" is not listed: _stage_grade grades with
# skip_post_effects=True and only forwards halation/grain/CA/lut, so it
# never renders. Glows are skipped by the engine whenever bloom is on.
_PRESET_SPATIAL_KEYS: Dict[str, str] = {
    "clarity": "clarity",
    "haze": "haze",
    "glow": "glow",
    "orton_glow": "glow",
    "sparkles": "sparkles",
    "halation": "halation",
    "chromatic_aberration": "chromatic aberration",
    "grain": "film grain",
}
_PRESET_GLOW_KEYS = ("glow", "orton_glow")


@staticmethod
def _identity(img, *_args, **_kwargs):
    return img


class _LatticeGrader(ColorGrader):
    """ColorGrader whose spatial effects are no-ops.

    Preset settings can carry clarity, haze, glow, vignette, halation,
    sparkles, chromatic aberration and grain; all of them read neighbouring
    pixels (or add noise), which would smear the lattice, so they pass the
    image through unchanged here.
    """

    _add_glow = _identity
    _add_clarity = _identity
    _F_add_clarity = _identity
    _add_clarity_noise_aware = _identity
    _F_add_clarity_noise_aware = _identity
    _add_vignette = _identity
    _add_chromatic_aberration = _identity
    _add_halation = _identity
    _add_grain = _identity
    _add_orton_glow = _identity
    _add_haze = _identity
    _F_add_haze = _identity
    _add_sparkles = _identity
    airy_haze = _identity
    clarity_split = _identity


def _lattice_engine():
    """A bare RetouchEngine carrying only what the colour stages touch.

    ``_stage_global`` / ``_stage_grade`` only use ``self._grader`` plus two
    static helpers, so skipping ``__init__`` avoids loading MediaPipe and
    starting the face-processor pool just to build a LUT.
    """
    from .engine import RetouchEngine

    engine = RetouchEngine.__new__(RetouchEngine)
    engine._grader = _LatticeGrader()
    return engine


def _recipe_context(recipe: str):
    from .engine import build_context
    from .params import resolve_recipe

    return build_context(recipe, resolve_recipe(recipe), {})


def skipped_steps(recipe: str) -> List[str]:
    """Plain-language names of the steps *recipe* uses that the LUT leaves out."""
    ctx = _recipe_context(recipe)
    found: List[str] = []
    for field, label in SKIPPED_CONTEXT_FIELDS.items():
        if getattr(ctx, field, 0) and label not in found:
            found.append(label)
    for field, label in _SKIPPED_OPTIONAL_FIELDS.items():
        if getattr(ctx, field, None) is not None and label not in found:
            found.append(label)
    # Spatial effects carried inside the colour-grade preset itself.
    from .grading import PRESETS

    preset = PRESETS.get(ctx.color_grade) if ctx.color_grade else None
    if preset and ctx.grade_intensity > 0:
        for key, label in _PRESET_SPATIAL_KEYS.items():
            if key in _PRESET_GLOW_KEYS and getattr(ctx, "bloom", 0) > 0:
                continue
            v = preset.get(key)
            active = bool(v) if not isinstance(v, (int, float)) else v > 0
            if key in preset and active and label not in found:
                found.append(label)
    if getattr(ctx, "sharpen", 0) > 0:
        found.append("sharpening")
    if getattr(ctx, "impact", 0) > 0:
        found.append("impact finish")
    return found


def changes_colour(lut: CubeLUT) -> bool:
    """False when *lut* is (within one 8-bit level) the identity.

    Retouch-only recipes such as ``natural`` have no colour look, so their
    LUT does nothing; callers use this to say so instead of silently
    handing over a no-op file.
    """
    identity = _build_identity_lut(lut.size)
    return float(np.abs(lut.array - identity).max()) > 1.0 / 255.0


def _render_colour_stages(img: np.ndarray, ctx) -> np.ndarray:
    """Run the per-pixel colour stages on a float32 [0,1] BGR image."""
    from .grading import apply_split_toning

    engine = _lattice_engine()
    zeros = np.zeros(img.shape[:2], dtype=np.float32)
    out = engine._stage_global(img, ctx)
    out = engine._stage_grade(out, ctx, zeros, zeros, None)
    # BB4 split toning lives in the finish stage but is per-pixel.
    if getattr(ctx, "split_toning", 0.0) > 0:
        u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        u8 = apply_split_toning(u8, strength=float(ctx.split_toning) / 100.0)
        out = u8.astype(np.float32) / 255.0
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _lut_context(recipe: str):
    ctx = _recipe_context(recipe)
    changes = {f: 0 for f in SKIPPED_CONTEXT_FIELDS if hasattr(ctx, f)}
    changes.update({f: None for f in _SKIPPED_OPTIONAL_FIELDS if hasattr(ctx, f)})
    if dataclasses.is_dataclass(ctx):
        return dataclasses.replace(ctx, **changes)
    for k, v in changes.items():
        setattr(ctx, k, v)
    return ctx


def recipe_lut(recipe: str, size: int = DEFAULT_LUT_SIZE) -> CubeLUT:
    """Bake *recipe*'s per-pixel colour steps into a ``size``-point CubeLUT."""
    if not 2 <= size <= 256:
        raise ValueError(f"LUT size must be between 2 and 256; got {size}")
    ctx = _lut_context(recipe)
    identity = _build_identity_lut(size)  # (b, g, r, 3) BGR
    lattice = identity.reshape(size * size, size, 3)
    graded = _render_colour_stages(lattice, ctx)
    return CubeLUT(graded.reshape(size, size, size, 3))


def write_cube(lut: CubeLUT, path: Union[str, Path], title: str = "") -> Path:
    """Write *lut* as an Adobe ``.cube`` file (red varies fastest)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    size = lut.size
    # CubeLUT is indexed [b, g, r] in BGR; .cube lists R G B with R fastest,
    # which is exactly C order over (b, g, r) with the channels reversed.
    rgb = lut.array[..., ::-1].reshape(-1, 3)
    lines = []
    if title:
        safe = title.replace('"', "'").replace("\n", " ")
        lines.append(f'TITLE "{safe}"')
    lines.append("# Made by retouch (per-pixel colour steps only)")
    lines.append(f"LUT_3D_SIZE {size}")
    lines.append("DOMAIN_MIN 0.0 0.0 0.0")
    lines.append("DOMAIN_MAX 1.0 1.0 1.0")
    body = "\n".join(f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in rgb.tolist())
    p.write_text("\n".join(lines) + "\n" + body + "\n", encoding="utf-8")
    return p


def export_recipe_lut(
    recipe: str,
    output: Union[str, Path],
    size: int = DEFAULT_LUT_SIZE,
) -> Tuple[Path, List[str]]:
    """Write *recipe*'s look to *output* and return (path, skipped steps).

    *output* may be a ``.cube`` file path or a directory, in which case the
    file is named ``<recipe>.cube`` inside it.
    """
    from .recipes import RECIPES

    if recipe not in RECIPES:
        # resolve_recipe() silently falls back to "natural"; refuse instead.
        raise KeyError(f"unknown recipe {recipe!r} (see ./run recipes)")
    out = Path(output)
    if out.suffix.lower() != ".cube":
        out = out / f"{recipe}.cube"
    path = write_cube(recipe_lut(recipe, size), out, title=recipe)
    return path, skipped_steps(recipe)


def export_all_recipe_luts(
    output_dir: Union[str, Path],
    size: int = DEFAULT_LUT_SIZE,
    recipes: Optional[List[str]] = None,
) -> List[Path]:
    """Write one ``<recipe>.cube`` per curated recipe into *output_dir*.

    Recipes with no colour look (retouch-only, see :func:`changes_colour`)
    are skipped, since their LUT would do nothing.
    """
    from .recipes import CURATED_RECIPE_NAMES

    names = list(recipes) if recipes is not None else list(CURATED_RECIPE_NAMES)
    written: List[Path] = []
    for name in names:
        lut = recipe_lut(name, size)
        if changes_colour(lut):
            written.append(write_cube(lut, Path(output_dir) / f"{name}.cube", title=name))
    return written


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m retouch.lut_export",
        description=(
            "Save a recipe's colour look as a .cube 3D LUT for Photoshop, "
            "Premiere, Resolve or Final Cut. Only per-pixel colour and tone "
            "steps are baked in; face retouching, masks, sharpening, "
            "vignette, glow and grain are not."
        ),
    )
    parser.add_argument("recipe", nargs="?", help="recipe name, e.g. cosplay_clear_v1")
    parser.add_argument("--all", action="store_true", help="export every curated recipe")
    parser.add_argument(
        "-o", "--output", default=".",
        help="output .cube file, or a folder (default: current folder)",
    )
    parser.add_argument(
        "--size", type=int, default=DEFAULT_LUT_SIZE,
        help=f"points per axis (default {DEFAULT_LUT_SIZE})",
    )
    args = parser.parse_args(argv)
    if bool(args.recipe) == bool(args.all):
        parser.error("give one recipe name or --all")
    try:
        if args.all:
            paths = export_all_recipe_luts(args.output, args.size)
            print(
                f"Wrote {len(paths)} LUTs to {Path(args.output).resolve()} "
                "(recipes with no colour look were skipped)"
            )
            return 0
        path, skipped = export_recipe_lut(args.recipe, args.output, args.size)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {path}")
    if not changes_colour(load_cube(path)):
        print(
            f"Note: {args.recipe} has no colour look (it only retouches), "
            "so this LUT leaves colours unchanged."
        )
    if skipped:
        print("Not in the LUT (needs the full app): " + ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
