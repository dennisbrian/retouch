"""Central parameter specifications for the retouch pipeline.

This module is the single source of truth for every processing parameter that
flows through the engine, the GUI, and the CLI. Adding a new tunable means
adding one ``ParamSpec`` entry here — no more 7-way edits across the codebase.

Conversion semantics
--------------------
Each ``ParamSpec`` carries a ``conversion`` code that controls how values are
translated between three representation spaces:

* **Recipe dict** (0.0–1.0 ratios, the human-editable values in ``recipes.py``)
* **GUI / sliders** (0–100, the integer scale used by the Gradio controls and
  the ``--flag`` CLI arguments)
* **Engine** (the actual numeric values consumed by ``ProcessingContext`` and
  the per-stage retouch code — sometimes 0–100, sometimes 0.0–1.0, sometimes
  raw recipe ratios, depending on what the stage was historically written to
  expect)

The supported ``conversion`` codes are:

* ``"recipe_pct"`` — recipe ratio × 100 in BOTH the GUI and the engine
  (e.g. ``smooth``, ``whiten``, ``equalize``).  Recipe 0.30 → GUI 30 → engine 30.
* ``"engine_pct"`` — recipe ratio × 100 for the GUI; the engine expects the
  percentage form (e.g. ``bloom``, ``impact``, ``pore_synthesis``).  Recipe
  0.05 → GUI 5 → engine 5.
* ``"recipe_direct"`` — recipe ratio passes through unchanged; GUI also
  receives the recipe value (e.g. ``mid_reduction``, ``texture_opacity``).
* ``"recipe_int_pct"`` — same as ``recipe_pct`` but truncated to ``int`` for
  the GUI sliders that only accept integers (e.g. ``catchlight``).
* ``"gui_mul500"`` — GUI stores the value × 500; the engine divides by 500
  (e.g. ``grain``).  Recipe 0.05 → GUI 25 → engine 0.05.
* ``"gui_div100"`` — GUI stores the value × 100; the engine divides by 100
  (e.g. ``halation``).  Recipe 0.10 → GUI 10 → engine 0.10.
* ``"gui_direct"`` — no recipe path; the value is GUI-only (engine reads it
  raw, e.g. ``chromatic_aberration``, ``lut``).
* ``"static"`` — no recipe path; the value is hard-coded in the engine
  default.  The GUI just initialises to the engine default (e.g.
  ``relight_azimuth``, ``bloom_threshold``).
* ``"alias"`` — derives from another parameter (used when two GUI sliders
  share the same recipe source, like ``blemish`` mirroring ``smooth``).
* ``"dropdown"`` — string-typed parameter (e.g. ``lip_tint``, ``whiten_tone``,
  ``color_grade``).  Uses the recipe value verbatim and falls back to the
  engine default when the recipe has no entry.
* ``"bool_flag"`` — boolean toggle (e.g. ``nose_blush``).  Uses the recipe
  value verbatim.

The conversion is applied automatically by ``recipe_to_params()`` when
populating the GUI-side dictionary, and by ``build_context()`` when populating
the engine ``ProcessingContext``.  CLI arguments are mapped by
``build_params()`` based on the same spec.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import copy

from .recipes import RECIPES


# ---------------------------------------------------------------------------
# ParamSpec — the canonical parameter declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """One retouch processing parameter.

    Attributes
    ----------
    name:
        Python attribute name on ``ProcessingContext`` AND the dictionary key
        in the GUI defaults dict.  Must be a valid Python identifier.
    cli_flag:
        The CLI argument for this parameter, *without* the leading dashes
        (e.g. ``"smooth"`` → ``--smooth``).  ``None`` for parameters that
        have no direct CLI flag (alias/static/internal-only parameters).
    cli_type:
        The ``argparse`` ``type=`` to use, or ``None`` for boolean flag
        arguments.  ``bool`` here means "store_true / store_false".
    default:
        The engine-side default value, used both as the ``ProcessingContext``
        field default and as the GUI initialisation value when the recipe
        supplies no override.  ``None`` is allowed for "leave at engine
        default" parameters (e.g. ``nose_smooth``, ``chromatic_aberration``).
    gui_default:
        Optional override for the value the GUI uses when the recipe has no
        entry.  Falls back to ``default`` when unset.  Some parameters
        (e.g. ``brightness``, ``highlights``) want ``None`` on the engine
        side but ``0`` on the GUI side; the two fields let us express that.
    recipe_key:
        Dot-separated path inside the resolved recipe dict (e.g.
        ``"frequency.smooth"``, ``"eyes.catchlight"``, ``"color_harmony.preset"``).
        ``None`` for parameters that are not derived from the recipe.
    gui_recipe_key:
        Optional override — when the GUI and the engine read the value from
        *different* recipe locations, the GUI uses this key and the engine
        uses ``recipe_key``.  When unset, both lookups use ``recipe_key``.
    engine_recipe_key:
        Optional override for the engine-side lookup path.  When set, the
        engine uses this key while the GUI uses ``recipe_key`` (or
        ``gui_recipe_key``).  Used for parameters where the historical
        engine and GUI recipes lived at different paths.
    conversion:
        One of the conversion codes documented in the module docstring.
    min_val / max_val:
        Optional inclusive bounds for slider rendering and CLI validation.
    alias_of:
        For ``"alias"`` conversion — the name of the parameter to mirror
        (e.g. ``blemish`` mirrors ``smooth``).
    fallback_default:
        For ``"alias"`` conversion — what to fall back to when the alias
        source recipe value is missing.
    """

    name: str
    cli_flag: Optional[str] = None
    cli_type: Optional[type] = None
    default: Any = None
    gui_default: Any = None
    recipe_key: Optional[str] = None
    gui_recipe_key: Optional[str] = None
    engine_recipe_key: Optional[str] = None
    conversion: str = "static"
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    alias_of: Optional[str] = None
    fallback_default: Any = None
    choices: Optional[Sequence[str]] = None


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _lookup_recipe(recipe: Dict[str, Any], path: str) -> Any:
    """Walk a dot-separated path inside a recipe dict.

    Returns ``None`` if any segment is missing.
    """
    node: Any = recipe
    for segment in path.split("."):
        if not isinstance(node, dict) or segment not in node:
            return None
        node = node[segment]
    return node


def _gui_for_recipe_value(conversion: str, recipe_value: Any) -> Any:
    """Translate a recipe value into the value the GUI slider expects.

    ``recipe_value`` may be ``None``; callers that want a guaranteed non-``None``
    GUI value should pass ``default`` separately.
    """
    if recipe_value is None:
        return None
    if conversion == "recipe_pct":
        return int(round(float(recipe_value) * 100.0))
    if conversion == "recipe_int_pct":
        return int(round(float(recipe_value) * 100.0))
    if conversion == "engine_pct":
        return int(round(float(recipe_value) * 100.0))
    if conversion == "recipe_direct":
        return float(recipe_value)
    if conversion == "dropdown":
        return str(recipe_value) if recipe_value else "none"
    if conversion == "bool_flag":
        return bool(recipe_value)
    if conversion in ("gui_direct", "static", "alias", "dodge_burn_pct",
                      "lip_tint_direct"):
        return recipe_value
    if conversion == "relight_pct":
        return int(round(float(recipe_value) * 100.0))
    if conversion == "gui_mul500":
        return int(round(float(recipe_value) * 500.0))
    if conversion == "gui_div100":
        return int(round(float(recipe_value) * 100.0))
    raise ValueError(f"Unknown conversion code: {conversion!r}")


def _engine_for_recipe_value(conversion: str, recipe_value: Any) -> Any:
    """Translate a recipe value into the value the engine expects.

    This is the same as the GUI translation for the percentage conversions
    (the engine reads percentages too) and identical for direct/dropdown/bool
    conversions.  ``gui_mul500`` and ``gui_div100`` need their own translation
    because the recipe stores the engine-side value, not the GUI-side.
    """
    if recipe_value is None:
        return None
    if conversion in (
        "recipe_pct",
        "recipe_int_pct",
        "engine_pct",
    ):
        return float(recipe_value) * 100.0
    if conversion == "recipe_direct":
        return float(recipe_value)
    if conversion == "dropdown":
        return recipe_value
    if conversion == "bool_flag":
        return bool(recipe_value)
    if conversion == "gui_mul500":
        return float(recipe_value)
    if conversion == "gui_div100":
        return float(recipe_value)
    if conversion in ("gui_direct", "static", "alias"):
        return recipe_value
    raise ValueError(f"Unknown conversion code: {conversion!r}")


# ---------------------------------------------------------------------------
# PROCESSING_PARAMS — the canonical list
# ---------------------------------------------------------------------------


# Skin / smoothing
_SKIN_PARAMS = [
    ParamSpec(
        name="smooth",
        cli_flag="smooth",
        cli_type=int,
        default=30,
        recipe_key="frequency.smooth",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="mid_reduction",
        cli_flag="mid-reduction",
        cli_type=float,
        default=0.35,
        recipe_key="frequency.mid_reduction",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="blotch_reduction",
        cli_flag="blotch-reduction",
        cli_type=float,
        default=0.0,
        recipe_key="frequency.blotch_reduction",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="texture_opacity",
        cli_flag="texture-opacity",
        cli_type=float,
        default=1.0,
        recipe_key="texture.opacity",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="pore_synthesis",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="texture.pore_synthesis",
        conversion="engine_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="nose_smooth",
        cli_flag="nose-smooth",
        cli_type=int,
        default=0,
        recipe_key="frequency.nose_smooth",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="regional_modulation",
        cli_flag="regional-modulation",
        cli_type=float,
        default=0.0,
        recipe_key="frequency.regional_modulation",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="smooth_engine",
        cli_flag="smooth-engine",
        cli_type=str,
        default="guided",
        recipe_key="frequency.smooth_engine",
        conversion="dropdown",
        min_val=None,
        max_val=None,
    ),
    ParamSpec(
        name="undereye_shadow_strength",
        cli_flag="undereye-shadow-strength",
        cli_type=float,
        default=0.0,
        recipe_key="eyes.undereye_shadow_strength",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="freckle_removal",
        cli_flag="freckle-removal",
        cli_type=float,
        default=0.0,
        recipe_key="frequency.freckle_removal",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=100.0,
    ),
    ParamSpec(
        name="heal_engine",
        cli_flag="heal-engine",
        cli_type=str,
        default="telea",
        recipe_key="frequency.heal_engine",
        conversion="dropdown",
        choices=("telea", "patchmatch"),
    ),
    ParamSpec(
        name="freckle_preserve_mask",
        cli_flag=None,
        cli_type=None,
        default=None,
        recipe_key=None,
        conversion="gui_direct",
        min_val=None,
        max_val=None,
    ),
    ParamSpec(
        name="mark_policy",
        cli_flag="mark-policy",
        cli_type=str,
        default="legacy",
        recipe_key="mark_policy",
        conversion="dropdown",
        choices=("legacy", "protect_identity", "preserve_all"),
    ),
    ParamSpec(
        # FA-02 production texture-restoration mode. "legacy" (default) is the
        # existing behaviour: skin.restore_micro_texture runs whenever
        # micro_restore > 0, which is true for nearly every recipe because
        # micro_restore's own default is 20 and the `natural` base recipe never
        # overrides it. Any other value routes to the NEW opt-in FA-02 path,
        # which first runs an eligibility gate (retouch/fa02_texture_eligibility)
        # and abstains unless the face clears every threshold.
        #
        # "dog" and "multiscale" are EXPERIMENTAL research arms with no evidence
        # behind them. No recipe sets this key and none should until the FA-02
        # scoring lock has a verdict — see the report referenced in
        # docs/plans/EXPERIMENT_FA02_TEXTURE_REPRESENTATIONS_2026_09_06.md.
        name="fa02_texture_mode",
        cli_flag="fa02-texture-mode",
        cli_type=str,
        default="legacy",
        recipe_key="fa02_texture_mode",
        conversion="dropdown",
        choices=("legacy", "raw_residual", "dog", "multiscale"),
    ),
    ParamSpec(
        name="micro_restore",
        cli_flag="micro-restore",
        cli_type=int,
        default=20,
        recipe_key="micro_restore",
        conversion="gui_direct",
        min_val=0,
        max_val=50,
    ),
    ParamSpec(
        name="micro_dodge_burn",
        cli_flag="micro-dodge-burn",
        cli_type=int,
        default=0,
        recipe_key="skin.micro_db",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="redness_even",
        cli_flag="redness-even",
        cli_type=int,
        default=0,
        recipe_key="skin.redness_even",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="hb_even",
        cli_flag="hb-even",
        cli_type=float,
        default=0.0,
        recipe_key="skin.hb_even",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="hb_shift",
        cli_flag="hb-shift",
        cli_type=float,
        default=0.0,
        recipe_key="skin.hb_shift",
        conversion="recipe_direct",
        min_val=-1.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="whiten_hue_stable",
        cli_flag="whiten-hue-stable",
        cli_type=bool,
        default=False,
        recipe_key="skin.whiten_hue_stable",
        conversion="bool_flag",
    ),
    ParamSpec(
        name="whiten",
        cli_flag="whiten",
        cli_type=int,
        default=10,
        recipe_key="skin.rosy",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="equalize",
        cli_flag="equalize",
        cli_type=int,
        default=0,
        recipe_key="skin.equalize",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="blemish",
        cli_flag="blemish",
        cli_type=int,
        default=30,
        recipe_key=None,
        conversion="alias",
        alias_of="smooth",
    ),
    ParamSpec(
        name="whiten_tone",
        cli_flag="whiten-tone",
        cli_type=str,
        default="rosy",
        recipe_key="skin.porcelain",
        conversion="dropdown",
    ),
    ParamSpec(
        name="nose_blush",
        cli_flag="nose-blush",
        cli_type=bool,
        default=False,
        recipe_key="nose_blush",
        conversion="bool_flag",
    ),
    ParamSpec(
        name="under_eye_blush",
        cli_flag="under-eye-blush",
        cli_type=bool,
        default=False,
        recipe_key="under_eye_blush",
        conversion="bool_flag",
    ),
    ParamSpec(
        name="white_costume_lift",
        cli_flag="white-costume-lift",
        cli_type=bool,
        default=False,
        recipe_key="white_costume_lift",
        conversion="bool_flag",
    ),
    ParamSpec(
        name="dodge_burn",
        cli_flag="dodge-burn",
        cli_type=int,
        default=0,
        recipe_key="dodge_burn",
        conversion="dodge_burn_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="relight",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="skin.relight",
        conversion="relight_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="relight_azimuth",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="skin.relight_azimuth",
        engine_recipe_key="relight_azimuth",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="relight_elevation",
        cli_flag=None,
        cli_type=None,
        default=30.0,
        recipe_key="skin.relight_elevation",
        engine_recipe_key="relight_elevation",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="sculpt",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="skin.sculpt",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="shine_removal",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="skin.shine_removal",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="wrinkle_soften",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="skin.wrinkle_soften",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="wrinkle_soften_forehead",
        cli_flag="wrinkle-soften-forehead",
        cli_type=float,
        default=0.0,
        recipe_key="skin.wrinkle_soften_forehead",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="wrinkle_soften_nasolabial",
        cli_flag="wrinkle-soften-nasolabial",
        cli_type=float,
        default=0.0,
        recipe_key="skin.wrinkle_soften_nasolabial",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="wrinkle_soften_neck",
        cli_flag="wrinkle-soften-neck",
        cli_type=float,
        default=0.0,
        recipe_key="skin.wrinkle_soften_neck",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="texture_transplant",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="skin.texture_transplant",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_smooth",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="body_skin.smooth",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_equalize",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="body_skin.equalize",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_whiten",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="body_skin.whiten",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_match_face",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="body_skin.match_face",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_relight",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="body_skin.relight",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_dodge_burn",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="body_skin.dodge_burn",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="shadow_lift",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="skin.shadow_lift",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="face_exposure",
        cli_flag="face-exposure",
        cli_type=float,
        default=0,
        recipe_key="skin.face_exposure",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_shadow_lift",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="body_skin.shadow_lift",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="nose_restore",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="skin.nose_restore",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="skin_sss",
        cli_flag="skin-sss",
        cli_type=float,
        default=0,
        recipe_key="skin.sss",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="specular_bloom",
        cli_flag="specular-bloom",
        cli_type=int,
        default=0,
        recipe_key="specular_bloom",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="specular_bloom_tone",
        cli_flag="specular-bloom-tone",
        cli_type=str,
        default="rosy",
        recipe_key="specular_bloom_tone",
        conversion="dropdown",
    ),
    ParamSpec(
        name="specular_finish",
        cli_flag="specular-finish",
        cli_type=str,
        default="matte",
        recipe_key="skin.specular_finish",
        conversion="dropdown",
        min_val=None,
        max_val=None,
    ),
    ParamSpec(
        name="specular_finish_strength",
        cli_flag="specular-finish-strength",
        cli_type=float,
        default=0.5,
        recipe_key="skin.specular_finish_strength",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="specular_recolor",
        cli_flag="specular-recolor",
        cli_type=float,
        default=0.0,
        recipe_key="skin.specular_recolor",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="albedo_even",
        cli_flag="albedo-even",
        cli_type=float,
        default=0.0,
        recipe_key="skin.albedo_even",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="makeup_coverage_even",
        cli_flag="makeup-coverage-even",
        cli_type=float,
        default=0.0,
        recipe_key="skin.makeup_coverage_even",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="makeup_cake_reduce",
        cli_flag="makeup-cake-reduce",
        cli_type=float,
        default=0.0,
        recipe_key="skin.makeup_cake_reduce",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="hemoglobin_smooth",
        cli_flag="hemoglobin-smooth",
        cli_type=float,
        default=0.0,
        recipe_key="skin.hemoglobin_smooth",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="mole_protect",
        cli_flag="mole-protect",
        cli_type=float,
        default=0.0,
        recipe_key="skin.mole_protect",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="vein_attenuate",
        cli_flag="vein-attenuate",
        cli_type=float,
        default=0.0,
        recipe_key="skin.vein_attenuate",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="skin_flatten",
        cli_flag="skin-flatten",
        cli_type=int,
        default=0,
        recipe_key="skin.flatten",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="skin_quantize",
        cli_flag="skin-quantize",
        cli_type=int,
        default=0,
        recipe_key="skin.quantize",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="skin_unify",
        cli_flag="skin-unify",
        cli_type=int,
        default=0,
        recipe_key="skin.unify",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="skin_unify_hue",
        cli_flag="skin-unify-hue",
        cli_type=float,
        default=-1.0,
        recipe_key="skin.unify_hue",
        conversion="recipe_direct",
        min_val=-1.0,
        max_val=360.0,
    ),
    ParamSpec(
        name="skin_hue_unify",
        cli_flag="skin-hue-unify",
        cli_type=int,
        default=0,
        recipe_key="skin.hue_unify",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="skin_chroma_even",
        cli_flag="skin-chroma-even",
        cli_type=int,
        default=0,
        recipe_key="skin.chroma_even",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="skin_glow",
        cli_flag="skin-glow",
        cli_type=int,
        default=0,
        recipe_key="skin.glow",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="mask_feather_mode",
        cli_flag="mask-feather-mode",
        cli_type=str,
        default="gaussian",
        recipe_key="mask.feather_mode",
        conversion="dropdown",
        min_val=None,
        max_val=None,
    ),
]


# Eyes / lips / teeth / hair
_FACE_FEATURE_PARAMS = [
    ParamSpec(
        name="eye_enhance",
        cli_flag="eye-enhance",
        cli_type=int,
        default=5,
        recipe_key="eyes.whites",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="catchlight",
        cli_flag="catchlight",
        cli_type=int,
        default=5,
        recipe_key="eyes.catchlight",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="corneal_shading",
        cli_flag="corneal-shading",
        cli_type=int,
        default=0,
        recipe_key="eyes.corneal_shading",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="dark_circles",
        cli_flag="dark-circles",
        cli_type=int,
        default=0,
        recipe_key="eyes.dark_circles",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="undereye_darken_removal",
        cli_flag="undereye-darken-removal",
        cli_type=int,
        default=0,
        recipe_key="undereye.darken_removal",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="undereye_puffiness_reduction",
        cli_flag="undereye-puffiness-reduction",
        cli_type=int,
        default=0,
        recipe_key="undereye.puffiness_reduction",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="eye_sclera_brighten",
        cli_flag="eye-sclera-brighten",
        cli_type=int,
        default=0,
        recipe_key="eye.sclera_brighten",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="eye_sclera_vessel_remove",
        cli_flag="eye-sclera-vessel-remove",
        cli_type=int,
        default=0,
        recipe_key="eyes.sclera_vessel_remove",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="eye_gate",
        cli_flag="eye-gate",
        cli_type=bool,
        default=True,
        recipe_key="eyes.gate",
        conversion="bool_flag",
    ),
    ParamSpec(
        name="backdrop_cleanup",
        cli_flag="backdrop-cleanup",
        cli_type=int,
        default=0,
        recipe_key="background.backdrop_cleanup",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="fabric_wrinkle_smooth",
        cli_flag="fabric-wrinkle-smooth",
        cli_type=float,
        default=0.0,
        recipe_key="fabric.wrinkle_smooth",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="eye_iris_saturate",
        cli_flag="eye-iris-saturate",
        cli_type=int,
        default=0,
        recipe_key="eye.iris_saturate",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="eye_iris_hue_shift",
        cli_flag="eye-iris-hue-shift",
        cli_type=int,
        default=0,
        recipe_key="eye.iris_hue_shift",
        conversion="recipe_direct",
        min_val=-30,
        max_val=30,
    ),
    ParamSpec(
        name="eye_iris_brightness",
        cli_flag="eye-iris-brightness",
        cli_type=int,
        default=0,
        recipe_key="eye.iris_brightness",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="teeth_whiten",
        cli_flag="teeth-whiten",
        cli_type=int,
        default=5,
        recipe_key="eyes.whites",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="lip_enhance",
        cli_flag="lip-enhance",
        cli_type=int,
        default=5,
        recipe_key="lips.gloss",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="lip_tint",
        cli_flag="lip-tint",
        cli_type=str,
        default=None,
        gui_default="none",
        recipe_key="lips.tint",
        conversion="lip_tint_direct",
    ),
    ParamSpec(
        name="lip_finish",
        cli_flag="lip-finish",
        cli_type=str,
        default="gloss",
        recipe_key="lip_finish",
        conversion="dropdown",
    ),
    ParamSpec(
        name="blush",
        cli_flag="blush",
        cli_type=int,
        default=0,
        recipe_key="blush",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="slimming",
        cli_flag="slimming",
        cli_type=int,
        default=0,
        recipe_key="slimming",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="reshape_eye_size",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.eye_size",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_eye_distance",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.eye_distance",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_nose_width",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.nose_width",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_nose_length",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.nose_length",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_jaw_width",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.jaw_width",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_chin_length",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.chin_length",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_mouth_size",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.mouth_size",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_smile",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.smile",
        conversion="gui_direct",
        min_val=-20,
        max_val=30,
    ),
    ParamSpec(
        name="reshape_forehead",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.forehead",
        conversion="gui_direct",
        min_val=-30,
        max_val=30,
    ),
    ParamSpec(
        name="reshape_jaw_width_l",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.jaw_width_l",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_jaw_width_r",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.jaw_width_r",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_nose_width_l",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.nose_width_l",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_nose_width_r",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.nose_width_r",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_eye_size_l",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.eye_size_l",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_eye_size_r",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.eye_size_r",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_neck_width",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.neck_width",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="reshape_neck_length",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="reshape.neck_length",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="hair_enhance",
        cli_flag="hair-enhance",
        cli_type=int,
        default=5,
        recipe_key="hair.shine",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="hair_deglare",
        cli_flag="hair-deglare",
        cli_type=int,
        default=0,
        recipe_key="hair.deglare",
        # 0-100 raw pass-through: the engine consumer (hairwork.deglare_wig)
        # expects strength in 0-100, and the recipe stores 0-100 (e.g.
        # auto_clean_v1 -> 30).  recipe_pct here would ×100 the recipe value
        # (30 -> 3000), feeding 30x-over-range garbage to the engine.
        conversion="recipe_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="hair_ring_position",
        cli_flag="hair-ring-position",
        cli_type=int,
        default=30,
        recipe_key="hair.ring_position",
        # 0-100 raw pass-through (hairwork.add_angel_ring position is 0-100).
        conversion="recipe_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="hair_ring_tint",
        cli_flag="hair-ring-tint",
        cli_type=int,
        default=40,
        recipe_key="hair.ring_tint",
        # 0-100 raw pass-through (hairwork.add_angel_ring tint is 0-100).
        conversion="recipe_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="hair_remove_flyaways",
        cli_flag="hair-remove-flyaways",
        cli_type=int,
        default=0,
        recipe_key="hair.remove_flyaways",
        # 0-100 raw pass-through (hairwork.remove_flyaways strength is 0-100).
        conversion="recipe_direct",
        min_val=0,
        max_val=100,
    ),
]


# Tonal / global
_TONAL_PARAMS = [
    ParamSpec(
        name="contrast",
        cli_flag="contrast",
        cli_type=int,
        default=0.0,
        recipe_key="contrast",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="brightness",
        cli_flag="brightness",
        cli_type=int,
        default=None,
        gui_default=0,
        recipe_key="brightness",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="highlights",
        cli_flag="highlights",
        cli_type=int,
        default=None,
        gui_default=0,
        recipe_key="highlights",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ),
    ParamSpec(
        name="shadows",
        cli_flag="shadows",
        cli_type=int,
        default=None,
        gui_default=0,
        recipe_key="shadows",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ),
    ParamSpec(
        name="whites",
        cli_flag="whites",
        cli_type=int,
        default=None,
        gui_default=0,
        recipe_key="whites",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ),
    ParamSpec(
        name="blacks",
        cli_flag="blacks",
        cli_type=int,
        default=None,
        gui_default=0,
        recipe_key="blacks",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ),
    ParamSpec(
        name="clarity",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="clarity",
        conversion="gui_direct",
        min_val=-50,
        max_val=50,
    ),
    ParamSpec(
        name="vibrance",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="vibrance",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ),
    ParamSpec(
        name="saturation",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="saturation",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ),
    ParamSpec(
        name="auto_exposure",
        cli_flag="auto-exposure",
        cli_type=bool,
        default=False,
        recipe_key="auto_exposure",
        conversion="bool_flag",
    ),
]


# Bloom / lens / finish
_LENS_PARAMS = [
    ParamSpec(
        name="bloom",
        cli_flag=None,
        cli_type=float,
        default=0.0,
        recipe_key="bloom.opacity",
        conversion="recipe_pct",
        min_val=0.0,
        max_val=100.0,
    ),
    ParamSpec(
        name="bloom_threshold",
        cli_flag=None,
        cli_type=None,
        default=210.0,
        recipe_key="bloom.threshold",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="bloom_softness",
        cli_flag=None,
        cli_type=None,
        default=30.0,
        recipe_key="bloom.softness",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="glow",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="glow",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="vignette",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="vignette",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ),
    ParamSpec(
        name="sharpen",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="sharpen",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="sharpen_radius",
        cli_flag=None,
        cli_type=None,
        default=1.0,
        recipe_key="sharpen_radius",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="subject_separation",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="subject_separation",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="impact",
        cli_flag="impact",
        cli_type=int,
        default=0.0,
        recipe_key="finish.impact",
        conversion="engine_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="fade_toe",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="finish.fade_toe",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="highlight_drift",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="finish.highlight_drift",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="airy_haze",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="finish.airy_haze",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="clarity_split_neg",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="finish.clarity_split_neg",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="clarity_split_pos",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="finish.clarity_split_pos",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
]


# Color grading / film effects
_GRADING_PARAMS = [
    ParamSpec(
        name="color_grade",
        cli_flag="color-grade",
        cli_type=str,
        default="none",
        recipe_key="color_harmony.preset",
        conversion="dropdown",
    ),
    ParamSpec(
        name="grade_intensity",
        cli_flag="grade-intensity",
        cli_type=float,
        default=0.0,
        recipe_key="color_harmony.amount",
        conversion="gui_div100",
    ),
    ParamSpec(
        name="color_transfer_intensity",
        cli_flag="color-transfer-intensity",
        cli_type=float,
        default=1.0,
        recipe_key="color_transfer_intensity",
        conversion="gui_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="chromatic_aberration",
        cli_flag="chromatic-aberration",
        cli_type=float,
        default=0.0,
        recipe_key="chromatic_aberration",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="grain",
        cli_flag="grain",
        cli_type=float,
        default=0.0,
        recipe_key="grain",
        conversion="gui_mul500",
    ),
    ParamSpec(
        name="halation",
        cli_flag=None,
        cli_type=float,
        default=0.0,
        recipe_key="halation",
        conversion="gui_div100",
    ),
    ParamSpec(
        name="lut",
        cli_flag="lut",
        cli_type=str,
        default="none",
        recipe_key="lut",
        conversion="dropdown",
    ),
    ParamSpec(
        name="tonal_curve_strength",
        cli_flag="tonal-curve-strength",
        cli_type=float,
        default=0.0,
        recipe_key="tonal_curve_strength",
        conversion="gui_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="skin_protect_strength",
        cli_flag="skin-protect",
        cli_type=float,
        default=0.0,
        recipe_key="skin_protect",
        conversion="gui_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="grain_strength",
        cli_flag="film-grain",
        cli_type=float,
        default=0.0,
        recipe_key="grain_strength",
        conversion="gui_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="highlight_rolloff_strength",
        cli_flag="highlight-rolloff",
        cli_type=float,
        default=0.0,
        recipe_key="highlight_rolloff",
        conversion="gui_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="gamut_compress",
        cli_flag="gamut-compress",
        cli_type=bool,
        default=True,
        recipe_key="gamut_compress",
        conversion="bool_flag",
    ),
    ParamSpec(
        name="saturation_mode",
        cli_flag="saturation-mode",
        cli_type=str,
        default="additive",
        recipe_key="saturation_mode",
        conversion="dropdown",
    ),
]


# Film density engine (C3 — parametric film-density model)
_FILM_DENSITY_PARAMS = [
    ParamSpec(
        name="film_enable",
        cli_flag=None,
        cli_type=None,
        default=False,
        recipe_key="film.enable",
        conversion="bool_flag",
    ),
    ParamSpec(
        name="film_strength",
        cli_flag=None,
        cli_type=None,
        default=1.0,
        recipe_key="film.strength",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="film_toe_r",
        cli_flag=None,
        cli_type=None,
        default=0.10,
        recipe_key="film.toe.r",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=0.5,
    ),
    ParamSpec(
        name="film_toe_g",
        cli_flag=None,
        cli_type=None,
        default=0.10,
        recipe_key="film.toe.g",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=0.5,
    ),
    ParamSpec(
        name="film_toe_b",
        cli_flag=None,
        cli_type=None,
        default=0.10,
        recipe_key="film.toe.b",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=0.5,
    ),
    ParamSpec(
        name="film_shoulder_r",
        cli_flag=None,
        cli_type=None,
        default=0.10,
        recipe_key="film.shoulder.r",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=0.5,
    ),
    ParamSpec(
        name="film_shoulder_g",
        cli_flag=None,
        cli_type=None,
        default=0.10,
        recipe_key="film.shoulder.g",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=0.5,
    ),
    ParamSpec(
        name="film_shoulder_b",
        cli_flag=None,
        cli_type=None,
        default=0.10,
        recipe_key="film.shoulder.b",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=0.5,
    ),
    ParamSpec(
        name="film_midpoint",
        cli_flag=None,
        cli_type=None,
        default=0.50,
        recipe_key="film.midpoint",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="film_gamma",
        cli_flag=None,
        cli_type=None,
        default=1.0,
        recipe_key="film.gamma",
        conversion="recipe_direct",
        min_val=0.5,
        max_val=2.5,
    ),
    ParamSpec(
        name="film_crosstalk_cy_mg",
        cli_flag=None,
        cli_type=None,
        default=0.06,
        recipe_key="film.crosstalk.cy_mg",
        conversion="recipe_direct",
        min_val=-0.15,
        max_val=0.15,
    ),
    ParamSpec(
        name="film_crosstalk_cy_ye",
        cli_flag=None,
        cli_type=None,
        default=0.03,
        recipe_key="film.crosstalk.cy_ye",
        conversion="recipe_direct",
        min_val=-0.15,
        max_val=0.15,
    ),
    ParamSpec(
        name="film_crosstalk_mg_ye",
        cli_flag=None,
        cli_type=None,
        default=0.02,
        recipe_key="film.crosstalk.mg_ye",
        conversion="recipe_direct",
        min_val=-0.15,
        max_val=0.15,
    ),
    ParamSpec(
        name="film_tonemap_strength",
        cli_flag=None,
        cli_type=None,
        default=0.7,
        recipe_key="film.tonemap.strength",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="film_tonemap_toe",
        cli_flag=None,
        cli_type=None,
        default=0.10,
        recipe_key="film.tonemap.toe",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=0.5,
    ),
    ParamSpec(
        name="film_tonemap_shoulder",
        cli_flag=None,
        cli_type=None,
        default=0.15,
        recipe_key="film.tonemap.shoulder",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=0.5,
    ),
    ParamSpec(
        name="film_skew",
        cli_flag=None,
        cli_type=None,
        default=0.3,
        recipe_key="film.tonemap.skew",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
    ParamSpec(
        name="film_highlight_purity",
        cli_flag="film-highlight-purity",
        cli_type=float,
        default=0.0,
        recipe_key="film.tonemap.highlight_purity",
        conversion="recipe_direct",
        min_val=0.0,
        max_val=1.0,
    ),
]


# C5 — Skin-anchored background color harmonization.
# background_harmonize: 0-100 strength of the background auto-grade that
# complements the corrected skin tone. 0 = no-op.
# background_harmonize_mode: harmony strategy on the hue wheel
# ("split" = split-complementary, the default; also "complementary",
# "analogous_warm", "analogous_cool").
_HARMONIZE_PARAMS = [
    ParamSpec(
        name="background_harmonize",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="harmony.background_harmonize",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="background_harmonize_mode",
        cli_flag=None,
        cli_type=None,
        default="split",
        recipe_key="harmony.background_harmonize_mode",
        conversion="dropdown",
    ),
]


# Add HSL adjustments and Calibration parameters dynamically to _GRADING_PARAMS
for color in ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]:
    _GRADING_PARAMS.append(ParamSpec(
        name=f"hsl_hue_{color}",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key=f"hsl_adjustments.hue.{color}",
        conversion="gui_direct",
        min_val=-180,
        max_val=180,
    ))
    _GRADING_PARAMS.append(ParamSpec(
        name=f"hsl_sat_{color}",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key=f"hsl_adjustments.saturation.{color}",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ))
    _GRADING_PARAMS.append(ParamSpec(
        name=f"hsl_lum_{color}",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key=f"hsl_adjustments.luminance.{color}",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ))

for color in ["red", "green", "blue"]:
    _GRADING_PARAMS.append(ParamSpec(
        name=f"calibration_{color}_hue",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key=f"calibration.{color}.hue",
        conversion="gui_direct",
        min_val=-180,
        max_val=180,
    ))
    _GRADING_PARAMS.append(ParamSpec(
        name=f"calibration_{color}_sat",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key=f"calibration.{color}.sat",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ))
    _GRADING_PARAMS.append(ParamSpec(
        name=f"calibration_{color}_lum",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key=f"calibration.{color}.lum",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ))


# T1 — Background replace & scene relight.
# These 7 keys are the historically-dead ``anime_crystal_void`` params,
# finally wired (T1). Each maps to a BackgroundReplacer operation gated
# by the feathered person mask so the subject is always protected.
# Recipe path is ``background.<key>`` (a new nested root). All are 0-100
# strength scales; 0 = no-op.
_BACKGROUND_PARAMS = [
    ParamSpec(
        name="background_blur",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="background.background_blur",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="lens_blur",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="background.lens_blur",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="background_desaturation",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="background.background_desaturation",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="light_wrap",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="background.light_wrap",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="blue_shadow_grade",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="background.blue_shadow_grade",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="cyan_midtone_grade",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="background.cyan_midtone_grade",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="subject_sharpen",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="background.subject_sharpen",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="matte_black",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="background.matte_black",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
]


# Split toning
_SPLIT_TONING_PARAMS = [
    ParamSpec(
        name="shadow_hue",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="shadow_hue",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="shadow_sat",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="shadow_sat",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="midtone_hue",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="midtone_hue",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="midtone_sat",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="midtone_sat",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="highlight_hue",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="highlight_hue",
        conversion="gui_direct",
    ),
    ParamSpec(
        name="highlight_sat",
        cli_flag=None,
        cli_type=None,
        default=0.0,
        recipe_key="highlight_sat",
        conversion="gui_direct",
    ),
]

# White balance / B&W mixer / negative split tone
_WB_BW_PARAMS = [
    ParamSpec(
        name="white_balance_kelvin",
        cli_flag="wb-kelvin",
        cli_type=int,
        default=6500,
        recipe_key="white_balance_kelvin",
        conversion="gui_direct",
        min_val=2000,
        max_val=12000,
    ),
    ParamSpec(
        name="white_balance_tint",
        cli_flag="wb-tint",
        cli_type=float,
        default=0.0,
        recipe_key="white_balance_tint",
        conversion="gui_direct",
        min_val=-100.0,
        max_val=100.0,
    ),
    ParamSpec(
        name="bw_channel_mixer_r",
        cli_flag="bw-r",
        cli_type=int,
        default=30,
        recipe_key="bw_channel_mixer_r",
        conversion="gui_direct",
        min_val=-100,
        max_val=200,
    ),
    ParamSpec(
        name="bw_channel_mixer_g",
        cli_flag="bw-g",
        cli_type=int,
        default=59,
        recipe_key="bw_channel_mixer_g",
        conversion="gui_direct",
        min_val=-100,
        max_val=200,
    ),
    ParamSpec(
        name="bw_channel_mixer_b",
        cli_flag="bw-b",
        cli_type=int,
        default=11,
        recipe_key="bw_channel_mixer_b",
        conversion="gui_direct",
        min_val=-100,
        max_val=200,
    ),
    ParamSpec(
        name="negative_split_tone_shadow",
        cli_flag="neg-split-shadow",
        cli_type=float,
        default=0.0,
        recipe_key="negative_split_tone_shadow",
        conversion="gui_direct",
        min_val=0.0,
        max_val=100.0,
    ),
    ParamSpec(
        name="negative_split_tone_highlight",
        cli_flag="neg-split-highlight",
        cli_type=float,
        default=0.0,
        recipe_key="negative_split_tone_highlight",
        conversion="gui_direct",
        min_val=0.0,
        max_val=100.0,
    ),
    ParamSpec(
        name="hsl_hue_global",
        cli_flag="hsl-hue",
        cli_type=int,
        default=0,
        recipe_key="hsl_hue_global",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ),
    ParamSpec(
        name="hsl_sat_global",
        cli_flag="hsl-sat",
        cli_type=int,
        default=0,
        recipe_key="hsl_sat_global",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ),
    ParamSpec(
        name="hsl_lum_global",
        cli_flag="hsl-lum",
        cli_type=int,
        default=0,
        recipe_key="hsl_lum_global",
        conversion="gui_direct",
        min_val=-100,
        max_val=100,
    ),
]


# F7 — AI denoise + super-resolution.
# ai_denoise: 0-100 opacity blend between original and denoised image
# (model runs once; this controls the mix). 0 = no-op.
# ai_sr_scale: export-time upscale factor (1=off, 2, 4). GUI flag, not a
# recipe-driven parameter — the recipe never sets it; callers opt in.
_AI_ENHANCE_PARAMS = [
    ParamSpec(
        name="ai_denoise",
        cli_flag="ai-denoise",
        cli_type=int,
        default=0,
        recipe_key="ai.denoise",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="ai_sr_scale",
        cli_flag="ai-sr-scale",
        cli_type=int,
        default=1,
        recipe_key=None,
        conversion="gui_direct",
        min_val=1,
        max_val=4,
    ),
]


# T2 — Makeup engine v2 (eyeshadow / liner / contour / brows / ombre).
# All keys live under the recipe root ``makeup_v2.<key>``. Each is a 0-100
# strength scale (0 = no-op) except the dropdown/style selectors which store
# the string verbatim.
_MAKEUP_V2_PARAMS = [
    ParamSpec(
        name="mv2_eyeshadow",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="makeup_v2.eyeshadow",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="mv2_eyeshadow_color",
        cli_flag=None,
        cli_type=None,
        default="rose",
        recipe_key="makeup_v2.eyeshadow_color",
        conversion="dropdown",
    ),
    ParamSpec(
        name="mv2_eyeshadow_style",
        cli_flag=None,
        cli_type=None,
        default="natural",
        recipe_key="makeup_v2.eyeshadow_style",
        conversion="dropdown",
    ),
    ParamSpec(
        name="mv2_eyeliner",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="makeup_v2.eyeliner",
        conversion="gui_direct",
        min_val=0,
        max_val=10,
    ),
    ParamSpec(
        name="mv2_eyeliner_color",
        cli_flag=None,
        cli_type=None,
        default="black",
        recipe_key="makeup_v2.eyeliner_color",
        conversion="dropdown",
    ),
    ParamSpec(
        name="mv2_eyeliner_style",
        cli_flag=None,
        cli_type=None,
        default="classic",
        recipe_key="makeup_v2.eyeliner_style",
        conversion="dropdown",
    ),
    ParamSpec(
        name="mv2_contour",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="makeup_v2.contour",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="mv2_brows",
        cli_flag=None,
        cli_type=None,
        default=0,
        recipe_key="makeup_v2.brows",
        conversion="gui_direct",
        min_val=0,
        max_val=10,
    ),
    ParamSpec(
        name="mv2_brows_color",
        cli_flag=None,
        cli_type=None,
        default="brown",
        recipe_key="makeup_v2.brows_color",
        conversion="dropdown",
    ),
    ParamSpec(
        name="mv2_ombre",
        cli_flag=None,
        cli_type=None,
        default=False,
        recipe_key="makeup_v2.ombre",
        conversion="bool_flag",
    ),
    ParamSpec(
        name="mv2_ombre_color1",
        cli_flag=None,
        cli_type=None,
        default="red",
        recipe_key="makeup_v2.ombre_color1",
        conversion="dropdown",
    ),
    ParamSpec(
        name="mv2_ombre_color2",
        cli_flag=None,
        cli_type=None,
        default="pink",
        recipe_key="makeup_v2.ombre_color2",
        conversion="dropdown",
    ),
]


# --- A4: Neural boosters (PARKED — compatibility-only) ---
# Keep the parameter names for loading old sessions, but do not expose CLI
# controls for effects whose segmenters intentionally return empty masks.
# See retouch/neural_boosters.py for the explicit unavailable status.
_NEURAL_BOOSTER_PARAMS = [
    ParamSpec(
        name="neural_stray_hair_boost",
        cli_flag=None,
        cli_type=int,
        default=0,
        gui_default=0,
        recipe_key="neural.stray_hair_boost",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="neural_defect_boost",
        cli_flag=None,
        cli_type=int,
        default=0,
        gui_default=0,
        recipe_key="neural.defect_boost",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
]

_COSPLAY_MOAT_PARAMS = [
    ParamSpec(
        name="cosplay_wig_lace_blend",
        cli_flag="cosplay-wig-lace-blend",
        cli_type=int,
        default=0,
        gui_default=0,
        recipe_key="cosplay.wig_lace_blend",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="cosplay_stockings_smooth",
        cli_flag="cosplay-stockings-smooth",
        cli_type=int,
        default=0,
        gui_default=0,
        recipe_key="cosplay.stockings_smooth",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="cosplay_consistency_strength",
        cli_flag="cosplay-consistency-strength",
        cli_type=int,
        default=0,
        gui_default=0,
        recipe_key="cosplay.consistency_strength",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
]


# T3 — Body reshape via MediaPipe Pose
# body_reshape_arm_length, body_reshape_leg_length, body_reshape_torso_width,
# body_reshape_shoulder_width, body_reshape_hip_width: all 0-100 strength scales;
# 0 = no-op; ±100 slider causes ±15% segment displacement (proportional).
# Recipe path is ``body_reshape.<key>`` (new nested root).
_BODY_RESHAPE_PARAMS = [
    ParamSpec(
        name="body_reshape_arm_length",
        cli_flag=None,
        cli_type=None,
        default=50.0,
        recipe_key="body_reshape.arm_length",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_reshape_leg_length",
        cli_flag=None,
        cli_type=None,
        default=50.0,
        recipe_key="body_reshape.leg_length",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_reshape_torso_width",
        cli_flag=None,
        cli_type=None,
        default=50.0,
        recipe_key="body_reshape.torso_width",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_reshape_shoulder_width",
        cli_flag=None,
        cli_type=None,
        default=50.0,
        recipe_key="body_reshape.shoulder_width",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="body_reshape_hip_width",
        cli_flag=None,
        cli_type=None,
        default=50.0,
        recipe_key="body_reshape.hip_width",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="auto_body_reshape",
        cli_flag="auto-body-reshape",
        cli_type=float,
        default=0.0,
        recipe_key="body_reshape.auto",
        conversion="gui_direct",
        min_val=0,
        max_val=100,
    ),
]

_NEW_FEATURE_PARAMS: List[ParamSpec] = [
    ParamSpec(
        name="purple_fringing",
        cli_flag="purple-fringing",
        cli_type=int,
        default=0,
        recipe_key="finish.purple_fringing",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="flyaway_cleanup",
        cli_flag="flyaway-cleanup",
        cli_type=int,
        default=0,
        recipe_key="hair.flyaway_cleanup",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="micro_grain",
        cli_flag="micro-grain",
        cli_type=int,
        default=0,
        recipe_key="finish.micro_grain",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
    ParamSpec(
        name="split_toning",
        cli_flag="split-toning",
        cli_type=int,
        default=0,
        recipe_key="finish.split_toning",
        conversion="recipe_pct",
        min_val=0,
        max_val=100,
    ),
]



PROCESSING_PARAMS: List[ParamSpec] = (
    _SKIN_PARAMS
    + _FACE_FEATURE_PARAMS
    + _TONAL_PARAMS
    + _LENS_PARAMS
    + _GRADING_PARAMS
    + _FILM_DENSITY_PARAMS
    + _HARMONIZE_PARAMS
    + _BACKGROUND_PARAMS
    + _SPLIT_TONING_PARAMS
    + _WB_BW_PARAMS
    + _AI_ENHANCE_PARAMS
    + _MAKEUP_V2_PARAMS
    + _NEURAL_BOOSTER_PARAMS
    + _COSPLAY_MOAT_PARAMS
    + _BODY_RESHAPE_PARAMS
    + _NEW_FEATURE_PARAMS
)



# ParamSpec lookup by name (lazy-built)
_BY_NAME: Dict[str, ParamSpec] = {p.name: p for p in PROCESSING_PARAMS}


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_param(name: str) -> ParamSpec:
    """Return the ``ParamSpec`` for *name*.  Raises ``KeyError`` if unknown."""
    return _BY_NAME[name]


def param_names() -> List[str]:
    """Return the ordered list of all parameter names."""
    return [p.name for p in PROCESSING_PARAMS]


# ---------------------------------------------------------------------------
# Recipe → params
# ---------------------------------------------------------------------------


def resolve_recipe(name: str, _seen: Optional[set] = None) -> Dict[str, Any]:
    """Return a fully-merged recipe dict, resolving 'extends' recursively.

    This is the single source of truth for recipe resolution.  It lives in
    ``retouch.params`` (not ``retouch.engine``) because it is pure data
    manipulation over the ``RECIPES`` table and has no engine dependencies —
    that placement keeps both the engine and the recipe loader free of
    circular imports.
    """
    if _seen is None:
        _seen = set()
    if name in _seen:
        return RECIPES.get("natural", {})
    _seen.add(name)
    rec = RECIPES.get(name)
    if rec is None:
        rec = RECIPES.get("natural", {})
    parent_name = rec.get("extends")
    if parent_name and parent_name in RECIPES:
        resolved_parent = resolve_recipe(parent_name, _seen)
        merged = copy.deepcopy(resolved_parent)
        _deep_merge(merged, rec)
        return merged
    return rec


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """Merge *override* into *base* in-place, recursing into nested dicts."""
    for key, val in override.items():
        if key == "extends":
            continue
        if isinstance(val, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val


def _resolve_dodge_burn(rec: Dict[str, Any]) -> Any:
    """``dodge_burn`` in a recipe can be a dict or a raw number.

    The engine expects a percentage; the GUI does too.  We also support the
    "raw float that already looks like a percent" form that some recipes
    use (e.g. ``anime_cinematic_v1`` writes 18.0 directly).
    """
    raw = rec.get("dodge_burn", 0.0)
    if isinstance(raw, dict):
        amount = raw.get("amount", 0.0)
    else:
        amount = float(raw)
    if amount <= 1.0:
        return amount * 100.0
    return amount


def _resolve_recipe_value(
    spec: ParamSpec,
    rec: Dict[str, Any],
    *,
    use_engine_key: bool = False,
) -> Any:
    """Apply the spec's ``conversion`` to a recipe dict and return the engine-side value.

    Returns the spec's ``default`` if no recipe source is available.

    This is the *generic* converter used by ``build_context()`` for the
    ~50 parameters that follow a simple ``recipe_key → conversion`` shape.
    The handful of parameters that need bespoke lookups (e.g. ``whiten``,
    which falls back from ``porcelain`` to ``rosy``) are handled by
    ``build_context()`` directly before falling back to this function.

    Set ``use_engine_key=True`` to honour ``engine_recipe_key`` (used by
    the engine side) rather than ``recipe_key`` (used by the GUI side).
    """
    if use_engine_key and spec.engine_recipe_key is not None:
        recipe_key = spec.engine_recipe_key
    elif (not use_engine_key) and spec.gui_recipe_key is not None:
        recipe_key = spec.gui_recipe_key
    else:
        recipe_key = spec.recipe_key

    if spec.conversion == "static":
        return spec.default
    if spec.conversion in ("alias", "dodge_burn_pct", "relight_pct",
                          "whiten_pct", "whiten_tone_dropdown",
                          "catchlight_pct", "teeth_whiten_pct",
                          "eye_enhance_pct", "lip_enhance_pct",
                          "lip_tint_direct", "hair_enhance_pct",
                          "color_grade_direct"):
        # Handled by build_context(); this function is never called for
        # these names.  Return the default for safety.
        return spec.default
    if spec.conversion == "gui_direct":
        if recipe_key is None:
            return spec.default
        val = _lookup_recipe(rec, recipe_key)
        return spec.default if val is None else val
    if spec.conversion == "dropdown":
        if recipe_key is None:
            return spec.default
        val = _lookup_recipe(rec, recipe_key)
        if val is None or val == "":
            return spec.default
        return val
    if spec.conversion == "bool_flag":
        if recipe_key is None:
            return spec.default
        val = _lookup_recipe(rec, recipe_key)
        return spec.default if val is None else bool(val)
    if recipe_key is None:
        return spec.default
    raw = _lookup_recipe(rec, recipe_key)
    if raw is None:
        return spec.default
    return _engine_for_recipe_value(spec.conversion, raw)


def _resolve_gui_value(spec: ParamSpec, rec: Dict[str, Any]) -> Any:
    """Resolve a spec to a GUI-side value from a recipe dict.

    Returns the engine ``default`` when the recipe has no information and the
    GUI should fall back to the engine default.

    The returned value is coerced to ``int`` when ``cli_type`` is ``int`` so
    the GUI sliders see the same type they always have (the test suite in
    particular is sensitive to ``nose_smooth`` being an int).
    """
    # The GUI may use a different recipe path than the engine (e.g. relight).
    lookup_key = spec.gui_recipe_key or spec.recipe_key
    # The GUI may have its own default distinct from the engine's.  This is
    # useful for parameters that want ``None`` on the engine side but a
    # numeric slider default on the GUI side.
    fallback = spec.gui_default if spec.gui_default is not None else spec.default

    val: Any
    if spec.conversion == "static":
        val = spec.default
    elif spec.conversion == "alias":
        # Mirror the alias source's GUI value
        if spec.alias_of is None:
            val = spec.default
        else:
            src = _BY_NAME.get(spec.alias_of)
            if src is None:
                val = spec.default
            else:
                val = _resolve_gui_value(src, rec)
    elif spec.conversion == "dodge_burn_pct":
        if "dodge_burn" not in rec:
            val = spec.default
        else:
            val = int(round(_resolve_dodge_burn(rec)))
    elif spec.conversion == "relight_pct":
        if "skin" in rec and "relight" in rec["skin"]:
            val = int(round(float(rec["skin"]["relight"]) * 100.0))
        else:
            val = int(round(float(rec.get("relight_strength", 0.0))))
    elif spec.conversion == "dropdown":
        if lookup_key is None:
            val = spec.default
        else:
            v = _lookup_recipe(rec, lookup_key)
            val = spec.default if v is None or v == "" else v
            if val is None:
                val = "none"
    elif spec.conversion == "lip_tint_direct":
        if lookup_key is None:
            val = fallback
        else:
            v = _lookup_recipe(rec, lookup_key)
            if v is None or v == "":
                val = fallback if fallback is not None else "none"
            else:
                val = v
    elif spec.conversion == "bool_flag":
        if lookup_key is None:
            val = spec.default
        else:
            v = _lookup_recipe(rec, lookup_key)
            val = spec.default if v is None else bool(v)
    elif spec.conversion == "gui_direct":
        if lookup_key is None:
            val = fallback
        else:
            v = _lookup_recipe(rec, lookup_key)
            val = fallback if v is None else v
    elif lookup_key is None:
        val = spec.default
    else:
        raw = _lookup_recipe(rec, lookup_key)
        if raw is None:
            val = spec.default
        else:
            val = _gui_for_recipe_value(spec.conversion, raw)

    # Coerce to the expected GUI type
    if spec.cli_type is int and val is not None and not isinstance(val, bool):
        try:
            val = int(val)
        except (TypeError, ValueError):
            pass
    elif spec.cli_type is float and val is not None and not isinstance(val, bool):
        try:
            val = float(val)
        except (TypeError, ValueError):
            pass
    elif spec.cli_type is bool and val is not None:
        val = bool(val)
    return val


def _special_cases_gui(spec: ParamSpec, rec: Dict[str, Any]) -> Any:
    """Apply recipe-specific quirks that the generic conversion cannot express.

    These are the *documented* differences between the recipe and what the GUI
    had been hard-coding.  Centralising them here means the GUI and the
    engine can agree on a single source of truth.
    """
    name = spec.name
    if name == "whiten":
        # Engine `whiten` is the recipe's "rosy" or "porcelain" tone, × 100
        skin = rec.get("skin", {})
        if "porcelain" in skin:
            return int(round(float(skin["porcelain"]) * 100.0))
        return int(round(float(skin.get("rosy", 0.0)) * 100.0))
    if name == "whiten_tone":
        return "porcelain" if "porcelain" in rec.get("skin", {}) else "rosy"
    if name == "blemish":
        # Historically mirrored `smooth` in the GUI
        return int(round(float(rec.get("frequency", {}).get("smooth", 0.30)) * 100.0))
    if name == "catchlight":
        eyes = rec.get("eyes", {})
        if "catchlight" in eyes:
            return int(round(float(eyes["catchlight"]) * 100.0))
        if "iris" in eyes:
            return int(round(float(eyes["iris"]) * 100.0))
        return spec.default
    if name == "teeth_whiten":
        eyes = rec.get("eyes", {})
        if "teeth_whiten" in eyes:
            return int(round(float(eyes["teeth_whiten"]) * 100.0))
        if "whites" in eyes:
            return int(round(float(eyes["whites"]) * 100.0))
        return spec.default
    if name == "subject_separation":
        v = rec.get("subject_separation", 0.0) or 0.0
        return int(v if v > 1.0 else v * 100.0)
    return None


def recipe_to_params(recipe_name: str) -> Dict[str, Any]:
    """Return a {param_name: gui_value} dict for a given recipe.

    This is the single canonical function the GUI uses to populate its
    default values; the engine's ``build_context()`` uses the same spec
    list (via a different conversion path) to build the
    ``ProcessingContext``.

    Falls back to the ``"natural"`` recipe for unknown names, matching
    the behaviour of ``engine.resolve_recipe``.
    """
    name = recipe_name if recipe_name in RECIPES else "natural"
    rec = resolve_recipe(name)
    out: Dict[str, Any] = {}
    for spec in PROCESSING_PARAMS:
        special = _special_cases_gui(spec, rec)
        if special is not None:
            out[spec.name] = special
            continue
        out[spec.name] = _resolve_gui_value(spec, rec)
    return out


# ---------------------------------------------------------------------------
# GUI → engine kwarg translation
# ---------------------------------------------------------------------------

# A handful of GUI controls encode "no value" as a non-``None`` sentinel
# string (e.g. ``"none"``) or as a 0.  The engine wants ``None`` for those
# cases so it can fall through to the recipe default.  This set lists the
# param names that need the ``"none"`` → ``None`` swap.
_GUI_NONE_SENTINELS = {"lip_tint", "color_grade", "lut"}

# Params whose GUI value of 0 should be forwarded to the engine as ``None``
# (i.e. "leave at recipe default").  ``nose_smooth`` and the film effects
# are the canonical examples.
_GUI_ZERO_IS_NONE = {"nose_smooth", "chromatic_aberration"}


def _gui_to_engine_value(spec: ParamSpec, gui_value: Any) -> Any:
    """Convert a single GUI value into the value ``engine.process`` expects.

    Handles the dropdown sentinels, the zero-as-None mappings, and the
    inverse of the ``gui_mul500`` / ``gui_div100`` GUI scaling.
    """
    if gui_value is None:
        return None
    if spec.name in _GUI_NONE_SENTINELS and gui_value == "none":
        return None
    if spec.name in _GUI_ZERO_IS_NONE and gui_value == 0:
        return None
    if spec.conversion == "gui_mul500":
        return float(gui_value) / 500.0
    if spec.conversion == "gui_div100":
        return float(gui_value) / 100.0
    if spec.name == "grade_intensity":
        # The GUI stores grade_intensity as 0-100; the engine reads 0-1.
        return float(gui_value) / 100.0
    if spec.name == "subject_separation":
        # The GUI stores subject_separation as 0-100; the engine reads the
        # raw recipe value (0-1) — so divide.
        return float(gui_value) / 100.0
    return gui_value


def gui_values_to_engine_kwargs(
    gui_values: Dict[str, Any],
    *,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Translate a flat ``{param_name: gui_value}`` dict into the kwargs
    ``RetouchEngine.process`` expects.

    Unknown keys are passed through unchanged so callers can mix in
    transport keys (``color_ref``, ``fast``, ``debug_dir``, ...) without
    having to list them all here.
    """
    out: Dict[str, Any] = {}
    for spec in PROCESSING_PARAMS:
        if spec.name in gui_values:
            out[spec.name] = _gui_to_engine_value(spec, gui_values[spec.name])
    if extra:
        out.update(extra)
    return out


__all__ = [
    "ParamSpec",
    "PROCESSING_PARAMS",
    "get_param",
    "param_names",
    "recipe_to_params",
    "gui_values_to_engine_kwargs",
]
