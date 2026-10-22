"""One strength per face: scale every face-local amount of a face's recipe.

Like Capture One's Impact slider: 100% is the face's recipe as set, 0%
turns that face's retouch off, 150% pushes it further. The control sits on
top of the per-face recipe picker: each face gets its own value (stored as
``"strength"`` in its position-anchored ``face_params`` entry, so it stays on
the right person when detection order changes), and ``face_strength`` is
the "All faces" value. The two multiply.

Which settings scale is not a hand-picked list. :data:`SCALED` and
:data:`NOT_SCALED` together must equal the settings the per-face core and
face reshape actually read, which :func:`core_param_reads` derives from
their source; ``tests/test_face_strength.py`` fails when a newly read
setting is in neither table. Amounts (``smooth``, ``whiten``, ``iris_pop``,
a reshape slider ...) scale toward their "does nothing" value; settings
that say *how* or *which* (a colour, a mode, an angle, a protection) don't.
On/off switches stay on unless the face is at 0%.
"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from typing import Any, Dict, FrozenSet, Optional, Tuple

#: Highest per-face, all-faces and combined factor (150%).
MAX_FACTOR = 1.5

# Amount settings and their "does nothing" value. Everything here scales as
# neutral + factor * (value - neutral), clamped to the ParamSpec range.
SCALED: Dict[str, float] = {
    # skin
    "smooth": 0.0, "nose_smooth": 0.0, "mid_reduction": 0.0,
    "blotch_reduction": 0.0, "pore_synthesis": 0.0, "micro_restore": 0.0,
    "micro_dodge_burn": 0.0, "albedo_even": 0.0, "makeup_coverage_even": 0.0,
    "makeup_cake_reduce": 0.0, "equalize": 0.0, "whiten": 0.0,
    "redness_even": 0.0, "skin_hue_unify": 0.0, "skin_chroma_even": 0.0,
    "skin_unify": 0.0, "skin_flatten": 0.0, "skin_quantize": 0.0,
    "hb_even": 0.0, "hb_shift": 0.0, "hemoglobin_smooth": 0.0,
    "vein_attenuate": 0.0, "blemish": 0.0, "spot_heal": 0.0,
    "freckle_removal": 0.0, "highlight_repair": 0.0, "shine_removal": 0.0,
    "powder_finish": 0.0, "specular_bloom": 0.0, "specular_finish_strength": 0.0,
    "specular_recolor": 0.0, "skin_sss": 0.0, "texture_transplant": 0.0,
    "wrinkle_soften": 0.0, "wrinkle_soften_forehead": 0.0,
    "wrinkle_soften_nasolabial": 0.0, "wrinkle_soften_neck": 0.0,
    "undereye_shadow_strength": 0.0, "dark_circles": 0.0,
    "undereye_darken_removal": 0.0, "undereye_puffiness_reduction": 0.0,
    "face_polish": 0.0,
    # light and shape of the face
    "relight": 0.0, "sculpt": 0.0, "face_exposure": 0.0, "dodge_burn": 0.0,
    "shadow_lift": 0.0, "face_shadow_fill": 0.0, "nose_highlight": 0.0,
    "clarity": 0.0,
    # eyes, teeth, lips, makeup
    "eye_enhance": 0.0, "catchlight": 0.0, "eye_sclera_vessel_remove": 0.0,
    "corneal_shading": 0.0, "eye_sclera_brighten": 0.0,
    "eye_iris_saturate": 0.0, "eye_iris_hue_shift": 0.0,
    "eye_iris_brightness": 0.0, "iris_pop": 0.0, "teeth_whiten": 0.0,
    "teeth_brighten": 0.0, "lip_enhance": 0.0, "blush": 0.0,
    "nose_tip_blush": 0.0, "mv2_eyeshadow": 0.0, "mv2_eyeliner": 0.0,
    "mv2_contour": 0.0, "mv2_brows": 0.0,
    # hair in the face crop
    "hair_enhance": 0.0, "hair_deglare": 0.0, "hair_remove_flyaways": 0.0,
    "flyaway_cleanup": 0.0,
    # face reshape (read by the reshape stage per face)
    "slimming": 0.0, "reshape_eye_size": 0.0, "reshape_eye_distance": 0.0,
    "reshape_nose_width": 0.0, "reshape_nose_length": 0.0,
    "reshape_jaw_width": 0.0, "reshape_chin_length": 0.0,
    "reshape_mouth_size": 0.0, "reshape_smile": 0.0, "reshape_forehead": 0.0,
    "reshape_jaw_width_l": 0.0, "reshape_jaw_width_r": 0.0,
    "reshape_nose_width_l": 0.0, "reshape_nose_width_r": 0.0,
    "reshape_eye_size_l": 0.0, "reshape_eye_size_r": 0.0,
    "reshape_neck_width": 0.0, "reshape_neck_length": 0.0,
}

# On/off switches: kept as set while the face is above 0%, off at 0%.
SWITCHES: FrozenSet[str] = frozenset({"mv2_ombre"})

# Read by the core, deliberately not scaled, and why.
NOT_SCALED: Dict[str, str] = {
    "mv2_ombre": "on/off switch (off at 0%, see SWITCHES)",
    "nose_blush": "sub-option of blush, which scales",
    "under_eye_blush": "sub-option of blush, which scales",
    "nose_restore": "puts back original nose pixels in proportion to the (scaled) edits",
    "nose_shape": "puts back nose shading in proportion to the (scaled) smoothing",
    "smooth_exposure_lock": "compensates the (scaled) smoothing's darkening",
    "texture_opacity": "how smoothing keeps texture, not how much it smooths",
    "regional_modulation": "how smoothing varies over the face",
    "mole_protect": "a protection; never weakened",
    "mark_policy": "a protection; never weakened",
    "eye_gate": "a safety gate",
    "safe_auto": "a safety gate",
    "fa02_texture_mode": "which texture method",
    "smooth_engine": "which smoothing method",
    "heal_engine": "which healing method",
    "specular_finish": "which finish (its strength scales)",
    "specular_bloom_tone": "colour of the bloom",
    "whiten_tone": "colour of the whitening",
    "whiten_hue_stable": "how whitening treats hue",
    "skin_locus": "target skin hue line",
    "skin_unify_hue": "target hue (its amount, skin_unify, scales)",
    "relight_azimuth": "light direction",
    "relight_elevation": "light direction",
    "lip_tint": "lip colour",
    "lip_finish": "lip finish type",
    "hair_ring_position": "where the angel ring sits",
    "hair_ring_tint": "angel ring colour mix",
    "catchlight_synthetic": "how catchlights are made",
    "mv2_eyeshadow_color": "makeup colour",
    "mv2_eyeshadow_style": "makeup style",
    "mv2_eyeliner_color": "makeup colour",
    "mv2_eyeliner_style": "makeup style",
    "mv2_brows_color": "makeup colour",
    "mv2_ombre_color1": "makeup colour",
    "mv2_ombre_color2": "makeup colour",
    "neck_tone_match": "a whole-frame stage; the core only reads it to skip the auto neck pass",
}


def _spec_ranges() -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    from .params import PROCESSING_PARAMS

    return {p.name: (p.min_val, p.max_val) for p in PROCESSING_PARAMS}


def clamp_factor(value: Any, default: float = 1.0) -> float:
    """A strength factor (1.0 = 100%) from a 0-150 percent value."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v:  # NaN
        return default
    return min(max(v / 100.0, 0.0), MAX_FACTOR)


def _specular_active(ctx: Any) -> bool:
    # The core re-renders the finish only off its default ("matte", 0.5, 0);
    # that default is a no-op stand-in, not an amount to scale.
    return (
        getattr(ctx, "specular_finish", "matte") != "matte"
        or abs(float(getattr(ctx, "specular_finish_strength", 0.5) or 0.0) - 0.5) > 1e-6
        or float(getattr(ctx, "specular_recolor", 0.0) or 0.0) != 0.0
    )


def scale_context(ctx: Any, factor: float) -> Any:
    """Copy of ``ctx`` with every face-local amount scaled by ``factor``.

    ``factor == 1`` returns ``ctx`` itself (bit-identical renders).
    """
    if abs(factor - 1.0) <= 1e-12:
        return ctx
    factor = min(max(float(factor), 0.0), MAX_FACTOR)
    ranges = _spec_ranges()
    out = copy.copy(ctx)
    specular = _specular_active(ctx)
    for name, neutral in SCALED.items():
        if name in ("specular_finish_strength", "specular_recolor") and not specular:
            continue
        value = getattr(ctx, name, None)
        if value is None or isinstance(value, (bool, str)):
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        scaled = neutral + factor * (v - neutral)
        lo, hi = ranges.get(name, (None, None))
        if lo is not None:
            scaled = max(scaled, float(lo))
        if hi is not None:
            scaled = min(scaled, float(hi))
        setattr(out, name, scaled)
    if factor <= 0.0:
        for name in SWITCHES:
            if getattr(ctx, name, None):
                setattr(out, name, False)
    return out


def core_param_reads() -> FrozenSet[str]:
    """Settings the per-face core and per-face reshape read, from their source.

    Scans ``perf_optimizations._process_face_core`` and ``_face_polish_ctx``
    for ``ctx.NAME`` / ``getattr(ctx, "NAME")`` / ``_ctx_get(ctx, "NAME")``,
    plus the reshape settings ``RetouchEngine._any_reshape_active`` checks
    (reshape runs per face with that face's context). Private (``_``) names
    are left out. Used by the tests; needs the source files.
    """
    from . import perf_optimizations
    from .engine import RetouchEngine

    names = set()
    tree = ast.parse(inspect.getsource(perf_optimizations))
    for fn in tree.body:
        if not (isinstance(fn, ast.FunctionDef)
                and fn.name in ("_process_face_core", "_face_polish_ctx")):
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "ctx"):
                names.add(node.attr)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id in ("getattr", "_ctx_get") and len(node.args) >= 2
                    and isinstance(node.args[0], ast.Name) and node.args[0].id == "ctx"
                    and isinstance(node.args[1], ast.Constant)):
                names.add(node.args[1].value)

    reshape_src = textwrap.dedent(inspect.getsource(RetouchEngine._any_reshape_active))
    for node in ast.walk(ast.parse(reshape_src)):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("reshape_"):
                names.add(node.value)
        elif isinstance(node, ast.Attribute) and node.attr == "slimming":
            names.add(node.attr)
    return frozenset(n for n in names if not n.startswith("_"))
