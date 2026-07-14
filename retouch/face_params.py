"""Per-face recipe / param overrides (Slice 1).

Merge face-local fields onto a base ProcessingContext. Image-global grade/WB/body
never leak from a face recipe expand.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

from .params import resolve_recipe

_log = logging.getLogger(__name__)

# Explicit allowlist — new globals stay global by default.
FACE_LOCAL_PARAM_NAMES: frozenset[str] = frozenset({
    "smooth", "whiten", "whiten_tone", "equalize", "blemish", "nose_smooth",
    "micro_restore", "micro_dodge_burn", "mid_reduction", "blotch_reduction",
    "texture_opacity", "pore_synthesis", "regional_modulation", "smooth_engine",
    "specular_bloom", "specular_bloom_tone", "specular_finish",
    "specular_finish_strength", "specular_recolor", "albedo_even",
    "hemoglobin_smooth", "mole_protect", "vein_attenuate", "dodge_burn",
    "relight", "relight_azimuth", "relight_elevation", "face_exposure", "sculpt",
    "skin_flatten", "skin_quantize", "skin_unify", "skin_unify_hue",
    "skin_hue_unify", "skin_chroma_even", "redness_even", "skin_glow",
    "whiten_hue_stable", "skin_locus", "smooth_exposure_lock", "shine_removal",
    "wrinkle_soften", "wrinkle_soften_forehead", "wrinkle_soften_nasolabial",
    "wrinkle_soften_neck", "texture_transplant", "shadow_lift", "nose_restore",
    "freckle_removal",
    "eye_enhance", "eye_sclera_vessel_remove", "dark_circles",
    "undereye_darken_removal", "undereye_puffiness_reduction",
    "undereye_shadow_strength", "catchlight", "eye_sclera_brighten",
    "eye_iris_saturate", "eye_iris_hue_shift", "eye_iris_brightness",
    "lip_enhance", "lip_tint", "lip_finish", "teeth_whiten",
    "blush", "nose_blush", "under_eye_blush",
    "hair_enhance", "hair_deglare", "hair_ring_position", "hair_ring_tint",
    "hair_remove_flyaways",
    "slimming",
    "reshape_eye_size", "reshape_eye_distance", "reshape_nose_width",
    "reshape_nose_length", "reshape_jaw_width", "reshape_chin_length",
    "reshape_mouth_size", "reshape_smile", "reshape_forehead",
    "reshape_jaw_width_l", "reshape_jaw_width_r", "reshape_nose_width_l",
    "reshape_nose_width_r", "reshape_eye_size_l", "reshape_eye_size_r",
    "reshape_neck_width", "reshape_neck_length",
    "mv2_eyeshadow", "mv2_eyeshadow_color", "mv2_eyeshadow_style",
    "mv2_eyeliner", "mv2_eyeliner_color", "mv2_eyeliner_style",
    "mv2_contour", "mv2_brows", "mv2_brows_color",
    "mv2_ombre", "mv2_ombre_color1", "mv2_ombre_color2",
    "cosplay_wig_lace_blend", "cosplay_stockings_smooth",
    "cosplay_consistency_strength",
})


def filter_face_local(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only FACE_LOCAL keys; drop recipe + globals (log once per key)."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if k == "recipe":
            continue
        if k in FACE_LOCAL_PARAM_NAMES:
            out[k] = v
        else:
            _log.warning("face override ignoring non-face-local key %r", k)
    return out


def coerce_face_params(
    face_params: Optional[Mapping[Any, Mapping[str, Any]]],
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Normalize keys to int; return None if empty."""
    if not face_params:
        return None
    out: Dict[int, Dict[str, Any]] = {}
    for k, v in face_params.items():
        if v is None:
            continue
        out[int(k)] = dict(v)
    return out or None


def load_face_params_json(path: Union[str, Path]) -> Dict[int, Dict[str, Any]]:
    """Load faces.json: {\"0\": {\"recipe\": \"cosplay\", \"smooth\": 70}, ...}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("face_params JSON must be an object keyed by face index")
    coerced = coerce_face_params(data)
    return coerced or {}


def resolve_face_context(base: Any, raw: Optional[Mapping[str, Any]]) -> Any:
    """Merge face recipe + explicit overrides onto base. FACE_LOCAL only.

    When raw is empty/None, returns the same base object (golden path identity).
    """
    if not raw:
        return base

    from .engine import build_context  # lazy: avoid import cycle

    explicit = filter_face_local(raw)
    recipe_name = raw.get("recipe")

    if recipe_name:
        rec = resolve_recipe(str(recipe_name))
        face_full = build_context(str(recipe_name), rec, explicit)
        patch = {
            name: getattr(face_full, name)
            for name in FACE_LOCAL_PARAM_NAMES
            if hasattr(face_full, name)
        }
        patch.update(explicit)
        return dataclasses.replace(base, **patch)

    if not explicit:
        return base
    return dataclasses.replace(base, **explicit)
