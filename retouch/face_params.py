"""Per-face recipe / param overrides (Slice 1).

Merge face-local fields onto a base ProcessingContext. Image-global grade/WB/body
never leak from a face recipe expand.
"""

from __future__ import annotations

import dataclasses
import json
import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union, Tuple

from .params import resolve_recipe

_log = logging.getLogger(__name__)

# Explicit allowlist — new globals stay global by default.
FACE_LOCAL_PARAM_NAMES: frozenset[str] = frozenset({
    "smooth", "whiten", "whiten_tone", "equalize", "blemish", "nose_smooth",
    "micro_restore", "micro_dodge_burn", "mid_reduction", "blotch_reduction",
    "texture_opacity", "pore_synthesis", "regional_modulation", "smooth_engine",
    "specular_bloom", "specular_bloom_tone", "specular_finish",
    "specular_finish_strength", "specular_recolor", "albedo_even",
    "makeup_coverage_even", "makeup_cake_reduce",
    "hemoglobin_smooth", "mole_protect", "vein_attenuate", "dodge_burn",
    "relight", "relight_azimuth", "relight_elevation", "face_exposure", "sculpt",
    "skin_flatten", "skin_quantize", "skin_unify", "skin_unify_hue",
    "skin_hue_unify", "skin_chroma_even", "redness_even", "skin_glow",
    "whiten_hue_stable", "skin_locus", "smooth_exposure_lock", "shine_removal",
    "wrinkle_soften", "wrinkle_soften_forehead", "wrinkle_soften_nasolabial",
    "wrinkle_soften_neck", "texture_transplant", "shadow_lift", "nose_restore",
    "skin_sss", "freckle_removal",
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


#: Optional entry key: the selected face's bbox as ``[x, y, w, h]`` fractions of
#: the image it was selected on. Lets the engine bind the entry to the same
#: person even when its own detection (proxy/fast scale) orders faces
#: differently. Entries without it keep plain index semantics.
ANCHOR_KEY = "anchor_bbox_norm"
#: Optional entry key: source path the anchor was measured on.
ANCHOR_SOURCE_KEY = "anchor_source"
#: Minimum IoU between an anchor and a detected face to accept the binding.
ANCHOR_MIN_IOU = 0.3


def make_anchor(bbox: Tuple[int, int, int, int], frame_size: Tuple[int, int]) -> list:
    """Normalize a pixel ``(x, y, w, h)`` bbox by ``(width, height)``."""
    fw, fh = float(frame_size[0]), float(frame_size[1])
    x, y, w, h = bbox
    return [x / fw, y / fh, w / fw, h / fh]


def _iou_norm(a, b) -> float:
    ix = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def bind_face_params(
    face_params: Optional[Dict[int, Dict[str, Any]]],
    face_bboxes,
    frame_size: Tuple[int, int],
) -> Tuple[Optional[Dict[int, Dict[str, Any]]], list]:
    """Re-key anchored entries to the engine's own detection indices.

    Anchored entries are matched one-to-one to detected faces by IoU (best
    pairs first, ``ANCHOR_MIN_IOU`` minimum). An anchored entry that matches
    nothing is dropped rather than applied to whichever face happens to hold
    its old index. Unanchored entries keep their index unless that index was
    claimed by an anchored match. Returns ``(bound, report)``; ``report`` has
    one dict per anchored entry.
    """
    if not face_params or not isinstance(face_params, dict):
        return face_params, []
    anchored = {k: v for k, v in face_params.items() if v.get(ANCHOR_KEY) is not None}
    if not anchored:
        return face_params, []

    detected = [make_anchor(tuple(b), frame_size) for b in face_bboxes]
    pairs = sorted(
        (
            (_iou_norm([float(x) for x in entry[ANCHOR_KEY]], det), key, j)
            for key, entry in anchored.items()
            for j, det in enumerate(detected)
        ),
        reverse=True,
    )
    bound: Dict[int, Dict[str, Any]] = {}
    report = []
    used_keys, used_faces = set(), set()
    for iou, key, j in pairs:
        if iou < ANCHOR_MIN_IOU or key in used_keys or j in used_faces:
            continue
        used_keys.add(key)
        used_faces.add(j)
        bound[j] = dict(anchored[key])
        report.append({"selected_index": key, "bound_index": j, "iou": round(iou, 3)})
    for key in anchored:
        if key not in used_keys:
            _log.warning(
                "face override for selected face %s matched no detected face "
                "(IoU < %.2f); not applied", key, ANCHOR_MIN_IOU,
            )
            report.append({"selected_index": key, "bound_index": None, "iou": None})
    for key, entry in face_params.items():
        if key not in anchored and key not in bound:
            bound[key] = entry
    return (bound or None), report


def face_params_for_source(
    face_params: Optional[Dict[int, Dict[str, Any]]],
    source: Optional[str],
) -> Optional[Dict[int, Dict[str, Any]]]:
    """Drop anchors measured on a different source image.

    A position in one photo says nothing about who sits there in another, so
    for other images the entry falls back to plain index semantics.
    """
    if not face_params or not isinstance(face_params, dict):
        return face_params
    out: Dict[int, Dict[str, Any]] = {}
    for key, entry in face_params.items():
        anchor_source = entry.get(ANCHOR_SOURCE_KEY)
        if anchor_source is not None and anchor_source != source:
            entry = {
                k: v for k, v in entry.items()
                if k not in (ANCHOR_KEY, ANCHOR_SOURCE_KEY)
            }
        out[key] = entry
    return out


def filter_face_local(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only FACE_LOCAL keys; drop recipe + globals (log once per key)."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if k in {"recipe", "tone_observation", "selection_reason", ANCHOR_KEY, ANCHOR_SOURCE_KEY}:
            continue
        if k in FACE_LOCAL_PARAM_NAMES:
            out[k] = v
        else:
            _log.warning("face override ignoring non-face-local key %r", k)
    return out


def coerce_face_params(
    face_params: Optional[Union[Mapping[Any, Mapping[str, Any]], str]],
) -> Optional[Union[Dict[int, Dict[str, Any]], str]]:
    """Normalize keys to int; return None if empty."""
    if not face_params:
        return None
    if isinstance(face_params, str) and face_params.lower() == "auto":
        return "auto"
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
        resolved = dataclasses.replace(base, **patch)
        provenance = dict(getattr(base, "_parameter_provenance", {}) or {})
        provenance["per_face_recipe"] = str(recipe_name)
        provenance["per_face_explicit_overrides"] = sorted(str(name) for name in explicit)
        resolved._parameter_provenance = provenance
        return resolved

    if not explicit:
        return base
    resolved = dataclasses.replace(base, **explicit)
    provenance = dict(getattr(base, "_parameter_provenance", {}) or {})
    provenance["per_face_explicit_overrides"] = sorted(str(name) for name in explicit)
    resolved._parameter_provenance = provenance
    return resolved


def suggest_face_recipe(
    img_bgr: np.ndarray,
    landmarks: Any,
    bbox: Tuple[int, int, int, int],
    ied: float,
) -> str:
    """Return a neutral suggestion instead of inferring demographics.

    The old implementation guessed child/senior/male/female from geometry,
    texture, and color.  Those signals are not reliable identity attributes or
    treatment preferences, so automatic selection is now deliberately neutral.
    The arguments remain for API compatibility; explicit user-selected recipes
    still work through :func:`resolve_face_context`.
    """
    return "natural"
