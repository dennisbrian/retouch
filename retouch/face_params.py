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

import cv2
import numpy as np

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
        return dataclasses.replace(base, **patch)

    if not explicit:
        return base
    return dataclasses.replace(base, **explicit)


def suggest_face_recipe(
    img_bgr: np.ndarray,
    landmarks: Any,
    bbox: Tuple[int, int, int, int],
    ied: float,
) -> str:
    """Auto-suggests a recipe name ('child', 'senior', 'male', 'female') based on demographics heuristics."""
    h_img, w_img = img_bgr.shape[:2]

    # 1. Fetch key landmarks
    indices = [10, 151, 168, 152, 13, 117, 346, 70, 300]
    pts = {}
    try:
        for idx in indices:
            lm = landmarks.landmark[idx]
            pts[idx] = (int(lm.x * w_img), int(lm.y * h_img))
    except (IndexError, AttributeError):
        return "female"

    x_face, y_face, w_face, h_face = bbox
    if w_face <= 0 or h_face <= 0:
        return "female"

    # 2. Child heuristic: eyes lower on head & rounder face
    y_top = pts[10][1]
    y_mid = pts[168][1]
    y_bottom = pts[152][1]

    top_dist = max(y_mid - y_top, 1.0)
    bottom_dist = max(y_bottom - y_mid, 1.0)
    bottom_ratio = bottom_dist / top_dist

    aspect_ratio = h_face / w_face
    eye_width_ratio = ied / w_face

    is_child = (bottom_ratio < 0.96 and aspect_ratio < 1.22) or (eye_width_ratio > 0.43)
    if is_child:
        return "child"

    # 3. Senior heuristic: wrinkle detection (high gradient variance on forehead / crows feet)
    fh_x, fh_y = pts[151]
    r_patch = max(int(ied * 0.25), 4)
    x1 = max(fh_x - r_patch, 0)
    y1 = max(fh_y - r_patch, 0)
    x2 = min(fh_x + r_patch, w_img)
    y2 = min(fh_y + r_patch, h_img)

    if (x2 - x1) >= 8 and (y2 - y1) >= 8:
        patch = img_bgr[y1:y2, x1:x2]
        patch_resized = cv2.resize(patch, (32, 32), interpolation=cv2.INTER_LINEAR)
        gray_patch = cv2.cvtColor(patch_resized, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(gray_patch, cv2.CV_32F)
        texture_std = float(np.std(lap))
        if texture_std > 12.0:
            return "senior"

    # 4. Male vs Female heuristic
    chin_x, chin_y = pts[152]
    # Go slightly above chin 152 to be in the beard area
    beard_y = int(0.7 * pts[152][1] + 0.3 * pts[13][1])
    beard_x = chin_x

    def sample_lab_mean(cx: int, cy: int) -> Tuple[float, float, float]:
        x_start = max(cx - 2, 0)
        y_start = max(cy - 2, 0)
        x_end = min(cx + 3, w_img)
        y_end = min(cy + 3, h_img)
        p = img_bgr[y_start:y_end, x_start:x_end]
        if p.size == 0:
            return 128.0, 128.0, 128.0
        p_rgb = cv2.cvtColor(p, cv2.COLOR_BGR2RGB)
        p_lab = cv2.cvtColor(p_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        return float(p_lab[..., 0].mean()), float(p_lab[..., 1].mean()), float(p_lab[..., 2].mean())

    L_fh, a_fh, b_fh = sample_lab_mean(fh_x, fh_y)
    L_bd, a_bd, b_bd = sample_lab_mean(beard_x, beard_y)

    # Check lip redness/contrast
    lip_x, lip_y = pts[13]
    L_lip, a_lip, b_lip = sample_lab_mean(lip_x, lip_y)

    lip_redness = a_lip - a_fh

    # Beard shadow: lower face is darker and cooler (less yellow/red, more blue/green)
    is_male = (lip_redness < 4.0) and ((L_fh - L_bd > 5.0) or (b_fh - b_bd > 3.0))
    if is_male:
        return "male"

    return "female"
