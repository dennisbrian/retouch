"""Per-photo "what was changed" report.

Each retouched photo can get a small JSON report that says, in plain terms,
what the app did to it: which recipe ran, which edits were active (grouped
into skin, face/body shape, makeup, eyes, colour and so on), whether any face
or body *shape* was changed, whether an AI model touched pixels, how much of
the frame actually changed, and what happened to the source photo's own
Content Credentials.

The report exists for disclosure: several countries and platforms ask for
retouched or reshaped images to be labelled, and a client or agency may ask
what was done to a photo. It is also the payload of the optional signed
Content Credentials (see :mod:`retouch.content_credentials`).

The report lists *settings that were active*, measured from the resolved
``ProcessingContext`` the engine actually ran with, not from the command line,
so recipe values are included. It does not claim an edit was visible: an
active slimming setting on a face turned too far sideways may still be gated
to zero by the engine. ``pixels`` gives the measured change as a cross-check.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA = "retouch.edit_report/1"
REPORT_DIRNAME = "edit-reports"

# Category ids, in the order the summary lists them. The label is what a
# person reads; the id is stable for anything parsing the JSON.
CATEGORIES = (
    ("shape", "Face and body shape"),
    ("skin", "Skin"),
    ("makeup", "Makeup"),
    ("eyes_teeth", "Eyes, teeth and glasses"),
    ("hair_costume", "Hair, wig and costume"),
    ("light", "Relighting"),
    ("background", "Background"),
    ("ai", "AI models"),
    ("colour", "Colour, tone and finish"),
)
_CATEGORY_LABELS = dict(CATEGORIES)

# Recipe-key roots (first segment of the ParamSpec recipe key) per category.
_ROOT_CATEGORY = {
    "reshape": "shape",
    "body_reshape": "shape",
    "slimming": "shape",
    "frequency": "skin",
    "skin": "skin",
    "texture": "skin",
    "body_skin": "skin",
    "undereye": "skin",
    "dodge_burn": "skin",
    "micro_restore": "skin",
    "makeup_v2": "makeup",
    "blush": "makeup",
    "lips": "makeup",
    "lip_finish": "makeup",
    "nose_blush": "makeup",
    "under_eye_blush": "makeup",
    "eye": "eyes_teeth",
    "eyes": "eyes_teeth",
    "glasses": "eyes_teeth",
    "hair": "hair_costume",
    "cosplay": "hair_costume",
    "fabric": "hair_costume",
    "white_costume_lift": "hair_costume",
    "relight_azimuth": "light",
    "relight_elevation": "light",
    "background": "background",
    "harmony": "background",
    "subject_separation": "background",
    "ai": "ai",
    "neural": "ai",
}
_NAME_CATEGORY = {
    "blemish": "skin",
    "cross_region_skin": "skin",
    "auto_body_reshape": "shape",
    "ai_sr_scale": "ai",
    "relight": "light",
}

# Settings that tune *how* another edit behaves (its engine, colour, radius,
# mix) rather than being an edit themselves. They never appear on their own.
_MODIFIERS = frozenset({
    "mid_reduction", "texture_opacity", "regional_modulation", "smooth_engine",
    "heal_engine", "mark_policy", "fa02_texture_mode", "freckle_preserve_mask",
    "whiten_tone", "relight_elevation", "relight_azimuth", "specular_bloom_tone",
    "specular_finish", "specular_finish_strength", "skin_unify_hue",
    "skin_protect_strength", "mask_feather_mode", "eye_gate", "lip_finish",
    "hair_ring_position", "hair_ring_tint", "bloom_threshold", "bloom_softness",
    "sharpen_radius", "color_transfer_intensity", "grade_intensity",
    "gamut_compress", "gamut_target", "saturation_mode",
    "background_harmonize_mode", "multi_illuminant_key_kelvin",
    "multi_illuminant_fill_kelvin", "multi_illuminant_mix",
    "bw_channel_mixer_r", "bw_channel_mixer_g", "bw_channel_mixer_b",
    "tonal_curve_strength", "neural_stray_hair_boost", "neural_defect_boost",
})
_MODIFIER_PREFIXES = ("film_", "mv2_eyeshadow_", "mv2_eyeliner_", "mv2_brows_", "mv2_ombre_")
_EFFECT_OVERRIDES = frozenset({"film_enable"})

# Settings whose "does nothing" value is not zero.
_NEUTRAL = {
    "body_reshape_arm_length": 50.0,
    "body_reshape_leg_length": 50.0,
    "body_reshape_torso_width": 50.0,
    "body_reshape_shoulder_width": 50.0,
    "body_reshape_hip_width": 50.0,
    "white_balance_kelvin": 6500,
    "ai_sr_scale": 1,
}
_NEUTRAL_STRINGS = frozenset({"", "none", "off", "false", "0"})

# Model-backed edits. None of them generate new image content; they are
# listed because some disclosure rules ask about any AI use.
_AI_DESCRIPTIONS = {
    "ai_denoise": "NAFNet noise reduction (restores existing detail; generates no new content)",
    "ai_sr_scale": "Super-resolution upscaling",
    "neural_stray_hair_boost": "Neural stray-hair detection",
    "neural_defect_boost": "Neural blemish detection",
}


def _is_modifier(name: str) -> bool:
    if name in _EFFECT_OVERRIDES:
        return False
    return name in _MODIFIERS or name.startswith(_MODIFIER_PREFIXES)


def _is_active(name: str, value: Any) -> bool:
    """True when *value* for setting *name* means the edit does something."""
    if value is None or _is_modifier(name):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, np.integer, np.floating)):
        neutral = _NEUTRAL.get(name, 0)
        return abs(float(value) - float(neutral)) > 1e-9
    if isinstance(value, str):
        return value.strip().lower() not in _NEUTRAL_STRINGS
    return False


def _category_for(spec: Any) -> str:
    if spec.name in _NAME_CATEGORY:
        return _NAME_CATEGORY[spec.name]
    key = spec.engine_recipe_key or spec.recipe_key or spec.name
    return _ROOT_CATEGORY.get(key.split(".")[0], "colour")


def _clean_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        rounded = round(float(value), 3)
        return int(rounded) if rounded.is_integer() else rounded
    return value


def _params_dict(params: Any) -> Dict[str, Any]:
    if params is None:
        return {}
    if dataclasses.is_dataclass(params) and not isinstance(params, type):
        return {f.name: getattr(params, f.name) for f in dataclasses.fields(params)}
    return dict(params)


def active_edits(params: Any) -> Dict[str, Dict[str, Any]]:
    """Group the active edit settings in *params* by category id.

    *params* is the engine's resolved ``ProcessingContext`` (or a dict of the
    same names). Only registered ``ParamSpec`` settings are considered.
    """
    from .params import PROCESSING_PARAMS

    values = _params_dict(params)
    grouped: Dict[str, Dict[str, Any]] = {}
    for spec in PROCESSING_PARAMS:
        value = values.get(spec.name)
        if not _is_active(spec.name, value):
            continue
        grouped.setdefault(_category_for(spec), {})[spec.name] = _clean_value(value)
    return {cat: grouped[cat] for cat, _ in CATEGORIES if cat in grouped}


def _is_face_only(category: str, name: str) -> bool:
    if category in ("makeup", "eyes_teeth"):
        return True
    if category == "shape":
        return name == "slimming" or name.startswith("reshape_")
    if category == "skin":
        return not name.startswith("body_")
    return False


def _split_face_only(edits: Mapping[str, Mapping[str, Any]]):
    """Split *edits* into (applied, face-only settings that could not run)."""
    applied: Dict[str, Dict[str, Any]] = {}
    skipped: Dict[str, Any] = {}
    for cat, settings in edits.items():
        for name, value in settings.items():
            if _is_face_only(cat, name):
                skipped[name] = value
            else:
                applied.setdefault(cat, {})[name] = value
    return applied, skipped


def pixel_change(before: Optional[np.ndarray], after: Optional[np.ndarray],
                 max_dim: int = 1024, threshold: int = 3) -> Optional[Dict[str, float]]:
    """Measure how much of the frame changed between *before* and *after*.

    Both are compared at up to *max_dim* on the long side. Returns ``None``
    when either is missing or their aspect ratios differ (a crop or rotate,
    where a per-pixel comparison means nothing).
    """
    if before is None or after is None:
        return None
    import cv2

    b = np.asarray(before)
    a = np.asarray(after)
    if b.ndim != 3 or a.ndim != 3:
        return None
    bh, bw = b.shape[:2]
    ah, aw = a.shape[:2]
    if abs(bh / bw - ah / aw) > 0.01:
        return None
    scale = min(1.0, max_dim / max(bh, bw))
    size = (max(1, int(round(bw * scale))), max(1, int(round(bh * scale))))

    def _prep(img: np.ndarray) -> np.ndarray:
        img = img[..., :3]
        if img.dtype == np.uint16:
            img = img.astype(np.float32) / 257.0
        img = np.clip(img.astype(np.float32), 0, 255)
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    diff = np.abs(_prep(a) - _prep(b)).max(axis=2)
    return {
        "changed_pct": round(float((diff > threshold).mean() * 100.0), 2),
        "mean_change": round(float(diff.mean()), 2),
        "threshold_levels": threshold,
    }


def sha256_file(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(edits: Mapping[str, Mapping[str, Any]], face_count: int,
             ai_used: Iterable[str]) -> List[str]:
    lines = []
    if "shape" in edits:
        lines.append(
            "Face or body shape was changed ("
            + ", ".join(f"{k} {v}" for k, v in edits["shape"].items()) + ")."
        )
    else:
        lines.append("No face or body shape change.")
    for cat, label in CATEGORIES:
        if cat in ("shape", "ai") or cat not in edits:
            continue
        lines.append(f"{label}: " + ", ".join(f"{k} {v}" for k, v in edits[cat].items()) + ".")
    ai_used = list(ai_used)
    if ai_used:
        lines.append("AI models used: " + "; ".join(ai_used) + ".")
    lines.append("No generative AI: no image content was synthesised by a generative model.")
    if face_count == 0:
        lines.append("No face was detected, so face edits in the recipe did not run.")
    return lines


def build_edit_report(
    params: Any,
    *,
    source_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    recipe: Optional[str] = None,
    face_count: Optional[int] = None,
    before: Optional[np.ndarray] = None,
    after: Optional[np.ndarray] = None,
    global_only: bool = False,
    source_had_credentials: bool = False,
    signed: bool = False,
    include_output_hash: bool = True,
) -> Dict[str, Any]:
    """Build the "what was changed" report for one photo.

    ``params`` is the resolved ``ProcessingContext`` (``result.params``) or a
    dict of setting names. ``before``/``after`` are the source and retouched
    pixels (any size; compared downscaled). ``include_output_hash`` is off
    when the report is going *inside* the output (signed credentials), where
    the file's own hash cannot be known yet.
    """
    from . import __version__

    values = _params_dict(params)
    edits = active_edits(values)
    if global_only:
        edits = {k: v for k, v in edits.items() if k not in ("shape", "skin", "makeup", "eyes_teeth")}
    if recipe is None:
        recipe = values.get("active_recipe") or values.get("recipe")
    faces = int(face_count or 0)
    not_applied: Dict[str, Any] = {}
    if face_count is not None and faces == 0:
        edits, not_applied = _split_face_only(edits)
    ai_used = [
        _AI_DESCRIPTIONS.get(name, name)
        for name in edits.get("ai", {})
    ]

    report: Dict[str, Any] = {
        "schema": SCHEMA,
        "app": {"name": "retouch", "version": __version__},
        "created": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "recipe": recipe,
        "faces_detected": faces,
        "shape_changed": "shape" in edits,
        "ai_used": ai_used,
        "generative_fill": False,
        "edits": {
            cat: {"label": _CATEGORY_LABELS[cat], "settings": dict(settings)}
            for cat, settings in edits.items()
        },
        "summary": _summary(edits, faces, ai_used),
    }
    if not_applied:
        report["not_applied_no_face"] = not_applied
    change = pixel_change(before, after)
    if change is not None:
        report["pixels"] = change
    if source_path is not None:
        src = Path(source_path)
        report["source"] = {"file": src.name}
        try:
            report["source"]["sha256"] = sha256_file(src)
        except OSError as exc:
            logger.debug("Could not hash %s: %s", src, exc)
    if output_path is not None:
        out = Path(output_path)
        report["output"] = {"file": out.name}
        if include_output_hash:
            try:
                report["output"]["sha256"] = sha256_file(out)
            except OSError as exc:
                logger.debug("Could not hash %s: %s", out, exc)
    report["content_credentials"] = {
        "source_had_credentials": bool(source_had_credentials),
        "signed": bool(signed),
        "note": credentials_note(source_had_credentials, signed),
    }
    return report


def credentials_note(source_had_credentials: bool, signed: bool) -> str:
    """One plain sentence on what happened to Content Credentials."""
    if signed and source_had_credentials:
        return ("Signed with new Content Credentials that record these edits and "
                "keep the camera's original credentials as the parent.")
    if signed:
        return "Signed with new Content Credentials that record these edits."
    if source_had_credentials:
        return ("The source photo had Content Credentials. They were not copied to the "
                "retouched file, because they describe the unedited pixels and would "
                "fail verification. Sign the output to carry them forward.")
    return "No Content Credentials in the source; the output is not signed."


def report_path_for(output_path: Union[str, Path], report_dir: Optional[Union[str, Path]] = None) -> Path:
    """Where the report for *output_path* goes: ``<dir>/edit-reports/<file>.json``."""
    out = Path(output_path)
    base = Path(report_dir) if report_dir is not None else out.parent / REPORT_DIRNAME
    return base / f"{out.name}.json"


def write_edit_report(report: Mapping[str, Any], output_path: Union[str, Path],
                      report_dir: Optional[Union[str, Path]] = None) -> Path:
    """Write *report* as JSON next to the output and return its path."""
    target = report_path_for(output_path, report_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.tmp")
    tmp.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target
