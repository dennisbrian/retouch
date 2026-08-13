"""Interactive Advanced Retouch operations.

This module is deliberately independent of Gradio.  The GUI owns the editor
state, while this module owns the image contract and delegates all pixel math
to the existing healing, region, parsing, and geometry implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from .geometry import FaceReshaper
from .heal import heal_region, mask_to_b64, b64_to_mask
from .model_fetch import model_exists
from .regions import apply_local_adjustment
from .spot_heal import LamaHealer


ADVANCED_OPERATIONS = ("exposure", "dodge", "burn", "warmth", "saturation", "smooth", "clarity")
ADVANCED_SEMANTIC_MASKS = ("None", "Skin", "Hair", "Face", "Lips", "Eyes", "Clothing", "Person", "Background")
ADVANCED_HEAL_METHODS = ("telea", "ns", "patchmatch")
ADVANCED_REMOVE_ENGINES = ("Auto (LaMa if installed)", "Telea fallback")

_RESHAPE_KEYS = (
    "eye_size",
    "eye_distance",
    "nose_width",
    "nose_length",
    "jaw_width",
    "chin_length",
    "mouth_size",
    "smile",
    "forehead",
    "eye_size_l",
    "eye_size_r",
    "nose_width_l",
    "nose_width_r",
    "jaw_width_l",
    "jaw_width_r",
)


@dataclass
class AdvancedEditResult:
    """Result of one non-destructive Advanced Retouch action."""

    image_rgb: np.ndarray
    mask: Optional[np.ndarray]
    status: str
    edit: Dict[str, Any]


def _coerce_rgb(image: Any) -> np.ndarray:
    """Return a contiguous uint8 RGB image or raise a useful ValueError."""
    if image is None:
        raise ValueError("No image is loaded.")
    if hasattr(image, "convert"):
        image = np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected an RGB/RGBA image, got shape {arr.shape}.")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        scale = 255.0 if float(np.nanmax(arr)) <= 1.0 + 1e-6 else 1.0
        arr = np.clip(arr.astype(np.float32) * scale, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _array_from_editor_part(part: Any) -> Optional[np.ndarray]:
    if part is None:
        return None
    if isinstance(part, Mapping):
        for key in ("image", "data", "background", "composite"):
            if key in part:
                found = _array_from_editor_part(part[key])
                if found is not None:
                    return found
        return None
    try:
        arr = np.asarray(part)
    except Exception:
        return None
    return arr if arr.ndim in (2, 3) else None


def _normalise_alpha(alpha: np.ndarray) -> np.ndarray:
    alpha = np.asarray(alpha)
    if alpha.ndim == 3:
        alpha = alpha[:, :, 0]
    if alpha.dtype == np.bool_:
        return alpha.astype(np.float32)
    alpha_f = alpha.astype(np.float32)
    if alpha_f.size and float(np.nanmax(alpha_f)) > 1.0 + 1e-6:
        alpha_f /= 255.0
    return np.clip(alpha_f, 0.0, 1.0)


def _editor_layer_parts(editor_value: Any) -> Tuple[Optional[np.ndarray], Iterable[Any]]:
    if not isinstance(editor_value, Mapping):
        arr = _array_from_editor_part(editor_value)
        return arr, ()
    background = _array_from_editor_part(editor_value.get("background"))
    layers = editor_value.get("layers") or ()
    if isinstance(layers, Mapping):
        layers = (layers,)
    return background, layers


def extract_editor_image_and_mask(
    editor_value: Any,
    fallback_rgb: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the current canvas and the painted alpha mask from Gradio.

    Gradio 4 returns an ``EditorValue`` mapping with ``background`` and
    ``layers``.  Older/compatibility payloads may be a direct ndarray, so the
    parser accepts both forms and never treats the opaque background as a
    painted mask.
    """
    fallback = _coerce_rgb(fallback_rgb) if fallback_rgb is not None else None
    background, layers = _editor_layer_parts(editor_value)
    base = _coerce_rgb(background) if background is not None and background.ndim == 3 else fallback
    direct = _array_from_editor_part(editor_value) if not isinstance(editor_value, Mapping) else None

    if base is None:
        if direct is None:
            raise ValueError("The Advanced Retouch canvas is empty.")
        base = _coerce_rgb(direct)
        if direct.ndim == 3 and direct.shape[2] >= 4:
            mask = _normalise_alpha(direct[:, :, 3])
        else:
            mask = np.zeros(base.shape[:2], dtype=np.float32)
    else:
        mask = np.zeros(base.shape[:2], dtype=np.float32)

    for layer in layers:
        arr = _array_from_editor_part(layer)
        if arr is None:
            continue
        if arr.ndim == 3 and arr.shape[2] >= 4:
            layer_mask = _normalise_alpha(arr[:, :, 3])
        elif arr.ndim == 2:
            layer_mask = _normalise_alpha(arr)
        else:
            # A non-alpha layer is not a brush mask.  Avoid interpreting a
            # fully opaque background or a preview composite as a selection.
            continue
        if layer_mask.shape != mask.shape:
            layer_mask = cv2.resize(layer_mask, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = np.maximum(mask, layer_mask.astype(np.float32))
    return base, np.clip(mask, 0.0, 1.0)


def mask_overlay(image_rgb: np.ndarray, mask: Optional[np.ndarray], opacity: float = 0.42) -> np.ndarray:
    """Render a red, soft mask overlay without changing the source image."""
    image = _coerce_rgb(image_rgb)
    if mask is None:
        return image
    m = np.asarray(mask, dtype=np.float32)
    if m.shape != image.shape[:2]:
        m = cv2.resize(m, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    m = np.clip(m, 0.0, 1.0) * float(np.clip(opacity, 0.0, 1.0))
    red = np.zeros_like(image)
    red[:, :, 0] = 255
    return np.clip(image.astype(np.float32) * (1.0 - m[:, :, None]) + red.astype(np.float32) * m[:, :, None], 0, 255).astype(np.uint8)


def before_after(original_rgb: np.ndarray, current_rgb: np.ndarray) -> np.ndarray:
    """Build a compact side-by-side RGB preview."""
    original = _coerce_rgb(original_rgb)
    current = _coerce_rgb(current_rgb)
    h = min(original.shape[0], current.shape[0])
    if original.shape[0] != h:
        original = cv2.resize(original, (int(original.shape[1] * h / original.shape[0]), h), interpolation=cv2.INTER_AREA)
    if current.shape[0] != h:
        current = cv2.resize(current, (int(current.shape[1] * h / current.shape[0]), h), interpolation=cv2.INTER_AREA)
    sep = np.full((h, 4, 3), 200, dtype=np.uint8)
    result = np.hstack((original[:h], sep, current[:h]))
    if result.shape[0] > 900:
        scale = 900.0 / result.shape[0]
        result = cv2.resize(result, (int(result.shape[1] * scale), 900), interpolation=cv2.INTER_AREA)
    return result


def _face_index(selection: Any, count: int) -> Optional[int]:
    if selection in (None, "", "All faces", "all", "All"):
        return None
    try:
        index = int(selection)
    except (TypeError, ValueError):
        return None
    if index < 0 or index >= count:
        raise ValueError(f"Face {selection} is not available; detect faces again.")
    return index


def _face_context(values: Mapping[str, Any]) -> SimpleNamespace:
    attrs = {"slimming": 0.0}
    for key in _RESHAPE_KEYS:
        attrs[f"reshape_{key}"] = float(values.get(key, 0.0) or 0.0)
    for key in ("reshape_neck_width", "reshape_neck_length"):
        attrs[key] = 0.0
    return SimpleNamespace(**attrs)


def apply_face_reshape(
    image_rgb: np.ndarray,
    detector: Any,
    selection: Any,
    values: Mapping[str, Any],
) -> Tuple[np.ndarray, str]:
    """Apply the existing FaceReshaper to all or one detected face."""
    if detector is None:
        raise ValueError("Face-aware reshape requires the bundled face detector.")
    image = _coerce_rgb(image_rgb)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    faces = detector.detect(bgr)
    if not faces:
        return image, "Reshape skipped: no faces detected."
    selected = _face_index(selection, len(faces))
    contexts = [_face_context(values) if selected is None or i == selected else _face_context({}) for i in range(len(faces))]
    result_bgr = FaceReshaper().reshape(bgr, faces, ctx=None, face_ctxs=contexts)
    target = "all faces" if selected is None else f"face {selected}"
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), f"Reshape applied to {target}."


def semantic_masks_for_selection(
    image_rgb: np.ndarray,
    detector: Any,
    parser: Any,
    selection: Any,
) -> Dict[str, np.ndarray]:
    """Return full-frame semantic masks for the selected face(s)."""
    if detector is None or parser is None:
        raise ValueError("Semantic intersection requires the bundled face detector and parser.")
    image = _coerce_rgb(image_rgb)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    faces = detector.detect(bgr)
    if not faces:
        return {}
    selected = _face_index(selection, len(faces))
    targets = range(len(faces)) if selected is None else (selected,)
    output: Dict[str, np.ndarray] = {}
    for index in targets:
        face = faces[index]
        regions = parser.parse(face.landmarks, bgr, face.bbox, person_mask=None, ied=face.ied)
        for key, aliases in {
            "Skin": ("skin",),
            "Hair": ("hair",),
            "Face": ("face_oval",),
            "Lips": ("lips",),
            "Eyes": ("left_eye", "right_eye"),
            "Clothing": ("cloth",),
        }.items():
            masks = [getattr(regions, alias, None) for alias in aliases]
            masks = [np.asarray(mask, dtype=np.float32) for mask in masks if mask is not None]
            if not masks:
                continue
            combined = np.maximum.reduce(masks)
            if key in output:
                output[key] = np.maximum(output[key], combined)
            else:
                output[key] = combined
    return output


def apply_advanced_edit(
    current_rgb: np.ndarray,
    editor_value: Any,
    mode: str,
    operation: str,
    strength: float,
    semantic: str,
    selection: Any,
    reshape_values: Mapping[str, Any],
    detector: Any,
    parser: Any,
    heal_method: str = "telea",
    remove_engine: str = ADVANCED_REMOVE_ENGINES[0],
) -> AdvancedEditResult:
    """Apply one Advanced Retouch action and return a replayable edit record."""
    current = _coerce_rgb(current_rgb)
    mode = str(mode or "Adjust")
    operation = str(operation or "exposure").lower()
    semantic = str(semantic or "None")
    if mode == "Reshape":
        image, status = apply_face_reshape(current, detector, selection, reshape_values)
        return AdvancedEditResult(image, None, status, {
            "version": 1, "image_shape": list(current.shape[:2]),
            "mode": "Reshape", "selection": selection, "reshape": dict(reshape_values),
        })

    base, mask = extract_editor_image_and_mask(editor_value, fallback_rgb=current)
    if float(mask.max()) <= 1e-6:
        raise ValueError("Paint a mask on the canvas before applying this operation.")
    effective_mask = mask
    bgr = cv2.cvtColor(current, cv2.COLOR_RGB2BGR)
    if semantic != "None":
        if semantic in {"Person", "Background"}:
            if detector is None or not hasattr(detector, "segment_person"):
                raise ValueError("Person/background intersection requires the bundled person segmenter.")
            semantic_mask = np.asarray(detector.segment_person(bgr), dtype=np.float32)
            if semantic == "Background":
                semantic_mask = 1.0 - np.clip(semantic_mask, 0.0, 1.0)
        else:
            semantic_masks = semantic_masks_for_selection(current, detector, parser, selection)
            semantic_mask = semantic_masks.get(semantic)
        if semantic_mask is None:
            raise ValueError(f"The {semantic.lower()} semantic mask is unavailable for the selected face.")
        if semantic_mask.shape != effective_mask.shape:
            semantic_mask = cv2.resize(semantic_mask, (effective_mask.shape[1], effective_mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        effective_mask = np.clip(effective_mask * semantic_mask, 0.0, 1.0)
    if float(effective_mask.max()) <= 1e-6:
        raise ValueError("The painted mask does not overlap the selected semantic region.")

    if mode == "Adjust":
        if operation not in ADVANCED_OPERATIONS:
            raise ValueError(f"Unsupported local operation: {operation}")
        result_bgr = apply_local_adjustment(bgr, effective_mask, operation, float(strength) / 100.0)
        status = f"Applied {operation} at {float(strength):g}%" + (f" inside {semantic.lower()}" if semantic != "None" else "") + "."
    elif mode == "Heal":
        result_bgr = heal_region(bgr, effective_mask, method=str(heal_method or "telea"))
        status = f"Healed mask with {str(heal_method or 'telea').capitalize()}."
    elif mode == "Remove":
        has_lama = model_exists("lama_inpaint")
        if str(remove_engine) == "Telea fallback":
            result_bgr = heal_region(bgr, effective_mask, method="telea")
            used = "Telea"
        else:
            result_bgr = LamaHealer().heal_large(bgr, effective_mask)
            used = "LaMa" if has_lama else "Telea fallback (LaMa model unavailable)"
        status = f"Removed mask with {used}."
    else:
        raise ValueError(f"Unsupported Advanced Retouch mode: {mode}")
    edit = {
        "version": 1,
        "image_shape": list(current.shape[:2]),
        "mode": mode,
        "operation": operation,
        "strength": float(strength),
        "semantic": semantic,
        "selection": selection,
        "mask_png_b64": mask_to_b64(effective_mask.astype(np.float32)),
        "heal_method": heal_method,
        "remove_engine": remove_engine,
    }
    return AdvancedEditResult(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), effective_mask, status, edit)


def replay_advanced_edits(
    source_rgb: np.ndarray,
    edits: Sequence[Mapping[str, Any]],
    detector: Any,
    parser: Any,
) -> np.ndarray:
    """Replay serialized Advanced Retouch edits against a source image."""
    current = _coerce_rgb(source_rgb)
    for edit in edits or ():
        expected_shape = edit.get("image_shape")
        if expected_shape and list(current.shape[:2]) != list(expected_shape):
            raise ValueError(
                "Advanced Retouch session image dimensions do not match the loaded source "
                f"({list(current.shape[:2])} vs {list(expected_shape)})."
            )
        mode = str(edit.get("mode", "Adjust"))
        if mode == "Reshape":
            current, _ = apply_face_reshape(current, detector, edit.get("selection"), edit.get("reshape", {}))
            continue
        mask = b64_to_mask(str(edit.get("mask_png_b64", "")), current.shape[:2])
        editor = {"background": current, "layers": [mask]}
        # The stored mask is already the post-semantic-intersection mask (see
        # apply_advanced_edit below); re-passing edit["semantic"] here would
        # intersect it a second time and, for any non-binary (feathered)
        # semantic mask, silently diverge from the original result.
        result = apply_advanced_edit(
            current, editor, mode, str(edit.get("operation", "exposure")), float(edit.get("strength", 0.0)),
            "None", edit.get("selection"), edit.get("reshape", {}), detector, parser,
            heal_method=str(edit.get("heal_method", "telea")),
            remove_engine=str(edit.get("remove_engine", ADVANCED_REMOVE_ENGINES[0])),
        )
        current = result.image_rgb
    return current
