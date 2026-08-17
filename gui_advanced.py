#!/usr/bin/env python3
"""Advanced Retouch editor cluster extracted from gui.py.

Pure move — handlers, preview/history helpers and state loaders for the
Advanced Retouch workspace. gui.py re-exports every public name and wires
``get_engine`` (engine singleton lives in gui.py so tests can patch
``gui._engine`` / ``gui.get_engine``).
"""

import base64
import logging

import cv2
import numpy as np
import gradio as gr

from retouch.io import imread_engine, imread_engine_with_context
from retouch.advanced_retouch import (
    apply_advanced_edit,
    before_after,
    extract_editor_image_and_mask,
    mask_overlay,
    replay_advanced_edits,
)
from retouch.advanced_history import AdvancedHistory
from retouch.advanced_contract import (
    BASE_KIND_PROCESSED,
    BASE_KIND_SOURCE,
    AdvancedContractError,
    build_base_contract,
    compare_base_contracts,
    parse_session_payload as parse_advanced_session_payload,
    pixel_sha256 as advanced_pixel_sha256,
    replay_result_matches,
)
from retouch.gui_preview_cache import GuiPreviewCache
from retouch.render_manifest import canonical_sha256

_logger = logging.getLogger(__name__)

ADVANCED_PREVIEW_MAX_DIM = 640
ADVANCED_SNAPSHOT_MAX_COUNT = 20


def get_engine():
    """Placeholder — overwritten by gui.py with the real singleton factory."""
    raise RuntimeError("gui_advanced used without gui; import gui first")


def _resolve_image_path(value):
    """Best-effort extraction of a filesystem path from a Gradio input value."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("name") or value.get("path")
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _resolve_image_path(value[0])
    if hasattr(value, "name"):
        return value.name
    return str(value)
def _advanced_preview_image(image_rgb, max_dim=ADVANCED_PREVIEW_MAX_DIM):
    """Return one bounded preview copy for Advanced history/snapshots."""
    if image_rgb is None:
        return None
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        return None
    if image.shape[2] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    image = np.ascontiguousarray(image)
    height, width = image.shape[:2]
    scale = min(float(max_dim) / max(width, height), 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return image.copy()


def _advanced_preview_payload(image_rgb):
    """Encode a bounded preview so snapshots do not retain NumPy arrays."""
    preview = _advanced_preview_image(image_rgb)
    if preview is None:
        return None
    try:
        bgr = cv2.cvtColor(preview, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )
        if ok:
            return {
                "encoding": "jpeg",
                "data": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "width": int(preview.shape[1]),
                "height": int(preview.shape[0]),
            }
    except Exception as exc:
        _logger.debug("Advanced preview encoding failed: %s", exc)
    return {"encoding": "unavailable", "shape": list(preview.shape)}


def _advanced_preview_from_payload(value):
    """Decode a compact snapshot preview; accept legacy array snapshots."""
    if isinstance(value, dict) and value.get("encoding") == "jpeg":
        try:
            raw = base64.b64decode(value.get("data", ""), validate=True)
            decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is not None:
                return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        except Exception as exc:
            _logger.debug("Advanced preview decoding failed: %s", exc)
        return None
    if isinstance(value, np.ndarray):
        return _advanced_preview_image(value)
    return value if isinstance(value, (list, tuple)) else None


def _advanced_history_from_edits(edit_log=None):
    """Build compact history from a serialized edit log."""
    history = AdvancedHistory()
    for edit in list(edit_log or []):
        outcome = history.append(edit)
        if not outcome.changed:
            _logger.warning("Advanced edit was not retained in compact history: %s", outcome.message)
            break
    return history


def _advanced_history_object(history, edit_log=None):
    """Accept the new compact state and migrate the old undo/redo shape."""
    if isinstance(history, AdvancedHistory):
        return history
    if isinstance(history, dict) and "entries" in history and "cursor" in history:
        try:
            return AdvancedHistory.from_state(history)
        except (TypeError, ValueError, KeyError) as exc:
            _logger.warning("Invalid compact Advanced Retouch history; rebuilding: %s", exc)
    # Older sessions contained full RGB arrays under undo/redo.  Do not copy
    # them into the new state; the edit log is the replayable source of truth.
    return _advanced_history_from_edits(edit_log)


def _replay_advanced_state(source_rgb, edits):
    """Replay compact edits, loading face backends only when required."""
    needs_face_models = any(
        str(edit.get("mode", "Adjust")) == "Reshape"
        or str(edit.get("semantic", "None")) != "None"
        for edit in list(edits or [])
    )
    detector = parser = None
    if needs_face_models:
        engine = get_engine()
        detector, parser = engine._detector, engine._parser
    return replay_advanced_edits(source_rgb, edits, detector, parser)


def _advanced_empty_state(source_rgb=None, status=""):
    current = source_rgb.copy() if isinstance(source_rgb, np.ndarray) else source_rgb
    return (
        source_rgb,
        current,
        current,
        None,
        before_after(source_rgb, current) if source_rgb is not None and current is not None else None,
        AdvancedHistory().to_state(),
        [],
        None,
        status,
        [],
        None,
    )


def _advanced_load_rgb_source(
    source_rgb,
    pending_edits=None,
    status_prefix="Loaded source image",
    *,
    base_kind=BASE_KIND_SOURCE,
    source_path=None,
    render_evidence=None,
    explicit_legacy_replay=False,
):
    """Initialize the Advanced Retouch state from an already-decoded RGB image."""
    if source_rgb is None:
        return _advanced_empty_state(None, "Process an image or upload one to begin Advanced Retouch.")
    source = np.ascontiguousarray(source_rgb).copy()
    base_contract = build_base_contract(
        source,
        kind=base_kind,
        source_path=source_path,
        render_evidence=render_evidence,
    )
    pending_payload = None
    if isinstance(pending_edits, dict) and pending_edits:
        pending_payload = parse_advanced_session_payload(pending_edits)
    elif isinstance(pending_edits, list) and pending_edits:
        pending_payload = parse_advanced_session_payload({"version": 1, "edits": pending_edits})

    edits = list(pending_payload.get("edits") or []) if pending_payload else []
    if pending_payload and pending_payload.get("legacy_unverified") and not explicit_legacy_replay:
        return (
            source,
            source.copy(),
            source.copy(),
            None,
            before_after(source, source),
            AdvancedHistory().to_state(),
            [],
            None,
            f"{status_prefix}. {len(edits)} legacy edit(s) remain pending because their base cannot be verified; "
            "use Bind legacy session edits to apply them explicitly.",
            pending_payload,
            base_contract,
        )
    if pending_payload and not pending_payload.get("legacy_unverified"):
        comparison = compare_base_contracts(pending_payload.get("base"), base_contract)
        if not comparison["matches"]:
            return (
                source,
                source.copy(),
                source.copy(),
                None,
                before_after(source, source),
                AdvancedHistory().to_state(),
                [],
                None,
                "{}; saved edits remain pending because the base differs ({}).".format(
                    status_prefix, ", ".join(comparison["reasons"])
                ),
                pending_payload,
                base_contract,
            )
    try:
        current = _replay_advanced_state(source, edits) if edits else source.copy()
        if pending_payload and not pending_payload.get("legacy_unverified"):
            replay_check = replay_result_matches(pending_payload, current)
            if not replay_check["matches"]:
                raise AdvancedContractError(
                    "replay result mismatch: %s" % ", ".join(replay_check["reasons"])
                )
    except Exception as exc:
        return (
            source,
            source.copy(),
            source.copy(),
            None,
            before_after(source, source),
            AdvancedHistory().to_state(),
            [],
            None,
            f"{status_prefix}; saved edits remain pending because replay could not be verified: {exc}",
            pending_payload or [],
            base_contract,
        )
    history = _advanced_history_from_edits(edits)
    detail = base_contract.get("render", {}).get("effective_detail")
    if base_kind == BASE_KIND_PROCESSED and detail != "native":
        delivery_status = " This is proxy/unverified detail; delivery export is blocked until a verified Full Quality render is used."
    elif base_kind == BASE_KIND_PROCESSED:
        delivery_status = " Native Full Quality render evidence verified."
    else:
        delivery_status = " Native uploaded-source evidence recorded."
    replay_status = f" and replayed {len(edits)} edit(s)" if edits else ""
    status = f"{status_prefix}{replay_status}.{delivery_status}"
    return (
        source,
        current,
        current,
        None,
        before_after(source, current),
        history.to_state(),
        history.current_edit_log(),
        None,
        status,
        [],
        base_contract,
    )


def on_advanced_source_change(img_paths, pending_edits=None):
    """Load the first source image into the Advanced Retouch canvas."""
    path = _resolve_image_path(img_paths)
    if not path:
        return _advanced_empty_state(None, "Upload an image to begin Advanced Retouch.")
    try:
        # Advanced Retouch edits the same display-referred sRGB pixels as the
        # main GUI. Keep tagged source pixels on that managed ingest path so
        # the source ICC is not merely copied onto converted pixels later.
        image_bgr, _color_context = imread_engine_with_context(path)
        if image_bgr.dtype != np.uint8:
            image_bgr = np.clip(image_bgr, 0, 255).astype(np.uint8)
        source = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return _advanced_load_rgb_source(
            source,
            pending_edits,
            "Loaded uploaded source image",
            base_kind=BASE_KIND_SOURCE,
            source_path=path,
        )
    except Exception as exc:
        _logger.warning("Advanced Retouch source load failed: %s", exc)
        return _advanced_empty_state(None, f"Could not load image: {exc}")


def on_advanced_processed_result(processed_rgb, pending_edits=None, preview_cache=None, current_edits=None):
    """Switch the Advanced Retouch workspace to the latest recipe result."""
    evidence = (
        preview_cache.latest_render_evidence
        if isinstance(preview_cache, GuiPreviewCache)
        else preview_cache if isinstance(preview_cache, dict) else {}
    )
    explicit_rebase = not pending_edits and bool(current_edits)
    replay_payload = pending_edits or list(current_edits or [])
    return _advanced_load_rgb_source(
        processed_rgb,
        replay_payload,
        "Loaded processed recipe result",
        base_kind=BASE_KIND_PROCESSED,
        source_path=evidence.get("source_path") if isinstance(evidence, dict) else None,
        render_evidence=evidence,
        explicit_legacy_replay=explicit_rebase,
    )


def advanced_bind_legacy_handler(source_rgb, pending_payload, base_contract):
    """Explicitly bind a legacy edit-only session to the visible base."""
    if source_rgb is None or base_contract is None:
        return (
            source_rgb,
            source_rgb,
            before_after(source_rgb, source_rgb) if source_rgb is not None else None,
            AdvancedHistory().to_state(),
            [],
            pending_payload,
            "Load a source before binding legacy Advanced Retouch edits.",
        )
    try:
        parsed = parse_advanced_session_payload(pending_payload or {})
        if not parsed.get("legacy_unverified"):
            return (
                source_rgb,
                source_rgb,
                before_after(source_rgb, source_rgb),
                AdvancedHistory().to_state(),
                [],
                pending_payload,
                "No legacy Advanced Retouch edits are pending.",
            )
        edits = list(parsed.get("edits") or [])
        current = _replay_advanced_state(source_rgb, edits)
        history = _advanced_history_from_edits(edits)
        return (
            current,
            current,
            before_after(source_rgb, current),
            history.to_state(),
            history.current_edit_log(),
            [],
            f"Explicitly bound and replayed {len(edits)} legacy edit(s). Save the session to upgrade it to v2 evidence.",
        )
    except Exception as exc:
        return (
            source_rgb,
            source_rgb,
            before_after(source_rgb, source_rgb),
            AdvancedHistory().to_state(),
            [],
            pending_payload,
            f"Legacy Advanced Retouch edits were not applied: {exc}",
        )


def on_advanced_face_choices(img_paths):
    path = _resolve_image_path(img_paths)
    if not path:
        return gr.update(choices=["All faces"], value="All faces"), "Upload an image first."
    try:
        image = imread_engine(path)
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        faces = get_engine()._detector.detect(image)
        choices = ["All faces"] + [str(i) for i in range(len(faces))]
        return gr.update(choices=choices, value="All faces"), f"Detected {len(faces)} face(s)."
    except Exception as exc:
        return gr.update(choices=["All faces"], value="All faces"), f"Face detection unavailable: {exc}"


def _advanced_history_push(history, current, edit_log, edit=None):
    """Append a compact operation and never retain a full-resolution frame."""
    compact = _advanced_history_object(history, edit_log)
    if edit is not None:
        compact.append(edit, preview=_advanced_preview_image(current))
    return compact


def advanced_apply_handler(
    editor_value,
    current_rgb,
    source_rgb,
    history,
    edit_log,
    mode,
    operation,
    strength,
    semantic,
    selection,
    heal_method,
    remove_engine,
    overlay_visible,
    overlay_opacity,
    *reshape_values,
):
    if current_rgb is None:
        return (gr.update(), current_rgb, history, edit_log, None, None, None, "Load an image first.")
    reshape_keys = (
        "eye_size", "eye_distance", "nose_width", "nose_length", "jaw_width",
        "chin_length", "mouth_size", "smile", "forehead",
        "eye_size_l", "eye_size_r", "nose_width_l", "nose_width_r",
        "jaw_width_l", "jaw_width_r",
    )
    reshape = dict(zip(reshape_keys, list(reshape_values)[:len(reshape_keys)]))
    mask_action = str(reshape_values[len(reshape_keys)] if len(reshape_values) > len(reshape_keys) else "Keep")
    mask_feather = float(reshape_values[len(reshape_keys) + 1] if len(reshape_values) > len(reshape_keys) + 1 else 0.0)
    try:
        needs_face_models = str(mode or "Adjust") == "Reshape" or str(semantic or "None") != "None"
        detector = parser = None
        if needs_face_models:
            engine = get_engine()
            detector, parser = engine._detector, engine._parser
        editor_for_apply = editor_value
        if str(mode or "Adjust") != "Reshape":
            base, painted_mask = extract_editor_image_and_mask(editor_value, fallback_rgb=current_rgb)
            if mask_action == "Clear":
                painted_mask = np.zeros_like(painted_mask)
            elif mask_action == "Invert":
                painted_mask = 1.0 - painted_mask
            if mask_feather > 0:
                painted_mask = cv2.GaussianBlur(painted_mask.astype(np.float32), (0, 0), sigmaX=mask_feather)
            editor_for_apply = {"background": base, "layers": [painted_mask]}
        result = apply_advanced_edit(
            current_rgb,
            editor_for_apply,
            mode,
            operation,
            float(strength or 0.0),
            semantic,
            selection,
            reshape,
            detector,
            parser,
            heal_method=heal_method,
            remove_engine=remove_engine,
        )
        compact_history = _advanced_history_object(history, edit_log)
        previous_history_state = compact_history.to_state()
        history_outcome = compact_history.append(
            result.edit,
            preview=_advanced_preview_image(result.image_rgb),
        )
        if not history_outcome.changed:
            return (
                gr.update(),
                current_rgb,
                history,
                list(edit_log or []),
                None,
                None,
                before_after(source_rgb, current_rgb) if source_rgb is not None else None,
                f"Advanced Retouch history rejected the edit: {history_outcome.message}",
            )
        if compact_history.history_truncated:
            return (
                gr.update(),
                current_rgb,
                previous_history_state,
                list(edit_log or []),
                None,
                None,
                before_after(source_rgb, current_rgb) if source_rgb is not None else None,
                "Advanced Retouch edit was not applied because it would cross the replay-safe history boundary. "
                "Save/export the canvas or reset the workspace before continuing.",
            )
        new_history = compact_history.to_state()
        new_log = compact_history.current_edit_log()
        overlay = mask_overlay(result.image_rgb, result.mask, float(overlay_opacity or 42) / 100.0) if overlay_visible else result.image_rgb
        return result.image_rgb, result.image_rgb, new_history, new_log, result.mask, overlay, before_after(source_rgb, result.image_rgb), result.status
    except Exception as exc:
        _logger.warning("Advanced Retouch action failed: %s", exc)
        return gr.update(), current_rgb, history, edit_log, None, None, before_after(source_rgb, current_rgb) if source_rgb is not None else None, f"Advanced Retouch failed: {exc}"


def advanced_undo_handler(current_rgb, source_rgb, history, edit_log=None):
    compact_history = _advanced_history_object(history, edit_log)
    if compact_history.history_truncated:
        return gr.update(), current_rgb, compact_history.to_state(), list(edit_log or []), None, None, before_after(source_rgb, current_rgb) if source_rgb is not None else None, "Undo unavailable: the compact history boundary has no full-resolution replay baseline."
    outcome = compact_history.undo()
    if not outcome.changed:
        return gr.update(), current_rgb, compact_history.to_state(), list(edit_log or []), None, None, before_after(source_rgb, current_rgb) if source_rgb is not None else None, "Nothing to undo."
    edits = compact_history.current_edit_log()
    try:
        previous = _replay_advanced_state(source_rgb, edits)
    except Exception as exc:
        _logger.warning("Advanced Retouch undo replay failed: %s", exc)
        return gr.update(), current_rgb, history, list(edit_log or []), None, None, before_after(source_rgb, current_rgb) if source_rgb is not None else None, f"Undo replay failed: {exc}"
    return previous, previous, compact_history.to_state(), edits, None, None, before_after(source_rgb, previous) if source_rgb is not None else None, "Undid the last Advanced Retouch edit."


def advanced_redo_handler(current_rgb, source_rgb, history, edit_log=None):
    compact_history = _advanced_history_object(history, edit_log)
    if compact_history.history_truncated:
        return gr.update(), current_rgb, compact_history.to_state(), list(edit_log or []), None, None, before_after(source_rgb, current_rgb) if source_rgb is not None else None, "Redo unavailable: the compact history boundary has no full-resolution replay baseline."
    outcome = compact_history.redo()
    if not outcome.changed:
        return gr.update(), current_rgb, compact_history.to_state(), list(edit_log or []), None, None, before_after(source_rgb, current_rgb) if source_rgb is not None else None, "Nothing to redo."
    edits = compact_history.current_edit_log()
    try:
        next_image = _replay_advanced_state(source_rgb, edits)
    except Exception as exc:
        _logger.warning("Advanced Retouch redo replay failed: %s", exc)
        return gr.update(), current_rgb, history, list(edit_log or []), None, None, before_after(source_rgb, current_rgb) if source_rgb is not None else None, f"Redo replay failed: {exc}"
    return next_image, next_image, compact_history.to_state(), edits, None, None, before_after(source_rgb, next_image) if source_rgb is not None else None, "Redid the last Advanced Retouch edit."


def advanced_reset_handler(source_rgb):
    return _advanced_empty_state(source_rgb, "Advanced Retouch edits reset.")[1:9]


def advanced_overlay_handler(current_rgb, mask, visible, opacity):
    if current_rgb is None:
        return None
    return mask_overlay(current_rgb, mask, float(opacity or 42) / 100.0) if visible and mask is not None else current_rgb


def advanced_clear_mask_handler(editor_value, current_rgb, source_rgb):
    """Clear only the painted mask; leave the current pixels untouched."""
    if current_rgb is None:
        return gr.update(), None, None, None, "Load an image first."
    base, _ = extract_editor_image_and_mask(editor_value, fallback_rgb=current_rgb)
    cleared_editor = {"background": base, "layers": []}
    return cleared_editor, None, current_rgb, before_after(source_rgb, current_rgb), "Cleared Advanced Retouch mask."


def advanced_save_snapshot_handler(name, current_rgb, edit_log, snapshots, base_contract=None):
    if not name or not str(name).strip():
        return gr.update(), snapshots or {}, "Enter a snapshot name."
    if current_rgb is None:
        return gr.update(), snapshots or {}, "Load an image first."
    snapshots = dict(snapshots or {})
    key = str(name).strip()
    snapshots[key] = {
        "preview": _advanced_preview_payload(current_rgb),
        "edits": list(edit_log or []),
        "edits_sha256": canonical_sha256(list(edit_log or [])),
        "result_sha256": advanced_pixel_sha256(current_rgb),
        "base_contract_sha256": (
            base_contract.get("contract_sha256")
            if isinstance(base_contract, dict)
            else None
        ),
        "preview_resolution": ADVANCED_PREVIEW_MAX_DIM,
    }
    while len(snapshots) > ADVANCED_SNAPSHOT_MAX_COUNT:
        snapshots.pop(next(iter(snapshots)))
    return gr.update(choices=list(snapshots.keys()), value=key), snapshots, f"Saved Advanced Retouch snapshot '{key}'."


def advanced_compare_snapshot_handler(name, current_rgb, snapshots):
    snapshots = snapshots or {}
    if not name or name not in snapshots or current_rgb is None:
        return None, "Select a snapshot and load an image first."
    snapshot = snapshots[name]
    snapshot_image = _advanced_preview_from_payload(snapshot.get("preview"))
    if snapshot_image is None:
        snapshot_image = _advanced_preview_from_payload(snapshot.get("image"))
    if snapshot_image is None:
        return None, f"Snapshot '{name}' has no preview image."
    return before_after(snapshot_image, _advanced_preview_image(current_rgb)), f"Comparing current edit with snapshot '{name}'."
