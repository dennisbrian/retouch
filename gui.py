#!/usr/bin/env python3

import base64
import json
import logging
import sys
import os
import shutil
import subprocess
import time
import tempfile
import threading
import zipfile
from pathlib import Path

import cv2
import numpy as np
import gradio as gr

# Some FileData-bearing components (image/file uploaders) resolve their
# pydantic JSON schema's `additionalProperties` to a bare `bool` rather than
# a dict on newer huggingface-hub/pydantic combinations. gradio_client's
# schema walker assumes a dict and crashes with
# "TypeError: argument of type 'bool' is not iterable" on every page load
# (GET /), which gradio's own launch() self-check then reports as
# "localhost is not accessible". Patch defensively at import time so a
# fresh `pip install` of gradio_client (which lacks this guard) doesn't
# resurrect the crash; no-ops once upstream ships the same guard.
try:
    import gradio_client.utils as _gc_utils

    _orig_json_schema_to_python_type = _gc_utils._json_schema_to_python_type

    def _patched_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return _orig_json_schema_to_python_type(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from retouch import RetouchEngine, __version__
from retouch.engine import resolve_recipe
from retouch.io import (
    EXT_MAP,
    EXPORT_RES_MAP,
    RAW_EXTENSIONS,
    color_context_for_path,
    encode_write_params,
    get_working_srgb_icc,
    imread_engine,
    imread_engine_with_context,
    imread_exif,
    read_icc_profile,
    read_exif_bytes,
    read_c2pa_manifest,
    write_image_with_icc,
    write_image_with_color_context,
)
from retouch.color_context import ColorContext
from retouch.lips import LIP_TINT_NAMES
from retouch.recipes import CURATED_RECIPE_NAMES, RECIPE_UI_CHOICES, RECIPES
from retouch.params import recipe_to_params, PROCESSING_PARAMS, param_names, gui_values_to_engine_kwargs
from retouch.grading import list_available_presets
from retouch.style_library import list_styles, save_style_profile, learn_dataset_style
from retouch.batch_processor import BatchProcessor, validate_batch_roots
from retouch.style import StyleProfile
from retouch.look_extractor import LookExtractor
from retouch.recipe_cookbook import search_recipes, list_recipes, list_categories
from retouch.diagnostics import clear_diagnostics, diagnostics_report
from retouch.runtime_doctor import (
    check_detector,
    format_runtime_report,
    runtime_doctor_report,
)
from retouch.lut import get_registry, watch_luts_dir
from retouch.marks import MARK_POLICY_PRESET_NAMES
from retouch.model_fetch import model_exists, model_status
from retouch.project_profiles import (
    LookBoard,
    LookReference,
    ProjectProfile,
    ProjectProfileStore,
)
from retouch.shoot_intelligence import (
    ProjectGraph,
    ProjectNode,
    group_bursts,
    inspect_asset,
    rank_burst_candidates,
)
from retouch.shoot_review import (
    DECISIONS,
    ShootReviewManifest,
    ShootReviewManifestStore,
    build_review_manifest,
)
from retouch.detection import FaceDetector
from retouch.face_quality import FaceQualityAnalyzer
from retouch.watch_folder import WatchFolder
from retouch.advanced_retouch import (
    ADVANCED_HEAL_METHODS,
    ADVANCED_OPERATIONS,
    ADVANCED_REMOVE_ENGINES,
    ADVANCED_SEMANTIC_MASKS,
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
    build_session_payload as build_advanced_session_payload,
    compare_base_contracts,
    delivery_decision as advanced_delivery_decision,
    parse_session_payload as parse_advanced_session_payload,
    pixel_sha256 as advanced_pixel_sha256,
    replay_result_matches,
    source_file_matches,
)
from retouch.gui_preview_cache import (
    GuiPreviewCache,
    make_preview_cache_key,
    source_identity,
)
from retouch.gui_render_modes import (
    MODE_EXPORT_ALL,
    MODE_EXPORT_FULL_QUALITY,
    MODE_RENDER_PREVIEW,
    RenderContractError,
    build_export_all,
    build_export_full_quality,
    build_render_preview,
    capture_settings_snapshot,
    validate_render_contract,
)
from retouch.gui_inspection import (
    MODE_100_PERCENT as INSPECTION_MODE_100_PERCENT,
    MODE_FACE as INSPECTION_MODE_FACE,
    MODE_FIT as INSPECTION_MODE_FIT,
    MODE_ROI as INSPECTION_MODE_ROI,
    InspectionContractError,
    build_inspection_contract,
    crop_from_inspection_contract,
)
from retouch.render_manifest import RenderManifest, canonical_sha256
from retouch.gui_workspace import SessionWorkspace, cleanup_workspace
from gui_advanced import (
    ADVANCED_PREVIEW_MAX_DIM,
    ADVANCED_SNAPSHOT_MAX_COUNT,
    _resolve_image_path,
    _advanced_preview_image,
    _advanced_preview_payload,
    _advanced_preview_from_payload,
    _advanced_history_from_edits,
    _advanced_history_object,
    _advanced_history_push,
    _replay_advanced_state,
    _advanced_empty_state,
    _advanced_load_rgb_source,
    on_advanced_source_change,
    on_advanced_processed_result,
    on_advanced_face_choices,
    advanced_bind_legacy_handler,
    advanced_apply_handler,
    advanced_undo_handler,
    advanced_redo_handler,
    advanced_reset_handler,
    advanced_overlay_handler,
    advanced_clear_mask_handler,
    advanced_save_snapshot_handler,
    advanced_compare_snapshot_handler,
)

_logger = logging.getLogger(__name__)

# Keep the normal editing flow focused. Legacy recipes remain available to
# existing saved sessions but are not presented as default creative choices.
RECIPE_NAMES = list(CURATED_RECIPE_NAMES)
COLOR_GRADE_NAMES = ["none"] + list_available_presets()
LUT_CHOICES = ["none", "kodak", "fuji"]
_TONE_CHOICES = ["rosy", "porcelain", "neutral"]
WHITEN_TONE_CHOICES = _TONE_CHOICES
LIP_FINISH_CHOICES = ["gloss", "matte", "velvet"]
SPECULAR_BLOOM_TONE_CHOICES = _TONE_CHOICES

PREVIEW_MAX_HEIGHT = 900
COMPARE_SEPARATOR_WIDTH = 4
COMPARE_SEPARATOR_COLOR = 200
TEMP_CLEANUP_AGE_SEC = 300
GUI_WORKSPACE_ROOT_ENV = "RETOUCH_GUI_WORKSPACE_ROOT"
_RENDER_EXPORT_SENTINEL = object()

_engine = None
_engine_lock = threading.Lock()
GUI_ENGINE_CONCURRENCY_ID = "retouch-engine"


def get_engine():
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                _engine = RetouchEngine()
    return _engine


# The Advanced Retouch handlers live in gui_advanced and resolve the engine
# singleton (including test monkeypatches of gui.get_engine/_engine) through
# this function object.
import gui_advanced

gui_advanced.get_engine = get_engine

# Smart Color card handlers (T2a — Color card only, no engine/model
# dependency, no get_engine wiring needed).
import gui_smart


# Shoot Intelligence / Watch Folder / Review / Profiles / Look Board handlers
# live in gui_shoot and resolve patchable shared names (FaceDetector,
# BatchProcessor, ProjectProfileStore, imread_exif, model_status) through
# this module reference at call time.
from gui_shoot import (
    _resolve_image_paths,
    on_shoot_intelligence_scan,
    on_watch_folder_process,
    _review_store,
    on_save_shoot_review,
    on_export_shoot_review_json,
    on_export_shoot_review_csv,
    on_save_project_profile,
    on_save_look_board,
    on_apply_look_board,
)

import gui_shoot

gui_shoot.gui = sys.modules[__name__]


# Batch / style-learn / job-dashboard handlers live in gui_batch and resolve
# patchable shared names (gr, _run_batch, get_custom_style_names) through
# this module reference at call time.
from gui_batch import (
    on_save_style,
    on_learn_style,
    on_process_folder,
    _run_batch,
    _job_row,
    on_refresh_jobs,
    _file_row,
    on_select_job,
    on_rerun_flagged,
)

import gui_batch

gui_batch.gui = sys.modules[__name__]


def _workspace_for_request(request=None):
    """Return the session-owned workspace for a Gradio request, if present."""
    session_hash = getattr(request, "session_hash", None)
    if session_hash in (None, ""):
        return None
    try:
        return SessionWorkspace(session_hash)
    except Exception as exc:  # noqa: BLE001 - workspace must not block fallback renders
        _logger.warning("Session workspace unavailable; using legacy temp fallback: %s", exc)
        return None


def cleanup_gui_request(request=None):
    """Clean one browser session's workspace at Gradio unload/shutdown."""
    workspace = _workspace_for_request(request)
    return cleanup_workspace(workspace)


def _source_color_cache_descriptor(path):
    """Describe source color/decode policy without decoding image pixels."""
    suffix = Path(path).suffix.lower()
    if suffix in RAW_EXTENSIONS:
        return {"source_kind": "raw-srgb", "profile_sha256": None}
    try:
        profile = read_icc_profile(path)
    except Exception:
        profile = None
    if profile:
        import hashlib
        return {
            "source_kind": "embedded-icc",
            "profile_sha256": hashlib.sha256(profile).hexdigest(),
        }
    return {"source_kind": "assumed-srgb", "profile_sha256": None}


def _detector_cache_descriptor(engine):
    detector = getattr(engine, "_detector", None)
    return {
        "backend": str(getattr(detector, "backend_name", "unknown")),
        "available": bool(getattr(detector, "available", False)),
        "model": str(getattr(detector, "model_path", "unknown")),
    }


def advanced_model_status_text():
    """Truthful status line for optional model-backed Advanced Retouch paths."""
    from retouch.body_reshape import body_reshape_capability_status
    from retouch.update_check import offline_mode_enabled

    lama = "available" if model_exists("lama_inpaint") else "unavailable — Telea fallback"
    sr = "available" if model_exists("sr_real_esrgan") else "unavailable — standard resize fallback"
    nafnet = "verified local" if model_exists("denoise_nafnet") else "unavailable"
    try:
        parsing_status = model_status("resnet18_bisenet")
        parsing = "verified local" if parsing_status.get("available") else "unavailable — landmark fallback"
    except Exception as exc:  # noqa: BLE001 — capability text must never block startup
        parsing = f"unavailable — {type(exc).__name__}"
    pose_status = body_reshape_capability_status()
    pose = "available" if pose_status.get("available") else "unavailable — model not verified"
    try:
        face_model_status = model_status("face_landmarker")
        face_check = check_detector({"face_landmarker": face_model_status}, probe=False)
        if face_check.get("available"):
            face = (
                f"Face-aware — {face_check.get('backend', 'detector')} "
                "(native initialization unprobed)"
            )
        else:
            face = f"Global-only — {face_check.get('reason', 'detector unavailable')}"
    except Exception as exc:  # noqa: BLE001 — status must never block the GUI
        face = f"Global-only — {type(exc).__name__}: {exc}"
    return (
        f"**Model status:** LaMa: {lama} · Real-ESRGAN: {sr} · "
        f"NAFNet denoise: {nafnet} · Face parsing: {parsing} · Body reshape: {pose}.  "
        f"**Face-aware status:** {face}.  **Network:** "
        f"{'offline/privacy mode' if offline_mode_enabled() else 'update checks enabled'}."
    )


def runtime_doctor_text() -> str:
    """Return a copyable, side-effect-free Runtime Doctor report for the GUI."""
    try:
        report = runtime_doctor_report(probe_detector=False, probe_parser=False)
        return format_runtime_report(report) + "\n\n" + json.dumps(
            report, indent=2, sort_keys=True
        )
    except Exception as exc:  # noqa: BLE001 — diagnostics must never block the GUI
        _logger.exception("Runtime Doctor failed: %s", exc)
        return f"Runtime Doctor failed: {type(exc).__name__}: {exc}"


def _face_runtime_label(detector) -> str:
    """Format the engine detector's explicit capability mode for the GUI."""
    if detector is None:
        return "Face-aware: unprobed"
    try:
        runtime = detector.runtime_status()
    except AttributeError:
        available = bool(getattr(detector, "available", False))
        runtime = {
            "mode": "face_aware" if available else "global_only",
            "backend": getattr(detector, "backend_name", "unknown"),
            "reason": getattr(detector, "unavailable_reason", None),
        }
    if runtime.get("mode") == "face_aware":
        return f"Face-aware: {runtime.get('backend', 'initialized')}"
    return f"Global-only: {runtime.get('reason') or 'face detector unavailable'}"


def _coerce_settings_revision(value) -> int:
    """Return a non-negative settings revision for UI/state boundaries."""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def on_settings_changed(current_revision=0):
    """Advance a session-owned draft revision without touching controls.

    The revision is deliberately supplied by ``gr.State``.  Keeping this
    callback pure prevents one browser session's slider edits from marking a
    different session's render stale.
    """
    current = _coerce_settings_revision(current_revision)
    revision = current + 1
    return revision, (
        f"Settings revision {revision} — Settings changed — preview is out of date."
    )


def _remember_settings_revision(revision: int) -> int:
    """Compatibility shim that never stores process-global revision state."""
    return _coerce_settings_revision(revision)


def _current_settings_revision() -> int:
    """Return the neutral value for callers that have no session State."""
    return 0


def render_status_for_revision(status, snapshot_revision, current_revision=None):
    """Annotate a completed render with its immutable settings snapshot."""
    snapshot = _coerce_settings_revision(snapshot_revision)
    current = _coerce_settings_revision(current_revision)
    message = (
        f"{status} | Rendering settings revision {snapshot} | "
        f"Preview revision {snapshot} | Draft revision {current}"
    )
    if current > snapshot:
        message += " | Settings changed while rendering — preview is out of date"
    return message


def render_start_status(snapshot_revision):
    """Show the snapshot revision before a long render begins."""
    revision = _coerce_settings_revision(snapshot_revision)
    return f"Rendering settings revision {revision} | Draft revision {revision}…"


def smart_start_status(snapshot_revision):
    """Show the Smart Process snapshot before analysis begins."""
    return f"Smart analysis settings revision {_coerce_settings_revision(snapshot_revision)}…"


def _render_contract_value(value):
    """Convert transient Gradio values to a bounded JSON-safe snapshot value."""
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not np.isfinite(value):
            return repr(value)
        return value
    if isinstance(value, np.generic):
        return _render_contract_value(value.item())
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _render_contract_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_render_contract_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": [int(item) for item in value.shape],
            "dtype": str(value.dtype),
        }
    name = getattr(value, "name", None) or getattr(value, "path", None)
    if name is not None:
        return str(name)
    return {"__type__": type(value).__name__, "repr": repr(value)[:256]}


def _render_contract_paths(img_paths):
    values = img_paths if isinstance(img_paths, list) else [img_paths]
    result = []
    for value in values:
        if isinstance(value, dict):
            value = value.get("name") or value.get("path")
        elif not isinstance(value, (str, os.PathLike)):
            value = getattr(value, "name", None) or getattr(value, "path", None)
        if value:
            result.append(os.fspath(value))
    return result


def _build_gui_render_contract(
    mode,
    process_args,
    revision,
    preserve_source_profile=False,
):
    """Build the pure render contract when the browser supplied valid paths."""
    params = dict(zip(PROCESS_INPUT_KEYS, process_args))
    settings = {
        key: _render_contract_value(value)
        for key, value in params.items()
        if key != "img_paths"
    }
    # Hash the effective render policy, not a stale UI checkbox.  The mode
    # boundary is authoritative: Preview is fast; Full Quality is full-speed
    # and always uses the native/full quality tier.
    settings["render_mode"] = mode
    if mode == MODE_RENDER_PREVIEW:
        settings["fast"] = True
    elif mode == MODE_EXPORT_FULL_QUALITY:
        settings["fast"] = False
        settings["quality_tier"] = "Full (native face crops)"
    settings["preserve_source_profile"] = bool(preserve_source_profile)
    snapshot = capture_settings_snapshot(settings, revision)
    paths = _render_contract_paths(params.get("img_paths"))
    if not paths:
        raise RenderContractError("no source paths")
    if mode == MODE_RENDER_PREVIEW:
        return build_render_preview(paths, snapshot, revision)
    if mode == MODE_EXPORT_FULL_QUALITY:
        return build_export_full_quality(paths, snapshot, revision)
    return build_export_all(paths, snapshot, revision)


def capture_render_snapshot(*args):
    """Capture all render inputs before the chained long-running event."""
    process_args = tuple(args[:len(PROCESS_INPUT_KEYS)])
    snapshot_revision = args[len(PROCESS_INPUT_KEYS)] if len(args) > len(PROCESS_INPUT_KEYS) else 0
    render_mode = args[len(PROCESS_INPUT_KEYS) + 1] if len(args) > len(PROCESS_INPUT_KEYS) + 1 else None
    preserve_source_profile = (
        bool(args[len(PROCESS_INPUT_KEYS) + 2])
        if len(args) > len(PROCESS_INPUT_KEYS) + 2
        else False
    )
    snapshot = {
        "process_args": process_args,
        "settings_revision": _coerce_settings_revision(snapshot_revision),
        "preserve_source_profile": preserve_source_profile,
    }
    if render_mode in {MODE_RENDER_PREVIEW, MODE_EXPORT_FULL_QUALITY, MODE_EXPORT_ALL}:
        snapshot["render_mode"] = render_mode
        try:
            snapshot["render_contract"] = _build_gui_render_contract(
                render_mode,
                process_args,
                _coerce_settings_revision(snapshot_revision),
                preserve_source_profile,
            )
        except Exception as exc:  # contract details are diagnostic; the handler remains fail-closed
            snapshot["render_contract_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot, render_start_status(snapshot_revision)


def capture_render_preview_snapshot(*args):
    return capture_render_snapshot(*args, MODE_RENDER_PREVIEW)


def capture_export_full_snapshot(*args):
    return capture_render_snapshot(*args, MODE_EXPORT_FULL_QUALITY)


def capture_smart_snapshot(img_paths, recipe, settings_revision=0):
    """Capture Smart Process inputs before analysis is queued."""
    return {
        "img_paths": img_paths,
        "recipe": recipe,
        "settings_revision": _coerce_settings_revision(settings_revision),
    }, smart_start_status(settings_revision)


def render_completion_status(
    status,
    snapshot,
    current_revision=0,
    export_file=_RENDER_EXPORT_SENTINEL,
):
    """Finalize a render status using the current session revision.

    This runs after the expensive render and reads the live ``gr.State`` value
    at completion, so edits made while the engine was busy are reported as a
    stale preview instead of being lost in a process-global counter.
    """
    snapshot_revision = snapshot.get("settings_revision", 0) if isinstance(snapshot, dict) else 0
    result = render_status_for_revision(status, snapshot_revision, current_revision)
    if isinstance(snapshot, dict) and snapshot.get("render_contract_error"):
        result += f" | Render contract: {snapshot['render_contract_error']}"
    if export_file is not _RENDER_EXPORT_SENTINEL:
        stale = _coerce_settings_revision(current_revision) > _coerce_settings_revision(snapshot_revision)
        if stale and isinstance(snapshot, dict) and snapshot.get("render_mode") == MODE_EXPORT_FULL_QUALITY:
            return result + " | Full export discarded because settings changed during rendering", gr.update(value=None, visible=False)
        return result, gr.skip()
    return result


def inspect_render_handler(
    processed_image,
    render_snapshot,
    current_revision=0,
    inspection_mode=INSPECTION_MODE_FIT,
    face_index=0,
    roi_json="",
    preview_cache=None,
):
    """Inspect a native crop from the exact render revision shown in State."""
    if processed_image is None:
        return gr.update(value=None, visible=False), "No rendered image is available for inspection."
    if not isinstance(render_snapshot, dict):
        return gr.update(value=None, visible=False), "Inspection unavailable: render revision evidence is missing."

    evidence = {}
    face_contexts = []
    if isinstance(preview_cache, GuiPreviewCache):
        evidence = preview_cache.latest_render_evidence
        face_contexts = list(preview_cache.latest_render_face_contexts)

    native_height, native_width = np.asarray(processed_image).shape[:2]
    native = evidence.get("native") if isinstance(evidence, dict) else None
    if isinstance(native, dict):
        native_width = int(native.get("width", native_width))
        native_height = int(native.get("height", native_height))
    render_revision = _coerce_settings_revision(
        render_snapshot.get("settings_revision", evidence.get("render_revision", 0))
    )
    try:
        requested_face = int(face_index or 0)
    except (TypeError, ValueError):
        requested_face = 0
    roi = None
    if isinstance(roi_json, str) and roi_json.strip():
        try:
            roi = json.loads(roi_json)
        except json.JSONDecodeError as exc:
            return gr.update(value=None, visible=False), f"Inspection unavailable: ROI JSON is invalid ({exc})."
    elif isinstance(roi_json, (dict, list, tuple)):
        roi = roi_json

    try:
        contract = build_inspection_contract(
            mode=inspection_mode,
            native_size=(native_width, native_height),
            render_revision=render_revision,
            current_revision=_coerce_settings_revision(current_revision),
            faces=face_contexts,
            face_index=requested_face,
            face_padding=24,
            roi=roi,
        )
        cropped = crop_from_inspection_contract(
            np.asarray(processed_image),
            contract,
            _coerce_settings_revision(current_revision),
        )
    except InspectionContractError as exc:
        stale_suffix = " (stale render)" if _coerce_settings_revision(current_revision) > render_revision else ""
        return gr.update(value=None, visible=False), f"Inspection unavailable: {exc}{stale_suffix}"
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        return gr.update(value=None, visible=False), f"Inspection unavailable: {type(exc).__name__}: {exc}"

    return (
        gr.update(value=cropped, visible=True),
        json.dumps(
            {
                "mode": contract["mode"],
                "render_revision": contract["render"]["revision"],
                "crop": contract["crop"],
                "native": contract["native"],
                "download_enabled": False,
            },
            indent=2,
            sort_keys=True,
        ),
    )


def build_render_manifest_handler(
    render_snapshot,
    current_revision=0,
    preview_cache=None,
    export_file=None,
):
    """Build a pixel-free manifest for the latest GUI render."""
    snapshot = render_snapshot if isinstance(render_snapshot, dict) else {}
    evidence = (
        preview_cache.latest_render_evidence
        if isinstance(preview_cache, GuiPreviewCache)
        else {}
    )
    render_revision = _coerce_settings_revision(
        snapshot.get("settings_revision", evidence.get("render_revision", 0))
    )
    current = _coerce_settings_revision(current_revision)
    mode_value = snapshot.get("render_mode", MODE_RENDER_PREVIEW)
    mode = "full" if mode_value == MODE_EXPORT_FULL_QUALITY else "preview"
    settings_sha256 = None
    contract = snapshot.get("render_contract")
    if isinstance(contract, dict):
        settings_sha256 = (
            contract.get("settings", {})
            if isinstance(contract.get("settings"), dict)
            else {}
        ).get("settings_sha256")
    if not isinstance(settings_sha256, str) or len(settings_sha256) != 64:
        settings_sha256 = canonical_sha256(
            {
                "revision": render_revision,
                "mode": mode_value,
                "preserve_source_profile": bool(
                    snapshot.get("preserve_source_profile", False)
                ),
            }
        )

    source_path = evidence.get("source_path")
    source: Dict[str, Any] = {
        "id": str(source_path or "unknown-source"),
    }
    if source_path:
        try:
            identity = source_identity(source_path)
            source.update(
                {
                    "path": identity.path,
                    "mtime_ns": identity.mtime_ns,
                    "size_bytes": identity.size_bytes,
                    "sha256": identity.content_sha256,
                }
            )
        except (FileNotFoundError, OSError, ValueError):
            pass

    output_id = export_file or "preview-%d" % render_revision
    output: Dict[str, Any] = {
        "id": str(output_id),
        "dimensions": {
            "width": int(evidence.get("output", {}).get("width", 1)),
            "height": int(evidence.get("output", {}).get("height", 1)),
            "channels": 3,
        },
        "dtype": str(evidence.get("output", {}).get("dtype", "uint8")),
    }
    hashes = {"settings_sha256": settings_sha256}
    if export_file and isinstance(export_file, (str, os.PathLike)):
        try:
            output_identity = source_identity(export_file)
            output.update(
                {
                    "path": output_identity.path,
                    "mtime_ns": output_identity.mtime_ns,
                    "size_bytes": output_identity.size_bytes,
                    "sha256": output_identity.content_sha256,
                }
            )
            hashes["output_sha256"] = output_identity.content_sha256
        except (FileNotFoundError, OSError, ValueError):
            pass

    cache_status = str(evidence.get("cache_status", "unknown"))
    if cache_status not in {"hit", "miss", "bypassed", "unknown"}:
        cache_status = "unknown"
    manifest = RenderManifest(
        render_revision=render_revision,
        settings_sha256=settings_sha256,
        mode=mode,
        status="completed",
        source=source,
        output=output,
        cache={
            "status": cache_status,
            "hit": cache_status == "hit" if cache_status != "unknown" else False,
        },
        color_context=evidence.get("color_context", {}),
        metadata_result=evidence.get("metadata_result", {}),
        face_count=evidence.get("face_count"),
        timings_ms=evidence.get("timings_ms", {}),
        qa=evidence.get("qa", {}),
        safe_auto_decisions=evidence.get("safe_auto_decisions", []),
        backend=evidence.get("backend", {}),
        provider=evidence.get("provider"),
        hashes=hashes,
        extensions={
            "stale": current > render_revision,
            "render_mode": mode_value,
            "precision": evidence.get("precision", {}),
        },
    )
    return manifest.to_json(indent=2)


def smart_completion_status(status, snapshot, current_revision=0):
    """Annotate Smart analysis with the revision it actually analyzed."""
    if not isinstance(snapshot, dict):
        return status
    snapshot_revision = _coerce_settings_revision(snapshot.get("settings_revision", 0))
    current = _coerce_settings_revision(current_revision)
    suffix = f" Smart analysis settings revision {snapshot_revision}."
    if current > snapshot_revision:
        suffix += " Settings changed during analysis; proposal remains unapplied."
    return f"{status}{suffix}"


def process_image_event(*args, request=None):
    """UI wrapper that preserves ``process_image``'s legacy eight outputs."""
    current_revision = None
    snapshot = None
    preview_cache = None
    if isinstance(args[0], dict) and "process_args" in args[0]:
        snapshot = args[0]
        process_args = tuple(snapshot.get("process_args") or ())
        snapshot_revision = snapshot.get("settings_revision", 0)
        render_mode = snapshot.get("render_mode")
        preserve_source_profile = bool(snapshot.get("preserve_source_profile", False))
        if len(args) > 1:
            current_revision = args[1]
        if len(args) > 2 and isinstance(args[2], GuiPreviewCache):
            preview_cache = args[2]
    else:
        process_args = args[:len(PROCESS_INPUT_KEYS)]
        snapshot_revision = args[len(PROCESS_INPUT_KEYS)] if len(args) > len(PROCESS_INPUT_KEYS) else 0
        render_mode = None
        preserve_source_profile = False
        if len(args) > len(PROCESS_INPUT_KEYS) + 1:
            current_revision = args[len(PROCESS_INPUT_KEYS) + 1]
    current_for_contract = (
        _coerce_settings_revision(snapshot_revision)
        if current_revision is None
        else _coerce_settings_revision(current_revision)
    )
    def _contract_failure(message):
        result = (None, gr.update(visible=False), None, None, message, None, gr.update(visible=False), "")
        return result + (preview_cache,) if preview_cache is not None else result
    if isinstance(snapshot, dict) and snapshot.get("render_contract"):
        contract_validation = validate_render_contract(snapshot["render_contract"])
        if not contract_validation["valid"]:
            message = "Render contract rejected: " + ", ".join(contract_validation["errors"])
            return _contract_failure(message)
        if (
            render_mode == MODE_EXPORT_FULL_QUALITY
            and current_for_contract > _coerce_settings_revision(snapshot_revision)
        ):
            return _contract_failure(
                "Export Full Quality cancelled: settings changed before rendering started."
            )
    process_kwargs = {}
    if render_mode in {MODE_RENDER_PREVIEW, MODE_EXPORT_FULL_QUALITY}:
        process_kwargs["render_mode"] = render_mode
    if preview_cache is not None:
        process_kwargs["preview_cache"] = preview_cache
    if isinstance(snapshot, dict) and preserve_source_profile:
        process_kwargs["preserve_source_profile"] = preserve_source_profile
    workspace = _workspace_for_request(request)
    if workspace is not None:
        process_kwargs["workspace"] = workspace
    result = process_image(*process_args, **process_kwargs)
    if preview_cache is not None and isinstance(result, tuple) and len(result) >= 8:
        latest = preview_cache.latest_render_evidence
        latest["render_revision"] = _coerce_settings_revision(snapshot_revision)
        latest["render_mode"] = render_mode or MODE_RENDER_PREVIEW
        if isinstance(snapshot, dict) and snapshot.get("render_contract"):
            settings = snapshot["render_contract"].get("settings", {})
            latest["settings_sha256"] = settings.get("settings_sha256")
        preview_cache.set_latest_render(
            latest,
            face_contexts=preview_cache.latest_render_face_contexts,
        )
    if isinstance(result, tuple) and len(result) > 4 and result[0] is not None:
        result = list(result)
        result[4] = render_status_for_revision(
            result[4], snapshot_revision,
            current_for_contract,
        )
        result = tuple(result)
    if preview_cache is not None and isinstance(result, tuple) and len(result) == 8:
        return result + (preview_cache,)
    return result


def recipe_defaults(recipe_name):
    """Return the per-slider defaults for a recipe.

    Delegates to ``retouch.params.recipe_to_params`` so there is exactly
    one place that knows how to turn a recipe dict into a UI-side values
    bag.  The engine uses the same spec list from a different code path
    (``engine.build_context``).
    """
    return recipe_to_params(recipe_name)


def get_custom_style_names():
    styles = list_styles()
    return [s["name"] for s in styles]


def apply_custom_style(style_name, current_recipe="natural"):
    if not style_name:
        return tuple([gr.update()] * len(_recipe_outputs))

    styles = list_styles()
    target = None
    for s in styles:
        if s["name"] == style_name:
            target = s
            break

    if not target:
        return tuple([gr.update()] * len(_recipe_outputs))
        
    p_dict = target["profile"]
    profile = StyleProfile(**p_dict)
    
    d = recipe_defaults(current_recipe)
    
    d["smooth"] = int(np.clip(profile.skin_smooth_strength * 100.0, 0.0, 100.0))
    d["whiten"] = int(np.clip(profile.skin_l_mean_delta * 4.0, 0.0, 100.0))
    d["mid_reduction"] = float(np.clip(profile.skin_mid_reduction, 0.0, 1.0))
    d["texture_opacity"] = float(np.clip(profile.skin_texture_opacity, 0.0, 1.0))
    d["contrast"] = int(profile.contrast_delta)
    d["brightness"] = int(np.clip(profile.brightness_delta, -50.0, 50.0))

    return tuple(d[k] for k in RECIPE_OUTPUT_KEYS)


# PROCESS_INPUT_KEYS — the ordered list of inputs the process_image() Gradio
# event handler expects.  Generated from PROCESSING_PARAMS (in declaration
# order) so the slider order stays in lock-step with the spec, plus the
# fixed-prefix transport / session keys at the end.
PROCESS_INPUT_KEYS = (
    ["img_paths", "recipe"]
    + [n for n in param_names() if n not in ("color_transfer_intensity", "freckle_preserve_mask")]
    + [
        "color_ref_img", "color_ref_strength",
        "show_compare", "fast",
        "export_fmt", "export_quality", "export_res",
        "quality_tier",
        "debug_mode",
        "optical_correction",
        "look_params",
        "face_params",
        "face_params_json",
    ]
)

def _coerce_float(value: object, default: float = 0.0) -> float:
    """Safely coerce an arbitrary Gradio value to float, falling back on parse failure.

    Slider components always return ``float``, but transient states (watchfiles
    reload races, empty Form payloads, deprecated field names mapped onto the
    same key) can deliver strings or ``None`` here.  Prefer the numeric value
    if one can be parsed; otherwise fall back to *default* rather than 500-ing
    the whole request.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    if value is None:
        return default
    return default


# ---------------------------------------------------------------------------
# F2: Session handlers
# ---------------------------------------------------------------------------

def save_session_handler(*args):
    """Save current params as a session JSON file for download."""
    from retouch.session import create_session_from_params
    import tempfile, os

    process_args = args[:len(PROCESS_INPUT_KEYS)]
    advanced_edits = args[len(PROCESS_INPUT_KEYS)] if len(args) > len(PROCESS_INPUT_KEYS) else []
    advanced_base = args[len(PROCESS_INPUT_KEYS) + 1] if len(args) > len(PROCESS_INPUT_KEYS) + 1 else None
    advanced_history = args[len(PROCESS_INPUT_KEYS) + 2] if len(args) > len(PROCESS_INPUT_KEYS) + 2 else None
    advanced_current = args[len(PROCESS_INPUT_KEYS) + 3] if len(args) > len(PROCESS_INPUT_KEYS) + 3 else None
    params = dict(zip(PROCESS_INPUT_KEYS, process_args))
    img_paths = params.get("img_paths")
    image_path = None
    if img_paths and isinstance(img_paths, list) and len(img_paths) > 0:
        if isinstance(img_paths[0], str):
            image_path = img_paths[0]
        elif hasattr(img_paths[0], "name"):
            image_path = img_paths[0].name

    recipe = params.get("recipe", "natural")
    session = create_session_from_params(params, recipe=recipe, image_path=image_path)
    if advanced_edits or advanced_base is not None:
        try:
            session.advanced_retouch = build_advanced_session_payload(
                list(advanced_edits or []),
                advanced_base,
                history_state=advanced_history,
                result_rgb=advanced_current,
            )
        except AdvancedContractError as exc:
            _logger.warning("Advanced Retouch session save refused: %s", exc)
            gr.Warning(f"Session was not saved: {exc}")
            return gr.update(value=None, visible=False)

    tmpdir = tempfile.mkdtemp()
    filepath = os.path.join(tmpdir, f"{recipe or 'session'}.session.json")
    session.to_file(filepath)

    return gr.update(value=filepath, visible=True)


def load_session_handler(session_file, *current_args):
    """Load a session JSON file and return params as a tuple for Gradio."""
    from retouch.session import Session
    import logging

    if session_file is None:
        return current_args

    try:
        if hasattr(session_file, "name"):
            path = session_file.name
        else:
            path = str(session_file)
        session = Session.from_file(path)
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to load session: {e}")
        gr.Warning(f"Failed to load session: {e}")
        return current_args

    gr.Info(f"Loaded session (recipe: {session.recipe or 'unknown'})")

    result = list(current_args)
    for i, key in enumerate(PROCESS_INPUT_KEYS):
        if key in session.params:
            val = session.params[key]
            if val is not None:
                # Session files are shared between users: never let a loaded
                # JSON steer reads/writes at local paths (img_paths etc.).
                if "path" in key:
                    continue
                if i < len(result):
                    result[i] = val
    return tuple(result)


def load_advanced_session_handler(session_file, source_rgb, base_contract=None):
    """Restore serialized Advanced Retouch edits when a source is available."""
    from retouch.session import Session

    if session_file is None:
        return (
            source_rgb,
            [],
            source_rgb,
            source_rgb,
            None,
            before_after(source_rgb, source_rgb) if source_rgb is not None else None,
            AdvancedHistory().to_state(),
            [],
            None,
            "",
        )
    try:
        path = session_file.name if hasattr(session_file, "name") else str(session_file)
        session = Session.from_file(path)
        raw_advanced = session.advanced_retouch or {}
        if not raw_advanced:
            return (
                source_rgb,
                [],
                source_rgb,
                source_rgb,
                None,
                before_after(source_rgb, source_rgb) if source_rgb is not None else None,
                AdvancedHistory().to_state(),
                [],
                None,
                "Session contains no Advanced Retouch edits.",
            )
        advanced = parse_advanced_session_payload(raw_advanced)
        edits = list(advanced.get("edits") or [])
    except Exception as exc:
        _logger.warning("Failed to load Advanced Retouch session data: %s", exc)
        return (
            source_rgb,
            [],
            source_rgb,
            source_rgb,
            None,
            before_after(source_rgb, source_rgb) if source_rgb is not None else None,
            AdvancedHistory().to_state(),
            [],
            None,
            f"Advanced Retouch session data unavailable: {exc}",
        )

    if advanced.get("legacy_unverified"):
        return (
            source_rgb,
            advanced,
            source_rgb,
            source_rgb,
            None,
            before_after(source_rgb, source_rgb) if source_rgb is not None else None,
            AdvancedHistory().to_state(),
            [],
            None,
            f"Loaded {len(edits)} legacy Advanced Retouch edit(s) as pending. "
            "Legacy sessions have no base hash; use Bind legacy session edits to apply them explicitly.",
        )
    if source_rgb is None or base_contract is None:
        return (
            source_rgb,
            advanced,
            source_rgb,
            None,
            None,
            None,
            _advanced_history_from_edits(edits).to_state(),
            edits,
            None,
            f"Loaded {len(edits)} Advanced Retouch edit(s) as pending. "
            "Load the exact recorded base to verify and replay them.",
        )
    comparison = compare_base_contracts(advanced.get("base"), base_contract)
    if not comparison["matches"]:
        return (
            source_rgb,
            advanced,
            source_rgb,
            source_rgb,
            None,
            before_after(source_rgb, source_rgb),
            AdvancedHistory().to_state(),
            [],
            None,
            "Advanced Retouch session base mismatch; edits remain pending ({}).".format(
                ", ".join(comparison["reasons"])
            ),
        )
    try:
        current = _replay_advanced_state(source_rgb, edits)
        replay_check = replay_result_matches(advanced, current)
        if not replay_check["matches"]:
            raise AdvancedContractError(
                "replay result mismatch: %s" % ", ".join(replay_check["reasons"])
            )
        return (
            source_rgb,
            [],
            current,
            current,
            None,
            before_after(source_rgb, current),
            _advanced_history_from_edits(edits).to_state(),
            edits,
            None,
            f"Verified and replayed {len(edits)} Advanced Retouch edit(s) on the recorded base.",
        )
    except Exception as exc:
        _logger.warning("Failed to replay Advanced Retouch session data: %s", exc)
        return (
            source_rgb,
            advanced,
            source_rgb,
            source_rgb,
            None,
            before_after(source_rgb, source_rgb),
            _advanced_history_from_edits(edits).to_state(),
            [],
            None,
            f"Could not verify Advanced Retouch replay; edits remain pending: {exc}",
        )


def _history_button_updates(undo_stack):
    """Derive button state from the session history cursor."""
    if undo_stack is None:
        return gr.update(interactive=False), gr.update(interactive=False)
    return (
        gr.update(interactive=bool(undo_stack.can_undo)),
        gr.update(interactive=bool(undo_stack.can_redo)),
    )


def _history_args(args, undo_stack=None):
    """Split process parameters from the optional session history state."""
    process_args = tuple(args[:len(PROCESS_INPUT_KEYS)])
    if undo_stack is None and len(args) > len(PROCESS_INPUT_KEYS):
        undo_stack = args[len(PROCESS_INPUT_KEYS)]
    return process_args, undo_stack


def record_history(*args):
    """Record the post-mutation settings in the session-owned history.

    The callback is intentionally small and can run unqueued.  It receives the
    updated component values only after a mutation event's ``success`` chain,
    or after a slider's ``release`` event, so the history cursor never stores
    the pre-mutation values.
    """
    from retouch.session import UndoRedoStack

    process_args, undo_stack = _history_args(args)
    params = dict(zip(PROCESS_INPUT_KEYS, process_args))
    if not isinstance(undo_stack, UndoRedoStack):
        undo_stack = UndoRedoStack()
    undo_stack._last_history_change = False
    if undo_stack.current() != params:
        undo_stack.push(params)
    undo_btn_update, redo_btn_update = _history_button_updates(undo_stack)
    return undo_stack, undo_btn_update, redo_btn_update


def initialize_history(*args):
    """Seed a new browser session with its baseline settings."""
    return record_history(*args, None)


def push_undo_handler(*args, undo_stack=None):
    """Backward-compatible alias for recording a post-mutation state."""
    if undo_stack is not None:
        args = tuple(args[:len(PROCESS_INPUT_KEYS)]) + (undo_stack,)
    return record_history(*args)


def _skip_process_inputs():
    """Preserve all processing controls when history cannot move."""
    return tuple(gr.skip() for _ in PROCESS_INPUT_KEYS)


def undo_handler(undo_stack):
    """Undo settings, preserving controls when the cursor is at the start."""
    if undo_stack is None:
        undo_btn_update, redo_btn_update = _history_button_updates(None)
        return (*_skip_process_inputs(), gr.skip(), undo_btn_update, redo_btn_update)
    params = undo_stack.undo()
    undo_stack._last_history_change = params is not None
    if params is None:
        undo_btn_update, redo_btn_update = _history_button_updates(undo_stack)
        return (*_skip_process_inputs(), gr.skip(), undo_btn_update, redo_btn_update)
    result = tuple(params.get(k) for k in PROCESS_INPUT_KEYS)
    undo_btn_update, redo_btn_update = _history_button_updates(undo_stack)
    return (*result, undo_stack, undo_btn_update, redo_btn_update)


def redo_handler(undo_stack):
    """Redo settings, preserving controls when the cursor is at the end."""
    if undo_stack is None:
        undo_btn_update, redo_btn_update = _history_button_updates(None)
        return (*_skip_process_inputs(), gr.skip(), undo_btn_update, redo_btn_update)
    params = undo_stack.redo()
    undo_stack._last_history_change = params is not None
    if params is None:
        undo_btn_update, redo_btn_update = _history_button_updates(undo_stack)
        return (*_skip_process_inputs(), gr.skip(), undo_btn_update, redo_btn_update)
    result = tuple(params.get(k) for k in PROCESS_INPUT_KEYS)
    undo_btn_update, redo_btn_update = _history_button_updates(undo_stack)
    return (*result, undo_stack, undo_btn_update, redo_btn_update)


def revision_after_history(current_revision, undo_stack):
    """Advance the draft only when Undo/Redo actually moved the cursor."""
    if getattr(undo_stack, "_last_history_change", False):
        return on_settings_changed(current_revision)
    return _coerce_settings_revision(current_revision), gr.skip()


def save_snapshot_handler(name, *args, snapshots=None):
    """Save a named snapshot of current params."""
    from retouch.session import Session, Snapshot

    if not name or not name.strip():
        gr.Warning("Please enter a snapshot name")
        return gr.update(), snapshots or {}

    process_args = args[:len(PROCESS_INPUT_KEYS)]
    advanced_edits = args[len(PROCESS_INPUT_KEYS)] if len(args) > len(PROCESS_INPUT_KEYS) else []
    advanced_base = args[len(PROCESS_INPUT_KEYS) + 1] if len(args) > len(PROCESS_INPUT_KEYS) + 1 else None
    advanced_history = args[len(PROCESS_INPUT_KEYS) + 2] if len(args) > len(PROCESS_INPUT_KEYS) + 2 else None
    advanced_current = args[len(PROCESS_INPUT_KEYS) + 3] if len(args) > len(PROCESS_INPUT_KEYS) + 3 else None
    if snapshots is None and len(args) > len(PROCESS_INPUT_KEYS) + 4:
        snapshots = args[len(PROCESS_INPUT_KEYS) + 4]
    params = dict(zip(PROCESS_INPUT_KEYS, process_args))
    recipe = params.get("recipe", "natural")
    session = Session(recipe=recipe, params=params)
    if advanced_edits or advanced_base is not None:
        try:
            session.advanced_retouch = build_advanced_session_payload(
                list(advanced_edits or []),
                advanced_base,
                history_state=advanced_history,
                result_rgb=advanced_current,
            )
        except AdvancedContractError as exc:
            gr.Warning(f"Snapshot was not saved: {exc}")
            return gr.update(), snapshots or {}
    snap = Snapshot(name=name.strip(), session=session)

    if snapshots is None:
        snapshots = {}
    snapshots[name.strip()] = snap

    gr.Info(f"Snapshot '{name.strip()}' saved")
    return gr.update(choices=list(snapshots.keys())), snapshots


def compare_snapshot_handler(selected_name, *args, snapshots=None):
    """Inspect the serialized settings for a saved snapshot.

    Standard snapshots currently contain settings, not a second rendered
    image.  The UI therefore calls this action ``Inspect Settings`` rather
    than implying an image comparison that it does not perform.
    """
    if snapshots is None and len(args) > len(PROCESS_INPUT_KEYS):
        snapshots = args[len(PROCESS_INPUT_KEYS)]
    if not selected_name or snapshots is None or selected_name not in snapshots:
        gr.Warning("Select a snapshot to compare")
        return None

    snap = snapshots[selected_name]
    gr.Info(f"Inspecting settings for snapshot '{selected_name}' (recipe: {snap.session.recipe})")
    return snap.session.to_json()


def process_image(
    *args,
    render_mode=None,
    preview_cache=None,
    preserve_source_profile=False,
    workspace=None,
):
    params = dict(zip(PROCESS_INPUT_KEYS, args))
    img_paths = params.get("img_paths")
    recipe = params.get("recipe")
    color_ref_img = params.get("color_ref_img")
    color_ref_strength = _coerce_float(params.get("color_ref_strength"))
    show_compare = params.get("show_compare")
    fast = params.get("fast")
    preview_only = render_mode == MODE_RENDER_PREVIEW
    full_quality_export = render_mode == MODE_EXPORT_FULL_QUALITY
    if preview_only:
        fast = True
    elif full_quality_export:
        fast = False
    export_fmt = params.get("export_fmt")
    export_quality = params.get("export_quality")
    export_res = params.get("export_res")
    debug_mode = params.get("debug_mode")
    quality_tier = params.get("quality_tier")
    optical_correction = bool(params.get("optical_correction"))
    preserve_source_profile = bool(preserve_source_profile)
    quality = "draft" if quality_tier and quality_tier.startswith("Draft") else "full"
    if full_quality_export:
        quality = "full"

    if not img_paths:
        return None, gr.update(visible=False), None, None, "Please upload an image first.", None, gr.update(visible=False), ""

    if full_quality_export and isinstance(img_paths, list) and len(img_paths) != 1:
        return (
            None,
            gr.update(visible=False),
            None,
            None,
            "Export Full Quality accepts one image. Use Export All in the Batch workflow for a shoot.",
            None,
            gr.update(visible=False),
            "",
        )

    if not isinstance(img_paths, list):
        img_paths = [img_paths]

    gr.Info(f"Processing {len(img_paths)} image(s)...")

    exported_paths = []
    first_result_rgb = None
    first_combined = None
    first_original = None
    first_result = None
    first_processed_rgb = None
    first_color_context = None
    debug_images = []
    capture_notes = []
    successful_count = 0

    color_ref_bgr = None
    if color_ref_img is not None and color_ref_strength > 0:
        # Defensive check: only process if color_ref_img is a valid file path (not bool/invalid type)
        if isinstance(color_ref_img, dict):
            color_ref_img = color_ref_img.get("name") or color_ref_img.get("path")
        if color_ref_img and (isinstance(color_ref_img, (str, bytes)) or hasattr(color_ref_img, '__fspath__')):
            try:
                color_ref_bgr = imread_exif(color_ref_img)
            except (TypeError, FileNotFoundError) as e:
                _logger.warning("Failed to load color reference image: %s", e)

    engine = get_engine()
    runtime_note = _face_runtime_label(getattr(engine, "_detector", None))
    start = time.time()

    # Legacy direct callers retain the old temporary fallback. GUI requests
    # use a session-owned request workspace so cleanup cannot touch another
    # browser session's files.
    if workspace is None:
        try:
            temp_root = Path(tempfile.gettempdir())
            now = time.time()
            for p in temp_root.glob("retouch_tmp_*"):
                if p.is_dir() and (now - p.stat().st_mtime > TEMP_CLEANUP_AGE_SEC):
                    shutil.rmtree(p, ignore_errors=True)
            for p in temp_root.glob("retouch_export_*.zip"):
                if p.is_file() and (now - p.stat().st_mtime > TEMP_CLEANUP_AGE_SEC):
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception as e:
            _logger.warning("Temp directory cleanup warning: %s", e)
        temp_dir = tempfile.mkdtemp(prefix="retouch_tmp_")
    else:
        temp_dir = str(workspace.request_workspace("render"))
    debug_dir = os.path.join(temp_dir, "debug") if debug_mode else None
    qa_warnings = []
    qa_html = ""

    # Translate the GUI-side values dict into the engine-side kwargs dict.
    # The spec list (in retouch.params) is the source of truth for the
    # unit conversions (e.g. grain × 500 → grain / 500, "none" → None,
    # grade_intensity / 100, etc.).
    engine_kwargs = gui_values_to_engine_kwargs(
        params,
        extra={
            "recipe": recipe,
            "color_ref": color_ref_bgr,
            "color_transfer_intensity": color_ref_strength,
            "fast": fast,
            "debug_dir": None,  # set per-image below
        },
    )

    # F6: overlay extracted-look params over recipe defaults (look wins).
    look_params = params.get("look_params")
    if isinstance(look_params, dict):
        for k, v in look_params.items():
            if str(k).startswith("_"):
                continue
            engine_kwargs[k] = v

    # Per-face overrides: State dict wins; JSON textbox is advanced fallback.
    from retouch.face_params import coerce_face_params
    fp = params.get("face_params")
    if not fp:
        face_params_json = params.get("face_params_json") or ""
        if isinstance(face_params_json, str) and face_params_json.strip():
            try:
                import json as _json
                fp = _json.loads(face_params_json)
            except Exception as e:
                _logger.warning("face_params_json parse failed: %s", e)
                fp = None
    if fp:
        coerced = coerce_face_params(fp)
        if coerced:
            engine_kwargs["face_params"] = coerced

    for idx, path_item in enumerate(img_paths):
        try:
            curr_path = path_item
            if isinstance(path_item, dict):
                curr_path = path_item.get("name") or path_item.get("path")

            cache_key = None
            cache_entry = None
            cache_hit = False
            color_context = None
            identity = None
            if isinstance(preview_cache, GuiPreviewCache):
                try:
                    identity = source_identity(curr_path)
                    engine_for_cache = get_engine()
                    cache_key = make_preview_cache_key(
                        identity,
                        "fast-800" if fast else "native",
                        geometry={"quality": quality, "fast": bool(fast)},
                        optical_correction=bool(optical_correction),
                        color_contract=_source_color_cache_descriptor(curr_path),
                        bit_depth=(
                            "float32"
                            if Path(curr_path).suffix.lower() in RAW_EXTENSIONS
                            else "uint8"
                        ),
                        raw_settings={"prefer_16bit": True, "raw_decoder": "rawpy"},
                        detector_backend=_detector_cache_descriptor(engine_for_cache),
                        engine_version=__version__,
                    )
                    cache_entry = preview_cache.get(cache_key)
                except Exception as exc:
                    _logger.debug("Preview cache key unavailable for %s: %s", curr_path, exc)
            
            # T5: RAW via 16-bit path; JPEG/PNG unchanged (imread_engine).
            # The caller-owned status dict makes an optional Lensfun fallback
            # or non-8-bit precision-preserving skip visible in this GUI.
            correction_status = {}
            if (
                cache_entry is not None
                and cache_entry.decoded_source is not None
                and isinstance(cache_entry.runtime_color_context, ColorContext)
            ):
                img_bgr = np.asarray(cache_entry.decoded_source).copy()
                color_context = cache_entry.runtime_color_context
                cache_hit = True
                correction_status["cached"] = True
            else:
                img_bgr, color_context = imread_engine_with_context(
                    curr_path,
                    optical_correction=optical_correction,
                    correction_status=correction_status,
                )
            if optical_correction:
                detail = correction_status.get("reason") or ", ".join(correction_status.get("applied", ()))
                capture_notes.append(f"{Path(curr_path).name}: {detail or 'no correction applied'}")
            original = (
                np.clip(img_bgr, 0, 255).astype(np.uint8)
                if img_bgr.dtype != np.uint8
                else img_bgr.copy()
            )

            engine_kwargs["debug_dir"] = (
                debug_dir if (first_result_rgb is None and first_combined is None) else None
            )

            engine_kwargs["quality"] = quality
            if cache_hit and cache_entry.runtime_face_contexts:
                engine_kwargs["face_contexts"] = list(cache_entry.runtime_face_contexts)
            else:
                engine_kwargs.pop("face_contexts", None)

            result = engine.process(img_bgr, **engine_kwargs)
            if cache_key is not None:
                try:
                    runtime_contexts = list(getattr(result, "face_contexts", None) or [])
                    preview_cache.put(
                        cache_key,
                        decoded_source=img_bgr,
                        decoded_source_metadata={
                            "path": str(curr_path),
                            "width": int(img_bgr.shape[1]),
                            "height": int(img_bgr.shape[0]),
                            "dtype": str(img_bgr.dtype),
                            "cached": bool(cache_hit),
                        },
                        face_contexts=runtime_contexts,
                        runtime_face_contexts=runtime_contexts,
                        color_context=color_context,
                        runtime_color_context=color_context,
                    )
                except Exception as exc:
                    _logger.warning("Preview cache store skipped for %s: %s", curr_path, exc)
            qa_warnings = getattr(result, 'qa', [])

            if (
                isinstance(preview_cache, GuiPreviewCache)
                and first_result_rgb is None
                and first_combined is None
            ):
                runtime_contexts = list(getattr(result, "face_contexts", None) or [])
                runtime_diagnostics = getattr(result, "runtime_diagnostics", {}) or {}
                preview_cache.set_latest_render(
                    {
                        "source_path": str(curr_path),
                        "native": {
                            "width": int(img_bgr.shape[1]),
                            "height": int(img_bgr.shape[0]),
                        },
                        "output": {
                            "width": int(result.shape[1]),
                            "height": int(result.shape[0]),
                            "dtype": str(result.dtype),
                        },
                        "render_revision": 0,
                        "cache_status": "hit" if cache_hit else "miss",
                        "color_context": (
                            color_context.to_dict()
                            if isinstance(color_context, ColorContext)
                            else {}
                        ),
                        "metadata_result": {
                            "source_icc": bool(
                                isinstance(color_context, ColorContext)
                                and color_context.is_tagged
                            ),
                            "source_profile_preserved": bool(preserve_source_profile),
                            "c2pa": "copied-unverified-passthrough",
                        },
                        "face_count": int(getattr(result, "face_count", 0)),
                        "timings_ms": dict(getattr(result, "timings", {}) or {}),
                        "qa": {
                            "warning_count": len(qa_warnings),
                            "warnings": [
                                getattr(item, "message", str(item))
                                for item in qa_warnings
                            ],
                        },
                        "safe_auto_decisions": list(
                            getattr(result, "safe_auto_decisions", []) or []
                        ),
                        "backend": runtime_diagnostics,
                        "precision": getattr(result, "precision_metadata", {}),
                    },
                    face_contexts=runtime_contexts,
                )

            if first_result_rgb is None and first_combined is None:
                first_original = original
                first_result = result
                first_color_context = color_context
                first_result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
                first_processed_rgb = first_result_rgb.copy()
                if show_compare:
                    h = min(original.shape[0], result.shape[0])
                    sep = np.full((h, COMPARE_SEPARATOR_WIDTH, 3), COMPARE_SEPARATOR_COLOR, dtype=np.uint8)
                    orig_rgb = cv2.cvtColor(original[:h], cv2.COLOR_BGR2RGB)
                    res_rgb = first_result_rgb[:h]
                    combined = np.hstack([orig_rgb, sep, res_rgb])
                    max_h = PREVIEW_MAX_HEIGHT
                    if combined.shape[0] > max_h:
                        scale = max_h / combined.shape[0]
                        new_w = int(combined.shape[1] * scale)
                        combined = cv2.resize(combined, (new_w, max_h), interpolation=cv2.INTER_AREA)
                    first_combined = combined
                else:
                    if first_result_rgb.shape[0] > PREVIEW_MAX_HEIGHT:
                        scale = PREVIEW_MAX_HEIGHT / first_result_rgb.shape[0]
                        new_w = int(first_result_rgb.shape[1] * scale)
                        first_result_rgb = cv2.resize(first_result_rgb, (new_w, PREVIEW_MAX_HEIGHT), interpolation=cv2.INTER_AREA)

                if debug_mode and debug_dir and os.path.isdir(debug_dir):
                    mask_files = [
                        ("Skin Mask", "skin_mask.png"),
                        ("Skin+Hair Mask", "skin_hair_mask.png"),
                        ("Lips Mask", "lips_mask.png"),
                        ("Sharpen Mask", "sharpen_mask.png"),
                        ("Glow Mask", "glow_mask.png"),
                        ("Freq Low", "freq_low.png"),
                        ("Freq Mid", "freq_mid.png"),
                        ("Freq High", "freq_high.png"),
                    ]
                    for label, fname in mask_files:
                        mpath = os.path.join(debug_dir, fname)
                        if os.path.exists(mpath):
                            mask_img = cv2.imread(mpath)
                            if mask_img is not None:
                                debug_images.append((cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB), label))

            # Render Preview intentionally stops at the in-memory preview. It
            # never creates an export file or ZIP as a side effect.
            if preview_only:
                successful_count += 1
                break

            export_img = result
            export_max = EXPORT_RES_MAP.get(export_res)
            if export_max is not None:
                h, w = export_img.shape[:2]
                if max(h, w) > export_max:
                    scale = export_max / max(h, w)
                    export_img = cv2.resize(export_img, (int(w * scale), int(h * scale)),
                                            interpolation=cv2.INTER_AREA)

            ext = EXT_MAP.get(export_fmt, ".jpg")
            filename = Path(curr_path).stem
            if workspace is not None:
                out_path = str(
                    workspace.allocate_artifact(
                        f"{filename}_{idx:03d}_retouched",
                        ext,
                        request_dir=temp_dir,
                    )
                )
            else:
                out_path = os.path.join(temp_dir, f"{filename}_{idx:03d}_retouched{ext}")
            write_kwargs = {
                "exif": read_exif_bytes(curr_path),
                "c2pa_manifest": read_c2pa_manifest(curr_path),
                "quality": export_quality,
            }
            if isinstance(color_context, ColorContext):
                write_image_with_color_context(
                    out_path,
                    export_img,
                    color_context,
                    preserve_source_profile=preserve_source_profile,
                    bit_depth=16 if export_fmt == "PNG-16" else 8,
                    **write_kwargs,
                )
            elif export_fmt == "PNG-16":
                # Fail-safe compatibility for an old cache entry without a
                # color context: label/metadata remain explicit, but do not
                # attach a possibly incorrect source ICC profile.
                write_image_with_icc(
                    out_path,
                    export_img,
                    icc_profile=None,
                    bit_depth=16,
                    **write_kwargs,
                )
            else:
                write_image_with_icc(
                    out_path,
                    export_img,
                    icc_profile=None,
                    bit_depth=8,
                    **write_kwargs,
                )
            exported_paths.append(out_path)
            successful_count += 1

        except Exception as e:
            _logger.exception("Failed to process %s", path_item)
            from retouch.utils import log_crash
            crash_path = log_crash(e, {
                "recipe": recipe,
                "image_path": str(curr_path),
                "show_compare": show_compare,
                "fast": fast
            })
            if crash_path:
                _logger.info("Crash details saved to: %s", crash_path)

    if successful_count == 0:
        gr.Warning("No images were successfully processed.")
        return None, gr.update(visible=False), None, None, "Error: No images were successfully processed.", None, gr.update(visible=False), qa_html

    preview = first_combined if show_compare else first_result_rgb
    debug_gallery = debug_images if debug_images else None
    debug_vis = gr.update(visible=bool(debug_images))

    qa_html = ""
    if qa_warnings:
        items = "".join(f'<li>⚠️ {w.message} (score: {w.score:.2f})</li>' for w in qa_warnings)
        qa_html = f'<div style="background:#fff3cd;border:1px solid #ffc107;padding:8px 12px;border-radius:6px;margin:8px 0;font-size:13px"><strong>Quality Warnings:</strong><ul style="margin:4px 0 0 16px;padding:0">{items}</ul></div>'

    elapsed = time.time() - start
    slide_html = _make_comparison_html(first_original, first_result) if (first_original is not None and first_result is not None) else ""
    delivery_notes = []
    precision = getattr(first_result, "precision", None)
    if precision is not None:
        if getattr(precision, "downgraded", False):
            delivery_notes.append(
                "Precision: 8-bit processed data; PNG-16 is a 16-bit container"
            )
        else:
            delivery_notes.append(
                f"Precision: {getattr(precision, 'precision_status', 'unknown')}"
            )
    if isinstance(first_color_context, ColorContext):
        if first_color_context.is_tagged:
            delivery_notes.append(
                "Color: embedded ICC converted to working sRGB"
                + ("; source profile preserved on export" if preserve_source_profile else "")
            )
        else:
            delivery_notes.append("Color: untagged input assumed sRGB")

    def _append_delivery_notes(message):
        if capture_notes:
            message += " | Lens: " + " ; ".join(capture_notes)
        if delivery_notes:
            message += " | " + " ; ".join(delivery_notes)
        return message

    if preview_only:
        message = f"Preview rendered in {elapsed:.1f}s ✓ | {runtime_note}"
        message = _append_delivery_notes(message)
        if show_compare:
            return gr.update(visible=False), gr.update(value=slide_html, visible=True), first_processed_rgb, None, message, debug_gallery, debug_vis, qa_html
        return first_processed_rgb, gr.update(visible=False), first_processed_rgb, None, message, debug_gallery, debug_vis, qa_html

    if len(exported_paths) > 1:
        zip_stamp = time.strftime("%Y%m%d_%H%M%S")
        if workspace is not None:
            zip_path = str(
                workspace.allocate_artifact(
                    f"retouch_export_{zip_stamp}",
                    ".zip",
                    request_dir=temp_dir,
                )
            )
        else:
            # Legacy fallback: mkstemp (O_EXCL) — a predictable stamped name
            # in the shared tmpdir is a symlink-clobber race on multi-user
            # hosts.
            fd, zip_tmp = tempfile.mkstemp(
                prefix=f"retouch_export_{zip_stamp}_", suffix=".zip"
            )
            os.close(fd)
            zip_path = zip_tmp
        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for exp_path in exported_paths:
                zipf.write(exp_path, arcname=os.path.basename(exp_path))
        gr.Info(f"Processed {len(exported_paths)}/{len(img_paths)} images in {elapsed:.1f}s")

        if show_compare:
            message = f"Processed {len(exported_paths)}/{len(img_paths)} images in {elapsed:.1f}s ✓ | {runtime_note}"
            message = _append_delivery_notes(message)
            return gr.update(visible=False), gr.update(value=slide_html, visible=True), first_processed_rgb, zip_path, message, debug_gallery, debug_vis, qa_html
        message = f"Processed {len(exported_paths)}/{len(img_paths)} images in {elapsed:.1f}s ✓ | {runtime_note}"
        message = _append_delivery_notes(message)
        return preview, gr.update(visible=False), first_processed_rgb, zip_path, message, debug_gallery, debug_vis, qa_html
    else:
        gr.Info(f"Done in {elapsed:.1f}s")

        if show_compare:
            message = f"Done in {elapsed:.1f}s ✓ | {runtime_note}"
            message = _append_delivery_notes(message)
            return gr.update(visible=False), gr.update(value=slide_html, visible=True), first_processed_rgb, exported_paths[0], message, debug_gallery, debug_vis, qa_html
        message = f"Done in {elapsed:.1f}s ✓ | {runtime_note}"
        message = _append_delivery_notes(message)
        return preview, gr.update(visible=False), first_processed_rgb, exported_paths[0], message, debug_gallery, debug_vis, qa_html


def export_all_handler(*args):
    """Direct multi-image final delivery to the existing Batch workflow."""
    process_args = tuple(args[:len(PROCESS_INPUT_KEYS)])
    current_revision = args[len(PROCESS_INPUT_KEYS)] if len(args) > len(PROCESS_INPUT_KEYS) else 0
    preserve_source_profile = bool(
        args[len(PROCESS_INPUT_KEYS) + 1]
        if len(args) > len(PROCESS_INPUT_KEYS) + 1
        else False
    )
    params = dict(zip(PROCESS_INPUT_KEYS, process_args))
    paths = _render_contract_paths(params.get("img_paths"))
    if not paths:
        return "Export All: upload one or more images first."
    try:
        settings = {
            key: _render_contract_value(value)
            for key, value in params.items()
            if key != "img_paths"
        }
        settings["render_mode"] = MODE_EXPORT_ALL
        settings["fast"] = False
        settings["quality_tier"] = "Full (native face crops)"
        settings["preserve_source_profile"] = preserve_source_profile
        settings_snapshot = capture_settings_snapshot(
            settings,
            _coerce_settings_revision(current_revision),
        )
        contract = build_export_all(
            paths,
            settings_snapshot,
            _coerce_settings_revision(current_revision),
        )
        return (
            f"Export All is ready for the Batch workflow ({len(paths)} image(s), "
            f"Draft revision {_coerce_settings_revision(current_revision)}). "
            "Open Batch, choose the output settings, and run the final job. "
            f"Handoff hash: {contract['batch_handoff']['settings_snapshot_sha256']}"
        )
    except Exception as exc:
        return f"Export All unavailable: {type(exc).__name__}: {exc}"


def invalidate_preview_cache_handler(cache):
    """Invalidate session preview/runtime contexts after an input change."""
    if not isinstance(cache, GuiPreviewCache):
        return GuiPreviewCache()
    cache.clear()
    return cache


def on_recipe_change(recipe):
    d = recipe_defaults(recipe)
    return tuple(d[k] for k in RECIPE_OUTPUT_KEYS)


def reset_skin_smoothing(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["smooth"], d["nose_smooth"], d["mid_reduction"], d["texture_opacity"], d["micro_restore"], d["pore_synthesis"], d["blemish"], d["skin_flatten"], d["skin_quantize"]

def reset_skin_tone(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["whiten"], d["whiten_tone"], d["equalize"], d["shadow_lift"], d["nose_restore"], d["skin_sss"], d["skin_unify"], d["skin_unify_hue"], d["auto_exposure"], d["white_costume_lift"], d["face_exposure"]

def reset_basic_tone(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["contrast"], d["brightness"], d["clarity"], d["vibrance"], d["saturation"]

def reset_tone_curve(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["highlights"], d["shadows"], d["whites"], d["blacks"]

def reset_relighting(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["relight"], d["relight_azimuth"], d["relight_elevation"]

def reset_eyes_lips(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["eye_enhance"], d["catchlight"], d["dark_circles"], d["undereye_darken_removal"], d["undereye_puffiness_reduction"], d["eye_sclera_brighten"], d["eye_iris_saturate"], d["eye_iris_hue_shift"], d["eye_iris_brightness"], d["teeth_whiten"], d["lip_enhance"], d["lip_tint"], d["lip_finish"], d["blush"], d["nose_blush"], d["under_eye_blush"], d["eye_gate"]

def reset_face_reshaping(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["slimming"]

def reset_structure_effects(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["hair_enhance"], d["dodge_burn"], d["impact"], d["specular_bloom"], d["specular_bloom_tone"], d["bloom"], d["bloom_threshold"], d["bloom_softness"], d["sharpen"], d["sharpen_radius"], d["glow"], d["skin_glow"], d["mask_feather_mode"], d["vignette"], d["subject_separation"]

def reset_color_grading(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["color_grade"], d["grade_intensity"]

def reset_film_effects(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["chromatic_aberration"], d["grain"], d["halation"], d["lut"], d["tonal_curve_strength"], d["skin_protect_strength"], d["grain_strength"], d["highlight_rolloff_strength"], d["film_enable"], d["film_highlight_purity"]

def reset_split_toning(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["shadow_hue"], d["shadow_sat"], d["midtone_hue"], d["midtone_sat"], d["highlight_hue"], d["highlight_sat"]

def reset_color_transfer():
    return None, 1.0

def reset_debug(recipe_name):
    return False


def reset_body_skin(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["body_smooth"], d["body_equalize"], d["body_whiten"], d["body_match_face"], d["body_relight"], d["body_dodge_burn"], d["body_shadow_lift"]


def reset_lch(recipe_name):
    d = recipe_defaults(recipe_name)
    return d["white_balance_kelvin"], d["white_balance_tint"], d["bw_channel_mixer_r"], d["bw_channel_mixer_g"], d["bw_channel_mixer_b"], d["negative_split_tone_shadow"], d["negative_split_tone_highlight"], d["hsl_hue_global"], d["hsl_sat_global"], d["hsl_lum_global"]


def on_detect_faces(img_paths):
    """Detect faces → Gallery thumbs + face index choices. Clears face_params."""
    if not img_paths:
        return [], {}, gr.update(choices=[], value=None), "Upload an image first."
    path = _resolve_image_path(img_paths)
    if not path:
        return [], {}, gr.update(choices=[], value=None), "Could not resolve image path."
    try:
        img = imread_engine(path)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        faces = get_engine()._detector.detect(img)
    except Exception as e:
        _logger.warning("Detect faces failed: %s", e)
        return [], {}, gr.update(choices=[], value=None), f"Detect failed: {e}"
    if not faces:
        return [], {}, gr.update(choices=[], value=None), "No faces detected."
    thumbs = []
    choices = []
    for i, f in enumerate(faces):
        x, y, w, h = f.bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        thumbs.append((rgb, f"Face {i}"))
        choices.append(str(i))
    
    suggested = get_engine().suggest_face_params(img, faces_data=faces)
    
    status = f"{len(choices)} face(s) detected.\n\nSuggested default recipes:\n"
    for k, v in suggested.items():
        status += f"- Face {k}: {v['recipe']}\n"
    status += "\nAssign a recipe per face (optional), then Process."

    return (
        thumbs,
        suggested,
        gr.update(choices=choices, value=choices[0] if choices else None),
        status,
    )


def on_apply_face_recipe(face_idx, recipe_name, face_params):
    """Merge {idx: {recipe: name}} into face_params State."""
    fp = dict(face_params or {})
    if face_idx is None or face_idx == "":
        return fp, "Select a face index first."
    if not recipe_name:
        return fp, "Select a recipe."
    try:
        i = int(face_idx)
    except (TypeError, ValueError):
        return fp, f"Bad face index: {face_idx!r}"
    entry = dict(fp.get(i) or fp.get(str(i)) or {})
    entry["recipe"] = recipe_name
    fp[i] = entry
    # drop string-key duplicate if any
    fp.pop(str(i), None)
    summary = ", ".join(f"{k}→{v.get('recipe', v)}" for k, v in sorted(fp.items()))
    return fp, f"Face {i} → {recipe_name}. Active: {summary}"


def on_clear_face_params():
    return {}, [], gr.update(choices=[], value=None), "Per-face overrides cleared."


def on_img_change_clear_faces():
    return {}, [], gr.update(choices=[], value=None), "Image changed — re-detect faces."


def advanced_export_handler(current_rgb, export_fmt, source_paths=None, base_contract=None):
    """Export the full-resolution canvas with context-aware metadata."""
    if current_rgb is None:
        return gr.update(visible=False), "Load an image first."
    delivery = advanced_delivery_decision(base_contract)
    if not delivery["allowed"]:
        return (
            gr.update(value=None, visible=False),
            "Advanced Retouch export blocked: {} ({}). Use an uploaded source or rerun Export Full Quality, "
            "then choose Edit processed result.".format(delivery["reason"], delivery["detail"]),
        )
    fmt = str(export_fmt or "PNG").upper()
    ext = {"JPEG": ".jpg", "PNG": ".png", "PNG-16": ".png", "TIFF-16": ".tiff", "WEBP": ".webp"}.get(fmt, ".png")
    output_dir = tempfile.mkdtemp(prefix="retouch_advanced_export_")
    output_path = os.path.join(output_dir, f"advanced_retouch{ext}")
    bgr = cv2.cvtColor(np.asarray(current_rgb).astype(np.uint8), cv2.COLOR_RGB2BGR)
    source_path = _resolve_image_path(source_paths)
    trusted_source_path = (
        source_path
        if source_path and source_file_matches(base_contract, source_path)
        else None
    )
    exif = read_exif_bytes(trusted_source_path) if trusted_source_path else None
    source_had_c2pa = bool(read_c2pa_manifest(trusted_source_path)) if trusted_source_path else False
    color_context = (
        color_context_for_path(trusted_source_path)
        if trusted_source_path
        else ColorContext.assumed_srgb_context(get_working_srgb_icc())
    )
    write_image_with_color_context(
        output_path,
        bgr,
        color_context,
        exif=exif,
        # Copying the source assertion after changing pixels would not create a
        # valid derived-work claim. Keep it out until a signing path exists.
        c2pa_manifest=None,
        bit_depth=16 if fmt in {"PNG-16", "TIFF-16"} else 8,
        quality=95,
        float_range="byte",
    )
    if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
        return gr.update(visible=False), "Advanced Retouch export failed."
    metadata = " with verified-source ICC/EXIF" if trusted_source_path else " with working-sRGB color metadata"
    precision = "; 16-bit container from an effective 8-bit Advanced canvas" if fmt in {"PNG-16", "TIFF-16"} else ""
    provenance = "; source C2PA was intentionally not copied after pixel edits" if source_had_c2pa else ""
    source_warning = (
        "; source metadata was not reused because file identity did not match the recorded base"
        if source_path and not trusted_source_path
        else ""
    )
    return (
        gr.update(value=output_path, visible=True),
        f"Exported verified-native Advanced Retouch canvas as {fmt}{metadata}{precision}{provenance}{source_warning}.",
    )


def on_extract_look(look_ref_file, img_input):
    """F6 — Extract an editable look from a reference image.

    Loads the uploaded reference, optionally pairs it with the main input
    image as the base, runs ``LookExtractor.extract``, strips ``_``-prefixed
    keys (internal transport params) and returns the cleaned ``engine_params``
    dict for storage in the hidden ``look_params`` State. The subsequent
    Process call overlays these params over the recipe defaults (look wins).
    Failures are reported without raising.
    """
    if look_ref_file is None:
        return {}, "Please upload a reference image first."

    ref_path = _resolve_image_path(look_ref_file)
    if not ref_path:
        return {}, "Could not resolve the reference image path."

    try:
        ref_bgr = imread_exif(ref_path)
    except (TypeError, FileNotFoundError, OSError) as e:
        _logger.warning("Extract Look: failed to load reference %s: %s", ref_path, e)
        return {}, f"Extract Look failed to load reference: {e}"

    base_bgr = None
    base_path = _resolve_image_path(img_input)
    if base_path:
        try:
            base_bgr = imread_exif(base_path)
        except (TypeError, FileNotFoundError, OSError) as e:
            _logger.warning("Extract Look: failed to load base %s: %s", base_path, e)
            base_bgr = None

    try:
        result = LookExtractor().extract(ref_bgr, base_bgr)
    except Exception as e:
        _logger.exception("LookExtractor failed: %s", e)
        return {}, f"Extract Look failed: {e}"

    engine_params = result.get("engine_params") or {}
    clean = {k: v for k, v in engine_params.items() if not str(k).startswith("_")}
    mode = result.get("mode", "unknown")
    return clean, f"Look extracted ({mode}) — {len(clean)} params. Click Process to apply."


def on_search_recipes(query, category=None):
    """T4 — Search/browse cookbook; optional category filter."""
    try:
        cat = category if category and category != "All" else None
        if query and str(query).strip():
            results = search_recipes(query.strip())
            if cat:
                results = [r for r in results if r.category == cat]
        else:
            results = list_recipes(cat)
        # The cookbook follows the same curated policy as the main recipe
        # selector, so an archived experiment cannot be selected into a
        # dropdown that intentionally does not expose it.
        results = [r for r in results if r.name in RECIPE_NAMES]
    except Exception as e:
        _logger.exception("Recipe search failed: %s", e)
        return gr.update(choices=[]), f"Recipe search failed: {e}"

    choices = [r.name for r in results]
    msg = f"{len(results)} recipe(s)"
    if query:
        msg += f" matching '{query}'"
    if category and category != "All":
        msg += f" in {category}"
    return gr.update(choices=choices, value=None), msg + "."


def on_select_cookbook(name):
    """T4 — Apply a cookbook recipe selection to the active recipe Radio."""
    if not name:
        return gr.update(), ""
    return gr.update(value=name), f"Selected recipe: {name}"


def on_browse_category(category):
    """T4 — List recipes in category (or all)."""
    return on_search_recipes("", category)


def pick_folder_dialog(current_value=None):
    """Open a native macOS folder-picker and return the chosen absolute path.

    No-ops (keeps the existing textbox value) on cancel or on non-macOS
    platforms, since there is no cross-platform native dialog available
    from a server-side Gradio callback.
    """
    if sys.platform != "darwin":
        _logger.info("Folder picker is only available on macOS; leave path as-is.")
        return gr.update()
    try:
        result = subprocess.run(
            ["osascript", "-e", "POSIX path of (choose folder)"],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            return gr.update()  # user canceled
        path = result.stdout.strip()
        return gr.update(value=path) if path else gr.update()
    except Exception as e:
        _logger.warning("Folder picker failed: %s", e)
        return gr.update()


def on_reload_luts():
    """Clear LUT discovery and live render caches."""
    try:
        get_registry().reload()
        _invalidate_live_engine_luts()
        return "LUTs reloaded successfully."
    except Exception as e:
        _logger.warning("LUT reload failed: %s", e)
        return f"LUT reload failed: {e}"


def _invalidate_live_engine_luts(stem=None):
    """Invalidate the active engine without forcing lazy initialization."""
    with _engine_lock:
        live_engine = _engine
    if live_engine is not None:
        live_engine.invalidate_lut_cache(stem)


# The registry watcher already evicts its discovery cache. The callback also
# evicts the independent ColorGrader cache used by the live render path.
def _on_lut_changed(stem: str) -> None:
    try:
        _invalidate_live_engine_luts(stem)
    except Exception as exc:
        _logger.warning("Live LUT cache invalidation failed for %s: %s", stem, exc)


try:
    watch_luts_dir(_on_lut_changed, interval=2.0)
except Exception as exc:
    _logger.warning("LUT watcher daemon did not start: %s", exc)


def build_smart_proposal(suggestion):
    """Build a serializable Smart Process proposal without changing UI state."""
    defaults = recipe_defaults(suggestion.recipe)
    params = dict(getattr(suggestion, "params", {}) or {})
    for key, value in params.items():
        if key in defaults:
            defaults[key] = value
    explanation_html = _format_smart_explanations(suggestion)
    return {
        "proposal_version": 1,
        "recipe": suggestion.recipe,
        "params": params,
        "slider_values": {key: defaults[key] for key in RECIPE_OUTPUT_KEYS},
        "explanations": list(getattr(suggestion, "explanations", []) or []),
        "explanation_html": explanation_html,
    }


def apply_smart_suggestion(proposal, current_revision=0):
    """Apply a stored proposal only after the user explicitly requests it."""
    revision = _coerce_settings_revision(current_revision)
    if not isinstance(proposal, dict):
        return (
            gr.update(),
            *([gr.update()] * len(RECIPE_OUTPUT_KEYS)),
            proposal,
            "No Smart suggestion is waiting to be applied.",
            gr.update(),
            gr.update(interactive=False),
            revision,
        )

    recipe_name = proposal.get("recipe")
    slider_values = proposal.get("slider_values")
    if not recipe_name or not isinstance(slider_values, dict):
        return (
            gr.update(),
            *([gr.update()] * len(RECIPE_OUTPUT_KEYS)),
            proposal,
            "Smart suggestion is invalid or incomplete; nothing was applied.",
            gr.update(),
            gr.update(interactive=False),
            revision,
        )

    missing = [key for key in RECIPE_OUTPUT_KEYS if key not in slider_values]
    if missing:
        return (
            gr.update(),
            *([gr.update()] * len(RECIPE_OUTPUT_KEYS)),
            proposal,
            f"Smart suggestion is missing {len(missing)} control value(s); nothing was applied.",
            gr.update(),
            gr.update(interactive=False),
            revision,
        )

    next_revision = revision + 1
    explanation_html = proposal.get("explanation_html") or ""
    status = (
        f"🧠 Applied Smart suggestion ({recipe_name}). Click Process to render. "
        f"Settings revision {next_revision}."
    )
    return (
        recipe_name,
        *(slider_values[key] for key in RECIPE_OUTPUT_KEYS),
        None,
        status,
        explanation_html,
        gr.update(interactive=False),
        next_revision,
    )


def on_smart_process(img_paths, recipe, *args, prg=gr.Progress()):
    """F10 Smart Process — analyze and store a proposal without changing sliders."""
    from retouch.smart_default import SmartProcessor

    if not img_paths:
        gr.Warning("Please upload an image first.")
        return None, "Please upload an image first.", gr.update()

    curr_path = img_paths[0]
    if isinstance(curr_path, dict):
        curr_path = curr_path.get("name") or curr_path.get("path")

    try:
        img_bgr = imread_exif(curr_path)
    except (TypeError, FileNotFoundError, OSError) as e:
        _logger.warning("Smart Process: failed to load %s: %s", curr_path, e)
        gr.Warning(f"Failed to load image: {e}")
        return None, f"Failed to load image: {e}", gr.update()

    if img_bgr is None:
        gr.Warning("Could not read the image.")
        return None, "Could not read the image.", gr.update()

    gr.Info("🧠 Analyzing image...")
    sp = SmartProcessor()
    try:
        suggestion = sp.analyze_and_suggest(img_bgr)
    except ValueError as e:
        _logger.exception("Smart Process analysis failed: %s", e)
        gr.Warning(f"Analysis failed: {e}")
        return None, f"Analysis failed: {e}", gr.update()

    proposal = build_smart_proposal(suggestion)
    explanation_html = _format_smart_explanations(suggestion)
    status_msg = (
        f"🧠 Smart suggestion ready ({suggestion.recipe}); sliders were not changed. "
        "Review it, then click Apply Smart Suggestion."
    )
    gr.Info(f"Smart suggestion ready: {suggestion.recipe} ({len(suggestion.params)} overrides)")

    return proposal, status_msg, explanation_html


def on_smart_process_event(img_paths, recipe=None, settings_revision=0):
    """Queue wrapper that preserves the analysis snapshot and enables Apply."""
    if isinstance(img_paths, dict) and "img_paths" in img_paths:
        snapshot = img_paths
        recipe = snapshot.get("recipe")
        settings_revision = snapshot.get("settings_revision", 0)
        img_paths = snapshot.get("img_paths")
    proposal, status, explanation = on_smart_process(img_paths, recipe)
    return proposal, status, explanation, gr.update(interactive=proposal is not None)


def _format_smart_explanations(suggestion) -> str:
    """Format SmartSuggestion.explanations as an HTML readout for the GUI."""
    if not suggestion.explanations:
        return ""
    items = "".join(f"<li>{e}</li>" for e in suggestion.explanations)
    return (
        '<div style="background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.25);'
        'padding:8px 12px;border-radius:6px;margin:8px 0;font-size:13px">'
        f'<strong>🧠 Smart Analysis — recipe: {suggestion.recipe}</strong>'
        f'<ul style="margin:4px 0 0 16px;padding:0">{items}</ul>'
        '</div>'
    )



LIP_TINTS = ["none"] + LIP_TINT_NAMES
custom_style_choices = get_custom_style_names()

COMPARE_TPL = """
<div id="cmp-%(uid)s" style="position:relative;width:100%%;user-select:none;overflow:hidden;border-radius:4px">
  <img src="%(result)s" style="width:100%%;display:block;pointer-events:none">
  <div class="cmp-overlay" style="position:absolute;top:0;left:0;width:50%%;height:100%%;overflow:hidden">
    <img src="%(orig)s" style="width:100%%;display:block;max-width:none;position:absolute;left:0;top:0;pointer-events:none">
  </div>
  <div class="cmp-handle" style="position:absolute;top:0;left:50%%;width:3px;height:100%%;background:#fff;cursor:ew-resize;z-index:10;box-shadow:0 0 6px rgba(0,0,0,0.4)"></div>
  <div class="cmp-label" style="position:absolute;top:10px;left:10px;background:rgba(0,0,0,0.55);color:#fff;padding:2px 10px;border-radius:3px;font-size:11px;letter-spacing:1px;pointer-events:none">BEFORE</div>
  <div class="cmp-label" style="position:absolute;top:10px;right:10px;background:rgba(0,0,0,0.55);color:#fff;padding:2px 10px;border-radius:3px;font-size:11px;letter-spacing:1px;pointer-events:none">AFTER</div>
</div>
"""


def _make_comparison_html(orig_bgr, result_bgr, max_height=600):
    scale = max_height / max(orig_bgr.shape[0], result_bgr.shape[0])
    if scale < 1.0:
        new_w = int(orig_bgr.shape[1] * scale)
        new_h = int(orig_bgr.shape[0] * scale)
        orig_bgr = cv2.resize(orig_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        result_bgr = cv2.resize(result_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    jpeg_params = encode_write_params("jpg", 92)
    _, ob = cv2.imencode('.jpg', orig_bgr, jpeg_params)
    _, rb = cv2.imencode('.jpg', result_bgr, jpeg_params)
    uid = hex(int(time.time() * 1e6))[2:]
    return COMPARE_TPL % {
        "uid": uid,
        "orig": f"data:image/jpeg;base64,{base64.b64encode(ob).decode()}",
        "result": f"data:image/jpeg;base64,{base64.b64encode(rb).decode()}",
    }

with gr.Blocks(title="🪄 Retouch — AI Portrait Workflow Platform", theme=gr.themes.Soft(primary_hue="sky", secondary_hue="slate"), css="""
    /* --- Liquid Glass Theme --- */
    
    /* Scoped under .dark to avoid light mode text visibility issues */
    .dark.gradio-container, .dark .gradio-container {
        --background-fill-primary: transparent !important;
        --background-fill-primary-dark: transparent !important;
        --block-background-fill: transparent !important;
        --block-background-fill-dark: transparent !important;
        --block-background-fill-light: transparent !important;
        --input-background-fill: rgba(255,255,255,0.06) !important;
        --input-background-fill-dark: rgba(255,255,255,0.06) !important;
        --border-color-primary: rgba(255,255,255,0.08) !important;
        --border-color-primary-dark: rgba(255,255,255,0.08) !important;
        --body-text-color: #e8edf5 !important;
        --body-text-color-dark: #e8edf5 !important;
        --block-label-text-color: rgba(255,255,255,0.55) !important;
        --block-label-text-color-dark: rgba(255,255,255,0.55) !important;
        --button-primary-background-fill: rgba(0, 162, 237, 0.7) !important;
        --button-primary-background-fill-dark: rgba(0, 162, 237, 0.7) !important;
        --button-secondary-background-fill: rgba(255,255,255,0.06) !important;
        --button-secondary-background-fill-dark: rgba(255,255,255,0.06) !important;
        --slider-color: #60a5fa !important;
        --slider-color-dark: #60a5fa !important;
        --checkbox-background-color-selected: #60a5fa !important;
        --checkbox-background-color-selected-dark: #60a5fa !important;
        --shadow-drop: 0 8px 32px rgba(0,0,0,0.25) !important;
        --shadow-drop-dark: 0 8px 32px rgba(0,0,0,0.25) !important;
    }

    /* Full-width container */
    html, body {
        max-width: 100vw !important;
        overflow-x: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    body.dark {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        background-attachment: fixed !important;
    }
    .gradio-container-outer {
        max-width: 100vw !important;
        width: 100vw !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    .gradio-container, .gradio-container .contain, .gradio-container .main-wrap {
        max-width: 100vw !important;
        width: 100vw !important;
        padding-left: 12px !important;
        padding-right: 12px !important;
        background: transparent !important;
    }
    .dark .gradio-container, .dark .gradio-container .contain, .dark .gradio-container .main-wrap {
        color: #e8edf5 !important;
    }
    
    /* Global font */
    body, input, button, select, textarea, span, p, div, label {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
    }
    
    /* Glass panels with frost effect (Dark Mode Only) */
    .dark .gr-group, .dark .group, .dark .form, .dark .block, .dark .panel, .dark .padded {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25), inset 0 1px 0 rgba(255, 255, 255, 0.06) !important;
    }
    
    /* Navigation tabs (Dark Mode Only) */
    .dark .tabs {
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 12px !important;
        padding: 4px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 20px !important;
    }
    
    .tab-nav {
        border-bottom: none !important;
        display: flex !important;
        gap: 4px !important;
    }
    
    .dark .tab-nav button {
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 18px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        color: rgba(255, 255, 255, 0.5) !important;
        background: transparent !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.03em !important;
    }
    
    .dark .tab-nav button.selected {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        box-shadow: 0 1px 8px rgba(0, 0, 0, 0.15) !important;
    }
    
    /* Glass buttons */
    .primary-btn {
        background: rgba(0, 162, 237, 0.7) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 16px rgba(0, 162, 237, 0.25) !important;
    }
    .primary-btn:hover {
        background: rgba(0, 162, 237, 0.85) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 24px rgba(0, 162, 237, 0.35) !important;
        border-color: rgba(255, 255, 255, 0.25) !important;
    }
    .primary-btn:active {
        transform: translateY(0px) !important;
    }
    
    .dark .secondary-btn {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.04em !important;
        border-radius: 10px !important;
        padding: 10px 22px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }
    .dark .secondary-btn:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #ffffff !important;
        transform: translateY(-2px) !important;
        border-color: rgba(255, 255, 255, 0.15) !important;
    }
    .dark .secondary-btn:active {
        transform: translateY(0px) !important;
    }
    
    /* Glass accordions (Dark Mode Only) */
    .dark .accordion {
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 10px !important;
        margin-bottom: 8px !important;
        overflow: visible !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
        transition: border-color 0.2s ease !important;
    }
    .dark .accordion:hover {
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    
    .dark .accordion > summary, .dark .accordion .label-wrap {
        background: rgba(255, 255, 255, 0.04) !important;
        padding: 8px 14px !important;
        color: rgba(255, 255, 255, 0.7) !important;
        font-size: 0.75rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.04) !important;
        border-radius: 10px 10px 0 0 !important;
    }
    
    /* Develop panel container */
    .develop-panel {
        max-height: 84vh !important;
        overflow-y: auto !important;
        padding-right: 6px !important;
        background: transparent !important;
        border: none !important;
    }
    
    .develop-panel::-webkit-scrollbar { width: 4px !important; }
    .develop-panel::-webkit-scrollbar-track { background: transparent !important; }
    .develop-panel::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.15) !important; border-radius: 2px !important; }
    .develop-panel::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.25) !important; }
    .develop-panel { scrollbar-width: thin !important; scrollbar-color: rgba(255,255,255,0.15) transparent !important; }
    
    /* Sliders */
    .dark .gr-slider input[type=range] {
        accent-color: #60a5fa !important;
        background: rgba(255, 255, 255, 0.08) !important;
    }
    
    /* Text inputs (Dark Mode Only) */
    .dark input[type="text"], .dark input[type="number"], .dark select, .dark textarea {
        background: rgba(255, 255, 255, 0.06) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        color: #e8edf5 !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
    }
    .dark input[type="text"]:focus, .dark input[type="number"]:focus, .dark select:focus, .dark textarea:focus {
        border-color: #60a5fa !important;
        box-shadow: 0 0 0 2px rgba(96, 165, 250, 0.2) !important;
    }
    
    /* Checkbox & Radio */
    input[type="checkbox"] { accent-color: #60a5fa !important; }
    
    /* Typography (Dark Mode Only) */
    .dark h1, .dark h2, .dark h3,
    .dark h4, .dark h5, .dark h6,
    .dark p, .dark strong, .dark .prose,
    .dark .prose h1, .dark .prose h2,
    .dark .prose h3, .dark .prose h4,
    .dark .prose p, .dark .markdown-text h1,
    .dark .markdown-text h2, .dark .markdown-text h3,
    .dark .markdown-text p, .dark div.markdown {
        color: #e8edf5 !important;
    }
    
    /* Form labels (Dark Mode Only) */
    .dark label span,
    .dark .form-label,
    .dark label .form-label-text,
    .dark .label-val {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        color: rgba(255, 255, 255, 0.55) !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.04em !important;
    }
    
    /* Preset chips: scroll cap applies in both themes — with 65+ recipes,
       an uncapped list buries the primary action buttons below the fold
       (this was previously .dark-only, so light-mode users got no cap). */
    .preset-chips { border: none !important; background: transparent !important; padding: 0 !important; }
    .preset-chips .wrap {
        display: flex !important;
        flex-direction: column !important;
        flex-wrap: nowrap !important;
        max-height: 220px !important;
        overflow-y: auto !important;
        gap: 4px !important;
        padding: 0 4px 0 0 !important;
    }
    .preset-chips .wrap::-webkit-scrollbar { width: 3px !important; }
    .preset-chips .wrap::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.15) !important; border-radius: 2px !important; }
    .dark .preset-chips .wrap::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12) !important; }

    /* Preset chips (glass style - Dark Mode Only) */

    .dark .preset-chips label {
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        background: rgba(255, 255, 255, 0.04) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 8px !important;
        padding: 8px 14px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        color: rgba(255, 255, 255, 0.6) !important;
        width: 100% !important;
    }
    .dark .preset-chips label:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        color: rgba(255, 255, 255, 0.85) !important;
        transform: translateY(-1px) !important;
        border-color: rgba(255, 255, 255, 0.12) !important;
    }
    .dark .preset-chips label.selected {
        background: rgba(96, 165, 250, 0.15) !important;
        color: #93c5fd !important;
        border-left: 3px solid #60a5fa !important;
        border-radius: 0 8px 8px 0 !important;
    }
    .preset-chips input[type="radio"] { display: none !important; }
    .preset-chips label .radio-circle { display: none !important; }
    
    /* Dropdown options (Dark Mode Only) */
    .dark ul.options, .dark .options {
        background: rgba(30, 27, 75, 0.95) !important;
        backdrop-filter: blur(24px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(180%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4) !important;
        z-index: 9999 !important;
        position: absolute !important;
        overflow-y: auto !important;
        max-height: 280px !important;
        scrollbar-width: thin !important;
        scrollbar-color: rgba(255,255,255,0.15) rgba(255,255,255,0.02) !important;
    }
    .dark ul.options > li, .dark .options > li {
        color: rgba(255, 255, 255, 0.7) !important;
        padding: 8px 16px !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }
    .dark ul.options > li:hover, .dark .options > li:hover {
        background: rgba(255, 255, 255, 0.06) !important;
        color: #ffffff !important;
    }
    .dark ul.options > li.selected, .dark .options > li.selected {
        color: #93c5fd !important;
        background: rgba(96, 165, 250, 0.12) !important;
    }
    
    /* Click-to-zoom */
    #retouch-output { cursor: zoom-in !important; }
    .cmp-label { font-weight: 600 !important; }
    #retouch-compare { cursor: ew-resize !important; }
    
    /* File upload */
    .dark .gr-file {
        background: rgba(255, 255, 255, 0.03) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        border: 1px dashed rgba(255, 255, 255, 0.1) !important;
        border-radius: 10px !important;
    }
    .dark .gr-file:hover { border-color: rgba(96, 165, 250, 0.5) !important; }
    
    /* Overflow fix */
    .gradio-container .block, .gradio-container .wrap,
    .gradio-container .group, .gradio-container .form,
    .gradio-container .row, .gradio-container .col,
    .gradio-container .column, .gradio-container .panel,
    .gradio-container .padded, .gradio-container .tabs,
    .gradio-container .tabitem, .gradio-container .dropdown,
    .gradio-container .dropdown-container {
        overflow: visible !important;
    }

    /* Re-enable internal scroll on develop-panel after the overflow fix above.
       The .gradio-container .column rule (specificity 0,2,0) overrides
       .develop-panel (0,1,0), killing overflow-y:auto and clipping content
       past max-height:84vh. This higher-specificity rule restores it. */
    .gradio-container .column.develop-panel {
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }

    /* Header Bar styling adapting to both Light and Dark mode */
    .header-bar {
        display: flex !important;
        justify-content: space-between !important;
        align-items: center !important;
        padding: 0.6rem 1.5rem !important;
        background: rgba(0, 0, 0, 0.03) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        margin-bottom: 18px !important;
        font-family: -apple-system, sans-serif !important;
        border-radius: 14px !important;
    }
    .header-left {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }
    .header-badge {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
        color: #ffffff !important;
        padding: 3px 8px !important;
        border-radius: 6px !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.3px !important;
    }
    .header-title {
        font-weight: 600 !important;
        font-size: 1rem !important;
        color: rgba(15, 23, 42, 0.9) !important;
        letter-spacing: 0.3px !important;
    }
    .header-version {
        font-size: 0.7rem !important;
        color: rgba(15, 23, 42, 0.4) !important;
        border-left: 1px solid rgba(15, 23, 42, 0.1) !important;
        padding-left: 10px !important;
        margin-left: 2px !important;
        font-weight: 500 !important;
    }
    .header-workspace {
        font-size: 0.75rem !important;
        color: rgba(15, 23, 42, 0.5) !important;
        font-weight: 500 !important;
        letter-spacing: 0.05em !important;
    }

    /* Dark mode overrides for Header Bar */
    .dark .header-bar {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    .dark .header-badge {
        background: linear-gradient(135deg, #60a5fa, #a78bfa) !important;
        color: #ffffff !important;
    }
    .dark .header-title {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    .dark .header-version {
        color: rgba(255, 255, 255, 0.3) !important;
        border-left: 1px solid rgba(255, 255, 255, 0.08) !important;
    }
    .dark .header-workspace {
        color: rgba(255, 255, 255, 0.35) !important;
    }
""", head="""
    <script>
    (function() {
        // Keyboard Shortcuts
        document.addEventListener('keydown', function(e) {
            if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                e.preventDefault();
                var btn = document.querySelector('.primary-btn');
                if (btn) btn.click();
            }
            if ((e.metaKey || e.ctrlKey) && e.key === 'r') {
                e.preventDefault();
                var openDetails = document.querySelector('.develop-panel details[open]');
                if (openDetails) {
                    var resetBtn = openDetails.querySelector('.section-reset-btn');
                    if (resetBtn) resetBtn.click();
                }
            }
        });

        // Click-to-Zoom Modal
        function openZoomModal(src) {
            var overlay = document.createElement('div');
            overlay.id = 'zoom-overlay';
            overlay.style.position = 'fixed';
            overlay.style.top = '0';
            overlay.style.left = '0';
            overlay.style.width = '100vw';
            overlay.style.height = '100vh';
            overlay.style.backgroundColor = 'rgba(10, 8, 25, 0.95)';
            overlay.style.backdropFilter = 'blur(24px)';
            overlay.style.webkitBackdropFilter = 'blur(24px)';
            overlay.style.zIndex = '99999';
            overlay.style.display = 'flex';
            overlay.style.alignItems = 'center';
            overlay.style.justifyContent = 'center';
            overlay.style.cursor = 'zoom-out';
            overlay.style.opacity = '0';
            overlay.style.transition = 'opacity 0.25s cubic-bezier(0.16, 1, 0.3, 1)';
            
            var img = document.createElement('img');
            img.src = src;
            img.style.maxHeight = '92vh';
            img.style.maxWidth = '92vw';
            img.style.objectFit = 'contain';
            img.style.borderRadius = '8px';
            img.style.boxShadow = '0 24px 60px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.1)';
            img.style.transform = 'scale(0.95)';
            img.style.transition = 'transform 0.25s cubic-bezier(0.16, 1, 0.3, 1)';
            
            overlay.appendChild(img);
            document.body.appendChild(overlay);
            
            setTimeout(function() {
                overlay.style.opacity = '1';
                img.style.transform = 'scale(1)';
            }, 10);
            
            overlay.addEventListener('click', function() {
                overlay.style.opacity = '0';
                img.style.transform = 'scale(0.95)';
                setTimeout(function() {
                    overlay.remove();
                }, 250);
            });
        }

        document.addEventListener('click', function(e) {
            var target = e.target;
            if (target.tagName === 'IMG' && (target.closest('#retouch-output') || target.closest('#retouch-compare'))) {
                e.preventDefault();
                openZoomModal(target.src);
            }
        });

        // Theme management and layout helpers
        function forceFullWidth() {
            document.querySelectorAll('.gradio-container-outer, .gradio-container').forEach(function(el){
                if (el.style.maxWidth !== 'none') {
                    el.style.setProperty('max-width', 'none', 'important');
                }
                if (el.style.width !== '100vw') {
                    el.style.setProperty('width', '100vw', 'important');
                }
                if (el.style.minWidth !== '100vw') {
                    el.style.setProperty('min-width', '100vw', 'important');
                }
            });
        }
        function forceDarkMode() {
            var isDark = window.location.search.includes('__theme=dark') || localStorage.getItem('theme') === 'dark' || window.matchMedia('(prefers-color-scheme: dark)').matches;
            if (isDark) {
                if (!document.documentElement.classList.contains('dark')) {
                    document.documentElement.classList.add('dark');
                }
                if (document.body && !document.body.classList.contains('dark')) {
                    document.body.classList.add('dark');
                }
                document.querySelectorAll('.gradio-container, .gradio-container-outer').forEach(function(el){
                    if (!el.classList.contains('dark')) {
                        el.classList.add('dark');
                    }
                });
            } else {
                if (document.documentElement.classList.contains('dark')) {
                    document.documentElement.classList.remove('dark');
                }
                if (document.body && document.body.classList.contains('dark')) {
                    document.body.classList.remove('dark');
                }
                document.querySelectorAll('.gradio-container, .gradio-container-outer').forEach(function(el){
                    if (el.classList.contains('dark')) {
                        el.classList.remove('dark');
                    }
                });
            }
        }
        
        var observer = new MutationObserver(function(){
            observer.disconnect();
            forceFullWidth();
            forceDarkMode();
            observer.observe(document.documentElement, {attributes: true, subtree: true, attributeFilter: ['style', 'class']});
        });
        observer.observe(document.documentElement, {attributes: true, subtree: true, attributeFilter: ['style', 'class']});
        ['load', 'DOMContentLoaded', 'gradio:ready'].forEach(function(e){ window.addEventListener(e, function(){ forceFullWidth(); forceDarkMode(); }); });
        setTimeout(function(){ forceFullWidth(); forceDarkMode(); }, 200);
        setTimeout(function(){ forceFullWidth(); forceDarkMode(); }, 1000);
        setTimeout(function(){ forceFullWidth(); forceDarkMode(); }, 3000);

        // Comparison Slider: drag handle + syncSize (delegated).
        // NOTE: COMPARE_TPL's <script> cannot run because Gradio's gr.HTML
        // injects the value via element.innerHTML, and HTML5 spec says
        // scripts inserted that way are inert. So the drag/sync logic lives
        // here and uses event delegation on document so listeners survive
        // every innerHTML replacement.
        (function() {
            function findCmp(node) {
                if (!node) return null;
                if (node.id && node.id.indexOf('cmp-') === 0) return node;
                var c = node.closest && node.closest('[id^="cmp-"]');
                if (c) return c;
                var byId = node.querySelector && node.querySelector('[id^="cmp-"]');
                return byId || null;
            }

            function setupCmp(c) {
                if (!c || c.__cmpInited) return;
                c.__cmpInited = true;
                var o = c.querySelector('.cmp-overlay');
                var imgResult = c.children[0];
                var imgOrig = o && o.children[0];
                if (!o || !imgResult || !imgOrig) return;

                function syncSize() {
                    var rect = imgResult.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        imgOrig.style.width = rect.width + 'px';
                        imgOrig.style.height = rect.height + 'px';
                    }
                }
                if (imgResult.complete && imgResult.naturalWidth > 0) {
                    syncSize();
                } else {
                    imgResult.addEventListener('load', syncSize);
                }
                window.addEventListener('resize', syncSize);
                setTimeout(syncSize, 50);
                setTimeout(syncSize, 200);
                setTimeout(syncSize, 1000);
            }

            var activeDrag = null;

            function beginDrag(handle, e) {
                var c = findCmp(handle);
                if (!c) return;
                setupCmp(c);
                var o = c.querySelector('.cmp-overlay');
                var h = c.querySelector('.cmp-handle');
                var imgResult = c.children[0];
                var imgOrig = o.children[0];

                function syncSize() {
                    var rect = imgResult.getBoundingClientRect();
                    if (rect.width > 0 && rect.height > 0) {
                        imgOrig.style.width = rect.width + 'px';
                        imgOrig.style.height = rect.height + 'px';
                    }
                }
                function move(x) {
                    var r = c.getBoundingClientRect();
                    var p = Math.max(0, Math.min(100, (x - r.left) / r.width * 100));
                    o.style.width = p + '%';
                    h.style.left = p + '%';
                    syncSize();
                }
                activeDrag = { move: move };
                if (e && e.preventDefault) e.preventDefault();
            }

            document.addEventListener('mousedown', function(e) {
                var handle = e.target.closest && e.target.closest('.cmp-handle');
                if (handle) beginDrag(handle, e);
            });
            document.addEventListener('mousemove', function(e) {
                if (activeDrag) activeDrag.move(e.clientX);
            });
            document.addEventListener('mouseup', function() { activeDrag = null; });

            document.addEventListener('touchstart', function(e) {
                var handle = e.target.closest && e.target.closest('.cmp-handle');
                if (handle) beginDrag(handle, e);
            }, { passive: false });
            document.addEventListener('touchmove', function(e) {
                if (activeDrag && e.touches[0]) activeDrag.move(e.touches[0].clientX);
            }, { passive: false });
            document.addEventListener('touchend', function() { activeDrag = null; });

            // Auto-setup whenever a new comparison is inserted.
            var mo = new MutationObserver(function(muts) {
                for (var i = 0; i < muts.length; i++) {
                    var added = muts[i].addedNodes;
                    for (var j = 0; j < added.length; j++) {
                        var n = added[j];
                        if (n.nodeType !== 1) continue;
                        var c = findCmp(n);
                        if (c) setupCmp(c);
                    }
                }
            });
            mo.observe(document.body, { childList: true, subtree: true });
        })();
    })();
    </script>
""") as app:

    # Liquid Glass Header bar
    gr.HTML(f"""
    <div class="header-bar">
        <div class="header-left">
            <span class="header-badge">RP</span>
            <span class="header-title">Retouch Pro</span>
            <span class="header-version">v{__version__}</span>
        </div>
        <div class="header-workspace">
            Liquid Glass Workspace
        </div>
    </div>
    """)


    with gr.Tabs():
        with gr.Tab("Single Photo Editor"):
            with gr.Row():
                # Column 1: Presets & Library Panel (Left)
                with gr.Column(scale=2, elem_classes=["library-panel"]):
                    img_input = gr.File(label="Input Image(s) (RAW supported)", file_types=["image", *sorted(RAW_EXTENSIONS)], file_count="multiple")

                    with gr.Group():
                        recipe = gr.Radio(
                            choices=RECIPE_UI_CHOICES, value="natural", label="Base Preset Recipe",
                            info="Recommended recipes are correction-first. Scene / creative recipes need matching light or intent.",
                            elem_classes=["preset-chips"]
                        )

                        with gr.Row():
                            process_btn = gr.Button("Render Preview ⚡", variant="primary", size="lg", elem_classes=["primary-btn"])
                            export_full_btn = gr.Button("Export Full Quality", variant="primary", size="lg", elem_classes=["primary-btn"])
                        with gr.Row():
                            smart_process_btn = gr.Button("🧠 Smart Process", variant="secondary", size="sm", elem_classes=["primary-btn"])
                            export_all_btn = gr.Button("Export All → Batch", variant="secondary", size="sm", elem_classes=["secondary-btn"])
                            apply_smart_btn = gr.Button("Apply Smart Suggestion", variant="secondary", size="sm", interactive=False, elem_classes=["secondary-btn"])
                            reset_btn = gr.Button("Reload Recipe Defaults 🔄", variant="secondary", size="sm", elem_classes=["secondary-btn"], elem_id="reset-btn")

                        custom_style_preset = gr.Dropdown(
                            choices=custom_style_choices, value=None, label="Or Load Custom Style Profile", interactive=True,
                            info="Select an extracted style from your library"
                        )

                        with gr.Accordion("📖 Recipe Cookbook", open=False):
                            with gr.Row():
                                cookbook_category = gr.Dropdown(
                                    label="Category",
                                    choices=["All"] + list_categories(),
                                    value="All",
                                    interactive=True,
                                    scale=1,
                                )
                                cookbook_search = gr.Textbox(
                                    label="Search",
                                    placeholder="e.g. cosplay, portrait, fuji",
                                    scale=2,
                                )
                            with gr.Row():
                                cookbook_search_btn = gr.Button(
                                    "Search / Browse", variant="secondary", size="sm",
                                    elem_classes=["secondary-btn"],
                                )
                            cookbook_dropdown = gr.Dropdown(
                                label="Cookbook Recipe",
                                choices=[],
                                interactive=True,
                                value=None,
                                allow_custom_value=False,
                                info="Browse by category or search, then select to load into Base Recipe",
                            )
                            cookbook_status = gr.Markdown(
                                "Open accordion → pick category or search → select recipe."
                            )

                        show_compare = gr.Checkbox(label="Show side-by-side comparison screen", value=True, info="Split view: original | separator | retouched result")

                        with gr.Row():
                            fast = gr.Checkbox(label="Legacy fast override", value=True,
                                               info="Render Preview is always fast; Export Full Quality is always full resolution.")

                        with gr.Accordion("💾 Session & History", open=False):
                            with gr.Row():
                                save_session_btn = gr.Button("Save Session", variant="secondary", size="sm")
                                load_session_file = gr.File(label="Load Session", file_types=[".json"], file_count="single")
                                undo_btn = gr.Button("↩ Undo", variant="secondary", size="sm", interactive=False)
                                redo_btn = gr.Button("↻ Redo", variant="secondary", size="sm", interactive=False)
                            with gr.Row():
                                snapshot_name = gr.Textbox(label="Snapshot Name", placeholder="e.g. 'warm tone'", scale=3)
                                save_snapshot_btn = gr.Button("📸 Save Snapshot", variant="secondary", size="sm", scale=1)
                                snapshot_dropdown = gr.Dropdown(label="Snapshots", choices=[], scale=3)
                                compare_snapshot_btn = gr.Button("Inspect Settings", variant="secondary", size="sm", scale=1)
                            session_download = gr.File(label="Download Session", visible=False)
                            _undo_stack_state = gr.State(value=None)
                            _snapshot_state = gr.State(value={})
                            _settings_revision_state = gr.State(value=0)
                            _render_preview_mode_state = gr.State(value=MODE_RENDER_PREVIEW)
                            _export_full_mode_state = gr.State(value=MODE_EXPORT_FULL_QUALITY)
                            _render_snapshot_state = gr.State(value=None)
                            _preview_cache_state = gr.State(value=GuiPreviewCache())
                            _smart_analysis_snapshot_state = gr.State(value=None)
                            _smart_proposal_state = gr.State(value=None)
                            snapshot_inspection_out = gr.Textbox(
                                label="Snapshot settings (JSON) — not an image comparison",
                                lines=5,
                                interactive=False,
                                visible=True,
                            )

                    with gr.Group():
                        gr.Markdown("### ⚙️ Export Settings")
                        with gr.Row():
                            export_fmt = gr.Radio(choices=["JPEG", "PNG", "PNG-16", "WebP"], value="JPEG", label="Format", interactive=True, info="PNG-16 = 16-bit container; processed precision is reported in the render status/manifest")
                            export_quality = gr.Slider(10, 100, 95, step=1, label="Compression Quality", info="For JPEG/WebP formats")
                        export_res = gr.Dropdown(
                            choices=["Original", "4K (3840px)", "2K (2048px)", "Full HD (1920px)", "HD (1280px)", "720px"],
                            value="Original", label="Resize / Limit Resolution", interactive=True,
                            info="Downscales image if it exceeds target dimension while maintaining aspect ratio"
                        )
                        quality_tier = gr.Radio(choices=["Full (native face crops)", "Draft (proxy, fast)"], value="Full (native face crops)", label="Processing Quality", interactive=True, info="Full: faces processed at native resolution (F8.2). Draft: legacy proxy path for fast batch contact sheets.")
                        optical_correction = gr.Checkbox(label="Apply Lensfun corrections", value=False, info="Uses camera/lens EXIF. Non-8-bit sources are skipped rather than downconverted; the status explains why.")
                        preserve_source_profile = gr.Checkbox(
                            label="Preserve source ICC profile on export",
                            value=False,
                            info="Off: export working-space sRGB. On: convert back to the embedded source profile explicitly.",
                        )

                # Column 2: Workspace Canvas (Center)
                with gr.Column(scale=4, elem_classes=["viewer-panel"]):
                    with gr.Group():
                        gr.Markdown("### 🖼️ Preview Canvas")
                        img_output = gr.Image(height=600, show_label=False, elem_id="retouch-output")
                        compare_viewer = gr.HTML(visible=False, elem_id="retouch-compare")
                        with gr.Row():
                            inspection_mode = gr.Dropdown(
                                choices=[
                                    ("Fit", INSPECTION_MODE_FIT),
                                    ("100%", INSPECTION_MODE_100_PERCENT),
                                    ("Face", INSPECTION_MODE_FACE),
                                    ("ROI", INSPECTION_MODE_ROI),
                                ],
                                value=INSPECTION_MODE_FIT,
                                label="Native inspection",
                                scale=2,
                            )
                            inspection_face_index = gr.Number(
                                value=0,
                                precision=0,
                                label="Face #",
                                minimum=0,
                                scale=1,
                            )
                            inspection_roi = gr.Textbox(
                                value="",
                                label="ROI JSON",
                                placeholder='{"x": 100, "y": 100, "width": 400, "height": 400}',
                                scale=3,
                            )
                            inspect_render_btn = gr.Button(
                                "Inspect Native Crop",
                                variant="secondary",
                                size="sm",
                                scale=2,
                            )
                        inspection_output = gr.Image(
                            label="Native inspection (download disabled)",
                            height=520,
                            show_download_button=False,
                            visible=False,
                        )
                        inspection_status = gr.Code(
                            label="Inspection contract",
                            language="json",
                            interactive=False,
                            visible=False,
                        )
                        # Latest full-resolution processed recipe result used by
                        # the Advanced Retouch "Edit processed result" path.
                        _processed_result_state = gr.State(value=None)
                        with gr.Accordion("✨ Advanced Retouch", open=False):
                            gr.Markdown(
                                "Paint a mask on the canvas, choose an operation, and apply it. "
                                "Reshape uses the selected detected face; local edits can be restricted "
                                "to semantic regions."
                            )
                            advanced_model_status = gr.Markdown(advanced_model_status_text())
                            advanced_edit_processed_btn = gr.Button(
                                "Edit processed result",
                                variant="secondary",
                            )
                            advanced_bind_legacy_btn = gr.Button(
                                "Bind legacy session edits",
                                variant="secondary",
                                size="sm",
                            )
                            gr.Markdown("Uses the latest recipe output as the Advanced Retouch source.")
                            advanced_editor = gr.ImageEditor(
                                label="Brush mask canvas",
                                type="numpy",
                                image_mode="RGBA",
                                height=520,
                                sources=(),
                                brush=gr.Brush(default_size=32, default_color="#ff304f"),
                                eraser=gr.Eraser(default_size=40),
                                show_download_button=False,
                                elem_id="advanced-retouch-editor",
                            )
                            with gr.Row():
                                advanced_mode = gr.Radio(
                                    choices=["Adjust", "Heal", "Remove", "Reshape"],
                                    value="Adjust", label="Mode", scale=1,
                                )
                                advanced_operation = gr.Dropdown(
                                    choices=list(ADVANCED_OPERATIONS), value="exposure", label="Operation", scale=1,
                                )
                                advanced_strength = gr.Slider(
                                    -100, 100, 25, step=1, label="Strength", scale=1,
                                )
                            with gr.Row():
                                advanced_semantic = gr.Dropdown(
                                    choices=list(ADVANCED_SEMANTIC_MASKS), value="None", label="Semantic intersection",
                                    info="Keeps the brush effect inside the selected parsed region.", scale=2,
                                )
                                advanced_face_select = gr.Dropdown(
                                    choices=["All faces"], value="All faces", label="Face selection", scale=1,
                                )
                                advanced_face_detect_btn = gr.Button("Detect faces", size="sm", variant="secondary", scale=1)
                            with gr.Row():
                                advanced_heal_method = gr.Dropdown(
                                    choices=list(ADVANCED_HEAL_METHODS), value="telea", label="Heal engine", scale=1,
                                )
                                advanced_remove_engine = gr.Dropdown(
                                    choices=list(ADVANCED_REMOVE_ENGINES), value=ADVANCED_REMOVE_ENGINES[0],
                                    label="Remove engine", scale=2,
                                    info="LaMa is used only when models/lama.onnx is installed; otherwise the status reports Telea fallback.",
                                )
                            with gr.Row():
                                advanced_mask_action = gr.Dropdown(
                                    choices=["Keep", "Clear", "Invert"], value="Keep", label="Mask action",
                                    info="Clear or invert the painted mask before applying the edit.", scale=1,
                                )
                                advanced_mask_feather = gr.Slider(
                                    0, 40, 0, step=1, label="Mask feather", info="Softens the painted mask edge.", scale=1,
                                )
                                advanced_clear_mask_btn = gr.Button("Clear mask", size="sm", variant="secondary", scale=1)
                            with gr.Accordion("Face reshape controls", open=False):
                                with gr.Row():
                                    advanced_eye_size = gr.Slider(-100, 100, 0, step=1, label="Eye size")
                                    advanced_eye_distance = gr.Slider(-100, 100, 0, step=1, label="Eye distance")
                                    advanced_nose_width = gr.Slider(-100, 100, 0, step=1, label="Nose width")
                                with gr.Row():
                                    advanced_nose_length = gr.Slider(-100, 100, 0, step=1, label="Nose length")
                                    advanced_jaw_width = gr.Slider(-100, 100, 0, step=1, label="Jaw width")
                                    advanced_chin_length = gr.Slider(-100, 100, 0, step=1, label="Chin length")
                                with gr.Row():
                                    advanced_mouth_size = gr.Slider(-100, 100, 0, step=1, label="Mouth size")
                                    advanced_smile = gr.Slider(-100, 100, 0, step=1, label="Smile")
                                    advanced_forehead = gr.Slider(-100, 100, 0, step=1, label="Forehead")
                                with gr.Row():
                                    advanced_eye_size_l = gr.Slider(-100, 100, 0, step=1, label="Left eye size")
                                    advanced_eye_size_r = gr.Slider(-100, 100, 0, step=1, label="Right eye size")
                                    advanced_nose_width_l = gr.Slider(-100, 100, 0, step=1, label="Left nose width")
                                with gr.Row():
                                    advanced_nose_width_r = gr.Slider(-100, 100, 0, step=1, label="Right nose width")
                                    advanced_jaw_width_l = gr.Slider(-100, 100, 0, step=1, label="Left jaw width")
                                    advanced_jaw_width_r = gr.Slider(-100, 100, 0, step=1, label="Right jaw width")
                            with gr.Row():
                                advanced_apply_btn = gr.Button("Apply edit", variant="primary")
                                advanced_undo_btn = gr.Button("↩ Undo", variant="secondary")
                                advanced_redo_btn = gr.Button("↻ Redo", variant="secondary")
                                advanced_reset_btn = gr.Button("Reset", variant="secondary")
                            with gr.Row():
                                advanced_overlay_visible = gr.Checkbox(label="Show mask overlay", value=True)
                                advanced_overlay_opacity = gr.Slider(0, 100, 42, step=1, label="Overlay opacity")
                            advanced_mask_overlay = gr.Image(label="Mask overlay", show_label=True, height=260)
                            advanced_before_after = gr.Image(label="Before / after", show_label=True, height=260)
                            with gr.Row():
                                advanced_snapshot_name = gr.Textbox(label="Advanced snapshot name", scale=2)
                                advanced_save_snapshot_btn = gr.Button("📸 Save snapshot", size="sm", variant="secondary")
                                advanced_snapshot_dropdown = gr.Dropdown(label="Snapshots", choices=[], scale=2)
                                advanced_compare_snapshot_btn = gr.Button("Compare", size="sm", variant="secondary")
                            with gr.Row():
                                advanced_export_fmt = gr.Radio(choices=["PNG", "PNG-16", "TIFF-16", "JPEG", "WEBP"], value="PNG", label="Export format", scale=1)
                                advanced_export_btn = gr.Button("Download current canvas", size="sm", variant="secondary", scale=1)
                                advanced_export_file = gr.File(label="Advanced export", visible=False, scale=2)
                            advanced_status = gr.Markdown("Advanced Retouch is ready.")
                            _advanced_source_state = gr.State(value=None)
                            _advanced_current_state = gr.State(value=None)
                            _advanced_history_state = gr.State(value=AdvancedHistory().to_state())
                            _advanced_edit_log_state = gr.State(value=[])
                            _advanced_mask_state = gr.State(value=None)
                            _advanced_snapshots_state = gr.State(value={})
                            _advanced_pending_session_state = gr.State(value=[])
                            _advanced_base_contract_state = gr.State(value=None)
                        # Hidden state variables for newly-added parameters (skin_hue_unify, skin_chroma_even)
                        # These maintain alignment with PROCESS_INPUT_KEYS but don't have visible UI yet.
                        _skin_hue_unify_state = gr.State(value=0)
                        _skin_chroma_even_state = gr.State(value=0)
                        _blotch_reduction_state = gr.State(value=0.0)
                        _specular_finish_state = gr.State(value="matte")
                        _specular_finish_strength_state = gr.State(value=0.5)
                        _specular_recolor_state = gr.State(value=0.0)
                        _albedo_even_state = gr.State(value=0.0)
                        _makeup_coverage_even_state = gr.State(value=0.0)
                        _makeup_cake_reduce_state = gr.State(value=0.0)
                        _hemoglobin_smooth_state = gr.State(value=0.0)
                        # mole_protect is a visible slider (below freckle_removal); no State
                        with gr.Accordion("👥 Per-face recipes", open=False):
                            detect_faces_btn = gr.Button("Detect Faces", size="sm", variant="secondary")
                            face_gallery = gr.Gallery(
                                label="Detected faces (detection order)",
                                columns=4, height=140, object_fit="cover",
                            )
                            with gr.Row():
                                face_select = gr.Dropdown(
                                    label="Face index", choices=[], interactive=True, scale=1,
                                )
                                face_recipe = gr.Dropdown(
                                    label="Recipe for face",
                                    choices=RECIPE_UI_CHOICES, value=None, scale=2,
                                )
                                apply_face_btn = gr.Button("Apply to face", size="sm", scale=1)
                            clear_faces_btn = gr.Button("Clear per-face", size="sm")
                            face_params_status = gr.Markdown("")
                            _face_params_state = gr.State(value={})
                            face_params_json = gr.Textbox(
                                label="Advanced: face_params JSON",
                                placeholder='{"0": {"recipe": "cosplay", "smooth": 70}}',
                                lines=2,
                                info="Optional. State from picker wins if set. Engine units 0–100.",
                            )
                        _vein_attenuate_state = gr.State(value=0.0)
                        _gamut_compress_state = gr.State(value=True)
                        _saturation_mode_state = gr.State(value="additive")
                        _reshape_eye_size_state = gr.State(value=0.0)
                        _reshape_eye_distance_state = gr.State(value=0.0)
                        _reshape_nose_width_state = gr.State(value=0.0)
                        _reshape_nose_length_state = gr.State(value=0.0)
                        _reshape_jaw_width_state = gr.State(value=0.0)
                        _reshape_chin_length_state = gr.State(value=0.0)
                        _reshape_mouth_size_state = gr.State(value=0.0)
                        _reshape_smile_state = gr.State(value=0.0)
                        _reshape_forehead_state = gr.State(value=0.0)
                        _hair_deglare_state = gr.State(value=0.0)
                        _hair_ring_position_state = gr.State(value=0.5)
                        _hair_ring_tint_state = gr.State(value=0.0)
                        _hair_remove_flyaways_state = gr.State(value=0.0)
                        _purple_fringing_state = gr.State(value=0.0)
                        _flyaway_cleanup_state = gr.State(value=0.0)
                        _micro_grain_state = gr.State(value=0.0)
                        _split_toning_state = gr.State(value=0.0)
                        _film_strength_state = gr.State(value=0.0)
                        _film_toe_r_state = gr.State(value=0.0)
                        _film_toe_g_state = gr.State(value=0.0)
                        _film_toe_b_state = gr.State(value=0.0)
                        _film_shoulder_r_state = gr.State(value=0.0)
                        _film_shoulder_g_state = gr.State(value=0.0)
                        _film_shoulder_b_state = gr.State(value=0.0)
                        _film_midpoint_state = gr.State(value=0.5)
                        _film_gamma_state = gr.State(value=1.0)
                        _film_crosstalk_cy_mg_state = gr.State(value=0.0)
                        _film_crosstalk_cy_ye_state = gr.State(value=0.0)
                        _film_crosstalk_mg_ye_state = gr.State(value=0.0)
                        _film_tonemap_strength_state = gr.State(value=0.0)
                        _film_tonemap_toe_state = gr.State(value=0.1)
                        _film_tonemap_shoulder_state = gr.State(value=0.1)
                        _film_skew_state = gr.State(value=0.0)
                        _background_harmonize_state = gr.State(value=0.0)
                        _background_harmonize_mode_state = gr.State(value="split")
                        _background_blur_state = gr.State(value=0.0)
                        _background_desaturation_state = gr.State(value=0.0)
                        _light_wrap_state = gr.State(value=0.0)
                        _blue_shadow_grade_state = gr.State(value=0.0)
                        _cyan_midtone_grade_state = gr.State(value=0.0)
                        _subject_sharpen_state = gr.State(value=0.0)
                        _matte_black_state = gr.State(value=0.0)
                        _ai_denoise_state = gr.State(value=0.0)
                        _ai_sr_scale_state = gr.State(value=1)
                        _mv2_eyeshadow_state = gr.State(value=0.0)
                        _mv2_eyeshadow_color_state = gr.State(value="brown")
                        _mv2_eyeshadow_style_state = gr.State(value="natural")
                        _mv2_eyeliner_state = gr.State(value=0.0)
                        _mv2_eyeliner_color_state = gr.State(value="black")
                        _mv2_eyeliner_style_state = gr.State(value="classic")
                        _mv2_contour_state = gr.State(value=0.0)
                        _mv2_brows_state = gr.State(value=0.0)
                        _mv2_brows_color_state = gr.State(value="brown")
                        _mv2_ombre_state = gr.State(value=0.0)
                        _mv2_ombre_color1_state = gr.State(value="red")
                        _mv2_ombre_color2_state = gr.State(value="pink")
                        # Hidden states for params present in params.py but with no visible
                        # slider yet (wiring-debt alignment: every PROCESSING_PARAMS name
                        # must appear in _process_inputs in param_names() order).
                        _wrinkle_soften_forehead_state = gr.State(value=0)
                        _wrinkle_soften_nasolabial_state = gr.State(value=0)
                        _wrinkle_soften_neck_state = gr.State(value=0)
                        _eye_sclera_vessel_remove_state = gr.State(value=0)
                        _backdrop_cleanup_state = gr.State(value=0)
                        _fabric_wrinkle_smooth_state = gr.State(value=0.0)
                        _reshape_jaw_width_l_state = gr.State(value=0.0)
                        _reshape_jaw_width_r_state = gr.State(value=0.0)
                        _reshape_nose_width_l_state = gr.State(value=0.0)
                        _reshape_nose_width_r_state = gr.State(value=0.0)
                        _reshape_eye_size_l_state = gr.State(value=0.0)
                        _reshape_eye_size_r_state = gr.State(value=0.0)
                        _reshape_neck_width_state = gr.State(value=0.0)
                        _reshape_neck_length_state = gr.State(value=0.0)
                        _neural_stray_hair_boost_state = gr.State(value=0)
                        _neural_defect_boost_state = gr.State(value=0)
                        _lens_blur_state = gr.State(value=0.0)
                        _hsl_hue_red_state = gr.State(value=0.0)
                        _hsl_sat_red_state = gr.State(value=0.0)
                        _hsl_lum_red_state = gr.State(value=0.0)
                        _hsl_hue_orange_state = gr.State(value=0.0)
                        _hsl_sat_orange_state = gr.State(value=0.0)
                        _hsl_lum_orange_state = gr.State(value=0.0)
                        _hsl_hue_yellow_state = gr.State(value=0.0)
                        _hsl_sat_yellow_state = gr.State(value=0.0)
                        _hsl_lum_yellow_state = gr.State(value=0.0)
                        _hsl_hue_green_state = gr.State(value=0.0)
                        _hsl_sat_green_state = gr.State(value=0.0)
                        _hsl_lum_green_state = gr.State(value=0.0)
                        _hsl_hue_cyan_state = gr.State(value=0.0)
                        _hsl_sat_cyan_state = gr.State(value=0.0)
                        _hsl_lum_cyan_state = gr.State(value=0.0)
                        _hsl_hue_blue_state = gr.State(value=0.0)
                        _hsl_sat_blue_state = gr.State(value=0.0)
                        _hsl_lum_blue_state = gr.State(value=0.0)
                        _hsl_hue_purple_state = gr.State(value=0.0)
                        _hsl_sat_purple_state = gr.State(value=0.0)
                        _hsl_lum_purple_state = gr.State(value=0.0)
                        _hsl_hue_magenta_state = gr.State(value=0.0)
                        _hsl_sat_magenta_state = gr.State(value=0.0)
                        _hsl_lum_magenta_state = gr.State(value=0.0)
                        _calibration_red_hue_state = gr.State(value=0.0)
                        _calibration_red_sat_state = gr.State(value=0.0)
                        _calibration_red_lum_state = gr.State(value=0.0)
                        _calibration_green_hue_state = gr.State(value=0.0)
                        _calibration_green_sat_state = gr.State(value=0.0)
                        _calibration_green_lum_state = gr.State(value=0.0)
                        _calibration_blue_hue_state = gr.State(value=0.0)
                        _calibration_blue_sat_state = gr.State(value=0.0)
                        _calibration_blue_lum_state = gr.State(value=0.0)
                        _look_params_state = gr.State(value={})
                        status = gr.Textbox(label="Status", interactive=False, placeholder="Upload an image and click Process to start...")
                        smart_analysis_html = gr.HTML(visible=True)
                        qa_status = gr.HTML(visible=True)
                        export_file = gr.File(label="📥 Download Exported Assets")

                    with gr.Group(visible=False) as debug_panel:
                        gr.Markdown("### 🔍 Debug Masks & Frequency Layers")
                        debug_gallery = gr.Gallery(label="Masks (skin, skin+hair, lips, sharpen, glow, freq_low, freq_mid, freq_high)", columns=4, height=300)

                # Column 3: Adjustment Panel (Right)
                with gr.Column(scale=3, elem_classes=["develop-panel"]):
                    with gr.Group():
                        smart_mode_switch = gr.Radio(
                            choices=["Classic", "Smart Color (preview)"],
                            value="Classic",
                            label="Editing mode",
                            info="Smart Color (preview) is a thin, reviewable control surface over the same Classic sliders below — it does not add a second processing engine.",
                        )
                        with gr.Group(visible=False) as smart_color_group:
                            gr.Markdown("### 🎨 Smart Color")
                            gr.Markdown("Preview — Color card only; Face/Auto Polish/Background/Clean not yet available. See `retouch/smart_intents.py` for the underlying contract.")
                            smart_amount = gr.Slider(-1.0, 1.0, 0.0, step=0.05, label="Amount", info="Vibrance + saturation, protecting skin tones less than Classic Vibrance alone")
                            smart_warmth = gr.Slider(-1.0, 1.0, 0.0, step=0.05, label="Warmth", info="White balance temperature + tint")
                            smart_contrast_macro = gr.Slider(-1.0, 1.0, 0.0, step=0.05, label="Contrast", info="Contrast + highlight/shadow rolloff")

                    with gr.Group():
                        gr.Markdown("### ⚙️ Develop Adjustments")
                        
                        with gr.Accordion("✨ Skin Smoothing & Texture", open=True):
                            reset_skin_smooth_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            smooth = gr.Slider(0, 100, 30, step=1, label="Smooth", info="Strength of skin smoothing (blur/median blend)")
                            nose_smooth = gr.Slider(0, 100, 0, step=1, label="Nose Smooth (0 = follow face)", info="Additional smoothing for nose bridge highlights")
                            smooth_engine = gr.Dropdown(choices=["guided", "bilateral", "anisotropic"], value="guided", label="Smoothing Engine", info="guided=isotropic (fast); anisotropic=orientation-aware (preserves wrinkle direction)")
                            regional_modulation = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Region-Aware Modulation", info="Per-region smoothing strength (0=off, 1=full modulation)")
                            mid_reduction = gr.Slider(0.0, 1.0, 0.45, step=0.05, label="Mid Frequency Reduction", info="Target mid-level skin blemishes while preserving high-frequency pores")
                            texture_opacity = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Texture Opacity", info="Control original pore structure opacity overlay")
                            micro_restore = gr.Slider(0, 50, 20, step=1, label="Micro-Texture Restore", info="Re-inject dimensional micro-contrast in cheek/nose/under-eye zones after smoothing (0 = off, 25 = subtle, 50 = strong)")
                            _micro_dodge_burn_state = gr.State(value=0)
                            _redness_even_state = gr.State(value=0)
                            hb_even = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Hemoglobin Even", info="Even redness variation in linear pigment space while preserving melanin marks")
                            hb_shift = gr.Slider(-1.0, 1.0, 0.0, step=0.05, label="Hemoglobin Shift", info="Negative reduces facial flush; positive adds it. Uses the face's own pigment range")
                            _whiten_hue_stable_state = gr.State(value=0)
                            pore_synthesis = gr.Slider(0, 100, 0, step=1, label="Pore Synthesis", info="Add micro-texture/synthesized pores to prevent artificial plastic skin")
                            blemish = gr.Slider(0, 100, 30, step=1, label="Blemish Removal", info="AI blemish detection and inpainting for acne/spots")
                            freckle_removal = gr.Slider(0, 100, 0, step=1, label="Freckle Removal", info="Remove freckles while preserving beauty marks (0=off)")
                            heal_engine = gr.Dropdown(choices=["telea", "patchmatch"], value="telea", label="Auto Heal Engine", info="PatchMatch synthesizes from nearby skin texture; Telea remains the fast default")
                            mark_policy = gr.Dropdown(
                                choices=list(MARK_POLICY_PRESET_NAMES),
                                value="legacy",
                                label="Identity Mark Policy",
                                info="Optional preserve mask for freckles/moles and H4 QA. Legacy keeps existing behavior.",
                            )
                            # FA-02 experimental texture-restoration mode.
                            # Deliberately a gr.State, NOT a visible Dropdown:
                            # "dog"/"multiscale" are unevaluated research arms
                            # with no evidence behind them, so the flag must not
                            # be reachable by clicking around the UI. API/CLI
                            # callers can still set it explicitly.
                            _fa02_texture_mode_state = gr.State(value="legacy")
                            mole_protect = gr.Slider(
                                0.0, 1.0, 0.0, step=0.05,
                                label="Mole / Beauty-Mark Protect",
                                info="R10: protect compact melanin spots from blemish+freckle heals (0=off, 1=full). Classical, no paid corpus.",
                            )
                            skin_flatten = gr.Slider(0, 100, 0, step=1, label="Skin Flatten (Anime)", info="Edge-preserving cel flatten for anime-style shading · 0=off, 80=aggressive")
                            skin_quantize = gr.Slider(0, 100, 0, step=1, label="Tone Quantize (Anime)", info="Cel shading colour bands on skin · 0=off, 60=dramatic bands")

                        with gr.Accordion("🎨 Skin Tone", open=False):
                            reset_skin_tone_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            whiten = gr.Slider(0, 100, 10, step=1, label="Whitening", info="Luminance boost and porcelain skin color match")
                            whiten_tone = gr.Dropdown(choices=WHITEN_TONE_CHOICES, value="rosy", label="Whitening Tone", interactive=True, info="Tone direction: rosy (warm pink), porcelain (cool neutral), neutral")
                            equalize = gr.Slider(0, 100, 20, step=1, label="Equalize", info="Even out skin redness and regional color inconsistencies")
                            shadow_lift = gr.Slider(0, 100, 0, step=1, label="Shadow Lift", info="Brighten small localized face shadows relative to local neighborhood")
                            nose_restore = gr.Slider(0, 100, 0, step=1, label="Nose Restore", info="Blend original (pre-retouch) nose pixels back in, to preserve natural nose shading")
                            skin_sss = gr.Slider(0, 100, 0, step=1, label="Subsurface Scatter", info="Game-render skin translucency: red-weighted shading diffusion + warm shadow terminators (pores stay crisp)")
                            skin_unify = gr.Slider(0, 100, 0, step=1, label="Skin Hue Unify (Anime)", info="Pull skin hues toward a single cel color · 0=off, 60=strong unified look")
                            skin_unify_hue = gr.Slider(-1.0, 360.0, -1.0, step=1.0, label="Target Hue (Anime)", info="Target skin hue angle · -1=auto (detect from face), 0=red, 50=orange, 180=cyan")
                            auto_exposure = gr.Checkbox(label="Auto Exposure Correction", value=False, info="Automatically correct under/over-exposed images before processing")
                            white_costume_lift = gr.Checkbox(label="White Costume Lift", value=False, info="Selectively boost bright clothing to create separation")
                            face_exposure = gr.Slider(0, 100, 0, step=1, label="Face Exposure Lift", info="Brighten/darken the exposed face relative to the body (skin.face_exposure) · 0 = off")

                        with gr.Accordion("🦵 Body Skin", open=False):
                            reset_body_skin_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            body_smooth = gr.Slider(0, 100, 0, step=1, label="Body Smooth", info="Smoothing for arms, legs, décolletage · milder curve than face to preserve texture")
                            body_equalize = gr.Slider(0, 100, 0, step=1, label="Body Equalize", info="Even out tone in body skin regions · tone harmonization at body scale")
                            body_whiten = gr.Slider(0, 100, 0, step=1, label="Body Whiten", info="Lighten body skin to match face whitening treatment")
                            body_match_face = gr.Slider(0, 100, 0, step=1, label="Body Match Face", info="Pull body skin L/a/b toward retouched face skin color · bounded ±8L ±6a/b")
                            body_relight = gr.Slider(0, 100, 0, step=1, label="Body Relight", info="Landmark-free directional shading on exposed body skin · matches face relight intensity")
                            body_dodge_burn = gr.Slider(0, 100, 0, step=1, label="Body Dodge & Burn", info="Local-contrast sculpting on body skin (CLAHE-based highlight/shadow)")
                            body_shadow_lift = gr.Slider(0, 100, 0, step=1, label="Body Shadow Lift", info="Brighten small localized shadows on body skin")

                        with gr.Accordion("📊 Basic Tone & Color", open=False):
                            reset_basic_tone_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            contrast = gr.Slider(-50, 50, 0, step=1, label="Contrast", info="Adjust global image contrast")
                            brightness = gr.Slider(-50, 50, 0, step=1, label="Brightness", info="Adjust global image brightness")
                            clarity = gr.Slider(-100, 100, 0, step=1, label="Clarity", info="Mid-tone contrast / local contrast enhancement (negative = soften)")
                            vibrance = gr.Slider(-100, 100, 0, step=1, label="Vibrance", info="Smart saturation boost that protects skin tones")
                            saturation = gr.Slider(-100, 100, 0, step=1, label="Saturation", info="Uniform global saturation adjustment")

                        with gr.Accordion("📈 Tone Curve", open=False):
                            reset_tone_curve_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            highlights = gr.Slider(-100, 100, 0, step=1, label="Highlights", info="Recover or boost bright highlight regions")
                            shadows = gr.Slider(-100, 100, 0, step=1, label="Shadows", info="Open up or deepen shadow regions")
                            whites = gr.Slider(-100, 100, 0, step=1, label="Whites", info="Control absolute white point ceiling")
                            blacks = gr.Slider(-100, 100, 0, step=1, label="Blacks", info="Control absolute black point floor")

                        with gr.Accordion("💡 Virtual Studio Relighting", open=False):
                            reset_relighting_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            relight = gr.Slider(0, 100, 0, step=1, label="Relight Strength", info="Intensity of 3D virtual studio light source redirection")
                            relight_azimuth = gr.Slider(-180, 180, 0, step=1, label="Light Azimuth", info="Horizontal light source direction angle (-180° to 180°)")
                            relight_elevation = gr.Slider(-90, 90, 30, step=1, label="Light Elevation", info="Vertical light source direction angle (-90° to 90°)")
                            sculpt = gr.Slider(0, 100, 0, step=1, label="Facial Sculpting", info="Shape reflectance: deepen cheekbones, nose ridge, and jawline via low-band shading")
                            shine_removal = gr.Slider(0, 100, 0, step=1, label="Shine Removal", info="Remove oily/sweaty shine: compress specular highlights and reconstruct chroma")
                            wrinkle_soften = gr.Slider(0, 100, 0, step=1, label="Wrinkle & Line Softening", info="Reduce nasolabial folds, forehead lines, and crow's feet via ridge-aware attenuation")
                            texture_transplant = gr.Slider(0, 100, 0, step=1, label="Texture Transplant", info="Clone pore texture from clean skin regions to over-smoothed/inpainted zones for realistic texture")

                        with gr.Accordion("👁️ Eyes & Lips", open=False):
                            reset_eyes_lips_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            eye_enhance = gr.Slider(0, 100, 5, step=1, label="Eye Enhance", info="Boost eye clarity, iris reflection details, and whites brightness")
                            catchlight = gr.Slider(0, 100, 0, step=1, label="Catchlight Boost", info="Amplify existing catchlight highlights in the iris (0 = follow Eye Enhance)")
                            corneal_shading = gr.Slider(0, 100, 0, step=1, label="Corneal Curvature", info="3D spherical corneal shading for eye depth and wetness (0 = off)")
                            dark_circles = gr.Slider(0, 100, 0, step=1, label="Dark Circle Repair", info="Under-eye dark circle detection and repair")
                            undereye_shadow_strength = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Under-Eye Shadow Smooth", info="Soften under-eye shadows conservatively (0=off)")
                            undereye_darken_removal = gr.Slider(0, 100, 0, step=1, label="Under-Eye Darken Removal", info="Lift under-eye darkening / discoloration (0=off)")
                            undereye_puffiness_reduction = gr.Slider(0, 100, 0, step=1, label="Under-Eye Puffiness Reduction", info="Reduce under-eye puffiness / bag volume (0=off)")
                            eye_sclera_brighten = gr.Slider(0, 100, 0, step=1, label="Sclera Brighten", info="Whiten/brighten the eye whites (sclera) for a clean look")
                            eye_iris_saturate = gr.Slider(0, 100, 0, step=1, label="Iris Saturate", info="Deepen iris color saturation")
                            eye_iris_brightness = gr.Slider(0, 100, 0, step=1, label="Iris Brightness", info="Brighten iris detail and reflection")
                            eye_iris_hue_shift = gr.Slider(-30, 30, 0, step=1, label="Iris Hue Shift", info="Rotate iris hue for colored-contact effects (-30..30°)")
                            teeth_whiten = gr.Slider(0, 100, 5, step=1, label="Teeth Whiten", info="Naturally whiten and brighten teeth enamel")
                            lip_enhance = gr.Slider(0, 100, 5, step=1, label="Lip Enhance", info="Enhance lip texture definition, gloss, and contour")
                            lip_tint = gr.Dropdown(choices=LIP_TINTS, value="none", label="Lip Tint Color", interactive=True, info="Apply a natural cosmetic tint overlay")
                            lip_finish = gr.Dropdown(choices=LIP_FINISH_CHOICES, value="gloss", label="Lip Finish", interactive=True, info="Surface finish style: gloss (shiny), matte (flat), velvet (soft)")
                            blush = gr.Slider(0, 100, 0, step=1, label="Blush Strength", info="Intensity of virtual cosmetic blush on cheeks")
                            with gr.Row():
                                nose_blush = gr.Checkbox(label="Nose Blush", value=False, info="Add cosmetic pink tone to nose tip")
                                under_eye_blush = gr.Checkbox(label="Under-Eye Blush", value=False, info="Apply soft under-eye blush for a fresh/cosplay look")
                                eye_gate = gr.Checkbox(label="Eye Occlusion Gate", value=True, info="Skip enhancing eyes detected as closed/occluded (prevents painting an iris onto hair or a closed lid)")

                        with gr.Accordion("🧬 Face Reshaping", open=False):
                            reset_face_reshaping_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            slimming = gr.Slider(0, 100, 0, step=1, label="Face Slimming", info="Liquify-based face slimming/reshaping via landmark-driven warp")

                        with gr.Accordion("🌟 Structure & Effects", open=False):
                            reset_structure_effects_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            hair_enhance = gr.Slider(0, 100, 5, step=1, label="Hair Shine", info="Boost highlight reflections and depth in hair strands")
                            dodge_burn = gr.Slider(0, 100, 0, step=1, label="Dodge & Burn", info="Sculpt face structure with local highlight/shadow contouring")
                            impact = gr.Slider(0, 100, 0, step=1, label="Global Impact Finish", info="Final punch: combined clarity, sharpening, and micro-contrast boost")
                            specular_bloom = gr.Slider(0, 100, 0, step=1, label="Specular Bloom", info="Dreamy bloom glow applied specifically to skin highlight zones")
                            specular_bloom_tone = gr.Dropdown(choices=SPECULAR_BLOOM_TONE_CHOICES, value="rosy", label="Specular Bloom Tone", interactive=True, info="Color tint of the specular bloom glow")
                            bloom = gr.Slider(0, 100, 0, step=1, label="Orton Bloom (Overall Glow)", info="High-key glow blending for high-fashion portraits")
                            bloom_threshold = gr.Slider(150, 250, 210, step=1, label="Bloom Threshold", info="Brightness threshold where the glow begins to bleed")
                            bloom_softness = gr.Slider(1, 100, 30, step=1, label="Bloom Softness", info="Softness blur radius of the bloom filter")
                            sharpen = gr.Slider(0, 100, 0, step=1, label="Selective Sharpening", info="Sharpen eyes, eyebrows, and hair edges (mask-driven)")
                            sharpen_radius = gr.Slider(0.1, 5.0, 1.0, step=0.1, label="Sharpen Radius", info="Blur radius for unsharp mask kernel")
                            glow = gr.Slider(0, 100, 0, step=1, label="Atmospheric Glow", info="Multi-scale atmospheric glow/bloom effect")
                            skin_glow = gr.Slider(0, 100, 0, step=1, label="Skin Light-Wrap (Anime)", info="Skin-scoped diffusion glow / light-wrap for anime cel blending · 0=off, 30=visible halo")
                            mask_feather_mode = gr.Dropdown(
                                choices=["gaussian", "guided"],
                                value="gaussian",
                                label="Mask Edge Refinement",
                                info="Gaussian is conservative; guided preserves fine wig, hairline, and lash edges.",
                            )
                            vignette = gr.Slider(0, 100, 0, step=1, label="Vignette", info="Darken image corners for a focused portrait look")
                            fade_toe = gr.Slider(0, 100, 0, step=1, label="Fade Toe", info="Lift shadows while preserving hue (L-only LAB fade for 透明感)")
                            highlight_drift = gr.Slider(0, 100, 0, step=1, label="Highlight Drift", info="Bounded cyan hue rotation in highlights with skin protection")
                            airy_haze = gr.Slider(0, 100, 0, step=1, label="Airy Haze", info="L-threshold-scoped atmospheric glow for 空気感 effect")
                            clarity_split_neg = gr.Slider(0, 100, 0, step=1, label="Clarity Split (Form)", info="Reduce form-band local contrast for soft look")
                            clarity_split_pos = gr.Slider(0, 100, 0, step=1, label="Clarity Split (Texture)", info="Boost texture-band micro-contrast for detail")
                            subject_separation = gr.Slider(0, 100, 0, step=1, label="Subject-Background Separation", info="Brighten subject / darken background using person segmentation mask")

                        with gr.Accordion("🎭 Cosplay Moat (A3)", open=False):
                            gr.Markdown("Cosplay-specific skin / wardrobe continuity (wig lace blend, stockings smooth, cross-shot consistency).")
                            cosplay_wig_lace_blend = gr.Slider(0, 100, 0, step=1, label="Wig Lace Blend", info="Fade wig lace edge into forehead skin")
                            cosplay_stockings_smooth = gr.Slider(0, 100, 0, step=1, label="Stockings Smooth", info="Smooth hosiery / stocking texture")
                            cosplay_consistency_strength = gr.Slider(0, 100, 0, step=1, label="Consistency Strength", info="Cross-shot lighting / white-balance continuity for a cosplay set")

                        with gr.Accordion("🦵 Body Reshape (T3)", open=False):
                            gr.Markdown("Landmark-driven body reshape via MediaPipe Pose (±15% segment displacement at ±100). 50 = no change.")
                            body_reshape_arm_length = gr.Slider(0, 100, 50, step=1, label="Arm Length", info="0 = shorter, 100 = longer arms")
                            body_reshape_leg_length = gr.Slider(0, 100, 50, step=1, label="Leg Length", info="0 = shorter, 100 = longer legs")
                            body_reshape_torso_width = gr.Slider(0, 100, 50, step=1, label="Torso Width", info="0 = narrower, 100 = wider torso")
                            body_reshape_shoulder_width = gr.Slider(0, 100, 50, step=1, label="Shoulder Width", info="0 = narrower, 100 = wider shoulders")
                            body_reshape_hip_width = gr.Slider(0, 100, 50, step=1, label="Hip Width", info="0 = narrower, 100 = wider hips")
                            auto_body_reshape = gr.Slider(0, 100, 0, step=1, label="Auto Body Reshape", info="Automatic proportional reshape strength (0 = off)")

                        with gr.Accordion("🎬 Film Color Grading", open=False):
                            reset_color_grading_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            color_grade = gr.Dropdown(choices=COLOR_GRADE_NAMES, value="natural", label="Color Grade Preset", interactive=True, info="Apply a film/color grading preset from the presets library")
                            grade_intensity = gr.Slider(0, 100, 0, step=1, label="Grade Intensity", info="Blend strength of the color grade (0-100%)")

                        with gr.Accordion("🎞️ Film & Analog Effects", open=False):
                            reset_film_effects_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            film_enable = gr.Checkbox(label="Enable Film Density Engine", value=False, info="Required for film highlight controls")
                            film_highlight_purity = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Highlight Purity", info="Roll saturated highlights naturally toward white instead of neon yellow or orange")
                            chromatic_aberration = gr.Slider(0, 20, 0, step=0.5, label="Chromatic Aberration", info="Lens fringing effect (RGB channel shift in pixels)")
                            grain = gr.Slider(0, 100, 0, step=1, label="Film Grain", info="Analog film grain noise overlay (0-100 maps to engine 0.0-0.2)")
                            halation = gr.Slider(0, 100, 0, step=1, label="Halation", info="Red light bloom around bright highlights (0-100 maps to engine 0.0-1.0)")
                            tonal_curve_strength = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Tonal Curve", info="Film H&D tonal curve strength (lifted blacks + S-curve)")
                            skin_protect_strength = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Skin Protection", info="Preserve skin hues during color grading ops")
                            grain_strength = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Organic Grain", info="Clumped luminance-correlated film grain (Fuji-style)")
                            highlight_rolloff_strength = gr.Slider(0.0, 1.0, 0.0, step=0.05, label="Highlight Rolloff", info="Soft C¹-continuous highlight compression")
                            lut = gr.Dropdown(choices=LUT_CHOICES, value="none", label="Film Emulation LUT", interactive=True, info="Apply a film stock emulation LUT (Kodak / Fuji)")

                        with gr.Accordion("🌈 Split Toning", open=False):
                            reset_split_toning_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            gr.Markdown("**Shadows**")
                            shadow_hue = gr.Slider(0, 360, 0, step=1, label="Shadow Hue", info="Hue shift applied to shadow tones (degrees)")
                            shadow_sat = gr.Slider(0, 100, 0, step=1, label="Shadow Saturation", info="Saturation boost for shadow tones")
                            gr.Markdown("**Midtones**")
                            midtone_hue = gr.Slider(0, 360, 0, step=1, label="Midtone Hue", info="Hue shift applied to midtone tones (degrees)")
                            midtone_sat = gr.Slider(0, 100, 0, step=1, label="Midtone Saturation", info="Saturation boost for midtone tones")
                            gr.Markdown("**Highlights**")
                            highlight_hue = gr.Slider(0, 360, 0, step=1, label="Highlight Hue", info="Hue shift applied to highlight tones (degrees)")
                            highlight_sat = gr.Slider(0, 100, 0, step=1, label="Highlight Saturation", info="Saturation boost for highlight tones")

                        with gr.Accordion("🎨 LCH Color Tools", open=False):
                            reset_lch_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            gr.Markdown("**White Balance**")
                            white_balance_kelvin = gr.Slider(2000, 12000, 6500, step=100, label="Temperature (K)", info="2000=warm candlelight, 6500=neutral daylight, 12000=cool shade")
                            white_balance_tint = gr.Slider(-100, 100, 0, step=1, label="Tint", info="Negative=green correction, positive=magenta correction")
                            gr.Markdown("**B&W Channel Mixer**")
                            bw_channel_mixer_r = gr.Slider(-100, 200, 30, step=1, label="Red Weight", info="Red channel weight for B&W conversion")
                            bw_channel_mixer_g = gr.Slider(-100, 200, 59, step=1, label="Green Weight", info="Green channel weight for B&W conversion")
                            bw_channel_mixer_b = gr.Slider(-100, 200, 11, step=1, label="Blue Weight", info="Blue channel weight for B&W conversion")
                            gr.Markdown("**Negative Split Tone**")
                            negative_split_tone_shadow = gr.Slider(0, 100, 0, step=1, label="Shadow Desaturation", info="Fade shadows toward grayscale")
                            negative_split_tone_highlight = gr.Slider(0, 100, 0, step=1, label="Highlight Desaturation", info="Fade highlights toward grayscale")
                            gr.Markdown("**Master HSL**")
                            hsl_hue_global = gr.Slider(-100, 100, 0, step=1, label="Hue Shift", info="Global hue rotation in LCH space")
                            hsl_sat_global = gr.Slider(-100, 100, 0, step=1, label="Saturation", info="Global perceptual saturation ±100%")
                            hsl_lum_global = gr.Slider(-100, 100, 0, step=1, label="Luminance", info="Global L* lightness ±100")

                        with gr.Accordion("🔮 Color Transfer", open=False):
                            reset_color_transfer_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            gr.Markdown("Upload a reference image to match its color tone using CDF-based histogram transfer")
                            color_ref_img = gr.Image(type="filepath", label="Reference Image", show_label=True, height=160)
                            color_ref_strength = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Transfer Strength", info="Mix ratio between original grade and matched reference grade (1.0 = full transfer, 0.0 = no transfer)")

                        with gr.Accordion("🎨 Look Extractor (F6)", open=False):
                            gr.Markdown("Upload a reference image to reverse-engineer an editable tone/color look. The extracted params are applied on the next Process (overriding recipe defaults).")
                            look_ref_file = gr.File(label="Reference Image (look source)", file_types=["image", *sorted(RAW_EXTENSIONS)], file_count="single")
                            look_extract_btn = gr.Button("✨ Extract Look", variant="secondary", size="sm", elem_classes=["secondary-btn"])
                            look_status = gr.Markdown("")

                        with gr.Accordion("🎞️ LUT Library", open=False):
                            gr.Markdown("Hot-reload the 3D LUT registry after adding/removing `.cube` files in the luts directory.")
                            reload_luts_btn = gr.Button("🔄 Reload LUTs", variant="secondary", size="sm", elem_classes=["secondary-btn"])
                            reload_luts_status = gr.Markdown("")

                        with gr.Accordion("🔍 Debug & Mask Preview", open=False):
                            reset_debug_btn = gr.Button("↺ Reset Section", size="sm", elem_classes=["secondary-btn", "section-reset-btn"])
                            debug_mode = gr.Checkbox(label="Generate Debug Masks", value=False, info="Save skin/lips/frequency-layer masks and display them for tuning")

                        with gr.Accordion("🩺 Diagnostics", open=False):
                            gr.Markdown("Version/environment bundle for bug reports. Click, then copy the text below.")
                            with gr.Row():
                                diagnostics_btn = gr.Button("📋 Generate Diagnostics", variant="secondary", size="sm", elem_classes=["secondary-btn"])
                                runtime_doctor_btn = gr.Button("🩺 Runtime Doctor", variant="secondary", size="sm", elem_classes=["secondary-btn"])
                                clear_diagnostics_btn = gr.Button("🗑️ Clear Diagnostics", variant="secondary", size="sm", elem_classes=["secondary-btn"])
                            diagnostics_status = gr.Markdown("")
                            diagnostics_out = gr.Textbox(label="Diagnostics", lines=8, interactive=False)
                            runtime_doctor_out = gr.Textbox(
                                label="Runtime Doctor (static; no native probe)",
                                lines=16,
                                interactive=False,
                            )
                            render_manifest_out = gr.Code(
                                label="Render Manifest (pixel-free)",
                                language="json",
                                lines=18,
                                interactive=False,
                            )

                        process_btn_bottom = gr.Button("Render Preview ⚡", variant="primary", size="lg", elem_classes=["primary-btn"])

                        gr.HTML("""
                        <div style="margin-top:12px;text-align:center;font-size:0.7rem;color:rgba(255,255,255,0.4);border-top:1px solid rgba(255,255,255,0.06);padding-top:10px">
                            <span>⌘+Enter Process · ⌘+R Reset · Click preview for full-size</span>
                        </div>
                        """)


        with gr.Tab("Batch Library Ingestion"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Row():
                        folder_in = gr.Textbox(label="Input Folder Path", placeholder="/path/to/photos", info="Absolute path to directory containing raw/jpeg source photos.", scale=4)
                        folder_in_browse = gr.Button("📂 Browse", scale=1, min_width=90)
                    with gr.Row():
                        folder_out = gr.Textbox(label="Output Folder Path", placeholder="/path/to/exports", info="Absolute path where processed results will be written.", scale=4)
                        folder_out_browse = gr.Button("📂 Browse", scale=1, min_width=90)

                    folder_in_browse.click(fn=pick_folder_dialog, inputs=[folder_in], outputs=[folder_in])
                    folder_out_browse.click(fn=pick_folder_dialog, inputs=[folder_out], outputs=[folder_out])

                    with gr.Group():
                        gr.Markdown("### Style Mode")
                        batch_style_type = gr.Radio(choices=["Use Standard Recipe", "Use Custom Style"], value="Use Standard Recipe", label="Style Mode", info="Choose whether to apply a built-in recipe preset or a custom learned style profile.")
                        batch_recipe = gr.Dropdown(choices=RECIPE_UI_CHOICES, value="natural", label="Standard Recipe", info="Recommended recipes are correction-first; scene / creative recipes are conditional.")

                        with gr.Accordion("📖 Recipe Cookbook", open=False):
                            with gr.Row():
                                batch_cookbook_category = gr.Dropdown(
                                    label="Category",
                                    choices=["All"] + list_categories(),
                                    value="All",
                                    interactive=True,
                                    scale=1,
                                )
                                batch_cookbook_search = gr.Textbox(
                                    label="Search",
                                    placeholder="e.g. cosplay, portrait, fuji",
                                    scale=2,
                                )
                            with gr.Row():
                                batch_cookbook_search_btn = gr.Button(
                                    "Search / Browse", variant="secondary", size="sm",
                                    elem_classes=["secondary-btn"],
                                )
                            batch_cookbook_dropdown = gr.Dropdown(
                                label="Cookbook Recipe",
                                choices=[],
                                interactive=True,
                                value=None,
                                allow_custom_value=False,
                                info="Browse by category or search, then select to load into Standard Recipe",
                            )
                            batch_cookbook_status = gr.Markdown(
                                "Open accordion → pick category or search → select recipe."
                            )

                        batch_custom_style = gr.Dropdown(choices=custom_style_choices, value=None, label="Custom Style Profile", interactive=True, visible=False, info="Select a custom style profile from your library.")
                    
                    with gr.Group():
                        gr.Markdown("### Export Formatting")
                        batch_fmt = gr.Radio(choices=["JPEG", "PNG", "WebP"], value="JPEG", label="Format", interactive=True)
                        batch_quality = gr.Slider(10, 100, 95, step=1, label="Quality")
                        batch_res = gr.Dropdown(
                            choices=["Original", "4K (3840px)", "2K (2048px)", "Full HD (1920px)", "HD (1280px)", "720px"],
                            value="Original", label="Export Resolution", interactive=True
                        )
                        
                    with gr.Row():
                        auto_group_toggle = gr.Checkbox(label="Enable Rule-Based Auto-Grouping", value=True, info="Group similar scenes to ensure visual coherence across outputs.")
                        sheet_toggle = gr.Checkbox(label="Generate Contact Sheet", value=True, info="Generate a printable contact grid sheet for all processed photos.")
                        zip_toggle = gr.Checkbox(label="Package into ZIP", value=True, info="Archive all output files into a single downloadable .zip file.")
                        
                    batch_btn = gr.Button("Process Entire Folder 🚀", variant="primary", size="lg", elem_classes=["primary-btn"])
                    
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 📋 Automation Output")
                        batch_sheet_out = gr.Image(label="Generated Contact Sheet", height=320)
                        batch_zip_out = gr.File(label="Download Packaged ZIP")
                        batch_status = gr.Textbox(label="Execution Log & Statistics", lines=12, interactive=False, placeholder="Click 'Process Entire Folder' to start batch processing...")

        with gr.Tab("Job Dashboard"):
            with gr.Group() as job_list_group:
                gr.Markdown("### Batch Job History")
                job_refresh_btn = gr.Button("Refresh 🔄")
                job_list_df = gr.Dataframe(
                    headers=["job_id", "created", "status", "recipe", "total_files", "flagged"],
                    interactive=False,
                    label="Past batch jobs (click a row to view details)",
                )

            with gr.Group(visible=False) as job_detail_group:
                gr.Markdown("### Job Detail")
                job_detail_log = gr.Textbox(label="Job Log", lines=4, interactive=False)
                job_detail_df = gr.Dataframe(
                    headers=["source_path", "status", "qa_state", "detectors"],
                    interactive=False,
                    label="Per-file results",
                )
                job_selected_id = gr.State(value=None)
                job_rerun_btn = gr.Button("Re-run Flagged Files ⚠️", variant="primary")
                job_rerun_status = gr.Textbox(label="Re-run Status", lines=3, interactive=False)

        with gr.Tab("Shoot Intelligence"):
            gr.Markdown(
                "Build an explainable shoot map, review burst candidates, and keep subject/look metadata linked without storing face pixels. "
                "Culling recommendations never delete or reject source files."
            )
            with gr.Row():
                shoot_folder = gr.Textbox(
                    label="Shoot folder",
                    placeholder="/path/to/shoot",
                    info="Folder containing JPEG/PNG/TIFF captures."
                )
                shoot_recursive = gr.Checkbox(label="Scan subfolders", value=True)
                shoot_face_quality = gr.Checkbox(
                    label="Measure face/eye sharpness",
                    value=False,
                    info="Optional evidence only; blink analysis remains unavailable.",
                )
                shoot_scan_btn = gr.Button("Scan shoot", variant="primary")
            shoot_status = gr.Markdown("No shoot scanned yet.")
            shoot_rows = gr.Dataframe(
                headers=["kind", "group", "asset ID", "path", "score", "evidence", "rank/time", "status"],
                interactive=False,
                label="Assets and burst candidates",
                wrap=True,
            )
            shoot_graph_json = gr.Code(label="Dependency/status graph", language="json", interactive=False)
            with gr.Row():
                watch_folder = gr.Textbox(label="Watch folder", placeholder="/path/to/incoming-shoot")
                watch_state = gr.Textbox(label="State file (optional)", placeholder="/path/to/watch-state.json")
                watch_limit = gr.Number(label="Max files per pass", value=20, precision=0)
                watch_btn = gr.Button("Scan stable files", variant="secondary")
            with gr.Row():
                watch_output_dir = gr.Textbox(
                    label="Job output folder (optional)",
                    placeholder="Leave empty for scan-only; must be outside watch folder",
                )
                watch_job_kind = gr.Dropdown(label="Job kind", choices=["preview", "final"], value="preview")
                watch_recipe = gr.Dropdown(label="Job recipe", choices=RECIPE_UI_CHOICES, value="natural")
            watch_status = gr.Markdown("")
            with gr.Accordion("Persistent Shoot Review Manifest", open=False):
                shoot_manifest_path = gr.Textbox(
                    label="Manifest path (optional for scan)",
                    placeholder="Defaults to <shoot-folder>/.retouch-shoot-review.json",
                )
                with gr.Row():
                    review_asset_id = gr.Textbox(label="Asset ID")
                    review_decision = gr.Dropdown(label="Human decision", choices=sorted(DECISIONS), value="hold")
                    review_rating = gr.Number(label="Rating (0-5)", precision=0)
                with gr.Row():
                    review_labels = gr.Textbox(label="Labels (comma-separated)")
                    review_reviewer = gr.Textbox(label="Reviewer")
                review_note = gr.Textbox(label="Review note")
                review_save_btn = gr.Button("Save human review", variant="secondary")
                review_export_selected = gr.Checkbox(label="CSV: selected assets only", value=False)
                with gr.Row():
                    review_json_btn = gr.Button("Export JSON")
                    review_csv_btn = gr.Button("Export CSV")
                review_export_file = gr.File(label="Review export", interactive=False)
                review_status = gr.Markdown("")
            with gr.Accordion("Subject-linked project profile", open=False):
                with gr.Row():
                    profile_id = gr.Textbox(label="Profile ID", placeholder="subject-001")
                    profile_subject_key = gr.Textbox(label="Subject key", placeholder="subject-001")
                    profile_display_name = gr.Textbox(label="Display name", placeholder="Portrait subject")
                with gr.Row():
                    profile_recipe = gr.Dropdown(label="Default recipe", choices=RECIPE_UI_CHOICES, value="natural")
                    profile_style = gr.Textbox(label="Style profile (optional)")
                profile_params = gr.Code(label="Preferred parameters JSON", language="json", value="{}")
                profile_marks = gr.Code(label="Protected marks JSON", language="json", value="[]")
                profile_save_btn = gr.Button("Save project profile", variant="secondary")
                profile_status = gr.Markdown("")
            with gr.Accordion("Multi-reference Look Board", open=False):
                with gr.Row():
                    look_board_id = gr.Textbox(label="Look Board ID", placeholder="wedding-daylight")
                    look_board_project = gr.Textbox(label="Linked profile ID (optional)")
                    look_board_refs = gr.File(label="Reference images", file_types=["image", *sorted(RAW_EXTENSIONS)], file_count="multiple")
                look_board_save_btn = gr.Button("Save Look Board", variant="secondary")
                look_board_apply_btn = gr.Button("Apply Look Board to Process", variant="primary")
                look_board_status = gr.Markdown("")
                look_board_json = gr.Code(label="Look Board manifest", language="json", interactive=False)

        with gr.Tab("Custom Style Library"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Save Sliders as Custom Style")
                    save_name = gr.Textbox(label="Style Name", placeholder="e.g. Dennis Cosplay v4", info="Give your custom style preset a unique name.")
                    save_author = gr.Textbox(label="Author", value="Dennis")
                    save_tags = gr.Textbox(label="Tags (comma-separated)", placeholder="moody, cosplay, soft")
                    save_style_btn = gr.Button("Save Sliders to Style Library 💾", variant="primary", elem_classes=["primary-btn"])
                    save_status = gr.Textbox(label="Save Status", interactive=False)
                    
                with gr.Column(scale=1):
                    gr.Markdown("### Style Dataset Learning (Pairs Extractor)")
                    learn_orig_dir = gr.Textbox(label="Original Folder Path", placeholder="/path/to/originals", info="Directory containing original, un-retouched photos.")
                    learn_edit_dir = gr.Textbox(label="Edited Folder Path", placeholder="/path/to/edited", info="Directory containing matching edited/retouched photos (same filenames).")
                    learn_name = gr.Textbox(label="Learned Style Name", placeholder="e.g. Dennis_Cosplay_V7", info="Unique name for the learned style profile.")
                    learn_author = gr.Textbox(label="Author", value="Dennis")
                    learn_tags = gr.Textbox(label="Tags (comma-separated)", placeholder="learned, cosplay")
                    learn_style_btn = gr.Button("Extract & Learn Style from Dataset 🧠", variant="primary", elem_classes=["primary-btn"])
                    learn_status = gr.Textbox(label="Learning Status", lines=5, interactive=False)

    def on_batch_style_change(style_type):
        if style_type == "Use Custom Style":
            return [gr.update(visible=False), gr.update(visible=True)]
        else:
            return [gr.update(visible=True), gr.update(visible=False)]

    # Event binding setup
    # RECIPE_OUTPUT_KEYS -- canonical ordered list of UI-output param names.
    # It is the single ordering contract for every producer that fills the
    # recipe / custom-style / smart-process slider tuple.  Producers must emit
    # values keyed by THIS list (never a hand-ordered tuple), so a newly added
    # param cannot silently shift every later slider into the wrong value
    # (the historical "brightness lands in the blush slider" footgun).
    RECIPE_OUTPUT_KEYS = (
        "smooth",
 "mid_reduction",
 "texture_opacity",
 "pore_synthesis",
 "nose_smooth",
        "regional_modulation",
 "smooth_engine",
 "undereye_shadow_strength",
 "freckle_removal",
 "micro_restore",
        "whiten",
 "equalize",
 "blemish",
 "whiten_tone",
 "nose_blush",
        "under_eye_blush",
 "white_costume_lift",
 "body_smooth",
 "body_equalize",
 "body_whiten",
        "body_match_face",
 "dodge_burn",
 "relight",
 "relight_azimuth",
 "relight_elevation",
        "sculpt",
 "shine_removal",
 "wrinkle_soften",
 "specular_bloom",
 "specular_bloom_tone",
        "skin_flatten",
 "skin_quantize",
 "skin_unify",
 "skin_unify_hue",
 "skin_glow",
 "mask_feather_mode",
 "skin_sss",
        "eye_enhance",
 "catchlight",
 "corneal_shading",
 "dark_circles",
 "undereye_darken_removal",
 "undereye_puffiness_reduction",
        "eye_sclera_brighten",
 "eye_iris_saturate",
 "eye_iris_hue_shift",
 "eye_iris_brightness",
 "teeth_whiten",
        "lip_enhance",
 "lip_tint",
 "lip_finish",
 "blush",
 "slimming",
        "hair_enhance",
 "contrast",
 "brightness",
 "highlights",
 "shadows",
        "whites",
 "blacks",
 "clarity",
 "vibrance",
 "saturation",
        "auto_exposure",
 "bloom",
 "bloom_threshold",
 "bloom_softness",
 "glow",
        "vignette",
 "sharpen",
 "sharpen_radius",
 "fade_toe",
 "highlight_drift",
        "airy_haze",
 "clarity_split_neg",
 "clarity_split_pos",
 "subject_separation",
 "impact",
        "color_grade",
 "grade_intensity",
 "chromatic_aberration",
 "grain",
 "halation",
        "lut",
 "tonal_curve_strength",
 "skin_protect_strength",
 "grain_strength",
 "highlight_rolloff_strength",
        "shadow_hue",
 "shadow_sat",
 "midtone_hue",
 "midtone_sat",
 "highlight_hue",
        "highlight_sat",
 "white_balance_kelvin",
 "white_balance_tint",
 "bw_channel_mixer_r",
 "bw_channel_mixer_g",
        "bw_channel_mixer_b",
 "negative_split_tone_shadow",
 "negative_split_tone_highlight",
 "hsl_hue_global",
 "hsl_sat_global",
        "hsl_lum_global",
 "face_exposure",
 "cosplay_wig_lace_blend",
 "cosplay_stockings_smooth",
 "cosplay_consistency_strength",
        "body_reshape_arm_length",
 "body_reshape_leg_length",
 "body_reshape_torso_width",
 "body_reshape_shoulder_width",
 "body_reshape_hip_width",
        "auto_body_reshape",
        # Recipe-driven face/body skin sliders that are VISIBLE gr.Slider
        # components (not gr.State placeholders): they must be synced on recipe
        # change or the stale slider value stomps the recipe value in GUI
        # renders (CLI was already correct via build_context).  Appended so
        # every existing RECIPE_OUTPUT_KEYS index stays stable.
        "shadow_lift",
 "nose_restore",
 "mole_protect",
 "texture_transplant",
 "body_relight",
        "body_dodge_burn",
 "body_shadow_lift",
        "mark_policy",
        # These controls are visible and recipe-driven. Keep them appended so
        # existing callback positions remain stable while a film/heal recipe
        # can actually activate the capability it declares.
        "heal_engine",
        "hb_even",
        "hb_shift",
        "film_enable",
        "film_highlight_purity",
    )

    # Name -> Gradio component map for the recipe-output tuple.  Mirrors the
    # `_process_input_components` guard: every RECIPE_OUTPUT_KEYS name must map
    # to exactly one component and vice-versa, enforced at import time so a
    # missing/extra entry fails loudly instead of shifting slider values.
    _recipe_output_components = {
        "smooth": smooth,
 "mid_reduction": mid_reduction,
 "texture_opacity": texture_opacity,
        "pore_synthesis": pore_synthesis,
 "nose_smooth": nose_smooth,
 "regional_modulation": regional_modulation,
        "smooth_engine": smooth_engine,
 "undereye_shadow_strength": undereye_shadow_strength,
        "freckle_removal": freckle_removal,
        "micro_restore": micro_restore,
 "whiten": whiten,
 "equalize": equalize,
        "blemish": blemish,
 "whiten_tone": whiten_tone,
 "nose_blush": nose_blush,
        "under_eye_blush": under_eye_blush,
 "white_costume_lift": white_costume_lift,
 "body_smooth": body_smooth,
        "body_equalize": body_equalize,
 "body_whiten": body_whiten,
 "body_match_face": body_match_face,
        "dodge_burn": dodge_burn,
 "relight": relight,
 "relight_azimuth": relight_azimuth,
        "relight_elevation": relight_elevation,
 "sculpt": sculpt,
 "shine_removal": shine_removal,
        "wrinkle_soften": wrinkle_soften,
 "specular_bloom": specular_bloom,
 "specular_bloom_tone": specular_bloom_tone,
        "skin_flatten": skin_flatten,
 "skin_quantize": skin_quantize,
 "skin_unify": skin_unify,
 "skin_unify_hue": skin_unify_hue,
 "skin_glow": skin_glow,
 "mask_feather_mode": mask_feather_mode,
 "skin_sss": skin_sss,
 "eye_enhance": eye_enhance,
        "catchlight": catchlight,
 "corneal_shading": corneal_shading,
 "dark_circles": dark_circles,
 "undereye_darken_removal": undereye_darken_removal,
        "undereye_puffiness_reduction": undereye_puffiness_reduction,
 "eye_sclera_brighten": eye_sclera_brighten,
 "eye_iris_saturate": eye_iris_saturate,
        "eye_iris_hue_shift": eye_iris_hue_shift,
 "eye_iris_brightness": eye_iris_brightness,
 "teeth_whiten": teeth_whiten,
        "lip_enhance": lip_enhance,
 "lip_tint": lip_tint,
 "lip_finish": lip_finish,
        "blush": blush,
 "slimming": slimming,
 "hair_enhance": hair_enhance,
        "contrast": contrast,
 "brightness": brightness,
 "highlights": highlights,
        "shadows": shadows,
 "whites": whites,
 "blacks": blacks,
        "clarity": clarity,
 "vibrance": vibrance,
 "saturation": saturation,
        "auto_exposure": auto_exposure,
 "bloom": bloom,
 "bloom_threshold": bloom_threshold,
        "bloom_softness": bloom_softness,
 "glow": glow,
 "vignette": vignette,
        "sharpen": sharpen,
 "sharpen_radius": sharpen_radius,
 "fade_toe": fade_toe,
        "highlight_drift": highlight_drift,
 "airy_haze": airy_haze,
 "clarity_split_neg": clarity_split_neg,
        "clarity_split_pos": clarity_split_pos,
 "subject_separation": subject_separation,
 "impact": impact,
        "color_grade": color_grade,
 "grade_intensity": grade_intensity,
 "chromatic_aberration": chromatic_aberration,
        "grain": grain,
 "halation": halation,
 "lut": lut,
        "tonal_curve_strength": tonal_curve_strength,
 "skin_protect_strength": skin_protect_strength,
 "grain_strength": grain_strength,
        "highlight_rolloff_strength": highlight_rolloff_strength,
 "shadow_hue": shadow_hue,
 "shadow_sat": shadow_sat,
        "midtone_hue": midtone_hue,
 "midtone_sat": midtone_sat,
 "highlight_hue": highlight_hue,
        "highlight_sat": highlight_sat,
 "white_balance_kelvin": white_balance_kelvin,
 "white_balance_tint": white_balance_tint,
        "bw_channel_mixer_r": bw_channel_mixer_r,
 "bw_channel_mixer_g": bw_channel_mixer_g,
 "bw_channel_mixer_b": bw_channel_mixer_b,
        "negative_split_tone_shadow": negative_split_tone_shadow,
 "negative_split_tone_highlight": negative_split_tone_highlight,
 "hsl_hue_global": hsl_hue_global,
        "hsl_sat_global": hsl_sat_global,
 "hsl_lum_global": hsl_lum_global,
 "face_exposure": face_exposure,
        "cosplay_wig_lace_blend": cosplay_wig_lace_blend,
 "cosplay_stockings_smooth": cosplay_stockings_smooth,
 "cosplay_consistency_strength": cosplay_consistency_strength,
        "body_reshape_arm_length": body_reshape_arm_length,
 "body_reshape_leg_length": body_reshape_leg_length,
 "body_reshape_torso_width": body_reshape_torso_width,
        "body_reshape_shoulder_width": body_reshape_shoulder_width,
 "body_reshape_hip_width": body_reshape_hip_width,
 "auto_body_reshape": auto_body_reshape,
        # Visible recipe-driven skin sliders (see RECIPE_OUTPUT_KEYS note above).
        "shadow_lift": shadow_lift,
 "nose_restore": nose_restore,
 "mole_protect": mole_protect,
 "texture_transplant": texture_transplant,
 "body_relight": body_relight,
        "body_dodge_burn": body_dodge_burn,
 "body_shadow_lift": body_shadow_lift,
        "mark_policy": mark_policy,
        "heal_engine": heal_engine,
        "hb_even": hb_even,
        "hb_shift": hb_shift,
        "film_enable": film_enable,
        "film_highlight_purity": film_highlight_purity,
    }
    _missing_outputs = set(RECIPE_OUTPUT_KEYS) - set(_recipe_output_components)
    _extra_outputs = set(_recipe_output_components) - set(RECIPE_OUTPUT_KEYS)
    if _missing_outputs or _extra_outputs:
        raise AssertionError(
            "_recipe_output_components drift vs RECIPE_OUTPUT_KEYS: "
            f"missing={sorted(_missing_outputs)} extra={sorted(_extra_outputs)}"
        )
    _recipe_outputs = [_recipe_output_components[k] for k in RECIPE_OUTPUT_KEYS]

    _mutation_events = []
    _reset_mutation_events = []

    def _track_reset_event(event):
        _mutation_events.append(event)
        _reset_mutation_events.append(event)
        return event

    # Smart mode switch: purely additive visibility toggle, no changes to
    # any Classic component's identity/callbacks (T2a).
    smart_mode_switch.change(
        fn=gui_smart.on_smart_mode_toggle,
        inputs=[smart_mode_switch],
        outputs=[smart_color_group],
        queue=False,
        show_progress="hidden",
    )

    _smart_color_outputs = [
        vibrance, saturation, white_balance_kelvin, white_balance_tint,
        contrast, highlights, shadows,
    ]
    _smart_color_inputs = [
        smart_amount, smart_warmth, smart_contrast_macro,
        vibrance, saturation, white_balance_kelvin, white_balance_tint,
        contrast, highlights, shadows,
    ]
    for _smart_slider in (smart_amount, smart_warmth, smart_contrast_macro):
        _mutation_events.append(_smart_slider.change(
            fn=gui_smart.apply_color_macros,
            inputs=_smart_color_inputs,
            outputs=_smart_color_outputs,
            queue=False,
            show_progress="hidden",
        ))

    _mutation_events.append(recipe.change(
        fn=on_recipe_change,
        inputs=[recipe],
        outputs=_recipe_outputs,
        queue=False,
        show_progress="hidden",
    ))

    _mutation_events.append(custom_style_preset.change(
        fn=apply_custom_style,
        inputs=[custom_style_preset, recipe],
        outputs=_recipe_outputs,
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_btn.click(
        fn=on_recipe_change,
        inputs=[recipe],
        outputs=_recipe_outputs,
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_skin_smooth_btn.click(
        fn=reset_skin_smoothing,
        inputs=[recipe],
        outputs=[smooth, nose_smooth, mid_reduction, texture_opacity, micro_restore, pore_synthesis, blemish, skin_flatten, skin_quantize],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_skin_tone_btn.click(
        fn=reset_skin_tone,
        inputs=[recipe],
        outputs=[whiten, whiten_tone, equalize, shadow_lift, nose_restore, skin_sss, skin_unify, skin_unify_hue, auto_exposure, white_costume_lift, face_exposure],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_basic_tone_btn.click(
        fn=reset_basic_tone,
        inputs=[recipe],
        outputs=[contrast, brightness, clarity, vibrance, saturation],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_tone_curve_btn.click(
        fn=reset_tone_curve,
        inputs=[recipe],
        outputs=[highlights, shadows, whites, blacks],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_relighting_btn.click(
        fn=reset_relighting,
        inputs=[recipe],
        outputs=[relight, relight_azimuth, relight_elevation],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_eyes_lips_btn.click(
        fn=reset_eyes_lips,
        inputs=[recipe],
        outputs=[eye_enhance, catchlight, dark_circles, undereye_darken_removal, undereye_puffiness_reduction, eye_sclera_brighten, eye_iris_saturate, eye_iris_hue_shift, eye_iris_brightness, teeth_whiten, lip_enhance, lip_tint, lip_finish, blush, nose_blush, under_eye_blush, eye_gate],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_face_reshaping_btn.click(
        fn=reset_face_reshaping,
        inputs=[recipe],
        outputs=[slimming],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_structure_effects_btn.click(
        fn=reset_structure_effects,
        inputs=[recipe],
        outputs=[hair_enhance, dodge_burn, impact, specular_bloom, specular_bloom_tone, bloom, bloom_threshold, bloom_softness, sharpen, sharpen_radius, glow, skin_glow, mask_feather_mode, vignette, subject_separation],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_color_grading_btn.click(
        fn=reset_color_grading,
        inputs=[recipe],
        outputs=[color_grade, grade_intensity],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_film_effects_btn.click(
        fn=reset_film_effects,
        inputs=[recipe],
        outputs=[chromatic_aberration, grain, halation, lut, tonal_curve_strength, skin_protect_strength, grain_strength, highlight_rolloff_strength, film_enable, film_highlight_purity],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_split_toning_btn.click(
        fn=reset_split_toning,
        inputs=[recipe],
        outputs=[shadow_hue, shadow_sat, midtone_hue, midtone_sat, highlight_hue, highlight_sat],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_color_transfer_btn.click(
        fn=reset_color_transfer,
        inputs=[],
        outputs=[color_ref_img, color_ref_strength],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_debug_btn.click(
        fn=reset_debug,
        inputs=[recipe],
        outputs=[debug_mode],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_body_skin_btn.click(
        fn=reset_body_skin,
        inputs=[recipe],
        outputs=[body_smooth, body_equalize, body_whiten, body_match_face, body_relight, body_dodge_burn, body_shadow_lift],
        queue=False,
        show_progress="hidden",
    ))

    _track_reset_event(reset_lch_btn.click(
        fn=reset_lch,
        inputs=[recipe],
        outputs=[white_balance_kelvin, white_balance_tint, bw_channel_mixer_r, bw_channel_mixer_g, bw_channel_mixer_b, negative_split_tone_shadow, negative_split_tone_highlight, hsl_hue_global, hsl_sat_global, hsl_lum_global],
        queue=False,
        show_progress="hidden",
    ))

    # F6: Look Extractor wiring
    _look_extract_event = look_extract_btn.click(
        fn=on_extract_look,
        inputs=[look_ref_file, img_input],
        outputs=[_look_params_state, look_status],
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
    )
    _mutation_events.append(_look_extract_event)

    # Per-face recipe picker
    _detect_faces_event = detect_faces_btn.click(
        fn=on_detect_faces,
        inputs=[img_input],
        outputs=[face_gallery, _face_params_state, face_select, face_params_status],
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
    )
    _mutation_events.append(_detect_faces_event)
    _apply_face_event = apply_face_btn.click(
        fn=on_apply_face_recipe,
        inputs=[face_select, face_recipe, _face_params_state],
        outputs=[_face_params_state, face_params_status],
        show_progress="hidden",
        queue=False,
    )
    _mutation_events.append(_apply_face_event)
    _clear_faces_event = clear_faces_btn.click(
        fn=on_clear_face_params,
        inputs=[],
        outputs=[_face_params_state, face_gallery, face_select, face_params_status],
        show_progress="hidden",
        queue=False,
    )
    _mutation_events.append(_clear_faces_event)
    _source_faces_event = img_input.change(
        fn=on_img_change_clear_faces,
        inputs=[],
        outputs=[_face_params_state, face_gallery, face_select, face_params_status],
        show_progress="hidden",
        queue=False,
    )
    _mutation_events.append(_source_faces_event)

    # Advanced Retouch workspace.  The editor is kept separate from the main
    # recipe canvas so manual edits remain reversible and can be serialized as
    # compact mask/action records in sessions and snapshots.
    img_input.change(
        fn=on_advanced_source_change,
        inputs=[img_input, _advanced_pending_session_state],
        outputs=[
            _advanced_source_state, _advanced_current_state, advanced_editor,
            advanced_mask_overlay, advanced_before_after, _advanced_history_state,
            _advanced_edit_log_state, _advanced_mask_state, advanced_status,
            _advanced_pending_session_state, _advanced_base_contract_state,
        ],
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
    )
    advanced_edit_processed_btn.click(
        fn=on_advanced_processed_result,
        inputs=[
            _processed_result_state, _advanced_pending_session_state,
            _preview_cache_state, _advanced_edit_log_state,
        ],
        outputs=[
            _advanced_source_state, _advanced_current_state, advanced_editor,
            advanced_mask_overlay, advanced_before_after, _advanced_history_state,
            _advanced_edit_log_state, _advanced_mask_state, advanced_status,
            _advanced_pending_session_state, _advanced_base_contract_state,
        ],
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
    )
    advanced_bind_legacy_btn.click(
        fn=advanced_bind_legacy_handler,
        inputs=[
            _advanced_source_state, _advanced_pending_session_state,
            _advanced_base_contract_state,
        ],
        outputs=[
            _advanced_current_state, advanced_editor, advanced_before_after,
            _advanced_history_state, _advanced_edit_log_state,
            _advanced_pending_session_state, advanced_status,
        ],
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
        show_progress="minimal",
    )
    advanced_face_detect_btn.click(
        fn=on_advanced_face_choices,
        inputs=[img_input],
        outputs=[advanced_face_select, advanced_status],
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
    )
    advanced_apply_btn.click(
        fn=advanced_apply_handler,
        inputs=[
            advanced_editor, _advanced_current_state, _advanced_source_state,
            _advanced_history_state, _advanced_edit_log_state,
            advanced_mode, advanced_operation, advanced_strength,
            advanced_semantic, advanced_face_select, advanced_heal_method,
            advanced_remove_engine, advanced_overlay_visible, advanced_overlay_opacity,
            advanced_eye_size, advanced_eye_distance,
            advanced_nose_width, advanced_nose_length, advanced_jaw_width,
            advanced_chin_length, advanced_mouth_size, advanced_smile,
            advanced_forehead, advanced_eye_size_l, advanced_eye_size_r,
            advanced_nose_width_l, advanced_nose_width_r,
            advanced_jaw_width_l, advanced_jaw_width_r,
            advanced_mask_action, advanced_mask_feather,
        ],
        outputs=[
            advanced_editor, _advanced_current_state, _advanced_history_state,
            _advanced_edit_log_state, _advanced_mask_state, advanced_mask_overlay,
            advanced_before_after, advanced_status,
        ],
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
        show_progress="minimal",
    )
    advanced_undo_btn.click(
        fn=advanced_undo_handler,
        inputs=[_advanced_current_state, _advanced_source_state, _advanced_history_state, _advanced_edit_log_state],
        outputs=[advanced_editor, _advanced_current_state, _advanced_history_state, _advanced_edit_log_state, _advanced_mask_state, advanced_mask_overlay, advanced_before_after, advanced_status],
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
        show_progress="minimal",
    )
    advanced_redo_btn.click(
        fn=advanced_redo_handler,
        inputs=[_advanced_current_state, _advanced_source_state, _advanced_history_state, _advanced_edit_log_state],
        outputs=[advanced_editor, _advanced_current_state, _advanced_history_state, _advanced_edit_log_state, _advanced_mask_state, advanced_mask_overlay, advanced_before_after, advanced_status],
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
        show_progress="minimal",
    )
    advanced_reset_btn.click(
        fn=advanced_reset_handler,
        inputs=[_advanced_source_state],
        outputs=[_advanced_current_state, advanced_editor, advanced_mask_overlay, advanced_before_after, _advanced_history_state, _advanced_edit_log_state, _advanced_mask_state, advanced_status],
        queue=False,
        show_progress="hidden",
    )
    advanced_overlay_visible.change(
        fn=advanced_overlay_handler,
        inputs=[_advanced_current_state, _advanced_mask_state, advanced_overlay_visible, advanced_overlay_opacity],
        outputs=[advanced_mask_overlay],
    )
    advanced_overlay_opacity.change(
        fn=advanced_overlay_handler,
        inputs=[_advanced_current_state, _advanced_mask_state, advanced_overlay_visible, advanced_overlay_opacity],
        outputs=[advanced_mask_overlay],
    )
    advanced_clear_mask_btn.click(
        fn=advanced_clear_mask_handler,
        inputs=[advanced_editor, _advanced_current_state, _advanced_source_state],
        outputs=[advanced_editor, _advanced_mask_state, advanced_mask_overlay, advanced_before_after, advanced_status],
    )
    advanced_save_snapshot_btn.click(
        fn=advanced_save_snapshot_handler,
        inputs=[
            advanced_snapshot_name, _advanced_current_state,
            _advanced_edit_log_state, _advanced_snapshots_state,
            _advanced_base_contract_state,
        ],
        outputs=[advanced_snapshot_dropdown, _advanced_snapshots_state, advanced_status],
    )
    advanced_compare_snapshot_btn.click(
        fn=advanced_compare_snapshot_handler,
        inputs=[advanced_snapshot_dropdown, _advanced_current_state, _advanced_snapshots_state],
        outputs=[advanced_before_after, advanced_status],
    )
    advanced_export_btn.click(
        fn=advanced_export_handler,
        inputs=[
            _advanced_current_state, advanced_export_fmt, img_input,
            _advanced_base_contract_state,
        ],
        outputs=[advanced_export_file, advanced_status],
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
        show_progress="minimal",
    )

    # T4: Recipe Cookbook wiring
    cookbook_search_btn.click(
        fn=on_search_recipes,
        inputs=[cookbook_search, cookbook_category],
        outputs=[cookbook_dropdown, cookbook_status],
    )
    cookbook_category.change(
        fn=on_browse_category,
        inputs=[cookbook_category],
        outputs=[cookbook_dropdown, cookbook_status],
    )
    cookbook_dropdown.change(
        fn=on_select_cookbook,
        inputs=[cookbook_dropdown],
        outputs=[recipe, cookbook_status],
    )

    # Batch tab Recipe Cookbook wiring (same T4 handlers, targets batch_recipe)
    batch_cookbook_search_btn.click(
        fn=on_search_recipes,
        inputs=[batch_cookbook_search, batch_cookbook_category],
        outputs=[batch_cookbook_dropdown, batch_cookbook_status],
    )
    batch_cookbook_category.change(
        fn=on_browse_category,
        inputs=[batch_cookbook_category],
        outputs=[batch_cookbook_dropdown, batch_cookbook_status],
    )
    batch_cookbook_dropdown.change(
        fn=on_select_cookbook,
        inputs=[batch_cookbook_dropdown],
        outputs=[batch_recipe, batch_cookbook_status],
    )

    # LUT hot-reload wiring
    reload_luts_btn.click(
        fn=on_reload_luts,
        inputs=[],
        outputs=[reload_luts_status],
    )

    # Diagnostics bundle wiring
    diagnostics_btn.click(
        fn=lambda: diagnostics_report(),
        inputs=[],
        outputs=[diagnostics_out],
    )
    runtime_doctor_btn.click(
        fn=runtime_doctor_text,
        inputs=[],
        outputs=[runtime_doctor_out],
    )
    clear_diagnostics_btn.click(
        fn=clear_diagnostics,
        inputs=[],
        outputs=[diagnostics_status],
    )


    save_style_btn.click(
        fn=on_save_style,
        inputs=[save_name, save_author, save_tags,
                smooth, mid_reduction, texture_opacity,
                whiten, contrast, brightness],
        outputs=[custom_style_preset, batch_custom_style, save_status]
    )

    learn_style_btn.click(
        fn=on_learn_style,
        inputs=[learn_orig_dir, learn_edit_dir, learn_name, learn_author, learn_tags],
        outputs=[custom_style_preset, batch_custom_style, learn_status]
    )

    batch_style_type.change(
        fn=on_batch_style_change,
        inputs=[batch_style_type],
        outputs=[batch_recipe, batch_custom_style],
    )

    batch_btn.click(
        fn=on_process_folder,
        inputs=[folder_in, folder_out, batch_style_type, batch_custom_style, batch_recipe,
                batch_fmt, batch_quality, batch_res, auto_group_toggle, sheet_toggle, zip_toggle],
        outputs=[batch_sheet_out, batch_zip_out, batch_status],
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
        show_progress="minimal",
    )

    job_refresh_btn.click(fn=on_refresh_jobs, inputs=[], outputs=[job_list_df])

    job_list_df.select(
        fn=on_select_job,
        inputs=[job_list_df],
        outputs=[job_detail_group, job_detail_df, job_detail_log, job_selected_id],
    )

    job_rerun_btn.click(
        fn=on_rerun_flagged,
        inputs=[job_selected_id],
        outputs=[job_rerun_status],
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
        show_progress="minimal",
    )

    # Shoot Intelligence: explainable scan, safe watch pass, profiles, and
    # multi-reference Look Board persistence. These handlers are intentionally
    # separate from the image renderer and never mutate source captures.
    shoot_scan_btn.click(
        fn=on_shoot_intelligence_scan,
        inputs=[shoot_folder, shoot_recursive, shoot_manifest_path, shoot_face_quality],
        outputs=[shoot_rows, shoot_status, shoot_graph_json],
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
        show_progress="minimal",
    )
    watch_btn.click(
        fn=on_watch_folder_process,
        inputs=[watch_folder, watch_state, watch_limit, watch_output_dir, watch_job_kind,
                watch_recipe, shoot_manifest_path],
        outputs=[watch_status],
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
        show_progress="minimal",
    )
    review_save_btn.click(
        fn=on_save_shoot_review,
        inputs=[shoot_manifest_path, review_asset_id, review_decision, review_rating,
                review_labels, review_reviewer, review_note],
        outputs=[review_status],
    )
    review_json_btn.click(
        fn=on_export_shoot_review_json,
        inputs=[shoot_manifest_path],
        outputs=[review_export_file, review_status],
    )
    review_csv_btn.click(
        fn=on_export_shoot_review_csv,
        inputs=[shoot_manifest_path, review_export_selected],
        outputs=[review_export_file, review_status],
    )
    profile_save_btn.click(
        fn=on_save_project_profile,
        inputs=[profile_id, profile_subject_key, profile_display_name, profile_recipe,
                profile_style, profile_params, profile_marks],
        outputs=[profile_status],
    )
    look_board_save_btn.click(
        fn=on_save_look_board,
        inputs=[look_board_id, look_board_project, look_board_refs],
        outputs=[look_board_status, look_board_json],
    )
    _look_board_apply_event = look_board_apply_btn.click(
        fn=on_apply_look_board,
        inputs=[look_board_id, img_input],
        outputs=[_look_params_state, look_board_status],
    )
    _mutation_events.append(_look_board_apply_event)

    # Name → Gradio component map.  Keyed by PROCESS_INPUT_KEYS so that adding
    # a ParamSpec (which auto-inserts a name into PROCESS_INPUT_KEYS via
    # param_names()) forces a matching component entry here.  The positional
    # `_process_inputs` list is ALWAYS DERIVED from this map, never hand-ordered,
    # so a forgotten/misplaced insertion can no longer silently shift every
    # later argument by one (the historical "brightness reads as contrast"
    # footgun).  The import-time drift guard below turns any mismatch into a
    # loud AssertionError instead of corrupted output.
    _process_input_components = {
        "img_paths": img_input,
        "recipe": recipe,
        "smooth": smooth,
        "mid_reduction": mid_reduction,
        "blotch_reduction": _blotch_reduction_state,
        "texture_opacity": texture_opacity,
        "pore_synthesis": pore_synthesis,
        "nose_smooth": nose_smooth,
        "regional_modulation": regional_modulation,
        "smooth_engine": smooth_engine,
        "undereye_shadow_strength": undereye_shadow_strength,
        "freckle_removal": freckle_removal,
        "heal_engine": heal_engine,
        "mark_policy": mark_policy,
        "fa02_texture_mode": _fa02_texture_mode_state,
        "micro_restore": micro_restore,
        "micro_dodge_burn": _micro_dodge_burn_state,
        "redness_even": _redness_even_state,
        "hb_even": hb_even,
        "hb_shift": hb_shift,
        "whiten_hue_stable": _whiten_hue_stable_state,
        "whiten": whiten,
        "equalize": equalize,
        "blemish": blemish,
        "whiten_tone": whiten_tone,
        "nose_blush": nose_blush,
        "under_eye_blush": under_eye_blush,
        "white_costume_lift": white_costume_lift,
        "dodge_burn": dodge_burn,
        "relight": relight,
        "relight_azimuth": relight_azimuth,
        "relight_elevation": relight_elevation,
        "sculpt": sculpt,
        "shine_removal": shine_removal,
        "wrinkle_soften": wrinkle_soften,
        "wrinkle_soften_forehead": _wrinkle_soften_forehead_state,
        "wrinkle_soften_nasolabial": _wrinkle_soften_nasolabial_state,
        "wrinkle_soften_neck": _wrinkle_soften_neck_state,
        "texture_transplant": texture_transplant,
        "body_smooth": body_smooth,
        "body_equalize": body_equalize,
        "body_whiten": body_whiten,
        "body_match_face": body_match_face,
        "body_relight": body_relight,
        "body_dodge_burn": body_dodge_burn,
        "shadow_lift": shadow_lift,
        "face_exposure": face_exposure,
        "body_shadow_lift": body_shadow_lift,
        "nose_restore": nose_restore,
        "skin_sss": skin_sss,
        "specular_bloom": specular_bloom,
        "specular_bloom_tone": specular_bloom_tone,
        "specular_finish": _specular_finish_state,
        "specular_finish_strength": _specular_finish_strength_state,
        "specular_recolor": _specular_recolor_state,
        "albedo_even": _albedo_even_state,
        "makeup_coverage_even": _makeup_coverage_even_state,
        "makeup_cake_reduce": _makeup_cake_reduce_state,
        "hemoglobin_smooth": _hemoglobin_smooth_state,
        "mole_protect": mole_protect,
        "vein_attenuate": _vein_attenuate_state,
        "skin_flatten": skin_flatten,
        "skin_quantize": skin_quantize,
        "skin_unify": skin_unify,
        "skin_unify_hue": skin_unify_hue,
        "skin_hue_unify": _skin_hue_unify_state,
        "skin_chroma_even": _skin_chroma_even_state,
        "skin_glow": skin_glow,
        "mask_feather_mode": mask_feather_mode,
        "eye_enhance": eye_enhance,
        "catchlight": catchlight,
        "corneal_shading": corneal_shading,
        "dark_circles": dark_circles,
        "undereye_darken_removal": undereye_darken_removal,
        "undereye_puffiness_reduction": undereye_puffiness_reduction,
        "eye_sclera_brighten": eye_sclera_brighten,
        "eye_sclera_vessel_remove": _eye_sclera_vessel_remove_state,
        "eye_gate": eye_gate,
        "backdrop_cleanup": _backdrop_cleanup_state,
        "fabric_wrinkle_smooth": _fabric_wrinkle_smooth_state,
        "eye_iris_saturate": eye_iris_saturate,
        "eye_iris_hue_shift": eye_iris_hue_shift,
        "eye_iris_brightness": eye_iris_brightness,
        "teeth_whiten": teeth_whiten,
        "lip_enhance": lip_enhance,
        "lip_tint": lip_tint,
        "lip_finish": lip_finish,
        "blush": blush,
        "slimming": slimming,
        "reshape_eye_size": _reshape_eye_size_state,
        "reshape_eye_distance": _reshape_eye_distance_state,
        "reshape_nose_width": _reshape_nose_width_state,
        "reshape_nose_length": _reshape_nose_length_state,
        "reshape_jaw_width": _reshape_jaw_width_state,
        "reshape_chin_length": _reshape_chin_length_state,
        "reshape_mouth_size": _reshape_mouth_size_state,
        "reshape_smile": _reshape_smile_state,
        "reshape_forehead": _reshape_forehead_state,
        "reshape_jaw_width_l": _reshape_jaw_width_l_state,
        "reshape_jaw_width_r": _reshape_jaw_width_r_state,
        "reshape_nose_width_l": _reshape_nose_width_l_state,
        "reshape_nose_width_r": _reshape_nose_width_r_state,
        "reshape_eye_size_l": _reshape_eye_size_l_state,
        "reshape_eye_size_r": _reshape_eye_size_r_state,
        "reshape_neck_width": _reshape_neck_width_state,
        "reshape_neck_length": _reshape_neck_length_state,
        "hair_enhance": hair_enhance,
        "hair_deglare": _hair_deglare_state,
        "hair_ring_position": _hair_ring_position_state,
        "hair_ring_tint": _hair_ring_tint_state,
        "hair_remove_flyaways": _hair_remove_flyaways_state,
        "contrast": contrast,
        "brightness": brightness,
        "highlights": highlights,
        "shadows": shadows,
        "whites": whites,
        "blacks": blacks,
        "clarity": clarity,
        "vibrance": vibrance,
        "saturation": saturation,
        "auto_exposure": auto_exposure,
        "bloom": bloom,
        "bloom_threshold": bloom_threshold,
        "bloom_softness": bloom_softness,
        "glow": glow,
        "vignette": vignette,
        "sharpen": sharpen,
        "sharpen_radius": sharpen_radius,
        "subject_separation": subject_separation,
        "impact": impact,
        "fade_toe": fade_toe,
        "highlight_drift": highlight_drift,
        "airy_haze": airy_haze,
        "clarity_split_neg": clarity_split_neg,
        "clarity_split_pos": clarity_split_pos,
        "color_grade": color_grade,
        "grade_intensity": grade_intensity,
        "chromatic_aberration": chromatic_aberration,
        "grain": grain,
        "halation": halation,
        "lut": lut,
        "tonal_curve_strength": tonal_curve_strength,
        "skin_protect_strength": skin_protect_strength,
        "grain_strength": grain_strength,
        "highlight_rolloff_strength": highlight_rolloff_strength,
        "gamut_compress": _gamut_compress_state,
        "saturation_mode": _saturation_mode_state,
        "hsl_hue_red": _hsl_hue_red_state,
        "hsl_sat_red": _hsl_sat_red_state,
        "hsl_lum_red": _hsl_lum_red_state,
        "hsl_hue_orange": _hsl_hue_orange_state,
        "hsl_sat_orange": _hsl_sat_orange_state,
        "hsl_lum_orange": _hsl_lum_orange_state,
        "hsl_hue_yellow": _hsl_hue_yellow_state,
        "hsl_sat_yellow": _hsl_sat_yellow_state,
        "hsl_lum_yellow": _hsl_lum_yellow_state,
        "hsl_hue_green": _hsl_hue_green_state,
        "hsl_sat_green": _hsl_sat_green_state,
        "hsl_lum_green": _hsl_lum_green_state,
        "hsl_hue_cyan": _hsl_hue_cyan_state,
        "hsl_sat_cyan": _hsl_sat_cyan_state,
        "hsl_lum_cyan": _hsl_lum_cyan_state,
        "hsl_hue_blue": _hsl_hue_blue_state,
        "hsl_sat_blue": _hsl_sat_blue_state,
        "hsl_lum_blue": _hsl_lum_blue_state,
        "hsl_hue_purple": _hsl_hue_purple_state,
        "hsl_sat_purple": _hsl_sat_purple_state,
        "hsl_lum_purple": _hsl_lum_purple_state,
        "hsl_hue_magenta": _hsl_hue_magenta_state,
        "hsl_sat_magenta": _hsl_sat_magenta_state,
        "hsl_lum_magenta": _hsl_lum_magenta_state,
        "calibration_red_hue": _calibration_red_hue_state,
        "calibration_red_sat": _calibration_red_sat_state,
        "calibration_red_lum": _calibration_red_lum_state,
        "calibration_green_hue": _calibration_green_hue_state,
        "calibration_green_sat": _calibration_green_sat_state,
        "calibration_green_lum": _calibration_green_lum_state,
        "calibration_blue_hue": _calibration_blue_hue_state,
        "calibration_blue_sat": _calibration_blue_sat_state,
        "calibration_blue_lum": _calibration_blue_lum_state,
        "film_enable": film_enable,
        "film_strength": _film_strength_state,
        "film_toe_r": _film_toe_r_state,
        "film_toe_g": _film_toe_g_state,
        "film_toe_b": _film_toe_b_state,
        "film_shoulder_r": _film_shoulder_r_state,
        "film_shoulder_g": _film_shoulder_g_state,
        "film_shoulder_b": _film_shoulder_b_state,
        "film_midpoint": _film_midpoint_state,
        "film_gamma": _film_gamma_state,
        "film_crosstalk_cy_mg": _film_crosstalk_cy_mg_state,
        "film_crosstalk_cy_ye": _film_crosstalk_cy_ye_state,
        "film_crosstalk_mg_ye": _film_crosstalk_mg_ye_state,
        "film_tonemap_strength": _film_tonemap_strength_state,
        "film_tonemap_toe": _film_tonemap_toe_state,
        "film_tonemap_shoulder": _film_tonemap_shoulder_state,
        "film_skew": _film_skew_state,
        "film_highlight_purity": film_highlight_purity,
        "background_harmonize": _background_harmonize_state,
        "background_harmonize_mode": _background_harmonize_mode_state,
        "background_blur": _background_blur_state,
        "lens_blur": _lens_blur_state,
        "background_desaturation": _background_desaturation_state,
        "light_wrap": _light_wrap_state,
        "blue_shadow_grade": _blue_shadow_grade_state,
        "cyan_midtone_grade": _cyan_midtone_grade_state,
        "subject_sharpen": _subject_sharpen_state,
        "matte_black": _matte_black_state,
        "shadow_hue": shadow_hue,
        "shadow_sat": shadow_sat,
        "midtone_hue": midtone_hue,
        "midtone_sat": midtone_sat,
        "highlight_hue": highlight_hue,
        "highlight_sat": highlight_sat,
        "white_balance_kelvin": white_balance_kelvin,
        "white_balance_tint": white_balance_tint,
        "bw_channel_mixer_r": bw_channel_mixer_r,
        "bw_channel_mixer_g": bw_channel_mixer_g,
        "bw_channel_mixer_b": bw_channel_mixer_b,
        "negative_split_tone_shadow": negative_split_tone_shadow,
        "negative_split_tone_highlight": negative_split_tone_highlight,
        "hsl_hue_global": hsl_hue_global,
        "hsl_sat_global": hsl_sat_global,
        "hsl_lum_global": hsl_lum_global,
        "ai_denoise": _ai_denoise_state,
        "ai_sr_scale": _ai_sr_scale_state,
        "mv2_eyeshadow": _mv2_eyeshadow_state,
        "mv2_eyeshadow_color": _mv2_eyeshadow_color_state,
        "mv2_eyeshadow_style": _mv2_eyeshadow_style_state,
        "mv2_eyeliner": _mv2_eyeliner_state,
        "mv2_eyeliner_color": _mv2_eyeliner_color_state,
        "mv2_eyeliner_style": _mv2_eyeliner_style_state,
        "mv2_contour": _mv2_contour_state,
        "mv2_brows": _mv2_brows_state,
        "mv2_brows_color": _mv2_brows_color_state,
        "mv2_ombre": _mv2_ombre_state,
        "mv2_ombre_color1": _mv2_ombre_color1_state,
        "mv2_ombre_color2": _mv2_ombre_color2_state,
        "neural_stray_hair_boost": _neural_stray_hair_boost_state,
        "neural_defect_boost": _neural_defect_boost_state,
        "cosplay_wig_lace_blend": cosplay_wig_lace_blend,
        "cosplay_stockings_smooth": cosplay_stockings_smooth,
        "cosplay_consistency_strength": cosplay_consistency_strength,
        "body_reshape_arm_length": body_reshape_arm_length,
        "body_reshape_leg_length": body_reshape_leg_length,
        "body_reshape_torso_width": body_reshape_torso_width,
        "body_reshape_shoulder_width": body_reshape_shoulder_width,
        "body_reshape_hip_width": body_reshape_hip_width,
        "auto_body_reshape": auto_body_reshape,
        "purple_fringing": _purple_fringing_state,
        "flyaway_cleanup": _flyaway_cleanup_state,
        "micro_grain": _micro_grain_state,
        "split_toning": _split_toning_state,
        "color_ref_img": color_ref_img,
        "color_ref_strength": color_ref_strength,
        "show_compare": show_compare,
        "fast": fast,
        "export_fmt": export_fmt,
        "export_quality": export_quality,
        "export_res": export_res,
        "quality_tier": quality_tier,
        "debug_mode": debug_mode,
        "optical_correction": optical_correction,
        "look_params": _look_params_state,
        "face_params": _face_params_state,
        "face_params_json": face_params_json,
    }
    # Drift guard: every PROCESS_INPUT_KEYS name must map to exactly one
    # component, and every component key must be a PROCESS_INPUT_KEYS name.
    # A new ParamSpec that forgets to add its component here will fail import
    # loudly instead of silently corrupting positional argument binding.
    _missing_components = set(PROCESS_INPUT_KEYS) - set(_process_input_components)
    _extra_components = set(_process_input_components) - set(PROCESS_INPUT_KEYS)
    if _missing_components or _extra_components:
        raise AssertionError(
            "_process_input_components drift vs PROCESS_INPUT_KEYS: "
            f"missing={sorted(_missing_components)} extra={sorted(_extra_components)}"
        )
    _process_inputs = [_process_input_components[k] for k in PROCESS_INPUT_KEYS]
    _process_outputs = [img_output, compare_viewer, _processed_result_state, export_file, status, debug_gallery, debug_panel, qa_status]

    # Detection/source caches are invalidated before a subsequent render when
    # any input that changes decoded pixels or the processing resolution moves.
    # The callback is unqueued and only replaces compact session state.
    for _cache_invalidation_component in (
        img_input,
        optical_correction,
        export_res,
        quality_tier,
        fast,
    ):
        _cache_invalidation_component.change(
            fn=invalidate_preview_cache_handler,
            inputs=[_preview_cache_state],
            outputs=[_preview_cache_state],
            show_progress="hidden",
            queue=False,
        )

    # Draft changes update only session State and the lightweight status badge.
    # One consolidated dependency replaces hundreds of per-control listeners.
    # Slider input is intentionally never used to write controls or history.
    _settings_revision_triggers = []
    _history_triggers = []
    _history_outputs = [_undo_stack_state, undo_btn, redo_btn]
    for _settings_component in _process_inputs[1:]:
        # Recipe changes are revisioned after their defaults have been
        # applied by the mutation event below; registering both paths would
        # advance the draft twice for one user action.
        if _settings_component is recipe:
            continue
        _input_event = getattr(_settings_component, "input", None)
        if callable(_input_event):
            _settings_revision_triggers.append(_input_event)
        if isinstance(_settings_component, gr.Slider):
            _release_event = getattr(_settings_component, "release", None)
            if callable(_release_event):
                _history_triggers.append(_release_event)
        elif _settings_component is not recipe and _settings_component is not custom_style_preset:
            # Dropdowns/checkboxes have no release event; their user input is
            # still a lightweight, non-queued history boundary.
            if callable(_input_event):
                _history_triggers.append(_input_event)

    gr.on(
        triggers=_settings_revision_triggers,
        fn=on_settings_changed,
        inputs=[_settings_revision_state],
        outputs=[_settings_revision_state, status],
        show_progress="hidden",
        queue=False,
    )
    if _history_triggers:
        gr.on(
            triggers=_history_triggers,
            fn=record_history,
            inputs=_process_inputs + [_undo_stack_state],
            outputs=_history_outputs,
            show_progress="hidden",
            queue=False,
        )

    # Seed each browser session with a baseline.  The stack and button state
    # remain session-owned and deepcopy-compatible through gr.State.
    app.load(
        fn=initialize_history,
        inputs=_process_inputs,
        outputs=_history_outputs,
        show_progress="hidden",
        queue=False,
    )

    # Mutation events were registered before _process_inputs was assembled.
    # Attach history and one revision increment only after their outputs have
    # committed, so recipe/style/reset mutations cannot race the draft state.
    for _mutation_event in _mutation_events:
        _history_event = _mutation_event.success(
            fn=record_history,
            inputs=_process_inputs + [_undo_stack_state],
            outputs=_history_outputs,
            show_progress="hidden",
            queue=False,
        )
        _history_event.success(
            fn=on_settings_changed,
            inputs=[_settings_revision_state],
            outputs=[_settings_revision_state, status],
            show_progress="hidden",
            queue=False,
        )

    def _bind_process_event(_button, _mode_state):
        _capture_event = _button.click(
            fn=capture_render_snapshot,
            inputs=_process_inputs + [
                _settings_revision_state,
                _mode_state,
                preserve_source_profile,
            ],
            outputs=[_render_snapshot_state, status],
            show_progress="hidden",
            queue=False,
        )
        _render_event = _capture_event.then(
            fn=process_image_event,
            inputs=[_render_snapshot_state, _settings_revision_state, _preview_cache_state],
            outputs=_process_outputs + [_preview_cache_state],
            show_progress="minimal",
            concurrency_limit=1,
            concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
        )
        _completion_event = _render_event.then(
            fn=render_completion_status,
            inputs=[status, _render_snapshot_state, _settings_revision_state, export_file],
            outputs=[status, export_file],
            show_progress="hidden",
            queue=False,
        )
        _completion_event.then(
            fn=build_render_manifest_handler,
            inputs=[
                _render_snapshot_state,
                _settings_revision_state,
                _preview_cache_state,
                export_file,
            ],
            outputs=[render_manifest_out],
            show_progress="hidden",
            queue=False,
        )

    _bind_process_event(process_btn, _render_preview_mode_state)
    _bind_process_event(process_btn_bottom, _render_preview_mode_state)
    _bind_process_event(export_full_btn, _export_full_mode_state)
    export_all_btn.click(
        fn=export_all_handler,
        inputs=_process_inputs + [_settings_revision_state, preserve_source_profile],
        outputs=[status],
        show_progress="hidden",
        queue=False,
    )

    # F10: Smart Process stores a proposal in State. It never writes recipe
    # controls until Apply Smart Suggestion is explicitly clicked.
    _smart_capture_event = smart_process_btn.click(
        fn=capture_smart_snapshot,
        inputs=[img_input, recipe, _settings_revision_state],
        outputs=[_smart_analysis_snapshot_state, status],
        show_progress="hidden",
        queue=False,
    )
    _smart_analysis_event = _smart_capture_event.then(
        fn=on_smart_process_event,
        inputs=[_smart_analysis_snapshot_state],
        outputs=[_smart_proposal_state, status, smart_analysis_html, apply_smart_btn],
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
    )
    _smart_analysis_event.then(
        fn=smart_completion_status,
        inputs=[status, _smart_analysis_snapshot_state, _settings_revision_state],
        outputs=[status],
        show_progress="hidden",
        queue=False,
    )

    _smart_apply_event = apply_smart_btn.click(
        fn=apply_smart_suggestion,
        inputs=[_smart_proposal_state, _settings_revision_state],
        outputs=[recipe] + list(_recipe_outputs) + [
            _smart_proposal_state, status, smart_analysis_html, apply_smart_btn,
            _settings_revision_state,
        ],
        show_progress="hidden",
        queue=False,
    )
    _smart_apply_event.success(
        fn=record_history,
        inputs=_process_inputs + [_undo_stack_state],
        outputs=_history_outputs,
        show_progress="hidden",
        queue=False,
    )

    # F2: Session wiring
    save_session_btn.click(
        fn=save_session_handler,
        inputs=_process_inputs + [
            _advanced_edit_log_state, _advanced_base_contract_state,
            _advanced_history_state, _advanced_current_state,
        ],
        outputs=[session_download],
    )

    _load_session_event = load_session_file.change(
        fn=load_session_handler,
        inputs=[load_session_file] + _process_inputs,
        outputs=list(_process_inputs),
    )
    _load_history_event = _load_session_event.success(
        fn=record_history,
        inputs=_process_inputs + [_undo_stack_state],
        outputs=_history_outputs,
        show_progress="hidden",
        queue=False,
    )
    _load_history_event.success(
        fn=on_settings_changed,
        inputs=[_settings_revision_state],
        outputs=[_settings_revision_state, status],
        show_progress="hidden",
        queue=False,
    )

    load_session_file.change(
        fn=load_advanced_session_handler,
        inputs=[load_session_file, _advanced_source_state, _advanced_base_contract_state],
        outputs=[
            _advanced_source_state, _advanced_pending_session_state, _advanced_current_state,
            advanced_editor, advanced_mask_overlay, advanced_before_after,
            _advanced_history_state, _advanced_edit_log_state, _advanced_mask_state,
            advanced_status,
        ],
        show_progress="minimal",
        concurrency_limit=1,
        concurrency_id=GUI_ENGINE_CONCURRENCY_ID,
    )

    _undo_event = undo_btn.click(
        fn=undo_handler,
        inputs=[_undo_stack_state],
        outputs=_process_inputs + [_undo_stack_state, undo_btn, redo_btn],
        show_progress="hidden",
        queue=False,
    )
    _undo_event.success(
        fn=revision_after_history,
        inputs=[_settings_revision_state, _undo_stack_state],
        outputs=[_settings_revision_state, status],
        show_progress="hidden",
        queue=False,
    )

    _redo_event = redo_btn.click(
        fn=redo_handler,
        inputs=[_undo_stack_state],
        outputs=_process_inputs + [_undo_stack_state, undo_btn, redo_btn],
        show_progress="hidden",
        queue=False,
    )
    _redo_event.success(
        fn=revision_after_history,
        inputs=[_settings_revision_state, _undo_stack_state],
        outputs=[_settings_revision_state, status],
        show_progress="hidden",
        queue=False,
    )

    save_snapshot_btn.click(
        fn=save_snapshot_handler,
        inputs=[snapshot_name] + _process_inputs + [
            _advanced_edit_log_state, _advanced_base_contract_state,
            _advanced_history_state, _advanced_current_state, _snapshot_state,
        ],
        outputs=[snapshot_dropdown, _snapshot_state],
    )

    compare_snapshot_btn.click(
        fn=compare_snapshot_handler,
        inputs=[snapshot_dropdown] + _process_inputs + [_snapshot_state],
        outputs=[snapshot_inspection_out],
        show_progress="hidden",
        queue=False,
    )

    inspect_render_btn.click(
        fn=inspect_render_handler,
        inputs=[
            _processed_result_state,
            _render_snapshot_state,
            _settings_revision_state,
            inspection_mode,
            inspection_face_index,
            inspection_roi,
            _preview_cache_state,
        ],
        outputs=[inspection_output, inspection_status],
        show_progress="hidden",
        queue=False,
    )

    # Session-owned artifact cleanup is scoped by Gradio's request hash. Keep
    # the hook optional for older Gradio versions used by headless tests.
    if callable(getattr(app, "unload", None)):
        app.unload(fn=cleanup_gui_request)

if __name__ == "__main__":
    from retouch.diagnostics import setup_file_logging
    setup_file_logging()

    # Non-blocking update check; surfaces a toast once the UI is up.
    import threading
    from retouch.update_check import check_for_update, offline_mode_enabled

    def _bg_update_check():
        info = check_for_update()
        if info is not None:
            try:
                gr.Info(f"Update available: {info.latest_version} — {info.url}", duration=20)
            except Exception:  # noqa: BLE001 — UI not ready yet; drop silently
                pass

    if not offline_mode_enabled():
        threading.Thread(target=_bg_update_check, daemon=True).start()
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=7860)
