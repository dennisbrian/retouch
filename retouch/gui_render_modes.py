"""Pure render-mode contracts for the Retouch GUI.

This module deliberately does not import Gradio, OpenCV, or the Retouch
engine.  It describes the boundary between a draft GUI state and work that a
later integration layer may queue:

* ``Render Preview`` renders only the first image at ``fast=True`` and never
  exports automatically.
* ``Export Full Quality`` renders one image at ``fast=False`` with an
  explicit metadata-preserving export intent.
* ``Export All`` creates a handoff for the existing batch workflow instead of
  pretending that a single-image event exported the shoot.

The returned structures contain only JSON-compatible values.  Settings are
hashed at capture time so a queued callback can prove which immutable draft
it received.  A stale render can still be described and displayed, but its
``safe_to_commit`` flag is false; callers must not silently treat it as the
current preview or export.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
CONTRACT_NAME = "retouch.gui.render_modes"

MODE_RENDER_PREVIEW = "render_preview"
MODE_EXPORT_FULL_QUALITY = "export_full_quality"
MODE_EXPORT_ALL = "export_all"
SUPPORTED_MODES = (
    MODE_RENDER_PREVIEW,
    MODE_EXPORT_FULL_QUALITY,
    MODE_EXPORT_ALL,
)

_MODE_ALIASES = {
    "preview": MODE_RENDER_PREVIEW,
    "render_preview": MODE_RENDER_PREVIEW,
    "render-preview": MODE_RENDER_PREVIEW,
    "export_full": MODE_EXPORT_FULL_QUALITY,
    "export_full_quality": MODE_EXPORT_FULL_QUALITY,
    "export-full-quality": MODE_EXPORT_FULL_QUALITY,
    "full_quality": MODE_EXPORT_FULL_QUALITY,
    "full-quality": MODE_EXPORT_FULL_QUALITY,
    "export_all": MODE_EXPORT_ALL,
    "export-all": MODE_EXPORT_ALL,
    "all": MODE_EXPORT_ALL,
}


class RenderContractError(ValueError):
    """Raised when a render contract cannot be constructed safely."""


def _json_safe(value: Any, path: str = "$") -> Any:
    """Copy a value while rejecting values that cannot be represented in JSON."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RenderContractError("non-finite value at %s" % path)
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        copied: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RenderContractError("non-string JSON key at %s" % path)
            copied[key] = _json_safe(item, "%s.%s" % (path, key))
        return copied
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, "%s[%d]" % (path, index)) for index, item in enumerate(value)]
    raise RenderContractError(
        "unsupported non-JSON value at %s: %s" % (path, type(value).__name__)
    )


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for hashing and handoff comparison."""

    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    """Hash a JSON-safe value deterministically."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_revision(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RenderContractError("%s must be a non-negative integer" % name)
    return value


def _optional_revision(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    return _require_revision(value, name)


def _revision_label(kind: str, revision: Optional[int]) -> str:
    if revision is None:
        return "%s revision unavailable" % kind.capitalize()
    return "%s revision %d" % (kind.capitalize(), revision)


def _normalise_paths(source_paths: Any) -> List[str]:
    if isinstance(source_paths, (str, os.PathLike)):
        values = [source_paths]
    elif isinstance(source_paths, Sequence) and not isinstance(source_paths, (bytes, bytearray)):
        values = list(source_paths)
    else:
        raise RenderContractError("source_paths must be a path or a sequence of paths")

    if not values:
        raise RenderContractError("source_paths must contain at least one path")

    result = []
    for index, value in enumerate(values):
        if not isinstance(value, (str, os.PathLike)):
            raise RenderContractError("source_paths[%d] is not a path" % index)
        path = os.fspath(value)
        if not isinstance(path, str) or not path.strip():
            raise RenderContractError("source_paths[%d] is empty" % index)
        result.append(path)
    return result


def _snapshot_errors(snapshot: Any) -> List[str]:
    errors: List[str] = []
    if not isinstance(snapshot, Mapping):
        return ["settings_snapshot must be a mapping"]

    if snapshot.get("schema_version") != SCHEMA_VERSION:
        errors.append("settings_snapshot_schema_version")

    try:
        revision = _require_revision(snapshot.get("revision"), "settings_snapshot.revision")
    except RenderContractError:
        errors.append("settings_snapshot_revision")
        revision = None

    settings = snapshot.get("settings")
    if not isinstance(settings, Mapping):
        errors.append("settings_snapshot_settings")
    else:
        try:
            safe_settings = _json_safe(settings, "$.settings_snapshot.settings")
        except RenderContractError:
            errors.append("settings_snapshot_not_json_safe")
            safe_settings = None

        digest = snapshot.get("settings_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append("settings_snapshot_hash")
        elif safe_settings is not None and digest != canonical_sha256(safe_settings):
            errors.append("settings_snapshot_hash_mismatch")

    # Keep this read to make the local variable intentional when a malformed
    # snapshot has no valid revision.  The revision is checked above.
    _ = revision
    return errors


def capture_settings_snapshot(settings: Mapping[str, Any], revision: int) -> Dict[str, Any]:
    """Capture a deep, hashable settings snapshot for queued work.

    The caller's mapping is copied recursively.  Unsupported values and
    invalid revisions raise instead of being stringified or dropped.
    """

    if not isinstance(settings, Mapping):
        raise RenderContractError("settings must be a mapping")
    captured_revision = _require_revision(revision, "revision")
    safe_settings = _json_safe(settings, "$.settings")
    return {
        "schema_version": SCHEMA_VERSION,
        "revision": captured_revision,
        "settings": safe_settings,
        "settings_sha256": canonical_sha256(safe_settings),
    }


def compare_revisions(rendered_revision: Any, current_revision: Any) -> Dict[str, Any]:
    """Compare a completed render with the current draft revision.

    Missing, malformed, or impossible (future-rendered) revisions are
    unresolved and stale.  That is intentionally fail-closed for callers
    deciding whether a result may replace the visible preview.
    """

    try:
        rendered = _require_revision(rendered_revision, "rendered_revision")
        current = _require_revision(current_revision, "current_revision")
    except RenderContractError as exc:
        return {
            "valid": False,
            "stale": True,
            "status": "unresolved",
            "reason": str(exc),
            "rendered_revision": None,
            "current_revision": None,
            "rendered_label": _revision_label("preview", None),
            "current_label": _revision_label("draft", None),
        }

    if rendered > current:
        return {
            "valid": False,
            "stale": True,
            "status": "invalid_order",
            "reason": "rendered_revision_is_newer_than_current_revision",
            "rendered_revision": rendered,
            "current_revision": current,
            "rendered_label": _revision_label("preview", rendered),
            "current_label": _revision_label("draft", current),
        }

    stale = current > rendered
    return {
        "valid": True,
        "stale": stale,
        "status": "stale" if stale else "current",
        "reason": "settings_changed_during_render" if stale else None,
        "rendered_revision": rendered,
        "current_revision": current,
        "rendered_label": _revision_label("preview", rendered),
        "current_label": _revision_label("draft", current),
    }


def _revision_state(
    draft_revision: Any,
    preview_revision: Any,
    captured_revision: int,
) -> Dict[str, Any]:
    draft = _require_revision(draft_revision, "draft_revision")
    preview = _optional_revision(preview_revision, "preview_revision")
    captured = _require_revision(captured_revision, "captured_revision")
    render_comparison = compare_revisions(captured, draft)
    preview_comparison = (
        compare_revisions(preview, draft)
        if preview is not None
        else {
            "valid": True,
            "stale": False,
            "status": "unavailable",
            "reason": "no_previous_preview",
            "rendered_revision": None,
            "current_revision": draft,
            "rendered_label": _revision_label("preview", None),
            "current_label": _revision_label("draft", draft),
        }
    )
    errors = []
    if not render_comparison["valid"]:
        errors.append("render_revision_comparison")
    if not preview_comparison["valid"]:
        errors.append("preview_revision_comparison")
    return {
        "valid": not errors,
        "errors": errors,
        "draft_revision": draft,
        "preview_revision": preview,
        "captured_revision": captured,
        "labels": {
            "draft": _revision_label("draft", draft),
            "preview": _revision_label("preview", preview),
            "captured": "Captured settings revision %d" % captured,
            "render": "Rendering settings revision %d" % captured,
        },
        "render_vs_draft": render_comparison,
        "preview_vs_draft": preview_comparison,
        "stale": bool(render_comparison["stale"]),
        "safe_to_commit": bool(
            not errors and render_comparison["valid"] and not render_comparison["stale"]
        ),
    }


def normalize_mode(mode: Any) -> str:
    """Return a canonical mode name or raise for an unknown mode."""

    if not isinstance(mode, str):
        raise RenderContractError("mode must be a string")
    key = mode.strip().lower().replace(" ", "_")
    canonical = _MODE_ALIASES.get(key)
    if canonical is None:
        raise RenderContractError("unsupported render mode: %r" % mode)
    return canonical


def _export_intent(mode: str) -> Dict[str, Any]:
    if mode == MODE_RENDER_PREVIEW:
        return {
            "requested": False,
            "automatic": False,
            "quality": "preview",
            "fast": True,
            "metadata_preserving": False,
            "writer_policy": None,
            "reason": "preview_only_no_automatic_export",
        }

    return {
        "requested": True,
        "automatic": mode == MODE_EXPORT_FULL_QUALITY,
        "quality": "full",
        "fast": False,
        "metadata_preserving": True,
        "writer_policy": "metadata_preserving",
        "reason": "full_quality_export",
    }


def build_render_contract(
    mode: Any,
    source_paths: Any,
    settings_snapshot: Mapping[str, Any],
    draft_revision: int,
    preview_revision: Optional[int] = None,
    *,
    batch_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one of the three stable render-mode contracts.

    This function performs strict input validation.  Use
    :func:`validate_render_contract` at a later boundary when consuming a
    contract received from browser/session state.
    """

    canonical_mode = normalize_mode(mode)
    paths = _normalise_paths(source_paths)
    snapshot_errors = _snapshot_errors(settings_snapshot)
    if snapshot_errors:
        raise RenderContractError(
            "invalid settings snapshot: %s" % ", ".join(snapshot_errors)
        )
    snapshot = _json_safe(settings_snapshot, "$.settings_snapshot")
    revisions = _revision_state(
        draft_revision,
        preview_revision,
        snapshot["revision"],
    )

    if canonical_mode == MODE_RENDER_PREVIEW:
        selected_paths = paths[:1]
        render = {
            "execute": True,
            "scope": "first_image",
            "first_image_only": True,
            "selected_count": 1,
            "fast": True,
            "automatic_export": False,
            "output": "preview_only",
        }
        mode_label = "Render Preview"
        batch_handoff = None
    elif canonical_mode == MODE_EXPORT_FULL_QUALITY:
        if len(paths) != 1:
            raise RenderContractError(
                "Export Full Quality accepts exactly one source path; use Export All for a shoot"
            )
        selected_paths = paths
        render = {
            "execute": True,
            "scope": "single_image",
            "first_image_only": False,
            "selected_count": 1,
            "fast": False,
            "automatic_export": True,
            "output": "full_quality_export",
        }
        mode_label = "Export Full Quality"
        batch_handoff = None
    else:
        selected_paths = paths
        job_id = None
        if batch_job_id is not None:
            if not isinstance(batch_job_id, str) or not batch_job_id.strip():
                raise RenderContractError("batch_job_id must be a non-empty string")
            job_id = batch_job_id
        render = {
            "execute": False,
            "scope": "all_images",
            "first_image_only": False,
            "selected_count": len(paths),
            "fast": False,
            "automatic_export": False,
            "output": "batch_handoff",
        }
        mode_label = "Export All"
        batch_handoff = {
            "required": True,
            "kind": "batch",
            "workflow": "Batch",
            "job_kind": "final",
            "source_paths": list(paths),
            "settings_snapshot_sha256": snapshot["settings_sha256"],
            "captured_revision": snapshot["revision"],
            "job_id": job_id,
        }

    contract: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": CONTRACT_NAME,
        "mode": canonical_mode,
        "mode_label": mode_label,
        "source": {
            "requested_paths": list(paths),
            "selected_paths": selected_paths,
            "requested_count": len(paths),
            "selected_count": len(selected_paths),
            "first_image_only": canonical_mode == MODE_RENDER_PREVIEW,
        },
        "settings": {
            "snapshot": snapshot,
            "captured_revision": snapshot["revision"],
            "settings_sha256": snapshot["settings_sha256"],
        },
        "revisions": revisions,
        "revision_labels": revisions["labels"],
        "render": render,
        "export_intent": _export_intent(canonical_mode),
        "batch_handoff": batch_handoff,
    }
    validation = validate_render_contract(contract)
    contract["validation"] = validation
    contract["stale"] = revisions["stale"]
    contract["safe_to_commit"] = validation["safe_to_commit"]
    return contract


def build_render_preview(
    source_paths: Any,
    settings_snapshot: Mapping[str, Any],
    draft_revision: int,
    preview_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Describe a first-image, fast preview with no automatic export."""

    return build_render_contract(
        MODE_RENDER_PREVIEW,
        source_paths,
        settings_snapshot,
        draft_revision,
        preview_revision,
    )


def build_export_full_quality(
    source_path: Any,
    settings_snapshot: Mapping[str, Any],
    draft_revision: int,
    preview_revision: Optional[int] = None,
) -> Dict[str, Any]:
    """Describe a single-image full-quality, metadata-preserving export."""

    return build_render_contract(
        MODE_EXPORT_FULL_QUALITY,
        source_path,
        settings_snapshot,
        draft_revision,
        preview_revision,
    )


def build_export_all(
    source_paths: Any,
    settings_snapshot: Mapping[str, Any],
    draft_revision: int,
    preview_revision: Optional[int] = None,
    *,
    batch_job_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Describe a handoff to the full-quality batch workflow."""

    return build_render_contract(
        MODE_EXPORT_ALL,
        source_paths,
        settings_snapshot,
        draft_revision,
        preview_revision,
        batch_job_id=batch_job_id,
    )


def validate_render_contract(contract: Any) -> Dict[str, Any]:
    """Validate a render contract without ever approving malformed input.

    ``valid`` describes structural and semantic correctness.  ``safe_to_commit``
    additionally requires that the captured render is current with the draft.
    A stale result therefore remains inspectable but cannot silently replace a
    newer preview or be treated as the requested export.
    """

    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "valid": False,
        "safe_to_commit": False,
        "errors": [],
        "warnings": [],
    }
    if not isinstance(contract, Mapping):
        result["errors"].append("contract must be a mapping")
        return result

    try:
        _json_safe(contract, "$.contract")
    except RenderContractError as exc:
        result["errors"].append(str(exc))
        return result

    if contract.get("schema_version") != SCHEMA_VERSION:
        result["errors"].append("schema_version")
    if contract.get("contract") != CONTRACT_NAME:
        result["errors"].append("contract_name")

    try:
        mode = normalize_mode(contract.get("mode"))
    except RenderContractError:
        result["errors"].append("mode")
        mode = None

    source = contract.get("source")
    if not isinstance(source, Mapping):
        result["errors"].append("source")
    else:
        requested = source.get("requested_paths")
        selected = source.get("selected_paths")
        if not isinstance(requested, list) or not requested:
            result["errors"].append("source.requested_paths")
        if not isinstance(selected, list) or not selected:
            result["errors"].append("source.selected_paths")
        if isinstance(requested, list) and isinstance(selected, list):
            if any(not isinstance(path, str) or not path.strip() for path in requested + selected):
                result["errors"].append("source.paths_json_safe")
            if mode == MODE_RENDER_PREVIEW and (
                len(selected) != 1 or selected[0] != requested[0]
            ):
                result["errors"].append("preview_first_image_only")
            if mode == MODE_EXPORT_FULL_QUALITY and len(requested) != 1:
                result["errors"].append("full_quality_single_image")
            if mode == MODE_EXPORT_ALL and selected != requested:
                result["errors"].append("export_all_source_selection")

    settings = contract.get("settings")
    captured_revision = None
    if not isinstance(settings, Mapping):
        result["errors"].append("settings")
    else:
        snapshot = settings.get("snapshot")
        snapshot_errors = _snapshot_errors(snapshot)
        result["errors"].extend(snapshot_errors)
        if isinstance(snapshot, Mapping):
            captured_revision = snapshot.get("revision")
            if settings.get("captured_revision") != captured_revision:
                result["errors"].append("captured_revision")
            if settings.get("settings_sha256") != snapshot.get("settings_sha256"):
                result["errors"].append("settings_sha256")

    revisions = contract.get("revisions")
    if not isinstance(revisions, Mapping):
        result["errors"].append("revisions")
    else:
        try:
            draft = _require_revision(revisions.get("draft_revision"), "draft_revision")
            preview = _optional_revision(revisions.get("preview_revision"), "preview_revision")
            captured = _require_revision(revisions.get("captured_revision"), "captured_revision")
            if captured_revision != captured:
                result["errors"].append("revision_snapshot_mismatch")
            expected_revisions = _revision_state(draft, preview, captured)
            if revisions.get("labels") != expected_revisions["labels"]:
                result["errors"].append("revision_labels")
            if revisions.get("stale") != expected_revisions["stale"]:
                result["errors"].append("revision_stale_state")
            if revisions.get("safe_to_commit") != expected_revisions["safe_to_commit"]:
                result["errors"].append("revision_commit_state")
            if expected_revisions["stale"]:
                result["warnings"].append("render_stale_against_draft")
            if not expected_revisions["safe_to_commit"]:
                result["warnings"].append("render_not_current")
        except RenderContractError:
            result["errors"].append("revision_values")

    render = contract.get("render")
    if not isinstance(render, Mapping) or mode is None:
        result["errors"].append("render")
    else:
        expected = {
            MODE_RENDER_PREVIEW: {
                "execute": True,
                "scope": "first_image",
                "first_image_only": True,
                "fast": True,
                "automatic_export": False,
                "output": "preview_only",
            },
            MODE_EXPORT_FULL_QUALITY: {
                "execute": True,
                "scope": "single_image",
                "first_image_only": False,
                "fast": False,
                "automatic_export": True,
                "output": "full_quality_export",
            },
            MODE_EXPORT_ALL: {
                "execute": False,
                "scope": "all_images",
                "first_image_only": False,
                "fast": False,
                "automatic_export": False,
                "output": "batch_handoff",
            },
        }[mode]
        for key, value in expected.items():
            if render.get(key) != value:
                result["errors"].append("render.%s" % key)

    export_intent = contract.get("export_intent")
    if not isinstance(export_intent, Mapping) or mode is None:
        result["errors"].append("export_intent")
    else:
        expected_requested = mode != MODE_RENDER_PREVIEW
        if export_intent.get("requested") is not expected_requested:
            result["errors"].append("export_intent.requested")
        if mode == MODE_RENDER_PREVIEW:
            if export_intent.get("automatic") is not False:
                result["errors"].append("preview_automatic_export")
        else:
            for key, value in {
                "automatic": mode == MODE_EXPORT_FULL_QUALITY,
                "quality": "full",
                "fast": False,
                "metadata_preserving": True,
                "writer_policy": "metadata_preserving",
            }.items():
                if export_intent.get(key) != value:
                    result["errors"].append("export_intent.%s" % key)

    handoff = contract.get("batch_handoff")
    if mode == MODE_EXPORT_ALL:
        if not isinstance(handoff, Mapping) or handoff.get("required") is not True:
            result["errors"].append("batch_handoff")
        elif handoff.get("kind") != "batch" or handoff.get("workflow") != "Batch":
            result["errors"].append("batch_handoff.workflow")
    elif handoff is not None:
        result["errors"].append("unexpected_batch_handoff")

    result["valid"] = not result["errors"]
    revisions_valid = isinstance(revisions, Mapping) and revisions.get("safe_to_commit") is True
    result["safe_to_commit"] = bool(result["valid"] and revisions_valid)
    return result


def require_current_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a contract only when it is valid and current; otherwise raise."""

    validation = validate_render_contract(contract)
    if not validation["valid"] or not validation["safe_to_commit"]:
        details = validation["errors"] or validation["warnings"] or ["contract is not current"]
        raise RenderContractError("render contract rejected: %s" % ", ".join(details))
    return _json_safe(contract)


__all__ = [
    "CONTRACT_NAME",
    "MODE_EXPORT_ALL",
    "MODE_EXPORT_FULL_QUALITY",
    "MODE_RENDER_PREVIEW",
    "RenderContractError",
    "SUPPORTED_MODES",
    "build_export_all",
    "build_export_full_quality",
    "build_render_contract",
    "build_render_preview",
    "canonical_json",
    "canonical_sha256",
    "capture_settings_snapshot",
    "compare_revisions",
    "normalize_mode",
    "require_current_contract",
    "validate_render_contract",
]
