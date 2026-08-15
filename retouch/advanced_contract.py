"""Deterministic provenance contracts for Advanced Retouch.

Advanced Retouch replays compact manual operations onto a base image.  The
operation log is reproducible only when it is tied to the exact pixels it was
created against.  This module keeps that contract independent of Gradio so it
can be validated by sessions, exports, tests, and future certification tools.

The contract deliberately stores hashes and dimensions, never image pixels.
Processed recipe results additionally carry render evidence; preview/proxy
renders may be edited, but are not eligible for delivery exports.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


BASE_CONTRACT_NAME = "retouch.advanced.base"
BASE_SCHEMA_VERSION = 1
SESSION_SCHEMA_VERSION = 2

BASE_KIND_SOURCE = "source_original"
BASE_KIND_PROCESSED = "processed_recipe"
SUPPORTED_BASE_KINDS = (BASE_KIND_SOURCE, BASE_KIND_PROCESSED)

DETAIL_NATIVE = "native"
DETAIL_PROXY = "proxy"
DETAIL_UNVERIFIED = "unverified"

RENDER_PREVIEW = "render_preview"
RENDER_FULL_QUALITY = "export_full_quality"


class AdvancedContractError(ValueError):
    """Raised when Advanced Retouch evidence is incomplete or inconsistent."""


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise AdvancedContractError("Advanced Retouch evidence is not JSON-safe: %s" % exc)


def canonical_sha256(value: Any) -> str:
    """Return a deterministic digest for a JSON-safe evidence value."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _json_copy(value: Any) -> Any:
    return json.loads(_canonical_json(value))


def pixel_sha256(image: Any) -> str:
    """Hash exact array identity, including shape and dtype."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise AdvancedContractError("Advanced base must be an RGB/RGBA image")
    contiguous = np.ascontiguousarray(array)
    header = {
        "dtype": contiguous.dtype.str,
        "shape": [int(value) for value in contiguous.shape],
    }
    digest = hashlib.sha256()
    digest.update(_canonical_json(header).encode("utf-8"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def pixel_descriptor(image: Any) -> Dict[str, Any]:
    """Return the replay identity for an RGB/RGBA array."""

    array = np.asarray(image)
    digest = pixel_sha256(array)
    return {
        "sha256": digest,
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "channels": int(array.shape[2]),
        "dtype": str(array.dtype),
        "encoding": "working-srgb",
        "effective_bits": 8 if array.dtype == np.uint8 else None,
    }


def _source_file_identity(path: Optional[Any]) -> Dict[str, Any]:
    if path is None:
        return {"path": None, "content_sha256": None, "size_bytes": None, "mtime_ns": None}
    path_value = os.fspath(path)
    try:
        stat = os.stat(path_value)
        digest = hashlib.sha256()
        with open(path_value, "rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return {
            "path": path_value,
            "content_sha256": digest.hexdigest(),
            "size_bytes": int(stat.st_size),
            "mtime_ns": int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000))),
        }
    except OSError:
        return {"path": path_value, "content_sha256": None, "size_bytes": None, "mtime_ns": None}


def _dimension_pair(value: Any) -> Optional[List[int]]:
    if not isinstance(value, Mapping):
        return None
    try:
        width = int(value.get("width"))
        height = int(value.get("height"))
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return [width, height]


def _valid_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


def _processed_render_contract(
    image: np.ndarray,
    render_evidence: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    evidence = render_evidence if isinstance(render_evidence, Mapping) else {}
    mode = evidence.get("render_mode")
    revision = evidence.get("render_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
        revision = None
    settings_sha256 = evidence.get("settings_sha256")
    if not _valid_sha256(settings_sha256):
        settings_sha256 = None

    native_size = _dimension_pair(evidence.get("native"))
    output_size = _dimension_pair(evidence.get("output"))
    actual_size = [int(image.shape[1]), int(image.shape[0])]
    dimensions_match = bool(
        native_size
        and output_size
        and native_size == output_size
        and output_size == actual_size
    )

    if mode == RENDER_PREVIEW:
        detail = DETAIL_PROXY
        reason = "render_preview_uses_proxy_detail"
    elif (
        mode == RENDER_FULL_QUALITY
        and revision is not None
        and settings_sha256 is not None
        and dimensions_match
    ):
        detail = DETAIL_NATIVE
        reason = None
    else:
        detail = DETAIL_UNVERIFIED
        reason = "native_render_evidence_incomplete"

    return {
        "mode": mode,
        "revision": revision,
        "settings_sha256": settings_sha256,
        "native_size": native_size,
        "output_size": output_size or actual_size,
        "effective_detail": detail,
        "delivery_eligible": detail == DETAIL_NATIVE,
        "delivery_block_reason": reason,
    }


def build_base_contract(
    image_rgb: Any,
    *,
    kind: str,
    source_path: Optional[Any] = None,
    render_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build pixel and render evidence for an Advanced Retouch base."""

    if kind not in SUPPORTED_BASE_KINDS:
        raise AdvancedContractError("unsupported Advanced base kind: %r" % kind)
    array = np.asarray(image_rgb)
    pixels = pixel_descriptor(array)
    path_value = os.fspath(source_path) if source_path is not None else None

    if kind == BASE_KIND_PROCESSED:
        render = _processed_render_contract(array, render_evidence)
        if path_value is None and isinstance(render_evidence, Mapping):
            evidence_path = render_evidence.get("source_path")
            if isinstance(evidence_path, str) and evidence_path:
                path_value = evidence_path
    else:
        render = {
            "mode": BASE_KIND_SOURCE,
            "revision": None,
            "settings_sha256": None,
            "native_size": [int(array.shape[1]), int(array.shape[0])],
            "output_size": [int(array.shape[1]), int(array.shape[0])],
            "effective_detail": DETAIL_NATIVE,
            "delivery_eligible": True,
            "delivery_block_reason": None,
        }

    payload: Dict[str, Any] = {
        "contract": BASE_CONTRACT_NAME,
        "schema_version": BASE_SCHEMA_VERSION,
        "kind": kind,
        "pixels": pixels,
        "source": _source_file_identity(path_value),
        "render": render,
    }
    payload["contract_sha256"] = canonical_sha256(payload)
    return payload


def validate_base_contract(contract: Any) -> Dict[str, Any]:
    """Validate contract integrity without requiring its image pixels."""

    reasons: List[str] = []
    if not isinstance(contract, Mapping):
        return {"valid": False, "reasons": ["base_contract_missing"]}
    if contract.get("contract") != BASE_CONTRACT_NAME:
        reasons.append("contract_name")
    if contract.get("schema_version") != BASE_SCHEMA_VERSION:
        reasons.append("schema_version")
    if contract.get("kind") not in SUPPORTED_BASE_KINDS:
        reasons.append("base_kind")
    pixels = contract.get("pixels")
    if not isinstance(pixels, Mapping) or not _valid_sha256(pixels.get("sha256")):
        reasons.append("pixel_hash")
    render = contract.get("render")
    if not isinstance(render, Mapping):
        reasons.append("render_evidence")
    expected_digest = contract.get("contract_sha256")
    unsigned = dict(contract)
    unsigned.pop("contract_sha256", None)
    if not _valid_sha256(expected_digest) or expected_digest != canonical_sha256(unsigned):
        reasons.append("contract_hash")
    return {"valid": not reasons, "reasons": reasons}


def compare_base_contracts(expected: Any, actual: Any) -> Dict[str, Any]:
    """Compare the exact replay identity while allowing a source file to move."""

    expected_validation = validate_base_contract(expected)
    actual_validation = validate_base_contract(actual)
    reasons: List[str] = []
    if not expected_validation["valid"]:
        reasons.append("expected_base_invalid")
    if not actual_validation["valid"]:
        reasons.append("actual_base_invalid")
    if reasons:
        return {"matches": False, "reasons": reasons}

    if expected.get("kind") != actual.get("kind"):
        reasons.append("base_kind_mismatch")
    expected_pixels = expected.get("pixels", {})
    actual_pixels = actual.get("pixels", {})
    for key in ("sha256", "width", "height", "channels", "dtype"):
        if expected_pixels.get(key) != actual_pixels.get(key):
            reasons.append("pixel_%s_mismatch" % key)

    if expected.get("kind") == BASE_KIND_PROCESSED:
        expected_render = expected.get("render", {})
        actual_render = actual.get("render", {})
        for key in ("mode", "revision", "settings_sha256", "effective_detail"):
            if expected_render.get(key) != actual_render.get(key):
                reasons.append("render_%s_mismatch" % key)
    return {"matches": not reasons, "reasons": reasons}


def build_session_payload(
    edits: Any,
    base_contract: Any,
    *,
    history_state: Optional[Mapping[str, Any]] = None,
    result_rgb: Any = None,
) -> Dict[str, Any]:
    """Build a deterministic Advanced session payload.

    A compact history whose beginning has been evicted cannot be replayed from
    the original base.  Saving therefore fails closed instead of producing a
    session that appears complete.
    """

    validation = validate_base_contract(base_contract)
    if not validation["valid"]:
        raise AdvancedContractError(
            "Advanced base evidence is invalid: %s" % ", ".join(validation["reasons"])
        )
    if not isinstance(edits, list) or any(not isinstance(edit, Mapping) for edit in edits):
        raise AdvancedContractError("Advanced edits must be a list of mappings")
    copied_edits = _json_copy(edits)

    base_cursor = 0
    cursor = len(copied_edits)
    if isinstance(history_state, Mapping):
        try:
            base_cursor = int(history_state.get("base_cursor", 0))
            cursor = int(history_state.get("cursor", len(copied_edits)))
        except (TypeError, ValueError):
            raise AdvancedContractError("Advanced history cursors are invalid")
    if base_cursor != 0:
        raise AdvancedContractError(
            "Advanced history was truncated; export the canvas or reset before saving a replay session"
        )
    if cursor != len(copied_edits):
        raise AdvancedContractError("Advanced edit log does not match the history cursor")
    if result_rgb is None:
        raise AdvancedContractError("Advanced result pixels are required for deterministic replay")

    return {
        "version": SESSION_SCHEMA_VERSION,
        "base": _json_copy(base_contract),
        "edits": copied_edits,
        "edits_sha256": canonical_sha256(copied_edits),
        "history": {
            "base_cursor": base_cursor,
            "cursor": cursor,
            "complete": True,
        },
        "result": pixel_descriptor(result_rgb),
    }


def parse_session_payload(payload: Any) -> Dict[str, Any]:
    """Validate v2 payloads and migrate legacy v1 edit-only sessions."""

    if not isinstance(payload, Mapping):
        raise AdvancedContractError("Advanced session payload must be a mapping")
    version = payload.get("version", 1)
    edits = payload.get("edits", [])
    if not isinstance(edits, list) or any(not isinstance(edit, Mapping) for edit in edits):
        raise AdvancedContractError("Advanced session edits must be a list of mappings")
    copied_edits = _json_copy(edits)

    if version == 1:
        return {
            "version": 1,
            "base": None,
            "edits": copied_edits,
            "legacy_unverified": True,
        }
    if version != SESSION_SCHEMA_VERSION:
        raise AdvancedContractError("unsupported Advanced session version: %r" % version)

    base = payload.get("base")
    validation = validate_base_contract(base)
    if not validation["valid"]:
        raise AdvancedContractError(
            "Advanced session base evidence is invalid: %s" % ", ".join(validation["reasons"])
        )
    if payload.get("edits_sha256") != canonical_sha256(copied_edits):
        raise AdvancedContractError("Advanced session edit hash does not match")
    history = payload.get("history")
    if not isinstance(history, Mapping) or history.get("complete") is not True:
        raise AdvancedContractError("Advanced session history is incomplete")
    try:
        base_cursor = int(history.get("base_cursor"))
        cursor = int(history.get("cursor"))
    except (TypeError, ValueError):
        raise AdvancedContractError("Advanced session history cursors are invalid")
    if base_cursor != 0 or cursor != len(copied_edits):
        raise AdvancedContractError("Advanced session history does not match its edit log")
    result = payload.get("result")
    if not isinstance(result, Mapping) or not _valid_sha256(result.get("sha256")):
        raise AdvancedContractError("Advanced session result evidence is invalid")
    return {
        "version": SESSION_SCHEMA_VERSION,
        "base": _json_copy(base),
        "edits": copied_edits,
        "edits_sha256": payload.get("edits_sha256"),
        "history": _json_copy(history),
        "result": _json_copy(result),
        "legacy_unverified": False,
    }


def replay_result_matches(payload: Mapping[str, Any], image_rgb: Any) -> Dict[str, Any]:
    """Verify that replay reproduced the result recorded when the session was saved."""

    expected = payload.get("result") if isinstance(payload, Mapping) else None
    if not isinstance(expected, Mapping):
        return {"matches": False, "reasons": ["result_evidence_missing"]}
    actual = pixel_descriptor(image_rgb)
    reasons = []
    for key in ("sha256", "width", "height", "channels", "dtype", "encoding", "effective_bits"):
        if expected.get(key) != actual.get(key):
            reasons.append("result_%s_mismatch" % key)
    return {"matches": not reasons, "reasons": reasons, "actual": actual}


def source_file_matches(base_contract: Any, path: Any) -> bool:
    """Return whether *path* is the exact source file recorded by the base."""

    validation = validate_base_contract(base_contract)
    if not validation["valid"] or path is None:
        return False
    expected = base_contract.get("source", {})
    expected_digest = expected.get("content_sha256") if isinstance(expected, Mapping) else None
    if not _valid_sha256(expected_digest):
        return False
    actual = _source_file_identity(path)
    return bool(
        actual.get("content_sha256") == expected_digest
        and actual.get("size_bytes") == expected.get("size_bytes")
    )


def delivery_decision(base_contract: Any) -> Dict[str, Any]:
    """Return a fail-closed delivery decision for an Advanced canvas."""

    validation = validate_base_contract(base_contract)
    if not validation["valid"]:
        return {
            "allowed": False,
            "reason": "base_evidence_invalid",
            "detail": ", ".join(validation["reasons"]),
        }
    render = base_contract.get("render", {})
    if render.get("delivery_eligible") is not True:
        return {
            "allowed": False,
            "reason": render.get("delivery_block_reason") or "base_not_delivery_eligible",
            "detail": render.get("effective_detail", DETAIL_UNVERIFIED),
        }
    return {
        "allowed": True,
        "reason": None,
        "detail": render.get("effective_detail", DETAIL_NATIVE),
    }
