"""Pixel-free render manifest and inspector contract.

This module records what a Retouch render did without retaining image pixels.
It is intentionally independent from the GUI, processing engine, and
certification evidence modules so it can be used by previews, exports, batch
workers, or a future inspector UI.

The public payload is versioned and contains only JSON-compatible values:

* render revision, settings digest, mode, and lifecycle status;
* source/output identities with dimensions, dtype, and content hashes;
* cache, color, metadata, face-count, timing, QA, Safe Auto, and runtime
  backend/provider observations; and
* a deterministic manifest digest computed over the payload without that
  digest field.

No array, image object, byte buffer, or other pixel-bearing value is accepted.
Callers should store large previews or exports in session-owned artifacts and
reference them by ``path``/``uri`` and SHA-256 here.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CONTRACT_NAME = "retouch.render_manifest"
SCHEMA_VERSION = 1
SUPPORTED_MODES = ("preview", "full")
SUPPORTED_STATUSES = ("queued", "running", "completed", "failed", "cancelled")
SUPPORTED_CACHE_STATUSES = ("hit", "miss", "bypassed", "unknown")

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


class RenderManifestError(ValueError):
    """Raised when a render manifest is malformed or not JSON-safe."""


def _json_safe(value: Any, path: str = "$") -> Any:
    """Return a detached JSON-safe copy, rejecting pixel-bearing objects.

    Dataclasses and enum values are supported because engine observations are
    commonly represented that way. NumPy arrays, PIL images, bytes, sets, and
    arbitrary objects are rejected rather than summarized: silently accepting
    one would make a supposedly pixel-free manifest unsafe to persist.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RenderManifestError("non-finite number at %s" % path)
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Enum):
        return _json_safe(value.value, path)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        raise RenderManifestError("array-like pixel data is not allowed at %s" % path)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(
            {item.name: getattr(value, item.name) for item in dataclasses.fields(value)},
            path,
        )
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise RenderManifestError("non-string JSON key at %s" % path)
            result[key] = _json_safe(item, "%s.%s" % (path, key))
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, "%s[%d]" % (path, index)) for index, item in enumerate(value)]
    if isinstance(value, (bytes, bytearray, memoryview)):
        raise RenderManifestError("pixel/byte buffer is not allowed at %s" % path)
    raise RenderManifestError(
        "unsupported non-JSON value at %s: %s" % (path, type(value).__name__)
    )


def canonical_json(value: Any) -> str:
    """Return deterministic JSON for manifest hashing."""

    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 digest of a JSON-safe value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, name: str) -> Dict[str, Any]:
    safe = _json_safe(value, "$%s" % name)
    if not isinstance(safe, dict):
        raise RenderManifestError("%s must be a mapping" % name)
    return safe


def _require_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RenderManifestError("%s must be a non-negative integer" % name)
    return value


def _optional_non_negative_int(value: Any, name: str) -> Optional[int]:
    if value is None:
        return None
    return _require_non_negative_int(value, name)


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise RenderManifestError("%s must be a 64-character SHA-256 hex digest" % name)
    return value.lower()


def _optional_sha256(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_sha256(value, name)


def _normalise_mode(value: Any) -> str:
    if not isinstance(value, str):
        raise RenderManifestError("mode must be a string")
    aliases = {
        "render_preview": "preview",
        "preview": "preview",
        "export_full_quality": "full",
        "export_full": "full",
        "full_quality": "full",
        "full": "full",
    }
    mode = aliases.get(value.strip().lower())
    if mode is None:
        raise RenderManifestError("mode must be one of: %s" % ", ".join(SUPPORTED_MODES))
    return mode


def _validate_dimensions(value: Any, name: str) -> Dict[str, Any]:
    dimensions = _require_mapping(value, name)
    for field_name in ("width", "height", "channels", "depth"):
        if field_name not in dimensions or dimensions[field_name] is None:
            continue
        _require_non_negative_int(dimensions[field_name], "%s.%s" % (name, field_name))
        if dimensions[field_name] == 0:
            raise RenderManifestError("%s.%s must be greater than zero" % (name, field_name))
    if "width" not in dimensions or "height" not in dimensions:
        raise RenderManifestError("%s must include width and height" % name)
    return dimensions


def _normalise_artifact(value: Any, name: str, *, required: bool) -> Optional[Dict[str, Any]]:
    if value is None:
        if required:
            raise RenderManifestError("%s is required" % name)
        return None

    artifact = _require_mapping(value, name)
    identity_keys = ("id", "name", "path", "uri", "sha256")
    if not any(artifact.get(key) not in (None, "") for key in identity_keys):
        raise RenderManifestError(
            "%s must include an identity field: %s" % (name, ", ".join(identity_keys))
        )
    if artifact.get("sha256") is not None:
        artifact["sha256"] = _require_sha256(artifact["sha256"], "%s.sha256" % name)
    if artifact.get("size_bytes") is not None:
        artifact["size_bytes"] = _require_non_negative_int(
            artifact["size_bytes"], "%s.size_bytes" % name
        )
    if artifact.get("mtime_ns") is not None:
        artifact["mtime_ns"] = _require_non_negative_int(
            artifact["mtime_ns"], "%s.mtime_ns" % name
        )
    if artifact.get("dimensions") is not None:
        artifact["dimensions"] = _validate_dimensions(
            artifact["dimensions"], "%s.dimensions" % name
        )
    if artifact.get("dtype") is not None and (
        not isinstance(artifact["dtype"], str) or not artifact["dtype"].strip()
    ):
        raise RenderManifestError("%s.dtype must be a non-empty string" % name)
    return artifact


def _validate_timing_values(value: Any, name: str) -> Dict[str, Any]:
    timings = _require_mapping(value, name)
    for key, item in timings.items():
        child_name = "%s.%s" % (name, key)
        if item is None:
            continue
        if isinstance(item, Mapping):
            _validate_timing_values(item, child_name)
            continue
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise RenderManifestError("%s must be a finite non-negative number" % child_name)
        if not math.isfinite(float(item)) or float(item) < 0:
            raise RenderManifestError("%s must be a finite non-negative number" % child_name)
    return timings


def _normalise_cache(value: Any) -> Dict[str, Any]:
    cache = _require_mapping(value, "cache")
    status = cache.get("status", "unknown")
    if status not in SUPPORTED_CACHE_STATUSES:
        raise RenderManifestError(
            "cache.status must be one of: %s" % ", ".join(SUPPORTED_CACHE_STATUSES)
        )
    cache["status"] = status
    if "hit" in cache and not isinstance(cache["hit"], bool):
        raise RenderManifestError("cache.hit must be boolean when supplied")
    if "hit" in cache:
        expected_hit = status == "hit"
        if status in {"hit", "miss", "bypassed"} and cache["hit"] != expected_hit:
            raise RenderManifestError("cache.hit contradicts cache.status")
    if "key" in cache and cache["key"] is not None and not isinstance(cache["key"], str):
        raise RenderManifestError("cache.key must be a string when supplied")
    return cache


def _normalise_provider(value: Any) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RenderManifestError("provider must be a non-empty string when supplied")
    return value


def _normalise_decisions(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise RenderManifestError("safe_auto_decisions must be a sequence")
    decisions: List[Dict[str, Any]] = []
    for index, decision in enumerate(value):
        decisions.append(_require_mapping(decision, "safe_auto_decisions[%d]" % index))
    return decisions


def _known_payload_keys() -> Tuple[str, ...]:
    return (
        "contract",
        "schema_version",
        "render_revision",
        "settings_sha256",
        "mode",
        "status",
        "source",
        "output",
        "cache",
        "color_context",
        "metadata_result",
        "face_count",
        "timings_ms",
        "qa",
        "safe_auto_decisions",
        "backend",
        "provider",
        "hashes",
        "error",
        "extensions",
    )


@dataclass(frozen=True)
class RenderManifest:
    """Immutable, pixel-free render observation.

    ``source`` and ``output`` are artifact identity mappings. A completed
    render must provide output dimensions and dtype; failed/cancelled records
    may omit output while retaining the failure reason and inspector data.
    """

    render_revision: int
    settings_sha256: str
    mode: str
    source: Mapping[str, Any]
    output: Optional[Mapping[str, Any]] = None
    cache: Mapping[str, Any] = field(default_factory=lambda: {"status": "unknown"})
    color_context: Mapping[str, Any] = field(default_factory=dict)
    metadata_result: Mapping[str, Any] = field(default_factory=dict)
    face_count: Optional[int] = None
    timings_ms: Mapping[str, Any] = field(default_factory=dict)
    qa: Mapping[str, Any] = field(default_factory=dict)
    safe_auto_decisions: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    backend: Mapping[str, Any] = field(default_factory=dict)
    provider: Optional[str] = None
    hashes: Mapping[str, Optional[str]] = field(default_factory=dict)
    status: str = "completed"
    error: Optional[str] = None
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        revision = _require_non_negative_int(self.render_revision, "render_revision")
        settings_sha256 = _require_sha256(self.settings_sha256, "settings_sha256")
        mode = _normalise_mode(self.mode)
        status = self.status
        if status not in SUPPORTED_STATUSES:
            raise RenderManifestError(
                "status must be one of: %s" % ", ".join(SUPPORTED_STATUSES)
            )

        source = _normalise_artifact(self.source, "source", required=True)
        output = _normalise_artifact(
            self.output,
            "output",
            required=status == "completed",
        )
        if status == "completed" and output is not None:
            if output.get("dimensions") is None:
                raise RenderManifestError("completed output must include dimensions")
            if not isinstance(output.get("dtype"), str) or not output["dtype"].strip():
                raise RenderManifestError("completed output must include dtype")

        cache = _normalise_cache(self.cache)
        color_context = _require_mapping(self.color_context, "color_context")
        metadata_result = _require_mapping(self.metadata_result, "metadata_result")
        backend = _require_mapping(self.backend, "backend")
        qa = _require_mapping(self.qa, "qa")
        extensions = _require_mapping(self.extensions, "extensions")
        timings_ms = _validate_timing_values(self.timings_ms, "timings_ms")
        decisions = _normalise_decisions(self.safe_auto_decisions)
        face_count = _optional_non_negative_int(self.face_count, "face_count")
        provider = _normalise_provider(self.provider)

        hashes_input = _require_mapping(self.hashes, "hashes")
        hashes: Dict[str, Optional[str]] = {}
        for key, value in hashes_input.items():
            if not isinstance(key, str) or not key.strip():
                raise RenderManifestError("hashes keys must be non-empty strings")
            if value is not None:
                hashes[key] = _require_sha256(value, "hashes.%s" % key)
            else:
                hashes[key] = None
        if "settings_sha256" in hashes and hashes["settings_sha256"] != settings_sha256:
            raise RenderManifestError("hashes.settings_sha256 must match settings_sha256")
        hashes["settings_sha256"] = settings_sha256
        if source is not None and source.get("sha256") is not None:
            _match_artifact_hash(hashes, "source_sha256", source["sha256"])
        if output is not None and output.get("sha256") is not None:
            _match_artifact_hash(hashes, "output_sha256", output["sha256"])

        if self.error is not None and (
            not isinstance(self.error, str) or not self.error.strip()
        ):
            raise RenderManifestError("error must be a non-empty string when supplied")
        if status == "failed" and not self.error:
            raise RenderManifestError("failed manifests must include error")

        object.__setattr__(self, "render_revision", revision)
        object.__setattr__(self, "settings_sha256", settings_sha256)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "source", copy.deepcopy(source))
        object.__setattr__(self, "output", copy.deepcopy(output))
        object.__setattr__(self, "cache", copy.deepcopy(cache))
        object.__setattr__(self, "color_context", copy.deepcopy(color_context))
        object.__setattr__(self, "metadata_result", copy.deepcopy(metadata_result))
        object.__setattr__(self, "face_count", face_count)
        object.__setattr__(self, "timings_ms", copy.deepcopy(timings_ms))
        object.__setattr__(self, "qa", copy.deepcopy(qa))
        object.__setattr__(self, "safe_auto_decisions", tuple(copy.deepcopy(decisions)))
        object.__setattr__(self, "backend", copy.deepcopy(backend))
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "hashes", copy.deepcopy(hashes))
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "extensions", copy.deepcopy(extensions))

    @property
    def settings_hash(self) -> str:
        """Compatibility/readability alias for ``settings_sha256``."""

        return self.settings_sha256

    @property
    def cache_status(self) -> str:
        return str(self.cache.get("status", "unknown"))

    @property
    def manifest_sha256(self) -> str:
        """Digest of the canonical payload excluding this digest field."""

        return canonical_sha256(self._payload_without_manifest_hash())

    def _payload_without_manifest_hash(self) -> Dict[str, Any]:
        hashes = {
            key: value for key, value in self.hashes.items() if key != "manifest_sha256"
        }
        payload: Dict[str, Any] = {
            "contract": CONTRACT_NAME,
            "schema_version": SCHEMA_VERSION,
            "render_revision": self.render_revision,
            "settings_sha256": self.settings_sha256,
            "mode": self.mode,
            "status": self.status,
            "source": copy.deepcopy(self.source),
            "output": copy.deepcopy(self.output),
            "cache": copy.deepcopy(self.cache),
            "color_context": copy.deepcopy(self.color_context),
            "metadata_result": copy.deepcopy(self.metadata_result),
            "face_count": self.face_count,
            "timings_ms": copy.deepcopy(self.timings_ms),
            "qa": copy.deepcopy(self.qa),
            "safe_auto_decisions": copy.deepcopy(list(self.safe_auto_decisions)),
            "backend": copy.deepcopy(self.backend),
            "provider": self.provider,
            "hashes": hashes,
            "error": self.error,
            "extensions": copy.deepcopy(self.extensions),
        }
        return payload

    def to_dict(self) -> Dict[str, Any]:
        """Return a detached JSON-safe payload including ``manifest_sha256``."""

        payload = self._payload_without_manifest_hash()
        payload["hashes"]["manifest_sha256"] = canonical_sha256(payload)
        return copy.deepcopy(_json_safe(payload))

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        """Serialize the manifest as deterministic JSON."""

        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=indent,
        )

    def deepcopy_state(self) -> Dict[str, Any]:
        """Return a detached state payload suitable for a session inspector."""

        return copy.deepcopy(self.to_dict())

    to_state = deepcopy_state

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RenderManifest":
        """Validate and restore a manifest from a mapping."""

        if not isinstance(payload, Mapping):
            raise RenderManifestError("render manifest must be a mapping")
        safe = _json_safe(payload)
        unknown = sorted(set(safe) - set(_known_payload_keys()))
        if unknown:
            raise RenderManifestError("unknown manifest fields: %s" % ", ".join(unknown))
        if safe.get("contract") != CONTRACT_NAME:
            raise RenderManifestError("contract must be %s" % CONTRACT_NAME)
        if safe.get("schema_version") != SCHEMA_VERSION:
            raise RenderManifestError("unsupported schema_version")
        if "render_revision" not in safe or "settings_sha256" not in safe:
            raise RenderManifestError("render_revision and settings_sha256 are required")

        supplied_hashes = safe.get("hashes", {})
        if not isinstance(supplied_hashes, Mapping):
            raise RenderManifestError("hashes must be a mapping")
        supplied_manifest_hash = supplied_hashes.get("manifest_sha256")
        hashes_without_manifest = dict(supplied_hashes)
        hashes_without_manifest.pop("manifest_sha256", None)
        values = dict(safe)
        values["hashes"] = hashes_without_manifest
        manifest = cls(
            render_revision=values["render_revision"],
            settings_sha256=values["settings_sha256"],
            mode=values.get("mode"),
            source=values.get("source"),
            output=values.get("output"),
            cache=values.get("cache", {"status": "unknown"}),
            color_context=values.get("color_context", {}),
            metadata_result=values.get("metadata_result", {}),
            face_count=values.get("face_count"),
            timings_ms=values.get("timings_ms", {}),
            qa=values.get("qa", {}),
            safe_auto_decisions=values.get("safe_auto_decisions", []),
            backend=values.get("backend", {}),
            provider=values.get("provider"),
            hashes=values["hashes"],
            status=values.get("status", "completed"),
            error=values.get("error"),
            extensions=values.get("extensions", {}),
        )
        if supplied_manifest_hash is not None:
            expected = manifest.manifest_sha256
            supplied = _require_sha256(supplied_manifest_hash, "hashes.manifest_sha256")
            if supplied != expected:
                raise RenderManifestError("hashes.manifest_sha256 does not match payload")
        return manifest

    @classmethod
    def from_json(cls, payload: str) -> "RenderManifest":
        try:
            value = json.loads(payload)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RenderManifestError("invalid render manifest JSON") from exc
        return cls.from_dict(value)

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "RenderManifest":
        return cls.from_dict(payload)

    def validate(self) -> Dict[str, Any]:
        """Return a non-throwing inspector validation report."""

        return validate_render_manifest(self.to_dict())


def _match_artifact_hash(
    hashes: Dict[str, Optional[str]], key: str, expected: str
) -> None:
    current = hashes.get(key)
    if current is not None and current != expected:
        raise RenderManifestError("hashes.%s does not match artifact sha256" % key)
    hashes[key] = expected


def validate_render_manifest(payload: Any) -> Dict[str, Any]:
    """Validate a manifest payload without raising.

    This is intentionally a structural inspector check, not a quality gate.
    It does not certify pixels, approve a recipe, or replace the separate
    certification evidence contract.
    """

    try:
        manifest = RenderManifest.from_dict(payload)
    except RenderManifestError as exc:
        schema_version = payload.get("schema_version") if isinstance(payload, Mapping) else None
        return {
            "valid": False,
            "contract": CONTRACT_NAME,
            "schema_version": schema_version,
            "errors": [str(exc)],
            "warnings": [],
        }

    return {
        "valid": True,
        "contract": CONTRACT_NAME,
        "schema_version": SCHEMA_VERSION,
        "errors": [],
        "warnings": [],
        "manifest_sha256": manifest.manifest_sha256,
    }


def build_render_manifest(**values: Any) -> RenderManifest:
    """Convenience factory kept independent from GUI and certification code."""

    return RenderManifest(**values)


__all__ = [
    "CONTRACT_NAME",
    "SCHEMA_VERSION",
    "SUPPORTED_CACHE_STATUSES",
    "SUPPORTED_MODES",
    "SUPPORTED_STATUSES",
    "RenderManifest",
    "RenderManifestError",
    "build_render_manifest",
    "canonical_json",
    "canonical_sha256",
    "validate_render_manifest",
]
