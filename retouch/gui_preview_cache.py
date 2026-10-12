"""Small, session-scoped cache primitives for GUI preview work.

The GUI can reuse decoded-source metadata and face analysis between preview
renders, but a Gradio ``State`` value must remain cheap and deepcopy-safe.
This module deliberately stores *references* to image artifacts rather than
image arrays.  Arrays and other binary values are represented by bounded
content references (shape, dtype, size, and a digest when it can be obtained).

The cache is intentionally independent of Gradio and the processing engine.
One :class:`GuiPreviewCache` should be created per browser/session.  Its key
contains the source identity and every input that can make the decoded source
or face context invalid: orientation, geometry, optical correction, and
render resolution.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


CACHE_SCHEMA_VERSION = 1
DEFAULT_MAX_ENTRIES = 4
DEFAULT_MAX_FACE_CONTEXTS = 32
_MAX_COMPACT_DEPTH = 6
_MAX_COMPACT_ITEMS = 64
_MAX_COMPACT_STRING = 2048
_MISSING = object()


def _canonical_json(value: Any) -> str:
    """Return stable JSON for already compact, JSON-compatible values."""

    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _qualified_name(value: Any) -> str:
    cls = type(value)
    return "%s.%s" % (cls.__module__, cls.__qualname__)


def _array_reference(value: Any) -> Dict[str, Any]:
    """Describe an array-like value without retaining its pixels.

    ``memoryview`` hashes contiguous NumPy arrays without making a second
    full-size byte copy.  For unusual/non-contiguous array implementations we
    prefer an explicit missing digest over calling a method that might create
    a very large temporary buffer.
    """

    try:
        shape = [int(item) for item in tuple(getattr(value, "shape"))]
    except Exception:
        shape = []
    try:
        dtype = str(getattr(value, "dtype"))
    except Exception:
        dtype = "unknown"
    try:
        nbytes = max(0, int(getattr(value, "nbytes")))
    except Exception:
        nbytes = None

    digest: Optional[str] = None
    try:
        view = memoryview(value)
        if view.c_contiguous:
            digest = hashlib.sha256(view.cast("B")).hexdigest()
    except Exception:
        # A reference without a digest is still safer than storing the array.
        digest = None

    result: Dict[str, Any] = {
        "__type__": "array_ref",
        "shape": shape,
        "dtype": dtype,
        "nbytes": nbytes,
    }
    if digest is not None:
        result["sha256"] = digest
    return result


def _compact_value(value: Any, *, depth: int = 0) -> Any:
    """Convert values to bounded, deepcopy-safe JSON-compatible data.

    Face contexts contain dataclasses, protobuf-adjacent objects, and often a
    ``face_image`` array.  The cache keeps useful scalar/landmark metadata but
    turns arrays, bytes, and oversized strings into content references.  The
    limits are intentionally conservative because this result is suitable for
    a session state object.
    """

    if depth > _MAX_COMPACT_DEPTH:
        return {"__truncated__": "max_depth"}

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            return value
        return {"__type__": "non_finite_float", "value": repr(value)}
    if isinstance(value, str):
        if len(value) <= _MAX_COMPACT_STRING:
            return value
        return {
            "__type__": "string_ref",
            "sha256": _sha256_bytes(value.encode("utf-8")),
            "size": len(value),
            "preview": value[:256],
        }
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            view = memoryview(value)
            size = view.nbytes
            digest = hashlib.sha256(view.cast("B")).hexdigest()
        except Exception:
            size = None
            digest = None
        return {"__type__": "bytes_ref", "sha256": digest, "size": size}
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if hasattr(value, "shape") and hasattr(value, "dtype") and hasattr(value, "nbytes"):
        return _array_reference(value)

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        result: Dict[str, Any] = {"__type__": _qualified_name(value)}
        for field in dataclasses.fields(value):
            try:
                field_value = getattr(value, field.name)
            except Exception:
                continue
            result[field.name] = _compact_value(field_value, depth=depth + 1)
        return result

    if isinstance(value, Mapping):
        result = {}
        items = sorted(value.items(), key=lambda item: str(item[0]))
        for key, item_value in items[:_MAX_COMPACT_ITEMS]:
            result[str(key)] = _compact_value(item_value, depth=depth + 1)
        if len(items) > _MAX_COMPACT_ITEMS:
            result["__truncated_items__"] = len(items) - _MAX_COMPACT_ITEMS
        return result

    if isinstance(value, (list, tuple)):
        result = [
            _compact_value(item, depth=depth + 1)
            for item in value[:_MAX_COMPACT_ITEMS]
        ]
        if len(value) > _MAX_COMPACT_ITEMS:
            result.append({"__truncated_items__": len(value) - _MAX_COMPACT_ITEMS})
        return result

    if isinstance(value, (set, frozenset)):
        compacted = [_compact_value(item, depth=depth + 1) for item in value]
        compacted.sort(key=_canonical_json)
        if len(compacted) > _MAX_COMPACT_ITEMS:
            compacted = compacted[:_MAX_COMPACT_ITEMS]
            compacted.append({"__truncated_items__": len(value) - _MAX_COMPACT_ITEMS})
        return compacted

    # Do not walk arbitrary objects through __dict__: model objects can own
    # native handles or full-resolution arrays.  Their type is useful evidence
    # but the object itself must never enter a Gradio State value.
    return {"__type__": _qualified_name(value), "__repr__": repr(value)[:256]}


def _stable_token(value: Any) -> str:
    compacted = _compact_value(value)
    return _sha256_bytes(_canonical_json(compacted).encode("utf-8"))


def _validate_sha256(value: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("content_sha256 must be a 64-character hexadecimal digest")
    return normalized


@dataclass(frozen=True)
class SourceIdentity:
    """Filesystem identity used to prevent stale preview reuse."""

    path: str
    mtime_ns: int
    size_bytes: int
    content_sha256: str

    def __post_init__(self) -> None:
        path_value = os.fsdecode(os.fspath(self.path))
        object.__setattr__(self, "path", os.path.realpath(os.path.abspath(path_value)))
        object.__setattr__(self, "mtime_ns", int(self.mtime_ns))
        object.__setattr__(self, "size_bytes", int(self.size_bytes))
        object.__setattr__(self, "content_sha256", _validate_sha256(self.content_sha256))

    @classmethod
    def from_path(
        cls,
        path: Union[str, os.PathLike],
        *,
        content_sha256: Optional[str] = None,
    ) -> "SourceIdentity":
        canonical_path = os.path.realpath(os.path.abspath(os.fsdecode(os.fspath(path))))
        stat_result = os.stat(canonical_path)
        digest = content_sha256 or _sha256_file(canonical_path)
        return cls(
            path=canonical_path,
            mtime_ns=int(stat_result.st_mtime_ns),
            size_bytes=int(stat_result.st_size),
            content_sha256=digest,
        )

    @property
    def identity_key(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict()).encode("utf-8"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "mtime_ns": self.mtime_ns,
            "size_bytes": self.size_bytes,
            "content_sha256": self.content_sha256,
        }


def source_identity(
    path: Union[str, os.PathLike], *, content_sha256: Optional[str] = None
) -> SourceIdentity:
    """Build a source identity from the current filesystem state."""

    return SourceIdentity.from_path(path, content_sha256=content_sha256)


def render_resolution_key(
    resolution: Any,
    height: Optional[int] = None,
) -> str:
    """Normalize common resolution inputs to a deterministic cache key."""

    if height is not None:
        width_value = resolution
        height_value = height
        try:
            return "%dx%d" % (int(width_value), int(height_value))
        except (TypeError, ValueError):
            raise ValueError("render resolution dimensions must be integers")

    if isinstance(resolution, Mapping):
        if "width" not in resolution or "height" not in resolution:
            raise ValueError("resolution mapping requires width and height")
        return render_resolution_key(resolution["width"], int(resolution["height"]))

    if isinstance(resolution, Sequence) and not isinstance(resolution, (str, bytes)):
        if len(resolution) != 2:
            raise ValueError("resolution sequence must contain width and height")
        return render_resolution_key(resolution[0], int(resolution[1]))

    if isinstance(resolution, str):
        normalized = "".join(resolution.strip().lower().split())
        if not normalized:
            raise ValueError("render resolution key cannot be empty")
        return normalized

    raise ValueError("resolution must be a width/height pair or a non-empty key")


@dataclass(frozen=True)
class PreviewCacheKey:
    """All inputs that identify one decoded-source/face-context variant."""

    source_identity: SourceIdentity
    render_resolution_key: str
    orientation_key: str
    geometry_key: str
    optical_correction_key: str
    color_contract_key: str = ""
    bit_depth_key: str = ""
    raw_settings_key: str = ""
    detector_backend_key: str = ""
    engine_version_key: str = ""

    @classmethod
    def create(
        cls,
        source: SourceIdentity,
        resolution: Any,
        *,
        orientation: Any = 1,
        geometry: Any = None,
        optical_correction: Any = False,
        color_contract: Any = None,
        bit_depth: Any = None,
        raw_settings: Any = None,
        detector_backend: Any = None,
        engine_version: Any = None,
    ) -> "PreviewCacheKey":
        if not isinstance(source, SourceIdentity):
            raise TypeError("source must be a SourceIdentity")
        return cls(
            source_identity=source,
            render_resolution_key=render_resolution_key(resolution),
            orientation_key=_stable_token(orientation),
            geometry_key=_stable_token(geometry),
            optical_correction_key=_stable_token(optical_correction),
            color_contract_key=_stable_token(color_contract),
            bit_depth_key=_stable_token(bit_depth),
            raw_settings_key=_stable_token(raw_settings),
            detector_backend_key=_stable_token(detector_backend),
            engine_version_key=_stable_token(engine_version),
        )

    @property
    def digest(self) -> str:
        return _sha256_bytes(_canonical_json(self.to_dict(include_digest=False)).encode("utf-8"))

    @property
    def key(self) -> str:
        return self.digest

    def changed_fields(self, other: "PreviewCacheKey") -> Tuple[str, ...]:
        if not isinstance(other, PreviewCacheKey):
            raise TypeError("other must be a PreviewCacheKey")
        changed = []
        if self.source_identity != other.source_identity:
            changed.append("source")
        if self.render_resolution_key != other.render_resolution_key:
            changed.append("resolution")
        if self.orientation_key != other.orientation_key:
            changed.append("orientation")
        if self.geometry_key != other.geometry_key:
            changed.append("geometry")
        if self.optical_correction_key != other.optical_correction_key:
            changed.append("optical_correction")
        if self.color_contract_key != other.color_contract_key:
            changed.append("color_contract")
        if self.bit_depth_key != other.bit_depth_key:
            changed.append("bit_depth")
        if self.raw_settings_key != other.raw_settings_key:
            changed.append("raw_settings")
        if self.detector_backend_key != other.detector_backend_key:
            changed.append("detector_backend")
        if self.engine_version_key != other.engine_version_key:
            changed.append("engine_version")
        return tuple(changed)

    def to_dict(self, *, include_digest: bool = True) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "source_identity": self.source_identity.to_dict(),
            "render_resolution_key": self.render_resolution_key,
            "orientation_key": self.orientation_key,
            "geometry_key": self.geometry_key,
            "optical_correction_key": self.optical_correction_key,
            "color_contract_key": self.color_contract_key,
            "bit_depth_key": self.bit_depth_key,
            "raw_settings_key": self.raw_settings_key,
            "detector_backend_key": self.detector_backend_key,
            "engine_version_key": self.engine_version_key,
        }
        if include_digest:
            result["digest"] = self.digest
        return result


def make_preview_cache_key(
    source: SourceIdentity,
    resolution: Any,
    *,
    orientation: Any = 1,
    geometry: Any = None,
    optical_correction: Any = False,
    color_contract: Any = None,
    bit_depth: Any = None,
    raw_settings: Any = None,
    detector_backend: Any = None,
    engine_version: Any = None,
) -> PreviewCacheKey:
    return PreviewCacheKey.create(
        source,
        resolution,
        orientation=orientation,
        geometry=geometry,
        optical_correction=optical_correction,
        color_contract=color_contract,
        bit_depth=bit_depth,
        raw_settings=raw_settings,
        detector_backend=detector_backend,
        engine_version=engine_version,
    )


def _compact_mapping(value: Any) -> Dict[str, Any]:
    compacted = _compact_value(value)
    if isinstance(compacted, dict):
        return compacted
    return {"value": compacted}


def _compact_reference(value: Any, kind: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    compacted = _compact_mapping(value)
    compacted.setdefault("kind", kind)
    return compacted


def _normalize_face_contexts(
    values: Optional[Iterable[Any]],
    max_count: int,
) -> Tuple[Tuple[Any, ...], bool]:
    if values is None:
        items: List[Any] = []
    elif isinstance(values, Mapping) or isinstance(values, (str, bytes)):
        items = [values]
    else:
        try:
            items = list(values)
        except TypeError:
            items = [values]
    truncated = len(items) > max_count
    compacted = tuple(_compact_value(item) for item in items[:max_count])
    return compacted, truncated


@dataclass(frozen=True)
class PreviewCacheEntry:
    """Compact, immutable-in-practice value returned from the cache."""

    cache_key: str
    source_identity: SourceIdentity
    decoded_source_metadata: Dict[str, Any]
    render_resolution_key: str
    orientation_key: str
    geometry_key: str
    optical_correction_key: str
    face_contexts: Tuple[Any, ...]
    color_context_metadata: Dict[str, Any] = field(default_factory=dict)
    color_contract_key: str = ""
    bit_depth_key: str = ""
    raw_settings_key: str = ""
    detector_backend_key: str = ""
    engine_version_key: str = ""
    face_contexts_truncated: bool = False
    preview_reference: Optional[Dict[str, Any]] = None
    checkpoint_reference: Optional[Dict[str, Any]] = None
    # Runtime-only values are deliberately excluded from ``to_dict``.  They
    # let the GUI reuse a live decoded source and engine FaceContext objects
    # while the serializable State representation remains compact.
    decoded_source: Any = None
    runtime_face_contexts: Tuple[Any, ...] = ()
    runtime_color_context: Any = None

    @property
    def key(self) -> str:
        return self.cache_key

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "source_identity": self.source_identity.to_dict(),
            "decoded_source_metadata": copy.deepcopy(self.decoded_source_metadata),
            "render_resolution_key": self.render_resolution_key,
            "orientation_key": self.orientation_key,
            "geometry_key": self.geometry_key,
            "optical_correction_key": self.optical_correction_key,
            "color_contract_key": self.color_contract_key,
            "bit_depth_key": self.bit_depth_key,
            "raw_settings_key": self.raw_settings_key,
            "detector_backend_key": self.detector_backend_key,
            "engine_version_key": self.engine_version_key,
            "face_contexts": copy.deepcopy(list(self.face_contexts)),
            "face_contexts_truncated": self.face_contexts_truncated,
            "color_context": copy.deepcopy(self.color_context_metadata),
            "preview_reference": copy.deepcopy(self.preview_reference),
            "checkpoint_reference": copy.deepcopy(self.checkpoint_reference),
            "decoded_source": _compact_value(self.decoded_source),
            "runtime_face_contexts_available": bool(self.runtime_face_contexts),
        }


class GuiPreviewCache:
    """Bounded cache intended to live inside one GUI session.

    There is no module-global singleton and no lock/native handle in this
    object.  Consequently ``copy.deepcopy(cache)`` is safe for Gradio State.
    Values returned by ``get`` and ``put`` are deep copies, so callers cannot
    mutate the session cache through a previously returned entry.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        max_face_contexts: int = DEFAULT_MAX_FACE_CONTEXTS,
    ) -> None:
        if int(max_entries) < 1:
            raise ValueError("max_entries must be at least 1")
        if int(max_face_contexts) < 1:
            raise ValueError("max_face_contexts must be at least 1")
        self.max_entries = int(max_entries)
        self.max_face_contexts = int(max_face_contexts)
        self._entries: Dict[str, PreviewCacheEntry] = {}
        self._access_order: List[str] = []
        self._latest_render: Dict[str, Any] = {}
        self._latest_render_face_contexts: Tuple[Any, ...] = ()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def size(self) -> int:
        return len(self)

    @staticmethod
    def _digest_for(key: Union[PreviewCacheKey, str]) -> str:
        if isinstance(key, PreviewCacheKey):
            return key.digest
        if isinstance(key, str) and key:
            return key
        raise TypeError("cache key must be a PreviewCacheKey or non-empty digest")

    def get(self, key: Union[PreviewCacheKey, str]) -> Optional[PreviewCacheEntry]:
        digest = self._digest_for(key)
        entry = self._entries.get(digest)
        if entry is None:
            return None
        if digest in self._access_order:
            self._access_order.remove(digest)
        self._access_order.append(digest)
        return copy.deepcopy(entry)

    def put(
        self,
        key: PreviewCacheKey,
        *,
        decoded_source_metadata: Optional[Mapping[str, Any]] = None,
        face_contexts: Optional[Iterable[Any]] = None,
        preview_reference: Any = None,
        checkpoint_reference: Any = None,
        decoded_source: Any = None,
        runtime_face_contexts: Optional[Iterable[Any]] = None,
        color_context: Any = None,
        runtime_color_context: Any = None,
    ) -> PreviewCacheEntry:
        if not isinstance(key, PreviewCacheKey):
            raise TypeError("put requires a PreviewCacheKey")
        contexts, truncated = _normalize_face_contexts(
            face_contexts,
            self.max_face_contexts,
        )
        entry = PreviewCacheEntry(
            cache_key=key.digest,
            source_identity=key.source_identity,
            decoded_source_metadata=_compact_mapping(decoded_source_metadata or {}),
            render_resolution_key=key.render_resolution_key,
            orientation_key=key.orientation_key,
            geometry_key=key.geometry_key,
            optical_correction_key=key.optical_correction_key,
            color_contract_key=key.color_contract_key,
            bit_depth_key=key.bit_depth_key,
            raw_settings_key=key.raw_settings_key,
            detector_backend_key=key.detector_backend_key,
            engine_version_key=key.engine_version_key,
            face_contexts=contexts,
            face_contexts_truncated=truncated,
            color_context_metadata=_compact_mapping(
                color_context.to_dict()
                if hasattr(color_context, "to_dict")
                else (color_context or {})
            ),
            preview_reference=_compact_reference(preview_reference, "preview"),
            checkpoint_reference=_compact_reference(checkpoint_reference, "checkpoint"),
            decoded_source=copy.deepcopy(decoded_source),
            runtime_face_contexts=tuple(
                copy.deepcopy(list(runtime_face_contexts or ()))[: self.max_face_contexts]
            ),
            runtime_color_context=copy.deepcopy(runtime_color_context),
        )
        digest = key.digest
        self._entries[digest] = entry
        if digest in self._access_order:
            self._access_order.remove(digest)
        self._access_order.append(digest)
        while len(self._access_order) > self.max_entries:
            evicted = self._access_order.pop(0)
            self._entries.pop(evicted, None)
        return copy.deepcopy(entry)

    def invalidate(
        self,
        key: Optional[Union[PreviewCacheKey, str]] = None,
        *,
        source_identity: Optional[SourceIdentity] = None,
        source_path: Optional[Union[str, os.PathLike]] = None,
        resolution: Any = _MISSING,
        orientation: Any = _MISSING,
        geometry: Any = _MISSING,
        optical_correction: Any = _MISSING,
    ) -> int:
        """Remove matching entries and return the number removed.

        Calling without filters clears the session cache.  Dimension filters
        are intentionally explicit; callers can invalidate only the old
        orientation, geometry, optical-correction, or resolution variant.
        """

        exact_digest = self._digest_for(key) if key is not None else None
        normalized_resolution = (
            render_resolution_key(resolution) if resolution is not _MISSING else _MISSING
        )
        normalized_orientation = (
            _stable_token(orientation) if orientation is not _MISSING else _MISSING
        )
        normalized_geometry = (
            _stable_token(geometry) if geometry is not _MISSING else _MISSING
        )
        normalized_optical = (
            _stable_token(optical_correction)
            if optical_correction is not _MISSING
            else _MISSING
        )
        canonical_source_path = (
            os.path.realpath(os.path.abspath(os.fspath(source_path)))
            if source_path is not None
            else None
        )

        if (
            exact_digest is None
            and source_identity is None
            and canonical_source_path is None
            and normalized_resolution is _MISSING
            and normalized_orientation is _MISSING
            and normalized_geometry is _MISSING
            and normalized_optical is _MISSING
        ):
            removed = len(self._entries)
            self._entries.clear()
            self._access_order.clear()
            return removed

        removed = 0
        for digest in list(self._access_order):
            entry = self._entries.get(digest)
            if entry is None:
                continue
            if exact_digest is not None and digest != exact_digest:
                continue
            if source_identity is not None and entry.source_identity != source_identity:
                continue
            if canonical_source_path is not None and entry.source_identity.path != canonical_source_path:
                continue
            if normalized_resolution is not _MISSING and entry.render_resolution_key != normalized_resolution:
                continue
            if normalized_orientation is not _MISSING and entry.orientation_key != normalized_orientation:
                continue
            if normalized_geometry is not _MISSING and entry.geometry_key != normalized_geometry:
                continue
            if normalized_optical is not _MISSING and entry.optical_correction_key != normalized_optical:
                continue
            self._entries.pop(digest, None)
            self._access_order.remove(digest)
            removed += 1
        return removed

    def invalidate_for_change(
        self,
        previous: PreviewCacheKey,
        current: PreviewCacheKey,
    ) -> int:
        """Invalidate the old variants affected by a changed render input."""

        changed = previous.changed_fields(current)
        if not changed:
            return 0

        # Source changes can make every old source variant unsafe.  For the
        # other dimensions, invalidate all variants at or below the changed
        # stage while retaining unrelated source entries.
        if "source" in changed:
            return self.invalidate(source_identity=previous.source_identity)
        if "orientation" in changed:
            return self._invalidate_key_fields(previous)
        if "geometry" in changed:
            return self._invalidate_key_fields(
                previous,
                include_resolution=False,
                include_geometry=True,
            )
        if "optical_correction" in changed:
            return self._invalidate_key_fields(
                previous,
                include_resolution=False,
                include_geometry=True,
                include_optical=True,
            )
        return self.invalidate(key=previous)

    def _invalidate_key_fields(
        self,
        key: PreviewCacheKey,
        *,
        include_resolution: bool = False,
        include_geometry: bool = False,
        include_optical: bool = False,
    ) -> int:
        removed = 0
        for digest in list(self._access_order):
            entry = self._entries.get(digest)
            if entry is None or entry.source_identity != key.source_identity:
                continue
            if entry.orientation_key != key.orientation_key:
                continue
            if include_geometry and entry.geometry_key != key.geometry_key:
                continue
            if include_optical and entry.optical_correction_key != key.optical_correction_key:
                continue
            if include_resolution and entry.render_resolution_key != key.render_resolution_key:
                continue
            self._entries.pop(digest, None)
            self._access_order.remove(digest)
            removed += 1
        return removed

    def clear(self) -> int:
        removed = self.invalidate()
        self._latest_render = {}
        self._latest_render_face_contexts = ()
        return removed

    def set_latest_render(
        self,
        evidence: Optional[Mapping[str, Any]],
        *,
        face_contexts: Optional[Iterable[Any]] = None,
    ) -> None:
        """Store compact, pixel-free evidence for the latest visible render.

        The GUI's native inspection view needs face boxes and the render
        revision, but those values should not be reconstructed from image
        pixels or retained in a separate global.  Live face contexts remain
        runtime-only; the compact evidence is safe for Gradio State.
        """
        self._latest_render = _compact_mapping(evidence or {})
        contexts, _ = _normalize_face_contexts(face_contexts, self.max_face_contexts)
        self._latest_render_face_contexts = tuple(copy.deepcopy(contexts))

    @property
    def latest_render_evidence(self) -> Dict[str, Any]:
        """Return compact evidence for the latest visible render."""
        return copy.deepcopy(self._latest_render)

    @property
    def latest_render_face_contexts(self) -> Tuple[Any, ...]:
        """Return runtime/compact face contexts for native inspection."""
        return copy.deepcopy(self._latest_render_face_contexts)

    def has_runtime_entry(self, key: Union[PreviewCacheKey, str]) -> bool:
        """Return whether a key has reusable in-process image/context values."""
        digest = self._digest_for(key)
        entry = self._entries.get(digest)
        return bool(entry is not None and (entry.decoded_source is not None or entry.runtime_face_contexts))

    def to_state(self) -> Dict[str, Any]:
        """Return a compact JSON/deepcopy-safe session-state snapshot."""

        entries = [self._entries[digest].to_dict() for digest in self._access_order]
        return {
            "schema_version": CACHE_SCHEMA_VERSION,
            "max_entries": self.max_entries,
            "max_face_contexts": self.max_face_contexts,
            "entries": entries,
            "latest_render": copy.deepcopy(self._latest_render),
            "latest_render_face_contexts": copy.deepcopy(
                list(self._latest_render_face_contexts)
            ),
        }

    def __deepcopy__(self, memo: Dict[int, Any]) -> "GuiPreviewCache":
        clone = type(self)(
            max_entries=self.max_entries,
            max_face_contexts=self.max_face_contexts,
        )
        memo[id(self)] = clone
        clone._entries = copy.deepcopy(self._entries, memo)
        clone._access_order = copy.deepcopy(self._access_order, memo)
        clone._latest_render = copy.deepcopy(self._latest_render, memo)
        clone._latest_render_face_contexts = copy.deepcopy(
            self._latest_render_face_contexts, memo
        )
        return clone

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "GuiPreviewCache":
        """Restore compact evidence state without inventing live runtime values.

        A browser/session round-trip must retain the cache's diagnostic
        entries and LRU order, even though decoded arrays and live face
        contexts cannot be reconstructed from a deepcopy-safe State value.
        Restored entries therefore remain evidence-only cache entries and
        ``has_runtime_entry`` correctly stays false until the GUI repopulates
        the in-process values.
        """
        if not isinstance(state, Mapping):
            raise TypeError("preview cache state must be a mapping")
        cache = cls(
            max_entries=int(state.get("max_entries", DEFAULT_MAX_ENTRIES)),
            max_face_contexts=int(state.get("max_face_contexts", DEFAULT_MAX_FACE_CONTEXTS)),
        )

        raw_entries = state.get("entries", ())
        if isinstance(raw_entries, (str, bytes)) or not isinstance(raw_entries, Sequence):
            raise TypeError("preview cache entries must be a sequence")

        for raw_entry in raw_entries:
            if not isinstance(raw_entry, Mapping):
                continue
            source_data = raw_entry.get("source_identity")
            if not isinstance(source_data, Mapping):
                continue
            cache_key = str(raw_entry.get("cache_key", ""))
            if not cache_key or cache_key in cache._entries:
                continue
            try:
                identity = SourceIdentity(
                    path=source_data["path"],
                    mtime_ns=source_data["mtime_ns"],
                    size_bytes=source_data["size_bytes"],
                    content_sha256=source_data["content_sha256"],
                )
            except (KeyError, TypeError, ValueError, OSError):
                # A stale/corrupt session state should not prevent the GUI
                # from starting; the invalid entry is simply discarded.
                continue

            raw_contexts = raw_entry.get("face_contexts", ())
            if isinstance(raw_contexts, (str, bytes)):
                raw_contexts = (raw_contexts,)
            elif not isinstance(raw_contexts, Sequence):
                raw_contexts = (raw_contexts,)
            context_items = list(raw_contexts)
            truncated = bool(raw_entry.get("face_contexts_truncated", False))
            if len(context_items) > cache.max_face_contexts:
                context_items = context_items[: cache.max_face_contexts]
                truncated = True

            entry = PreviewCacheEntry(
                cache_key=cache_key,
                source_identity=identity,
                decoded_source_metadata=_compact_mapping(
                    raw_entry.get("decoded_source_metadata", {})
                ),
                render_resolution_key=str(raw_entry.get("render_resolution_key", "")),
                orientation_key=str(raw_entry.get("orientation_key", "")),
                geometry_key=str(raw_entry.get("geometry_key", "")),
                optical_correction_key=str(
                    raw_entry.get("optical_correction_key", "")
                ),
                color_contract_key=str(raw_entry.get("color_contract_key", "")),
                bit_depth_key=str(raw_entry.get("bit_depth_key", "")),
                raw_settings_key=str(raw_entry.get("raw_settings_key", "")),
                detector_backend_key=str(
                    raw_entry.get("detector_backend_key", "")
                ),
                engine_version_key=str(raw_entry.get("engine_version_key", "")),
                face_contexts=tuple(copy.deepcopy(context_items)),
                color_context_metadata=_compact_mapping(
                    raw_entry.get("color_context", {})
                ),
                face_contexts_truncated=truncated,
                preview_reference=_compact_reference(
                    raw_entry.get("preview_reference"), "preview"
                ),
                checkpoint_reference=_compact_reference(
                    raw_entry.get("checkpoint_reference"), "checkpoint"
                ),
                # Runtime values are intentionally unavailable after a JSON
                # or Gradio State round-trip.
                decoded_source=None,
                runtime_face_contexts=(),
                runtime_color_context=None,
            )
            cache._entries[cache_key] = entry
            cache._access_order.append(cache_key)

        while len(cache._access_order) > cache.max_entries:
            evicted = cache._access_order.pop(0)
            cache._entries.pop(evicted, None)
        cache._latest_render = _compact_mapping(state.get("latest_render", {}))
        raw_latest_contexts = state.get("latest_render_face_contexts", ())
        if isinstance(raw_latest_contexts, (str, bytes)):
            raw_latest_contexts = (raw_latest_contexts,)
        elif not isinstance(raw_latest_contexts, Sequence):
            raw_latest_contexts = (raw_latest_contexts,)
        cache._latest_render_face_contexts = tuple(
            copy.deepcopy(list(raw_latest_contexts)[: cache.max_face_contexts])
        )
        return cache


# Small functional helpers make the contract easy to use from callbacks while
# keeping the cache itself a session-owned object.
def get_preview(
    cache: GuiPreviewCache,
    key: Union[PreviewCacheKey, str],
) -> Optional[PreviewCacheEntry]:
    return cache.get(key)


def put_preview(
    cache: GuiPreviewCache,
    key: PreviewCacheKey,
    **kwargs: Any,
) -> PreviewCacheEntry:
    return cache.put(key, **kwargs)


def invalidate_preview(cache: GuiPreviewCache, *args: Any, **kwargs: Any) -> int:
    return cache.invalidate(*args, **kwargs)


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MAX_FACE_CONTEXTS",
    "GuiPreviewCache",
    "PreviewCacheEntry",
    "PreviewCacheKey",
    "SourceIdentity",
    "get_preview",
    "invalidate_preview",
    "make_preview_cache_key",
    "put_preview",
    "render_resolution_key",
    "source_identity",
]
