"""Compact history primitives for Advanced Retouch.

The Advanced Retouch canvas used to keep a copy of the complete RGB image for
every undo step.  That is convenient for a small demo, but it is an unsafe
default for a large photograph: a single 6240 x 4160 RGB frame is about 74 MiB
before Python and array overhead are counted.

This module deliberately does not know about the GUI or the image engine.  It
keeps a bounded log of JSON-safe edit descriptions and, when the caller
supplies one, an occasional *preview*-resolution checkpoint.  A checkpoint is
not a full-resolution export and is never treated as one.  The caller can use
``ReplayPlan`` with its source/current image and an engine-specific edit
function when it needs to render a state.

The public state is suitable for a Gradio ``State`` value or a session JSON
file.  In particular, ``to_state`` contains no NumPy arrays, PIL images, or
other live objects.  Array-like values in an edit description are represented
by a small descriptor rather than copied into history.  Preview arrays may be
stored in a capped compressed representation when they are small enough.

The module is stdlib-only so that importing the history state does not force
the optional image stack to initialize.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import zlib


SCHEMA_VERSION = 1
DEFAULT_MAX_ENTRIES = 100
DEFAULT_MEMORY_BUDGET_BYTES = 4 * 1024 * 1024
DEFAULT_CHECKPOINT_INTERVAL = 5
DEFAULT_PREVIEW_MAX_BYTES = 128 * 1024
DEFAULT_MAX_CHECKPOINTS = 16

STATUS_APPLIED = "applied"
STATUS_SKIP = "skip"
STATUS_REJECTED = "rejected"


def _digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_float(value: float) -> Any:
    """Return a strict-JSON representation of a float."""

    if math.isfinite(value):
        return value
    return {"__type__": "non-finite-float", "value": repr(value)}


def _array_descriptor(value: Any) -> Optional[Dict[str, Any]]:
    """Describe an array-like value without retaining its elements.

    NumPy is intentionally not imported.  The small protocol below also
    works for array objects supplied by test doubles or other image libraries.
    """

    shape = getattr(value, "shape", None)
    tobytes = getattr(value, "tobytes", None)
    if shape is None or not callable(tobytes):
        return None

    try:
        raw = bytes(tobytes())
    except Exception:
        raw = b""
    try:
        shape_value = [int(part) for part in shape]
    except Exception:
        shape_value = [str(shape)]
    dtype = str(getattr(value, "dtype", type(value).__name__))
    return {
        "__type__": "array-descriptor",
        "shape": shape_value,
        "dtype": dtype,
        "bytes": len(raw),
        "sha256": _digest_bytes(raw),
    }


def _json_safe(value: Any, *, depth: int = 0, seen: Optional[set] = None) -> Any:
    """Make a recursive, detached, strict-JSON-safe copy of ``value``.

    The function is intentionally conservative for unknown objects.  Edit
    records should describe an operation, not retain the object or image that
    happened to produce it.  Array-like values therefore become descriptors.
    """

    if depth > 32:
        return {"__type__": "depth-limit"}

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        return {
            "__type__": "bytes",
            "bytes": len(raw),
            "sha256": _digest_bytes(raw),
        }
    if isinstance(value, Path):
        return os.fspath(value)

    array_value = _array_descriptor(value)
    if array_value is not None:
        return array_value

    if seen is None:
        seen = set()
    container = isinstance(value, (Mapping, list, tuple, set, frozenset)) or dataclasses.is_dataclass(value)
    value_id = id(value)
    if container:
        if value_id in seen:
            return {"__type__": "cycle"}
        seen.add(value_id)

    if dataclasses.is_dataclass(value):
        result = {
            str(item.name): _json_safe(getattr(value, item.name), depth=depth + 1, seen=seen)
            for item in dataclasses.fields(value)
        }
        seen.discard(value_id)
        return result

    if isinstance(value, Mapping):
        result = {
            str(key): _json_safe(item, depth=depth + 1, seen=seen)
            for key, item in value.items()
        }
        seen.discard(value_id)
        return result

    if isinstance(value, (list, tuple)):
        result = [_json_safe(item, depth=depth + 1, seen=seen) for item in value]
        seen.discard(value_id)
        return result

    if isinstance(value, (set, frozenset)):
        result = [
            _json_safe(item, depth=depth + 1, seen=seen)
            for item in sorted(value, key=lambda item: repr(item))
        ]
        seen.discard(value_id)
        return result

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _json_safe(to_dict(), depth=depth + 1, seen=seen)
        except Exception:
            pass

    return {"__type__": "repr", "class": type(value).__name__, "value": repr(value)}


def json_safe_copy(value: Any) -> Any:
    """Return a detached value that can be passed to ``json.dumps``.

    This helper is public because GUI/session adapters can use it before
    placing arbitrary edit metadata in a deepcopy-sensitive state object.
    """

    safe = _json_safe(value)
    # ``allow_nan=False`` is a useful assertion that no accidental live value
    # or non-standard scalar escaped the conversion.
    json.dumps(safe, ensure_ascii=False, allow_nan=False)
    return safe


def _compact_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _preview_payload(value: Any, max_bytes: int) -> Any:
    """Encode a preview while ensuring it cannot consume an unbounded state."""

    if value is None:
        return None

    # For a supplied low-resolution array, retain pixels only when the raw
    # compressed representation fits the explicit preview budget.  A large
    # source-sized array becomes metadata only.
    shape = getattr(value, "shape", None)
    tobytes = getattr(value, "tobytes", None)
    if shape is not None and callable(tobytes):
        try:
            raw = bytes(tobytes())
        except Exception:
            raw = b""
        compressed = zlib.compress(raw, level=6)
        descriptor = _array_descriptor(value) or {
            "shape": [str(shape)],
            "dtype": type(value).__name__,
            "bytes": len(raw),
            "sha256": _digest_bytes(raw),
        }
        if len(compressed) <= max_bytes:
            return {
                "__type__": "preview-array",
                "shape": descriptor["shape"],
                "dtype": descriptor["dtype"],
                "bytes": len(raw),
                "sha256": descriptor["sha256"],
                "encoding": "zlib-base64",
                "data": base64.b64encode(compressed).decode("ascii"),
            }
        return {
            "__type__": "preview-omitted",
            "reason": "preview_budget",
            "shape": descriptor["shape"],
            "dtype": descriptor["dtype"],
            "bytes": len(raw),
            "sha256": descriptor["sha256"],
        }

    safe = json_safe_copy(value)
    encoded = _compact_json_bytes(safe)
    if len(encoded) <= max_bytes:
        return safe
    return {
        "__type__": "preview-omitted",
        "reason": "preview_budget",
        "bytes": len(encoded),
        "sha256": _digest_bytes(encoded),
    }


@dataclass(frozen=True)
class HistoryOutcome:
    """Result of a history mutation.

    ``status == "skip"`` and ``outcome == "noop"`` are deliberate.  A GUI
    adapter must be able to distinguish "there was nothing to undo" from a
    valid state whose value happens to be ``None``.
    """

    action: str
    status: str
    outcome: str
    cursor: int
    base_cursor: int
    entry_count: int
    checkpoint_count: int
    message: str = ""
    edit_id: Optional[str] = None
    truncated: bool = False

    @property
    def changed(self) -> bool:
        return self.status == STATUS_APPLIED

    @property
    def is_skip(self) -> bool:
        return self.status == STATUS_SKIP

    @property
    def is_noop(self) -> bool:
        return self.outcome == "noop"

    def __bool__(self) -> bool:
        return self.changed

    def to_state(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "status": self.status,
            "outcome": self.outcome,
            "cursor": self.cursor,
            "base_cursor": self.base_cursor,
            "entry_count": self.entry_count,
            "checkpoint_count": self.checkpoint_count,
            "message": self.message,
            "edit_id": self.edit_id,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class EditRecord:
    """One compact operation in the retained edit log."""

    cursor: int
    edit_id: str
    edit: Any
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_state(self) -> Dict[str, Any]:
        return {
            "cursor": self.cursor,
            "edit_id": self.edit_id,
            "edit": json_safe_copy(self.edit),
            "metadata": json_safe_copy(self.metadata),
        }

    @classmethod
    def from_state(cls, value: Mapping[str, Any], fallback_cursor: int) -> "EditRecord":
        return cls(
            cursor=int(value.get("cursor", fallback_cursor)),
            edit_id=str(value.get("edit_id", "edit-{}".format(fallback_cursor))),
            edit=json_safe_copy(value.get("edit")),
            metadata=json_safe_copy(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class PreviewCheckpoint:
    """A bounded preview checkpoint at an absolute edit cursor."""

    cursor: int
    preview: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    byte_size: int = 0

    def to_state(self) -> Dict[str, Any]:
        return {
            "cursor": self.cursor,
            "preview": json_safe_copy(self.preview),
            "metadata": json_safe_copy(self.metadata),
            "byte_size": self.byte_size,
        }

    @classmethod
    def from_state(cls, value: Mapping[str, Any]) -> "PreviewCheckpoint":
        return cls(
            cursor=int(value.get("cursor", 0)),
            preview=json_safe_copy(value.get("preview")),
            metadata=json_safe_copy(value.get("metadata") or {}),
            byte_size=int(value.get("byte_size", 0)),
        )


@dataclass(frozen=True)
class ReplayPlan:
    """Ordered operations needed to reach a target cursor.

    ``checkpoint`` is a preview only.  The caller chooses how to obtain a
    full-resolution starting image; ``replay`` applies the ordered edit
    records to an already suitable starting value.
    """

    from_cursor: int
    to_cursor: int
    base_cursor: int
    checkpoint: Optional[PreviewCheckpoint]
    operations: Tuple[EditRecord, ...]
    history_truncated: bool = False

    @property
    def edits(self) -> Tuple[Any, ...]:
        return tuple(json_safe_copy(record.edit) for record in self.operations)

    @property
    def edit_ids(self) -> Tuple[str, ...]:
        return tuple(record.edit_id for record in self.operations)

    @property
    def complete_from_source(self) -> bool:
        return not self.history_truncated and self.from_cursor == 0

    def replay(self, value: Any, apply_edit: Any) -> Any:
        """Apply operations in log order using ``apply_edit(value, edit)``."""

        current = value
        for record in self.operations:
            current = apply_edit(current, json_safe_copy(record.edit))
        return current

    def to_state(self) -> Dict[str, Any]:
        return {
            "from_cursor": self.from_cursor,
            "to_cursor": self.to_cursor,
            "base_cursor": self.base_cursor,
            "checkpoint": self.checkpoint.to_state() if self.checkpoint else None,
            "operations": [record.to_state() for record in self.operations],
            "history_truncated": self.history_truncated,
            "complete_from_source": self.complete_from_source,
        }


class AdvancedHistory:
    """Bounded edit-log history for a single Advanced Retouch session.

    ``cursor`` is the number of edits applied in the absolute timeline.  The
    retained entries start at ``base_cursor``.  If the memory budget trims old
    entries, ``base_cursor`` advances and undo stops at that explicit boundary
    instead of pretending that a missing full-resolution state exists.
    """

    def __init__(
        self,
        *,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        memory_budget_bytes: int = DEFAULT_MEMORY_BUDGET_BYTES,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        preview_max_bytes: int = DEFAULT_PREVIEW_MAX_BYTES,
        max_checkpoints: int = DEFAULT_MAX_CHECKPOINTS,
    ) -> None:
        self.max_entries = self._positive_int("max_entries", max_entries)
        self.memory_budget_bytes = self._positive_int("memory_budget_bytes", memory_budget_bytes)
        self.checkpoint_interval = self._positive_int("checkpoint_interval", checkpoint_interval)
        self.preview_max_bytes = self._positive_int("preview_max_bytes", preview_max_bytes)
        self.max_checkpoints = self._positive_int("max_checkpoints", max_checkpoints)
        self._base_cursor = 0
        self._cursor = 0
        self._next_edit_number = 1
        self._entries: List[EditRecord] = []
        self._checkpoints: List[PreviewCheckpoint] = []

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        try:
            result = int(value)
        except (TypeError, ValueError):
            raise ValueError("{} must be a positive integer".format(name))
        if result <= 0:
            raise ValueError("{} must be a positive integer".format(name))
        return result

    @property
    def base_cursor(self) -> int:
        return self._base_cursor

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def end_cursor(self) -> int:
        return self._base_cursor + len(self._entries)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def checkpoint_count(self) -> int:
        return len(self._checkpoints)

    @property
    def can_undo(self) -> bool:
        return self._cursor > self._base_cursor

    @property
    def can_redo(self) -> bool:
        return self._cursor < self.end_cursor

    @property
    def history_truncated(self) -> bool:
        return self._base_cursor > 0

    @property
    def memory_size_bytes(self) -> int:
        return len(_compact_json_bytes(self.to_state()))

    @property
    def entries(self) -> Tuple[EditRecord, ...]:
        return tuple(
            EditRecord(record.cursor, record.edit_id, json_safe_copy(record.edit), json_safe_copy(record.metadata))
            for record in self._entries
        )

    @property
    def checkpoints(self) -> Tuple[PreviewCheckpoint, ...]:
        return tuple(
            PreviewCheckpoint(cp.cursor, json_safe_copy(cp.preview), json_safe_copy(cp.metadata), cp.byte_size)
            for cp in self._checkpoints
        )

    def current_edit(self) -> Optional[Any]:
        """Return the current retained edit description, if one exists."""

        if self._cursor <= self._base_cursor:
            return None
        index = self._cursor - self._base_cursor - 1
        if index < 0 or index >= len(self._entries):
            return None
        return json_safe_copy(self._entries[index].edit)

    def current_edit_log(self) -> List[Any]:
        """Return a detached list of retained edits through the cursor."""

        count = max(0, min(len(self._entries), self._cursor - self._base_cursor))
        return [json_safe_copy(record.edit) for record in self._entries[:count]]

    def _outcome(
        self,
        action: str,
        status: str,
        message: str,
        *,
        edit_id: Optional[str] = None,
        truncated: bool = False,
    ) -> HistoryOutcome:
        return HistoryOutcome(
            action=action,
            status=status,
            outcome="noop" if status == STATUS_SKIP else status,
            cursor=self._cursor,
            base_cursor=self._base_cursor,
            entry_count=len(self._entries),
            checkpoint_count=len(self._checkpoints),
            message=message,
            edit_id=edit_id,
            truncated=truncated or self.history_truncated,
        )

    def _drop_oldest_entry(self) -> None:
        if not self._entries:
            return
        removed = self._entries.pop(0)
        self._base_cursor = removed.cursor
        if self._cursor < self._base_cursor:
            self._cursor = self._base_cursor
        self._checkpoints = [
            checkpoint
            for checkpoint in self._checkpoints
            if checkpoint.cursor >= self._base_cursor
        ]

    def _trim_checkpoints_before_base(self) -> None:
        self._checkpoints = [
            checkpoint
            for checkpoint in self._checkpoints
            if self._base_cursor <= checkpoint.cursor <= self.end_cursor
        ]

    def _state_size(self) -> int:
        return len(_compact_json_bytes(self.to_state()))

    def _enforce_limits(self) -> None:
        while len(self._entries) > self.max_entries:
            self._drop_oldest_entry()

        self._trim_checkpoints_before_base()
        while len(self._checkpoints) > self.max_checkpoints:
            self._checkpoints.pop(0)

        # Checkpoints are disposable previews; discard them before discarding
        # edit records.  The latter are needed for deterministic replay.
        while self._state_size() > self.memory_budget_bytes and self._checkpoints:
            self._checkpoints.pop(0)

        while self._state_size() > self.memory_budget_bytes and self._entries:
            self._drop_oldest_entry()
            while self._state_size() > self.memory_budget_bytes and self._checkpoints:
                self._checkpoints.pop(0)

        self._trim_checkpoints_before_base()

    def append(
        self,
        edit: Any,
        *,
        edit_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        preview: Any = None,
        checkpoint: bool = False,
    ) -> HistoryOutcome:
        """Append an edit and optionally capture a preview checkpoint.

        Redo entries are discarded when a new edit follows an undo.  A
        checkpoint is created when ``checkpoint=True`` or when the append is
        on the configured interval.  Supplying ``preview`` is required for a
        useful checkpoint; no full-resolution value is ever inferred here.
        """

        before = self.to_state()
        safe_edit = json_safe_copy(edit)
        safe_metadata = json_safe_copy(dict(metadata or {}))
        if not isinstance(safe_metadata, dict):
            safe_metadata = {"value": safe_metadata}

        retained_count = max(0, self._cursor - self._base_cursor)
        if retained_count < len(self._entries):
            self._entries = self._entries[:retained_count]
            self._checkpoints = [
                item for item in self._checkpoints if item.cursor <= self._cursor
            ]

        record_cursor = self._cursor + 1
        record_id = str(edit_id) if edit_id is not None else "edit-{}".format(self._next_edit_number)
        self._next_edit_number += 1
        self._entries.append(EditRecord(record_cursor, record_id, safe_edit, safe_metadata))
        self._cursor = record_cursor

        should_checkpoint = preview is not None and (
            checkpoint
            or record_cursor % self.checkpoint_interval == 0
            or not self._checkpoints
        )
        if should_checkpoint:
            self._store_checkpoint(record_cursor, preview, {"automatic": not checkpoint})

        self._enforce_limits()
        retained = any(record.edit_id == record_id for record in self._entries)
        if not retained or self._state_size() > self.memory_budget_bytes:
            self._restore_state(before)
            return self._outcome(
                "append",
                STATUS_REJECTED,
                "Edit was not retained because it exceeds the history memory budget.",
            )

        return self._outcome(
            "append",
            STATUS_APPLIED,
            "Edit appended.",
            edit_id=record_id,
            truncated=self.history_truncated,
        )

    def _store_checkpoint(
        self,
        cursor: int,
        preview: Any,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> PreviewCheckpoint:
        payload = _preview_payload(preview, self.preview_max_bytes)
        byte_size = len(_compact_json_bytes(payload)) if payload is not None else 0
        checkpoint = PreviewCheckpoint(
            cursor=int(cursor),
            preview=payload,
            metadata=json_safe_copy(dict(metadata or {})),
            byte_size=byte_size,
        )
        self._checkpoints = [item for item in self._checkpoints if item.cursor != checkpoint.cursor]
        self._checkpoints.append(checkpoint)
        self._checkpoints.sort(key=lambda item: item.cursor)
        return checkpoint

    def add_checkpoint(
        self,
        preview: Any,
        *,
        cursor: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> HistoryOutcome:
        """Add a bounded preview checkpoint at an existing cursor."""

        if preview is None:
            return self._outcome(
                "checkpoint",
                STATUS_SKIP,
                "No preview supplied; checkpoint skipped.",
            )
        target = self._cursor if cursor is None else int(cursor)
        if target < self._base_cursor or target > self.end_cursor:
            raise ValueError(
                "checkpoint cursor {} is outside [{}, {}]".format(
                    target, self._base_cursor, self.end_cursor
                )
            )
        before = self.to_state()
        self._store_checkpoint(target, preview, metadata)
        self._enforce_limits()
        retained = any(item.cursor == target for item in self._checkpoints)
        if not retained or self._state_size() > self.memory_budget_bytes:
            self._restore_state(before)
            return self._outcome(
                "checkpoint",
                STATUS_REJECTED,
                "Checkpoint was not retained because it exceeds the history memory budget.",
            )
        return self._outcome("checkpoint", STATUS_APPLIED, "Preview checkpoint stored.")

    def select_checkpoint(self, cursor: Optional[int] = None) -> Optional[PreviewCheckpoint]:
        """Return the nearest retained checkpoint at or before ``cursor``."""

        target = self._cursor if cursor is None else int(cursor)
        if target < self._base_cursor or target > self.end_cursor:
            raise ValueError(
                "cursor {} is outside [{}, {}]".format(
                    target, self._base_cursor, self.end_cursor
                )
            )
        choices = [item for item in self._checkpoints if item.cursor <= target]
        if not choices:
            return None
        selected = choices[-1]
        return PreviewCheckpoint(
            selected.cursor,
            json_safe_copy(selected.preview),
            json_safe_copy(selected.metadata),
            selected.byte_size,
        )

    # A descriptive alias for callers that prefer the noun used in the
    # evidence/replay documentation.
    checkpoint_for = select_checkpoint

    def undo(self) -> HistoryOutcome:
        """Move one edit backward, or return an explicit skip/no-op."""

        if not self.can_undo:
            return self._outcome("undo", STATUS_SKIP, "Nothing to undo.")
        self._cursor -= 1
        return self._outcome("undo", STATUS_APPLIED, "Undo applied.")

    def redo(self) -> HistoryOutcome:
        """Move one edit forward, or return an explicit skip/no-op."""

        if not self.can_redo:
            return self._outcome("redo", STATUS_SKIP, "Nothing to redo.")
        self._cursor += 1
        return self._outcome("redo", STATUS_APPLIED, "Redo applied.")

    def replay_plan(
        self,
        cursor: Optional[int] = None,
        *,
        checkpoint_cursor: Optional[int] = None,
    ) -> ReplayPlan:
        """Build an ordered plan from the best checkpoint to ``cursor``."""

        target = self._cursor if cursor is None else int(cursor)
        if target < self._base_cursor or target > self.end_cursor:
            raise ValueError(
                "cursor {} is outside [{}, {}]".format(
                    target, self._base_cursor, self.end_cursor
                )
            )

        if checkpoint_cursor is not None:
            checkpoint = next(
                (item for item in self._checkpoints if item.cursor == int(checkpoint_cursor)),
                None,
            )
            if checkpoint is None or checkpoint.cursor > target:
                raise ValueError("requested checkpoint is not available for this replay target")
            checkpoint = PreviewCheckpoint(
                checkpoint.cursor,
                json_safe_copy(checkpoint.preview),
                json_safe_copy(checkpoint.metadata),
                checkpoint.byte_size,
            )
        else:
            checkpoint = self.select_checkpoint(target)

        start = checkpoint.cursor if checkpoint is not None else self._base_cursor
        operations = tuple(
            EditRecord(
                record.cursor,
                record.edit_id,
                json_safe_copy(record.edit),
                json_safe_copy(record.metadata),
            )
            for record in self._entries
            if start < record.cursor <= target
        )
        return ReplayPlan(
            from_cursor=start,
            to_cursor=target,
            base_cursor=self._base_cursor,
            checkpoint=checkpoint,
            operations=operations,
            history_truncated=self.history_truncated,
        )

    def to_state(self) -> Dict[str, Any]:
        """Return a detached JSON/deepcopy-safe state dictionary."""

        return {
            "schema_version": SCHEMA_VERSION,
            "max_entries": self.max_entries,
            "memory_budget_bytes": self.memory_budget_bytes,
            "checkpoint_interval": self.checkpoint_interval,
            "preview_max_bytes": self.preview_max_bytes,
            "max_checkpoints": self.max_checkpoints,
            "base_cursor": self._base_cursor,
            "cursor": self._cursor,
            "next_edit_number": self._next_edit_number,
            "entries": [record.to_state() for record in self._entries],
            "checkpoints": [checkpoint.to_state() for checkpoint in self._checkpoints],
        }

    # ``state`` is convenient when the object is used as a Gradio adapter.
    state = to_state

    def deepcopy_state(self) -> Dict[str, Any]:
        """Return a fresh deepcopy-safe state suitable for ``gr.State``."""

        return copy.deepcopy(self.to_state())

    def __deepcopy__(self, memo: Dict[int, Any]) -> "AdvancedHistory":
        del memo
        return self.from_state(self.to_state())

    def _restore_state(self, state: Mapping[str, Any]) -> None:
        restored = self.from_state(state)
        self.__dict__.clear()
        self.__dict__.update(restored.__dict__)

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> "AdvancedHistory":
        """Restore history from ``to_state`` output."""

        if not isinstance(state, Mapping):
            raise TypeError("history state must be a mapping")
        history = cls(
            max_entries=int(state.get("max_entries", DEFAULT_MAX_ENTRIES)),
            memory_budget_bytes=int(state.get("memory_budget_bytes", DEFAULT_MEMORY_BUDGET_BYTES)),
            checkpoint_interval=int(state.get("checkpoint_interval", DEFAULT_CHECKPOINT_INTERVAL)),
            preview_max_bytes=int(state.get("preview_max_bytes", DEFAULT_PREVIEW_MAX_BYTES)),
            max_checkpoints=int(state.get("max_checkpoints", DEFAULT_MAX_CHECKPOINTS)),
        )
        history._base_cursor = max(0, int(state.get("base_cursor", 0)))
        history._cursor = max(history._base_cursor, int(state.get("cursor", history._base_cursor)))
        history._next_edit_number = max(1, int(state.get("next_edit_number", 1)))

        raw_entries = state.get("entries") or []
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes, bytearray)):
            raise TypeError("history entries must be a sequence")
        history._entries = []
        fallback = history._base_cursor + 1
        for item in raw_entries:
            if not isinstance(item, Mapping):
                continue
            history._entries.append(EditRecord.from_state(item, fallback))
            fallback += 1

        # Normalize a hand-edited/older state to the contiguous cursor model.
        for offset, record in enumerate(history._entries):
            expected = history._base_cursor + offset + 1
            if record.cursor != expected:
                history._entries[offset] = EditRecord(
                    expected, record.edit_id, record.edit, record.metadata
                )
        history._cursor = min(history._cursor, history.end_cursor)
        history._cursor = max(history._cursor, history._base_cursor)

        raw_checkpoints = state.get("checkpoints") or []
        if not isinstance(raw_checkpoints, Sequence) or isinstance(raw_checkpoints, (str, bytes, bytearray)):
            raise TypeError("history checkpoints must be a sequence")
        history._checkpoints = []
        for item in raw_checkpoints:
            if not isinstance(item, Mapping):
                continue
            checkpoint = PreviewCheckpoint.from_state(item)
            if history._base_cursor <= checkpoint.cursor <= history.end_cursor:
                history._checkpoints.append(checkpoint)
        history._checkpoints.sort(key=lambda item: item.cursor)
        history._enforce_limits()
        return history

    def to_json(self, *, indent: Optional[int] = 2) -> str:
        return json.dumps(
            self.to_state(),
            indent=indent,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, value: str) -> "AdvancedHistory":
        return cls.from_state(json.loads(value))


# Functional helpers keep adapters that store only a dict from needing to
# instantiate the class just to make a safe copy.
def history_state(value: Any) -> Dict[str, Any]:
    """Return JSON-safe state from an ``AdvancedHistory`` or state mapping."""

    if isinstance(value, AdvancedHistory):
        return value.to_state()
    if not isinstance(value, Mapping):
        raise TypeError("value must be AdvancedHistory or a state mapping")
    return json_safe_copy(value)


def history_from_state(value: Mapping[str, Any]) -> AdvancedHistory:
    return AdvancedHistory.from_state(value)


__all__ = [
    "AdvancedHistory",
    "EditRecord",
    "HistoryOutcome",
    "PreviewCheckpoint",
    "ReplayPlan",
    "SCHEMA_VERSION",
    "STATUS_APPLIED",
    "STATUS_REJECTED",
    "STATUS_SKIP",
    "DEFAULT_MAX_ENTRIES",
    "DEFAULT_MEMORY_BUDGET_BYTES",
    "DEFAULT_CHECKPOINT_INTERVAL",
    "DEFAULT_PREVIEW_MAX_BYTES",
    "json_safe_copy",
    "history_state",
    "history_from_state",
]
