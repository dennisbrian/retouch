"""Session model for the Retouch Engine (F2).

A Session captures the complete edit state: recipe name, all processing
parameters, image path, and a timestamp. Sessions serialize to JSON for
save/load, batch reproduction, and CLI reproducibility.

The session IS the params dict that ``gui.py`` already materializes at
``params = dict(zip(PROCESS_INPUT_KEYS, args))`` — plus metadata.

Public API:
    Session — dataclass with to_json/from_json
    UndoRedoStack — bounded undo/redo history with cursor
    Snapshot — named session save for A/B comparison

Design principles:
    - Forward-compatible: unknown params in loaded JSON warn, don't crash
    - Tiny: sessions are ~2-4 KB JSON, memory is trivial
    - Reproducible: same params dict → same output hash
    - Aligned: load-session returns a tuple in PROCESS_INPUT_KEYS order
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SESSION_VERSION = 1


@dataclass
class Session:
    """Complete edit state for a single image processing session.

    Attributes:
        version: Schema version for forward compatibility.
        recipe: Recipe name (e.g. "natural", "porcelain_unified_v1") or None.
        params: Dict of processing parameters keyed by PROCESS_INPUT_KEYS.
        image_path: Path to the source image (for reference, not embedded).
        created: ISO 8601 timestamp of session creation.
        image_hash: Optional SHA-256 hash of the source image (first 16 chars)
            for verifying the session matches the right image.
    """

    version: int = SESSION_VERSION
    recipe: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    image_path: Optional[str] = None
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    image_hash: Optional[str] = None
    local_adjustments: List[Dict[str, Any]] = field(default_factory=list)
    advanced_retouch: Dict[str, Any] = field(default_factory=dict)
    # P4 instrumentation: parameter/evidence metadata only. Face pixels and
    # image arrays are intentionally never stored here.
    style_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize session to JSON string."""
        payload = {
            "version": self.version,
            "recipe": self.recipe,
            "params": self.params,
            "image_path": self.image_path,
            "created": self.created,
            "image_hash": self.image_hash,
            "local_adjustments": self.local_adjustments,
        }
        if self.style_events:
            payload["style_events"] = self.style_events
        # Keep legacy sessions byte/schema-compatible when no Advanced
        # Retouch actions exist; populated sessions carry the versioned edit
        # log needed to reproduce the manual canvas.
        if self.advanced_retouch:
            payload["advanced_retouch"] = self.advanced_retouch
        return json.dumps(payload, indent=indent, sort_keys=True, ensure_ascii=False)

    def to_file(self, path: str) -> str:
        """Save session to a JSON file. Returns the absolute path."""
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path) or ".", exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
        return abs_path

    @classmethod
    def from_json(cls, json_str: str) -> "Session":
        """Deserialize session from JSON string.

        Forward-compatible: unknown keys are warned and ignored.
        Missing keys use defaults.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Create session from a dict (e.g. parsed JSON).

        Forward-compatible: unknown keys warn, missing keys use defaults.
        """
        known_keys = {"version", "recipe", "params", "image_path", "created", "image_hash", "local_adjustments", "advanced_retouch", "style_events"}
        unknown = set(data.keys()) - known_keys
        if unknown:
            logger.warning(
                f"Session JSON contains unknown keys (ignored): {sorted(unknown)}"
            )

        version = data.get("version", SESSION_VERSION)
        if version > SESSION_VERSION:
            logger.warning(
                f"Session version {version} is newer than supported {SESSION_VERSION} — "
                f"loading with best-effort compatibility"
            )

        return cls(
            version=version,
            recipe=data.get("recipe"),
            params=data.get("params", {}),
            image_path=data.get("image_path"),
            created=data.get("created", datetime.now(timezone.utc).isoformat()),
            image_hash=data.get("image_hash"),
            local_adjustments=data.get("local_adjustments", []),
            advanced_retouch=data.get("advanced_retouch", {}),
            style_events=data.get("style_events", []),
        )

    @classmethod
    def from_file(cls, path: str) -> "Session":
        """Load session from a JSON file."""
        abs_path = os.path.abspath(path)
        with open(abs_path, "r", encoding="utf-8") as f:
            return cls.from_json(f.read())

    def to_params_tuple(self, input_keys: Tuple[str, ...]) -> Tuple[Any, ...]:
        """Convert session params to a tuple in PROCESS_INPUT_KEYS order.

        This is the bridge between a loaded session and the Gradio UI:
        load_session returns this tuple, which Gradio distributes to
        the component values.

        Missing params use their defaults from the input_keys defaults.
        """
        from gui import PROCESS_INPUT_KEYS
        if input_keys != PROCESS_INPUT_KEYS:
            input_keys = PROCESS_INPUT_KEYS
        return tuple(self.params.get(k) for k in input_keys)

    def merge_params(self, overrides: Dict[str, Any]) -> "Session":
        """Return a new session with params updated by overrides.

        Non-mutating — returns a copy for undo/redo safety.
        """
        new_params = {**self.params, **overrides}
        return Session(
            version=self.version,
            recipe=self.recipe,
            params=new_params,
            image_path=self.image_path,
            created=self.created,
            image_hash=self.image_hash,
            local_adjustments=self.local_adjustments,
            advanced_retouch=self.advanced_retouch,
            style_events=self.style_events,
        )

    def record_style_event(
        self,
        *,
        stage: str,
        suggested: Optional[Dict[str, Any]] = None,
        final: Optional[Dict[str, Any]] = None,
        outcome: str = "accepted",
        scene_features: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record inspectable style deltas without storing face pixels."""
        if outcome not in {"accepted", "rejected", "skipped", "review"}:
            raise ValueError(f"unknown style event outcome: {outcome!r}")

        def safe_mapping(values: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            out: Dict[str, Any] = {}
            for key, value in (values or {}).items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    out[str(key)] = value
            return out

        self.style_events.append({
            "stage": str(stage),
            "suggested": safe_mapping(suggested),
            "final": safe_mapping(final),
            "outcome": outcome,
            "scene_features": safe_mapping(scene_features),
        })


@dataclass
class UndoRedoStack:
    """Bounded undo/redo history for session params.

    Stores up to ``max_size`` param dicts with a cursor. Push on every
    Process click; undo/redo move the cursor and return the params at
    that position.

    The stack stores param dicts only (not full Sessions) to minimize
    memory — the recipe and image_path don't change between undo states.
    """

    max_size: int = 50
    _stack: List[Dict[str, Any]] = field(default_factory=list)
    _cursor: int = -1

    def push(self, params: Dict[str, Any]) -> None:
        """Push a new params state onto the stack.

        If the cursor is not at the end (i.e. user did some undos),
        the redo history is truncated before pushing.
        """
        if self._cursor < len(self._stack) - 1:
            self._stack = self._stack[: self._cursor + 1]

        self._stack.append(dict(params))
        self._cursor = len(self._stack) - 1

        if len(self._stack) > self.max_size:
            trim = len(self._stack) - self.max_size
            self._stack = self._stack[trim:]
            self._cursor -= trim

    def undo(self) -> Optional[Dict[str, Any]]:
        """Move cursor back and return the params at the new position.

        Returns None if at the beginning (no more undo).
        """
        if self._cursor <= 0:
            return None
        self._cursor -= 1
        return dict(self._stack[self._cursor])

    def redo(self) -> Optional[Dict[str, Any]]:
        """Move cursor forward and return the params at the new position.

        Returns None if at the end (no more redo).
        """
        if self._cursor >= len(self._stack) - 1:
            return None
        self._cursor += 1
        return dict(self._stack[self._cursor])

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor < len(self._stack) - 1

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def size(self) -> int:
        return len(self._stack)

    def current(self) -> Optional[Dict[str, Any]]:
        """Return the params at the current cursor position."""
        if 0 <= self._cursor < len(self._stack):
            return dict(self._stack[self._cursor])
        return None

    def clear(self) -> None:
        """Clear all history."""
        self._stack.clear()
        self._cursor = -1


@dataclass
class Snapshot:
    """Named session save for A/B comparison.

    Snapshots are stored in a Gradio State dict, not persisted to disk
    (that's what Session.to_file is for). The "Compare" button renders
    the snapshot vs current side-by-side.
    """

    name: str
    session: Session
    created: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "session": json.loads(self.session.to_json()),
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Snapshot":
        return cls(
            name=data["name"],
            session=Session.from_dict(data["session"]),
            created=data.get("created", datetime.now(timezone.utc).isoformat()),
        )


def compute_image_hash(image_path: str) -> Optional[str]:
    """Compute a short SHA-256 hash of an image file (first 16 chars).

    Used for verifying a session matches the right image.
    Returns None if the file can't be read.
    """
    import hashlib

    try:
        with open(image_path, "rb") as f:
            data = f.read()
        return hashlib.sha256(data).hexdigest()[:16]
    except (OSError, IOError) as e:
        logger.warning(f"Could not hash image {image_path}: {e}")
        return None


def create_session_from_params(
    params: Dict[str, Any],
    recipe: Optional[str] = None,
    image_path: Optional[str] = None,
) -> Session:
    """Convenience: create a Session from a params dict.

    If image_path is provided, computes the image hash for verification.
    """
    image_hash = compute_image_hash(image_path) if image_path else None
    return Session(
        recipe=recipe,
        params=dict(params),
        image_path=image_path,
        image_hash=image_hash,
    )
