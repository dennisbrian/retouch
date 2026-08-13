"""Subject-linked project profiles and multi-reference Look Boards.

Profiles store photographer intent and protected-feature metadata, not face
pixels. Look Boards keep references linked to a project so later processing can
use several references without collapsing them into one opaque style.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProjectProfile:
    profile_id: str
    subject_key: str
    display_name: str
    recipe: str = "natural"
    style_name: Optional[str] = None
    preferred_params: Dict[str, Any] = field(default_factory=dict)
    protected_marks: List[Dict[str, Any]] = field(default_factory=list)
    reference_paths: List[str] = field(default_factory=list)
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ProjectProfile":
        return cls(
            profile_id=str(payload["profile_id"]),
            subject_key=str(payload.get("subject_key", payload["profile_id"])),
            display_name=str(payload.get("display_name", payload["profile_id"])),
            recipe=str(payload.get("recipe", "natural")),
            style_name=payload.get("style_name"),
            preferred_params=dict(payload.get("preferred_params") or {}),
            protected_marks=[dict(item) for item in payload.get("protected_marks", [])],
            reference_paths=[str(item) for item in payload.get("reference_paths", [])],
            created=str(payload.get("created", _now())),
            updated=str(payload.get("updated", _now())),
        )


@dataclass(frozen=True)
class LookReference:
    path: str
    label: str
    weight: float = 1.0
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("Look Board reference path is required")
        if not 0.0 <= float(self.weight) <= 1.0:
            raise ValueError("Look Board reference weight must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LookBoard:
    board_id: str
    project_id: Optional[str] = None
    references: List[LookReference] = field(default_factory=list)
    created: str = field(default_factory=_now)
    updated: str = field(default_factory=_now)

    def add_reference(self, reference: LookReference) -> None:
        if any(item.path == reference.path for item in self.references):
            self.references = [item for item in self.references if item.path != reference.path]
        self.references.append(reference)
        self.updated = _now()

    def remove_reference(self, path: str) -> None:
        self.references = [item for item in self.references if item.path != path]
        self.updated = _now()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "board_id": self.board_id,
            "project_id": self.project_id,
            "references": [item.to_dict() for item in self.references],
            "created": self.created,
            "updated": self.updated,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LookBoard":
        return cls(
            board_id=str(payload["board_id"]),
            project_id=payload.get("project_id"),
            references=[LookReference(**dict(item)) for item in payload.get("references", [])],
            created=str(payload.get("created", _now())),
            updated=str(payload.get("updated", _now())),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "LookBoard":
        return cls.from_dict(json.loads(payload))


class ProjectProfileStore:
    """Atomic JSON store for project profiles and Look Boards."""

    def __init__(self, path: Optional[Union[str, Path]] = None) -> None:
        self.path = Path(path).expanduser() if path else Path.home() / ".retouch" / "projects.json"
        self.profiles: Dict[str, ProjectProfile] = {}
        self.boards: Dict[str, LookBoard] = {}
        self.load()

    def load(self) -> None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return
        self.profiles = {
            str(key): ProjectProfile.from_dict(value)
            for key, value in (payload.get("profiles") or {}).items()
        }
        self.boards = {
            str(key): LookBoard.from_dict(value)
            for key, value in (payload.get("boards") or {}).items()
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "profiles": {key: value.to_dict() for key, value in self.profiles.items()},
            "boards": {key: value.to_dict() for key, value in self.boards.items()},
        }
        fd, temp_name = tempfile.mkstemp(prefix=".retouch-project-", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
            os.replace(temp_name, self.path)
        except OSError:
            try:
                os.remove(temp_name)
            except OSError:
                pass
            raise

    def upsert_profile(self, profile: ProjectProfile) -> None:
        profile.updated = _now()
        self.profiles[profile.profile_id] = profile
        self.save()

    def upsert_board(self, board: LookBoard) -> None:
        board.updated = _now()
        self.boards[board.board_id] = board
        self.save()

    def list_profiles(self) -> List[ProjectProfile]:
        return sorted(self.profiles.values(), key=lambda item: item.display_name.lower())

    def list_boards(self) -> List[LookBoard]:
        return sorted(self.boards.values(), key=lambda item: item.board_id.lower())
