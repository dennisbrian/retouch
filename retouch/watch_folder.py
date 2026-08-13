"""Resumable, non-destructive watch-folder ingestion.

The service uses polling rather than requiring a platform-specific watcher.
Files are emitted only after their size/mtime signature is unchanged across
two scans. Processing callbacks are supplied by the caller; failures remain
pending and are recorded with an explanation.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

from .io import IMAGE_EXTENSIONS


@dataclass
class WatchRecord:
    path: str
    signature: str
    status: str = "pending"  # pending | processing | done | failed
    attempts: int = 0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WatchFolderState:
    version: int = 1
    records: Dict[str, WatchRecord] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "records": {path: record.to_dict() for path, record in self.records.items()},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WatchFolderState":
        records = {
            str(path): WatchRecord(**dict(value))
            for path, value in (payload.get("records") or {}).items()
        }
        return cls(version=int(payload.get("version", 1)), records=records)


class WatchFolder:
    """Poll a folder and process only stable, changed image files."""

    def __init__(
        self,
        input_dir: Union[str, Path],
        *,
        state_path: Optional[Union[str, Path]] = None,
        recursive: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir).expanduser().resolve()
        if not self.input_dir.is_dir():
            raise NotADirectoryError(str(self.input_dir))
        self.recursive = bool(recursive)
        self.state_path = Path(state_path).expanduser() if state_path else self.input_dir / ".retouch-watch.json"
        self.state = self._load_state()

    def _load_state(self) -> WatchFolderState:
        try:
            return WatchFolderState.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        except (FileNotFoundError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return WatchFolderState()

    def save(self) -> None:
        """Persist state atomically so interruption cannot corrupt the queue."""
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=".retouch-watch-", suffix=".tmp", dir=str(self.state_path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.state.to_dict(), handle, indent=2, sort_keys=True)
            os.replace(temp_name, self.state_path)
        except OSError:
            try:
                os.remove(temp_name)
            except OSError:
                pass
            raise

    def _paths(self) -> Iterable[Path]:
        iterator = self.input_dir.rglob("*") if self.recursive else self.input_dir.iterdir()
        return sorted(
            path for path in iterator
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.name != self.state_path.name
        )

    @staticmethod
    def _signature(path: Path) -> str:
        stat = path.stat()
        return f"{stat.st_size}:{stat.st_mtime_ns}"

    def scan_once(self) -> List[WatchRecord]:
        """Update stability state and return files ready for processing.

        A newly observed or changed file is held for one scan. It becomes
        ready only when its signature is observed unchanged on the next scan.
        """
        ready: List[WatchRecord] = []
        current_paths = set()
        for path in self._paths():
            resolved = str(path.resolve())
            current_paths.add(resolved)
            try:
                signature = self._signature(path)
            except OSError:
                continue
            previous = self.state.records.get(resolved)
            if previous is None or previous.signature != signature:
                self.state.records[resolved] = WatchRecord(path=resolved, signature=signature)
                continue
            if previous.status in {"pending", "failed"}:
                ready.append(previous)
        # Stale records are retained for audit/retry, but no longer appear in
        # the active queue when the source has been removed.
        for path, record in self.state.records.items():
            if path not in current_paths and record.status == "processing":
                record.status = "failed"
                record.error = "source disappeared while processing"
        self.save()
        return ready

    def process_pending(
        self,
        processor: Callable[[Path], Any],
        *,
        limit: Optional[int] = None,
    ) -> List[WatchRecord]:
        """Process stable files; failed callbacks remain retryable."""
        ready = self.scan_once()
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            ready = ready[:limit]
        completed: List[WatchRecord] = []
        for record in ready:
            record.status = "processing"
            record.attempts += 1
            record.error = None
            self.save()
            try:
                processor(Path(record.path))
            except Exception as exc:  # callback errors are queue state, not watcher crashes
                record.status = "failed"
                record.error = f"{type(exc).__name__}: {exc}"
            else:
                record.status = "done"
            completed.append(record)
            self.save()
        return completed
