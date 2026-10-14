"""Crash recovery for the Layer Editor: autosave unsaved work, offer it back.

Each editor session owns one :class:`Autosaver`. After an edit leaves the
document dirty, the current revision is written in the background as a
normal ``.comp`` package under :func:`recovery_dir`, followed by a small JSON
description. Saving (or undoing back to the saved revision) deletes the
entry; so does opening another document with "discard". A process crash,
force quit or closed window leaves it on disk, and :func:`list_recoveries`
offers it back on the next start.

Writes reuse ``save_project``: the package is staged and swapped in whole,
so an entry is either the previous or the new revision, never half of one.
Unchanged layers are copied rather than re-encoded (milestone 6 fast saves),
so after the first write an autosave is a small fraction of a full save.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .project_store import (_STALE_SAVING_SECONDS, _sweep_interrupted_saves,
                            load_project, save_project)

_logger = logging.getLogger(__name__)

RECOVERY_DIR_ENV = 'RETOUCH_EDITOR_RECOVERY_DIR'
# Entries nobody restored or discarded are removed after this long.
MAX_AGE_SECONDS = 30 * 24 * 3600
_META_VERSION = 1

# Entry ids written by autosavers alive in this process; never offered back,
# so a second browser tab cannot restore (and fork) a live session.
_active_ids = set()
_active_lock = threading.Lock()


def recovery_dir() -> Path:
    """Per-user folder for autosaved editor work (not created here)."""
    override = os.environ.get(RECOVERY_DIR_ENV)
    if override:
        return Path(override).expanduser()
    if sys.platform == 'darwin':
        return (Path.home() / 'Library' / 'Application Support' / 'ProMaxRetouch'
                / 'Recovery')
    if sys.platform == 'win32':
        base = os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData' / 'Local'))
        return Path(base) / 'ProMaxRetouch' / 'Recovery'
    base = os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local' / 'share')
    return Path(base) / 'promaxretouch' / 'recovery'


@dataclass(frozen=True)
class RecoveryEntry:
    """One autosaved document that no live session owns."""
    entry_id: str
    title: str
    width: int
    height: int
    layers: int
    updated: float
    package: Path

    @property
    def label(self) -> str:
        # Local wall-clock time, as the user reads it.
        when = datetime.fromtimestamp(self.updated).astimezone().strftime('%Y-%m-%d %H:%M')
        return (f'{self.title} · {self.width} × {self.height} · '
                f'{self.layers} layers · autosaved {when}')


def _paths(root: Path, entry_id: str):
    return root / f'{entry_id}.comp', root / f'{entry_id}.json'


def _valid_id(entry_id) -> bool:
    try:
        return uuid.UUID(entry_id).hex == entry_id
    except (TypeError, ValueError, AttributeError):
        return False


def list_recoveries(root: Path | None = None, now: float | None = None
                    ) -> list[RecoveryEntry]:
    """Recoverable entries, newest first. Removes expired and half-written ones."""
    root = Path(root) if root is not None else recovery_dir()
    if not root.is_dir():
        return []
    now = time.time() if now is None else now
    with _active_lock:
        active = set(_active_ids)
    entries = []
    for meta_path in root.glob('*.json'):
        entry_id = meta_path.stem
        if not _valid_id(entry_id) or entry_id in active:
            continue
        package, _ = _paths(root, entry_id)
        # Repair a package stranded by a crash between the swap renames
        # before deciding the entry is gone.
        _sweep_interrupted_saves(package)
        try:
            meta = json.loads(meta_path.read_text('utf-8'))
            entry = RecoveryEntry(entry_id, str(meta['title']), int(meta['width']),
                                  int(meta['height']), int(meta['layers']),
                                  float(meta['updated']), package)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if not package.is_dir() or now - entry.updated > MAX_AGE_SECONDS:
            _remove(root, entry_id)
        else:
            entries.append(entry)
    # Staging folders of interrupted writes (a crash mid-autosave). A fresh
    # one may belong to a concurrent autosave in another process, so only
    # aged dirs are removed.
    for staging in root.glob('.*.comp.saving-*'):
        owner = staging.name[1:].split('.comp.saving-')[0]
        if owner not in active:
            try:
                if staging.stat().st_mtime >= now - _STALE_SAVING_SECONDS:
                    continue
            except FileNotFoundError:
                continue
            shutil.rmtree(staging, ignore_errors=True)
    # Retired packages orphaned by a crash (their entry is long gone).
    for retired in root.glob('.*.comp.replaced-*'):
        owner = retired.name[1:].split('.comp.replaced-')[0]
        if not (root / f'{owner}.json').is_file():
            shutil.rmtree(retired, ignore_errors=True)
    return sorted(entries, key=lambda e: e.updated, reverse=True)


def load_recovery(entry_id: str, root: Path | None = None):
    """Open an entry's document. Raises ``ValueError`` if it is gone or invalid."""
    root = Path(root) if root is not None else recovery_dir()
    if not _valid_id(entry_id):
        raise ValueError('Choose an autosaved document first')
    package, meta_path = _paths(root, entry_id)
    if not meta_path.is_file():
        raise ValueError('That autosave no longer exists')
    # A crash between the swap renames leaves the target missing with the
    # only copy under ".replaced-"; repair before refusing.
    _sweep_interrupted_saves(package)
    if not package.is_dir():
        raise ValueError('That autosave no longer exists')
    return load_project(package)


def discard_recovery(entry_id: str, root: Path | None = None) -> None:
    """Delete an entry. Live entries of this process are left alone."""
    root = Path(root) if root is not None else recovery_dir()
    if not _valid_id(entry_id):
        raise ValueError('Choose an autosaved document first')
    with _active_lock:
        if entry_id in _active_ids:
            raise ValueError('That document is open in another editor window')
    _remove(root, entry_id)


def _remove(root: Path, entry_id: str) -> None:
    package, meta_path = _paths(root, entry_id)
    # Description first: without it the entry is never listed.
    try:
        meta_path.unlink()
    except FileNotFoundError:
        pass
    shutil.rmtree(package, ignore_errors=True)


class Autosaver:
    """Background writer for one editor session's recovery entry.

    ``checkpoint`` is cheap to call after every UI action: it returns at once
    and the newest pending revision wins if several arrive during a write.
    Failures are logged, never raised into the editor.
    """

    def __init__(self, title: str, root: Path | None = None,
                 entry_id: str | None = None) -> None:
        # Pass ``entry_id`` to adopt a restored entry, so later edits replace it.
        if entry_id is not None and not _valid_id(entry_id):
            raise ValueError('invalid recovery entry id')
        self.root = Path(root) if root is not None else recovery_dir()
        self.entry_id = entry_id or uuid.uuid4().hex
        self.title = title or 'Untitled'
        self._cond = threading.Condition()
        self._pending = None
        # Revision on disk: 0 none, -1 an entry this session restored.
        self._written = -1 if entry_id is not None else 0
        self._busy = False
        self._closed = False
        self._thread = None
        self.last_error: BaseException | None = None
        with _active_lock:
            _active_ids.add(self.entry_id)

    @property
    def package(self) -> Path:
        return _paths(self.root, self.entry_id)[0]

    def checkpoint(self, history) -> None:
        """Queue the history's current revision, or drop the entry if clean."""
        if history.dirty:
            revision = history.revision
            with self._cond:
                if self._closed:
                    return
                if revision == self._written and self._pending is None:
                    return
                self._pending = (revision, history.document)
                if self._thread is None:
                    self._thread = threading.Thread(
                        target=self._run, name='layer-editor-autosave', daemon=True)
                    self._thread.start()
                self._cond.notify_all()
        else:
            self._clear()

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until queued writes finish; ``False`` on timeout."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._pending is not None or self._busy:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def close(self, keep: bool) -> None:
        """Stop the writer. ``keep`` leaves the latest revision on disk."""
        if keep:
            self.flush()
        with self._cond:
            self._closed = True
            self._pending = None
            self._cond.notify_all()
        self.flush()
        if not keep:
            _remove(self.root, self.entry_id)
        with _active_lock:
            _active_ids.discard(self.entry_id)

    def _clear(self) -> None:
        with self._cond:
            self._pending = None
            while self._busy:
                self._cond.wait()
            if self._written:
                self._written = 0
                _remove(self.root, self.entry_id)

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._closed:
                    self._cond.wait()
                if self._pending is None:
                    return
                revision, document = self._pending
                self._pending = None
                self._busy = True
            try:
                self._write(document)
                written = revision
            except Exception as exc:  # noqa: BLE001 - disk full, permissions: keep editing
                self.last_error = exc
                written = None
                _logger.warning('Layer Editor autosave failed: %s', exc)
            with self._cond:
                if written is not None:
                    self._written = written
                self._busy = False
                self._cond.notify_all()

    def _write(self, document) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        package, meta_path = _paths(self.root, self.entry_id)
        save_project(document, package)
        meta = {'version': _META_VERSION, 'title': self.title,
                'width': document.width, 'height': document.height,
                'layers': len(document.layers), 'updated': time.time(),
                'updated_utc': datetime.now(timezone.utc).isoformat(timespec='seconds')}
        staging = meta_path.with_name(f'.{meta_path.name}.{uuid.uuid4().hex[:8]}')
        staging.write_text(json.dumps(meta, indent=2), 'utf-8')
        os.replace(staging, meta_path)
