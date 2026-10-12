"""Session-owned temporary workspace primitives for the GUI.

The GUI currently creates temporary files in process-global locations.  This
module provides the lifecycle boundary that a later GUI integration can use
without allowing one browser session to remove another session's files.

The public object is :class:`SessionWorkspace`.  A workspace has a directory
whose name is derived only from a SHA-256 digest of the request/session hash;
the raw hash is never used as a path component.  Request directories and
artifact files add fresh random tokens and are allocated atomically with
``O_EXCL``.  Every destructive operation checks the session ownership marker
and resolved path containment before deleting anything.

This module has no Gradio, engine, or desktop dependencies.  It is suitable
for use from request handlers as well as ``Blocks.unload``/desktop shutdown
callbacks.  ``shutdown`` and ``unload`` are intentionally idempotent.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union


WORKSPACE_SCHEMA_VERSION = 1
OWNER_FILENAME = ".retouch-workspace-owner.json"
SESSION_DIR_PREFIX = "session-"
REQUEST_DIR_PREFIX = "request-"
DEFAULT_ROOT_NAME = "retouch-gui-workspaces"
_MAX_ALLOCATION_ATTEMPTS = 64
_LABEL_PATTERN = re.compile(r"[^A-Za-z0-9_-]+")
_SUFFIX_PATTERN = re.compile(r"\.[A-Za-z0-9][A-Za-z0-9._-]{0,31}\Z")

PathLike = Union[str, os.PathLike]


class WorkspaceOwnershipError(PermissionError):
    """Raised when a path is not owned by the current session workspace."""


class WorkspaceClosedError(RuntimeError):
    """Raised when an operation is attempted after workspace shutdown."""


def _sha256_text(value: Any) -> str:
    """Return a stable digest for an untrusted session/request identifier."""

    if value is None:
        raise ValueError("session/request identifiers cannot be None")
    text = str(value)
    if not text:
        raise ValueError("session/request identifiers cannot be empty")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def default_workspace_root() -> Path:
    """Return the default ephemeral root used by GUI sessions.

    ``RETOUCH_GUI_WORKSPACE_ROOT`` is an explicit integration seam for the
    desktop app and tests.  It is not consulted for ownership: multiple
    sessions are expected to share this root, while each session owns only
    its digest-named child.
    """

    configured = os.environ.get("RETOUCH_GUI_WORKSPACE_ROOT")
    if configured:
        return Path(configured).expanduser().absolute()
    return Path(tempfile.gettempdir()).absolute() / DEFAULT_ROOT_NAME


def _lexists(path: Path) -> bool:
    """Like ``Path.exists`` but also true for a broken symlink."""

    return os.path.lexists(os.fspath(path))


def _ensure_directory(path: Path, *, mode: int = 0o700) -> Path:
    """Create a private directory, rejecting symlinks and non-directories."""

    if path.is_symlink():
        raise WorkspaceOwnershipError("workspace path must not be a symlink: %s" % path)
    if _lexists(path):
        if not path.is_dir():
            raise NotADirectoryError(str(path))
        return path
    path.mkdir(parents=True, mode=mode, exist_ok=False)
    return path


def _safe_label(value: Any, *, fallback: str) -> str:
    """Turn a display label into one harmless filename component."""

    text = str(value).strip()
    text = _LABEL_PATTERN.sub("-", text).strip("-._")
    return (text or fallback)[:48]


def _safe_suffix(value: Any) -> str:
    """Validate/normalize a file suffix without accepting path separators."""

    if value in (None, ""):
        return ""
    suffix = str(value)
    if not suffix.startswith("."):
        suffix = "." + suffix
    if "/" in suffix or "\\" in suffix or "\x00" in suffix:
        raise ValueError("artifact suffix must be a single filename suffix")
    if not _SUFFIX_PATTERN.fullmatch(suffix):
        raise ValueError("invalid artifact suffix: %r" % value)
    return suffix


def _owner_payload(
    *,
    scope: str,
    session_digest: str,
    request_digest: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": WORKSPACE_SCHEMA_VERSION,
        "scope": scope,
        "session_hash_sha256": session_digest,
    }
    if request_digest is not None:
        payload["request_id_sha256"] = request_digest
    return payload


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, ValueError) as exc:
        raise WorkspaceOwnershipError("invalid workspace ownership marker: %s" % path) from exc
    if not isinstance(value, dict):
        raise WorkspaceOwnershipError("workspace ownership marker is not an object: %s" % path)
    return value


def _claim_directory(path: Path, payload: Dict[str, Any]) -> None:
    """Create or verify a directory ownership marker atomically."""

    _ensure_directory(path)
    marker = path / OWNER_FILENAME
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(os.fspath(marker), flags, 0o600)
    except FileExistsError:
        observed = _read_json(marker)
        if observed != payload:
            raise WorkspaceOwnershipError("workspace ownership mismatch: %s" % path)
        return

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.write("\n")
    except Exception:
        try:
            marker.unlink()
        except OSError:
            pass
        raise


def _remove_path(path: Path) -> int:
    """Remove one path without following a symlink at its root."""

    if not _lexists(path):
        return 0
    if path.is_symlink() or not path.is_dir():
        path.unlink()
        return 1
    shutil.rmtree(os.fspath(path))
    return 1


def _clear_directory(path: Path) -> int:
    """Delete direct children of a directory while retaining that directory."""

    if not _lexists(path):
        path.mkdir(parents=True, mode=0o700, exist_ok=False)
        return 0
    if path.is_symlink() or not path.is_dir():
        raise WorkspaceOwnershipError("cache path is not a directory: %s" % path)

    removed = 0
    for child in list(path.iterdir()):
        _remove_path(child)
        removed += 1
    return removed


class SessionWorkspace:
    """Private artifact and cache workspace for one request session.

    Parameters
    ----------
    session_hash:
        The opaque request/session identifier supplied by the GUI framework.
        It is hashed before being used in a path or marker.
    root:
        Shared parent for all sessions.  Tests and desktop integration should
        inject a temporary or application-specific root.
    """

    def __init__(self, session_hash: Any, *, root: Optional[PathLike] = None) -> None:
        self.session_hash = str(session_hash)
        self.session_hash_sha256 = _sha256_text(session_hash)
        self.root = (Path(root) if root is not None else default_workspace_root()).expanduser().absolute()
        self.session_dir = self.root / (SESSION_DIR_PREFIX + self.session_hash_sha256)
        self.cache_dir = self.session_dir / "cache"
        self.artifacts_dir = self.session_dir / "artifacts"
        self.requests_dir = self.session_dir / "requests"
        self._lock = threading.RLock()
        self._closed = False

        _ensure_directory(self.root)
        _claim_directory(
            self.session_dir,
            _owner_payload(scope="session", session_digest=self.session_hash_sha256),
        )
        self._ensure_owned_subdirectory(self.cache_dir)
        self._ensure_owned_subdirectory(self.artifacts_dir)
        self._ensure_owned_subdirectory(self.requests_dir)

    @property
    def is_closed(self) -> bool:
        """Whether this lifecycle object has already been shut down."""

        return self._closed

    @property
    def session_id(self) -> str:
        """Return the non-sensitive digest used for the session directory."""

        return self.session_hash_sha256

    def _ensure_open(self) -> None:
        if self._closed:
            raise WorkspaceClosedError("session workspace is already closed")

    def _ensure_owned_subdirectory(self, path: Path) -> Path:
        self._assert_session_owner()
        return _ensure_directory(path)

    def _assert_session_owner(self) -> None:
        if not _lexists(self.session_dir):
            raise WorkspaceOwnershipError("session workspace no longer exists")
        if self.session_dir.is_symlink() or not self.session_dir.is_dir():
            raise WorkspaceOwnershipError("invalid session workspace: %s" % self.session_dir)
        marker = self.session_dir / OWNER_FILENAME
        observed = _read_json(marker)
        expected = _owner_payload(scope="session", session_digest=self.session_hash_sha256)
        if observed != expected:
            raise WorkspaceOwnershipError("session workspace is owned by another session")

    def _assert_owned_path(self, path: PathLike, *, allow_session_root: bool = False) -> Path:
        """Validate resolved containment and the session marker.

        The returned path is the original lexical path, so deleting a symlink
        removes the link itself rather than following it.  Containment is
        checked on the resolved path to reject links that escape the session.
        """

        self._assert_session_owner()
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.session_dir / candidate
        candidate = candidate.absolute()
        session_root = self.session_dir.resolve()
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(session_root)
        except ValueError as exc:
            raise WorkspaceOwnershipError(
                "path is outside this session workspace: %s" % candidate
            ) from exc
        if not allow_session_root and resolved == session_root:
            raise WorkspaceOwnershipError("the session workspace root is not an artifact")
        return candidate

    def _assert_request_directory(self, request_dir: PathLike) -> Path:
        candidate = self._assert_owned_path(request_dir)
        requests_root = self.requests_dir.resolve()
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(requests_root)
        except ValueError as exc:
            raise WorkspaceOwnershipError("path is not a request workspace: %s" % candidate) from exc
        if resolved == requests_root:
            raise WorkspaceOwnershipError("the requests root cannot be cleaned directly")
        if not _lexists(candidate):
            return candidate
        if candidate.is_symlink() or not candidate.is_dir():
            raise WorkspaceOwnershipError("request workspace is not a directory: %s" % candidate)
        marker = candidate / OWNER_FILENAME
        observed = _read_json(marker)
        if observed.get("scope") != "request" or observed.get("session_hash_sha256") != self.session_hash_sha256:
            raise WorkspaceOwnershipError("request workspace is owned by another session")
        return candidate

    def request_workspace(self, request_id: Optional[Any] = None) -> Path:
        """Create a fresh request directory owned by this session.

        The request id is represented by a digest-only hint plus a fresh
        random token.  Consequently repeated request ids still receive
        independent directories and untrusted ids cannot escape the root.
        """

        with self._lock:
            self._ensure_open()
            self._assert_session_owner()
            request_digest = _sha256_text(request_id) if request_id is not None else None
            hint = request_digest[:20] if request_digest else "anonymous"
            for _ in range(_MAX_ALLOCATION_ATTEMPTS):
                token = secrets.token_hex(16)
                candidate = self.requests_dir / (REQUEST_DIR_PREFIX + hint + "-" + token)
                if _lexists(candidate):
                    continue
                try:
                    candidate.mkdir(mode=0o700, exist_ok=False)
                except FileExistsError:
                    continue
                try:
                    _claim_directory(
                        candidate,
                        _owner_payload(
                            scope="request",
                            session_digest=self.session_hash_sha256,
                            request_digest=request_digest,
                        ),
                    )
                    _ensure_directory(candidate / "artifacts")
                    _ensure_directory(candidate / "cache")
                except Exception:
                    shutil.rmtree(os.fspath(candidate), ignore_errors=True)
                    raise
                return candidate
            raise FileExistsError("unable to allocate a unique request workspace")

    # A shorter alias is convenient for request handlers and keeps the API
    # readable when a caller already has a request id in a local variable.
    new_request = request_workspace

    def _artifact_directory(
        self,
        *,
        request_dir: Optional[PathLike],
        request_id: Optional[Any],
        cache: bool,
    ) -> Path:
        if request_dir is not None and request_id is not None:
            raise ValueError("pass request_dir or request_id, not both")
        if request_id is not None:
            request_dir = self.request_workspace(request_id)
        if request_dir is None:
            directory = self.cache_dir if cache else self.artifacts_dir
            self._assert_owned_path(directory)
            return _ensure_directory(directory)

        request_root = self._assert_request_directory(request_dir)
        directory = request_root / ("cache" if cache else "artifacts")
        self._assert_owned_path(directory)
        return _ensure_directory(directory)

    def allocate_artifact(
        self,
        prefix: Any = "artifact",
        suffix: Any = "",
        *,
        request_dir: Optional[PathLike] = None,
        request_id: Optional[Any] = None,
    ) -> Path:
        """Atomically reserve a collision-safe artifact filename.

        The returned path is an empty, owner-created file.  Callers may write
        their result to it normally; reserving it first closes the race where
        two queued callbacks otherwise choose the same output name.
        """

        with self._lock:
            self._ensure_open()
            directory = self._artifact_directory(
                request_dir=request_dir,
                request_id=request_id,
                cache=False,
            )
            label = _safe_label(prefix, fallback="artifact")
            normalized_suffix = _safe_suffix(suffix)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            for _ in range(_MAX_ALLOCATION_ATTEMPTS):
                candidate = directory / (
                    label + "-" + secrets.token_hex(16) + normalized_suffix
                )
                try:
                    fd = os.open(os.fspath(candidate), flags, 0o600)
                except FileExistsError:
                    continue
                os.close(fd)
                return candidate
            raise FileExistsError("unable to allocate a unique artifact filename")

    # These aliases make the primitive easy to discover without creating
    # multiple implementations with subtly different ownership behavior.
    new_artifact = allocate_artifact
    artifact_path = allocate_artifact

    def allocate_cache(
        self,
        prefix: Any = "cache",
        suffix: Any = "",
        *,
        request_dir: Optional[PathLike] = None,
        request_id: Optional[Any] = None,
    ) -> Path:
        """Atomically reserve a cache filename in the session-owned cache."""

        with self._lock:
            self._ensure_open()
            directory = self._artifact_directory(
                request_dir=request_dir,
                request_id=request_id,
                cache=True,
            )
            label = _safe_label(prefix, fallback="cache")
            normalized_suffix = _safe_suffix(suffix)
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            for _ in range(_MAX_ALLOCATION_ATTEMPTS):
                candidate = directory / (
                    label + "-" + secrets.token_hex(16) + normalized_suffix
                )
                try:
                    fd = os.open(os.fspath(candidate), flags, 0o600)
                except FileExistsError:
                    continue
                os.close(fd)
                return candidate
            raise FileExistsError("unable to allocate a unique cache filename")

    def delete_cache(self, path: Optional[PathLike] = None) -> int:
        """Delete session-owned cache contents, or one cache entry.

        ``path`` must be inside this session's cache directory.  A path from
        another session, a traversal path, or a symlink resolving outside the
        cache is rejected before any deletion occurs.  The operation is
        idempotent and returns the number of direct entries removed.
        """

        with self._lock:
            self._ensure_open()
            cache_root = self._assert_owned_path(self.cache_dir)
            cache_root = _ensure_directory(cache_root)
            target = cache_root if path is None else Path(path)
            if not target.is_absolute():
                target = cache_root / target
            target = self._assert_owned_path(target)
            cache_resolved = cache_root.resolve()
            target_resolved = target.resolve(strict=False)
            try:
                target_resolved.relative_to(cache_resolved)
            except ValueError as exc:
                raise WorkspaceOwnershipError(
                    "cache path is outside this session cache: %s" % target
                ) from exc
            if target_resolved == cache_resolved:
                return _clear_directory(cache_root)
            return _remove_path(target)

    def cleanup(self, request_dir: Optional[PathLike] = None) -> int:
        """Clean one request or the complete session workspace.

        A request cleanup leaves the session usable.  A workspace cleanup
        removes the session directory and marks this object closed.  Repeated
        calls after shutdown are no-ops, which makes this safe to register
        with both application shutdown and ``Blocks.unload``.
        """

        with self._lock:
            if self._closed:
                return 0
            if request_dir is not None:
                target = self._assert_request_directory(request_dir)
                if not _lexists(target):
                    return 0
                return _remove_path(target)

            if not _lexists(self.session_dir):
                self._closed = True
                return 0
            self._assert_session_owner()
            removed = _remove_path(self.session_dir)
            self._closed = True
            return removed

    def close(self) -> int:
        """Alias for complete session cleanup."""

        return self.cleanup()

    def shutdown(self) -> int:
        """Idempotent application-shutdown helper."""

        return self.cleanup()

    def unload(self) -> int:
        """Idempotent UI-unload helper."""

        return self.cleanup()

    def __enter__(self) -> "SessionWorkspace":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.shutdown()


def create_session_workspace(
    session_hash: Any,
    *,
    root: Optional[PathLike] = None,
) -> SessionWorkspace:
    """Factory used by future GUI integration points."""

    return SessionWorkspace(session_hash, root=root)


def cleanup_workspace(
    workspace: Optional[SessionWorkspace],
    request_dir: Optional[PathLike] = None,
) -> int:
    """Null-safe cleanup callback suitable for framework lifecycle hooks."""

    if workspace is None:
        return 0
    return workspace.cleanup(request_dir=request_dir)


def delete_cache(
    workspace: Optional[SessionWorkspace],
    path: Optional[PathLike] = None,
) -> int:
    """Null-safe module-level wrapper for :meth:`SessionWorkspace.delete_cache`."""

    if workspace is None:
        return 0
    return workspace.delete_cache(path=path)


def shutdown_workspace(workspace: Optional[SessionWorkspace]) -> int:
    """Idempotent null-safe shutdown callback."""

    if workspace is None:
        return 0
    return workspace.shutdown()


def unload_workspace(workspace: Optional[SessionWorkspace]) -> int:
    """Idempotent null-safe unload callback."""

    if workspace is None:
        return 0
    return workspace.unload()


__all__ = [
    "OWNER_FILENAME",
    "WORKSPACE_SCHEMA_VERSION",
    "SessionWorkspace",
    "WorkspaceClosedError",
    "WorkspaceOwnershipError",
    "cleanup_workspace",
    "create_session_workspace",
    "default_workspace_root",
    "delete_cache",
    "shutdown_workspace",
    "unload_workspace",
]
