"""Focused lifecycle and ownership tests for session-owned GUI workspaces."""

from __future__ import annotations

import json

import pytest

from retouch.gui_workspace import (
    OWNER_FILENAME,
    SessionWorkspace,
    WorkspaceClosedError,
    WorkspaceOwnershipError,
    cleanup_workspace,
    delete_cache,
    shutdown_workspace,
    unload_workspace,
)


def test_sessions_have_private_roots_and_cannot_clean_each_other(tmp_path):
    first = SessionWorkspace("browser-session-A", root=tmp_path)
    second = SessionWorkspace("browser-session-B", root=tmp_path)
    first_request = first.request_workspace("render-1")
    second_request = second.request_workspace("render-1")
    first_cache = first.allocate_cache("preview", ".webp")
    second_cache = second.allocate_cache("preview", ".webp")
    first_artifact = first.allocate_artifact("render", ".jpg", request_dir=first_request)
    second_artifact = second.allocate_artifact("render", ".jpg", request_dir=second_request)

    assert first.session_dir != second.session_dir
    assert first.session_dir.name != second.session_dir.name
    assert first_cache.exists() and second_cache.exists()
    assert first_artifact.exists() and second_artifact.exists()

    with pytest.raises(WorkspaceOwnershipError):
        first.delete_cache(second_cache)
    with pytest.raises(WorkspaceOwnershipError):
        first.cleanup(request_dir=second_request)
    with pytest.raises(WorkspaceOwnershipError):
        second.delete_cache(first_cache)

    assert second_cache.exists()
    assert second_request.exists()
    assert first_cache.exists()
    assert first_request.exists()

    first.shutdown()
    assert not first.session_dir.exists()
    assert second.session_dir.exists()
    assert second_cache.exists()
    assert second_artifact.exists()
    second.unload()
    assert not second.session_dir.exists()


def test_artifact_and_request_names_are_collision_safe(tmp_path):
    workspace = SessionWorkspace("same-session", root=tmp_path)
    request_one = workspace.request_workspace("same-request")
    request_two = workspace.request_workspace("same-request")

    artifact_paths = {
        workspace.allocate_artifact("preview output", ".jpg", request_dir=request_one)
        for _ in range(100)
    }
    cache_paths = {
        workspace.allocate_cache("preview output", ".webp")
        for _ in range(100)
    }

    assert len(artifact_paths) == 100
    assert len(cache_paths) == 100
    assert request_one != request_two
    assert all(path.name.startswith("preview-output-") for path in artifact_paths)
    assert all(path.suffix == ".jpg" for path in artifact_paths)
    assert all(path.suffix == ".webp" for path in cache_paths)
    assert all(path.parent == request_one / "artifacts" for path in artifact_paths)
    assert all(path.parent == workspace.cache_dir for path in cache_paths)

    workspace.shutdown()


def test_ownership_marker_does_not_store_raw_session_hash(tmp_path):
    raw_session_hash = "session/with spaces?and-secrets"
    workspace = SessionWorkspace(raw_session_hash, root=tmp_path)
    marker = workspace.session_dir / OWNER_FILENAME
    payload = json.loads(marker.read_text(encoding="utf-8"))

    assert raw_session_hash not in marker.read_text(encoding="utf-8")
    assert payload["scope"] == "session"
    assert payload["session_hash_sha256"] == workspace.session_id
    assert raw_session_hash not in workspace.session_dir.name
    workspace.shutdown()


def test_cache_cleanup_is_scoped_and_lifecycle_helpers_are_idempotent(tmp_path):
    workspace = SessionWorkspace("lifecycle-session", root=tmp_path)
    cache_file = workspace.allocate_cache("old", ".bin")
    artifact_file = workspace.allocate_artifact("keep", ".jpg")
    request_dir = workspace.request_workspace("request")
    request_file = workspace.allocate_artifact("request", ".jpg", request_dir=request_dir)

    assert delete_cache(workspace) == 1
    assert not cache_file.exists()
    assert artifact_file.exists()
    assert request_file.exists()
    assert workspace.cleanup(request_dir=request_dir) == 1
    assert not request_dir.exists()
    assert workspace.cleanup(request_dir=request_dir) == 0

    assert shutdown_workspace(workspace) == 1
    assert unload_workspace(workspace) == 0
    assert cleanup_workspace(workspace) == 0
    assert not workspace.session_dir.exists()

    with pytest.raises(WorkspaceClosedError):
        workspace.allocate_artifact("after-close")


def test_traversal_and_external_paths_are_rejected_before_deletion(tmp_path):
    workspace = SessionWorkspace("safe-session", root=tmp_path / "workspaces")
    outside = tmp_path / "outside.txt"
    outside.write_text("must survive", encoding="utf-8")
    cache_file = workspace.allocate_cache("owned", ".bin")

    with pytest.raises(WorkspaceOwnershipError):
        workspace.delete_cache(outside)
    with pytest.raises(WorkspaceOwnershipError):
        workspace.delete_cache(workspace.cache_dir / ".." / ".." / "outside.txt")

    assert cache_file.exists()
    assert outside.exists()
    assert outside.read_text(encoding="utf-8") == "must survive"
    workspace.shutdown()
