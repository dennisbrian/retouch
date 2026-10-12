"""Contract tests for the session-scoped GUI preview cache."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass

import pytest

from retouch.gui_preview_cache import (
    GuiPreviewCache,
    PreviewCacheKey,
    SourceIdentity,
    make_preview_cache_key,
    render_resolution_key,
    source_identity,
)


@dataclass
class FakeArray:
    shape: tuple
    dtype: str = "uint8"
    nbytes: int = 0


@dataclass
class FakeFaceContext:
    index: int
    landmarks: object
    face_image: object


def _source(tmp_path, name="portrait.jpg", payload=b"source-v1"):
    path = tmp_path / name
    path.write_bytes(payload)
    return path, source_identity(path)


def _key(identity, resolution=(320, 240), **kwargs):
    return make_preview_cache_key(identity, resolution, **kwargs)


def test_source_identity_contains_path_mtime_size_and_content_hash(tmp_path):
    path, identity = _source(tmp_path)

    assert identity.path == str(path.resolve())
    assert identity.mtime_ns == path.stat().st_mtime_ns
    assert identity.size_bytes == len(b"source-v1")
    assert len(identity.content_sha256) == 64
    assert identity.identity_key == SourceIdentity.from_path(path).identity_key

    path.write_bytes(b"source-v2")
    changed = SourceIdentity.from_path(path)
    assert changed.content_sha256 != identity.content_sha256
    assert changed != identity


def test_resolution_and_cache_key_are_deterministic_and_dimension_sensitive(tmp_path):
    _, identity = _source(tmp_path)
    assert render_resolution_key((320, 240)) == "320x240"
    assert render_resolution_key({"height": 240, "width": 320}) == "320x240"
    assert render_resolution_key(" 320 x 240 ") == "320x240"

    base = _key(identity, geometry={"crop": [0, 0, 1, 1]})
    same = _key(identity, geometry={"crop": [0, 0, 1, 1]})
    assert base.digest == same.digest
    assert base.changed_fields(same) == ()

    assert base.digest != _key(identity, (640, 480), geometry={"crop": [0, 0, 1, 1]}).digest
    assert base.digest != _key(identity, orientation=6, geometry={"crop": [0, 0, 1, 1]}).digest
    assert base.digest != _key(identity, geometry={"crop": [0, 0, 0.9, 1]}).digest
    assert base.digest != _key(identity, optical_correction=True, geometry={"crop": [0, 0, 1, 1]}).digest
    assert base.digest != _key(identity, color_contract="source-adobe-rgb").digest
    assert base.digest != _key(identity, bit_depth=16).digest
    assert base.digest != _key(identity, detector_backend="tasks-v2").digest


def test_put_get_returns_deepcopy_safe_compact_values(tmp_path):
    _, identity = _source(tmp_path)
    key = _key(identity)
    huge = FakeArray(shape=(6240, 4160, 3), nbytes=6240 * 4160 * 3)
    face = FakeFaceContext(
        index=0,
        landmarks=[{"x": 0.25, "y": 0.5}],
        face_image=huge,
    )
    cache = GuiPreviewCache(max_entries=2)
    stored = cache.put(
        key,
        decoded_source_metadata={"width": 6240, "height": 4160, "dtype": "uint8"},
        face_contexts=[face],
        decoded_source=huge,
        runtime_face_contexts=[face],
        preview_reference={"uri": "/tmp/preview.webp", "sha256": "a" * 64},
        checkpoint_reference={"uri": "/tmp/checkpoint.npz", "size_bytes": 1234},
    )

    assert stored.cache_key == key.digest
    assert stored.face_contexts[0]["face_image"]["__type__"] == "array_ref"
    assert stored.face_contexts[0]["face_image"]["nbytes"] == huge.nbytes
    assert not isinstance(stored.face_contexts[0]["face_image"], FakeArray)
    assert stored.decoded_source is not None
    assert stored.runtime_face_contexts
    assert cache.has_runtime_entry(key) is True

    returned = cache.get(key)
    assert returned is not None
    returned.decoded_source_metadata["width"] = 1
    returned.face_contexts[0]["landmarks"][0]["x"] = 0.99
    fresh = cache.get(key)
    assert fresh.decoded_source_metadata["width"] == 6240
    assert fresh.face_contexts[0]["landmarks"][0]["x"] == 0.25

    cloned = copy.deepcopy(cache)
    assert cloned.get(key) == fresh
    state = cache.to_state()
    encoded = json.dumps(state, sort_keys=True)
    assert len(encoded) < 20_000
    assert "FakeArray" not in encoded
    assert "face_image" in encoded
    assert state["entries"][0]["decoded_source"]["__type__"] == "array_ref"

    restored = GuiPreviewCache.from_state(state)
    restored_entry = restored.get(key)
    assert restored_entry is not None
    assert restored_entry.face_contexts == fresh.face_contexts
    assert restored.has_runtime_entry(key) is False

    cache.set_latest_render(
        {"render_revision": 4, "native": {"width": 6240, "height": 4160}},
        face_contexts=[face],
    )
    latest = cache.latest_render_evidence
    assert latest["render_revision"] == 4
    assert cache.latest_render_face_contexts[0]["face_image"]["__type__"] == "array_ref"
    roundtrip = GuiPreviewCache.from_state(cache.to_state())
    assert roundtrip.latest_render_evidence["render_revision"] == 4
    assert roundtrip.latest_render_face_contexts[0]["face_image"]["__type__"] == "array_ref"


def test_face_contexts_and_entries_are_bounded(tmp_path):
    _, identity = _source(tmp_path)
    cache = GuiPreviewCache(max_entries=2, max_face_contexts=2)
    key1 = _key(identity, (320, 240))
    key2 = _key(identity, (640, 480))
    key3 = _key(identity, (1280, 960))
    contexts = [{"index": index} for index in range(4)]

    entry = cache.put(key1, face_contexts=contexts)
    assert len(entry.face_contexts) == 2
    assert entry.face_contexts_truncated is True
    cache.put(key2)
    assert cache.get(key1) is not None
    cache.put(key3)
    assert cache.get(key2) is None
    assert cache.get(key1) is not None
    assert cache.size == 2


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "resolution",
        "orientation",
        "geometry",
        "optical_correction",
    ],
)
def test_invalidate_for_change_removes_old_variant(tmp_path, change):
    path, identity = _source(tmp_path)
    previous = _key(identity, geometry={"crop": [0, 0, 1, 1]})
    if change == "source":
        path.write_bytes(b"source-v2")
        current = _key(SourceIdentity.from_path(path), geometry={"crop": [0, 0, 1, 1]})
    elif change == "resolution":
        current = _key(identity, (640, 480), geometry={"crop": [0, 0, 1, 1]})
    elif change == "orientation":
        current = _key(identity, orientation=6, geometry={"crop": [0, 0, 1, 1]})
    elif change == "geometry":
        current = _key(identity, geometry={"crop": [0, 0, 0.9, 1]})
    else:
        current = _key(
            identity,
            optical_correction=True,
            geometry={"crop": [0, 0, 1, 1]},
        )

    cache = GuiPreviewCache()
    cache.put(previous)
    assert cache.invalidate_for_change(previous, current) == 1
    assert cache.get(previous) is None
    assert cache.get(current) is None


def test_explicit_filters_invalidate_only_matching_source_and_variant(tmp_path):
    _, first = _source(tmp_path, "first.jpg", b"first")
    _, second = _source(tmp_path, "second.jpg", b"second")
    first_key = _key(first, orientation=1)
    first_rotated = _key(first, orientation=6)
    second_key = _key(second, orientation=1)
    cache = GuiPreviewCache()
    cache.put(first_key)
    cache.put(first_rotated)
    cache.put(second_key)

    assert cache.invalidate(source_identity=first, orientation=1) == 1
    assert cache.get(first_key) is None
    assert cache.get(first_rotated) is not None
    assert cache.get(second_key) is not None

    assert cache.invalidate(source_path=second.path) == 1
    assert cache.get(second_key) is None
