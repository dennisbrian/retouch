"""Tests for retouch/lut.py — LUTRegistry hot-load mechanism."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import retouch.lut as lut_mod
from retouch.lut import (
    CubeLUT,
    LUTRegistry,
    get_registry,
    load_cube,
    watch_luts_dir,
)


MINIMAL_CUBE_2 = (
    "LUT_3D_SIZE 2\n"
    "0.0 0.0 0.0\n"
    "0.2 0.0 0.0\n"
    "0.0 0.3 0.0\n"
    "0.2 0.3 0.0\n"
    "0.0 0.0 0.5\n"
    "0.2 0.0 0.5\n"
    "0.0 0.3 0.5\n"
    "0.2 0.3 0.5\n"
)

MODIFIED_CUBE_2 = (
    "LUT_3D_SIZE 2\n"
    "0.0 0.0 0.0\n"
    "0.9 0.0 0.0\n"
    "0.0 0.8 0.0\n"
    "0.9 0.8 0.0\n"
    "0.0 0.0 0.7\n"
    "0.9 0.0 0.7\n"
    "0.0 0.8 0.7\n"
    "0.9 0.8 0.7\n"
)


def _write_cube(path: Path, content: str = MINIMAL_CUBE_2, mtime: float = 1_000_000.0) -> None:
    path.write_text(content)
    os.utime(path, (mtime, mtime))


class TestLUTRegistryInit:
    def test_initial_cache_is_empty(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        assert reg.cache_size == 0
        assert reg.luts_dir == tmp_path

    def test_default_dir_is_canonical_luts_dir(self, monkeypatch):
        monkeypatch.setattr(lut_mod, "luts_dir", lambda: Path("/tmp/fake_luts"))
        reg = LUTRegistry()
        assert reg.luts_dir == Path("/tmp/fake_luts")

    def test_luts_dir_property_uses_default(self, monkeypatch):
        sentinel = Path("/tmp/sentinel_luts")
        monkeypatch.setattr(lut_mod, "luts_dir", lambda: sentinel)
        reg = LUTRegistry()
        assert reg.luts_dir == sentinel


class TestLUTRegistryRegister:
    def test_register_then_get_returns_lut(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        lut = CubeLUT(2)
        reg.register(lut, "memory_lut")
        got = reg.get("memory_lut")
        assert got is lut

    def test_register_default_name_is_unique(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        first = reg.register(CubeLUT(2))
        second = reg.register(CubeLUT(2))
        assert first != second
        assert reg.get(first) is not reg.get(second)

    def test_register_strips_cube_suffix(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        lut = CubeLUT(2)
        name = reg.register(lut, "foo.cube")
        assert name == "foo"
        assert reg.get("foo.cube") is lut
        assert reg.get("foo") is lut

    def test_registered_entry_not_replaced_by_disk(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        lut = CubeLUT(2)
        reg.register(lut, "kodak")
        _write_cube(tmp_path / "kodak.cube")
        assert reg.get("kodak") is lut


class TestLUTRegistryReload:
    def test_reload_clears_cache(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        _write_cube(tmp_path / "alpha.cube")
        reg.get("alpha")
        assert reg.cache_size == 1
        reg.reload()
        assert reg.cache_size == 0

    def test_reload_clears_registered_entries(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        reg.register(CubeLUT(2), "memory")
        assert reg.cache_size == 1
        reg.reload()
        assert reg.cache_size == 0
        with pytest.raises(FileNotFoundError):
            reg.get("memory")

    def test_reload_resets_poll_baseline(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        _write_cube(tmp_path / "alpha.cube", mtime=2_000_000.0)
        first = reg.poll_changes()
        assert first == []
        _write_cube(tmp_path / "beta.cube", mtime=3_000_000.0)
        reg.reload()
        first_after_reload = reg.poll_changes()
        assert first_after_reload == []


class TestLUTRegistryMtimeInvalidation:
    def test_mtime_change_invalidates_cache(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        p = tmp_path / "test.cube"
        _write_cube(p, MINIMAL_CUBE_2, mtime=1_000_000.0)
        first = reg.get("test")
        first_id = id(first.array)
        _write_cube(p, MODIFIED_CUBE_2, mtime=2_000_000.0)
        second = reg.get("test")
        second_id = id(second.array)
        assert first_id != second_id, "cache should be invalidated on mtime change"
        np.testing.assert_allclose(second.array[1, 0, 0], (0.7, 0.0, 0.0), atol=1e-6)

    def test_unchanged_mtime_uses_cache(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        p = tmp_path / "stable.cube"
        _write_cube(p, mtime=1_500_000.0)
        first = reg.get("stable")
        second = reg.get("stable")
        assert first is second

    def test_mtime_only_change_no_content_rewrite(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        p = tmp_path / "bumped.cube"
        _write_cube(p, mtime=1_000_000.0)
        first = reg.get("bumped")
        os.utime(p, (3_000_000.0, 3_000_000.0))
        second = reg.get("bumped")
        assert first is not second


class TestLUTRegistryListAvailable:
    def test_empty_dir_returns_empty_list(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        assert reg.list_available() == []

    def test_returns_actual_files_in_dir(self, tmp_path):
        _write_cube(tmp_path / "zeta.cube", mtime=1_000_000.0)
        _write_cube(tmp_path / "alpha.cube", mtime=2_000_000.0)
        (tmp_path / "ignore.txt").write_text("not a lut")
        _write_cube(tmp_path / "beta.cube", mtime=3_000_000.0)
        reg = LUTRegistry(tmp_path)
        assert reg.list_available() == ["alpha", "beta", "zeta"]

    def test_reflects_new_files_without_reload(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        assert reg.list_available() == []
        _write_cube(tmp_path / "fresh.cube", mtime=1_000_000.0)
        assert reg.list_available() == ["fresh"]


class TestLUTRegistryPollChanges:
    def test_first_call_returns_empty(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        _write_cube(tmp_path / "a.cube", mtime=1_000_000.0)
        assert reg.poll_changes() == []

    def test_detects_added_file(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        _write_cube(tmp_path / "a.cube", mtime=1_000_000.0)
        reg.poll_changes()
        _write_cube(tmp_path / "b.cube", mtime=2_000_000.0)
        assert reg.poll_changes() == ["b"]

    def test_detects_removed_file(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        p = tmp_path / "a.cube"
        _write_cube(p, mtime=1_000_000.0)
        reg.poll_changes()
        p.unlink()
        assert reg.poll_changes() == ["a"]

    def test_detects_mtime_modified_file(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        p = tmp_path / "a.cube"
        _write_cube(p, mtime=1_000_000.0)
        reg.poll_changes()
        os.utime(p, (5_000_000.0, 5_000_000.0))
        assert reg.poll_changes() == ["a"]

    def test_no_change_returns_empty(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        _write_cube(tmp_path / "a.cube", mtime=1_000_000.0)
        reg.poll_changes()
        assert reg.poll_changes() == []

    def test_poll_changes_evicts_from_cache(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        p = tmp_path / "a.cube"
        _write_cube(p, mtime=1_000_000.0)
        reg.get("a")
        assert reg.cache_size == 1
        reg.poll_changes()
        os.utime(p, (9_000_000.0, 9_000_000.0))
        reg.poll_changes()
        assert reg.cache_size == 0


class TestLUTRegistryGetMissing:
    def test_missing_stem_raises(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        with pytest.raises(FileNotFoundError):
            reg.get("nonexistent")

    def test_get_after_delete_falls_back_to_cached(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        p = tmp_path / "transient.cube"
        _write_cube(p, mtime=1_000_000.0)
        first = reg.get("transient")
        p.unlink()
        second = reg.get("transient")
        assert second is first


class TestLUTRegistryGetByPath:
    def test_get_with_cube_suffix(self, tmp_path):
        reg = LUTRegistry(tmp_path)
        _write_cube(tmp_path / "foo.cube", mtime=1_000_000.0)
        lut = reg.get("foo.cube")
        assert isinstance(lut, CubeLUT)
        assert lut.size == 2


class TestGetRegistrySingleton:
    def test_returns_same_instance(self):
        a = get_registry()
        b = get_registry()
        assert a is b

    def test_default_registry_uses_luts_dir(self, monkeypatch):
        monkeypatch.setattr(lut_mod, "_DEFAULT_REGISTRY", LUTRegistry(Path("/tmp/x")))
        sentinel = get_registry()
        assert sentinel.luts_dir == Path("/tmp/x")


class TestWatchLutsDir:
    def test_watch_luts_dir_returns_thread(self):
        callback_calls: list = []

        def cb(stem: str) -> None:
            callback_calls.append(stem)

        t = watch_luts_dir(cb, interval=60.0)
        try:
            assert t.is_alive()
            assert t.daemon is True
            assert t.name == "lut-watcher"
        finally:
            pass

    def test_watch_luts_dir_invalid_interval_raises(self):
        with pytest.raises(ValueError):
            watch_luts_dir(lambda s: None, interval=0.0)
        with pytest.raises(ValueError):
            watch_luts_dir(lambda s: None, interval=-1.0)


class TestIntegrationWithExistingAPI:
    def test_list_available_luts_still_works(self, tmp_path, monkeypatch):
        _write_cube(tmp_path / "alpha.cube", mtime=1_000_000.0)
        _write_cube(tmp_path / "beta.cube", mtime=2_000_000.0)
        monkeypatch.setattr(lut_mod, "luts_dir", lambda: tmp_path)
        from retouch.lut import list_available_luts
        assert list_available_luts() == ["alpha", "beta"]

    def test_load_cube_unchanged(self, tmp_path):
        p = tmp_path / "plain.cube"
        p.write_text(MINIMAL_CUBE_2)
        lut = load_cube(p)
        assert lut.size == 2
