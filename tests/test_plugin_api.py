"""Tests for retouch.plugin_api (T4 Plugin API v0)."""
from __future__ import annotations

import os
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from retouch.stages import BaseStage, PipelineState, Stage, StageRegistry
from retouch.plugin_api import (
    BasePlugin,
    Plugin,
    PluginError,
    PluginManager,
    register_stage,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PLUGIN_TEMPLATE_CLASS = textwrap.dedent(
    """
    from retouch.plugin_api import BasePlugin
    from retouch.stages import BaseStage

    class PluginClass(BasePlugin):
        name = "{name}"
        version = "0.1.0"
        def __init__(self):
            self.initialized = False
            self.tore_down = False
        def init(self, engine):
            self.initialized = True
            self.engine = engine
        def register_stages(self, registry):
            registry.add(self._make_stage())
        def teardown(self):
            self.tore_down = True
        def _make_stage(self):
            stage = BaseStage()
            stage.name = "{stage_name}"
            stage.phase = "global"
            stage.run = lambda state: state
            return stage
    """
)

PLUGIN_TEMPLATE_FACTORY = textwrap.dedent(
    """
    from retouch.plugin_api import BasePlugin

    def create_plugin():
        return _P()

    class _P(BasePlugin):
        name = "factory_plugin"
        version = "0.2.0"
    """
)

PLUGIN_TEMPLATE_INSTANCE = textwrap.dedent(
    """
    from retouch.plugin_api import BasePlugin

    class _P(BasePlugin):
        name = "instance_plugin"
        version = "0.3.0"

    plugin = _P()
    """
)

PLUGIN_TEMPLATE_BROKEN = textwrap.dedent(
    """
    raise RuntimeError("boom on import")
    """
)

PLUGIN_TEMPLATE_NOT_A_PLUGIN = textwrap.dedent(
    """
    class NotAPlugin:
        pass
    PluginClass = NotAPlugin
    """
)


@pytest.fixture
def plugins_dir(tmp_path: Path) -> Path:
    d = tmp_path / "plugins"
    d.mkdir()
    return d


def _write_plugin(plugins_dir: Path, filename: str, body: str) -> Path:
    p = plugins_dir / filename
    p.write_text(body)
    return p


# ---------------------------------------------------------------------------
# Plugin protocol / base class
# ---------------------------------------------------------------------------

class TestPluginProtocol:
    def test_base_plugin_satisfies_protocol(self):
        plugin = BasePlugin()
        assert isinstance(plugin, Plugin)

    def test_custom_subclass_satisfies_protocol(self):
        class MyPlugin(BasePlugin):
            name = "mine"
            version = "1.0.0"

        assert isinstance(MyPlugin(), Plugin)

    def test_non_plugin_object_fails_protocol(self):
        class Empty:
            pass

        assert not isinstance(Empty(), Plugin)


# ---------------------------------------------------------------------------
# Discovery / loading
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_discover_empty_dir(self, plugins_dir: Path):
        mgr = PluginManager(plugins_dir=plugins_dir)
        assert mgr.discover() == []
        assert mgr.errors == []

    def test_discover_missing_dir(self, tmp_path: Path):
        mgr = PluginManager(plugins_dir=tmp_path / "nope")
        assert mgr.discover() == []
        assert mgr.errors == []

    def test_discover_class_plugin(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "alpha.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="alpha", stage_name="alpha_stage"))
        mgr = PluginManager(plugins_dir=plugins_dir)
        loaded = mgr.discover()
        assert len(loaded) == 1
        assert loaded[0].name == "alpha"
        assert loaded[0].version == "0.1.0"
        assert mgr.errors == []

    def test_discover_factory_plugin(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "factory.py", PLUGIN_TEMPLATE_FACTORY)
        mgr = PluginManager(plugins_dir=plugins_dir)
        loaded = mgr.discover()
        assert len(loaded) == 1
        assert loaded[0].name == "factory_plugin"

    def test_discover_instance_plugin(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "instance.py", PLUGIN_TEMPLATE_INSTANCE)
        mgr = PluginManager(plugins_dir=plugins_dir)
        loaded = mgr.discover()
        assert len(loaded) == 1
        assert loaded[0].name == "instance_plugin"

    def test_discover_skips_dunder_files(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "_helper.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="hidden", stage_name="h"))
        _write_plugin(plugins_dir, "visible.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="visible", stage_name="v"))
        mgr = PluginManager(plugins_dir=plugins_dir)
        loaded = mgr.discover()
        assert [p.name for p in loaded] == ["visible"]

    def test_discover_idempotent(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "alpha.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="alpha", stage_name="alpha_stage"))
        mgr = PluginManager(plugins_dir=plugins_dir)
        first = mgr.discover()
        second = mgr.discover()
        assert len(first) == 1
        assert second == []

    def test_discover_broken_plugin_recorded_not_raised(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "broken.py", PLUGIN_TEMPLATE_BROKEN)
        _write_plugin(plugins_dir, "good.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="good", stage_name="g"))
        mgr = PluginManager(plugins_dir=plugins_dir)
        loaded = mgr.discover()
        assert [p.name for p in loaded] == ["good"]
        assert len(mgr.errors) == 1
        assert "boom on import" in mgr.errors[0].message
        assert mgr.errors[0].exc_type == "RuntimeError"

    def test_discover_non_plugin_recorded(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "bad.py", PLUGIN_TEMPLATE_NOT_A_PLUGIN)
        mgr = PluginManager(plugins_dir=plugins_dir)
        assert mgr.discover() == []
        assert len(mgr.errors) == 1
        assert "Plugin protocol" in mgr.errors[0].message


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_init_all_calls_init(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "alpha.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="alpha", stage_name="alpha_stage"))
        mgr = PluginManager(plugins_dir=plugins_dir)
        mgr.discover()
        sentinel = object()
        mgr.init_all(sentinel)
        assert mgr.plugins[0].initialized is True
        assert mgr.plugins[0].engine is sentinel

    def test_register_all_adds_stages(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "alpha.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="alpha", stage_name="alpha_stage"))
        registry = StageRegistry()
        mgr = PluginManager(plugins_dir=plugins_dir, registry=registry)
        mgr.discover()
        mgr.register_all()
        assert "alpha_stage" in registry.names()

    def test_register_all_uses_external_registry(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "alpha.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="alpha", stage_name="alpha_stage"))
        mgr = PluginManager(plugins_dir=plugins_dir)
        mgr.discover()
        external = StageRegistry()
        mgr.register_all(external)
        assert "alpha_stage" in external.names()
        assert "alpha_stage" not in mgr.registry.names()

    def test_teardown_all_calls_teardown(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "alpha.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="alpha", stage_name="alpha_stage"))
        mgr = PluginManager(plugins_dir=plugins_dir)
        mgr.discover()
        mgr.teardown_all()
        assert mgr.plugins[0].tore_down is True

    def test_init_failure_does_not_crash_others(self, plugins_dir: Path):
        bad = textwrap.dedent(
            """
            from retouch.plugin_api import BasePlugin
            class PluginClass(BasePlugin):
                name = "bad"
                version = "0.1.0"
                def init(self, engine):
                    raise ValueError("init boom")
            """
        )
        good = PLUGIN_TEMPLATE_CLASS.format(name="good", stage_name="good_stage")
        _write_plugin(plugins_dir, "bad.py", bad)
        _write_plugin(plugins_dir, "good.py", good)
        mgr = PluginManager(plugins_dir=plugins_dir)
        mgr.discover()
        mgr.init_all(object())
        assert any("init boom" in e.message for e in mgr.errors)
        good_plugin = next(p for p in mgr.plugins if p.name == "good")
        assert good_plugin.initialized is True


# ---------------------------------------------------------------------------
# Manual registration
# ---------------------------------------------------------------------------

class TestManualRegistration:
    def test_register_plugin_adds_to_list(self, plugins_dir: Path):
        mgr = PluginManager(plugins_dir=plugins_dir)

        class P(BasePlugin):
            name = "manual"
            version = "0.1.0"

        p = P()
        mgr.register_plugin(p)
        assert mgr.plugins == [p]

    def test_register_plugin_rejects_non_plugin(self, plugins_dir: Path):
        mgr = PluginManager(plugins_dir=plugins_dir)
        with pytest.raises(TypeError):
            mgr.register_plugin(object())  # type: ignore[arg-type]

    def test_register_plugin_after_init_calls_init(self, plugins_dir: Path):
        mgr = PluginManager(plugins_dir=plugins_dir)
        mgr.init_all(object())

        class P(BasePlugin):
            name = "late"
            version = "0.1.0"
            def __init__(self):
                self.initialized = False
            def init(self, engine):
                self.initialized = True

        p = P()
        mgr.register_plugin(p)
        assert p.initialized is True


# ---------------------------------------------------------------------------
# register_stage helper
# ---------------------------------------------------------------------------

class TestRegisterStageHelper:
    def test_register_stage_adds_to_registry(self):
        registry = StageRegistry()

        class S(BaseStage):
            name = "helper_stage"
            phase = "global"
            def run(self, state):
                return state

        register_stage(registry, S())
        assert "helper_stage" in registry.names()

    def test_register_stage_rejects_non_stage(self):
        registry = StageRegistry()
        with pytest.raises(TypeError):
            register_stage(registry, object())  # type: ignore[arg-type]

    def test_register_stage_duplicate_raises(self):
        registry = StageRegistry()

        class S(BaseStage):
            name = "dup"
            phase = "global"
            def run(self, state):
                return state

        register_stage(registry, S())
        with pytest.raises(ValueError):
            register_stage(registry, S())


# ---------------------------------------------------------------------------
# Path-traversal guard
# ---------------------------------------------------------------------------

class TestPathTraversalGuard:
    def test_symlink_escape_rejected(self, plugins_dir: Path, tmp_path: Path):
        # Write a real plugin file OUTSIDE the plugins dir, then symlink it in.
        outside = tmp_path / "outside.py"
        outside.write_text(PLUGIN_TEMPLATE_CLASS.format(name="escaped", stage_name="e"))
        link = plugins_dir / "link.py"
        link.symlink_to(outside)

        mgr = PluginManager(plugins_dir=plugins_dir)
        mgr.discover()
        assert mgr.plugins == []
        assert any("Path traversal" in e.message for e in mgr.errors)

    def test_env_override_dir_used(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        env_dir = tmp_path / "env_plugins"
        env_dir.mkdir()
        _write_plugin(env_dir, "alpha.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="alpha", stage_name="alpha_stage"))
        monkeypatch.setenv("RETOUCH_PLUGINS_DIR", str(env_dir))

        mgr = PluginManager()
        assert mgr.plugins_dir == env_dir.resolve() or mgr.plugins_dir == env_dir
        loaded = mgr.discover()
        assert len(loaded) == 1


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    def test_concurrent_discover_safe(self, plugins_dir: Path):
        for i in range(8):
            _write_plugin(plugins_dir, f"p{i}.py",
                          PLUGIN_TEMPLATE_CLASS.format(name=f"p{i}", stage_name=f"s{i}"))

        mgr = PluginManager(plugins_dir=plugins_dir)
        errors: List[Exception] = []

        def worker():
            try:
                mgr.discover()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(mgr.plugins) == 8

    def test_concurrent_register_all_safe(self, plugins_dir: Path):
        _write_plugin(plugins_dir, "alpha.py",
                      PLUGIN_TEMPLATE_CLASS.format(name="alpha", stage_name="alpha_stage"))
        registry = StageRegistry()
        mgr = PluginManager(plugins_dir=plugins_dir, registry=registry)
        mgr.discover()

        errors: List[Exception] = []

        def worker():
            try:
                mgr.register_all()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert "alpha_stage" in registry.names()
