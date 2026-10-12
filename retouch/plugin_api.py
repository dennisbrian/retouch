"""Plugin API v0 for the Retouch Engine (T4).

Allows external plugins to register custom pipeline stages and hook into
the engine lifecycle. Plugins live in ``~/.retouch/plugins/`` as plain
Python modules; each module must expose a callable (class or factory)
that returns an object satisfying the :class:`Plugin` protocol.

Public API:
    Plugin           — protocol/ABC every plugin must implement
    BasePlugin       — convenience base class with no-op defaults
    PluginManager    — discover, load, and lifecycle-manage plugins
    register_stage   — module-level helper to add a stage to a registry

Design notes:
    - Thread-safe: a ``threading.RLock`` guards all mutable manager state.
      Plugins are loaded once at startup; stage registration from already-
      loaded plugins is also locked.
    - Path-traversal guard: plugin files are resolved and confined to the
      plugins directory (symlinks that escape are rejected). No plugin file
      outside the resolved plugins root is importable through this API.
    - No bare ``except``. ``ImportError``/``AttributeError`` from a plugin
      module are caught, logged, and reported in ``PluginManager.errors``;
      a single broken plugin never prevents the rest from loading.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union, runtime_checkable

from .stages import Stage, StageRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plugin protocol / base class
# ---------------------------------------------------------------------------

@runtime_checkable
class Plugin(Protocol):
    """Protocol every plugin must satisfy.

    A plugin is a small object that:
        - identifies itself (``name``, ``version``)
        - is handed the engine once at startup (``init``)
        - registers its custom stages with the registry (``register_stages``)
        - cleans up on shutdown (``teardown``)
    """

    name: str
    version: str

    def init(self, engine: Any) -> None: ...

    def register_stages(self, registry: StageRegistry) -> None: ...

    def teardown(self) -> None: ...


class BasePlugin:
    """Convenience base class with no-op defaults.

    Subclass and override only the hooks you need::

        class MyPlugin(BasePlugin):
            name = "my_plugin"
            version = "0.1.0"

            def register_stages(self, registry):
                registry.add(MyStage())
    """

    name: str = "base_plugin"
    version: str = "0.0.1"

    def init(self, engine: Any) -> None:
        return None

    def register_stages(self, registry: StageRegistry) -> None:
        return None

    def teardown(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Plugin error record
# ---------------------------------------------------------------------------

@dataclass
class PluginError:
    """Record of a failed plugin load/init, exposed for UI diagnostics."""

    plugin_path: str
    message: str
    exc_type: str = ""


# ---------------------------------------------------------------------------
# Plugin manager
# ---------------------------------------------------------------------------

_DEFAULT_PLUGINS_DIR = Path.home() / ".retouch" / "plugins"


def _default_plugins_dir() -> Path:
    """Return the default plugins directory (env-overridable)."""
    env = os.environ.get("RETOUCH_PLUGINS_DIR")
    if env:
        return Path(env).expanduser()
    return _DEFAULT_PLUGINS_DIR


def _safe_resolve_plugin_path(path: Path, plugins_root: Path) -> Path:
    """Resolve *path* and confirm it stays inside *plugins_root*.

    Symlinks that escape the plugins directory are rejected. This is the
    path-traversal guard: a plugin file must live (after symlink
    resolution) inside the configured plugins root.
    """
    resolved = path.expanduser().resolve(strict=False)
    root = plugins_root.expanduser().resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Path traversal rejected: plugin {path!s} escapes plugins root {root!s}"
        ) from exc
    return resolved


class PluginManager:
    """Discover, load, and lifecycle-manage Retouch Engine plugins.

    Usage::

        mgr = PluginManager()
        mgr.discover()                 # scan ~/.retouch/plugins/*.py
        mgr.init_all(engine)           # call init() on every loaded plugin
        mgr.register_all(registry)     # let plugins add custom stages
        ...
        mgr.teardown_all()             # call teardown() on every plugin
    """

    def __init__(
        self,
        plugins_dir: Optional[Union[str, Path]] = None,
        registry: Optional[StageRegistry] = None,
    ) -> None:
        self._plugins_dir: Path = (
            Path(plugins_dir).expanduser() if plugins_dir is not None
            else _default_plugins_dir()
        )
        self._registry: StageRegistry = registry if registry is not None else StageRegistry()
        self._plugins: List[Plugin] = []
        self._errors: List[PluginError] = []
        self._loaded_modules: List[str] = []
        self._lock = threading.RLock()
        self._engine: Any = None

    # -- properties --------------------------------------------------------

    @property
    def plugins_dir(self) -> Path:
        return self._plugins_dir

    @property
    def registry(self) -> StageRegistry:
        return self._registry

    @property
    def plugins(self) -> List[Plugin]:
        with self._lock:
            return list(self._plugins)

    @property
    def errors(self) -> List[PluginError]:
        with self._lock:
            return list(self._errors)

    # -- discovery / loading ----------------------------------------------

    def discover(self) -> List[Plugin]:
        """Scan the plugins directory for ``*.py`` files and load them.

        Each file is imported in isolation, then inspected for a plugin
        factory. The factory may be exposed as either:
            - a top-level ``PluginClass`` attribute (a class), or
            - a top-level ``create_plugin`` callable, or
            - a top-level ``plugin`` attribute that is already an instance.

        Already-loaded modules are skipped (idempotent discovery).

        Returns the list of newly loaded plugin instances. Failures are
        recorded in :attr:`errors` and do not raise.
        """
        newly_loaded: List[Plugin] = []
        if not self._plugins_dir.is_dir():
            return newly_loaded

        with self._lock:
            for py_file in sorted(self._plugins_dir.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                mod_name = f"retouch_plugin_{py_file.stem}"
                if mod_name in self._loaded_modules:
                    continue
                try:
                    plugin = self._load_plugin_file(py_file, mod_name)
                except Exception as exc:  # noqa: BLE001 — report, don't crash
                    self._errors.append(PluginError(
                        plugin_path=str(py_file),
                        message=str(exc),
                        exc_type=type(exc).__name__,
                    ))
                    logger.warning("Failed to load plugin %s: %s", py_file, exc, exc_info=True)
                    continue
                if plugin is None:
                    self._errors.append(PluginError(
                        plugin_path=str(py_file),
                        message="No plugin factory found (expected PluginClass, create_plugin, or plugin).",
                        exc_type="AttributeError",
                    ))
                    continue
                self._loaded_modules.append(mod_name)
                self._plugins.append(plugin)
                newly_loaded.append(plugin)
        return newly_loaded

    def _load_plugin_file(self, path: Path, mod_name: str) -> Optional[Plugin]:
        """Import *path* as an isolated module and return a plugin instance."""
        safe_path = _safe_resolve_plugin_path(path, self._plugins_dir)
        spec = importlib.util.spec_from_file_location(mod_name, str(safe_path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not build module spec for {path!s}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(mod_name, None)
            raise
        return self._extract_plugin(module)

    @staticmethod
    def _extract_plugin(module: Any) -> Optional[Plugin]:
        """Pull a plugin instance out of a freshly imported module."""
        candidate: Any = None
        if hasattr(module, "PluginClass"):
            candidate = module.PluginClass
        elif hasattr(module, "create_plugin"):
            candidate = module.create_plugin
        elif hasattr(module, "plugin"):
            candidate = module.plugin

        if isinstance(candidate, type):
            instance = candidate()
        elif callable(candidate):
            instance = candidate()
        else:
            instance = candidate

        if instance is None:
            return None
        if not isinstance(instance, Plugin):
            raise TypeError(
                f"Plugin factory in {module.__name__!r} returned object "
                f"that does not satisfy the Plugin protocol: {type(instance).__name__}"
            )
        return instance

    # -- lifecycle ---------------------------------------------------------

    def init_all(self, engine: Any) -> None:
        """Call ``init(engine)`` on every loaded plugin."""
        with self._lock:
            self._engine = engine
            for plugin in list(self._plugins):
                try:
                    plugin.init(engine)
                except Exception as exc:  # noqa: BLE001 — report, don't crash
                    self._errors.append(PluginError(
                        plugin_path=plugin.name,
                        message=f"init() failed: {exc}",
                        exc_type=type(exc).__name__,
                    ))
                    logger.warning("Plugin %s init() failed: %s", plugin.name, exc, exc_info=True)

    def register_all(self, registry: Optional[StageRegistry] = None) -> StageRegistry:
        """Call ``register_stages(registry)`` on every loaded plugin.

        If *registry* is None, the manager's own registry is used.
        Returns the registry stages were registered into.
        """
        target = registry if registry is not None else self._registry
        with self._lock:
            for plugin in list(self._plugins):
                try:
                    plugin.register_stages(target)
                except Exception as exc:  # noqa: BLE001 — report, don't crash
                    self._errors.append(PluginError(
                        plugin_path=plugin.name,
                        message=f"register_stages() failed: {exc}",
                        exc_type=type(exc).__name__,
                    ))
                    logger.warning(
                        "Plugin %s register_stages() failed: %s",
                        plugin.name, exc, exc_info=True,
                    )
        return target

    def teardown_all(self) -> None:
        """Call ``teardown()`` on every loaded plugin (best-effort)."""
        with self._lock:
            for plugin in list(self._plugins):
                try:
                    plugin.teardown()
                except Exception as exc:  # noqa: BLE001 — report, don't crash
                    self._errors.append(PluginError(
                        plugin_path=plugin.name,
                        message=f"teardown() failed: {exc}",
                        exc_type=type(exc).__name__,
                    ))
                    logger.warning(
                        "Plugin %s teardown() failed: %s",
                        plugin.name, exc, exc_info=True,
                    )

    # -- manual registration ----------------------------------------------

    def register_plugin(self, plugin: Plugin) -> None:
        """Manually register an already-instantiated plugin."""
        if not isinstance(plugin, Plugin):
            raise TypeError("plugin must satisfy the Plugin protocol")
        with self._lock:
            self._plugins.append(plugin)
            if self._engine is not None:
                try:
                    plugin.init(self._engine)
                except Exception as exc:  # noqa: BLE001
                    self._errors.append(PluginError(
                        plugin_path=plugin.name,
                        message=f"init() failed: {exc}",
                        exc_type=type(exc).__name__,
                    ))
            try:
                plugin.register_stages(self._registry)
            except Exception as exc:  # noqa: BLE001
                self._errors.append(PluginError(
                    plugin_path=plugin.name,
                    message=f"register_stages() failed: {exc}",
                    exc_type=type(exc).__name__,
                ))


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def register_stage(registry: StageRegistry, stage: Stage) -> StageRegistry:
    """Add *stage* to *registry*.

    Thin convenience wrapper around ``StageRegistry.add`` so plugin authors
    can write ``register_stage(registry, MyStage())`` without thinking about
    the registry's chaining API. Raises ``ValueError`` on duplicate names
    (propagated from the registry).
    """
    if not isinstance(stage, Stage):
        raise TypeError("stage must satisfy the Stage protocol")
    return registry.add(stage)


# Re-export Union for type-checker friendliness in downstream signatures.
__all__ = [
    "Plugin",
    "BasePlugin",
    "PluginError",
    "PluginManager",
    "register_stage",
]
