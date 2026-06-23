"""Structural tests for desktop.py — the PyWebView wrapper.

`desktop.py` is a thin launcher that boots Gradio in a background thread
and opens a native OS window via pywebview. We intentionally avoid running
the launcher (no display, no real web server) and instead verify the
module's structure and that ``start_gradio`` correctly delegates to
``gui.app.launch`` with the expected arguments.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest


_DESKTOP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "desktop.py",
)


@pytest.fixture
def desktop_module():
    """Import ``desktop.py`` with ``webview`` and ``gui.app`` mocked.

    Yields:
        tuple: ``(module, mock_app)`` so tests can introspect the
        mocked ``gui.app`` object that the launcher talks to.
    """
    mock_webview = MagicMock()
    mock_gui = MagicMock()
    mock_app = MagicMock()
    mock_gui.app = mock_app

    saved_modules = {}
    for key in ("webview", "gui"):
        if key in sys.modules:
            saved_modules[key] = sys.modules[key]
    sys.modules["webview"] = mock_webview
    sys.modules["gui"] = mock_gui

    saved_desktop = sys.modules.pop("desktop", None)
    try:
        spec = importlib.util.spec_from_file_location(
            "_desktop_under_test", _DESKTOP_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        yield module, mock_app
    finally:
        sys.modules.pop("_desktop_under_test", None)
        if saved_desktop is not None:
            sys.modules["desktop"] = saved_desktop
        for key in ("webview", "gui"):
            if key in saved_modules:
                sys.modules[key] = saved_modules[key]
            else:
                sys.modules.pop(key, None)


def test_module_imports_without_error(desktop_module):
    """``desktop.py`` should import cleanly when its heavy deps are mocked."""
    module, _ = desktop_module
    assert module is not None


def test_start_gradio_is_callable(desktop_module):
    """The module must expose a callable ``start_gradio`` entry point."""
    module, _ = desktop_module
    assert hasattr(module, "start_gradio")
    assert callable(module.start_gradio)


def test_start_gradio_invokes_app_launch_with_expected_kwargs(desktop_module):
    """``start_gradio`` must delegate to ``gui.app.launch`` with the
    localhost binding used by the native wrapper."""
    module, mock_app = desktop_module
    module.start_gradio()
    assert mock_app.launch.called
    kwargs = mock_app.launch.call_args.kwargs
    assert kwargs.get("server_name") == "127.0.0.1"
    assert kwargs.get("server_port") == 7860
    assert kwargs.get("inbrowser") is False
    assert kwargs.get("show_error") is True
