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
from urllib.error import URLError

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


class _FakeSocket:
    instances = []
    busy_ports = {7860, 7861}

    def __init__(self, *_args):
        self.bound_port = None
        self.closed = False
        self.options = []
        self.__class__.instances.append(self)

    def setsockopt(self, *args):
        self.options.append(args)

    def bind(self, address):
        _host, port = address
        if port in self.busy_ports:
            raise OSError("address already in use")
        self.bound_port = port

    def getsockname(self):
        return ("127.0.0.1", 49152)

    def close(self):
        self.closed = True


def test_select_available_port_skips_collisions_and_closes_probe_sockets(desktop_module):
    module, _ = desktop_module
    _FakeSocket.instances = []

    selected = module.select_available_port(
        preferred_port=7860,
        attempts=4,
        socket_factory=_FakeSocket,
    )

    assert selected == 7862
    assert len(_FakeSocket.instances) == 3
    assert all(sock.closed for sock in _FakeSocket.instances)


class _Response:
    def __init__(self, status=200):
        self.status = status
        self.closed = False

    def close(self):
        self.closed = True


def test_wait_for_readiness_retries_transient_http_failures(desktop_module):
    module, _ = desktop_module
    responses = [URLError("not listening"), _Response(status=503), _Response(status=200)]
    sleeps = []
    now = [0.0]

    def opener(_url, timeout=None):
        del timeout
        result = responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def sleeper(delay):
        sleeps.append(delay)
        now[0] += delay

    assert module.wait_for_readiness(
        "http://127.0.0.1:7860",
        timeout=2.0,
        interval=0.25,
        opener=opener,
        sleeper=sleeper,
        clock=lambda: now[0],
    ) is True
    assert len(sleeps) == 2
    assert responses == []


def test_start_gradio_hands_backend_exception_to_startup_state(desktop_module):
    module, _ = desktop_module
    app = MagicMock()
    app.launch.side_effect = RuntimeError("port bind failed")
    state = module.StartupState()

    result = module.start_gradio(
        app_instance=app,
        port=7862,
        startup_state=state,
    )

    assert result is state
    assert state.finished.is_set()
    assert state.error is not None
    assert state.error.phase == "gradio"
    assert "port bind failed" in state.error.message


def test_handoff_startup_error_loads_escaped_error_document(desktop_module):
    module, _ = desktop_module
    window = MagicMock()

    document = module.handoff_startup_error(
        window,
        RuntimeError("bad <binding>"),
        url="http://127.0.0.1:7862",
    )

    window.load_html.assert_called_once_with(document)
    assert "bad &lt;binding&gt;" in document
    assert "127.0.0.1:7862" in document


class _ClosedEvent:
    def __init__(self):
        self.callbacks = []

    def __iadd__(self, callback):
        self.callbacks.append(callback)
        return self


class _FakeWindow:
    def __init__(self):
        self.events = type("Events", (), {"closed": _ClosedEvent()})()


class _FakeWebview:
    def __init__(self):
        self.create_calls = []
        self.window = _FakeWindow()
        self.start_calls = 0

    def create_window(self, *args, **kwargs):
        self.create_calls.append((args, kwargs))
        return self.window

    def start(self):
        self.start_calls += 1
        for callback in list(self.window.events.closed.callbacks):
            callback()


class _FakeThread:
    def __init__(self, target, daemon=False):
        self.target = target
        self.daemon = daemon
        self.started = False
        self.join_calls = []
        self._alive = True

    def start(self):
        self.started = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        self.join_calls.append(timeout)
        self._alive = False


def test_desktop_runtime_waits_for_ready_and_shuts_down_on_window_close(desktop_module):
    module, _ = desktop_module
    app = MagicMock()
    webview = _FakeWebview()
    thread = _FakeThread(target=lambda: None)

    def thread_factory(target, daemon=False):
        thread.target = target
        thread.daemon = daemon
        return thread

    runtime = module.DesktopRuntime(
        app_instance=app,
        webview_module=webview,
        socket_factory=_FakeSocket,
        readiness_opener=lambda _url, timeout=None: _Response(status=200),
        thread_factory=thread_factory,
    )

    assert runtime.run() is True
    assert runtime.url == "http://127.0.0.1:7862"
    assert webview.start_calls == 1
    assert app.close.called
    assert thread.join_calls == [module.DEFAULT_JOIN_TIMEOUT]
    assert runtime.state.stop_requested.is_set()
    assert webview.create_calls[0][0] == (module.DEFAULT_WINDOW_TITLE, runtime.url)


def test_desktop_runtime_hands_readiness_timeout_to_error_window(desktop_module):
    module, _ = desktop_module
    app = MagicMock()
    webview = _FakeWebview()
    thread = _FakeThread(target=lambda: None)

    def thread_factory(target, daemon=False):
        thread.target = target
        thread.daemon = daemon
        return thread

    def never_ready(_url, timeout=None):
        del timeout
        raise URLError("connection refused")

    runtime = module.DesktopRuntime(
        app_instance=app,
        webview_module=webview,
        socket_factory=_FakeSocket,
        readiness_opener=never_ready,
        startup_timeout=0,
        thread_factory=thread_factory,
    )

    assert runtime.run() is False
    assert "Retouch could not start" in webview.create_calls[0][1]["html"]
    assert "Timed out waiting" in webview.create_calls[0][1]["html"]
    assert app.close.called
    assert thread.join_calls == [module.DEFAULT_JOIN_TIMEOUT]
