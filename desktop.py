#!/usr/bin/env python3
"""Reliable desktop launcher for Pro Max Retouch Studio.

The native shell is deliberately kept separate from the Gradio application.
Only the standard-library startup helpers are imported at module import time;
``gui`` and ``pywebview`` are loaded when the launcher is actually run.  This
keeps packaging, headless diagnostics, and unit tests independent of native
desktop bindings.
"""

import html
import importlib
import socket
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860
DEFAULT_WINDOW_TITLE = "Pro Max Retouch Studio"
DEFAULT_WINDOW_WIDTH = 1600
DEFAULT_WINDOW_HEIGHT = 1000
DEFAULT_STARTUP_TIMEOUT = 30.0
DEFAULT_READINESS_INTERVAL = 0.1
DEFAULT_JOIN_TIMEOUT = 5.0


class PortSelectionError(RuntimeError):
    """Raised when no candidate local port can be bound for inspection."""


class StartupFailure(RuntimeError):
    """A backend failure that can be handed off to the native window."""

    def __init__(
        self,
        message: str,
        *,
        phase: str = "startup",
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.phase = phase
        self.cause = cause


class ReadinessTimeout(StartupFailure):
    """Raised when the local Gradio endpoint never becomes reachable."""


@dataclass
class StartupErrorInfo:
    """Serializable details from an exception in the Gradio thread."""

    phase: str
    exception_type: str
    message: str
    traceback_text: str = ""

    @classmethod
    def from_exception(
        cls,
        error: BaseException,
        *,
        phase: str,
    ) -> "StartupErrorInfo":
        return cls(
            phase=phase,
            exception_type=type(error).__name__,
            message=str(error) or repr(error),
            traceback_text="".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        )

    def display_message(self) -> str:
        return "%s: %s" % (self.exception_type, self.message)


class StartupState:
    """Thread-safe handoff state shared by the launcher and Gradio thread."""

    def __init__(self) -> None:
        self.ready = threading.Event()
        self.finished = threading.Event()
        self.stop_requested = threading.Event()
        self._lock = threading.Lock()
        self._error: Optional[StartupErrorInfo] = None

    @property
    def error(self) -> Optional[StartupErrorInfo]:
        with self._lock:
            return self._error

    def mark_ready(self) -> None:
        self.ready.set()

    def mark_finished(self) -> None:
        self.finished.set()

    def request_stop(self) -> None:
        self.stop_requested.set()

    def mark_error(
        self,
        error: BaseException,
        *,
        phase: str = "gradio",
    ) -> StartupErrorInfo:
        details = (
            error
            if isinstance(error, StartupErrorInfo)
            else StartupErrorInfo.from_exception(error, phase=phase)
        )
        with self._lock:
            if self._error is None:
                self._error = details
        self.finished.set()
        return details


def _load_gui_app() -> Any:
    """Load ``gui.app`` only when the desktop launcher needs it."""

    return getattr(importlib.import_module("gui"), "app")


def _load_webview() -> Any:
    """Load pywebview lazily so importing this module stays headless-safe."""

    return importlib.import_module("webview")


def _candidate_ports(preferred_port: int, attempts: int):
    if preferred_port == 0:
        yield 0
        return
    for offset in range(attempts):
        candidate = preferred_port + offset
        if candidate > 65535:
            break
        yield candidate


def select_available_port(
    host: str = DEFAULT_HOST,
    preferred_port: int = DEFAULT_PORT,
    *,
    attempts: int = 32,
    socket_factory: Callable[..., Any] = socket.socket,
) -> int:
    """Return the first bindable port, preserving the historical default.

    The check uses the same address family and loopback host as the Gradio
    launch.  The socket is closed immediately after the check because Gradio
    owns the listening socket; the short selection-to-launch window is still
    inherently subject to an external process racing for the port, so the
    launcher also captures and reports a bind failure from the Gradio thread.
    """

    if not isinstance(preferred_port, int) or not 0 <= preferred_port <= 65535:
        raise ValueError("preferred_port must be an integer between 0 and 65535")
    if not isinstance(attempts, int) or attempts < 1:
        raise ValueError("attempts must be a positive integer")

    failures = []
    for candidate in _candidate_ports(preferred_port, attempts):
        sock = socket_factory(socket.AF_INET, socket.SOCK_STREAM)
        try:
            set_sockopt = getattr(sock, "setsockopt", None)
            if callable(set_sockopt):
                set_sockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, candidate))
            if candidate == 0:
                return int(sock.getsockname()[1])
            return candidate
        except OSError as error:
            failures.append((candidate, error))
        finally:
            close = getattr(sock, "close", None)
            if callable(close):
                close()

    attempted = ", ".join(str(port) for port, _ in failures) or str(preferred_port)
    raise PortSelectionError(
        "No available local port for %s after checking %s" % (host, attempted)
    )


# A descriptive alias makes the helper convenient to discover without
# changing the single implementation used by the launcher.
find_available_port = select_available_port


def _response_status(response: Any) -> Optional[int]:
    status = getattr(response, "status", None)
    if status is None:
        getcode = getattr(response, "getcode", None)
        status = getcode() if callable(getcode) else None
    return int(status) if status is not None else None


def wait_for_readiness(
    url: str,
    *,
    timeout: float = DEFAULT_STARTUP_TIMEOUT,
    interval: float = DEFAULT_READINESS_INTERVAL,
    opener: Callable[..., Any] = urlopen,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    startup_state: Optional[StartupState] = None,
) -> bool:
    """Poll a local HTTP URL until it responds or startup fails.

    ``opener``, ``sleeper``, and ``clock`` are injectable so tests can model
    slow startup without real sockets or wall-clock delays.  A 2xx/3xx
    response proves that an HTTP server is serving; backend exceptions and a
    process that exits before readiness are handed back immediately.
    """

    if timeout < 0:
        raise ValueError("timeout must not be negative")
    if interval < 0:
        raise ValueError("interval must not be negative")

    deadline = clock() + timeout
    last_error: Optional[BaseException] = None

    while True:
        if startup_state is not None:
            startup_error = startup_state.error
            if startup_error is not None:
                raise StartupFailure(
                    startup_error.display_message(),
                    phase=startup_error.phase,
                    cause=startup_error,
                )
            if startup_state.finished.is_set() and not startup_state.ready.is_set():
                raise StartupFailure(
                    "Gradio exited before %s became ready" % url,
                    phase="readiness",
                )

        try:
            response = opener(
                url,
                timeout=max(
                    0.01,
                    min(interval or 0.01, max(0.01, deadline - clock())),
                ),
            )
            try:
                status = _response_status(response)
                if status is None or 200 <= status < 400:
                    if startup_state is not None:
                        startup_state.mark_ready()
                    return True
                last_error = RuntimeError("HTTP readiness status %s" % status)
            finally:
                close = getattr(response, "close", None)
                if callable(close):
                    close()
        except (HTTPError, URLError, OSError, TimeoutError) as error:
            last_error = error

        remaining = deadline - clock()
        if remaining <= 0:
            detail = ": %s" % last_error if last_error is not None else ""
            raise ReadinessTimeout(
                "Timed out waiting for %s after %.1fs%s" % (url, timeout, detail),
                phase="readiness",
                cause=last_error,
            )
        sleeper(min(interval, remaining) if interval else 0)


def startup_error_html(error: Any, *, url: Optional[str] = None) -> str:
    """Create a small, safe error document suitable for ``window.load_html``."""

    if isinstance(error, StartupErrorInfo):
        message = error.display_message()
        phase = error.phase
    else:
        message = str(error) or repr(error)
        phase = getattr(error, "phase", "startup")
    endpoint = "" if url is None else "<p>Endpoint: <code>%s</code></p>" % html.escape(url)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>Retouch startup error</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;"
        "max-width:760px;margin:12vh auto;padding:2rem;color:#242424}"
        "code{background:#f1f1f1;padding:.15rem .3rem;border-radius:4px}"
        "</style></head><body>"
        "<h1>Retouch could not start</h1>"
        "<p><strong>%s</strong></p><p>%s</p>%s"
        "</body></html>"
        % (html.escape(phase), html.escape(message), endpoint)
    )


def handoff_startup_error(target: Any, error: Any, *, url: Optional[str] = None) -> str:
    """Hand a startup error to a window or callback and return its HTML.

    The callback form is useful for embedding the launcher in another host;
    pywebview windows normally expose ``load_html``.  ``load_url`` is retained
    as a fallback for small window fakes and older integrations.
    """

    document = startup_error_html(error, url=url)
    if callable(target) and not callable(getattr(target, "load_html", None)):
        target(error)
        return document

    load_html = getattr(target, "load_html", None)
    if callable(load_html):
        load_html(document)
        return document

    load_url = getattr(target, "load_url", None)
    if callable(load_url):
        load_url("data:text/html;charset=utf-8," + quote(document))
        return document

    raise TypeError("startup error target must be a callback or pywebview window")


def _attach_closed_handler(window: Any, callback: Callable[..., Any]) -> bool:
    """Attach a pywebview ``closed`` callback while tolerating test doubles."""

    events = getattr(window, "events", None)
    closed = getattr(events, "closed", None)
    if closed is None:
        return False
    try:
        closed += callback
        return True
    except (AttributeError, TypeError):
        append = getattr(closed, "append", None)
        if callable(append):
            append(callback)
            return True
    return False


class DesktopRuntime:
    """Own the backend thread, native window, and shutdown lifecycle."""

    def __init__(
        self,
        *,
        app_instance: Any = None,
        webview_module: Any = None,
        host: str = DEFAULT_HOST,
        preferred_port: int = DEFAULT_PORT,
        port_attempts: int = 32,
        startup_timeout: float = DEFAULT_STARTUP_TIMEOUT,
        readiness_interval: float = DEFAULT_READINESS_INTERVAL,
        socket_factory: Callable[..., Any] = socket.socket,
        readiness_opener: Callable[..., Any] = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        thread_factory: Callable[..., Any] = threading.Thread,
        join_timeout: float = DEFAULT_JOIN_TIMEOUT,
        window_title: str = DEFAULT_WINDOW_TITLE,
        window_width: int = DEFAULT_WINDOW_WIDTH,
        window_height: int = DEFAULT_WINDOW_HEIGHT,
    ) -> None:
        self.app = app_instance
        self.webview = webview_module
        self.host = host
        self.preferred_port = preferred_port
        self.port_attempts = port_attempts
        self.startup_timeout = startup_timeout
        self.readiness_interval = readiness_interval
        self.socket_factory = socket_factory
        self.readiness_opener = readiness_opener
        self.sleeper = sleeper
        self.clock = clock
        self.thread_factory = thread_factory
        self.join_timeout = join_timeout
        self.window_title = window_title
        self.window_width = window_width
        self.window_height = window_height
        self.state = StartupState()
        self.port: Optional[int] = None
        self.url: Optional[str] = None
        self.thread: Any = None
        self.window: Any = None
        self._shutdown_lock = threading.Lock()
        self._shutdown_complete = False

    def _ensure_dependencies(self) -> None:
        if self.app is None:
            self.app = _load_gui_app()
        if self.webview is None:
            self.webview = _load_webview()

    def _launch_backend(self) -> None:
        start_gradio(
            app_instance=self.app,
            host=self.host,
            port=self.port,
            startup_state=self.state,
        )

    def _create_window(self, *, error: Optional[Any] = None) -> Any:
        if error is None:
            window = self.webview.create_window(
                self.window_title,
                self.url,
                width=self.window_width,
                height=self.window_height,
            )
        else:
            window = self.webview.create_window(
                self.window_title,
                html=startup_error_html(error, url=self.url),
                width=self.window_width,
                height=self.window_height,
            )
        self.window = window
        _attach_closed_handler(window, self.shutdown)
        return window

    def run(self) -> bool:
        """Start Gradio, show a native window, and clean up on exit.

        Returns ``True`` when the backend became reachable and ``False`` when
        a startup error was handed to an error window.  Exceptions raised by
        pywebview itself are still propagated after backend cleanup.
        """

        self._ensure_dependencies()
        self.port = select_available_port(
            self.host,
            self.preferred_port,
            attempts=self.port_attempts,
            socket_factory=self.socket_factory,
        )
        self.url = "http://%s:%s" % (self.host, self.port)
        try:
            self.thread = self.thread_factory(target=self._launch_backend, daemon=True)
        except TypeError:
            # Small embedding hosts and test doubles may expose the older
            # target-only Thread constructor; set daemon below when possible.
            self.thread = self.thread_factory(target=self._launch_backend)
        try:
            self.thread.daemon = True
        except (AttributeError, RuntimeError):
            pass
        self.thread.start()

        try:
            wait_for_readiness(
                self.url,
                timeout=self.startup_timeout,
                interval=self.readiness_interval,
                opener=self.readiness_opener,
                sleeper=self.sleeper,
                clock=self.clock,
                startup_state=self.state,
            )
        except StartupFailure as error:
            self._create_window(error=error)
            try:
                self.webview.start()
            finally:
                self.shutdown()
            return False

        self._create_window()
        try:
            self.webview.start()
        finally:
            self.shutdown()
        return True

    def shutdown(self, *_args: Any, **_kwargs: Any) -> None:
        """Stop Gradio and join its thread; safe to call more than once."""

        with self._shutdown_lock:
            if self._shutdown_complete:
                return
            self._shutdown_complete = True

        self.state.request_stop()
        close = getattr(self.app, "close", None)
        try:
            if callable(close):
                close()
        finally:
            thread = self.thread
            if (
                thread is not None
                and thread is not threading.current_thread()
                and getattr(thread, "is_alive", lambda: False)()
            ):
                thread.join(timeout=self.join_timeout)


def start_gradio(
    app_instance: Any = None,
    *,
    host: str = DEFAULT_HOST,
    port: Optional[int] = DEFAULT_PORT,
    startup_state: Optional[StartupState] = None,
    **launch_kwargs: Any,
) -> Optional[StartupState]:
    """Launch Gradio using the historical defaults and capture failures."""

    app_instance = _load_gui_app() if app_instance is None else app_instance
    state = startup_state
    kwargs = {
        "server_name": host,
        "server_port": DEFAULT_PORT if port is None else port,
        "inbrowser": False,
        "show_error": True,
    }
    kwargs.update(launch_kwargs)
    try:
        app_instance.launch(**kwargs)
    except Exception as error:
        if state is None:
            raise
        state.mark_error(error, phase="gradio")
    finally:
        if state is not None:
            state.mark_finished()
    return state


def run_desktop(**runtime_kwargs: Any) -> bool:
    """Compatibility entry point used by the script and desktop bundle."""

    return DesktopRuntime(**runtime_kwargs).run()


if __name__ == "__main__":
    run_desktop()
