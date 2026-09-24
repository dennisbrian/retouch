"""Diagnostics: rotating log file + copy-paste diagnostics bundle.

Lives in the user log dir (``~/Library/Logs/ProMaxRetouch`` on macOS,
``%LOCALAPPDATA%\\ProMaxRetouch\\Logs`` on Windows, ``~/.cache/promaxretouch/logs``
elsewhere). Setup is opt-in: GUI/desktop entry points call
:func:`setup_file_logging` at startup; library use is untouched.

:func:`diagnostics_report` returns a plain-text bundle (versions, ORT
providers, platform, last errors) suitable for pasting into a bug report.
"""

from __future__ import annotations

import logging
import os
import platform
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import IO, Optional


def log_dir() -> Path:
    """Per-user log directory, created on demand."""
    if sys.platform == "darwin":
        d = Path.home() / "Library" / "Logs" / "ProMaxRetouch"
    elif sys.platform == "win32":
        import os
        d = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ProMaxRetouch" / "Logs"
    else:
        d = Path.home() / ".cache" / "promaxretouch" / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def setup_file_logging(level: int = logging.INFO) -> Path:
    """Attach a rotating file handler to the root logger. Idempotent.

    Returns the log file path. 1 MB x 3 files — enough for a session's
    worth of errors without growing unbounded.
    """
    path = log_dir() / "retouch.log"
    root = logging.getLogger()
    for h in root.handlers:
        if isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", None) == str(path):
            return path
    handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    handler.setLevel(level)
    root.addHandler(handler)
    if root.level > level:
        root.setLevel(level)
    return path


_native_crash_stream: Optional[IO[str]] = None


def enable_native_crash_log(max_bytes: int = 1_000_000) -> Optional[Path]:
    """Record Python tracebacks in ``native-crash.log`` if the process dies in native code.

    A segfault or abort inside MediaPipe, onnxruntime or OpenCV kills the
    process before any ``except`` block or :func:`retouch.utils.log_crash`
    runs, so without this the window just disappears and nothing is left to
    report. Uses :mod:`faulthandler`; idempotent. The file is truncated once
    it passes ``max_bytes``. Returns the log path, or ``None`` if it cannot
    be opened (faulthandler then stays on stderr).
    """
    import faulthandler

    global _native_crash_stream
    path = log_dir() / "native-crash.log"
    if _native_crash_stream is not None:
        return path
    try:
        mode = "w" if path.exists() and path.stat().st_size > max_bytes else "a"
        stream = open(path, mode, encoding="utf-8")
        stream.write(
            f"--- retouch started {datetime.now().isoformat(timespec='seconds')} "
            f"(pid {os.getpid()})\n"
        )
        stream.flush()
    except OSError as exc:
        logging.getLogger(__name__).warning("Could not open %s: %s", path, exc)
        faulthandler.enable()
        return None
    faulthandler.enable(file=stream, all_threads=True)
    _native_crash_stream = stream
    return path


def clear_diagnostics() -> str:
    """Remove Retouch log/crash artifacts from the user's diagnostic stores."""
    from .utils import get_cache_dir

    removed = 0
    targets = [log_dir(), get_cache_dir()]
    for directory in targets:
        if not directory.exists():
            continue
        for pattern in ("retouch.log*", "crash.log*", "native-crash.log*"):
            for path in directory.glob(pattern):
                try:
                    path.unlink()
                    removed += 1
                except OSError:
                    logging.getLogger(__name__).warning("Could not clear diagnostic file %s", path)
    return f"Cleared {removed} diagnostic file(s)."


def diagnostics_report() -> str:
    """Plain-text diagnostics bundle for bug reports."""
    from . import __version__

    from .utils import offline_mode_enabled, redact_diagnostics_text

    lines = [
        f"retouch version: {__version__}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        f"offline mode: {'enabled' if offline_mode_enabled() else 'disabled'}",
    ]
    try:
        import cv2
        lines.append(f"opencv: {cv2.__version__}")
    except ImportError:
        lines.append("opencv: NOT INSTALLED")
    try:
        import numpy as np
        lines.append(f"numpy: {np.__version__}")
    except ImportError:
        lines.append("numpy: NOT INSTALLED")
    try:
        import onnxruntime as ort
        lines.append(f"onnxruntime: {ort.__version__}")
        try:
            from .perf_optimizations import build_ort_providers
            lines.append(f"ort providers: {build_ort_providers()}")
        except Exception as e:  # noqa: BLE001 — diagnostics must never raise
            lines.append(f"ort providers: unavailable ({e})")
    except ImportError:
        lines.append("onnxruntime: NOT INSTALLED")
    try:
        import mediapipe as mp
        lines.append(f"mediapipe: {mp.__version__}")
    except ImportError:
        lines.append("mediapipe: NOT INSTALLED")
    try:
        from .model_fetch import load_manifest, model_status
        for model_name, entry in load_manifest().get("models", {}).items():
            status = model_status(model_name)
            availability = entry.get("availability", "unspecified")
            lines.append(
                f"model {model_name}: availability={availability} "
                f"local={'yes' if status.get('available') else 'no'} "
                f"downloadable={'yes' if status.get('downloadable') else 'no'}"
            )
    except Exception as e:  # noqa: BLE001 — diagnostics must never raise
        lines.append(f"model manifest: unavailable ({e})")
    try:
        from .body_reshape import body_reshape_capability_status
        pose = body_reshape_capability_status()
        lines.append(
            f"body reshape: {'available' if pose.get('available') else 'unavailable'} "
            f"({pose.get('reason', 'unknown')})"
        )
    except Exception as e:  # noqa: BLE001
        lines.append(f"body reshape: unavailable ({e})")
    try:
        lines.append(f"log file: {redact_diagnostics_text(log_dir() / 'retouch.log')}")
        lines.append(
            f"native crash log: {redact_diagnostics_text(log_dir() / 'native-crash.log')}"
        )
    except Exception:  # noqa: BLE001
        pass
    return redact_diagnostics_text("\n".join(lines))
