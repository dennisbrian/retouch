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
import platform
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


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


def diagnostics_report() -> str:
    """Plain-text diagnostics bundle for bug reports."""
    from . import __version__

    lines = [
        f"retouch version: {__version__}",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
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
        lines.append(f"log file: {log_dir() / 'retouch.log'}")
    except Exception:  # noqa: BLE001
        pass
    return "\n".join(lines)
