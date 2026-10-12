"""Tests for retouch/diagnostics.py."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from retouch.diagnostics import clear_diagnostics, diagnostics_report, log_dir, setup_file_logging


def test_log_dir_created(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    d = log_dir()
    assert d.exists() and d.is_dir()


def test_setup_file_logging_writes(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    path = setup_file_logging()
    logging.getLogger().info("diagnostics test marker")
    for h in logging.getLogger().handlers:
        if isinstance(h, RotatingFileHandler):
            h.flush()
    assert "diagnostics test marker" in path.read_text()


def test_setup_file_logging_idempotent(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    setup_file_logging()
    n = len([h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)])
    setup_file_logging()
    m = len([h for h in logging.getLogger().handlers if isinstance(h, RotatingFileHandler)])
    assert n == m


def test_diagnostics_report_contents():
    report = diagnostics_report()
    assert "retouch version:" in report
    assert "python:" in report
    assert "platform:" in report
    assert "opencv:" in report
    assert "numpy:" in report


def test_diagnostics_report_never_raises_on_missing_ort(monkeypatch):
    # Simulate onnxruntime absence
    import sys
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    report = diagnostics_report()
    assert "onnxruntime:" in report


def test_crash_diagnostics_redact_private_image_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("RETOUCH_CACHE_DIR", str(tmp_path / "cache"))
    private_image = tmp_path / "private" / "portrait.jpg"
    from retouch.utils import log_crash

    log_crash(RuntimeError(f"could not open {private_image}"), {"file_path": str(private_image)})
    crash_log = (tmp_path / "cache" / "crash.log").read_text(encoding="utf-8")
    assert str(private_image) not in crash_log
    assert "<redacted-path>" in crash_log


def test_clear_diagnostics_removes_crash_files(tmp_path, monkeypatch):
    monkeypatch.setenv("RETOUCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "crash.log").write_text("private", encoding="utf-8")
    result = clear_diagnostics()
    assert "Cleared" in result
    assert not (tmp_path / "cache" / "crash.log").exists()


def test_enable_native_crash_log_points_faulthandler_at_log_dir(tmp_path, monkeypatch):
    import faulthandler

    import retouch.diagnostics as diagnostics

    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setattr(diagnostics, "_native_crash_stream", None)
    calls = []
    # Don't hijack pytest's own faulthandler; just record the call.
    monkeypatch.setattr(faulthandler, "enable", lambda **kw: calls.append(kw))

    path = diagnostics.enable_native_crash_log()
    try:
        assert path == log_dir() / "native-crash.log"
        assert "retouch started" in path.read_text(encoding="utf-8")
        assert calls and calls[0]["file"].name == str(path)
        assert calls[0]["all_threads"] is True
        # Idempotent: a second call does not reopen or re-register.
        assert diagnostics.enable_native_crash_log() == path
        assert len(calls) == 1
    finally:
        diagnostics._native_crash_stream.close()


def test_native_crash_logging_fails_soft_without_file_or_stderr(tmp_path, monkeypatch):
    import faulthandler
    import io
    import retouch.diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "_native_crash_stream", None)
    monkeypatch.setattr(diagnostics, "log_dir", lambda: tmp_path)
    monkeypatch.setattr(
        diagnostics, "open", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("denied")),
        raising=False,
    )
    monkeypatch.setattr(
        faulthandler, "enable", lambda **kwargs: (_ for _ in ()).throw(io.UnsupportedOperation("fileno")),
    )

    assert diagnostics.enable_native_crash_log() is None


def test_native_crash_leaves_a_traceback_on_disk(tmp_path):
    """A real segfault in a child process must end up in native-crash.log."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, HOME=str(tmp_path), LOCALAPPDATA=str(tmp_path))
    code = (
        "from retouch.diagnostics import enable_native_crash_log; "
        "p = enable_native_crash_log(); print(p, flush=True); "
        "import ctypes; ctypes.string_at(0)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    log_path = proc.stdout.strip().splitlines()[0]
    text = open(log_path, encoding="utf-8").read()
    assert "Fatal Python error" in text
    assert "string_at" in text or "<string>" in text


def test_clear_diagnostics_removes_native_crash_log(tmp_path, monkeypatch):
    monkeypatch.setenv("RETOUCH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    native = log_dir() / "native-crash.log"
    native.write_text("Fatal Python error", encoding="utf-8")
    clear_diagnostics()
    assert not native.exists()


def test_diagnostics_report_names_native_crash_log():
    assert "native crash log:" in diagnostics_report()
