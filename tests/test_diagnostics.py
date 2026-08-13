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
