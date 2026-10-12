"""CLI wiring for --edit-report and --sign-cert/--sign-key."""

import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import cli
from retouch.content_credentials import CredentialsError


def _args(**kw):
    base = dict(edit_report=False, signing=None)
    base.update(kw)
    return SimpleNamespace(**base)


def test_options_off_by_default():
    assert cli._credentials_options(_args()) is None


def test_options_report_only():
    assert cli._credentials_options(_args(edit_report=True)) == {"edit_report": True, "signing": None}


def test_report_thumb_is_small_copy():
    img = np.zeros((3000, 2000, 3), np.uint8)
    thumb = cli._report_thumb(img)
    assert max(thumb.shape[:2]) == 1024
    small = np.zeros((10, 10, 3), np.uint8)
    copy = cli._report_thumb(small)
    copy[:] = 1
    assert small.max() == 0


def _write_pair(tmp_path):
    src = tmp_path / "in.jpg"
    out = tmp_path / "out" / "in.jpg"
    out.parent.mkdir()
    before = np.full((40, 60, 3), 100, np.uint8)
    after = before.copy()
    after[:10] = 160
    cv2.imwrite(str(src), before)
    cv2.imwrite(str(out), after)
    return src, out, before, after


def test_write_report_from_engine_result(tmp_path):
    src, out, before, after = _write_pair(tmp_path)
    processed = SimpleNamespace(params={"smooth": 30, "active_recipe": "natural"}, face_count=1)
    info = cli._write_credentials({"edit_report": True, "signing": None}, src, out,
                                  processed, after, before, {"recipe": "natural"})
    report = json.loads((out.parent / "edit-reports" / "in.jpg.json").read_text())
    assert info == {"edit_report": str(out.parent / "edit-reports" / "in.jpg.json")}
    assert report["recipe"] == "natural"
    assert report["edits"]["skin"]["settings"] == {"smooth": 30}
    assert report["pixels"]["changed_pct"] == pytest.approx(25.0, abs=1.0)
    assert report["content_credentials"]["signed"] is False


def test_global_only_result_falls_back_to_cli_params(tmp_path):
    src, out, before, after = _write_pair(tmp_path)
    cli._write_credentials({"edit_report": True, "signing": None}, src, out,
                           after, after, before, {"recipe": "natural", "saturation": 10},
                           global_only=True)
    report = json.loads((out.parent / "edit-reports" / "in.jpg.json").read_text())
    assert report["edits"] == {"colour": {"label": "Colour, tone and finish",
                                          "settings": {"saturation": 10}}}


def test_signing_failure_removes_unsigned_output(tmp_path, monkeypatch):
    src, out, before, after = _write_pair(tmp_path)

    def boom(*_a, **_k):
        raise CredentialsError("bad key")

    monkeypatch.setattr("retouch.content_credentials.sign_output", boom)
    with pytest.raises(CredentialsError):
        cli._write_credentials({"edit_report": True, "signing": object()}, src, out,
                               None, after, before, {})
    assert not out.exists()
    assert not (out.parent / "edit-reports").exists()


def test_cert_without_key_is_a_usage_error(tmp_path, monkeypatch, capsys):
    cert = tmp_path / "c.pem"
    cert.write_text("-----BEGIN CERTIFICATE-----\n")
    monkeypatch.setattr(sys, "argv", ["cli.py", str(tmp_path), "--sign-cert", str(cert)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "--sign-cert and --sign-key must be given together" in capsys.readouterr().err
