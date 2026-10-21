"""Tests for --set / --list-params (generic engine-param exposure) in cli.py."""

import argparse

import pytest

import cli
from cli import _build_set_params, _parse_set_value, _snapshot_original


class TestBuildSetParams:
    def test_valid_int_with_range(self):
        assert _build_set_params(["smooth=42"]) == {"smooth": 42}

    def test_valid_float_no_flag_param(self):
        # relight_azimuth has no dedicated CLI flag; --set reaches it.
        assert _build_set_params(["relight_azimuth=90.5"]) == {
            "relight_azimuth": 90.5
        }

    def test_valid_str_dropdown(self):
        assert _build_set_params(["mv2_eyeshadow_color=rose"]) == {
            "mv2_eyeshadow_color": "rose"
        }

    def test_bool_true_false(self):
        assert _build_set_params(["nose_blush=true"]) == {"nose_blush": True}
        assert _build_set_params(["nose_blush=false"]) == {"nose_blush": False}

    def test_repeatable(self):
        out = _build_set_params(["smooth=1", "pore_synthesis=2"])
        assert out == {"smooth": 1, "pore_synthesis": 2}

    def test_unknown_param_lists_valid_names(self):
        with pytest.raises(ValueError, match="valid names"):
            _build_set_params(["bogus_param=1"])

    def test_bad_int_value(self):
        with pytest.raises(ValueError, match="expected int"):
            _build_set_params(["smooth=notanint"])

    def test_out_of_range(self):
        with pytest.raises(ValueError, match="above maximum"):
            _build_set_params(["smooth=999"])

    def test_missing_equals(self):
        with pytest.raises(ValueError, match="name=value"):
            _build_set_params(["smooth"])

    def test_none_or_empty(self):
        assert _build_set_params(None) == {}
        assert _build_set_params([]) == {}


class TestBuildParamsSetPrecedence:
    def _args(self, **kw):
        base = dict(
            recipe=None, preset=None, face_params=None, color_grade=None,
            lip_tint=None, color_ref=None, match_hero=None, set=None,
        )
        base.update(kw)
        return argparse.Namespace(**base)

    def test_set_beats_specific_flag(self):
        params = cli.build_params(self._args(recipe="natural", smooth=10, set=["smooth=99"]))
        assert params["smooth"] == 99

    def test_set_reaches_flagless_param(self):
        params = cli.build_params(self._args(recipe="natural", set=["relight_azimuth=-45"]))
        assert params["relight_azimuth"] == -45


class TestSnapshotOriginal:
    def test_no_compare_returns_none(self):
        import numpy as np

        assert _snapshot_original(np.zeros((4, 4, 3), np.uint8), False) is None

    def test_uint8_copies(self):
        import numpy as np

        img = np.zeros((4, 4, 3), np.uint8)
        snap = _snapshot_original(img, True)
        assert snap is not img and snap.dtype == np.uint8

    def test_float32_normalized_to_uint8(self):
        import numpy as np

        img = np.full((4, 4, 3), 128.5, np.float32)
        snap = _snapshot_original(img, True)
        assert snap.dtype == np.uint8

    def test_clips_out_of_range_floats(self):
        import numpy as np

        img = np.full((4, 4, 3), 300.0, np.float32)
        assert _snapshot_original(img, True).max() == 255


class TestCliFlags:
    """Subprocess tests mirroring tests/test_cli.py's _run_cli pattern."""

    def _run_cli(self, *args, timeout=60):
        import subprocess
        import sys
        from pathlib import Path

        cli_path = Path(__file__).resolve().parent.parent / "cli.py"
        result = subprocess.run(
            [sys.executable, str(cli_path), *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    def test_list_params_runs(self):
        rc, out, err = self._run_cli("--list-params")
        assert rc == 0, err
        assert "smooth" in out
        assert "relight_azimuth" in out
        assert out.rstrip().endswith(")")

    def test_unknown_param_errors(self):
        rc, out, err = self._run_cli("--set", "no_such_param=1", "x.jpg")
        assert rc != 0
        assert "Unknown engine param" in err

    def test_set_reaches_engine(self, tmp_path, monkeypatch):
        """--set with a valid param must reach engine.process (spy via main)."""
        import numpy as np

        cv2 = pytest.importorskip("cv2")
        engine = cli.RetouchEngine()
        calls = {}

        real_process = engine.process

        def spy(img, progress_cb=None, **kw):
            calls.update(kw)
            return real_process(img, progress_cb=progress_cb, **kw)

        monkeypatch.setattr(cli, "RetouchEngine", lambda: engine)
        monkeypatch.setattr(engine, "process", spy)

        rng = np.random.default_rng(0)
        face = np.zeros((256, 256, 3), np.uint8)
        cv2.circle(face, (128, 128), 60, (200, 180, 170), -1)
        cv2.circle(face, (108, 120), 8, (60, 50, 50), -1)
        cv2.circle(face, (148, 120), 8, (60, 50, 50), -1)
        inp = tmp_path / "in.jpg"
        cv2.imwrite(str(inp), face)
        outdir = tmp_path / "out"
        outdir.mkdir()

        import sys

        rc = 0
        old_argv = sys.argv
        sys.argv = [
            "cli.py", str(inp), "-o", str(outdir), "--recipe", "natural",
            "--max-dim", "200", "--no-compare", "--force", "--workers", "1",
            "--set", "smooth=77",
        ]
        try:
            cli.main()
        except SystemExit as e:
            rc = e.code or 0
        finally:
            sys.argv = old_argv

        assert rc == 0
        assert calls.get("smooth") == 77
        assert list(outdir.iterdir()), "no output written"
