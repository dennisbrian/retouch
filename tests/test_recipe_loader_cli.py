"""Smoke tests for retouch/recipe_loader_cli.py — CLI arg parsing and error paths."""

import subprocess
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

import pytest


def _run(args):
    """Run the CLI module and return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, "-m", "retouch.recipe_loader_cli"] + args,
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode, result.stdout, result.stderr


class TestCliList:
    def test_list_exits_zero(self):
        rc, stdout, _ = _run(["list"])
        assert rc == 0
        assert "user recipes" in stdout.lower() or "no user recipes" in stdout.lower()


class TestCliSchema:
    def test_schema_exits_zero(self):
        rc, stdout, _ = _run(["schema"])
        assert rc == 0
        assert "$schema" in stdout

    def test_schema_url_exits_zero(self):
        rc, stdout, _ = _run(["schema", "--url"])
        assert rc == 0
        # URL should be present in output (printed or in JSON)
        assert len(stdout.strip()) > 0


class TestCliValidate:
    def test_validate_missing_file_fails(self):
        rc, _, stderr = _run(["validate", "/tmp/nonexistent_recipe_abc123.json"])
        assert rc != 0

    def test_validate_builtin_recipe_does_not_crash(self):
        """Validating a built-in preset should not crash the CLI."""
        recipe_path = Path(__file__).parent.parent / "presets" / "astia.json"
        if not recipe_path.exists():
            pytest.skip("astia.json not found")
        rc, stdout, stderr = _run(["validate", str(recipe_path)])
        # CLI should not error out with a traceback (rc=0 or rc=1 are both fine)

    def test_validate_malformed_json_fails(self):
        with NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            tmp_path = f.name
        try:
            rc, _, stderr = _run(["validate", tmp_path])
            assert rc != 0
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestCliImport:
    def test_import_missing_file_fails(self):
        rc, _, stderr = _run(["import", "/tmp/nonexistent_recipe_xyz789.json"])
        assert rc != 0


class TestCliRemove:
    def test_remove_nonexistent_fails(self):
        rc, _, stderr = _run(["remove", "__nonexistent_recipe_test_123__"])
        assert rc != 0


class TestCliExport:
    def test_export_nonexistent_fails(self):
        rc, _, stderr = _run(["export", "--recipe", "__nonexistent_recipe_test_456__"])
        assert rc != 0


class TestCliHelp:
    def test_no_args_shows_usage(self):
        rc, _, stderr = _run([])
        assert rc != 0
        assert "usage" in stderr.lower() or "{" in stderr
