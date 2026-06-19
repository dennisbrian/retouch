"""Tests for the CLI entry point (cli.py).

Uses subprocess to invoke the CLI as a standalone script.
"""

import subprocess
import sys
import os
from pathlib import Path

import pytest


CLI_PATH = Path(__file__).resolve().parent.parent / "cli.py"


def _run_cli(*args, timeout=30):
    """Run cli.py with given args; return (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


# ---------------------------------------------------------------------------
# --help and argument parsing
# ---------------------------------------------------------------------------

class TestHelpAndArgs:
    def test_help(self):
        rc, out, err = _run_cli("--help")
        assert rc == 0
        assert "usage:" in out.lower() or "usage:" in err.lower()

    def test_dry_run_no_input(self):
        """--dry-run without input should fail with usage."""
        rc, out, err = _run_cli("--dry-run")
        assert rc != 0

    def test_dry_run_on_synthetic(self, tmp_path, synthetic_face):
        """Write synthetic face to temp dir and dry-run it."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        cv2 = pytest.importorskip("cv2")
        cv2.imwrite(str(input_dir / "face.jpg"), synthetic_face)

        rc, out, err = _run_cli(str(input_dir), "--dry-run", "--recipe", "natural")
        # dry-run should exit 0 after printing what it would process
        assert rc == 0, f"stderr: {err}"


# ---------------------------------------------------------------------------
# Basic processing
# ---------------------------------------------------------------------------

class TestProcessing:
    def test_process_single_image(self, tmp_path, synthetic_face):
        """Process a single synthetic image and write output."""
        cv2 = pytest.importorskip("cv2")

        input_path = tmp_path / "input.jpg"
        cv2.imwrite(str(input_path), synthetic_face)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--recipe", "natural",
            "--max-dim", "200",
            "--no-compare",
            "--force",
            timeout=60,
        )
        assert rc == 0, f"CLI failed: {err}"

        # Check that output was written
        out_files = list(output_dir.iterdir())
        assert len(out_files) >= 1, f"No output files in {output_dir}"

    def test_process_with_recipe(self, tmp_path, synthetic_face):
        """Try a few different recipes."""
        cv2 = pytest.importorskip("cv2")

        input_path = tmp_path / "input.jpg"
        cv2.imwrite(str(input_path), synthetic_face)

        for recipe in ("natural", "cosplay", "beauty"):
            output_dir = tmp_path / f"output_{recipe}"
            output_dir.mkdir()

            rc, out, err = _run_cli(
                str(input_path),
            "-o", str(output_dir),
                "--recipe", recipe,
                "--max-dim", "200",
                "--no-compare",
                "--force",
                timeout=60,
            )
            assert rc == 0, f"CLI failed for recipe '{recipe}': {err}"

    def test_global_only_mode(self, tmp_path, synthetic_face):
        """--global-only should skip face detection."""
        cv2 = pytest.importorskip("cv2")

        input_path = tmp_path / "input.jpg"
        cv2.imwrite(str(input_path), synthetic_face)

        output_dir = tmp_path / "global_only"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--global-only",
            "--max-dim", "200",
            "--no-compare",
            "--force",
            timeout=60,
        )
        assert rc == 0, f"CLI failed: {err}"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unsupported_format(self, tmp_path):
        input_path = tmp_path / "input.txt"
        input_path.write_text("not an image")

        rc, out, err = _run_cli(
            str(input_path),
            "--dry-run",
        )
        # CLI skips unsupported files (prints warning, exits 0 when no work)
        assert rc == 0

    def test_nonexistent_input(self, tmp_path):
        rc, out, err = _run_cli(
            str(tmp_path / "does_not_exist.jpg"),
            "--dry-run",
        )
        assert rc != 0

    def test_directory_recursion(self, tmp_path, synthetic_face):
        """Process a directory of images."""
        cv2 = pytest.importorskip("cv2")
        (tmp_path / "sub").mkdir()
        cv2.imwrite(str(tmp_path / "sub" / "img1.jpg"), synthetic_face)
        cv2.imwrite(str(tmp_path / "sub" / "img2.jpg"), synthetic_face)

        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(tmp_path / "sub"),
            "-o", str(output_dir),
            "--max-dim", "200",
            "--no-compare",
            "--force",
            "--workers", "1",
            timeout=120,
        )
        assert rc == 0, f"CLI failed: {err}"
        out_files = list(output_dir.iterdir())
        assert len(out_files) >= 2


import numpy as np
