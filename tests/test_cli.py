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


# ---------------------------------------------------------------------------
# --micro-restore flag smoke test
# ---------------------------------------------------------------------------

class TestMicroRestoreFlag:
    """End-to-end smoke test for the --micro-restore CLI flag.

    The flag is auto-derived from ``ParamSpec(cli_flag="micro-restore")`` in
    ``retouch/params.py`` and is exposed on the CLI as ``--micro-restore``.
    argparse converts the hyphen to an underscore, so the parsed attribute
    is ``args.micro_restore`` and it flows into ``engine.process(micro_restore=...)``.
    """

    def _run_with_micro_restore(self, value, image_path, tmp_path):
        """Invoke the CLI with --micro-restore=<value> and return (rc, out_path)."""
        cv2 = pytest.importorskip("cv2")

        input_path = Path(str(image_path))
        assert input_path.exists(), f"Test image missing: {input_path}"

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        out_path = output_dir / f"{input_path.stem}.jpg"

        args = [
            str(input_path),
            "-o", str(output_dir),
            "--max-dim", "400",
            "--no-compare",
            "--force",
        ]
        if value is not None:
            args.extend(["--micro-restore", str(value)])

        rc, out, err = _run_cli(*args, timeout=120)
        return rc, out, err, out_path

    def test_cli_accepts_micro_restore_flag(self, natural_image_path, tmp_path):
        """--micro-restore=25 must succeed end-to-end and produce a valid JPEG."""
        if natural_image_path is None:
            pytest.skip("No natural test image available in test_output/")

        rc, out, err, out_path = self._run_with_micro_restore(
            25, natural_image_path, tmp_path,
        )
        assert rc == 0, f"CLI failed (rc={rc}): {err}\nstdout: {out}"

        assert out_path.exists(), f"Output file not created: {out_path}"
        assert out_path.stat().st_size > 0, "Output file is empty"

        cv2 = pytest.importorskip("cv2")
        img = cv2.imread(str(out_path))
        assert img is not None, f"Output is not a valid readable JPEG: {out_path}"
        assert img.ndim == 3 and img.shape[2] == 3, (
            f"Output has unexpected shape: {img.shape}"
        )

    def test_cli_micro_restore_zero_is_valid(self, natural_image_path, tmp_path):
        """--micro-restore=0 should be accepted (treated as off) and succeed."""
        if natural_image_path is None:
            pytest.skip("No natural test image available in test_output/")

        rc, out, err, out_path = self._run_with_micro_restore(
            0, natural_image_path, tmp_path,
        )
        assert rc == 0, f"CLI failed (rc={rc}): {err}\nstdout: {out}"
        assert out_path.exists(), f"Output file not created: {out_path}"
        assert out_path.stat().st_size > 0, "Output file is empty"

        cv2 = pytest.importorskip("cv2")
        img = cv2.imread(str(out_path))
        assert img is not None, f"Output is not a valid readable JPEG: {out_path}"

    def test_cli_micro_restore_default_is_twenty(self, natural_image_path, tmp_path):
        """Invoking without --micro-restore should succeed using the spec default (20)."""
        if natural_image_path is None:
            pytest.skip("No natural test image available in test_output/")

        rc, out, err, out_path = self._run_with_micro_restore(
            None, natural_image_path, tmp_path,
        )
        assert rc == 0, f"CLI failed (rc={rc}): {err}\nstdout: {out}"
        assert out_path.exists(), f"Output file not created: {out_path}"
        assert out_path.stat().st_size > 0, "Output file is empty"

        cv2 = pytest.importorskip("cv2")
        img = cv2.imread(str(out_path))
        assert img is not None, f"Output is not a valid readable JPEG: {out_path}"


import numpy as np
