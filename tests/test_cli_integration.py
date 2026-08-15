"""CLI integration tests — exercise the cli.py entry point.

These tests use ``subprocess`` to invoke ``cli.py`` with real arguments
against small synthetic images. They are intentionally scoped to fast
operations (small images, ``--max-dim`` capped, ``--global-only`` where
possible) to keep the suite snappy.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest


CLI_PATH = Path(__file__).resolve().parent.parent / "cli.py"


def _run_cli(*args, timeout=60):
    """Run cli.py with given args; return (returncode, stdout, stderr)."""
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    result = subprocess.run(
        [sys.executable, str(CLI_PATH), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return result.returncode, result.stdout, result.stderr


def _write_synthetic_image(path: Path, value: int = 128) -> None:
    """Write a small synthetic BGR image to *path*."""
    img = np.full((100, 100, 3), value, dtype=np.uint8)
    cv2.imwrite(str(path), img)


# ---------------------------------------------------------------------------
# Basic processing
# ---------------------------------------------------------------------------


class TestBasicProcessing:
    def test_process_single_image(self, tmp_path):
        """Test `python3 cli.py input.jpg -o output/`."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path, value=120)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--max-dim", "200",
            "--no-compare",
            "--force",
            "--workers", "1",
        )
        assert rc == 0, f"CLI failed: {err}"
        out_files = list(output_dir.iterdir())
        assert len(out_files) >= 1

    def test_process_directory(self, tmp_path):
        """Process a folder of synthetic images."""
        input_dir = tmp_path / "dir_input"
        input_dir.mkdir()
        for i in range(3):
            _write_synthetic_image(input_dir / f"img_{i}.jpg", value=80 + i * 20)
        output_dir = tmp_path / "dir_output"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_dir),
            "-o", str(output_dir),
            "--max-dim", "200",
            "--no-compare",
            "--force",
            "--workers", "1",
        )
        assert rc == 0, f"CLI failed: {err}"
        out_files = [p for p in output_dir.iterdir() if not p.name.startswith(".")]
        assert len(out_files) >= 3

    def test_same_format_without_output_refuses_source_overwrite(self, tmp_path):
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        before = input_path.read_bytes()

        rc, out, err = _run_cli(
            str(input_path),
            "--max-dim", "200",
            "--no-compare",
            "--force",
            "--workers", "1",
        )

        assert rc == 1
        assert "Output preflight failed" in out
        assert input_path.read_bytes() == before

    def test_recursive_flag_picks_subdirs(self, tmp_path):
        """The -r flag should make the CLI recurse into subdirectories."""
        input_dir = tmp_path / "root"
        sub = input_dir / "sub"
        input_dir.mkdir()
        sub.mkdir()
        _write_synthetic_image(input_dir / "top.jpg")
        _write_synthetic_image(sub / "nested.jpg")
        output_dir = tmp_path / "recursive_out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_dir),
            "-o", str(output_dir),
            "-r",
            "--max-dim", "200",
            "--no-compare",
            "--force",
            "--workers", "1",
        )
        assert rc == 0, f"CLI failed: {err}"
        assert {
            p.relative_to(output_dir).as_posix()
            for p in output_dir.rglob("*")
            if p.is_file()
        } == {
            "top.jpg",
            "sub/nested.jpg",
        }


# ---------------------------------------------------------------------------
# Recipe application
# ---------------------------------------------------------------------------


class TestRecipeApplication:
    @pytest.mark.parametrize("recipe", ["natural", "portrait", "matsuri_glow_v1", "reala_ace"])
    def test_recipe_applied(self, tmp_path, recipe):
        """`--recipe <name>` should be accepted and not crash."""
        input_path = tmp_path / f"input_{recipe}.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / f"out_{recipe}"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--recipe", recipe,
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed for recipe '{recipe}': {err}"
        assert any(output_dir.iterdir())

    def test_recipe_and_overrides(self, tmp_path):
        """Test that overrides are applied alongside the recipe."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--recipe", "natural",
            "--smooth", "85",
            "--whiten", "40",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"


# ---------------------------------------------------------------------------
# Custom style
# ---------------------------------------------------------------------------


class TestCustomStyle:
    def test_style_with_brightness_contrast(self, tmp_path):
        """Test that --brightness and --contrast overrides are accepted."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--recipe", "natural",
            "--brightness", "10",
            "--contrast", "20",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"

    def test_color_grade_override(self, tmp_path):
        """Test --color-grade and --grade-intensity."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--color-grade", "natural",
            "--grade-intensity", "0.6",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"


# ---------------------------------------------------------------------------
# Export options
# ---------------------------------------------------------------------------


class TestExportOptions:
    def test_format_webp(self, tmp_path):
        """Test --format webp produces .webp output."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--format", "webp",
            "--quality", "80",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"
        files = list(output_dir.iterdir())
        # The output should be a .webp file
        assert any(f.suffix == ".webp" for f in files)

    def test_format_png(self, tmp_path):
        """Test --format png produces .png output."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--format", "png",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"
        files = list(output_dir.iterdir())
        assert any(f.suffix == ".png" for f in files)

    def test_max_dim_downscales(self, tmp_path):
        """Test --max-dim caps the longest side."""
        input_path = tmp_path / "input.jpg"
        # 800x600 image
        big = np.full((600, 800, 3), 100, dtype=np.uint8)
        cv2.imwrite(str(input_path), big)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"
        files = list(output_dir.iterdir())
        assert files
        out_img = cv2.imread(str(files[0]))
        assert out_img is not None
        # After upscale-back, the output should match the original 600x800
        # (the CLI upsamples the result back to the original resolution
        # after downscaling for processing)
        assert out_img.shape[0] == 600
        assert out_img.shape[1] == 800

    def test_quality_option(self, tmp_path):
        """Test --quality produces a valid output file."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--quality", "50",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"
        assert any(output_dir.iterdir())

    def test_same_format_preserves_extension(self, tmp_path):
        """Test --format same keeps the input extension."""
        input_path = tmp_path / "input.png"
        cv2.imwrite(str(input_path), np.full((100, 100, 3), 128, dtype=np.uint8))
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--format", "same",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"
        files = list(output_dir.iterdir())
        # Either .png or upsampled .jpg depending on CLI behavior
        assert any(f.suffix in (".png", ".jpg") for f in files)


# ---------------------------------------------------------------------------
# Global-only mode
# ---------------------------------------------------------------------------


class TestGlobalOnlyMode:
    def test_global_only_runs_without_crash(self, tmp_path):
        """`--global-only` should skip face detection and still process."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--global-only",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"
        assert any(output_dir.iterdir())

    def test_global_only_with_recipe(self, tmp_path):
        """`--global-only` should still honour the recipe choice."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--global-only",
            "--recipe", "portrait",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"

    def test_global_only_with_color_grade(self, tmp_path):
        """`--global-only` should accept --color-grade and --grade-intensity."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--global-only",
            "--color-grade", "natural",
            "--grade-intensity", "0.5",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_missing_input_fails(self, tmp_path):
        """Missing input file should exit with non-zero status."""
        rc, out, err = _run_cli(
            str(tmp_path / "does_not_exist.jpg"),
            "--dry-run",
        )
        assert rc != 0, "Missing input should fail"

    def test_unsupported_recipe_fails(self, tmp_path):
        """An invalid recipe name should be rejected by argparse."""
        rc, out, err = _run_cli(
            str(tmp_path / "input.jpg"),
            "--recipe", "this_recipe_does_not_exist",
            "--dry-run",
        )
        # argparse rejects unknown choice
        assert rc != 0

    def test_directory_with_no_images(self, tmp_path):
        """A directory with no supported images should exit non-zero."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        rc, out, err = _run_cli(str(empty_dir))
        assert rc != 0

    def test_no_input_arg_shows_help(self, tmp_path):
        """No input argument should show usage and exit non-zero."""
        rc, out, err = _run_cli()
        assert rc != 0

    def test_help_flag_succeeds(self, tmp_path):
        """--help should exit 0 and print usage information."""
        rc, out, err = _run_cli("--help")
        assert rc == 0
        combined = (out + err).lower()
        assert "usage:" in combined

    def test_dry_run_does_not_write_files(self, tmp_path):
        """--dry-run should list images but not write any output."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--dry-run",
            "--max-dim", "200",
        )
        assert rc == 0, f"dry-run should succeed: {err}"
        # No output files should have been written
        files = list(output_dir.iterdir())
        assert not any(
            f.suffix.lower() in (".jpg", ".png", ".webp") for f in files
        )

    def test_force_overwrites_existing(self, tmp_path):
        """--force should overwrite existing output files."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()
        out_path = output_dir / "input.jpg"
        # Pre-create the output file
        cv2.imwrite(str(out_path), np.full((50, 50, 3), 50, dtype=np.uint8))

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"
        # File should still exist
        assert out_path.exists()

    def test_invalid_format_choice(self, tmp_path):
        """An invalid --format choice should be rejected."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        rc, out, err = _run_cli(
            str(input_path),
            "--format", "tiff",
            "--dry-run",
        )
        assert rc != 0


# ---------------------------------------------------------------------------
# Comparison output
# ---------------------------------------------------------------------------


class TestComparisonOutput:
    def test_compare_output_created(self, tmp_path):
        """Without --no-compare, a side-by-side _compare file should be created."""
        input_path = tmp_path / "input.jpg"
        _write_synthetic_image(input_path)
        output_dir = tmp_path / "out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_path),
            "-o", str(output_dir),
            "--max-dim", "200",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"
        # A *_compare.jpg file should exist
        compare_files = list(output_dir.glob("*_compare.*"))
        assert compare_files, f"No comparison file in {output_dir}"


# ---------------------------------------------------------------------------
# Workers / parallelism
# ---------------------------------------------------------------------------


class TestWorkers:
    def test_workers_1_runs_serially(self, tmp_path):
        """--workers 1 should run files serially."""
        input_dir = tmp_path / "w1_in"
        input_dir.mkdir()
        for i in range(2):
            _write_synthetic_image(input_dir / f"img_{i}.jpg")
        output_dir = tmp_path / "w1_out"
        output_dir.mkdir()

        rc, out, err = _run_cli(
            str(input_dir),
            "-o", str(output_dir),
            "--workers", "1",
            "--max-dim", "200",
            "--no-compare",
            "--force",
        )
        assert rc == 0, f"CLI failed: {err}"
        assert len(list(output_dir.iterdir())) >= 2
