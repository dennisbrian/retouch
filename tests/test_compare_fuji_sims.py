"""Tests for scripts/review/compare_fuji_sims.py -- Phase 1.c comparison script.

The script produces a side-by-side comparison of the 3 official Fuji film
simulations. These tests verify the script:

    1. Exists at the expected path.
    2. Can be imported without syntax errors.
    3. Exposes a ``main()`` function.
    4. Generates a synthetic test image (no engine required).
    5. Runs end-to-end and produces all expected output files (engine + models
       required -- skipped gracefully if either is missing).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Make sure the project root is importable so ``retouch`` resolves.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "review" / "compare_fuji_sims.py"


# ---------------------------------------------------------------------------
# Fixture: import the script as a module (no execution of main())
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def compare_module():
    """Import ``scripts/review/compare_fuji_sims.py`` as a module without running it."""
    if not SCRIPT_PATH.exists():
        pytest.skip(f"Script not found at {SCRIPT_PATH}")
    spec = importlib.util.spec_from_file_location("compare_fuji_sims", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Tests -- script structure
# ---------------------------------------------------------------------------

def test_script_file_exists():
    """The script must exist at scripts/review/compare_fuji_sims.py."""
    assert SCRIPT_PATH.is_file(), f"missing script: {SCRIPT_PATH}"


def test_script_imports_without_syntax_error(compare_module):
    """Importing the script must not raise (catches syntax + import errors)."""
    assert compare_module is not None


def test_script_exposes_main(compare_module):
    """The script must expose a callable main() function."""
    assert hasattr(compare_module, "main")
    assert callable(compare_module.main)


def test_script_exposes_helpers(compare_module):
    """All public helpers used by main() must be present."""
    for name in (
        "make_synthetic_image",
        "apply_sim",
        "make_comparison_grid",
        "compute_color_stats",
        "print_stats_table",
    ):
        assert hasattr(compare_module, name), f"missing helper: {name}"


def test_out_dir_constant(compare_module):
    """The script should write outputs under /tmp/sim_comparison."""
    out = Path(compare_module.OUT_DIR)
    assert str(out).startswith("/tmp/")


# ---------------------------------------------------------------------------
# Tests -- synthetic image generation (no engine required)
# ---------------------------------------------------------------------------

def test_make_synthetic_image_shape(compare_module):
    """Default synthetic image is 1280x720 uint8 BGR."""
    img = compare_module.make_synthetic_image(1280, 720)
    assert img.shape == (720, 1280, 3)
    assert img.dtype == np.uint8


def test_make_synthetic_image_has_content(compare_module):
    """Synthetic image must contain a real gradient (non-zero variance)."""
    img = compare_module.make_synthetic_image(1280, 720)
    # Standard deviation > 0 means there is real variation (not a flat colour).
    assert float(img.std()) > 5.0, f"image appears flat: std={img.std()}"


def test_make_synthetic_image_contains_color_patches(compare_module):
    """Verify the saturated red/green/blue swatches are present.

    The test image is supposed to contain RGB swatches for color-grade
    validation. We sample the centre of each swatch and check the channel
    values are roughly in the expected ballpark.
    """
    img = compare_module.make_synthetic_image(1280, 720)
    h, w = img.shape[:2]

    def sample(x_frac: float, y_frac: float) -> np.ndarray:
        return img[int(h * y_frac), int(w * x_frac), :]

    # Red swatch: BGR ~ (50, 50, 220)
    r = sample(0.585, 0.67)
    assert int(r[2]) > 180, f"red swatch R channel too low: {r}"
    # Green swatch: BGR ~ (60, 200, 60)
    g = sample(0.685, 0.67)
    assert int(g[1]) > 150, f"green swatch G channel too low: {g}"
    # Blue swatch: BGR ~ (220, 130, 30)
    b = sample(0.785, 0.67)
    assert int(b[0]) > 180, f"blue swatch B channel too low: {b}"


# ---------------------------------------------------------------------------
# Tests -- color stats helper
# ---------------------------------------------------------------------------

def test_compute_color_stats_identity(compare_module):
    """Identity comparison: source == output must yield zero shift."""
    img = compare_module.make_synthetic_image(128, 128)
    stats = compare_module.compute_color_stats(img, img)
    assert stats["mean_abs_diff"] == 0.0
    assert stats["b_shift"] == 0.0
    assert stats["g_shift"] == 0.0
    assert stats["r_shift"] == 0.0


def test_compute_color_stats_brightness(compare_module):
    """Brightening by +10 should give mean_abs_diff ~ 10 and positive shifts."""
    img = compare_module.make_synthetic_image(128, 128)
    out = np.clip(img.astype(np.int16) + 10, 0, 255).astype(np.uint8)
    stats = compare_module.compute_color_stats(img, out)
    assert 9.0 < stats["mean_abs_diff"] < 11.0
    assert stats["b_shift"] == pytest.approx(10.0, abs=1.0)
    assert stats["g_shift"] == pytest.approx(10.0, abs=0.5)
    assert stats["r_shift"] == pytest.approx(10.0, abs=0.5)


# ---------------------------------------------------------------------------
# Tests -- main() end-to-end (engine + models required)
# ---------------------------------------------------------------------------

def _engine_available() -> bool:
    """Return True if RetouchEngine can be initialised (models + deps present)."""
    models_dir = _PROJECT_ROOT / "models"
    return (models_dir / "face_landmarker.task").exists()


@pytest.mark.skipif(
    not _engine_available(),
    reason="Face landmarker model not available -- engine init would fail",
)
def test_main_runs_end_to_end(tmp_path, compare_module, monkeypatch):
    """Run main() and verify all expected output files are written.

    Uses a temporary OUT_DIR so the test does not pollute /tmp.
    """
    test_out = tmp_path / "sim_comparison"
    test_out.mkdir()
    monkeypatch.setattr(compare_module, "OUT_DIR", test_out)

    rc = compare_module.main()
    assert rc == 0

    for name in (
        "source.png",
        "classic_chrome.png",
        "astia.png",
        "provia.png",
        "comparison.png",
    ):
        path = test_out / name
        assert path.is_file(), f"expected output missing: {path}"
        assert path.stat().st_size > 0, f"output is empty: {path}"
