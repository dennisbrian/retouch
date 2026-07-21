"""Tests for scripts/recipes/generate_demo_luts.py — verifies generated demo LUTs are valid.

These tests regenerate the demo LUTs into a temp dir on each fixture
invocation so they are self-contained and do not depend on prior runs
of the generator.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from retouch.lut import list_available_luts, load_cube, luts_dir  # noqa: E402


def _load_generate_module():
    spec = importlib.util.spec_from_file_location(
        "generate_demo_luts", _SCRIPTS_DIR / "recipes" / "generate_demo_luts.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("could not load generate_demo_luts")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EXPECTED_NAMES: tuple = (
    "identity_33",
    "warm_boost_17",
    "cool_shadows_17",
    "kodak_ish_17",
    "kodak",
    "fuji",
    "bleach_bypass_17",
    "teal_orange_17",
    "cross_process_17",
)
EXPECTED_SIZES = {
    "identity_33": 33,
    "warm_boost_17": 17,
    "cool_shadows_17": 17,
    "kodak_ish_17": 17,
    "kodak": 17,
    "fuji": 17,
    "bleach_bypass_17": 17,
    "teal_orange_17": 17,
    "cross_process_17": 17,
}


@pytest.fixture
def gen_dir():
    """Generate demo LUTs into a temp dir and yield (out_dir, paths)."""
    gen = _load_generate_module()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        paths = gen.generate(out_dir)
        yield out_dir, paths


def test_generator_creates_all_expected_files(gen_dir):
    out_dir, paths = gen_dir
    assert len(paths) == len(EXPECTED_NAMES)
    for name in EXPECTED_NAMES:
        p = out_dir / f"{name}.cube"
        assert p.exists()
        assert p.stat().st_size > 0


def test_generator_is_idempotent():
    """Re-running generate() in the same dir must not error."""
    gen = _load_generate_module()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        gen.generate(out_dir)
        second = gen.generate(out_dir)
        assert len(second) == len(EXPECTED_NAMES)
        for p in second:
            assert p.exists()


def test_each_lut_loads_with_correct_size(gen_dir):
    out_dir, _ = gen_dir
    for name in EXPECTED_NAMES:
        lut = load_cube(out_dir / f"{name}.cube")
        assert lut.size == EXPECTED_SIZES[name]
        assert lut.array.shape == (EXPECTED_SIZES[name],) * 3 + (3,)


def test_identity_round_trip(gen_dir):
    """identity_33 LUT applied to a random image must reproduce the image."""
    out_dir, _ = gen_dir
    lut = load_cube(out_dir / "identity_33.cube")
    rng = np.random.default_rng(7)
    img = rng.integers(0, 256, size=(16, 24, 3), dtype=np.uint8)
    out = lut.apply(img)
    np.testing.assert_allclose(out, img, atol=1)


@pytest.mark.parametrize("lut_name", EXPECTED_NAMES)
def test_luts_preserve_endpoints(gen_dir, lut_name):
    """Pure black in -> pure black out; pure white in -> pure white out."""
    out_dir, _ = gen_dir
    lut = load_cube(out_dir / f"{lut_name}.cube")
    black = np.zeros((1, 1, 3), dtype=np.uint8)
    white = np.full((1, 1, 3), 255, dtype=np.uint8)
    np.testing.assert_allclose(lut.apply(black), black, atol=1)
    np.testing.assert_allclose(lut.apply(white), white, atol=1)


def test_warm_boost_warms_midtones(gen_dir):
    out_dir, _ = gen_dir
    lut = load_cube(out_dir / "warm_boost_17.cube")
    mid = np.full((1, 1, 3), 128, dtype=np.uint8)
    out = lut.apply(mid)
    r = int(out[0, 0, 2])
    b = int(out[0, 0, 0])
    assert r > b, f"warm boost must produce R > B on neutral mid-grey, got R={r} B={b}"


def test_cool_shadows_cools_dark_pixels(gen_dir):
    out_dir, _ = gen_dir
    lut = load_cube(out_dir / "cool_shadows_17.cube")
    dark = np.full((1, 1, 3), 10, dtype=np.uint8)
    out = lut.apply(dark)
    r = int(out[0, 0, 2])
    b = int(out[0, 0, 0])
    assert b > r, f"cool shadows must produce B > R on dark grey, got B={b} R={r}"
    assert r <= 10, f"cool shadows must not raise R, got R={r}"


def test_kodak_ish_lifts_midtones(gen_dir):
    out_dir, _ = gen_dir
    lut = load_cube(out_dir / "kodak_ish_17.cube")
    mid = np.full((1, 1, 3), 128, dtype=np.uint8)
    out = lut.apply(mid)
    g = int(out[0, 0, 1])
    assert g > 128, f"kodak_ish G midtone lift must raise G above 128, got G={g}"


def test_bleach_bypass_desaturates_midtones(gen_dir):
    """Bleach bypass pulls chroma toward neutral in midtones."""
    out_dir, _ = gen_dir
    lut = load_cube(out_dir / "bleach_bypass_17.cube")
    # Saturated red midtone.
    red_mid = np.array([[[40, 20, 200]]], dtype=np.uint8)  # BGR
    out = lut.apply(red_mid)
    r_in, b_in = 200, 40
    r_out, b_out = int(out[0, 0, 2]), int(out[0, 0, 0])
    assert (r_in - b_in) > (r_out - b_out), (
        f"bleach bypass must reduce R-B spread on saturated red mid, "
        f"got in={(r_in, b_in)} out={(r_out, b_out)}"
    )


def test_teal_orange_splits_shadows_and_highlights(gen_dir):
    """Shadows drift teal (B>R), highlights drift orange (R>B)."""
    out_dir, _ = gen_dir
    lut = load_cube(out_dir / "teal_orange_17.cube")
    dark = np.full((1, 1, 3), 40, dtype=np.uint8)
    bright = np.full((1, 1, 3), 215, dtype=np.uint8)
    dark_out = lut.apply(dark)[0, 0]
    bright_out = lut.apply(bright)[0, 0]
    assert int(dark_out[0]) > int(dark_out[2]), (
        f"shadow must be teal (B>R), got B={dark_out[0]} R={dark_out[2]}"
    )
    assert int(bright_out[2]) > int(bright_out[0]), (
        f"highlight must be orange (R>B), got R={bright_out[2]} B={bright_out[0]}"
    )


def test_cross_process_lifts_green_in_midtones(gen_dir):
    """Cross-process dominant tell: green midtone lift."""
    out_dir, _ = gen_dir
    lut = load_cube(out_dir / "cross_process_17.cube")
    mid = np.full((1, 1, 3), 128, dtype=np.uint8)
    out = lut.apply(mid)
    g = int(out[0, 0, 1])
    assert g > 128, f"cross_process G midtone lift must raise G above 128, got G={g}"


def test_luts_dir_sees_generated_files():
    """If the canonical luts/ dir has been populated, list_available_luts() sees them."""
    available = set(list_available_luts())
    missing = set(EXPECTED_NAMES) - available
    if missing:
        pytest.skip(
            f"Demo LUTs not present in {luts_dir()}: missing {sorted(missing)}. "
            f"Run: python3 scripts/recipes/generate_demo_luts.py"
        )
    for name in EXPECTED_NAMES:
        lut = load_cube(luts_dir() / f"{name}.cube")
        assert lut.size == EXPECTED_SIZES[name]
