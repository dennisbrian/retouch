#!/usr/bin/env python3
"""Phase 1.b -- Generate demo ``.cube`` LUT files.

Produces a small set of safe-to-redistribute demo LUTs in ``luts/``:

  * ``identity_33.cube``         -- 33^3 identity LUT (round-trip test)
  * ``warm_boost_17.cube``       -- Mild warm midtone shift (R+, B-)
  * ``cool_shadows_17.cube``     -- Cool (cyan) tint in shadow regions
  * ``kodak_ish_17.cube``        -- Per-channel curves suggesting a "warm" film stock.
                                    NOT an authoritative Kodak profile.
  * ``kodak.cube``               -- Slightly stronger kodak_ish variant (film preset)
  * ``fuji.cube``                -- Classic Chrome-inspired cool greens (film preset alt)
  * ``bleach_bypass_17.cube``    -- Silver-retention look: high contrast + desaturated
  * ``teal_orange_17.cube``      -- Complementary split-tone (shadows teal, highlights orange)
  * ``cross_process_17.cube``    -- Slide-in-C-41 look: green mids, blue shadow lift

Re-runnable: regenerates all files each invocation (idempotent).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from retouch.lut import luts_dir  # noqa: E402


_LUMA_R = 0.2126
_LUMA_G = 0.7152
_LUMA_B = 0.0722


def _build_input_grid(size: int) -> np.ndarray:
    """Build a (size, size, size, 3) RGB float32 grid.

    grid[r_idx, g_idx, b_idx, :] = (R, G, B) with each axis sampled
    uniformly in [0, 1].
    """
    if size < 2:
        raise ValueError(f"size must be >= 2; got {size}")
    axis = np.linspace(0.0, 1.0, size, dtype=np.float32)
    grid = np.empty((size, size, size, 3), dtype=np.float32)
    grid[..., 0] = axis.reshape(size, 1, 1)
    grid[..., 1] = axis.reshape(1, size, 1)
    grid[..., 2] = axis.reshape(1, 1, size)
    return grid


def _warm_boost(grid: np.ndarray) -> np.ndarray:
    """Midtone-biased warm shift: R*1.10, B*0.95 weighted by a triangular bell.

    Black and white are preserved (bell is 0 at luma=0 and luma=1).
    """
    out = grid.copy()
    luma = _LUMA_R * out[..., 0] + _LUMA_G * out[..., 1] + _LUMA_B * out[..., 2]
    bell = np.clip(1.0 - np.abs(luma - 0.5) * 2.0, 0.0, 1.0)
    out[..., 0] = np.clip(out[..., 0] * (1.0 + 0.10 * bell), 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] * (1.0 - 0.05 * bell), 0.0, 1.0)
    return out


def _cool_shadows(grid: np.ndarray) -> np.ndarray:
    """Cool (cyan) tint in the shadow region, peaking around luma=0.1.

    Reduces R, boosts B. Pure black and white are preserved.
    """
    out = grid.copy()
    luma = _LUMA_R * out[..., 0] + _LUMA_G * out[..., 1] + _LUMA_B * out[..., 2]
    sigma = 0.1
    bump = luma * np.exp(-((luma - 0.1) / sigma) ** 2)
    shadow_w = np.clip(10.0 * bump, 0.0, 1.0)
    out[..., 0] = np.clip(out[..., 0] - 0.05 * shadow_w, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + 0.05 * shadow_w, 0.0, 1.0)
    return out


def _s_curve(x: np.ndarray, strength: float) -> np.ndarray:
    """Linear contrast: midpoint (0.5) preserved; values stretched by (1+strength)."""
    t = 2.0 * x - 1.0
    return np.clip((t * (1.0 + strength) + 1.0) * 0.5, 0.0, 1.0)


def _midtone_lift(x: np.ndarray, lift: float) -> np.ndarray:
    """Add a bump centered at x=0.5 (bell). Endpoints untouched."""
    bell = 4.0 * x * (1.0 - x)
    return np.clip(x + lift * bell, 0.0, 1.0)


def _shadow_crush(x: np.ndarray, crush: float) -> np.ndarray:
    """Lift the shadow toe. 0 at x=0 and x=1, peaks in the lower midtones."""
    return np.clip(x + crush * 4.0 * x * (1.0 - x) * (1.0 - x), 0.0, 1.0)


def _kodak_ish(grid: np.ndarray) -> np.ndarray:
    """Per-channel curves suggesting a 'warm' film stock.

    R: gentle S-curve (linear contrast +10%).
    G: midtone lift (+0.04 at L~0.5).
    B: shadow crush (lower mids lifted via 4x(1-x)^2 weighting).
    """
    out = np.empty_like(grid)
    out[..., 0] = _s_curve(grid[..., 0], strength=0.10)
    out[..., 1] = _midtone_lift(grid[..., 1], lift=0.04)
    out[..., 2] = _shadow_crush(grid[..., 2], crush=0.06)
    return out


def _kodak_warm(grid: np.ndarray) -> np.ndarray:
    """Demo 'kodak' preset LUT — warm midtones, gentle S-curve.

    Same as kodak_ish but slightly stronger to give presets a distinct look.
    """
    out = np.empty_like(grid)
    out[..., 0] = _s_curve(grid[..., 0], strength=0.12)
    out[..., 1] = _midtone_lift(grid[..., 1], lift=0.05)
    out[..., 2] = _shadow_crush(grid[..., 2], crush=0.05)
    return out


def _fuji_chrome(grid: np.ndarray) -> np.ndarray:
    """Demo 'fuji' preset LUT — cool greens/cyans, classic Chrome look.

    R: gentle highlight crush (slight reduction in highlights).
    G: cool shift in midtones (slight G reduction).
    B: linear (preserves blue fidelity).

    Endpoints preserved by using a bell weight (peaks at L=0.5, zero at L=0/1).
    """
    out = grid.copy()
    luma = _LUMA_R * out[..., 0] + _LUMA_G * out[..., 1] + _LUMA_B * out[..., 2]
    bell = np.clip(1.0 - np.abs(luma - 0.5) * 2.0, 0.0, 1.0)
    out[..., 0] = np.clip(out[..., 0] * (1.0 - 0.05 * bell), 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] - 0.04 * bell, 0.0, 1.0)
    out[..., 2] = out[..., 2]
    return out


def _bleach_bypass(grid: np.ndarray) -> np.ndarray:
    """Bleach bypass (silver-retention) demo: high contrast, desaturated.

    Lifts contrast (+18% S-curve on luma) and pulls chroma toward neutral by
    blending each channel 55% toward luma. Black/white endpoints preserved
    (the S-curve is anchored at 0 and 1; the desaturation weight is a bell
    that is 0 at luma=0 and luma=1).
    """
    out = grid.copy()
    luma = _LUMA_R * out[..., 0] + _LUMA_G * out[..., 1] + _LUMA_B * out[..., 2]
    bell = np.clip(1.0 - np.abs(luma - 0.5) * 2.0, 0.0, 1.0)
    contrast_luma = _s_curve(luma, strength=0.18)
    for ch in range(3):
        # Pull each channel 45% of the way toward the contrasty luma,
        # only in the midtones (bell weight).
        desat = out[..., ch] * (1.0 - 0.45 * bell) + contrast_luma * (0.45 * bell)
        out[..., ch] = np.clip(desat, 0.0, 1.0)
    return out


def _teal_orange(grid: np.ndarray) -> np.ndarray:
    """Teal-and-orange demo: complementary split-tone.

    Shadows drift toward teal (B up, R down), weighted by a shadow bell
    peaking near luma=0.2. Highlights drift toward orange (R up, B down),
    weighted by a highlight bell peaking near luma=0.8. Midtones and
    endpoints left alone.
    """
    out = grid.copy()
    luma = _LUMA_R * out[..., 0] + _LUMA_G * out[..., 1] + _LUMA_B * out[..., 2]
    # Multiply by luma / (1-luma) so the bells are exactly 0 at the endpoints
    # (pure black and pure white must round-trip — see test_luts_preserve_endpoints).
    shadow_w = luma * np.exp(-((luma - 0.2) / 0.15) ** 2)
    highlight_w = (1.0 - luma) * np.exp(-((luma - 0.8) / 0.15) ** 2)
    out[..., 0] = np.clip(out[..., 0] - 0.06 * shadow_w + 0.06 * highlight_w, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] - 0.02 * highlight_w, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + 0.06 * shadow_w - 0.06 * highlight_w, 0.0, 1.0)
    return out


def _cross_process(grid: np.ndarray) -> np.ndarray:
    """Cross-process demo (slide-film processed as C-41).

    Classic tells: blue/minor-green lift in shadows, heavy green push in
    midtones, magenta/red tint in highlights. Endpoints preserved by
    weighting with a bell that is 0 at luma=0 and luma=1.
    """
    out = grid.copy()
    luma = _LUMA_R * out[..., 0] + _LUMA_G * out[..., 1] + _LUMA_B * out[..., 2]
    bell = np.clip(1.0 - np.abs(luma - 0.5) * 2.0, 0.0, 1.0)
    shadow_w = luma * np.exp(-((luma - 0.15) / 0.12) ** 2)
    highlight_w = (1.0 - luma) * np.exp(-((luma - 0.85) / 0.12) ** 2)
    out[..., 0] = np.clip(out[..., 0] + 0.05 * shadow_w - 0.03 * highlight_w + 0.02 * bell, 0.0, 1.0)
    out[..., 1] = np.clip(out[..., 1] + 0.06 * bell + 0.02 * shadow_w, 0.0, 1.0)
    out[..., 2] = np.clip(out[..., 2] + 0.08 * shadow_w - 0.05 * highlight_w, 0.0, 1.0)
    return out


def _write_cube(
    out_path: Path,
    size: int,
    grid: np.ndarray,
    title: str,
    header_lines: Tuple[str, ...],
) -> None:
    lines: List[str] = ["# Generated by scripts/recipes/generate_demo_luts.py"]
    for line in header_lines:
        lines.append(f"# {line}")
    lines.append(f'TITLE "{title}"')
    lines.append(f"LUT_3D_SIZE {size}")
    lines.append("DOMAIN_MIN 0.0 0.0 0.0")
    lines.append("DOMAIN_MAX 1.0 1.0 1.0")
    for bi in range(size):
        for gi in range(size):
            for ri in range(size):
                r, g, b = (float(v) for v in grid[ri, gi, bi])
                lines.append(f"{r:.6f} {g:.6f} {b:.6f}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate(out_dir: Path) -> Tuple[Path, ...]:
    """Generate the four demo LUTs into ``out_dir``. Returns the file paths."""
    out_dir.mkdir(parents=True, exist_ok=True)

    n33 = 33
    grid33 = _build_input_grid(n33)
    p_id = out_dir / "identity_33.cube"
    _write_cube(
        p_id,
        n33,
        grid33,
        title="Demo Identity 33x33x33",
        header_lines=(
            "Retouch Engine -- demo LUT (safe to redistribute).",
            "Identity: each voxel output equals its (R, G, B) input.",
            "Use for pipeline round-trip tests.",
        ),
    )

    n17 = 17
    grid17 = _build_input_grid(n17)

    p_warm = out_dir / "warm_boost_17.cube"
    _write_cube(
        p_warm,
        n17,
        _warm_boost(grid17),
        title="Demo Warm Boost 17x17x17",
        header_lines=(
            "Retouch Engine -- demo LUT (safe to redistribute).",
            "Mild warm midtone shift: R * 1.10 and B * 0.95 in midtones,",
            "weighted by a triangular bell peaking at L=0.5.",
            "Demonstrates a per-channel 'look' without being a real film stock.",
        ),
    )

    p_cool = out_dir / "cool_shadows_17.cube"
    _write_cube(
        p_cool,
        n17,
        _cool_shadows(grid17),
        title="Demo Cool Shadows 17x17x17",
        header_lines=(
            "Retouch Engine -- demo LUT (safe to redistribute).",
            "Cool tint (R - 0.05, B + 0.05) in shadow regions,",
            "weighted by a luma bump centered at L=0.1.",
            "Demonstrates a luminance-dependent 'look' (e.g., split-toning).",
        ),
    )

    p_kodak = out_dir / "kodak_ish_17.cube"
    _write_cube(
        p_kodak,
        n17,
        _kodak_ish(grid17),
        title="Demo Kodak-ish 17x17x17",
        header_lines=(
            "Retouch Engine -- demo LUT (safe to redistribute).",
            "NOT an authoritative Kodak profile. An illustrative approximation:",
            "  R: gentle S-curve (linear contrast +10%)",
            "  G: midtone lift (+0.04 at L~0.5)",
            "  B: shadow crush (lower mids lifted via 4x(1-x)^2 weighting)",
            "For real Kodak/Fuji film stock emulation, install commercial LUTs",
            "(see luts/ACQUISITION.md).",
        ),
    )

    p_kodak_warm = out_dir / "kodak.cube"
    _write_cube(
        p_kodak_warm,
        n17,
        _kodak_warm(grid17),
        title="Demo Kodak Warm 17x17x17",
        header_lines=(
            "Retouch Engine -- demo LUT (safe to redistribute).",
            "Referenced by the 'film' preset. NOT an authoritative Kodak profile.",
            "Per-channel curves: R S-curve (+12%), G midtone lift (+0.05), B shadow crush (+0.05).",
            "For real Kodak/Fuji film stock emulation, install commercial LUTs.",
        ),
    )

    p_fuji_chrome = out_dir / "fuji.cube"
    _write_cube(
        p_fuji_chrome,
        n17,
        _fuji_chrome(grid17),
        title="Demo Fuji Chrome 17x17x17",
        header_lines=(
            "Retouch Engine -- demo LUT (safe to redistribute).",
            "Referenced by the 'film' preset (alt name). NOT an authoritative Fuji profile.",
            "Classic Chrome-inspired curves: R highlight crush (-5%), G cool midtones (-0.04), B linear.",
            "For real Fuji film stock emulation, install commercial LUTs.",
        ),
    )

    p_bleach = out_dir / "bleach_bypass_17.cube"
    _write_cube(
        p_bleach,
        n17,
        _bleach_bypass(grid17),
        title="Demo Bleach Bypass 17x17x17",
        header_lines=(
            "Retouch Engine -- demo LUT (safe to redistribute).",
            "Bleach-bypass / silver-retention look: high contrast + desaturated midtones.",
            "Luma S-curve (+18%), channels blended 45% toward luma in midtones only.",
            "Endpoints preserved; not a real film stock profile.",
        ),
    )

    p_teal_orange = out_dir / "teal_orange_17.cube"
    _write_cube(
        p_teal_orange,
        n17,
        _teal_orange(grid17),
        title="Demo Teal & Orange 17x17x17",
        header_lines=(
            "Retouch Engine -- demo LUT (safe to redistribute).",
            "Complementary split-tone: shadows -> teal (B+ R-), highlights -> orange (R+ B-).",
            "Gaussian bells peaked at L=0.2 and L=0.8; midtones and endpoints untouched.",
            "Demonstrates luminance-masked complementary grading.",
        ),
    )

    p_cross = out_dir / "cross_process_17.cube"
    _write_cube(
        p_cross,
        n17,
        _cross_process(grid17),
        title="Demo Cross Process 17x17x17",
        header_lines=(
            "Retouch Engine -- demo LUT (safe to redistribute).",
            "Cross-process look (slide film developed as C-41):",
            "  R: shadow lift +5%, highlight crush -3%, midtone +2%",
            "  G: midtone lift +6% (the dominant tell), shadow +2%",
            "  B: shadow lift +8%, highlight crush -5%",
            "Endpoints preserved by bell weighting.",
        ),
    )

    return (p_id, p_warm, p_cool, p_kodak, p_kodak_warm, p_fuji_chrome,
            p_bleach, p_teal_orange, p_cross)


def main() -> int:
    out = generate(luts_dir())
    print(f"Generated {len(out)} demo LUT(s) in {luts_dir()}:")
    for p in out:
        size_kb = p.stat().st_size / 1024.0
        print(f"  {p.name}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
