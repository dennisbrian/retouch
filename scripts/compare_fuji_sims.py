#!/usr/bin/env python3
"""Phase 1.c -- Side-by-Side Comparison of the 3 Official Fuji Film Simulations.

Renders a synthetic 1280x720 test image (sky-to-ground gradient + skin-tone
patches + saturated RGB patches) and applies each of the three official Fuji
film simulations from Phase 1.c:

    * Classic Chrome -- flagship "Fuji look"
    * Astia         -- portrait specialist with strong skin protection
    * Provia        -- neutral, accurate reproduction

Outputs (all under ``/tmp/sim_comparison/``):
    * ``source.png``           -- the synthetic test image
    * ``classic_chrome.png``   -- Classic Chrome sim output
    * ``astia.png``            -- Astia sim output
    * ``provia.png``           -- Provia sim output
    * ``comparison.png``       -- 2x2 grid with title bar and cell labels

A summary table of file size, mean color shift, and per-channel mean shift
(BGR) is printed to stdout. The script is safe to run on a CPU-only machine
(no GPU/CUDA assumed) and does not require a face in the input image.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the project root importable when running this script directly
# (python3 scripts/compare_fuji_sims.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from retouch.recipes import FUJI_SIM_NAMES, list_fuji_sims


# ---------------------------------------------------------------------------
# Paths and output dir
# ---------------------------------------------------------------------------

OUT_DIR = Path("/tmp/sim_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic test image
# ---------------------------------------------------------------------------

def make_synthetic_image(width: int = 1280, height: int = 720) -> np.ndarray:
    """Create a 1280x720 HD test image for visual sim comparison.

    Layout (top to bottom):
        1. Sky-to-horizon-to-ground vertical gradient.
        2. Soft sun disc in the upper right (highlights).
        3. Two peach / brown "skin-tone" patches (skin protection validation).
        4. Saturated red, green, blue colour swatches (colour grade validation).
        5. Foliage green patch and a cobalt blue sky patch.
        6. Subtle Gaussian noise so the grain module has organic texture.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # 1. Sky -> horizon -> ground gradient.
    sky_top = np.array([180, 200, 235], dtype=np.float32)         # BGR
    sky_horizon = np.array([240, 220, 200], dtype=np.float32)
    ground_bottom = np.array([60, 50, 40], dtype=np.float32)
    horizon_y = int(height * 0.55)
    for y in range(height):
        if y < horizon_y:
            t = y / float(horizon_y)
            row = sky_top * (1.0 - t) + sky_horizon * t
        else:
            t = (y - horizon_y) / float(height - horizon_y)
            row = sky_horizon * (1.0 - t) + ground_bottom * t
        img[y, :] = row.astype(np.uint8)

    # 2. Sun disc (top right) -- exercises highlight rolloff.
    sun_cx, sun_cy, sun_r = int(width * 0.78), int(height * 0.22), 60
    cv2.circle(img, (sun_cx, sun_cy), sun_r, (245, 240, 220), -1)
    cv2.circle(img, (sun_cx, sun_cy), sun_r, (255, 250, 235), 2)

    # 3. Skin-tone patches (peach + brown) -- these will be protected by
    #    the Astia sim's strong skin_protect weight (0.85).
    peach = (160, 170, 220)   # BGR -- warm peach
    brown = (110, 130, 180)   # BGR -- brown skin
    cv2.rectangle(img, (int(width * 0.10), int(height * 0.62)),
                  (int(width * 0.20), int(height * 0.72)), peach, -1)
    cv2.rectangle(img, (int(width * 0.30), int(height * 0.62)),
                  (int(width * 0.40), int(height * 0.72)), brown, -1)

    # 4. Saturated RGB swatches (for colour grade validation).
    cv2.rectangle(img, (int(width * 0.55), int(height * 0.62)),
                  (int(width * 0.62), int(height * 0.72)), (50, 50, 220), -1)   # red (BGR)
    cv2.rectangle(img, (int(width * 0.65), int(height * 0.62)),
                  (int(width * 0.72), int(height * 0.72)), (60, 200, 60), -1)  # green
    cv2.rectangle(img, (int(width * 0.75), int(height * 0.62)),
                  (int(width * 0.82), int(height * 0.72)), (220, 130, 30), -1)  # blue

    # 5. Foliage + cobalt patches (extra colour sanity).
    cv2.rectangle(img, (int(width * 0.30), int(height * 0.78)),
                  (int(width * 0.50), int(height * 0.88)), (40, 90, 60), -1)
    cv2.rectangle(img, (int(width * 0.55), int(height * 0.20)),
                  (int(width * 0.72), int(height * 0.30)), (200, 110, 30), -1)

    # 6. Subtle noise -- gives the grain module something organic.
    noise = np.random.normal(0, 5, img.shape).astype(np.float32)
    img_f = img.astype(np.float32) + noise
    return np.clip(img_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Apply a single sim
# ---------------------------------------------------------------------------

def apply_sim(engine, img_bgr: np.ndarray, sim_name: str) -> np.ndarray:
    """Apply a single Fuji sim via ``engine.process(recipe=sim_name)``.

    Returns the output as a plain ``np.ndarray`` (BGR uint8) for easy
    post-processing. Handles the engine's ProcessingResult wrapper by
    extracting ``.image`` if present, otherwise returning the array as-is.
    """
    result = engine.process(img_bgr, recipe=sim_name)
    # ProcessingResult is a numpy ndarray subclass with a richer .image
    # attribute; prefer the explicit .image when present.
    if hasattr(result, "image") and result.image is not None:
        return np.asarray(result.image)
    return np.asarray(result)


# ---------------------------------------------------------------------------
# 2x2 comparison grid
# ---------------------------------------------------------------------------

def make_comparison_grid(
    images: List[Tuple[str, np.ndarray]],
    out_path: Path,
    title: str = "Fuji Film Simulation Comparison",
) -> None:
    """Render a 2x2 grid with a title bar and per-cell labels.

    Each cell is resized to ``cell_w x cell_h`` for a uniform grid.
    The title is burned in at the top, cell labels are burned in just
    above each image cell. All images must already be the same shape
    (the grid layout does not handle mismatched sizes gracefully).
    """
    if not images:
        raise ValueError("images list is empty")

    # Canonical cell size = first image; cap to keep file small.
    cell_h, cell_w = images[0][1].shape[:2]
    max_cell = 640
    if max(cell_h, cell_w) > max_cell:
        s = max_cell / float(max(cell_h, cell_w))
        cell_h = int(cell_h * s)
        cell_w = int(cell_w * s)
        cell_h -= cell_h % 2
        cell_w -= cell_w % 2

    cols = 2
    rows = (len(images) + cols - 1) // cols
    label_h = 32          # cell-label strip
    title_h = 48          # top title bar
    pad = 8               # small gap between cells

    canvas_h = title_h + rows * (cell_h + label_h + pad)
    canvas_w = cols * cell_w + (cols + 1) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    # Title bar
    cv2.putText(
        canvas, title,
        (pad + 8, title_h - 14),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
    )

    for i, (label, img) in enumerate(images):
        r, c = divmod(i, cols)
        cell_resized = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        y0 = title_h + r * (cell_h + label_h + pad) + label_h
        x0 = pad + c * (cell_w + pad)
        canvas[y0:y0 + cell_h, x0:x0 + cell_w] = cell_resized
        cv2.putText(
            canvas, label,
            (x0 + 8, y0 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

    cv2.imwrite(str(out_path), canvas)


# ---------------------------------------------------------------------------
# Per-sim statistics
# ---------------------------------------------------------------------------

def compute_color_stats(source: np.ndarray, output: np.ndarray) -> Dict[str, float]:
    """Compute objective color-shift stats between a source and a sim output.

    Both arrays must be uint8 BGR with the same shape. Returns:
        * mean_abs_diff: mean of |out - src| across all pixels / channels.
        * b_shift, g_shift, r_shift: signed mean (output - source) per channel
          in 8-bit units. Negative = channel darkened, positive = brightened.
    """
    if source.shape != output.shape:
        raise ValueError(
            f"shape mismatch: source={source.shape}, output={output.shape}"
        )
    diff = output.astype(np.float32) - source.astype(np.float32)
    mean_abs = float(np.abs(diff).mean())
    b_shift = float(diff[:, :, 0].mean())
    g_shift = float(diff[:, :, 1].mean())
    r_shift = float(diff[:, :, 2].mean())
    return {
        "mean_abs_diff": mean_abs,
        "b_shift": b_shift,
        "g_shift": g_shift,
        "r_shift": r_shift,
    }


def print_stats_table(rows: List[Tuple[str, Path, Dict[str, float]]]) -> None:
    """Print a formatted table of per-sim statistics to stdout."""
    header = (
        f"{'sim':<18s} {'file':<32s} {'size':>10s} "
        f"{'mean_abs':>10s} {'B shift':>9s} {'G shift':>9s} {'R shift':>9s}"
    )
    print(header)
    print("-" * len(header))
    for sim_name, path, stats in rows:
        size_kb = path.stat().st_size / 1024.0
        print(
            f"{sim_name:<18s} {path.name:<32s} {size_kb:>8.1f}KB "
            f"{stats['mean_abs_diff']:>10.2f} "
            f"{stats['b_shift']:>+9.2f} "
            f"{stats['g_shift']:>+9.2f} "
            f"{stats['r_shift']:>+9.2f}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 80)
    print("Phase 1.c -- Fuji Film Simulation Side-by-Side Comparison")
    print("=" * 80)

    # 0. Verify the 3 sims are actually registered in RECIPES.
    available = list_fuji_sims()
    print(f"\nFuji sims registered: {available}")
    missing = [s for s in FUJI_SIM_NAMES if s not in available]
    if missing:
        print(f"  WARNING: sims not in RECIPES (engine will fall back to 'natural'): {missing}")

    # 1. Build the synthetic test image.
    print("\n[1/4] Generating synthetic 1280x720 test image ...")
    img = make_synthetic_image(1280, 720)
    h, w = img.shape[:2]
    source_path = OUT_DIR / "source.png"
    cv2.imwrite(str(source_path), img)
    print(f"  saved {source_path} ({source_path.stat().st_size / 1024.0:.0f} KB)")

    # 2. Apply each sim via the full engine.
    print("\n[2/4] Applying each sim via RetouchEngine.process() ...")
    from retouch.engine import RetouchEngine

    timings: Dict[str, float] = {}
    sim_outputs: Dict[str, np.ndarray] = {}
    with RetouchEngine() as engine:
        for sim_name in FUJI_SIM_NAMES:
            t0 = time.perf_counter()
            out = apply_sim(engine, img, sim_name)
            dt = time.perf_counter() - t0
            timings[sim_name] = dt
            sim_outputs[sim_name] = out
            out_path = OUT_DIR / f"{sim_name}.png"
            cv2.imwrite(str(out_path), out)
            size_kb = out_path.stat().st_size / 1024.0
            print(f"  {sim_name:<18s}  {dt:6.2f} s   saved {out_path.name} ({size_kb:.0f} KB)")

    # 3. Build the 2x2 comparison grid.
    print("\n[3/4] Building 2x2 comparison grid ...")
    grid_images: List[Tuple[str, np.ndarray]] = [
        ("1. Source", img),
        ("2. Classic Chrome", sim_outputs.get("classic_chrome", img)),
        ("3. Astia", sim_outputs.get("astia", img)),
        ("4. Provia", sim_outputs.get("provia", img)),
    ]
    grid_path = OUT_DIR / "comparison.png"
    make_comparison_grid(grid_images, grid_path)
    print(f"  saved {grid_path} ({grid_path.stat().st_size / 1024.0:.0f} KB)")

    # 4. Per-sim statistics table.
    print("\n[4/4] Per-sim color-shift statistics (vs source) ...")
    rows: List[Tuple[str, Path, Dict[str, float]]] = []
    for sim_name in FUJI_SIM_NAMES:
        out_path = OUT_DIR / f"{sim_name}.png"
        if out_path.exists() and sim_name in sim_outputs:
            stats = compute_color_stats(img, sim_outputs[sim_name])
        else:
            stats = {"mean_abs_diff": 0.0, "b_shift": 0.0, "g_shift": 0.0, "r_shift": 0.0}
        rows.append((sim_name, out_path, stats))
    print_stats_table(rows)

    # Final summary.
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"  Image:        {w}x{h}")
    print(f"  Output dir:   {OUT_DIR}")
    print(f"  Sim timings:")
    for sim_name, dt in timings.items():
        print(f"    {sim_name:<18s} {dt:6.2f} s")
    print(f"  Comparison:   {grid_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
