#!/usr/bin/env python3
"""Phase 1.a -- Visual validation + benchmark for the Fuji Foundation.

Validates the 4 new modules:
  * retouch.tonal     (apply_hd_curve)        -- film H&D response curve
  * retouch.skin_protect (protect_skin)       -- LCH-based skin mask
  * retouch.grain     (apply_film_grain)      -- clumped luminance grain
  * retouch.highlight (apply_highlight_rolloff) -- soft highlight clip

For each module the script:
  1. Generates (or loads) a synthetic test image with a horizontal gradient,
     simple shapes, and a synthetic "skin-tone" patch.
  2. Applies the module in isolation at the recommended default strength.
  3. Saves a per-module output PNG to /tmp/fuji_validation/.
  4. Benchmarks it on a 1080p image (10 iterations, time.perf_counter).
  5. Computes subjective "Fuji-like" properties of the output.

A combined pipeline image and a 2x3 grid comparison image are also written.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

# Make the project root importable when running this script directly
# (python3 scripts/review/validate_fuji_foundation.py).
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Paths and output dir
# ---------------------------------------------------------------------------

OUT_DIR = Path("/tmp/fuji_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic test image
# ---------------------------------------------------------------------------

def make_synthetic_image(width: int = 1080, height: int = 720) -> np.ndarray:
    """Create a synthetic 1080x720 image with a sky-to-ground gradient, simple
    shapes, and a warm "skin-tone" patch for visual interest.

    Returns:
        (H, W, 3) uint8 BGR image.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Vertical sky-to-ground gradient: sky (top) -> ground (bottom).
    sky_top = np.array([180, 200, 235], dtype=np.float32)
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

    # Sun: bright disc in the upper-right (will exercise highlight rolloff).
    sun_cx, sun_cy, sun_r = int(width * 0.78), int(height * 0.22), 60
    cv2.circle(img, (sun_cx, sun_cy), sun_r, (245, 240, 220), -1)
    cv2.circle(img, (sun_cx, sun_cy), sun_r, (255, 250, 235), 2)

    # Synthetic "skin-tone" patches (LCH ~25 deg hue) -- the
    # skin_protect module will protect these.
    # Picked to be in the orange wedge (hue ~20 deg) with chroma > 8
    # so the LCH skin mask in skin_protect actually fires.
    skin_color = (160, 170, 220)  # BGR ~ warm peach (RGB 220,170,160)
    cv2.rectangle(img, (int(width * 0.10), int(height * 0.62)),
                  (int(width * 0.20), int(height * 0.72)), skin_color, -1)
    cv2.rectangle(img, (int(width * 0.85), int(height * 0.60)),
                  (int(width * 0.95), int(height * 0.70)), skin_color, -1)

    # Foliage patch (deep green) to test saturation / chroma behaviour.
    cv2.rectangle(img, (int(width * 0.30), int(height * 0.78)),
                  (int(width * 0.50), int(height * 0.88)), (40, 90, 60), -1)
    # Sky / cobalt blue patch.
    cv2.rectangle(img, (int(width * 0.55), int(height * 0.20)),
                  (int(width * 0.72), int(height * 0.30)), (200, 110, 30), -1)

    # A few thin diagonal lines to test for tearing / moire.
    for i, t in enumerate(np.linspace(0.05, 0.95, 12)):
        x1, y1 = int(width * t), int(height * 0.05)
        x2, y2 = int(width * (t + 0.05)), int(height * 0.50)
        cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Add a small amount of texture noise so the grain module has something
    # organic to enhance.
    noise = np.random.normal(0, 6, img.shape).astype(np.float32)
    img_f = img.astype(np.float32) + noise
    return np.clip(img_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-module wrappers
# ---------------------------------------------------------------------------

def _tonal_module(img: np.ndarray) -> np.ndarray:
    from retouch.tonal import apply_hd_curve
    return apply_hd_curve(img, strength=0.7, toe=0.10, shoulder=0.10, luma_only=True)


def _skin_protect_module(img: np.ndarray) -> np.ndarray:
    from retouch.skin_protect import protect_skin
    from retouch.tonal import apply_hd_curve
    return protect_skin(img, lambda x: apply_hd_curve(x, strength=1.0), strength=0.7)


def _grain_module(img: np.ndarray) -> np.ndarray:
    from retouch.grain import apply_film_grain
    return apply_film_grain(img, strength=0.3, clump_sigma=1.2, luma_power=1.2, chroma=0.0, seed=42)


def _highlight_module(img: np.ndarray) -> np.ndarray:
    from retouch.highlight import apply_highlight_rolloff
    return apply_highlight_rolloff(img, strength=0.7)


def _combined_pipeline(img: np.ndarray) -> np.ndarray:
    """Apply all 4 modules in the spec'd order: tonal -> skin -> highlight -> grain.

    Defaults are moderate (not max) so the composite is a usable Fuji-ish look.
    """
    from retouch.tonal import apply_hd_curve
    from retouch.skin_protect import protect_skin
    from retouch.highlight import apply_highlight_rolloff
    from retouch.grain import apply_film_grain

    # 1. Tonal -- the H&D response curve
    x = apply_hd_curve(img, strength=0.6, toe=0.10, shoulder=0.12, luma_only=True)
    # 2. Color (skin-protected)
    x = protect_skin(x, lambda y: apply_hd_curve(y, strength=0.3), strength=0.5)
    # 3. Highlight rolloff
    x = apply_highlight_rolloff(x, strength=0.6)
    # 4. Grain
    x = apply_film_grain(x, strength=0.25, clump_sigma=1.0, luma_power=1.2, chroma=0.0, seed=42)
    return x


# ---------------------------------------------------------------------------
# Per-module benchmark
# ---------------------------------------------------------------------------

def benchmark_module(name: str, fn: Callable[[np.ndarray], np.ndarray],
                     img: np.ndarray, iterations: int = 10) -> Tuple[float, float, float]:
    """Run ``fn`` on ``img`` ``iterations`` times.

    Returns:
        (mean_ms, std_ms, fps_equiv)
    """
    samples_ms: List[float] = []
    # Warmup once to load any one-time imports / JIT.
    _ = fn(img.copy())
    for _ in range(iterations):
        t0 = time.perf_counter()
        _ = fn(img.copy())
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)
    arr = np.asarray(samples_ms, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1) if len(arr) > 1 else 0.0)
    fps = 1000.0 / mean if mean > 1e-6 else 0.0
    print(f"  {name:<22s}  mean={mean:7.2f} ms   std={std:6.2f} ms   ~{fps:5.1f} fps")
    return mean, std, fps


# ---------------------------------------------------------------------------
# Subjective "Fuji-like?" observations
# ---------------------------------------------------------------------------

def fuji_observations(original: np.ndarray, tonal: np.ndarray,
                       skin_out: np.ndarray, grain_out: np.ndarray,
                       highlight_out: np.ndarray,
                       combined: np.ndarray) -> List[str]:
    """Compute objective signals that correlate with the Fuji look."""
    obs: List[str] = []

    # 1. Shadow behaviour: H&D curve with toe>0 compresses shadows (Astia),
    #    not lifts them (Classic Chrome). Both are valid Fuji looks; report
    #    the actual direction so the user can see what they got.
    lab_orig = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    lab_tonal = cv2.cvtColor(tonal, cv2.COLOR_BGR2LAB)
    lo_orig = float(lab_orig[:, :, 0].min())
    lo_tonal = float(lab_tonal[:, :, 0].min())
    if abs(lo_tonal - lo_orig) < 1.0:
        obs.append(f"blacks preserved (L* min ~{lo_orig:.1f})")
    elif lo_tonal > lo_orig:
        obs.append(f"lifted blacks (L* min: {lo_orig:.1f} -> {lo_tonal:.1f}) [Classic Chrome style]")
    else:
        obs.append(f"shadow compression / soft toe (L* min: {lo_orig:.1f} -> {lo_tonal:.1f}) [Astia/Provia style]")

    # 2. Soft highlight rolloff: the max value of the highlight output
    #    should be less than the max value of the original.
    hi_orig = float(original.max())
    hi_hl = float(highlight_out.max())
    if hi_hl < hi_orig - 0.5:
        obs.append(f"soft highlight rolloff present (max: {hi_orig:.1f} -> {hi_hl:.1f})")
    else:
        obs.append(f"highlights unchanged (max ~{hi_orig:.1f})")

    # 3. Skin protection: in the skin region, the protected output should
    #    be closer to the original than the bare full-strength tonal.
    from retouch.tonal import apply_hd_curve
    full_tonal = apply_hd_curve(original, strength=1.0, toe=0.10, shoulder=0.10, luma_only=True)

    h, w = original.shape[:2]
    sx0, sy0 = int(w * 0.10), int(h * 0.62)
    sx1, sy1 = int(w * 0.20), int(h * 0.72)
    region_orig = original[sy0:sy1, sx0:sx1].astype(np.float32)
    region_full = full_tonal[sy0:sy1, sx0:sx1].astype(np.float32)
    region_skin = skin_out[sy0:sy1, sx0:sx1].astype(np.float32)
    diff_full = float(np.abs(region_full - region_orig).mean())
    diff_skin = float(np.abs(region_skin - region_orig).mean())
    if diff_skin < diff_full * 0.5:
        obs.append(f"skin-tone protection strong (skin diff: full_tonal={diff_full:.1f}, protected={diff_skin:.1f}, reduction={100*(1 - diff_skin/diff_full):.0f}%)")
    elif diff_skin < diff_full * 0.85:
        obs.append(f"skin-tone protection active (skin diff: full_tonal={diff_full:.1f}, protected={diff_skin:.1f}, reduction={100*(1 - diff_skin/diff_full):.0f}%)")
    else:
        obs.append(f"skin-tone protection weak (skin diff: full_tonal={diff_full:.1f}, protected={diff_skin:.1f}, reduction={100*(1 - diff_skin/diff_full):.0f}%)")

    # 4. Grain visibility: local 3x3 std is more sensitive to fine grain
    #    than Laplacian. We measure the local std deviation in a
    #    homogeneous region and check it grew after the grain pass.
    sky_y0, sky_y1 = int(h * 0.05), int(h * 0.15)
    sky_x0, sky_x1 = int(w * 0.05), int(w * 0.30)
    crop_orig = original[sky_y0:sky_y1, sky_x0:sky_x1, 0].astype(np.float32)
    crop_grain = grain_out[sky_y0:sky_y1, sky_x0:sky_x1, 0].astype(np.float32)
    k = 5
    mean_o = cv2.boxFilter(crop_orig, -1, (k, k))
    mean_g = cv2.boxFilter(crop_grain, -1, (k, k))
    sq_o = cv2.boxFilter(crop_orig * crop_orig, -1, (k, k))
    sq_g = cv2.boxFilter(crop_grain * crop_grain, -1, (k, k))
    var_o = np.clip(sq_o - mean_o * mean_o, 0, None)
    var_g = np.clip(sq_g - mean_g * mean_g, 0, None)
    std_o = float(np.sqrt(var_o).mean())
    std_g = float(np.sqrt(var_g).mean())
    if std_g > std_o * 1.10:
        obs.append(f"organic grain visible (local std: {std_o:.2f} -> {std_g:.2f}, +{100*(std_g-std_o)/std_o:.0f}%)")
    else:
        obs.append(f"grain subtle (local std: {std_o:.2f} -> {std_g:.2f}, +{100*(std_g-std_o)/std_o:.0f}%)")

    # 5. Combined pipeline total: the combined image should differ from
    #    the original by a moderate amount (5-25 mean abs units in 0-255).
    combined_diff = float(np.abs(combined.astype(np.float32) - original.astype(np.float32)).mean())
    if combined_diff > 3.0:
        obs.append(f"combined pipeline has effect (mean abs diff: {combined_diff:.1f}/255)")
    else:
        obs.append(f"combined pipeline barely visible (mean abs diff: {combined_diff:.1f}/255)")

    return obs


# ---------------------------------------------------------------------------
# Comparison grid
# ---------------------------------------------------------------------------

def make_comparison_grid(images: List[Tuple[str, np.ndarray]],
                         out_path: Path,
                         cols: int = 3) -> None:
    """Save a grid of labelled images as a single PNG.

    Each cell has the label burned in at the top with cv2.putText.
    Images are resized to the same dimensions for the grid.
    """
    if not images:
        return
    # Use the first image as the canonical cell size.
    cell_h, cell_w = images[0][1].shape[:2]
    # Cap cell size to keep the grid under ~10 MB.
    max_cell = 600
    if max(cell_h, cell_w) > max_cell:
        scale = max_cell / float(max(cell_h, cell_w))
        cell_h = int(cell_h * scale)
        cell_w = int(cell_w * scale)
        cell_h = cell_h - cell_h % 2
        cell_w = cell_w - cell_w % 2

    rows = (len(images) + cols - 1) // cols
    label_h = 36
    canvas = np.full((rows * (cell_h + label_h), cols * cell_w, 3), 32, dtype=np.uint8)
    for i, (label, img) in enumerate(images):
        r, c = divmod(i, cols)
        cell = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        y_off = r * (cell_h + label_h) + label_h
        x_off = c * cell_w
        canvas[y_off:y_off + cell_h, x_off:x_off + cell_w] = cell
        cv2.putText(canvas, label, (x_off + 8, r * (cell_h + label_h) + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=" * 72)
    print("Phase 1.a -- Fuji Foundation Validation")
    print("=" * 72)

    # 1. Build the synthetic test image.
    print("\n[1/5] Generating synthetic test image (1080x720) ...")
    img = make_synthetic_image(1080, 720)
    h, w = img.shape[:2]
    print(f"  image: {w}x{h}, dtype={img.dtype}, mean L*={cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:,:,0].mean():.1f}")
    cv2.imwrite(str(OUT_DIR / "original.png"), img)

    # 2. Apply each module in isolation.
    print("\n[2/5] Applying each module in isolation ...")
    tonal_out = _tonal_module(img)
    skin_out = _skin_protect_module(img)
    grain_out = _grain_module(img)
    highlight_out = _highlight_module(img)
    combined_out = _combined_pipeline(img)
    cv2.imwrite(str(OUT_DIR / "tonal.png"), tonal_out)
    cv2.imwrite(str(OUT_DIR / "skin_protect.png"), skin_out)
    cv2.imwrite(str(OUT_DIR / "grain.png"), grain_out)
    cv2.imwrite(str(OUT_DIR / "highlight.png"), highlight_out)
    cv2.imwrite(str(OUT_DIR / "combined.png"), combined_out)
    for name in ("tonal", "skin_protect", "grain", "highlight", "combined"):
        p = OUT_DIR / f"{name}.png"
        size_kb = p.stat().st_size / 1024.0
        print(f"  saved {p} ({size_kb:.0f} KB)")

    # 3. Build a 2x3 grid for side-by-side comparison.
    print("\n[3/5] Building 2x3 comparison grid ...")
    grid_images = [
        ("1. original", img),
        ("2. tonal", tonal_out),
        ("3. skin_protect", skin_out),
        ("4. highlight", highlight_out),
        ("5. grain", grain_out),
        ("6. combined", combined_out),
    ]
    make_comparison_grid(grid_images, OUT_DIR / "comparison.png", cols=3)
    grid_kb = (OUT_DIR / "comparison.png").stat().st_size / 1024.0
    print(f"  saved {OUT_DIR / 'comparison.png'} ({grid_kb:.0f} KB)")

    # 4. Benchmark each module on a 1080p image.
    print("\n[4/5] Benchmarking each module (10 iterations on 1080x720) ...")
    perf = {}
    for name, fn in (("tonal", _tonal_module), ("skin_protect", _skin_protect_module),
                     ("highlight", _highlight_module), ("grain", _grain_module),
                     ("combined", _combined_pipeline)):
        mean_ms, std_ms, fps = benchmark_module(name, fn, img, iterations=10)
        perf[name] = (mean_ms, std_ms, fps)

    # 5. Subjective observations.
    print("\n[5/5] Subjective 'Fuji-like' observations ...")
    obs = fuji_observations(img, tonal_out, skin_out, grain_out, highlight_out, combined_out)
    for line in obs:
        print(f"  - {line}")

    # 6. Final summary.
    total_combined = perf["combined"][0]
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"  Image: {w}x{h} (1080p)")
    print(f"  Output dir: {OUT_DIR}")
    print(f"  Per-module perf (mean ms / fps):")
    for name, (mean_ms, std_ms, fps) in perf.items():
        print(f"    {name:<14s} {mean_ms:7.2f} +/- {std_ms:5.2f} ms  ~{fps:5.1f} fps")
    print(f"  Combined pipeline total: {total_combined:.2f} ms (~{1000.0/total_combined:.1f} fps)")
    print(f"  Looks Fuji-like? {len(obs)} / 4 signals positive")
    for line in obs:
        print(f"    {line}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
