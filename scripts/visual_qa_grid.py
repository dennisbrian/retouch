"""Visual-QA comparison grid generator for the "everything auto" backlog features.

Reads the Retouch engine only (never edits it) and writes ON-vs-baseline
comparison images + montages for a set of new features, on BOTH the JPEG
reference and the RAF source.

Outputs (per source):
    test_output/visual_qa_grid/<source>/<feature>_on.jpg
    test_output/visual_qa_grid/<source>/<feature>_montage.jpg
    test_output/visual_qa_grid/<source>/combo_on.jpg
    test_output/visual_qa_grid/<source>/combo_montage.jpg

Run:
    python3 scripts/visual_qa_grid.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

# Engine imports (READ ONLY usage).
from retouch.io import imread_exif, resize_for_processing
from retouch.engine import RetouchEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("visual_qa_grid")

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
JPEG_PATH = Path("test_output/masterwork_v1/DSCF8007.jpg")
RAF_PATH = Path("/Users/dennis/Pictures/2025/2025-08-09/_DSF1853.RAF")

# Downscale longest side for speed.
MAX_DIM = 1280

OUT_ROOT = Path("test_output/visual_qa_grid")

# Amplified-diff gain to reveal subtle background/fabric changes.
DIFF_GAIN = 4.0


# ---------------------------------------------------------------------------
# Feature definitions: name -> kwargs applied ON TOP of the baseline.
# ---------------------------------------------------------------------------
FEATURES: List[Tuple[str, Dict[str, float]]] = [
    ("backdrop_cleanup", {"backdrop_cleanup": 60.0}),
    ("fabric_wrinkle_smooth", {"fabric_wrinkle_smooth": 60.0}),
    ("eye_sclera_vessel_remove", {"eye_sclera_vessel_remove": 60.0, "eye_enhance": 20.0}),
    ("wrinkle_soften_forehead", {"wrinkle_soften_forehead": 60.0}),
    ("wrinkle_soften_nasolabial", {"wrinkle_soften_nasolabial": 60.0}),
    ("wrinkle_soften_neck", {"wrinkle_soften_neck": 60.0}),
    ("reshape_jaw", {"reshape_jaw_width_l": 40.0, "reshape_jaw_width_r": -20.0}),
    ("reshape_neck", {"reshape_neck_width": 30.0, "reshape_neck_length": 20.0}),
    ("auto_body_reshape", {"auto_body_reshape": 60.0}),
]

# Sensible "everything auto" subset.
COMBO_KWARGS: Dict[str, float] = {
    "backdrop_cleanup": 60.0,
    "fabric_wrinkle_smooth": 60.0,
    "eye_sclera_vessel_remove": 60.0,
    "eye_enhance": 20.0,
    "wrinkle_soften_forehead": 60.0,
    "wrinkle_soften_nasolabial": 60.0,
    "wrinkle_soften_neck": 60.0,
    "reshape_jaw_width_l": 40.0,
    "reshape_jaw_width_r": -20.0,
    "reshape_neck_width": 30.0,
    "reshape_neck_length": 20.0,
    "auto_body_reshape": 60.0,
}


def load_source(path: Path, source_name: str) -> np.ndarray:
    """Load a source as uint8 BGR (handles JPEG and RAW via the engine's
    own loader in retouch.io.imread_exif) and downscale to MAX_DIM."""
    # PREFER the engine's own loader: imread_exif decodes RAW via rawpy with
    # camera WB + sRGB, and applies EXIF orientation for JPEG. Returns uint8 BGR.
    img = imread_exif(path)
    img, _ = resize_for_processing(img, MAX_DIM)
    logger.info("%s: loaded %s -> %s", source_name, path.name, img.shape)
    return img


def amplified_diff(baseline: np.ndarray, on: np.ndarray) -> np.ndarray:
    """clip(|on-baseline| * GAIN, 0, 255) uint8, revealed-subsidy diff."""
    diff = np.abs(on.astype(np.float32) - baseline.astype(np.float32)) * DIFF_GAIN
    return np.clip(diff, 0, 255).astype(np.uint8)


def add_title(strip: np.ndarray, text: str) -> np.ndarray:
    """Prepend a title bar above the image strip."""
    h, w = strip.shape[:2]
    bar_h = 36
    bar = np.full((bar_h, w, 3), 24, dtype=np.uint8)
    cv2.putText(
        bar, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA
    )
    return np.vstack([bar, strip])


def make_montage(baseline: np.ndarray, on: np.ndarray, title: str) -> np.ndarray:
    """Horizontal stack [baseline | on | amplified-abs-diff] with a title."""
    diff = amplified_diff(baseline, on)
    sep = np.full((on.shape[0], 4, 3), 180, dtype=np.uint8)
    stacked = np.hstack([baseline, sep, on, sep, diff])
    return add_title(stacked, title)


def process_safe(engine: RetouchEngine, img: np.ndarray, **kwargs: float) -> np.ndarray:
    """Run engine.process; returns uint8 BGR. Raises on failure (caller catches)."""
    result = engine.process(img, **kwargs)
    return np.asarray(result)


def mean_abs_delta(baseline: np.ndarray, on: np.ndarray) -> float:
    return float(np.mean(np.abs(on.astype(np.float32) - baseline.astype(np.float32))))


def run_source(engine: RetouchEngine, img: np.ndarray, source_name: str) -> Dict[str, float]:
    out_dir = OUT_ROOT / source_name
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, float] = {}

    # Baseline (no new features).
    baseline = process_safe(engine, img)
    cv2.imwrite(str(out_dir / "baseline.jpg"), baseline)

    # Single-feature ON runs.
    for feat_name, kwargs in FEATURES:
        try:
            on = process_safe(engine, img, **kwargs)
        except Exception as exc:  # never abort the whole run
            logger.error("FEATURE %s FAILED: %s", feat_name, exc)
            print(f"[ERROR] {source_name}/{feat_name}: {exc!r}", file=sys.stderr)
            continue
        cv2.imwrite(str(out_dir / f"{feat_name}_on.jpg"), on)
        montage = make_montage(baseline, on, f"{feat_name} ON")
        cv2.imwrite(str(out_dir / f"{feat_name}_montage.jpg"), montage)
        delta = mean_abs_delta(baseline, on)
        summary[feat_name] = delta
        logger.info("%s/%s: mean|Δ| = %.3f", source_name, feat_name, delta)

    # Combo "everything auto".
    try:
        combo = process_safe(engine, img, **COMBO_KWARGS)
        cv2.imwrite(str(out_dir / "combo_on.jpg"), combo)
        combo_montage = make_montage(baseline, combo, "COMBO everything auto")
        cv2.imwrite(str(out_dir / "combo_montage.jpg"), combo_montage)
        combo_delta = mean_abs_delta(baseline, combo)
        summary["combo"] = combo_delta
        logger.info("%s/combo: mean|Δ| = %.3f", source_name, combo_delta)
    except Exception as exc:
        logger.error("COMBO FAILED: %s", exc)
        print(f"[ERROR] {source_name}/combo: {exc!r}", file=sys.stderr)

    return summary


def main() -> int:
    sources = [
        ("jpeg", JPEG_PATH),
        ("raf", RAF_PATH),
    ]

    engine = RetouchEngine()
    all_summaries: Dict[str, Dict[str, float]] = {}

    for source_name, path in sources:
        if not path.exists():
            logger.warning("Source missing, skipping: %s", path)
            print(f"[SKIP] {source_name}: {path} not found", file=sys.stderr)
            continue
        img = load_source(path, source_name)
        summary = run_source(engine, img, source_name)
        all_summaries[source_name] = summary

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VISUAL QA GRID — MANIFEST")
    print("=" * 70)
    for source_name, summary in all_summaries.items():
        out_dir = OUT_ROOT / source_name
        print(f"\n[{source_name}] outputs in {out_dir}/")
        files = sorted(p.name for p in out_dir.glob("*.jpg"))
        for f in files:
            print(f"   {f}")
        print("  per-feature whole-frame mean|Δ|:")
        for feat, delta in summary.items():
            print(f"     {feat:30s} {delta:8.3f}")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
