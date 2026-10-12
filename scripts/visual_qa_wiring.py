"""Render the previously unreachable stages to visual-QA comparisons.

Each case compares a stage-off baseline against the same recipe with the
stage enabled. This avoids attributing ordinary recipe styling to a newly
wired stage. The runner emits individual ON outputs, baseline outputs,
amplified-diff montages, and one contact sheet for rapid human review.

Example:
    python3 scripts/visual_qa_wiring.py \
        --input "/path/to/photo-a.jpg" --input "/path/to/photo-b.jpg"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retouch.engine import RetouchEngine
from retouch.io import imread_exif, resize_for_processing
from retouch.regions import radial_mask


DEFAULT_INPUTS = [ROOT / "test_output/masterwork_v1/DSCF8007.jpg"]
DEFAULT_OUT = ROOT / "test_output/visual_qa_wiring"
DEFAULT_MAX_DIM = 1600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", action="append", dest="inputs", type=Path,
        help="Input image or RAW file. Repeat for multiple photos.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--max-dim", type=int, default=DEFAULT_MAX_DIM,
        help=f"Longest processing edge for QA only (default: {DEFAULT_MAX_DIM}).",
    )
    parser.add_argument(
        "--case", action="append", dest="case_names",
        help="Only render this QA case. Repeat for multiple cases.",
    )
    return parser.parse_args()


def load(path: Path, max_dim: int) -> np.ndarray:
    """Load through the engine path and downscale only for quick QA."""
    image = imread_exif(path)
    image, _ = resize_for_processing(image, max_dim)
    return image


def montage(base: np.ndarray, on: np.ndarray, title: str) -> np.ndarray:
    """Return labelled [stage off | stage on | amplified difference]."""
    diff = np.clip(
        np.abs(on.astype(np.float32) - base.astype(np.float32)) * 4.0,
        0,
        255,
    ).astype(np.uint8)
    divider = np.full((base.shape[0], 6, 3), 180, dtype=np.uint8)
    strip = np.hstack([base, divider, on, divider, diff])
    header = np.full((42, strip.shape[1], 3), 24, dtype=np.uint8)
    cv2.putText(header, title, (12, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (235, 235, 235), 2, cv2.LINE_AA)
    labels = ("STAGE OFF", "STAGE ON", "AMPLIFIED DIFF x4")
    panel_w = base.shape[1] + 6
    for index, label in enumerate(labels):
        cv2.putText(strip, label, (10 + index * panel_w, base.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 2, cv2.LINE_AA)
    return np.vstack([header, strip])


def _local_adjustment(img: np.ndarray) -> List[Dict[str, Any]]:
    """A moderate, feathered face/upper-body dodge brush for F3 QA."""
    h, w = img.shape[:2]
    return [{
        "mask": radial_mask(h, w, center_x=0.5, center_y=0.38,
                            radius=0.16, feather=0.55),
        "op": "dodge",
        "strength": 0.25,
    }]


def cases(img: np.ndarray) -> Iterable[Tuple[str, Dict[str, Any], Dict[str, Any]]]:
    """Yield (name, baseline kwargs, stage-on kwargs) for the four fixed stages."""
    yield (
        "background_anime_crystal_void",
        {
            "recipe": "anime_crystal_void",
            "background_blur": 0.0,
            "background_desaturation": 0.0,
            "light_wrap": 0.0,
            "blue_shadow_grade": 0.0,
            "cyan_midtone_grade": 0.0,
            "subject_sharpen": 0.0,
            "matte_black": 0.0,
        },
        {"recipe": "anime_crystal_void"},
    )
    yield (
        "cosplay_moat",
        {
            "recipe": "cosplay_wiring_demo_v1",
            "cosplay_wig_lace_blend": 0.0,
            "cosplay_stockings_smooth": 0.0,
            "cosplay_consistency_strength": 0.0,
        },
        {"recipe": "cosplay_wiring_demo_v1"},
    )
    yield (
        "local_adjustment_dodge",
        {"recipe": "natural_polish_v1"},
        {
            "recipe": "natural_polish_v1",
            "local_adjustments": _local_adjustment(img),
        },
    )
    yield (
        "body_reshape",
        {
            "recipe": "body_reshape_demo_v1",
            "body_reshape_arm_length": 50.0,
            "body_reshape_leg_length": 50.0,
            "body_reshape_torso_width": 50.0,
            "body_reshape_shoulder_width": 50.0,
            "body_reshape_hip_width": 50.0,
        },
        {"recipe": "body_reshape_demo_v1"},
    )


def contact_sheet(items: List[Tuple[str, np.ndarray]], destination: Path) -> None:
    """Write a compact review sheet without hiding the full montages."""
    if not items:
        return
    thumb_w = 560
    thumbs: List[np.ndarray] = []
    for title, image in items:
        scale = thumb_w / image.shape[1]
        thumb = cv2.resize(image, (thumb_w, max(1, int(image.shape[0] * scale))),
                           interpolation=cv2.INTER_AREA)
        label = np.full((32, thumb_w, 3), 24, dtype=np.uint8)
        cv2.putText(label, title, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (235, 235, 235), 1, cv2.LINE_AA)
        thumbs.append(np.vstack([label, thumb]))

    cell_h = max(image.shape[0] for image in thumbs)
    padded = [cv2.copyMakeBorder(image, 0, cell_h - image.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(12, 12, 12))
              for image in thumbs]
    rows = []
    for start in range(0, len(padded), 3):
        row = padded[start:start + 3]
        while len(row) < 3:
            row.append(np.full((cell_h, thumb_w, 3), 12, dtype=np.uint8))
        rows.append(np.hstack(row))
    cv2.imwrite(str(destination), np.vstack(rows))


def main() -> int:
    args = parse_args()
    if args.max_dim <= 0:
        raise SystemExit("--max-dim must be positive")

    inputs = args.inputs or DEFAULT_INPUTS
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    engine = RetouchEngine()
    sheet_items: List[Tuple[str, np.ndarray]] = []

    for path in inputs:
        if not path.is_file():
            print(f"[SKIP] missing input: {path}")
            continue
        image = load(path, args.max_dim)
        source_dir = output / path.stem
        source_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(source_dir / "source_qa_resolution.jpg"), image)

        for name, base_kwargs, on_kwargs in cases(image):
            if args.case_names and name not in args.case_names:
                continue
            try:
                base = np.asarray(engine.process(image, **base_kwargs))
                on = np.asarray(engine.process(image, **on_kwargs))
                view = montage(base, on, f"{path.stem}: {name}")
                cv2.imwrite(str(source_dir / f"{name}_stage_off.jpg"), base)
                cv2.imwrite(str(source_dir / f"{name}_stage_on.jpg"), on)
                cv2.imwrite(str(source_dir / f"{name}_montage.jpg"), view)
                delta = float(np.mean(np.abs(on.astype(np.float32) - base.astype(np.float32))))
                print(f"[OK] {path.stem}/{name}: mean|delta|={delta:.3f}")
                sheet_items.append((f"{path.stem} - {name}", view))
            except Exception as exc:  # Keep one failed stage from hiding others.
                print(f"[FAIL] {path.stem}/{name}: {exc}")

    contact_sheet(sheet_items, output / "contact_sheet.jpg")
    print(f"Outputs: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
