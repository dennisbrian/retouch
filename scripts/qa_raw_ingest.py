"""Visual QA: OLD 8-bit RAF ingest vs NEW 16-bit float ingest through process().

Renders the same RAF two ways at ~1280px and writes old/new/montage into
test_output/visual_qa_raw/. The amplified diff panel exposes the (subtle)
precision/headroom difference the 16-bit path buys.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retouch import RetouchEngine
from retouch.io import imread_exif, read_image_16bit, resize_for_processing

RAF = Path("/Users/dennis/Pictures/2025/2025-08-09/_DSF1853.RAF")
OUT = Path(__file__).resolve().parents[1] / "test_output" / "visual_qa_raw"
MAX_DIM = 1280
RECIPE = "natural"


def _to_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8) if img.dtype != np.uint8 else img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    engine = RetouchEngine()

    # OLD path: 8-bit imread_exif (uint8 BGR sRGB)
    old_in = imread_exif(RAF)
    old_in, _ = resize_for_processing(old_in, MAX_DIM)
    print(f"OLD ingest: dtype={old_in.dtype} shape={old_in.shape} "
          f"min={old_in.min()} max={old_in.max()}")
    old_res = np.asarray(engine.process(old_in, recipe=RECIPE).image)

    # NEW path: 16-bit read_image_16bit (float32 BGR [0,255] sRGB)
    new_in = read_image_16bit(RAF)
    new_in, _ = resize_for_processing(new_in, MAX_DIM)
    print(f"NEW ingest: dtype={new_in.dtype} shape={new_in.shape} "
          f"min={new_in.min():.2f} max={new_in.max():.2f}")
    new_res = np.asarray(engine.process(new_in, recipe=RECIPE).image)

    print(f"OLD result: dtype={old_res.dtype} shape={old_res.shape}")
    print(f"NEW result: dtype={new_res.dtype} shape={new_res.shape}")

    if old_res.shape != new_res.shape:
        new_res = cv2.resize(new_res, (old_res.shape[1], old_res.shape[0]))

    diff = np.abs(old_res.astype(np.float32) - new_res.astype(np.float32))
    print(f"result diff: mean={diff.mean():.3f} max={diff.max():.1f} "
          f"px>2={(diff.max(axis=2) > 2).mean() * 100:.2f}%")

    amp = np.clip(diff * 12.0, 0, 255).astype(np.uint8)

    cv2.imwrite(str(OUT / "old_path.jpg"), _to_u8(old_res),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(OUT / "new_path.jpg"), _to_u8(new_res),
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    h = old_res.shape[0]
    sep = np.full((h, 4, 3), 200, dtype=np.uint8)
    montage = np.hstack([_to_u8(old_res), sep, _to_u8(new_res), sep, amp])
    cv2.imwrite(str(OUT / "raw_compare_montage.jpg"), montage,
                [cv2.IMWRITE_JPEG_QUALITY, 95])

    engine.close()
    print(f"wrote: {OUT}/old_path.jpg, new_path.jpg, raw_compare_montage.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
