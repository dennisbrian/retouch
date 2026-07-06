#!/usr/bin/env python3
"""Batch process nikke folder with natural_polish_v1, one image at a time."""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from retouch import RetouchEngine
from retouch.io import imread_exif, encode_write_params

INPUT = Path.home() / "Desktop/duotian nikke"
OUTPUT = Path.home() / "Desktop/duotian_nikke_retouched"
RECIPE = "natural_polish_v1"

OUTPUT.mkdir(parents=True, exist_ok=True)
exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
files = sorted([f for f in INPUT.iterdir() if f.suffix.lower() in exts])
print(f"Found {len(files)} images", flush=True)

engine = RetouchEngine()
done = 0
try:
    for i, f in enumerate(files, 1):
        try:
            img = imread_exif(f)
            result = engine.process(img, recipe=RECIPE)
            out = OUTPUT / f"{f.stem}_retouched.jpg"
            cv2.imwrite(str(out), result, encode_write_params("jpg", 95))
            done += 1
            print(f"[{i}/{len(files)}] {f.name}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(files)}] {f.name}: FAILED {e}", flush=True)
finally:
    engine.close()
print(f"\nDone: {done}/{len(files)} processed", flush=True)
