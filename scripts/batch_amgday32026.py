"""One-off batch run for amgday32026 con photo set (cosplay_portrait_polish_v1, draft quality)."""
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retouch import RetouchEngine

SRC = Path.home() / "Desktop" / "amgday32026"
DST = Path.home() / "Desktop" / "amgday32026_retouched"
RECIPE = "cosplay_portrait_polish_v1"

DST.mkdir(exist_ok=True)

engine = RetouchEngine()

images = sorted(SRC.glob("DSCF*.jpg"))
print(f"Found {len(images)} images")

for i, path in enumerate(images, 1):
    t0 = time.time()
    img = cv2.imread(str(path))
    if img is None:
        print(f"[{i}/{len(images)}] SKIP unreadable: {path.name}")
        continue
    result = engine.process(img, recipe=RECIPE, quality="draft")
    out_path = DST / path.name
    cv2.imwrite(str(out_path), result.image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    dt = time.time() - t0
    print(f"[{i}/{len(images)}] {path.name} -> {out_path.name} ({dt:.1f}s)")

print("Done.")
