"""Visual QA — skin_sss (subsurface-scatter finish) via game_character_v2.

Renders the side-lit stress shot with game_character_v1 and game_character_v2
(identical except skin.sss=0.40), locates the face from the render diff (sss
only touches skin), and writes 1:1 crops + an amplified diff map to
test_output/. Also prints each render's engine QA warnings.

Run: python3 scripts/review/visual_qa_skin_sss.py [path-to-photo]
Exit 0 always (reporting tool); os._exit dodges MediaPipe teardown deadlock.
"""

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser(
    "~/Pictures/2026/2026-06-14/DSCF7142.jpg"
)
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "test_output")


def main() -> int:
    from retouch import RetouchEngine

    img = cv2.imread(SRC)
    if img is None:
        print(f"cannot read {SRC}")
        return 1
    print(f"input {SRC} {img.shape}")

    engine = RetouchEngine()
    renders = {}
    for recipe in ("game_character_v1", "game_character_v2"):
        res = engine.process(img.copy(), recipe=recipe)
        renders[recipe] = np.asarray(res).astype(np.uint8)
        print(f"{recipe}: qa={list(getattr(res, 'qa', []) or [])}")

    v1 = renders["game_character_v1"]
    v2 = renders["game_character_v2"]

    diff = np.abs(v2.astype(np.float32) - v1.astype(np.float32)).sum(axis=-1)
    print(f"v1-vs-v2 diff: mean={diff.mean():.3f} max={diff.max():.1f}")

    # sss only touches skin, so the diff centroid + spread locates the face.
    ys, xs = np.nonzero(diff > max(8.0, 0.25 * diff.max()))
    if len(xs) == 0:
        print("NO DIFF FOUND — skin_sss appears inert end-to-end")
        return 1
    cx, cy = int(np.median(xs)), int(np.median(ys))
    half = int(max(xs.std(), ys.std()) * 2.2) or 300
    h, w = v1.shape[:2]
    x1, x2 = max(cx - half, 0), min(cx + half, w)
    y1, y2 = max(cy - half, 0), min(cy + half, h)
    print(f"face crop: x[{x1}:{x2}] y[{y1}:{y2}]")

    strip = np.hstack([img[y1:y2, x1:x2], v1[y1:y2, x1:x2], v2[y1:y2, x1:x2]])
    base = os.path.splitext(os.path.basename(SRC))[0]
    cv2.imwrite(os.path.join(OUT, f"sss_qa_{base}_face_orig_v1_v2.jpg"), strip,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    amp = np.clip(diff / max(diff.max(), 1e-3) * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, f"sss_qa_{base}_diff.jpg"),
                cv2.applyColorMap(amp, cv2.COLORMAP_INFERNO))
    cv2.imwrite(os.path.join(OUT, f"sss_qa_{base}_full_v2.jpg"), v2,
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("wrote 3 files to test_output/")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    os._exit(rc)
