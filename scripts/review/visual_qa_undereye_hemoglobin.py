#!/usr/bin/env python3
"""Create human-review montages for the E-EYE-4 under-eye hemoglobin spike.

This deliberately avoids RetouchEngine/MediaPipe so it can run in local
environments where MediaPipe initialization is unavailable. Haar eye masks are
only a QA approximation; they are not part of the product implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.undereye import UndereyeProcessor  # noqa: E402


def _resize(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale == 1.0:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def _detect_masks(
    img: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int] | None, str]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return np.zeros(gray.shape, dtype=np.float32), None, "none"

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])
    # Haar eye cascades routinely confuse costume details and mouths for an
    # eye. A fixed, visible face-relative approximation is more honest for
    # this non-product QA helper: reviewers can reject a bad mask at a glance.
    centers = [
        (x + int(w * 0.34), y + int(h * 0.55)),
        (x + int(w * 0.66), y + int(h * 0.55)),
    ]

    mask = np.zeros(gray.shape, dtype=np.float32)
    for cx, cy in centers:
        cv2.ellipse(
            mask,
            (cx, cy),
            (max(8, int(w * 0.13)), max(5, int(h * 0.07))),
            0,
            0,
            360,
            1.0,
            -1,
        )
    return (
        cv2.GaussianBlur(mask, (0, 0), 4.0),
        (x, y, w, h),
        "face geometry approximation",
    )


def _panel(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 340), 34), (20, 20, 20), -1)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    return out


def _face_crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    pad_x, pad_y = int(w * 0.15), int(h * 0.10)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(img.shape[1], x + w + pad_x), min(img.shape[0], y + h + pad_y)
    return img[y1:y2, x1:x2]


def _write_case(src: Path, out_dir: Path, strength: float, max_dim: int) -> str:
    original = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if original is None:
        return f"SKIP {src}: unreadable"
    original = _resize(original, max_dim)
    mask, face_box, mask_mode = _detect_masks(original)
    if face_box is None:
        return f"SKIP {src}: no face detected by Haar QA helper"

    processor = UndereyeProcessor()
    processed = processor.attenuate_hemoglobin(original, mask, strength=strength)
    mask_view = cv2.applyColorMap(np.clip(mask * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    overlay = cv2.addWeighted(original, 0.65, mask_view, 0.35, 0.0)

    before = _face_crop(original, face_box)
    after = _face_crop(processed, face_box)
    masked = _face_crop(overlay, face_box)
    montage = np.hstack((
        _panel(before, "BEFORE"),
        _panel(after, f"HB SPIKE {strength:.1f}"),
        _panel(masked, f"QA MASK: {mask_mode}"),
    ))
    stem = src.stem
    output = out_dir / f"{stem}_undereye_hemoglobin_montage.jpg"
    cv2.imwrite(str(output), montage, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return f"OK {src.name}: {output}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path, help="Portrait image paths")
    parser.add_argument("--out", type=Path, default=ROOT / "test_output" / "visual_qa_undereye_hemoglobin")
    parser.add_argument("--strength", type=float, default=0.8)
    parser.add_argument("--max-dim", type=int, default=1600)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    results = [_write_case(src, args.out, args.strength, args.max_dim) for src in args.sources]
    (args.out / "REPORT.txt").write_text("\n".join(results) + "\n", encoding="utf-8")
    print("\n".join(results))
    print(f"Human review: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
