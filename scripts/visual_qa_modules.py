#!/usr/bin/env python3
"""Module-level visual QA — numbers + paths only (no pixels to stdout).

Writes montages under test_output/visual_qa_modules/ and REPORT.md.
Never prints image arrays. Human gates stay PENDING with paths.

Usage:
  python3 scripts/visual_qa_modules.py
  python3 scripts/visual_qa_modules.py --src path/to.jpg --maxd 960
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from retouch.engine import RetouchEngine  # noqa: E402


MODULES = {
    "face_exposure": dict(face_exposure=40),
    "blotch_r4": dict(blotch_reduction=0.7),
    "f5_liquify": dict(reshape_eye_size=40, reshape_jaw_width=-25, reshape_chin_length=15),
    "sclera": dict(eye_sclera_vessel_remove=60, eye_enhance=20),
    "backdrop": dict(backdrop_cleanup=60),
    "fabric": dict(fabric_wrinkle_smooth=60),
    "wrinkle": dict(
        wrinkle_soften=40,
        wrinkle_soften_forehead=50,
        wrinkle_soften_nasolabial=40,
    ),
    "makeup_p4": dict(makeup_coverage_even=0.5, makeup_cake_reduce=0.3),
}


def load(src: Path, maxd: int) -> np.ndarray:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(src)
    h, w = img.shape[:2]
    if max(h, w) > maxd:
        s = maxd / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), cv2.INTER_AREA)
    return img


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))


def montage(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = min(a.shape[0], b.shape[0])
    sep = np.full((h, 4, 3), 200, dtype=np.uint8)
    return np.hstack([a[:h], sep, b[:h]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="test_output/masterwork_v1/DSCF8007.jpg")
    ap.add_argument("--out", default="test_output/visual_qa_modules")
    ap.add_argument("--maxd", type=int, default=960)
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_file():
        # fallback candidates
        for cand in sorted(Path("test_output").glob("DSCF*.jpg"))[:1]:
            src = cand
            break
    if not src.is_file():
        print(f"FAIL: no source image at {args.src}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    img = load(src, args.maxd)
    eng = RetouchEngine()
    base = np.asarray(eng.process(img, recipe="natural", fast=True))
    if base.dtype != np.uint8:
        base = np.clip(base, 0, 255).astype(np.uint8)

    lines = [
        f"# Module Visual QA — {date.today().isoformat()}",
        "",
        f"**Source:** `{src}` (maxd={args.maxd})",
        f"**Out:** `{out_dir}/`",
        f"**Harness:** `scripts/visual_qa_modules.py`",
        "",
        "Human gates (Natural / Edge Tear) = PENDING + montage path — never auto PASS.",
        "",
        "| module | MAE vs base | montage | auto note |",
        "|--------|------------:|---------|-----------|",
    ]

    print(f"src={src} out={out_dir}")
    for name, kw in MODULES.items():
        try:
            on = np.asarray(eng.process(img, recipe="natural", fast=True, **kw))
            if on.dtype != np.uint8:
                on = np.clip(on, 0, 255).astype(np.uint8)
            m = mae(base, on)
            mon = montage(base, on)
            mon_path = out_dir / f"{name}_montage.jpg"
            cv2.imwrite(str(mon_path), mon)
            note = "changed" if m > 0.05 else "near-identity"
            lines.append(
                f"| {name} | {m:.3f} | `{mon_path}` | {note}; Natural PENDING |"
            )
            print(f"{name:14s} MAE={m:.3f}  montage={mon_path}")
        except Exception as e:
            lines.append(f"| {name} | ERROR | — | {e} |")
            print(f"{name:14s} ERROR {e}")

    report = out_dir / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    print("Human: open *_montage.jpg in OS viewer. Do not feed images to LLM.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
