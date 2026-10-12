#!/usr/bin/env python3
"""Naive per-frame video render for the V0 problem baseline.

V0 review-only tool: pushes each frame of a local clip through the image
engine independently — no tracking reuse, no temporal stabilization. This is
deliberately the wrong way to retouch video; its QA metrics quantify the
flicker problem the video track exists to solve. It is not a product render
path and must not be promoted into one.

Usage:
    python3 scripts/review/render_video_naive.py clip.mov --out naive.mp4 --max-frames 60
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Local source clip")
    parser.add_argument("--out", type=Path, required=True, help="Rendered MP4 output path")
    parser.add_argument("--max-frames", type=int, help="Optional cap for a quick baseline")
    args = parser.parse_args()

    import cv2

    from retouch import RetouchEngine

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError(f"Could not decode {args.video}.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rendered = 0
    started = time.perf_counter()
    try:
        with RetouchEngine() as engine:
            while args.max_frames is None or rendered < args.max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                result = engine.process(frame)
                writer.write(result[:, :, :3].astype("uint8"))
                rendered += 1
                if rendered % 10 == 0:
                    per_frame = (time.perf_counter() - started) / rendered
                    print(f"  frame {rendered} ({per_frame:.2f}s/frame)")
    finally:
        capture.release()
        writer.release()
    if rendered == 0:
        raise ValueError(f"No readable frames in {args.video}.")
    elapsed = time.perf_counter() - started
    print(f"wrote {args.out}: {rendered} frames in {elapsed:.1f}s ({elapsed / rendered:.2f}s/frame)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
