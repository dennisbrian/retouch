#!/usr/bin/env python3
"""Extract a single-face track JSON for video_qa_report.py.

V0 review-only tool: runs the repo's FaceDetector per frame on a local clip
and writes the harness track contract
(``{"frames": [{"frame", "face": [x, y, w, h], "confidence"}]}``). It selects
the largest detected face per frame and omits frames with no detection, which
the QA harness treats as untracked. Detection runs on a downscaled proxy for
speed; boxes are scaled back to source resolution.

This script processes local files only and never uploads or retains facial
data beyond the JSON boxes it writes.

Usage:
    python3 scripts/review/extract_face_track.py clip.mov --out tracks.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

Bbox = tuple[int, int, int, int]


def select_primary_face(candidates: Sequence[tuple[Bbox, float]]) -> tuple[Bbox, float] | None:
    """Pick the largest-area face; the V0 contract is one face per frame."""
    best: tuple[Bbox, float] | None = None
    best_area = 0
    for bbox, confidence in candidates:
        area = bbox[2] * bbox[3]
        if area > best_area:
            best = (bbox, float(confidence))
            best_area = area
    return best


def scale_bbox(bbox: Bbox, scale: float, frame_width: int, frame_height: int) -> Bbox:
    """Scale a proxy-space box back to source resolution and clamp to frame."""
    x = max(0, int(round(bbox[0] * scale)))
    y = max(0, int(round(bbox[1] * scale)))
    w = min(frame_width - x, int(round(bbox[2] * scale)))
    h = min(frame_height - y, int(round(bbox[3] * scale)))
    return (x, y, max(w, 1), max(h, 1))


def build_payload(records: list[dict[str, object]], source: str, fps: float, detect_width: int) -> dict[str, object]:
    """Assemble the track JSON; extra metadata keys are ignored by the harness."""
    return {
        "source": source,
        "fps": fps,
        "detect_width": detect_width,
        "frames": records,
    }


def extract_tracks(
    video_path: Path,
    *,
    detect_width: int = 1280,
    max_frames: int | None = None,
) -> tuple[list[dict[str, object]], float, int]:
    """Run per-frame detection; returns (records, fps, frames_read)."""
    import cv2

    from retouch.detection import FaceDetector

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not decode {video_path}.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    records: list[dict[str, object]] = []
    frames_read = 0
    try:
        with FaceDetector(max_faces=4) as detector:
            while max_frames is None or frames_read < max_frames:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                height, width = frame.shape[:2]
                scale = width / detect_width if width > detect_width else 1.0
                proxy = (
                    cv2.resize(frame, (detect_width, int(round(height / scale))), interpolation=cv2.INTER_AREA)
                    if scale > 1.0
                    else frame
                )
                faces = detector.detect(proxy)
                primary = select_primary_face(
                    [(face.bbox, min(max(face.confidence, 0.0), 1.0)) for face in faces]
                )
                if primary is not None:
                    bbox = scale_bbox(primary[0], scale, width, height)
                    records.append(
                        {"frame": frames_read, "face": list(bbox), "confidence": primary[1]}
                    )
                frames_read += 1
                if frames_read % 30 == 0:
                    print(f"  frame {frames_read}: {len(records)} tracked")
    finally:
        capture.release()
    if frames_read == 0:
        raise ValueError(f"No readable frames in {video_path}.")
    return records, fps, frames_read


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Local source clip")
    parser.add_argument("--out", type=Path, required=True, help="Track JSON output path")
    parser.add_argument("--detect-width", type=int, default=1280, help="Proxy width for detection")
    parser.add_argument("--max-frames", type=int, help="Optional cap for a quick run")
    args = parser.parse_args()
    if args.detect_width < 64:
        parser.error("--detect-width must be at least 64")

    records, fps, frames_read = extract_tracks(
        args.video, detect_width=args.detect_width, max_frames=args.max_frames
    )
    payload = build_payload(records, args.video.name, fps, args.detect_width)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    coverage = len(records) / frames_read
    print(f"wrote {args.out}: {len(records)}/{frames_read} frames tracked ({coverage:.1%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
