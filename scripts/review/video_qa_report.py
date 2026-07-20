#!/usr/bin/env python3
"""Measure temporal stability of a face-retouch video render.

This is a V0 research/QA harness, not a retouch pipeline. It compares a source
clip and a result clip in face-track-normalized coordinates so camera movement
does not dominate the temporal measurements. It never uploads, retains, or
creates facial-video data.

The supplied track JSON must contain one selected face per frame:

    {"frames": [{"frame": 0, "face": [x, y, width, height], "confidence": 0.98}]}

Usage:
    python3 scripts/review/video_qa_report.py source.mp4 result.mp4 tracks.json --out test_output/video_qa/run_01
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class FaceTrack:
    frame: int
    x: int
    y: int
    width: int
    height: int
    confidence: float


def load_tracks(path: Path) -> dict[int, FaceTrack]:
    """Load and validate the single-face JSON track contract."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("frames")
    if not isinstance(records, list):
        raise ValueError("Track JSON must contain a 'frames' list.")
    tracks: dict[int, FaceTrack] = {}
    for record in records:
        try:
            frame = int(record["frame"])
            x, y, width, height = (int(value) for value in record["face"])
            confidence = float(record["confidence"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid track record: {record!r}") from exc
        if frame < 0 or width <= 0 or height <= 0 or not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Invalid track values at frame {frame}.")
        if frame in tracks:
            raise ValueError(f"Duplicate track record for frame {frame}.")
        tracks[frame] = FaceTrack(frame, x, y, width, height, confidence)
    return tracks


def canonical_face_crop(frame: np.ndarray, track: FaceTrack, size: int = 128) -> np.ndarray:
    """Crop a tracked face and resize it into a comparison-friendly space."""
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Frames must be BGR images with three channels.")
    height, width = frame.shape[:2]
    x0, y0 = max(0, track.x), max(0, track.y)
    x1, y1 = min(width, track.x + track.width), min(height, track.y + track.height)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Track at frame {track.frame} is outside the video frame.")
    return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)


def percentile(values: list[float], level: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float32), level)) if values else 0.0


def temporal_metrics(
    source_frames: Iterable[np.ndarray],
    result_frames: Iterable[np.ndarray],
    tracks: dict[int, FaceTrack],
    *,
    min_confidence: float = 0.75,
    canonical_size: int = 128,
) -> dict[str, float | int]:
    """Measure retouch-map stability after normalizing by the face track.

    ``effect`` is result minus source. Its change between adjacent normalized
    face crops is the useful flicker signal: source movement is largely removed
    and ordinary face motion remains measurable as a separate baseline.
    """
    source_list = list(source_frames)
    result_list = list(result_frames)
    if len(source_list) != len(result_list):
        raise ValueError("Source and result clips have different frame counts.")
    effect_energy: list[float] = []
    effect_flicker: list[float] = []
    source_motion: list[float] = []
    previous_effect: np.ndarray | None = None
    previous_source: np.ndarray | None = None
    tracked_frames = 0
    for frame_index, (source, result) in enumerate(zip(source_list, result_list)):
        if source.shape != result.shape:
            raise ValueError(f"Source/result dimensions differ at frame {frame_index}.")
        track = tracks.get(frame_index)
        if track is None or track.confidence < min_confidence:
            previous_effect = None
            previous_source = None
            continue
        source_crop = canonical_face_crop(source, track, canonical_size)
        result_crop = canonical_face_crop(result, track, canonical_size)
        effect = result_crop - source_crop
        effect_energy.append(float(np.mean(np.abs(effect))))
        if previous_effect is not None and previous_source is not None:
            effect_flicker.append(float(np.mean(np.abs(effect - previous_effect))))
            source_motion.append(float(np.mean(np.abs(source_crop - previous_source))))
        previous_effect = effect
        previous_source = source_crop
        tracked_frames += 1
    mean_effect_flicker = float(np.mean(effect_flicker)) if effect_flicker else 0.0
    mean_source_motion = float(np.mean(source_motion)) if source_motion else 0.0
    return {
        "frames_compared": len(source_list),
        "tracked_frames": tracked_frames,
        "track_coverage": tracked_frames / len(source_list) if source_list else 0.0,
        "mean_effect_energy": float(np.mean(effect_energy)) if effect_energy else 0.0,
        "mean_effect_flicker": mean_effect_flicker,
        "p95_effect_flicker": percentile(effect_flicker, 95.0),
        "mean_source_motion": mean_source_motion,
        "effect_flicker_to_motion": mean_effect_flicker / max(mean_source_motion, 1e-6),
    }


def read_video(path: Path, max_frames: int | None = None) -> tuple[list[np.ndarray], float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not decode {path}.")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No readable frames in {path}.")
    return frames, fps


def write_report(out_dir: Path, metrics: dict[str, float | int], tracks: dict[int, FaceTrack]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Video Temporal QA Report",
        "",
        "This is an automated measurement aid. Human review remains required; a low",
        "score does not prove that a render is natural or that mask edges are safe.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in metrics.items():
        rendered = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| `{name}` | {rendered} |")
    lines.extend(
        [
            "",
            f"Track records supplied: {len(tracks)}",
            "",
            "## Human gates",
            "",
            "- [ ] No skin-mask leakage across eyes, brows, lips, hairline, or occluders.",
            "- [ ] No temporal flicker or sliding during turns, speech, blink, or exposure changes.",
            "- [ ] No texture loss, plastic skin, or geometry wobble.",
            "- [ ] Audio duration and A/V sync verified separately.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Original local video")
    parser.add_argument("result", type=Path, help="Retouched local video")
    parser.add_argument("tracks", type=Path, help="Single-face track JSON")
    parser.add_argument("--out", type=Path, required=True, help="Report output directory")
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--max-frames", type=int, help="Optional cap for a quick review run")
    args = parser.parse_args()
    if not 0.0 <= args.min_confidence <= 1.0:
        parser.error("--min-confidence must be between 0 and 1")

    source_frames, source_fps = read_video(args.source, args.max_frames)
    result_frames, result_fps = read_video(args.result, args.max_frames)
    if source_fps > 0 and result_fps > 0 and abs(source_fps - result_fps) > 0.01:
        raise ValueError(f"Source/result FPS differ ({source_fps:.3f} vs {result_fps:.3f}).")
    tracks = load_tracks(args.tracks)
    metrics = temporal_metrics(
        source_frames,
        result_frames,
        tracks,
        min_confidence=args.min_confidence,
    )
    metrics["fps"] = source_fps
    write_report(args.out, metrics, tracks)
    print(f"wrote {args.out / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
