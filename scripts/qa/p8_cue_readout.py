#!/usr/bin/env python3
"""Run the P8 R1 observable-cue readout without rendering an edit.

Usage:

    RETOUCH_GPU=0 RETOUCH_MEDIAPIPE_BACKEND=legacy .venv/bin/python \
        scripts/qa/p8_cue_readout.py portrait.jpg --json readout.json

The command performs face detection and region parsing, then writes only
measurement/provenance JSON.  It never calls ``RetouchEngine.process`` and
does not write edited images.  The readout describes observable appearance;
it is not an age estimate or an aging-vector control.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("RETOUCH_GPU", "0")
os.environ.setdefault("RETOUCH_MEDIAPIPE_BACKEND", "legacy")

import cv2  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retouch.aging_cues import measure_p8_cues  # noqa: E402
from retouch.detection import FaceDetector  # noqa: E402
from retouch.io import imread_exif  # noqa: E402
from retouch.parsing import FaceParser  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resize_for_measurement(image: Any, max_dim: Optional[int]) -> Any:
    if max_dim is None or max_dim < 1:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return image
    scale = float(max_dim) / float(longest)
    return cv2.resize(
        image,
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def _measure_one(
    path: Path,
    detector: FaceDetector,
    parser: FaceParser,
    *,
    max_dim: Optional[int],
    min_pixels: int,
) -> Dict[str, Any]:
    source_image = imread_exif(path)
    if source_image is None:
        raise ValueError(f"could not decode image: {path}")
    image = _resize_for_measurement(source_image, max_dim)
    faces = detector.detect(image)
    person_mask = detector.segment_person(image) if faces else None
    records: List[Dict[str, Any]] = []

    for index, face in enumerate(faces):
        regions = parser.parse(
            face.landmarks,
            image,
            face.bbox,
            person_mask=person_mask,
            ied=face.ied,
        )
        readout = measure_p8_cues(image, regions, min_pixels=min_pixels)
        records.append({
            "face_index": int(index),
            "bbox": [int(v) for v in face.bbox],
            "detector_confidence": float(face.confidence),
            "detector_confidence_source": str(face.confidence_source),
            "readout": readout.to_dict(),
        })

    return {
        "status": "ok",
        "path": str(path),
        "sha256": _sha256(path),
        "source_shape": [int(v) for v in source_image.shape],
        "measured_shape": [int(v) for v in image.shape],
        "measurement_scope": "observable appearance only; no age or tissue inference",
        "face_count": int(len(faces)),
        "faces": records,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("input", nargs="+", type=Path, help="image path(s)")
    parser.add_argument("--json", type=Path, default=None, dest="json_path",
                        help="write the JSON readout to this path")
    parser.add_argument("--max-dim", type=int, default=None,
                        help="optionally downscale the measurement input")
    parser.add_argument("--min-pixels", type=int, default=32,
                        help="minimum support pixels (default: 32)")
    parser.add_argument("--max-faces", type=int, default=10,
                        help="maximum faces per image (default: 10)")
    parser.add_argument("--min-confidence", type=float, default=0.4,
                        help="detector confidence floor (default: 0.4)")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.min_pixels < 32:
        raise SystemExit("--min-pixels must be at least 32")
    if args.max_faces < 1:
        raise SystemExit("--max-faces must be positive")

    try:
        detector = FaceDetector(
            max_faces=args.max_faces,
            min_confidence=args.min_confidence,
            refine_landmarks=True,
            allow_unavailable=False,
        )
        parser = FaceParser()
    except Exception as exc:
        payload = {"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 2

    results: List[Dict[str, Any]] = []
    try:
        for raw_path in args.input:
            path = raw_path.expanduser().resolve()
            try:
                results.append(
                    _measure_one(
                        path,
                        detector,
                        parser,
                        max_dim=args.max_dim,
                        min_pixels=args.min_pixels,
                    )
                )
            except Exception as exc:  # keep a batch auditable when one asset fails
                results.append({
                    "status": "error",
                    "path": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                })
    finally:
        detector.close()

    payload = {
        "descriptor_revision": "p8-r1-appearance-v1",
        "measurement_scope": "observable appearance only; no age or tissue inference",
        "detector": detector.runtime_status(),
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if any(item.get("status") != "ok" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
