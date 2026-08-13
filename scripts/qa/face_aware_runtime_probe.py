"""Probe the installed face-aware runtime with a real image.

This is intentionally smaller than the certification runner. It answers one
question: can this Python environment construct the CPU landmark backend and
return dense landmarks? A successful probe is necessary, but not sufficient,
for certification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from retouch.detection import FaceDetector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()

    image = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if image is None:
        print(json.dumps({"status": "error", "error": f"Could not read {args.input}"}))
        return 2

    detector = None
    try:
        detector = FaceDetector(allow_unavailable=False)
        faces = detector.detect(image)
        landmarks = [len(face.landmarks.landmark) for face in faces]
        result = {
            "status": "pass" if faces and all(count >= 468 for count in landmarks) else "fail",
            "mediapipe": __import__("mediapipe").__version__,
            "backend": "legacy_cpu" if detector._legacy_mesh is not None else "tasks",
            "face_count": len(faces),
            "landmark_counts": landmarks,
        }
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "pass" else 1
    except Exception as exc:  # noqa: BLE001 - probe reports the environment failure
        print(json.dumps({"status": "blocked", "error": f"{type(exc).__name__}: {exc}"}, indent=2))
        return 2
    finally:
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    raise SystemExit(main())
