"""Single-face video tracker with landmarks (V1 slice S2).

Emits one selected face per frame: a box, 478 landmarks and a confidence.
Two models do the work:

* The engine's ``FaceDetector`` finds faces on a downscaled copy of the
  frame. It runs on the first frame, and again whenever the selected face is
  lost. It is the same detector the V0 extractor uses, so small faces in 4K
  frames are found (MediaPipe FaceLandmarker's own detector misses a face
  about 11% of the frame wide: 0 of 95 frames on DSCF4322).
* While the face is tracked, the repo's FaceLandmarker (``face_landmarker.task``)
  runs in VIDEO mode on a square crop around the previous box, so the face
  fills the model's input and consecutive frames reuse the previous face
  instead of re-detecting from scratch (lower landmark jitter).

Selection policy (plan S2): the largest face in the first frame that has
one, then the face nearest the previous box on every later frame. The
tracker never switches to a different face mid-clip; a face is only accepted
if it is close to where the selected one was last seen (the allowed distance
grows while the face is lost, so a subject who moves during an occlusion is
picked up again). Frames where the selected face is not found are left out,
which the QA harness and later slices treat as untracked. Records say which
model found them (``via``: ``"video"`` or ``"detect"``).

The JSON written by :func:`build_contract` is the V0 harness track contract
(``scripts/review/video_qa_report.py``) at version 2: the V0 per-frame keys
are unchanged and ``landmarks``/``time``/``via`` are added, along with
top-level ``version`` and ``landmark_format``.

Usage::

    python -m retouch.video.tracker clip.mp4 --out tracks.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Callable, List, Optional, Protocol, Union

import cv2
import numpy as np

PathLike = Union[str, Path]
Bbox = tuple  # (x, y, w, h) in source pixels

CONTRACT_VERSION = 2
LANDMARK_FORMAT = "mediapipe_face_478_xyz_normalized"
NUM_LANDMARKS = 478

# A face counts as the same one if its box centre is within this many
# previous-box diagonals of the last selected box...
_MATCH_RADIUS = 0.5
# ...growing by this much per frame the face has been lost, up to the cap.
_MATCH_RADIUS_GROWTH = 0.05
_MATCH_RADIUS_MAX = 2.0
# Box size (sqrt of area ratio) allowed between consecutive matches, and
# after a loss.
_SIZE_RATIO_MAX = 1.5
_SIZE_RATIO_MAX_LOST = 2.0


# The VIDEO-mode crop around the previous box: this many box sides wide,
# resized so its long edge is _CROP_SIZE px (FaceLandmarker's input is 256).
_CROP_SCALE = 2.0
_CROP_SIZE = 384


class TrackerBackend(Protocol):
    """The two models; each returns every face's landmarks, (478, 3) normalized to its input."""

    def find_faces(self, image_bgr: np.ndarray) -> List[np.ndarray]: ...

    def landmark_video(self, crop_bgr: np.ndarray, timestamp_ms: int) -> List[np.ndarray]: ...

    def close(self) -> None: ...


class MediaPipeBackend:
    """The engine's FaceDetector plus a VIDEO-mode FaceLandmarker.

    Must be closed on the main thread (see ``FaceDetector.close``): left to the
    garbage collector, MediaPipe's task finalizer can hang the process.
    """

    def __init__(self, *, max_faces: int = 4, min_confidence: float = 0.5) -> None:
        import mediapipe as mp

        from ..detection import FaceDetector
        from ..model_fetch import get_model_path

        self._mp = mp
        self._detector = FaceDetector(max_faces=max_faces)
        if not self._detector.available:
            reason = self._detector.unavailable_reason
            self._detector.close()
            raise RuntimeError(f"face detection is unavailable: {reason}")
        vision = mp.tasks.vision
        self._video = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=get_model_path("face_landmarker")),
                running_mode=vision.RunningMode.VIDEO,
                num_faces=max_faces,
                min_face_detection_confidence=min_confidence,
                min_face_presence_confidence=min_confidence,
                min_tracking_confidence=min_confidence,
            )
        )

    def find_faces(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        return [
            np.array([(p.x, p.y, p.z) for p in face.landmarks.landmark], dtype=np.float32)
            for face in self._detector.detect(image_bgr)
        ]

    def landmark_video(self, crop_bgr: np.ndarray, timestamp_ms: int) -> List[np.ndarray]:
        rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        result = self._video.detect_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        return [np.array([(p.x, p.y, p.z) for p in face], dtype=np.float32) for face in result.face_landmarks]

    def close(self) -> None:
        if self._video is not None:
            try:
                self._video.close()
            except Exception:  # teardown must not mask the caller's error
                pass
            self._video = None
        self._detector.close()


@dataclasses.dataclass(frozen=True)
class TrackedFace:
    """The selected face on one frame."""

    frame: int
    time: float  # seconds, from the frame's presentation timestamp
    bbox: Bbox  # (x, y, w, h), source pixels, clamped to the frame
    landmarks: np.ndarray  # (478, 3) float32; x, y normalized to the frame
    confidence: float
    via: str  # "video" (VIDEO-mode landmarker on a crop) or "detect" (FaceDetector)


@dataclasses.dataclass(frozen=True)
class VideoTrack:
    """Tracker output for a whole clip."""

    width: int
    height: int
    fps: Fraction
    frame_count: int  # frames decoded, tracked or not
    faces: List[TrackedFace]

    @property
    def coverage(self) -> float:
        return len(self.faces) / self.frame_count if self.frame_count else 0.0


def landmarks_bbox(landmarks: np.ndarray, width: int, height: int) -> Bbox:
    """Landmark box in pixels (edges rounded), clamped to the frame (at least 1x1)."""
    xs = landmarks[:, 0] * width
    ys = landmarks[:, 1] * height
    x0 = max(0, int(round(float(xs.min()))))
    y0 = max(0, int(round(float(ys.min()))))
    x1 = min(width, int(round(float(xs.max()))))
    y1 = min(height, int(round(float(ys.max()))))
    x0, y0 = min(x0, width - 1), min(y0, height - 1)
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _centre(b: Bbox) -> tuple:
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def select_initial(boxes: Sequence[Bbox]) -> Optional[int]:
    """Index of the largest box (the first one wins a tie), or None."""
    best, best_area = None, 0
    for i, b in enumerate(boxes):
        area = b[2] * b[3]
        if area > best_area:
            best, best_area = i, area
    return best


def match_previous(boxes: Sequence[Bbox], previous: Bbox, frames_lost: int = 0) -> Optional[int]:
    """Index of the box nearest ``previous`` that could be the same face, or None.

    A candidate must lie within a radius of the previous box's diagonal (wider
    the longer the face has been lost) and be of similar size.
    """
    diag = math.hypot(previous[2], previous[3])
    radius = min(_MATCH_RADIUS + _MATCH_RADIUS_GROWTH * frames_lost, _MATCH_RADIUS_MAX) * diag
    size_limit = _SIZE_RATIO_MAX if frames_lost == 0 else _SIZE_RATIO_MAX_LOST
    px, py = _centre(previous)
    prev_size = math.sqrt(previous[2] * previous[3])
    best, best_dist = None, None
    for i, b in enumerate(boxes):
        ratio = math.sqrt(b[2] * b[3]) / prev_size
        if not 1.0 / size_limit <= ratio <= size_limit:
            continue
        cx, cy = _centre(b)
        dist = math.hypot(cx - px, cy - py)
        if dist <= radius and (best_dist is None or dist < best_dist):
            best, best_dist = i, dist
    return best


def crop_around(box: Bbox, width: int, height: int, scale: float = _CROP_SCALE) -> Bbox:
    """Square region ``scale`` box sides wide centred on ``box``, clipped to the frame."""
    side = scale * max(box[2], box[3])
    cx, cy = _centre(box)
    x0 = max(0, int(round(cx - side / 2)))
    y0 = max(0, int(round(cy - side / 2)))
    x1 = min(width, int(round(cx + side / 2)))
    y1 = min(height, int(round(cy + side / 2)))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


class FaceTracker:
    """Follows one face through consecutive frames; see the module docstring.

    Feed frames in presentation order with :meth:`update`. Use as a context
    manager (or call :meth:`close`) on the main thread.
    """

    def __init__(
        self,
        *,
        max_faces: int = 4,
        min_confidence: float = 0.5,
        detect_max_dim: int = 1280,
        backend: Optional[TrackerBackend] = None,
    ) -> None:
        if detect_max_dim < 64:
            raise ValueError("detect_max_dim must be at least 64")
        self.max_faces = max_faces
        self.min_confidence = min_confidence
        self.detect_max_dim = detect_max_dim
        self._backend = backend if backend is not None else MediaPipeBackend(
            max_faces=max_faces, min_confidence=min_confidence
        )
        self._frame = 0
        self._t0: Optional[float] = None
        self._last_ms = -1
        self._last_box: Optional[Bbox] = None
        self._frames_lost = 0

    def _proxy(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        scale = self.detect_max_dim / max(height, width)
        if scale >= 1.0:
            return image
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    def _timestamp_ms(self, time_s: float) -> int:
        # VIDEO mode needs strictly increasing integer milliseconds.
        ms = max(int(round((time_s - self._t0) * 1000.0)), self._last_ms + 1)
        self._last_ms = ms
        return ms

    def _pick(self, faces: List[np.ndarray], width: int, height: int) -> Optional[int]:
        boxes = [landmarks_bbox(f, width, height) for f in faces]
        if self._last_box is None:
            return select_initial(boxes)
        return match_previous(boxes, self._last_box, self._frames_lost)

    def _track_in_crop(self, image: np.ndarray, time_s: float) -> List[np.ndarray]:
        """VIDEO-mode landmarks on a crop around the last box, mapped to frame coordinates."""
        height, width = image.shape[:2]
        x, y, w, h = crop_around(self._last_box, width, height)
        crop = image[y : y + h, x : x + w]
        size = _CROP_SIZE / max(w, h)
        resized = cv2.resize(
            crop,
            (max(1, round(w * size)), max(1, round(h * size))),
            interpolation=cv2.INTER_AREA if size < 1.0 else cv2.INTER_LINEAR,
        )
        faces = []
        for face in self._backend.landmark_video(resized, self._timestamp_ms(time_s)):
            mapped = np.asarray(face, dtype=np.float32).copy()
            if mapped.ndim == 2 and mapped.shape[1] == 3:
                # Normalized to the crop, which was resized uniformly: map back via its size.
                mapped[:, 0] = (x + mapped[:, 0] * w) / width
                mapped[:, 1] = (y + mapped[:, 1] * h) / height
                # MediaPipe depth uses the same normalized scale as x.
                mapped[:, 2] *= w / width
            faces.append(mapped)
        return faces

    def update(self, image: np.ndarray, time_s: float) -> Optional[TrackedFace]:
        """Track the selected face on the next frame; None when it isn't found."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an H x W x 3 uint8 BGR image, got {image.dtype} {image.shape}")
        height, width = image.shape[:2]
        index = self._frame
        self._frame += 1
        if self._t0 is None:
            self._t0 = time_s

        chosen = None
        if self._last_box is not None:
            via = "video"
            faces = self._track_in_crop(image, time_s)
            chosen = self._pick(faces, width, height)
        if chosen is None:
            # Not tracked yet, or lost: detect on the whole (downscaled) frame.
            via = "detect"
            faces = self._backend.find_faces(self._proxy(image))
            chosen = self._pick(faces, width, height)
        if chosen is None:
            if self._last_box is not None:
                self._frames_lost += 1
            return None

        landmarks = np.asarray(faces[chosen], dtype=np.float32)
        if landmarks.shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"backend returned landmarks of shape {landmarks.shape}")
        box = landmarks_bbox(landmarks, width, height)
        self._last_box = box
        self._frames_lost = 0
        # Neither model reports a per-face score; a returned face has passed the
        # detection/presence gates, so it is recorded as 1.0.
        return TrackedFace(index, float(time_s), box, landmarks, 1.0, via)

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> FaceTracker:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def track_video(
    path: PathLike,
    *,
    max_frames: Optional[int] = None,
    progress: Optional[Callable[[int, int], None]] = None,
    **tracker_options,
) -> VideoTrack:
    """Decode a clip with ``media.read_frames`` and track its selected face."""
    from .media import probe, read_frames

    info = probe(path)
    faces: List[TrackedFace] = []
    frames = 0
    with FaceTracker(**tracker_options) as tracker:
        for frame in read_frames(path):
            if max_frames is not None and frames >= max_frames:
                break
            face = tracker.update(frame.image, frame.time)
            if face is not None:
                faces.append(face)
            frames += 1
            if progress is not None:
                progress(frames, len(faces))
    if frames == 0:
        raise ValueError(f"{path}: no frames decoded")
    return VideoTrack(info.width, info.height, info.fps, frames, faces)


def build_contract(track: VideoTrack, source: str, *, decimals: int = 5) -> dict:
    """The version-2 track JSON (a superset of the V0 harness contract)."""
    frames = [
        {
            "frame": face.frame,
            "face": [int(v) for v in face.bbox],
            "confidence": face.confidence,
            "time": round(face.time, 6),
            "via": face.via,
            "landmarks": np.round(face.landmarks.astype(np.float64), decimals).tolist(),
        }
        for face in track.faces
    ]
    return {
        "version": CONTRACT_VERSION,
        "source": source,
        "width": track.width,
        "height": track.height,
        "fps": float(track.fps),
        "frame_count": track.frame_count,
        "landmark_format": LANDMARK_FORMAT,
        "confidence_source": "presence_gate",
        "frames": frames,
    }


def load_contract(path: PathLike) -> dict:
    """Read and validate a track JSON; version-1 files (no ``version``) have no landmarks."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    version = payload.get("version", 1)
    if version not in (1, CONTRACT_VERSION):
        raise ValueError(f"{path}: unsupported track contract version {version!r}")
    frames = payload.get("frames")
    if not isinstance(frames, list):
        raise ValueError(f"{path}: track JSON must contain a 'frames' list")
    seen = set()
    for record in frames:
        frame = int(record["frame"])
        if frame in seen:
            raise ValueError(f"{path}: duplicate record for frame {frame}")
        seen.add(frame)
        if len(record["face"]) != 4 or not 0.0 <= float(record["confidence"]) <= 1.0:
            raise ValueError(f"{path}: invalid record at frame {frame}")
        if version >= 2 and np.asarray(record["landmarks"]).shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"{path}: frame {frame} needs {NUM_LANDMARKS} x 3 landmarks")
    return payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Track one face through a clip and write track JSON.")
    parser.add_argument("video", type=Path, help="Local source clip")
    parser.add_argument("--out", type=Path, required=True, help="Track JSON output path")
    parser.add_argument("--detect-max-dim", type=int, default=1280, help="Long edge of the detection proxy")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    args = parser.parse_args(argv)

    def report(frames: int, tracked: int) -> None:
        if frames % 60 == 0:
            print(f"  frame {frames}: {tracked} tracked", flush=True)

    track = track_video(
        args.video,
        max_frames=args.max_frames,
        progress=report,
        detect_max_dim=args.detect_max_dim,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build_contract(track, args.video.name)) + "\n", encoding="utf-8")
    detected = sum(face.via == "detect" for face in track.faces)
    print(
        f"wrote {args.out}: {len(track.faces)}/{track.frame_count} frames tracked "
        f"({track.coverage:.1%}), {detected} from the detector"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
