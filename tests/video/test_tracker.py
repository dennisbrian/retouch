"""S2 video-mode tracker: selection, continuity, re-acquire and the version-2 track contract.

Most tests drive the tracker through a scripted fake backend. One test runs the
real MediaPipe landmarker on frames composed from a repo reference photo (no
facial video is committed); it is skipped when the model can't be resolved.
"""

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from retouch.video.tracker import (
    CONTRACT_VERSION,
    NUM_LANDMARKS,
    FaceTracker,
    VideoTrack,
    build_contract,
    landmarks_bbox,
    load_contract,
    match_previous,
    select_initial,
    track_video,
)

REPO = Path(__file__).resolve().parents[2]


def _landmarks(box):
    """(478, 3) points spanning a normalized (x0, y0, x1, y1) box exactly."""
    x0, y0, x1, y1 = box
    u = np.linspace(0.0, 1.0, NUM_LANDMARKS)
    pts = np.zeros((NUM_LANDMARKS, 3), np.float32)
    pts[:, 0] = x0 + (x1 - x0) * u
    pts[:, 1] = y0 + (y1 - y0) * u[::-1]
    return pts


class FakeBackend:
    """Scripted faces per call: ``video[i]`` / ``image[i]`` list normalized boxes for frame i."""

    def __init__(self, video, image=None):
        self.video, self.image = video, image or {}
        self.video_calls, self.image_calls, self.shapes = [], [], []
        self.closed = False
        self._frame = -1

    def detect_video(self, image_bgr, timestamp_ms):
        self._frame += 1
        self.video_calls.append(timestamp_ms)
        self.shapes.append(image_bgr.shape)
        return [_landmarks(b) for b in self.video[self._frame]]

    def detect_image(self, image_bgr):
        self.image_calls.append(self._frame)
        return [_landmarks(b) for b in self.image.get(self._frame, [])]

    def close(self):
        self.closed = True


def _frame(w=200, h=100):
    return np.zeros((h, w, 3), np.uint8)


A = (0.10, 0.20, 0.30, 0.60)  # the subject: 40 x 40 px on a 200 x 100 frame
BIG_ELSEWHERE = (0.60, 0.10, 0.95, 0.90)


def test_select_initial_takes_largest_first_on_tie():
    assert select_initial([(0, 0, 10, 10), (5, 5, 40, 30), (0, 0, 30, 40)]) == 1
    assert select_initial([]) is None


def test_match_previous_nearest_within_radius_and_size():
    prev = (100, 100, 40, 40)  # diagonal 56.6, so radius 28.3 px
    near, nearer = (120, 100, 40, 40), (110, 105, 40, 40)
    assert match_previous([near, nearer], prev) == 1
    assert match_previous([(140, 100, 40, 40)], prev) is None  # 40 px away
    assert match_previous([(100, 100, 70, 70)], prev) is None  # 1.75x the size
    # While lost, the radius grows: 40 px is reachable after 3 lost frames (0.65 diag = 36.8)...
    assert match_previous([(140, 100, 40, 40)], prev, frames_lost=3) is None
    assert match_previous([(140, 100, 40, 40)], prev, frames_lost=5) == 0
    # ...and a bigger size change is allowed, but never past the caps.
    assert match_previous([(100, 100, 70, 70)], prev, frames_lost=1) == 0
    assert match_previous([(400, 100, 40, 40)], prev, frames_lost=1000) is None


def test_landmarks_bbox_clamps_to_frame():
    assert landmarks_bbox(_landmarks((0.1, 0.2, 0.3, 0.6)), 200, 100) == (20, 20, 40, 40)
    assert landmarks_bbox(_landmarks((-0.1, -0.2, 0.3, 1.4)), 200, 100) == (0, 0, 60, 100)


def test_keeps_first_face_when_a_bigger_one_appears():
    backend = FakeBackend([[A], [A, BIG_ELSEWHERE], [BIG_ELSEWHERE, A]])
    with FaceTracker(backend=backend) as tracker:
        boxes = [tracker.update(_frame(), i / 30).bbox for i in range(3)]
    assert boxes == [(20, 20, 40, 40)] * 3
    assert backend.closed


def test_initial_pick_is_largest_face():
    backend = FakeBackend([[A, BIG_ELSEWHERE]])
    with FaceTracker(backend=backend) as tracker:
        assert tracker.update(_frame(), 0.0).bbox == landmarks_bbox(_landmarks(BIG_ELSEWHERE), 200, 100)


def test_lost_face_is_not_replaced_by_another_face():
    moved = (0.30, 0.20, 0.50, 0.60)  # 40 px right of A: reachable only after 5 lost frames
    script = [[A], [BIG_ELSEWHERE], []] + [[moved]] * 5
    backend = FakeBackend(script, image={i: [BIG_ELSEWHERE] for i in range(8)})
    with FaceTracker(backend=backend) as tracker:
        out = [tracker.update(_frame(), i / 30) for i in range(8)]
    assert [o is not None for o in out] == [True, False, False, False, False, False, True, True]
    assert out[6].bbox == (60, 20, 40, 40)
    assert out[7].via == "video"


def test_dropout_is_reacquired_in_image_mode():
    backend = FakeBackend([[A], [], [A]], image={1: [A]})
    with FaceTracker(backend=backend) as tracker:
        out = [tracker.update(_frame(), i / 30) for i in range(3)]
    assert [o.via for o in out] == ["video", "reacquire", "video"]
    assert backend.image_calls == [1]


def test_reacquire_can_be_turned_off():
    backend = FakeBackend([[A], [], [A]], image={1: [A]})
    with FaceTracker(backend=backend, reacquire=False) as tracker:
        out = [tracker.update(_frame(), i / 30) for i in range(3)]
    assert out[1] is None and out[2].via == "video"
    assert backend.image_calls == []


def test_timestamps_start_at_zero_and_strictly_increase():
    backend = FakeBackend([[A]] * 5)
    with FaceTracker(backend=backend) as tracker:
        for t in (10.0, 10.0167, 10.0167, 10.0170, 10.05):  # duplicates and sub-ms steps
            tracker.update(_frame(), t)
    assert backend.video_calls == [0, 17, 18, 19, 50]


def test_detection_runs_on_a_proxy_but_boxes_are_in_source_pixels():
    backend = FakeBackend([[A]])
    with FaceTracker(backend=backend, detect_max_dim=100) as tracker:
        face = tracker.update(_frame(400, 200), 0.0)
    assert backend.shapes == [(50, 100, 3)]
    assert face.bbox == (40, 40, 80, 80)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        FaceTracker(backend=FakeBackend([]), detect_max_dim=10)
    bad = FakeBackend([[A]])
    bad.detect_video = lambda image, ts: [np.zeros((468, 3), np.float32)]
    with FaceTracker(backend=bad) as tracker:
        with pytest.raises(ValueError, match="H x W x 3"):
            tracker.update(np.zeros((10, 10), np.uint8), 0.0)
        with pytest.raises(ValueError, match="shape"):
            tracker.update(_frame(), 0.0)


def _load_harness():
    path = REPO / "scripts" / "review" / "video_qa_report.py"
    spec = importlib.util.spec_from_file_location("video_qa_report_s2", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _track():
    backend = FakeBackend([[A], [], [A]])
    with FaceTracker(backend=backend, reacquire=False) as tracker:
        faces = [f for f in (tracker.update(_frame(), i / 30) for i in range(3)) if f is not None]
    return VideoTrack(200, 100, Fraction(30), 3, faces)


def test_contract_is_accepted_by_the_v0_harness(tmp_path):
    payload = build_contract(_track(), "clip.mp4")
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    tracks = _load_harness().load_tracks(path)
    assert sorted(tracks) == [0, 2]
    assert (tracks[2].x, tracks[2].y, tracks[2].width, tracks[2].height) == (20, 20, 40, 40)

    loaded = load_contract(path)
    assert loaded["version"] == CONTRACT_VERSION and loaded["frame_count"] == 3
    assert np.asarray(loaded["frames"][1]["landmarks"]).shape == (NUM_LANDMARKS, 3)
    assert loaded["frames"][1]["time"] == pytest.approx(2 / 30, abs=1e-6)


def test_load_contract_validates(tmp_path):
    path = tmp_path / "tracks.json"
    v1 = {"frames": [{"frame": 0, "face": [1, 2, 3, 4], "confidence": 0.9}]}
    path.write_text(json.dumps(v1), encoding="utf-8")
    assert load_contract(path)["frames"][0]["face"] == [1, 2, 3, 4]

    for bad, message in (
        ({"version": 3, "frames": []}, "version"),
        ({"version": 2, "frames": [dict(v1["frames"][0], landmarks=[[0, 0, 0]])]}, "landmarks"),
        ({"frames": v1["frames"] * 2}, "duplicate"),
    ):
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_contract(path)


def test_track_video_reads_every_frame(tmp_path, monkeypatch):
    pytest.importorskip("av")
    from tests.video.test_media import H, W, _make_clip

    clip = _make_clip(tmp_path / "clip.mp4", [i * 1001 for i in range(6)])
    backend = FakeBackend([[A], [A], [], [A], [A], [A]])
    monkeypatch.setattr("retouch.video.tracker.MediaPipeBackend", lambda **kw: backend)

    seen = []
    track = track_video(clip, reacquire=False, progress=lambda n, t: seen.append((n, t)))
    assert (track.width, track.height, track.frame_count) == (W, H, 6)
    assert [f.frame for f in track.faces] == [0, 1, 3, 4, 5]
    assert track.coverage == pytest.approx(5 / 6)
    assert track.faces[3].time == pytest.approx(4 * 1001 / 30000)
    assert seen[-1] == (6, 5) and backend.closed

    track = track_video(clip, max_frames=2, backend=FakeBackend([[A]] * 6))
    assert track.frame_count == 2


def _model_available():
    try:
        import mediapipe  # noqa: F401

        from retouch.model_fetch import get_model_path

        get_model_path("face_landmarker")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="MediaPipe face landmarker model not available")
def test_real_landmarker_follows_the_subject_past_a_growing_second_face():
    import cv2

    sheet = cv2.imread(str(REPO / "docs" / "reference_targets" / "chang_e_cosplay_tamed_shine.jpg"))

    def tile(r, c):
        return cv2.resize(sheet[r * 480 : (r + 1) * 480, c * 360 : (c + 1) * 360], None, fx=2, fy=2,
                          interpolation=cv2.INTER_CUBIC)

    subject = tile(0, 1)[100:500, 160:480]
    other = tile(0, 2)[100:480, 180:480]
    frames = 40
    with FaceTracker() as tracker:
        out = []
        for i in range(frames):
            canvas = np.full((540, 960, 3), 110, np.uint8)
            x = 40 + 8 * i
            canvas[60:460, x : x + 320] = subject
            # The second face starts smaller and ends bigger than the subject.
            o = cv2.resize(other, None, fx=0.8 + 0.3 * i / (frames - 1), fy=0.8 + 0.3 * i / (frames - 1),
                           interpolation=cv2.INTER_AREA)
            canvas[540 - o.shape[0] :, 950 - o.shape[1] : 950] = o
            out.append(tracker.update(canvas, i / 30))

    assert all(face is not None for face in out)
    first = out[0].bbox
    for i, face in enumerate(out):
        assert abs(face.bbox[0] - (first[0] + 8 * i)) <= 12  # follows the 8 px/frame pan
        assert face.bbox[0] + face.bbox[2] < 700  # never the face on the right
