"""Tests for retouch.video.stabilize (V1 slice S3)."""

from __future__ import annotations

import numpy as np
import pytest

from retouch.video import stabilize as st
from retouch.video.stabilize import StabilizeParams

W, H, FPS = 1000, 800, 30.0
IED_PX = 100.0


def _base_landmarks(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lm = np.empty((478, 3))
    lm[:, 0] = rng.uniform(0.40, 0.60, 478)
    lm[:, 1] = rng.uniform(0.35, 0.65, 478)
    lm[:, 2] = rng.uniform(-0.02, 0.02, 478)
    lm[33] = (0.45, 0.45, 0.0)
    lm[263] = (0.55, 0.45, 0.0)  # IED = 100 px at W = 1000
    return lm


def _contract(landmarks_by_frame: dict, n: int) -> dict:
    frames = []
    for i in sorted(landmarks_by_frame):
        lm = landmarks_by_frame[i]
        x0, y0 = lm[:, 0].min() * W, lm[:, 1].min() * H
        x1, y1 = lm[:, 0].max() * W, lm[:, 1].max() * H
        frames.append(
            {
                "frame": i,
                "face": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)],
                "confidence": 1.0,
                "time": i / FPS,
                "via": "video",
                "landmarks": lm.tolist(),
            }
        )
    return {"version": 2, "width": W, "height": H, "fps": FPS, "frame_count": n, "frames": frames}


def _jitter_px(seq: np.ndarray, index: int = 1) -> float:
    pts = seq[:, index, :2]
    return float(np.mean(np.linalg.norm(pts[2:] - 2 * pts[1:-1] + pts[:-2], axis=1)))


# --------------------------------------------------------------------------
# One-Euro smoothing
# --------------------------------------------------------------------------


class TestSmoothing:
    def test_static_face_jitter_drops(self):
        rng = np.random.default_rng(1)
        n = 120
        times = np.arange(n) / FPS
        base = _base_landmarks()[None] * (W, H, W)
        noisy = base + rng.normal(0, 0.02 * IED_PX, (n, 478, 3))
        smooth = st.smooth_zero_lag(times, noisy, np.full(n, IED_PX), StabilizeParams())
        assert _jitter_px(smooth) < _jitter_px(noisy) / 4
        # and it stays on the face: mean error well under the noise level
        assert np.abs(smooth - base).mean() < 0.5 * np.abs(noisy - base).mean()

    def test_constant_motion_has_no_lag(self):
        n = 90
        times = np.arange(n) / FPS
        speed = 2.0 * IED_PX  # px / s
        base = _base_landmarks()[None] * (W, H, W)
        moving = base + np.zeros((n, 478, 3))
        moving[:, :, 0] += speed * times[:, None]
        params = StabilizeParams()
        causal = st.one_euro(times, moving, np.full(n, IED_PX), params)
        zero_lag = st.smooth_zero_lag(times, moving, np.full(n, IED_PX), params)
        mid = slice(20, 70)
        causal_lag = np.abs(causal[mid, :, 0] - moving[mid, :, 0]).mean()
        assert causal_lag > 0.02 * IED_PX  # the forward pass alone lags...
        assert np.abs(zero_lag[mid, :, 0] - moving[mid, :, 0]).mean() < 0.1 * causal_lag  # ...the pair doesn't

    def test_fast_turn_is_followed(self):
        # a 1-IED move in 0.2 s (a quick head turn) must not leave the face behind
        n = 60
        times = np.arange(n) / FPS
        offset = np.clip((times - 1.0) / 0.2, 0, 1) * IED_PX
        base = _base_landmarks()[None] * (W, H, W)
        moving = base + np.zeros((n, 478, 3))
        moving[:, :, 0] += offset[:, None]
        smooth = st.smooth_zero_lag(times, moving, np.full(n, IED_PX), StabilizeParams())
        assert np.abs(smooth[:, :, 0] - moving[:, :, 0]).max() < 0.15 * IED_PX

    def test_variable_frame_rate(self):
        rng = np.random.default_rng(2)
        times = np.cumsum(rng.uniform(1 / 60, 1 / 20, 100))
        base = _base_landmarks()[None] * (W, H, W)
        noisy = base + rng.normal(0, 0.02 * IED_PX, (100, 478, 3))
        smooth = st.smooth_zero_lag(times, noisy, np.full(100, IED_PX), StabilizeParams())
        assert np.isfinite(smooth).all()
        assert np.abs(smooth - base).mean() < 0.5 * np.abs(noisy - base).mean()


# --------------------------------------------------------------------------
# Cuts
# --------------------------------------------------------------------------


class TestCuts:
    def test_difference_spike_is_a_cut(self):
        rng = np.random.default_rng(3)
        times = np.arange(100) / FPS
        diffs = rng.uniform(1.0, 3.0, 100)
        diffs[0] = 0
        diffs[60] = 40.0
        assert st.find_cuts(times, diffs, StabilizeParams()) == [60]

    def test_steady_motion_is_not_a_cut(self):
        # a camera pan: every frame differs a lot, but evenly
        times = np.arange(100) / FPS
        diffs = np.full(100, 20.0)
        diffs[0] = 0
        assert st.find_cuts(times, diffs, StabilizeParams()) == []

    def test_small_spike_below_floor_is_not_a_cut(self):
        times = np.arange(100) / FPS
        diffs = np.full(100, 0.5)
        diffs[50] = 5.0  # 10x the level, but under the grey-level floor
        assert st.find_cuts(times, diffs, StabilizeParams()) == []

    def test_box_jump_is_a_cut(self):
        times = np.arange(10) / FPS
        boxes = {i: (100.0, 100.0, 50.0, 50.0) for i in range(5)}
        boxes.update({i: (600.0, 400.0, 50.0, 50.0) for i in range(5, 10)})
        assert st.find_cuts(times, np.zeros(10), StabilizeParams(), boxes) == [5]

    def test_box_jump_across_a_gap_is_not_a_cut(self):
        times = np.arange(20) / FPS
        boxes = {i: (100.0, 100.0, 50.0, 50.0) for i in range(5)}
        boxes.update({i: (600.0, 400.0, 50.0, 50.0) for i in range(15, 20)})
        assert st.find_cuts(times, np.zeros(20), StabilizeParams(), boxes) == []


# --------------------------------------------------------------------------
# Whole track: gaps, fades, shots
# --------------------------------------------------------------------------


class TestStabilize:
    def test_short_gap_is_filled(self):
        base = _base_landmarks()
        frames = {i: base + [0.001 * i, 0, 0] for i in range(30) if i not in (10, 11, 12)}
        track = st.stabilize(_contract(frames, 30))
        by = {f.frame: f for f in track.frames}
        assert sorted(by) == list(range(30))
        assert [by[i].source for i in (10, 11, 12)] == ["interpolated"] * 3
        assert all(f.weight == 1.0 for f in track.frames)
        # filled in between its neighbours
        assert by[9].landmarks[0, 0] < by[11].landmarks[0, 0] < by[13].landmarks[0, 0]

    def test_long_gap_fades_out_and_in(self):
        base = _base_landmarks()
        frames = {i: base for i in list(range(0, 30)) + list(range(60, 90))}
        track = st.stabilize(_contract(frames, 90))
        by = {f.frame: f for f in track.frames}
        assert 30 not in by and 59 not in by
        assert by[0].weight == 1.0 and by[89].weight == 1.0  # clip edges: no fade
        assert by[15].weight == 1.0 and by[75].weight == 1.0
        assert by[29].weight < 0.25 and by[60].weight < 0.25
        w_out = [by[i].weight for i in range(20, 30)]
        w_in = [by[i].weight for i in range(60, 70)]
        assert w_out == sorted(w_out, reverse=True) and w_in == sorted(w_in)

    def test_cut_resets_smoothing(self):
        base = _base_landmarks()
        frames = {i: base for i in range(20)}
        frames.update({i: base + [0.35, 0.25, 0] for i in range(20, 40)})
        contract = _contract(frames, 40)
        track = st.stabilize(contract)
        by = {f.frame: f for f in track.frames}
        assert track.cuts == [20]
        assert by[19].shot == 0 and by[20].shot == 1
        np.testing.assert_allclose(by[19].landmarks[:, :2], base[:, :2], atol=1e-5)
        np.testing.assert_allclose(by[20].landmarks[:, :2], (base + [0.35, 0.25, 0])[:, :2], atol=1e-5)
        assert by[20].weight == 1.0  # a cut is not a fade

    def test_without_evidence_every_region_is_visible(self):
        frames = {i: _base_landmarks() for i in range(10)}
        track = st.stabilize(_contract(frames, 10))
        assert all(v == 1.0 for f in track.frames for v in f.visibility.values())

    def test_contract_roundtrip(self):
        frames = {i: _base_landmarks() for i in range(5)}
        out = st.build_stable_contract(st.stabilize(_contract(frames, 5)), "clip.mp4")
        assert out["format"] == st.STABLE_FORMAT and out["regions"] == st.REGION_NAMES
        rec = out["frames"][0]
        assert np.asarray(rec["landmarks"]).shape == (478, 3)
        assert set(rec["visibility"]) == set(st.REGION_NAMES)


# --------------------------------------------------------------------------
# Covered parts
# --------------------------------------------------------------------------


def _skin_stats(n: int, rng, tone: float = 1.0) -> np.ndarray:
    """(n, R, 4) region stats of a steady face with realistic frame noise."""
    R = len(st.REGION_NAMES)
    lab = np.array([[65, 14, 16]] * R, float)
    lab[st.REGION_NAMES.index("lips")] = (52, 30, 14)
    lab[st.REGION_NAMES.index("eye_l")] = lab[st.REGION_NAMES.index("eye_r")] = (40, 6, 6)
    lab[st.REGION_NAMES.index("jaw_l")] = (58, 13, 15)  # shaded side
    tex = np.full(R, 2.0)
    tex[st.REGION_NAMES.index("eye_l")] = tex[st.REGION_NAMES.index("eye_r")] = 6.0
    stats = np.empty((n, R, 4))
    light = 1.0 + 0.04 * np.sin(np.arange(n) / 15.0)  # slow lighting drift, whole face
    stats[:, :, 0] = lab[None, :, 0] * light[:, None] * tone + rng.normal(0, 0.6, (n, R))
    stats[:, :, 1:3] = lab[None, :, 1:3] * tone + rng.normal(0, 0.5, (n, R, 2))
    stats[:, :, 3] = tex[None] * tone * np.exp(rng.normal(0, 0.05, (n, R)))
    return stats


def _evidence(stats: np.ndarray):
    return [st.FrameEvidence(i, i / FPS, 1.0 if i else 0.0, stats[i].astype(np.float32)) for i in range(len(stats))]


def _ball(stats: np.ndarray, frames, regions, tone: float = 1.0) -> None:
    """A smooth orange ball over some regions on some frames."""
    for name in regions:
        r = st.REGION_NAMES.index(name)
        stats[frames, r, :3] = np.array([70, 35, 60]) * tone
        stats[frames, r, 3] = 0.6 * tone


class TestCoveredParts:
    @pytest.mark.parametrize("tone", [1.0, 0.45])
    def test_ball_over_lower_face(self, tone):
        rng = np.random.default_rng(4)
        n = 200
        stats = _skin_stats(n, rng, tone)
        _ball(stats, slice(80, 140), ["lips", "chin", "nose"], tone)
        frames = {i: _base_landmarks() for i in range(n)}
        track = st.stabilize(_contract(frames, n), _evidence(stats))
        by = {f.frame: f for f in track.frames}
        for i in range(82, 138):
            assert by[i].visibility["lips"] == 0.0 and by[i].visibility["chin"] == 0.0
            assert by[i].visibility["nose"] == 0.0
            assert by[i].visibility["forehead"] == 1.0 and by[i].visibility["eye_l"] == 1.0
        for i in list(range(0, 60)) + list(range(165, 200)):
            assert all(v == 1.0 for v in by[i].visibility.values()), (i, by[i].visibility)
        # fades rather than pops
        lips = [by[i].visibility["lips"] for i in range(136, 165)]
        assert lips == sorted(lips) and lips[0] == 0.0 and lips[-1] == 1.0
        assert any(0.0 < v < 1.0 for v in lips)

    def test_ball_lifting_off_briefly_does_not_flash_the_edit(self):
        # the ball leaves the chin for a few frames: too short to fade fully
        # back in, so the chin stays hidden instead of half-showing the edit
        rng = np.random.default_rng(9)
        n = 200
        stats = _skin_stats(n, rng)
        _ball(stats, slice(60, 90), ["chin"])
        _ball(stats, slice(94, 130), ["chin"])
        frames = {i: _base_landmarks() for i in range(n)}
        track = st.stabilize(_contract(frames, n), _evidence(stats))
        by = {f.frame: f for f in track.frames}
        assert all(by[i].visibility["chin"] == 0.0 for i in range(62, 128))

    def test_steady_face_never_reads_covered(self):
        rng = np.random.default_rng(5)
        n = 300
        stats = _skin_stats(n, rng)
        frames = {i: _base_landmarks() for i in range(n)}
        track = st.stabilize(_contract(frames, n), _evidence(stats))
        assert all(v == 1.0 for f in track.frames for v in f.visibility.values())

    def test_lighting_change_on_the_whole_face_is_not_cover(self):
        # a light switched off half-way, with a white-balance shift: every
        # region changes together, so none of them reads covered
        rng = np.random.default_rng(8)
        n = 200
        stats = _skin_stats(n, rng)
        stats[100:, :, 0] *= 0.75
        stats[100:, :, 3] *= 0.75
        stats[100:, :, 2] += 4.0
        frames = {i: _base_landmarks() for i in range(n)}
        track = st.stabilize(_contract(frames, n), _evidence(stats))
        assert all(v == 1.0 for f in track.frames for v in f.visibility.values())

    def test_covered_reference_region_is_outvoted(self):
        # a hand over one cheek: that cheek reads covered, the rest stays clear
        rng = np.random.default_rng(6)
        n = 200
        stats = _skin_stats(n, rng)
        _ball(stats, slice(50, 100), ["cheek_l"])
        frames = {i: _base_landmarks() for i in range(n)}
        track = st.stabilize(_contract(frames, n), _evidence(stats))
        by = {f.frame: f for f in track.frames}
        assert by[75].visibility["cheek_l"] == 0.0
        assert all(v == 1.0 for k, v in by[75].visibility.items() if k != "cheek_l")

    def test_region_stats_on_a_drawn_face(self):
        # region_stats reads the right pixels: paint the lips hull red
        import cv2

        img = np.full((H, W, 3), (120, 150, 200), np.uint8)
        lm = _base_landmarks()
        pts = (lm[:, :2] * (W, H)).astype(np.int32)
        cv2.fillConvexPoly(img, cv2.convexHull(pts[st.REGIONS["eye_l"]]), (30, 30, 200))
        stats = st.region_stats(img, lm.astype(np.float32))
        background = cv2.cvtColor(img[:1, :1].astype(np.float32) / 255, cv2.COLOR_BGR2LAB)[0, 0]
        eye = st.REGION_NAMES.index("eye_l")
        assert stats[eye, 1] > background[1] + 20  # redder (a*)
        assert np.isfinite(stats).all()

    def test_one_frame_blip_is_ignored(self):
        # the detector's first fit can differ from the video-mode fits for a frame
        rng = np.random.default_rng(7)
        n = 120
        stats = _skin_stats(n, rng)
        _ball(stats, slice(0, 1), ["jaw_l"])
        _ball(stats, slice(60, 61), ["lips"])
        frames = {i: _base_landmarks() for i in range(n)}
        track = st.stabilize(_contract(frames, n), _evidence(stats))
        assert all(v == 1.0 for f in track.frames for v in f.visibility.values())


def _model_available() -> bool:
    try:
        import mediapipe  # noqa: F401

        from retouch.model_fetch import get_model_path

        get_model_path("face_landmarker")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _model_available(), reason="MediaPipe face models not available")
def test_real_models_mark_a_ball_over_the_mouth():
    """A ball held over mouth and chin (like clip 042): lower face covered, eyes and forehead not."""
    from fractions import Fraction
    from pathlib import Path

    import cv2

    from retouch.video.media import Frame
    from retouch.video.tracker import FaceTracker, VideoTrack, build_contract

    sheet = cv2.imread(str(Path(__file__).resolve().parents[2] / "docs" / "reference_targets" / "chang_e_cosplay_tamed_shine.jpg"))
    subject = cv2.resize(sheet[0:480, 360:720], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)[100:500, 160:480]
    rng = np.random.default_rng(0)
    n, ball = 60, range(25, 40)
    frames, faces, first = [], [], None
    with FaceTracker() as tracker:
        for i in range(n):
            canvas = np.full((540, 960, 3), 110, np.uint8)
            canvas[60:460, 200 + 2 * i : 520 + 2 * i] = subject
            canvas = np.clip(canvas + rng.normal(0, 2, canvas.shape), 0, 255).astype(np.uint8)
            if i in ball:
                px = first[:, :2] * (960, 540) + (2 * i, 0)
                centre = (px[13] + px[152]) / 2  # between the lips and the chin
                radius = 0.75 * np.linalg.norm(px[33] - px[263])
                cv2.circle(canvas, tuple(int(v) for v in centre), int(radius), (40, 120, 230), -1)
            face = tracker.update(canvas, i / 30)
            if i == 0:
                first = face.landmarks.copy()
            if face is not None:
                faces.append(face)
            frames.append(Frame(i, i, Fraction(1, 30), canvas))

    contract = build_contract(VideoTrack(960, 540, Fraction(30), n, faces), "synthetic")
    evidence = st.gather_evidence(frames, {f.frame: f.landmarks for f in faces})
    by = {f.frame: f for f in st.stabilize(contract, evidence).frames}

    assert sorted(by) == list(range(n))  # short losses under the ball are filled
    for i in range(27, 38):
        assert by[i].visibility["lips"] == 0.0 and by[i].visibility["chin"] == 0.0, i
    for f in by.values():
        assert f.visibility["forehead"] == 1.0 and f.visibility["eye_l"] == 1.0, f.frame
    for i in list(range(0, 12)) + list(range(52, 60)):
        assert all(v == 1.0 for v in by[i].visibility.values()), (i, by[i].visibility)
