"""S1 media I/O: generated A/V clips round-trip with correct duration, sync and rotation.

Fixtures are generated with PyAV itself (no system ffmpeg, no committed video).
"""

import struct
from fractions import Fraction

import numpy as np
import pytest

av = pytest.importorskip("av")

from retouch.video.media import VideoWriter, probe, read_frames  # noqa: E402

W, H = 96, 64
RATE = Fraction(30000, 1001)
SAMPLE_RATE = 48000


def _pattern(i: int) -> np.ndarray:
    """Flat colour blocks (lossless under 4:2:0) that change per frame, with a red marker top-left."""
    img = np.zeros((H, W, 3), np.uint8)
    img[:, : W // 2] = (40, 90 + i % 50, 160)
    img[:, W // 2 :] = (170, 120, 60 + i % 50)
    img[:16, :16] = (0, 0, 255)
    return img


def _make_clip(path, pts, *, time_base=Fraction(1, 30000), audio_seconds=None, lossless=False):
    """Write frames at the given pts (time_base units) plus an optional 440 Hz AAC track."""
    with av.open(str(path), "w") as out:
        video = out.add_stream("ffv1" if lossless else "libx264", rate=RATE)
        video.width, video.height = W, H
        video.pix_fmt = "bgr0" if lossless else "yuv420p"
        video.time_base = time_base
        video.codec_context.time_base = time_base
        audio = None
        if audio_seconds:
            audio = out.add_stream("aac", rate=SAMPLE_RATE)
            audio.layout = "mono"
        for i, p in enumerate(pts):
            frame = av.VideoFrame.from_ndarray(_pattern(i), format="bgr24")
            frame.pts = p
            frame.time_base = time_base
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        if audio is not None:
            n = int(audio_seconds * SAMPLE_RATE)
            tone = (0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / SAMPLE_RATE)).astype(np.float32)
            for start in range(0, n, 1024):
                chunk = av.AudioFrame.from_ndarray(tone[None, start : start + 1024], format="flt", layout="mono")
                chunk.sample_rate = SAMPLE_RATE
                chunk.pts = start
                for packet in audio.encode(chunk):
                    out.mux(packet)
            for packet in audio.encode():
                out.mux(packet)
    return path


def _set_display_rotation(path, matrix):
    """Patch the first track's tkhd matrix, as `ffmpeg -display_rotation` writes it."""
    data = bytearray(path.read_bytes())
    i = data.index(b"tkhd")
    version = data[i + 4]
    offset = i + 4 + 4 + (32 if version == 1 else 20) + 8 + 8
    data[offset : offset + 36] = struct.pack(">9i", *matrix)
    path.write_bytes(bytes(data))


def _audio_packets(path):
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        return [
            (bytes(p), float(p.pts * p.time_base)) for p in container.demux(stream) if p.dts is not None
        ]


def _first_audio_time(path):
    return _audio_packets(path)[0][1]


CFR = [round(i * 1001) for i in range(60)]  # 60 frames at 29.97 in 1/30000
VFR = [i * 1000 for i in range(30)] + [30000 + i * 2000 for i in range(15)]  # 30 fps for 1 s, then 15 fps


@pytest.fixture
def av_clip(tmp_path):
    return _make_clip(tmp_path / "av.mp4", CFR, audio_seconds=2.0)


def test_probe_reports_size_rate_duration_and_audio(av_clip):
    info = probe(av_clip)
    assert (info.width, info.height) == (W, H)
    assert info.fps == RATE
    assert info.frame_count == 60
    assert info.duration == pytest.approx(60 * 1001 / 30000, abs=1e-6)
    assert info.rotation == 0
    assert info.has_audio and info.audio_codec == "aac"


def test_frames_stream_in_order_with_source_timestamps(av_clip):
    frames = list(read_frames(av_clip))
    assert [f.index for f in frames] == list(range(60))
    assert [f.pts for f in frames] == CFR
    assert frames[1].time == pytest.approx(1001 / 30000)
    assert frames[0].image.shape == (H, W, 3) and frames[0].image.dtype == np.uint8


@pytest.mark.parametrize("matrix, rotation, marker", [
    # ffmpeg -display_rotation 90 (counter-clockwise display): stored top-left ends bottom-left
    ((0, -65536, 0, 65536, 0, 0, 0, 0, 1 << 30), 270, "bottom_left"),
    # ffmpeg -display_rotation -90, a phone held upright: stored top-left ends top-right
    ((0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30), 90, "top_right"),
    ((-65536, 0, 0, 0, -65536, 0, 0, 0, 1 << 30), 180, "bottom_right"),
])
def test_rotated_clip_decodes_upright(tmp_path, matrix, rotation, marker):
    clip = _make_clip(tmp_path / "rot.mp4", CFR[:5])
    _set_display_rotation(clip, matrix)
    info = probe(clip)
    image = next(read_frames(clip)).image
    expected = (H, W) if rotation == 180 else (W, H)
    assert info.rotation == rotation
    assert (info.height, info.width) == expected and image.shape[:2] == expected
    h, w = image.shape[:2]
    corners = {
        "top_left": image[:8, :8], "top_right": image[:8, w - 8 :],
        "bottom_left": image[h - 8 :, :8], "bottom_right": image[h - 8 :, w - 8 :],
    }
    red = {name: bool((block[..., 2] > 200).all() and (block[..., 0] < 50).all()) for name, block in corners.items()}
    assert red == {name: name == marker for name in corners}


def test_rotated_clip_round_trips_upright_without_rotation_tag(tmp_path):
    clip = _make_clip(tmp_path / "rot.mp4", CFR[:10], audio_seconds=0.5)
    _set_display_rotation(clip, (0, 65536, 0, -65536, 0, 0, 0, 0, 1 << 30))
    out = tmp_path / "out.mp4"
    frames = list(read_frames(clip))
    with VideoWriter(out, clip) as writer:
        for frame in frames:
            writer.write(frame, frame.image)
    info = probe(out)
    assert info.rotation == 0
    assert (info.width, info.height) == (H, W)
    diff = np.abs(next(read_frames(out)).image.astype(int) - frames[0].image)
    assert np.percentile(diff, 90) <= 1  # only the colour-block edges lose chroma detail under 4:2:0


def test_h264_round_trip_keeps_frames_timing_and_audio(av_clip, tmp_path):
    out = tmp_path / "out.mp4"
    frames = list(read_frames(av_clip))
    with VideoWriter(out, av_clip) as writer:
        for frame in frames:
            writer.write(frame, frame.image)
    back = list(read_frames(out))
    assert [f.time for f in back] == pytest.approx([f.time for f in frames], abs=1e-9)
    assert probe(out).duration == pytest.approx(probe(av_clip).duration, abs=1e-6)
    # Flat blocks: H.264 at crf 18 stays within a few levels, away from block edges.
    for a, b in zip(frames, back):
        inner = np.s_[20:44, 20:28]
        assert np.abs(a.image[inner].astype(int) - b.image[inner]).max() <= 3
    # Audio packets copied byte for byte, at the same times.
    assert _audio_packets(out) == _audio_packets(av_clip)


def test_output_is_tagged_bt709(av_clip, tmp_path):
    out = tmp_path / "out.mp4"
    with VideoWriter(out, av_clip) as writer:
        for frame in read_frames(av_clip):
            writer.write(frame, frame.image)
    with av.open(str(out)) as container:
        ctx = container.streams.video[0].codec_context
        assert (ctx.colorspace, ctx.color_primaries, ctx.color_trc) == (1, 1, 1)


def test_lossless_is_bit_exact_and_keeps_av_offset(av_clip, tmp_path):
    out = tmp_path / "out.mkv"
    frames = list(read_frames(av_clip))
    rng = np.random.default_rng(0)
    written = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in frames]  # noise defeats any lossy codec
    with VideoWriter(out, av_clip, lossless=True) as writer:
        for frame, image in zip(frames, written):
            writer.write(frame, image)
    back = list(read_frames(out))
    assert len(back) == len(frames)
    assert all(np.array_equal(a, b.image) for a, b in zip(written, back))
    # Matroska can't store the AAC priming delay as a negative time, so it shifts the whole file
    # by it: frame spacing and the audio/video offset stay the same.
    shift = back[0].time - frames[0].time
    assert [f.time - shift for f in back] == pytest.approx([f.time for f in frames], abs=1e-3)
    assert _first_audio_time(out) - back[0].time == pytest.approx(_first_audio_time(av_clip) - frames[0].time, abs=1e-3)
    assert [p for p, _ in _audio_packets(out)] == [p for p, _ in _audio_packets(av_clip)]


def test_variable_frame_rate_timing_survives(tmp_path):
    clip = _make_clip(tmp_path / "vfr.mp4", VFR, audio_seconds=2.0)
    out = tmp_path / "out.mp4"
    frames = list(read_frames(clip))
    with VideoWriter(out, clip) as writer:
        for frame in frames:
            writer.write(frame, frame.image)
    assert [f.pts * f.time_base for f in read_frames(out)] == [Fraction(p, 30000) for p in VFR]


def test_audio_can_be_dropped_and_silent_sources_work(av_clip, tmp_path):
    silent = _make_clip(tmp_path / "silent.mp4", CFR[:10])
    for source, kwargs in ((av_clip, {"audio": False}), (silent, {})):
        out = tmp_path / f"out_{source.stem}.mp4"
        with VideoWriter(out, source, **kwargs) as writer:
            for frame in read_frames(source):
                writer.write(frame, frame.image)
        assert not probe(out).has_audio


def test_writer_rejects_bad_frames(av_clip, tmp_path):
    frame = next(read_frames(av_clip))
    with VideoWriter(tmp_path / "out.mp4", av_clip) as writer:
        with pytest.raises(ValueError, match="uint8 BGR"):
            writer.write(frame, frame.image.astype(np.float32))
        with pytest.raises(ValueError, match="even dimensions"):
            writer.write(frame, np.zeros((63, 96, 3), np.uint8))
        writer.write(frame, frame.image)
        with pytest.raises(ValueError, match="frame size changed"):
            writer.write(frame, np.zeros((32, 32, 3), np.uint8))


def test_probe_rejects_audio_only_files(tmp_path):
    path = tmp_path / "audio.m4a"
    with av.open(str(path), "w") as out:
        audio = out.add_stream("aac", rate=SAMPLE_RATE)
        audio.layout = "mono"
        chunk = av.AudioFrame.from_ndarray(np.zeros((1, 1024), np.float32), format="flt", layout="mono")
        chunk.sample_rate = SAMPLE_RATE
        chunk.pts = 0
        for packet in list(audio.encode(chunk)) + list(audio.encode()):
            out.mux(packet)
    with pytest.raises(ValueError, match="no video stream"):
        probe(path)
    with pytest.raises(ValueError, match="no video stream"):
        next(read_frames(path))
