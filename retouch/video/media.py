"""Streaming video decode/encode with audio passthrough (V1 slice S1).

Frames are decoded one at a time, so memory stays flat however long the clip
is. Every frame comes out upright (the container's rotation is applied to the
pixels) as a BGR ``uint8`` array, the same layout the image engine uses, and
keeps its source presentation timestamp. The writer re-uses those timestamps,
so variable frame rate footage keeps its timing, and copies the source audio
packets untouched, so audio is sample-exact and stays in sync.

Two output modes:

* ``lossless=False``: H.264 (libx264, ``crf``), yuv420p, tagged BT.709, for
  delivery. Use a ``.mp4`` path.
* ``lossless=True``: FFV1 in RGB (``bgr0``), bit-exact to the frames written,
  for QA measurements that a lossy intermediate would swamp. Use a ``.mkv``
  path.

Needs the optional ``video`` extra (PyAV, which bundles its own FFmpeg)::

    pip install 'retouch-engine[video]'
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction
from pathlib import Path
from typing import Iterator, Optional, Union

import cv2
import numpy as np

try:
    import av
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "retouch.video needs PyAV; install the video extra: pip install 'retouch-engine[video]'"
    ) from exc

PathLike = Union[str, Path]


@dataclasses.dataclass(frozen=True)
class VideoInfo:
    """What a clip looks like once decoded: upright size, timing and audio."""

    width: int  # upright (after rotation)
    height: int
    fps: Fraction  # average frame rate
    duration: float  # seconds, from the video stream
    frame_count: int  # from the container; 0 when it doesn't say
    rotation: int  # degrees applied to make frames upright: 0, 90, 180 or 270
    has_audio: bool
    audio_codec: Optional[str]


@dataclasses.dataclass(frozen=True)
class Frame:
    """One decoded frame: upright BGR pixels plus its source timing."""

    index: int
    pts: int  # presentation timestamp in ``time_base`` units
    time_base: Fraction
    image: np.ndarray  # H x W x 3, uint8, BGR, upright

    @property
    def time(self) -> float:
        return float(self.pts * self.time_base)


def _rotation(stream) -> int:
    """Clockwise degrees to turn stored pixels upright, from the display matrix."""
    for frame in stream.container.decode(stream):
        # PyAV reports the display matrix angle counter-clockwise.
        return int(round(-frame.rotation)) % 360
    return 0


# YUV <-> RGB is done here in numpy with exact coefficients: swscale's default path (the only
# one PyAV exposes) is 3-5 levels off on flat colours in each direction, and PyAV also ignores
# src_colorspace on frames that carry a colorspace tag.
# AVColorSpace -> (Kr, Kb). Untagged video is BT.601, as ffmpeg assumes.
_KR_KB = {
    1: (0.2126, 0.0722),  # BT.709
    4: (0.30, 0.11),  # FCC
    5: (0.299, 0.114),  # BT.470BG / BT.601
    6: (0.299, 0.114),  # SMPTE 170M / BT.601
    7: (0.212, 0.087),  # SMPTE 240M
    9: (0.2627, 0.0593),  # BT.2020 non-constant
    10: (0.2627, 0.0593),  # BT.2020 constant (approximated as non-constant)
}
_BT601 = _KR_KB[5]
_AVCOL_RANGE_JPEG = 2
_YUV_FORMATS = {"yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "yuv444p", "yuvj444p",
                "yuv420p10le", "yuv422p10le", "yuv444p10le"}


def _plane(frame, i: int, bits: int) -> np.ndarray:
    plane = frame.planes[i]
    dtype = np.dtype("<u2" if bits > 8 else np.uint8)
    rows = np.frombuffer(plane, dtype).reshape(plane.height, plane.line_size // dtype.itemsize)
    return rows[:, : plane.width].astype(np.float32)


def _to_bgr(frame) -> np.ndarray:
    """Decode a frame to BGR uint8 with the YUV matrix and range its stream is tagged with."""
    name = frame.format.name
    if name not in _YUV_FORMATS:
        return frame.to_ndarray(format="bgr24")  # RGB formats (e.g. lossless FFV1) convert exactly
    bits = 10 if name.endswith("10le") else 8
    scale = float(1 << (bits - 8))
    full = name.startswith("yuvj") or frame.color_range == _AVCOL_RANGE_JPEG
    kr, kb = _KR_KB.get(frame.colorspace, _BT601)
    y, cb, cr = (_plane(frame, i, bits) / scale for i in range(3))
    height, width = y.shape
    if cb.shape != y.shape:
        cb = cv2.resize(cb, (width, height), interpolation=cv2.INTER_LINEAR)
        cr = cv2.resize(cr, (width, height), interpolation=cv2.INTER_LINEAR)
    if full:
        y, cb, cr = y / 255.0, (cb - 128.0) / 255.0, (cr - 128.0) / 255.0
    else:
        y, cb, cr = (y - 16.0) / 219.0, (cb - 128.0) / 224.0, (cr - 128.0) / 224.0
    kg = 1.0 - kr - kb
    r = y + 2.0 * (1.0 - kr) * cr
    b = y + 2.0 * (1.0 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    bgr = np.stack([b, g, r], axis=-1) * 255.0
    return np.clip(np.rint(bgr), 0, 255).astype(np.uint8)


def _to_yuv420p_bt709(image: np.ndarray):
    """BGR uint8 -> a limited-range BT.709 yuv420p VideoFrame, chroma averaged over 2x2."""
    kr, kb = _KR_KB[1]
    bgr = image.astype(np.float32) / 255.0
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    y = kr * r + (1.0 - kr - kb) * g + kb * b
    cb = (b - y) / (2.0 * (1.0 - kb))
    cr = (r - y) / (2.0 * (1.0 - kr))
    height, width = y.shape
    half = (width // 2, height // 2)
    cb = cv2.resize(cb, half, interpolation=cv2.INTER_AREA)
    cr = cv2.resize(cr, half, interpolation=cv2.INTER_AREA)
    planes = np.concatenate([
        (16.0 + 219.0 * y).ravel(),
        (128.0 + 224.0 * cb).ravel(),
        (128.0 + 224.0 * cr).ravel(),
    ])
    packed = np.clip(np.rint(planes), 0, 255).astype(np.uint8).reshape(height * 3 // 2, width)
    return av.VideoFrame.from_ndarray(packed, format="yuv420p")


def _upright(pixels: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return pixels
    # np.rot90 turns counter-clockwise for positive k.
    return np.ascontiguousarray(np.rot90(pixels, k=-(rotation // 90)))


def probe(path: PathLike) -> VideoInfo:
    """Read a clip's upright size, frame rate, duration, rotation and audio."""
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream")
        stream = container.streams.video[0]
        rotation = _rotation(stream)
        width, height = stream.codec_context.width, stream.codec_context.height
        if rotation in (90, 270):
            width, height = height, width
        if stream.duration is not None:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = float(container.duration or 0) / av.time_base
        audio = container.streams.audio[0] if container.streams.audio else None
        return VideoInfo(
            width=width,
            height=height,
            fps=Fraction(stream.average_rate or stream.guessed_rate or 0),
            duration=duration,
            frame_count=stream.frames,
            rotation=rotation,
            has_audio=audio is not None,
            audio_codec=audio.codec_context.name if audio is not None else None,
        )


def read_frames(path: PathLike) -> Iterator[Frame]:
    """Decode a clip frame by frame, upright, in presentation order."""
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rotation = None
        index = 0
        for frame in container.decode(stream):
            if rotation is None:
                rotation = int(round(-frame.rotation)) % 360
            pixels = _to_bgr(frame)
            yield Frame(
                index=index,
                pts=frame.pts,
                time_base=Fraction(frame.time_base),
                image=_upright(pixels, rotation),
            )
            index += 1


class VideoWriter:
    """Encode upright BGR frames and copy the source clip's audio alongside.

    Frames must be written in the order ``read_frames`` produced them; each
    keeps its source timestamp. Audio packets are interleaved as the video
    advances and the rest are flushed on ``close``::

        with VideoWriter("out.mp4", "in.mp4") as writer:
            for frame in read_frames("in.mp4"):
                writer.write(frame, retouch(frame.image))
    """

    def __init__(
        self,
        output: PathLike,
        source: PathLike,
        *,
        lossless: bool = False,
        crf: int = 18,
        audio: bool = True,
    ) -> None:
        self._source = av.open(str(source))
        src_video = self._source.streams.video[0]
        self._time_base = Fraction(src_video.time_base)
        self._out = av.open(str(output), "w")
        self._lossless = lossless
        if lossless:
            self._video = self._out.add_stream("ffv1", rate=src_video.average_rate)
            self._video.pix_fmt = "bgr0"
        else:
            self._video = self._out.add_stream(
                "libx264", rate=src_video.average_rate, options={"crf": str(crf), "preset": "medium"}
            )
            self._video.pix_fmt = "yuv420p"
            ctx = self._video.codec_context
            ctx.colorspace = 1  # AVCOL_SPC_BT709
            ctx.color_primaries = 1  # AVCOL_PRI_BT709
            ctx.color_trc = 1  # AVCOL_TRC_BT709
            ctx.color_range = 1  # AVCOL_RANGE_MPEG
        self._video.time_base = self._time_base
        self._video.codec_context.time_base = self._time_base

        src_audio = self._source.streams.audio[0] if audio and self._source.streams.audio else None
        self._audio_out = self._out.add_stream_from_template(src_audio) if src_audio is not None else None
        self._packets = self._source.demux(src_audio) if src_audio is not None else iter(())
        self._pending = None  # next audio packet not yet muxed
        self._size: Optional[tuple] = None

    def _mux_audio_until(self, seconds: Optional[float]) -> None:
        """Copy source audio packets that start at or before ``seconds`` (all if None)."""
        if self._audio_out is None:
            return
        while True:
            packet = self._pending if self._pending is not None else next(self._packets, None)
            self._pending = None
            if packet is None:
                return
            if packet.dts is None:  # demuxer flush packet
                continue
            if seconds is not None and packet.pts is not None and float(packet.pts * packet.time_base) > seconds:
                self._pending = packet
                return
            packet.stream = self._audio_out
            self._out.mux(packet)

    def write(self, frame: Frame, image: np.ndarray) -> None:
        """Encode ``image`` at ``frame``'s timestamp."""
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected an H x W x 3 uint8 BGR image, got {image.dtype} {image.shape}")
        height, width = image.shape[:2]
        if self._size is None:
            if not self._lossless and (width % 2 or height % 2):
                raise ValueError(f"H.264 yuv420p needs even dimensions, got {width}x{height}")
            self._video.width, self._video.height = width, height
            self._size = (width, height)
        elif (width, height) != self._size:
            raise ValueError(f"frame size changed from {self._size} to {(width, height)}")
        if self._lossless:
            out = av.VideoFrame.from_ndarray(image, format="bgr24")
        else:
            out = _to_yuv420p_bt709(image)
        out.pts = frame.pts
        out.time_base = self._time_base
        for packet in self._video.encode(out):
            self._out.mux(packet)
        self._mux_audio_until(frame.time)

    def close(self) -> None:
        for packet in self._video.encode():
            self._out.mux(packet)
        self._mux_audio_until(None)
        self._out.close()
        self._source.close()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
