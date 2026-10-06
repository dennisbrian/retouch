"""Selected-face render adapter (V1 slice S4).

Turns each frame of a stable track (S3) into one retouched frame. The engine
is reused unchanged: for every frame the adapter builds the ``FaceContext``
the engine would have built itself and passes it in, so the engine skips
detection and parsing and works on the tracked face only.

Per frame:

* **Crop.** The engine's own face region (the box plus its padding for
  forehead, neck and sides) is cut out of the frame and processed on its own.
  Everything outside it is copied from the source untouched, so other faces
  (a printed banner, a passer-by) stay byte-identical, and a 4K frame never
  reaches the engine's proxy path (which re-detects and re-parses). When the
  region is longer than the engine's proxy limit, the neck padding is trimmed
  first, then the sides; only a face too big even for that is downscaled.
* **Face parts.** The parser (BiSeNet, or the landmark + segmenter fallback)
  runs on keyframes only: every ``parse_every_s`` seconds, at each new shot,
  and never on a frame where S3 marks a part covered or the track is fading
  (masks from a frame with a ball over the mouth would follow the face
  afterwards). Between keyframes the keyframe masks are carried to the
  current frame by a piecewise-affine warp over the smoothed landmarks, so
  they move with the face, mouth opening included. After a new keyframe the
  masks cross-fade from the old keyframe to the new one over one interval,
  so a re-parse never pops.
* **Retouch.** ``engine.process(crop, face_contexts=[ctx], **params)`` with a
  fixed video-safe parameter set (:data:`VIDEO_PARAMS`): the ``natural``
  recipe with every global stage and every per-frame automatic choice off.
* **Fade.** The change is applied as ``source + (retouched - source) * w``.
  ``w`` is the frame weight from S3 (fades around track losses), limited to
  the face parts the parser found (grown and feathered, so the engine's
  float round-trip cannot shimmer the background), lowered inside each face
  part S3 marks covered, and tapered to 0 at the crop border so nothing the
  engine does near the edge leaves a seam.

Usage::

    python -m retouch.video.adapter clip.mp4 stable.json --out retouched.mp4
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from .stabilize import REGIONS, STABLE_FORMAT, STABLE_VERSION, StableFrame, StableTrack

PathLike = Union[str, Path]

# The engine's proxy limit (engine.PROXY_MAX_DIM). Crops at or under it are
# processed at native resolution with the cached face context honoured.
MAX_CROP_DIM = 2048

# The video-safe parameter set: the ``natural`` recipe (light frequency
# separation smoothing, skin evening, small eye/lip/hair touches) with the
# stages that would change the whole crop, or choose something anew on every
# frame, switched off. Growth beyond this set is V2's job (plan S4).
VIDEO_RECIPE = "natural"
VIDEO_PARAMS: Dict[str, Any] = {
    "auto_exposure": False,
    "safe_auto": False,
}

_EYE_CORNERS = (33, 263)


@dataclasses.dataclass(frozen=True)
class AdapterParams:
    # Seconds between parser runs (keyframes). The plan's V0 cadence bake-off
    # has not run; 0.2 s is every 6 frames at 30 fps, every 12 at 60, inside
    # the 4-8 frame range the research note proposes for 30 fps.
    parse_every_s: float = 0.2
    # Feather of the covered-part fade, in inter-eye distances.
    cover_feather_ied: float = 0.15
    # Width of the taper to the crop border, as a fraction of the crop's
    # short side.
    border_taper: float = 0.04

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be finite and positive")


# --------------------------------------------------------------------------
# Stable-track JSON
# --------------------------------------------------------------------------


def load_stable_track(path: PathLike) -> StableTrack:
    """Read the JSON written by ``python -m retouch.video.stabilize``."""
    data = json.loads(Path(path).read_text())
    if data.get("format") != STABLE_FORMAT or data.get("version") != STABLE_VERSION:
        raise ValueError(f"{path}: not a {STABLE_FORMAT} v{STABLE_VERSION} file")
    width, height = int(data["width"]), int(data["height"])
    frames = []
    for r in data["frames"]:
        landmarks = np.asarray(r["landmarks"], np.float32)
        if landmarks.shape != (478, 3) or not np.isfinite(landmarks).all():
            raise ValueError(f"frame {r.get('frame')}: expected finite 478 x 3 landmarks")
        frames.append(
            StableFrame(
                frame=int(r["frame"]),
                time=float(r["time"]),
                shot=int(r["shot"]),
                source=str(r["source"]),
                weight=float(r["weight"]),
                bbox=tuple(int(v) for v in r["face"]),
                landmarks=landmarks,
                visibility={k: float(v) for k, v in r["visibility"].items()},
            )
        )
    return StableTrack(width, height, float(data["fps"]), int(data["frame_count"]), list(data["cuts"]), frames)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def roi_padding(face_w: int, face_h: int) -> Tuple[int, int, int, int]:
    """(top, bottom, left, right): the engine's per-face ROI padding."""
    from ..engine import RetouchEngine

    return RetouchEngine._compute_face_roi_padding(face_w, face_h)


def crop_box(bbox: Sequence[int], width: int, height: int, max_dim: int = MAX_CROP_DIM) -> Tuple[int, int, int, int]:
    """(x0, y0, x1, y1): the engine's face ROI, clipped to the frame and to max_dim.

    Over max_dim the neck padding goes first (the face and forehead are what
    the face ops need), then both sides evenly. A face too big even then
    returns a box over max_dim; the caller downscales it.
    """
    x, y, w, h = (int(v) for v in bbox)
    top, bottom, left, right = roi_padding(w, h)
    x0, y0 = max(0, x - left), max(0, y - top)
    x1, y1 = min(width, x + w + right), min(height, y + h + bottom)
    if y1 - y0 > max_dim:
        y1 = max(min(y + h, height), y0 + max_dim)
    if x1 - x0 > max_dim:
        excess = x1 - x0 - max_dim
        x0, x1 = x0 + excess // 2, x1 - (excess - excess // 2)
        x0, x1 = min(x0, max(0, x)), max(x1, min(width, x + w))
    return x0, y0, x1, y1


def _anchor_points(w: int, h: int) -> np.ndarray:
    """Points around and outside a w x h canvas so every pixel lies in the mesh."""
    xs = np.linspace(-0.25 * w, 1.25 * w, 7)
    ys = np.linspace(-0.25 * h, 1.25 * h, 7)
    ring = [(x, ys[0]) for x in xs] + [(x, ys[-1]) for x in xs]
    ring += [(xs[0], y) for y in ys[1:-1]] + [(xs[-1], y) for y in ys[1:-1]]
    return np.array(ring, np.float64)


def _triangles(points: np.ndarray, w: int, h: int) -> np.ndarray:
    lo = points.min(0) - 1
    hi = points.max(0) + 1
    subdiv = cv2.Subdiv2D((int(math.floor(lo[0])), int(math.floor(lo[1])), int(math.ceil(hi[0] - lo[0])) + 1, int(math.ceil(hi[1] - lo[1])) + 1))
    index = {}
    for i, p in enumerate(points):
        key = (float(p[0]), float(p[1]))
        if key in index:
            continue
        index[key] = i
        subdiv.insert(key)
    tris = []
    for t in subdiv.getTriangleList():
        ids = [index.get((float(t[0]), float(t[1]))), index.get((float(t[2]), float(t[3]))), index.get((float(t[4]), float(t[5])))]
        if None not in ids:
            tris.append(ids)
    return np.array(tris, np.int64)


def warp_map(src_pts: np.ndarray, dst_pts: np.ndarray, dst_shape: Tuple[int, int], src_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """remap() maps taking an image whose landmarks are src_pts to dst_pts.

    Piecewise affine over a Delaunay mesh of the destination landmarks plus
    an outer ring of anchors; the anchors follow the similarity transform
    that best fits the landmarks, so pixels away from the face (hair, neck)
    move with the head as a whole.
    """
    h, w = dst_shape
    sh, sw = src_shape
    sim, _ = cv2.estimateAffinePartial2D(dst_pts.astype(np.float32), src_pts.astype(np.float32), method=cv2.LMEDS)
    if sim is None:
        sim = np.array([[1, 0, 0], [0, 1, 0]], np.float64)
    anchors_dst = _anchor_points(w, h)
    anchors_src = anchors_dst @ sim[:, :2].T + sim[:, 2]
    dst = np.vstack([dst_pts, anchors_dst])
    src = np.vstack([src_pts, anchors_src])
    tris = _triangles(dst, w, h)
    label = np.full((h, w), -1, np.int32)
    affines = np.zeros((len(tris) + 1, 2, 3), np.float64)
    affines[-1] = sim  # pixels the mesh misses fall back to the similarity
    for k, t in enumerate(tris):
        a = cv2.getAffineTransform(dst[t].astype(np.float32), src[t].astype(np.float32))
        affines[k] = a
        cv2.fillConvexPoly(label, np.round(dst[t]).astype(np.int32), int(k), lineType=cv2.LINE_8)
    label[label < 0] = len(tris)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    A = affines[label].astype(np.float32)
    map_x = A[..., 0, 0] * xx + A[..., 0, 1] * yy + A[..., 0, 2]
    map_y = A[..., 1, 0] * xx + A[..., 1, 1] * yy + A[..., 1, 2]
    return map_x, map_y


def warp_regions(regions, maps: Tuple[np.ndarray, np.ndarray]):
    """A copy of FaceRegions with every mask remapped (eye-gate cache cleared)."""
    out = copy.copy(regions)
    for name in type(regions).__slots__:
        value = getattr(regions, name, None)
        if isinstance(value, np.ndarray) and value.ndim == 2:
            warped = cv2.remap(value, maps[0], maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            setattr(out, name, warped.astype(value.dtype, copy=False))
    out._eye_gate_cache = None
    return out


def blend_regions(a, b, t: float):
    """(1 - t) * a + t * b mask by mask; masks one side lacks come from the other."""
    if t <= 0:
        return a
    if t >= 1:
        return b
    out = copy.copy(b)
    for name in type(b).__slots__:
        va, vb = getattr(a, name, None), getattr(b, name, None)
        if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray) and va.shape == vb.shape and va.ndim == 2:
            mixed = (1 - t) * va.astype(np.float32) + t * vb.astype(np.float32)
            setattr(out, name, mixed.astype(vb.dtype, copy=False))
    out._eye_gate_cache = None
    return out


# --------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------


@dataclasses.dataclass
class _Keyframe:
    shot: int
    time: float
    points: np.ndarray  # landmarks in its crop's pixels (478, 2)
    shape: Tuple[int, int]
    regions: Any
    light_direction: Any


class FaceRenderer:
    """Renders one frame at a time; frames must arrive in order."""

    def __init__(
        self,
        engine=None,
        params: AdapterParams = AdapterParams(),
        recipe: str = VIDEO_RECIPE,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        if engine is None:
            from ..engine import RetouchEngine

            engine = RetouchEngine()
        self.engine = engine
        self.params = params
        self.recipe = recipe
        self.render_kwargs = dict(VIDEO_PARAMS)
        self.render_kwargs.update(overrides or {})
        self.mask_feather_mode = self.render_kwargs.get("mask_feather_mode", "gaussian")
        self._current: Optional[_Keyframe] = None
        self._previous: Optional[_Keyframe] = None
        self.stats = {"frames": 0, "rendered": 0, "parsed": 0, "downscaled": 0, "seconds": 0.0}

    # -- per-frame pieces ---------------------------------------------------

    def _face_context(self, crop: np.ndarray, pts: np.ndarray, z: np.ndarray, regions, light):
        from ..detection import FaceContext, FaceData, _Landmark, _LandmarkCompat

        h, w = crop.shape[:2]
        compat = _LandmarkCompat(
            [_Landmark(float(x) / w, float(y) / h, float(zz)) for (x, y), zz in zip(pts, z)]
        )
        x0, y0 = np.floor(pts.min(0)).astype(int)
        x1, y1 = np.ceil(pts.max(0)).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
        ied = float(np.linalg.norm(pts[_EYE_CORNERS[0]] - pts[_EYE_CORNERS[1]]))
        data = FaceData(
            landmarks=compat,
            bbox=(int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))),
            ied=ied,
            confidence=1.0,
            confidence_source="video_track",
        )
        return FaceContext(
            face_data=data,
            regions=regions,
            index=0,
            face_image=None,
            light_direction=light,
            frame_size=(w, h),
            mask_feather_mode=self.mask_feather_mode,
        ), compat, ied

    def _parse(self, crop: np.ndarray, compat, bbox, ied: float):
        from ..lighting import estimate_light_direction

        engine = self.engine
        person = engine._detector.segment_person(crop)
        regions = engine._parser.parse_batch(
            [crop], [compat], [bbox], [person], [ied], mask_feather_mode=self.mask_feather_mode
        )[0]
        light = estimate_light_direction(crop, regions, face_width=bbox[2])
        self.stats["parsed"] += 1
        return regions, light

    def _wants_keyframe(self, sf: StableFrame) -> bool:
        if self._current is None or self._current.shot != sf.shot:
            return True
        due = sf.time - self._current.time >= self.params.parse_every_s - 1e-6
        clear = sf.weight >= 1.0 and all(v >= 1.0 for v in sf.visibility.values())
        return due and clear

    def _regions_for(self, sf: StableFrame, crop, pts, compat, bbox, ied):
        shape = crop.shape[:2]
        if self._wants_keyframe(sf):
            regions, light = self._parse(crop, compat, bbox, ied)
            new = _Keyframe(sf.shot, sf.time, pts.copy(), shape, regions, light)
            same_shot = self._current is not None and self._current.shot == sf.shot
            self._previous = self._current if same_shot else None
            self._current = new
        cur = self._current
        regions = cur.regions
        if cur.time != sf.time or cur.shape != shape:
            regions = warp_regions(cur.regions, warp_map(cur.points, pts, shape, cur.shape))
        prev = self._previous
        if prev is not None:
            t = (sf.time - cur.time) / self.params.parse_every_s
            if t < 1:
                old = warp_regions(prev.regions, warp_map(prev.points, pts, shape, prev.shape))
                regions = blend_regions(old, regions, t)
            else:
                self._previous = None
        return regions, cur.light_direction

    def fade_map(self, sf: StableFrame, pts: np.ndarray, shape: Tuple[int, int], ied: float, regions=None) -> np.ndarray:
        """Per-pixel share of the retouch to keep, 0..1, over the crop."""
        h, w = shape
        weight = np.full((h, w), float(sf.weight), np.float32)
        sigma = max(1.0, self.params.cover_feather_ied * ied)
        if regions is not None:
            support = np.zeros((h, w), np.float32)
            for name in type(regions).__slots__:
                m = getattr(regions, name, None)
                if isinstance(m, np.ndarray) and m.shape == (h, w):
                    m = m.astype(np.float32)
                    if m.max() > 1.5:
                        m = m / 255.0
                    np.maximum(support, np.clip(m, 0, 1), out=support)
            k = int(2 * round(sigma) + 1)
            support = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            weight *= cv2.GaussianBlur(support, (0, 0), sigma)
        for name, idx in REGIONS.items():
            v = float(sf.visibility.get(name, 1.0))
            if v >= 1.0:
                continue
            hull = np.zeros((h, w), np.float32)
            cv2.fillConvexPoly(hull, cv2.convexHull(np.round(pts[list(idx)]).astype(np.int32)), 1.0)
            # grow by the feather so the covered part itself gets the full fade
            k = int(2 * round(sigma) + 1)
            hull = cv2.dilate(hull, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            hull = cv2.GaussianBlur(hull, (0, 0), sigma)
            weight *= 1.0 - (1.0 - v) * hull
        taper = max(1.0, self.params.border_taper * min(h, w))
        ys = np.minimum(np.arange(h), np.arange(h)[::-1]).astype(np.float32)
        xs = np.minimum(np.arange(w), np.arange(w)[::-1]).astype(np.float32)
        edge = np.minimum(ys[:, None], xs[None, :])
        weight *= np.clip(edge / taper, 0.0, 1.0)
        return weight

    # -- public -------------------------------------------------------------

    def render(self, image: np.ndarray, sf: Optional[StableFrame]) -> np.ndarray:
        """The retouched frame (a new array); the source frame when there is no face."""
        self.stats["frames"] += 1
        if sf is None or sf.weight <= 0:
            return image.copy()
        t0 = time.perf_counter()
        height, width = image.shape[:2]
        x0, y0, x1, y1 = crop_box(sf.bbox, width, height)
        crop = image[y0:y1, x0:x1]
        scale = 1.0
        if max(crop.shape[:2]) > MAX_CROP_DIM:
            scale = MAX_CROP_DIM / float(max(crop.shape[:2]))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            self.stats["downscaled"] += 1
        else:
            crop = crop.copy()
        lm = sf.landmarks.astype(np.float64)
        pts = (lm[:, :2] * (width, height) - (x0, y0)) * scale
        ctx, compat, ied = self._face_context(crop, pts, lm[:, 2], None, None)
        regions, light = self._regions_for(sf, crop, pts, compat, ctx.face_data.bbox, ied)
        ctx.regions, ctx.light_direction = regions, light

        out = np.asarray(self.engine.process(crop, recipe=self.recipe, face_contexts=[ctx], **self.render_kwargs))
        if out.shape != crop.shape:
            raise RuntimeError(f"engine returned {out.shape} for a {crop.shape} crop")
        keep = self.fade_map(sf, pts, crop.shape[:2], ied, regions)[..., None]
        delta = (out.astype(np.float32) - crop.astype(np.float32)) * keep
        if scale != 1.0:
            delta = cv2.resize(delta, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
            base = image[y0:y1, x0:x1].astype(np.float32)
        else:
            base = crop.astype(np.float32)
        result = image.copy()
        result[y0:y1, x0:x1] = np.clip(np.round(base + delta), 0, 255).astype(image.dtype)
        self.stats["rendered"] += 1
        self.stats["seconds"] += time.perf_counter() - t0
        return result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def render_video(video: PathLike, track: StableTrack, out: PathLike, renderer: FaceRenderer, *, lossless: bool = False, progress=None) -> dict:
    from .media import VideoWriter, read_frames

    by_frame = {f.frame: f for f in track.frames}
    with VideoWriter(out, video, lossless=lossless) as writer:
        for frame in read_frames(video):
            if frame.image.shape[:2] != (track.height, track.width):
                raise ValueError("stable track dimensions do not match the decoded video")
            writer.write(frame, renderer.render(frame.image, by_frame.get(frame.index)))
            if progress is not None:
                progress(renderer.stats)
    return dict(renderer.stats)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Retouch the tracked face of a clip (selected face only).")
    parser.add_argument("video", type=Path, help="Local source clip (read only)")
    parser.add_argument("stable", type=Path, help="Stable track JSON from retouch.video.stabilize")
    parser.add_argument("--out", type=Path, required=True, help="Output video (.mp4 H.264, or .mkv with --lossless)")
    parser.add_argument("--lossless", action="store_true", help="FFV1 output for QA")
    parser.add_argument("--parse-every", type=float, default=AdapterParams.parse_every_s, help="Seconds between face-part parses")
    parser.add_argument("--recipe", default=VIDEO_RECIPE)
    args = parser.parse_args(argv)

    sources = {args.video.resolve(), args.stable.resolve()}
    if args.out.resolve() in sources:
        parser.error("--out must differ from the inputs")
    track = load_stable_track(args.stable)
    renderer = FaceRenderer(params=AdapterParams(parse_every_s=args.parse_every), recipe=args.recipe)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def report(stats):
        if stats["frames"] % 30 == 0:
            per = stats["seconds"] / max(stats["rendered"], 1)
            print(f"  frame {stats['frames']}: {stats['rendered']} retouched, {stats['parsed']} parses, {per:.2f} s/frame", flush=True)

    stats = render_video(args.video, track, args.out, renderer, lossless=args.lossless, progress=report)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
