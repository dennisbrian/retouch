"""Temporal stabilization of a face track (V1 slice S3).

Takes the per-frame track from :mod:`retouch.video.tracker` and returns, for
every frame of the clip, what the retouch stage needs to keep edits steady:

* **Smoothed landmarks and box.** A One-Euro filter run forward and backward
  over each shot and averaged, so it has no lag. Its cutoff rises with the
  face's speed, measured in inter-eye distances (IED) per second, so slow
  drift is smoothed hard and fast moves are followed. Time-based, so variable
  frame rate is fine.
* **Scene cuts.** A spike in the frame-to-frame difference of a small grey
  thumbnail (relative to the clip's own recent level), or the tracked box
  jumping more than a box diagonal between consecutive frames. Smoothing,
  gap filling and covered-part baselines restart at each cut.
* **Gaps.** Untracked runs up to ``max_gap_s`` inside a shot are filled by
  interpolation. Around longer gaps the frame ``weight`` (0..1, how much of
  the retouch to apply) fades out before the face is lost and back in after
  it returns, instead of popping.
* **Covered parts.** The tracker keeps placing nose, lips and chin landmarks
  on a ball held in front of the face, at full confidence (042, frames
  226-327). Each landmark region's colour and texture are compared with the
  same region over the rest of the shot, after removing the change every
  region shares on that frame (exposure, white balance, shade). A region
  that changes on its own, beyond its own usual spread, is marked covered,
  with hysteresis, and its ``visibility`` fades to 0 so later slices can
  skip the edits that would land on the occluder. Every measure is relative
  to the same face in the same clip, so skin tone and exposure cancel.

Usage::

    python -m retouch.video.stabilize clip.mp4 tracks.json --out stable.json \
        [--overlay review.mp4]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np

from .tracker import NUM_LANDMARKS, load_contract

PathLike = Union[str, Path]

STABLE_FORMAT = "retouch_stable_track"
STABLE_VERSION = 1

# Landmark regions whose visibility is reported. Small convex hulls inside
# the face; LEFT/RIGHT are camera-viewer side, as in retouch/parsing.py.
REGIONS: Dict[str, List[int]] = {
    "forehead": [103, 67, 109, 10, 338, 297, 332, 333, 299, 337, 151, 108, 69, 104],
    "cheek_l": [117, 118, 101, 36, 205, 187, 123, 116],
    "cheek_r": [346, 347, 330, 266, 425, 411, 352, 345],
    "eye_l": [33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7],
    "eye_r": [263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249],
    "nose": [168, 6, 197, 195, 5, 4, 1, 2, 98, 327, 294, 278, 344, 440, 275, 45, 220, 115, 48, 64],
    "lips": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185],
    "chin": [18, 83, 313, 175, 152, 148, 377, 176, 400, 201, 421],
    "jaw_l": [132, 58, 172, 136, 207, 214, 192],
    "jaw_r": [361, 288, 397, 365, 427, 434, 416],
}
REGION_NAMES = list(REGIONS)
_EYE_CORNERS = (33, 263)
_THUMB = (64, 36)
# Regions that set the change shared by the whole face on a frame.
_COMMON_REGIONS = 3
_MIN_REGION_PX = 9


@dataclasses.dataclass(frozen=True)
class StabilizeParams:
    """Tunable constants. Units are seconds, IED and the clip's own spreads."""

    # One-Euro: cutoff (Hz) = min_cutoff + beta * speed (IED / s).
    min_cutoff: float = 1.0
    beta: float = 1.5
    d_cutoff: float = 1.0
    # Gaps inside a shot up to this long are interpolated; longer ones fade.
    max_gap_s: float = 0.2
    fade_s: float = 0.2
    # Scene cut: thumbnail difference this many times its recent median
    # (window in seconds) and at least the floor (grey levels, 0-255).
    cut_ratio: float = 4.0
    cut_window_s: float = 1.0
    cut_floor: float = 8.0
    # Box centre jump between consecutive tracked frames, in box diagonals.
    cut_jump: float = 1.0
    # Covered parts: deviation from the region's own normal in units of its
    # spread over the shot. Covered above z_on, clear again below z_off.
    z_on: float = 6.0
    z_off: float = 3.0
    # Spread floors: colour difference (CIELab, L 0-100) and log texture ratio.
    colour_floor: float = 1.5
    texture_floor: float = 0.08
    # Covered spans shorter than this are ignored (a one-frame landmark blip);
    # the rest are widened by cover_margin_s each side and faded over fade_s.
    min_cover_s: float = 0.05
    cover_margin_s: float = 0.1


# --------------------------------------------------------------------------
# Per-frame evidence (needs the decoded frame)
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FrameEvidence:
    """What one decoded frame says, beyond the tracker's landmarks."""

    frame: int
    time: float
    diff: float  # mean |thumbnail - previous thumbnail|, grey levels; 0 on frame 0
    # (len(REGIONS), 4): median L, a, b (CIELab, L 0-100) and mean |Laplacian of L|
    # inside each region; NaN rows where the face is untracked or a region is tiny.
    regions: Optional[np.ndarray]


def thumbnail(image: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(grey, _THUMB, interpolation=cv2.INTER_AREA).astype(np.float32)


def region_stats(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Median Lab and texture inside each landmark region (rows in REGIONS order)."""
    h, w = image.shape[:2]
    pts = landmarks[:, :2].astype(np.float32) * np.array([w, h], np.float32)
    x0, y0 = np.maximum(np.floor(pts.min(0)).astype(int) - 4, 0)
    x1 = min(int(math.ceil(float(pts[:, 0].max()))) + 4, w)
    y1 = min(int(math.ceil(float(pts[:, 1].max()))) + 4, h)
    out = np.full((len(REGIONS), 4), np.nan, np.float32)
    if x1 - x0 < 3 or y1 - y0 < 3:
        return out
    crop = image[y0:y1, x0:x1].astype(np.float32) / 255.0
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    lap = np.abs(cv2.Laplacian(lab[..., 0], cv2.CV_32F, ksize=3))
    local = pts - np.array([x0, y0], np.float32)
    for r, name in enumerate(REGION_NAMES):
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, cv2.convexHull(local[REGIONS[name]].astype(np.int32)), 1)
        sel = mask.astype(bool)
        if int(sel.sum()) < _MIN_REGION_PX:
            continue
        out[r, :3] = np.median(lab[sel], axis=0)
        out[r, 3] = float(lap[sel].mean())
    return out


def gather_evidence(
    frames: Iterable,
    landmarks_by_frame: Dict[int, np.ndarray],
    *,
    progress: Optional[Callable[[int], None]] = None,
) -> List[FrameEvidence]:
    """Evidence for every frame of ``frames`` (``media.Frame`` objects, in order)."""
    out: List[FrameEvidence] = []
    previous = None
    for frame in frames:
        thumb = thumbnail(frame.image)
        diff = 0.0 if previous is None else float(np.mean(np.abs(thumb - previous)))
        previous = thumb
        landmarks = landmarks_by_frame.get(frame.index)
        stats = None if landmarks is None else region_stats(frame.image, landmarks)
        out.append(FrameEvidence(frame.index, frame.time, diff, stats))
        if progress is not None:
            progress(len(out))
    return out


# --------------------------------------------------------------------------
# Offline pieces (pure numpy, no frames needed)
# --------------------------------------------------------------------------


def _alpha(cutoff: np.ndarray, dt: float) -> np.ndarray:
    tau = 1.0 / (2.0 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


def one_euro(
    times: np.ndarray, values: np.ndarray, scale: np.ndarray, params: StabilizeParams
) -> np.ndarray:
    """Causal One-Euro filter over (T, ...) ``values``; one cutoff per frame.

    Speed is the mean absolute filtered velocity over all values divided by
    ``scale[t]`` (the face's IED), so every landmark gets the same cutoff and
    the face keeps its shape.
    """
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    velocity = np.zeros_like(values[0], dtype=np.float64)
    for t in range(1, len(values)):
        dt = max(float(times[t] - times[t - 1]), 1e-6)
        raw_velocity = (values[t] - out[t - 1]) / dt
        a_d = _alpha(np.float64(params.d_cutoff), dt)
        velocity = velocity + a_d * (raw_velocity - velocity)
        speed = float(np.mean(np.abs(velocity))) / max(float(scale[t]), 1e-6)
        a = _alpha(np.float64(params.min_cutoff + params.beta * speed), dt)
        out[t] = out[t - 1] + a * (values[t] - out[t - 1])
    return out


def smooth_zero_lag(
    times: np.ndarray, values: np.ndarray, scale: np.ndarray, params: StabilizeParams
) -> np.ndarray:
    """Average of a forward and a backward One-Euro pass: no net lag."""
    if len(values) < 2:
        return values.astype(np.float64)
    forward = one_euro(times, values, scale, params)
    backward = one_euro(-times[::-1], values[::-1], scale[::-1], params)[::-1]
    return 0.5 * (forward + backward)


def find_cuts(
    times: np.ndarray,
    diffs: np.ndarray,
    params: StabilizeParams,
    boxes: Optional[Dict[int, Tuple[float, float, float, float]]] = None,
) -> List[int]:
    """Frame indices that start a new shot (0 is implied, not listed).

    ``diffs[t]`` compares frame t with frame t-1. A cut is a difference at
    least ``cut_ratio`` times the median over the previous ``cut_window_s``
    and above ``cut_floor``; or a tracked box whose centre moved more than
    ``cut_jump`` diagonals since the previous frame (consecutive frames only).
    """
    cuts = set()
    for t in range(1, len(diffs)):
        start = max(1, int(np.searchsorted(times, times[t] - params.cut_window_s)))
        window = diffs[start:t]
        if not len(window):
            continue
        level = float(np.median(window))
        if diffs[t] >= params.cut_floor and diffs[t] >= params.cut_ratio * max(level, 1e-6):
            cuts.add(t)
    if boxes:
        for t in sorted(boxes):
            if t - 1 in boxes:
                (x0, y0, w0, h0), (x1, y1, w1, h1) = boxes[t - 1], boxes[t]
                jump = math.hypot(x1 + w1 / 2 - x0 - w0 / 2, y1 + h1 / 2 - y0 - h0 / 2)
                if jump > params.cut_jump * math.hypot(w0, h0):
                    cuts.add(t)
    return sorted(cuts)


def _robust_spread(x: np.ndarray, floor: float) -> float:
    med = np.median(x)
    return max(1.4826 * float(np.median(np.abs(x - med))), floor)


def region_deviation(stats: np.ndarray, params: StabilizeParams) -> np.ndarray:
    """(T, R) deviation of each region from its own normal, in spread units.

    ``stats`` is (T, R, 4) for one shot (NaN where unmeasured). Each region is
    compared with its own median over the shot: lightness as a log ratio
    (x50, about L* units at mid tones), a* and b* as differences, texture as a
    log ratio. What all regions share on a frame (the median change over the
    regions that look clear: exposure, white balance, a head turn into shade)
    is removed, so only a region that changes on its own scores. The shared
    change is the median over the three clear regions that changed least, so
    it holds while most of the face is covered. Spreads are
    each region's own over the shot. Measured three times so frames that
    look covered don't set the normal or the shared change.
    """
    T, R = stats.shape[:2]
    with np.errstate(all="ignore"):
        x = np.stack(
            [
                50.0 * np.log(np.maximum(stats[..., 0], 1e-3)),
                stats[..., 1],
                stats[..., 2],
                np.log(np.maximum(stats[..., 3], 1e-3)),
            ],
            axis=-1,
        )
    ok = np.isfinite(stats).all(-1)  # (T, R)
    keep = ok.copy()
    score = np.full((T, R), np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN slices (untracked, all covered)
        for _ in range(3):
            masked = np.where(keep[..., None], x, np.nan)
            d = x - np.nanmedian(masked, axis=0)  # (T, R, 4)
            # shared change: median over the few clear regions that changed least
            size = np.where(keep, np.linalg.norm(d[..., :3], axis=-1), np.inf)
            calmest = np.argsort(size, axis=1)[:, :_COMMON_REGIONS]
            picked = np.take_along_axis(np.where(keep[..., None], d, np.nan), calmest[..., None], axis=1)
            common = np.nanmedian(picked, axis=1)  # (T, 4)
            e = d - np.nan_to_num(common)[:, None, :]
            dev_c = np.linalg.norm(e[..., :3], axis=-1)
            dev_t = np.abs(e[..., 3])
            for r in range(R):
                k = keep[:, r]
                if k.sum() < 3:
                    continue
                score[:, r] = np.maximum(
                    dev_c[:, r] / _robust_spread(dev_c[k, r], params.colour_floor),
                    dev_t[:, r] / _robust_spread(dev_t[k, r], params.texture_floor),
                )
            calm = ok & (np.nan_to_num(score, nan=np.inf) < params.z_on)
            if (calm.sum(0) < 3).any():
                break
            keep = calm
    return np.where(ok, score, np.nan)


def _ramp(mask: np.ndarray, times: np.ndarray, margin_s: float, fade_s: float) -> np.ndarray:
    """1 inside ``mask``, widened by margin_s, then falling to 0 over fade_s."""
    on = np.flatnonzero(mask)
    if not len(on):
        return np.zeros(len(times))
    # distance in seconds from each frame to the nearest masked frame
    dist = np.min(np.abs(times[:, None] - times[on][None, :]), axis=1)
    return np.clip(1.0 - (dist - margin_s) / max(fade_s, 1e-6), 0.0, 1.0)


def hysteresis(z: np.ndarray, on: float, off: float) -> np.ndarray:
    """True from a frame above ``on`` until z drops below ``off``; NaN keeps state."""
    state = False
    out = np.zeros(len(z), bool)
    for t, v in enumerate(z):
        if np.isfinite(v):
            if v > on:
                state = True
            elif v < off:
                state = False
        out[t] = state
    return out


# --------------------------------------------------------------------------
# Whole-track stabilization
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class StableFrame:
    frame: int
    time: float
    shot: int
    source: str  # "tracked" or "interpolated"
    weight: float  # 0..1: how much of the retouch to apply on this frame
    bbox: Tuple[int, int, int, int]
    landmarks: np.ndarray  # (478, 3) float32, x, y normalized to the frame
    visibility: Dict[str, float]  # region -> 0 (covered) .. 1 (clear)


@dataclasses.dataclass(frozen=True)
class StableTrack:
    width: int
    height: int
    fps: float
    frame_count: int
    cuts: List[int]
    frames: List[StableFrame]  # only frames with a face (tracked or filled)


def _segments(n: int, cuts: Sequence[int]) -> List[Tuple[int, int]]:
    edges = [0] + [c for c in cuts if 0 < c < n] + [n]
    return [(a, b) for a, b in zip(edges, edges[1:]) if b > a]


def _runs(indices: np.ndarray) -> List[Tuple[int, int]]:
    """Consecutive runs of sorted ints as (first, last)."""
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[splits + 1]]
    ends = np.r_[indices[splits], indices[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def stabilize(
    contract: dict,
    evidence: Optional[Sequence[FrameEvidence]] = None,
    params: StabilizeParams = StabilizeParams(),
) -> StableTrack:
    """Stabilize a version-2 track contract.

    Without ``evidence`` there is no cut detection beyond box jumps and every
    region is reported visible; frame times come from the records and fps.
    """
    width, height = int(contract["width"]), int(contract["height"])
    fps = float(contract.get("fps") or 30.0)
    records = {int(r["frame"]): r for r in contract["frames"]}
    n = int(contract.get("frame_count") or (max(records) + 1 if records else 0))
    if evidence is not None:
        n = max(n, len(evidence))

    times = np.arange(n, dtype=np.float64) / fps
    for i, r in records.items():
        if "time" in r and i < n:
            times[i] = float(r["time"])
    diffs = np.zeros(n)
    stats = np.full((n, len(REGIONS), 4), np.nan)
    if evidence is not None:
        for e in evidence:
            if e.frame < n:
                times[e.frame] = e.time
                diffs[e.frame] = e.diff
                if e.regions is not None:
                    stats[e.frame] = e.regions

    boxes = {i: tuple(map(float, r["face"])) for i, r in records.items()}
    cuts = find_cuts(times, diffs, params, boxes)
    scale_px = np.array([width, height, width], np.float64)

    out: List[StableFrame] = []
    for shot, (a, b) in enumerate(_segments(n, cuts)):
        idx = np.array(sorted(i for i in records if a <= i < b), int)
        if not len(idx):
            continue
        pts = np.stack([np.asarray(records[i]["landmarks"], np.float64) for i in idx]) * scale_px
        ied = np.linalg.norm(pts[:, _EYE_CORNERS[0], :2] - pts[:, _EYE_CORNERS[1], :2], axis=1)

        # Group tracked frames into runs separated by long gaps; fill short gaps.
        runs: List[List[int]] = []
        for first, last in _runs(idx):
            if runs and times[first] - times[runs[-1][-1]] <= params.max_gap_s + 1e-9:
                runs[-1].extend(range(runs[-1][-1] + 1, last + 1))
            else:
                runs.append(list(range(first, last + 1)))

        z = region_deviation(stats[a:b], params)
        visibility = np.ones((b - a, len(REGIONS)))
        local_t = times[a:b]
        for r in range(len(REGIONS)):
            covered = hysteresis(z[:, r], params.z_on, params.z_off)
            for first, last in _runs(np.flatnonzero(covered)):
                if local_t[last] - local_t[first] < params.min_cover_s:
                    covered[first : last + 1] = False
            visibility[:, r] = 1.0 - _ramp(covered, local_t, params.cover_margin_s, params.fade_s)

        pos = {int(i): k for k, i in enumerate(idx)}
        for run in runs:
            run = np.array(run)
            have = np.array([pos[i] for i in run if i in pos])
            have_frames = idx[have]
            # interpolate missing frames of the run in time
            filled = np.empty((len(run), NUM_LANDMARKS, 3))
            for c in range(3):
                flat = pts[have][:, :, c]
                for k in range(NUM_LANDMARKS):
                    filled[:, k, c] = np.interp(times[run], times[have_frames], flat[:, k])
            run_ied = np.interp(times[run], times[have_frames], ied[have])
            smooth = smooth_zero_lag(times[run], filled, run_ied, params)

            # fade only where the run borders a long gap, not a cut or clip edge
            t_run = times[run]
            weight = np.ones(len(run))
            if run[0] > a:
                weight = np.minimum(weight, np.clip((t_run - t_run[0] + 1 / fps) / params.fade_s, 0, 1))
            if run[-1] < b - 1:
                weight = np.minimum(weight, np.clip((t_run[-1] - t_run + 1 / fps) / params.fade_s, 0, 1))

            for k, i in enumerate(run):
                lm = (smooth[k] / scale_px).astype(np.float32)
                x0, y0 = lm[:, 0].min() * width, lm[:, 1].min() * height
                x1, y1 = lm[:, 0].max() * width, lm[:, 1].max() * height
                x0, y0 = max(0, round(x0)), max(0, round(y0))
                x1, y1 = min(width, round(x1)), min(height, round(y1))
                out.append(
                    StableFrame(
                        frame=int(i),
                        time=float(times[i]),
                        shot=shot,
                        source="tracked" if i in pos else "interpolated",
                        weight=float(weight[k]),
                        bbox=(int(x0), int(y0), int(max(1, x1 - x0)), int(max(1, y1 - y0))),
                        landmarks=lm,
                        visibility={
                            name: float(visibility[i - a, r]) for r, name in enumerate(REGION_NAMES)
                        },
                    )
                )
    return StableTrack(width, height, fps, n, list(cuts), out)


def build_stable_contract(track: StableTrack, source: str, *, decimals: int = 5) -> dict:
    return {
        "format": STABLE_FORMAT,
        "version": STABLE_VERSION,
        "source": source,
        "width": track.width,
        "height": track.height,
        "fps": track.fps,
        "frame_count": track.frame_count,
        "cuts": track.cuts,
        "regions": REGION_NAMES,
        "frames": [
            {
                "frame": f.frame,
                "time": round(f.time, 6),
                "shot": f.shot,
                "source": f.source,
                "weight": round(f.weight, 4),
                "face": list(f.bbox),
                "visibility": {k: round(v, 3) for k, v in f.visibility.items()},
                "landmarks": np.round(f.landmarks.astype(np.float64), decimals).tolist(),
            }
            for f in track.frames
        ],
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def landmark_jitter(frames: Sequence[np.ndarray], ied: Sequence[float], index: int) -> np.ndarray:
    """|second difference| of one landmark per frame, in IED (consecutive frames only)."""
    pts = np.stack([f[index, :2] for f in frames])
    acc = np.linalg.norm(pts[2:] - 2 * pts[1:-1] + pts[:-2], axis=1)
    return acc / np.maximum(np.asarray(ied)[1:-1], 1e-6)


def _draw_overlay(image: np.ndarray, raw: Optional[np.ndarray], stable: Optional[StableFrame]) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    radius = max(1, round(max(w, h) / 900))
    if raw is not None:
        for x, y in raw[:, :2] * (w, h):
            cv2.circle(out, (int(x), int(y)), radius, (0, 0, 255), -1)
    if stable is not None:
        pts = stable.landmarks[:, :2] * (w, h)
        for name in REGION_NAMES:
            v = stable.visibility[name]
            colour = (0, int(255 * v), int(255 * (1 - v)))
            hull = cv2.convexHull(pts[REGIONS[name]].astype(np.int32))
            cv2.polylines(out, [hull], True, colour, radius)
        for x, y in pts:
            cv2.circle(out, (int(x), int(y)), radius, (0, 255, 0), -1)
        label = f"w={stable.weight:.2f} shot={stable.shot} {stable.source}"
        covered = [n for n in REGION_NAMES if stable.visibility[n] < 0.5]
        if covered:
            label += " covered: " + ",".join(covered)
        cv2.putText(out, label, (20, 40 * radius), cv2.FONT_HERSHEY_SIMPLEX, radius, (255, 255, 255), 2 * radius)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    from .media import read_frames

    parser = argparse.ArgumentParser(description="Stabilize a face track and mark covered face parts.")
    parser.add_argument("video", type=Path, help="Local source clip (read only)")
    parser.add_argument("tracks", type=Path, help="Track JSON from retouch.video.tracker")
    parser.add_argument("--out", type=Path, required=True, help="Stable track JSON output path")
    parser.add_argument("--overlay", type=Path, help="Also write a review video: raw (red) vs stable (green)")
    args = parser.parse_args(argv)

    contract = load_contract(args.tracks)
    if int(contract.get("version", 1)) < 2:
        raise SystemExit("track JSON needs landmarks (version 2)")
    by_frame = {int(r["frame"]): np.asarray(r["landmarks"], np.float32) for r in contract["frames"]}

    def report(count: int) -> None:
        if count % 60 == 0:
            print(f"  frame {count}", flush=True)

    evidence = gather_evidence(read_frames(args.video), by_frame, progress=report)
    track = stabilize(contract, evidence)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(build_stable_contract(track, args.video.name)) + "\n", encoding="utf-8")

    covered = {n: sum(f.visibility[n] < 0.5 for f in track.frames) for n in REGION_NAMES}
    filled = sum(f.source == "interpolated" for f in track.frames)
    print(
        f"wrote {args.out}: {len(track.frames)}/{track.frame_count} frames with a face "
        f"({filled} filled), cuts at {track.cuts or 'none'}"
    )
    print("  frames with region covered: " + ", ".join(f"{k} {v}" for k, v in covered.items()))

    stable_by = {f.frame: f for f in track.frames}
    both = [i for i in sorted(by_frame) if i in stable_by and i - 1 in by_frame and i + 1 in by_frame]
    if len(both) > 2:
        for name, index in (("nose", 1), ("eye_l", 33), ("eye_r", 263), ("chin", 152)):
            raw_j, stable_j = [], []
            for i in both:
                ied = float(np.linalg.norm((by_frame[i][33, :2] - by_frame[i][263, :2]) * (track.width, track.height)))
                tri_raw = [by_frame[j] * (track.width, track.height, track.width) for j in (i - 1, i, i + 1)]
                if all(j in stable_by for j in (i - 1, i + 1)):
                    tri_st = [stable_by[j].landmarks * (track.width, track.height, track.width) for j in (i - 1, i, i + 1)]
                    stable_j.append(landmark_jitter(tri_st, [ied] * 3, index)[0])
                raw_j.append(landmark_jitter(tri_raw, [ied] * 3, index)[0])
            print(
                f"  jitter p95/IED {name}: raw {np.percentile(raw_j, 95):.4f} "
                f"stable {np.percentile(stable_j, 95):.4f}"
            )

    if args.overlay:
        from .media import VideoWriter

        with VideoWriter(args.overlay, args.video) as writer:
            for frame in read_frames(args.video):
                writer.write(frame, _draw_overlay(frame.image, by_frame.get(frame.index), stable_by.get(frame.index)))
        print(f"wrote {args.overlay}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
