"""Shoot-wide near-duplicate detection (classical, no model weights).

Burst grouping (:func:`retouch.shoot_intelligence.group_bursts`) only joins
frames shot within a couple of seconds of each other. At a convention the same
pose is often shot again minutes later, after a chat or a lens change, and
those repeats end up scattered through the shoot. This module finds them
across the whole shoot, whatever the time gap, groups them, and suggests one
keeper per group. It never deletes anything; moving the extras aside is a
separate, opt-in step.

What counts as a duplicate
--------------------------
Same pose and framing: the subject stands the same way, arms and head in the
same place, and the framing moved by no more than about 10% (the photographer
shifted or zoomed slightly). A blink, a small head tilt, a different exposure
or white balance still count as duplicates; that is what the keeper choice is
for. An arm raised, a turn of the body or a different crop does not.

How two frames are compared
---------------------------
1. **Signature.** Each photo is decoded once at low resolution (JPEG draft
   mode, EXIF orientation applied) into a 128 px grey thumbnail, its gradient
   magnitude and a 16x16 colour grid.
2. **Pre-filter.** A cosine similarity over coarse 16x16 thumbnails and the
   average colour rule out most pairs with one matrix product.
3. **Alignment.** The central 80% of one frame is matched against the other
   at five scales (0.9-1.1) and five tilts (-2 to +2 degrees) with
   normalised cross-correlation, which gives the reframing (shift, zoom,
   tilt) and a global similarity, both ways round.
4. **Pose check.** With that alignment applied, the gradient maps are
   compared cell by cell on a 12x12 grid. The share of the frame's structure
   (gradient energy) sitting in cells that no longer match is the *changed
   fraction*. A moved arm on a small full-length figure changes only a few
   cells, which is why this local check exists: the global score alone
   cannot see it.
5. **Grouping.** Complete linkage: a frame joins a group only if it is a
   duplicate of every frame already in it, so slow drifts through a pose
   sequence do not chain into one large group.

The keeper is ranked with :func:`retouch.shoot_intelligence.rank_burst_candidates`,
so it prefers the sharpest face with open eyes when face evidence is supplied,
and the sharpest, best-exposed frame otherwise.

Thresholds were measured on 1,944 frames from 17 public talking-head and
full-length video clips (pairs 0.1-8 s apart and across clips) and on real
26 MP con photos with simulated reframing; see ``DUPLICATE_*`` below.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageOps

from .io import open_image

logger = logging.getLogger(__name__)

__all__ = [
    "FrameSignature",
    "PairEvidence",
    "DuplicateGroup",
    "frame_signature",
    "compare_frames",
    "find_duplicates",
    "choose_keepers",
    "write_report",
    "move_extras",
    "face_evidence_for",
    "list_images",
    "signature_from_array",
    "main",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"}

_THUMB = 128            # signature thumbnail edge (square resample)
_COARSE = 16            # pre-filter thumbnail edge
_MATCH = 64             # alignment runs at this edge
_CORE = 0.8             # central share of the frame used as the match template
_SCALES = (0.9, 0.95, 1.0, 1.05, 1.1)
_ANGLES = (0.0, -1.0, 1.0, -2.0, 2.0)  # degrees of camera tilt tried
_GRID = 12              # pose-check grid (cells per side)
_CELL_MATCH = 0.5       # a cell whose gradient NCC is below this has changed

# Decision thresholds. Measured 2026-09-26: pairs 0.1 s apart in real video
# clips have global similarity p5 0.965; pairs across different clips p95
# 0.645. Real 26 MP con frames with 3-10% simulated reframing, +-0.5 EV or a
# white balance shift score 0.985-1.0. See the module docstring.
DUPLICATE_MIN_SIMILARITY = 0.90
DUPLICATE_MAX_CHANGED = 0.10
DUPLICATE_MAX_COLOUR = 6.0      # mean a/b difference of the aligned 16x16 grids (LAB, 0-255 scale)
_ASPECT_TOLERANCE = 0.03
_PREFILTER_COSINE = 0.55


@dataclass(frozen=True)
class FrameSignature:
    """Low-resolution description of one photo, enough to compare poses."""

    path: str
    width: int
    height: int
    capture_time: Optional[float]
    gray: np.ndarray = field(repr=False)      # (_THUMB, _THUMB) float32
    grad: np.ndarray = field(repr=False)      # (_THUMB, _THUMB) float32
    colour: np.ndarray = field(repr=False)    # (16, 16, 3) float32 LAB
    coarse: np.ndarray = field(repr=False)    # (_COARSE*_COARSE,) unit vector

    @property
    def aspect(self) -> float:
        return self.width / max(self.height, 1)


@dataclass(frozen=True)
class PairEvidence:
    """Why two frames are (or are not) duplicates."""

    similarity: float
    changed_fraction: float
    colour_difference: float
    scale: float
    shift: Tuple[float, float]
    angle: float
    duplicate: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "similarity": round(self.similarity, 4),
            "changed_fraction": round(self.changed_fraction, 4),
            "colour_difference": round(self.colour_difference, 3),
            "scale": round(self.scale, 3),
            "shift": [round(self.shift[0], 4), round(self.shift[1], 4)],
            "angle": round(self.angle, 2),
            "duplicate": self.duplicate,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DuplicateGroup:
    """Frames that show the same pose and framing. Review before culling."""

    group_id: str
    asset_paths: Tuple[str, ...]
    keeper: str
    reasons: Tuple[str, ...]
    evidence: Mapping[str, Any]
    review_required: bool = True

    @property
    def extras(self) -> Tuple[str, ...]:
        return tuple(p for p in self.asset_paths if p != self.keeper)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "asset_paths": list(self.asset_paths),
            "keeper": self.keeper,
            "extras": list(self.extras),
            "reasons": list(self.reasons),
            "evidence": dict(self.evidence),
            "review_required": self.review_required,
        }


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------

def _capture_time(image: Image.Image) -> Optional[float]:
    try:
        from datetime import datetime, timezone

        exif = image.getexif()
        raw = exif.get_ifd(0x8769).get(36867) or exif.get(36867) or exif.get(306)
        if not raw:
            return None
        parsed = datetime.strptime(str(raw).strip().rstrip("\x00"), "%Y:%m:%d %H:%M:%S")
        return parsed.replace(tzinfo=timezone.utc).timestamp()
    except (ValueError, TypeError, KeyError, AttributeError, OSError):
        return None


def signature_from_array(
    rgb: np.ndarray,
    *,
    path: str = "",
    size: Optional[Tuple[int, int]] = None,
    capture_time: Optional[float] = None,
) -> FrameSignature:
    """Build a signature from an RGB uint8 array (already upright)."""
    h, w = rgb.shape[:2]
    width, height = size if size is not None else (w, h)
    square = cv2.resize(rgb, (_THUMB * 2, _THUMB * 2), interpolation=cv2.INTER_AREA)
    grey2 = cv2.cvtColor(square, cv2.COLOR_RGB2GRAY).astype(np.float32)
    soft = cv2.GaussianBlur(grey2, (0, 0), 1.5)
    gx = cv2.Sobel(soft, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(soft, cv2.CV_32F, 0, 1)
    grad = cv2.resize(np.hypot(gx, gy), (_THUMB, _THUMB), interpolation=cv2.INTER_AREA)
    grey = cv2.resize(grey2, (_THUMB, _THUMB), interpolation=cv2.INTER_AREA)
    colour = cv2.resize(
        cv2.cvtColor(square, cv2.COLOR_RGB2LAB).astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA
    )
    coarse = cv2.resize(grey, (_COARSE, _COARSE), interpolation=cv2.INTER_AREA).ravel()
    coarse = coarse - coarse.mean()
    coarse = coarse / (np.linalg.norm(coarse) + 1e-6)
    return FrameSignature(
        path=path,
        width=int(width),
        height=int(height),
        capture_time=capture_time,
        gray=grey,
        grad=grad,
        colour=colour,
        coarse=coarse.astype(np.float32),
    )


def frame_signature(path: Union[str, Path]) -> FrameSignature:
    """Decode ``path`` at low resolution and describe it.

    JPEGs use Pillow's draft mode, so a 26 MP file decodes at 1/8 size in a
    few milliseconds. EXIF orientation is applied so portrait frames compare
    as portrait.
    """
    source = Path(path).expanduser().resolve()
    with open_image(source) as image:
        capture_time = _capture_time(image)
        width, height = image.size
        try:
            image.draft("RGB", (_THUMB * 4, _THUMB * 4))
        except Exception:  # draft is a JPEG-only speed-up; other formats decode fully
            pass
        upright = ImageOps.exif_transpose(image)
        rgb = np.asarray(upright.convert("RGB"), dtype=np.uint8)
        orientation = image.getexif().get(0x0112, 1)
    if orientation in (5, 6, 7, 8):
        width, height = height, width
    return signature_from_array(rgb, path=str(source), size=(width, height), capture_time=capture_time)


# ---------------------------------------------------------------------------
# Pair comparison
# ---------------------------------------------------------------------------

def _align(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, float, Tuple[float, float]]:
    """Best NCC of ``a``'s centre inside ``b`` over scales and small tilts.

    Returns ``(score, scale, angle_deg, (dx, dy))``: ``a`` maps onto ``b`` by
    rotating ``angle_deg`` and scaling by ``scale`` about the centre, then
    shifting by ``(dx, dy)`` (fractions of the frame).
    """
    n = a.shape[0]
    m = int(round(n * (1 - _CORE) / 2))
    core = a[m:n - m, m:n - m]
    c = core.shape[0]
    best = (-1.0, 1.0, 0.0, (0.0, 0.0))
    for angle in _ANGLES:
        if angle:
            R = cv2.getRotationMatrix2D((c / 2.0, c / 2.0), angle, 1.0)
            turned = cv2.warpAffine(core, R, (c, c), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            turned = core
        for k in _SCALES:
            size = int(round(c * k))
            if size >= n - 1 or size < 8:
                continue
            tmpl = cv2.resize(turned, (size, size), interpolation=cv2.INTER_AREA)
            res = cv2.matchTemplate(b, tmpl, cv2.TM_CCOEFF_NORMED)
            _, score, _, loc = cv2.minMaxLoc(res)
            if score > best[0]:
                cx = loc[0] + size / 2.0
                cy = loc[1] + size / 2.0
                best = (float(score), float(k), float(angle), ((cx - n / 2.0) / n, (cy - n / 2.0) / n))
    return best


def _changed_fraction(a: FrameSignature, b: FrameSignature, scale: float, angle: float, shift: Tuple[float, float]) -> float:
    """Share of structure in cells that differ once ``a`` is aligned onto ``b``."""
    n = _THUMB
    M = _affine(n, scale, angle, shift)
    warped = cv2.warpAffine(a.grad, M, (n, n), flags=cv2.INTER_LINEAR, borderValue=-1.0)
    valid = cv2.warpAffine(np.ones((n, n), np.float32), M, (n, n), flags=cv2.INTER_NEAREST, borderValue=0.0) > 0.5
    ga = cv2.GaussianBlur(np.where(valid, warped, 0.0).astype(np.float32), (0, 0), 1.0)
    gb = cv2.GaussianBlur(b.grad, (0, 0), 1.0)
    cell = n // _GRID
    energy_total = 0.0
    energy_changed = 0.0
    floor = 0.05 * float(max(gb.mean(), ga[valid].mean() if valid.any() else 0.0, 1e-6))
    for i in range(_GRID):
        for j in range(_GRID):
            ys, xs = slice(i * cell, (i + 1) * cell), slice(j * cell, (j + 1) * cell)
            if valid[ys, xs].mean() < 0.9:
                continue
            pa = ga[ys, xs].ravel()
            pb = gb[ys, xs].ravel()
            energy = float(max(pa.mean(), pb.mean()))
            if energy < floor:
                continue
            da, db = pa - pa.mean(), pb - pb.mean()
            denom = float(np.sqrt((da * da).sum() * (db * db).sum()))
            if denom < 1e-6:
                ncc = 1.0 if abs(pa.mean() - pb.mean()) < 0.25 * energy else 0.0
            else:
                ncc = float((da * db).sum() / denom)
                # Same shape but a very different amount of structure (a
                # patterned sleeve where there was bare arm) is a change too.
                ratio = min(pa.mean(), pb.mean()) / max(pa.mean(), pb.mean(), 1e-6)
                if ratio < 0.5:
                    ncc = min(ncc, ratio)
            energy_total += energy
            if ncc < _CELL_MATCH:
                energy_changed += energy
    if energy_total <= 0.0:
        return 0.0
    return energy_changed / energy_total


def _small(x: np.ndarray) -> np.ndarray:
    return cv2.resize(x, (_MATCH, _MATCH), interpolation=cv2.INTER_AREA)


def compare_frames(
    a: FrameSignature,
    b: FrameSignature,
    *,
    min_similarity: float = DUPLICATE_MIN_SIMILARITY,
    max_changed: float = DUPLICATE_MAX_CHANGED,
    max_colour: float = DUPLICATE_MAX_COLOUR,
) -> PairEvidence:
    """Decide whether two frames show the same pose and framing."""
    # Average colour first: cheap, and position-free so reframing can't move it.
    mean_gap = float(np.abs(a.colour[..., 1:].mean(axis=(0, 1)) - b.colour[..., 1:].mean(axis=(0, 1))).mean())

    def verdict(sim, changed, colour, scale, shift, ok: bool, why: str, angle: float = 0.0) -> PairEvidence:
        return PairEvidence(float(sim), float(changed), float(colour), float(scale),
                            (float(shift[0]), float(shift[1])), float(angle), ok, why)

    ra, rb = a.aspect, b.aspect
    if abs(ra - rb) / max(ra, rb) > _ASPECT_TOLERANCE:
        return verdict(0.0, 1.0, mean_gap, 1.0, (0, 0), False, "different_aspect")
    if mean_gap > max_colour:
        return verdict(0.0, 1.0, mean_gap, 1.0, (0, 0), False, "different_colour")
    ga, gb = _small(a.gray), _small(b.gray)
    s1, k1, r1, t1 = _align(ga, gb)
    s2, _k2, _r2, _t2 = _align(gb, ga)
    sim = min(s1, s2)
    if sim < min_similarity:
        return verdict(sim, 1.0, mean_gap, k1, t1, False, "different_composition", r1)
    colour = _aligned_colour_difference(a, b, k1, r1, t1)
    if colour > max_colour:
        return verdict(sim, 1.0, colour, k1, t1, False, "different_colour", r1)
    # Pose check on the gradient maps, with the same alignment.
    changed = _changed_fraction(a, b, k1, r1, t1)
    if changed > max_changed:
        return verdict(sim, changed, colour, k1, t1, False, "pose_changed", r1)
    return verdict(sim, changed, colour, k1, t1, True, "same_pose_and_framing", r1)


def _affine(n: int, scale: float, angle: float, shift: Tuple[float, float]) -> np.ndarray:
    """``a`` -> ``b`` map on an ``n``-pixel grid: rotate and scale about the centre, then shift."""
    M = cv2.getRotationMatrix2D((n / 2.0, n / 2.0), angle, scale).astype(np.float32)
    M[0, 2] += shift[0] * n
    M[1, 2] += shift[1] * n
    return M


def _aligned_colour_difference(a: FrameSignature, b: FrameSignature, scale: float, angle: float, shift: Tuple[float, float]) -> float:
    """Mean a/b difference of the colour grids once ``a`` is aligned onto ``b``."""
    n = a.colour.shape[0]
    M = _affine(n, scale, angle, shift)
    warped = cv2.warpAffine(a.colour, M, (n, n), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    valid = cv2.warpAffine(np.ones((n, n), np.float32), M, (n, n), flags=cv2.INTER_LINEAR, borderValue=0.0) > 0.99
    if not valid.any():
        return float(np.abs(a.colour[..., 1:] - b.colour[..., 1:]).mean())
    return float(np.abs(warped[..., 1:] - b.colour[..., 1:])[valid].mean())


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _prefilter(signatures: Sequence[FrameSignature]) -> List[Tuple[int, int]]:
    n = len(signatures)
    if n < 2:
        return []
    coarse = np.stack([s.coarse for s in signatures])
    cos = coarse @ coarse.T
    colours = np.stack([s.colour[..., 1:].mean(axis=(0, 1)) for s in signatures])
    aspects = np.asarray([s.aspect for s in signatures])
    pairs: List[Tuple[int, int]] = []
    for i in range(n):
        js = np.nonzero(cos[i, i + 1:] >= _PREFILTER_COSINE)[0] + i + 1
        for j in js:
            if abs(aspects[i] - aspects[j]) / max(aspects[i], aspects[j]) > _ASPECT_TOLERANCE:
                continue
            if float(np.abs(colours[i] - colours[j]).mean()) > DUPLICATE_MAX_COLOUR:
                continue
            pairs.append((i, int(j)))
    return pairs


def _group_indices(n: int, edges: Mapping[Tuple[int, int], PairEvidence], order: Sequence[int]) -> List[List[int]]:
    """Greedy complete-linkage groups over the duplicate edges."""
    adj: Dict[int, Dict[int, float]] = {}
    for (i, j), ev in edges.items():
        if ev.duplicate:
            adj.setdefault(i, {})[j] = ev.similarity
            adj.setdefault(j, {})[i] = ev.similarity
    assigned: set = set()
    groups: List[List[int]] = []
    for i in order:
        if i in assigned or i not in adj:
            continue
        group = [i]
        for j, _sim in sorted(adj[i].items(), key=lambda kv: -kv[1]):
            if j in assigned:
                continue
            if all(j in adj.get(g, {}) for g in group):
                group.append(j)
        if len(group) > 1:
            assigned.update(group)
            groups.append(group)
    return groups


def find_duplicates(
    items: Iterable[Union[str, Path, FrameSignature]],
    *,
    min_similarity: float = DUPLICATE_MIN_SIMILARITY,
    max_changed: float = DUPLICATE_MAX_CHANGED,
    face_evidence_by_path: Optional[Mapping[str, Sequence[Any]]] = None,
    progress: Optional[Any] = None,
) -> List[DuplicateGroup]:
    """Group near-duplicate frames across a whole shoot.

    ``items`` are image paths or ready signatures. Unreadable files are
    skipped with a warning. Groups come back in capture order, each with a
    suggested keeper; nothing is moved or deleted.
    """
    signatures: List[FrameSignature] = []
    for item in items:
        if isinstance(item, FrameSignature):
            signatures.append(item)
            continue
        try:
            signatures.append(frame_signature(item))
        except Exception as exc:  # one unreadable file must not stop the scan
            logger.warning("Duplicate scan skipped %s: %s", item, exc)
        if progress is not None:
            progress(len(signatures))
    order = sorted(
        range(len(signatures)),
        key=lambda i: (signatures[i].capture_time is None, signatures[i].capture_time or 0.0, signatures[i].path),
    )
    edges: Dict[Tuple[int, int], PairEvidence] = {}
    for i, j in _prefilter(signatures):
        edges[(i, j)] = compare_frames(
            signatures[i], signatures[j], min_similarity=min_similarity, max_changed=max_changed
        )
    groups = _group_indices(len(signatures), edges, order)
    rank = {idx: pos for pos, idx in enumerate(order)}
    result: List[DuplicateGroup] = []
    for gi, group in enumerate(sorted(groups, key=lambda g: min(rank[i] for i in g))):
        group = sorted(group, key=lambda i: rank[i])
        paths = tuple(signatures[i].path for i in group)
        pair_ev = [
            edges.get((min(x, y), max(x, y)))
            for x_i, x in enumerate(group) for y in group[x_i + 1:]
        ]
        pair_ev = [ev for ev in pair_ev if ev is not None]
        times = [signatures[i].capture_time for i in group if signatures[i].capture_time is not None]
        span = (max(times) - min(times)) if len(times) >= 2 else None
        evidence = {
            "count": len(group),
            "min_similarity": round(min(ev.similarity for ev in pair_ev), 4),
            "max_changed_fraction": round(max(ev.changed_fraction for ev in pair_ev), 4),
            "time_span_seconds": span,
        }
        reasons = ["same pose and framing", "no clear pose change between any two frames"]
        if span is not None and span > 2.0:
            reasons.append(f"shot {span / 60.0:.1f} min apart" if span >= 60 else f"shot {span:.0f} s apart")
        result.append(DuplicateGroup(
            group_id=f"dup-{gi + 1:04d}",
            asset_paths=paths,
            keeper=paths[0],
            reasons=tuple(reasons),
            evidence=evidence,
        ))
    return choose_keepers(result, face_evidence_by_path)


def choose_keepers(
    groups: Sequence[DuplicateGroup],
    face_evidence_by_path: Optional[Mapping[str, Sequence[Any]]] = None,
) -> List[DuplicateGroup]:
    """Rank each group's frames and make the best one the keeper.

    Uses the burst ranker, so with face evidence the keeper is the frame with
    the sharpest face and open eyes; without it, the sharpest, best-exposed
    frame. The ranking is stored in the group's evidence.
    """
    from .shoot_intelligence import BurstGroup, rank_burst_candidates

    out: List[DuplicateGroup] = []
    for group in groups:
        burst = BurstGroup(
            group_id=group.group_id,
            asset_paths=group.asset_paths,
            confidence=1.0,
            reasons=group.reasons,
            evidence={},
        )
        ranked = rank_burst_candidates(burst, face_evidence_by_path)
        if not ranked:
            out.append(group)
            continue
        evidence = dict(group.evidence)
        evidence["ranking"] = [
            {
                "path": c.path,
                "rank": c.rank,
                "score": round(c.score, 4),
                "flags": list(c.evidence.get("flags") or []),
                "policy": c.evidence.get("scoring_policy"),
            }
            for c in ranked
        ]
        out.append(DuplicateGroup(
            group_id=group.group_id,
            asset_paths=group.asset_paths,
            keeper=ranked[0].path,
            reasons=group.reasons,
            evidence=evidence,
            review_required=True,
        ))
    return out


# ---------------------------------------------------------------------------
# Face evidence (optional keeper refinement)
# ---------------------------------------------------------------------------

def face_evidence_for(paths: Sequence[str]) -> Dict[str, Sequence[Any]]:
    """Face-quality evidence for ``paths`` (only frames that are in a group).

    Lets the keeper prefer the sharpest face with open eyes. Frames that fail
    to load or analyse are left out; the ranker then treats them as having no
    face evidence.
    """
    from .detection import FaceDetector
    from .face_quality import FaceQualityAnalyzer
    from .io import imread_exif

    evidence: Dict[str, Sequence[Any]] = {}
    detector = FaceDetector(allow_unavailable=True)
    try:
        if not detector.available:
            logger.warning("Face detector unavailable; keepers use sharpness and exposure only")
            return evidence
        analyzer = FaceQualityAnalyzer()
        for path in paths:
            try:
                image = imread_exif(path)
                evidence[path] = analyzer.analyze(image, detector.detect(image))
            except Exception as exc:  # one bad frame must not stop the others
                logger.warning("Face check failed for %s: %s", path, exc)
    finally:
        detector.close()
    return evidence


# ---------------------------------------------------------------------------
# Report and moving extras aside
# ---------------------------------------------------------------------------

REPORT_DIRNAME = "duplicates-report"
EXTRAS_DIRNAME = "duplicates"
_REPORT_THUMB = 360


def _thumb(path: str, target: Path) -> bool:
    try:
        with open_image(path) as image:
            try:
                image.draft("RGB", (_REPORT_THUMB * 2, _REPORT_THUMB * 2))
            except Exception:
                pass
            upright = ImageOps.exif_transpose(image).convert("RGB")
            upright.thumbnail((_REPORT_THUMB, _REPORT_THUMB))
            target.parent.mkdir(parents=True, exist_ok=True)
            upright.save(target, "JPEG", quality=85)
        return True
    except Exception as exc:
        logger.warning("Thumbnail failed for %s: %s", path, exc)
        return False


def _rel(path: Union[str, Path], root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return Path(path).name


def write_report(
    groups: Sequence[DuplicateGroup],
    shoot_root: Union[str, Path],
    report_dir: Union[str, Path],
    *,
    scanned: int,
) -> Path:
    """Write ``duplicates.json`` and a contact-sheet ``index.html``.

    Every image path in the page is relative to the report folder, so the
    folder can be moved or zipped with the shoot.
    """
    root = Path(shoot_root).expanduser().resolve()
    out = Path(report_dir).expanduser().resolve()
    thumbs = out / "thumbs"
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "shoot": str(root),
        "scanned": scanned,
        "groups": [
            {**g.to_dict(), "relative_paths": [_rel(p, root) for p in g.asset_paths], "keeper_relative": _rel(g.keeper, root)}
            for g in groups
        ],
        "thresholds": {
            "min_similarity": DUPLICATE_MIN_SIMILARITY,
            "max_changed_fraction": DUPLICATE_MAX_CHANGED,
            "max_colour_difference": DUPLICATE_MAX_COLOUR,
        },
    }
    (out / "duplicates.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    extras = sum(len(g.extras) for g in groups)
    cards: List[str] = []
    for g in groups:
        ranking = {r["path"]: r for r in g.evidence.get("ranking", [])}
        tiles: List[str] = []
        for i, path in enumerate(g.asset_paths):
            thumb = thumbs / g.group_id / f"{i:03d}.jpg"
            ok = _thumb(path, thumb)
            rel = _rel(path, root)
            r = ranking.get(path, {})
            flags = ", ".join(r.get("flags") or [])
            badge = '<span class="keep">Keeper</span>' if path == g.keeper else '<span class="extra">Extra</span>'
            img = (
                f'<img loading="lazy" src="thumbs/{html.escape(g.group_id)}/{i:03d}.jpg" alt="">'
                if ok else '<div class="missing">no preview</div>'
            )
            meta = f"rank {r['rank']} &middot; score {r['score']:.2f}" if r else ""
            if flags:
                meta += f" &middot; {html.escape(flags)}"
            tiles.append(
                f'<figure class="{"is-keeper" if path == g.keeper else ""}">{img}'
                f'<figcaption>{badge}<span class="name">{html.escape(rel)}</span>'
                f'<span class="meta">{meta}</span></figcaption></figure>'
            )
        span = g.evidence.get("time_span_seconds")
        when = ""
        if span is not None:
            when = f" &middot; {span / 60:.1f} min apart" if span >= 60 else f" &middot; {span:.0f} s apart"
        cards.append(
            f'<section><h2>{html.escape(g.group_id)} &middot; {len(g.asset_paths)} frames{when}</h2>'
            f'<div class="row">{"".join(tiles)}</div></section>'
        )
    body = "".join(cards) or '<p class="none">No duplicates found.</p>'
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Duplicates</title>
<style>
:root {{ --bg:#fafaf9; --fg:#1c1917; --muted:#57534e; --card:#fff; --line:#e7e5e4; --keep:#15803d; --extra:#a16207; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#1c1917; --fg:#f5f5f4; --muted:#a8a29e; --card:#292524; --line:#44403c; --keep:#4ade80; --extra:#facc15; }} }}
body {{ margin:0; padding:24px 16px; background:var(--bg); color:var(--fg); font:15px/1.45 system-ui, sans-serif; }}
h1 {{ font-size:22px; margin:0 0 4px; }} p.lead {{ color:var(--muted); margin:0 0 24px; max-width:70ch; }}
section {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; margin:0 0 16px; }}
h2 {{ font-size:15px; margin:0 0 10px; font-weight:600; }}
.row {{ display:flex; gap:12px; overflow-x:auto; padding-bottom:4px; }}
figure {{ margin:0; flex:0 0 auto; width:220px; }}
figure img, .missing {{ width:220px; height:220px; object-fit:contain; background:var(--bg); border-radius:6px; display:block; border:2px solid transparent; }}
figure.is-keeper img {{ border-color:var(--keep); }}
figcaption {{ display:flex; flex-direction:column; gap:2px; font-size:12px; margin-top:6px; }}
.keep {{ color:var(--keep); font-weight:600; }} .extra {{ color:var(--extra); font-weight:600; }}
.name {{ word-break:break-all; }} .meta {{ color:var(--muted); }} .none {{ color:var(--muted); }}
</style></head><body>
<h1>Duplicates in {html.escape(root.name)}</h1>
<p class="lead">{scanned} photos scanned, {len(groups)} groups of near-identical frames, {extras} extra frames.
Each group shows the same pose and framing; the suggested keeper has a green border. Nothing has been moved or deleted.</p>
{body}
</body></html>
"""
    index = out / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def move_extras(
    groups: Sequence[DuplicateGroup],
    shoot_root: Union[str, Path],
    dest_dirname: str = EXTRAS_DIRNAME,
) -> Dict[str, int]:
    """Move every non-keeper frame into ``<shoot>/<dest_dirname>/``.

    Relative folders are kept, nothing is overwritten (an existing file at
    the destination is counted and skipped), and nothing is deleted. Moving
    the files back undoes it.
    """
    root = Path(shoot_root).expanduser().resolve()
    dest_root = root / dest_dirname
    counts = {"moved": 0, "exists": 0, "missing": 0}
    for group in groups:
        for path in group.extras:
            src = Path(path)
            if not src.is_file():
                counts["missing"] += 1
                continue
            dest = dest_root / _rel(src, root)
            if dest.exists():
                counts["exists"] += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            counts["moved"] += 1
    return counts


def list_images(folder: Union[str, Path], *, recursive: bool = False,
                skip_dirs: Sequence[str] = (REPORT_DIRNAME, EXTRAS_DIRNAME)) -> List[Path]:
    """Image files in ``folder`` (sorted), skipping this tool's own folders."""
    base = Path(folder).expanduser().resolve()
    candidates = base.rglob("*") if recursive else base.iterdir()
    out: List[Path] = []
    for path in candidates:
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel_parts = path.relative_to(base).parts
        if rel_parts and rel_parts[0] in skip_dirs:
            continue
        if any(part.startswith(".") for part in rel_parts):
            continue
        out.append(path)
    return sorted(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m retouch.duplicates",
        description=(
            "Find near-identical frames (same pose and framing) anywhere in a shoot, "
            "whatever the time gap, and suggest one keeper per group. Writes a report; "
            "moves nothing unless --move-extras is given, and never deletes."
        ),
    )
    parser.add_argument("folder", type=Path, help="Shoot folder with the original photos")
    parser.add_argument("-r", "--recursive", action="store_true", help="Include subfolders")
    parser.add_argument("-o", "--report", type=Path, default=None,
                        help=f"Report folder (default: FOLDER/{REPORT_DIRNAME})")
    parser.add_argument("--faces", action="store_true",
                        help="Check faces in grouped frames so the keeper has the sharpest face and open eyes (slower)")
    parser.add_argument("--strict", action="store_true",
                        help="Only group frames that are almost identical (fewer, surer groups)")
    parser.add_argument("--move-extras", action="store_true",
                        help=f"Move every non-keeper into FOLDER/{EXTRAS_DIRNAME}/ (never deletes or overwrites)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point for ``python -m retouch.duplicates``. Returns an exit code."""
    args = _build_parser().parse_args(argv)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    folder: Path = args.folder.expanduser()
    if not folder.is_dir():
        print(f"error: folder not found: {folder}", file=sys.stderr)
        return 1
    files = list_images(folder, recursive=args.recursive)
    if not files:
        print(f"error: no photos found in {folder}", file=sys.stderr)
        return 1
    total = len(files)

    def progress(done: int) -> None:
        if done == total or done % 50 == 0:
            print(f"  read {done}/{total}", flush=True)

    print(f"Scanning {total} photos in {folder}")
    kwargs: Dict[str, Any] = {}
    if args.strict:
        kwargs = {"min_similarity": 0.95, "max_changed": 0.05}
    groups = find_duplicates(files, progress=progress, **kwargs)
    if args.faces and groups:
        grouped = [p for g in groups for p in g.asset_paths]
        print(f"Checking faces in {len(grouped)} grouped frames")
        groups = choose_keepers(groups, face_evidence_for(grouped))
    report_dir = args.report or (folder / REPORT_DIRNAME)
    index = write_report(groups, folder, report_dir, scanned=total)
    extras = sum(len(g.extras) for g in groups)
    print(f"Found {len(groups)} duplicate groups ({extras} extra frames)")
    print(f"Report → {index}")
    if args.move_extras and extras:
        counts = move_extras(groups, folder)
        print(f"Moved {counts['moved']} extra frames → {folder.expanduser().resolve() / EXTRAS_DIRNAME}")
        if counts["exists"]:
            print(f"Skipped {counts['exists']} already present at the destination")
        if counts["missing"]:
            print(f"Missing {counts['missing']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
