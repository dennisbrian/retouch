"""Group a shoot by cosplayer, without face recognition.

At a convention the subjects rotate: one cosplayer for ten minutes, the next
for five, the first one again after lunch. This module sorts a shoot folder
into one group per cosplayer so each person's photos can be found and handed
over together.

It never looks at who a face belongs to. Instead it reads what a cosplayer
*wears*: the MediaPipe multiclass selfie segmenter (already shipped for hair
masks) splits each photo into hair, clothes, accessories and skin, and the
colours of the costume, wig and props become a small colour signature. Two
cues then decide the groups:

1. **Sessions.** Shots in capture order stay together while the pause between
   them is short (``--gap``, default 10 min) and the costume does not change.
   A costume change is a jump in the signature that holds for the next shots
   too, so one odd frame (a close-up of a prop, a wide shot) does not split a
   session.
2. **Linking.** Sessions from different times of day join when their average
   signatures match closely (``--link``). The bar is deliberately strict:
   leaving one cosplayer in two folders costs a drag-and-drop, while mixing
   two people's photos would send someone the wrong pictures.

Shots with two or more people, and shots with no person in them (props,
venue, details), stay in the session they were taken in but never decide a
split or a link.

Limits (calibration in ``docs/guides/GROUP_BY_COSPLAYER.md``): people in
the same colours look alike to it. Two cosplayers in matching black suits or
a group cosplay in one uniform can share a group, most often when shot back
to back. One cosplayer shot very differently (full length by a window, then a
tight close-up) may stay as two groups. The HTML report shows every group so
either is quick to fix by hand.

Nothing is moved unless asked: the default is a preview plus
``people-report/index.html``. ``--move``/``--copy`` put each group in
``by-person/person-NN/`` and write a manifest so a move can be undone.

Usage::

    python -m retouch.cosplayer_groups ~/shoots/con-day1            # preview + report
    python -m retouch.cosplayer_groups ~/shoots/con-day1 --move     # sort into folders
    python -m retouch.cosplayer_groups --undo ~/shoots/con-day1/by-person/people-manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import logging
import struct
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .shoot_split import (
    SHOT_EXTENSIONS,
    Shot,
    ShotSet,
    apply_moves,
    collect_shots,
    find_conflicts,
    parse_duration,
    plan_moves,
    undo_moves,
)

_logger = logging.getLogger(__name__)

MANIFEST_NAME = "people-manifest.json"
REPORT_DIR = "people-report"
OUTPUT_DIR = "by-person"
UNSORTED = "unsorted"

# Multiclass selfie segmenter channels (model card order).
_BACKGROUND, _HAIR, _BODY_SKIN, _FACE_SKIN, _CLOTHES, _OTHERS = range(6)

# Long side the preview is decoded and segmented at.
PREVIEW_DIM = 512
# A shot needs at least this share of its pixels in costume/wig/props to get
# a signature; below it (a venue shot, a tight face crop) it cannot vote.
MIN_COSTUME_SHARE = 0.01
# Colour signature: joint Lab histogram, L x a x b bins over a* and b* in
# [78, 178) (8-bit OpenCV Lab, 128 = neutral). Coarse on purpose: finer or
# blurred bins matched different outfits more often on the calibration set.
_BINS = (4, 10, 10)
_AB_RANGE = (78, 178)

# Hellinger distances between signatures (0 = same colours, 1 = disjoint).
# Calibration (docs/guides/GROUP_BY_COSPLAYER.md): single shots of one
# costume in different locations and framings scored up to 0.46 apart, and 4%
# of pairs of different outfits scored below 0.45, so single shots cannot be
# trusted alone; session averages and the time order carry the rest.
DEFAULT_SPLIT = 0.40   # a costume change inside a session
DEFAULT_LINK = 0.20    # two sessions are the same cosplayer
DEFAULT_GAP = timedelta(minutes=10)
# Usable shots before a boundary averaged when looking for a costume change.
_CHANGE_WINDOW = 4


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------

_RAW_EXTENSIONS = SHOT_EXTENSIONS - {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".exr"}


def _raf_preview_bytes(path: Path) -> Optional[bytes]:
    """The JPEG preview Fuji embeds in every RAF (offset/length at byte 84)."""
    with open(path, "rb") as fh:
        head = fh.read(92)
        if len(head) < 92 or not head.startswith(b"FUJIFILMCCD-RAW"):
            return None
        offset, length = struct.unpack(">II", head[84:92])
        fh.seek(offset)
        data = fh.read(length)
    return data if data.startswith(b"\xff\xd8") else None


def load_preview(path: Path, max_dim: int = PREVIEW_DIM) -> Optional[np.ndarray]:
    """Decode a small, upright BGR preview of one photo, or None.

    JPEGs decode at reduced scale (Pillow draft mode), so a 26 MP file costs
    a few tens of milliseconds. RAF uses its embedded preview; other RAW
    formats are read only through a JPEG with the same name (see
    :func:`preview_source`).
    """
    from PIL import Image, ImageOps

    try:
        suffix = path.suffix.lower()
        if suffix == ".raf":
            data = _raf_preview_bytes(path)
            if data is None:
                return None
            img = Image.open(io.BytesIO(data))
        elif suffix in _RAW_EXTENSIONS:
            return None
        else:
            img = Image.open(path)
        with img:
            img.draft("RGB", (max_dim, max_dim))
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img.thumbnail((max_dim, max_dim))
            rgb = np.asarray(img)
    except Exception as exc:  # unreadable or truncated file
        _logger.info("No preview for %s: %s", path, exc)
        return None
    return np.ascontiguousarray(rgb[:, :, ::-1])


def preview_source(shot: Shot) -> Optional[Path]:
    """The shot's file best suited for a preview: a JPEG/PNG/TIFF first, then RAF."""
    photos = [p for p in shot.files if p.suffix.lower() in SHOT_EXTENSIONS]
    photos.sort(key=lambda p: (p.suffix.lower() in _RAW_EXTENSIONS,
                               p.suffix.lower() != ".raf", p.name))
    for path in photos:
        if path.suffix.lower() not in _RAW_EXTENSIONS or path.suffix.lower() == ".raf":
            return path
    return None


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


class CostumeSegmenter:
    """Per-pixel class confidences from the MediaPipe multiclass selfie
    segmenter (the model the hair masks already download and verify)."""

    def __init__(self) -> None:
        self._seg = None
        self._lock = threading.Lock()

    def _get(self):
        if self._seg is None:
            import mediapipe as mp

            from .model_fetch import get_model_path

            vision = mp.tasks.vision
            base = mp.tasks.BaseOptions
            self._seg = vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(
                    base_options=base(model_asset_path=get_model_path("selfie_multiclass"),
                                      delegate=base.Delegate.CPU),
                    running_mode=vision.RunningMode.IMAGE,
                    output_category_mask=False,
                    output_confidence_masks=True,
                )
            )
        return self._seg

    def __call__(self, img_bgr: np.ndarray) -> np.ndarray:
        """``(H, W, 6)`` float32 confidences for a uint8 BGR image."""
        import cv2
        import mediapipe as mp

        rgb = np.ascontiguousarray(img_bgr[:, :, ::-1])
        with self._lock:
            result = self._get().segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        h, w = img_bgr.shape[:2]
        chans = []
        for mask in result.confidence_masks:
            arr = np.squeeze(np.asarray(mask.numpy_view(), dtype=np.float32))
            if arr.shape != (h, w):
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
            chans.append(arr)
        return np.clip(np.stack(chans, axis=-1), 0.0, 1.0)

    def close(self) -> None:
        with self._lock:
            seg, self._seg = self._seg, None
        if seg is not None:
            try:
                seg.close()
            except Exception as exc:
                _logger.debug("Segmenter close raised: %s", exc)


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


@dataclass
class Look:
    """What one shot shows: how many people, and its costume colours."""

    faces: int = 0
    costume_share: float = 0.0
    whole: Optional[np.ndarray] = None   # costume + wig + props
    head: Optional[np.ndarray] = None    # the part around the face(s): wig, headpiece
    body: Optional[np.ndarray] = None    # the rest: costume, props

    @property
    def usable(self) -> bool:
        """Can this shot decide a split or a link? One person, enough costume."""
        return self.faces <= 1 and self.whole is not None


def _histogram(lab: np.ndarray, mask: np.ndarray, min_pixels: int = 150) -> Optional[np.ndarray]:
    if int(mask.sum()) < min_pixels:
        return None
    px = lab[mask].astype(np.float32)
    # Very saturated colours (a pure red or blue costume) sit outside the
    # a*/b* range; clip them into the edge bins instead of dropping them.
    px[:, 1:] = np.clip(px[:, 1:], _AB_RANGE[0], _AB_RANGE[1] - 1e-3)
    hist, _ = np.histogramdd(px, bins=_BINS, range=((0, 256), _AB_RANGE, _AB_RANGE))
    hist = hist.ravel()
    return hist / hist.sum()


def _face_boxes(face_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Face-skin blobs big enough to be a subject: at least a quarter of the
    largest one (so a crowd face behind the cosplayer does not count)."""
    import cv2

    n, _, stats, _ = cv2.connectedComponentsWithStats(face_mask.astype(np.uint8))
    if n <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA]
    floor = max(0.0005 * face_mask.size, 0.25 * areas.max())
    order = np.argsort(-areas)
    return [tuple(int(v) for v in stats[i + 1, :4]) for i in order if areas[i] >= floor]


def look_from_classes(img_bgr: np.ndarray, probs: np.ndarray) -> Look:
    """Build a :class:`Look` from a preview and its segmenter confidences.

    Costume = hair + clothes + accessories (a white or pastel wig often reads
    as clothes, so wig and costume are pooled rather than trusted apart).
    The head zone is a box around each face: 1 face height above it, 1.3 below,
    1.6 face widths either side of its centre.
    """
    import cv2

    costume = (probs[..., _HAIR] + probs[..., _CLOTHES] + probs[..., _OTHERS]) > 0.5
    faces = _face_boxes(probs[..., _FACE_SKIN] > 0.5)
    share = float(costume.mean())
    look = Look(faces=len(faces), costume_share=share)
    if share < MIN_COSTUME_SHARE:
        return look
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    head = np.zeros_like(costume)
    for x, y, w, h in faces:
        cx = x + w / 2.0
        head[max(0, int(y - 1.0 * h)):int(y + 1.3 * h),
             max(0, int(cx - 1.6 * w)):max(0, int(cx + 1.6 * w))] = True
    look.whole = _histogram(lab, costume)
    look.head = _histogram(lab, costume & head)
    look.body = _histogram(lab, costume & ~head)
    return look


def _hellinger(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(max(0.0, 1.0 - float(np.sum(np.sqrt(a * b))))))


def look_distance(a: Look, b: Look) -> float:
    """0 (same colours) .. 1 (nothing in common). Averages the whole-costume
    distance with the mean of the head and body distances, so a matching
    costume under a different wig (or the reverse) still reads as different."""
    if a.whole is None or b.whole is None:
        return 1.0
    whole = _hellinger(a.whole, b.whole)
    parts = [_hellinger(x, y) for x, y in ((a.head, b.head), (a.body, b.body))
             if x is not None and y is not None]
    if not parts:
        return whole
    return 0.5 * (whole + sum(parts) / len(parts))


def mean_look(looks: Sequence[Look]) -> Optional[Look]:
    """Average signature of the usable looks, or None if there are none."""
    usable = [lk for lk in looks if lk.usable]
    if not usable:
        return None

    def avg(attr: str) -> Optional[np.ndarray]:
        arrs = [getattr(lk, attr) for lk in usable if getattr(lk, attr) is not None]
        if not arrs:
            return None
        m = np.mean(arrs, axis=0)
        return m / m.sum()

    return Look(faces=1, costume_share=float(np.mean([lk.costume_share for lk in usable])),
                whole=avg("whole"), head=avg("head"), body=avg("body"))


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


@dataclass
class ShotLook:
    shot: Shot
    look: Look
    preview: Optional[Path] = None


@dataclass
class Session:
    items: List[ShotLook] = field(default_factory=list)

    @property
    def start(self) -> Optional[datetime]:
        return self.items[0].shot.time if self.items else None

    @property
    def end(self) -> Optional[datetime]:
        return self.items[-1].shot.time if self.items else None

    @property
    def look(self) -> Optional[Look]:
        return mean_look([it.look for it in self.items])


@dataclass
class PersonGroup:
    name: str
    sessions: List[Session]

    @property
    def items(self) -> List[ShotLook]:
        return [it for s in self.sessions for it in s.items]


def _change_points(items: Sequence[ShotLook], split: float) -> List[int]:
    """Indices where a costume change starts inside one time-run of shots.

    Usable shot i starts a new costume when it differs from the average of
    the few usable shots before it (since the last change) by more than
    ``split``, and so does the next usable shot, if there is one. Needing two
    in a row stops one odd frame (a prop close-up, a wide shot) from
    splitting a session. The cut goes right before shot i, so a group or
    detail shot taken in between stays with the session before it.
    """
    usable = [i for i, it in enumerate(items) if it.look.usable]
    cuts: List[int] = []
    last_cut = 0
    for pos in range(1, len(usable)):
        i = usable[pos]
        before = [items[j].look for j in usable[max(0, pos - _CHANGE_WINDOW):pos] if j >= last_cut]
        if not before:
            continue
        ref_before = mean_look(before)
        if look_distance(items[i].look, ref_before) <= split:
            continue
        if pos + 1 < len(usable) and look_distance(items[usable[pos + 1]].look, ref_before) <= split:
            continue
        cuts.append(i)
        last_cut = i
    return cuts


def build_sessions(items: Sequence[ShotLook], gap: timedelta = DEFAULT_GAP,
                   split: float = DEFAULT_SPLIT) -> List[Session]:
    """Cut the timed shots into sessions: at pauses longer than ``gap`` and at
    costume changes. Untimed shots each become their own session."""
    timed = sorted((it for it in items if it.shot.time is not None),
                   key=lambda it: (it.shot.time, it.shot.key))
    runs: List[List[ShotLook]] = []
    for it in timed:
        if runs and it.shot.time - runs[-1][-1].shot.time <= gap:
            runs[-1].append(it)
        else:
            runs.append([it])
    sessions: List[Session] = []
    for run in runs:
        bounds = [0] + _change_points(run, split) + [len(run)]
        for a, b in zip(bounds, bounds[1:]):
            if b > a:
                sessions.append(Session(list(run[a:b])))
    untimed = sorted((it for it in items if it.shot.time is None), key=lambda it: it.shot.key)
    sessions.extend(Session([it]) for it in untimed)
    return sessions


def link_sessions(sessions: Sequence[Session], link: float = DEFAULT_LINK) -> List[List[Session]]:
    """Join sessions of the same cosplayer (average linkage, strict threshold).

    Sessions without a usable look (only group shots, only details) are never
    joined to anything. Two clusters join only when the average distance
    between their sessions is at most ``link``; the closest pair joins first.
    """
    looks = [s.look for s in sessions]
    clusters: List[List[int]] = [[i] for i in range(len(sessions))]
    n = len(sessions)
    dist = np.full((n, n), np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            if looks[i] is not None and looks[j] is not None:
                dist[i, j] = dist[j, i] = look_distance(looks[i], looks[j])

    def cluster_dist(a: List[int], b: List[int]) -> float:
        return float(np.mean([dist[i, j] for i in a for j in b]))

    while len(clusters) > 1:
        best, pair = np.inf, None
        for x in range(len(clusters)):
            for y in range(x + 1, len(clusters)):
                d = cluster_dist(clusters[x], clusters[y])
                if d < best:
                    best, pair = d, (x, y)
        if pair is None or best > link:
            break
        x, y = pair
        clusters[x] = clusters[x] + clusters[y]
        del clusters[y]
    return [[sessions[i] for i in sorted(c)] for c in clusters]


def group_people(items: Sequence[ShotLook], gap: timedelta = DEFAULT_GAP,
                 split: float = DEFAULT_SPLIT, link: float = DEFAULT_LINK) -> List[PersonGroup]:
    """Sessions, linked into one group per cosplayer, named in order of first
    appearance. Single shots that have no person in them and no timed
    neighbour go to ``unsorted``."""
    sessions = build_sessions(items, gap, split)
    clusters = link_sessions(sessions, link)
    loose: List[Session] = []
    kept: List[List[Session]] = []
    for c in clusters:
        if all(s.look is None for s in c) and all(it.look.faces == 0 for s in c for it in s.items):
            loose.extend(c)
        else:
            kept.append(c)

    def first_key(c: List[Session]):
        times = [s.start for s in c if s.start is not None]
        return (0, min(times), "") if times else (1, datetime.min, c[0].items[0].shot.key)

    kept.sort(key=first_key)
    width = max(2, len(str(len(kept))))
    groups = [PersonGroup(f"person-{k:0{width}d}", c) for k, c in enumerate(kept, 1)]
    if loose:
        groups.append(PersonGroup(UNSORTED, loose))
    return groups


# ---------------------------------------------------------------------------
# Scanning a folder
# ---------------------------------------------------------------------------


def scan(shots: Sequence[Shot], segmenter: Optional[Callable[[np.ndarray], np.ndarray]] = None,
         progress: Optional[Callable[[int, int], None]] = None,
         keep_previews: Optional[Dict[str, np.ndarray]] = None) -> List[ShotLook]:
    """Read a preview of every shot and compute its look.

    ``keep_previews`` (key -> BGR) collects the previews for the report.
    """
    own = segmenter is None
    seg = segmenter or CostumeSegmenter()
    items: List[ShotLook] = []
    try:
        for n, shot in enumerate(shots, 1):
            src = preview_source(shot)
            img = load_preview(src) if src is not None else None
            if img is None:
                items.append(ShotLook(shot, Look(), src))
            else:
                items.append(ShotLook(shot, look_from_classes(img, seg(img)), src))
                if keep_previews is not None:
                    keep_previews[shot.key] = img
            if progress is not None:
                progress(n, len(shots))
    finally:
        if own:
            seg.close()
    return items


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

_CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e2dc;--warn:#9a5b00}
@media (prefers-color-scheme:dark){:root{--bg:#171716;--fg:#ecebe6;--muted:#a09f98;--card:#222220;--line:#3a3a36;--warn:#f0b35a}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}p.sub{color:var(--muted);margin:0 0 24px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:0 0 16px}
h2{font-size:17px;margin:0 0 2px}.meta{color:var(--muted);font-size:13px;margin:0 0 10px}
.grid{display:flex;flex-wrap:wrap;gap:8px}.sess{display:contents}
figure{margin:0;width:132px}figure img{width:132px;height:132px;object-fit:cover;border-radius:6px;display:block;background:var(--line)}
figcaption{font-size:11px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tag{color:var(--warn)}.brk{flex-basis:100%;height:0;border-top:1px dashed var(--line);margin:2px 0}
"""


def _fmt_time(t: Optional[datetime]) -> str:
    return t.strftime("%a %H:%M") if t else "no time"


def write_report(groups: Sequence[PersonGroup], out_dir: Path,
                 previews: Dict[str, np.ndarray], title: str) -> Path:
    """Contact sheet: one card per cosplayer, sessions separated by a dashed line."""
    import cv2

    thumbs = out_dir / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    parts = [f"<h1>{html.escape(title)}</h1>",
             f"<p class=sub>{sum(len(g.items) for g in groups)} photos in "
             f"{sum(1 for g in groups if g.name != UNSORTED)} cosplayer groups. "
             "Grouped by costume and wig colours and capture time, not by face. "
             "Dashed lines separate the times of day a group was shot.</p>"]
    for g in groups:
        span = [s for s in g.sessions if s.start]
        when = ", ".join(f"{_fmt_time(s.start)}–{s.end.strftime('%H:%M')}" for s in span) or "no capture time"
        multi = sum(1 for it in g.items if it.look.faces >= 2)
        extra = f" · <span class=tag>{multi} with 2+ people</span>" if multi else ""
        parts.append(f"<section><h2>{g.name}</h2><p class=meta>{len(g.items)} photo{'' if len(g.items) == 1 else 's'} · {html.escape(when)}{extra}</p><div class=grid>")
        for k, s in enumerate(g.sessions):
            if k:
                parts.append("<div class=brk></div>")
            for it in s.items:
                img = previews.get(it.shot.key)
                name = (it.preview or it.shot.files[0]).name
                src = ""
                if img is not None:
                    fname = hashlib.sha1(it.shot.key.encode("utf-8")).hexdigest()[:12] + ".jpg"
                    small = img
                    scale = 264 / max(img.shape[:2])
                    if scale < 1:
                        small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(thumbs / fname), small, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    src = f"thumbs/{fname}"
                note = " · group" if it.look.faces >= 2 else ""
                t = it.shot.time.strftime("%H:%M:%S") if it.shot.time else ""
                parts.append(f"<figure><img loading=lazy src='{src}' alt=''><figcaption title='{html.escape(name)}'>"
                             f"{html.escape(name)}</figcaption><figcaption>{t}{note}</figcaption></figure>")
        parts.append("</div></section>")
    page = ("<!doctype html><html lang=en><head><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Cosplayer groups</title><style>{_CSS}</style></head><body><main>"
            + "".join(parts) + "</main></body></html>")
    index = out_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def groups_json(groups: Sequence[PersonGroup], folder: Path) -> dict:
    return {
        "folder": str(folder.resolve()),
        "groups": [{
            "name": g.name,
            "sessions": [{
                "start": s.start.isoformat() if s.start else None,
                "end": s.end.isoformat() if s.end else None,
                "shots": [{
                    "files": [str(f) for f in it.shot.files],
                    "time": it.shot.time.isoformat() if it.shot.time else None,
                    "people": it.look.faces,
                } for it in s.items],
            } for s in g.sessions],
        } for g in groups],
    }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="./run people",
        description="Group a shoot by cosplayer, from costume and wig colours plus capture time "
                    "(no face recognition). Previews and writes an HTML report by default; "
                    "pass --move or --copy to sort files into by-person/person-NN folders.",
    )
    p.add_argument("folder", nargs="?", help="Folder of photos (RAW, JPEG and sidecars)")
    p.add_argument("-o", "--output", help=f"Where to create the person folders (default: FOLDER/{OUTPUT_DIR})")
    p.add_argument("--gap", default="10m",
                   help="A pause longer than this always ends a session, e.g. 5m, 20m (default: 10m)")
    p.add_argument("--link", type=float, default=DEFAULT_LINK,
                   help=f"How alike two sessions' costumes must be to join, 0-1; lower is stricter "
                        f"(default: {DEFAULT_LINK})")
    p.add_argument("--split", type=float, default=DEFAULT_SPLIT,
                   help=f"How big a costume change splits a session, 0-1 (default: {DEFAULT_SPLIT})")
    p.add_argument("--no-link", action="store_true",
                   help="Only split by time and costume changes; never join sessions across the day")
    action = p.add_mutually_exclusive_group()
    action.add_argument("--move", action="store_true", help="Move files into person folders")
    action.add_argument("--copy", action="store_true", help="Copy files into person folders")
    action.add_argument("--undo", metavar="MANIFEST", help=f"Undo a --move using its {MANIFEST_NAME}")
    p.add_argument("-r", "--recursive", action="store_true", help="Include subfolders")
    p.add_argument("--mtime-fallback", action="store_true",
                   help="Use file modified time for photos with no EXIF capture time")
    p.add_argument("--no-report", action="store_true", help=f"Skip {REPORT_DIR}/index.html")
    p.add_argument("--json", action="store_true", help="Print the groups as JSON")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.undo:
        try:
            restored, problems = undo_moves(Path(args.undo))
        except (OSError, ValueError, KeyError) as exc:
            print(f"✖ Undo failed: {exc}", file=sys.stderr)
            return 1
        print(f"✓ Restored {restored} file(s)")
        for problem in problems:
            print(f"  ! {problem}", file=sys.stderr)
        if not problems:
            try:
                Path(args.undo).parent.rmdir()  # the by-person folder, once empty
            except OSError:
                pass
        return 1 if problems else 0

    if not args.folder:
        parser.error("FOLDER is required")
    folder = Path(args.folder).expanduser()
    if not folder.is_dir():
        parser.error(f"not a folder: {folder}")
    try:
        gap = parse_duration(args.gap)
    except ValueError as exc:
        parser.error(str(exc))
    for name in ("link", "split"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name} must be between 0 and 1")
    output = Path(args.output).expanduser() if args.output else folder / OUTPUT_DIR

    shots, _ = collect_shots(folder, args.recursive, args.mtime_fallback)
    # Never re-scan an earlier sort or report.
    shots = [s for s in shots if not any(part in (OUTPUT_DIR, REPORT_DIR)
                                         for part in Path(s.key).relative_to(folder).parts)]
    if not shots:
        print(f"No photos found in {folder}")
        return 0

    def progress(n: int, total: int) -> None:
        if not args.json and (n == total or n % 25 == 0):
            print(f"  read {n}/{total}", flush=True)

    previews: Dict[str, np.ndarray] = {}
    items = scan(shots, progress=progress, keep_previews=None if args.no_report else previews)
    link = -1.0 if args.no_link else args.link
    groups = group_people(items, gap, args.split, link)

    report = None
    if not args.no_report:
        report = write_report(groups, folder / REPORT_DIR, previews, f"Cosplayers in {folder.name}")
        with open(folder / REPORT_DIR / "groups.json", "w", encoding="utf-8") as fh:
            json.dump(groups_json(groups, folder), fh, indent=2)

    if args.json:
        print(json.dumps(groups_json(groups, folder), indent=2))
    else:
        people = [g for g in groups if g.name != UNSORTED]
        print(f"{len(shots)} shot(s) → {len(people)} cosplayer group(s)")
        for g in groups:
            times = ", ".join(_fmt_time(s.start) for s in g.sessions if s.start) or "no capture time"
            multi = sum(1 for it in g.items if it.look.faces >= 2)
            note = f"  ({multi} with 2+ people)" if multi else ""
            print(f"  {g.name + '/':<14} {len(g.items):>4} shot(s)  {times}{note}")
        no_preview = sum(1 for it in items if it.preview is None)
        if no_preview:
            print(f"  {no_preview} shot(s) had no readable preview (RAW without a JPEG) "
                  "and were placed by capture time only")
        if report:
            print(f"  Report: {report}")

    if not (args.move or args.copy):
        if not args.json:
            print("Preview only. Check the report, then re-run with --move or --copy.")
        return 0

    sets = [ShotSet(g.name, [it.shot for it in g.items]) for g in groups]
    moves = plan_moves(sets, folder, output)
    conflicts = find_conflicts(moves)
    if conflicts:
        print("✖ Nothing was changed, because:", file=sys.stderr)
        for problem in conflicts[:20]:
            print(f"  {problem}", file=sys.stderr)
        return 1
    manifest = output / MANIFEST_NAME
    if manifest.exists():
        print(f"✖ {manifest} already exists from an earlier sort. Undo it or move it away first.",
              file=sys.stderr)
        return 1
    action = "move" if args.move else "copy"
    settings = {"gap": args.gap, "link": link, "split": args.split, "recursive": args.recursive}
    count = apply_moves(moves, manifest, action, settings)
    verb = "Moved" if action == "move" else "Copied"
    print(f"✓ {verb} {count} file(s) into {len(sets)} folder(s) under {output}")
    if action == "move":
        print(f"  Undo with: ./run people --undo \"{manifest}\"")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
