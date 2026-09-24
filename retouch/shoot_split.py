"""Split a shoot folder into sets by capture time.

Reads each photo's EXIF capture time (``DateTimeOriginal``, falling back to
``DateTimeDigitized`` then ``DateTime``) and groups shots into sets, either
wherever the gap between consecutive shots exceeds a threshold, or into fixed
hour/day buckets. A shot is every file sharing one stem, so a Fuji
``DSCF1234.RAF`` + ``DSCF1234.JPG`` pair and its ``.xmp`` sidecars always land
in the same folder.

The EXIF reader is stdlib-only and seeks rather than decoding pixels, so it
reads a 50 MB RAF in well under a millisecond and runs without OpenCV,
MediaPipe or Pillow installed. Supported containers: JPEG, TIFF and the
TIFF-based RAWs (DNG, NEF, NRW, CR2, ARW, ORF, RW2, PEF, SRW), Fuji RAF
(via its embedded JPEG), Canon CR3, PNG ``eXIf`` and WebP ``EXIF``.

Nothing is moved unless asked: the default is a preview. ``--move`` and
``--copy`` refuse to start if any destination already exists, and every run
that changes files writes ``split-manifest.json`` so a move can be undone with
``--undo``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Sequence, Tuple

# Mirrors ``retouch.io.IMAGE_EXTENSIONS`` (kept local so this module does not
# import OpenCV); ``tests/test_shoot_split.py`` guards the two against drift.
SHOT_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".exr",
    ".raf", ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".dng", ".orf", ".rw2", ".pef", ".srw", ".x3f",
}

MANIFEST_NAME = "split-manifest.json"
NO_TIME_FOLDER = "no-capture-time"

_TAG_DATETIME = 306
_TAG_MODEL = 272
_TAG_EXIF_IFD = 34665
_TAG_DATETIME_ORIGINAL = 36867
_TAG_DATETIME_DIGITIZED = 36868
_TAG_SUBSEC_ORIGINAL = 37521
_TAG_SUBSEC_DIGITIZED = 37522

_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8}
_MAX_IFD_ENTRIES = 1000


# ---------------------------------------------------------------------------
# EXIF reading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptureInfo:
    """What one file says about when (and on which camera) it was shot."""

    time: Optional[datetime]
    source: Optional[str] = None  # EXIF tag the time came from, or "mtime"
    camera: Optional[str] = None


def _parse_exif_datetime(raw: Optional[str], subsec: Optional[str] = None) -> Optional[datetime]:
    if not raw:
        return None
    text = raw.strip().strip("\x00").strip()
    try:
        parsed = datetime.strptime(text[:19], "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None  # blank "    :  :  " or zeroed "0000:00:00" clocks
    digits = re.sub(r"\D", "", subsec or "")
    if digits:
        parsed = parsed.replace(microsecond=int(digits[:6].ljust(6, "0")))
    return parsed


class _TiffReader:
    """Minimal bounds-checked TIFF IFD reader over a seekable file."""

    def __init__(self, fh: BinaryIO, base: int, limit: Optional[int] = None):
        self.fh = fh
        self.base = base
        self.limit = limit
        fh.seek(base)
        header = fh.read(8)
        if len(header) < 8 or header[:2] not in (b"II", b"MM"):
            raise ValueError("not a TIFF header")
        self.endian = "<" if header[:2] == b"II" else ">"
        # Magic is 42 for TIFF, but ORF ("RO"/"SR") and RW2 (0x55) differ;
        # the IFD layout is the same, so accept any.
        self.first_ifd = struct.unpack(self.endian + "I", header[4:8])[0]

    def _read(self, offset: int, size: int) -> bytes:
        if offset < 0 or (self.limit is not None and offset + size > self.limit):
            return b""
        self.fh.seek(self.base + offset)
        return self.fh.read(size)

    def ifd(self, offset: int) -> Dict[int, object]:
        """Return ``{tag: value}`` for ASCII and LONG/SHORT scalar entries."""
        raw_count = self._read(offset, 2)
        if len(raw_count) < 2:
            return {}
        count = struct.unpack(self.endian + "H", raw_count)[0]
        if count > _MAX_IFD_ENTRIES:
            return {}
        table = self._read(offset + 2, count * 12)
        tags: Dict[int, object] = {}
        for i in range(len(table) // 12):
            tag, typ, n = struct.unpack(self.endian + "HHI", table[i * 12:i * 12 + 8])
            value_field = table[i * 12 + 8:i * 12 + 12]
            size = _TYPE_SIZES.get(typ, 0) * n
            if size == 0:
                continue
            data = value_field[:size] if size <= 4 else self._read(
                struct.unpack(self.endian + "I", value_field)[0], min(size, 256))
            if typ == 2:
                tags[tag] = data.split(b"\x00", 1)[0].decode("ascii", "replace")
            elif typ in (4, 13) and n == 1 and len(data) >= 4:
                tags[tag] = struct.unpack(self.endian + "I", data[:4])[0]
            elif typ == 3 and n == 1 and len(data) >= 2:
                tags[tag] = struct.unpack(self.endian + "H", data[:2])[0]
        return tags


def _info_from_tiff(fh: BinaryIO, base: int, limit: Optional[int] = None,
                    first_ifd_is_exif: bool = False) -> CaptureInfo:
    reader = _TiffReader(fh, base, limit)
    ifd0 = reader.ifd(reader.first_ifd)
    exif = ifd0 if first_ifd_is_exif else {}
    pointer = ifd0.get(_TAG_EXIF_IFD)
    if isinstance(pointer, int) and not first_ifd_is_exif:
        exif = reader.ifd(pointer)
    camera = ifd0.get(_TAG_MODEL)
    camera = camera.strip() if isinstance(camera, str) and camera.strip() else None
    for tag, subsec_tag, name, table in (
        (_TAG_DATETIME_ORIGINAL, _TAG_SUBSEC_ORIGINAL, "DateTimeOriginal", exif),
        (_TAG_DATETIME_DIGITIZED, _TAG_SUBSEC_DIGITIZED, "DateTimeDigitized", exif),
        (_TAG_DATETIME, None, "DateTime", ifd0),
    ):
        value = table.get(tag)
        subsec = table.get(subsec_tag) if subsec_tag else None
        when = _parse_exif_datetime(value if isinstance(value, str) else None,
                                    subsec if isinstance(subsec, str) else None)
        if when is not None:
            return CaptureInfo(when, name, camera)
    return CaptureInfo(None, None, camera)


def _info_from_jpeg(fh: BinaryIO, start: int = 0, end: Optional[int] = None) -> CaptureInfo:
    fh.seek(start)
    if fh.read(2) != b"\xff\xd8":
        raise ValueError("not a JPEG")
    pos = start + 2
    while end is None or pos < end:
        fh.seek(pos)
        marker = fh.read(4)
        if len(marker) < 4 or marker[0] != 0xFF:
            break
        kind = marker[1]
        if kind == 0xFF:  # fill byte
            pos += 1
            continue
        if kind in (0xD9, 0xDA):  # end of image / start of scan: no more metadata
            break
        length = struct.unpack(">H", marker[2:4])[0]
        if kind == 0xE1 and fh.read(6) == b"Exif\x00\x00":
            return _info_from_tiff(fh, pos + 10, limit=length - 8)
        pos += 2 + length
    return CaptureInfo(None)


def _info_from_raf(fh: BinaryIO) -> CaptureInfo:
    # RAF header: 16-byte magic ... big-endian JPEG preview offset/length at 84.
    fh.seek(84)
    offset, length = struct.unpack(">II", fh.read(8))
    return _info_from_jpeg(fh, offset, offset + length)


def _info_from_cr3(fh: BinaryIO) -> CaptureInfo:
    # CR3 keeps IFD0 in a CMT1 box and the Exif IFD in CMT2, both near the start.
    head = fh.read(1 << 20)
    cmt1, cmt2 = head.find(b"CMT1"), head.find(b"CMT2")
    ifd0 = _info_from_tiff(fh, cmt1 + 4) if cmt1 >= 4 else CaptureInfo(None)
    exif = _info_from_tiff(fh, cmt2 + 4, first_ifd_is_exif=True) if cmt2 >= 4 else CaptureInfo(None)
    if exif.time is not None:
        return CaptureInfo(exif.time, exif.source, ifd0.camera)
    return ifd0


def _info_from_png(fh: BinaryIO) -> CaptureInfo:
    fh.seek(8)
    while True:
        header = fh.read(8)
        if len(header) < 8:
            return CaptureInfo(None)
        length, kind = struct.unpack(">I4s", header)
        if kind == b"eXIf":
            return _info_from_tiff(fh, fh.tell(), limit=length)
        if kind == b"IEND":
            return CaptureInfo(None)
        fh.seek(length + 4, os.SEEK_CUR)


def _info_from_webp(fh: BinaryIO) -> CaptureInfo:
    fh.seek(12)
    while True:
        header = fh.read(8)
        if len(header) < 8:
            return CaptureInfo(None)
        kind, length = struct.unpack("<4sI", header)
        if kind == b"EXIF":
            start = fh.tell()
            if fh.read(6) != b"Exif\x00\x00":
                fh.seek(start)
            return _info_from_tiff(fh, fh.tell(), limit=length)
        fh.seek(length + (length & 1), os.SEEK_CUR)


def read_capture_info(path: Path) -> CaptureInfo:
    """Read the capture time and camera model from one file's EXIF.

    Never raises for unreadable or unsupported files; returns ``time=None``.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(16)
            if head.startswith(b"FUJIFILMCCD-RAW"):
                return _info_from_raf(fh)
            if head.startswith(b"\xff\xd8"):
                return _info_from_jpeg(fh)
            if head[:2] in (b"II", b"MM"):
                return _info_from_tiff(fh, 0)
            if head[4:8] == b"ftyp" and head[8:11] == b"crx":
                fh.seek(0)
                return _info_from_cr3(fh)
            if head.startswith(b"\x89PNG\r\n\x1a\n"):
                return _info_from_png(fh)
            if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
                return _info_from_webp(fh)
    except (OSError, ValueError, struct.error):
        pass
    return CaptureInfo(None)


# ---------------------------------------------------------------------------
# Shots and sets
# ---------------------------------------------------------------------------


@dataclass
class Shot:
    """Every file that belongs to one exposure (RAW, JPEG, sidecars)."""

    key: str
    files: List[Path]
    time: Optional[datetime] = None
    source: Optional[str] = None
    camera: Optional[str] = None


@dataclass
class ShotSet:
    name: str
    shots: List[Shot] = field(default_factory=list)

    @property
    def start(self) -> Optional[datetime]:
        times = [s.time for s in self.shots if s.time is not None]
        return min(times) if times else None

    @property
    def end(self) -> Optional[datetime]:
        times = [s.time for s in self.shots if s.time is not None]
        return max(times) if times else None

    @property
    def files(self) -> List[Path]:
        return [f for shot in self.shots for f in shot.files]


def _shot_key(path: Path) -> str:
    # "DSCF1234.RAF.xmp" and "DSCF1234.xmp" both belong to DSCF1234.
    return str(path.parent / path.name.split(".", 1)[0].lower())


def collect_shots(
    folder: Path,
    recursive: bool = False,
    mtime_fallback: bool = False,
    camera_offsets: Optional[Dict[str, timedelta]] = None,
) -> Tuple[List[Shot], List[Path]]:
    """Group the folder's files into shots and read each shot's capture time.

    Returns ``(shots, left_alone)``: files that are neither photos nor share a
    stem with one (notes, hidden files, a previous manifest) are left alone.
    """
    walker = folder.rglob("*") if recursive else folder.glob("*")
    files = sorted(p for p in walker if p.is_file() and not p.name.startswith("."))
    photo_keys = {_shot_key(p) for p in files if p.suffix.lower() in SHOT_EXTENSIONS}
    by_key: Dict[str, List[Path]] = {}
    left_alone: List[Path] = []
    for path in files:
        key = _shot_key(path)
        if key in photo_keys and path.name != MANIFEST_NAME:
            by_key.setdefault(key, []).append(path)
        else:
            left_alone.append(path)

    offsets = {k.casefold(): v for k, v in (camera_offsets or {}).items()}
    shots = []
    for key, members in by_key.items():
        shot = Shot(key=key, files=members)
        # Photos before sidecars; RAW before JPEG only matters for the camera
        # name, both carry the same capture time.
        for path in sorted(members, key=lambda p: p.suffix.lower() not in SHOT_EXTENSIONS):
            if path.suffix.lower() not in SHOT_EXTENSIONS:
                continue
            info = read_capture_info(path)
            shot.camera = shot.camera or info.camera
            if info.time is not None:
                shot.time, shot.source = info.time, info.source
                break
        if shot.time is None and mtime_fallback:
            photo = next(p for p in members if p.suffix.lower() in SHOT_EXTENSIONS)
            shot.time = datetime.fromtimestamp(photo.stat().st_mtime).replace(microsecond=0)
            shot.source = "mtime"
        if shot.time is not None and shot.camera and shot.camera.casefold() in offsets:
            shot.time += offsets[shot.camera.casefold()]
        shots.append(shot)
    return shots, left_alone


def plan_sets(shots: Sequence[Shot], mode: str = "gap",
              gap: timedelta = timedelta(minutes=15)) -> List[ShotSet]:
    """Group shots into named sets. Shots without a time get their own set."""
    timed = sorted((s for s in shots if s.time is not None), key=lambda s: (s.time, s.key))
    untimed = sorted((s for s in shots if s.time is None), key=lambda s: s.key)

    groups: List[List[Shot]] = []
    for shot in timed:
        if not groups:
            groups.append([shot])
            continue
        prev = groups[-1][-1]
        if mode == "gap":
            same = shot.time - prev.time <= gap
        elif mode == "hour":
            same = shot.time.replace(minute=0, second=0, microsecond=0) == \
                prev.time.replace(minute=0, second=0, microsecond=0)
        elif mode == "day":
            same = shot.time.date() == prev.time.date()
        else:
            raise ValueError(f"unknown split mode: {mode}")
        if same:
            groups[-1].append(shot)
        else:
            groups.append([shot])

    width = max(2, len(str(len(groups))))
    sets = []
    for number, group in enumerate(groups, 1):
        start = group[0].time
        if mode == "hour":
            name = start.strftime("%Y-%m-%d_%Hh")
        elif mode == "day":
            name = start.strftime("%Y-%m-%d")
        else:
            name = f"{number:0{width}d}_{start.strftime('%Y-%m-%d_%H%M')}"
        sets.append(ShotSet(name, list(group)))
    if untimed:
        sets.append(ShotSet(NO_TIME_FOLDER, untimed))
    return sets


def plan_moves(sets: Sequence[ShotSet], source_root: Path,
               output_root: Path) -> List[Tuple[Path, Path]]:
    """Map every file to ``output_root/<set>/<file name>``."""
    moves = []
    for shot_set in sets:
        for path in shot_set.files:
            moves.append((path, output_root / shot_set.name / path.name))
    return moves


def find_conflicts(moves: Sequence[Tuple[Path, Path]]) -> List[str]:
    """Problems that would make a move or copy overwrite something."""
    problems = []
    seen: Dict[str, Path] = {}
    for src, dst in moves:
        key = os.path.normcase(str(dst)).casefold()
        if key in seen:
            problems.append(f"{src} and {seen[key]} would both become {dst}")
        seen[key] = src
        if dst.exists() and dst.resolve() != src.resolve():
            problems.append(f"{dst} already exists")
    return problems


# ---------------------------------------------------------------------------
# Applying and undoing
# ---------------------------------------------------------------------------


def _write_manifest(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def apply_moves(moves: Sequence[Tuple[Path, Path]], manifest_path: Path, action: str,
                settings: dict) -> int:
    """Move or copy files, recording each one in the manifest as it happens."""
    entries = [{"from": str(src.resolve()), "to": str(dst.resolve()), "done": False}
               for src, dst in moves]
    payload = {"version": 1, "action": action, "created": datetime.now().isoformat(timespec="seconds"),
               "settings": settings, "files": entries}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifest_path, payload)
    done = 0
    for entry, (src, dst) in zip(entries, moves):
        if src.resolve() == dst.resolve():
            entry["done"] = True
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if action == "move":
            shutil.move(str(src), str(dst))
        else:
            shutil.copy2(str(src), str(dst))
        entry["done"] = True
        done += 1
        if done % 50 == 0:
            _write_manifest(manifest_path, payload)
    _write_manifest(manifest_path, payload)
    return done


def undo_moves(manifest_path: Path) -> Tuple[int, List[str]]:
    """Put moved files back where the manifest says they came from."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("action") != "move":
        raise ValueError("only a --move can be undone; for a --copy, delete the set folders")
    restored, problems = 0, []
    for entry in payload["files"]:
        src, dst = Path(entry["from"]), Path(entry["to"])
        if src == dst:
            continue
        if not dst.exists():
            if not src.exists():
                problems.append(f"{dst} is missing")
            continue  # never moved, or already restored
        if src.exists():
            problems.append(f"{src} exists again, left {dst} where it is")
            continue
        src.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst), str(src))
        restored += 1
    for folder in sorted({Path(e["to"]).parent for e in payload["files"]}, reverse=True):
        try:
            folder.rmdir()  # only removes folders the undo left empty
        except OSError:
            pass
    if not problems:
        manifest_path.unlink()
    return restored, problems


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def parse_duration(text: str) -> timedelta:
    """Parse ``90s``, ``15m``, ``2h``, ``1h30m`` or ``HH:MM[:SS]`` (optionally signed)."""
    raw = text.strip()
    sign = -1 if raw.startswith("-") else 1
    body = raw.lstrip("+-")
    if ":" in body:
        parts = body.split(":")
        if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
            raise ValueError(f"bad duration: {text}")
        h, m, s = (int(p) for p in (parts + ["0"])[:3])
        return sign * timedelta(hours=h, minutes=m, seconds=s)
    matches = re.fullmatch(r"(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?", body)
    if not body or not matches:
        if body.isdigit():
            return sign * timedelta(minutes=int(body))
        raise ValueError(f"bad duration: {text}")
    h, m, s = (int(g or 0) for g in matches.groups())
    return sign * timedelta(hours=h, minutes=m, seconds=s)


def _parse_offset(text: str) -> Tuple[str, timedelta]:
    model, sep, amount = text.rpartition("=")
    if not sep or not model.strip():
        raise argparse.ArgumentTypeError("use MODEL=OFFSET, e.g. \"X-T5=+00:03:20\"")
    try:
        return model.strip(), parse_duration(amount)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _fmt_span(shot_set: ShotSet) -> str:
    if shot_set.start is None:
        return "no capture time"
    start, end = shot_set.start, shot_set.end
    span = f"{start:%Y-%m-%d %H:%M}"
    return span + (f"–{end:%H:%M}" if end.date() == start.date() else f" – {end:%Y-%m-%d %H:%M}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="split_shoot.py",
        description="Split a shoot folder into sets by capture time. Previews by default; "
                    "pass --move or --copy to act.",
    )
    parser.add_argument("folder", nargs="?", help="Folder of photos (RAW, JPEG and sidecars)")
    parser.add_argument("-o", "--output", help="Where to create the set folders (default: inside FOLDER)")
    parser.add_argument("--by", choices=["gap", "hour", "day"], default="gap",
                        help="gap: new set after a pause (default); hour/day: fixed buckets")
    parser.add_argument("--gap", default="15m",
                        help="Pause that starts a new set with --by gap, e.g. 10m, 1h, 90s (default: 15m)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--move", action="store_true", help="Move files into set folders")
    action.add_argument("--copy", action="store_true", help="Copy files into set folders")
    action.add_argument("--undo", metavar="MANIFEST",
                        help=f"Undo a --move using its {MANIFEST_NAME}")
    parser.add_argument("-r", "--recursive", action="store_true", help="Include subfolders")
    parser.add_argument("--mtime-fallback", action="store_true",
                        help="Use file modified time for photos with no EXIF capture time "
                             f"(default: put them in {NO_TIME_FOLDER}/)")
    parser.add_argument("--camera-offset", action="append", type=_parse_offset, default=[],
                        metavar="MODEL=OFFSET",
                        help="Shift one camera's clock before grouping, e.g. \"X-T5=+00:03:20\" "
                             "or \"X-H2=-2m\" (repeatable; MODEL is the EXIF model name)")
    parser.add_argument("--json", action="store_true", help="Print the plan as JSON")
    return parser


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
    if gap <= timedelta(0):
        parser.error("--gap must be positive")
    output = Path(args.output).expanduser() if args.output else folder

    shots, left_alone = collect_shots(folder, args.recursive, args.mtime_fallback,
                                      dict(args.camera_offset))
    if not shots:
        print(f"No photos found in {folder}")
        if (output / MANIFEST_NAME).exists():
            print(f"  It was already split; see {output / MANIFEST_NAME}")
        return 0
    sets = plan_sets(shots, args.by, gap)
    moves = plan_moves(sets, folder, output)

    if args.json:
        print(json.dumps({
            "folder": str(folder.resolve()), "output": str(output.resolve()), "by": args.by,
            "gap_seconds": gap.total_seconds(),
            "sets": [{
                "name": s.name,
                "start": s.start.isoformat() if s.start else None,
                "end": s.end.isoformat() if s.end else None,
                "shots": len(s.shots),
                "files": [str(f) for f in s.files],
            } for s in sets],
            "left_alone": [str(p) for p in left_alone],
        }, indent=2))
    else:
        rule = f"gaps over {args.gap}" if args.by == "gap" else f"one folder per {args.by}"
        print(f"{len(shots)} shot(s), {len(moves)} file(s) → {len(sets)} set(s) by {rule}")
        cameras = sorted({s.camera for s in shots if s.camera})
        if len(cameras) > 1:
            print(f"  Cameras: {', '.join(cameras)} (use --camera-offset if their clocks differ)")
        for shot_set in sets:
            print(f"  {shot_set.name + '/':<28} {len(shot_set.shots):>4} shot(s)  {_fmt_span(shot_set)}")
        mtime = sum(1 for s in shots if s.source == "mtime")
        if mtime:
            print(f"  {mtime} shot(s) had no EXIF time and were placed by file modified time")
        if left_alone:
            print(f"  Leaving {len(left_alone)} other file(s) where they are")

    if not (args.move or args.copy):
        if not args.json:
            print("Preview only. Re-run with --move or --copy to create the folders.")
        return 0

    conflicts = find_conflicts(moves)
    if conflicts:
        print("✖ Nothing was changed, because:", file=sys.stderr)
        for problem in conflicts[:20]:
            print(f"  {problem}", file=sys.stderr)
        if len(conflicts) > 20:
            print(f"  … and {len(conflicts) - 20} more", file=sys.stderr)
        return 1

    action = "move" if args.move else "copy"
    manifest = output / MANIFEST_NAME
    if manifest.exists():
        print(f"✖ {manifest} already exists from an earlier split. Undo it or move it away first.",
              file=sys.stderr)
        return 1
    settings = {"by": args.by, "gap": args.gap, "recursive": args.recursive,
                "mtime_fallback": args.mtime_fallback,
                "camera_offsets": {m: o.total_seconds() for m, o in args.camera_offset}}
    count = apply_moves(moves, manifest, action, settings)
    verb = "Moved" if action == "move" else "Copied"
    print(f"✓ {verb} {count} file(s) into {len(sets)} folder(s) under {output}")
    first = next((s for s in sets if s.name != NO_TIME_FOLDER), sets[0])
    print(f"  Retouch a set: ./run batch \"{output / first.name}\" -o <output> --recipe <recipe>")
    if action == "move":
        print(f"  Undo with: ./run split --undo \"{manifest}\"")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
