"""Tests for shoot_split: grouping and organizing photos by capture time."""

from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image
import json
import struct
import pytest

from retouch.shoot_split import (
    Shot,
    ShotSet,
    collect_shots,
    plan_sets,
    plan_moves,
    find_conflicts,
    apply_moves,
    undo_moves,
    parse_duration,
    main,
    MANIFEST_NAME,
    NO_TIME_FOLDER,
)


# ---------------------------------------------------------------------------
# Helpers for creating test images
# ---------------------------------------------------------------------------


def _write_jpeg_with_time(path: Path, model: str, dt: datetime) -> None:
    """Create a JPEG with EXIF DateTime."""
    image = Image.new("RGB", (8, 8), (128, 128, 128))
    exif = image.getexif()
    exif[272] = model  # Model tag (272)
    exif[306] = dt.strftime("%Y:%m:%d %H:%M:%S")  # DateTime tag (306)
    image.save(path, exif=exif.tobytes())


def _write_raf_with_time(path: Path, model: str, dt: datetime) -> None:
    """Create a minimal RAF with embedded JPEG containing the time."""
    # Create the embedded JPEG
    jpeg_image = Image.new("RGB", (8, 8), (64, 64, 64))
    exif = jpeg_image.getexif()
    exif[272] = model
    exif[306] = dt.strftime("%Y:%m:%d %H:%M:%S")  # DateTime tag

    import io
    jpeg_buf = io.BytesIO()
    jpeg_image.save(jpeg_buf, format="JPEG", exif=exif.tobytes())
    jpeg_bytes = jpeg_buf.getvalue()

    # RAF header: "FUJIFILMCCD-RAW " (16 bytes) + padding to offset 84
    # At offset 84: big-endian JPEG offset and length (8 bytes)
    # Then padding to JPEG offset and the JPEG data

    jpeg_offset = 92  # After header (16) + JPEG offset/length (8) + some padding

    with open(path, "wb") as f:
        # Write RAF header and padding
        f.write(b"FUJIFILMCCD-RAW " + b"\x00" * (84 - 16))
        # Write JPEG offset and length
        f.write(struct.pack(">II", jpeg_offset, len(jpeg_bytes)))
        # Padding to JPEG offset
        f.write(b"\x00" * (jpeg_offset - f.tell()))
        # Write JPEG
        f.write(jpeg_bytes)


def _write_xmp_sidecar(path: Path) -> None:
    """Create an empty XMP sidecar file."""
    path.write_text('<?xml version="1.0"?>\n<x:xmpmeta></x:xmpmeta>\n')


# ---------------------------------------------------------------------------
# Tests for collect_shots
# ---------------------------------------------------------------------------


def test_collect_shots_groups_raf_jpg_and_xmp_by_stem(tmp_path):
    """RAW, JPEG, and XMP with the same stem land in one shot."""
    dt = datetime(2026, 9, 20, 14, 32, 5)

    _write_raf_with_time(tmp_path / "DSCF0001.RAF", "X-T5", dt)
    _write_jpeg_with_time(tmp_path / "DSCF0001.JPG", "X-T5", dt)
    _write_xmp_sidecar(tmp_path / "DSCF0001.RAF.xmp")
    _write_xmp_sidecar(tmp_path / "DSCF0001.xmp")

    shots, left_alone = collect_shots(tmp_path)

    assert len(shots) == 1
    shot = shots[0]
    assert shot.time == dt
    assert shot.camera == "X-T5"
    # All four files should be in the shot
    assert len(shot.files) == 4
    assert all(p.parent == tmp_path for p in shot.files)


def test_collect_shots_stem_matching_is_case_insensitive(tmp_path):
    """DSCF0001.RAF and dscf0001.jpg are matched as one stem."""
    dt = datetime(2026, 9, 20, 14, 32, 5)

    _write_raf_with_time(tmp_path / "DSCF0001.RAF", "X-T5", dt)
    _write_jpeg_with_time(tmp_path / "dscf0001.jpg", "X-T5", dt)

    shots, left_alone = collect_shots(tmp_path)

    assert len(shots) == 1
    assert len(shots[0].files) == 2


def test_collect_shots_ignores_non_photo_files(tmp_path):
    """Non-photo files (notes, hidden files) are left alone or filtered."""
    dt = datetime(2026, 9, 20, 14, 32, 5)

    _write_jpeg_with_time(tmp_path / "photo.jpg", "X-T5", dt)
    (tmp_path / "notes.txt").write_text("some notes")
    (tmp_path / ".DS_Store").write_text("mac thing")

    shots, left_alone = collect_shots(tmp_path)

    assert len(shots) == 1
    # .DS_Store is filtered out because it starts with ".", only notes.txt is left alone
    assert len(left_alone) == 1
    assert left_alone[0].name == "notes.txt"


def test_collect_shots_returns_mtime_when_exif_missing_and_fallback_enabled(tmp_path):
    """Photos without EXIF time use mtime if mtime_fallback=True."""
    # Create a plain image without EXIF time
    image = Image.new("RGB", (8, 8), (128, 128, 128))
    path = tmp_path / "notimed.jpg"
    image.save(path)

    shots, _ = collect_shots(tmp_path, mtime_fallback=True)

    assert len(shots) == 1
    assert shots[0].time is not None
    assert shots[0].source == "mtime"


def test_collect_shots_puts_untimed_photos_in_no_capture_time(tmp_path):
    """Photos without EXIF time and no mtime fallback go to a no-capture-time folder."""
    image = Image.new("RGB", (8, 8), (128, 128, 128))
    (tmp_path / "notimed.jpg").write_bytes(image.tobytes())

    shots, _ = collect_shots(tmp_path)

    # Should still be collected as a shot, just without time
    assert len(shots) == 1
    assert shots[0].time is None


def test_collect_shots_applies_camera_offset(tmp_path):
    """camera_offsets shifts the time of matching cameras."""
    dt = datetime(2026, 9, 20, 14, 32, 5)

    _write_jpeg_with_time(tmp_path / "photo1.jpg", "X-T5", dt)
    _write_jpeg_with_time(tmp_path / "photo2.jpg", "X-H2", dt)

    # Shift X-T5 by +2 minutes, leave X-H2 alone
    offsets = {"X-T5": timedelta(minutes=2)}
    shots, _ = collect_shots(tmp_path, camera_offsets=offsets)

    assert len(shots) == 2
    t5_shot = next(s for s in shots if "photo1" in str(s.files))
    h2_shot = next(s for s in shots if "photo2" in str(s.files))

    assert t5_shot.time == datetime(2026, 9, 20, 14, 34, 5)
    assert h2_shot.time == dt


def test_collect_shots_camera_offset_is_case_insensitive(tmp_path):
    """Camera model matching in offsets is case-insensitive."""
    dt = datetime(2026, 9, 20, 14, 32, 5)

    _write_jpeg_with_time(tmp_path / "photo.jpg", "X-T5", dt)

    # Offset keyed as lowercase
    offsets = {"x-t5": timedelta(minutes=3)}
    shots, _ = collect_shots(tmp_path, camera_offsets=offsets)

    assert shots[0].time == datetime(2026, 9, 20, 14, 35, 5)


def test_collect_shots_recursive_includes_subfolders(tmp_path):
    """With recursive=True, subfolders are included."""
    dt = datetime(2026, 9, 20, 14, 32, 5)

    subfolder = tmp_path / "batch1"
    subfolder.mkdir()

    _write_jpeg_with_time(tmp_path / "root.jpg", "X-T5", dt)
    _write_jpeg_with_time(subfolder / "sub.jpg", "X-T5", dt)

    shots, _ = collect_shots(tmp_path, recursive=True)

    assert len(shots) == 2


# ---------------------------------------------------------------------------
# Tests for plan_sets
# ---------------------------------------------------------------------------


def test_plan_sets_by_gap_groups_with_15m_threshold(tmp_path):
    """By gap mode: 14:00, 14:05, 14:40, 14:41 with 15m threshold -> 2 sets."""
    times = [
        datetime(2026, 9, 20, 14, 0, 0),
        datetime(2026, 9, 20, 14, 5, 0),
        datetime(2026, 9, 20, 14, 40, 0),
        datetime(2026, 9, 20, 14, 41, 0),
    ]

    shots = [
        Shot(f"shot_{i}", [Path(f"dummy_{i}.jpg")], time=t)
        for i, t in enumerate(times)
    ]

    sets = plan_sets(shots, mode="gap", gap=timedelta(minutes=15))

    assert len(sets) == 2
    assert len(sets[0].shots) == 2  # 14:00, 14:05
    assert len(sets[1].shots) == 2  # 14:40, 14:41
    # Names should be numbered
    assert sets[0].name.startswith("01_")
    assert sets[1].name.startswith("02_")


def test_plan_sets_by_gap_stays_same_set_when_gap_equals_threshold(tmp_path):
    """Gap exactly equal to threshold stays in same set."""
    times = [
        datetime(2026, 9, 20, 14, 0, 0),
        datetime(2026, 9, 20, 14, 15, 0),  # Exactly 15 minutes later
    ]

    shots = [
        Shot(f"shot_{i}", [Path(f"dummy_{i}.jpg")], time=t)
        for i, t in enumerate(times)
    ]

    sets = plan_sets(shots, mode="gap", gap=timedelta(minutes=15))

    assert len(sets) == 1
    assert len(sets[0].shots) == 2


def test_plan_sets_by_gap_handles_3_sets(tmp_path):
    """Three sets are numbered with 2 digits."""
    times = [
        datetime(2026, 9, 20, 14, 0, 0),
        datetime(2026, 9, 20, 14, 20, 0),
        datetime(2026, 9, 20, 14, 40, 0),
    ]

    shots = [
        Shot(f"shot_{i}", [Path(f"dummy_{i}.jpg")], time=t)
        for i, t in enumerate(times)
    ]

    sets = plan_sets(shots, mode="gap", gap=timedelta(minutes=15))

    assert len(sets) == 3
    assert sets[0].name == "01_2026-09-20_1400"
    assert sets[1].name == "02_2026-09-20_1420"
    assert sets[2].name == "03_2026-09-20_1440"


def test_plan_sets_by_gap_pads_to_3_digits_for_100_sets(tmp_path):
    """100+ sets are padded to 3 digits."""
    # Create 101 shots spread across time
    shots = [
        Shot(f"shot_{i}", [Path(f"dummy_{i}.jpg")],
             time=datetime(2026, 9, 20, 0, 0) + timedelta(hours=i))
        for i in range(101)
    ]

    sets = plan_sets(shots, mode="gap", gap=timedelta(minutes=15))

    assert len(sets) == 101
    assert sets[0].name.startswith("001_")
    assert sets[100].name.startswith("101_")


def test_plan_sets_by_hour_groups_into_hour_buckets(tmp_path):
    """By hour mode: shots in the same hour go to one set."""
    times = [
        datetime(2026, 9, 20, 14, 5, 0),
        datetime(2026, 9, 20, 14, 30, 0),
        datetime(2026, 9, 20, 15, 0, 0),
    ]

    shots = [
        Shot(f"shot_{i}", [Path(f"dummy_{i}.jpg")], time=t)
        for i, t in enumerate(times)
    ]

    sets = plan_sets(shots, mode="hour")

    assert len(sets) == 2
    assert sets[0].name == "2026-09-20_14h"
    assert sets[1].name == "2026-09-20_15h"
    assert len(sets[0].shots) == 2
    assert len(sets[1].shots) == 1


def test_plan_sets_by_day_groups_into_date_buckets(tmp_path):
    """By day mode: shots on the same date go to one set."""
    times = [
        datetime(2026, 9, 20, 14, 0, 0),
        datetime(2026, 9, 20, 23, 0, 0),
        datetime(2026, 9, 21, 0, 0, 0),
    ]

    shots = [
        Shot(f"shot_{i}", [Path(f"dummy_{i}.jpg")], time=t)
        for i, t in enumerate(times)
    ]

    sets = plan_sets(shots, mode="day")

    assert len(sets) == 2
    assert sets[0].name == "2026-09-20"
    assert sets[1].name == "2026-09-21"


def test_plan_sets_by_gap_crossing_midnight_stays_same_set_if_gap_small(tmp_path):
    """Shots with small gap crossing midnight stay in same set."""
    times = [
        datetime(2026, 9, 20, 23, 58, 0),
        datetime(2026, 9, 21, 0, 3, 0),  # 5 minutes later
    ]

    shots = [
        Shot(f"shot_{i}", [Path(f"dummy_{i}.jpg")], time=t)
        for i, t in enumerate(times)
    ]

    sets = plan_sets(shots, mode="gap", gap=timedelta(minutes=15))

    assert len(sets) == 1


def test_plan_sets_puts_untimed_shots_in_no_capture_time_set(tmp_path):
    """Shots with no time go to NO_TIME_FOLDER set."""
    timed = Shot("timed", [Path("dummy.jpg")], time=datetime(2026, 9, 20, 14, 0, 0))
    untimed1 = Shot("untimed1", [Path("dummy2.jpg")], time=None)
    untimed2 = Shot("untimed2", [Path("dummy3.jpg")], time=None)

    sets = plan_sets([timed, untimed1, untimed2])

    assert len(sets) == 2
    assert sets[0].shots == [timed]
    assert sets[1].name == NO_TIME_FOLDER
    assert len(sets[1].shots) == 2


# ---------------------------------------------------------------------------
# Tests for plan_moves
# ---------------------------------------------------------------------------


def test_plan_moves_creates_output_paths(tmp_path):
    """Files are mapped to output_root/<set>/<filename>."""
    shot1 = Shot("shot1", [tmp_path / "photo1.jpg"], time=datetime(2026, 9, 20, 14, 0))
    shot2 = Shot("shot2", [tmp_path / "photo2.jpg"], time=datetime(2026, 9, 20, 15, 0))

    sets = [
        ShotSet("set1", [shot1]),
        ShotSet("set2", [shot2]),
    ]

    output = tmp_path / "output"
    moves = plan_moves(sets, tmp_path, output)

    assert len(moves) == 2
    assert moves[0] == (tmp_path / "photo1.jpg", output / "set1" / "photo1.jpg")
    assert moves[1] == (tmp_path / "photo2.jpg", output / "set2" / "photo2.jpg")


# ---------------------------------------------------------------------------
# Tests for find_conflicts
# ---------------------------------------------------------------------------


def test_find_conflicts_detects_existing_destination(tmp_path):
    """Existing file at destination is reported as conflict."""
    src = tmp_path / "source.jpg"
    dst = tmp_path / "dest.jpg"
    src.write_text("source")
    dst.write_text("dest")

    moves = [(src, dst)]
    conflicts = find_conflicts(moves)

    assert len(conflicts) == 1
    assert "already exists" in conflicts[0]


def test_find_conflicts_detects_duplicate_destinations(tmp_path):
    """Two sources mapping to same destination is a conflict."""
    src1 = tmp_path / "source1.jpg"
    src2 = tmp_path / "source2.jpg"
    dst = tmp_path / "dest" / "same.jpg"
    src1.write_text("1")
    src2.write_text("2")

    moves = [(src1, dst), (src2, dst)]
    conflicts = find_conflicts(moves)

    assert len(conflicts) == 1
    assert "would both become" in conflicts[0]


def test_find_conflicts_ignores_same_source_and_dest(tmp_path):
    """Source == destination (no move needed) is not a conflict."""
    path = tmp_path / "photo.jpg"
    path.write_text("content")

    moves = [(path, path)]
    conflicts = find_conflicts(moves)

    assert len(conflicts) == 0


def test_find_conflicts_treats_case_only_differences_as_collisions(tmp_path):
    """macOS and Windows volumes are case-insensitive, so flag it everywhere."""
    moves = [
        (tmp_path / "a" / "Photo.jpg", tmp_path / "dest" / "Photo.jpg"),
        (tmp_path / "b" / "photo.jpg", tmp_path / "dest" / "photo.jpg"),
    ]

    conflicts = find_conflicts(moves)

    assert len(conflicts) == 1
    assert "would both become" in conflicts[0]


# ---------------------------------------------------------------------------
# Tests for apply_moves and undo_moves
# ---------------------------------------------------------------------------


def test_apply_moves_moves_files_and_writes_manifest(tmp_path):
    """apply_moves creates output dirs, moves files, and writes manifest."""
    src_dir = tmp_path / "source"
    out_dir = tmp_path / "output"
    src_dir.mkdir()

    photo = src_dir / "photo.jpg"
    photo.write_text("image data")

    dst = out_dir / "set1" / "photo.jpg"
    manifest = out_dir / MANIFEST_NAME

    moves = [(photo, dst)]
    count = apply_moves(moves, manifest, "move", {"test": True})

    assert count == 1
    assert dst.exists()
    assert not photo.exists()
    assert manifest.exists()

    payload = json.loads(manifest.read_text())
    assert payload["action"] == "move"
    assert len(payload["files"]) == 1
    assert payload["files"][0]["done"] is True


def test_apply_moves_copies_files_when_action_is_copy(tmp_path):
    """apply_moves copies instead of moving when action='copy'."""
    src_dir = tmp_path / "source"
    out_dir = tmp_path / "output"
    src_dir.mkdir()

    photo = src_dir / "photo.jpg"
    photo.write_text("image data")

    dst = out_dir / "set1" / "photo.jpg"
    manifest = out_dir / MANIFEST_NAME

    moves = [(photo, dst)]
    count = apply_moves(moves, manifest, "copy", {})

    assert count == 1
    assert dst.exists()
    assert photo.exists()  # Still there after copy


def test_apply_moves_handles_same_source_and_dest(tmp_path):
    """apply_moves skips if source and destination are the same."""
    path = tmp_path / "photo.jpg"
    path.write_text("data")
    manifest = tmp_path / MANIFEST_NAME

    moves = [(path, path)]
    count = apply_moves(moves, manifest, "move", {})

    assert count == 0
    payload = json.loads(manifest.read_text())
    assert payload["files"][0]["done"] is True


def test_undo_moves_restores_files_and_removes_manifest(tmp_path):
    """undo_moves restores files from manifest and removes manifest."""
    src_dir = tmp_path / "source"
    out_dir = tmp_path / "output" / "set1"
    src_dir.mkdir()
    out_dir.mkdir(parents=True)

    photo = src_dir / "photo.jpg"
    photo.write_text("data")

    # Create manifest as if a move was done
    manifest = tmp_path / MANIFEST_NAME
    payload = {
        "version": 1,
        "action": "move",
        "created": datetime.now().isoformat(),
        "settings": {},
        "files": [
            {
                "from": str(photo),
                "to": str(out_dir / "photo.jpg"),
                "done": True,
            }
        ],
    }
    manifest.write_text(json.dumps(payload))

    # Now move the file
    moved = out_dir / "photo.jpg"
    photo.rename(moved)
    assert not photo.exists()
    assert moved.exists()

    # Undo
    restored, problems = undo_moves(manifest)

    assert restored == 1
    assert len(problems) == 0
    assert photo.exists()
    assert not moved.exists()
    assert not manifest.exists()


def test_undo_moves_never_overwrites_a_file_that_reappeared(tmp_path):
    """A new file at the original path is kept; the moved one stays put."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    _write_jpeg_with_time(folder / "DSCF0001.JPG", "X-T5", datetime(2026, 9, 20, 14, 0))
    assert main([str(folder), "--move"]) == 0
    moved = next(folder.glob("*/DSCF0001.JPG"))
    (folder / "DSCF0001.JPG").write_text("new card import")

    restored, problems = undo_moves(folder / MANIFEST_NAME)

    assert restored == 0
    assert len(problems) == 1 and "exists again" in problems[0]
    assert (folder / "DSCF0001.JPG").read_text() == "new card import"
    assert moved.exists()
    assert (folder / MANIFEST_NAME).exists()


def test_undo_moves_rejects_copy_manifests(tmp_path):
    """undo_moves raises for --copy manifests."""
    manifest = tmp_path / MANIFEST_NAME
    payload = {
        "version": 1,
        "action": "copy",
        "created": datetime.now().isoformat(),
        "settings": {},
        "files": [],
    }
    manifest.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="only a --move"):
        undo_moves(manifest)


def test_undo_moves_handles_partially_restored_state(tmp_path):
    """undo_moves can resume from a partially-restored manifest."""
    src_dir = tmp_path / "source"
    out_dir = tmp_path / "output" / "set1"
    src_dir.mkdir()
    out_dir.mkdir(parents=True)

    photo1 = src_dir / "photo1.jpg"
    photo2 = src_dir / "photo2.jpg"
    photo1.write_text("data1")
    photo2.write_text("data2")

    # Manifest for two files
    manifest = tmp_path / MANIFEST_NAME
    payload = {
        "version": 1,
        "action": "move",
        "created": datetime.now().isoformat(),
        "settings": {},
        "files": [
            {
                "from": str(photo1),
                "to": str(out_dir / "photo1.jpg"),
                "done": True,
            },
            {
                "from": str(photo2),
                "to": str(out_dir / "photo2.jpg"),
                "done": True,
            },
        ],
    }
    manifest.write_text(json.dumps(payload))

    # Move both files
    photo1.rename(out_dir / "photo1.jpg")
    photo2.rename(out_dir / "photo2.jpg")

    # Manually restore one file to simulate partial undo
    (out_dir / "photo1.jpg").rename(photo1)

    # Now undo should restore the remaining file
    restored, problems = undo_moves(manifest)

    assert restored == 1
    assert photo2.exists()
    assert not (out_dir / "photo2.jpg").exists()


# ---------------------------------------------------------------------------
# Tests for parse_duration
# ---------------------------------------------------------------------------


def test_parse_duration_minutes(tmp_path):
    """Bare number is interpreted as minutes."""
    assert parse_duration("15") == timedelta(minutes=15)
    assert parse_duration("90") == timedelta(minutes=90)


def test_parse_duration_with_suffixes(tmp_path):
    """Suffixes: s (seconds), m (minutes), h (hours)."""
    assert parse_duration("15s") == timedelta(seconds=15)
    assert parse_duration("90s") == timedelta(seconds=90)
    assert parse_duration("15m") == timedelta(minutes=15)
    assert parse_duration("2h") == timedelta(hours=2)


def test_parse_duration_combined_suffixes(tmp_path):
    """Multiple components: 1h30m, 1h30m45s."""
    assert parse_duration("1h30m") == timedelta(hours=1, minutes=30)
    assert parse_duration("1h30m45s") == timedelta(hours=1, minutes=30, seconds=45)
    assert parse_duration("2h") == timedelta(hours=2)


def test_parse_duration_colon_format(tmp_path):
    """HH:MM or HH:MM:SS format."""
    assert parse_duration("00:15:00") == timedelta(hours=0, minutes=15, seconds=0)
    assert parse_duration("1:30:45") == timedelta(hours=1, minutes=30, seconds=45)
    assert parse_duration("02:30") == timedelta(hours=2, minutes=30)


def test_parse_duration_signed(tmp_path):
    """Negative durations: -2m, +00:01:00."""
    assert parse_duration("-2m") == timedelta(minutes=-2)
    assert parse_duration("+00:01:00") == timedelta(minutes=1)
    assert parse_duration("-90s") == timedelta(seconds=-90)


def test_parse_duration_rejects_bad_input(tmp_path):
    """Invalid formats raise ValueError."""
    with pytest.raises(ValueError):
        parse_duration("invalid")

    with pytest.raises(ValueError):
        parse_duration("x")

    with pytest.raises(ValueError):
        parse_duration("1:2:3:4")  # Too many colons

    with pytest.raises(ValueError):
        parse_duration("1:x")  # Non-digit in colon format


# ---------------------------------------------------------------------------
# Tests for main() CLI
# ---------------------------------------------------------------------------


def test_cli_preview_returns_0_and_changes_nothing(tmp_path, capsys):
    """Preview (no flags) returns 0, doesn't move anything."""
    dt = datetime(2026, 9, 20, 14, 32, 5)
    _write_jpeg_with_time(tmp_path / "photo.jpg", "X-T5", dt)

    result = main([str(tmp_path)])

    assert result == 0
    assert (tmp_path / "photo.jpg").exists()
    assert not (tmp_path / MANIFEST_NAME).exists()

    captured = capsys.readouterr()
    assert "Preview only" in captured.out


def test_cli_move_creates_set_folders_and_manifest(tmp_path, capsys):
    """--move creates folders, moves files, writes manifest."""
    times = [
        datetime(2026, 9, 20, 14, 0, 0),
        datetime(2026, 9, 20, 14, 20, 0),
    ]

    for i, t in enumerate(times):
        _write_jpeg_with_time(tmp_path / f"photo{i}.jpg", "X-T5", t)

    result = main([str(tmp_path), "--move"])

    assert result == 0
    assert (tmp_path / MANIFEST_NAME).exists()
    # Original files should not exist
    assert not (tmp_path / "photo0.jpg").exists()
    assert not (tmp_path / "photo1.jpg").exists()


def test_cli_undo_restores_files_and_removes_manifest(tmp_path, capsys):
    """--undo with manifest path restores files."""
    dt = datetime(2026, 9, 20, 14, 32, 5)
    _write_jpeg_with_time(tmp_path / "photo.jpg", "X-T5", dt)

    # First do a move
    result = main([str(tmp_path), "--move"])
    assert result == 0

    manifest = tmp_path / MANIFEST_NAME
    assert manifest.exists()

    # Now undo
    result = main(["--undo", str(manifest)])

    assert result == 0
    assert (tmp_path / "photo.jpg").exists()
    assert not manifest.exists()


def test_cli_copy_preserves_originals(tmp_path, capsys):
    """--copy leaves original files."""
    dt = datetime(2026, 9, 20, 14, 32, 5)
    _write_jpeg_with_time(tmp_path / "photo.jpg", "X-T5", dt)

    result = main([str(tmp_path), "--copy"])

    assert result == 0
    assert (tmp_path / "photo.jpg").exists()  # Original still there
    assert (tmp_path / MANIFEST_NAME).exists()


def test_cli_undo_refuses_copy_manifest(tmp_path, capsys):
    """--undo on a --copy manifest returns 1."""
    dt = datetime(2026, 9, 20, 14, 32, 5)
    _write_jpeg_with_time(tmp_path / "photo.jpg", "X-T5", dt)

    # Do a copy
    result = main([str(tmp_path), "--copy"])
    assert result == 0

    manifest = tmp_path / MANIFEST_NAME
    # Try to undo a copy
    result = main(["--undo", str(manifest)])

    assert result == 1


def test_cli_move_rejects_existing_destination(tmp_path, capsys):
    """--move returns 1 if destination file exists."""
    dt = datetime(2026, 9, 20, 14, 32, 5)

    # Create photo and destination
    _write_jpeg_with_time(tmp_path / "photo.jpg", "X-T5", dt)

    # Pre-create a set folder with a file (folder name based on timestamp)
    set_dir = tmp_path / "01_2026-09-20_1432"  # 14:32 = 1432 in %H%M format
    set_dir.mkdir()
    (set_dir / "photo.jpg").write_text("existing")

    result = main([str(tmp_path), "--move"])

    assert result == 1
    # Original file should still exist (nothing moved)
    assert (tmp_path / "photo.jpg").exists()


def test_cli_move_rejects_if_manifest_exists(tmp_path, capsys):
    """--move returns 1 if manifest already exists."""
    dt = datetime(2026, 9, 20, 14, 32, 5)
    _write_jpeg_with_time(tmp_path / "photo.jpg", "X-T5", dt)

    # Pre-create manifest
    manifest = tmp_path / MANIFEST_NAME
    manifest.write_text("{}")

    result = main([str(tmp_path), "--move"])

    assert result == 1


def test_cli_output_option_puts_sets_in_specified_dir(tmp_path, capsys):
    """--output puts set folders elsewhere."""
    dt = datetime(2026, 9, 20, 14, 32, 5)
    _write_jpeg_with_time(tmp_path / "photo.jpg", "X-T5", dt)

    output = tmp_path / "output"
    result = main([str(tmp_path), "-o", str(output), "--move"])

    assert result == 0
    assert (output / MANIFEST_NAME).exists()
    # Original should be moved (not in source anymore)
    assert not (tmp_path / "photo.jpg").exists()
    # File should now be in output directory
    assert (output / "01_2026-09-20_1432" / "photo.jpg").exists()


def test_cli_json_output_is_valid_json(tmp_path, capsys):
    """--json outputs valid JSON with sets list."""
    times = [
        datetime(2026, 9, 20, 14, 0, 0),
        datetime(2026, 9, 20, 15, 0, 0),
    ]

    for i, t in enumerate(times):
        _write_jpeg_with_time(tmp_path / f"photo{i}.jpg", "X-T5", t)

    result = main([str(tmp_path), "--json"])

    assert result == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert "sets" in data
    assert len(data["sets"]) == 2
    assert data["sets"][0]["name"].startswith("01_")


def test_cli_recursive_includes_subfolder_photos(tmp_path, capsys):
    """--recursive includes photos in subfolders."""
    dt = datetime(2026, 9, 20, 14, 32, 5)

    subfolder = tmp_path / "batch1"
    subfolder.mkdir()

    _write_jpeg_with_time(tmp_path / "root.jpg", "X-T5", dt)
    _write_jpeg_with_time(subfolder / "sub.jpg", "X-T5", dt)

    result = main([str(tmp_path), "-r", "--move"])

    assert result == 0
    # Both should be moved
    assert not (tmp_path / "root.jpg").exists()
    assert not (subfolder / "sub.jpg").exists()


def test_cli_recursive_conflict_on_duplicate_stems_from_different_dirs(tmp_path, capsys):
    """Two same-named files from different subfolders landing in one set = conflict."""
    dt = datetime(2026, 9, 20, 14, 32, 5)

    dir1 = tmp_path / "batch1"
    dir2 = tmp_path / "batch2"
    dir1.mkdir()
    dir2.mkdir()

    # Same filename in different directories, same time
    _write_jpeg_with_time(dir1 / "photo.jpg", "X-T5", dt)
    _write_jpeg_with_time(dir2 / "photo.jpg", "X-T5", dt)

    result = main([str(tmp_path), "-r", "--move"])

    assert result == 1  # Conflict


def test_cli_by_hour_creates_hourly_sets(tmp_path, capsys):
    """--by hour creates one folder per hour."""
    times = [
        datetime(2026, 9, 20, 14, 5, 0),
        datetime(2026, 9, 20, 15, 30, 0),
    ]

    for i, t in enumerate(times):
        _write_jpeg_with_time(tmp_path / f"photo{i}.jpg", "X-T5", t)

    result = main([str(tmp_path), "--by", "hour", "--json"])

    assert result == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert len(data["sets"]) == 2
    assert data["sets"][0]["name"] == "2026-09-20_14h"
    assert data["sets"][1]["name"] == "2026-09-20_15h"


def test_cli_by_day_creates_daily_sets(tmp_path, capsys):
    """--by day creates one folder per day."""
    times = [
        datetime(2026, 9, 20, 14, 0, 0),
        datetime(2026, 9, 21, 10, 0, 0),
    ]

    for i, t in enumerate(times):
        _write_jpeg_with_time(tmp_path / f"photo{i}.jpg", "X-T5", t)

    result = main([str(tmp_path), "--by", "day", "--json"])

    assert result == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert len(data["sets"]) == 2
    assert data["sets"][0]["name"] == "2026-09-20"
    assert data["sets"][1]["name"] == "2026-09-21"


def test_cli_no_photos_found(tmp_path, capsys):
    """Empty folder prints message and returns 0."""
    result = main([str(tmp_path)])

    assert result == 0
    captured = capsys.readouterr()
    assert "No photos found" in captured.out
