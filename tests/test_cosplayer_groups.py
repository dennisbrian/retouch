"""Tests for cosplayer_groups: grouping a shoot by costume colours and capture time."""

import io
import json
import struct
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from retouch import cosplayer_groups as cg
from retouch.cosplayer_groups import (
    Look,
    ShotLook,
    build_sessions,
    group_people,
    link_sessions,
    load_preview,
    look_distance,
    look_from_classes,
    preview_source,
)
from retouch.shoot_split import Shot

SKIN_BGR = (140, 170, 220)
RED, BLUE, GREEN, PURPLE = (40, 40, 200), (200, 90, 30), (60, 170, 60), (150, 60, 140)
WIG_WHITE, WIG_PINK = (235, 235, 235), (190, 150, 250)


def _person(costume, wig=None, size=(200, 160), faces=1):
    """White backdrop, a costume block, a wig block and a skin-coloured face."""
    h, w = size
    img = np.full((h, w, 3), 255, np.uint8)
    img[90:190, 40:120] = costume
    for k in range(faces):
        x = 60 + k * 50 if faces > 1 else 65
        if wig is not None:
            img[20:80, x - 12:x + 42] = wig
        img[40:75, x:x + 30] = SKIN_BGR
    return img


def _fake_classes(img):
    """Stand-in for the multiclass segmenter: white = background, the skin
    colour = face skin, anything else = clothes."""
    h, w = img.shape[:2]
    probs = np.zeros((h, w, 6), np.float32)
    white = np.all(img >= 250, axis=-1)
    skin = np.all(img == np.array(SKIN_BGR, np.uint8), axis=-1)
    probs[..., 0] = white
    probs[..., 3] = skin
    probs[..., 4] = ~white & ~skin
    return probs


def _look(costume, wig=None, faces=1):
    img = _person(costume, wig, faces=faces)
    return look_from_classes(img, _fake_classes(img))


def _items(spec, start=datetime(2026, 9, 20, 10, 0)):
    """spec: list of (minutes_from_start, Look)."""
    return [ShotLook(Shot(key=f"/shoot/dscf{n:04d}", files=[Path(f"/shoot/DSCF{n:04d}.JPG")],
                          time=start + timedelta(minutes=m)), look)
            for n, (m, look) in enumerate(spec)]


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def test_look_reads_one_person_with_a_costume():
    look = _look(RED, WIG_PINK)
    assert look.faces == 1
    assert look.usable
    assert look.whole is not None and look.head is not None and look.body is not None
    assert look.whole.sum() == pytest.approx(1.0)


def test_look_without_costume_cannot_vote():
    img = np.full((200, 160, 3), 255, np.uint8)
    img[40:75, 65:95] = SKIN_BGR
    look = look_from_classes(img, _fake_classes(img))
    assert look.whole is None
    assert not look.usable


def test_look_counts_two_people_and_does_not_vote():
    look = _look(RED, faces=2)
    assert look.faces == 2
    assert not look.usable


def test_distance_is_small_for_same_costume_and_large_for_different():
    same = look_distance(_look(RED, WIG_PINK), _look(RED, WIG_PINK))
    other = look_distance(_look(RED, WIG_PINK), _look(BLUE, WIG_WHITE))
    assert same < 0.05
    assert other > 0.9


def test_same_costume_with_different_wig_reads_as_different():
    d = look_distance(_look(RED, WIG_PINK), _look(RED, WIG_WHITE))
    assert d > cg.DEFAULT_LINK


def test_distance_to_a_look_without_signature_is_max():
    assert look_distance(Look(), _look(RED)) == 1.0


# ---------------------------------------------------------------------------
# Sessions and linking
# ---------------------------------------------------------------------------


def test_long_pause_starts_a_new_session():
    items = _items([(0, _look(RED)), (2, _look(RED)), (30, _look(BLUE))])
    sessions = build_sessions(items, gap=timedelta(minutes=10))
    assert [len(s.items) for s in sessions] == [2, 1]


def test_costume_change_splits_back_to_back_shots():
    red, blue = _look(RED, WIG_PINK), _look(BLUE, WIG_WHITE)
    items = _items([(0, red), (1, red), (2, red), (3, blue), (4, blue), (5, blue)])
    sessions = build_sessions(items)
    assert [len(s.items) for s in sessions] == [3, 3]


def test_one_odd_frame_does_not_split_a_session():
    red, blue = _look(RED, WIG_PINK), _look(BLUE, WIG_WHITE)
    items = _items([(0, red), (1, red), (2, red), (3, blue), (4, red), (5, red)])
    sessions = build_sessions(items)
    assert len(sessions) == 1


def test_group_and_detail_shots_stay_in_their_session():
    red, blue = _look(RED, WIG_PINK), _look(BLUE, WIG_WHITE)
    duo, detail = _look(GREEN, faces=2), Look()
    items = _items([(0, red), (1, duo), (2, red), (3, detail), (4, blue), (5, blue)])
    sessions = build_sessions(items)
    assert [len(s.items) for s in sessions] == [4, 2]


def test_sessions_of_the_same_costume_hours_apart_are_linked():
    red, blue = _look(RED, WIG_PINK), _look(BLUE, WIG_WHITE)
    items = _items([(0, red), (1, red), (60, blue), (61, blue), (240, red), (241, red)])
    groups = group_people(items)
    assert [g.name for g in groups] == ["person-01", "person-02"]
    assert len(groups[0].items) == 4 and len(groups[0].sessions) == 2
    assert len(groups[1].items) == 2


def test_no_link_keeps_sessions_apart():
    red = _look(RED, WIG_PINK)
    items = _items([(0, red), (240, red)])
    assert len(group_people(items, link=-1.0)) == 2


def test_sessions_with_only_group_shots_are_never_linked():
    duo = _look(GREEN, faces=2)
    items = _items([(0, duo), (240, duo)])
    sessions = build_sessions(items)
    assert len(link_sessions(sessions)) == 2


def test_untimed_detail_shots_go_to_unsorted():
    red = _look(RED, WIG_PINK)
    items = _items([(0, red), (1, red)])
    items.append(ShotLook(Shot(key="/shoot/prop", files=[Path("/shoot/PROP.JPG")]), Look()))
    groups = group_people(items)
    assert [g.name for g in groups] == ["person-01", cg.UNSORTED]


def test_groups_are_named_in_order_of_first_appearance():
    red, blue, green = _look(RED, WIG_PINK), _look(BLUE, WIG_WHITE), _look(GREEN, WIG_PINK)
    items = _items([(0, blue), (1, blue), (30, green), (60, red), (90, blue)])
    groups = group_people(items)
    first_times = [g.sessions[0].start for g in groups]
    assert first_times == sorted(first_times)
    assert len(groups) == 3


# ---------------------------------------------------------------------------
# Previews
# ---------------------------------------------------------------------------


def _jpeg_bytes(img_bgr, orientation=None):
    pil = Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1]))
    buf = io.BytesIO()
    if orientation is not None:
        exif = pil.getexif()
        exif[0x0112] = orientation
        pil.save(buf, "JPEG", quality=95, exif=exif)
    else:
        pil.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def test_load_preview_downscales_and_applies_orientation(tmp_path):
    img = np.zeros((600, 1200, 3), np.uint8)
    path = tmp_path / "a.jpg"
    path.write_bytes(_jpeg_bytes(img, orientation=6))  # rotate 90° on display
    preview = load_preview(path, max_dim=300)
    assert preview is not None
    assert preview.shape[0] > preview.shape[1]  # upright portrait
    assert max(preview.shape[:2]) <= 300


def test_load_preview_reads_the_jpeg_inside_a_raf(tmp_path):
    jpeg = _jpeg_bytes(_person(RED))
    path = tmp_path / "DSCF0001.RAF"
    header = b"FUJIFILMCCD-RAW " + b"\x00" * (84 - 16)
    offset = len(header) + 8
    path.write_bytes(header + struct.pack(">II", offset, len(jpeg)) + jpeg)
    preview = load_preview(path)
    assert preview is not None and preview.shape[:2] == (200, 160)


def test_load_preview_returns_none_for_other_raw_and_broken_files(tmp_path):
    (tmp_path / "a.nef").write_bytes(b"not really a nef")
    (tmp_path / "b.jpg").write_bytes(b"broken")
    assert load_preview(tmp_path / "a.nef") is None
    assert load_preview(tmp_path / "b.jpg") is None


def test_preview_source_prefers_jpeg_then_raf():
    shot = Shot(key="k", files=[Path("X.RAF"), Path("X.JPG"), Path("X.xmp")])
    assert preview_source(shot) == Path("X.JPG")
    assert preview_source(Shot(key="k", files=[Path("X.RAF")])) == Path("X.RAF")
    assert preview_source(Shot(key="k", files=[Path("X.NEF")])) is None


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


class _FakeSegmenter:
    def __call__(self, img):
        return _fake_classes(img)

    def close(self):
        pass


def _write_shot(path, img, when):
    pil = Image.fromarray(np.ascontiguousarray(img[:, :, ::-1]))
    exif = pil.getexif()
    exif[0x8769] = {36867: when.strftime("%Y:%m:%d %H:%M:%S")}
    pil.save(path, "JPEG", quality=95, exif=exif)


@pytest.fixture
def shoot(tmp_path, monkeypatch):
    monkeypatch.setattr(cg, "CostumeSegmenter", _FakeSegmenter)
    day = datetime(2026, 9, 20, 10, 0)
    plan = [(0, RED, WIG_PINK), (1, RED, WIG_PINK), (2, BLUE, WIG_WHITE), (3, BLUE, WIG_WHITE),
            (200, RED, WIG_PINK)]
    for n, (m, costume, wig) in enumerate(plan):
        _write_shot(tmp_path / f"DSCF{n:04d}.JPG", _person(costume, wig), day + timedelta(minutes=m))
    (tmp_path / "DSCF0000.xmp").write_text("<x/>")
    return tmp_path


def test_main_previews_and_writes_report(shoot, capsys):
    assert cg.main([str(shoot)]) == 0
    out = capsys.readouterr().out
    assert "2 cosplayer group(s)" in out
    assert (shoot / cg.REPORT_DIR / "index.html").exists()
    data = json.loads((shoot / cg.REPORT_DIR / "groups.json").read_text())
    sizes = [sum(len(s["shots"]) for s in g["sessions"]) for g in data["groups"]]
    assert sizes == [3, 2]
    assert not (shoot / cg.OUTPUT_DIR).exists()  # preview moves nothing


def test_main_move_and_undo(shoot, capsys):
    assert cg.main([str(shoot), "--move", "--no-report"]) == 0
    first = shoot / cg.OUTPUT_DIR / "person-01"
    assert sorted(p.name for p in first.iterdir()) == ["DSCF0000.JPG", "DSCF0000.xmp",
                                                        "DSCF0001.JPG", "DSCF0004.JPG"]
    manifest = shoot / cg.OUTPUT_DIR / cg.MANIFEST_NAME
    assert cg.main(["--undo", str(manifest)]) == 0
    assert (shoot / "DSCF0004.JPG").exists()
    assert not (shoot / cg.OUTPUT_DIR).exists()


def test_main_rescan_skips_earlier_output(shoot, capsys):
    assert cg.main([str(shoot), "--copy", "--no-report"]) == 0
    capsys.readouterr()
    assert cg.main([str(shoot), "-r", "--no-report", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    files = [f for g in data["groups"] for s in g["sessions"] for sh in s["shots"] for f in sh["files"]]
    assert not any(cg.OUTPUT_DIR in f for f in files)


def test_main_rejects_out_of_range_thresholds(shoot):
    with pytest.raises(SystemExit):
        cg.main([str(shoot), "--link", "1.5"])
