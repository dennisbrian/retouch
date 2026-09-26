"""Shoot-wide near-duplicate detection (retouch/duplicates.py).

Synthetic scenes stand in for photos: a textured backdrop and a stick-figure
subject whose arm position is the "pose". No models or network.
"""

import json

import cv2
import numpy as np
import pytest
from PIL import Image

from retouch.duplicates import (
    DuplicateGroup,
    compare_frames,
    find_duplicates,
    frame_signature,
    list_images,
    main,
    move_extras,
    signature_from_array,
    write_report,
)


def _scene(arm_angle=30.0, seed=0, size=(480, 720), backdrop_seed=0):
    """RGB uint8 portrait-ish frame: textured backdrop plus a figure."""
    w, h = size
    rng = np.random.default_rng(backdrop_seed)
    base = cv2.resize(rng.uniform(70, 170, (5, 4, 3)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    img = base + cv2.GaussianBlur(rng.normal(0, 6, (h, w, 3)).astype(np.float32), (0, 0), 2)
    # Soft backdrop structure (a window frame and a shelf), as in a con hall.
    cv2.line(img, (0, 120), (w, 135), (190, 190, 185), 4)
    cv2.rectangle(img, (30, 40), (150, 260), (20, 20, 30), 5)
    cx = w // 2
    skin = (205, 160, 140)
    cv2.circle(img, (cx, 200), 48, skin, -1)
    cv2.rectangle(img, (cx - 60, 250), (cx + 60, 520), (40, 40, 160), -1)
    cv2.line(img, (cx - 30, 520), (cx - 40, 690), (30, 30, 30), 26)
    cv2.line(img, (cx + 30, 520), (cx + 40, 690), (30, 30, 30), 26)
    # Left arm fixed, right arm is the pose.
    cv2.line(img, (cx - 60, 270), (cx - 120, 430), skin, 30)
    a = np.radians(arm_angle)
    end = (int(cx + 60 + 170 * np.cos(a)), int(270 + 170 * np.sin(a)))
    cv2.line(img, (cx + 60, 270), end, skin, 30)
    noise = np.random.default_rng(seed).normal(0, 2.0, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def _reframe(img, dx=0.04, dy=0.02, zoom=1.08, gain=1.0):
    h, w = img.shape[:2]
    ch, cw = int(h / zoom), int(w / zoom)
    y0 = int((h - ch) / 2 + dy * h)
    x0 = int((w - cw) / 2 + dx * w)
    out = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(out.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def _sig(img, path=""):
    return signature_from_array(img, path=path)


def _save(path, rgb, **kw):
    Image.fromarray(rgb).save(path, "JPEG", quality=92, **kw)
    return path


class TestCompareFrames:
    def test_identical_frames_are_duplicates(self):
        img = _scene()
        ev = compare_frames(_sig(img), _sig(img))
        assert ev.duplicate and ev.similarity > 0.99 and ev.changed_fraction < 0.02

    def test_reframed_and_brighter_copy_is_duplicate(self):
        img = _scene()
        ev = compare_frames(_sig(img), _sig(_reframe(img, gain=1.15)))
        assert ev.duplicate, ev.to_dict()
        assert 1.0 < ev.scale <= 1.1

    def test_raised_arm_is_a_different_pose(self):
        # Mutation guard for the local pose check: the global score alone
        # passes this pair.
        a, b = _scene(arm_angle=30), _scene(arm_angle=-60, seed=1)
        ev = compare_frames(_sig(a), _sig(b))
        assert ev.similarity >= 0.9, "global match should not be what rejects it"
        assert not ev.duplicate and ev.reason == "pose_changed"

    def test_different_backdrop_is_not_duplicate(self):
        ev = compare_frames(_sig(_scene(backdrop_seed=0)), _sig(_scene(backdrop_seed=7)))
        assert not ev.duplicate

    def test_portrait_and_landscape_never_match(self):
        img = _scene()
        ev = compare_frames(_sig(img), _sig(np.ascontiguousarray(np.rot90(img))))
        assert not ev.duplicate and ev.reason == "different_aspect"

    def test_colour_gate_is_relative_not_absolute(self):
        # A darker exposure of the same frame keeps its chroma and still matches.
        img = _scene()
        ev = compare_frames(_sig(img), _sig(_reframe(img, 0, 0, 1.0, gain=0.7)))
        assert ev.duplicate, ev.to_dict()


class TestFindDuplicates:
    def test_groups_repeats_and_keeps_poses_apart(self, tmp_path):
        base = _scene(arm_angle=30)
        paths = [
            _save(tmp_path / "a1.jpg", base),
            _save(tmp_path / "a2.jpg", _reframe(base)),
            _save(tmp_path / "b1.jpg", _scene(arm_angle=-60, seed=2)),
            _save(tmp_path / "c1.jpg", _scene(backdrop_seed=5, seed=3)),
        ]
        groups = find_duplicates(paths)
        assert len(groups) == 1
        names = sorted(p.split("/")[-1] for p in groups[0].asset_paths)
        assert names == ["a1.jpg", "a2.jpg"]
        assert groups[0].review_required

    def test_keeper_is_the_sharpest_frame(self, tmp_path):
        base = _scene()
        blurred = cv2.GaussianBlur(base, (0, 0), 2.5)
        paths = [_save(tmp_path / "soft.jpg", blurred), _save(tmp_path / "sharp.jpg", base)]
        (group,) = find_duplicates(paths)
        assert group.keeper.endswith("sharp.jpg")
        assert [p.split("/")[-1] for p in group.extras] == ["soft.jpg"]
        assert group.evidence["ranking"][0]["path"] == group.keeper

    def test_complete_linkage_does_not_chain(self):
        # a~b and b~c but a and c differ: never one group of three.
        a = _scene(arm_angle=30)
        c = _scene(arm_angle=-60, seed=1)
        sigs = [_sig(a, "a"), _sig(a, "b"), _sig(c, "c")]
        groups = find_duplicates(sigs)
        for g in groups:
            assert not ({"a", "c"} <= set(g.asset_paths))

    def test_unreadable_file_is_skipped(self, tmp_path):
        bad = tmp_path / "bad.jpg"
        bad.write_bytes(b"not a jpeg")
        base = _scene()
        paths = [bad, _save(tmp_path / "x.jpg", base), _save(tmp_path / "y.jpg", base)]
        groups = find_duplicates(paths)
        assert len(groups) == 1

    def test_exif_orientation_is_applied(self, tmp_path):
        # Stored sideways with orientation 6 (rotate 90 CW to view): must
        # compare as the upright portrait it is.
        base = _scene()
        stored = np.ascontiguousarray(np.rot90(base, k=1))  # CCW, undone by tag 6
        exif = Image.Exif()
        exif[0x0112] = 6
        rotated = _save(tmp_path / "rot.jpg", stored, exif=exif)
        upright = _save(tmp_path / "up.jpg", base)
        sig = frame_signature(rotated)
        assert sig.height > sig.width
        assert compare_frames(sig, frame_signature(upright)).duplicate


class TestReportAndMove:
    def _groups(self, tmp_path):
        base = _scene()
        paths = [_save(tmp_path / "one.jpg", base), _save(tmp_path / "two.jpg", _reframe(base))]
        return find_duplicates(paths)

    def test_report_is_written_with_relative_paths(self, tmp_path):
        groups = self._groups(tmp_path)
        index = write_report(groups, tmp_path, tmp_path / "duplicates-report", scanned=2)
        page = index.read_text()
        assert "thumbs/dup-0001/000.jpg" in page and str(tmp_path) not in page
        data = json.loads((tmp_path / "duplicates-report" / "duplicates.json").read_text())
        assert data["scanned"] == 2 and len(data["groups"]) == 1
        assert sorted(data["groups"][0]["relative_paths"]) == ["one.jpg", "two.jpg"]
        assert (tmp_path / "duplicates-report" / "thumbs" / "dup-0001" / "001.jpg").is_file()

    def test_move_extras_moves_only_non_keepers_and_never_overwrites(self, tmp_path):
        groups = self._groups(tmp_path)
        (extra,) = groups[0].extras
        name = extra.split("/")[-1]
        (tmp_path / "duplicates").mkdir()
        (tmp_path / "duplicates" / name).write_bytes(b"existing")
        counts = move_extras(groups, tmp_path)
        assert counts == {"moved": 0, "exists": 1, "missing": 0}
        assert (tmp_path / name).is_file()
        (tmp_path / "duplicates" / name).unlink()
        counts = move_extras(groups, tmp_path)
        assert counts["moved"] == 1
        assert (tmp_path / "duplicates" / name).is_file() and not (tmp_path / name).exists()
        assert (tmp_path / groups[0].keeper.split("/")[-1]).is_file()

    def test_list_images_skips_own_folders(self, tmp_path):
        base = _scene()
        _save(tmp_path / "a.jpg", base)
        (tmp_path / "duplicates").mkdir()
        _save(tmp_path / "duplicates" / "b.jpg", base)
        (tmp_path / "sub").mkdir()
        _save(tmp_path / "sub" / "c.jpg", base)
        assert [p.name for p in list_images(tmp_path)] == ["a.jpg"]
        assert [p.name for p in list_images(tmp_path, recursive=True)] == ["a.jpg", "c.jpg"]


class TestCli:
    def test_report_only_by_default(self, tmp_path, capsys):
        base = _scene()
        _save(tmp_path / "one.jpg", base)
        _save(tmp_path / "two.jpg", _reframe(base))
        assert main([str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "Found 1 duplicate groups (1 extra frames)" in out
        assert (tmp_path / "duplicates-report" / "index.html").is_file()
        assert (tmp_path / "one.jpg").is_file() and (tmp_path / "two.jpg").is_file()

    def test_move_extras_flag(self, tmp_path):
        base = _scene()
        _save(tmp_path / "one.jpg", base)
        _save(tmp_path / "two.jpg", _reframe(base))
        assert main([str(tmp_path), "--move-extras"]) == 0
        assert len(list((tmp_path / "duplicates").glob("*.jpg"))) == 1

    def test_missing_folder_is_an_error(self, tmp_path, capsys):
        assert main([str(tmp_path / "nope")]) == 1
        assert "folder not found" in capsys.readouterr().err

    @pytest.mark.parametrize("strict", [False, True])
    def test_strict_is_never_looser(self, tmp_path, strict):
        base = _scene()
        _save(tmp_path / "one.jpg", base)
        _save(tmp_path / "two.jpg", _reframe(base, dx=0.08, dy=0.05, zoom=1.1))
        args = [str(tmp_path)] + (["--strict"] if strict else [])
        assert main(args) == 0
        data = json.loads((tmp_path / "duplicates-report" / "duplicates.json").read_text())
        if strict:
            assert len(data["groups"]) <= 1


def test_group_extras_excludes_keeper():
    g = DuplicateGroup("dup-0001", ("a", "b", "c"), "b", (), {})
    assert g.extras == ("a", "c")
