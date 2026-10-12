"""cli.py live-progress wiring: keys, result metadata, temp-file hygiene."""

import queue
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import cli
from cli import (
    _ATOMIC_TEMP_RE,
    _progress_key,
    _remove_partial_outputs,
    _result_info,
    find_images,
)


class TestProgressKey:
    def test_plain_name_without_root(self, tmp_path):
        assert _progress_key(tmp_path / "a" / "DSCF1.jpg") == "DSCF1.jpg"

    def test_relative_path_under_recursive_root(self, tmp_path):
        root = tmp_path.resolve()
        img = root / "set1" / "DSCF1.jpg"
        assert _progress_key(img, root) == "set1/DSCF1.jpg"

    def test_same_name_in_two_folders_stays_distinct(self, tmp_path):
        root = tmp_path.resolve()
        a = _progress_key(root / "set1" / "DSCF1.jpg", root)
        b = _progress_key(root / "set2" / "DSCF1.jpg", root)
        assert a != b

    def test_outside_root_falls_back_to_name(self, tmp_path):
        root = (tmp_path / "in").resolve()
        assert _progress_key(tmp_path / "other" / "x.jpg", root) == "x.jpg"


class TestResultInfo:
    def test_copies_faces_timings_and_qa(self):
        warning = SimpleNamespace(detector="banding", score=0.83, threshold=0.15,
                                  flagged=True, message="Banding visible")
        result = SimpleNamespace(face_count=2,
                                 timings={"per_face": 1234.56, "note": "x"},
                                 qa=[warning])
        info = {}
        _result_info(result, info)
        assert info["faces"] == 2
        assert info["timings"] == {"per_face": 1234.6}
        assert info["qa"] == [{"detector": "banding", "score": 0.83,
                               "threshold": 0.15, "flagged": True,
                               "message": "Banding visible"}]

    def test_plain_ndarray_gives_empty_qa(self):
        info = {}
        _result_info(np.zeros((2, 2, 3), np.uint8), info)
        assert info == {"qa": []}


class TestPartialOutputs:
    def test_find_images_skips_hidden_partial_writes(self, tmp_path):
        img = np.zeros((4, 4, 3), np.uint8)
        cv2.imwrite(str(tmp_path / "DSCF1.jpg"), img)
        cv2.imwrite(str(tmp_path / ".DSCF2.tmp-123-deadbeef.jpg"), img)
        assert [p.name for p in find_images(str(tmp_path), False)] == ["DSCF1.jpg"]

    def test_remove_partial_outputs_only_touches_atomic_temps(self, tmp_path):
        (tmp_path / "sub").mkdir()
        temp = tmp_path / "sub" / ".DSCF2.tmp-123-deadbeef.jpg"
        keep = [tmp_path / "DSCF1.jpg", tmp_path / ".retouch-progress.json",
                tmp_path / ".notes.tmp-backup.txt"]
        for p in [temp, *keep]:
            p.write_bytes(b"x")
        _remove_partial_outputs(tmp_path)
        assert not temp.exists()
        assert all(p.exists() for p in keep)

    def test_temp_pattern_matches_io_writer_names(self, tmp_path):
        from retouch.io import _atomic_output_path

        with _atomic_output_path(tmp_path / "DSCF0001.jpg") as tmp:
            assert _ATOMIC_TEMP_RE.match(tmp.name)
            tmp.write_bytes(b"x")

    def test_missing_dir_is_a_no_op(self, tmp_path):
        _remove_partial_outputs(tmp_path / "nope")
        _remove_partial_outputs(None)


class TestProcessSingleEvents:
    def test_skip_streams_start_and_returns_info(self, tmp_path, monkeypatch):
        src = tmp_path / "in" / "DSCF1.jpg"
        src.parent.mkdir()
        cv2.imwrite(str(src), np.zeros((8, 8, 3), np.uint8))
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        cv2.imwrite(str(out_dir / "DSCF1.jpg"), np.zeros((8, 8, 3), np.uint8))
        q = queue.Queue()
        monkeypatch.setattr(cli, "_worker_progress_q", q)
        args = (src, out_dir, {}, "jpg", 95, False, False, None, False, False,
                8, False, None, False, False, 0.0, 1.0, "rawpy", None, 95,
                1.0, False, None, None, False)
        name, status, info = cli._process_single(args)
        assert (name, status) == ("DSCF1.jpg", "skipped")
        assert info["seconds"] >= 0
        event = q.get_nowait()
        assert event["image"] == "DSCF1.jpg" and event["kind"] == "start"

    def test_no_queue_means_no_events(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cli, "_worker_progress_q", None)
        src = tmp_path / "missing.jpg"
        args = (src, tmp_path, {}, "jpg", 95, False, False, None, False, False,
                8, False, None, False, False, 0.0, 1.0, "rawpy", None, 95,
                1.0, False, None, None, False)
        name, status, info = cli._process_single(args)
        assert name == "missing.jpg" and status.startswith("failed")
        assert Path(tmp_path).exists()
