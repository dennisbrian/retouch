"""Tests for retouch.review_page — the batch review page backend.

Per CLAUDE.md conventions: synthetic images only (cv2, tmp_path), no model
downloads, no RetouchEngine — ``result_review_meta`` is exercised with fake
result objects (``SimpleNamespace`` / the real ``QAWarning`` dataclass, which
is cheap to construct and carries no model dependency).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from retouch.qa_detectors import QAWarning
from retouch.review_page import (
    PAGE_NAME,
    REVIEW_DIRNAME,
    STATUSES,
    ReviewRecord,
    apply_decisions,
    backfill_records,
    build_review_page,
    load_review_records,
    main,
    record_key,
    result_review_meta,
    review_root_for,
    write_review_record,
)


def _write_image(path: Path, value: int = 120, size=(24, 20)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size
    cv2.imwrite(str(path), np.full((h, w, 3), value, dtype=np.uint8))


def _extract_review_data(html: str) -> dict:
    match = re.search(
        r'<script id="review-data" type="application/json">(.*?)</script>',
        html,
        re.S,
    )
    assert match, "review-data JSON script block not found in review.html"
    return json.loads(match.group(1))


class TestConstants:
    def test_constants_match_spec(self):
        assert REVIEW_DIRNAME == ".retouch-review"
        assert PAGE_NAME == "review.html"
        assert set(STATUSES) == {"done", "qa_fail", "failed", "skipped"}


class TestReviewRootFor:
    def test_uses_output_dir_when_given(self, tmp_path):
        input_path = tmp_path / "in"
        output_dir = tmp_path / "out"
        assert review_root_for(output_dir, input_path) == output_dir

    def test_uses_input_dir_when_input_is_a_directory(self, tmp_path):
        input_dir = tmp_path / "in"
        input_dir.mkdir()
        assert review_root_for(None, input_dir) == input_dir

    def test_uses_input_parent_when_input_is_a_file(self, tmp_path):
        input_file = tmp_path / "in" / "a.jpg"
        input_file.parent.mkdir()
        input_file.touch()
        assert review_root_for(None, input_file) == input_file.parent


class TestReviewRecordRoundTrip:
    def test_to_dict_from_dict_roundtrip(self):
        rec = ReviewRecord(
            source="/in/a.jpg",
            output="/out/a.jpg",
            compare="/out/a_compare.jpg",
            status="done",
            recipe="cosplay",
            faces=[[1, 2, 3, 4]],
            qa=[{
                "detector": "halo", "score": 0.5, "threshold": 0.6,
                "flagged": False, "message": "ok", "available": True,
            }],
            error=None,
            elapsed_s=1.5,
            created_at="2026-09-24T00:00:00Z",
            relative="a.jpg",
        )
        restored = ReviewRecord.from_dict(rec.to_dict())
        assert restored == rec

    def test_from_dict_ignores_unknown_keys(self):
        d = {"source": "/in/a.jpg", "totally_new_field_from_the_future": 123}
        rec = ReviewRecord.from_dict(d)
        assert rec.source == "/in/a.jpg"

    def test_default_status_is_a_known_status(self):
        rec = ReviewRecord(source="/in/a.jpg")
        assert rec.status in STATUSES

    def test_default_qa_recorded_is_true(self):
        # backfill_records is documented to set qa_recorded=False explicitly;
        # ordinary records (real QA run) default to True.
        rec = ReviewRecord(source="/in/a.jpg")
        assert rec.qa_recorded is True


class TestRecordKey:
    def test_deterministic_for_same_path(self):
        p = Path("/in/set1/DSCF4463.JPG")
        assert record_key(p) == record_key(p)
        assert record_key(p) == record_key(str(p))

    def test_different_paths_get_different_keys(self):
        assert record_key("/in/a/DSCF1.jpg") != record_key("/in/b/DSCF1.jpg")

    def test_collision_safe_for_same_stem_different_dirs(self):
        k1 = record_key("/in/a/img.jpg")
        k2 = record_key("/in/b/img.jpg")
        assert k1 != k2

    def test_filesystem_safe(self):
        key = record_key("/in/weird name (1)!@#.jpg")
        for bad in ("/", "\\", " ", "!", "@", "#", "(", ")"):
            assert bad not in key


class TestWriteReviewRecord:
    def test_writes_json_under_records_dir(self, tmp_path):
        rec = ReviewRecord(source=str(tmp_path / "a.jpg"), relative="a.jpg")
        path = write_review_record(tmp_path, rec)
        assert path.exists()
        assert path.parent == tmp_path / REVIEW_DIRNAME / "records"
        data = json.loads(path.read_text())
        assert data["source"] == rec.source

    def test_fills_created_at_when_empty(self, tmp_path):
        rec = ReviewRecord(source="/in/a.jpg", relative="a.jpg")
        assert rec.created_at == ""
        path = write_review_record(tmp_path, rec)
        data = json.loads(path.read_text())
        assert data["created_at"]

    def test_preserves_explicit_created_at(self, tmp_path):
        rec = ReviewRecord(
            source="/in/a.jpg", relative="a.jpg", created_at="2020-01-01T00:00:00Z",
        )
        path = write_review_record(tmp_path, rec)
        data = json.loads(path.read_text())
        assert data["created_at"] == "2020-01-01T00:00:00Z"

    def test_no_leftover_temp_files(self, tmp_path):
        rec = ReviewRecord(source="/in/a.jpg", relative="a.jpg")
        write_review_record(tmp_path, rec)
        records_dir = tmp_path / REVIEW_DIRNAME / "records"
        assert list(records_dir.glob("*.tmp*")) == []

    def test_uses_atomic_replace(self, tmp_path, monkeypatch):
        calls = []
        orig_replace = os.replace

        def spy(src, dst):
            calls.append((src, dst))
            return orig_replace(src, dst)

        monkeypatch.setattr("retouch.review_page.os.replace", spy)
        rec = ReviewRecord(source="/in/a.jpg", relative="a.jpg")
        write_review_record(tmp_path, rec)
        assert len(calls) == 1


class TestLoadReviewRecords:
    def test_sorted_by_relative_or_source(self, tmp_path):
        for name in ["b.jpg", "a.jpg", "c.jpg"]:
            write_review_record(
                tmp_path, ReviewRecord(source=f"/in/{name}", relative=name),
            )
        records = load_review_records(tmp_path)
        assert [r.relative for r in records] == ["a.jpg", "b.jpg", "c.jpg"]

    def test_skips_corrupt_file_and_logs_warning(self, tmp_path, caplog):
        write_review_record(
            tmp_path, ReviewRecord(source="/in/a.jpg", relative="a.jpg"),
        )
        records_dir = tmp_path / REVIEW_DIRNAME / "records"
        (records_dir / "corrupt.json").write_text("{not valid json")

        with caplog.at_level(logging.WARNING):
            records = load_review_records(tmp_path)
        assert [r.relative for r in records] == ["a.jpg"]

    def test_empty_root_returns_empty_list(self, tmp_path):
        assert load_review_records(tmp_path) == []


class TestResultReviewMeta:
    def test_extracts_faces_and_qa_at_scale_1(self):
        fc = SimpleNamespace(face_data=SimpleNamespace(bbox=(10, 20, 30, 40)))
        qa = QAWarning(
            detector="halo", score=0.5, flagged=False, message="ok",
            threshold=0.6,
        )
        result = SimpleNamespace(face_contexts=[fc], qa=[qa])
        meta = result_review_meta(result, scale=1.0)
        assert meta["faces"] == [[10, 20, 30, 40]]
        assert len(meta["qa"]) == 1
        assert meta["qa"][0]["detector"] == "halo"
        assert meta["qa"][0]["flagged"] is False
        assert meta["qa"][0]["threshold"] == 0.6

    def test_divides_bbox_by_scale_for_native_coords(self):
        fc = SimpleNamespace(face_data=SimpleNamespace(bbox=(10, 20, 30, 40)))
        result = SimpleNamespace(face_contexts=[fc], qa=[])
        meta = result_review_meta(result, scale=0.5)
        assert meta["faces"] == [[20, 40, 60, 80]]

    def test_missing_attributes_return_empty_lists(self):
        # Plain ndarray as returned by the global-only path: no .qa/.face_contexts.
        result = np.zeros((4, 4, 3), dtype=np.uint8)
        meta = result_review_meta(result)
        assert meta == {"faces": [], "qa": []}

    def test_none_valued_attributes_return_empty_lists(self):
        result = SimpleNamespace(face_contexts=None, qa=None)
        meta = result_review_meta(result)
        assert meta == {"faces": [], "qa": []}

    def test_never_raises_on_arbitrary_object(self):
        meta = result_review_meta(object())
        assert meta == {"faces": [], "qa": []}

    def test_default_scale_is_1(self):
        fc = SimpleNamespace(face_data=SimpleNamespace(bbox=(4, 4, 8, 8)))
        result = SimpleNamespace(face_contexts=[fc], qa=[])
        assert result_review_meta(result)["faces"] == [[4, 4, 8, 8]]


class TestBuildReviewPage:
    def _make_record(self, root, input_dir, name="set1/DSCF0001.jpg", value=50):
        src = input_dir / name
        out = root / name
        _write_image(src, value=value)
        _write_image(out, value=value + 10)
        rec = ReviewRecord(
            source=str(src),
            output=str(out),
            status="done",
            recipe="cosplay",
            relative=name,
            faces=[[2, 2, 10, 8]],
            qa=[{
                "detector": "plastic_skin", "score": 0.1, "threshold": 0.6,
                "flagged": False, "message": "fine", "available": True,
            }],
            elapsed_s=12.3,
        )
        write_review_record(root, rec)
        return rec, src, out

    def test_builds_html_with_relative_thumb_and_output_paths(self, tmp_path):
        root = tmp_path / "out"
        input_dir = tmp_path / "in"
        root.mkdir()
        rec, src, out = self._make_record(root, input_dir)

        page = build_review_page(root, title="Shoot 1", workers=1)
        assert page == root / PAGE_NAME
        assert page.exists()

        data = _extract_review_data(page.read_text(encoding="utf-8"))
        assert data["title"] == "Shoot 1"
        assert len(data["items"]) == 1
        item = data["items"][0]
        assert item["name"] == "set1/DSCF0001.jpg"
        assert item["status"] == "done"
        assert item["recipe"] == "cosplay"

        # `root` is documented as the absolute path (localStorage/apply_command
        # context); every per-item path must be *relative* to review.html.
        assert Path(data["root"]) == root.resolve() or data["root"] == str(root)

        for key in ("before", "after"):
            rel = item[key]
            assert rel is not None
            assert not Path(rel).is_absolute()
            assert (root / rel).exists()

        assert item["output_href"] == "set1/DSCF0001.jpg"
        assert (root / item["output_href"]).resolve() == out.resolve()

        assert len(item["faces"]) == 1
        for key in ("before", "after"):
            rel = item["faces"][0][key]
            assert not Path(rel).is_absolute()
            assert (root / rel).exists()

        assert item["qa"][0]["detector"] == "plastic_skin"
        assert item["flagged"] is False

    def test_thumbs_written_under_review_dirname(self, tmp_path):
        root = tmp_path / "out"
        input_dir = tmp_path / "in"
        root.mkdir()
        self._make_record(root, input_dir)

        build_review_page(root, workers=1)
        thumbs_dir = root / REVIEW_DIRNAME / "thumbs"
        assert thumbs_dir.is_dir()
        assert list(thumbs_dir.glob("*_after.jpg"))
        assert list(thumbs_dir.glob("*_before.jpg"))

    def test_grid_thumb_is_small_and_used_by_card(self, tmp_path):
        from retouch.review_page import GRID_EDGE, build_page_data

        root = tmp_path / "out"
        input_dir = tmp_path / "in"
        root.mkdir()
        self._make_record(root, input_dir)

        build_review_page(root, workers=1)
        item = build_page_data(root)["items"][0]
        assert item["grid"].endswith("_after_grid.jpg")
        grid = cv2.imread(str(root / item["grid"]))
        assert grid is not None
        assert max(grid.shape[:2]) <= GRID_EDGE

    def test_summary_counts_match_records(self, tmp_path):
        root = tmp_path / "out"
        input_dir = tmp_path / "in"
        root.mkdir()
        self._make_record(root, input_dir, name="a.jpg")
        # A second, failed record with no output.
        write_review_record(root, ReviewRecord(
            source=str(input_dir / "b.jpg"), output=None, status="failed",
            error="boom", relative="b.jpg",
        ))

        page = build_review_page(root, workers=1)
        data = _extract_review_data(page.read_text(encoding="utf-8"))
        assert data["summary"]["total"] == 2
        assert data["summary"]["done"] == 1
        assert data["summary"]["failed"] == 1

    def test_stale_thumb_is_regenerated_on_source_change(self, tmp_path):
        root = tmp_path / "out"
        input_dir = tmp_path / "in"
        root.mkdir()
        name = "a.jpg"
        src = input_dir / name
        out = root / name
        _write_image(src, value=10)
        _write_image(out, value=10)
        write_review_record(root, ReviewRecord(
            source=str(src), output=str(out), status="done", relative=name,
        ))

        build_review_page(root, workers=1)
        thumb = root / REVIEW_DIRNAME / "thumbs" / f"{record_key(src)}_after.jpg"
        assert thumb.exists()
        first_bytes = thumb.read_bytes()

        # Bump the output's mtime into the future *and* change its pixel
        # content, so a regenerated thumb is unambiguously detectable.
        future = time.time() + 10
        _write_image(out, value=240)
        os.utime(out, (future, future))

        build_review_page(root, workers=1)
        assert thumb.read_bytes() != first_bytes

    def test_missing_output_file_is_tolerated(self, tmp_path):
        root = tmp_path / "out"
        input_dir = tmp_path / "in"
        root.mkdir()
        src = input_dir / "a.jpg"
        _write_image(src)
        write_review_record(root, ReviewRecord(
            source=str(src), output=str(root / "does_not_exist.jpg"),
            status="qa_fail", relative="a.jpg",
        ))

        page = build_review_page(root, workers=1)  # must not raise
        data = _extract_review_data(page.read_text(encoding="utf-8"))
        item = data["items"][0]
        assert item["status"] == "qa_fail"
        assert item["after"] is None

    def test_missing_source_file_is_tolerated(self, tmp_path):
        root = tmp_path / "out"
        root.mkdir()
        write_review_record(root, ReviewRecord(
            source=str(tmp_path / "missing_src.jpg"), output=None,
            status="failed", error="could not read", relative="missing.jpg",
        ))

        page = build_review_page(root, workers=1)  # must not raise
        assert page.exists()
        data = _extract_review_data(page.read_text(encoding="utf-8"))
        assert data["items"][0]["error"] == "could not read"

    def test_no_records_still_builds_a_page(self, tmp_path):
        root = tmp_path
        page = build_review_page(root, workers=1)
        assert page.exists()
        data = _extract_review_data(page.read_text(encoding="utf-8"))
        assert data["items"] == []
        assert data["summary"]["total"] == 0


class TestScriptEscapingSecurity:
    """`</script>` inside embedded item data must never break out of the
    `<script id="review-data" type="application/json">` tag."""

    def test_item_name_with_script_tag_is_escaped_and_recoverable(self, tmp_path):
        malicious = "</script><img src=x onerror=alert(1)>"
        write_review_record(tmp_path, ReviewRecord(
            source=str(tmp_path / "src.jpg"), output=None, status="failed",
            error=malicious, relative=malicious,
        ))

        page = build_review_page(tmp_path, workers=1)
        html = page.read_text(encoding="utf-8")

        # The raw breakout sequence must never appear verbatim.
        assert "</script><img" not in html

        data = _extract_review_data(html)
        names = [item["name"] for item in data["items"]]
        assert malicious in names
        errors = [item["error"] for item in data["items"]]
        assert malicious in errors


class TestBackfillRecords:
    def test_pairs_outputs_with_sources_excludes_compare(self, tmp_path):
        source_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        _write_image(source_dir / "a.jpg", value=10)
        _write_image(source_dir / "b.jpg", value=20)
        _write_image(output_dir / "a.jpg", value=15)
        _write_image(output_dir / "b.jpg", value=25)
        _write_image(output_dir / "a_compare.jpg", value=99)

        count = backfill_records(output_dir, source_dir)
        assert count == 2

        records = load_review_records(output_dir)
        assert len(records) == 2
        for r in records:
            assert r.status == "done"
            assert r.qa == []
            assert r.qa_recorded is False

    def test_recursive_pairs_nested_outputs(self, tmp_path):
        source_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        (source_dir / "set1").mkdir(parents=True)
        (output_dir / "set1").mkdir(parents=True)
        _write_image(source_dir / "set1" / "c.jpg")
        _write_image(output_dir / "set1" / "c.jpg")

        count = backfill_records(output_dir, source_dir, recursive=True)
        assert count == 1

    def test_does_not_write_a_record_for_an_output_with_no_matching_source(self, tmp_path):
        source_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        source_dir.mkdir()
        output_dir.mkdir()
        _write_image(output_dir / "orphan.jpg")

        count = backfill_records(output_dir, source_dir)
        assert count == 0


class TestApplyDecisions:
    def _record_with_output(self, root, rel="set1/a.jpg", value=77, compare=False):
        out = root / rel
        _write_image(out, value=value)
        compare_path = None
        if compare:
            compare_path = out.with_name(out.stem + "_compare" + out.suffix)
            _write_image(compare_path, value=value + 1)
        rec = ReviewRecord(
            source=str(root / "src" / rel), output=str(out),
            compare=str(compare_path) if compare_path else None,
            status="done", relative=rel,
        )
        write_review_record(root, rec)
        return rec, out, compare_path

    def test_copy_picks_preserves_relative_subdirs(self, tmp_path):
        root = tmp_path
        rec, out, _ = self._record_with_output(root)
        key = record_key(rec.source)

        result = apply_decisions(root, {key: "pick"})
        assert result["picked"] == 1
        picked = root / "picks" / "set1" / "a.jpg"
        assert picked.exists()
        assert picked.read_bytes() == out.read_bytes()
        assert out.exists()  # source output untouched

    def test_reject_does_not_move_unless_flag_set(self, tmp_path):
        root = tmp_path
        rec, out, compare = self._record_with_output(root, rel="b.jpg", compare=True)
        key = record_key(rec.source)

        result = apply_decisions(root, {key: "reject"})
        # Without move_rejects a "reject" decision is a deliberate no-op
        # (nothing to move, nothing counted).
        assert result["rejected"] == 0
        assert out.exists()
        assert compare.exists()
        assert not (root / "rejected" / "b.jpg").exists()

    def test_reject_moves_output_and_compare_when_flag_set(self, tmp_path):
        root = tmp_path
        rec, out, compare = self._record_with_output(root, rel="c.jpg", compare=True)
        key = record_key(rec.source)

        result = apply_decisions(root, {key: "reject"}, move_rejects=True)
        assert result["rejected"] == 1
        assert (root / "rejected" / "c.jpg").exists()
        assert (root / "rejected" / "c_compare.jpg").exists()
        assert not out.exists()
        assert not compare.exists()

    def test_never_overwrites_existing_destination(self, tmp_path):
        root = tmp_path
        rec, out, _ = self._record_with_output(root, rel="d.jpg", value=1)
        key = record_key(rec.source)

        dest = root / "picks" / "d.jpg"
        dest.parent.mkdir(parents=True)
        _write_image(dest, value=250)
        existing_bytes = dest.read_bytes()

        result = apply_decisions(root, {key: "pick"})
        assert result["exists"] == 1
        assert result["picked"] == 0
        assert dest.read_bytes() == existing_bytes

    def test_missing_output_is_counted_not_raised(self, tmp_path):
        root = tmp_path
        rec = ReviewRecord(
            source=str(root / "src" / "e.jpg"), output=str(root / "nope.jpg"),
            status="done", relative="e.jpg",
        )
        write_review_record(root, rec)
        key = record_key(rec.source)

        result = apply_decisions(root, {key: "pick"})
        assert result["missing"] == 1
        assert result["picked"] == 0

    def test_accepts_decisions_from_exported_json_file(self, tmp_path):
        root = tmp_path
        rec, out, _ = self._record_with_output(root, rel="f.jpg")
        key = record_key(rec.source)

        decisions_path = tmp_path / "decisions.json"
        decisions_path.write_text(json.dumps({
            "version": 1, "batch_id": "abc123", "root": str(root),
            "decisions": {key: "pick"},
        }))

        result = apply_decisions(root, decisions_path)
        assert result["picked"] == 1
        assert (root / "picks" / "f.jpg").exists()

    def test_custom_picks_and_rejects_dirs(self, tmp_path):
        root = tmp_path
        rec, out, _ = self._record_with_output(root, rel="g.jpg")
        key = record_key(rec.source)

        result = apply_decisions(root, {key: "pick"}, picks_dir="chosen")
        assert result["picked"] == 1
        assert (root / "chosen" / "g.jpg").exists()


class TestMainArgvParsing:
    def test_build_subcommand(self, tmp_path):
        root = tmp_path
        src = root / "a.jpg"
        _write_image(src)
        write_review_record(root, ReviewRecord(
            source=str(src), output=str(src), status="done", relative="a.jpg",
        ))

        rc = main(["build", str(root), "--workers", "1"])
        assert rc in (0, None)
        assert (root / PAGE_NAME).exists()

    def test_apply_subcommand(self, tmp_path):
        root = tmp_path
        out = root / "a.jpg"
        _write_image(out)
        rec = ReviewRecord(
            source=str(root / "src" / "a.jpg"), output=str(out),
            status="done", relative="a.jpg",
        )
        write_review_record(root, rec)
        key = record_key(rec.source)
        decisions_path = tmp_path / "decisions.json"
        decisions_path.write_text(json.dumps({"decisions": {key: "pick"}}))

        rc = main(["apply", str(root), str(decisions_path)])
        assert rc in (0, None)
        assert (root / "picks" / "a.jpg").exists()

    def test_apply_move_rejects_flag(self, tmp_path):
        root = tmp_path
        out = root / "b.jpg"
        _write_image(out)
        rec = ReviewRecord(
            source=str(root / "src" / "b.jpg"), output=str(out),
            status="done", relative="b.jpg",
        )
        write_review_record(root, rec)
        key = record_key(rec.source)
        decisions_path = tmp_path / "decisions.json"
        decisions_path.write_text(json.dumps({"decisions": {key: "reject"}}))

        rc = main(["apply", str(root), str(decisions_path), "--move-rejects"])
        assert rc in (0, None)
        assert (root / "rejected" / "b.jpg").exists()
        assert not out.exists()

    def test_unknown_command_fails(self):
        try:
            rc = main(["bogus-subcommand"])
        except SystemExit as exc:
            rc = exc.code
        assert rc not in (0, None)
