"""Tests for retouch/batch_progress.py (no engine, fake clock, queue.Queue)."""

import json
import os
import pickle
import queue
import threading
import time

import numpy as np
import pytest

from retouch.batch_progress import (
    BatchProgress,
    ProgressReporter,
    emit,
    format_duration,
    image_megapixels,
    make_event_sink,
)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = float(t)
        self._lock = threading.Lock()

    def __call__(self):
        with self._lock:
            return self.t

    def advance(self, dt):
        with self._lock:
            self.t += dt


def ev(image, kind, t, **info):
    return {"image": image, "kind": kind, "t": t, **info}


def finish(bp, name, t0, secs, status="done", **info):
    bp.on_event(ev(name, "start", t0, worker=0))
    bp.on_event(ev(name, "finish", t0 + secs, status=status, seconds=secs, **info))


# ---------------------------------------------------------------- helpers

class TestFormatDuration:
    @pytest.mark.parametrize("secs,expected", [
        (None, "?"), (0, "0s"), (45.4, "45s"), (190, "3m10s"), (3725, "1h02m"),
    ])
    def test_format(self, secs, expected):
        assert format_duration(secs) == expected


class TestImageMegapixels:
    def test_small_jpeg(self, tmp_path):
        cv2 = pytest.importorskip("cv2")
        p = tmp_path / "a.jpg"
        assert cv2.imwrite(str(p), np.zeros((300, 400, 3), np.uint8))
        assert image_megapixels(p) == pytest.approx(0.12)
        assert image_megapixels(str(p)) == pytest.approx(0.12)

    def test_garbage_file_returns_none(self, tmp_path):
        p = tmp_path / "junk.jpg"
        p.write_bytes(b"not an image at all" * 10)
        assert image_megapixels(p) is None

    def test_missing_and_garbage_raw_return_none(self, tmp_path):
        assert image_megapixels(tmp_path / "missing.png") is None
        raw = tmp_path / "x.raf"
        raw.write_bytes(b"\x00" * 64)
        assert image_megapixels(raw) is None


# ---------------------------------------------------------------- sinks

class _BrokenQueue:
    def put(self, *_a, **_k):
        raise BrokenPipeError("manager gone")

    def get(self, *_a, **_k):
        raise EOFError

    def get_nowait(self):
        raise EOFError


class TestEventSink:
    def test_sink_puts_event(self):
        q = queue.Queue()
        sink = make_event_sink(q, "DSCF1.jpg")
        sink("stage", {"stage": "per_face"})
        sink("face", {"index": 0, "total": 3})
        e1, e2 = q.get_nowait(), q.get_nowait()
        assert e1["image"] == "DSCF1.jpg" and e1["kind"] == "stage"
        assert e1["stage"] == "per_face" and isinstance(e1["t"], float)
        assert e2["index"] == 0 and e2["total"] == 3

    def test_sink_swallows_broken_queue(self):
        sink = make_event_sink(_BrokenQueue(), "x.jpg")
        sink("stage", {"stage": "grading"})  # must not raise
        sink("stage", None)
        emit(_BrokenQueue(), "x.jpg", "finish", status="done")
        emit(None, "x.jpg", "start")

    def test_sink_is_picklable_and_emit(self):
        sink = pickle.loads(pickle.dumps(make_event_sink(None, "a.jpg")))
        sink("stage", {"stage": "s"})  # q None → no-op
        q = queue.Queue()
        emit(q, "a.jpg", "start", worker=3)
        e = q.get_nowait()
        assert e["kind"] == "start" and e["worker"] == 3 and e["image"] == "a.jpg"


# ---------------------------------------------------------------- state

class TestBatchProgress:
    def test_unknown_mp_uses_median(self):
        bp = BatchProgress([("a", 10.0), ("b", 20.0), ("c", 30.0), ("d", None)],
                           clock=FakeClock())
        assert bp.images["d"]["mp"] == 20.0
        assert BatchProgress([("a", None)], clock=FakeClock()).images["a"]["mp"] == 1.0

    def test_eta_none_until_first_finish_then_math(self):
        clock = FakeClock()
        bp = BatchProgress([("a", 20.0), ("b", 20.0), ("c", 40.0)], clock=clock)
        assert bp.eta_seconds() is None
        bp.on_event(ev("a", "start", 1000.0))
        assert bp.eta_seconds() is None
        bp.on_event(ev("a", "finish", 1100.0, status="done", seconds=100.0))
        # 0.2 MP/s, one worker, 60 MP left → 300 s
        assert bp.done_mp == 20.0 and bp.total_mp == 80.0
        assert bp.eta_seconds() == pytest.approx(300.0)

    def test_eta_excludes_skipped_images_both_sides(self):
        clock = FakeClock()
        bp = BatchProgress([("s1", 50.0), ("s2", 50.0), ("a", 20.0), ("b", 40.0)],
                           clock=clock)
        for n in ("s1", "s2"):
            bp.on_event(ev(n, "start", 1000.0))
            bp.on_event(ev(n, "finish", 1000.01, status="skipped", seconds=0.01))
        # skipped images alone give no rate
        assert bp.eta_seconds() is None
        assert bp.total_mp == 60.0 and bp.done_mp == 0.0
        finish(bp, "a", 1001.0, 100.0)
        assert bp.total_mp == 60.0 and bp.done_mp == 20.0
        # rate 0.2 MP/s from "a" only (skipped 0.01 s must not inflate it)
        assert bp.eta_seconds() == pytest.approx(200.0)
        assert bp.counts["skipped"] == 2 and bp.counts["done"] == 1

    def test_eta_scales_with_observed_parallelism(self):
        bp = BatchProgress([(n, 10.0) for n in "abcdef"], clock=FakeClock())
        for n in "abcd":
            bp.on_event(ev(n, "start", 1000.0))
        bp.on_event(ev("a", "finish", 1100.0, status="done", seconds=100.0))
        # 0.1 MP/s/worker x 4 workers; 50 MP left (b,c,d running + e,f)
        assert bp.eta_seconds() == pytest.approx(50.0 / 0.4)
        # tail: only one image left → parallelism capped at 1
        for n in "bcde":
            bp.on_event(ev(n, "finish", 1100.0, status="done", seconds=100.0))
        assert bp.eta_seconds() == pytest.approx(10.0 / 0.1)
        bp.on_event(ev("f", "finish", 1200.0, status="done", seconds=100.0))
        assert bp.eta_seconds() == 0.0

    def test_status_mapping_and_active(self):
        clock = FakeClock(2000.0)
        bp = BatchProgress([("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", 1.0)], clock=clock)
        finish(bp, "a", 1000.0, 5.0, status="failed: boom")
        finish(bp, "b", 1000.0, 5.0, status="QA_FAIL: skin_mask")
        finish(bp, "c", 1000.0, 5.0, status="weird")
        c = bp.counts
        assert (c["failed"], c["qa_fail"], c["pending"]) == (2, 1, 1)
        assert bp.images["a"]["reason"] == "failed: boom"
        bp.on_event(ev("d", "start", 1940.0, worker=2))
        bp.on_event(ev("d", "stage", 1950.0, stage="per_face"))
        bp.on_event(ev("d", "face", 1960.0, index=0, total=5))
        bp.on_event(ev("d", "face", 1970.0, index=3, total=5))
        (a,) = bp.active()
        assert a["image"] == "d" and a["stage"] == "per_face"
        assert (a["faces_done"], a["faces_total"]) == (2, 5)
        assert a["elapsed"] == pytest.approx(60.0)
        assert a["stage_elapsed"] == pytest.approx(50.0)

    def test_late_events_after_finish_are_ignored(self):
        bp = BatchProgress([("a", 1.0)], clock=FakeClock())
        bp.on_event(ev("a", "start", 1000.0))
        bp.on_event(ev("a", "finish", 1010.0, status="done", seconds=10.0))
        bp.on_event(ev("a", "stage", 1009.0, stage="grading"))
        bp.on_event(ev("a", "face", 1009.5, index=1, total=2))
        bp.on_event(ev("a", "start", 1009.0))
        r = bp.images["a"]
        assert r["status"] == "done" and r["stage"] is None and r["faces_done"] == 0
        assert r["last_event_at"] == 1010.0
        assert bp.active() == [] and bp.counts["running"] == 0

    def test_stage_without_start_counts_as_running(self):
        bp = BatchProgress([("a", 1.0)], clock=FakeClock())
        bp.on_event(ev("a", "stage", 1000.0, stage="detect"))
        assert bp.counts["running"] == 1
        bp.on_event({"kind": "stage"})  # malformed: ignored
        bp.on_event(ev("new.jpg", "start", 1000.0))  # unknown image added
        assert "new.jpg" in bp.images and bp.counts["total"] == 2

    def test_stalled(self):
        clock = FakeClock(1000.0)
        bp = BatchProgress([("a", 1.0), ("b", 1.0)], clock=clock)
        bp.on_event(ev("a", "start", 1000.0))
        bp.on_event(ev("b", "start", 1000.0))
        bp.on_event(ev("b", "stage", 1080.0, stage="grading"))
        assert bp.stalled(now=1089.0, after=90.0) == []
        (s,) = bp.stalled(now=1091.0, after=90.0)
        assert s["image"] == "a" and s["silent_for"] == pytest.approx(91.0)
        clock.advance(200.0)
        assert {s["image"] for s in bp.stalled(after=90.0)} == {"a", "b"}
        bp.on_event(ev("a", "finish", 1200.0, status="done"))
        assert [s["image"] for s in bp.stalled(after=90.0)] == ["b"]
        # seconds inferred from start when the event omits it
        assert bp.images["a"]["seconds"] == pytest.approx(200.0)

    def test_snapshot_roundtrip_and_passthrough(self, tmp_path):
        clock = FakeClock(1000.0)
        bp = BatchProgress([("a.jpg", 24.0), ("b.jpg", None)], clock=clock)
        qa = [{"detector": "halo", "score": np.float32(0.8), "threshold": 0.5,
               "flagged": True, "message": "halo at hairline"},
              {"detector": "banding", "score": 0.1, "threshold": 0.5,
               "flagged": False, "message": ""}]
        bp.on_event(ev("a.jpg", "start", 1000.0, worker=1))
        bp.on_event(ev("a.jpg", "finish", 1030.0, status="done", seconds=30.0, faces=2,
                       timings={"per_face": np.float64(12.5)}, qa=qa,
                       out_path="/out/a.jpg", compare_path=None, megapixels=26.0))
        bp.on_event(ev("b.jpg", "start", 1030.0))
        clock.advance(40.0)
        snap = bp.snapshot()
        back = json.loads(json.dumps(snap))
        assert back == snap
        assert back["version"] == 1 and back["started_at"] == 1000.0
        assert back["updated_at"] == 1040.0
        assert back["counts"]["done"] == 1 and back["counts"]["running"] == 1
        a = back["images"]["a.jpg"]
        assert a["out_path"] == "/out/a.jpg" and a["compare_path"] is None
        assert a["mp"] == 26.0  # finish megapixels overrides the estimate
        assert a["qa"][0]["detector"] == "halo" and a["qa"][0]["score"] == pytest.approx(0.8)
        assert a["timings"] == {"per_face": 12.5}
        assert back["images"]["b.jpg"]["elapsed"] == pytest.approx(10.0)
        assert back["total_mp"] == pytest.approx(26.0 + 24.0)

    def test_write_json_atomic(self, tmp_path, monkeypatch):
        bp = BatchProgress([("a", 1.0)], clock=FakeClock())
        path = tmp_path / "sub" / "progress.json"
        bp.write_json(path)
        assert json.loads(path.read_text())["counts"]["pending"] == 1
        assert [p.name for p in path.parent.iterdir()] == ["progress.json"]

        replaced = []
        real_replace = os.replace

        def spy(src, dst):
            replaced.append((os.path.dirname(src), os.fspath(dst)))
            return real_replace(src, dst)

        monkeypatch.setattr(os, "replace", spy)
        bp.on_event(ev("a", "start", 1000.0))
        bp.write_json(path)
        assert replaced == [(str(path.parent), str(path))]
        assert json.loads(path.read_text())["counts"]["running"] == 1

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            bp.write_json(path)
        # old file intact, no stray tmp files
        assert json.loads(path.read_text())["counts"]["running"] == 1
        assert [p.name for p in path.parent.iterdir()] == ["progress.json"]

    def test_summary_lines(self):
        bp = BatchProgress([(f"img{i}.jpg", 10.0) for i in range(6)], clock=FakeClock())
        finish(bp, "img0.jpg", 1000.0, 50.0, faces=1)
        finish(bp, "img1.jpg", 1000.0, 300.0, faces=5)
        finish(bp, "img2.jpg", 1000.0, 100.0,
               qa=[{"detector": "halo", "score": 0.9, "threshold": 0.5,
                    "flagged": True, "message": "m"}])
        finish(bp, "img3.jpg", 1000.0, 0.01, status="skipped")
        finish(bp, "img4.jpg", 1000.0, 10.0, status="failed: OOM")
        finish(bp, "img5.jpg", 1000.0, 40.0, status="QA_FAIL: skin")
        text = "\n".join(bp.summary_lines())
        assert "6 images" in text and "3 done" in text and "1 skipped" in text
        assert "1 failed" in text and "1 QA fail" in text
        assert "Total time 5m00s" in text
        # 5 processed images: 500 s / 5 = 100 s/image, 500 s / 50 MP = 10 s/MP
        assert "avg 100.0 s/image" in text and "10.0 s/MP" in text
        slow = next(l for l in bp.summary_lines() if l.startswith("Slowest"))
        assert slow.index("img1.jpg 300.0s") < slow.index("img2.jpg 100.0s") < slow.index("img0.jpg 50.0s")
        assert "5 faces" in slow and "img3" not in slow
        assert "FAILED img4.jpg: failed: OOM" in text
        assert "QA_FAIL img5.jpg: QA_FAIL: skin" in text
        assert "QA flagged img2.jpg: halo 0.9/0.5" in text


# ---------------------------------------------------------------- reporter

class TestProgressReporter:
    def test_reporter_consumes_events_and_summarises(self, tmp_path):
        clock = FakeClock(1000.0)
        bp = BatchProgress([("DSCF1.jpg", 20.0), ("DSCF2.jpg", 20.0), ("DSCF3.jpg", 10.0)],
                           clock=clock)
        q = queue.Queue()
        lines = []
        path = tmp_path / "progress.json"
        rep = ProgressReporter(bp, q, path, stall_after=90.0, write=lines.append,
                               disable=True, poll_interval=0.02, json_interval=0.0)
        with rep:
            q.put(ev("DSCF1.jpg", "start", 1000.0, worker=0))
            q.put(ev("DSCF1.jpg", "stage", 1001.0, stage="per_face"))
            q.put(ev("DSCF1.jpg", "face", 1010.0, index=0, total=2))
            _wait(lambda: bp.images["DSCF1.jpg"]["faces_done"] == 1)
            clock.advance(72.0)
            _wait(lambda: "DSCF1 per_face 1/2 1m12s" in rep._postfix())
            q.put(ev("DSCF1.jpg", "finish", 1100.0, status="done", seconds=100.0, faces=2))
            q.put(ev("DSCF3.jpg", "start", 1100.0))
            q.put(ev("DSCF3.jpg", "finish", 1100.1, status="skipped", seconds=0.1))
            _wait(lambda: rep.shown_mp == pytest.approx(20.0) and rep.bar.total == pytest.approx(40.0))
            _wait(lambda: path.exists() and json.loads(path.read_text())["counts"]["done"] == 1)
            q.put(ev("DSCF2.jpg", "start", 1100.0))
            q.put(ev("DSCF2.jpg", "finish", 1150.0, status="failed: boom", seconds=50.0))
        summary = rep.summary
        assert summary == rep.close()  # idempotent
        text = "\n".join(summary)
        assert "1 done" in text and "1 skipped" in text and "1 failed" in text
        assert "FAILED DSCF2.jpg: failed: boom" in text
        final = json.loads(path.read_text())
        assert final["counts"]["finished"] == 3 and final["eta_seconds"] == 0.0
        assert not rep._thread.is_alive()
        assert lines == []  # nothing stalled

    def test_stall_lines_rate_limited(self):
        clock = FakeClock(1000.0)
        bp = BatchProgress([("DSCF9.jpg", 26.0)], clock=clock)
        q = queue.Queue()
        lines = []
        with ProgressReporter(bp, q, None, stall_after=90.0, write=lines.append,
                              disable=True, poll_interval=0.02) as rep:
            q.put(ev("DSCF9.jpg", "start", 1000.0))
            q.put(ev("DSCF9.jpg", "stage", 1000.0, stage="per_face"))
            _wait(lambda: bp.counts["running"] == 1)
            clock.advance(190.0)
            _wait(lambda: len(lines) == 1)
            assert lines[0].endswith("still working on DSCF9.jpg: per_face for 3m10s")
            time.sleep(0.1)
            assert len(lines) == 1  # not repeated within stall_after
            clock.advance(91.0)
            _wait(lambda: len(lines) == 2)
            q.put(ev("DSCF9.jpg", "finish", 1300.0, status="done"))
        assert len(lines) == 2
        assert rep.summary[0].startswith("Batch: 1 images")

    def test_dead_queue_does_not_kill_reporter(self, tmp_path):
        bp = BatchProgress([("a", 1.0)], clock=FakeClock())
        rep = ProgressReporter(bp, _BrokenQueue(), tmp_path / "p.json",
                               write=lambda s: None, disable=True, poll_interval=0.02)
        _wait(lambda: rep._queue_dead)
        rep.handle(ev("a", "start", 1000.0))
        rep.handle(ev("a", "finish", 1005.0, status="done", seconds=5.0))
        summary = rep.close()
        assert "1 done" in summary[0]
        assert json.loads((tmp_path / "p.json").read_text())["counts"]["done"] == 1


def _wait(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if pred():
                return
        except Exception:
            pass
        time.sleep(0.01)
    assert pred(), "condition not reached before timeout"
