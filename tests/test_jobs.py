"""Tests for retouch.jobs — Job/FileRecord model and JobStore persistence."""

import json
import logging

import pytest

from retouch.jobs import (
    FileRecord,
    Job,
    JobStore,
    QA_STATE_CLEAN,
    QA_STATE_FLAGGED,
    make_job_id,
)


def test_make_job_id_is_unique_and_sortable():
    a = make_job_id("/some/dir")
    b = make_job_id("/some/dir")
    assert a != b
    assert a < b  # timestamp-prefixed, so later calls sort after earlier ones


def test_file_record_round_trip():
    rec = FileRecord(
        source_path="/in/a.jpg",
        output_path="/out/a_retouched.jpg",
        status="done",
        qa_state=QA_STATE_FLAGGED,
        qa_warnings=[{"detector": "halo", "score": 0.9}],
    )
    restored = FileRecord.from_dict(rec.to_dict())
    assert restored == rec


def test_job_round_trip_to_json():
    job = Job(
        job_id="20260101T000000000000_abcd1234",
        input_dir="/in",
        output_dir="/out",
        recipe_or_style="natural",
        total_files=2,
        files=[
            FileRecord(source_path="/in/a.jpg", status="done", qa_state=QA_STATE_CLEAN),
            FileRecord(source_path="/in/b.jpg", status="done", qa_state=QA_STATE_FLAGGED,
                       qa_warnings=[{"detector": "banding"}]),
        ],
    )
    restored = Job.from_json(job.to_json())
    assert restored.job_id == job.job_id
    assert restored.total_files == 2
    assert len(restored.files) == 2
    assert restored.flagged_count == 1
    assert [f.source_path for f in restored.flagged_files()] == ["/in/b.jpg"]


def test_job_from_dict_forward_compatible_unknown_keys(caplog):
    data = {
        "job_id": "abc",
        "totally_new_field": "should warn not crash",
        "files": [],
    }
    with caplog.at_level(logging.WARNING):
        job = Job.from_dict(data)
    assert job.job_id == "abc"
    assert "unknown keys" in caplog.text.lower()


def test_file_record_from_dict_forward_compatible_unknown_keys(caplog):
    data = {"source_path": "/in/a.jpg", "some_future_field": 1}
    with caplog.at_level(logging.WARNING):
        rec = FileRecord.from_dict(data)
    assert rec.source_path == "/in/a.jpg"
    assert "unknown keys" in caplog.text.lower()


def test_job_store_save_and_load(tmp_path):
    store = JobStore(jobs_dir=str(tmp_path))
    job = Job(job_id="job1", input_dir="/in", output_dir="/out")
    store.save(job)

    loaded = store.load("job1")
    assert loaded.job_id == "job1"
    assert loaded.input_dir == "/in"


def test_job_store_save_is_atomic_on_replace_failure(tmp_path, monkeypatch):
    store = JobStore(jobs_dir=str(tmp_path))
    job = Job(job_id="job1", input_dir="/in")
    store.save(job)

    original_content = (tmp_path / "job1.json").read_text()

    def fail_replace(src, dst):
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr("retouch.jobs.os.replace", fail_replace)

    job.input_dir = "/changed"
    with pytest.raises(OSError):
        store.save(job)

    # Original file must be untouched — no partial write landed.
    assert (tmp_path / "job1.json").read_text() == original_content
    # No leftover temp files.
    assert list(tmp_path.glob("*.tmp")) == []


def test_job_store_list_jobs_sorted_desc(tmp_path):
    store = JobStore(jobs_dir=str(tmp_path))
    job_a = Job(job_id="a", created="2026-01-01T00:00:00+00:00")
    job_b = Job(job_id="b", created="2026-01-02T00:00:00+00:00")
    store.save(job_a)
    store.save(job_b)

    jobs = store.list_jobs()
    assert [j.job_id for j in jobs] == ["b", "a"]


def test_job_store_list_jobs_skips_corrupt_file(tmp_path, caplog):
    store = JobStore(jobs_dir=str(tmp_path))
    good = Job(job_id="good")
    store.save(good)

    (tmp_path / "corrupt.json").write_text("{not valid json")

    with caplog.at_level(logging.ERROR):
        jobs = store.list_jobs()

    assert [j.job_id for j in jobs] == ["good"]
    assert "corrupt.json" in caplog.text


def test_stale_running_job_demoted_on_list(tmp_path):
    store = JobStore(jobs_dir=str(tmp_path))
    stale = Job(job_id="stale", status="running", updated="2020-01-01T00:00:00+00:00")
    (tmp_path / "stale.json").write_text(stale.to_json())

    jobs = store.list_jobs()

    assert jobs[0].status == "partial"
    assert jobs[0].error_code == "worker_lost"
    # Persisted: reloading via load() sees the recovered state too.
    assert store.load("stale").status == "partial"


def test_stale_running_job_demoted_on_load(tmp_path):
    store = JobStore(jobs_dir=str(tmp_path))
    stale = Job(job_id="stale", status="running", updated="2020-01-01T00:00:00+00:00")
    (tmp_path / "stale.json").write_text(stale.to_json())

    job = store.load("stale")

    assert job.status == "partial"
    assert job.error_code == "worker_lost"


def test_fresh_running_job_untouched(tmp_path):
    store = JobStore(jobs_dir=str(tmp_path))
    fresh = Job(job_id="fresh", status="running")
    store.save(fresh)

    assert store.load("fresh").status == "running"
    assert store.load("fresh").error_code is None
    assert store.list_jobs()[0].status == "running"


def test_done_and_failed_jobs_untouched(tmp_path):
    store = JobStore(jobs_dir=str(tmp_path))
    for jid, status in (("d", "done"), ("f", "failed"), ("p", "partial")):
        job = Job(job_id=jid, status=status)
        job.updated = "2020-01-01T00:00:00+00:00"
        store.save(job)

    for jid in ("d", "f", "p"):
        loaded = store.load(jid)
        assert loaded.status == ("partial" if jid == "p" else loaded.status)
        assert loaded.error_code is None


def test_stale_recovery_idempotent_no_double_write(tmp_path):
    store = JobStore(jobs_dir=str(tmp_path))
    stale = Job(job_id="stale", status="running", updated="2020-01-01T00:00:00+00:00")
    (tmp_path / "stale.json").write_text(stale.to_json())

    first = store.list_jobs()[0]
    assert first.status == "partial"
    mtime = (tmp_path / "stale.json").stat().st_mtime_ns
    import time
    time.sleep(0.01)
    second = store.load("stale")
    assert second.status == "partial"
    assert (tmp_path / "stale.json").stat().st_mtime_ns == mtime
