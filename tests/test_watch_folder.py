"""Watch-folder stability and retry tests."""

import json
import multiprocessing
import time
from pathlib import Path

from PIL import Image

import pytest

from retouch.watch_folder import (
    PermanentWatchJobError,
    StateLoadError,
    WatchFolder,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None


def _write_image(path: Path, value: int = 100) -> None:
    Image.new("RGB", (8, 8), (value, value, value)).save(path)


def _wait_for(path: Path, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"peer never signalled via {path}")
        time.sleep(0.05)


def _concurrent_writer(input_dir: str, state_path: str, image_name: str, ready: str, peer_ready: str) -> None:
    """Load shared state, wait until the peer also loaded, then record one file and save."""
    from retouch.watch_folder import WatchFolder, WatchRecord

    watcher = WatchFolder(input_dir, state_path=state_path)
    Path(ready).write_text("ready", encoding="utf-8")
    _wait_for(Path(peer_ready))
    record_path = str((Path(input_dir) / image_name).resolve())
    watcher.state.records[record_path] = WatchRecord(path=record_path, signature="1:1")
    watcher.save()


def test_watch_folder_requires_two_stable_scans(tmp_path: Path):
    source = tmp_path / "portrait.jpg"
    _write_image(source)
    watcher = WatchFolder(tmp_path)

    assert watcher.scan_once() == []
    ready = watcher.scan_once()

    assert len(ready) == 1
    assert ready[0].path == str(source.resolve())
    assert ready[0].status == "pending"


def test_watch_folder_processes_success_and_persists_state(tmp_path: Path):
    source = tmp_path / "portrait.jpg"
    state = tmp_path / "state.json"
    _write_image(source)
    watcher = WatchFolder(tmp_path, state_path=state)
    watcher.scan_once()

    processed = []
    result = watcher.process_pending(lambda path: processed.append(str(path)))

    assert result[0].status == "done"
    assert processed == [str(source.resolve())]
    restored = WatchFolder(tmp_path, state_path=state)
    assert restored.state.records[str(source.resolve())].status == "done"


def test_watch_folder_failures_are_retryable(tmp_path: Path):
    source = tmp_path / "portrait.jpg"
    _write_image(source)
    watcher = WatchFolder(tmp_path)
    watcher.scan_once()

    failed = watcher.process_pending(lambda _path: (_ for _ in ()).throw(RuntimeError("temporary")))
    assert failed[0].status == "failed"
    assert failed[0].attempts == 1
    assert "temporary" in failed[0].error

    succeeded = watcher.process_pending(lambda _path: None)
    assert succeeded[0].status == "done"
    assert succeeded[0].attempts == 2


def test_watch_folder_ignores_non_images_and_supports_limit(tmp_path: Path):
    _write_image(tmp_path / "a.jpg", 80)
    _write_image(tmp_path / "b.jpg", 120)
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")
    watcher = WatchFolder(tmp_path)
    watcher.scan_once()
    watcher.scan_once()

    completed = watcher.process_pending(lambda _path: None, limit=1)

    assert len(completed) == 1
    assert sum(record.status == "done" for record in watcher.state.records.values()) == 1


def test_watch_jobs_are_idempotent_and_preview_final_are_independent(tmp_path: Path):
    source = tmp_path / "card-a" / "IMG_0001.jpg"
    source.parent.mkdir()
    _write_image(source)
    watcher = WatchFolder(tmp_path, state_path=tmp_path / "watch.json")
    output = tmp_path.parent / (tmp_path.name + "-deliveries")
    settings = {"recipe": "natural", "export_fmt": "JPEG"}

    assert watcher.queue_jobs(
        kind="preview", output_root=output, settings=settings,
        pipeline_fingerprint="test-v1",
    ) == []
    queued = watcher.queue_jobs(
        kind="preview", output_root=output, settings=settings,
        pipeline_fingerprint="test-v1",
    )
    assert len(queued) == 1
    preview_id = queued[0].job_id

    def write_output(job):
        Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(job.output_path).write_bytes(b"preview bytes")
        return Path(job.output_path)

    done = watcher.run_queued(write_output)
    assert done[0].status == "done"
    assert done[0].output_sha256
    assert next(iter(watcher.state.records.values())).status == "done"
    assert watcher.queue_jobs(
        kind="preview", output_root=output, settings=settings,
        pipeline_fingerprint="test-v1",
    ) == []

    final = watcher.queue_jobs(
        kind="final", output_root=output, settings=settings,
        pipeline_fingerprint="test-v1",
    )
    assert len(final) == 1
    assert final[0].job_id != preview_id
    assert final[0].kind == "final"


def test_watch_jobs_disambiguate_same_stem_source_formats(tmp_path: Path):
    _write_image(tmp_path / "photo.jpg", 80)
    _write_image(tmp_path / "photo.png", 120)
    watcher = WatchFolder(tmp_path, state_path=tmp_path / "watch.json")
    output = tmp_path.parent / (tmp_path.name + "-deliveries")
    kwargs = {
        "kind": "preview",
        "output_root": output,
        "settings": {"recipe": "natural", "export_fmt": "JPEG"},
        "pipeline_fingerprint": "test-v1",
    }

    watcher.queue_jobs(**kwargs)
    queued = watcher.queue_jobs(**kwargs)

    assert len(queued) == 2
    assert len({job.output_path for job in queued}) == 2
    assert any("_jpg_" in Path(job.output_path).name for job in queued)
    assert any("_png_" in Path(job.output_path).name for job in queued)


def test_watch_done_job_is_replaced_when_processing_contract_changes(tmp_path: Path):
    source = tmp_path / "portrait.jpg"
    _write_image(source)
    watcher = WatchFolder(tmp_path, state_path=tmp_path / "watch.json")
    first_output = tmp_path.parent / (tmp_path.name + "-out-a")
    base_kwargs = {
        "kind": "preview",
        "output_root": first_output,
        "settings": {"recipe": "natural", "export_fmt": "JPEG"},
        "pipeline_fingerprint": "pipeline-a",
    }
    watcher.queue_jobs(**base_kwargs)
    original = watcher.queue_jobs(**base_kwargs)[0]

    def write_output(job):
        Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(job.output_path).write_bytes(job.settings["recipe"].encode("utf-8"))
        return Path(job.output_path)

    watcher.run_queued(write_output)
    changed = watcher.queue_jobs(
        **{**base_kwargs, "settings": {"recipe": "classic_chrome", "export_fmt": "JPEG"}}
    )
    assert len(changed) == 1
    assert changed[0].job_id != original.job_id
    assert changed[0].output_path != original.output_path

    pipeline_changed = watcher.queue_jobs(
        **{
            **base_kwargs,
            "settings": {"recipe": "classic_chrome", "export_fmt": "JPEG"},
            "pipeline_fingerprint": "pipeline-b",
        }
    )
    assert len(pipeline_changed) == 1
    assert watcher.state.jobs[changed[0].job_id].status == "cancelled"

    second_output = tmp_path.parent / (tmp_path.name + "-out-b")
    root_changed = watcher.queue_jobs(
        **{
            **base_kwargs,
            "output_root": second_output,
            "settings": {"recipe": "classic_chrome", "export_fmt": "JPEG"},
            "pipeline_fingerprint": "pipeline-b",
        }
    )
    assert len(root_changed) == 1
    Path(root_changed[0].output_path).resolve().relative_to(second_output.resolve())


def test_watch_v1_record_only_state_migrates_to_v2(tmp_path: Path):
    source = tmp_path / "portrait.jpg"
    _write_image(source)
    state_path = tmp_path / "watch-v1.json"
    state_path.write_text(
        json.dumps({
            "version": 1,
            "records": {
                str(source.resolve()): {
                    "path": str(source.resolve()),
                    "signature": "legacy-signature",
                    "status": "done",
                    "attempts": 1,
                    "error": None,
                }
            },
        }),
        encoding="utf-8",
    )

    watcher = WatchFolder(tmp_path, state_path=state_path)

    assert watcher.state.version == 2
    assert watcher.state.jobs == {}
    assert str(source.resolve()) in watcher.state.records


def test_run_queued_filters_kind_and_applies_limit_per_pass(tmp_path: Path):
    for name, value in (("a.jpg", 80), ("b.jpg", 120)):
        _write_image(tmp_path / name, value)
    watcher = WatchFolder(tmp_path, state_path=tmp_path / "watch.json")
    output = tmp_path.parent / (tmp_path.name + "-deliveries")
    settings = {"recipe": "natural", "export_fmt": "JPEG"}

    watcher.queue_jobs(
        kind="preview", output_root=output, settings=settings,
        pipeline_fingerprint="test-v1",
    )
    previews = watcher.queue_jobs(
        kind="preview", output_root=output, settings=settings,
        pipeline_fingerprint="test-v1",
    )
    finals = watcher.queue_jobs(
        kind="final", output_root=output,
        settings={"recipe": "classic_chrome", "export_fmt": "PNG"},
        pipeline_fingerprint="test-v1",
    )
    assert len(previews) == 2
    assert len(finals) == 2

    seen = []

    def write_output(job):
        seen.append((job.kind, job.settings["recipe"], job.settings["export_fmt"]))
        Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(job.output_path).write_bytes(job.kind.encode("ascii"))
        return Path(job.output_path)

    completed = watcher.run_queued(write_output, kind="preview", limit=1)

    assert len(completed) == 1
    assert seen == [("preview", "natural", "JPEG")]
    assert sum(job.status == "done" for job in watcher.state.jobs.values() if job.kind == "preview") == 1
    assert sum(job.status == "queued" for job in watcher.state.jobs.values() if job.kind == "preview") == 1
    assert all(job.status == "queued" for job in watcher.state.jobs.values() if job.kind == "final")


def test_malformed_watch_state_is_preserved_and_fails_closed(tmp_path: Path):
    state_path = tmp_path / "watch.json"
    original = "{ not valid json"
    state_path.write_text(original, encoding="utf-8")

    with pytest.raises(StateLoadError, match="Malformed watch-folder state"):
        WatchFolder(tmp_path, state_path=state_path)

    assert state_path.read_text(encoding="utf-8") == original
    quarantined = list(tmp_path.glob("watch.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == original


def test_watch_job_failure_is_retryable_and_never_done_without_verified_output(tmp_path: Path):
    source = tmp_path / "capture.jpg"
    _write_image(source)
    watcher = WatchFolder(tmp_path, state_path=tmp_path / "watch.json")
    kwargs = {
        "kind": "preview", "output_root": tmp_path.parent / (tmp_path.name + "-out"),
        "settings": {"recipe": "natural"}, "pipeline_fingerprint": "test-v1",
    }
    watcher.queue_jobs(**kwargs)
    jobs = watcher.queue_jobs(**kwargs)
    assert len(jobs) == 1

    failed = watcher.run_queued(lambda _job: None)
    assert failed[0].status == "failed"
    assert failed[0].output_sha256 is None
    assert next(iter(watcher.state.records.values())).status == "failed"

    retry = watcher.queue_jobs(**kwargs)
    assert len(retry) == 1
    assert retry[0].job_id == failed[0].job_id


def test_permanent_watch_job_failure_requires_explicit_retry(tmp_path: Path):
    source = tmp_path / "capture.jpg"
    _write_image(source)
    watcher = WatchFolder(tmp_path, state_path=tmp_path / "watch.json")
    kwargs = {
        "kind": "preview", "output_root": tmp_path.parent / (tmp_path.name + "-out"),
        "settings": {"recipe": "natural"}, "pipeline_fingerprint": "test-v1",
    }
    watcher.queue_jobs(**kwargs)
    jobs = watcher.queue_jobs(**kwargs)
    assert len(jobs) == 1
    watcher.run_queued(lambda _job: (_ for _ in ()).throw(PermanentWatchJobError("model unavailable")))

    assert watcher.queue_jobs(**kwargs) == []
    explicit = watcher.queue_jobs(**kwargs, retry_failed=True)
    assert len(explicit) == 1


def test_state_record_path_outside_input_dir_fails_closed(tmp_path: Path):
    outside = tmp_path.parent / (tmp_path.name + "-outside") / "evil.jpg"
    state_path = tmp_path / "watch.json"
    payload = {
        "version": 2,
        "records": {
            str(outside.resolve()): {
                "path": str(outside.resolve()),
                "signature": "1:1",
                "status": "pending",
            }
        },
        "jobs": {},
    }
    original = json.dumps(payload)
    state_path.write_text(original, encoding="utf-8")

    with pytest.raises(StateLoadError, match="escapes the watched folder"):
        WatchFolder(tmp_path, state_path=state_path)

    assert state_path.read_text(encoding="utf-8") == original
    quarantined = list(tmp_path.glob("watch.json.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == original


def test_state_job_output_path_outside_root_fails_closed(tmp_path: Path):
    source = tmp_path / "portrait.jpg"
    _write_image(source)
    output_root = tmp_path.parent / (tmp_path.name + "-deliveries")
    outside = tmp_path.parent / (tmp_path.name + "-evil") / "stolen.jpg"
    job = {
        "job_id": "watch-job-tampered",
        "source_path": str(source.resolve()),
        "content_id": "abc",
        "asset_instance_id": "abc",
        "kind": "preview",
        "settings": {},
        "settings_fingerprint": "f",
        "pipeline_fingerprint": "p",
        "output_path": str(outside),
        "output_root": str(output_root),
        "status": "queued",
    }
    state_path = tmp_path / "watch.json"
    state_path.write_text(
        json.dumps({"version": 2, "records": {}, "jobs": {job["job_id"]: job}}),
        encoding="utf-8",
    )

    with pytest.raises(StateLoadError, match="escapes its output root"):
        WatchFolder(tmp_path, state_path=state_path)

    assert not outside.exists()
    assert len(list(tmp_path.glob("watch.json.corrupt-*"))) == 1


def test_run_queued_rejects_tampered_job_output_without_processing(tmp_path: Path):
    source = tmp_path / "portrait.jpg"
    _write_image(source)
    watcher = WatchFolder(tmp_path, state_path=tmp_path / "watch.json")
    kwargs = {
        "kind": "preview", "output_root": tmp_path.parent / (tmp_path.name + "-out"),
        "settings": {"recipe": "natural"}, "pipeline_fingerprint": "test-v1",
    }
    watcher.queue_jobs(**kwargs)
    job = watcher.queue_jobs(**kwargs)[0]
    outside = tmp_path.parent / (tmp_path.name + "-evil") / "stolen.jpg"

    def evil_writer(_job):
        outside.parent.mkdir(parents=True, exist_ok=True)
        outside.write_bytes(b"stolen")
        return outside

    job.output_path = str(outside)
    with pytest.raises(StateLoadError, match="escapes its output root"):
        watcher.run_queued(evil_writer)
    assert not outside.exists()

    # A job whose persisted output_root is blank but whose output path sits
    # under the RUNTIME root (pinned by queue_jobs) is now valid — the
    # state-file root no longer self-attests. It must process normally:
    # no StateLoadError, output verified, job completes.
    job.output_path = str(tmp_path.parent / (tmp_path.name + "-out") / "preview" / "ok.jpg")
    job.output_root = ""

    def good_writer(j):
        out = Path(j.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"ok")
        return out

    watcher.run_queued(good_writer)
    assert Path(job.output_path).exists()


def test_state_with_contained_paths_round_trips(tmp_path: Path):
    source = tmp_path / "portrait.jpg"
    _write_image(source)
    state_path = tmp_path / "watch.json"
    watcher = WatchFolder(tmp_path, state_path=state_path)
    output = tmp_path.parent / (tmp_path.name + "-deliveries")
    kwargs = {
        "kind": "preview", "output_root": output,
        "settings": {"recipe": "natural", "export_fmt": "JPEG"},
        "pipeline_fingerprint": "test-v1",
    }
    watcher.queue_jobs(**kwargs)
    job = watcher.queue_jobs(**kwargs)[0]

    def write_output(job):
        Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(job.output_path).write_bytes(b"preview bytes")
        return Path(job.output_path)

    watcher.run_queued(write_output)

    restored = WatchFolder(tmp_path, state_path=state_path)
    assert restored.state.jobs[job.job_id].status == "done"
    assert restored.queue_jobs(**kwargs) == []


def test_concurrent_watchers_keep_both_updates(tmp_path: Path):
    """Two processes on one state file must not lose each other's records."""
    state = tmp_path / "watch.json"
    ready_a = tmp_path.parent / (tmp_path.name + "-a.ready")
    ready_b = tmp_path.parent / (tmp_path.name + "-b.ready")
    ctx = multiprocessing.get_context("spawn")
    proc_a = ctx.Process(target=_concurrent_writer, args=(str(tmp_path), str(state), "a.jpg", str(ready_a), str(ready_b)))
    proc_b = ctx.Process(target=_concurrent_writer, args=(str(tmp_path), str(state), "b.jpg", str(ready_b), str(ready_a)))
    proc_a.start()
    proc_b.start()
    proc_a.join(timeout=60)
    proc_b.join(timeout=60)
    assert proc_a.exitcode == 0, "writer A crashed"
    assert proc_b.exitcode == 0, "writer B crashed"

    final = WatchFolder(tmp_path, state_path=state)
    paths = set(final.state.records)
    assert str((tmp_path / "a.jpg").resolve()) in paths
    assert str((tmp_path / "b.jpg").resolve()) in paths


def test_state_lock_released_after_save(tmp_path: Path):
    """Another process must acquire the lock immediately after save() returns."""
    source = tmp_path / "portrait.jpg"
    _write_image(source)
    watcher = WatchFolder(tmp_path, state_path=tmp_path / "watch.json")
    watcher.scan_once()
    watcher.scan_once()
    lock_path = tmp_path / "watch.json.lock"
    assert lock_path.exists(), "save must leave the lock file beside the state"
    with open(lock_path, "a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_state_lock_blocks_concurrent_save(tmp_path: Path):
    """A second writer must block (not interleave) while the first holds the lock."""
    from retouch.watch_folder import _state_file_lock

    source = tmp_path / "portrait.jpg"
    _write_image(source)
    watcher = WatchFolder(tmp_path, state_path=tmp_path / "watch.json")
    with _state_file_lock(watcher.state_path):
        lock_path = watcher.state_path.with_name(watcher.state_path.name + ".lock")
        with open(lock_path, "a") as handle:
            with pytest.raises(OSError):
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Released after the context exits: a non-blocking acquire now succeeds.
    with open(lock_path, "a") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def test_state_lock_not_held_during_processing(tmp_path: Path):
    """The processor callback runs outside the lock (long jobs don't block peers)."""
    from retouch.watch_folder import _state_file_lock

    source = tmp_path / "portrait.jpg"
    _write_image(source)
    watcher = WatchFolder(tmp_path, state_path=tmp_path / "watch.json")
    kwargs = {
        "kind": "preview", "output_root": tmp_path.parent / (tmp_path.name + "-out"),
        "settings": {"recipe": "natural"}, "pipeline_fingerprint": "test-v1",
    }
    watcher.queue_jobs(**kwargs)
    watcher.queue_jobs(**kwargs)

    def slow_writer(job):
        Path(job.output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(job.output_path).write_bytes(b"preview")
        lock_path = watcher.state_path.with_name(watcher.state_path.name + ".lock")
        with open(lock_path, "a") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return Path(job.output_path)

    assert watcher.run_queued(slow_writer)[0].status == "done"
