"""Watch-folder stability and retry tests."""

from pathlib import Path

from PIL import Image

from retouch.watch_folder import PermanentWatchJobError, WatchFolder


def _write_image(path: Path, value: int = 100) -> None:
    Image.new("RGB", (8, 8), (value, value, value)).save(path)


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
