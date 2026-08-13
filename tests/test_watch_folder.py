"""Watch-folder stability and retry tests."""

from pathlib import Path

from PIL import Image

from retouch.watch_folder import WatchFolder


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
