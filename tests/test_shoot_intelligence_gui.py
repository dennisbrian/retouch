"""GUI wiring checks for Shoot Intelligence handlers."""

from pathlib import Path

from PIL import Image


def test_gui_exposes_shoot_intelligence_controls_and_handlers():
    import gui

    source = Path("gui.py").read_text(encoding="utf-8")
    assert 'with gr.Tab("Shoot Intelligence")' in source
    assert "Scan shoot" in source
    assert "Save project profile" in source
    assert "Save Look Board" in source
    assert "Apply Look Board to Process" in source
    assert "Scan stable files" in source
    assert "Persistent Shoot Review Manifest" in source

    config = gui.app.get_config_file()
    api_names = {
        dependency.get("api_name")
        for dependency in config["dependencies"]
        if dependency.get("backend_fn")
    }
    assert {
        "on_shoot_intelligence_scan",
        "on_watch_folder_process",
        "on_save_project_profile",
        "on_save_look_board",
    }.issubset(api_names)


def test_gui_shoot_scan_is_non_native_and_explainable(tmp_path: Path):
    import gui

    image = Image.new("RGB", (16, 16), (100, 100, 100))
    image.save(tmp_path / "capture.jpg")

    rows, status, graph = gui.on_shoot_intelligence_scan(str(tmp_path), False)

    assert len(rows) == 1
    assert rows[0][0] == "asset"
    assert "Inspected 1 asset" in status
    assert '"human_cull_review"' in graph


def test_gui_watch_scan_does_not_mark_noop_processing_done(tmp_path: Path):
    import json
    import gui

    image = Image.new("RGB", (16, 16), (100, 100, 100))
    image.save(tmp_path / "capture.jpg")
    state = tmp_path / "watch-state.json"

    first = gui.on_watch_folder_process(str(tmp_path), str(state), None)
    second = gui.on_watch_folder_process(str(tmp_path), str(state), None)

    assert "no processing job was submitted" in first
    assert "no files were marked done" in second
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert all(record["status"] == "pending" for record in payload["records"].values())


def test_gui_watch_job_contract_verifies_preview_and_final_separately(monkeypatch, tmp_path: Path):
    import json
    import gui

    image = Image.new("RGB", (16, 16), (100, 100, 100))
    source = tmp_path / "card-a" / "capture.jpg"
    source.parent.mkdir()
    image.save(source)
    rows, status, _ = gui.on_shoot_intelligence_scan(str(tmp_path), True)
    assert rows and "Review manifest saved" in status

    class FakeBatchProcessor:
        def process_folder(self, input_dir, output_dir, only_files, **kwargs):
            path = Path(only_files[0])
            output = Path(kwargs["output_path_overrides"][path])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"verified " + str(kwargs["export_res"]).encode("utf-8"))
            return [str(output)], None, None, "processed one file"

        def close(self):
            return None

    monkeypatch.setattr(gui, "BatchProcessor", FakeBatchProcessor)
    state = tmp_path / "watch.json"
    output = tmp_path.parent / (tmp_path.name + "-watch-output")
    manifest = tmp_path / ".retouch-shoot-review.json"

    gui.on_watch_folder_process(str(tmp_path), str(state), None, str(output), "preview", "natural", str(manifest))
    preview_status = gui.on_watch_folder_process(
        str(tmp_path), str(state), None, str(output), "preview", "natural", str(manifest)
    )
    assert "1 preview succeeded" in preview_status
    payload = json.loads(state.read_text(encoding="utf-8"))
    jobs = list(payload["jobs"].values())
    assert len(jobs) == 1 and jobs[0]["status"] == "done"
    assert jobs[0]["output_sha256"]

    final_status = gui.on_watch_folder_process(
        str(tmp_path), str(state), None, str(output), "final", "natural", str(manifest)
    )
    assert "1 final succeeded" in final_status
    payload = json.loads(state.read_text(encoding="utf-8"))
    assert {job["kind"] for job in payload["jobs"].values()} == {"preview", "final"}
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert {job_kind for asset in manifest_payload["assets"] for job_kind in asset["job_provenance"]} == {"preview", "final"}


def test_gui_watch_pass_processes_only_selected_kind_from_recorded_job(monkeypatch, tmp_path: Path):
    import json
    import gui
    from retouch.watch_folder import WatchFolder

    source = tmp_path / "capture.jpg"
    Image.new("RGB", (16, 16), (100, 100, 100)).save(source)
    state = tmp_path / "watch.json"
    output = tmp_path.parent / (tmp_path.name + "-watch-output")

    queued_watcher = WatchFolder(tmp_path, state_path=state)
    queued_watcher.queue_jobs(
        kind="preview",
        output_root=output,
        settings={"recipe": "natural", "export_fmt": "JPEG", "export_res": "720px"},
        pipeline_fingerprint="test-v1",
    )
    preview_jobs = queued_watcher.queue_jobs(
        kind="preview",
        output_root=output,
        settings={"recipe": "natural", "export_fmt": "JPEG", "export_res": "720px"},
        pipeline_fingerprint="test-v1",
    )
    assert len(preview_jobs) == 1

    calls = []

    class FakeBatchProcessor:
        def process_folder(self, input_dir, output_dir, only_files, **kwargs):
            calls.append((kwargs["export_res"], kwargs["export_quality"], kwargs["style_name_or_recipe"]))
            path = Path(only_files[0])
            result = Path(kwargs["output_path_overrides"][path])
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_bytes(b"final output")
            return [str(result)], None, None, "processed one file"

        def close(self):
            return None

    monkeypatch.setattr(gui, "BatchProcessor", FakeBatchProcessor)
    manifest = tmp_path / ".retouch-shoot-review.json"
    status = gui.on_watch_folder_process(
        str(tmp_path), str(state), 1, str(output), "final", "classic_chrome", str(manifest)
    )

    assert "1 final succeeded" in status
    assert calls == [("Original", 95, "classic_chrome")]
    payload = json.loads(state.read_text(encoding="utf-8"))
    jobs_by_kind = {job["kind"]: job for job in payload["jobs"].values()}
    assert jobs_by_kind["preview"]["status"] == "queued"
    assert jobs_by_kind["final"]["status"] == "done"


def test_gui_face_quality_scan_is_opt_in_and_persists_evidence(monkeypatch, tmp_path: Path):
    import json
    import gui
    from retouch.detection import FaceData, _Landmark, _LandmarkCompat

    image = Image.new("RGB", (128, 128), (100, 100, 100))
    source = tmp_path / "portrait.jpg"
    image.save(source)
    points = [_Landmark(0.5, 0.5) for _ in range(468)]
    eye_positions = {
        33: (0.30, 0.42), 133: (0.40, 0.42), 159: (0.35, 0.40), 145: (0.35, 0.44),
        263: (0.60, 0.42), 362: (0.70, 0.42), 386: (0.65, 0.40), 374: (0.65, 0.44),
    }
    for index, (x, y) in eye_positions.items():
        points[index] = _Landmark(x, y)

    class FakeDetector:
        available = True
        unavailable_reason = None
        closed = False

        def __init__(self, allow_unavailable=False):
            assert allow_unavailable is True

        def detect(self, image_bgr):
            return [FaceData(
                landmarks=_LandmarkCompat(points),
                bbox=(20, 12, 88, 105),
                ied=40.0,
                confidence=0.93,
                confidence_source="retinaface",
            )]

        def close(self):
            type(self).closed = True

    monkeypatch.setattr(gui, "FaceDetector", FakeDetector)
    rows, status, _ = gui.on_shoot_intelligence_scan(
        str(tmp_path), False, None, True,
    )

    payload = json.loads((tmp_path / ".retouch-shoot-review.json").read_text(encoding="utf-8"))
    assert rows[0][-1] == "1 face evidence record(s); review required"
    assert rows[1][0] == "face"
    assert "confidence=0.93 (retinaface)" in rows[1][5]
    assert "blink_analysis_deferred" in rows[1][5]
    assert rows[1][-1] == "evidence only; human review required"
    assert "Face quality measured for 1 asset(s): 1 face(s)" in status
    assert len(payload["assets"][0]["faces"]) == 1
    assert payload["assets"][0]["faces"][0]["eyes_open"] == "uncertain"
    assert FakeDetector.closed is True


def test_gui_face_quality_unavailable_does_not_block_manifest_scan(monkeypatch, tmp_path: Path):
    import json
    import gui

    Image.new("RGB", (32, 32), (100, 100, 100)).save(tmp_path / "portrait.jpg")

    class UnavailableDetector:
        available = False
        unavailable_reason = "unsupported test runtime"

        def __init__(self, allow_unavailable=False):
            assert allow_unavailable is True

        def close(self):
            return None

    monkeypatch.setattr(gui, "FaceDetector", UnavailableDetector)
    rows, status, _ = gui.on_shoot_intelligence_scan(str(tmp_path), False, None, True)

    payload = json.loads((tmp_path / ".retouch-shoot-review.json").read_text(encoding="utf-8"))
    assert rows
    assert "Face quality unavailable: unsupported test runtime" in status
    assert "face_detector_unavailable" in payload["assets"][0]["uncertainty"]
    assert payload["assets"][0]["decision"] == "hold"
