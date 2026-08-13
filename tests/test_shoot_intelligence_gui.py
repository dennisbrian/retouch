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
    assert "Run watch pass" in source

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
