"""Focused contracts for session-safe GUI revisions and history boundaries."""

from pathlib import Path


def _gui():
    import gui

    return gui


def test_revision_and_completion_are_session_value_functions():
    gui = _gui()

    first_revision, first_status = gui.on_settings_changed(0)
    second_revision, second_status = gui.on_settings_changed(0)

    assert first_revision == second_revision == 1
    assert "preview is out of date" in first_status
    assert second_status == first_status
    assert "out of date" in gui.render_completion_status(
        "Done", {"settings_revision": 4}, current_revision=5
    )
    assert "out of date" not in gui.render_completion_status(
        "Done", {"settings_revision": 4}, current_revision=4
    )


def test_capture_snapshots_are_plain_values_for_the_next_queued_stage():
    gui = _gui()

    process_args = tuple(range(len(gui.PROCESS_INPUT_KEYS)))
    render_snapshot, render_status = gui.capture_render_snapshot(*process_args, 12)
    smart_snapshot, smart_status = gui.capture_smart_snapshot(
        ["portrait.jpg"], "natural", 12
    )

    assert render_snapshot["process_args"] == process_args
    assert render_snapshot["settings_revision"] == 12
    assert "revision 12" in render_status
    assert smart_snapshot == {
        "img_paths": ["portrait.jpg"],
        "recipe": "natural",
        "settings_revision": 12,
    }
    assert "revision 12" in smart_status


def test_empty_undo_redo_preserve_processing_controls_with_skip():
    gui = _gui()
    expected_skip = {"__type__": "update"}
    count = len(gui.PROCESS_INPUT_KEYS)

    for result in (gui.undo_handler(None), gui.redo_handler(None)):
        assert len(result) == count + 3
        assert all(value == expected_skip for value in result[:count])
        assert result[count] == expected_skip
        assert result[count + 1].get("interactive") is False
        assert result[count + 2].get("interactive") is False


def test_history_starts_at_baseline_and_moves_from_post_mutation_values():
    gui = _gui()
    baseline = tuple(range(len(gui.PROCESS_INPUT_KEYS)))
    changed = list(baseline)
    changed[gui.PROCESS_INPUT_KEYS.index("smooth")] = 77

    stack, undo_update, redo_update = gui.initialize_history(*baseline)
    assert stack.cursor == 0
    assert stack.current() == dict(zip(gui.PROCESS_INPUT_KEYS, baseline))
    assert undo_update.get("interactive") is False
    assert redo_update.get("interactive") is False

    stack, undo_update, redo_update = gui.record_history(*changed, stack)
    assert stack.cursor == 1
    assert undo_update.get("interactive") is True
    assert redo_update.get("interactive") is False

    undone = gui.undo_handler(stack)
    assert undone[gui.PROCESS_INPUT_KEYS.index("smooth")] == baseline[
        gui.PROCESS_INPUT_KEYS.index("smooth")
    ]
    assert undone[len(gui.PROCESS_INPUT_KEYS) + 1].get("interactive") is False
    assert undone[len(gui.PROCESS_INPUT_KEYS) + 2].get("interactive") is True


def test_gui_wiring_consolidates_revision_and_keeps_capture_unqueued():
    gui = _gui()
    source = Path("gui.py").read_text(encoding="utf-8")

    assert "_latest_settings_revision" not in source
    assert 'concurrency_id="retouch-settings-revision"' not in source
    assert "gr.on(" in source
    assert "_settings_revision_triggers" in source
    assert "_history_triggers" in source
    assert "_capture_event = _button.click" in source
    assert "_smart_capture_event = smart_process_btn.click" in source
    assert "fn=render_completion_status" in source
    assert "fn=smart_completion_status" in source
    assert 'compare_snapshot_btn = gr.Button("Inspect Settings"' in source
    assert "Snapshot settings (JSON) — not an image comparison" in source

    dependencies = gui.app.config.get("dependencies", [])

    def dependencies_for(name):
        return [
            dependency
            for dependency in dependencies
            if dependency.get("api_name", "").startswith(name)
        ]

    capture_dependencies = dependencies_for("capture_render_snapshot")
    render_dependencies = dependencies_for("process_image_event")
    smart_capture_dependencies = dependencies_for("capture_smart_snapshot")
    smart_dependencies = dependencies_for("on_smart_process_event")

    assert capture_dependencies and all(
        dependency["queue"] is False
        and dependency["show_progress"] == "hidden"
        for dependency in capture_dependencies
    )
    assert smart_capture_dependencies and smart_capture_dependencies[0]["queue"] is False
    assert smart_capture_dependencies[0]["show_progress"] == "hidden"
    assert render_dependencies and all(
        dependency["queue"] is True
        and dependency["show_progress"] == "minimal"
        for dependency in render_dependencies
    )
    assert smart_dependencies and smart_dependencies[0]["queue"] is True
    assert smart_dependencies[0]["show_progress"] == "minimal"


def test_live_revision_state_has_no_process_global_counter_and_finalizes_full_export():
    gui = _gui()
    source = Path("gui.py").read_text(encoding="utf-8")

    assert "ContextVar" not in source
    assert "_latest_settings_revision" not in source

    status, export_update = gui.render_completion_status(
        "Done",
        {"settings_revision": 4, "render_mode": gui.MODE_EXPORT_FULL_QUALITY},
        current_revision=5,
        export_file="/tmp/stale-export.jpg",
    )
    assert "out of date" in status
    assert export_update.get("visible") is False


def test_advanced_history_boundary_fails_closed_instead_of_replaying_a_truncated_prefix():
    gui = _gui()
    from retouch.advanced_history import AdvancedHistory

    history = AdvancedHistory(
        memory_budget_bytes=1400,
        checkpoint_interval=1000,
    )
    for index in range(30):
        history.append({"index": index, "note": "x" * 80})
    assert history.history_truncated is True

    source = __import__("numpy").zeros((8, 8, 3), dtype="uint8")
    result = gui.advanced_undo_handler(
        source,
        source,
        history.to_state(),
        history.current_edit_log(),
    )
    assert result[1] is source
    assert "boundary" in result[7]
