"""Focused contracts for responsive GUI processing and stale previews."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def test_smart_process_stores_a_proposal_without_writing_live_controls(monkeypatch):
    import gui
    import retouch.smart_default as smart_default

    class FakeSmartProcessor:
        def analyze_and_suggest(self, image):
            assert image.shape == (8, 8, 3)
            return SimpleNamespace(
                recipe="natural",
                params={"smooth": 77},
                explanations=["Keep texture visible"],
            )

    monkeypatch.setattr(gui, "imread_exif", lambda _path: np.zeros((8, 8, 3), dtype=np.uint8))
    monkeypatch.setattr(smart_default, "SmartProcessor", FakeSmartProcessor)

    result = gui.on_smart_process(["portrait.jpg"], "natural")

    assert len(result) == 3
    proposal, status, explanation = result
    assert proposal["slider_values"]["smooth"] == 77
    assert proposal["proposal_version"] == 1
    assert "sliders were not changed" in status
    assert "Keep texture visible" in explanation


def test_apply_smart_suggestion_is_the_only_slider_mutation_boundary():
    import gui

    proposal = gui.build_smart_proposal(
        SimpleNamespace(
            recipe="natural",
            params={"smooth": 61},
            explanations=["Use restrained smoothing"],
        )
    )
    current_revision = gui._current_settings_revision()

    result = gui.apply_smart_suggestion(proposal, current_revision)
    slider_start = 1
    slider_end = slider_start + len(gui.RECIPE_OUTPUT_KEYS)

    assert len(result) == slider_end + 5
    assert result[0] == "natural"
    assert result[slider_start + gui.RECIPE_OUTPUT_KEYS.index("smooth")] == 61
    assert result[slider_end] is None  # proposal State is consumed
    assert "Applied Smart suggestion" in result[slider_end + 1]
    assert result[slider_end + 3].get("interactive") is False
    assert result[slider_end + 4] > current_revision


def test_revision_and_stale_preview_helpers_are_explicit():
    import gui

    current = gui._current_settings_revision()
    revision, status = gui.on_settings_changed(current)

    assert revision > current
    assert f"Settings revision {revision}" in status
    assert "preview is out of date" in status
    assert "Rendering settings revision 12" in gui.render_status_for_revision(
        "Done", 12, current_revision=13
    )
    assert "out of date" not in gui.render_status_for_revision(
        "Done", 12, current_revision=12
    )


def test_render_event_keeps_legacy_outputs_and_labels_snapshot(monkeypatch):
    import gui

    captured = {}

    def fake_process(*args):
        captured["args"] = args
        return ("rendered", None, None, None, "Done", None, None, "")

    monkeypatch.setattr(
        gui,
        "process_image",
        fake_process,
    )
    source_args = tuple(range(len(gui.PROCESS_INPUT_KEYS)))
    snapshot, _ = gui.capture_render_snapshot(*source_args, 19)
    result = gui.process_image_event(snapshot, 20)

    assert len(result) == 8
    assert captured["args"] == source_args
    assert "Rendering settings revision 19" in result[4]
    assert "preview is out of date" in result[4]


def test_gui_wiring_keeps_long_events_minimal_and_smart_controls_deferred():
    import inspect
    import gui

    source = Path("gui.py").read_text(encoding="utf-8")
    assert 'show_progress="minimal"' in source
    assert 'concurrency_id=GUI_ENGINE_CONCURRENCY_ID' in source
    assert "Apply Smart Suggestion" in source
    assert "outputs=[_smart_proposal_state, status, smart_analysis_html, apply_smart_btn]" in source
    assert "fn=process_image_event" in source
    assert "fn=on_smart_process_event" in source
    assert "list(_recipe_outputs)" not in inspect.getsource(gui.on_smart_process)
