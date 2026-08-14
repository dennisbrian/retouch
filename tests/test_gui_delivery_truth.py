"""Focused GUI contracts for color, precision, inspection, and manifests."""

from __future__ import annotations

import json

import numpy as np


def test_render_snapshot_captures_explicit_source_profile_policy():
    import gui

    values = list(range(len(gui.PROCESS_INPUT_KEYS)))
    values[gui.PROCESS_INPUT_KEYS.index("img_paths")] = ["portrait.jpg"]
    snapshot, _ = gui.capture_render_snapshot(
        *tuple(values),
        7,
        gui.MODE_RENDER_PREVIEW,
        True,
    )

    assert snapshot["preserve_source_profile"] is True
    assert (
        snapshot["render_contract"]["settings"]["snapshot"]["settings"]
        ["preserve_source_profile"]
        is True
    )


def test_native_inspection_is_revision_gated_and_download_disabled():
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache

    cache = GuiPreviewCache()
    cache.set_latest_render(
        {
            "render_revision": 4,
            "native": {"width": 12, "height": 10},
        },
        face_contexts=[{"bbox": (2, 3, 4, 4)}],
    )
    image = np.zeros((10, 12, 3), dtype=np.uint8)
    snapshot = {"settings_revision": 4}

    visible, contract_json = gui.inspect_render_handler(
        image,
        snapshot,
        4,
        "face",
        0,
        "",
        cache,
    )
    assert visible["visible"] is True
    assert visible["value"].shape == (10, 12, 3)
    assert json.loads(contract_json)["download_enabled"] is False

    hidden, stale_message = gui.inspect_render_handler(
        image,
        snapshot,
        5,
        "face",
        0,
        "",
        cache,
    )
    assert hidden["visible"] is False
    assert "stale" in stale_message.lower()


def test_render_manifest_is_pixel_free_and_reports_precision_and_color():
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache

    cache = GuiPreviewCache()
    cache.set_latest_render(
        {
            "render_revision": 3,
            "source_path": "portrait.jpg",
            "native": {"width": 400, "height": 300},
            "output": {"width": 400, "height": 300, "dtype": "uint8"},
            "cache_status": "miss",
            "color_context": {"source_kind": "assumed-srgb"},
            "metadata_result": {"icc": "working-srgb"},
            "face_count": 0,
            "timings_ms": {"total": 12.5},
            "qa": {"warning_count": 0},
            "precision": {"precision_status": "downgraded_to_uint8"},
        }
    )
    snapshot = {
        "settings_revision": 3,
        "render_mode": gui.MODE_RENDER_PREVIEW,
    }
    payload = json.loads(gui.build_render_manifest_handler(snapshot, 3, cache, None))

    assert payload["contract"] == "retouch.render_manifest"
    assert payload["output"]["dtype"] == "uint8"
    assert payload["color_context"]["source_kind"] == "assumed-srgb"
    assert payload["extensions"]["precision"]["precision_status"] == "downgraded_to_uint8"
    assert "pixels" not in json.dumps(payload).lower()
