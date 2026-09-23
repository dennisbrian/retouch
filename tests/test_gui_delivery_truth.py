"""Focused GUI contracts for color, precision, inspection, and manifests."""

from __future__ import annotations

import json

import numpy as np
import pytest


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
            "qa_provenance": {
                "schema": "retouch_qa_reference_v1",
                "reference_stage": "pre_face_post_input_preprocess",
            },
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
    assert payload["extensions"]["qa_provenance"]["schema"] == "retouch_qa_reference_v1"
    assert "pixels" not in json.dumps(payload).lower()


def _render_event_snapshot(gui, source_path, mode=None, revision=8):
    process_args = [None] * len(gui.PROCESS_INPUT_KEYS)
    process_args[gui.PROCESS_INPUT_KEYS.index("img_paths")] = [str(source_path)]
    snapshot = {
        "process_args": tuple(process_args),
        "settings_revision": revision,
    }
    if mode is not None:
        snapshot["render_mode"] = mode
    return snapshot


def test_failed_render_manifest_does_not_relabel_prior_preview_as_success(monkeypatch, tmp_path):
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache
    from retouch.render_manifest import RenderManifest

    old_source = tmp_path / "old.jpg"
    new_source = tmp_path / "new.jpg"
    old_source.write_bytes(b"old source")
    new_source.write_bytes(b"new source")
    cache = GuiPreviewCache()
    cache.set_latest_render(
        {
            "render_revision": 4,
            "render_mode": gui.MODE_RENDER_PREVIEW,
            "source_path": str(old_source),
            "output": {"width": 400, "height": 300, "dtype": "uint8"},
            "face_count": 2,
            "qa": {"warning_count": 1},
            "backend": {"face_detection": {"mode": "face_aware"}},
        },
        face_contexts=[{"bbox": (1, 2, 3, 4)}],
    )
    monkeypatch.setattr(
        gui,
        "process_image",
        lambda *args, **kwargs: (
            None, None, None, None, "Error: No images were successfully processed.",
            None, None, "",
        ),
    )

    snapshot = _render_event_snapshot(gui, new_source)
    result = gui.process_image_event(snapshot, 8, cache)
    assert result[2] is None
    assert cache.latest_render_evidence["render_revision"] == 4
    assert cache.latest_render_evidence["source_path"] == str(old_source)
    assert cache.latest_render_evidence["render_attempt"]["status"] == "failed"

    payload = json.loads(gui.build_render_manifest_handler(snapshot, 8, cache, None))
    manifest = RenderManifest.from_dict(payload)
    assert manifest.status == "failed"
    assert manifest.output is None
    assert manifest.source["id"] == str(new_source)
    assert manifest.face_count is None
    assert manifest.qa == {}
    assert "No images" in manifest.error


def test_multi_input_preview_binds_manifest_to_first_successful_source(monkeypatch, tmp_path):
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache

    failed_source = tmp_path / "first.jpg"
    rendered_source = tmp_path / "second.jpg"
    failed_source.write_bytes(b"first source")
    rendered_source.write_bytes(b"second source")
    cache = GuiPreviewCache()
    snapshot = _render_event_snapshot(gui, failed_source)
    process_args = list(snapshot["process_args"])
    process_args[gui.PROCESS_INPUT_KEYS.index("img_paths")] = [
        str(failed_source), str(rendered_source)
    ]
    snapshot["process_args"] = tuple(process_args)

    def _render_later_source(*args, **kwargs):
        cache.set_latest_render(
            {
                "source_path": str(rendered_source),
                "_evidence_attempt_id": kwargs["render_attempt_id"],
                "face_count": 1,
                "qa": {"warning_count": 0},
                "output": {"width": 24, "height": 16, "dtype": "uint8"},
            },
            face_contexts=[{"bbox": (1, 2, 3, 4)}],
        )
        return (
            None, None, np.zeros((16, 24, 3), dtype=np.uint8), None,
            "Rendered", None, None, "",
        )

    monkeypatch.setattr(gui, "process_image", _render_later_source)
    result = gui.process_image_event(snapshot, 8, cache)

    assert result[2].shape == (16, 24, 3)
    evidence = cache.latest_render_evidence
    assert evidence["source_path"] == str(rendered_source)
    assert evidence["face_count"] == 1
    payload = json.loads(gui.build_render_manifest_handler(snapshot, 8, cache, None))
    assert payload["status"] == "completed"
    assert payload["source"]["id"] == str(rendered_source)
    assert payload["source"]["id"] != str(failed_source)
    assert payload["face_count"] == 1
    assert payload["output"]["dimensions"] == {
        "width": 24,
        "height": 16,
        "channels": 3,
    }


def test_multi_input_legacy_preview_does_not_claim_an_unidentified_source(
    monkeypatch, tmp_path
):
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache

    first_source = tmp_path / "first.jpg"
    second_source = tmp_path / "second.jpg"
    first_source.write_bytes(b"first source")
    second_source.write_bytes(b"second source")
    snapshot = _render_event_snapshot(gui, first_source)
    process_args = list(snapshot["process_args"])
    process_args[gui.PROCESS_INPUT_KEYS.index("img_paths")] = [
        str(first_source), str(second_source)
    ]
    snapshot["process_args"] = tuple(process_args)
    cache = GuiPreviewCache()
    cache.set_latest_render(
        {
            "source_path": str(first_source),
            "face_count": 12,
            "qa": {"warning_count": 4},
        }
    )
    monkeypatch.setattr(
        gui,
        "process_image",
        lambda *args, **kwargs: (
            None, None, np.zeros((16, 24, 3), dtype=np.uint8), None,
            "Rendered", None, None, "",
        ),
    )

    gui.process_image_event(snapshot, 8, cache)
    payload = json.loads(gui.build_render_manifest_handler(snapshot, 8, cache, None))
    assert payload["status"] == "completed"
    assert payload["source"]["id"] == "unknown-source"
    assert payload["face_count"] is None
    assert payload["qa"] == {}


def test_rejected_render_contract_records_failed_attempt(monkeypatch, tmp_path):
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache

    source = tmp_path / "portrait.jpg"
    source.write_bytes(b"source")
    cache = GuiPreviewCache()
    cache.set_latest_render(
        {"render_revision": 2, "source_path": "old.jpg",
         "output": {"width": 20, "height": 10, "dtype": "uint8"}}
    )
    called = []
    monkeypatch.setattr(gui, "process_image", lambda *a, **k: called.append(True))
    snapshot = _render_event_snapshot(gui, source, revision=9)
    snapshot["render_contract"] = {"invalid": True}

    result = gui.process_image_event(snapshot, 9, cache)
    assert result[2] is None
    assert called == []
    payload = json.loads(gui.build_render_manifest_handler(snapshot, 9, cache, None))
    assert payload["status"] == "failed"
    assert payload["render_revision"] == 9
    assert payload["output"] is None
    assert payload["source"]["id"] == str(source)


def test_full_quality_export_discarded_for_newer_settings_is_cancelled(monkeypatch, tmp_path):
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache

    source = tmp_path / "portrait.jpg"
    source.write_bytes(b"source")
    cache = GuiPreviewCache()
    cache.set_latest_render(
        {
            "render_revision": 5,
            "render_mode": gui.MODE_EXPORT_FULL_QUALITY,
            "source_path": str(source),
            "output": {"width": 40, "height": 30, "dtype": "uint8"},
        }
    )
    snapshot = _render_event_snapshot(
        gui, source, mode=gui.MODE_EXPORT_FULL_QUALITY, revision=5
    )
    monkeypatch.setattr(
        gui,
        "process_image",
        lambda *args, **kwargs: (
            None, None, np.zeros((30, 40, 3), dtype=np.uint8), None,
            "Full export rendered", None, None, "",
        ),
    )
    gui.process_image_event(
        snapshot, 5, cache
    )

    payload = json.loads(gui.build_render_manifest_handler(snapshot, 6, cache, None))
    assert payload["status"] == "cancelled"
    assert payload["output"] is None
    assert "settings changed" in payload["error"].lower()


def test_full_quality_export_stale_before_start_is_cancelled(monkeypatch, tmp_path):
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache

    source = tmp_path / "portrait.jpg"
    source.write_bytes(b"source")
    base = _render_event_snapshot(gui, source, revision=5)
    snapshot, _ = gui.capture_render_snapshot(
        *base["process_args"], 5, gui.MODE_EXPORT_FULL_QUALITY, False
    )
    cache = GuiPreviewCache()
    cache.set_latest_render(
        {"render_revision": 4, "source_path": "older.jpg",
         "output": {"width": 20, "height": 10, "dtype": "uint8"}}
    )
    called = []
    monkeypatch.setattr(gui, "process_image", lambda *a, **k: called.append(True))

    gui.process_image_event(snapshot, 6, cache)
    payload = json.loads(
        gui.build_render_manifest_handler(snapshot, 6, cache, None)
    )
    assert called == []
    assert payload["status"] == "cancelled"
    assert payload["output"] is None
    assert "before rendering started" in payload["error"].lower()


def test_full_quality_manifest_uses_written_export_dimensions_and_bit_depth(tmp_path):
    import gui
    from PIL import Image
    from retouch.gui_preview_cache import GuiPreviewCache

    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    output = tmp_path / "export.png"
    Image.new("I;16", (17, 11), 1200).save(output)
    cache = GuiPreviewCache()
    cache.set_latest_render(
        {
            "render_revision": 3,
            "render_mode": gui.MODE_EXPORT_FULL_QUALITY,
            "source_path": str(source),
            "output": {"width": 4000, "height": 3000, "dtype": "uint8"},
        }
    )
    snapshot = {
        "settings_revision": 3,
        "render_mode": gui.MODE_EXPORT_FULL_QUALITY,
    }

    payload = json.loads(
        gui.build_render_manifest_handler(snapshot, 3, cache, str(output))
    )
    assert payload["status"] == "completed"
    assert payload["output"]["dimensions"]["width"] == 17
    assert payload["output"]["dimensions"]["height"] == 11
    assert payload["output"]["dtype"] == "uint16"


@pytest.mark.parametrize("artifact_kind", ["missing", "corrupt"])
def test_full_quality_manifest_rejects_unreadable_export_artifact(
    tmp_path, artifact_kind
):
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache

    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    artifact = tmp_path / "export.png"
    if artifact_kind == "corrupt":
        artifact.write_bytes(b"not an image")
    cache = GuiPreviewCache()
    cache.set_latest_render(
        {
            "render_revision": 3,
            "render_mode": gui.MODE_EXPORT_FULL_QUALITY,
            "source_path": str(source),
            "output": {"width": 4000, "height": 3000, "dtype": "uint8"},
            "face_count": 2,
            "qa": {"warning_count": 1},
            "backend": {"face_detection": {"mode": "face_aware"}},
        }
    )

    payload = json.loads(
        gui.build_render_manifest_handler(
            {"settings_revision": 3, "render_mode": gui.MODE_EXPORT_FULL_QUALITY},
            3,
            cache,
            str(artifact),
        )
    )

    assert payload["status"] == "failed"
    assert payload["output"] is None
    assert payload["face_count"] is None
    assert payload["qa"] == {}
    assert "unreadable or unavailable" in payload["error"]
    assert "output_sha256" not in payload["hashes"]


def test_successful_same_source_legacy_handler_cannot_reuse_old_qa(monkeypatch, tmp_path):
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache

    source = tmp_path / "portrait.jpg"
    source.write_bytes(b"source")
    cache = GuiPreviewCache()
    cache.set_latest_render(
        {
            "render_revision": 4,
            "source_path": str(source),
            "native": {"width": 4000, "height": 3000},
            "output": {"width": 4000, "height": 3000, "dtype": "uint8"},
            "face_count": 7,
            "qa": {"warning_count": 3},
            "backend": {"face_detection": {"mode": "face_aware"}},
        },
        face_contexts=[{"bbox": (1, 2, 3, 4)}],
    )
    rendered = np.zeros((30, 40, 3), dtype=np.uint8)
    monkeypatch.setattr(
        gui,
        "process_image",
        lambda *args, **kwargs: (
            rendered, None, rendered, None, "Preview rendered", None, None, ""
        ),
    )
    snapshot = _render_event_snapshot(gui, source, revision=9)

    gui.process_image_event(snapshot, 9, cache)
    payload = json.loads(gui.build_render_manifest_handler(snapshot, 9, cache, None))
    assert payload["status"] == "completed"
    assert payload["output"]["dimensions"]["width"] == 40
    assert payload["output"]["dimensions"]["height"] == 30
    assert payload["face_count"] is None
    assert payload["qa"] == {}
    assert payload["backend"] == {}
    assert cache.latest_render_face_contexts == ()


def test_inspection_reports_effective_detail_not_just_display_zoom():
    # T2b (RESEARCH_RETOUCH_TARGET_AND_PREVIEW_PARITY_2026_09_23 §4): a fast
    # preview enlarged to native size was shown as "native_one_to_one" with no
    # indication that its detail came from ~800px processing.
    import gui
    from retouch.gui_preview_cache import GuiPreviewCache

    image = np.zeros((10, 12, 3), dtype=np.uint8)
    cache = GuiPreviewCache()
    cache.set_latest_render(
        {"render_revision": 4, "native": {"width": 12, "height": 10},
         "output": {"width": 12, "height": 10}},
        face_contexts=[{"bbox": (2, 3, 4, 4)}],
    )
    preview = {"settings_revision": 4, "render_mode": gui.MODE_RENDER_PREVIEW}
    _, contract_json = gui.inspect_render_handler(image, preview, 4, "100%", 0, "", cache)
    detail = json.loads(contract_json)["detail"]
    assert detail["effective_detail"] == "proxy"

    full = {"settings_revision": 4, "render_mode": gui.MODE_EXPORT_FULL_QUALITY,
            "settings_sha256": "a" * 64}
    _, contract_json = gui.inspect_render_handler(image, full, 4, "100%", 0, "", cache)
    assert json.loads(contract_json)["detail"]["effective_detail"] == "native"
