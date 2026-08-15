"""Focused tests for the Advanced Retouch composition layer."""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from retouch.advanced_retouch import (
    apply_advanced_edit,
    before_after,
    extract_editor_image_and_mask,
    mask_overlay,
    replay_advanced_edits,
)
from retouch.advanced_contract import (
    BASE_KIND_PROCESSED,
    BASE_KIND_SOURCE,
    build_base_contract,
    build_session_payload,
)
from retouch.advanced_history import AdvancedHistory
from retouch.session import Session


def _canvas():
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = (90, 110, 130)
    layer = np.zeros((48, 64, 4), dtype=np.uint8)
    layer[14:30, 20:42, 3] = 255
    return image, {"background": image, "layers": [layer]}


def test_editor_value_extracts_alpha_without_selecting_background():
    image, editor = _canvas()
    base, mask = extract_editor_image_and_mask(editor)
    assert np.array_equal(base, image)
    assert mask.shape == image.shape[:2]
    assert mask[20, 30] == 1.0
    assert mask[2, 2] == 0.0


def test_local_adjustment_is_masked_and_replayable():
    image, editor = _canvas()
    result = apply_advanced_edit(
        image, editor, "Adjust", "warmth", 40, "None", "All faces", {}, None, None
    )
    assert result.image_rgb.shape == image.shape
    assert result.image_rgb.dtype == np.uint8
    assert not np.array_equal(result.image_rgb[20, 30], image[20, 30])
    # The existing LAB conversion can quantize an untouched pixel by one code
    # value; it must not receive a material edit outside the feathered mask.
    assert np.max(np.abs(result.image_rgb[2, 2].astype(int) - image[2, 2].astype(int))) <= 1

    replayed = replay_advanced_edits(image, [result.edit], None, None)
    assert np.array_equal(replayed, result.image_rgb)


def test_replay_is_idempotent_for_feathered_semantic_masks():
    """The stored mask is already post-semantic-intersection; replay must not
    intersect it again, or a non-binary (feathered) parser mask would be
    squared and diverge from the original result."""
    image, editor = _canvas()

    class _FakeParser:
        def parse(self, landmarks, bgr, bbox, person_mask=None, ied=None):
            h, w = bgr.shape[:2]
            feathered = np.zeros((h, w), dtype=np.float32)
            feathered[10:35, 15:50] = 0.5
            return SimpleNamespace(skin=feathered)

    class _FakeFace:
        landmarks = None
        bbox = (0, 0, 64, 48)
        ied = 1.0

    class _FakeDetector:
        def detect(self, bgr):
            return [_FakeFace()]

    detector, parser = _FakeDetector(), _FakeParser()
    result = apply_advanced_edit(
        image, editor, "Adjust", "warmth", 40, "Skin", "All faces", {}, detector, parser
    )
    replayed = replay_advanced_edits(image, [result.edit], detector, parser)
    assert np.array_equal(replayed, result.image_rgb)


def test_visual_helpers_preserve_rgb_contract():
    image, editor = _canvas()
    _, mask = extract_editor_image_and_mask(editor)
    overlay = mask_overlay(image, mask)
    comparison = before_after(image, overlay)
    assert overlay.shape == image.shape
    assert overlay.dtype == np.uint8
    assert comparison.ndim == 3 and comparison.shape[2] == 3


def test_session_serializes_advanced_edit_log_only_when_present():
    session = Session(
        recipe="natural",
        params={},
        advanced_retouch={"version": 1, "edits": [{"mode": "Adjust", "operation": "clarity"}]},
    )
    payload = json.loads(session.to_json())
    assert payload["advanced_retouch"]["version"] == 1
    restored = Session.from_json(session.to_json())
    assert restored.advanced_retouch == session.advanced_retouch


def test_model_manifest_contains_no_placeholder_hosts():
    manifest = json.loads(Path("models/manifest.json").read_text())
    for entry in manifest["models"].values():
        assert "github.com/owner/retouch-models" not in str(entry.get("url", ""))


def test_gui_adjust_handler_does_not_construct_face_engine(tmp_path, monkeypatch):
    """Brush-only GUI edits must work in headless/native-model-free mode."""
    import gui

    image, editor = _canvas()
    path = tmp_path / "source.png"
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    monkeypatch.setattr(gui, "get_engine", lambda: (_ for _ in ()).throw(AssertionError("face engine should not load")))

    loaded = gui.on_advanced_source_change([str(path)], [])
    assert np.array_equal(loaded[0], image)
    result = gui.advanced_apply_handler(
        editor, loaded[1], loaded[0], loaded[5], loaded[6],
        "Adjust", "warmth", 40, "None", "All faces", "telea",
        "Auto (LaMa if installed)", True, 42,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    )
    assert result[1].shape == image.shape
    assert result[3] and result[7].startswith("Applied warmth")


def test_gui_advanced_undo_redo_round_trip():
    import gui

    image, editor = _canvas()
    applied = gui.advanced_apply_handler(
        editor, image, image, {"undo": [], "redo": []}, [],
        "Adjust", "warmth", 40, "None", "All faces", "telea",
        "Auto (LaMa if installed)", True, 42,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    )
    edited = applied[1]
    undone = gui.advanced_undo_handler(edited, image, applied[2], applied[3])
    assert np.array_equal(undone[1], image)
    redone = gui.advanced_redo_handler(undone[1], image, undone[2], undone[3])
    assert np.array_equal(redone[1], edited)


def test_gui_advanced_snapshots_compare_and_export(tmp_path, monkeypatch):
    import gui

    image, editor = _canvas()
    applied = gui.advanced_apply_handler(
        editor, image, image, {"undo": [], "redo": []}, [],
        "Adjust", "warmth", 40, "None", "All faces", "telea",
        "Auto (LaMa if installed)", True, 42,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    )
    edited = applied[1]
    choices, snapshots, status = gui.advanced_save_snapshot_handler(
        "warm", edited, applied[3], {},
    )
    assert snapshots["warm"]["edits"] == applied[3]
    assert "warm" in str(choices)
    assert "Saved" in status

    comparison, compare_status = gui.advanced_compare_snapshot_handler(
        "warm", image, snapshots,
    )
    assert comparison.ndim == 3 and comparison.shape[2] == 3
    assert "Comparing" in compare_status

    export_dir = tmp_path / "advanced-export"
    export_dir.mkdir()
    monkeypatch.setattr(gui.tempfile, "mkdtemp", lambda prefix: str(export_dir))
    base_contract = build_base_contract(image, kind=BASE_KIND_SOURCE)
    exported, export_status = gui.advanced_export_handler(
        edited, "PNG", None, base_contract,
    )
    assert exported["visible"] is True
    assert Path(exported["value"]).is_file()
    assert "Exported" in export_status


def test_gui_advanced_export_uses_source_color_context(tmp_path, monkeypatch):
    import gui

    image = np.full((24, 32, 3), 150, dtype=np.uint8)
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    export_dir = tmp_path / "advanced-export"
    export_dir.mkdir()
    monkeypatch.setattr(gui.tempfile, "mkdtemp", lambda prefix: str(export_dir))
    captured = {}

    def fake_write(path, pixels, context, **kwargs):
        captured["context"] = context
        Path(path).write_bytes(b"verified advanced export")

    monkeypatch.setattr(gui, "write_image_with_color_context", fake_write)
    base_contract = build_base_contract(
        image, kind=BASE_KIND_SOURCE, source_path=source,
    )
    exported, status = gui.advanced_export_handler(
        image, "PNG", [str(source)], base_contract,
    )

    assert exported["visible"] is True
    assert captured["context"].source_kind == "assumed-srgb"
    assert "verified-source ICC/EXIF" in status


def test_gui_source_declares_advanced_editor_and_visible_reshape_controls():
    source = Path("gui.py").read_text(encoding="utf-8")
    assert "gr.ImageEditor(" in source
    assert 'choices=["Adjust", "Heal", "Remove", "Reshape"]' in source
    for label in ("Eye size", "Eye distance", "Nose width", "Nose length", "Jaw width", "Chin length", "Mouth size", "Smile", "Forehead", "Left eye size", "Right eye size", "Left nose width", "Right nose width", "Left jaw width", "Right jaw width", "Mask action", "Mask feather"):
        assert f'label="{label}"' in source


def test_gui_advanced_event_graph_exposes_interactive_workspace():
    """Verify the built Gradio graph wires the editor controls end-to-end."""
    import gui

    config = gui.app.get_config_file()
    components = {
        item["id"]: item
        for item in config["components"]
    }

    def component_value(component_id):
        return components[component_id].get("props", {}).get("value")

    advanced_values = {
        component_value(item_id): item_id
        for item_id in components
        if components[item_id].get("type") == "button"
    }
    assert "Apply edit" in advanced_values
    assert "↩ Undo" in advanced_values
    assert "↻ Redo" in advanced_values
    assert "Reset" in advanced_values
    assert "Download current canvas" in advanced_values

    api_names = {
        dependency.get("api_name")
        for dependency in config["dependencies"]
        if dependency.get("backend_fn")
    }
    assert {
        "advanced_apply_handler",
        "advanced_undo_handler",
        "advanced_redo_handler",
        "advanced_reset_handler",
        "advanced_save_snapshot_handler",
        "advanced_compare_snapshot_handler",
        "advanced_export_handler",
    }.issubset(api_names)

    apply_dependency = next(
        dependency
        for dependency in config["dependencies"]
        if dependency.get("api_name") == "advanced_apply_handler"
    )
    assert len(apply_dependency["inputs"]) == 31
    assert len(apply_dependency["outputs"]) == 8


def test_semantic_intersection_respects_selected_face_without_native_models():
    """Exercise the face-aware mask contract with deterministic test doubles."""
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :] = (90, 110, 130)
    layer = np.zeros((48, 64, 4), dtype=np.uint8)
    layer[:, :, 3] = 255

    faces = [
        SimpleNamespace(landmarks=object(), bbox=(0, 0, 20, 20), ied=10.0),
        SimpleNamespace(landmarks=object(), bbox=(32, 0, 20, 20), ied=10.0),
    ]

    class Detector:
        def detect(self, _image):
            return faces

    left = np.zeros((48, 64), dtype=np.float32)
    left[10:30, 4:20] = 1.0
    right = np.zeros((48, 64), dtype=np.float32)
    right[10:30, 40:56] = 1.0

    class Parser:
        def parse(self, landmarks, *_args, **_kwargs):
            return SimpleNamespace(skin=left if landmarks is faces[0].landmarks else right)

    result = apply_advanced_edit(
        image, {"background": image, "layers": [layer]},
        "Adjust", "exposure", 40, "Skin", "1", {}, Detector(), Parser(),
    )
    assert result.mask[15, 45] == 1.0
    assert result.mask[15, 10] == 0.0
    assert not np.array_equal(result.image_rgb[15, 45], image[15, 45])
    assert np.max(np.abs(result.image_rgb[15, 10].astype(int) - image[15, 10].astype(int))) <= 1


def test_processed_recipe_result_can_become_advanced_source():
    import gui

    processed = np.full((24, 32, 3), 177, dtype=np.uint8)
    loaded = gui.on_advanced_processed_result(processed, [])
    assert np.array_equal(loaded[0], processed)
    assert np.array_equal(loaded[1], processed)
    assert "processed recipe result" in loaded[8].lower()
    assert loaded[10]["render"]["effective_detail"] == "unverified"


def test_preview_derived_advanced_export_is_blocked_before_writer(tmp_path, monkeypatch):
    import gui

    image = np.full((24, 32, 3), 177, dtype=np.uint8)
    preview_base = build_base_contract(
        image,
        kind=BASE_KIND_PROCESSED,
        render_evidence={
            "render_mode": "render_preview",
            "render_revision": 2,
            "settings_sha256": "b" * 64,
            "native": {"width": 32, "height": 24},
            "output": {"width": 32, "height": 24},
        },
    )
    monkeypatch.setattr(
        gui,
        "write_image_with_color_context",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("writer must not run")),
    )

    exported, status = gui.advanced_export_handler(image, "PNG", None, preview_base)

    assert exported["visible"] is False
    assert "blocked" in status.lower()


def test_full_quality_processed_result_records_native_delivery_evidence():
    import gui

    processed = np.full((24, 32, 3), 177, dtype=np.uint8)
    evidence = {
        "render_mode": "export_full_quality",
        "render_revision": 3,
        "settings_sha256": "c" * 64,
        "native": {"width": 32, "height": 24},
        "output": {"width": 32, "height": 24},
    }

    loaded = gui.on_advanced_processed_result(processed, [], evidence, [])

    assert loaded[10]["render"]["effective_detail"] == "native"
    assert loaded[10]["render"]["delivery_eligible"] is True
    assert "verified" in loaded[8].lower()


def test_v2_advanced_session_refuses_same_size_wrong_base(tmp_path):
    import gui

    expected_source = np.full((24, 32, 3), 80, dtype=np.uint8)
    active_source = np.full((24, 32, 3), 81, dtype=np.uint8)
    expected_base = build_base_contract(expected_source, kind=BASE_KIND_SOURCE)
    active_base = build_base_contract(active_source, kind=BASE_KIND_SOURCE)
    payload = build_session_payload(
        [],
        expected_base,
        history_state={"base_cursor": 0, "cursor": 0},
        result_rgb=expected_source,
    )
    session_path = tmp_path / "advanced-v2.session.json"
    Session(advanced_retouch=payload).to_file(str(session_path))

    loaded = gui.load_advanced_session_handler(str(session_path), active_source, active_base)

    assert np.array_equal(loaded[2], active_source)
    assert loaded[1]["version"] == 2
    assert "base mismatch" in loaded[9].lower()


def test_legacy_advanced_session_requires_explicit_binding(tmp_path):
    import gui

    source, editor = _canvas()
    edit = apply_advanced_edit(
        source, editor, "Adjust", "warmth", 40,
        "None", "All faces", {}, None, None,
    ).edit
    session_path = tmp_path / "legacy.session.json"
    Session(advanced_retouch={"version": 1, "edits": [edit]}).to_file(str(session_path))
    base = build_base_contract(source, kind=BASE_KIND_SOURCE)

    loaded = gui.load_advanced_session_handler(str(session_path), source, base)
    assert np.array_equal(loaded[2], source)
    assert "legacy" in loaded[9].lower()

    bound = gui.advanced_bind_legacy_handler(source, loaded[1], base)
    assert not np.array_equal(bound[0], source)
    assert bound[5] == []
    assert "explicitly bound" in bound[6].lower()


def test_advanced_edit_rejects_history_truncation():
    import gui

    image, editor = _canvas()
    first = apply_advanced_edit(
        image, editor, "Adjust", "warmth", 20,
        "None", "All faces", {}, None, None,
    )
    history = AdvancedHistory(max_entries=1)
    assert history.append(first.edit).changed

    applied = gui.advanced_apply_handler(
        editor, first.image_rgb, image, history.to_state(), [first.edit],
        "Adjust", "warmth", 40, "None", "All faces", "telea",
        "Auto (LaMa if installed)", True, 42,
        0, 0, 0, 0, 0, 0, 0, 0, 0,
    )

    assert np.array_equal(applied[1], first.image_rgb)
    assert "replay-safe history boundary" in applied[7]


def test_clear_mask_preserves_pixels_and_removes_selection():
    import gui

    image, editor = _canvas()
    cleared = gui.advanced_clear_mask_handler(editor, image, image)
    assert cleared[1] is None
    assert np.array_equal(cleared[2], image)
    assert cleared[0]["layers"] == []
    assert "Cleared" in cleared[4]


def test_person_and_background_semantic_masks_intersect_brush():
    image = np.full((24, 32, 3), 100, dtype=np.uint8)
    layer = np.zeros((24, 32, 4), dtype=np.uint8)
    layer[:, :, 3] = 255

    class Detector:
        def segment_person(self, _image):
            mask = np.zeros((24, 32), dtype=np.float32)
            mask[4:20, 8:24] = 1.0
            return mask

    for semantic, inside, outside in (
        ("Person", (12, 16), (2, 2)),
        ("Background", (2, 2), (12, 16)),
    ):
        result = apply_advanced_edit(
            image, {"background": image, "layers": [layer]},
            "Adjust", "exposure", 30, semantic, "All faces", {}, Detector(), None,
        )
        assert result.mask[inside] == 1.0
        assert result.mask[outside] == 0.0
