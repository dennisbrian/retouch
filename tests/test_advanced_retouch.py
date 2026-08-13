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
    exported, export_status = gui.advanced_export_handler(edited, "PNG")
    assert exported["visible"] is True
    assert Path(exported["value"]).is_file()
    assert "Exported" in export_status


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
