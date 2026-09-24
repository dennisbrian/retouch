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
    blocked, blocked_status = gui.advanced_export_handler(
        edited, "PNG", None, base_contract, True, False,
    )
    assert blocked["visible"] is False
    assert "review the highlighted" in blocked_status.lower()

    hidden_overlay_export, hidden_overlay_status = gui.advanced_export_handler(
        edited, "PNG", None, base_contract, True, True, False, 42,
    )
    zero_opacity_export, zero_opacity_status = gui.advanced_export_handler(
        edited, "PNG", None, base_contract, True, True, True, 0,
    )
    low_opacity_export, low_opacity_status = gui.advanced_export_handler(
        edited, "PNG", None, base_contract, True, True, True, 1,
    )
    assert hidden_overlay_export["visible"] is False
    assert zero_opacity_export["visible"] is False
    assert low_opacity_export["visible"] is False
    assert "review" in hidden_overlay_status.lower()
    assert "review" in zero_opacity_status.lower()
    assert "review" in low_opacity_status.lower()

    _, unchanged_snapshots, snapshot_status = gui.advanced_save_snapshot_handler(
        "unreviewed", edited, applied[3], {}, base_contract, True, True, False, 42,
    )
    assert unchanged_snapshots == {}
    assert "not saved" in snapshot_status.lower()

    regular_snapshot_args = [None] * len(gui.PROCESS_INPUT_KEYS) + [
        applied[3], base_contract, applied[2], edited, {}, True, True, False, 42,
    ]
    regular_snapshot, saved_snapshots = gui.save_snapshot_handler(
        "unreviewed", *regular_snapshot_args,
    )
    assert saved_snapshots == {}
    assert "choices" not in regular_snapshot

    session_args = [None] * len(gui.PROCESS_INPUT_KEYS) + [
        applied[3], base_contract, applied[2], edited, True, False,
    ]
    not_saved = gui.save_session_handler(*session_args)
    assert not_saved["visible"] is False
    hidden_overlay_session_args = [None] * len(gui.PROCESS_INPUT_KEYS) + [
        applied[3], base_contract, applied[2], edited, True, True, False, 42,
    ]
    hidden_overlay_session = gui.save_session_handler(*hidden_overlay_session_args)
    assert hidden_overlay_session["visible"] is False

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
    assert len(apply_dependency["inputs"]) == 33
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
    assert bound[9] is True
    assert bound[10]["visible"] is True
    assert bound[11]["value"] is True


def test_legacy_binding_refuses_reshape_edits_that_use_detection_order():
    import gui

    source, _editor = _canvas()
    base = build_base_contract(source, kind=BASE_KIND_SOURCE)
    pending = {
        "version": 1,
        "edits": [{"mode": "Reshape", "selection": "0", "reshape": {"eye_size": 15}}],
    }

    bound = gui.advanced_bind_legacy_handler(source, pending, base)

    assert not isinstance(bound[0], np.ndarray)
    assert bound[5]["__type__"] == "update"
    assert "binding stopped" in bound[6].lower()


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


def test_explicit_rebase_warns_that_masks_keep_old_coordinates():
    # T6 (RESEARCH_RETOUCH_TARGET_AND_PREVIEW_PARITY_2026_09_23 §6): loading a
    # new processed result with current edits replays the stored post-semantic
    # mask at its old pixel position; it must not be reported as a plain success.
    import gui
    from retouch.heal import mask_to_b64

    mask = np.zeros((24, 32), dtype=np.float32)
    mask[4:12, 6:18] = 1.0
    edit = {
        "version": 1,
        "image_shape": [24, 32],
        "mode": "Adjust",
        "operation": "exposure",
        "strength": 20.0,
        "semantic": "None",
        "selection": None,
        "mask_png_b64": mask_to_b64(mask),
    }
    processed = np.full((24, 32, 3), 120, dtype=np.uint8)

    rebased = gui.on_advanced_processed_result(processed, [], None, [edit])
    assert "rebase warning" in rebased[8].lower()
    assert "1 masked edit" in rebased[8]

    fresh = gui.on_advanced_processed_result(processed, [], None, [])
    assert "rebase warning" not in fresh[8].lower()


def test_mask_rebase_shows_old_support_and_requires_review():
    import gui
    from retouch.heal import mask_to_b64

    mask = np.zeros((24, 32), dtype=np.float32)
    mask[4:12, 6:18] = 1.0
    edit = {
        "version": 1,
        "image_shape": [24, 32],
        "mode": "Adjust",
        "operation": "exposure",
        "strength": 20.0,
        "semantic": "None",
        "selection": None,
        "mask_png_b64": mask_to_b64(mask),
    }
    processed = np.full((24, 32, 3), 120, dtype=np.uint8)

    rebased = gui.on_advanced_processed_result(processed, [], None, [edit])

    assert rebased[11] is True
    assert rebased[12]["visible"] is True
    assert rebased[12]["value"] is False
    assert rebased[13]["value"] is True
    assert rebased[14]["value"] == 42
    assert np.array_equal(rebased[7][6:12, 8:18], np.ones((6, 10), dtype=np.float32))
    assert not np.array_equal(rebased[3][6, 8], rebased[1][6, 8])
    assert np.array_equal(rebased[3][0, 0], rebased[1][0, 0])
    assert "review the overlay" in rebased[8].lower()

    blocked_apply = gui.advanced_apply_handler(
        rebased[2], rebased[1], rebased[0], rebased[5], rebased[6],
        "Adjust", "exposure", 20, "None", "All faces", "telea",
        "Auto (LaMa if installed)", True, 42,
        *([0] * 15), "Keep", 0, True, False,
    )
    assert not isinstance(blocked_apply[1], np.ndarray)
    assert "review the highlighted" in blocked_apply[7].lower()

    low_opacity_apply = gui.advanced_apply_handler(
        rebased[2], rebased[1], rebased[0], rebased[5], rebased[6],
        "Adjust", "exposure", 20, "None", "All faces", "telea",
        "Auto (LaMa if installed)", True, 1,
        *([0] * 15), "Keep", 0, True, True,
    )
    assert not isinstance(low_opacity_apply[1], np.ndarray)
    assert "review the highlighted" in low_opacity_apply[7].lower()


def test_rebase_overlay_visibility_and_opacity_clear_acknowledgement():
    import gui

    image = np.full((24, 32, 3), 120, dtype=np.uint8)
    mask = np.zeros((24, 32), dtype=np.float32)
    mask[4:12, 6:18] = 1.0

    hidden_overlay, hidden_ack = gui.advanced_overlay_handler(
        image, mask, False, 42, True,
    )
    assert np.array_equal(hidden_overlay, image)
    assert hidden_ack["value"] is False
    assert hidden_ack["interactive"] is False

    transparent_overlay, transparent_ack = gui.advanced_overlay_handler(
        image, mask, True, 0, True,
    )
    assert np.array_equal(transparent_overlay, image)
    assert transparent_ack["value"] is False
    assert transparent_ack["interactive"] is False

    too_subtle_overlay, too_subtle_ack = gui.advanced_overlay_handler(
        image, mask, True, 1, True,
    )
    assert not np.array_equal(too_subtle_overlay, image)
    assert too_subtle_ack["value"] is False
    assert too_subtle_ack["interactive"] is False

    visible_overlay, visible_ack = gui.advanced_overlay_handler(
        image, mask, True, 42, True,
    )
    assert not np.array_equal(visible_overlay, image)
    assert visible_ack["interactive"] is True


def test_exact_mask_session_load_requires_review_of_saved_support(tmp_path):
    import gui

    image, editor = _canvas()
    result = apply_advanced_edit(
        image, editor, "Adjust", "exposure", 20, "None", "All faces", {}, None, None,
    )
    base = build_base_contract(image, kind=BASE_KIND_SOURCE)
    payload = build_session_payload(
        [result.edit], base,
        history_state={"base_cursor": 0, "cursor": 1},
        result_rgb=result.image_rgb,
    )
    session_path = tmp_path / "masked-v2.session.json"
    Session(advanced_retouch=payload).to_file(str(session_path))

    loaded = gui.load_advanced_session_handler(str(session_path), image, base)

    assert len(loaded) == 14
    assert np.array_equal(loaded[2], result.image_rgb)
    assert loaded[10] is True
    assert loaded[11]["value"] is False
    assert loaded[11]["visible"] is True
    assert loaded[12]["value"] is True
    assert loaded[13]["value"] == 42
    assert not np.array_equal(loaded[4], loaded[2])
    assert "support is highlighted" in loaded[9].lower()


def test_pending_v2_mask_replay_requires_review_after_source_or_processed_load(tmp_path):
    import gui

    image, editor = _canvas()
    result = apply_advanced_edit(
        image, editor, "Adjust", "exposure", 20, "None", "All faces", {}, None, None,
    )
    source_path = tmp_path / "pending-source.png"
    cv2.imwrite(str(source_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    source_base = build_base_contract(
        image, kind=BASE_KIND_SOURCE, source_path=source_path,
    )
    source_payload = build_session_payload(
        [result.edit], source_base,
        history_state={"base_cursor": 0, "cursor": 1},
        result_rgb=result.image_rgb,
    )

    loaded_source = gui.on_advanced_source_change([str(source_path)], source_payload)
    assert loaded_source[11] is True
    assert loaded_source[12]["visible"] is True
    assert loaded_source[12]["value"] is False
    assert loaded_source[13]["value"] is True
    assert loaded_source[7] is not None

    evidence = {
        "render_mode": "export_full_quality",
        "render_revision": 3,
        "settings_sha256": "d" * 64,
        "native": {"width": image.shape[1], "height": image.shape[0]},
        "output": {"width": image.shape[1], "height": image.shape[0]},
    }
    processed_base = build_base_contract(
        image, kind=BASE_KIND_PROCESSED, render_evidence=evidence,
    )
    processed_payload = build_session_payload(
        [result.edit], processed_base,
        history_state={"base_cursor": 0, "cursor": 1},
        result_rgb=result.image_rgb,
    )
    loaded_processed = gui.on_advanced_processed_result(
        image, processed_payload, evidence, [],
    )
    assert loaded_processed[11] is True
    assert loaded_processed[12]["visible"] is True
    assert loaded_processed[12]["value"] is False
    assert loaded_processed[13]["value"] is True
    assert loaded_processed[7] is not None


def test_explicit_rebase_refuses_missing_or_mismatched_mask_geometry(monkeypatch):
    import gui
    from retouch.heal import mask_to_b64

    processed = np.full((24, 32, 3), 120, dtype=np.uint8)
    source_mask = np.ones((12, 16), dtype=np.float32)
    edit = {
        "version": 1,
        "image_shape": [24, 32],
        "mode": "Adjust",
        "operation": "exposure",
        "strength": 20.0,
        "mask_png_b64": mask_to_b64(source_mask),
    }
    monkeypatch.setattr(
        gui, "_replay_advanced_state",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unsafe mask replay must not run")),
    )

    result = gui.on_advanced_processed_result(processed, [], None, [edit])
    assert not isinstance(result[1], np.ndarray)
    assert "geometry cannot be transferred safely" in result[8].lower()
    assert isinstance(result[9], dict) and "__type__" in result[9]

    edit["image_shape"] = [24.9, 32.9]
    edit["mask_png_b64"] = mask_to_b64(np.ones((24, 32), dtype=np.float32))
    result = gui.on_advanced_processed_result(processed, [], None, [edit])
    assert not isinstance(result[1], np.ndarray)
    assert "geometry cannot be transferred safely" in result[8].lower()

    edit.pop("image_shape")
    result = gui.on_advanced_processed_result(processed, [], None, [edit])
    assert not isinstance(result[1], np.ndarray)
    assert "geometry cannot be transferred safely" in result[8].lower()


def test_edit_processed_without_result_preserves_advanced_workspace():
    import gui

    result = gui.on_advanced_processed_result(None, [], None, [])
    assert len(result) == 15
    assert not isinstance(result[0], np.ndarray)
    assert "processed result is unavailable" in result[8].lower()
    assert isinstance(result[9], dict) and "__type__" in result[9]


def test_replay_rejects_missing_or_inconsistent_mask_geometry():
    import pytest
    from retouch.heal import mask_to_b64

    image, _ = _canvas()
    mask = np.ones((12, 16), dtype=np.float32)
    edit = {
        "mode": "Adjust",
        "operation": "exposure",
        "strength": 20.0,
        "mask_png_b64": mask_to_b64(mask),
    }
    with pytest.raises(ValueError, match="missing recorded image dimensions"):
        replay_advanced_edits(image, [edit], None, None)

    edit["image_shape"] = [48, 64]
    with pytest.raises(ValueError, match="mask dimensions do not match"):
        replay_advanced_edits(image, [edit], None, None)

    edit["image_shape"] = [48.9, 64.9]
    edit["mask_png_b64"] = mask_to_b64(np.ones((48, 64), dtype=np.float32))
    with pytest.raises(ValueError, match="missing recorded image dimensions"):
        replay_advanced_edits(image, [edit], None, None)


def test_rebase_refuses_reshape_edits_that_use_detection_order():
    import gui

    processed = np.full((24, 32, 3), 120, dtype=np.uint8)
    edit = {
        "version": 1,
        "image_shape": [24, 32],
        "mode": "Reshape",
        "selection": "0",
        "reshape": {"eye_size": 15},
    }

    rebased = gui.on_advanced_processed_result(processed, [], None, [edit])

    assert not isinstance(rebased[1], np.ndarray)
    assert "rebase stopped" in rebased[8].lower()
    assert "detection-order" in rebased[8].lower()


def test_malformed_face_selection_is_rejected_not_treated_as_all_faces():
    # T5c (RESEARCH_RETOUCH_TARGET_AND_PREVIEW_PARITY_2026_09_23 §5): a
    # nonnumeric unrecognised value returned None, which means "all faces".
    import pytest
    from retouch.advanced_retouch import _face_index

    assert _face_index("All faces", 2) is None
    assert _face_index(None, 2) is None
    assert _face_index("1", 2) == 1
    for bad in ("Face 1", "left", object()):
        with pytest.raises(ValueError):
            _face_index(bad, 2)
