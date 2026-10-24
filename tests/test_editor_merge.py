"""Merge Down / Merge Group / Merge Layers (upstream ``LayerMerge.swift``)."""
import numpy as np
import pytest

from retouch.editor import Document, DocumentHistory, Layer, composite, load_project
from retouch.editor.adjustments import adjustment


def photo(height=12, width=16, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    a[..., 3] = 255
    return a


def solid(color, width=4, height=4, alpha=255):
    a = np.empty((height, width, 4), np.uint8)
    a[..., :3], a[..., 3] = color, alpha
    return a


def test_merge_down_bakes_blend_opacity_mask_and_adjustment_exactly():
    """Merging onto the opaque bottom layer leaves the canvas byte-identical."""
    base = Layer('Photo', photo())
    mask = np.zeros((6, 8), np.uint8); mask[:, 4:] = 255
    top = Layer('Overlay', photo(6, 8, seed=1), origin=(3, 2), opacity=0.6,
                blend_mode='Multiply', mask=mask)
    doc = Document(16, 12, [base, top])
    history = DocumentHistory(doc, saved=True)
    before = composite(history.document)

    assert history.merge_plan(top.id)[-1] == 'Merge Down'
    assert history.merge_layers(top.id)
    merged = history.document.layers
    assert len(merged) == 1 and merged[0].name == 'Photo'
    assert merged[0].blend_mode == 'Normal' and merged[0].opacity == 1
    assert merged[0].mask is None and history.document.active_layer_id == merged[0].id
    np.testing.assert_array_equal(composite(history.document), before)
    assert history.undo_label == 'Merge Down'

    # An adjustment layer merged down bakes its grade into the pixels.
    history.add_layer(Layer('Exposure', size=(16, 12), adjustment=adjustment('Invert')))
    graded = composite(history.document)
    assert history.merge_layers(history.document.layers[-1].id)
    assert len(history.document.layers) == 1
    assert history.document.layers[0].adjustment is None
    np.testing.assert_array_equal(composite(history.document), graded)


def test_merge_is_one_undo_step_and_survives_save(tmp_path):
    base = Layer('Photo', photo())
    top = Layer('Box', solid((200, 30, 10), 5, 3), origin=(4, 4), opacity=0.5)
    history = DocumentHistory(Document(16, 12, [base, top]), saved=True)
    ids = [layer.id for layer in history.document.layers]
    before = composite(history.document)
    history.merge_layers(top.id)
    after = composite(history.document)
    assert history.undo()
    assert [layer.id for layer in history.document.layers] == ids
    assert not history.dirty
    np.testing.assert_array_equal(composite(history.document), before)
    assert history.redo()
    reopened = load_project(history.save(tmp_path / 'merged.comp'))
    np.testing.assert_array_equal(composite(reopened), after)


def test_result_is_trimmed_to_its_pixels_and_keeps_the_slot():
    """Two small layers over a transparent canvas, with a layer above them."""
    bottom = Layer('A', solid((10, 20, 30), 3, 2), origin=(2, 5))
    middle = Layer('B', solid((90, 80, 70), 2, 4), origin=(6, 1))
    above = Layer('Above', solid((1, 2, 3), 1, 1), origin=(0, 0))
    history = DocumentHistory(Document(16, 12, [bottom, middle, above]))
    before = composite(history.document)
    history.merge_layers(middle.id)
    layers = history.document.layers
    assert [layer.name for layer in layers] == ['A', 'Above']
    merged = layers[0]
    assert merged.origin == (2.0, 1.0) and merged.size == (6, 6)
    assert merged.pixels.shape == (6, 6, 4)
    np.testing.assert_array_equal(composite(history.document), before)


def test_merge_group_replaces_the_folder_in_its_parent():
    base = Layer('Photo', photo())
    folder = Layer('Folder', size=(16, 12), is_group=True, opacity=0.5)
    child_a = Layer('Red', solid((255, 0, 0), 6, 6), origin=(1, 1), parent_id=folder.id)
    child_b = Layer('Blue', solid((0, 0, 255), 6, 6), origin=(4, 4), parent_id=folder.id,
                    blend_mode='Screen')
    history = DocumentHistory(Document(16, 12, [base, folder, child_a, child_b]))
    before = composite(history.document)
    plan = history.merge_plan(folder.id)
    assert plan[-1] == 'Merge Group' and plan[2] == 'Folder'
    history.merge_layers(folder.id)
    layers = history.document.layers
    assert [layer.name for layer in layers] == ['Photo', 'Folder']
    assert not layers[1].is_group and layers[1].parent_id is None
    # Screen inside a pass-through folder blends with the photo below it;
    # merged on its own over transparent it cannot, exactly as upstream.
    assert layers[1].blend_mode == 'Normal'
    # Pixels away from the Screen child agree with the live canvas, to the
    # one level an 8-bit half-transparent bake can round by (upstream bakes
    # into an 8-bit context too).
    diff = np.abs(composite(history.document)[1:4, 1:4].astype(int) - before[1:4, 1:4])
    assert diff.max() <= 1


def test_merge_down_refuses_a_folder_beneath_and_a_lone_layer():
    folder = Layer('Folder', size=(4, 4), is_group=True)
    child = Layer('Child', solid((1, 2, 3)), parent_id=folder.id)
    top = Layer('Top', solid((4, 5, 6)))
    history = DocumentHistory(Document(4, 4, [folder, child, top]))
    assert history.merge_plan(top.id) is None
    with pytest.raises(ValueError, match='Nothing to merge'):
        history.merge_layers(top.id)
    # The only layer inside a folder has nothing beneath it in that folder.
    assert history.merge_plan(child.id) is None
    empty = Layer('Empty', size=(4, 4), is_group=True)
    history.add_layer(empty)
    assert history.merge_plan(empty.id) is None


def test_layers_clipped_to_a_merged_layer_clip_to_the_result():
    base = Layer('Base', solid((80, 80, 80), 4, 4, alpha=255), origin=(2, 2))
    patch = Layer('Patch', solid((10, 200, 10), 2, 2), origin=(3, 3))
    clipped = Layer('Clipped', solid((0, 0, 255), 8, 8), mask_source_id=patch.id)
    history = DocumentHistory(Document(8, 8, [base, patch, clipped]))
    history.merge_layers(patch.id)
    layers = history.document.layers
    assert [layer.name for layer in layers] == ['Base', 'Clipped']
    assert layers[1].mask_source_id == layers[0].id


def test_merge_chosen_layers_takes_the_top_name_and_folder_contents():
    base = Layer('Photo', photo())
    lower = Layer('Lower', solid((255, 0, 0), 3, 3), origin=(1, 1))
    folder = Layer('Folder', size=(16, 12), is_group=True)
    inner = Layer('Inner', solid((0, 255, 0), 3, 3), origin=(8, 6), parent_id=folder.id)
    keep = Layer('Keep', solid((0, 0, 255), 2, 2), origin=(12, 0))
    history = DocumentHistory(Document(16, 12, [base, lower, folder, inner, keep]))
    before = composite(history.document)
    plan = history.merge_plan(keep.id, [lower.id, folder.id])
    assert plan[-1] == 'Merge Layers' and plan[2] == 'Folder'
    assert set(plan[1]) == {lower.id, folder.id, inner.id}
    history.merge_layers(keep.id, [lower.id, folder.id])
    layers = history.document.layers
    assert [layer.name for layer in layers] == ['Photo', 'Folder', 'Keep']
    assert layers[1].origin == (1.0, 1.0) and layers[1].size == (10, 8)
    np.testing.assert_array_equal(composite(history.document), before)


def test_merge_on_full_res_photo_matches_canvas():
    """A 26 MP-class stack stays exact; skipped unless the photo is provided."""
    import os
    path = os.environ.get('RETOUCH_MERGE_PHOTO')
    if not path:
        pytest.skip('set RETOUCH_MERGE_PHOTO to a photo to run the native-size check')
    from PIL import Image
    rgb = np.asarray(Image.open(path).convert('RGBA'))
    h, w = rgb.shape[:2]
    mask = np.zeros((h // 2, w // 2), np.uint8); mask[:, : w // 4] = 255
    top = Layer('Warm', solid((255, 160, 90), w // 2, h // 2), origin=(w // 4, h // 4),
                opacity=0.4, blend_mode='Soft Light', mask=mask)
    history = DocumentHistory(Document(w, h, [Layer('Photo', rgb), top]))
    before = composite(history.document)
    history.merge_layers(top.id)
    np.testing.assert_array_equal(composite(history.document), before)


def _edits():
    import gradio as gr
    from retouch.editor.ui import build_editor_tab
    with gr.Blocks() as app:
        with gr.Tabs():
            build_editor_tab()
    return {fn.__kwdefaults__['operation']: fn for fn in (v.fn for v in app.fns.values() if v.fn)
            if fn.__name__ == 'edit'}


def test_layer_editor_buttons_merge_and_undo():
    import gradio as gr
    from retouch.editor.ui import EditorSession
    doc = Document(16, 12, [Layer('Photo', photo())])
    session = EditorSession.create(doc)
    try:
        edits = _edits()
        session.history.add_layer(Layer('Box', solid((9, 9, 200), 4, 4), origin=(2, 2), opacity=0.5))
        session.selected_id = session.history.document.layers[-1].id
        before = composite(session.history.document)
        edits['merge_down'](session, session.selected_id, None)
        assert [layer.name for layer in session.history.document.layers] == ['Photo']
        assert session.selected_id == session.history.document.layers[0].id
        np.testing.assert_array_equal(composite(session.history.document), before)
        edits['undo'](session, session.selected_id, None)
        assert len(session.history.document.layers) == 2

        ids = [layer.id for layer in session.history.document.layers]
        with pytest.raises(gr.Error, match='at least two'):
            edits['merge_chosen'](session, session.selected_id, None, ids[:1])
        edits['merge_chosen'](session, session.selected_id, None, ids)
        assert [layer.name for layer in session.history.document.layers] == ['Box']
        # Nothing beneath the only layer: refused without a history entry.
        revision = session.history.revision
        with pytest.raises(gr.Error, match='Nothing to merge'):
            edits['merge_down'](session, session.selected_id, None)
        assert session.history.revision == revision
    finally:
        session.close()
