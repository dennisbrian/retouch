"""Layer UI contracts including native-pixel preservation and discard gates."""
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image, ImageCms

from retouch.editor import Document, load_project, composite
from retouch.editor.ui import EditorSession, _photo, build_editor_tab, preview_document


@pytest.fixture
def session():
    doc = Document(8, 6)
    doc.add_image(np.full((6, 8, 3), 100, np.uint8), 'Base')
    result = EditorSession.create(doc)
    yield result
    result.close()


def stroke_canvas():
    stroke = np.zeros((3, 4, 4), np.uint8)
    stroke[1, 1] = [0, 0, 0, 255]
    return {'layers': [stroke]}


def test_brush_changes_only_painted_support_and_is_undoable(session):
    layer_id = session.selected_id
    mask = np.arange(48, dtype=np.uint8).reshape(6, 8) + 100
    session.history.set_mask(layer_id, mask)
    before = composite(session.history.document)
    session.apply_mask(stroke_canvas())
    after = session.history.document.layer(layer_id).mask
    np.testing.assert_array_equal(after[0], mask[0])
    np.testing.assert_array_equal(after[:, -1], mask[:, -1])
    assert after[2:4, 2:4].min() < mask[2:4, 2:4].min()
    session.history.undo()
    np.testing.assert_array_equal(composite(session.history.document), before)


def test_empty_strokes_create_no_history_entry(session):
    revision = session.history.revision
    assert not session.apply_mask({'layers': []})
    assert session.history.revision == revision


def test_preview_is_bounded_and_does_not_change_native_source():
    doc = Document(2048, 1024)
    source = np.zeros((1024, 2048, 4), np.uint8)
    source[..., 3] = 255
    source[:, 1024:, 0] = 255
    layer = doc.add_image(source, 'Base')
    preview = preview_document(doc)
    assert preview.shape == (512, 1024, 4)
    assert (doc.width, doc.height) == (2048, 1024)
    np.testing.assert_array_equal(layer.pixels, source)


@pytest.fixture
def callbacks():
    with gr.Blocks() as app:
        with gr.Tabs():
            build_editor_tab()
    functions = [value.fn for value in app.fns.values() if value.fn]
    return functions


def test_open_requires_discard_and_failed_open_preserves_session(session, callbacks, tmp_path):
    opens = [fn for fn in callbacks if fn.__name__ == 'load']
    image_path = tmp_path / 'photo.png'
    Image.new('RGB', (3, 2), 'red').save(image_path)
    with pytest.raises(gr.Error, match='Discard'):
        opens[0](session, str(image_path), '', False, None)
    assert session.history.document.width == 8
    with pytest.raises(gr.Error):
        opens[1](session, None, str(tmp_path / 'missing.comp'), True, None)
    assert Path(session.workspace).is_dir()
    loaded = opens[0](session, str(image_path), '', True, None)
    replacement = loaded[0]
    try:
        assert replacement.history.document.width == 3
        assert replacement.history.dirty
        assert not Path(session.workspace).exists()
        assert len(loaded) == 19
    finally:
        replacement.close()


def test_save_and_export_gate_unapplied_strokes(session, callbacks, tmp_path):
    save = next(fn for fn in callbacks if fn.__name__ == 'saving')
    export = next(fn for fn in callbacks if fn.__name__ == 'exporting')
    target = tmp_path / 'out.comp'
    with pytest.raises(gr.Error):
        save(session, str(target), stroke_canvas())
    with pytest.raises(gr.Error):
        export(session, stroke_canvas())
    assert not target.exists() and session.history.dirty
    result = save(session, str(target), None)
    assert len(result) == 18 and not session.history.dirty
    np.testing.assert_array_equal(composite(load_project(target)),
                                  composite(session.history.document))
    path = export(session, None)
    with Image.open(path) as image:
        assert image.size == (8, 6)
        np.testing.assert_array_equal(np.asarray(image), composite(session.history.document))


def test_ui_edit_callbacks_change_layer_and_restore_with_undo(session, callbacks):
    edits = [fn for fn in callbacks if fn.__name__ == 'edit']
    result = edits[0](session, session.selected_id, None, 'New name', False, 50, True)
    assert len(result) == 18
    layer = session.selected()
    assert (layer.name, layer.visible, layer.opacity) == ('New name', False, 0.5)
    undo = next(fn for fn in edits if fn.__kwdefaults__['operation'] == 'undo')
    undo(session, session.selected_id, None)
    assert session.selected().name == 'Base'


def test_project_reopen_starts_clean_and_preserves_mask(session, callbacks, tmp_path):
    session.apply_mask(stroke_canvas())
    target = tmp_path / 'saved.comp'
    session.history.save(target)
    opener = [fn for fn in callbacks if fn.__name__ == 'load'][1]
    loaded = opener(None, None, str(target), False, None)[0]
    try:
        assert not loaded.history.dirty
        np.testing.assert_array_equal(composite(loaded.history.document),
                                      composite(session.history.document))
    finally:
        loaded.close()


def test_photo_import_preserves_alpha_and_converts_embedded_srgb(tmp_path):
    path = tmp_path / 'tagged.png'
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    Image.new('RGBA', (2, 3), (20, 100, 180, 128)).save(path, icc_profile=profile)
    pixels = _photo(path)
    assert pixels.shape == (3, 2, 4)
    assert pixels[0, 0].tolist() == [20, 100, 180, 128]


def test_programmatic_canvas_updates_do_not_mark_browser_strokes_dirty(callbacks):
    # Only user input enables the stroke unload guard; freshly opened canvas
    # values may contain empty internal brush layers after Gradio processing.
    with gr.Blocks() as app:
        build_editor_tab()
    input_events = [dep for dep in app.config['dependencies']
                    if dep.get('js') and 'retouchLayerStrokeDirty = !!' in dep['js']]
    assert len(input_events) == 1
    assert input_events[0]['targets'][0][1] == 'input'


def test_recipe_callbacks_attach_one_masked_layer_and_block_overlapping_jobs(session, callbacks):
    from retouch.editor.retouch_jobs import render_worker
    prepare = next(fn for fn in callbacks if fn.__name__ == 'prepare_recipe')
    finish = next(fn for fn in callbacks if fn.__name__ == 'finish_recipe')
    prepare(session, 'natural', 50, None)
    with pytest.raises(gr.Error, match='already active'):
        prepare(session, 'natural', 100, None)
    class Engine:
        def process(self, image, **kwargs):
            return np.full_like(image, 200)
    render_worker(session.job.directory, Engine)
    session.job.status = 'completed'
    result = finish(session, None)
    assert len(result) == 21 and session.job.finalized
    assert session.selected().name == 'Retouch · natural'
    assert session.selected().opacity == 0.5
    assert composite(session.history.document)[0, 0].tolist() == [150, 150, 150, 255]


def test_recipe_finalization_preserves_pending_brush_strokes(session, callbacks):
    from retouch.editor.retouch_jobs import render_worker
    prepare = next(fn for fn in callbacks if fn.__name__ == 'prepare_recipe')
    finish = next(fn for fn in callbacks if fn.__name__ == 'finish_recipe')
    prepare(session, 'natural', 100, None)
    class Engine:
        def process(self, image, **kwargs):
            return image
    render_worker(session.job.directory, Engine)
    session.job.status = 'completed'
    result = finish(session, stroke_canvas())
    assert session.job.status == 'stale' and session.job.finalized
    assert len(session.history.document.layers) == 1
    assert result[7] == gr.update()
    assert 'Unapplied strokes' in result[10]


def test_blend_selector_tracks_property_edits_and_undo(session, callbacks):
    edit = next(fn for fn in callbacks if fn.__name__ == 'edit'
                and fn.__kwdefaults__['operation'] == 'properties')
    undo = next(fn for fn in callbacks if fn.__name__ == 'edit'
                and fn.__kwdefaults__['operation'] == 'undo')
    result = edit(session, session.selected_id, None, 'Base', True, 100, True, 'Screen')
    assert session.selected().blend_mode == 'Screen'
    assert result[11]['value'] == 'Screen'
    result = undo(session, session.selected_id, None)
    assert session.selected().blend_mode == result[11]['value'] == 'Normal'


def test_revealing_unsupported_mode_is_rejected_before_mutation(session, callbacks):
    layer = session.selected()
    session.history.update_layer(layer.id, blend_mode='Color Dodge', visible=False)
    revision = session.history.revision
    edit = next(fn for fn in callbacks if fn.__name__ == 'edit'
                and fn.__kwdefaults__['operation'] == 'properties')
    with pytest.raises(gr.Error, match='supported blend'):
        edit(session, layer.id, None, 'Base', True, 100, True, 'Color Dodge')
    assert not session.selected().visible and session.history.revision == revision


def test_position_and_flips_are_undoable_and_saved(session, callbacks, tmp_path):
    transform = next(fn for fn in callbacks if fn.__name__ == 'edit'
                     and fn.__kwdefaults__['operation'] == 'transform')
    original = composite(session.history.document)
    lid = session.selected_id
    result = transform(session, lid, None, -2, 1, True, True)
    assert (session.selected().origin, session.selected().flip_x, session.selected().flip_y) == (
        (-2, 1), True, True)
    assert result[12:16] == (-2, 1, True, True)
    rendered = composite(session.history.document)
    assert not rendered[0].any()
    assert not rendered[:, -2:].any()
    target = tmp_path / 'position.comp'
    session.history.save(target)
    again = load_project(target)
    assert again.layers[0].origin == (-2, 1)
    assert again.layers[0].flip_x and again.layers[0].flip_y
    np.testing.assert_array_equal(composite(again), rendered)
    session.history.undo()
    np.testing.assert_array_equal(composite(session.history.document), original)
    session.history.redo()
    np.testing.assert_array_equal(composite(session.history.document), rendered)


@pytest.mark.parametrize('x, y', [(0.5, 0), (0, float('nan')), (None, 0), (True, 0), (1000001, 0)])
def test_invalid_transform_preserves_document_and_history(session, callbacks, x, y):
    transform = next(fn for fn in callbacks if fn.__name__ == 'edit'
                     and fn.__kwdefaults__['operation'] == 'transform')
    before = composite(session.history.document)
    with pytest.raises(gr.Error):
        transform(session, session.selected_id, None, x, y, True, False)
    assert session.history.revision == 0
    assert session.selected().origin == (0, 0)
    np.testing.assert_array_equal(composite(session.history.document), before)


@pytest.mark.parametrize('flip_x, flip_y', [(True, False), (False, True), (True, True)])
def test_painting_flipped_layer_keeps_canvas_and_native_mask_aligned(session, flip_x, flip_y):
    from retouch.editor.ui import _view
    lid = session.selected_id
    pixels = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    session.history.replace_pixels(lid, pixels)
    session.history.update_layer(lid, flip_x=flip_x, flip_y=flip_y)
    expected = session.selected().pixels
    if flip_y:
        expected = expected[::-1]
    if flip_x:
        expected = expected[:, ::-1]
    np.testing.assert_array_equal(_view(session)[7]['background'], expected)
    stroke = np.zeros((6, 8, 4), np.uint8)
    stroke[1, 2] = [0, 0, 0, 255]
    session.apply_mask({'layers': [stroke]})
    mask = session.selected().mask
    expected_y = 4 if flip_y else 1
    expected_x = 5 if flip_x else 2
    assert mask[expected_y, expected_x] == 0
    assert (mask == 0).sum() == 1
    rendered = composite(session.history.document)
    assert rendered[1, 2, 3] == 0
    assert (rendered[..., 3] == 0).sum() == 1
    session.history.undo()
    assert (composite(session.history.document)[..., 3] == 255).all()


def test_resize_callback_and_brush_keep_native_pixels(session, callbacks):
    resize = next(fn for fn in callbacks if fn.__name__ == 'edit'
                  and fn.__kwdefaults__['operation'] == 'resize')
    source = session.selected().pixels
    lid = session.selected_id
    result = resize(session, lid, None, 16, 12)
    assert result[16:18] == (16, 12)
    assert session.selected().pixels is source
    stroke = np.zeros((12, 16, 4), np.uint8)
    stroke[2:4, 4:6] = [0, 0, 0, 255]
    session.apply_mask({'layers': [stroke]})
    assert session.selected().mask.shape == (6, 8)
    assert session.selected().mask[1, 2] < 255
    assert session.selected().mask[-1, -1] == 255
    session.history.undo()
    session.history.undo()
    assert session.selected().size == (8, 6)
