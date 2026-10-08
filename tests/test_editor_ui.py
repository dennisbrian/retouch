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
        assert len(loaded) == 12
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
    assert len(result) == 11 and not session.history.dirty
    np.testing.assert_array_equal(composite(load_project(target)),
                                  composite(session.history.document))
    path = export(session, None)
    with Image.open(path) as image:
        assert image.size == (8, 6)
        np.testing.assert_array_equal(np.asarray(image), composite(session.history.document))


def test_ui_edit_callbacks_change_layer_and_restore_with_undo(session, callbacks):
    edits = [fn for fn in callbacks if fn.__name__ == 'edit']
    result = edits[0](session, session.selected_id, None, 'New name', False, 50, True)
    assert len(result) == 11
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
    assert len(result) == 14 and session.job.finalized
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
