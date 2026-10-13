"""Layer UI contracts including native-pixel preservation and discard gates."""
from pathlib import Path

import gradio as gr
import numpy as np
import pytest
from PIL import Image, ImageCms

from retouch.editor import Document, load_project, composite
from retouch.editor.ui import (
    EditorSession,
    _clear_preview_cache,
    _photo,
    build_editor_tab,
    preview_document,
)


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


def test_effects_callbacks_build_stroke_shadow_and_overlay_and_are_undoable(session, callbacks, tmp_path):
    edits = {fn.__kwdefaults__['operation']: fn for fn in callbacks if fn.__name__ == 'edit'}
    lid = session.selected_id
    before = composite(session.history.document)

    # (stroke on, size, inside, opacity%, colour, shadow on, angle, distance,
    #  blur, opacity%, colour, overlay on, opacity%, colour,
    #  inner shadow on, angle, distance, blur, opacity%, colour,
    #  outer glow on, size, opacity%, colour, inner glow on, size, opacity%, colour)
    result = edits['apply_effects'](session, lid, None, True, 2, False, 75, '#ff0000',
                                    True, 90, 3, 0, 100, '#000000',
                                    True, 40, '#0000ff',
                                    True, 90, 1, 0, 100, '#000000',
                                    True, 4, 100, '#00ff00',
                                    True, 4, 100, '#ffff00')
    effects = session.selected().effects
    assert effects.stroke.size == 2 and effects.stroke.inside is False
    assert effects.stroke.opacity == 0.75 and effects.stroke.red == 1.0
    assert effects.shadow.distance == 3 and effects.shadow.opacity == 1.0
    assert (effects.colorOverlay.red, effects.colorOverlay.blue) == (0.0, 1.0)
    assert effects.colorOverlay.opacity == 0.4
    assert effects.innerShadow.distance == 1 and effects.innerShadow.blur == 0
    assert effects.outerGlow.size == 4 and effects.outerGlow.opacity == 1.0
    assert (effects.innerGlow.red, effects.innerGlow.green) == (1.0, 1.0)
    painted = composite(session.history.document)
    assert not np.array_equal(painted, before)
    # The returned view exposes the new values back to the controls.
    assert result[44] is True and result[45] == 90  # shadow on, angle
    assert result[52]['value'] == '#0000ff'  # overlay colour
    assert result[53] is True and result[54] == 90  # inner shadow on, angle
    assert result[59] is True and result[60] == 4  # outer glow on, size
    assert result[66]['value'] == '#ffff00'  # inner glow colour

    target = tmp_path / 'effects.comp'
    session.history.save(target)
    np.testing.assert_array_equal(composite(load_project(target)), painted)

    assert session.history.undo()
    assert session.selected().effects is None
    np.testing.assert_array_equal(composite(session.history.document), before)

    edits['apply_effects'](session, lid, None, False, 4, False, 100, '#000000',
                           True, 90, 3, 0, 100, '#000000', False, 100, '#000000',
                           False, 90, 1, 0, 100, '#000000',
                           False, 4, 100, '#00ff00', False, 4, 100, '#ffff00')
    assert session.selected().effects.colorOverlay is None
    edits['clear_effects'](session, lid, None)
    assert session.selected().effects is None
    np.testing.assert_array_equal(composite(session.history.document), before)


def test_effects_on_a_folder_are_rejected_before_mutation(session, callbacks):
    edits = {fn.__kwdefaults__['operation']: fn for fn in callbacks if fn.__name__ == 'edit'}
    base = session.selected_id
    edits['group'](session, base, None, [base])
    folder = session.selected_id
    revision = session.history.revision
    with pytest.raises(gr.Error, match='raster layer'):
        edits['apply_effects'](session, folder, None, False, 4, False, 100, '#000000',
                               True, 90, 3, 0, 100, '#000000', False, 100, '#000000',
                               False, 90, 1, 0, 100, '#000000',
                               False, 4, 100, '#00ff00', False, 4, 100, '#ffff00')
    assert session.history.revision == revision and folder == session.selected_id


@pytest.mark.parametrize('color', ['nonsense', '#12', '#gggggg'])
def test_effects_reject_invalid_colour_before_mutation(session, callbacks, color):
    edit = next(fn for fn in callbacks if fn.__name__ == 'edit'
                and fn.__kwdefaults__['operation'] == 'apply_effects')
    revision = session.history.revision
    with pytest.raises(gr.Error, match='#rrggbb'):
        edit(session, session.selected_id, None, True, 2, False, 75, color,
             False, 90, 3, 0, 100, '#000000', False, 100, '#000000',
             False, 90, 1, 0, 100, '#000000',
             False, 4, 100, '#00ff00', False, 4, 100, '#ffff00')
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


def test_preview_cache_reuses_unchanged_layers_and_invalidates_replaced_pixels(monkeypatch):
    from retouch.editor import ui

    _clear_preview_cache()
    calls = {'pixels': 0, 'masks': 0}
    original_pixels, original_masks = ui._preview_lanczos, ui._preview_mask

    def count_pixels(array, size):
        calls['pixels'] += 1
        return original_pixels(array, size)

    def count_masks(array, size):
        calls['masks'] += 1
        return original_masks(array, size)

    monkeypatch.setattr(ui, '_preview_lanczos', count_pixels)
    monkeypatch.setattr(ui, '_preview_mask', count_masks)
    doc = Document(2048, 1024)
    first = np.zeros((1024, 2048, 4), np.uint8)
    first[..., 3] = 255
    second = first.copy()
    second[..., 1] = 180
    layer_a = doc.add_image(first, 'A')
    layer_b = doc.add_image(second, 'B')
    mask = np.full((1024, 2048), 255, np.uint8)
    layer_a.set_mask(mask)

    uncached = preview_document(doc)
    assert calls == {'pixels': 2, 'masks': 1}
    cached = preview_document(doc)
    np.testing.assert_array_equal(cached, uncached)
    assert calls == {'pixels': 2, 'masks': 1}

    replacement = second.copy()
    replacement[..., 2] = 77
    layer_b.replace_pixels(replacement)
    changed = preview_document(doc)
    assert calls == {'pixels': 3, 'masks': 1}
    np.testing.assert_array_equal(changed, preview_document(doc))
    np.testing.assert_array_equal(layer_a.pixels, first)
    np.testing.assert_array_equal(layer_b.pixels, replacement)


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
        assert len(loaded) == 68
    finally:
        replacement.close()


def test_save_and_export_gate_unapplied_strokes(session, callbacks, tmp_path):
    save = next(fn for fn in callbacks if fn.__name__ == 'saving')
    export = next(fn for fn in callbacks if fn.__name__ == 'exporting')
    target = tmp_path / 'out.comp'
    with pytest.raises(gr.Error):
        save(session, str(target), stroke_canvas())
    with pytest.raises(gr.Error):
        export(session, stroke_canvas(), 'PNG (8-bit sRGB)')
    assert not target.exists() and session.history.dirty
    result = save(session, str(target), None)
    assert len(result) == 67 and not session.history.dirty
    np.testing.assert_array_equal(composite(load_project(target)),
                                  composite(session.history.document))
    path = export(session, None, 'PNG (8-bit sRGB)')
    with Image.open(path) as image:
        assert image.size == (8, 6)
        assert image.info['icc_profile']
        np.testing.assert_array_equal(np.asarray(image), composite(session.history.document))
    high_bit = export(session, None, 'TIFF (16-bit sRGB)')
    assert high_bit.endswith('.tif')
    with Image.open(high_bit) as image:
        assert image.tag_v2[258] == (16, 16, 16, 16)
        assert image.info['icc_profile']


def test_ui_edit_callbacks_change_layer_and_restore_with_undo(session, callbacks):
    edits = [fn for fn in callbacks if fn.__name__ == 'edit']
    result = edits[0](session, session.selected_id, None, 'New name', False, 50, True)
    assert len(result) == 67
    layer = session.selected()
    assert (layer.name, layer.visible, layer.opacity) == ('New name', False, 0.5)
    undo = next(fn for fn in edits if fn.__kwdefaults__['operation'] == 'undo')
    undo(session, session.selected_id, None)
    assert session.selected().name == 'Base'


def test_duplicate_and_invert_callbacks_preserve_original_and_round_trip(session, callbacks, tmp_path):
    edits = {fn.__kwdefaults__['operation']: fn for fn in callbacks if fn.__name__ == 'edit'}
    original_id = session.selected_id
    original = composite(session.history.document)
    edits['duplicate'](session, original_id, None)
    duplicate_id = session.selected_id
    assert duplicate_id != original_id and session.selected().name == 'Base copy'
    assert len(session.history.document.layers) == 2
    edits['invert_mask'](session, duplicate_id, None)
    assert session.selected().mask[0, 0] == 0
    assert session.history.document.layer(original_id).mask is None
    np.testing.assert_array_equal(composite(session.history.document), original)
    target = tmp_path / 'new-actions.comp'
    session.history.save(target)
    np.testing.assert_array_equal(composite(load_project(target)), original)
    edits['undo'](session, duplicate_id, None)
    assert session.selected().mask is None
    edits['undo'](session, duplicate_id, None)
    assert session.selected_id == original_id
    edits['redo'](session, original_id, None)
    assert session.selected_id == original_id  # Undo/redo retain valid manual selection.
    assert session.history.document.active_layer_id == duplicate_id


@pytest.mark.parametrize('operation', ['duplicate', 'invert_mask', 'reset_mask'])
def test_new_actions_gate_pending_strokes_and_missing_selection(session, callbacks, operation):
    edit = next(fn for fn in callbacks if fn.__name__ == 'edit'
                and fn.__kwdefaults__['operation'] == operation)
    revision = session.history.revision
    with pytest.raises(gr.Error, match='brush strokes'):
        edit(session, session.selected_id, stroke_canvas())
    assert session.history.revision == revision
    session.history.remove_layer(session.selected_id)
    with pytest.raises(gr.Error, match='Select a layer'):
        edit(session, None, None)


def test_reset_mask_callback_reveals_layer_and_undo_restores_coverage(session, callbacks, tmp_path):
    edits = {fn.__kwdefaults__['operation']: fn for fn in callbacks if fn.__name__ == 'edit'}
    lid = session.selected_id
    expected = composite(session.history.document)
    session.apply_mask(stroke_canvas())
    painted = composite(session.history.document)
    assert not np.array_equal(painted, expected)
    result = edits['reset_mask'](session, lid, None)
    assert len(result) == 67 and session.selected_id == lid
    assert session.selected().mask is None
    np.testing.assert_array_equal(composite(session.history.document), expected)
    session.history.save(tmp_path / 'reset.comp')
    np.testing.assert_array_equal(composite(load_project(tmp_path / 'reset.comp')), expected)
    edits['undo'](session, lid, None)
    np.testing.assert_array_equal(composite(session.history.document), painted)
    edits['redo'](session, lid, None)
    assert not session.history.dirty
    np.testing.assert_array_equal(composite(session.history.document), expected)


def test_proportional_resize_ui_uses_selected_axis_and_can_unlock(session, callbacks):
    resize = next(fn for fn in callbacks if fn.__name__ == 'edit'
                  and fn.__kwdefaults__['operation'] == 'resize')
    lid = session.selected_id
    source = session.selected().pixels
    result = resize(session, lid, None, 16, 999, True, 'Width')
    assert session.selected().size == (16, 12)
    assert result[16:18] == (16, 12)
    resize(session, lid, None, 999, 9, True, 'Height')
    assert session.selected().size == (12, 9)
    resize(session, lid, None, 5, 10, False, 'Width')
    assert session.selected().size == (5, 10)
    assert session.selected().pixels is source
    revision = session.history.revision
    with pytest.raises(gr.Error, match='brush strokes'):
        resize(session, lid, stroke_canvas(), 20, 20, True, 'Width')
    assert session.history.revision == revision


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
    assert len(result) == 70 and session.job.finalized
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


def test_component_blend_mode_round_trips_through_ui(session, callbacks):
    edit = next(fn for fn in callbacks if fn.__name__ == 'edit'
                and fn.__kwdefaults__['operation'] == 'properties')
    result = edit(session, session.selected_id, None, 'Base', True, 100, True, 'Luminosity')
    assert session.selected().blend_mode == 'Luminosity'
    assert result[11]['value'] == 'Luminosity'


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


@pytest.mark.parametrize('x, y', [(0, float('nan')), (None, 0), (True, 0), (1000001.5, 0)])
def test_invalid_transform_preserves_document_and_history(session, callbacks, x, y):
    transform = next(fn for fn in callbacks if fn.__name__ == 'edit'
                     and fn.__kwdefaults__['operation'] == 'transform')
    before = composite(session.history.document)
    with pytest.raises(gr.Error):
        transform(session, session.selected_id, None, x, y, True, False)
    assert session.history.revision == 0
    assert session.selected().origin == (0, 0)
    np.testing.assert_array_equal(composite(session.history.document), before)


def test_fractional_position_moves_layer_and_round_trips(session, callbacks, tmp_path):
    transform = next(fn for fn in callbacks if fn.__name__ == 'edit'
                     and fn.__kwdefaults__['operation'] == 'transform')
    transform(session, session.selected_id, None, 0.5, 0.25, False, False)
    assert session.selected().origin == (0.5, 0.25)
    # Half-pixel x, quarter-pixel y: col 0 covered 0.5, row 0 covered 0.75.
    out = composite(session.history.document)
    assert out[0, 0, 3] == 96 and out[0, 0, :3].tolist() == [100, 100, 100]
    target = tmp_path / 'frac.comp'
    session.history.save(target)
    loaded = load_project(target)
    assert loaded.layers[0].origin == (0.5, 0.25)
    np.testing.assert_array_equal(composite(loaded), out)


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


def test_fit_callback_updates_size_and_position_and_gates_strokes(session, callbacks):
    edits = {fn.__kwdefaults__['operation']: fn for fn in callbacks if fn.__name__ == 'edit'}
    lid = session.selected_id
    session.history.update_layer(lid, size=(4, 4), origin=(-3, 2))
    source = session.selected().pixels
    result = edits['fit'](session, lid, None)
    assert result[12:14] == (1, 0) and result[16:18] == (6, 6)
    assert session.selected_id == lid and session.selected().pixels is source
    edits['undo'](session, lid, None)
    assert session.selected().size == (4, 4) and session.selected().origin == (-3, 2)
    revision = session.history.revision
    with pytest.raises(gr.Error, match='brush strokes'):
        edits['fit'](session, lid, stroke_canvas())
    assert session.history.revision == revision and session.history.can_redo
    session.history.remove_layer(lid)
    with pytest.raises(gr.Error, match='Select a layer'):
        edits['fit'](session, None, None)


def test_finishing_controls_apply_selection_rotation_group_and_grading(session, callbacks, tmp_path):
    edits = {fn.__kwdefaults__['operation']: fn for fn in callbacks if fn.__name__ == 'edit'}
    base = session.selected_id
    edits['transform'](session, base, None, 0, 0, False, False, 90)
    assert session.selected().rotation == 90
    edits['selection'](session, base, None, 'Rectangle', 2, 1, 3, 4, '[]', 0, 'Replace')
    assert np.count_nonzero(session.selection) == 12
    edits['selection_mask'](session, base, None, 'Replace')
    assert session.selected().mask is not None
    edits['group'](session, base, None, [base])
    folder_id = session.selected_id
    assert session.selected().is_group
    edits['add_grade'](session, folder_id, None, 'Exposure', 1, 0, 1, 0, 255, 1, 1, 1)
    assert session.selected().adjustment['kind'] == 'Exposure'
    assert session.selected().parent_id == folder_id
    edits['update_grade'](session, session.selected_id, None, 'Levels', 0, 0, 1.2, 0, 255, 1.1, 1, 1)
    assert session.selected().adjustment['kind'] == 'Levels'
    target = tmp_path / 'finishing.comp'
    session.history.save(target)
    np.testing.assert_array_equal(composite(load_project(target)), composite(session.history.document))
    edits['clear_selection'](session, session.selected_id, None)
    assert session.selection is None


@pytest.mark.parametrize('kind, expected_key', [
    ('Saturation', 'saturationSettings'),
    ('Vibrance', 'vibranceSettings'),
    ('Temperature', 'temperatureSettings'),
    ('Tint', 'tintSettings'),
    ('Brightness', 'brightnessSettings'),
    ('Contrast', 'contrastSettings'),
    ('Hue', 'hueSettings'),
])
def test_new_adjustment_kinds_are_available_through_ui(session, callbacks, kind, expected_key):
    add = next(fn for fn in callbacks if fn.__name__ == 'edit'
               and fn.__kwdefaults__['operation'] == 'add_grade')
    add(session, session.selected_id, None, kind, 0, 0, 1, 0, 255, 1, 1, 1,
        25, 30, 10, -10, 15, 20, 45)
    settings = session.selected().adjustment
    assert settings['kind'] == kind
    assert expected_key in settings


def test_folder_brush_and_invalid_links_preserve_history(session, callbacks):
    edits = {fn.__kwdefaults__['operation']: fn for fn in callbacks if fn.__name__ == 'edit'}
    base = session.selected_id
    edits['group'](session, base, None, [base])
    folder = session.selected_id
    edits['mask'](session, folder, stroke_canvas())
    assert session.selected().mask is not None
    revision = session.history.revision
    with pytest.raises(gr.Error):
        edits['links'](session, folder, None, folder, '')
    assert session.history.revision == revision
    with pytest.raises(gr.Error):
        edits['transform'](session, folder, stroke_canvas(), 0, 0, False, False, 90)
    assert session.history.revision == revision


def test_folder_mask_propagates_to_composite_and_is_undoable():
    doc = Document(8, 6)
    doc.add_image(np.full((6, 8, 3), 100, np.uint8), 'Base')
    session = EditorSession.create(doc)
    try:
        base = session.selected_id
        assert session.history.group_layers([base], 'Folder')
        folder = session.selected_id
        before = composite(session.history.document)
        assert before[0, 0, 3] == 255
        stroke = np.zeros((6, 8, 4), np.uint8)
        stroke[:, :4] = [0, 0, 0, 255]
        assert session.apply_mask({'layers': [stroke]})
        after = composite(session.history.document)
        assert after[0, 0, 3] == 0        # painted half of the folder is hidden
        assert after[0, 7, 3] == 255      # unpainted half still reveals
        session.history.undo()
        np.testing.assert_array_equal(composite(session.history.document), before)
    finally:
        session.close()


@pytest.mark.parametrize('color', [[255,255,255], [255,51,102], [0,0,0]])
def test_explicit_mask_action_overrides_palette_color_and_is_undoable(session, color):
    stroke = np.zeros((6, 8, 4), np.uint8)
    stroke[1:3, 2:4, :3] = color
    stroke[1:3, 2:4, 3] = 255
    before = composite(session.history.document)
    session.apply_mask({'layers': [stroke]}, 'Hide')
    mask = session.selected().mask
    assert (mask[1:3, 2:4] == 0).all() and (mask[0] == 255).all()
    session.apply_mask({'layers': [stroke]}, 'Reveal')
    assert (session.selected().mask == 255).all()
    assert session.history.undo()
    assert (session.selected().mask[1:3, 2:4] == 0).all()
    assert session.history.undo()
    np.testing.assert_array_equal(composite(session.history.document), before)


def test_mask_ui_callback_uses_explicit_action(session, callbacks):
    edit = next(fn for fn in callbacks if fn.__name__ == 'edit'
                and fn.__kwdefaults__['operation'] == 'mask')
    stroke = np.full((6,8,4), [255,51,102,255], np.uint8)
    edit(session, session.selected_id, {'layers':[stroke]}, 'Hide')
    assert not session.selected().mask.any()
    revision = session.history.revision
    with pytest.raises(gr.Error):
        edit(session, session.selected_id, {'layers':[stroke]}, 'Invalid')
    assert session.history.revision == revision
