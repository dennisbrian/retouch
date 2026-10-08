"""Editor session (retouch/editor/session.py) and bounded preview (preview.py).

Covers milestone 3's acceptance path without a UI: open a photo, edit a
mask, undo, save, reopen and export without losing edits.
"""
import cv2
import numpy as np
import pytest
from PIL import Image

from retouch.editor import composite, load_project
from retouch.editor.preview import PreviewRenderer, preview_scale
from retouch.editor.session import EditorError, EditorSession


def _photo(tmp_path, name='photo.png', width=120, height=80, seed=0):
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    path = tmp_path / name
    Image.fromarray(rgb).save(path)
    return path, rgb


def _session(tmp_path, **kwargs):
    path, rgb = _photo(tmp_path, **kwargs)
    session = EditorSession(preview_max_side=64)
    session.open(str(path))
    return session, rgb


def _top(session):
    return session.history.document.layers[-1]


def test_opening_a_photo_gives_one_clean_layer(tmp_path):
    session, rgb = _session(tmp_path)
    state = session.state()
    assert state['open'] and state['width'] == 120 and state['height'] == 80
    assert len(state['layers']) == 1 and not state['dirty'] and not state['can_undo']
    assert state['project_path'] is None and state['title'] == 'photo'
    assert np.array_equal(composite(session.history.document)[..., :3], rgb)


def test_open_round_trip_edit_undo_save_reopen_export(tmp_path):
    session, base = _session(tmp_path)
    overlay_path, overlay = _photo(tmp_path, 'retouched.png', seed=1)
    session.add_photo_layer(overlay_path)
    top = _top(session).id
    assert session.state()['selected_id'] == top

    # Hide a stroke: the mask is created and painted as one undo step.
    session.brush(top, [(20, 40), (100, 40)], diameter=16, hardness=1.0, reveal=False)
    state = session.state()
    assert state['dirty'] and state['undo_label'] == 'Edit mask'
    flat = composite(session.history.document)
    assert np.array_equal(flat[40, 60, :3], base[40, 60])       # hidden: base shows
    assert np.array_equal(flat[5, 60, :3], overlay[5, 60])      # untouched: overlay

    session.undo()
    assert _top(session).mask is None
    session.redo()
    assert _top(session).mask is not None

    target = tmp_path / 'edit.comp'
    state = session.save(str(target))
    assert not state['dirty'] and state['project_path'] == str(target)

    reopened = EditorSession(preview_max_side=64)
    reopened.open(str(target))
    assert not reopened.dirty
    assert np.array_equal(composite(reopened.history.document), flat)
    assert np.array_equal(_top(reopened).mask, _top(session).mask)

    out = reopened.export_png(str(tmp_path / 'flat.png'))
    assert np.array_equal(np.asarray(Image.open(out)), flat)
    assert not reopened.dirty


def test_reveal_brush_restores_what_was_hidden(tmp_path):
    session, _ = _session(tmp_path)
    layer = _top(session).id
    session.add_mask(layer, reveal=False)
    session.brush(layer, [(60, 40)], diameter=20, reveal=True)
    mask = _top(session).mask
    assert mask[40, 60] == 255 and mask[0, 0] == 0
    # Reveal on a maskless layer has nothing to do and adds no undo step.
    session.remove_mask(layer)
    revision = session.history.revision
    session.brush(layer, [(60, 40)], diameter=20, reveal=True)
    assert session.history.revision == revision


def test_hidden_layers_and_disabled_masks_refuse_painting(tmp_path):
    # Mirrors BrushTests.foldersHiddenLayersAndDisabledMasksRejectPainting.
    session, _ = _session(tmp_path)
    layer = _top(session).id
    session.update_layer(layer, visible=False)
    with pytest.raises(EditorError):
        session.brush(layer, [(10, 10)], diameter=8, reveal=False)
    session.update_layer(layer, visible=True)
    session.add_mask(layer)
    session.update_layer(layer, mask_enabled=False)
    with pytest.raises(EditorError):
        session.brush(layer, [(10, 10)], diameter=8, reveal=False)


def test_flipped_and_offset_layers_paint_where_the_pointer_is(tmp_path):
    session, _ = _session(tmp_path)
    layer = _top(session).id
    doc = session.history.document
    doc.layers[0].flip_x = True
    doc.layers[0].origin = (10, 0)
    session._start(doc, saved=True)
    session.brush(layer, [(25.5, 40.5)], diameter=3, hardness=1.0, reveal=False)
    flat = composite(session.history.document)
    assert flat[40, 25, 3] == 0              # hidden exactly under the pointer
    assert flat[40, 60, 3] == 255


def test_masks_of_another_size_are_painted_at_layer_size(tmp_path):
    session, _ = _session(tmp_path)
    layer = _top(session).id
    doc = session.history.document
    doc.layers[0].set_mask(np.full((1, 1), 255, np.uint8))
    session._start(doc, saved=True)
    session.brush(layer, [(60, 40)], diameter=10, reveal=False)
    assert _top(session).mask.shape == (80, 120)


def test_layer_panel_edits_are_each_one_undo_step(tmp_path):
    session, _ = _session(tmp_path)
    second, _ = _photo(tmp_path, 'second.png', seed=2)
    session.add_photo_layer(second)
    bottom, top = [layer.id for layer in session.history.document.layers]
    session.update_layer(top, opacity=0.5, name='Retouched')
    assert _top(session).opacity == 0.5 and _top(session).name == 'Retouched'
    session.move_layer(top, 0)
    assert session.history.document.layers[0].id == top
    session.remove_layer(bottom)
    assert [layer.id for layer in session.history.document.layers] == [top]
    for _ in range(3):
        session.undo()
    assert [layer.id for layer in session.history.document.layers] == [bottom, top]
    assert _top(session).opacity == 1.0


@pytest.mark.parametrize('changes', [{'opacity': 1.5}, {'opacity': float('nan')}, {'name': '  '},
                                     {'blend_mode': 'Multiply'}])
def test_invalid_layer_changes_are_refused(tmp_path, changes):
    session, _ = _session(tmp_path)
    with pytest.raises(EditorError):
        session.update_layer(_top(session).id, **changes)


def test_save_and_export_never_replace_other_files_silently(tmp_path):
    session, _ = _session(tmp_path)
    taken = tmp_path / 'taken.png'
    taken.write_bytes(b'not mine')
    with pytest.raises(FileExistsError):
        session.export_png(str(taken))
    assert taken.read_bytes() == b'not mine'
    session.export_png(str(taken), overwrite=True)
    assert Image.open(taken).size == (120, 80)

    folder = tmp_path / 'folder.comp'
    folder.mkdir()
    (folder / 'keep.txt').write_text('mine')
    with pytest.raises(FileExistsError):
        session.save(str(folder))
    with pytest.raises(EditorError):
        session.save(str(folder), overwrite=True)    # not a Compositor project
    assert (folder / 'keep.txt').read_text() == 'mine'
    with pytest.raises(EditorError):
        session.save()                               # no project path yet
    with pytest.raises(EditorError):
        session.save(str(tmp_path / 'missing' / 'x.comp'))


def test_save_as_adds_the_suffix_and_save_reuses_the_path(tmp_path):
    session, _ = _session(tmp_path)
    state = session.save(str(tmp_path / 'project'))
    assert state['project_path'].endswith('project.comp')
    session.update_layer(_top(session).id, opacity=0.25)
    assert session.dirty
    session.save()
    assert load_project(tmp_path / 'project.comp').layers[0].opacity == 0.25


def test_relative_paths_and_unknown_files_are_refused(tmp_path):
    session = EditorSession()
    with pytest.raises(EditorError):
        session.open('photo.jpg')
    with pytest.raises(EditorError):
        session.open(str(tmp_path / 'missing.jpg'))
    notes = tmp_path / 'notes.txt'
    notes.write_text('x')
    with pytest.raises(EditorError):
        session.open(str(notes))
    with pytest.raises(EditorError):
        session.undo()


def test_transparent_pngs_keep_their_alpha(tmp_path):
    rgba = np.zeros((10, 12, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[2:8, 3:9, 3] = 255
    path = tmp_path / 'cutout.png'
    Image.fromarray(rgba).save(path)
    session = EditorSession()
    session.open(str(path))
    assert np.array_equal(_top(session).pixels, rgba)


def test_jpeg_photos_open_through_the_engine_reader(tmp_path):
    rgb = np.zeros((30, 40, 3), np.uint8)
    rgb[:, :20] = (250, 20, 20)
    path = tmp_path / 'photo.jpg'
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 100])
    session = EditorSession()
    session.open(str(path))
    pixels = _top(session).pixels
    assert pixels.shape == (30, 40, 4)
    assert pixels[15, 5, 0] > 230 and pixels[15, 5, 2] < 40     # RGB order, not BGR


def test_preview_is_bounded_and_matches_the_downscaled_composite(tmp_path):
    session, _ = _session(tmp_path, width=400, height=200)
    preview, scale = session.preview()
    assert scale == preview_scale(400, 200, 64) == 0.16
    assert preview.shape == (32, 64, 4)
    full = composite(session.history.document)
    small = cv2.resize(full, (64, 32), interpolation=cv2.INTER_AREA)
    assert np.abs(preview.astype(int) - small.astype(int)).max() <= 1


def test_preview_reuses_scaled_layers_until_they_change(tmp_path):
    session, _ = _session(tmp_path)
    renderer = PreviewRenderer(64)
    doc = session.history.document
    renderer.render(doc)
    cached = {key: value[1] for key, value in renderer._cache.items()}
    renderer.render(session.history.document)
    assert all(renderer._cache[key][1] is small for key, small in cached.items())
    session.brush(_top(session).id, [(60, 40)], diameter=30, reveal=False)
    renderer.render(session.history.document)
    assert len(renderer._cache) == 2            # photo reused, new mask added


def test_preview_keeps_transparent_edges_clean():
    from retouch.editor.preview import _resize_rgba
    rgba = np.zeros((100, 100, 4), np.uint8)
    rgba[:, :50] = (255, 255, 255, 255)        # white, transparent black beside it
    small = _resize_rgba(rgba, 25, 25)
    edge = small[:, 12]
    assert edge[0, 3] > 0 and (edge[:, :3] == 255).all()    # no dark fringe


def test_mask_preview_follows_flips(tmp_path):
    session, _ = _session(tmp_path)
    layer = _top(session).id
    session.brush(layer, [(5, 5)], diameter=6, reveal=False)
    doc = session.history.document
    doc.layers[0].flip_x = True
    session._start(doc, saved=True)
    session._preview.max_side = 120
    mask = session.mask_preview(layer)
    assert mask.shape == (80, 120)
    assert mask[5, 120 - 6] < 128 and mask[5, 5] == 255
