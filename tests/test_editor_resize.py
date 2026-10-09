"""Non-destructive scale, alpha edges, masks, persistence and resource limits."""
import numpy as np
import pytest

from retouch.editor import Document, DocumentHistory, composite, load_project, save_project


def document():
    doc = Document(8, 6)
    doc.add_image(np.array([[[255, 0, 0, 255], [0, 255, 0, 255]],
                            [[0, 0, 255, 255], [255, 255, 255, 255]]], np.uint8), 'Patch')
    return doc


def test_resize_retains_source_and_round_trips_transform(tmp_path):
    doc = document()
    pixels = doc.layers[0].pixels
    h = DocumentHistory(doc, saved=True)
    lid = doc.layers[0].id
    h.update_layer(lid, size=(6, 4), sampling='Nearest')
    assert h.document.layers[0].pixels is pixels
    out = composite(h.document)
    assert out[0, 0].tolist() == [255, 0, 0, 255]
    assert out[3, 5].tolist() == [255, 255, 255, 255]
    assert not out[4:].any()
    again = load_project(save_project(h.document, tmp_path / 'scaled.comp'))
    assert again.layers[0].size == (6, 4)
    np.testing.assert_array_equal(again.layers[0].pixels, pixels)
    np.testing.assert_array_equal(composite(again), out)
    h.undo()
    assert not h.dirty
    np.testing.assert_array_equal(composite(h.document), composite(doc))
    h.redo()
    np.testing.assert_array_equal(composite(h.document), out)


def test_scaled_mask_follows_flips_and_placement():
    doc = document()
    layer = doc.layers[0]
    layer.size, layer.origin, layer.sampling, layer.flip_x = (4, 4), (2, 1), 'Nearest', True
    layer.set_mask(np.array([[0, 255], [0, 255]], np.uint8))
    out = composite(doc)
    assert (out[1:5, 2:4, 3] == 255).all()
    assert not out[1:5, 4:6].any()
    assert out[1, 2, :3].tolist() == [0, 255, 0]


@pytest.mark.parametrize('sampling', ['Smooth', 'High quality'])
def test_resizing_transparent_edges_has_no_hidden_color_halo(sampling):
    doc = Document(16, 1)
    layer = doc.add_image(np.array([[[255, 0, 0, 255], [0, 255, 0, 0]]], np.uint8), 'Edge')
    layer.size, layer.sampling = (16, 1), sampling
    out = composite(doc)[0]
    shown = out[:, 3] > 0
    assert (out[shown, 0] == 255).all()
    assert not out[shown, 1:3].any()


def test_replacement_preserves_scaled_size():
    doc = document()
    layer = doc.layers[0]
    layer.size = (6, 4)
    layer.replace_pixels(np.full((2, 2, 3), 100, np.uint8))
    assert layer.size == (6, 4)
    assert composite(doc)[3, 5].tolist() == [100, 100, 100, 255]


def test_render_budget_rejects_large_resize_before_allocation():
    h = DocumentHistory(document(), saved=True)
    with pytest.raises(ValueError, match='budget'):
        h.update_layer(h.document.layers[0].id, size=(20000, 20000))
    assert h.revision == 0 and not h.dirty


def test_scaled_project_budget_is_checked_before_png_decode(tmp_path, monkeypatch):
    import json
    import retouch.editor.project_store as store
    path = save_project(document(), tmp_path / 'scaled.comp')
    manifest = json.loads((path / 'manifest.json').read_text())
    manifest['layers'][0]['transform']['size'] = [20000, 20000]
    (path / 'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(store, '_read_png', lambda *args: pytest.fail('PNG decode reached'))
    with pytest.raises(ValueError, match='size limits'):
        load_project(path)


@pytest.mark.parametrize('axis,width,height,expected', [
    ('Width', 9, 999, (9, 6)), ('Height', 999, 8, (12, 8)),
])
def test_proportional_resize_uses_current_rectangle_and_round_trips(tmp_path, axis, width, height, expected):
    doc = document()
    layer = doc.layers[0]
    layer.size, layer.origin, layer.flip_x = (6, 4), (-1, 2), True
    layer.set_mask(np.array([[0, 255], [255, 80]], np.uint8))
    history = DocumentHistory(doc, saved=True)
    before = composite(doc)
    history.resize_layer(layer.id, width, height, keep_aspect=True, axis=axis)
    resized = history.document.layer(layer.id)
    assert resized.size == expected  # Native source is square, rendered ratio is 3:2.
    assert resized.pixels is layer.pixels and resized.mask is layer.mask
    assert resized.origin == layer.origin and resized.flip_x
    after = composite(history.document)
    history.save(tmp_path / 'proportional.comp')
    loaded = load_project(tmp_path / 'proportional.comp')
    assert loaded.layer(layer.id).size == expected
    np.testing.assert_array_equal(composite(loaded), after)
    history.undo()
    assert history.document.layer(layer.id).size == (6, 4)
    np.testing.assert_array_equal(composite(history.document), before)
    history.redo()
    assert not history.dirty
    np.testing.assert_array_equal(composite(history.document), after)


@pytest.mark.parametrize('old_size,axis,value,expected', [
    ((4, 2), 'Width', 3, (3, 2)),  # 1.5 rounds up, never a fractional size.
    ((2, 4), 'Height', 3, (2, 3)),
    ((20, 1), 'Width', 1, (1, 1)),  # Avoid a zero-height result.
    ((1, 20), 'Height', 1, (1, 1)),
])
def test_proportional_resize_rounding_and_one_pixel_floor(old_size, axis, value, expected):
    doc = document()
    doc.layers[0].size = old_size
    history = DocumentHistory(doc)
    history.resize_layer(doc.layers[0].id, value, value, keep_aspect=True, axis=axis)
    assert history.document.layers[0].size == expected


@pytest.mark.parametrize('value', [0, -1, 2.5, True, float('nan'), float('inf'), None])
def test_proportional_resize_rejects_invalid_primary_without_losing_redo(value):
    history = DocumentHistory(document(), saved=True)
    lid = history.document.layers[0].id
    history.resize_layer(lid, 4, 4)
    history.undo()
    with pytest.raises(ValueError, match='positive whole pixels'):
        history.resize_layer(lid, value, 10, keep_aspect=True)
    assert history.revision == 0 and not history.dirty and history.can_redo


def test_proportional_resize_preserves_budget_limits_and_noop_redo():
    history = DocumentHistory(document(), saved=True)
    lid = history.document.layers[0].id
    history.resize_layer(lid, 4, 4)
    history.undo()
    assert not history.resize_layer(lid, 2, 999, keep_aspect=True)
    assert history.can_redo and not history.dirty
    with pytest.raises(ValueError, match='budget'):
        history.resize_layer(lid, 20000, 1, keep_aspect=True)
    assert history.revision == 0 and not history.dirty and history.can_redo
