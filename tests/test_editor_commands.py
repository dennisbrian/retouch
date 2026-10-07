"""Editor history contracts: lossless undo, save points and bounded buffers."""
import numpy as np
import pytest

from retouch.editor import Document, DocumentHistory, Layer, composite, load_project


def history(**kwargs):
    doc = Document(3, 2)
    doc.add_image(np.full((2, 3, 3), 90, np.uint8), 'Base')
    return DocumentHistory(doc, **kwargs)


def test_layer_operations_undo_and_redo_restore_stack_and_active_layer():
    h = history(saved=True)
    base = h.document.layers[0].id
    top = Layer('Top', pixels=np.full((2, 3, 4), 255, np.uint8))
    h.add_layer(top)
    h.move_layer(top.id, 0)
    h.remove_layer(top.id)
    assert h.document.active_layer_id == base
    assert h.undo()
    assert [l.id for l in h.document.layers] == [top.id, base]
    assert h.document.active_layer_id == top.id
    assert h.undo()
    assert [l.id for l in h.document.layers] == [base, top.id]
    assert h.undo() and not h.dirty
    assert not h.undo()
    assert h.redo() and h.redo() and h.redo()
    assert [l.id for l in h.document.layers] == [base]
    assert not h.redo()


def test_properties_masks_and_pixels_restore_exact_composite():
    h = history(saved=True)
    layer_id = h.document.layers[0].id
    original = composite(h.document)
    h.update_layer(layer_id, opacity=0.5, name='Changed')
    half = composite(h.document)
    mask = np.array([[0, 255, 128], [255, 0, 255]], np.uint8)
    h.set_mask(layer_id, mask)
    masked = composite(h.document)
    mask[:] = 0  # caller cannot alter the captured edit
    h.replace_pixels(layer_id, np.full((2, 3, 3), 150, np.uint8))
    replaced = composite(h.document)
    for expected in (masked, half, original):
        assert h.undo()
        np.testing.assert_array_equal(composite(h.document), expected)
    for expected in (half, masked, replaced):
        assert h.redo()
        np.testing.assert_array_equal(composite(h.document), expected)


def test_visibility_and_mask_enable_are_undoable():
    h = history()
    lid = h.document.layers[0].id
    h.set_mask(lid, np.zeros((2, 3), np.uint8))
    h.update_layer(lid, mask_enabled=False)
    assert composite(h.document)[..., 3].min() == 255
    h.update_layer(lid, visible=False)
    assert not composite(h.document).any()
    h.undo()
    assert composite(h.document)[..., 3].min() == 255
    h.undo()
    assert not composite(h.document).any()


def test_branching_never_reuses_saved_revision(tmp_path):
    h = history()
    lid = h.document.layers[0].id
    assert h.dirty
    h.update_layer(lid, opacity=0.5)
    h.save(tmp_path / 'saved.comp')
    saved = h.revision
    assert not h.dirty
    h.update_layer(lid, opacity=0.8)
    assert h.dirty
    h.undo()
    assert not h.dirty
    h.undo()
    assert h.dirty
    h.update_layer(lid, opacity=0.3)
    assert h.revision > saved and h.dirty and not h.can_redo


def test_failed_save_keeps_dirty_state_and_save_point(tmp_path, monkeypatch):
    h = history()
    target = tmp_path / 'saved.comp'
    h.save(target)
    lid = h.document.layers[0].id
    h.update_layer(lid, opacity=0.5)
    import retouch.editor.commands as commands

    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(commands, 'save_project', fail)
    with pytest.raises(OSError):
        h.save(target)
    assert h.dirty
    assert load_project(target).layers[0].opacity == 1
    h.undo()
    assert not h.dirty


@pytest.mark.parametrize('changes', [dict(opacity=2), dict(visible='yes'),
                                     dict(origin=(0.5, 0)), dict(id='new')])
def test_rejected_edit_preserves_document_revision_and_redo(changes):
    h = history(saved=True)
    lid = h.document.layers[0].id
    h.update_layer(lid, opacity=0.5)
    h.undo()
    before = composite(h.document)
    with pytest.raises(ValueError):
        h.update_layer(lid, **changes)
    assert h.revision == 0 and not h.dirty and h.can_redo
    np.testing.assert_array_equal(composite(h.document), before)


def test_noop_preserves_redo_and_clean_state():
    h = history(saved=True)
    lid = h.document.layers[0].id
    h.update_layer(lid, opacity=0.5)
    h.undo()
    assert not h.update_layer(lid, opacity=1)
    assert not h.move_layer(lid, 0)
    assert h.can_redo and not h.dirty


def test_snapshots_detach_metadata_and_share_image_buffers():
    doc = Document(3, 2)
    layer = doc.add_image(np.zeros((2, 3, 3), np.uint8), 'Base')
    layer.text = {'content': 'Original'}
    h = DocumentHistory(doc)
    doc.layers[0].text['content'] = 'Outside edit'
    snapshot = h.document
    snapshot.layers[0].name = 'Outside name'
    snapshot.layers[0].text['content'] = 'Outside text'
    h.update_layer(layer.id, opacity=0.5)
    assert h.document.layers[0].name == 'Base'
    assert h.document.layers[0].text['content'] == 'Original'
    assert h.document.layers[0].pixels is layer.pixels
    h.replace_pixels(layer.id, np.ones((2, 3, 3), np.uint8))
    assert h.document.layers[0].text is None
    h.undo()
    assert h.document.layers[0].text['content'] == 'Original'


def test_history_freezes_arrays_assigned_directly_to_input_document():
    doc = Document(3, 2)
    layer = doc.add_image(np.zeros((2, 3, 3), np.uint8), 'Base')
    pixels = np.full((2, 3, 4), 100, np.uint8)
    mask = np.full((2, 3), 255, np.uint8)
    layer.pixels, layer.mask = pixels, mask
    h = DocumentHistory(doc)
    pixels[:] = 0
    mask[:] = 0
    assert h.document.layers[0].pixels[0, 0, 0] == 100
    assert h.document.layers[0].mask[0, 0] == 255


def test_count_limit_prunes_oldest_undo_without_false_clean_state():
    h = history(saved=True, max_entries=2)
    lid = h.document.layers[0].id
    for value in (0.8, 0.6, 0.4):
        h.update_layer(lid, opacity=value)
    assert h.undo() and h.undo() and not h.undo()
    assert h.document.layers[0].opacity == 0.8 and h.dirty


def test_budget_deduplicates_shared_pixels_and_prunes_replacements():
    h = history(saved=True, max_buffer_bytes=48)  # two 24-byte RGBA images
    lid = h.document.layers[0].id
    for value in (0.8, 0.6, 0.4):
        h.update_layer(lid, opacity=value)
    assert h.retained_buffer_bytes == 24
    h.replace_pixels(lid, np.ones((2, 3, 3), np.uint8))
    assert h.retained_buffer_bytes == 48
    h.replace_pixels(lid, np.full((2, 3, 3), 2, np.uint8))
    assert h.retained_buffer_bytes == 48
    assert h.undo() and not h.undo()
    assert h.dirty


def test_current_document_over_budget_remains_usable():
    h = history(max_buffer_bytes=1)
    lid = h.document.layers[0].id
    h.update_layer(lid, opacity=0.5)
    assert not h.can_undo and h.retained_buffer_bytes == 24
    assert composite(h.document)[0, 0, 3] == 128


@pytest.mark.parametrize('kwargs', [dict(max_entries=0), dict(max_entries=True),
                                  dict(max_buffer_bytes=-1)])
def test_invalid_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        history(**kwargs)
