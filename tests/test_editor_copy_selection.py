"""Copy/Cut selection as layer: parity with upstream Layer via Copy and Cut.

Upstream (Compositor SelectionClipboard.swift): Layer via Copy draws the
selection's pixels through the selection clip (feather becomes partial
alpha) into a new layer at the region origin, one undo step, and the edit
drops the selection. Cut is that copy plus clearSelectedPixels on the
active layer — transparent holes, never a content fill.
"""
from pathlib import Path

import gradio as gr
import numpy as np
import pytest

from retouch.editor import Document
from retouch.editor.commands import DocumentHistory
from retouch.editor.composite import composite
from retouch.editor.ui import EditorSession, build_editor_tab


def _doc():
    doc = Document(8, 6)
    base = np.zeros((6, 8, 4), np.uint8)
    base[..., :3] = (200, 40, 40)
    base[..., 3] = 255
    doc.add_image(base, 'Base')
    top = np.zeros((6, 8, 4), np.uint8)
    top[..., :3] = (30, 180, 90)
    top[..., 3] = 255
    doc.add_image(top, 'Top')
    return doc


def _selection():
    mask = np.zeros((6, 8), np.uint8)
    mask[1:4, 2:6] = 255
    return mask


def test_copy_bbox_extraction_and_identity():
    history = DocumentHistory(_doc())
    before = composite(history.document)
    history.copy_selection_as_layer(_selection())
    doc = history.document
    # New top layer, original two untouched below it.
    assert len(doc.layers) == 3
    new = doc.layers[-1]
    assert new.name == 'Layer 1'
    assert new.origin == (2.0, 1.0)
    assert new.size == (4, 3)
    assert (new.pixels == before[1:4, 2:6]).all()
    assert history.undo_label == 'Copy selection as layer'
    # Compositing the copy back over the unchanged stack is identity:
    # the selection region comes from the composite, alpha 255 inside.
    after = composite(doc)
    assert (after == before).all()


def test_copy_soft_selection_weights_alpha():
    history = DocumentHistory(_doc())
    mask = np.zeros((6, 8), np.uint8)
    mask[1:4, 2:6] = 255
    mask[1:4, 6] = 128  # feathered/partial coverage edge column
    history.copy_selection_as_layer(mask)
    new = history.document.layers[-1]
    assert new.size == (5, 3)
    assert (new.pixels[:, -1, 3] == 128).all()
    assert (new.pixels[:, :-1, 3] == 255).all()


def test_cut_clears_active_layer_pixels_not_fill():
    history = DocumentHistory(_doc())
    top_id = history.document.layers[-1].id
    history.copy_selection_as_layer(_selection(), top_id, cut=True)
    doc = history.document
    assert len(doc.layers) == 3
    cut_layer = doc.layers[1]  # the original Top, now with holes
    alpha = cut_layer.pixels[..., 3]
    assert (alpha[1:4, 2:6] == 0).all()
    assert (alpha[:, :2] == 255).all() and (alpha[:, 6:] == 255).all()
    assert (cut_layer.pixels[..., :3][1:4, 2:6] == 0).all()
    assert history.undo_label == 'Cut selection as layer'


def test_cut_requires_paintable_layer():
    history = DocumentHistory(_doc())
    doc = history.document
    history.group_layers([doc.layers[-1].id])
    group_id = history.document.active_layer_id
    with pytest.raises(ValueError, match='no pixels to cut'):
        history.copy_selection_as_layer(_selection(), group_id, cut=True)
    empty = Document(8, 6)
    empty.add_layer(__import__('retouch.editor.document', fromlist=['Layer'])
                    .Layer('Blank', size=(8, 6)))  # no pixels
    history3 = DocumentHistory(empty)
    with pytest.raises(ValueError, match='nothing to cut'):
        history3.copy_selection_as_layer(_selection(), empty.layers[0].id, cut=True)


def test_empty_selection_raises_like_crop():
    history = DocumentHistory(_doc())
    with pytest.raises(ValueError, match='Create a selection first'):
        history.copy_selection_as_layer(None)
    with pytest.raises(ValueError, match='Create a selection first'):
        history.copy_selection_as_layer(np.zeros((6, 8), np.uint8))
    with pytest.raises(ValueError, match='Create a selection first'):
        history.copy_selection_as_layer(np.zeros((3, 3), np.uint8))


def test_undo_redo_restores_both_layers():
    history = DocumentHistory(_doc())
    top_id = history.document.layers[-1].id
    original = history.document
    original_pixels = original.layer(top_id).pixels
    history.copy_selection_as_layer(_selection(), top_id, cut=True)
    assert history.undo()
    doc = history.document
    assert len(doc.layers) == 2
    assert (doc.layer(top_id).pixels == original_pixels).all()
    assert history.redo()
    doc = history.document
    assert len(doc.layers) == 3
    assert (doc.layer(top_id).pixels[1:4, 2:6, 3] == 0).all()


def test_off_canvas_clamp_via_selection_mask():
    history = DocumentHistory(_doc())
    mask = np.zeros((6, 8), np.uint8)
    mask[:, :] = 255  # whole canvas: the region cannot leave it
    history.copy_selection_as_layer(mask)
    new = history.document.layers[-1]
    assert new.origin == (0.0, 0.0)
    assert new.size == (8, 6)
    # Selections are canvas-sized arrays, so a bbox that would bleed past
    # the edge simply has no coverage there — the clamp is inherent: pixels
    # outside the canvas can never be selected.
    flat = composite(history.document)
    history.undo()
    assert (composite(history.document) == flat).all()


def test_copy_pixel_identity_with_composite_crop_multilayer():
    doc = _doc()
    doc.layers[1].opacity = 0.5  # blend the copy must reproduce
    history = DocumentHistory(doc)
    flat = composite(history.document)
    history.copy_selection_as_layer(_selection())
    # With the copy added the composite is unchanged (region drawn from it).
    assert (composite(history.document) == flat).all()


def test_save_load_roundtrip(tmp_path: Path):
    history = DocumentHistory(_doc())
    history.copy_selection_as_layer(_selection())
    path = history.save(tmp_path / 'copy.comp')
    from retouch.editor import load_project
    loaded = load_project(path)
    new = loaded.layers[-1]
    assert new.origin == (2.0, 1.0)
    assert new.size == (4, 3)
    assert new.pixels is not None and (new.pixels[..., 3] == 255).all()


def _edit(callbacks):
    return {fn.__kwdefaults__['operation']: fn
            for fn in callbacks if fn.__name__ == 'edit'}


@pytest.fixture
def session():
    result = EditorSession.create(_doc())
    yield result
    result.close()


@pytest.fixture
def callbacks():
    with gr.Blocks() as app:
        with gr.Tabs():
            build_editor_tab()
    return [value.fn for value in app.fns.values() if value.fn]


def test_ui_copy_and_cut_wiring(session, callbacks):
    edits = _edit(callbacks)
    session.selection = _selection()
    edits['copy_selection_as_layer'](session, session.selected_id, None)
    doc = session.history.document
    assert len(doc.layers) == 3
    assert session.selection is None  # dropped like upstream paste
    assert session.selected_id == doc.layers[-1].id

    session.selection = _selection()
    edits['cut_selection_as_layer'](session, session.selected_id, None)
    doc = session.history.document
    assert len(doc.layers) == 4
    assert session.selection is None
    # The cut cleared the selected pixels of the layer active at the time
    # (the copy from the previous step, whose only pixels were the selection).
    assert (doc.layers[-2].pixels[..., 3] == 0).all()
    assert (doc.layers[1].pixels[..., 3] == 255).all()  # untouched original
    assert history_labels(session) == ['Copy selection as layer', 'Cut selection as layer']


def history_labels(session):
    history = session.history
    saved = []
    while history.can_undo:
        saved.append(history.undo_label)
        assert history.undo()
    for _ in saved:
        assert history.redo()
    return list(reversed(saved))


def test_ui_requires_selection(session, callbacks):
    edits = _edit(callbacks)
    assert session.selection is None
    with pytest.raises(gr.Error, match='Create a selection first'):
        edits['copy_selection_as_layer'](session, session.selected_id, None)
    session.selection = np.zeros((6, 8), np.uint8)
    with pytest.raises(gr.Error, match='Create a selection first'):
        edits['cut_selection_as_layer'](session, session.selected_id, None)
