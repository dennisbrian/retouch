"""Crop UI wiring: selection cleared, gates match other document actions."""
import numpy as np
import pytest
import gradio as gr

from retouch.editor import Document
from retouch.editor.ui import EditorSession, build_editor_tab


@pytest.fixture
def session():
    doc = Document(8, 6)
    doc.add_image(np.full((6, 8, 3), 100, np.uint8), 'Base')
    result = EditorSession.create(doc)
    yield result
    result.close()


@pytest.fixture
def callbacks():
    with gr.Blocks() as app:
        with gr.Tabs():
            build_editor_tab()
    return [value.fn for value in app.fns.values() if value.fn]


def _edit(callbacks):
    return {fn.__kwdefaults__['operation']: fn
            for fn in callbacks if fn.__name__ == 'edit'}


def test_crop_canvas_clears_selection_and_shifts_content(session, callbacks):
    edits = _edit(callbacks)
    session.selection = np.zeros((6, 8), np.uint8)
    session.selection[1:4, 1:5] = 255
    edits['crop'](session, session.selected_id, None, 1, 1, 4, 3)
    doc = session.history.document
    assert (doc.width, doc.height) == (4, 3)
    assert session.selection is None
    assert doc.layers[0].origin == (-1.0, -1.0)
    assert session.history.undo_label == 'Crop'


def test_crop_selection_requires_selection(session, callbacks):
    edits = _edit(callbacks)
    assert session.selection is None
    with pytest.raises(gr.Error, match='Create a selection first'):
        edits['crop_selection'](session, session.selected_id, None)
    session.selection = np.zeros((6, 8), np.uint8)  # entirely empty
    with pytest.raises(gr.Error, match='Create a selection first'):
        edits['crop_selection'](session, session.selected_id, None)


def test_crop_selection_uses_bbox_and_clears(session, callbacks):
    edits = _edit(callbacks)
    session.selection = np.zeros((6, 8), np.uint8)
    session.selection[2:5, 3:7] = 255
    edits['crop_selection'](session, session.selected_id, None)
    doc = session.history.document
    assert (doc.width, doc.height) == (4, 3)
    assert session.selection is None
    assert doc.layers[0].origin == (-3.0, -2.0)
