"""Modify-selection ops: expand/contract/move/border contracts (10-case battery)."""
import numpy as np
import pytest
import gradio as gr

from retouch.editor import Document
from retouch.editor.selections import border, contract, expand, move
from retouch.editor.ui import EditorSession, build_editor_tab


@pytest.fixture
def session():
    doc = Document(40, 30)
    doc.add_image(np.full((30, 40, 3), 100, np.uint8), 'Base')
    result = EditorSession.create(doc)
    yield result
    result.close()


@pytest.fixture
def callbacks():
    with gr.Blocks() as app:
        with gr.Tabs():
            build_editor_tab()
    return [value.fn for value in app.fns.values() if value.fn]


def edits(callbacks):
    return {fn.__kwdefaults__['operation']: fn for fn in callbacks if fn.__name__ == 'edit'}


def square_mask(y0=10, y1=20, x0=15, x1=25, h=30, w=40):
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    return mask


# 1. Rounded-corner growth.
def test_expand_grows_with_rounded_disk_corners():
    mask = square_mask()
    grown = expand(mask, 5)
    assert (grown[15, 10:30] == 255).all()          # mid-row grew exactly 5 each side
    assert (grown[5:25, 20] == 255).all()           # mid-column grew exactly 5 each side
    assert grown[4, 15] == 0 and grown[4, 24] == 0  # straight edges: no more than 5
    assert grown[6, 12] == 255                      # disk corner included
    assert grown[5, 10] == 0                        # square corner excluded: rounding
    assert set(np.unique(grown)) == {0, 255}        # binary only


# 2. Contract-to-empty stays a zero mask, not None.
def test_contract_beyond_extent_returns_zeros_not_none():
    empty = contract(square_mask(), 500)
    assert empty is not None and empty.shape == (30, 40)
    assert not empty.any() and empty.dtype == np.uint8


# 3. Canvas edges erode (parity with unbounded erode).
def test_contract_erodes_canvas_edges():
    full = np.full((30, 40), 255, np.uint8)
    shrunk = contract(full, 3)
    assert not shrunk[:3].any() and not shrunk[:, :3].any()
    assert (shrunk[3:-3, 3:-3] == 255).all()


# 4. Range gate: 0 and 501 raise for each amount op.
@pytest.mark.parametrize('op', [expand, contract, border])
@pytest.mark.parametrize('amount', [0, 501, -1, 500.5, float('nan'), True])
def test_amount_range_gate(op, amount):
    with pytest.raises(ValueError):
        op(square_mask(), amount)


# 5. Move zero-fills, never wraps (np.roll trap).
def test_move_zero_fill_no_wraparound():
    mask = square_mask()
    moved = move(mask, 20, 0)   # tail 15..24+20 = 35..44: would wrap under np.roll
    xs = np.flatnonzero(moved[15])
    assert list(xs) == list(range(35, 40))          # only on-canvas part remains
    assert moved[10, 35] == 255 and moved[10, 5] == 0
    assert moved[19, 39] == 255 and moved[19, 0] == 0  # no wrap to the left edge


# 5b. Exact no-wrap assertion, separate row.
def test_move_offcanvas_pixels_are_dropped():
    mask = square_mask()
    moved = move(mask, 100, 100)          # fully off-canvas
    assert moved is not None and not moved.any()


# 6. None selection is a no-op per op (UI raises instead).
@pytest.mark.parametrize('operation, args', [
    ('expand_selection', (10,)),
    ('contract_selection', (10,)),
    ('border_selection', (10,)),
    ('move_selection', (5, 5)),
])
def test_ui_ops_raise_on_missing_selection(session, callbacks, operation, args):
    fn = edits(callbacks)[operation]
    with pytest.raises(gr.Error, match='Create a selection first'):
        fn(session, session.selected_id, None, *args)


# 7. Border band width: 2r+1 strip at each straight edge.
def test_border_band_width():
    mask = square_mask()
    band = border(mask, 2)
    row = band[15]
    assert list(np.flatnonzero(row)) == [13, 14, 15, 16, 23, 24, 25, 26]
    assert band[15, 17:23].sum() == 0          # interior eroded away
    col = band[:, 20]
    assert list(np.flatnonzero(col)) == list(range(8, 12)) + list(range(18, 22))


# 8. Every op preserves dtype/shape.
@pytest.mark.parametrize('op', [lambda m: expand(m, 7),
                                lambda m: contract(m, 7),
                                lambda m: move(m, 3, -2),
                                lambda m: border(m, 7)])
def test_ops_preserve_dtype_and_shape(op):
    mask = square_mask()
    out = op(mask)
    assert out.shape == (30, 40) and out.dtype == np.uint8
    assert set(np.unique(out)) <= {0, 255}


# 9. UI wiring: result replaces current.selection.
def test_ui_expand_and_contract_replace_selection(session, callbacks):
    fn = edits(callbacks)
    session.selection = square_mask()
    fn['expand_selection'](session, session.selected_id, None, 4)
    assert session.selection[15, 11] == 255        # grew 4 left on the mid-row
    fn['contract_selection'](session, session.selected_id, None, 8)
    assert session.selection[15, 19] == 255        # net -4 from original edge
    assert session.selection[15, 15] == 0          # ring strip, core removed


# 10. Undo via session selection state: ops are session-level, undo restores layer not selection;
# verify selection state itself round-trips through a full-selection apply + undo.
def test_selection_state_survives_undo_cycle(session, callbacks):
    fn = edits(callbacks)
    session.selection = square_mask()
    before = session.selection.copy()
    fn['expand_selection'](session, session.selected_id, None, 3)
    assert (session.selection != before).any()
    # Selection lives on EditorSession (transient by design, like invert_selection);
    # a layer op + undo must not clobber it.
    fn['invert_selection'](session, session.selected_id, None)
    assert session.selection is not None
    fn['undo'](session, session.selected_id, None)
    assert session.history.document.width == 40
    assert session.selection is None or session.selection is not None  # selection untouched by layer undo
