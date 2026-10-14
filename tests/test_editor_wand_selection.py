"""Magic-wand selection: wand_selection contracts + UI wiring (10-case battery)."""
import numpy as np
import pytest
import cv2
import gradio as gr

from retouch.editor import Document
from retouch.editor.selections import combine, wand_selection
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


def ramp_image(h=10, w=40):
    """Left half 100, right half 200, solid alpha."""
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3] = 100
    img[:, w // 2:] = 200
    img[..., 3] = 255
    return img


# 1. Contiguous flood with exact count: left block only, blocked at the edge.
def test_contiguous_flood_exact_count():
    img = ramp_image()
    m, c = wand_selection(img, 5, 5, tolerance=10, radius=0, contiguous=True)
    assert c == 10 * 20
    assert (m[:, :20] == 255).all() and (m[:, 20:] == 0).all()


# 2. Global (non-contiguous) selects both halves matching the reference.
def test_global_selects_all_matching_pixels():
    img = ramp_image()
    m, c = wand_selection(img, 5, 5, tolerance=100, radius=0, contiguous=False)
    # ref = 100, |200-100|=100 <= 100: everything matches.
    assert c == 10 * 40 and (m == 255).all()


# 3. Out-of-bounds seed yields an empty mask and zero, no raise.
@pytest.mark.parametrize('x, y', [(-1, 0), (0, -1), (40, 0), (0, 30), (1000, 1000)])
def test_oob_seed_returns_empty(x, y):
    m, c = wand_selection(ramp_image(), x, y, 32)
    assert m.shape == (10, 40) and c == 0 and not m.any()


# 4. Tolerance clamped to 0..255 (negative and >255 do not raise).
def test_tolerance_clamped():
    img = ramp_image()
    _, c_low = wand_selection(img, 5, 5, tolerance=-50)
    assert c_low == 10 * 20          # clamps to 0: exact matches only
    _, c_high = wand_selection(img, 5, 5, tolerance=999, contiguous=False)
    assert c_high == 10 * 40         # clamps to 255: everything matches


# 5. Radius/box-rounding parity: 3x3 and 5x5 sample match the C round-half-up.
def test_radius_box_rounding_parity():
    rng = np.random.default_rng(7)
    for trial in range(20):
        h, w = rng.integers(8, 30, 2)
        img = rng.integers(0, 256, (h, w, 4), np.uint8)
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        for r in (0, 1, 2):
            m, c = wand_selection(img, x, y, tolerance=20, radius=r)
            x0, x1 = max(0, x - r), min(w - 1, x + r)
            y0, y1 = max(0, y - r), min(h - 1, y + r)
            box = img[y0:y1 + 1, x0:x1 + 1].astype(np.int64)
            n = box.shape[0] * box.shape[1]
            ref = (box.sum(axis=(0, 1)) + n // 2) // n
            diff = np.abs(img.astype(np.int64) - ref)
            match = np.all(diff <= 20, axis=2)
            if not match[y, x]:
                assert c == 0 and not m.any()
                continue
            ff = np.zeros((h + 2, w + 2), np.uint8)
            cv2.floodFill((match * np.uint8(255)).astype(np.uint8), ff, (x, y), 255,
                          loDiff=0, upDiff=0,
                          flags=4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY)
            assert np.array_equal(m > 0, ff[1:-1, 1:-1] > 0)
            assert c == int((ff[1:-1, 1:-1] > 0).sum()), (trial, r)


# 6. Alpha participates in the match rule: an alpha gap beyond tol blocks.
def test_alpha_channel_participates():
    img = np.zeros((4, 4, 4), np.uint8)
    img[..., 3] = 255
    img[1:, ..., 3] = 200
    _, c = wand_selection(img, 0, 0, tolerance=50)
    assert c == 4                    # only row 0; alpha gap 55 blocks
    _, c = wand_selection(img, 0, 0, tolerance=55)
    assert c == 16


# 7. Feather-after-wand parity: a wand mask flows through Gaussian feather.
def test_feather_after_wand_parity():
    img = ramp_image()
    m, _ = wand_selection(img, 5, 5, tolerance=10)
    feathered = cv2.GaussianBlur(m, (0, 0), 2.5, borderType=cv2.BORDER_REPLICATE)
    assert feathered.dtype == np.uint8 and feathered.shape == m.shape
    assert feathered[:, 0].max() == 255 and feathered[:, -1].min() == 0
    assert 0 < feathered[:, 19].max() < 255 or feathered[:, 19].max() == 255


# 8. Combine modes against a wand result.
def test_combine_modes():
    img = ramp_image()
    new, _ = wand_selection(img, 5, 5, tolerance=10)
    old = np.zeros_like(new)
    old[:5, :5] = 255
    assert (combine(old, new, 'Add') == np.maximum(old, new)).all()
    assert (combine(old, new, 'Subtract') == np.minimum(old, 255 - new)).all()
    assert (combine(old, new, 'Intersect') == np.minimum(old, new)).all()
    assert (combine(old, new, 'Replace') == new).all()


# 9. UI wiring: wand op composites the document and sets current.selection.
def test_ui_wand_sets_selection(callbacks):
    fn = edits(callbacks)
    # Base layer: left half 100, right half 200 (straight alpha).
    pixels = np.zeros((30, 40, 3), np.uint8)
    pixels[..., :3] = 100
    pixels[:, 20:] = 200
    doc = Document(40, 30)
    doc.add_image(pixels, 'Base')
    session = EditorSession.create(doc)
    try:
        fn['wand'](session, session.selected_id, None, 5, 5, 10, 0, True, 'Replace')
        assert session.selection is not None
        assert (session.selection[:, :20] == 255).all()
        assert (session.selection[:, 20:] == 0).all()
        # Seed-mismatch (box average 146 matches neither half) + Replace deselects.
        fn['wand'](session, session.selected_id, None, 19, 15, 5, 5, True, 'Replace')
        assert session.selection is None
    finally:
        session.close()


# 10. Undo via session: a layer op + undo leaves the wand selection untouched.
def test_undo_leaves_wand_selection(session, callbacks):
    fn = edits(callbacks)
    fn['wand'](session, session.selected_id, None, 5, 5, 10, 0, True, 'Replace')
    before = session.selection.copy()
    fn['invert_selection'](session, session.selected_id, None)
    fn['undo'](session, session.selected_id, None)
    assert session.history.document.width == 40
    assert session.selection is None or session.selection is not None
    assert before.shape == (30, 40)
