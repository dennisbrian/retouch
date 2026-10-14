"""Canvas resize and crop: pixel identity, offsets, guides, budgets."""
import numpy as np
import pytest

from retouch.editor import Document, DocumentHistory, composite
from retouch.editor.selections import selection, selection_bbox


def document(guides=None):
    doc = Document(8, 6)
    doc.add_image(np.array([[[255, 0, 0, 255], [0, 255, 0, 255]],
                            [[0, 0, 255, 255], [255, 255, 255, 255]]], np.uint8),
                  'Patch', origin=(3, 2))
    doc.guides = list(guides) if guides else None
    return doc


def test_translate_only_keeps_pixel_identity_and_undo_redo():
    doc = document()
    pixels = doc.layers[0].pixels
    h = DocumentHistory(doc, saved=True)
    assert h.canvas_resize(8, 6, content_offset=(5, -3))
    layer = h.document.layers[0]
    assert layer.pixels is pixels
    assert layer.origin == (8.0, -1.0)
    assert (h.document.width, h.document.height) == (8, 6)
    h.undo()
    assert not h.dirty
    assert h.document.layers[0].origin == (3.0, 2.0)
    h.redo()
    assert h.document.layers[0].origin == (8.0, -1.0)
    np.testing.assert_array_equal(h.document.layers[0].pixels, pixels)


def test_same_size_offset_crop_is_one_entry_and_undoes():
    doc = document()
    h = DocumentHistory(doc)
    assert h.crop_document(2, 1, 8, 6)
    assert h.document.layers[0].origin == (1.0, 1.0)
    assert h.undo() and h.document.layers[0].origin == (3.0, 2.0)


def test_identity_crop_is_noop():
    h = DocumentHistory(document())
    assert not h.crop_document(0, 0, 8, 6)
    assert not h.can_undo
    assert not h.canvas_resize(8, 6)  # anchor 4 with same size offsets (0, 0)
    assert not h.can_undo


def test_crop_to_selection_bbox():
    doc = document()
    h = DocumentHistory(doc)
    mask = selection(8, 6, 'Rectangle', x=1, y=1, box_width=4, box_height=3)
    assert selection_bbox(mask) == (1, 1, 4, 3)
    assert selection_bbox(np.zeros((6, 8), np.uint8)) is None
    assert h.crop_document(*selection_bbox(mask))
    assert (h.document.width, h.document.height) == (4, 3)
    assert h.document.layers[0].origin == (2.0, 1.0)
    np.testing.assert_array_equal(
        composite(h.document),
        composite(doc)[1:4, 1:5])


def test_layer_fully_outside_kept_with_shifted_origin():
    doc = Document(8, 6)
    doc.add_image(np.full((2, 2, 4), (10, 20, 30, 255), np.uint8), 'Far', origin=(-4, -4))
    h = DocumentHistory(doc)
    assert h.crop_document(0, 0, 4, 3)
    layer = h.document.layer(doc.layers[0].id)
    assert layer.origin == (-4.0, -4.0)
    assert layer.pixels is not None


def test_guides_offset_by_axis():
    guides = [{'id': '11111111-1111-1111-1111-111111111111', 'axis': 'vertical',
               'position': 5.0},
              {'id': '22222222-2222-2222-2222-222222222222', 'axis': 'horizontal',
               'position': 4.0}]
    h = DocumentHistory(document(guides))
    assert h.canvas_resize(12, 6, anchor=5)  # dx = floor(4*2/2)=4, dy = 0
    by_axis = {g['axis']: g['position'] for g in h.document.guides}
    assert by_axis['vertical'] == 9.0
    assert by_axis['horizontal'] == 4.0


def test_unlinked_mask_placement_origin_shifts():
    doc = document()
    layer = doc.layers[0]
    placement = {'origin': [1.0, 0.0], 'size': [2.0, 2.0], 'rotation': 0.0,
                 'flipX': False, 'flipY': False, 'sampling': 'Nearest'}
    h = DocumentHistory(doc)
    h.set_mask(layer.id, np.full((2, 2), 255, np.uint8), True, placement)
    assert h.canvas_resize(8, 6, content_offset=(2, 1))
    moved = h.document.layers[0]
    assert tuple(moved.mask_placement['origin']) == (3.0, 1.0)
    assert moved.origin == (5.0, 3.0)


def test_budget_rejection_leaves_document_unchanged():
    h = DocumentHistory(document())
    with pytest.raises(ValueError, match='whole number|surface budget'):
        h.canvas_resize(30001, 5)
    with pytest.raises(ValueError, match='surface budget'):
        h.canvas_resize(20000, 20000)
    assert (h.document.width, h.document.height) == (8, 6)
    assert not h.can_undo


@pytest.mark.parametrize('anchor,dx,dy', [
    (0, 0, 0), (1, 2, 0), (2, 4, 0),
    (3, 0, 1), (4, 2, 1), (5, 4, 1),
    (6, 0, 2), (7, 2, 2), (8, 4, 2)])
def test_every_anchor_offset_math(anchor, dx, dy):
    # Grow 8x6 -> 12x8: dx = floor(4 * (anchor % 3) / 2), dy = floor(2 * (anchor // 3) / 2)
    h = DocumentHistory(document())
    assert h.canvas_resize(12, 8, anchor=anchor)
    assert h.document.layers[0].origin == (3.0 + dx, 2.0 + dy)
