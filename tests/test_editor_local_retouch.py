"""Local healing/dodge/burn: exact support, native coordinates and history."""
import cv2
import numpy as np
import pytest

from retouch.editor import Document, DocumentHistory, composite, load_project
from retouch.editor.local_retouch import capture_composite, painted_mask, apply_local_retouch
from retouch.regions import apply_local_adjustment


def history():
    doc = Document(96, 80)
    pixels = np.full((80, 96, 3), [160, 120, 90], np.uint8)
    pixels[36:42, 44:50] = 10
    doc.add_image(pixels, 'Original')
    return DocumentHistory(doc, saved=True)


def brush(capture, alpha=255):
    width, height = capture.preview_size
    strokes = np.zeros((height, width, 4), np.uint8)
    strokes[height // 3:2 * height // 3, width // 3:2 * width // 3] = [20, 70, 220, alpha]
    return {'layers': [strokes]}


@pytest.mark.parametrize('operation,method', [('Heal', 'Telea'), ('Heal', 'Navier-Stokes'),
                                            ('Dodge', 'Telea'), ('Burn', 'Telea')])
def test_each_operation_is_a_separate_undoable_saved_layer(tmp_path, operation, method):
    edits = history()
    before = composite(edits.document)
    source = edits.document.layers[0].pixels.copy()
    _, captured = capture_composite(edits)
    strokes = brush(captured)
    mask = painted_mask(strokes, captured)
    layer_id = apply_local_retouch(edits, captured, strokes, operation, method=method)
    layer = edits.document.layer(layer_id)
    assert len(edits.document.layers) == 2
    assert layer.mask is not None and layer.parent_id is None
    after = composite(edits.document)
    assert not np.array_equal(after[mask > 0], before[mask > 0])
    np.testing.assert_array_equal(after[mask == 0], before[mask == 0])
    np.testing.assert_array_equal(edits.document.layers[0].pixels, source)
    if operation == 'Heal':
        assert after[38, 47, :3].mean() > before[38, 47, :3].mean() + 40
    target = tmp_path / 'local.comp'
    edits.save(target)
    np.testing.assert_array_equal(composite(load_project(target)), after)
    assert edits.undo()
    np.testing.assert_array_equal(composite(edits.document), before)
    assert edits.redo()
    np.testing.assert_array_equal(composite(edits.document), after)


@pytest.mark.parametrize('operation', ['Dodge', 'Burn'])
def test_soft_strokes_are_not_applied_twice(operation):
    edits = history()
    before = composite(edits.document)
    _, captured = capture_composite(edits)
    strokes = brush(captured, alpha=128)
    mask = painted_mask(strokes, captured)
    expected_bgr = apply_local_adjustment(cv2.cvtColor(before, cv2.COLOR_RGBA2BGR),
                                          mask.astype(np.float32) / 255, operation.lower(), 1)
    expected = cv2.cvtColor(expected_bgr, cv2.COLOR_BGR2RGBA)
    expected[mask == 0] = before[mask == 0]
    apply_local_retouch(edits, captured, strokes, operation)
    np.testing.assert_array_equal(composite(edits.document), expected)


def test_one_level_healing_alpha_is_not_mistaken_for_full_opacity():
    edits = history()
    before = composite(edits.document)
    _, captured = capture_composite(edits)
    try:
        apply_local_retouch(edits, captured, brush(captured, alpha=1), 'Heal')
    except ValueError as exc:
        assert 'no visible change' in str(exc)
        assert edits.revision == captured.revision
    delta = np.abs(composite(edits.document).astype(np.int16) - before.astype(np.int16))
    assert delta.max() <= 1


def test_native_coordinates_and_selection_intersection():
    edits = history()
    before = composite(edits.document)
    _, captured = capture_composite(edits, preview_side=48)
    assert captured.preview_size == (48, 40)
    strokes = brush(captured)
    selection = np.zeros((80, 96), np.uint8)
    selection[:, :48] = 255
    mask = np.minimum(painted_mask(strokes, captured), selection)
    layer_id = apply_local_retouch(edits, captured, strokes, 'Dodge', 50, selection=selection)
    layer = edits.document.layer(layer_id)
    assert layer.origin[0] < 48 and layer.size[0] < 48
    assert layer.opacity == .5
    after = composite(edits.document)
    np.testing.assert_array_equal(after[mask == 0], before[mask == 0])
    assert not np.array_equal(after, before)


@pytest.mark.parametrize('failure', ['empty', 'size', 'stale', 'capture', 'operation', 'opacity',
                                   'method', 'selection_shape', 'selection_empty'])
def test_rejected_edits_preserve_pixels_and_history(failure):
    edits = history()
    before = composite(edits.document)
    _, captured = capture_composite(edits)
    strokes = brush(captured)
    options = {}
    if failure == 'empty':
        strokes = {'layers': []}
    elif failure == 'size':
        strokes = {'layers': [np.full((4, 4, 4), 255, np.uint8)]}
    elif failure == 'stale':
        edits.update_layer(edits.document.layers[0].id, name='Changed')
    elif failure == 'capture':
        captured = None
    elif failure == 'opacity':
        options['opacity'] = float('nan')
    elif failure == 'method':
        options['method'] = 'unknown'
    elif failure.startswith('selection_'):
        options['selection'] = np.zeros((2, 2) if failure == 'selection_shape' else (80, 96), np.uint8)
    revision = edits.revision
    with pytest.raises(ValueError):
        apply_local_retouch(edits, captured, strokes, 'Invalid' if failure == 'operation' else 'Heal', **options)
    assert edits.revision == revision and len(edits.document.layers) == 1
    np.testing.assert_array_equal(composite(edits.document), before)


def test_transparent_composite_cannot_be_captured():
    doc = Document(5, 4)
    doc.add_image(np.zeros((4, 5, 4), np.uint8), 'Transparent')
    with pytest.raises(ValueError, match='opaque'):
        capture_composite(DocumentHistory(doc))


def test_blank_canvas_does_not_create_a_healing_layer():
    doc = Document(96, 80)
    doc.add_image(np.zeros((80, 96, 3), np.uint8), 'Black')
    edits = DocumentHistory(doc)
    _, captured = capture_composite(edits)
    revision = edits.revision
    with pytest.raises(ValueError, match='no visible change'):
        apply_local_retouch(edits, captured, brush(captured), 'Heal')
    assert edits.revision == revision
