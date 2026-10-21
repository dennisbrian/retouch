"""Local healing/dodge/burn: exact support, native coordinates and history."""
import cv2
import numpy as np
import pytest

from retouch.editor import Document, DocumentHistory, composite, load_project
from retouch.editor.local_retouch import (
    capture_composite, painted_mask, apply_local_retouch, _warp_pixels)
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


@pytest.mark.parametrize('offset,alpha', [((-12, 0), 255), ((4, -9), 128), ((1, 0), 1)])
def test_clone_samples_original_snapshot_with_exact_soft_coverage(tmp_path, offset, alpha):
    doc = Document(96, 80)
    rng = np.random.default_rng(8)
    doc.add_image(rng.integers(0, 256, (80, 96, 3), dtype=np.uint8), 'Texture')
    edits = DocumentHistory(doc, saved=True)
    before = composite(edits.document)
    source = doc.layers[0].pixels.copy()
    _, captured = capture_composite(edits, preview_side=48)
    strokes = brush(captured, alpha)
    selection = np.zeros((80, 96), np.uint8)
    selection[:, :48] = 255
    mask = np.minimum(painted_mask(strokes, captured), selection)
    ys, xs = np.nonzero(mask)
    expected = before.copy()
    a = mask[ys, xs, None].astype(np.uint32)
    donor = before[ys + offset[1], xs + offset[0]].astype(np.uint32)
    expected[ys, xs] = ((donor * a + before[ys, xs].astype(np.uint32) * (255-a) + 127) // 255)
    lid = apply_local_retouch(edits, captured, strokes, 'Clone', selection=selection,
                             clone_offset=offset)
    np.testing.assert_array_equal(composite(edits.document), expected)
    np.testing.assert_array_equal(edits.document.layers[0].pixels, source)
    assert edits.document.layer(lid).name == 'Clone'
    path = tmp_path / 'clone.comp'
    edits.save(path)
    np.testing.assert_array_equal(composite(load_project(path)), expected)
    edits.undo()
    np.testing.assert_array_equal(composite(edits.document), before)
    edits.redo()
    np.testing.assert_array_equal(composite(edits.document), expected)


@pytest.mark.parametrize('offset', [(0, 0), (1.5, 0), (True, 0), (float('nan'), 0),
                                    (float('inf'), 0), (96, 0), (-40, 0), (0, 60), (1,), None])
def test_invalid_clone_source_preserves_document(offset):
    edits = history()
    before = composite(edits.document)
    _, captured = capture_composite(edits)
    with pytest.raises(ValueError, match='clone|Clone'):
        apply_local_retouch(edits, captured, brush(captured), 'Clone', clone_offset=offset)
    assert edits.revision == captured.revision and len(edits.document.layers) == 1
    np.testing.assert_array_equal(composite(edits.document), before)


# --- Blur / Smudge / Liquify -------------------------------------------------

def edge_history():
    """A hard vertical edge, as upstream BlurBrushTests uses."""
    doc = Document(200, 100)
    pixels = np.zeros((100, 200, 3), np.uint8)
    pixels[:, 100:] = 255
    doc.add_image(pixels, 'Edge')
    return DocumentHistory(doc)


def blur_brush(capture):
    width, height = capture.preview_size
    strokes = np.zeros((height, width, 4), np.uint8)
    strokes[35:65, 80:120] = [255, 0, 0, 255]
    return {'layers': [strokes]}


@pytest.mark.parametrize('operation', ['Blur', 'Smudge', 'Liquify'])
def test_warp_operation_is_separate_undoable_layer(tmp_path, operation):
    edits = edge_history()
    before = composite(edits.document)
    _, captured = capture_composite(edits)
    strokes = blur_brush(captured)
    mask = painted_mask(strokes, captured)
    layer_id = apply_local_retouch(edits, captured, strokes, operation,
                                   brush_size=24, brush_strength=90)
    layer = edits.document.layer(layer_id)
    assert layer.name == operation and len(edits.document.layers) == 2
    after = composite(edits.document)
    assert not np.array_equal(after[mask > 0], before[mask > 0])
    np.testing.assert_array_equal(after[mask == 0], before[mask == 0])
    target = tmp_path / 'warp.comp'
    edits.save(target)
    np.testing.assert_array_equal(composite(load_project(target)), after)
    assert edits.undo()
    np.testing.assert_array_equal(composite(edits.document), before)
    assert edits.redo()
    np.testing.assert_array_equal(composite(edits.document), after)


def test_blur_radius_sets_how_far_it_softens():
    """Translated from upstream BlurBrushTests.radiusSetsHowFarItSoftens:
    the softening reach grows with the brush radius, independent of stroke."""
    def spread(brush_size):
        edits = edge_history()
        _, captured = capture_composite(edits)
        apply_local_retouch(edits, captured, blur_brush(captured), 'Blur',
                            brush_size=brush_size)
        row = composite(edits.document)[50, :, 0]
        return int(((row > 10) & (row < 245)).sum())
    narrow, wide = spread(8), spread(48)
    assert wide > narrow * 2, f'wider radius spreads further: {narrow} vs {wide}'


def test_blur_matches_gaussian_oracle():
    edits = edge_history()
    _, captured = capture_composite(edits)
    strokes = blur_brush(captured)
    mask = painted_mask(strokes, captured)
    before = composite(edits.document)
    apply_local_retouch(edits, captured, strokes, 'Blur', brush_size=24)
    expected = cv2.GaussianBlur(before, (0, 0), sigmaX=12, sigmaY=12,
                                borderType=cv2.BORDER_REPLICATE)
    a = mask[..., None].astype(np.uint32)
    expected = ((expected.astype(np.uint32) * a + before.astype(np.uint32) * (255 - a) + 127)
                // 255).astype(np.uint8)
    np.testing.assert_array_equal(composite(edits.document), expected)


def test_smudge_drags_colour_along_the_stroke():
    """Upstream: a smudged bright region leaves one trail that fades, not ghost copies."""
    rng = np.random.default_rng(0)
    region = np.zeros((21, 41, 4), np.uint8)
    region[..., 3] = 255
    region[:, 20:] = 255
    mask = np.zeros((21, 41), np.uint8)
    mask[7:14, 8:33] = 255
    out = _warp_pixels(region.copy(), mask, 'Smudge', 6, 0.5, 0.9)
    # What the first dab picked up (black) darkens the far end of the stroke.
    assert out[10, 28:33, 0].mean() < 200
    # A dab's circle is wider than the painted line (native: the brush, not
    # the stroke shape, decides what moves). Rows/columns no dab reached are intact.
    np.testing.assert_array_equal(out[:1, :, :], region[:1, :, :])
    np.testing.assert_array_equal(out[20:, :, :], region[20:, :, :])
    np.testing.assert_array_equal(out[:, :3, :], region[:, :3, :])
    np.testing.assert_array_equal(out[:, 38:, :], region[:, 38:, :])


def test_liquify_pushes_pixels_with_the_stroke():
    region = np.zeros((21, 41, 4), np.uint8)
    region[..., 3] = 255
    region[:, 20:] = 255
    mask = np.zeros((21, 41), np.uint8)
    mask[7:14, 8:33] = 255
    out = _warp_pixels(region.copy(), mask, 'Liquify', 6, 0.5, 0.9)
    # The white half pushed left into the black half at the stroke centre.
    assert out[10, 18, 0] < 255
    np.testing.assert_array_equal(out[:1, :, :], region[:1, :, :])
    np.testing.assert_array_equal(out[20:, :, :], region[20:, :, :])
    np.testing.assert_array_equal(out[:, :3, :], region[:, :3, :])
    np.testing.assert_array_equal(out[:, 38:, :], region[:, 38:, :])


@pytest.mark.parametrize('operation', ['Blur', 'Smudge', 'Liquify'])
def test_soft_warp_strokes_blend_once(operation):
    edits = edge_history()
    before = composite(edits.document)
    _, captured = capture_composite(edits)
    strokes = blur_brush(captured)
    strokes['layers'] = [strokes['layers'][0].copy()]
    strokes['layers'][0][..., 3] = (strokes['layers'][0][..., 3].astype(np.uint32) * 128 // 255).astype(np.uint8)
    mask = painted_mask(strokes, captured)
    apply_local_retouch(edits, captured, strokes, operation, brush_size=24, brush_strength=90)
    after = composite(edits.document)
    # Half-alpha strokes land halfway between original and the full-strength result.
    edits2 = edge_history()
    _, captured2 = capture_composite(edits2)
    full = blur_brush(captured2)
    apply_local_retouch(edits2, captured2, full, operation, brush_size=24, brush_strength=90)
    strong = composite(edits2.document)
    midpoint = ((before.astype(np.uint32) + strong.astype(np.uint32)) // 2).astype(np.uint8)
    assert np.abs(after.astype(np.int16) - midpoint).max() <= 1


@pytest.mark.parametrize('args', [
    {'brush_size': 1}, {'brush_size': 201}, {'brush_size': 40.5},
    {'brush_size': True}, {'brush_strength': 0}, {'brush_strength': 101},
    {'brush_strength': float('nan')}, {'brush_hardness': -1}, {'brush_hardness': 101},
])
def test_invalid_warp_settings_never_touch_history(args):
    edits = edge_history()
    before = composite(edits.document)
    _, captured = capture_composite(edits)
    with pytest.raises(ValueError, match='Brush'):
        apply_local_retouch(edits, captured, blur_brush(captured), 'Smudge', **args)
    assert edits.revision == captured.revision
    np.testing.assert_array_equal(composite(edits.document), before)


def test_warp_area_budget_refusal():
    doc = Document(1200, 900)
    rng = np.random.default_rng(1)
    doc.add_image(rng.integers(0, 256, (900, 1200, 3), dtype=np.uint8), 'Noise')
    edits = DocumentHistory(doc)
    before = composite(edits.document)
    _, captured = capture_composite(edits)
    width, height = captured.preview_size  # (w, h); numpy rows are height
    strokes = np.zeros((height, width, 4), np.uint8)
    yy, xx = np.mgrid[0:height, 0:width]
    strokes[((yy - height / 2) ** 2 + (xx - width / 2) ** 2) < (width * 0.45) ** 2] = [255, 0, 0, 255]
    with pytest.raises(ValueError, match='too large'):
        apply_local_retouch(edits, captured, {'layers': [strokes]}, 'Smudge', brush_size=60)
    assert edits.revision == captured.revision
    np.testing.assert_array_equal(composite(edits.document), before)


@pytest.mark.parametrize('operation', ['Blur', 'Smudge', 'Liquify'])
def test_warp_rejects_stale_capture(operation):
    edits = edge_history()
    _, captured = capture_composite(edits)
    edits.update_layer(edits.document.layers[0].id, name='Changed')
    with pytest.raises(ValueError, match='Document changed'):
        apply_local_retouch(edits, captured, blur_brush(captured), operation, brush_size=24)
    assert len(edits.document.layers) == 1


def test_warp_selection_limits_the_stroke():
    edits = edge_history()
    _, captured = capture_composite(edits)
    strokes = blur_brush(captured)
    selection = np.zeros((100, 200), np.uint8)
    selection[:, :100] = 255  # left half only
    before = composite(edits.document)
    apply_local_retouch(edits, captured, strokes, 'Blur', brush_size=24,
                        selection=selection)
    after = composite(edits.document)
    assert not np.array_equal(after[:, 80:100], before[:, 80:100])
    np.testing.assert_array_equal(after[:, 100:], before[:, 100:])
