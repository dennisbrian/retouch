"""Colour painting and erasing on layer pixels: compositing, mapping, history."""
import numpy as np
import pytest
from PIL import Image

from retouch.editor import Document, DocumentHistory, composite, load_project
from retouch.editor.adjustments import adjustment as make_adjustment
from retouch.editor.document import Layer
from retouch.editor.layer_paint import (
    compose_paint,
    paint_canvas_background,
    paint_layer,
    selection_on_layer,
    stroke_coverage,
)


def history(width=96, height=80):
    doc = Document(width, height)
    pixels = np.full((height, width, 3), [160, 120, 90], np.uint8)
    doc.add_image(pixels, 'Original')
    return DocumentHistory(doc, saved=True)


def strokes(width, height, box, alpha=255):
    layer = np.zeros((height, width, 4), np.uint8)
    top, bottom, left, right = box
    layer[top:bottom, left:right] = [255, 51, 102, alpha]
    return {'layers': [layer]}


def test_paint_matches_porter_duff_over():
    rng = np.random.default_rng(3)
    base = rng.integers(0, 256, (40, 50, 4), dtype=np.uint8)
    coverage = rng.integers(0, 256, (40, 50)).astype(np.float32) / 255
    color = (0.2, 0.6, 0.9)
    painted = compose_paint(base, coverage, color, 0.7)
    src = np.zeros_like(base)
    src[..., :3] = np.rint(np.array(color) * 255)
    src[..., 3] = np.rint(coverage * 0.7 * 255)
    expected = np.asarray(Image.alpha_composite(Image.fromarray(base), Image.fromarray(src)))
    visible = expected[..., 3] > 32
    # Pillow rounds the source alpha to 8 bits first and works in integers;
    # colour of nearly transparent pixels is ill-conditioned, so compare
    # colour where there is something to see.
    assert np.abs(painted[visible].astype(int) - expected[visible]).max() <= 2
    assert np.abs(painted[..., 3].astype(int) - expected[..., 3]).max() <= 1


def test_zero_coverage_is_bit_identical_and_erase_keeps_colour():
    base = np.full((10, 10, 4), [10, 200, 30, 255], np.uint8)
    coverage = np.zeros((10, 10), np.float32)
    coverage[2:5, 2:5] = 1
    erased = compose_paint(base, coverage, (0, 0, 0), 0.5, erase=True)
    np.testing.assert_array_equal(erased[coverage == 0], base[coverage == 0])
    assert (erased[2:5, 2:5, 3] == 128).all()
    np.testing.assert_array_equal(erased[2:5, 2:5, :3], base[2:5, 2:5, :3])
    gone = compose_paint(base, coverage, (0, 0, 0), 1.0, erase=True)
    assert (gone[2:5, 2:5] == 0).all()


def test_paint_is_one_undo_step_and_round_trips(tmp_path):
    edits = history()
    before = composite(edits.document)
    layer = edits.document.layers[0]
    assert paint_layer(edits, layer.id, strokes(96, 80, (10, 20, 30, 50)),
                       color=(1.0, 0.0, 0.0), opacity=100)
    after = composite(edits.document)
    assert (after[10:20, 30:50, :3] == [255, 0, 0]).all()
    untouched = np.ones((80, 96), bool)
    untouched[10:20, 30:50] = False
    np.testing.assert_array_equal(after[untouched], before[untouched])
    assert edits.undo_label == 'Paint on layer'
    target = tmp_path / 'paint.comp'
    edits.save(target)
    np.testing.assert_array_equal(composite(load_project(target)), after)
    assert edits.undo()
    np.testing.assert_array_equal(composite(edits.document), before)


def test_opacity_caps_overlapping_strokes():
    edits = history()
    layer = edits.document.layers[0]
    first = strokes(96, 80, (10, 30, 10, 30))['layers'][0]
    second = strokes(96, 80, (20, 40, 20, 40))['layers'][0]
    paint_layer(edits, layer.id, {'layers': [first, second]}, color=(0, 0, 0), opacity=50)
    pixels = edits.document.layers[0].pixels
    # Overlap region gets the same 50% as single coverage: never 75%.
    np.testing.assert_array_equal(pixels[25, 25], pixels[15, 15])
    np.testing.assert_array_equal(pixels[25, 25, :3], [80, 60, 45])


def test_new_blank_layer_paints_into_transparent_pixels():
    edits = history()
    doc = edits.document
    edits.add_layer(Layer('Paint', size=(doc.width, doc.height)))
    paint = edits.document.layers[-1]
    assert paint.pixels is None
    with pytest.raises(ValueError, match='nothing to erase'):
        paint_layer(edits, paint.id, strokes(96, 80, (0, 5, 0, 5)), mode='Erase')
    paint_layer(edits, paint.id, strokes(96, 80, (0, 10, 0, 10)), color=(0, 0, 1), opacity=100)
    pixels = edits.document.layers[-1].pixels
    assert pixels.shape == (80, 96, 4)
    assert (pixels[:10, :10] == [0, 0, 255, 255]).all()
    assert (pixels[20:, 20:] == 0).all()
    np.testing.assert_array_equal(edits.document.layers[0].pixels[..., :3],
                                  np.full((80, 96, 3), [160, 120, 90], np.uint8))


def test_preview_strokes_land_on_native_grid_and_flips():
    doc = Document(200, 100)
    doc.add_image(np.full((100, 200, 3), 128, np.uint8), 'Big')
    layer = doc.layers[0]
    # A half-size preview stroke on the left quarter maps to native pixels.
    preview = strokes(100, 50, (0, 50, 0, 25))
    coverage = stroke_coverage(preview, layer)
    assert coverage.shape == (100, 200)
    assert coverage[:, :45].min() > 0.99 and coverage[:, 55:].max() == 0
    layer.flip_x = True
    flipped = stroke_coverage(preview, layer)
    np.testing.assert_array_equal(flipped, coverage[:, ::-1])


def test_selection_maps_through_placement():
    doc = Document(100, 100)
    doc.add_image(np.full((40, 60, 3), 50, np.uint8), 'Placed', origin=(20, 30))
    layer = doc.layers[0]
    selection = np.zeros((100, 100), np.uint8)
    selection[30:50, 20:50] = 255
    on_layer = selection_on_layer(selection, layer)
    assert on_layer.shape == (40, 60)
    assert on_layer[:19, :29].min() > 0.99
    assert on_layer[22:, :].max() < 0.01 and on_layer[:, 32:].max() < 0.01


def test_selection_limits_paint_and_misses_are_refused():
    edits = history()
    layer = edits.document.layers[0]
    selection = np.zeros((80, 96), np.uint8)
    selection[:, :48] = 255
    paint_layer(edits, layer.id, strokes(96, 80, (10, 20, 30, 70)), color=(0, 1, 0),
                selection=selection)
    pixels = edits.document.layers[0].pixels
    assert (pixels[15, 30:47, :3] == [0, 255, 0]).all()
    assert (pixels[15, 50:70, :3] == [160, 120, 90]).all()
    empty = np.zeros_like(selection)
    with pytest.raises(ValueError, match='do not overlap'):
        paint_layer(edits, layer.id, strokes(96, 80, (10, 20, 30, 70)), selection=empty)


def test_softness_feathers_the_edge():
    edits = history()
    layer = edits.document.layers[0]
    paint_layer(edits, layer.id, strokes(96, 80, (20, 60, 20, 76)), color=(0, 0, 0), softness=12)
    row = edits.document.layers[0].pixels[40, :, 0].astype(int)
    assert row[48] < 20 and row[5] == 160
    assert 20 < row[20] < 150  # a gradient across the stroke edge


def test_refuses_folders_adjustments_and_bad_settings():
    edits = history()
    layer = edits.document.layers[0]
    with pytest.raises(ValueError, match='opacity'):
        paint_layer(edits, layer.id, strokes(96, 80, (0, 5, 0, 5)), opacity=0)
    with pytest.raises(ValueError, match='Paint or Erase'):
        paint_layer(edits, layer.id, strokes(96, 80, (0, 5, 0, 5)), mode='Smudge')
    with pytest.raises(ValueError, match='colour'):
        paint_layer(edits, layer.id, strokes(96, 80, (0, 5, 0, 5)), color=(2, 0, 0))
    assert not paint_layer(edits, layer.id, {'layers': []})
    adjustment = Layer('Grade', size=(96, 80), adjustment=make_adjustment('Invert'))
    edits.add_layer(adjustment)
    with pytest.raises(ValueError, match='no pixels'):
        paint_layer(edits, adjustment.id, strokes(96, 80, (0, 5, 0, 5)))


def test_canvas_background_shows_composite_under_layer():
    doc = Document(100, 60)
    base = np.zeros((60, 100, 3), np.uint8)
    base[:, 50:] = 255
    doc.add_image(base, 'Half')
    doc.add_image(np.full((20, 40, 3), 90, np.uint8), 'Patch', origin=(30, 20))
    patch = doc.layers[-1]
    view = paint_canvas_background(doc, patch)['background']
    assert view.shape == (20, 40, 4)
    # The patch itself is opaque grey, so the canvas shows it.
    assert (view[..., :3] == 90).all()
    patch.opacity = 0
    view = paint_canvas_background(doc, patch)['background']
    assert (view[:, :19, :3] == 0).all() and (view[:, 21:, :3] == 255).all()
    # The canvas shows the layer as displayed, so a flip doesn't move what's under it.
    patch.flip_x = True
    flipped = paint_canvas_background(doc, patch)['background']
    np.testing.assert_array_equal(flipped, view)


def test_ui_new_paint_layer_and_apply_through_callbacks():
    import gradio as gr
    from retouch.editor.ui import EditorSession, build_editor_tab
    with gr.Blocks() as app:
        with gr.Tabs():
            build_editor_tab()
    callbacks = {value.fn.__name__: value.fn for value in app.fns.values() if value.fn}
    doc = Document(40, 30)
    doc.add_image(np.full((30, 40, 3), 100, np.uint8), 'Base')
    session = EditorSession.create(doc)
    try:
        result = callbacks['adding_paint_layer'](session, None, None, False)
        brush, target = result[-4], result[-3]
        assert session.history.document.layer(target).name == 'Paint'
        height, width = brush['background'].shape[:2]
        stroke = np.zeros((height, width, 4), np.uint8)
        stroke[:height // 2, :width // 2] = [0, 0, 0, 255]
        callbacks['applying_paint'](session, target, {'layers': [stroke]}, None, 'Paint',
                                    '#00ff00', 100, 0, False)
        out = composite(session.history.document)
        assert (out[:14, :19, :3] == [0, 255, 0]).all()
        assert (out[20:, 25:, :3] == 100).all()
        assert session.history.undo_label == 'Paint on layer'
    finally:
        session.close()
