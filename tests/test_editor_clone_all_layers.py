"""Clone Stamp, Sample: All Layers (upstream ``CloneSettings.sampleAllLayers``)."""
import cv2
import gradio as gr
import numpy as np
import pytest

from retouch.editor import Document, Layer, composite
from retouch.editor.healing import clone_stamp, sample_all_layers
from retouch.editor.ui import EditorSession, build_editor_tab


def photo():
    """A smooth ramp with a red block at x 20-30, so copies are easy to trace."""
    yy, xx = np.mgrid[0:80, 0:120]
    image = np.zeros((80, 120, 4), np.uint8)
    image[..., 0], image[..., 1], image[..., 2], image[..., 3] = xx * 2, yy * 3, 90, 255
    image[30:50, 20:30] = (230, 20, 20, 255)
    return image


def brushed(center, radius, shape=(80, 120)):
    stroke = np.zeros((*shape, 4), np.uint8)
    cv2.circle(stroke, center, radius, (255, 51, 102, 255), -1)
    return {'layers': [stroke]}


@pytest.fixture
def session():
    doc = Document(120, 80)
    doc.add_image(photo(), 'Photo')
    result = EditorSession.create(doc)
    result.history.add_layer(Layer('Retouch', size=(120, 80)))
    result.selected_id = result.history.document.layers[-1].id
    yield result
    result.close()


def test_clone_onto_an_empty_layer_copies_the_canvas_and_leaves_the_photo(session):
    before = composite(session.history.document)
    photo_layer = session.history.document.layers[0]
    original = photo_layer.pixels.copy()
    assert session.paint_pixels(brushed((70, 40), 4), 'clone', offset=(-45, 0), sample='All layers')
    assert session.history.undo_label == 'Clone Stamp'
    layer = session.selected()
    # Painted pixels hold the canvas 45 px to the left; the rest stays transparent.
    np.testing.assert_array_equal(layer.pixels[40, 70], before[40, 25])
    assert layer.pixels[40, 90, 3] == 0 and layer.pixels[10, 10, 3] == 0
    np.testing.assert_array_equal(session.history.document.layers[0].pixels, original)
    after = composite(session.history.document)
    np.testing.assert_array_equal(after[40, 70], before[40, 25])
    np.testing.assert_array_equal(after[:, :60], before[:, :60])
    session.history.undo()
    np.testing.assert_array_equal(composite(session.history.document), before)


def test_zero_offset_stamps_the_canvas_as_shown(session):
    before = composite(session.history.document)
    assert session.paint_pixels(brushed((60, 40), 10), 'clone', offset=(0, 0), sample='All layers')
    layer = session.selected()
    covered = layer.pixels[..., 3] > 0
    assert covered.sum() > 250
    np.testing.assert_array_equal(layer.pixels[covered], before[covered])


def test_all_layers_copies_what_shows_current_layer_copies_its_own(session):
    # A green patch layer above the photo, over the clone source.
    patch = np.zeros((80, 120, 4), np.uint8)
    patch[35:45, 15:35] = (0, 200, 0, 255)
    session.history.add_layer(Layer('Patch', pixels=patch), 1)
    session.selected_id = session.history.document.layers[0].id  # the photo
    session.paint_pixels(brushed((70, 40), 3), 'clone', offset=(-45, 0), sample='All layers')
    np.testing.assert_array_equal(session.selected().pixels[40, 70], (0, 200, 0, 255))
    session.history.undo()
    session.paint_pixels(brushed((70, 40), 3), 'clone', offset=(-45, 0), sample='Current layer')
    np.testing.assert_array_equal(session.selected().pixels[40, 70], (230, 20, 20, 255))


def test_offset_is_in_canvas_pixels_for_a_moved_layer():
    canvas = composite(_doc_with(Layer('Photo', pixels=photo())))
    layer = Layer('Retouch', size=(60, 40), origin=(30, 20))
    coverage = np.zeros((40, 60), np.uint8)
    coverage[18:23, 10:15] = 255
    source = sample_all_layers(canvas, layer, coverage, (-12, 5))
    # Layer pixel (x, y) shows at canvas (x + 30, y + 20).
    np.testing.assert_array_equal(source[20, 12], canvas[20 + 20 + 5, 12 + 30 - 12])
    assert not source[0, 0].any()


def test_scaled_and_flipped_layer_samples_where_it_shows():
    canvas = composite(_doc_with(Layer('Photo', pixels=photo())))
    # 60 x 40 pixels drawn over the whole 120 x 80 canvas, mirrored left-right.
    layer = Layer('Half', pixels=np.zeros((40, 60, 4), np.uint8), size=(120, 80), flip_x=True)
    coverage = np.zeros((40, 60), np.uint8)
    coverage[10:30, 40:50] = 255
    source = sample_all_layers(canvas, layer, coverage, (0, 0))
    # Grid pixel (x, y) has its centre at canvas x = 120 - 2 (x + 0.5), y = 2 (y + 0.5).
    x, y = 45, 20
    cx, cy = 120 - 2 * (x + 0.5) - 0.5, 2 * (y + 0.5) - 0.5
    x0, y0 = int(np.floor(cx)), int(np.floor(cy))
    fx, fy = cx - x0, cy - y0
    c = canvas.astype(float)
    expect = ((1 - fy) * ((1 - fx) * c[y0, x0] + fx * c[y0, x0 + 1])
              + fy * ((1 - fx) * c[y0 + 1, x0] + fx * c[y0 + 1, x0 + 1]))
    assert np.abs(source[y, x].astype(float) - expect).max() <= 1


def test_source_beyond_the_canvas_draws_nothing():
    canvas = composite(_doc_with(Layer('Photo', pixels=photo())))
    layer = Layer('Retouch', size=(120, 80))
    coverage = np.zeros((80, 120), np.uint8)
    coverage[38:43, 5:10] = 255
    source = sample_all_layers(canvas, layer, coverage, (-40, 0))
    assert not source.any()
    empty = np.zeros((80, 120, 4), np.uint8)
    np.testing.assert_array_equal(clone_stamp(empty, coverage, (-40, 0), source=source), empty)


def test_current_layer_still_needs_pixels_and_an_offset(session):
    with pytest.raises(ValueError, match='All layers'):
        session.paint_pixels(brushed((60, 40), 5), 'clone', offset=(-10, 0))
    with pytest.raises(ValueError, match='current layer or all layers'):
        session.paint_pixels(brushed((60, 40), 5), 'clone', offset=(-10, 0), sample='Merged')
    with pytest.raises(ValueError, match='All layers'):
        session.paint_pixels(brushed((60, 40), 5), 'heal')


def test_clone_button_passes_the_sample_choice(session):
    with gr.Blocks() as app, gr.Tabs():
        build_editor_tab()
    edits = {value.fn.__kwdefaults__['operation']: value.fn for value in app.fns.values()
             if value.fn and value.fn.__name__ == 'edit'}
    before = composite(session.history.document)
    edits['clone'](session, session.selected_id, brushed((70, 40), 3), 100, -45, 0, 0, 'All layers')
    assert session.history.undo_label == 'Clone Stamp'
    np.testing.assert_array_equal(session.selected().pixels[40, 70], before[40, 25])


def _doc_with(layer):
    doc = Document(120, 80)
    doc.layers.append(layer)
    return doc
