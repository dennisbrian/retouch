"""Shape layers: pixel rendering, persistence, undo, UI wiring.

Pixel cases port Compositor's ``ShapeToolTests`` at 11d8d7a (rect fill,
ellipse corners clear, pill clamp) against this port's supersampled ImageDraw
render; MIT, Copyright (c) 2026 Wonder Assembly LLC for the translated cases.
"""
import json
from pathlib import Path

import numpy as np
import pytest
import gradio as gr
from PIL import Image, ImageDraw

from retouch.editor import (
    Document, Layer, DocumentHistory, composite, load_project, save_project,
    shape_image, shape_layer, shape_style, validate_shape_style,
)
from tests import compositor_fixtures as fx


def _alpha(pixels, x, y):
    return int(pixels[y, x, 3])


# ------------------------------------------------------------------ style

class TestStyle:
    def test_rejects_unknown_kind_and_bad_channels(self):
        with pytest.raises(ValueError):
            shape_style('Triangle', 1, 0, 0)
        with pytest.raises(ValueError):
            shape_style('Rectangle', 2, 0, 0)
        with pytest.raises(ValueError):
            shape_style('Rectangle', 1, 0, 0, corner_radius=-1)

    def test_line_needs_width_and_ends(self):
        with pytest.raises(ValueError):
            shape_style('Line', 1, 0, 0)
        style = shape_style('Line', 1, 0, 0, line_width=4,
                            start=[0, 0], end=[1, 1])
        assert style['cornerRadius'] == 0.0 and style['lineWidth'] == 4.0

    def test_document_validation_rejects_bad_shape_record(self):
        doc = Document(10, 10)
        doc.add_image(np.zeros((10, 10, 3), np.uint8), 'Base')
        doc.layers.append(Layer('R', pixels=np.zeros((5, 5, 4), np.uint8),
                                origin=(0, 0), shape={'kind': 'Blob'}))
        with pytest.raises(ValueError):
            DocumentHistory(doc)


# ------------------------------------------------------------------ pixels

class TestPixels:
    def test_rectangle_fills_its_box(self):
        # ShapeToolTests.rectangleFillsANewLayerWithTheForegroundColor
        pixels = shape_image(shape_style('Rectangle', 1, 0, 0), 30, 20)
        assert pixels[10, 15].tolist() == [255, 0, 0, 255]
        assert _alpha(pixels, 0, 0) == 255 and _alpha(pixels, 29, 19) == 255
        # outside the layer box there is nothing by construction (h, w = 20, 30)
        assert pixels.shape == (20, 30, 4)

    def test_ellipse_leaves_corners_clear(self):
        # ShapeToolTests.ellipseLeavesItsCornersClear
        pixels = shape_image(shape_style('Ellipse', 1, 0, 0), 20, 20)
        assert _alpha(pixels, 10, 10) == 255
        assert _alpha(pixels, 10, 1) > 0 and _alpha(pixels, 1, 10) > 0
        assert _alpha(pixels, 0, 0) == 0 and _alpha(pixels, 19, 19) == 0

    def test_corner_radius_clamps_to_a_pill(self):
        # ShapeToolTests.roundedRectanglesFollowTheRadiusAndClampToAPill
        pixels = shape_image(shape_style('Rectangle', 1, 0, 0, corner_radius=8), 40, 30)
        assert _alpha(pixels, 0, 0) == 0      # the corner is cut away
        assert _alpha(pixels, 1, 1) == 0
        assert _alpha(pixels, 3, 3) == 255    # inside the rounded corner
        assert _alpha(pixels, 20, 0) == 255   # straight edges stay full
        pill = shape_image(shape_style('Rectangle', 1, 0, 0, corner_radius=500), 40, 20)
        assert _alpha(pill, 0, 0) == 0 and _alpha(pill, 20, 10) == 255

    def test_line_runs_between_its_ends_with_round_caps(self):
        pixels = shape_image(shape_style('Line', 0, 0, 1, line_width=4,
                                         start=[0.1, 0.1], end=[0.9, 0.9]), 40, 40)
        assert _alpha(pixels, 20, 20) == 255
        assert _alpha(pixels, 0, 39) == 0 and _alpha(pixels, 39, 0) == 0
        # round cap reaches just past the end point, not halfway across
        assert _alpha(pixels, 3, 3) > 0 and _alpha(pixels, 1, 1) == 0

    def test_supersampled_render_matches_area_coverage_oracle(self):
        # Independent oracle: an ellipse's pixel coverage equals the fraction
        # of 16x16 subsamples inside it (area sampling, like CGContext AA).
        w = h = 33
        style = shape_style('Ellipse', 1, 1, 1)
        pixels = shape_image(style, w, h)
        sub = 16
        ys, xs = np.mgrid[0:h * sub, 0:w * sub]
        cx = cy = (w - 1) / 2 * sub
        rx = ry = (w - 1) / 2 * sub
        inside = ((xs - cx) / rx) ** 2 + ((ys - cy) / ry) ** 2 <= 1.0
        oracle = inside.reshape(h, sub, w, sub).mean(axis=(1, 3))
        expected = np.round(oracle * 255)
        got = pixels[..., 3].astype(int)
        delta = np.abs(got - expected)
        # Interior agrees exactly.
        assert delta[expected == 255].max() == 0
        # PIL's integer ellipse rasterization can bulge a fraction of a
        # supersample past the true curve, so stray coverage is excused only
        # within 2 output pixels of the true edge.
        def grow(mask, iterations):
            mask = mask.copy()
            for _ in range(iterations):
                wider = mask.copy()
                wider[1:, :] |= mask[:-1, :]
                wider[:-1, :] |= mask[1:, :]
                wider[:, 1:] |= mask[:, :-1]
                wider[:, :-1] |= mask[:, 1:]
                mask = wider
            return mask
        near_edge = grow(oracle > 0, 2)
        stray = (expected == 0) & (delta > 0) & ~near_edge
        assert stray.sum() == 0
        # No systematic size bias: total coverage within 8% (PIL's integer
        # ellipse rasterization runs slightly fuller than the analytic area).
        assert abs(got.sum() - expected.sum()) <= 0.08 * expected.sum()

    def test_supersample_factor_controls_edge_smoothness(self):
        style = shape_style('Ellipse', 1, 1, 1)
        def edge_levels(factor):
            return len(np.unique(shape_image(style, 24, 24, supersample=factor)[..., 3]))
        assert edge_levels(4) >= 4      # several AA levels, not hard edges
        assert edge_levels(8) >= edge_levels(4)

    def test_unfilled_rectangle_draws_an_outline(self):
        pixels = shape_image(shape_style('Rectangle', 1, 0, 0), 21, 21, filled=False)
        assert _alpha(pixels, 10, 10) == 0
        # a 1-device-pixel outline: solid on the edge rows/cols
        assert _alpha(pixels, 10, 0) >= 255 * 0.9
        assert _alpha(pixels, 10, 20) >= 255 * 0.9
        assert _alpha(pixels, 0, 10) >= 255 * 0.9 and _alpha(pixels, 20, 10) >= 255 * 0.9


# ------------------------------------------------------------- persistence

class TestPersistence:
    def test_shape_record_round_trips_through_comp(self, tmp_path):
        style = shape_style('Line', 0.5, 0.25, 0.125, line_width=4.0,
                            start=[0.1, 0.2], end=[0.8, 0.9])
        doc = Document(50, 40)
        doc.add_image(np.zeros((40, 50, 3), np.uint8), 'Base')
        doc.layers.append(shape_layer('Diagonal', style, origin=(2, 3), size=(40, 30)))
        path = save_project(doc, tmp_path / 'shape.comp')
        reopened = load_project(path)
        assert reopened.layers[1].shape == style
        np.testing.assert_array_equal(reopened.layers[1].pixels,
                                      doc.layers[1].pixels)

    def test_shape_layer_rejects_manifest_without_image(self, tmp_path):
        # Upstream: shape rides an ordinary image asset; no image, no shape.
        record = fx.layer_record(0, 'Ghost', 4, 4)
        record['shape'] = {'kind': 'Rectangle', 'red': 1.0, 'green': 0.0,
                           'blue': 0.0, 'cornerRadius': 0}
        path = fx.write_package(tmp_path / 'ghost.comp', 4, 4, [record], images={})
        with pytest.raises(Exception):
            load_project(path)

    def test_shape_with_image_round_trips_from_manifest(self, tmp_path):
        record = fx.layer_record(0, 'Rect', 6, 4)
        record['shape'] = {'kind': 'Ellipse', 'red': 0.0, 'green': 1.0,
                           'blue': 0.0, 'cornerRadius': 0}
        pixels = fx.solid(4, 6, (0, 255, 0, 255))
        path = fx.write_package(tmp_path / 'e.comp', 6, 4, [record],
                                images={0: pixels})
        doc = load_project(path)
        assert doc.layers[0].shape['kind'] == 'Ellipse'
        again = save_project(doc, tmp_path / 'again.comp')
        assert load_project(again).layers[0].shape == doc.layers[0].shape


# --------------------------------------------------------------- undo/redo

class TestHistory:
    def _document(self):
        doc = Document(80, 60)
        doc.add_image(np.zeros((60, 80, 3), np.uint8), 'Base')
        return doc

    def test_add_shape_layer_renders_over_base_and_undoes(self):
        history = DocumentHistory(self._document())
        style = shape_style('Rectangle', 1, 0, 0)
        history.add_layer(shape_layer('Rectangle 1', style, origin=(10, 10),
                                      size=(30, 20)))
        doc = history.document
        rendered = composite(doc)
        assert tuple(rendered[20, 25][:3]) == (255, 0, 0)
        assert history.undo_label == 'Add layer'
        history.undo()
        assert len(history.document.layers) == 1

    def test_update_shape_changes_pixels_and_style_as_one_undo_step(self):
        history = DocumentHistory(self._document())
        history.add_layer(shape_layer('Ellipse 1', shape_style('Ellipse', 1, 0, 0),
                                      origin=(5, 5), size=(40, 40)))
        red = history.document.layers[1]
        new_style = shape_style('Ellipse', 0, 0, 1)
        assert history.update_shape(red.id, new_style)
        layer = history.document.layers[1]
        assert layer.shape == new_style
        assert layer.pixels[20, 20].tolist() == [0, 0, 255, 255]
        assert history.undo_label == 'Change shape'
        history.undo()
        reverted = history.document.layers[1]
        assert reverted.shape['blue'] == 0.0
        assert reverted.pixels[20, 20].tolist() == [255, 0, 0, 255]

    def test_update_shape_refuses_non_shape_layers(self):
        history = DocumentHistory(self._document())
        base = history.document.layers[0]
        with pytest.raises(ValueError, match='shape layer'):
            history.update_shape(base.id, shape_style('Rectangle', 1, 0, 0))

    def test_painted_pixels_stop_the_shape_redraw(self):
        # liveShape: once pixels differ from the shape's own raster (e.g. a
        # resize), the shape style no longer redraws over them.
        history = DocumentHistory(self._document())
        history.add_layer(shape_layer('Rectangle 1', shape_style('Rectangle', 1, 0, 0),
                                      origin=(0, 0), size=(10, 10)))
        lid = history.document.layers[1].id
        history.update_layer(lid, size=(20, 10))
        with pytest.raises(ValueError, match='shape layer'):
            history.update_shape(lid, shape_style('Rectangle', 0, 1, 0))


# --------------------------------------------------------------- UI wiring

@pytest.fixture
def session():
    doc = Document(80, 60)
    doc.add_image(np.zeros((60, 80, 3), np.uint8), 'Base')
    from retouch.editor.ui import EditorSession
    result = EditorSession.create(doc)
    yield result
    result.close()

@pytest.fixture
def callbacks():
    from retouch.editor.ui import build_editor_tab
    with gr.Blocks() as app:
        with gr.Tabs():
            build_editor_tab()
    return [value.fn for value in app.fns.values() if value.fn]

def _edits(callbacks):
    return {fn.__kwdefaults__['operation']: fn
            for fn in callbacks if fn.__name__ == 'edit'}


def test_add_shape_button_creates_layer_and_renders(session, callbacks):
    edits = _edits(callbacks)
    result = edits['add_shape'](session, session.selected_id, None,
                                'Rectangle', 10, 10, 30, 20,
                                0, 0, 30, 20, True, '#ff0000', 4, 0)
    layer = session.selected()
    assert layer.name == 'Rectangle 1' and layer.shape['kind'] == 'Rectangle'
    assert layer.origin == (10.0, 10.0) and layer.size == (30, 20)
    assert composite(session.history.document)[20, 25].tolist()[:3] == [255, 0, 0]

def test_add_line_shape_uses_drag_ends(session, callbacks):
    edits = _edits(callbacks)
    edits['add_shape'](session, session.selected_id, None,
                       'Line', 0, 0, 40, 30, 4, 3, 36, 27, True, '#0000ff', 5, 0)
    layer = session.selected()
    assert layer.shape['lineWidth'] == 5
    assert layer.shape['start'] == [0.1, 0.1] and layer.shape['end'] == [0.9, 0.9]

def test_update_shape_button_redraws_selected_shape_and_undoes(session, callbacks):
    edits = _edits(callbacks)
    edits['add_shape'](session, session.selected_id, None,
                       'Ellipse', 5, 5, 40, 40, 0, 0, 40, 40, True, '#ff0000', 4, 0)
    lid = session.selected_id
    edits['update_shape'](session, lid, None,
                          'Ellipse', 5, 5, 40, 40, 0, 0, 40, 40, True, '#00ff00', 4, 0)
    layer = session.selected()
    assert layer.shape['green'] == 1.0 and layer.pixels[20, 25, 1] == 255
    assert session.history.undo_label == 'Change shape'
    session.history.undo()
    assert session.history.document.layer(lid).pixels[20, 25].tolist() == [255, 0, 0, 255]

def test_update_shape_button_refuses_plain_layer(session, callbacks):
    import gradio as gr
    edits = _edits(callbacks)
    with pytest.raises(gr.Error, match='shape layer'):
        edits['update_shape'](session, session.selected_id, None,
                              'Rectangle', 0, 0, 10, 10, 0, 0, 10, 10,
                              True, '#ff0000', 4, 0)

def test_shape_layer_survives_save_reopen_from_ui(session, callbacks, tmp_path):
    edits = _edits(callbacks)
    edits['add_shape'](session, session.selected_id, None,
                       'Rectangle', 8, 6, 24, 12, 0, 0, 24, 12, True, '#123456', 4, 3)
    layer = session.selected()
    doc = session.history.document
    path = save_project(doc, tmp_path / 'ui-shape.comp')
    reopened = load_project(path)
    saved = reopened.layer(layer.id)
    assert saved.shape == layer.shape
    np.testing.assert_array_equal(saved.pixels, layer.pixels)
    rendered = composite(reopened)
    assert tuple(rendered[12, 20][:3]) == (0x12, 0x34, 0x56)
