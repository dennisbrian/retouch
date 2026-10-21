"""Gradient Map adjustment kind: LUT port, endpoints, round-trip, UI wiring."""
import numpy as np
import pytest
from pathlib import Path

import gradio as gr

from retouch.editor import (Document, DocumentHistory, Layer, UnsupportedFeature,
                            composite, load_project)
from retouch.editor.adjustments import KINDS, adjustment, grade, validate_adjustment
from retouch.editor.selections import selection
from retouch.editor.ui import EditorSession, build_editor_tab

BLUE_ORANGE = dict(shadows=(0.0, 0.2, 0.6), highlights=(1.0, 0.5, 0.1))


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


def _native_reference(rgb, gmap):
    """The Swift/C formulas transcribed line for line."""
    dark, light = ((gmap['highlights'], gmap['shadows']) if gmap['reversed']
                   else (gmap['shadows'], gmap['highlights']))
    dark = tuple(dark[c] for c in ('red', 'green', 'blue'))
    light = tuple(light[c] for c in ('red', 'green', 'blue'))

    def channel(f, t, to):
        value = f + (to - f) * t
        return int(min(255.0, max(0.0, round(value * 255))))

    table = []
    for index in range(256):
        t = index / 255
        table += [channel(dark[0], t, light[0]), channel(dark[1], t, light[1]),
                  channel(dark[2], t, light[2])]
    out = np.empty(rgb.shape, np.float32)
    flat = np.rint(np.clip(rgb, 0, 1) * 255).astype(np.int64).reshape(-1, 3)
    for i, (r, g, b) in enumerate(flat):
        level = min(255, (2126 * r + 7152 * g + 722 * b + 5000) // 10000)
        out.reshape(-1, 3)[i] = [table[level * 3], table[level * 3 + 1],
                                 table[level * 3 + 2]]
    return out / 255.0


@pytest.mark.parametrize('reversed_', [False, True])
@pytest.mark.parametrize('shadows, highlights', [
    ((0, 0, 0), (1, 1, 1)),
    ((0.0, 0.2, 0.6), (1.0, 0.5, 0.1)),
    ((0.13, 0.71, 0.29), (0.05, 0.05, 0.95)),
])
def test_gradient_map_matches_native_lut_formula(shadows, highlights, reversed_):
    rng = np.random.default_rng(11)
    rgb = rng.random((13, 17, 3)).astype(np.float32)
    record = adjustment('Gradient Map', shadows=shadows,
                        highlights=highlights, reversed=reversed_)
    result = grade(rgb, record)
    np.testing.assert_array_equal(
        result, _native_reference(rgb, record['gradientMapSettings']))


def test_gradient_map_endpoints_hit_the_two_colours_exactly():
    # Pure black and pure white land on LUT slots 0 and 255: the two ends.
    ends = np.array([[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]], np.float32)
    record = adjustment('Gradient Map', **BLUE_ORANGE)
    result = grade(ends, record)
    np.testing.assert_allclose(result[0, 0], BLUE_ORANGE['shadows'], atol=1 / 255)
    np.testing.assert_allclose(result[1, 0], BLUE_ORANGE['highlights'], atol=1 / 255)
    reversed_record = adjustment('Gradient Map', **BLUE_ORANGE, reversed=True)
    flipped = grade(ends, reversed_record)
    np.testing.assert_allclose(flipped[0, 0], BLUE_ORANGE['highlights'], atol=1 / 255)
    np.testing.assert_allclose(flipped[1, 0], BLUE_ORANGE['shadows'], atol=1 / 255)


def test_gradient_map_identity_default_keeps_gray_ramp():
    ramp = np.tile(np.arange(256, dtype=np.float32) / 255, (2, 1))
    gray = np.stack([ramp] * 3, axis=-1)
    np.testing.assert_array_equal(grade(gray, adjustment('Gradient Map')), gray)


def test_gradient_map_luminance_desaturates_colour_input():
    # A saturated red and the gray of its Rec.709 luminance land on the same
    # LUT slot, so both take the same colour.
    red = np.full((1, 1, 3), 0.5, np.float32)
    red[0, 0] = (1.0, 0.0, 0.0)
    gray = np.full((1, 1, 3), round((2126 * 255) / 10000) / 255, np.float32)
    out = grade(np.stack([red, gray]), adjustment('Gradient Map', **BLUE_ORANGE))
    np.testing.assert_array_equal(out[0], out[1])


# --- Records and validation ------------------------------------------------------

def test_gradient_map_kind_now_supported_but_others_still_refused():
    assert 'Gradient Map' in KINDS
    record = dict(kind='Gaussian Blur', hue=0, saturation=0, lightness=0, colorize=False,
                  range='master', invertRange=False,
                  levels=dict(channel='RGB', ranges=[dict(black=0, white=255, gamma=1,
                                                          outputBlack=0, outputWhite=255)] * 4),
                  curves=dict(channel='RGB',
                              channels=[[dict(x=0, y=0), dict(x=255, y=255)]] * 4))
    with pytest.raises(UnsupportedFeature, match='Supported adjustments'):
        validate_adjustment(record)


def test_gradient_map_settings_do_not_ride_other_kinds():
    record = adjustment('Gradient Map', **BLUE_ORANGE)
    record['grainSettings'] = dict(amount=1, size=1, roughness=0, seed=0)
    with pytest.raises(UnsupportedFeature, match='do not match its kind'):
        validate_adjustment(record)
    record = adjustment('Grain', amount=50, seed=3)
    record['gradientMapSettings'] = adjustment('Gradient Map')['gradientMapSettings']
    with pytest.raises(UnsupportedFeature, match='do not match its kind'):
        validate_adjustment(record)


@pytest.mark.parametrize('shadows, highlights, reversed_', [
    ((0, 0), (1, 1, 1), False),
    ((0, 0, 0), (1, 1, 1, 1), False),
    ((0, 0, 0), (1, 1, 1), 1),
    ((-0.1, 0, 0), (1, 1, 1), False),
    ((0, 0, 0), (1.1, 1, 1), False),
    ((0, 0, float('nan')), (1, 1, 1), False),
    ((None, 0, 0), (1, 1, 1), False),
])
def test_gradient_map_validation_rejects_out_of_range(shadows, highlights, reversed_):
    with pytest.raises(ValueError):
        adjustment('Gradient Map', shadows=shadows, highlights=highlights,
                   reversed=reversed_)


# --- Round trip ------------------------------------------------------------------

def _solid(color, alpha=255):
    a = np.empty((4, 4, 4), np.uint8)
    a[..., :3], a[..., 3] = color, alpha
    return a


def test_gradient_map_adjustment_layer_round_trips(tmp_path):
    base = Layer('Base', _solid((80, 100, 120), alpha=128))
    settings = adjustment('Gradient Map', **BLUE_ORANGE, reversed=True)
    adj = Layer('Gradient Map', size=(4, 4), adjustment=settings,
                mask=selection(4, 4, 'Rectangle', 0, 0, 2, 4), opacity=.5)
    h = DocumentHistory(Document(4, 4, [base]))
    before = composite(h.document)
    h.add_layer(adj)
    image = composite(h.document)
    np.testing.assert_array_equal(image[..., 3], before[..., 3])
    np.testing.assert_array_equal(image[:, 2:], before[:, 2:])
    assert not np.array_equal(image[:, :2], before[:, :2])
    h.save(tmp_path / 'gradient_map.comp')
    again = load_project(tmp_path / 'gradient_map.comp')
    assert again.layers[-1].adjustment == settings
    np.testing.assert_array_equal(composite(again), image)
    h.undo()
    np.testing.assert_array_equal(composite(h.document), before)


# --- UI wiring -------------------------------------------------------------------

def _gradient_args(shadow_hex, highlight_hex, reversed_=False):
    tables = [[[0, 0], [255, 255]]] * 4
    bw = (40, 60, 40, 60, 20, 80, False, 40, 20)
    cb = (0, 0, 0, 0, 0, 0, 0, 0, 0, True)
    tonal = (50, 16, 40, 60, 30)
    bloom = (40, 24)
    return ['Gradient Map', 0, 0, 1, 0, 255, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
            *tables, 35, 2.5, 65, 12345, 12.5, True, True,
            *tonal, *bloom, *bw, *cb, shadow_hex, highlight_hex, reversed_]


def test_gradient_map_ui_adds_updates_and_round_trips(session, callbacks, tmp_path):
    from retouch.editor.adjustment_controls import gradient_color
    edits = {fn.__kwdefaults__['operation']: fn for fn in callbacks
             if fn.__name__ == 'edit'}
    before = composite(session.history.document)
    view = edits['add_grade'](session, session.selected_id, None,
                              *_gradient_args('#003399', '#ff8020', True))
    settings = session.selected().adjustment
    assert settings['kind'] == 'Gradient Map'
    assert settings['gradientMapSettings'] == dict(
        shadows=gradient_color('#003399'), highlights=gradient_color('#ff8020'),
        reversed=True)
    rendered = composite(session.history.document)
    assert not np.array_equal(rendered, before)
    # The view hands the stored colours back to the pickers (indices shifted
    # past the Tonal Contrast and Bloom rows added before the B&W block).
    assert view[106] == gr.update(value='#003399')
    assert view[107] == gr.update(value='#ff8020')
    assert view[108] is True
    target = tmp_path / 'gradient.comp'
    session.history.save(target)
    reopened = load_project(target)
    assert reopened.layer(session.selected_id).adjustment == settings
    np.testing.assert_array_equal(composite(reopened), rendered)
    edits['update_grade'](session, session.selected_id, None,
                          *_gradient_args('#000000', '#ffffff', False))
    assert session.selected().adjustment['gradientMapSettings'] == dict(
        shadows=gradient_color('#000000'), highlights=gradient_color('#ffffff'),
        reversed=False)
    edits['undo'](session, session.selected_id, None)
    np.testing.assert_array_equal(composite(session.history.document), rendered)


def test_gradient_map_invalid_colour_preserves_history(session, callbacks):
    add = next(fn for fn in callbacks if fn.__name__ == 'edit'
               and fn.__kwdefaults__['operation'] == 'add_grade')
    revision = session.history.revision
    with pytest.raises(gr.Error):
        add(session, session.selected_id, None, *_gradient_args('nope', '#ffffff'))
    assert session.history.revision == revision


def test_gradient_map_visibility_callbacks_cover_the_new_kind():
    with gr.Blocks() as app:
        with gr.Tabs():
            build_editor_tab()
    kind = next(component for component in app.blocks.values()
                if isinstance(component, gr.Radio) and component.label == 'Adjustment')
    assert 'Gradient Map' in [choice[0] for choice in kind.choices]
    visibility = [event for event in app.fns.values()
                  if event.inputs == [kind] and event.fn]
    assert len(visibility) == 2
    for event in visibility:
        for value in KINDS:
            assert len(event.fn(value)) == len(event.outputs)
