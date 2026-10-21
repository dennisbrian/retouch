"""Dirty-band caching in the editor compositor (retouch/editor/composite.py).

After an edit that only swaps pixel/mask buffers or changes opacity,
visibility or blend mode of a raster layer, ``composite`` re-renders only the
256-row bands the edit reaches and copies the rest from the previous result.
Every check compares the patched result byte for byte with a fresh full
render of the same document.
"""
import sys

import numpy as np
import pytest

from retouch.editor import Document, Layer, composite
from retouch.editor.adjustments import adjustment
from retouch.editor.commands import DocumentHistory
from retouch.editor.document import LayerEffects, ShadowEffect

C = sys.modules['retouch.editor.composite']

W, H = 96, 1100  # five 256-row bands


@pytest.fixture(autouse=True)
def _clean_memo():
    C._COMPOSITE_MEMO.clear()
    yield
    C._COMPOSITE_MEMO.clear()


@pytest.fixture
def bands(monkeypatch):
    """Tops of the bands the next composite draws (margin windows excluded)."""
    drawn = []
    original = C._BandRenderer.render

    def render(self, top, bottom, up_to=None):
        if up_to is None and bottom - top <= C._BAND:
            drawn.append(top)
        return original(self, top, bottom, up_to)

    monkeypatch.setattr(C._BandRenderer, 'render', render)
    return drawn


def _fresh(document):
    """A full render that leaves the memo as the patch path will find it."""
    saved = dict(C._COMPOSITE_MEMO)
    C._COMPOSITE_MEMO.clear()
    try:
        return np.array(composite(document))
    finally:
        C._COMPOSITE_MEMO.clear()
        C._COMPOSITE_MEMO.update(saved)


def _photo(seed, h=H, w=W):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    pixels[..., 3] = rng.integers(120, 256, (h, w), dtype=np.uint8)
    return pixels


def _document(*extra):
    doc = Document(width=W, height=H)
    doc.add_image(_photo(1), 'photo')
    top = Layer('top', _photo(2), opacity=0.7, blend_mode='Multiply')
    top.set_mask(np.full((H, W), 128, np.uint8))
    doc.add_layer(top)
    for layer in extra:
        doc.add_layer(layer)
    return DocumentHistory(doc), top.id


def _stroke(history, layer_id, y, rows=20, value=255):
    mask = np.array(history.document.layer(layer_id).mask)
    mask[y:y + rows, 10:50] = value
    history.set_mask(layer_id, mask)


def test_mask_stroke_redraws_only_touched_band(bands):
    history, top = _document()
    composite(history.document)
    bands.clear()
    _stroke(history, top, 300)
    patched = np.array(composite(history.document))
    assert bands == [256]
    assert np.array_equal(patched, _fresh(history.document))


def test_stroke_across_a_band_edge_redraws_both(bands):
    history, top = _document()
    composite(history.document)
    bands.clear()
    _stroke(history, top, 500, rows=30)
    patched = np.array(composite(history.document))
    assert bands == [256, 512]
    assert np.array_equal(patched, _fresh(history.document))


def test_pixel_replacement_on_offset_layer():
    history, top = _document()
    small = Layer('patch', _photo(3, 120, 40), origin=(20, 610))
    history.add_layer(small)
    composite(history.document)
    pixels = np.array(small.pixels)
    pixels[50:60] = (255, 0, 0, 255)
    history.replace_pixels(small.id, pixels)
    assert np.array_equal(np.array(composite(history.document)), _fresh(history.document))


def test_unchanged_buffer_swap_draws_nothing(bands):
    history, top = _document()
    before = np.array(composite(history.document))
    bands.clear()
    history.set_mask(top, np.array(history.document.layer(top).mask))
    assert np.array_equal(np.array(composite(history.document)), before)
    assert bands == []


@pytest.mark.parametrize('changes', [
    {'opacity': 0.3}, {'visible': False}, {'blend_mode': 'Screen'},
    {'mask_enabled': False},
])
def test_property_edit_on_small_layer_redraws_its_extent(bands, changes):
    history, top = _document()
    small = Layer('patch', _photo(3, 100, 40), origin=(20, 700))
    history.add_layer(small)
    composite(history.document)
    small_id = history.document.layers[-1].id
    history.set_mask(small_id, np.full((100, 40), 200, np.uint8))
    composite(history.document)
    bands.clear()
    history.update_layer(small_id, **changes)
    patched = np.array(composite(history.document))
    assert bands == [512, 768]
    assert np.array_equal(patched, _fresh(history.document))


def test_rotated_and_fractional_layers_use_their_extent():
    history, top = _document()
    tilted = Layer('tilted', _photo(4, 80, 60), origin=(10.5, 400.25), rotation=17)
    scaled = Layer('scaled', _photo(5, 40, 30), origin=(5, 850), size=(60, 90))
    history.add_layer(tilted)
    history.add_layer(scaled)
    composite(history.document)
    for layer_id in (history.document.layers[-2].id, history.document.layers[-1].id):
        layer = history.document.layer(layer_id)
        pixels = np.array(layer.pixels)
        pixels[5:15] = (0, 255, 0, 255)
        history.replace_pixels(layer_id, pixels)
        assert np.array_equal(np.array(composite(history.document)),
                              _fresh(history.document))


def test_blur_adjustment_above_widens_the_dirty_rows():
    bloom = Layer('Bloom', size=(W, H), adjustment=adjustment(
        'Bloom', amount=100, bloom_radius=20, bloom_threshold=0))
    tonal = Layer('Tonal', size=(W, H), adjustment=adjustment(
        'Tonal Contrast', amount=70, radius=12))
    history, top = _document(bloom, tonal)
    composite(history.document)
    # Right next to a band edge, so only the margins reach the next band.
    _stroke(history, top, 250, rows=4)
    assert np.array_equal(np.array(composite(history.document)), _fresh(history.document))


def test_adjustment_mask_stroke_is_patched(bands):
    exposure = Layer('Exposure', size=(W, H), adjustment=adjustment('Exposure', exposure=1))
    history, top = _document(exposure)
    adj_id = history.document.layers[-1].id
    history.set_mask(adj_id, np.zeros((H, W), np.uint8))
    composite(history.document)
    bands.clear()
    _stroke(history, adj_id, 900)
    patched = np.array(composite(history.document))
    assert bands == [768]
    assert np.array_equal(patched, _fresh(history.document))


def test_layer_effects_spread_is_included():
    history, top = _document()
    history.update_layer(top, effects=LayerEffects(shadow=ShadowEffect(
        angle=90, distance=30, blur=6, opacity=1.0)))
    composite(history.document)
    _stroke(history, top, 236, rows=10, value=0)
    assert np.array_equal(np.array(composite(history.document)), _fresh(history.document))


def test_structure_change_renders_in_full(bands):
    history, top = _document()
    composite(history.document)
    bands.clear()
    history.add_layer(Layer('new', _photo(6, 10, 10), origin=(0, 0)))
    patched = np.array(composite(history.document))
    assert bands == [0, 256, 512, 768, 1024]
    assert np.array_equal(patched, _fresh(history.document))


def test_whole_plane_adjustment_renders_in_full(bands):
    scan = Layer('Scanlines', size=(W, H), adjustment=adjustment('Scanlines'))
    history, top = _document(scan)
    composite(history.document)
    bands.clear()
    _stroke(history, top, 300)
    patched = np.array(composite(history.document))
    # The plane under it is rendered too, so each band shows up twice.
    assert set(bands) == {0, 256, 512, 768, 1024}
    assert np.array_equal(patched, _fresh(history.document))


def test_hiding_a_clip_base_renders_in_full(bands):
    history, top = _document()
    base = Layer('base', _photo(7, 60, 60), origin=(0, 300))
    history.add_layer(base)
    base_id = history.document.layers[-1].id
    history.add_layer(Layer('clipped', _photo(8, H, W), mask_source_id=base_id))
    composite(history.document)
    bands.clear()
    history.update_layer(base_id, visible=False)
    patched = np.array(composite(history.document))
    assert bands == [0, 256, 512, 768, 1024]
    assert np.array_equal(patched, _fresh(history.document))


def test_undo_and_redo_stay_exact():
    history, top = _document()
    composite(history.document)
    _stroke(history, top, 300)
    composite(history.document)
    _stroke(history, top, 900)
    composite(history.document)
    for step in (history.undo, history.undo, history.redo, history.redo):
        step()
        assert np.array_equal(np.array(composite(history.document)),
                              _fresh(history.document))


def test_freed_old_buffer_falls_back_to_full_render(bands):
    doc = Document(width=W, height=H)
    doc.add_image(_photo(1), 'photo')
    layer = doc.layers[0]
    layer.set_mask(np.full((H, W), 255, np.uint8))
    composite(doc)
    mask = np.array(layer.mask)
    mask[300:320] = 0
    layer.set_mask(mask)  # no history keeps the old mask alive
    bands.clear()
    patched = np.array(composite(doc))
    assert bands == [0, 256, 512, 768, 1024]
    assert np.array_equal(patched, _fresh(doc))


def test_preview_and_native_slots_coexist():
    history, top = _document()
    native = composite(history.document)
    small = Document(width=W // 2, height=H // 2,
                     document_id=history.document.document_id)
    small.add_image(_photo(9, H // 2, W // 2), 'preview')
    composite(small)
    assert composite(history.document) is native


def test_random_edit_sequence_matches_full_renders():
    rng = np.random.default_rng(11)
    tilted = Layer('tilted', _photo(4, 80, 60), origin=(10.5, 400.25), rotation=17)
    history, top = _document(tilted)
    ids = [layer.id for layer in history.document.layers]
    for _ in range(25):
        layer_id = ids[rng.integers(len(ids))]
        layer = history.document.layer(layer_id)
        action = rng.integers(4)
        if action == 0:
            mask = (np.array(layer.mask) if layer.mask is not None
                    else np.full(layer.pixels.shape[:2], 255, np.uint8))
            y = rng.integers(0, mask.shape[0])
            mask[y:y + 15] = rng.integers(0, 256)
            history.set_mask(layer_id, mask)
        elif action == 1:
            pixels = np.array(layer.pixels)
            y = rng.integers(0, pixels.shape[0])
            pixels[y:y + 9] = rng.integers(0, 256, 4)
            history.replace_pixels(layer_id, pixels)
        elif action == 2:
            history.update_layer(layer_id, opacity=float(rng.uniform(0.1, 1)))
        else:
            history.undo()
        assert np.array_equal(np.array(composite(history.document)),
                              _fresh(history.document))
