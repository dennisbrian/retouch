"""Texture healing respects donor masks, soft coverage and bounded native work."""
from copy import deepcopy

import cv2
import numpy as np
import pytest

from retouch.editor import Document, DocumentHistory, composite, load_project
from retouch.editor import local_retouch as local


def setup(size=80):
    yy, xx = np.indices((size, size))
    texture = (130 + 35 * np.sin(xx * 1.1) + 20 * np.cos(yy * .7)).astype(np.uint8)
    pixels = np.stack([texture, texture // 2, texture // 3], axis=-1)
    pixels[38:43, 38:43] = 5
    doc = Document(size, size)
    doc.add_image(pixels, 'Original')
    history = DocumentHistory(doc, saved=True)
    _, captured = local.capture_composite(history, preview_side=size)
    stroke = np.zeros((size, size, 4), np.uint8)
    stroke[37:44, 37:44] = [255, 0, 0, 128]
    return history, captured, {'layers': [stroke]}


def test_real_texture_search_is_donor_constrained_soft_and_undoable(tmp_path, monkeypatch):
    history, captured, canvas = setup()
    before = composite(history.document)
    donors = np.zeros((80, 80), np.uint8)
    donors[20:60, 12:32] = 255
    actual = local.patchmatch_fill
    observed = {}
    def search(image, mask, **kwargs):
        result, mapping = actual(image, mask, return_source_map=True, **kwargs)
        ys, xs = np.nonzero(mask)
        sy, sx = mapping[ys, xs].T
        assert np.all(sy >= 0) and np.all(sx >= 0)
        assert np.all(kwargs['source_mask'][sy, sx] > 0)
        assert np.all(mask[sy, sx] == 0)
        observed['result'] = result.copy()
        observed['report'] = dict(kwargs['report'])
        return result
    monkeypatch.setattr(local, 'patchmatch_fill', search)
    lid = local.apply_local_retouch(history, captured, canvas, 'Heal', method='Texture-aware',
                                    donor_selection=donors, search_distance=32)
    after = composite(history.document)
    mask = local.painted_mask(canvas, captured)
    np.testing.assert_array_equal(after[mask == 0], before[mask == 0])
    np.testing.assert_array_equal(history.document.layers[0].pixels, before)
    assert np.any(after != before)
    assert observed['report']['executed'] == 'patchmatch'
    layer = history.document.layer(lid)
    assert layer.name == 'Heal · Texture-aware'
    x, y = map(int, layer.origin)
    h, w = layer.pixels.shape[:2]
    original = cv2.cvtColor(before[y:y+h, x:x+w], cv2.COLOR_RGBA2BGR)
    # The search result uses full texture; soft brush coverage is baked once.
    alpha = mask[y:y+h, x:x+w, None].astype(np.uint32)
    expected = ((observed['result'].astype(np.uint32) * alpha
                 + original.astype(np.uint32) * (255-alpha) + 127) // 255).astype(np.uint8)
    np.testing.assert_array_equal(cv2.cvtColor(layer.pixels, cv2.COLOR_RGBA2BGR), expected)
    path = tmp_path / 'texture.comp'
    history.save(path)
    np.testing.assert_array_equal(composite(load_project(path)), after)
    history.undo()
    np.testing.assert_array_equal(composite(history.document), before)
    history.redo()
    np.testing.assert_array_equal(composite(history.document), after)


def test_degenerate_donor_reports_actual_fallback():
    history, captured, canvas = setup()
    donors = np.zeros((80, 80), np.uint8)
    donors[25, 25] = 255
    lid = local.apply_local_retouch(history, captured, canvas, 'Heal', method='Texture-aware',
                                    donor_selection=donors)
    assert 'donor-constrained fallback' in history.document.layer(lid).name


@pytest.mark.parametrize('failure', ['empty_donor', 'donor_shape', 'distance', 'fractional_distance',
                                   'nan_distance', 'paint_budget', 'search_budget', 'stale'])
def test_invalid_texture_work_never_enters_search_or_changes_history(monkeypatch, failure):
    history, captured, canvas = setup(800 if failure == 'search_budget' else 80)
    options = dict(method='Texture-aware')
    if failure == 'empty_donor': options['donor_selection'] = np.zeros((80, 80), np.uint8)
    elif failure == 'donor_shape': options['donor_selection'] = np.zeros((2, 2), np.uint8)
    elif failure == 'distance': options['search_distance'] = 257
    elif failure == 'fractional_distance': options['search_distance'] = 32.5
    elif failure == 'nan_distance': options['search_distance'] = float('nan')
    elif failure == 'paint_budget': canvas['layers'][0][:70, :70] = 255
    elif failure == 'search_budget': canvas['layers'][0][700, 700] = 255
    else: history.update_layer(history.document.layers[0].id, name='Changed')
    revision = history.revision
    before = composite(history.document)
    def forbidden(*args, **kwargs): pytest.fail('Invalid work reached texture search')
    monkeypatch.setattr(local, 'patchmatch_fill', forbidden)
    with pytest.raises(ValueError):
        local.apply_local_retouch(history, captured, canvas, 'Heal', **options)
    assert history.revision == revision and len(history.document.layers) == 1
    np.testing.assert_array_equal(composite(history.document), before)


def test_texture_heal_is_deterministic_unmocked():
    history, captured, canvas = setup()
    donors = np.zeros((80, 80), np.uint8)
    donors[20:60, 12:32] = 255
    args = (history, captured, canvas, 'Heal')
    kw = dict(method='Texture-aware', donor_selection=donors, search_distance=32)
    local.apply_local_retouch(*args, **kw)
    first = deepcopy(history.document.layers[-1].pixels)
    expected = composite(history.document)
    history.undo()
    local.apply_local_retouch(*args, **kw)
    np.testing.assert_array_equal(history.document.layers[-1].pixels, first)
    np.testing.assert_array_equal(composite(history.document), expected)


def test_nearby_pixels_donor_none_finds_texture_outside_strokes():
    history, captured, canvas = setup()
    # Donor selection None = search the whole canvas minus the stroke.
    lid = local.apply_local_retouch(history, captured, canvas, 'Heal',
                                    method='Texture-aware',
                                    donor_selection=None, search_distance=64)
    assert history.document.layer(lid).name == 'Heal · Texture-aware'


def test_donor_equal_to_limit_selection_raises_when_exhausted():
    history, captured, canvas = setup()
    # Donor mask overlapping ONLY the painted region: nothing outside
    # strokes to copy from, so the op must refuse rather than fall back.
    donors = np.zeros((80, 80), np.uint8)
    donors[37:44, 37:44] = 255
    with pytest.raises(ValueError):
        local.apply_local_retouch(history, captured, canvas, 'Heal',
                                  method='Texture-aware',
                                  donor_selection=donors, search_distance=16)
