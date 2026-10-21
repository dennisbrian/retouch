"""Margin-aware adjustments: Tonal Contrast and Bloom band plumbing.

The two blur-based kinds must be invisible to band splits: the compositor
grows each band's fetch window by the kind's margin (ceil(3 sigma) for the
truncate-3.0 Gaussian; 3r+2 for Bloom, upstream's own blurMargin), grades,
crops back. Windows only fetch what exists, so margins larger than the
canvas still work.
"""
import numpy as np
import pytest

from retouch.editor.adjustments import (KINDS, adjustment, adjustment_margin,
                                        grade, validate_adjustment)
from retouch.editor.composite import _BandRenderer, composite
from retouch.editor.document import Document, Layer

RNG = np.random.default_rng(11)


def _photo_document(height=120, width=90):
    doc = Document(width=width, height=height)
    pixels = np.dstack([(RNG.random((height, width, 3)) * 255).astype(np.uint8),
                        np.full((height, width), 255, np.uint8)])
    doc.add_layer(Layer('photo', size=(width, height), pixels=pixels))
    return doc


def _float_bands(doc, band_rows):
    height = doc.height
    renderer = _BandRenderer(doc)
    rgb = np.empty((height, doc.width, 3), np.float32)
    alpha = np.empty((height, doc.width), np.float32)
    for top in range(0, height, band_rows):
        band_rgb, band_alpha = renderer.render(top, min(height, top + band_rows))
        rgb[top:top + band_rgb.shape[0]] = band_rgb
        alpha[top:top + band_alpha.shape[0]] = band_alpha
    return rgb, alpha


@pytest.mark.parametrize('kind', ['Tonal Contrast', 'Bloom'])
@pytest.mark.parametrize('band_rows', [1, 7, 256])
def test_band_invariance(kind, band_rows):
    """A banded render is bit-identical to the whole-canvas one (< 1e-6)."""
    doc = _photo_document()
    doc.add_layer(Layer(kind, size=(doc.width, doc.height),
                        adjustment=adjustment(kind)))
    full = _BandRenderer(doc).render(0, doc.height)
    banded = _float_bands(doc, band_rows)
    error = max(np.abs(full[0] - banded[0]).max(), np.abs(full[1] - banded[1]).max())
    assert error < 1e-6


@pytest.mark.parametrize('kind,radius,bloom_radius', [
    ('Tonal Contrast', 100, None), ('Bloom', None, 150)])
def test_margin_exceeds_canvas(kind, radius, bloom_radius):
    """Max radius margins (300/452 rows) exceed the canvas; fetch-what-exists."""
    doc = _photo_document(height=80, width=60)
    record = adjustment(kind, radius=radius) if radius is not None \
        else adjustment(kind, bloom_radius=bloom_radius)
    doc.add_layer(Layer(kind, size=(60, 80), adjustment=record))
    full = composite(doc)
    for band_rows in (1, 5, 80):
        assert np.array_equal(full, composite(doc, band_rows=band_rows))


def test_tonal_contrast_values():
    """Flat regions are a fixed point (detail 0); output stays clamped."""
    flat = np.full((10, 10, 3), 0.5, np.float32)
    out = grade(flat, adjustment('Tonal Contrast'))
    assert np.allclose(out, flat, atol=1e-6)
    ramp = np.tile(np.linspace(0, 1, 64, dtype=np.float32)[None, :, None], (16, 1, 3))
    out = grade(ramp, adjustment('Tonal Contrast', amount=100))
    assert (out >= -1e-7).all() and (out <= 1 + 1e-7).all()


def test_bloom_values():
    """Bloom only adds light (screen); black stays black."""
    black = np.zeros((10, 10, 3), np.float32)
    assert np.allclose(grade(black, adjustment('Bloom', amount=100)), 0.0)
    img = RNG.random((20, 20, 3)).astype(np.float32)
    out = grade(img, adjustment('Bloom', amount=100))
    assert (out >= img - 1e-7).all()


def test_margins_match_kernels():
    assert adjustment_margin(adjustment('Tonal Contrast')) == 48  # ceil(16 * 3)
    assert adjustment_margin(adjustment('Bloom')) == 74  # ceil(24 * 3 + 2)
    assert adjustment_margin(adjustment('Tonal Contrast', radius=100)) == 300
    assert adjustment_margin(adjustment('Bloom', bloom_radius=150)) == 452
    for kind in KINDS:
        if kind not in ('Tonal Contrast', 'Bloom'):
            assert adjustment_margin(adjustment(kind)) == 0
    # zero strength or radius needs no window
    assert adjustment_margin(adjustment('Tonal Contrast', amount=0)) == 0
    assert adjustment_margin(adjustment('Bloom', amount=0)) == 0


def test_param_ranges_and_round_trip():
    assert adjustment('Tonal Contrast')['tonalSettings'] == dict(
        amount=50, radius=16, shadows=40, midtones=60, highlights=30)
    assert adjustment('Bloom')['bloomSettings'] == dict(amount=40, radius=24)
    for bad in (dict(amount=101), dict(radius=0), dict(radius=101),
                dict(shadows=-101), dict(midtones=101), dict(highlights=-101)):
        record = adjustment('Tonal Contrast')
        record['tonalSettings'].update(bad)
        with pytest.raises(ValueError):
            validate_adjustment(record)
    for bad in (dict(amount=-1), dict(amount=101), dict(radius=0), dict(radius=151)):
        record = adjustment('Bloom')
        record['bloomSettings'].update(bad)
        with pytest.raises(ValueError):
            validate_adjustment(record)
    # round-trip through the controls builder (UI wiring contract: 16 base
    # sliders, then [4 curve tables] as one list, then grain/noise, tonal, bloom)
    from retouch.editor.adjustment_controls import controls_adjustment
    args = ('Tonal Contrast',) + (0,) * 15
    tables = [[[0, 0], [255, 255]]] * 4
    grain_noise = [25, 1.5, 50, 0, 25, False, False]
    tonal = (70, 24, -20, 40, 10)
    record = controls_adjustment(*args, tables, *grain_noise, *tonal, 60, 80)
    assert record['tonalSettings'] == dict(amount=70, radius=24, shadows=-20,
                                           midtones=40, highlights=10)


def test_bloom_sliders_round_trip():
    from retouch.editor.adjustment_controls import controls_adjustment
    args = ('Bloom',) + (0,) * 15
    tables = [[[0, 0], [255, 255]]] * 4
    grain_noise = [25, 1.5, 50, 0, 25, False, False]
    record = controls_adjustment(*args, tables, *grain_noise,
                                 50, 16, 40, 60, 30, 65, 100)
    assert record['bloomSettings'] == dict(amount=65, radius=100)


def test_save_load_round_trip(tmp_path):
    from retouch.editor.commands import DocumentHistory
    from retouch.editor.project_store import load_project, save_project
    doc = _photo_document()
    history = DocumentHistory(doc)
    history.add_layer(Layer('tc', size=(doc.width, doc.height),
                            adjustment=adjustment('Tonal Contrast', radius=42)))
    history.add_layer(Layer('bl', size=(doc.width, doc.height),
                            adjustment=adjustment('Bloom', amount=70)))
    target = tmp_path / 'doc.comp'
    save_project(history.document, target)
    loaded = load_project(target)
    kinds = [layer.adjustment['kind'] for layer in loaded.layers
             if layer.adjustment is not None]
    assert kinds == ['Tonal Contrast', 'Bloom']
    tonal = next(l.adjustment for l in loaded.layers
                 if l.adjustment and l.adjustment['kind'] == 'Tonal Contrast')
    assert tonal['tonalSettings']['radius'] == 42


def test_undo_restores_adjustment():
    from retouch.editor.commands import DocumentHistory
    doc = _photo_document()
    history = DocumentHistory(doc)
    history.add_layer(Layer('tc', size=(doc.width, doc.height),
                            adjustment=adjustment('Tonal Contrast')))
    assert any(l.adjustment for l in history.document.layers)
    history.undo()
    assert not any(l.adjustment for l in history.document.layers)
    history.redo()
    assert any(l.adjustment for l in history.document.layers)


def test_seam_free_gradient():
    """A smooth vertical gradient shows no band-seam after either kind."""
    height, width = 96, 8
    ramp = np.tile(np.linspace(0, 1, height, dtype=np.float32)[:, None, None],
                   (1, width, 3))
    doc = Document(width=width, height=height)
    pixels = np.dstack([(ramp * 255).astype(np.uint8),
                        np.full((height, width), 255, np.uint8)])
    doc.add_layer(Layer('ramp', size=(width, height), pixels=pixels))
    for kind in ('Tonal Contrast', 'Bloom'):
        doc.add_layer(Layer(kind, size=(width, height), adjustment=adjustment(kind)))
        full = composite(doc)
        banded = composite(doc, band_rows=7)
        assert np.array_equal(full, banded), kind
        doc.remove_layer(doc.layers[0].id if doc.layers[0].adjustment else doc.layers[1].id)
