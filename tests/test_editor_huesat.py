"""Hue/Saturation: 33^3 trilerp cube port of HueSaturation.swift (MIT,
Copyright 2026 Wonder Assembly LLC), pinned at 11d8d7a."""
import time

import numpy as np
import pytest

from retouch.editor.adjustments import (adjustment, grade, validate_adjustment,
                                        _to_hsl)

rng = np.random.default_rng(42)


def hs_record(hue=0, sat=0, light=0, colorize=False, band='master', invert=False):
    r = adjustment('Hue', hue=0 if colorize else hue, saturation=sat)
    r['hue'] = hue if colorize else 0
    r['lightness'] = light
    r['colorize'] = colorize
    r['range'] = band
    r['invertRange'] = invert
    validate_adjustment(r)
    return r


def test_identity_bit_exact():
    ramp = np.tile(np.linspace(0, 1, 256, dtype=np.float32)[:, None], (1, 3))[None]
    assert np.array_equal(grade(ramp, hs_record()), ramp)
    rand = rng.random((1000, 100, 3), dtype=np.float32)
    assert np.array_equal(grade(rand, hs_record()), rand)


def test_hue_shift_moves_primary_to_expected_hue():
    # master +30: pure red (hue 0) lands near hue 30 (orange).
    rec = hs_record(hue=30)
    out = grade(np.array([[[1.0, 0.0, 0.0]]], np.float32), rec)[0, 0]
    assert 29 < _to_hsl(*out)[0] < 31


def test_colorize_spot():
    rec = hs_record(hue=200, sat=40, light=-10, colorize=True)
    out = grade(np.full((1, 1, 3), 0.5, np.float32), rec)[0, 0]
    h, s, l = _to_hsl(*out)
    assert abs(h - 200) < 1.5 and abs(s - 0.4) < 0.01 and abs(l - 0.45) < 0.005


def test_invert_range_shifts_only_out_of_band():
    rec = hs_record(hue=30, band='reds', invert=True)
    red = grade(np.array([[[1.0, 0.0, 0.0]]], np.float32), rec)[0, 0]
    cyan = grade(np.array([[[0.0, 1.0, 1.0]]], np.float32), rec)[0, 0]
    # red is inside the inverted (non-red) band's complement: unshifted
    assert _to_hsl(*red)[0] == pytest.approx(0, abs=1)
    # cyan 180 -> ~210: outside reds, so inverted band applies the shift
    assert abs(_to_hsl(*cyan)[0] - 210) < 2


def test_band_invariance():
    from retouch.editor.document import Document, Layer
    from retouch.editor.composite import composite
    from retouch.editor.commands import DocumentHistory
    px = (rng.random((64, 64, 4), dtype=np.float32) * 255).astype(np.uint8)
    px[..., 3] = 255
    doc = Document(64, 64, [Layer('Base', px)])
    h = DocumentHistory(doc)
    h.add_layer(Layer('HS', size=(64, 64),
                      adjustment=hs_record(hue=40, sat=-20, light=10, band='blues')))
    full = composite(h.document)
    for band_rows in (7, 13, 64, 256):
        assert np.array_equal(composite(h.document, band_rows=band_rows), full), band_rows


def test_round_trip(tmp_path):
    from retouch.editor.document import Document, Layer
    from retouch.editor.composite import composite
    from retouch.editor.commands import DocumentHistory
    from retouch.editor.project_store import load_project
    px = np.zeros((4, 4, 4), np.uint8)
    px[..., 3] = 255
    px[..., :3] = (80, 100, 120)
    h = DocumentHistory(Document(4, 4, [Layer('Base', px)]))
    settings = hs_record(hue=25, sat=-30, band='greens', invert=True)
    h.add_layer(Layer('HS', size=(4, 4), adjustment=settings))
    image = composite(h.document)
    path = tmp_path / 'huesat.comp'
    h.save(path)
    again = load_project(path)
    assert again.layers[-1].adjustment == settings
    assert np.array_equal(composite(again), image)


def test_range_validation_rejects_unknown_band():
    record = hs_record()
    record['range'] = 'chartreuse'
    with pytest.raises(ValueError):
        validate_adjustment(record)


def test_perf_12mp():
    rgb = rng.random((1, 12_000_000, 3), dtype=np.float32)
    grade(rgb, hs_record())  # warm identity path
    t0 = time.time()
    grade(rgb, hs_record(hue=30, band='reds'))
    assert time.time() - t0 < 1.0
