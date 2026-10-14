"""Curves: monotone-cubic Hermite grading, ported from Curves.swift (MIT,
Copyright 2026 Wonder Assembly LLC), pinned at 11d8d7a."""
import pytest
import numpy as np
from retouch.editor.adjustments import (KINDS, adjustment, validate_adjustment,
                                        grade, _curve_value, _curves_lut)
from retouch.editor.document import Document, Layer, UnsupportedFeature
from retouch.editor.project_store import save_project, load_project

PTS = lambda *xy: [dict(x=float(a), y=float(b)) for a, b in xy]
MONO = PTS((0, 0), (64, 64), (128, 80), (255, 255))
HM = PTS((0, 0), (50, 100), (120, 120), (255, 200))
SIGN = PTS((0, 100), (80, 40), (180, 180), (255, 255))


def curves_record(channels=None):
    ident = PTS((0, 0), (255, 255))
    record = adjustment('Curves')
    record['curves']['channels'] = (channels or [ident] * 4)
    return record


def test_kind_registered_atomically():
    assert 'Curves' in KINDS


@pytest.mark.parametrize('points,x,expected', [
    (MONO, 32, 36.8), (MONO, 96, 71.814268440145), (MONO, 200, 166.320195392202),
    (HM, 85, 111.001506024096), (HM, 130, 124.150802264078),
    (SIGN, 130, 95.416666666667), (SIGN, 40, 62.5),
])
def test_swift_verified_curve_values(points, x, expected):
    assert _curve_value(x, points) == pytest.approx(expected, abs=1e-6)


def test_composite_lut_matches_swift_baked_tables():
    ident = PTS((0, 0), (255, 255))
    red = _curves_lut(dict(channel='RGB', channels=[
        ident, PTS((0, 0), (128, 64), (255, 255)), ident, ident]))
    assert red[0][100] == pytest.approx(45.719145229617 / 255, abs=1e-6)
    rgb = _curves_lut(dict(channel='RGB', channels=[
        ident, PTS((0, 0), (128, 200), (255, 255)), ident, ident]))
    assert rgb[0][100] == pytest.approx(171.362974493777 / 255, abs=1e-6)


def test_identity_is_bit_exact_noop():
    rng = np.random.default_rng(7)
    pixels = rng.random((16, 16, 3), np.float32)
    assert np.array_equal(grade(pixels.copy(), curves_record()), pixels)


def test_lut_is_monotone_for_monotone_points():
    lut = _curves_lut(dict(channel='RGB', channels=[MONO, MONO, MONO, MONO]))
    values = [entry[0] for entry in lut]
    assert all(b >= a - 1e-9 for a, b in zip(values, values[1:]))


def test_band_invariance_exact():
    rng = np.random.default_rng(11)
    pixels = rng.random((64, 77, 3), np.float32)
    record = curves_record([HM, MONO, SIGN, PTS((0, 0), (255, 255))])
    full = grade(pixels.copy(), record)
    bands = np.concatenate([grade(pixels[a:b].copy(), record)
                             for a, b in ((0, 10), (10, 44), (44, 64))])
    assert np.array_equal(full, bands)


def test_round_trip_preserves_curves_verbatim(tmp_path):
    record = curves_record([HM, MONO, SIGN, PTS((0, 0), (255, 255))])
    pixels = np.dstack([np.full((8, 8, 3), 102, np.uint8),
                        np.full((8, 8), 255, np.uint8)])
    doc = Document(8, 8, [Layer('Base', pixels),
                          Layer('Curves', adjustment=record, size=(8, 8))])
    save_project(doc, tmp_path / 'curves.comp')
    again = load_project(tmp_path / 'curves.comp')
    assert again.layers[-1].adjustment == record


@pytest.mark.parametrize('points', [
    PTS((0, 0)),                                        # fewer than 2 points
    PTS(*[(i * 255 / 32, 0) for i in range(33)]),       # more than 32
    PTS((1, 0), (255, 255)),                            # first x must be 0
    PTS((0, 0), (254, 255)),                            # last x must be 255
    PTS((0, 0), (200, 10), (100, 20), (255, 255)),      # x must strictly increase
    PTS((0, -1), (255, 255)),                           # y out of range
    PTS((0, 0), (256, 255)),                            # x out of range
])
def test_validation_rejects_bad_point_lists(points):
    record = curves_record()
    record['curves']['channels'][1] = points
    with pytest.raises((ValueError, UnsupportedFeature)):
        validate_adjustment(record)


def test_curves_record_never_renders_as_levels(monkeypatch):
    # Regression: without 'Curves' in KINDS, a curves record must be rejected
    # by validation, never silently graded through the Levels path.
    import retouch.editor.adjustments as adjustments
    record = curves_record([MONO, MONO, MONO, HM])
    monkeypatch.setattr(adjustments, 'KINDS',
                        tuple(k for k in adjustments.KINDS if k != 'Curves'))
    with pytest.raises((ValueError, UnsupportedFeature)):
        adjustments.validate_adjustment(record)


def test_rgb_to_hsl_red_max_branch_wraps_and_masks():
    # Regression: the red-max branch dropped its mask and the % 360 wrap, so
    # magenta-family hues (g < b in the red-max branch) computed as 0.
    from retouch.editor.adjustments import _rgb_to_hsl
    magenta = _rgb_to_hsl(np.array([[[213/255, 42/255, 211/255]]], np.float32))
    assert 300 < magenta[0, 0, 0] * 360 < 302
    # hue continuity across the wrap: r>g=b tie resolves to red (hue 0)
    tie = _rgb_to_hsl(np.array([[[255/255, 128/255, 128/255]]], np.float32))
    assert tie[0, 0, 0] * 360 == 0


def test_hsl_to_rgb_sector_four_receives_its_x():
    # Regression: sector 4 (hue 240-300, blue -> magenta) never received x,
    # collapsing the whole violet band to pure blue.
    from retouch.editor.adjustments import _hsl_to_rgb
    violet = _hsl_to_rgb(np.array([[[270/360, 1.0, 0.5]]], np.float32))
    np.testing.assert_allclose(violet[0, 0], [0.5, 0, 1], atol=1e-6)
    # round trip stays tight across the full hue wheel
    from retouch.editor.adjustments import _rgb_to_hsl
    hue = np.linspace(0, 1, 720, endpoint=False, dtype=np.float32)
    hsl = np.stack([hue, np.full_like(hue, .8), np.full_like(hue, .5)], -1)
    px = _hsl_to_rgb(hsl)
    back = _rgb_to_hsl(px)
    np.testing.assert_allclose(back[..., 0], (hue + .5/720) % 1, atol=5e-3)
