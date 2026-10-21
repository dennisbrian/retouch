"""Black & White and Color Balance adjustment kinds: records, kernels, parity.

Kinds 14/15 of the native adjustment set (Compositor 11d8d7a): BlackWhiteSettings
and ColorBalanceSettings (Document/ImageAdjustments.swift) driving the C kernels
``adjust_black_white`` and ``adjust_color_balance`` (Rendering/AdjustPixels.c).
Ground truth is ``tests/fixtures/cg_bw_cb_reference.txt``, emitted by
``scripts/dev/cg_bw_cb_fixture.swift`` against the real C kernels.
"""
import numpy as np
import pytest

from retouch.editor import (Document, DocumentHistory, Layer, UnsupportedFeature,
                            composite, load_project)
from retouch.editor.adjustments import adjustment, grade, validate_adjustment
from retouch.editor.selections import selection


def solid(color, width=4, height=4, alpha=255):
    a = np.empty((height, width, 4), np.uint8)
    a[..., :3], a[..., 3] = color, alpha
    return a


# --- Fixture probes: identical to scripts/dev/cg_bw_cb_fixture.swift ----------
def _probes():
    px = []
    for i in range(256):
        px.append((i, 255 - i, (i * 7) % 256))
    px += [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
           (0, 255, 255), (255, 255, 255), (0, 0, 0), (128, 128, 128), (64, 64, 64),
           (200, 100, 50), (64, 192, 32), (10, 20, 30), (240, 180, 160),
           (30, 90, 200), (120, 30, 180), (180, 200, 40), (90, 40, 40)]
    return np.array(px, np.float32) / 255.0


def _reference():
    reference = {}
    for line in (Path(__file__).parent / 'fixtures/cg_bw_cb_reference.txt'
                 ).read_text().splitlines():
        label, source, result = line.split()
        reference.setdefault(label, []).append(result)
    return reference


from pathlib import Path


_BW_CASES = {
    'bw_default': dict(),
    'bw_unweighted': dict(reds=100, yellows=100, greens=100, cyans=100,
                          blues=100, magentas=100),
    'bw_boosted': dict(reds=200, yellows=300, greens=-200, cyans=-100,
                       blues=40, magentas=0),
    'bw_tinted': dict(bw_tint=True, tint_hue=40, tint_saturation=20),
}

_CB_CASES = {
    'cb_shadows': dict(shadow_cyan_red=50, shadow_magenta_green=-25,
                       shadow_yellow_blue=75),
    'cb_mids': dict(mid_cyan_red=-40, mid_magenta_green=60, mid_yellow_blue=-20),
    'cb_highlights': dict(highlight_cyan_red=30, highlight_magenta_green=80,
                          highlight_yellow_blue=-60),
    'cb_all_nolum': dict(shadow_cyan_red=50, shadow_magenta_green=50,
                         shadow_yellow_blue=50, mid_cyan_red=50,
                         mid_magenta_green=50, mid_yellow_blue=50,
                         highlight_cyan_red=50, highlight_magenta_green=50,
                         highlight_yellow_blue=50, preserve_luminosity=False),
    'cb_all_keep': dict(shadow_cyan_red=50, shadow_magenta_green=50,
                        shadow_yellow_blue=50, mid_cyan_red=50,
                        mid_magenta_green=50, mid_yellow_blue=50,
                        highlight_cyan_red=50, highlight_magenta_green=50,
                        highlight_yellow_blue=50),
}


@pytest.mark.parametrize('label', sorted(_BW_CASES))
def test_bw_matches_native_c_kernel_byte_for_byte(label):
    record = adjustment('Black & White', **_BW_CASES[label])
    out = grade(_probes().copy(), record)
    got = np.rint(out * 255).astype(int)
    expected = np.array([[int(h[j:j + 2], 16) for j in (0, 2, 4)]
                         for h in _reference()[label]])
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize('label', sorted(_CB_CASES))
def test_cb_matches_native_c_kernel_byte_for_byte(label):
    record = adjustment('Color Balance', **_CB_CASES[label])
    out = grade(_probes().copy(), record)
    got = np.rint(out * 255).astype(int)
    expected = np.array([[int(h[j:j + 2], 16) for j in (0, 2, 4)]
                         for h in _reference()[label]])
    np.testing.assert_array_equal(got, expected)


def test_bw_defaults_are_photoshops_and_pure_primaries_hit_their_weight():
    # Native defaults (ImageAdjustments.swift): 40/60/40/60/20/80.
    record = adjustment('Black & White')['blackWhiteSettings']
    assert (record['reds'], record['yellows'], record['greens'],
            record['cyans'], record['blues'], record['magentas']) == (40, 60, 40, 60, 20, 80)
    red = np.array([[[1.0, 0.0, 0.0]]], np.float32)
    assert np.allclose(grade(red, adjustment('Black & White')), .4, atol=1e-6)
    blue = np.array([[[0.0, 0.0, 1.0]]], np.float32)
    assert np.allclose(grade(blue, adjustment('Black & White')), .2, atol=1e-6)
    # Equal weights give plain max-channel luminance: any color stays its max
    # (byte domain: .9 * 255 rounds to 230, i.e. 230/255).
    flat = adjustment('Black & White', reds=100, yellows=100, greens=100,
                      cyans=100, blues=100, magentas=100)
    ramp = np.array([[[.2, .9, .5]]], np.float32)
    assert np.allclose(grade(ramp, flat), round(.9 * 255) / 255, atol=1e-6)


def test_bw_weights_reach_beyond_one_and_clamp():
    boosted = adjustment('Black & White', reds=300)
    red = np.array([[[1.0, 0.0, 0.0]]], np.float32)
    assert np.allclose(grade(red, boosted), 1.0, atol=1e-6)
    cut = adjustment('Black & White', reds=-200)
    assert np.allclose(grade(red, cut), 0.0, atol=1e-6)


def test_bw_tint_keeps_gray_as_lightness():
    record = adjustment('Black & White', bw_tint=True, tint_hue=40,
                       tint_saturation=25)
    gray = np.full((1, 1, 3), .5, np.float32)
    out = grade(gray, record)
    # Same gray in, same tone out: the tinted result keeps the gray's lightness.
    lightness = (out.max() + out.min()) / 2
    assert abs(lightness - round(.5 * 255) / 255) < 2 / 255
    red = np.array([[[1.0, 0.0, 0.0]]], np.float32)
    out = grade(red, record)
    assert out.max() > out.min()  # sepia: red channel brighter than blue


def test_cb_identity_and_preserve_luminosity():
    gray = np.full((4, 4, 3), .5, np.float32)
    np.testing.assert_array_equal(grade(gray, adjustment('Color Balance')), gray)
    record = adjustment('Color Balance', mid_cyan_red=40, mid_magenta_green=-30)
    out = grade(np.full((1, 1, 3), .5, np.float32), record)
    before = .5 * (0.299 + 0.587 + 0.114)
    after = float(np.dot(out[0, 0], (0.299, 0.587, 0.114)))
    assert abs(after - before) < 2 / 255  # brightness put back
    plain = adjustment('Color Balance', mid_cyan_red=40, preserve_luminosity=False)
    out2 = grade(np.full((1, 1, 3), .5, np.float32), plain)
    assert abs(float(out2[0, 0, 0]) - .5) > 2 / 255  # color moves, brightness too


@pytest.mark.parametrize('kind, kwargs, bad', [
    ('Black & White', dict(reds=-201), ValueError),
    ('Black & White', dict(reds=301), ValueError),
    ('Black & White', dict(tint_hue=361), ValueError),
    ('Black & White', dict(tint_saturation=101), ValueError),
    ('Black & White', dict(tint_hue=-1), ValueError),
    ('Color Balance', dict(shadow_cyan_red=-101), ValueError),
    ('Color Balance', dict(mid_magenta_green=101), ValueError),
    ('Color Balance', dict(highlight_yellow_blue=-101), ValueError),
    ('Color Balance', dict(preserve_luminosity=1), ValueError),
])
def test_bw_cb_validation_rejects_out_of_range(kind, kwargs, bad):
    with pytest.raises(bad):
        adjustment(kind, **kwargs)


@pytest.mark.parametrize('kind', ['Black & White', 'Color Balance'])
def test_bw_cb_settings_do_not_ride_other_kinds(kind):
    mine, theirs = (('blackWhiteSettings', 'colorBalanceSettings')
                    if kind == 'Black & White'
                    else ('colorBalanceSettings', 'blackWhiteSettings'))
    record = adjustment(kind)
    record[theirs] = adjustment('Black & White')['blackWhiteSettings'] \
        if theirs == 'blackWhiteSettings' else \
        adjustment('Color Balance')['colorBalanceSettings']
    with pytest.raises(UnsupportedFeature, match='do not match its kind'):
        validate_adjustment(record)
    record = adjustment('Grain', amount=50, seed=3)
    record[mine] = adjustment(kind)[mine]
    with pytest.raises(UnsupportedFeature, match='do not match its kind'):
        validate_adjustment(record)


@pytest.mark.parametrize('kind', ['Black & White', 'Color Balance'])
def test_bw_cb_kind_now_supported_but_others_still_refused(kind):
    from retouch.editor.adjustments import KINDS
    assert kind in KINDS
    record = dict(kind='Gaussian Blur', hue=0, saturation=0, lightness=0, colorize=False,
                  range='master', invertRange=False,
                  levels=dict(channel='RGB', ranges=[dict(black=0, white=255, gamma=1,
                                                          outputBlack=0, outputWhite=255)] * 4),
                  curves=dict(channel='RGB',
                              channels=[[dict(x=0, y=0), dict(x=255, y=255)]] * 4))
    with pytest.raises(UnsupportedFeature, match='Supported adjustments'):
        validate_adjustment(record)


@pytest.mark.parametrize('kind, kwargs', [
    ('Black & White', dict(reds=120, yellows=-40, greens=90, cyans=15,
                           blues=250, magentas=-180, bw_tint=True,
                           tint_hue=210, tint_saturation=65)),
    ('Color Balance', dict(shadow_cyan_red=-60, shadow_magenta_green=25,
                            shadow_yellow_blue=90, mid_cyan_red=35,
                            mid_magenta_green=-45, mid_yellow_blue=10,
                            highlight_cyan_red=70, highlight_magenta_green=5,
                            highlight_yellow_blue=-80, preserve_luminosity=False)),
])
def test_bw_cb_adjustment_layer_round_trips(kind, kwargs, tmp_path):
    base = Layer('Base', solid((80, 100, 120), alpha=128))
    settings = adjustment(kind, **kwargs)
    adj = Layer(kind, size=(4, 4), adjustment=settings,
                mask=selection(4, 4, 'Rectangle', 0, 0, 2, 4), opacity=.5)
    h = DocumentHistory(Document(4, 4, [base]))
    before = composite(h.document)
    h.add_layer(adj)
    image = composite(h.document)
    np.testing.assert_array_equal(image[..., 3], before[..., 3])
    np.testing.assert_array_equal(image[:, 2:], before[:, 2:])
    assert not np.array_equal(image[:, :2], before[:, :2])
    h.save(tmp_path / (kind.replace(' ', '_').replace('&', 'and') + '.comp'))
    again = load_project(tmp_path / (kind.replace(' ', '_').replace('&', 'and') + '.comp'))
    assert again.layers[-1].adjustment == settings
    np.testing.assert_array_equal(composite(again), image)


def test_bw_partial_coverage_grades_only_covered_pixels():
    # The C kernel skips transparent pixels; the port must leave uncovered
    # pixels exactly as given, including at fractional coverage edges.
    rgb = np.full((1, 2, 3), .75, np.float32)
    a = np.array([[1.0, 0.0]], np.float32)
    from retouch.editor.adjustments import _grade_black_white, _grade_color_balance
    for kernel, settings in ((_grade_black_white, {}),
                            (_grade_color_balance,
                             dict(mid_cyan_red=50, preserveLuminosity=True))):
        out = kernel(rgb.copy(), a, settings)
        assert np.array_equal(out[0, 1], rgb[0, 1])
        assert not np.array_equal(out[0, 0], rgb[0, 0])
