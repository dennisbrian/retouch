"""Separable blends: independent pixel formulas, alpha, bands and persistence."""
import numpy as np
import pytest

from retouch.editor import Document, DocumentHistory, composite, load_project, save_project


def blend(back, src, mode):
    if mode == 'Multiply':
        return back * src
    if mode == 'Screen':
        return 1 - (1 - back) * (1 - src)
    if mode == 'Overlay':
        return 2 * back * src if back <= 0.5 else 1 - 2 * (1 - back) * (1 - src)
    return abs(back - src)


MODES = ['Multiply', 'Screen', 'Overlay', 'Difference']


@pytest.mark.parametrize('mode', MODES)
@pytest.mark.parametrize('back_alpha', [0, 90, 255])
def test_translucent_blend_matches_independent_float64_formula(mode, back_alpha):
    base = np.array([[[30, 130, 240, back_alpha]]], np.uint8)
    top = np.array([[[220, 90, 160, 170]]], np.uint8)
    doc = Document(1, 1)
    doc.add_image(base, 'Base')
    layer = doc.add_image(top, 'Top')
    layer.blend_mode, layer.opacity = mode, 0.6
    layer.set_mask(np.array([[120]], np.uint8))
    ab = back_alpha / 255
    a = 170 / 255 * 0.6 * 120 / 255
    alpha = a + ab * (1 - a)
    expected = []
    for b, s in zip(base[0, 0, :3] / 255, top[0, 0, :3] / 255):
        premultiplied = (a * (1 - ab) * s + ab * (1 - a) * b
                         + a * ab * blend(b, s, mode))
        expected.append(round(premultiplied / alpha * 255))
    expected.append(round(alpha * 255))
    assert np.abs(composite(doc)[0, 0].astype(int) - expected).max() <= 1


@pytest.mark.parametrize('mode', MODES)
def test_opaque_known_pixels_round_trip_and_undo(mode, tmp_path):
    doc = Document(3, 1)
    doc.add_image(np.array([[[0, 128, 255, 255]] * 3], np.uint8), 'Base')
    lid = doc.add_image(np.array([[[200, 100, 50, 255]] * 3], np.uint8), 'Top').id
    history = DocumentHistory(doc, saved=True)
    history.update_layer(lid, blend_mode=mode)
    out = composite(history.document)
    expected = [round(blend(b / 255, s / 255, mode) * 255)
                for b, s in zip([0, 128, 255], [200, 100, 50])]
    assert out[0, 0].tolist() == expected + [255]
    again = load_project(save_project(history.document, tmp_path / 'blend.comp'))
    assert again.layers[-1].blend_mode == mode
    np.testing.assert_array_equal(composite(again), out)
    assert history.undo() and not history.dirty
    assert composite(history.document)[0, 0].tolist() == [200, 100, 50, 255]
    assert history.redo()
    np.testing.assert_array_equal(composite(history.document), out)


@pytest.mark.parametrize('mode', MODES)
def test_bands_masks_and_hidden_layers(mode):
    rng = np.random.default_rng(4)
    doc = Document(13, 17)
    doc.add_image(rng.integers(0, 256, (17, 13, 4), dtype=np.uint8), 'Base')
    top = doc.add_image(rng.integers(0, 256, (17, 13, 4), dtype=np.uint8), 'Top')
    top.blend_mode = mode
    mask = rng.integers(0, 256, (17, 13), dtype=np.uint8)
    mask[0] = 0
    top.set_mask(mask)
    np.testing.assert_array_equal(composite(doc, band_rows=3), composite(doc, band_rows=100))
    shown = composite(doc)
    top.visible = False
    hidden = composite(doc)
    np.testing.assert_array_equal(shown[0], hidden[0])


def test_identical_opaque_layers_difference_is_black():
    doc = Document(4, 3)
    pixels = np.random.default_rng(1).integers(0, 256, (3, 4, 3), dtype=np.uint8)
    doc.add_image(pixels, 'Base')
    doc.add_image(pixels, 'Inspect').blend_mode = 'Difference'
    assert not composite(doc)[..., :3].any()
    assert (composite(doc)[..., 3] == 255).all()
