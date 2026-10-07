"""Reference compositor (retouch/editor/composite.py).

Expected values are the pixel checks of Compositor's own tests at 11d8d7a
(``LayerAppearanceTests.blendModesAndOpacityMatchKnownPixels``,
``LayerMaskTests.coverageOpacityAndDisabledMasksRenderCorrectly`` and
``transformedMaskResizeAndCanvasChangesStayAligned``), translated with the
same tolerances, plus analytic source-over checks. MIT, Copyright (c) 2026
Wonder Assembly LLC for the translated cases.
"""
import numpy as np
import pytest
from PIL import Image

from retouch.editor import (
    Document,
    Layer,
    UnsupportedFeature,
    composite,
    export_png,
    load_project,
)
from tests import compositor_fixtures as fx


def _rgba(document):
    return composite(document).astype(int)


class TestUpstreamPixelChecks:
    def test_normal_over_gray(self, tmp_path):
        out = _rgba(load_project(fx.two_grays(tmp_path / 'a.comp')))
        assert abs(out[0, 0, 0] / 255 - 0.8) < 0.02 and out[0, 0, 3] == 255

    @pytest.mark.parametrize('opacity, expected', [(0.5, 0.6), (0.0, 0.4)])
    def test_opacity(self, tmp_path, opacity, expected):
        out = _rgba(load_project(fx.two_grays(tmp_path / 'a.comp', opacity)))
        assert abs(out[0, 0, 0] / 255 - expected) < 0.02

    def test_mask_coverage(self, tmp_path):
        out = _rgba(load_project(fx.red_with_mask(tmp_path / 'a.comp')))
        assert out[..., 3].ravel().tolist() == [255, 0, 128, 255]

    def test_mask_with_half_opacity(self, tmp_path):
        out = _rgba(load_project(fx.red_with_mask(tmp_path / 'a.comp', opacity=0.5)))
        assert all(abs(a - b) <= 1 for a, b in
                   zip(out[..., 3].ravel(), [128, 0, 64, 128]))

    def test_disabled_mask_reveals_all(self, tmp_path):
        out = _rgba(load_project(fx.red_with_mask(tmp_path / 'a.comp', enabled=False)))
        assert out[..., 3].ravel().tolist() == [255, 255, 255, 255]

    @pytest.mark.parametrize('value, alphas', [(255, [255] * 4), (0, [0] * 4)])
    def test_uniform_1x1_mask(self, tmp_path, value, alphas):
        mask = np.array([[value]], np.uint8)
        out = _rgba(load_project(fx.red_with_mask(tmp_path / 'a.comp', mask=mask)))
        assert out[..., 3].ravel().tolist() == alphas

    def test_flip_moves_the_mask_with_the_layer(self, tmp_path):
        out = _rgba(load_project(fx.red_with_mask(tmp_path / 'a.comp', flip_x=True)))
        assert out[..., 3].ravel().tolist() == [0, 255, 255, 128]


class TestSourceOver:
    def test_matches_analytic_blend_in_gamma_srgb(self):
        rng = np.random.default_rng(7)
        base = rng.integers(0, 256, (37, 53, 3), dtype=np.uint8)
        top = rng.integers(0, 256, (37, 53, 4), dtype=np.uint8)
        mask = rng.integers(0, 256, (37, 53), dtype=np.uint8)
        doc = Document(53, 37)
        doc.add_image(base, 'Base')
        layer = doc.add_image(top, 'Top')
        layer.opacity = 0.7
        layer.set_mask(mask)
        a = top[..., 3:] / 255 * mask[..., None] / 255 * 0.7
        expected = np.rint(top[..., :3] * a + base * (1 - a))
        out = composite(doc, band_rows=8).astype(int)   # bands must not show
        assert np.abs(out[..., :3] - expected).max() <= 1
        assert (out[..., 3] == 255).all()

    def test_translucent_over_transparent_stays_straight_alpha(self):
        doc = Document(3, 1)
        doc.add_image(np.array([[[200, 100, 50, 128]] * 3], np.uint8), 'Half')
        out = composite(doc)
        assert out[0, 0].tolist() == [200, 100, 50, 128]

    def test_offset_layer_is_clipped_to_canvas(self):
        doc = Document(4, 4)
        doc.add_layer(Layer(name='Patch', pixels=fx.solid(3, 3, (9, 8, 7, 255)),
                            origin=(2, -1)))
        out = composite(doc)
        assert out[..., 3].tolist() == [[0, 0, 255, 255]] * 2 + [[0, 0, 0, 0]] * 2
        assert out[0, 2, :3].tolist() == [9, 8, 7]

    def test_hidden_and_blank_layers_draw_nothing(self):
        doc = Document(2, 2)
        doc.add_image(fx.solid(2, 2, (10, 20, 30, 255)), 'Base')
        doc.add_layer(Layer(name='Blank', size=(2, 2)))
        doc.add_image(fx.solid(2, 2, (250, 0, 0, 255)), 'Hidden').visible = False
        assert composite(doc)[0, 0].tolist() == [10, 20, 30, 255]

    def test_layer_order_is_bottom_to_top(self):
        doc = Document(1, 1)
        red = doc.add_image(fx.solid(1, 1, (255, 0, 0, 255)), 'Red')
        doc.add_image(fx.solid(1, 1, (0, 0, 255, 255)), 'Blue')
        assert composite(doc)[0, 0, :3].tolist() == [0, 0, 255]
        doc.move_layer(red.id, 1)
        assert composite(doc)[0, 0, :3].tolist() == [255, 0, 0]


class TestUnsupported:
    def test_visible_non_normal_blend_is_refused(self):
        doc = Document(1, 1)
        doc.add_image(fx.solid(1, 1, (1, 2, 3, 255)), 'Top').blend_mode = 'Multiply'
        with pytest.raises(UnsupportedFeature, match='Multiply'):
            composite(doc)

    def test_hidden_non_normal_blend_is_kept_but_not_drawn(self):
        doc = Document(1, 1)
        doc.add_image(fx.solid(1, 1, (1, 2, 3, 255)), 'Base')
        diff = doc.add_image(fx.solid(1, 1, (9, 9, 9, 255)), 'Diff')
        diff.blend_mode, diff.visible = 'Difference', False
        assert composite(doc)[0, 0].tolist() == [1, 2, 3, 255]


def test_png_export_carries_resolution(tmp_path):
    doc = Document(2, 2, resolution=300)
    doc.add_image(fx.solid(2, 2, (5, 6, 7, 128)), 'Base')
    path = export_png(doc, tmp_path / 'flat.png')
    with Image.open(path) as image:
        assert image.mode == 'RGBA'
        assert round(image.info['dpi'][0]) == 300
        np.testing.assert_array_equal(np.asarray(image), composite(doc))
    assert [p.name for p in tmp_path.iterdir()] == ['flat.png']
