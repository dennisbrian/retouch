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
import cv2
from pathlib import Path
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

    def test_unlinked_mask_same_placement_renders_as_linked(self, tmp_path):
        # Upstream clipImage: a placement equal to the layer's transform is the
        # linked case — identical pixels.
        linked = _rgba(load_project(fx.red_with_mask(tmp_path / 'l.comp')))
        extra = {'maskFile': fx.IDS[0] + '.mask.png', 'maskLinked': False,
                 'maskPlacement': fx.transform(2, 2, sampling='Nearest')}
        record = fx.layer_record(0, 'Red', 2, 2, **extra)
        record['transform'] = fx.transform(2, 2, sampling='Nearest')
        path = fx.write_package(tmp_path / 'u.comp', 2, 2, [record],
                                images={0: fx.solid(2, 2, (255, 0, 0, 255))},
                                masks={0: fx.MASK_2X2})
        np.testing.assert_array_equal(_rgba(load_project(path)), linked)

    def test_unlinked_mask_shift_moves_reveal_window(self):
        # Mask [255,0 / 128,255] placed one pixel right of the layer: layer
        # column x samples mask column x-1, so the reveal window slides right.
        # Outside the mask's rect the dominant (white) border keeps revealing.
        doc = Document(4, 4)
        layer = doc.add_image(np.full((2, 2, 4), (255, 0, 0, 255), np.uint8), 'Red')
        layer.set_mask(np.array([[255, 0], [128, 255]], np.uint8))
        layer.mask_placement = fx.transform(2, 2, origin=(1, 0), sampling='Nearest')
        out = composite(doc)
        # Col 0 samples off-rect (white background); col 1 samples mask col 0.
        assert out[0, 0, 3] == 255 and out[1, 0, 3] == 255
        assert out[0, 1, 3] == 255 and out[1, 1, 3] == 128

    def test_unlinked_mask_background_is_dominant_border_gray(self):
        # Mask [255,0 / 0,255] shrunk onto a 1x1 rect at the layer's corner:
        # that single pixel averages to 128, and the dominant white border
        # keeps revealing everywhere else (upstream LayerMask.background).
        doc = Document(4, 4)
        layer = doc.add_image(np.full((2, 2, 4), (255, 0, 0, 255), np.uint8), 'Red')
        layer.set_mask(np.array([[255, 0], [0, 255]], np.uint8))
        layer.mask_placement = fx.transform(1, 1, origin=(0, 0), sampling='Nearest')
        out = composite(doc)
        assert out[0, 0, 3] == 128
        assert out[0, 1, 3] == 255 and out[1, 0, 3] == 255 and out[1, 1, 3] == 255

    def test_unlinked_mask_black_border_keeps_hiding_outside(self):
        doc = Document(4, 4)
        layer = doc.add_image(np.full((2, 2, 4), (255, 0, 0, 255), np.uint8), 'Red')
        layer.set_mask(np.zeros((2, 2), np.uint8))
        layer.mask_placement = fx.transform(1, 1, origin=(0, 0), sampling='Nearest')
        out = composite(doc)
        assert out[..., 3].max() == 0  # black border: hide everywhere on the layer

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
    def test_hidden_non_normal_blend_is_kept_but_not_drawn(self):
        doc = Document(1, 1)
        doc.add_image(fx.solid(1, 1, (1, 2, 3, 255)), 'Base')
        diff = doc.add_image(fx.solid(1, 1, (9, 9, 9, 255)), 'Diff')
        diff.blend_mode, diff.visible = 'Difference', False
        assert composite(doc)[0, 0].tolist() == [1, 2, 3, 255]


@pytest.mark.parametrize('mode, expected', [
    ('Darken', [51, 128, 51]),
    ('Color Burn', [0, 2, 0]),
    ('Linear Burn', [0, 1, 0]),
    ('Lighten', [204, 128, 204]),
    ('Color Dodge', [255, 255, 255]),
    ('Linear Dodge (Add)', [255, 255, 255]),
    ('Soft Light', [89, 128, 180]),
    ('Hard Light', [173, 128, 82]),
    ('Vivid Light', [128, 129, 128]),
    ('Linear Light', [204, 129, 51]),
    ('Pin Light', [153, 128, 102]),
    ('Hard Mix', [255, 255, 255]),
    ('Exclusion', [173, 127, 173]),
    ('Subtract', [0, 0, 153]),
    ('Divide', [64, 255, 255]),
    # Component modes (W3C/PDF non-separable formulas, as Core Image implements).
    # Cb=(0.2, 0.502, 0.8) and Cs=(0.8, 0.502, 0.2) share Sat=0.6, so Hue's
    # SetSat pass is the identity here and Hue matches Color exactly.
    ('Hue', [175, 99, 22]),
    # Saturation: SetSat(Cb, Sat(Cs))=Cb, SetLum back to its own luminance is a no-op.
    ('Saturation', [51, 128, 204]),
    ('Color', [175, 99, 22]),
    ('Luminosity', [80, 157, 233]),
])
def test_separable_native_blend_modes_render_real_pixels(mode, expected):
    doc = Document(1, 1)
    doc.add_image(np.array([[[51, 128, 204, 255]]], np.uint8), 'Base')
    layer = doc.add_image(np.array([[[204, 128, 51, 255]]], np.uint8), 'Top')
    layer.blend_mode = mode
    assert composite(doc)[0, 0].tolist() == expected + [255]


CI_BLEND_PROBES = [(0.0, 1.0), (1 / 255.0, 1.0), (0.3, 0.15), (0.3, 0.6),
                   (1.0, 0.0), (0.8, 0.25), (0.9, 0.4),
                   (0.3, 0.7), (0.95, 0.05), (0.85, 0.15), (0.9, 0.1)]


def _ci_blend_reference():
    """Capture real Core Image values, or fall back to the recorded bytes.

    Mirrors the cg_adjustment_fixture pattern: swiftc on macOS (dev machine /
    CI runner with Xcode), recorded file otherwise. Linux runners ship swiftc
    too, but without Core Image the fixture cannot build there.
    """
    import shutil
    import subprocess
    import sys
    import tempfile
    if sys.platform == 'darwin' and shutil.which('swiftc'):
        src = Path(__file__).parents[1] / 'scripts/dev/ci_blend_fixture.swift'
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / 'fixture'
            subprocess.run(['swiftc', '-O', str(src), '-o', str(exe)], check=True)
            out = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout
    else:
        out = (Path(__file__).parent / 'fixtures/ci_blend_reference.txt').read_text()
    reference = {}
    for line in out.splitlines():
        parts = line.split()
        reference[parts[0]] = [int(h, 16) for h in parts[1:]]
    return reference


@pytest.mark.parametrize('mode', ['Color Dodge', 'Color Burn', 'Vivid Light'])
def test_singular_corners_match_core_image(mode):
    # The singular corners of dodge/burn (and vivid, which inherits both) are
    # ambiguous in the W3C spec; real Core Image resolves them as
    # dodge(Cb=0,Cs=1)=0 / burn(Cb=1,Cs=0)=1, and the port follows CI, not
    # spec-literal. Hard-mix ties ride on vivid's float32 result and were
    # verified to match CI across the full Cb+Cs=1 line.
    from retouch.editor.composite import _blend
    ref = _ci_blend_reference()[mode.lower().replace(' ', '_')]
    for (cb, cs), expected in zip(CI_BLEND_PROBES, ref):
        got = _blend(np.float32(cb), np.float32(cs), mode)
        assert int(np.rint(got * 255)) == expected, (mode, cb, cs)


def test_hard_mix_ties_match_core_image():
    from retouch.editor.composite import _blend
    for cb, cs, expected in [(0.3, 0.7, 255), (0.95, 0.05, 0),
                             (0.85, 0.15, 255), (0.9, 0.1, 0)]:
        vivid = _blend(np.float32(cb), np.float32(cs), 'Vivid Light')
        assert int(np.rint(vivid * 255)) in (127, 128)  # f32 lands at the tie
        assert (vivid >= 0.5) == (expected == 255), (cb, cs)


def test_png_export_carries_resolution_and_working_profile(tmp_path):
    doc = Document(2, 2, resolution=300)
    doc.add_image(fx.solid(2, 2, (5, 6, 7, 128)), 'Base')
    path = export_png(doc, tmp_path / 'flat.png')
    with Image.open(path) as image:
        assert image.mode == 'RGBA'
        assert round(image.info['dpi'][0]) == 300
        assert image.info['icc_profile']
        np.testing.assert_array_equal(np.asarray(image), composite(doc))
    assert [p.name for p in tmp_path.iterdir()] == ['flat.png']


def test_high_bit_export_widens_exact_pixels_and_keeps_alpha_and_icc(tmp_path):
    doc = Document(2, 1, resolution=240)
    source = np.array([[(0, 128, 255, 64), (5, 6, 7, 255)]], np.uint8)
    doc.add_image(source, 'Base')
    path = export_png(doc, tmp_path / 'flat.tif', bit_depth=16)

    # Pillow exposes multichannel TIFF through an 8-bit convenience view; its
    # tags and OpenCV's unchanged decoder verify the real stored samples.
    with Image.open(path) as image:
        assert image.tag_v2[258] == (16, 16, 16, 16)
        assert image.tag_v2[282] == 240
        assert image.tag_v2[283] == 240
        assert image.tag_v2[296] == 2
        assert image.info['icc_profile']
    stored = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    expected = cv2.cvtColor(composite(doc), cv2.COLOR_RGBA2BGRA).astype(np.uint16) * 257
    np.testing.assert_array_equal(stored, expected)


class TestLayerEffectsPixels:
    """Exact native pixel math contracts for layer effects (stroke, drop shadow, colour overlay)."""

    def test_outside_stroke_square_ring_geometry(self):
        from retouch.editor.document import LayerEffects, StrokeEffect
        doc = Document(5, 5)
        layer = doc.add_image(np.array([[[255, 0, 0, 255]]], np.uint8), 'Pixel', origin=(2, 2))
        layer.effects = LayerEffects(stroke=StrokeEffect(size=1, red=0, green=0, blue=1,
                                                         opacity=1.0, inside=False))
        out = composite(doc)
        assert out[2, 2].tolist() == [255, 0, 0, 255]   # opaque layer covers its behind-stroke
        # Square reach of 1: the full 8-neighbour ring is stroke blue.
        for row, col in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)):
            assert out[row, col].tolist() == [0, 0, 255, 255], (row, col)
        assert out[0, 2].tolist() == [0, 0, 0, 0]        # reach 1 stops at one pixel
        assert out[0, 0].tolist() == [0, 0, 0, 0]        # corners of reach are covered, 2 away is not
        assert out[4, 4].tolist() == [0, 0, 0, 0]

    def test_inside_stroke_ring_keeps_interior_colour(self):
        from retouch.editor.document import LayerEffects, StrokeEffect
        doc = Document(5, 5)
        pixels = np.zeros((3, 3, 4), np.uint8)
        pixels[:] = (255, 255, 255, 255)
        layer = doc.add_image(pixels, 'Box', origin=(1, 1))
        layer.effects = LayerEffects(stroke=StrokeEffect(size=1, red=1, green=0, blue=0,
                                                         opacity=1.0, inside=True))
        out = composite(doc)
        assert out[2, 2].tolist() == [255, 255, 255, 255]  # interior survives the erosion
        for row, col in ((1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)):
            assert out[row, col].tolist() == [255, 0, 0, 255], (row, col)
        assert out[0, 2].tolist() == [0, 0, 0, 0]          # inside stroke never leaves the shape

    def test_stroke_half_opacity_lands_at_half_coverage(self):
        from retouch.editor.document import LayerEffects, StrokeEffect
        doc = Document(3, 3)
        layer = doc.add_image(np.array([[[0, 255, 0, 255]]], np.uint8), 'Pixel', origin=(1, 1))
        layer.effects = LayerEffects(stroke=StrokeEffect(size=1, red=0, green=0, blue=1,
                                                         opacity=0.5, inside=False))
        out = composite(doc)
        assert out[1, 1].tolist() == [0, 255, 0, 255]
        # coverage 0.5 on transparent ground: straight alpha keeps full blue, alpha rint(127.5) = 128.
        assert out[0, 1].tolist() == [0, 0, 255, 128]

    def test_color_overlay_analytic_source_over(self):
        from retouch.editor.document import LayerEffects, ColorOverlayEffect
        doc = Document(1, 1)
        layer = doc.add_image(np.array([[[200, 50, 50, 255]]], np.uint8), 'Base')
        layer.effects = LayerEffects(colorOverlay=ColorOverlayEffect(red=0.0, green=0.0, blue=1.0, opacity=0.4))
        out = composite(doc)
        # Analytic: red 200 * 0.6 + 0 * 0.4 = 120; green 50 * 0.6 = 30; blue 50 * 0.6 + 255 * 0.4 = 132
        assert out[0, 0].tolist() == [120, 30, 132, 255]

    def test_color_overlay_respects_unshadowed_layer_silhouette(self):
        from retouch.editor.document import LayerEffects, ColorOverlayEffect
        doc = Document(2, 2)
        pixels = np.zeros((2, 2, 4), np.uint8)
        pixels[0, 0] = [200, 50, 50, 255]  # only top-left opaque
        layer = doc.add_image(pixels, 'Base')
        layer.effects = LayerEffects(colorOverlay=ColorOverlayEffect(red=0.0, green=0.0, blue=1.0, opacity=0.4))
        out = composite(doc)
        assert out[0, 0].tolist() == [120, 30, 132, 255]
        # Transparent regions remain uncoloured
        assert (out[0, 1:] == 0).all()
        assert (out[1, :] == 0).all()

    def test_extreme_effect_margin_is_budget_checked(self):
        # A max-range shadow (distance 5000, blur 500 -> margin ~6502) on a
        # 200 MP canvas would allocate a multi-GB band window; the port-only
        # live budget (upstream LayerEffectsSurface.swift:47, 80 MP) refuses.
        from retouch.editor.document import LayerEffects, ShadowEffect
        doc = Document(20000, 10000)
        layer = doc.add_image(np.full((4, 4, 4), (255, 0, 0, 255), np.uint8),
                              'Tiny', origin=(0, 0))
        layer.effects = LayerEffects(shadow=ShadowEffect(angle=90, distance=5000,
                                                         blur=500, opacity=0.6))
        with pytest.raises(ValueError, match='live budget'):
            composite(doc)

    def test_drop_shadow_zero_blur_geometry_and_placement(self):
        from retouch.editor.document import LayerEffects, ShadowEffect
        doc = Document(5, 5)
        layer = doc.add_image(np.array([[[255, 0, 0, 255]]], np.uint8), 'Pixel', origin=(2, 2))
        # angle 90 = light from straight above -> shadow sits straight down (dy = +2, dx = 0)
        layer.effects = LayerEffects(shadow=ShadowEffect(angle=90, distance=2, blur=0, opacity=1.0,
                                                        red=0.0, green=0.0, blue=0.0))
        out = composite(doc)
        assert out[2, 2].tolist() == [255, 0, 0, 255]     # layer pixel unchanged
        assert out[4, 2].tolist() == [0, 0, 0, 255]       # shadow cast 2 px down
        assert out[0, 2].tolist() == [0, 0, 0, 0]         # no shadow above
        assert out[2, 0].tolist() == [0, 0, 0, 0]         # no shadow to sides

    def test_drop_shadow_fractional_offset_blends_not_snaps(self):
        from retouch.editor.document import LayerEffects, ShadowEffect
        doc = Document(4, 4)
        layer = doc.add_image(np.array([[[255, 255, 255, 255]]], np.uint8), 'P', origin=(1, 1))
        # angle 90, distance 1.5 -> dy = 1.5: upstream's effects_shift blends
        # bilinearly between the rows (MetalLayerEffects.swift:278-289), it never
        # snaps to whole pixels. Half coverage on each of the two rows below.
        layer.effects = LayerEffects(shadow=ShadowEffect(angle=90, distance=1.5, blur=0,
                                                          opacity=1.0, red=0, green=0, blue=0))
        out = composite(doc)
        assert out[1, 1].tolist() == [255, 255, 255, 255]  # layer pixel
        assert out[2, 1].tolist() == [0, 0, 0, 128]        # rint(255*0.5)
        assert out[3, 1].tolist() == [0, 0, 0, 128]
        assert out[0, 1].tolist() == [0, 0, 0, 0]         # nothing above

    def test_drop_shadow_sits_behind_layer_pixels(self):
        from retouch.editor.document import LayerEffects, ShadowEffect
        doc = Document(1, 1)
        layer = doc.add_image(np.array([[[255, 255, 255, 255]]], np.uint8), 'White')
        # Zero-distance black shadow directly under opaque white pixel
        layer.effects = LayerEffects(shadow=ShadowEffect(angle=90, distance=0, blur=0, opacity=1.0,
                                                        red=0.0, green=0.0, blue=0.0))
        out = composite(doc)
        assert out[0, 0].tolist() == [255, 255, 255, 255]  # white layer covers black shadow completely

    def test_translucent_layer_reveals_shadow(self):
        from retouch.editor.document import LayerEffects, ShadowEffect
        doc = Document(1, 1)
        layer = doc.add_image(np.array([[[255, 255, 255, 128]]], np.uint8), 'TranslucentWhite')
        layer.effects = LayerEffects(shadow=ShadowEffect(angle=90, distance=0, blur=0, opacity=1.0,
                                                        red=0.0, green=0.0, blue=0.0))
        out = composite(doc)
        # Analytic: a=128/255, out_a = a*(2 - a) = 48896/65025 -> 192/255; straight RGB = 255*255/382 -> 170
        assert out[0, 0].tolist() == [170, 170, 170, 192]

    def test_inner_shadow_zero_blur_geometry_and_placement(self):
        from retouch.editor.document import LayerEffects, InnerShadowEffect
        doc = Document(5, 5)
        px = np.full((3, 3, 4), (255, 255, 255, 255), np.uint8)
        layer = doc.add_image(px, 'Box', origin=(1, 1))
        # Same offset as a drop shadow (angle 90 -> dy +1): the shifted-blurred
        # coverage's complement, clipped inside the layer, darkens the top edge.
        layer.effects = LayerEffects(innerShadow=InnerShadowEffect(
            angle=90, distance=1, blur=0, opacity=1.0))
        out = composite(doc)
        for col in (1, 2, 3):
            assert out[1, col].tolist() == [0, 0, 0, 255], col
        assert out[2, 2].tolist() == [255, 255, 255, 255]  # interior untouched
        assert out[0, 2].tolist() == [0, 0, 0, 0]  # never leaves the shape

    def test_inner_shadow_half_opacity_analytic_source_over(self):
        from retouch.editor.document import LayerEffects, InnerShadowEffect
        doc = Document(5, 5)
        layer = doc.add_image(np.full((3, 3, 4), (255, 255, 255, 255), np.uint8), 'Box', origin=(1, 1))
        layer.effects = LayerEffects(innerShadow=InnerShadowEffect(
            angle=90, distance=1, blur=0, opacity=0.5))
        out = composite(doc)
        # Black 0.5 over opaque white: rint(255*0.5) = 128.
        assert out[1, 2].tolist() == [128, 128, 128, 255]
        assert out[2, 2].tolist() == [255, 255, 255, 255]

    def test_inner_shadow_zero_distance_full_silhouette_is_invisible(self):
        from retouch.editor.document import LayerEffects, InnerShadowEffect
        doc_plain = Document(3, 3)
        doc_plain.add_image(np.full((3, 3, 4), (10, 200, 30, 255), np.uint8), 'Box')
        doc_fx = Document(3, 3)
        layer = doc_fx.add_image(np.full((3, 3, 4), (10, 200, 30, 255), np.uint8), 'Box')
        # coverage = a * (1 - a) = 0 everywhere on an opaque layer.
        layer.effects = LayerEffects(innerShadow=InnerShadowEffect(distance=0, blur=0, opacity=1))
        np.testing.assert_array_equal(composite(doc_fx), composite(doc_plain))

    def test_outer_glow_size_zero_is_invisible_behind_layer(self):
        from retouch.editor.document import LayerEffects, OuterGlowEffect
        doc_plain = Document(3, 3)
        doc_plain.add_image(np.array([[[255, 0, 0, 255]]], np.uint8), 'P', origin=(1, 1))
        doc_fx = Document(3, 3)
        layer = doc_fx.add_image(np.array([[[255, 0, 0, 255]]], np.uint8), 'P', origin=(1, 1))
        # sigma 0 -> glow equals the silhouette; masked by (1 - shape) it vanishes.
        layer.effects = LayerEffects(outerGlow=OuterGlowEffect(size=0, opacity=1))
        np.testing.assert_array_equal(composite(doc_fx), composite(doc_plain))

    def test_outer_glow_blurred_halo_lands_outside_layer(self):
        from retouch.editor.document import LayerEffects, OuterGlowEffect
        doc = Document(5, 5)
        layer = doc.add_image(np.array([[[255, 0, 0, 255]]], np.uint8), 'P', origin=(2, 2))
        layer.effects = LayerEffects(outerGlow=OuterGlowEffect(
            size=2, opacity=1, red=1, green=0, blue=0))
        out = composite(doc)
        assert out[2, 2].tolist() == [255, 0, 0, 255]  # layer pixel unchanged
        assert out[1, 2, 3] > 0 and out[1, 2, :3].tolist() == [255, 0, 0]  # red halo
        assert out[0, 2, 3] < out[1, 2, 3]  # halo fades with distance

    def test_inner_glow_brightens_edges_from_inside(self):
        from retouch.editor.document import LayerEffects, InnerGlowEffect
        doc = Document(5, 5)
        layer = doc.add_image(np.full((3, 3, 4), (0, 0, 0, 255), np.uint8), 'Box', origin=(1, 1))
        layer.effects = LayerEffects(innerGlow=InnerGlowEffect(
            size=6, opacity=1, red=1, green=1, blue=0))
        out = composite(doc)
        # Corners sit closer to the boundary than the center: brighter there.
        assert out[1, 1, 0] > out[2, 2, 0] > 0
        assert out[2, 2, 2] == 0  # pure yellow glow, no blue channel

    def test_new_effect_kinds_in_band_invariance(self):
        from retouch.editor.document import (LayerEffects, InnerShadowEffect,
                                             OuterGlowEffect, InnerGlowEffect)
        doc = Document(16, 16)
        layer = doc.add_image(np.full((4, 4, 4), (200, 100, 50, 255), np.uint8), 'Box', origin=(6, 6))
        layer.effects = LayerEffects(
            innerShadow=InnerShadowEffect(angle=45, distance=2, blur=1, opacity=0.6),
            outerGlow=OuterGlowEffect(size=3, opacity=0.7),
            innerGlow=InnerGlowEffect(size=2, opacity=0.5))
        c16 = composite(doc, band_rows=16)
        c4 = composite(doc, band_rows=4)
        c2 = composite(doc, band_rows=2)
        np.testing.assert_array_equal(c4, c16)
        np.testing.assert_array_equal(c2, c16)

    def test_disabled_new_effect_kinds_render_as_none(self):
        from retouch.editor.document import (LayerEffects, InnerShadowEffect,
                                             OuterGlowEffect, InnerGlowEffect)
        doc_plain = Document(8, 8)
        doc_plain.add_image(np.full((4, 4, 4), (100, 150, 200, 255), np.uint8), 'Box', origin=(2, 2))
        doc_fx = Document(8, 8)
        layer = doc_fx.add_image(np.full((4, 4, 4), (100, 150, 200, 255), np.uint8), 'Box', origin=(2, 2))
        layer.effects = LayerEffects(
            innerShadow=InnerShadowEffect(distance=3, blur=2, opacity=0.9, enabled=False),
            outerGlow=OuterGlowEffect(size=3, opacity=0.9, enabled=False),
            innerGlow=InnerGlowEffect(size=3, opacity=0.9, enabled=False))
        np.testing.assert_array_equal(composite(doc_fx), composite(doc_plain))


        from retouch.editor.document import LayerEffects, StrokeEffect, ShadowEffect, ColorOverlayEffect
        doc = Document(16, 16)
        layer = doc.add_image(np.full((4, 4, 4), (200, 100, 50, 255), np.uint8), 'Box', origin=(6, 6))
        layer.effects = LayerEffects(
            stroke=StrokeEffect(size=2, red=1, green=1, blue=0, opacity=0.7),
            shadow=ShadowEffect(angle=90, distance=4, blur=2, opacity=0.8, red=0, green=0, blue=0),
            colorOverlay=ColorOverlayEffect(red=0.1, green=0.2, blue=0.3, opacity=0.3),
        )
        c16 = composite(doc, band_rows=16)  # single band
        c8 = composite(doc, band_rows=8)    # two bands (split across shadow at row 8)
        c2 = composite(doc, band_rows=2)    # fine bands
        np.testing.assert_array_equal(c8, c16)
        np.testing.assert_array_equal(c2, c16)

    def test_disabled_effects_produce_identical_pixels_to_none(self):
        from retouch.editor.document import LayerEffects, StrokeEffect, ShadowEffect, ColorOverlayEffect
        doc_none = Document(8, 8)
        doc_none.add_image(np.full((4, 4, 4), (100, 150, 200, 255), np.uint8), 'Box', origin=(2, 2))

        doc_disabled = Document(8, 8)
        layer = doc_disabled.add_image(np.full((4, 4, 4), (100, 150, 200, 255), np.uint8), 'Box', origin=(2, 2))
        layer.effects = LayerEffects(
            stroke=StrokeEffect(size=3, red=0, green=1, blue=0, opacity=1.0, enabled=False),
            shadow=ShadowEffect(angle=90, distance=3, blur=2, opacity=0.9, red=0, green=0, blue=0, enabled=False),
            colorOverlay=ColorOverlayEffect(red=1, green=0, blue=0, opacity=1.0, enabled=False),
        )
        np.testing.assert_array_equal(composite(doc_disabled), composite(doc_none))


class TestEffectsPlaneCache:
    def test_cache_is_band_invariant_and_matches_fallback(self):
        # The cached full-canvas plane sliced per band must reproduce the
        # per-band render bit for bit at every band_rows.
        import importlib
        _cm = importlib.import_module('retouch.editor.composite')
        from retouch.editor.document import LayerEffects, ShadowEffect, OuterGlowEffect
        doc = Document(90, 120)
        layer = doc.add_image(np.full((30, 40, 4), (255, 120, 0, 255), np.uint8),
                              'Fx', origin=(17.3, 21.9))
        layer.rotation = 17.3
        layer.effects = LayerEffects(
            shadow=ShadowEffect(angle=90, distance=5, blur=4, opacity=0.7),
            outerGlow=OuterGlowEffect(size=6, opacity=0.5, red=1, green=0, blue=0))
        ref = composite(doc, band_rows=256)
        for band_rows in (37, 7, 1):
            assert np.array_equal(ref, composite(doc, band_rows=band_rows))
        # Forced per-band fallback (cache admission zeroed) is identical.
        saved = _cm._MAX_EFFECTS_CACHE_PIXELS
        _cm._MAX_EFFECTS_CACHE_PIXELS = 0
        try:
            assert np.array_equal(ref, composite(doc, band_rows=7))
        finally:
            _cm._MAX_EFFECTS_CACHE_PIXELS = saved

    def test_guard_still_fires_on_fallback_path(self, monkeypatch):
        # The 80 MP window guard protects the per-band path the cache
        # replaced; a cache-admissible canvas forced to fallback must still
        # refuse an extreme margin window.
        import importlib
        _cm = importlib.import_module('retouch.editor.composite')
        from retouch.editor.document import LayerEffects, ShadowEffect
        monkeypatch.setattr(_cm, '_MAX_EFFECTS_CACHE_PIXELS', 0)
        monkeypatch.setattr(_cm, '_MAX_EFFECTS_WINDOW_PIXELS', 2_000_000)
        doc = Document(3000, 4000)  # 12 MP: would be cache-admissible
        layer = doc.add_image(np.full((4, 4, 4), (255, 0, 0, 255), np.uint8), 'Tiny')
        layer.effects = LayerEffects(shadow=ShadowEffect(angle=90, distance=5000,
                                                          blur=500, opacity=0.6))
        with pytest.raises(ValueError, match='live budget'):
            composite(doc)
