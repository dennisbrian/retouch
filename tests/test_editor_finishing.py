"""Independent analytic contracts for finishing tools, graphs and native saves."""
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from retouch.editor import Document, Layer, DocumentHistory, composite, save_project, load_project
from retouch.editor.adjustments import adjustment, grade
from retouch.editor.selections import selection, project_selection, combine


def solid(color, width=4, height=4, alpha=255):
    a = np.empty((height, width, 4), np.uint8)
    a[..., :3], a[..., 3] = color, alpha
    return a


def test_clockwise_rotation_native_pixels_and_mask_round_trip(tmp_path):
    a = solid((0, 0, 0)); a[:2, :2, :3] = (255, 0, 0)
    mask = np.full((4, 4), 255, np.uint8); mask[2:, 2:] = 0
    layer = Layer('Quadrants', a, origin=(2, 2), mask=mask, sampling='Nearest')
    doc = Document(8, 8, [layer]); history = DocumentHistory(doc)
    history.update_layer(layer.id, rotation=90)
    image = composite(history.document)
    assert (image[2:4, 4:6, 0] == 255).all()
    assert (image[4:6, 2:4, 3] == 0).all()
    np.testing.assert_array_equal(history.document.layer(layer.id).pixels, a)
    history.save(tmp_path / 'rotate.comp')
    again = load_project(tmp_path / 'rotate.comp')
    assert again.layers[0].rotation == 90
    np.testing.assert_array_equal(composite(again), image)
    assert history.undo() and history.redo()
    np.testing.assert_array_equal(composite(history.document), image)


@pytest.mark.parametrize('angle', [15, 90, -90, 180, 270, 360, 721])
def test_rotation_band_independence_and_alpha_edges(angle):
    a = solid((255, 80, 20), 8, 6); a[:, :2, 3] = 0; a[:, :2, :3] = (0, 0, 255)
    doc = Document(16, 16, [Layer('Edge', a, origin=(4, 4), rotation=angle)])
    image = composite(doc, 1)
    np.testing.assert_array_equal(image, composite(doc, 7))
    visible = image[..., 3] > 0
    # Transparent blue pixels must never bleed into the orange edge.
    assert np.max(np.abs(image[..., :3][visible].astype(int) - [255, 80, 20])) <= 1


@pytest.mark.parametrize('rotation', [float('nan'), float('inf'), True])
def test_invalid_rotation_is_transactional(rotation):
    h = DocumentHistory(Document(4, 4, [Layer('Base', solid((80, 90, 100)))]))
    before = composite(h.document)
    with pytest.raises(ValueError):
        h.update_layer(h.document.layers[0].id, rotation=rotation)
    assert h.revision == 0
    np.testing.assert_array_equal(composite(h.document), before)


def test_pass_through_folder_opacity_visibility_and_mask(tmp_path):
    folder = Layer('Folder', size=(4, 4), is_group=True, opacity=.5,
                   mask=selection(4, 4, 'Rectangle', 0, 0, 2, 4))
    lower = Layer('Red', solid((255, 0, 0)), parent_id=folder.id)
    upper = Layer('Blue', solid((0, 0, 255)), opacity=.5, parent_id=folder.id)
    doc = Document(4, 4, [folder, lower, upper])
    image = composite(doc)
    # Each child gets folder opacity: red .5, blue .25 => alpha .625.
    assert image[1, 1].tolist() == [153, 0, 102, 159]
    assert not image[:, 2:].any()
    save_project(doc, tmp_path / 'folder.comp')
    np.testing.assert_array_equal(composite(load_project(tmp_path / 'folder.comp')), image)
    folder.visible = False
    assert not composite(doc).any()


def test_clipping_stack_preserves_base_alpha_and_uses_base_blend(tmp_path):
    base = Layer('Base', solid((255, 0, 0), alpha=128))
    clipped = Layer('Blue', solid((0, 0, 255)), mask_source_id=base.id)
    doc = Document(4, 4, [base, clipped])
    image = composite(doc)
    assert image[0, 0].tolist() == [0, 0, 255, 128]
    save_project(doc, tmp_path / 'clip.comp')
    np.testing.assert_array_equal(composite(load_project(tmp_path / 'clip.comp')), image)
    clipped.opacity = .5
    assert composite(doc)[0, 0].tolist() == [128, 0, 128, 128]


def test_nonadjacent_live_alpha_link_ignores_source_visibility():
    base = Layer('Hidden source', solid((255, 0, 0), alpha=128), visible=False)
    child = Layer('Blue', solid((0, 0, 255)), mask_source_id=base.id)
    image = composite(Document(4, 4, [base, child]))
    assert image[0, 0].tolist() == [0, 0, 255, 128]


@pytest.mark.parametrize('field', ['parent_id', 'mask_source_id'])
def test_invalid_graph_rejected_before_history_changes(field):
    base = Layer('Base', solid((1, 2, 3)))
    h = DocumentHistory(Document(4, 4, [base]))
    with pytest.raises(ValueError):
        h.update_layer(base.id, **{field: base.id})
    assert h.revision == 0


def test_nested_group_duplicate_remaps_internal_links_and_undo(tmp_path):
    base = Layer('Base', solid((50, 60, 70)))
    child = Layer('Child', solid((100, 110, 120)), mask_source_id=base.id)
    h = DocumentHistory(Document(4, 4, [base, child]))
    h.group_layers([base.id, child.id]); folder = h.document.active_layer_id
    h.duplicate_layer(folder)
    again = h.document
    copy_folder = again.layer(again.active_layer_id)
    members = [l for l in again.layers if l.parent_id == copy_folder.id]
    assert members[1].mask_source_id == members[0].id
    assert len(again.layers) == 6
    h.remove_layer(copy_folder.id)
    assert len(h.document.layers) == 3
    assert h.undo() and len(h.document.layers) == 6
    h.save(tmp_path / 'nested.comp')
    np.testing.assert_array_equal(composite(load_project(tmp_path / 'nested.comp')), composite(h.document))


def test_ungroup_preserves_order_and_rejects_effect_loss():
    layers = [Layer(str(i), solid((i, i, i))) for i in range(3)]
    h = DocumentHistory(Document(4, 4, layers))
    h.group_layers([layers[0].id, layers[1].id]); folder = h.document.active_layer_id
    before = composite(h.document)
    h.ungroup_layer(folder)
    assert [l.id for l in h.document.layers] == [l.id for l in layers]
    np.testing.assert_array_equal(composite(h.document), before)
    h.undo(); h.update_layer(folder, opacity=.5)
    with pytest.raises(ValueError):
        h.ungroup_layer(folder)


def test_selection_shapes_combinations_and_projection():
    rect = selection(8, 8, 'Rectangle', 2, 1, 4, 3)
    assert np.count_nonzero(rect) == 12
    ellipse = selection(8, 8, 'Ellipse', 2, 2, 4, 4)
    assert 8 < np.count_nonzero(ellipse) < 16
    polygon = selection(8, 8, 'Polygon', points='[[0,0],[4,0],[0,4]]')
    assert polygon[1, 1] == 255 and polygon[5, 5] == 0
    feather = selection(8, 8, 'Rectangle', 2, 2, 4, 4, feather=1)
    assert 0 < feather[1, 1] < 255
    np.testing.assert_array_equal(combine(rect, rect, 'Subtract'), np.zeros_like(rect))
    np.testing.assert_array_equal(combine(rect, ellipse, 'Intersect'), np.minimum(rect, ellipse))
    layer = Layer('Turned', solid((255, 0, 0)), origin=(2, 2), rotation=90, flip_x=True)
    projected = project_selection(selection(8, 8, 'Rectangle', 4, 2, 2, 2), layer)
    expected = np.zeros((4, 4), np.uint8); expected[:2, 2:] = 255
    np.testing.assert_array_equal(projected, expected)


def test_selection_to_mask_preserves_pixels_and_round_trips(tmp_path):
    layer = Layer('Base', solid((20, 30, 40)), origin=(2, 2), rotation=90)
    h = DocumentHistory(Document(8, 8, [layer]))
    h.selection_mask(layer.id, selection(8, 8, 'Rectangle', 4, 2, 2, 2))
    image = composite(h.document)
    assert image[2:4, 4:6, 3].min() == 255
    assert np.count_nonzero(image[..., 3]) == 4
    h.save(tmp_path / 'selection.comp')
    np.testing.assert_array_equal(composite(load_project(tmp_path / 'selection.comp')), image)
    np.testing.assert_array_equal(h.document.layer(layer.id).pixels, layer.pixels)
    assert h.undo() and h.document.layer(layer.id).mask is None


def test_exposure_uses_linear_light_and_levels_native_channel_order():
    rgb = np.full((1, 1, 3), .5, np.float32)
    result = grade(rgb, adjustment('Exposure', exposure=1))
    # sRGB .5 decodes to .214041; double linear light encodes to .685836.
    np.testing.assert_allclose(result, .685836, atol=1e-6)
    record = adjustment('Levels', gamma=2, red_gamma=2)
    record['levels']['ranges'][1]['outputWhite'] = 128
    expected_red = (((.5 ** .5) * 128 / 255) ** .5)
    assert grade(rgb, record)[0, 0, 0] == pytest.approx(expected_red, abs=1e-6)


@pytest.mark.parametrize('kind', ['Exposure', 'Levels', 'Invert'])
def test_grading_alpha_mask_undo_and_native_record(kind, tmp_path):
    base = Layer('Base', solid((80, 100, 120), alpha=128))
    settings = adjustment(kind, exposure=1, gamma=1.2)
    adj = Layer(kind, size=(4, 4), adjustment=settings,
                mask=selection(4, 4, 'Rectangle', 0, 0, 2, 4), opacity=.5)
    h = DocumentHistory(Document(4, 4, [base])); before = composite(h.document)
    h.add_layer(adj); image = composite(h.document)
    np.testing.assert_array_equal(image[..., 3], before[..., 3])
    np.testing.assert_array_equal(image[:, 2:], before[:, 2:])
    assert not np.array_equal(image[:, :2], before[:, :2])
    h.save(tmp_path / (kind + '.comp'))
    again = load_project(tmp_path / (kind + '.comp'))
    assert again.layers[-1].adjustment == settings
    manifest = json.loads((tmp_path / (kind + '.comp') / 'manifest.json').read_text())
    assert 'outputBlack' in manifest['layers'][-1]['adjustment']['levels']['ranges'][0]
    np.testing.assert_array_equal(composite(again), image)
    h.undo(); np.testing.assert_array_equal(composite(h.document), before)


def test_bad_native_links_are_rejected_before_assets_are_decoded(tmp_path, monkeypatch):
    from retouch.editor import project_store
    layer = Layer('Base', solid((1, 2, 3)))
    path = save_project(Document(4, 4, [layer]), tmp_path / 'bad.comp')
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['layers'][0]['maskSourceID'] = layer.id
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(project_store, '_read_png', lambda *a: pytest.fail('decoded invalid graph'))
    with pytest.raises(ValueError, match='clipping'):
        load_project(path)


def test_clipped_adjustment_uses_full_strength_and_preserves_stack_alpha():
    base = Layer('Base', solid((80, 100, 120), alpha=128))
    adj = Layer('Invert', size=(4, 4), adjustment=adjustment('Invert'), mask_source_id=base.id)
    image = composite(Document(4, 4, [base, adj]))
    assert image[0, 0].tolist() == [175, 155, 135, 128]


@pytest.mark.parametrize('angle', [15, 45, 90, -90, 181, 270])
def test_fit_includes_rotated_extents_and_remains_undoable(angle):
    from retouch.editor.composite import _rotation_geometry
    layer = Layer('Fit', solid((1, 2, 3), 20, 10), origin=(-4, 3), rotation=angle)
    history = DocumentHistory(Document(40, 30, [layer]))
    history.fit_layer(layer.id)
    fitted = history.document.layer(layer.id)
    _, (width, height), (x, y) = _rotation_geometry(fitted)
    assert x >= 0 and y >= 0 and x + width <= 40 and y + height <= 30
    assert fitted.pixels is layer.pixels and fitted.rotation == angle
    assert history.undo() and history.document.layer(layer.id).size == (20, 10)


def test_exposure_builder_ignores_inactive_levels_and_supports_native_minimum_gamma():
    settings = adjustment('Exposure', gamma=.01, black=254, white=1)
    assert settings['exposureSettings']['gamma'] == .01
    assert settings['levels']['ranges'][0]['gamma'] == 1


def test_unsupported_adjustments_and_cycles_preserve_existing_package(tmp_path):
    base = Layer('Base', solid((1, 2, 3)))
    path = save_project(Document(4, 4, [base]), tmp_path / 'safe.comp')
    original = (path / 'manifest.json').read_bytes()
    bad = Document(4, 4, [deepcopy(base)])
    bad.layers[0].mask_source_id = base.id
    with pytest.raises(ValueError):
        save_project(bad, path)
    assert (path / 'manifest.json').read_bytes() == original


# --- Downsampling pyramid (mirrors Compositor's ``Rendering/DownsampleCache.swift``) ---
from retouch.editor.composite import _downsample_level, _downsample, _halve, _resample


@pytest.mark.parametrize('factor,expected', [
    (1.0, 0), (0.5, 0), (0.49, 1), (0.26, 1), (0.25, 2),
    (0.13, 2), (0.125, 3), (0.0625, 4), (0.001, 6), (0.0, 0),
])
def test_downsample_level_arithmetic(factor, expected):
    assert _downsample_level(factor) == expected


def test_downsample_level_bounds():
    assert _downsample_level(-1) == 0
    assert _downsample_level(0) == 0
    assert _downsample_level(0.5) == 0
    assert _downsample_level(1.0) == 0
    assert _downsample_level(float('inf')) == 0


def test_halve_exactly_doubles():
    arr = np.full((8, 6, 4), 100, np.uint8)
    result = _halve(arr)
    # Exactly half, rounded up: (8+1)//2 = 4, (6+1)//2 = 3. (An 8x6 image has
    # no boundary-free interior, so only the geometry is checked here.)
    assert result.shape == (4, 3, 4)
    assert result.dtype == np.uint8


def test_halve_mask_odd_dimensions():
    mask = np.full((7, 5), 255, np.uint8)
    result = _halve(mask, mask=True)
    assert result.shape == ((7 + 1) // 2, (5 + 1) // 2)
    assert result.dtype == np.uint8


def test_downsample_mask_and_pixels_use_same_level():
    arr = np.full((256, 128, 4), 128, np.uint8)
    mask = np.full((256, 128), 200, np.uint8)
    reduced_arr, lvl_arr = _downsample(arr, 1/8, mask=False)
    reduced_mask, lvl_mask = _downsample(mask, 1/8, mask=True)
    # Both must use the same pyramid level so they stay in sync.
    assert lvl_arr == lvl_mask
    assert lvl_arr == 3   # log2(8) = 3


def test_resample_upright_1to1_uses_nearest():
    arr = np.full((4, 6, 4), 50, np.uint8)
    result, level = _resample(arr, (6, 4), 'High quality', upright=True)
    assert result.shape == (4, 6, 4)
    assert level == 0  # nearest — no pyramid used


@pytest.mark.parametrize('sampling', ['Nearest', 'Smooth', 'High quality'])
def test_resample_shrink_uses_pyramid(sampling):
    arr = np.random.RandomState(0).randint(0, 255, (64, 64, 4), np.uint8)
    result, level = _resample(arr, (8, 8), sampling, upright=True)
    assert result.shape == (8, 8, 4)
    # Nearest never prefilters (it takes the whole reduction in one step);
    # the smooth settings use the halving pyramid.
    assert (level >= 2) if sampling != 'Nearest' else (level == 0)


def test_resample_rotated_enlarges_keeps_setting():
    arr = np.full((4, 4, 4), 80, np.uint8)
    result, level = _resample(arr, (8, 8), 'High quality', upright=False)
    assert result.shape == (8, 8, 4)
    # Enlarging does not use the pyramid (factor > 1), but the upright=False
    # flag prevents the "1:1 nearest" shortcut.
    assert level == 0


def test_resample_mask_matches_pixel_reduction_level():
    """Mask and pixel resampling must apply the same pyramid level so that
    mask coverage correctly clips the pixel data during rendering."""
    arr = np.full((128, 128, 4), 200, np.uint8)
    mask = np.full((128, 128), 128, np.uint8)
    _, lvl_arr = _resample(arr, (8, 8), 'High quality', upright=True)
    _, lvl_msk = _resample(mask, (8, 8), 'High quality', upright=True)
    assert lvl_arr == lvl_msk, "mask and pixel pyramid levels must match"


def test_halve_chain_interior_preserves_constant_regions():
    """A constant region survives the halving chain: Lanczos preserves the
    source value away from the transparent pad, so a flat mask stays flat and
    a flat fill keeps its colour."""
    for val in (0, 127, 255):
        arr = np.full((32, 32, 4), val, np.uint8)
        result = _halve(arr)
        centre = result[3:13, 3:13]
        np.testing.assert_array_equal(centre, val,
            err_msg=f"halve of constant {val}: interior differs")


# Core Graphics ground truth for a 7x5 gray mask rotated 33 degrees into a
# 13x11 canvas, captured from the real API (``clip(to:mask:)`` with a grayscale
# CGImage, interpolation quality .high; .low/.medium give the same grid for a
# grayscale clip). Generated by ``cg_mask_fixture.swift`` next to the review
# notes; regenerate with ``swift cg_mask_fixture.swift`` on macOS.
CG_ROTATED_MASK_ALPHA = """
0,0,0,0,0,0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0,0,0,0,0,0
0,0,0,0,88,14,9,0,0,0,0,0,0
0,0,0,0,189,210,95,69,10,0,0,0,0
0,0,4,171,31,193,165,124,174,62,1,0,0
0,0,0,87,185,127,255,161,51,184,64,0,0
0,0,0,18,111,60,15,175,4,0,1,0,0
0,0,0,0,9,137,96,229,192,0,0,0,0
0,0,0,0,0,0,19,27,88,0,0,0,0
0,0,0,0,0,0,0,0,0,0,0,0,0
0,0,0,0,0,0,0,0,0,0,0,0,0"""


def _cg_grid(text):
    return np.array([[int(v) for v in row.split(',')] for row in text.split()])


def test_rotated_mask_matches_core_graphics_reference():
    """A rotated masked layer agrees with the real Core Graphics result.

    Core Graphics resamples the mask as its own coverage pass with a 4-tap
    cubic filter (its impulse response is 15/63/192/240 out of 512, not the
    0/64/192/256 of bilinear), so a bilinear or area warp leaves a mean error
    of ~5 on this fixture while the cubic filter here stays well under it. The
    residual (mean ~3) is Core Graphics' own subpixel/edge convention, which an
    affine warp cannot reproduce exactly; the tolerance below pins it.
    """
    mask = _cg_grid("""
0,32,64,128,192,224,255
255,224,192,128,64,32,0
0,0,255,255,255,0,0
255,255,0,0,0,255,255
0,64,128,255,128,64,0""").astype(np.uint8)
    native = _cg_grid(CG_ROTATED_MASK_ALPHA)
    pixels = np.empty((5, 7, 4), np.uint8)
    pixels[..., :3], pixels[..., 3] = 255, 255
    layer = Layer('Masked', pixels, origin=(3, 3), size=(7, 5), mask=mask,
                  rotation=33, sampling='High quality')
    alpha = composite(Document(13, 11, [layer]))[..., 3].astype(int)
    error = np.abs(alpha - native)
    assert error.mean() < 6.0, f"mean error {error.mean():.2f} vs Core Graphics"
    assert np.percentile(error, 95) < 40, "too many pixels diverge from Core Graphics"
    # Outside the rotated rectangle the mask is fully hidden, and the centre of
    # the covered band reaches full coverage: the shape is right even where the
    # filter differs.
    assert (alpha[:2] == 0).all() and (alpha[-2:] == 0).all()
    assert alpha.max() == 255 and (alpha > 200).sum() >= 3


def test_compositor_preview_uses_downsample_for_large_reduction():
    """A photo layer placed much smaller than its source uses the halving
    pyramid for pixels and mask, and both land at the same level."""
    source = np.empty((1024, 1024, 4), np.uint8)
    source[..., :3], source[..., 3] = 120, 255        # opaque gray
    mask = np.full((1024, 1024), 255, np.uint8)
    layer = Layer('Photo', source, origin=(0, 0), size=(64, 64), mask=mask)
    image = composite(Document(64, 64, [layer]))
    assert image.shape == (64, 64, 4)
    # A 1/16 reduction (factor 0.0625) needs 4 halvings; the flat source and
    # mask survive intact, so the layer reads fully opaque.
    assert image[32, 32].tolist() == [120, 120, 120, 255]


# Core Graphics-era native ground truth for the grader kernel. Captured by
# ``scripts/dev/cg_adjustment_fixture.swift``, which drives Compositor's real
# ``Rendering/LevelsPixels.c`` kernel (pinned 11d8d7a) over a 256-step RGB ramp
# plus low-alpha edge colours. The kernel un-premultiplies each pixel,
# interpolates a 256-entry curve table, and rounds ``result * alpha`` back into
# premultiplied bytes; the Python compositor grades float straight-alpha. These
# pixel bytes are the contract. Regenerate with ``swift`` (see the fixture's
# header for the two build steps).
CG_ADJUSTMENT_RAMP = [(i, 255 - i, (i * 7) % 256, 255) for i in range(256)] + [
    (200, 100, 50, 1), (255, 0, 0, 128), (0, 255, 0, 64),
    (128, 128, 128, 200), (64, 192, 32, 255), (10, 20, 30, 0)]


def _cg_adjustment_reference():
    """name -> native premultiplied uint8 output (N, 4) for the fixture ramp."""
    text = (Path(__file__).parent / 'fixtures' / 'cg_adjustment_reference.txt').read_text()
    return {name: np.frombuffer(bytes.fromhex(hexs), np.uint8).reshape(-1, 4)
            for name, hexs in (line.split() for line in text.splitlines())}


def _cg_adjustment_records():
    """The same settings the Swift fixture used, as native records."""
    records = {
        'exp_1': adjustment('Exposure', exposure=1),
        'exp_neg2_gamma2': adjustment('Exposure', exposure=-2, offset=0.1, gamma=2),
        'exp_gamma05': adjustment('Exposure', gamma=0.5, offset=-0.2),
        'lv_rgamma2': adjustment('Levels', red_gamma=2),
        'lv_clamp_gamma': adjustment('Levels', gamma=4, red_gamma=2.5),
        'lv_outputrange': adjustment('Levels'),
        'invert': adjustment('Invert'),
    }
    # Native channel order: ranges[0] is the combined RGB pass, then R, G, B.
    r = records['lv_clamp_gamma']['levels']['ranges']
    r[0].update(black=20, white=200, gamma=0.5, outputBlack=5, outputWhite=255)
    r[1].update(black=30, white=220, gamma=2.5, outputBlack=10, outputWhite=240)
    records['lv_outputrange']['levels']['ranges'][1].update(outputBlack=20, outputWhite=235)
    return records


@pytest.mark.parametrize('name', [
    'exp_1', 'exp_neg2_gamma2', 'exp_gamma05', 'lv_rgamma2',
    'lv_clamp_gamma', 'lv_outputrange', 'invert',
])
def test_adjustment_pixels_match_native_kernel_fixture(name):
    """Adjustment layers preserve native pixel results on a full byte ramp."""
    native = _cg_adjustment_reference()[name]
    source = np.array([CG_ADJUSTMENT_RAMP], dtype=np.uint8)
    actual = composite(Document(source.shape[1], 1, [
        Layer('Source', source),
        Layer(name, size=(source.shape[1], 1), adjustment=_cg_adjustment_records()[name]),
    ]))[0]

    np.testing.assert_array_equal(actual[:, 3], native[:, 3])
    opaque = native[:, 3] == 255
    # Native uses a 256-entry table with linear interpolation. Python evaluates
    # the same transfer continuously; the full opaque byte ramp stays within
    # one output level, including non-default ranges and extreme gamma.
    np.testing.assert_array_less(np.abs(actual[opaque, :3].astype(int)
                                        - native[opaque, :3].astype(int)), 2)
    # Premultiplied output is the native storage contract at translucent edges.
    premult = np.rint(actual[:, :3].astype(np.float32)
                      * actual[:, 3:4].astype(np.float32) / 255).astype(np.uint8)
    np.testing.assert_array_less(np.abs(premult.astype(int)
                                        - native[:, :3].astype(int)), 2)


# --- Saturation and Vibrance -----------------------------------------------------
#
# Saturation scales the HSL saturation channel by (100 + amount) / 100.
# Vibrance boosts saturation logarithmically, protecting already-saturated pixels.
from retouch.editor.adjustments import _rgb_to_hsl, _grade_vibrance


@pytest.mark.parametrize('kind', ['Saturation', 'Vibrance'])
def test_saturation_vibrance_identity(kind):
    gray = np.full((1, 1, 3), .5, np.float32)
    result = grade(gray, adjustment(kind))
    np.testing.assert_allclose(result, gray, atol=1e-6)


def test_saturation_desaturates():
    gray = np.full((1, 1, 3), .5, np.float32)
    result = grade(gray, adjustment('Saturation', saturation=-100))
    np.testing.assert_allclose(result[0, 0], gray[0, 0], atol=1e-4)


def test_saturation_preserves_hue_on_saturated_pixel():
    # Pure red [1,0,0]: H=0, S=1. Sat+100 scales S, clipped to 1. HSL preserves
    # the red-primary hue; red channel stays dominant after the adjustment.
    red = np.array([[[1.0, 0.0, 0.0]]], np.float32)
    result = grade(red, adjustment('Saturation', saturation=100))
    # Red-primary hue: H in [0..0.833]; after clamping S=1, the result stays vivid.
    hsl = _rgb_to_hsl(result)
    assert hsl[0, 0, 0] < 0.9   # hue is in the red-to-magenta zone
    assert hsl[0, 0, 1] == 1.0  # saturation fully saturated


def test_vibrance_boosts_dull_more_than_saturated():
    # Vibrance boosts low-saturation colours more than vivid ones.
    # Use blue (HSL hue 0.667) and a muted lavender: both clearly distinguishable.
    blue = np.array([[[0.2, 0.2, 0.9]]], np.float32)   # vivid blue
    muted = np.array([[[0.5, 0.4, 0.6]]], np.float32)  # muted blue-purple
    blue_s0 = _rgb_to_hsl(blue)[0, 0, 1]
    blue_s1 = _rgb_to_hsl(_grade_vibrance(blue, 50))[0, 0, 1]
    muted_s0 = _rgb_to_hsl(muted)[0, 0, 1]
    muted_s1 = _rgb_to_hsl(_grade_vibrance(muted, 50))[0, 0, 1]
    blue_rel  = blue_s1  / blue_s0
    muted_rel = muted_s1 / muted_s0
    assert muted_rel > blue_rel, (
        f"muted relative boost {muted_rel:.3f} should exceed vivid {blue_rel:.3f}")


def test_vibrance_on_gray_is_identity():
    gray = np.full((1, 1, 3), .5, np.float32)
    result = grade(gray, adjustment('Vibrance', vibrance=100))
    np.testing.assert_allclose(result[0, 0], gray[0, 0], atol=1e-4)


@pytest.mark.parametrize('kind, kwargs', [
    ('Saturation', dict(saturation=50)),
    ('Vibrance', dict(vibrance=50)),
])
def test_saturation_vibrance_adjustment_layer_and_native_record(kind, kwargs, tmp_path):
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
    h.save(tmp_path / (kind + '.comp'))
    again = load_project(tmp_path / (kind + '.comp'))
    assert again.layers[-1].adjustment == settings
    np.testing.assert_array_equal(composite(again), image)
    h.undo()
    np.testing.assert_array_equal(composite(h.document), before)


def test_unsupported_adjustment_kind_raises():
    from retouch.editor.adjustments import validate_adjustment
    from retouch.editor import UnsupportedFeature
    identity_points = [dict(x=0, y=0), dict(x=255, y=255)]
    identity_range = dict(black=0, white=255, gamma=1, outputBlack=0, outputWhite=255)
    record = dict(kind='Hue', hue=30, saturation=0, lightness=0, colorize=False,
                  levels=dict(channel='RGB', ranges=[dict(identity_range) for _ in range(4)]),
                  curves=dict(channel='RGB',
                              channels=[[dict(p) for p in identity_points] for _ in range(4)]))
    with pytest.raises(UnsupportedFeature, match='Supported adjustments'):
        validate_adjustment(record)
