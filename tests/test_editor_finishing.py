"""Independent analytic contracts for finishing tools, graphs and native saves."""
import json
from copy import deepcopy

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
