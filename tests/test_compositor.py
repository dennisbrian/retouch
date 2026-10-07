import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from retouch.compositor import export_project, archive_project


def test_handoff_reconstructs_result_and_keeps_base(tmp_path):
    base = np.full((9, 12, 3), 80, np.uint8)
    result = base.copy()
    result[2:5, 3:8] = [121, 102, 97]
    project = export_project(base, result, tmp_path / 'review.comp')
    manifest = json.loads((project / 'manifest.json').read_text())
    layers = manifest['layers']
    def pixels(layer):
        return np.array(Image.open(project / 'images' / layer['imageFile']))
    mask = np.array(Image.open(project / 'images' / layers[1]['maskFile']))
    reconstructed = np.where(mask[..., None] == 255, pixels(layers[1])[..., :3],
                             pixels(layers[0])[..., :3])
    np.testing.assert_array_equal(reconstructed, result)
    np.testing.assert_array_equal(pixels(layers[0])[..., :3], base)
    assert layers[2]['blendMode'] == 'Difference' and not layers[2]['isVisible']
    assert manifest['activeLayerID'] == layers[1]['id']
    for layer in layers:
        assert layer['imageFile'] == layer['id'] + '.png'
        assert layer['id'] == layer['id'].upper()
    archive = archive_project(project)
    with zipfile.ZipFile(archive) as z:
        assert 'review.comp/manifest.json' in z.namelist()
        assert all(n.startswith('review.comp/') for n in z.namelist())


def test_existing_manual_project_is_never_overwritten(tmp_path):
    p = tmp_path / 'review.comp'
    p.mkdir()
    (p / 'manual.txt').write_text('keep')
    image = np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(FileExistsError):
        export_project(image, image, p)
    assert (p / 'manual.txt').read_text() == 'keep'


@pytest.mark.parametrize('image', [np.zeros((2, 2, 3), np.float32),
                                  np.zeros((2, 2, 4), np.uint8),
                                  np.zeros((0, 2, 3), np.uint8)])
def test_rejects_ambiguous_precision_or_empty_canvas(tmp_path, image):
    with pytest.raises(ValueError):
        export_project(image, image, tmp_path / 'review.comp')
    assert not (tmp_path / 'review.comp').exists()


def test_rejects_different_coordinates(tmp_path):
    with pytest.raises(ValueError):
        export_project(np.zeros((2, 2, 3), np.uint8),
                       np.zeros((3, 2, 3), np.uint8), tmp_path / 'review.comp')


def test_failed_write_does_not_leave_a_partial_project(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(Image.Image, 'save', fail)
    image = np.zeros((2, 2, 3), np.uint8)
    with pytest.raises(OSError):
        export_project(image, image, tmp_path / 'review.comp')
    assert not (tmp_path / 'review.comp').exists()


def test_gui_handoff_respects_delivery_and_rebase_gates(tmp_path, monkeypatch):
    import gui
    image = np.zeros((2, 2, 3), np.uint8)
    monkeypatch.setattr(gui, 'advanced_delivery_decision',
                        lambda _: {'allowed': False, 'reason': 'preview_only'})
    update, status = gui.compositor_export_handler(image, image)
    assert update['value'] is None and 'preview_only' in status
    monkeypatch.setattr(gui, 'advanced_delivery_decision',
                        lambda _: {'allowed': True})
    update, status = gui.compositor_export_handler(
        image, image, rebase_review_required=True)
    assert update['value'] is None and 'Review' in status
    class Workspace:
        def request_workspace(self, label):
            return tmp_path
    monkeypatch.setattr(gui, '_workspace_for_request', lambda _: Workspace())
    update, status = gui.compositor_export_handler(image, image)
    assert update['visible'] and 'project ready' in status
    with zipfile.ZipFile(update['value']) as z:
        assert 'Retouch.comp/manifest.json' in z.namelist()
