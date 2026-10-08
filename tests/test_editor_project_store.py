"""Validated .comp reader/writer (retouch/editor/project_store.py).

Rejection cases follow Compositor's ``ProjectTests`` and ``LayerMaskTests``
at 11d8d7a (future version, path traversal, corrupt manifest, missing image or
mask, masks on an older schema, failed save keeps the previous package); MIT,
Copyright (c) 2026 Wonder Assembly LLC for the translated cases.
"""
import json

import numpy as np
import pytest
from PIL import Image

from retouch.compositor import export_project
from retouch.editor import (
    Document,
    Layer,
    ProjectError,
    UnsupportedFeature,
    composite,
    load_project,
    save_project,
)
from tests import compositor_fixtures as fx


def _manifest(path):
    return json.loads((path / 'manifest.json').read_text())


def _rewrite(path, edit):
    manifest = _manifest(path)
    edit(manifest)
    (path / 'manifest.json').write_text(json.dumps(manifest))


class TestRead:
    def test_documented_minimal_project(self, tmp_path):
        pixels = fx.solid(5, 3, (1, 2, 3, 255))
        path = fx.write_package(tmp_path / 'min.comp', 5, 3,
                                [fx.layer_record(0, 'Background', 5, 3)],
                                images={0: pixels})
        doc = load_project(path)
        assert (doc.width, doc.height, doc.resolution) == (5, 3, 72)
        assert doc.document_id == fx.DOC_ID and doc.active_layer_id == fx.IDS[0]
        np.testing.assert_array_equal(doc.layers[0].pixels, pixels)
        assert not doc.layers[0].pixels.flags.writeable

    def test_rgb_png_reads_as_opaque_rgba(self, tmp_path):
        path = fx.write_package(tmp_path / 'rgb.comp', 2, 2,
                                [fx.layer_record(0, 'RGB', 2, 2)],
                                images={0: fx.solid(2, 2, (4, 5, 6, 255))[..., :3].copy()})
        assert load_project(path).layers[0].pixels[0, 0].tolist() == [4, 5, 6, 255]

    def test_version_1_without_appearance_fields(self, tmp_path):
        record = fx.layer_record(0, 'Old', 2, 2)
        del record['opacity'], record['blendMode'], record['isGroup']
        path = fx.write_package(tmp_path / 'v1.comp', 2, 2, [record],
                                images={0: fx.solid(2, 2, (9, 9, 9, 255))}, version=1)
        layer = load_project(path).layers[0]
        assert (layer.opacity, layer.blend_mode) == (1.0, 'Normal')

    def test_our_handoff_bridge_reconstructs_the_result(self, tmp_path):
        rng = np.random.default_rng(3)
        base = rng.integers(0, 256, (21, 34, 3), dtype=np.uint8)
        result = base.copy()
        result[4:15, 6:20] = rng.integers(0, 256, (11, 14, 3), dtype=np.uint8)
        doc = load_project(export_project(base, result, tmp_path / 'r.comp'))
        assert [layer.blend_mode for layer in doc.layers] == ['Normal', 'Normal', 'Difference']
        np.testing.assert_array_equal(composite(doc)[..., :3], result)
        doc.layer(doc.active_layer_id).visible = False
        np.testing.assert_array_equal(composite(doc)[..., :3], base)


class TestRejections:
    @pytest.fixture
    def project(self, tmp_path):
        return fx.write_package(tmp_path / 'p.comp', 2, 2,
                                [fx.layer_record(0, 'Layer', 2, 2)],
                                images={0: fx.solid(2, 2, (1, 1, 1, 255))})

    def test_future_version(self, project):
        _rewrite(project, lambda m: m.update(version=42))
        with pytest.raises(ProjectError, match='42'):
            load_project(project)

    def test_path_traversal(self, project):
        _rewrite(project, lambda m: m['layers'][0].update(imageFile='../../outside.png'))
        with pytest.raises(ProjectError):
            load_project(project)

    def test_corrupt_manifest(self, project):
        (project / 'manifest.json').write_text('not json')
        with pytest.raises(ProjectError):
            load_project(project)

    def test_missing_image(self, project):
        (project / 'images' / (fx.IDS[0] + '.png')).unlink()
        with pytest.raises(ProjectError):
            load_project(project)

    def test_symlinked_image(self, project, tmp_path):
        image = project / 'images' / (fx.IDS[0] + '.png')
        outside = tmp_path / 'outside.png'
        image.rename(outside)
        image.symlink_to(outside)
        with pytest.raises(ProjectError):
            load_project(project)

    def test_missing_mask(self, tmp_path):
        path = fx.red_with_mask(tmp_path / 'm.comp')
        (path / 'images' / (fx.IDS[0] + '.mask.png')).unlink()
        with pytest.raises(ProjectError):
            load_project(path)

    def test_mask_needs_version_4(self, tmp_path):
        path = fx.red_with_mask(tmp_path / 'm.comp')
        _rewrite(path, lambda m: m.update(version=3))
        with pytest.raises(ProjectError):
            load_project(path)

    def test_mask_with_alpha_is_invalid(self, tmp_path):
        path = fx.red_with_mask(tmp_path / 'm.comp')
        Image.new('LA', (2, 2)).save(path / 'images' / (fx.IDS[0] + '.mask.png'))
        with pytest.raises(ProjectError, match='grayscale'):
            load_project(path)

    def test_16_bit_png_is_refused(self, project):
        Image.new('I;16', (2, 2)).save(project / 'images' / (fx.IDS[0] + '.png'))
        with pytest.raises(ProjectError, match='8-bit'):
            load_project(project)

    def test_image_over_pixel_budget_fails_before_decoding(self, project, monkeypatch):
        import retouch.editor.project_store as store
        monkeypatch.setattr(store, 'MAX_SURFACE_PIXELS', 3)
        monkeypatch.setattr(store.Image, 'open', None)   # never reached
        with pytest.raises(ProjectError, match='size limits'):
            load_project(project)

    def test_opacity_needs_version_3(self, project):
        _rewrite(project, lambda m: (m.update(version=2), m['layers'][0].update(opacity=0.5)))
        with pytest.raises(ProjectError):
            load_project(project)

    def test_misnamed_image_file(self, project):
        _rewrite(project, lambda m: m['layers'][0].update(imageFile='other.png'))
        with pytest.raises(ProjectError):
            load_project(project)

    def test_unknown_active_layer(self, project):
        _rewrite(project, lambda m: m.update(activeLayerID=fx.IDS[2]))
        with pytest.raises(ProjectError):
            load_project(project)


class TestUnsupportedIsExplicit:
    @pytest.mark.parametrize('extra', [
        {'parentID': fx.IDS[2]},
        {'maskSourceID': fx.IDS[2]},
        {'effects': {'stroke': {}}},
        {'adjustment': {'kind': 'Invert'}},
        {'futureField': 1},
    ])
    def test_unmodelled_layer_fields(self, tmp_path, extra):
        path = fx.write_package(tmp_path / 'u.comp', 2, 2,
                                [fx.layer_record(0, 'Layer', 2, 2, **extra)],
                                images={0: fx.solid(2, 2, (1, 1, 1, 255))})
        with pytest.raises(UnsupportedFeature):
            load_project(path)

    def test_folder(self, tmp_path):
        record = fx.layer_record(0, 'Folder', 2, 2, image=False, isGroup=True)
        path = fx.write_package(tmp_path / 'g.comp', 2, 2, [record])
        with pytest.raises(UnsupportedFeature, match='folders'):
            load_project(path)

    @pytest.mark.parametrize('geometry', [
        {'rotation': 15}, {'origin': [0.5, 0]}])
    def test_transformed_layers(self, tmp_path, geometry):
        record = fx.layer_record(0, 'Layer', 2, 2)
        record['transform'].update(geometry)
        path = fx.write_package(tmp_path / 't.comp', 2, 2, [record],
                                images={0: fx.solid(2, 2, (1, 1, 1, 255))})
        with pytest.raises(UnsupportedFeature):
            load_project(path)


class TestWrite:
    def _document(self):
        doc = Document(6, 4, resolution=240)
        base = doc.add_image(fx.solid(6, 4, (10, 20, 30, 255)), 'Base')
        top = doc.add_layer(Layer(name='Patch', pixels=fx.solid(2, 2, (200, 0, 0, 180)),
                                  origin=(3, 1), flip_x=True, opacity=0.4))
        top.set_mask(np.array([[255, 0], [64, 255]], np.uint8), enabled=False)
        doc.add_layer(Layer(name='Blank', size=(6, 4)))
        doc.active_layer_id = base.id
        return doc

    def test_round_trip(self, tmp_path):
        doc = self._document()
        path = save_project(doc, tmp_path / 'out.comp')
        again = load_project(path)
        assert _manifest(path)['version'] == 11
        assert [layer.name for layer in again.layers] == ['Base', 'Patch', 'Blank']
        patch = again.layers[1]
        assert (patch.origin, patch.flip_x, patch.opacity, patch.mask_enabled) == (
            (3, 1), True, 0.4, False)
        np.testing.assert_array_equal(patch.mask, doc.layers[1].mask)
        assert again.active_layer_id == doc.active_layer_id
        assert again.resolution == 240
        np.testing.assert_array_equal(composite(again), composite(doc))
        assert sorted(p.name for p in tmp_path.iterdir()) == ['out.comp']

    def test_overwrite_drops_removed_assets(self, tmp_path):
        doc = self._document()
        path = save_project(doc, tmp_path / 'out.comp')
        doc.remove_layer(doc.layers[1].id)
        save_project(doc, path)
        names = sorted(p.name for p in (path / 'images').iterdir())
        assert names == [doc.layers[0].id + '.png']

    def test_failed_save_keeps_previous_package(self, tmp_path, monkeypatch):
        doc = self._document()
        path = save_project(doc, tmp_path / 'out.comp')
        before = (path / 'manifest.json').read_bytes()
        doc.layers[0].opacity = 7          # invalid: refused before writing
        with pytest.raises(ValueError):
            save_project(doc, path)
        doc.layers[0].opacity = 1
        import retouch.editor.project_store as store

        def broken(*_args, **_kwargs):
            raise OSError('disk full')
        monkeypatch.setattr(store.Image.Image, 'save', broken)
        with pytest.raises(OSError):
            save_project(doc, path)
        assert (path / 'manifest.json').read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ['out.comp']

    def test_refuses_to_replace_a_non_project(self, tmp_path):
        target = tmp_path / 'notes.comp'
        target.mkdir()
        (target / 'keep.txt').write_text('mine')
        with pytest.raises(FileExistsError):
            save_project(self._document(), target)
        assert (target / 'keep.txt').read_text() == 'mine'

    @pytest.mark.parametrize('manifest', [
        'not json', '[]', '{"application": "other"}',
        '{"format": "com.compositor.project", "version": 11}',
    ])
    def test_foreign_manifest_does_not_authorize_replacement(self, tmp_path, manifest):
        target = tmp_path / 'notes.comp'
        target.mkdir()
        (target / 'manifest.json').write_text(manifest)
        (target / 'keep.txt').write_text('mine')
        with pytest.raises(FileExistsError):
            save_project(self._document(), target)
        assert (target / 'manifest.json').read_text() == manifest
        assert (target / 'keep.txt').read_text() == 'mine'
        assert list(tmp_path.iterdir()) == [target]

    def test_symlinked_manifest_does_not_authorize_replacement(self, tmp_path):
        source = save_project(self._document(), tmp_path / 'source.comp')
        target = tmp_path / 'notes.comp'
        target.mkdir()
        (target / 'manifest.json').symlink_to(source / 'manifest.json')
        (target / 'keep.txt').write_text('mine')
        with pytest.raises(FileExistsError):
            save_project(self._document(), target)
        assert (target / 'manifest.json').is_symlink()
        assert (target / 'keep.txt').read_text() == 'mine'
        load_project(source)

    @pytest.mark.parametrize('field, value', [
        ('document_id', 'invalid'), ('document_id', None),
        ('width', 1.5), ('height', True),
        ('guides', [{'invalid': True}]),
        ('guides', [{'id': fx.IDS[0], 'axis': 'diagonal', 'position': 0}]),
        ('guides', [{'id': fx.IDS[0], 'axis': 'vertical', 'position': 0}] * 2),
    ])
    @pytest.mark.parametrize('overwrite', [False, True])
    def test_invalid_document_is_rejected_before_writing(self, tmp_path, field, value, overwrite):
        target = tmp_path / 'out.comp'
        if overwrite:
            save_project(self._document(), target)
            before = {p.relative_to(target): p.read_bytes()
                      for p in target.rglob('*') if p.is_file()}
        doc = self._document()
        setattr(doc, field, value)
        with pytest.raises(ValueError):
            save_project(doc, target)
        if overwrite:
            after = {p.relative_to(target): p.read_bytes()
                     for p in target.rglob('*') if p.is_file()}
            assert after == before
            load_project(target)
        else:
            assert not target.exists()
        assert list(tmp_path.iterdir()) == ([target] if overwrite else [])

    def test_guides_round_trip(self, tmp_path):
        doc = self._document()
        doc.guides = [{'id': fx.IDS[0], 'axis': 'horizontal', 'position': 1.5}]
        again = load_project(save_project(doc, tmp_path / 'guided.comp'))
        assert again.guides == doc.guides

    def test_text_metadata_round_trips_until_pixels_change(self, tmp_path):
        doc = Document(2, 2)
        layer = doc.add_image(fx.solid(2, 2, (1, 2, 3, 255)), 'Title')
        layer.text = {'content': 'Hi', 'fontName': 'Helvetica', 'fontSize': 12}
        again = load_project(save_project(doc, tmp_path / 't.comp'))
        assert again.layers[0].text == layer.text
        again.layers[0].replace_pixels(fx.solid(2, 2, (9, 9, 9, 255)))
        assert again.layers[0].text is None

    def test_pixels_are_immutable(self):
        source = fx.solid(2, 2, (1, 2, 3, 255))
        layer = Layer(name='L', pixels=source)
        source[:] = 0
        assert layer.pixels[0, 0].tolist() == [1, 2, 3, 255]
        with pytest.raises(ValueError):
            layer.pixels[0, 0, 0] = 5
