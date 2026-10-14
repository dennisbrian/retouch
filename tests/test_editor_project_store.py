"""Validated .comp reader/writer (retouch/editor/project_store.py).

Rejection cases follow Compositor's ``ProjectTests`` and ``LayerMaskTests``
at 11d8d7a (future version, path traversal, corrupt manifest, missing image or
mask, masks on an older schema, failed save keeps the previous package); MIT,
Copyright (c) 2026 Wonder Assembly LLC for the translated cases.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import os
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

    def test_unlinked_mask_round_trips(self, tmp_path):
        mask = np.zeros((2, 2), np.uint8)
        mask[0, :] = 255  # reveal the mask rect's top row; border is white
        extra = {'maskFile': fx.IDS[0] + '.mask.png', 'maskLinked': False,
                 'maskPlacement': fx.transform(2, 2, origin=(1, 0), sampling='Nearest')}
        record = fx.layer_record(0, 'Red', 2, 2, **extra)
        record['transform'] = fx.transform(2, 2, origin=(0, 0), sampling='Nearest')
        path = fx.write_package(tmp_path / 'u.comp', 4, 4, [record],
                                images={0: fx.solid(2, 2, (255, 0, 0, 255))},
                                masks={0: mask})
        layer = load_project(path).layers[0]
        assert layer.mask_placement is not None
        assert tuple(layer.mask_placement['origin']) == (1.0, 0.0)
        assert layer.mask_linked is False
        # The mask's reveal row lands on the layer's top row; the layer's
        # bottom-right pixel samples the mask's hidden row. Its white border
        # keeps revealing beyond the mask's rect.
        out = composite(load_project(path))
        np.testing.assert_array_equal(
            out[:2, :2, 3], np.array([[255, 255], [255, 0]], np.uint8))

    def test_mask_linked_defaults_to_true_when_absent(self, tmp_path):
        path = fx.red_with_mask(tmp_path / 'm.comp')
        layer = load_project(path).layers[0]
        assert layer.mask_linked is True and layer.mask_placement is None

    def test_save_reopen_preserves_unlinked_mask(self, tmp_path):
        doc = Document(4, 4)
        layer = doc.add_image(fx.solid(2, 2, (255, 0, 0, 255)), 'Red', origin=(0, 0))
        layer.set_mask(np.array([[0, 255], [0, 255]], np.uint8))
        layer.mask_placement = fx.transform(2, 2, origin=(1, 0), sampling='Nearest')
        layer.mask_linked = False
        path = tmp_path / 'saved.comp'
        save_project(doc, path)
        reopened = load_project(path)
        r = reopened.layers[0]
        assert r.mask_placement == layer.mask_placement
        assert r.mask_linked is False
        np.testing.assert_array_equal(composite(reopened), composite(doc))

    def test_mask_placement_needs_mask_file(self, tmp_path):
        record = fx.layer_record(0, 'Red', 2, 2,
                                 maskPlacement=fx.transform(2, 2))
        path = fx.write_package(tmp_path / 'np.comp', 4, 4, [record],
                                images={0: fx.solid(2, 2, (255, 0, 0, 255))})
        with pytest.raises(ProjectError):
            load_project(path)

    def test_mask_placement_needs_full_transform(self, tmp_path):
        extra = {'maskFile': fx.IDS[0] + '.mask.png',
                 'maskPlacement': {'origin': [0, 0], 'size': [2, 2]}}
        record = fx.layer_record(0, 'Red', 2, 2, **extra)
        path = fx.write_package(tmp_path / 'nt.comp', 4, 4, [record],
                                images={0: fx.solid(2, 2, (255, 0, 0, 255))},
                                masks={0: fx.MASK_2X2})
        with pytest.raises(ProjectError):
            load_project(path)

    def test_mask_linked_false_without_mask_file_refused(self, tmp_path):
        record = fx.layer_record(0, 'Red', 2, 2, maskLinked=False)
        path = fx.write_package(tmp_path / 'nl.comp', 4, 4, [record],
                                images={0: fx.solid(2, 2, (255, 0, 0, 255))})
        with pytest.raises(ProjectError):
            load_project(path)

    def test_unlinked_folder_mask_refused(self, tmp_path):
        record = fx.layer_record(0, 'Folder', 2, 2, image=False, isGroup=True,
                                 maskFile=fx.IDS[0] + '.mask.png',
                                 maskPlacement=fx.transform(2, 2))
        path = fx.write_package(tmp_path / 'f.comp', 4, 4, [record],
                                masks={0: fx.MASK_2X2})
        with pytest.raises(UnsupportedFeature):
            load_project(path)

    def test_oversized_mask_placement_refused_before_decode(self, tmp_path):
        # _place_mask would resample the mask to the placement size first
        # (upstream never materializes it), so a crafted 300000^2 placement
        # must die at validation before any PNG decode or allocation.
        record = fx.layer_record(0, 'L', 2, 2,
                                  maskFile=fx.IDS[0] + '.mask.png',
                                  maskPlacement=fx.transform(300000, 300000))
        path = fx.write_package(tmp_path / 'huge.comp', 4, 4, [record],
                                masks={0: fx.MASK_2X2})
        with pytest.raises((ValueError, UnsupportedFeature)):
            load_project(path)

    def test_crash_between_swap_renames_recovers_old_package(self, tmp_path):
        # Simulate a process killed between _swap_in's two renames: the old
        # package sits retired under .replaced-<token>, target missing.
        doc = Document(2, 2)
        doc.add_image(fx.solid(2, 2, (255, 0, 0, 255)), 'Red')
        path = tmp_path / 'p.comp'
        save_project(doc, path)
        doc.layers[0].name = 'Renamed'
        save_project(doc, path)  # retired sibling is rmtree'd on success
        retired = tmp_path / '.p.comp.replaced-crash'
        os.rename(path, retired)
        # A load of the missing target sweeps the interrupted save aside and
        # restores the retired (older but complete) package.
        recovered = load_project(path)
        assert recovered.layers[0].name == 'Renamed'
        assert not retired.exists()

    def test_stale_saving_staging_dirs_are_swept(self, tmp_path):
        doc = Document(2, 2)
        doc.add_image(fx.solid(2, 2, (0, 255, 0, 255)), 'Green')
        path = tmp_path / 'p.comp'
        save_project(doc, path)
        stale = tmp_path / '.p.comp.saving-deadbeef'
        (stale / 'images').mkdir(parents=True)
        (stale / 'manifest.json').write_text('{}')
        reopened = load_project(path)
        assert reopened.layers[0].name == 'Green'
        assert not stale.exists()

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
        {'effects': {'gradientOverlay': {'opacity': 0.5}}},
        {'adjustment': {'kind': 'Gaussian Blur'}},
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
        assert load_project(path).layers[0].is_group

    def test_fractional_origin_loads_and_renders(self, tmp_path):
        record = fx.layer_record(0, 'Layer', 2, 2)
        record['transform']['origin'] = [0.5, 0]
        path = fx.write_package(tmp_path / 'f.comp', 4, 2, [record],
                                images={0: fx.solid(2, 2, (1, 1, 1, 255))})
        doc = load_project(path)
        assert doc.layers[0].origin == (0.5, 0.0)
        out = composite(doc)
        # 2px solid layer at x=0.5: col 0 half-covered, col 1 full, col 2 half.
        assert out[0, 0, 3] == 128 and out[0, 1, 3] == 255 and out[0, 2, 3] == 128

    @pytest.mark.parametrize('geometry', [
        {'size': [2.5, 2]}])
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
        monkeypatch.setattr(store.shutil, 'copyfile', broken)
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


class TestFastSave:
    """Unchanged assets are copied, not re-encoded; changed ones are encoded."""

    def _count_encodes(self, monkeypatch):
        import retouch.editor.project_store as store
        calls = []
        original = store.Image.Image.save

        def counting(image, path, *args, **kwargs):
            calls.append(Path(path).name)
            return original(image, path, *args, **kwargs)
        monkeypatch.setattr(store.Image.Image, 'save', counting)
        return calls

    def _document(self):
        doc = Document(6, 4)
        doc.add_image(fx.solid(6, 4, (10, 20, 30, 255)), 'Base')
        top = doc.add_layer(Layer(name='Top', pixels=fx.solid(2, 2, (200, 0, 0, 180))))
        top.set_mask(np.array([[255, 0], [64, 255]], np.uint8))
        return doc

    def test_resave_copies_unchanged_assets(self, tmp_path, monkeypatch):
        doc = self._document()
        path = save_project(doc, tmp_path / 'out.comp')
        calls = self._count_encodes(monkeypatch)
        doc.layers[1].opacity = 0.5
        save_project(doc, path)
        assert calls == []
        again = load_project(path)
        assert again.layers[1].opacity == 0.5
        np.testing.assert_array_equal(composite(again), composite(doc))

    def test_only_the_changed_layer_is_encoded(self, tmp_path, monkeypatch):
        doc = self._document()
        path = save_project(doc, tmp_path / 'out.comp')
        calls = self._count_encodes(monkeypatch)
        doc.layers[1].set_mask(np.full((2, 2), 7, np.uint8))
        save_project(doc, tmp_path / 'copy.comp')       # save-as copies too
        assert calls == [doc.layers[1].id + '.mask.png']
        np.testing.assert_array_equal(load_project(tmp_path / 'copy.comp').layers[1].mask, 7)
        np.testing.assert_array_equal(load_project(path).layers[1].mask,
                                      [[255, 0], [64, 255]])

    def test_opened_project_resaves_without_encoding(self, tmp_path, monkeypatch):
        path = save_project(self._document(), tmp_path / 'out.comp')
        doc = load_project(path)
        calls = self._count_encodes(monkeypatch)
        save_project(doc, path)
        assert calls == []
        np.testing.assert_array_equal(composite(load_project(path)), composite(doc))

    def test_file_changed_on_disk_is_re_encoded(self, tmp_path, monkeypatch):
        doc = self._document()
        path = save_project(doc, tmp_path / 'out.comp')
        base = path / 'images' / (doc.layers[0].id + '.png')
        Image.fromarray(fx.solid(6, 4, (0, 0, 0, 255)), 'RGBA').save(base)
        calls = self._count_encodes(monkeypatch)
        save_project(doc, tmp_path / 'copy.comp')
        assert calls == [base.name]
        np.testing.assert_array_equal(load_project(tmp_path / 'copy.comp').layers[0].pixels,
                                      doc.layers[0].pixels)

    def test_copy_failure_falls_back_to_encoding(self, tmp_path, monkeypatch):
        import retouch.editor.project_store as store
        doc = self._document()
        save_project(doc, tmp_path / 'out.comp')

        def broken(*_args, **_kwargs):
            raise OSError('gone')
        monkeypatch.setattr(store.shutil, 'copyfile', broken)
        path = save_project(doc, tmp_path / 'copy.comp')
        np.testing.assert_array_equal(composite(load_project(path)), composite(doc))


class TestLayerEffects:
    """Effects (drop shadow, color overlay) round-trip and are refused for unsupported kinds."""

    def test_drop_shadow_round_trips(self, tmp_path):
        record = fx.layer_record(0, 'Layer', 4, 4,
                                 effects={'shadow': {'angle': 135, 'distance': 10,
                                                     'blur': 5, 'red': 0, 'green': 0,
                                                     'blue': 0, 'opacity': 0.4}})
        path = fx.write_package(tmp_path / 'shadow.comp', 4, 4, [record],
                               images={0: fx.solid(4, 4, (80, 100, 120, 255))})
        doc = load_project(path)
        layer = doc.layers[0]
        assert layer.effects is not None
        assert layer.effects.shadow is not None
        assert layer.effects.shadow.angle == 135
        assert layer.effects.shadow.distance == 10
        assert layer.effects.shadow.blur == 5
        assert layer.effects.shadow.red == 0
        assert layer.effects.shadow.green == 0
        assert layer.effects.shadow.blue == 0
        assert layer.effects.shadow.opacity == 0.4

    def test_color_overlay_round_trips(self, tmp_path):
        record = fx.layer_record(0, 'Layer', 4, 4,
                                 effects={'colorOverlay': {'red': 1, 'green': 0.5,
                                                          'blue': 0, 'opacity': 0.5}})
        path = fx.write_package(tmp_path / 'overlay.comp', 4, 4, [record],
                               images={0: fx.solid(4, 4, (80, 100, 120, 255))})
        doc = load_project(path)
        layer = doc.layers[0]
        assert layer.effects is not None
        assert layer.effects.colorOverlay is not None
        assert layer.effects.colorOverlay.red == 1
        assert layer.effects.colorOverlay.green == 0.5
        assert layer.effects.colorOverlay.blue == 0
        assert layer.effects.colorOverlay.opacity == 0.5

    def test_shadow_and_overlay_combined_round_trip(self, tmp_path):
        record = fx.layer_record(0, 'Layer', 4, 4,
                                 effects={'shadow': {'angle': 90, 'distance': 5,
                                                     'blur': 3, 'red': 0, 'green': 0,
                                                     'blue': 0, 'opacity': 0.3},
                                         'colorOverlay': {'red': 0, 'green': 0,
                                                          'blue': 1, 'opacity': 0.2}})
        path = fx.write_package(tmp_path / 'both.comp', 4, 4, [record],
                               images={0: fx.solid(4, 4, (128, 128, 128, 255))})
        doc = load_project(path)
        layer = doc.layers[0]
        assert layer.effects.shadow is not None
        assert layer.effects.colorOverlay is not None

    def test_unsupported_effect_kind_raises(self, tmp_path):
        # gradientOverlay stays unmodelled: the UnsupportedFeature path stays covered.
        record = fx.layer_record(0, 'Layer', 4, 4,
                                 effects={'gradientOverlay': {'opacity': 0.5}})
        path = fx.write_package(tmp_path / 'bad.comp', 4, 4, [record],
                               images={0: fx.solid(4, 4, (128, 128, 128, 255))})
        with pytest.raises(UnsupportedFeature):
            load_project(path)

    def test_save_and_reopen_preserves_layer_effects(self, tmp_path):
        from retouch.editor.document import (LayerEffects, StrokeEffect, ShadowEffect,
                                             ColorOverlayEffect)
        doc = Document(8, 8)
        layer = doc.add_image(fx.solid(4, 4, (50, 100, 150, 255)), 'Box', origin=(2, 2))
        layer.effects = LayerEffects(
            stroke=StrokeEffect(size=7, red=0.2, green=0.4, blue=0.6, opacity=0.8,
                                inside=True, enabled=False),
            shadow=ShadowEffect(angle=45, distance=3, blur=1, red=0.1, green=0.2, blue=0.3, opacity=0.7),
            colorOverlay=ColorOverlayEffect(red=0.8, green=0.2, blue=0.1, opacity=0.5, enabled=False),
        )
        path = tmp_path / 'saved.comp'
        save_project(doc, path)
        reopened = load_project(path)
        assert reopened.layers[0].effects == layer.effects
        assert reopened.layers[0].effects.stroke.enabled is False
        assert reopened.layers[0].effects.colorOverlay.enabled is False
        np.testing.assert_array_equal(composite(reopened), composite(doc))

    def test_stroke_record_round_trips_with_defaults(self, tmp_path):
        record = fx.layer_record(0, 'Layer', 4, 4,
                                 effects={'stroke': {'size': 4}})
        path = fx.write_package(tmp_path / 'stroke.comp', 4, 4, [record],
                                images={0: fx.solid(4, 4, (128, 128, 128, 255))})
        stroke = load_project(path).layers[0].effects.stroke
        assert (stroke.size, stroke.inside, stroke.enabled) == (4.0, False, True)
        assert (stroke.opacity, stroke.red, stroke.green, stroke.blue) == (1.0, 0, 0, 0)

    def test_inner_shadow_round_trips_with_defaults(self, tmp_path):
        record = fx.layer_record(0, 'Layer', 4, 4,
                                 effects={'innerShadow': {'distance': 5}})
        path = fx.write_package(tmp_path / 'inner.comp', 4, 4, [record],
                                images={0: fx.solid(4, 4, (128, 128, 128, 255))})
        inner = load_project(path).layers[0].effects.innerShadow
        # Upstream InnerShadowEffect defaults: angle 90, blur 10, black, opacity 0.5.
        assert (inner.angle, inner.distance, inner.blur, inner.enabled) == (90.0, 5.0, 10.0, True)
        assert (inner.opacity, inner.red, inner.green, inner.blue) == (0.5, 0, 0, 0)

    @pytest.mark.parametrize('kind, expected', [
        ('outerGlow', (3.0, 1.0, 1.0, 1.0, 0.75)),   # upstream rest defaults: white, 0.75
        ('innerGlow', (3.0, 1.0, 1.0, 1.0, 0.75)),
    ])
    def test_glow_round_trips_with_defaults(self, tmp_path, kind, expected):
        record = fx.layer_record(0, 'Layer', 4, 4, effects={kind: {'size': 3}})
        path = fx.write_package(tmp_path / f'{kind}.comp', 4, 4, [record],
                                images={0: fx.solid(4, 4, (128, 128, 128, 255))})
        glow = getattr(load_project(path).layers[0].effects, kind)
        assert (glow.size, glow.red, glow.green, glow.blue, glow.opacity) == expected

    def test_inner_shadow_and_glow_values_round_trip(self, tmp_path):
        record = fx.layer_record(0, 'Layer', 4, 4, effects={
            'innerShadow': {'angle': 135, 'distance': 10, 'blur': 5,
                            'red': 0.1, 'green': 0.2, 'blue': 0.3, 'opacity': 0.4},
            'outerGlow': {'size': 3, 'red': 1, 'green': 0.5, 'blue': 0, 'opacity': 0.6},
            'innerGlow': {'size': 7, 'red': 0, 'green': 1, 'blue': 0.5, 'opacity': 0.8},
        })
        path = fx.write_package(tmp_path / 'fx.comp', 4, 4, [record],
                                images={0: fx.solid(4, 4, (80, 100, 120, 255))})
        effects = load_project(path).layers[0].effects
        assert (effects.innerShadow.angle, effects.innerShadow.distance) == (135.0, 10.0)
        assert (effects.innerShadow.red, effects.innerShadow.opacity) == (0.1, 0.4)
        assert (effects.outerGlow.size, effects.outerGlow.green, effects.outerGlow.opacity) == (3.0, 0.5, 0.6)
        assert (effects.innerGlow.size, effects.innerGlow.blue) == (7.0, 0.5)

    def test_save_reopen_preserves_new_effect_kinds(self, tmp_path):
        from retouch.editor.document import (LayerEffects, InnerShadowEffect,
                                             OuterGlowEffect, InnerGlowEffect)
        doc = Document(8, 8)
        layer = doc.add_image(fx.solid(4, 4, (50, 100, 150, 255)), 'Box', origin=(2, 2))
        layer.effects = LayerEffects(
            innerShadow=InnerShadowEffect(angle=45, distance=3, blur=2, red=0.1,
                                          green=0.2, blue=0.3, opacity=0.7, enabled=False),
            outerGlow=OuterGlowEffect(size=5, red=1, green=0.5, blue=0, opacity=0.6),
            innerGlow=InnerGlowEffect(size=4, red=0, green=1, blue=0.5, opacity=0.8),
        )
        path = tmp_path / 'saved.comp'
        save_project(doc, path)
        reopened = load_project(path)
        assert reopened.layers[0].effects == layer.effects
        assert reopened.layers[0].effects.innerShadow.enabled is False
        np.testing.assert_array_equal(composite(reopened), composite(doc))

    @pytest.mark.parametrize('kind, bads', [
        ('innerShadow', [{'angle': 361}, {'distance': -1}, {'blur': 501},
                         {'opacity': 1.5}, {'red': -0.1}, {'enabled': 1}]),
        ('outerGlow', [{'size': 501}, {'size': -1}, {'opacity': 1.5},
                       {'green': 2}, {'enabled': 1}]),
        ('innerGlow', [{'size': 501}, {'opacity': -0.1}, {'blue': -1}]),
    ])
    def test_invalid_new_effect_records_refused(self, tmp_path, kind, bads):
        base = {'innerShadow': {'distance': 5}, 'outerGlow': {'size': 3},
                'innerGlow': {'size': 3}}[kind]
        for i, bad in enumerate(bads):
            record = fx.layer_record(0, 'Layer', 4, 4,
                                     effects={kind: dict(base, **bad)})
            path = fx.write_package(tmp_path / f'bad{i}.comp', 4, 4, [record],
                                    images={0: fx.solid(4, 4, (1, 1, 1, 255))})
            with pytest.raises(ProjectError):
                load_project(path)

    def test_unknown_new_effect_sub_keys_refused(self, tmp_path):
        bad = fx.layer_record(0, 'Layer', 4, 4,
                              effects={'innerShadow': {'angle': 90, 'futureKey': 1}})
        path = fx.write_package(tmp_path / 'bad_inner.comp', 4, 4, [bad],
                                images={0: fx.solid(4, 4, (1, 1, 1, 255))})
        with pytest.raises(ProjectError, match='unknown innerShadow keys'):
            load_project(path)

    def test_new_effect_kinds_on_folder_refused(self, tmp_path):
        folder_rec = fx.layer_record(0, 'Folder', 4, 4, image=False, isGroup=True,
                                     effects={'innerShadow': {'distance': 5}})
        path = fx.write_package(tmp_path / 'folder.comp', 4, 4, [folder_rec])
        with pytest.raises(ProjectError, match='effects'):
            load_project(path)

    @pytest.mark.parametrize('bad', [
        {'size': 501}, {'size': -1}, {'opacity': 1.5}, {'red': -0.1},
        {'inside': 'yes'}, {'enabled': 1}, {'futureKey': 1},
    ])
    def test_invalid_stroke_records_refused(self, tmp_path, bad):
        record = fx.layer_record(0, 'Layer', 4, 4,
                                 effects={'stroke': dict({'size': 4}, **bad)})
        path = fx.write_package(tmp_path / 'bad_stroke.comp', 4, 4, [record],
                                images={0: fx.solid(4, 4, (1, 1, 1, 255))})
        with pytest.raises(ProjectError):
            load_project(path)

    def test_effects_on_folder_or_adjustment_refused(self, tmp_path):
        folder_rec = fx.layer_record(0, 'Folder', 4, 4, image=False, isGroup=True,
                                     effects={'shadow': {'angle': 90, 'distance': 5, 'blur': 3,
                                                         'red': 0, 'green': 0, 'blue': 0, 'opacity': 0.5}})
        path = fx.write_package(tmp_path / 'folder.comp', 4, 4, [folder_rec])
        with pytest.raises(ProjectError, match='effects'):
            load_project(path)

    def test_unknown_sub_keys_refused(self, tmp_path):
        bad_shadow = fx.layer_record(0, 'Layer', 4, 4,
                                     effects={'shadow': {'angle': 90, 'futureKey': 1}})
        path = fx.write_package(tmp_path / 'bad_shadow.comp', 4, 4, [bad_shadow],
                               images={0: fx.solid(4, 4, (1, 1, 1, 255))})
        with pytest.raises(ProjectError, match='unknown shadow keys'):
            load_project(path)
