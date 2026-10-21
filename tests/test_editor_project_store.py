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
import time
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
        # Crash leftovers are aged; a fresh dir could be a concurrent save.
        old = time.time() - 7200
        os.utime(stale, (old, old))
        reopened = load_project(path)
        assert reopened.layers[0].name == 'Green'
        assert not stale.exists()

    def test_fresh_saving_staging_dir_is_a_live_save(self, tmp_path):
        # A concurrent writer's staging dir matches the sweep prefix for its
        # whole write window; sweeping by name alone deletes a live save.
        doc = Document(2, 2)
        doc.add_image(fx.solid(2, 2, (0, 255, 0, 255)), 'Green')
        path = tmp_path / 'p.comp'
        save_project(doc, path)
        live = tmp_path / '.p.comp.saving-live'
        (live / 'images').mkdir(parents=True)
        (live / 'manifest.json').write_text('{}')
        reopened = load_project(path)
        assert reopened.layers[0].name == 'Green'
        assert live.exists()  # untouched — only aged dirs are swept

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
        with pytest.raises(ProjectError, match='document pixel budget'):
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


class TestFormats12To14:
    """Read-side rehearsal of upstream c70a12f's versions 12-14 (writes stay
    at 11): 16-bit layer PNGs (12), OpenEXR layer assets (13), shape
    smoothing (14). Upstream citations: ProjectStore.swift's
    ``depth == 16 && !isMask && manifest.version >= 12`` load gate, its
    ``imageFile == "<uuid>.exr"`` naming valid from 13, and validate()'s
    ``smoothing == true`` requiring 14; MIT, Copyright (c) 2026 Wonder
    Assembly LLC for the mirrored cases."""

    def _wide_layer(self, path, version=12):
        """A 2x2 layer whose 16-bit PNG carries 0x0000, 0x0101, 0x8080 and
        0xFFFF in the red channel (one per pixel): widened by >>8 that is
        0, 1, 128, 255. Alpha stays 255 so the straight-alpha decode is
        observable (a premultiplying decoder zeroes the low-alpha colours)."""
        record = fx.layer_record(0, 'Deep', 2, 2)
        path = fx.write_package(path, 2, 2, [record], version=version)
        reds = [0x0000, 0x0101, 0x8080, 0xFFFF]
        values = []
        for red in reds:
            values += [red, 0, 0, 0xFFFF]
        fx.png16(path / 'images' / (fx.IDS[0] + '.png'), 2, 2, values)
        return path

    def test_v12_16_bit_layer_loads_widened(self, tmp_path):
        doc = load_project(self._wide_layer(tmp_path / 'deep.comp'))
        # Widened with the upstream dither (hash_noise(x, y, draw)), not a
        # plain >>8: 0x8080 dithers to 129 on even draw draws, 0x0101 to 0/1
        # per (x, y, channel). Compare against the vectorised replica.
        from retouch.editor.project_store import _dither_quantize16
        rgba16 = np.array(
            [[0x0000, 0x0101], [0x8080, 0xFFFF]], '>u2')[:, :, None]
        rgba16 = np.concatenate([rgba16, np.zeros((2, 2, 3), '>u2')], axis=2)
        rgba16[..., 3] = 0xFFFF
        alpha16 = rgba16[..., 3:4].astype(np.float64)
        premult = np.rint(rgba16.astype(np.float64)
                          * (alpha16 / 65535.0)).astype(np.uint16)
        want = _dither_quantize16(premult)[..., 0]
        assert (doc.layers[0].pixels[..., 0] == want).all()
        assert (doc.layers[0].pixels[..., 3] == 255).all()

    def test_16_bit_layer_in_v11_refused(self, tmp_path):
        path = self._wide_layer(tmp_path / 'old.comp', version=11)
        with pytest.raises(ProjectError, match='8-bit'):
            load_project(path)

    def test_16_bit_mask_refused_even_on_v12(self, tmp_path):
        record = fx.layer_record(0, 'Red', 2, 2,
                                 maskFile=fx.IDS[0] + '.mask.png')
        path = fx.write_package(tmp_path / 'm.comp', 2, 2, [record],
                                images={0: fx.solid(2, 2, (255, 0, 0, 255))},
                                version=12)
        fx.png16(path / 'images' / (fx.IDS[0] + '.mask.png'), 2, 2, [0xFFFF] * 16)
        with pytest.raises(ProjectError, match='8-bit'):
            load_project(path)

    def test_v12_loaded_project_round_trips_as_v11_8_bit(self, tmp_path):
        doc = load_project(self._wide_layer(tmp_path / 'deep.comp'))
        path = tmp_path / 'saved.comp'
        save_project(doc, path)
        manifest = _manifest(path)
        assert manifest['version'] == 11
        with Image.open(path / 'images' / (fx.IDS[0] + '.png')) as image:
            assert image.mode == 'RGBA' and image.info.get('bits', 8) != 16
        # Re-saved pixels match the widened load exactly.
        assert (load_project(path).layers[0].pixels == doc.layers[0].pixels).all()

    def test_v13_exr_layer_refused_cleanly(self, tmp_path):
        # EXR is the one v13 feature refused outright: HDR half-float light
        # brighter than white cannot be represented in the port's 8-bit sRGB
        # compositing, and decoding via a plugin EXR codec is unsafe.
        record = fx.layer_record(0, 'HDR', 2, 2)
        record['imageFile'] = fx.IDS[0] + '.exr'
        path = fx.write_package(tmp_path / 'hdr.comp', 2, 2, [record], version=13)
        (path / 'images' / (fx.IDS[0] + '.exr')).write_bytes(b'\x76\x2f\x31\x01')
        with pytest.raises(UnsupportedFeature,
                           match=r'HDR \(EXR\) layers from format version 13'):
            load_project(path)

    def test_exr_layer_in_v12_refused_before_the_feature_check(self, tmp_path):
        # A v12 file cannot name an EXR at all (upstream validate(): the
        # .exr suffix is only legal from version 13), so the version gate
        # fires as a plain invalid, not as the HDR refusal.
        record = fx.layer_record(0, 'HDR', 2, 2)
        record['imageFile'] = fx.IDS[0] + '.exr'
        path = fx.write_package(tmp_path / 'old.exr.comp', 2, 2, [record], version=12)
        (path / 'images' / (fx.IDS[0] + '.exr')).write_bytes(b'\x76\x2f\x31\x01')
        with pytest.raises(ProjectError, match='format version 13'):
            load_project(path)

    def _smooth_shape(self, path, version):
        # As rasterized by upstream: a shape layer's pixels carry the
        # smoothing; the style only redraws it when the layer is scaled.
        shape = {'kind': 'Rectangle', 'red': 1, 'green': 0, 'blue': 0,
                 'cornerRadius': 4, 'smoothing': True}
        record = fx.layer_record(0, 'Smoothed rect', 2, 2, shape=shape)
        return fx.write_package(path, 2, 2, [record],
                                images={0: fx.solid(2, 2, (255, 0, 0, 255))},
                                version=version)

    def test_v14_smoothing_shape_loads_and_is_ignored(self, tmp_path):
        doc = load_project(self._smooth_shape(tmp_path / 's14.comp', version=14))
        layer = doc.layers[0]
        assert layer.shape is not None and layer.shape['kind'] == 'Rectangle'
        assert 'smoothing' not in layer.shape     # ignored, deliberately
        assert layer.shape['cornerRadius'] == 4
        # The style rides along; the pixels composite unchanged.
        np.testing.assert_array_equal(
            composite(doc)[..., :3], fx.solid(2, 2, (255, 0, 0, 255))[..., :3])

    def test_v14_smoothing_shape_saves_as_v11_without_the_flag(self, tmp_path):
        doc = load_project(self._smooth_shape(tmp_path / 's14.comp', version=14))
        path = save_project(doc, tmp_path / 'saved.comp')
        assert _manifest(path)['version'] == 11
        assert 'smoothing' not in _manifest(path)['layers'][0]['shape']
        again = load_project(path)
        assert again.layers[0].shape['cornerRadius'] == 4

    def test_smoothing_in_v13_refused(self, tmp_path):
        path = self._smooth_shape(tmp_path / 's13.comp', version=13)
        with pytest.raises(ProjectError, match='format version 14'):
            load_project(path)

    def test_smoothing_false_is_legal_below_v14(self, tmp_path):
        shape = {'kind': 'Rectangle', 'red': 1, 'green': 0, 'blue': 0,
                 'cornerRadius': 4, 'smoothing': False}
        record = fx.layer_record(0, 'Rect', 2, 2, shape=shape)
        path = fx.write_package(tmp_path / 'plain.comp', 2, 2, [record],
                                images={0: fx.solid(2, 2, (255, 0, 0, 255))},
                                version=13)
        assert load_project(path).layers[0].shape['cornerRadius'] == 4

    def test_v15_refused(self, tmp_path):
        path = fx.write_package(tmp_path / 'future.comp', 2, 2,
                                [fx.layer_record(0, 'Layer', 2, 2)],
                                images={0: fx.solid(2, 2, (1, 1, 1, 255))},
                                version=15)
        with pytest.raises(ProjectError, match='supported 1-14'):
            load_project(path)


def test_known_file_callback_does_not_deadlock_under_the_lock(tmp_path):
    """A registered array freed while the registry lock is held (garbage
    collection can run a weakref callback at any allocation, including inside
    _remember) must not deadlock on the lock."""
    import threading
    from retouch.editor import project_store as ps

    path = tmp_path / 'm.png'
    Image.fromarray(np.zeros((2, 2), np.uint8)).save(path)

    def run():
        arr = np.zeros((2, 2), np.uint8)
        ps._remember(arr, path)
        with ps._KNOWN_LOCK:
            del arr  # refcount drop runs the weakref callback right here

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(5)
    assert not worker.is_alive()


class TestRecursionGuard:
    """Deeply nested JSON must fail as a clean refusal, never a RecursionError.

    The stdlib json decoder recurses per nesting level; the original handlers
    only caught ValueError-family errors, so a crafted manifest crashed
    callers — worst case the recovery listing that runs at editor startup.
    """

    def test_deeply_nested_manifest_is_refused(self, tmp_path):
        path = fx.write_package(tmp_path / 'deep.comp', 5, 3,
                                [fx.layer_record(0, 'Background', 5, 3)],
                                images={0: fx.solid(5, 3, (1, 2, 3, 255))})
        (path / 'manifest.json').write_text('{"x": ' + '[' * 1500 + ']' * 1500 + '}')
        with pytest.raises(ProjectError, match='manifest is not JSON'):
            load_project(path)

    def test_deeply_nested_recipe_extension_is_refused(self, tmp_path):
        path = fx.write_package(tmp_path / 'recdeep.comp', 5, 3,
                                [fx.layer_record(0, 'Background', 5, 3)],
                                images={0: fx.solid(5, 3, (1, 2, 3, 255))})
        (path / 'retouch-recipes.json').write_text(
            '{"x": ' + '[' * 1500 + ']' * 1500 + '}')
        with pytest.raises(ProjectError, match='recipe extension'):
            load_project(path)

    def test_saving_over_deeply_nested_existing_refuses_cleanly(self, tmp_path):
        path = fx.write_package(tmp_path / 'over.comp', 5, 3,
                                [fx.layer_record(0, 'Background', 5, 3)],
                                images={0: fx.solid(5, 3, (1, 2, 3, 255))})
        (path / 'manifest.json').write_text('{"x": ' + '[' * 1500 + ']' * 1500 + '}')
        doc = Document(5, 3)
        doc.add_image(fx.solid(5, 3, (9, 9, 9, 255)), 'Base')
        with pytest.raises(FileExistsError):
            save_project(doc, path)
