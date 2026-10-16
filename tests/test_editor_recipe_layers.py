"""Recipe regeneration persists original pixels and preserves manual edits."""
import json

import numpy as np
import pytest
from PIL import Image

from retouch.editor import Document, DocumentHistory, composite, load_project
from retouch.editor.retouch_jobs import prepare_job, render_worker, attach_result


class Engine:
    def process(self, bgr, recipe, quality, **overrides):
        assert bgr[0, 0].tolist() == [80, 40, 10]  # always the original composite
        return np.full_like(bgr, [50 + int(overrides.get('smooth', 0)), 100, 150])


def setup(tmp_path):
    doc = Document(8, 6)
    doc.add_image(np.full((6, 8, 3), [10, 40, 80], np.uint8), 'Original')
    history = DocumentHistory(doc, saved=True)
    job = prepare_job(history, tmp_path, 'natural', 70)
    render_worker(job.directory, Engine)
    job.status = 'completed'
    lid = attach_result(history, job)
    return history, lid


def test_regenerate_after_save_preserves_properties_and_undo(tmp_path):
    history, lid = setup(tmp_path)
    mask = np.arange(48, dtype=np.uint8).reshape(6, 8) * 5
    history.set_mask(lid, mask)
    history.update_layer(lid, name='My face layer', origin=(1.5, -2), rotation=15,
                         flip_x=True, opacity=.35, size=(16, 12))
    target = tmp_path / 'recipe.comp'
    history.save(target)
    manifest = json.loads((target / 'manifest.json').read_text())
    assert not any('recipe' in key.lower() for layer in manifest['layers'] for key in layer)
    history = DocumentHistory(load_project(target), saved=True)
    before = composite(history.document)
    original = history.document.layer(lid)
    job = prepare_job(history, tmp_path, 'natural', 100, lid, {'smooth': 40, 'whiten': 0})
    render_worker(job.directory, Engine)
    job.status = 'completed'
    assert attach_result(history, job) == lid
    changed = history.document.layer(lid)
    assert changed.name == original.name and changed.opacity == original.opacity
    assert changed.origin == original.origin and changed.size == original.size
    assert changed.rotation == 15 and changed.flip_x
    np.testing.assert_array_equal(changed.mask, mask)
    np.testing.assert_array_equal(changed.recipe_source, original.recipe_source)
    assert changed.recipe_settings['overrides'] == {'smooth': 40, 'whiten': 0}
    assert len(history.document.layers) == 2
    after = composite(history.document)
    assert not np.array_equal(after, before)
    history.save(target)
    np.testing.assert_array_equal(composite(load_project(target)), after)
    history.undo()
    np.testing.assert_array_equal(composite(history.document), before)
    history.redo()
    np.testing.assert_array_equal(composite(history.document), after)


def test_metadata_detaches_duplicates_and_source_counts_toward_history_budget(tmp_path):
    history, lid = setup(tmp_path)
    layer = history.document.layer(lid)
    assert not layer.recipe_source.flags.writeable
    assert history.retained_buffer_bytes >= layer.recipe_source.nbytes * 3
    layer.recipe_settings['overrides']['smooth'] = 99
    assert history.document.layer(lid).recipe_settings['overrides'] == {}
    history.duplicate_layer(lid)
    duplicate = history.document.layers[-1]
    assert duplicate.id != lid and duplicate.recipe_settings == history.document.layer(lid).recipe_settings
    history.remove_layer(lid)
    target = tmp_path / 'duplicate.comp'
    history.save(target)
    assert load_project(target).layer(duplicate.id).recipe_source is not None


def test_replacing_pixels_explicitly_drops_provenance(tmp_path):
    history, lid = setup(tmp_path)
    history.replace_pixels(lid, np.full((6, 8, 4), 255, np.uint8))
    layer = history.document.layer(lid)
    assert layer.recipe_settings is None and layer.recipe_source is None
    with pytest.raises(ValueError, match='saved source'):
        prepare_job(history, tmp_path, 'natural', 100, lid)
    target = tmp_path / 'raster.comp'
    history.save(target)
    assert not (target / 'retouch-recipes.json').exists()
    assert not list((target / 'images').glob('*.recipe-source.png'))


@pytest.mark.parametrize('asset', ['source', 'result', 'missing', 'symlink'])
def test_changed_or_missing_recipe_assets_fail_closed(tmp_path, asset):
    history, lid = setup(tmp_path)
    target = tmp_path / 'safe.comp'
    history.save(target)
    path = target / 'images' / (lid + ('.png' if asset == 'result' else '.recipe-source.png'))
    if asset == 'missing':
        path.unlink()
    elif asset == 'symlink':
        path.unlink()
        path.symlink_to(target / 'images' / (lid + '.png'))
    else:
        Image.new('RGBA', (8, 6), (1, 2, 3, 255)).save(path)
    with pytest.raises(ValueError):
        load_project(target)


@pytest.mark.parametrize('change', ['version', 'document', 'orphan', 'settings', 'path'])
def test_invalid_extension_is_rejected_before_asset_decode(tmp_path, monkeypatch, change):
    from retouch.editor import project_store
    history, lid = setup(tmp_path)
    target = tmp_path / 'safe.comp'
    history.save(target)
    path = target / 'retouch-recipes.json'
    record = json.loads(path.read_text())
    if change == 'version': record['version'] = True
    elif change == 'document': record['documentID'] = 'foreign'
    elif change == 'orphan': record['layers']['../../outside'] = record['layers'].pop(lid)
    elif change == 'settings': record['layers'][lid]['settings']['overrides']['smooth'] = float('nan')
    else: record['layers'][lid]['sourceFile'] = '../outside.png'
    path.write_text(json.dumps(record))
    def forbidden(*args): pytest.fail('Invalid extension must fail before PNG decode')
    monkeypatch.setattr(project_store, '_read_png', forbidden)
    with pytest.raises(ValueError): load_project(target)


def test_regeneration_refuses_stale_results(tmp_path):
    history, lid = setup(tmp_path)
    job = prepare_job(history, tmp_path, 'natural', 100, lid)
    render_worker(job.directory, Engine)
    job.status = 'completed'
    history.update_layer(lid, name='Changed during render')
    revision = history.revision
    with pytest.raises(ValueError, match='changed'):
        attach_result(history, job)
    assert history.revision == revision


def test_preview_does_not_resize_or_drop_saved_native_source(tmp_path):
    from retouch.editor.document import Layer
    from retouch.editor.ui import preview_document
    pixels = np.full((1024, 2048, 4), 255, np.uint8)
    doc = Document(2048, 1024)
    doc.add_layer(Layer('Recipe', pixels=pixels, recipe_source=pixels,
                        recipe_settings=dict(version=1, recipe='natural', overrides={}, source_revision=0)))
    preview = preview_document(doc)
    assert preview.shape == (512, 1024, 4)
    assert doc.layers[0].recipe_source.shape == (1024, 2048, 4)
    assert doc.layers[0].recipe_settings['recipe'] == 'natural'
