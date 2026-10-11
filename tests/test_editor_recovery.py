"""Layer Editor crash recovery: autosave unsaved work and offer it back."""
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pytest

from retouch.editor import Document, composite
from retouch.editor.commands import DocumentHistory
from retouch.editor.recovery import (
    MAX_AGE_SECONDS,
    Autosaver,
    discard_recovery,
    list_recoveries,
    load_recovery,
    recovery_dir,
)
from retouch.editor.ui import EditorSession, build_editor_tab


def _document(value=100):
    doc = Document(16, 12)
    doc.add_image(np.full((12, 16, 3), value, np.uint8), 'Base')
    return doc


def _edited_history():
    history = DocumentHistory(_document(), saved=True)
    layer_id = history.document.layers[0].id
    history.set_mask(layer_id, np.tile(np.arange(16, dtype=np.uint8) * 16, (12, 1)))
    return history, layer_id


def test_recovery_dir_follows_override(tmp_path, monkeypatch):
    monkeypatch.setenv('RETOUCH_EDITOR_RECOVERY_DIR', str(tmp_path / 'r'))
    assert recovery_dir() == tmp_path / 'r'


def test_closed_session_is_offered_back_with_identical_pixels(tmp_path):
    history, _ = _edited_history()
    saver = Autosaver('photo', tmp_path)
    saver.checkpoint(history)
    assert saver.flush(30)
    # A live session is never offered back (a second tab could fork it).
    assert list_recoveries(tmp_path) == []
    saver.close(keep=True)
    [entry] = list_recoveries(tmp_path)
    assert (entry.title, entry.width, entry.height, entry.layers) == ('photo', 16, 12, 1)
    assert 'photo · 16 × 12 · 1 layers · autosaved' in entry.label
    restored = load_recovery(entry.entry_id, tmp_path)
    np.testing.assert_array_equal(composite(restored), composite(history.document))


def test_clean_history_deletes_the_entry(tmp_path):
    history, _ = _edited_history()
    saver = Autosaver('photo', tmp_path)
    saver.checkpoint(history)
    saver.flush(30)
    assert saver.package.is_dir()
    history.undo()  # back to the saved revision
    saver.checkpoint(history)
    assert not saver.package.exists()
    assert not list(tmp_path.glob('*.json'))
    saver.close(keep=True)
    assert list_recoveries(tmp_path) == []


def test_save_deletes_the_entry(tmp_path):
    history, _ = _edited_history()
    saver = Autosaver('photo', tmp_path)
    saver.checkpoint(history)
    saver.flush(30)
    history.save(tmp_path / 'kept.comp')
    saver.checkpoint(history)
    saver.close(keep=True)
    assert list_recoveries(tmp_path) == []
    assert (tmp_path / 'kept.comp').is_dir()


def test_discarding_close_deletes_the_entry(tmp_path):
    history, _ = _edited_history()
    saver = Autosaver('photo', tmp_path)
    saver.checkpoint(history)
    saver.close(keep=False)
    assert list_recoveries(tmp_path) == []
    assert not any(tmp_path.iterdir())


def test_latest_revision_wins_when_checkpoints_queue_up(tmp_path):
    history, layer_id = _edited_history()
    saver = Autosaver('photo', tmp_path)
    for value in range(5):
        history.update_layer(layer_id, opacity=0.1 + value * 0.2)
        saver.checkpoint(history)
    saver.close(keep=True)
    [entry] = list_recoveries(tmp_path)
    assert load_recovery(entry.entry_id, tmp_path).layers[0].opacity == pytest.approx(0.9)


def test_restored_entry_is_replaced_by_edits_and_deleted_on_save(tmp_path):
    history, layer_id = _edited_history()
    first = Autosaver('photo', tmp_path)
    first.checkpoint(history)
    first.close(keep=True)
    [entry] = list_recoveries(tmp_path)

    restored = DocumentHistory(load_recovery(entry.entry_id, tmp_path))
    adopted = Autosaver(entry.title, tmp_path, entry.entry_id)
    restored.update_layer(layer_id, opacity=0.25)
    adopted.checkpoint(restored)
    adopted.flush(30)
    assert len(list(tmp_path.glob('*.json'))) == 1
    with pytest.raises(ValueError, match='another editor window'):
        discard_recovery(entry.entry_id, tmp_path)
    restored.save(tmp_path / 'kept.comp')
    adopted.checkpoint(restored)
    adopted.close(keep=True)
    assert list_recoveries(tmp_path) == []


def test_process_crash_leaves_a_restorable_entry(tmp_path):
    script = textwrap.dedent('''
        import os, sys
        import numpy as np
        from retouch.editor import Document
        from retouch.editor.commands import DocumentHistory
        from retouch.editor.recovery import Autosaver
        doc = Document(16, 12)
        doc.add_image(np.full((12, 16, 3), 100, np.uint8), 'Base')
        history = DocumentHistory(doc, saved=True)
        history.update_layer(history.document.layers[0].id, opacity=0.4)
        saver = Autosaver('crashed', sys.argv[1])
        saver.checkpoint(history)
        saver.flush(60)
        os._exit(3)  # no cleanup, like a crash or force quit
    ''')
    result = subprocess.run([sys.executable, '-c', script, str(tmp_path)],
                            cwd=Path(__file__).resolve().parents[1], timeout=120, check=False)
    assert result.returncode == 3
    [entry] = list_recoveries(tmp_path)
    assert entry.title == 'crashed'
    assert load_recovery(entry.entry_id, tmp_path).layers[0].opacity == pytest.approx(0.4)


def test_expired_partial_and_interrupted_entries_are_cleaned(tmp_path):
    history, _ = _edited_history()
    saver = Autosaver('old', tmp_path)
    saver.checkpoint(history)
    saver.close(keep=True)
    [entry] = list_recoveries(tmp_path)
    # Description without a package (crash between the two writes).
    orphan = 'f' * 32
    (tmp_path / f'{orphan}.json').write_text(json.dumps(
        {'title': 'x', 'width': 1, 'height': 1, 'layers': 1, 'updated': time.time()}))
    staging = tmp_path / f'.{"e" * 32}.comp.saving-abc'
    staging.mkdir()
    (tmp_path / 'notes.json').write_text('{}')  # not ours: left alone
    assert [e.entry_id for e in list_recoveries(tmp_path)] == [entry.entry_id]
    assert not (tmp_path / f'{orphan}.json').exists()
    assert not staging.exists()
    assert (tmp_path / 'notes.json').exists()
    assert list_recoveries(tmp_path, now=entry.updated + MAX_AGE_SECONDS + 1) == []
    assert not entry.package.exists()


def test_write_failure_is_logged_not_raised(tmp_path):
    blocked = tmp_path / 'file'
    blocked.write_text('not a folder')
    history, _ = _edited_history()
    saver = Autosaver('photo', blocked)
    saver.checkpoint(history)
    assert saver.flush(30)
    assert saver.last_error is not None
    saver.close(keep=True)


def test_invalid_ids_are_refused(tmp_path):
    for bad in ('', '../x', 'g' * 32):
        with pytest.raises(ValueError):
            load_recovery(bad, tmp_path)
        with pytest.raises(ValueError):
            discard_recovery(bad, tmp_path)


def _callbacks():
    with gr.Blocks() as app, gr.Tabs():
        build_editor_tab()
    fns = [value.fn for value in app.fns.values() if value.fn]
    named = {fn.__name__: fn for fn in fns}
    named['load'] = [fn for fn in fns if fn.__name__ == 'load']
    named['edit'] = {fn.__kwdefaults__['operation']: fn for fn in fns if fn.__name__ == 'edit'}
    return named


def test_ui_restores_autosaved_work_as_unsaved(tmp_path):
    root = recovery_dir()
    history, _ = _edited_history()
    saver = Autosaver('bunny', root)
    saver.checkpoint(history)
    saver.close(keep=True)
    [entry] = list_recoveries(root)

    fns = _callbacks()
    offered, panel = fns['offering_recoveries']()
    assert offered['choices'] == [(entry.label, entry.entry_id)]
    assert panel['open'] is True

    view = fns['restoring'](None, entry.entry_id, False, None)
    session = view[0]
    try:
        assert 'Unsaved changes' in view[10] and 'Restored autosaved work' in view[10]
        np.testing.assert_array_equal(composite(session.history.document),
                                      composite(history.document))
        # Now owned by this session: not offered to another window.
        assert view[-1]['choices'] == []
        with pytest.raises(gr.Error):
            fns['discarding_recovery'](entry.entry_id)
    finally:
        session.close(keep_recovery=True)
    # Closing the window keeps it for next time.
    assert [e.entry_id for e in list_recoveries(root)] == [entry.entry_id]
    assert fns['discarding_recovery'](entry.entry_id)['choices'] == []


def test_ui_edits_autosave_and_window_close_keeps_them(tmp_path):
    doc = _document()
    session = EditorSession.create(doc, title='photo')
    fns = _callbacks()
    view = fns['edit']['properties'](session, session.selected_id, None, 'Renamed', True, 50, True, 'Normal')
    assert 'Unsaved changes' in view[10]
    session.close(keep_recovery=True)
    [entry] = list_recoveries(recovery_dir())
    restored = load_recovery(entry.entry_id)
    assert restored.layers[0].name == 'Renamed'
    assert restored.layers[0].opacity == pytest.approx(0.5)


def test_ui_discarding_open_deletes_previous_autosave(tmp_path):
    from PIL import Image
    session = EditorSession.create(_document(), title='photo')
    session.history.update_layer(session.selected_id, opacity=0.5)
    session.checkpoint()
    session.autosave.flush(30)
    image_path = tmp_path / 'next.png'
    Image.new('RGB', (3, 2), 'red').save(image_path)
    view = _callbacks()['load'][0](session, str(image_path), '', True, None)
    view[0].close()
    assert list_recoveries(recovery_dir()) == []
