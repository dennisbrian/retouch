"""Tier-1 hotkeys: source-level contract + wiring of the three elem_ids."""
import re

import gradio as gr

from retouch.editor.ui import EDITOR_HOTKEYS_JS, build_editor_tab


def test_chord_branches_present():
    assert "hit('editor-undo')" in EDITOR_HOTKEYS_JS
    assert EDITOR_HOTKEYS_JS.count("hit('editor-redo')") == 2  # shift+z and y
    assert "hit('editor-clear-selection')" in EDITOR_HOTKEYS_JS
    # shift+z must be tested before plain z, or it never fires.
    assert (EDITOR_HOTKEYS_JS.index("S && k === 'z'")
            < EDITOR_HOTKEYS_JS.index("hit('editor-undo')"))


def test_suppression_guard_present():
    for token in ("'INPUT'", "'TEXTAREA'", "'SELECT'", 'isContentEditable',
                  'metaKey || e.ctrlKey', 'preventDefault'):
        assert token in EDITOR_HOTKEYS_JS, token


def test_editor_buttons_carry_elem_ids():
    with gr.Blocks() as app:
        with gr.Tabs():
            build_editor_tab()
    elem_ids = {c['props'].get('elem_id')
                for c in app.get_config_file()['components']}
    assert {'editor-undo', 'editor-redo',
            'editor-clear-selection'} <= elem_ids


def test_gui_blocks_attaches_editor_js():
    source = open('gui.py').read()
    assert 'js=EDITOR_HOTKEYS_JS' in source
    assert re.search(r'from retouch\.editor\.ui import EDITOR_HOTKEYS_JS', source)
