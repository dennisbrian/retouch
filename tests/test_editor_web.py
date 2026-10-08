"""Layer editor HTTP routes (retouch/editor/web.py) through a test client."""
import io

import numpy as np
import pytest
from PIL import Image

fastapi = pytest.importorskip('fastapi')
from fastapi import FastAPI
from fastapi.testclient import TestClient

from retouch.editor import web

HEADERS = {'X-Editor-Token': web.TOKEN}


@pytest.fixture
def client():
    web._sessions.clear()
    app = FastAPI(routes=web.editor_routes())
    with TestClient(app, base_url='http://127.0.0.1:7860') as test_client:
        yield test_client
    web._sessions.clear()


def _png_bytes(width=64, height=48, seed=0):
    rgb = np.random.default_rng(seed).integers(0, 256, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format='PNG')
    return buffer.getvalue()


def _start(client):
    response = client.post('/editor/api/session', json={}, headers=HEADERS)
    assert response.status_code == 200
    return response.json()['session_id']


def _act(client, session_id, action, **args):
    return client.post(f'/editor/api/{session_id}/action',
                       json={'action': action, 'args': args}, headers=HEADERS)


def test_page_embeds_the_script_and_token(client):
    response = client.get('/editor/')
    assert response.status_code == 200
    assert web.TOKEN in response.text
    assert '/*EDITOR_SCRIPT*/' not in response.text and 'startSession' in response.text
    assert response.headers['cache-control'] == 'no-store'


def test_requests_naming_another_host_are_refused(client):
    # A rebinding site resolves to 127.0.0.1 but still sends its own Host.
    assert client.get('/editor/', headers={'Host': 'evil.example:7860'}).status_code == 403
    response = client.post('/editor/api/session', json={},
                           headers=dict(HEADERS, Host='evil.example'))
    assert response.status_code == 403
    assert client.get('/editor/', headers={'Host': 'localhost:7860'}).status_code == 200


def test_api_calls_need_the_page_token(client):
    assert client.post('/editor/api/session', json={}).status_code == 403
    assert client.post('/editor/api/session', json={},
                       headers={'X-Editor-Token': 'guess'}).status_code == 403


def test_open_brush_undo_save_reopen_export(client, tmp_path):
    photo = tmp_path / 'photo.png'
    photo.write_bytes(_png_bytes())
    session_id = _start(client)
    state = _act(client, session_id, 'open', path=str(photo)).json()
    assert state['open'] and state['width'] == 64 and not state['dirty']
    layer = state['selected_id']

    state = _act(client, session_id, 'brush', layer_id=layer, points=[[10, 10], [50, 10]],
                 diameter=8, hardness=1, opacity=1, reveal=False).json()
    assert state['dirty'] and state['layers'][0]['has_mask']

    preview = client.get(f'/editor/api/{session_id}/preview.png', headers=HEADERS)
    assert preview.headers['content-type'] == 'image/png'
    pixels = np.asarray(Image.open(io.BytesIO(preview.content)))
    assert pixels.shape == (48, 64, 4) and pixels[10, 30, 3] == 0 and pixels[40, 30, 3] == 255
    mask = client.get(f'/editor/api/{session_id}/mask.png', params={'layer': layer},
                      headers=HEADERS)
    assert np.asarray(Image.open(io.BytesIO(mask.content)))[10, 30] == 0

    assert not _act(client, session_id, 'undo').json()['layers'][0]['has_mask']
    assert _act(client, session_id, 'redo').json()['layers'][0]['has_mask']

    target = tmp_path / 'edit.comp'
    state = _act(client, session_id, 'save', path=str(target)).json()
    assert not state['dirty'] and target.is_dir()
    assert _act(client, session_id, 'save', path=str(target)).status_code == 200   # own path

    other = _start(client)
    state = _act(client, other, 'open', path=str(target)).json()
    assert state['layers'][0]['has_mask'] and not state['dirty']
    out = tmp_path / 'flat.png'
    state = _act(client, other, 'export', path=str(out)).json()
    assert 'Exported' in state['message']
    assert np.asarray(Image.open(out))[10, 30, 3] == 0

    # Replacing an existing file needs the page's confirmation.
    response = _act(client, other, 'export', path=str(out))
    assert response.status_code == 409 and response.json()['exists'] == str(out)
    assert _act(client, other, 'export', path=str(out), overwrite=True).status_code == 200


def test_upload_opens_or_adds_a_photo(client):
    session_id = _start(client)
    response = client.post(f'/editor/api/{session_id}/upload',
                           params={'name': 'DSCF0001.png', 'mode': 'open'},
                           content=_png_bytes(), headers=HEADERS)
    assert response.status_code == 200 and response.json()['title'] == 'DSCF0001'
    response = client.post(f'/editor/api/{session_id}/upload',
                           params={'name': '../../evil.png', 'mode': 'layer'},
                           content=_png_bytes(seed=1), headers=HEADERS)
    names = [layer['name'] for layer in response.json()['layers']]
    assert names == ['DSCF0001', 'evil']
    bad = client.post(f'/editor/api/{session_id}/upload', params={'name': 'x.png'},
                      content=b'not an image', headers=HEADERS)
    assert bad.status_code == 400


def test_errors_come_back_as_messages(client, tmp_path):
    session_id = _start(client)
    response = _act(client, session_id, 'open', path='relative.jpg')
    assert response.status_code == 400 and 'full path' in response.json()['error']
    assert _act(client, session_id, 'nonsense').status_code == 400
    assert _act(client, 'no-such-session', 'state').status_code == 404
    response = _act(client, session_id, 'brush', layer_id='x', points='bad')
    assert response.status_code == 400


def test_a_reloaded_page_gets_its_document_back(client, tmp_path):
    photo = tmp_path / 'photo.png'
    photo.write_bytes(_png_bytes())
    session_id = _start(client)
    _act(client, session_id, 'open', path=str(photo))
    again = client.post('/editor/api/session', json={'session_id': session_id}, headers=HEADERS)
    assert again.json()['session_id'] == session_id and again.json()['open']


def test_unsaved_work_is_reported_and_never_evicted(client, tmp_path, monkeypatch):
    photo = tmp_path / 'photo.png'
    photo.write_bytes(_png_bytes())
    monkeypatch.setattr(web, 'MAX_SESSIONS', 2)
    first = _start(client)
    state = _act(client, first, 'open', path=str(photo)).json()
    _act(client, first, 'brush', layer_id=state['selected_id'], points=[[5, 5]], diameter=4,
         reveal=False)
    assert web.unsaved_titles() == ['photo']
    _start(client)
    _start(client)                      # evicts the clean second session, not the first
    assert first in web._sessions
    _act(client, list(web._sessions)[-1], 'open', path=str(photo))
    second = list(web._sessions)[-1]
    state = _act(client, second, 'state').json()
    _act(client, second, 'brush', layer_id=state['selected_id'], points=[[5, 5]], diameter=4,
         reveal=False)
    response = client.post('/editor/api/session', json={}, headers=HEADERS)
    assert response.status_code == 400 and 'unsaved' in response.json()['error']


@pytest.mark.parametrize('host, local', [
    ('127.0.0.1:7860', True), ('localhost', True), ('[::1]:7860', True),
    ('LOCALHOST:1', True), ('example.com', False), ('127.0.0.1.evil.com:80', False), ('', False),
])
def test_host_check(host, local):
    assert web._host_is_local(host) is local


def test_editor_page_files_ship_in_the_wheel():
    # The page is read from package data at request time, so the wheel and
    # the PyInstaller bundle (which copies retouch/ whole) must carry it.
    try:
        import tomllib
    except ModuleNotFoundError:          # Python 3.9-3.10
        tomllib = pytest.importorskip('tomli')
    root = web.STATIC.parents[2]
    config = tomllib.loads((root / 'pyproject.toml').read_text(encoding='utf-8'))
    globs = config['tool']['setuptools']['package-data']['retouch.editor']
    for name in ('editor.html', 'editor.js'):
        assert (web.STATIC / name).is_file()
        assert any(web.STATIC.joinpath(name).match(pattern) for pattern in globs)
