"""HTTP routes for the layer editor page inside the Retouch app window.

The page (``static/editor.html``) is a canvas UI; this module serves it and a
small JSON API over :class:`EditorSession`, mounted on the Gradio server under
``/editor`` (see :func:`editor_routes`). Documents live in this process, so a
page reload picks its document back up.

The API reads and writes files at paths the user types, so it only answers
requests that name this computer as the host (no DNS rebinding) and carry the
per-process token embedded in the page (another web site the user visits
cannot read the page, so cannot forge requests).
"""
import secrets
import tempfile
import threading
import time
import uuid
from pathlib import Path

import cv2

from .document import UnsupportedFeature
from .project_store import ProjectError
from .session import EditorError, EditorSession, read_photo, user_path

STATIC = Path(__file__).with_name('static')
TOKEN = secrets.token_urlsafe(32)
MAX_SESSIONS = 8
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
LOCAL_HOSTS = frozenset({'127.0.0.1', 'localhost', '::1', '[::1]'})


class _Entry:
    def __init__(self):
        self.session = EditorSession()
        self.lock = threading.Lock()
        self.used = time.monotonic()


_sessions = {}
_registry_lock = threading.Lock()


def unsaved_titles():
    """Titles of open documents with unsaved changes (for a close prompt)."""
    with _registry_lock:
        entries = list(_sessions.values())
    return [entry.session.title for entry in entries if entry.session.dirty]


def get_session(session_id=None):
    """The session for ``session_id``, or a new one when it is unknown."""
    with _registry_lock:
        entry = _sessions.get(session_id) if session_id else None
        if entry is None:
            if len(_sessions) >= MAX_SESSIONS:
                clean = [(e.used, key) for key, e in _sessions.items() if not e.session.dirty]
                if not clean:
                    raise EditorError('Too many editor windows have unsaved work; '
                                      'save or close one first.')
                del _sessions[min(clean)[1]]
            session_id = uuid.uuid4().hex
            entry = _sessions[session_id] = _Entry()
        entry.used = time.monotonic()
        return session_id, entry


def _host_is_local(host):
    host = (host or '').strip().lower()
    if host.startswith('['):
        name = host.split(']')[0] + ']'
    else:
        name = host.rsplit(':', 1)[0] if host.count(':') == 1 else host
    return name in LOCAL_HOSTS


def handle_action(session, action, args):
    """Run one editor action; returns the new state (or extra data)."""
    args = dict(args or {})
    layer = args.get('layer_id')
    if action == 'state':
        return session.state()
    if action == 'open':
        return session.open(args.get('path', ''))
    if action == 'add_layer':
        return session.add_photo_layer(user_path(args.get('path', '')))
    if action == 'select':
        return session.select(layer)
    if action == 'update_layer':
        changes = {k: args[k] for k in ('name', 'visible', 'opacity', 'mask_enabled') if k in args}
        return session.update_layer(layer, **changes)
    if action == 'move_layer':
        return session.move_layer(layer, args.get('index', 0))
    if action == 'remove_layer':
        return session.remove_layer(layer)
    if action == 'add_mask':
        return session.add_mask(layer, reveal=bool(args.get('reveal', True)))
    if action == 'remove_mask':
        return session.remove_mask(layer)
    if action == 'brush':
        points = args.get('points') or []
        if not isinstance(points, list) or not all(
                isinstance(p, (list, tuple)) and len(p) == 2 for p in points):
            raise EditorError('Brush points must be [x, y] pairs.')
        return session.brush(layer, points, diameter=args.get('diameter', 40),
                             hardness=args.get('hardness', 1.0),
                             opacity=args.get('opacity', 1.0),
                             reveal=bool(args.get('reveal', True)))
    if action == 'undo':
        return session.undo()
    if action == 'redo':
        return session.redo()
    if action == 'save':
        return session.save(args.get('path') or None, overwrite=bool(args.get('overwrite')))
    if action == 'export':
        target = session.export_png(args.get('path', ''), overwrite=bool(args.get('overwrite')))
        state = session.state()
        state['message'] = f'Exported {target}'
        return state
    raise EditorError(f'Unknown editor action {action!r}.')


def _png(rgba_or_gray):
    if rgba_or_gray.ndim == 3:
        rgba_or_gray = cv2.cvtColor(rgba_or_gray, cv2.COLOR_RGBA2BGRA)
    ok, data = cv2.imencode('.png', rgba_or_gray, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    if not ok:
        raise EditorError('Could not encode the preview.')
    return data.tobytes()


def editor_routes():
    """Starlette/FastAPI routes for the editor, to pass as ``app_kwargs['routes']``."""
    from fastapi import APIRouter, Request
    from fastapi.responses import HTMLResponse, JSONResponse, Response

    router = APIRouter()

    def refuse(request):
        if not _host_is_local(request.headers.get('host')):
            return JSONResponse({'error': 'The layer editor only runs on this computer.'},
                                status_code=403)
        return None

    def guard(request):
        refused = refuse(request)
        if refused is not None:
            return refused
        if not secrets.compare_digest(request.headers.get('x-editor-token', ''), TOKEN):
            return JSONResponse({'error': 'Reload the editor page.'}, status_code=403)
        return None

    def run(entry, fn):
        with entry.lock:
            try:
                return fn()
            except FileExistsError as exc:
                return JSONResponse({'error': f'{exc} already exists.', 'exists': str(exc)},
                                    status_code=409)
            except (EditorError, ProjectError, UnsupportedFeature, ValueError, OSError) as exc:
                return JSONResponse({'error': str(exc)}, status_code=400)

    @router.get('/editor/', response_class=HTMLResponse)
    def page(request: Request):
        refused = refuse(request)
        if refused is not None:
            return HTMLResponse('<p>The layer editor only runs on this computer.</p>',
                                status_code=403)
        html = (STATIC / 'editor.html').read_text(encoding='utf-8')
        script = (STATIC / 'editor.js').read_text(encoding='utf-8')
        html = html.replace('/*EDITOR_SCRIPT*/', script).replace('__EDITOR_TOKEN__', TOKEN)
        return HTMLResponse(html, headers={'Cache-Control': 'no-store'})

    @router.post('/editor/api/session')
    async def session_route(request: Request):
        refused = guard(request)
        if refused is not None:
            return refused
        body = await _json(request)
        try:
            session_id, entry = get_session(body.get('session_id'))
        except EditorError as exc:
            return JSONResponse({'error': str(exc)}, status_code=400)
        result = run(entry, entry.session.state)
        if isinstance(result, dict):
            result = dict(result, session_id=session_id)
        return result

    @router.post('/editor/api/{session_id}/action')
    async def action_route(session_id: str, request: Request):
        refused = guard(request)
        if refused is not None:
            return refused
        entry = _sessions.get(session_id)
        if entry is None:
            return JSONResponse({'error': 'This editor session has ended; reload the page.',
                                 'expired': True}, status_code=404)
        body = await _json(request)
        return await _in_thread(run, entry, lambda: handle_action(
            entry.session, body.get('action'), body.get('args')))

    @router.post('/editor/api/{session_id}/upload')
    async def upload_route(session_id: str, request: Request, name: str = 'photo.jpg',
                           mode: str = 'open'):
        refused = guard(request)
        if refused is not None:
            return refused
        entry = _sessions.get(session_id)
        if entry is None:
            return JSONResponse({'error': 'Reload the page.', 'expired': True}, status_code=404)
        data = bytearray()
        async for chunk in request.stream():
            data += chunk
            if len(data) > MAX_UPLOAD_BYTES:
                return JSONResponse({'error': 'That file is too large to upload.'},
                                    status_code=413)
        filename = Path(name).name or 'photo.jpg'

        def load():
            with tempfile.TemporaryDirectory(prefix='retouch-editor-') as folder:
                path = Path(folder) / filename
                path.write_bytes(bytes(data))
                pixels = read_photo(path)
            if mode == 'layer':
                return entry.session.add_photo_layer(Path(filename), pixels=pixels)
            return entry.session.open_photo(Path(filename), pixels=pixels)
        return await _in_thread(run, entry, load)

    @router.get('/editor/api/{session_id}/preview.png')
    async def preview_route(session_id: str, request: Request):
        refused = guard(request)
        if refused is not None:
            return refused
        entry = _sessions.get(session_id)
        if entry is None:
            return JSONResponse({'error': 'Reload the page.', 'expired': True}, status_code=404)
        result = await _in_thread(run, entry, lambda: _png(entry.session.preview()[0]))
        if isinstance(result, bytes):
            return Response(result, media_type='image/png', headers={'Cache-Control': 'no-store'})
        return result

    @router.get('/editor/api/{session_id}/mask.png')
    async def mask_route(session_id: str, layer: str, request: Request):
        refused = guard(request)
        if refused is not None:
            return refused
        entry = _sessions.get(session_id)
        if entry is None:
            return JSONResponse({'error': 'Reload the page.', 'expired': True}, status_code=404)

        def mask():
            preview = entry.session.mask_preview(layer)
            if preview is None:
                raise EditorError('This layer has no mask.')
            return _png(preview)
        result = await _in_thread(run, entry, mask)
        if isinstance(result, bytes):
            return Response(result, media_type='image/png', headers={'Cache-Control': 'no-store'})
        return result

    return list(router.routes)


async def _json(request):
    try:
        body = await request.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


async def _in_thread(fn, *args):
    from starlette.concurrency import run_in_threadpool
    return await run_in_threadpool(fn, *args)
