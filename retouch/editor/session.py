"""One open editor document: what the editor window drives, without any UI.

``EditorSession`` wraps :class:`DocumentHistory` with the editor's actions
(open a photo or ``.comp`` project, add a photo as a layer, layer panel edits,
mask brush, undo/redo, save, export) and a bounded-resolution preview. Every
edit goes through the history, so each is one undo step and the saved
revision tracks unsaved changes. Native pixels are kept; only the preview is
scaled. Not thread-safe: callers serialise access (the web layer holds a lock
per session).
"""
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import ImageOps

from . import brush
from .commands import DocumentHistory
from .composite import _fit_mask, export_png
from .document import Document, Layer
from .preview import PreviewRenderer, preview_scale, scaled_rect
from .project_store import load_project

IMAGE_SUFFIXES = frozenset({'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp', '.bmp'})


class EditorError(ValueError):
    """An action the user asked for that cannot be done as asked."""


def user_path(text):
    """A full path from what the user typed (``~`` allowed)."""
    text = (text or '').strip()
    if not text:
        raise EditorError('Enter a file path.')
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise EditorError('Use a full path, for example ~/Pictures/photo.jpg.')
    return path


def read_photo(path):
    """A photo as uint8 RGB or RGBA (EXIF orientation applied)."""
    from ..io import RAW_EXTENSIONS, imread_exif, open_image

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in IMAGE_SUFFIXES and suffix not in {s.lower() for s in RAW_EXTENSIONS}:
        raise EditorError(f'{path.name} is not a photo this editor opens '
                          '(JPEG, PNG, TIFF, WebP, BMP or RAW).')
    if not path.is_file():
        raise EditorError(f'No file at {path}.')
    if suffix in {'.png', '.tif', '.tiff', '.webp'}:
        with open_image(path) as image:
            if 'A' in image.getbands() or image.mode == 'P' and 'transparency' in image.info:
                # Keep transparency; the editor works in 8-bit straight alpha.
                return np.asarray(ImageOps.exif_transpose(image).convert('RGBA'))
    bgr = imread_exif(path)
    if bgr is None:
        raise EditorError(f'Could not read {path.name}.')
    if bgr.dtype != np.uint8:
        raise EditorError(f'{path.name} did not decode to 8-bit colour.')
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


class EditorSession:
    """The document open in one editor view."""

    def __init__(self, preview_max_side=None):
        self.history = None
        self.project_path = None       # where Save writes; None until saved
        self.title = 'Untitled'
        self.selected_id = None
        self._preview = (PreviewRenderer() if preview_max_side is None
                         else PreviewRenderer(preview_max_side))

    # Opening -------------------------------------------------------------

    @property
    def dirty(self):
        return self.history is not None and self.history.dirty

    def open(self, path):
        """Open a ``.comp`` project or start a new document from a photo."""
        path = user_path(path) if isinstance(path, str) else Path(path)
        if path.suffix.lower() == '.comp':
            return self.open_project(path)
        return self.open_photo(path)

    def open_project(self, path):
        path = Path(path)
        if not path.is_dir():
            raise EditorError(f'No Compositor project at {path}.')
        document = load_project(path)
        self._start(document, saved=True)
        self.project_path = path
        self.title = path.stem
        return self.state()

    def open_photo(self, path, pixels=None):
        """New document holding one photo layer, sized to the photo."""
        path = Path(path)
        pixels = read_photo(path) if pixels is None else pixels
        height, width = pixels.shape[:2]
        document = Document(width=width, height=height)
        document.add_image(pixels, path.stem or 'Photo')
        # Nothing to lose until the first edit, though Save still asks where.
        self._start(document, saved=True)
        self.project_path = None
        self.title = path.stem or 'Untitled'
        return self.state()

    def _start(self, document, saved):
        self.history = DocumentHistory(document, saved=saved)
        doc = self.history.document
        self.selected_id = doc.active_layer_id or (doc.layers[-1].id if doc.layers else None)

    def _require(self):
        if self.history is None:
            raise EditorError('Open a photo or project first.')
        return self.history

    # Layer panel ---------------------------------------------------------

    def add_photo_layer(self, path, pixels=None, name=None):
        """Add a photo above the selected layer, at the canvas origin."""
        history = self._require()
        path = Path(path)
        pixels = read_photo(path) if pixels is None else pixels
        doc = history.document
        index = doc.index_of(self.selected_id) + 1 if self.selected_id else None
        layer = Layer(name=name or path.stem or 'Photo', pixels=pixels)
        history.add_layer(layer, index)
        self.selected_id = layer.id
        return self.state()

    def select(self, layer_id):
        self._layer(layer_id)
        self.selected_id = layer_id
        return self.state()

    def update_layer(self, layer_id, **changes):
        """Name, visibility, opacity or mask on/off, as one undo step."""
        self._layer(layer_id)
        allowed = {'name', 'visible', 'opacity', 'mask_enabled'}
        if set(changes) - allowed:
            raise EditorError('That layer setting cannot be changed here.')
        if 'name' in changes:
            changes['name'] = str(changes['name']).strip()
            if not changes['name']:
                raise EditorError('A layer needs a name.')
        if 'opacity' in changes:
            opacity = float(changes['opacity'])
            if not (math.isfinite(opacity) and 0 <= opacity <= 1):
                raise EditorError('Opacity must be between 0 and 100%.')
            changes['opacity'] = opacity
        for key in ('visible', 'mask_enabled'):
            if key in changes:
                changes[key] = bool(changes[key])
        self.history.update_layer(layer_id, **changes)
        return self.state()

    def move_layer(self, layer_id, index):
        self._layer(layer_id)
        self.history.move_layer(layer_id, int(index))
        return self.state()

    def remove_layer(self, layer_id):
        self._layer(layer_id)
        self.history.remove_layer(layer_id)
        doc = self.history.document
        self.selected_id = doc.layers[-1].id if doc.layers else None
        return self.state()

    def add_mask(self, layer_id, reveal=True):
        layer = self._layer(layer_id)
        if layer.mask is not None:
            raise EditorError('This layer already has a mask.')
        w, h = layer.size
        self.history.set_mask(layer_id, np.full((h, w), 255 if reveal else 0, np.uint8))
        return self.state()

    def remove_mask(self, layer_id):
        self._layer(layer_id)
        self.history.set_mask(layer_id, None)
        return self.state()

    # Mask brush ----------------------------------------------------------

    def brush(self, layer_id, points, *, diameter, hardness=1.0, opacity=1.0, reveal=True):
        """Paint one stroke on a layer's mask (``points`` in document pixels).

        A layer without a mask gets a reveal-all mask in the same undo step,
        so hiding works straight away.
        """
        layer = self._layer(layer_id)
        if layer.pixels is None:
            raise EditorError('This layer has no pixels to mask.')
        if not layer.visible:
            raise EditorError('Show the layer before painting its mask.')
        if layer.mask is not None and not layer.mask_enabled:
            raise EditorError('Turn the mask on before painting it.')
        w, h = layer.size
        if layer.mask is None:
            if reveal:
                return self.state()      # nothing hidden yet: nothing to reveal
            mask = np.full((h, w), 255, np.uint8)
        else:
            mask = _fit_mask(layer.mask, w, h, layer.sampling)
        local = []
        for x, y in points:
            x, y = float(x) - layer.origin[0], float(y) - layer.origin[1]
            if not (math.isfinite(x) and math.isfinite(y)):
                raise EditorError('Brush points must be numbers.')
            local.append((w - x if layer.flip_x else x, h - y if layer.flip_y else y))
        try:
            painted = brush.paint_mask(mask, local, diameter=diameter, hardness=hardness,
                                       opacity=opacity, reveal=reveal)
        except ValueError as exc:
            raise EditorError(str(exc)) from exc
        if painted is not mask:
            self.history.set_mask(layer_id, painted)
        return self.state()

    # History, saving, export ---------------------------------------------

    def undo(self):
        self._require().undo()
        self._fix_selection()
        return self.state()

    def redo(self):
        self._require().redo()
        self._fix_selection()
        return self.state()

    def save(self, path=None, overwrite=False):
        """Save to ``path`` (Save As) or the project's own path."""
        history = self._require()
        if path is None:
            if self.project_path is None:
                raise EditorError('Choose where to save the project first.')
            target = self.project_path
        else:
            target = user_path(path) if isinstance(path, str) else Path(path)
            if target.suffix.lower() != '.comp':
                target = target.with_name(target.name + '.comp')
            if target != self.project_path and target.exists() and not overwrite:
                raise FileExistsError(str(target))
        if not target.parent.is_dir():
            raise EditorError(f'Folder {target.parent} does not exist.')
        try:
            history.save(target)
        except FileExistsError as exc:
            raise EditorError(f'{target.name} exists and is not a Compositor project; '
                              'choose another name.') from exc
        self.project_path = target
        self.title = target.stem
        return self.state()

    def export_png(self, path, overwrite=False):
        """Flatten at native resolution to a PNG (never marks the project saved)."""
        history = self._require()
        target = user_path(path) if isinstance(path, str) else Path(path)
        if target.suffix.lower() != '.png':
            target = target.with_name(target.name + '.png')
        if target.exists() and not overwrite:
            raise FileExistsError(str(target))
        if not target.parent.is_dir():
            raise EditorError(f'Folder {target.parent} does not exist.')
        export_png(history.document, target)
        return target

    # Views ---------------------------------------------------------------

    def preview(self):
        """RGBA preview of the composite and its scale (document -> preview)."""
        return self._preview.render(self._require().document)

    def mask_preview(self, layer_id):
        layer = self._layer(layer_id)
        doc = self.history.document
        scale = self._preview_scale(doc)
        return self._preview.mask(layer, scale)

    def state(self):
        """Everything the editor view shows, as plain JSON-ready data."""
        if self.history is None:
            return {'open': False}
        history = self.history
        doc = history.document
        scale = self._preview_scale(doc)
        layers = []
        for layer in doc.layers:
            x, y, w, h = scaled_rect(layer.origin, layer.size, scale)
            layers.append({
                'id': layer.id, 'name': layer.name, 'visible': layer.visible,
                'opacity': layer.opacity, 'blend_mode': layer.blend_mode,
                'blank': layer.pixels is None, 'has_mask': layer.mask is not None,
                'mask_enabled': layer.mask_enabled, 'origin': list(layer.origin),
                'size': list(layer.size), 'flip_x': layer.flip_x, 'flip_y': layer.flip_y,
                'preview_rect': [x, y, w, h],
            })
        return {
            'open': True, 'title': self.title,
            'project_path': None if self.project_path is None else str(self.project_path),
            'width': doc.width, 'height': doc.height, 'preview_scale': scale,
            'preview_size': [max(1, int(round(doc.width * scale))),
                             max(1, int(round(doc.height * scale)))],
            'revision': history.revision, 'dirty': history.dirty,
            'can_undo': history.can_undo, 'can_redo': history.can_redo,
            'undo_label': history.undo_label, 'redo_label': history.redo_label,
            'selected_id': self.selected_id, 'layers': layers,
        }

    def _preview_scale(self, doc):
        return preview_scale(doc.width, doc.height, self._preview.max_side)

    def _layer(self, layer_id):
        try:
            return self._require().document.layer(layer_id)
        except KeyError:
            raise EditorError('That layer is no longer in the document.') from None

    def _fix_selection(self):
        doc = self.history.document
        ids = [layer.id for layer in doc.layers]
        if self.selected_id not in ids:
            self.selected_id = ids[-1] if ids else None
