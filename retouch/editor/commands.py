"""Transactional layer edits and bounded undo history for the editor core.

Metadata snapshots are detached; read-only pixel/mask buffers are shared.
The budget counts unique retained array bytes, including the current document.
If the current document alone exceeds it, all undo states are dropped. This
is a buffer budget, not a promise about total Python process memory.
"""
from copy import copy, deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .document import Document, Layer, _frozen, new_id, rgba_from
from .project_store import CURRENT_VERSION, _check_manifest, _encode, save_project


def _snapshot(document: Document) -> Document:
    result = copy(document)
    result.guides = deepcopy(document.guides)
    result.layers = []
    for layer in document.layers:
        detached = copy(layer)
        if layer.pixels is not None:
            detached.pixels = rgba_from(layer.pixels)
        if layer.mask is not None:
            detached.mask = _frozen(layer.mask, 2)
        detached.text = deepcopy(layer.text)
        detached.shape = deepcopy(layer.shape)
        result.layers.append(detached)
    return result


def _validate(document: Document) -> None:
    document.validate()
    _check_manifest(_encode(document), CURRENT_VERSION)


def _same(left: Document, right: Document) -> bool:
    if {k: v for k, v in vars(left).items() if k != 'layers'} != {
            k: v for k, v in vars(right).items() if k != 'layers'}:
        return False
    if len(left.layers) != len(right.layers):
        return False
    for a, b in zip(left.layers, right.layers):
        for key, value in vars(a).items():
            other = getattr(b, key)
            if key in ('pixels', 'mask'):
                if value is not other:
                    return False
            elif value != other:
                return False
    return True


@dataclass
class _State:
    document: Document
    revision: int
    label: str


class DocumentHistory:
    """Own document edits; use ``document`` snapshots for rendering/export.

    Direct edits to the input document or a returned snapshot do not update
    this history. Newly created documents start dirty; pass ``saved=True``
    for a document just opened from disk. ``save`` marks clean only after a
    successful project save. Calls are synchronous and belong to one session.
    """

    def __init__(self, document: Document, *, saved: bool = False,
                 max_entries: int = 100,
                 max_buffer_bytes: int = 512 * 1024 * 1024) -> None:
        for name, value in (('max_entries', max_entries),
                            ('max_buffer_bytes', max_buffer_bytes)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        _validate(document)
        self.max_entries = max_entries
        self.max_buffer_bytes = max_buffer_bytes
        self._states = [_State(_snapshot(document), 0, '')]
        self._cursor = 0
        self._next_revision = 1
        self._saved_revision = 0 if saved else None

    @property
    def document(self) -> Document:
        """Detached metadata with shared read-only image buffers."""
        return _snapshot(self._states[self._cursor].document)

    @property
    def revision(self) -> int:
        return self._states[self._cursor].revision

    @property
    def dirty(self) -> bool:
        return self.revision != self._saved_revision

    @property
    def can_undo(self) -> bool:
        return self._cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._cursor + 1 < len(self._states)

    @property
    def undo_label(self) -> Optional[str]:
        return self._states[self._cursor].label if self.can_undo else None

    @property
    def redo_label(self) -> Optional[str]:
        return self._states[self._cursor + 1].label if self.can_redo else None

    @property
    def retained_buffer_bytes(self) -> int:
        buffers = {}
        for state in self._states:
            for layer in state.document.layers:
                for array in (layer.pixels, layer.mask):
                    if array is not None:
                        # Count the whole backing ndarray, even for a view.
                        while isinstance(array.base, np.ndarray):
                            array = array.base
                        buffers[id(array)] = array.nbytes
        return sum(buffers.values())

    def _edit(self, label: str, operation: Callable[[Document], None]) -> bool:
        current = self._states[self._cursor].document
        candidate = _snapshot(current)
        operation(candidate)
        _validate(candidate)
        if _same(current, candidate):
            return False
        self._states = self._states[:self._cursor + 1]
        self._states.append(_State(candidate, self._next_revision, label))
        self._next_revision += 1
        self._cursor += 1
        while len(self._states) > 1 and (
                len(self._states) > self.max_entries + 1
                or self.retained_buffer_bytes > self.max_buffer_bytes):
            self._states.pop(0)
            self._cursor -= 1
        return True

    def add_layer(self, layer: Layer, index: Optional[int] = None) -> bool:
        # Reconstruct to freeze any caller-supplied writable buffers.
        detached = Layer(**deepcopy({k: v for k, v in vars(layer).items()
                                    if k not in ('pixels', 'mask')}),
                         pixels=layer.pixels, mask=layer.mask)
        return self._edit('Add layer', lambda doc: doc.add_layer(detached, index))

    def remove_layer(self, layer_id: str) -> bool:
        return self._edit('Remove layer', lambda doc: doc.remove_layer(layer_id))

    def duplicate_layer(self, layer_id: str) -> bool:
        """Insert an independent copy directly above the source and select it."""
        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            duplicate = copy(layer)
            duplicate.id = new_id()
            # A maximal valid name must remain valid after adding the suffix.
            suffix = ' copy'
            duplicate.name = (layer.name.encode('utf-8')[:16_384 - len(suffix)]
                              .decode('utf-8', errors='ignore') + suffix)
            duplicate.text = deepcopy(layer.text)
            duplicate.shape = deepcopy(layer.shape)
            doc.add_layer(duplicate, doc.index_of(layer_id) + 1)
        return self._edit('Duplicate layer', apply)

    def invert_mask(self, layer_id: str) -> bool:
        """Invert stored coverage, preserving mask enablement and source pixels.

        An absent mask means full coverage; its inverse is a compact black mask.
        """
        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            mask = (np.zeros((1, 1), np.uint8) if layer.mask is None
                    else 255 - layer.mask)
            layer.set_mask(mask, layer.mask_enabled)
        return self._edit('Invert mask', apply)

    def reset_mask(self, layer_id: str) -> bool:
        """Remove stored coverage as one undo step; future painting is enabled."""
        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            if layer.mask is not None:
                # .comp cannot store disabled enablement without a mask asset.
                layer.set_mask(None, True)
        return self._edit('Reset mask', apply)

    def move_layer(self, layer_id: str, index: int) -> bool:
        return self._edit('Reorder layer', lambda doc: doc.move_layer(layer_id, index))

    def update_layer(self, layer_id: str, **changes) -> bool:
        """Set layer properties together as one undoable edit."""
        allowed = {'name', 'visible', 'opacity', 'blend_mode', 'origin',
                   'size', 'flip_x', 'flip_y', 'sampling', 'mask_enabled'}
        if set(changes) - allowed:
            raise ValueError('unsupported layer properties')

        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            for key, value in changes.items():
                setattr(layer, key, deepcopy(value))
        return self._edit('Change layer', apply)

    def set_mask(self, layer_id: str, mask: Optional[np.ndarray],
                 enabled: bool = True) -> bool:
        return self._edit('Edit mask', lambda doc: doc.layer(layer_id).set_mask(mask, enabled))

    def replace_pixels(self, layer_id: str, pixels: np.ndarray) -> bool:
        return self._edit('Replace pixels', lambda doc: doc.layer(layer_id).replace_pixels(pixels))

    def undo(self) -> bool:
        if not self.can_undo:
            return False
        self._cursor -= 1
        return True

    def redo(self) -> bool:
        if not self.can_redo:
            return False
        self._cursor += 1
        return True

    def save(self, path: Path) -> Path:
        """Save the current revision; failed writes keep its dirty status."""
        result = save_project(self._states[self._cursor].document, path)
        self._saved_revision = self.revision
        return result
