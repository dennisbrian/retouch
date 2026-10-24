"""Transactional layer edits and bounded undo history for the editor core.

Metadata snapshots are detached; read-only pixel/mask buffers are shared.
The budget counts unique retained array bytes, including the current document.
If the current document alone exceeds it, all undo states are dropped. This
is a buffer budget, not a promise about total Python process memory.
"""
from copy import copy, deepcopy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .document import Document, Layer, _frozen, new_id, rgba_from
from .document import MAX_SIDE, MAX_SURFACE_PIXELS
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
        detached.recipe_settings = deepcopy(layer.recipe_settings)
        if layer.recipe_source is not None:
            detached.recipe_source = rgba_from(layer.recipe_source)
        detached.adjustment = deepcopy(layer.adjustment)
        detached.effects = deepcopy(layer.effects)
        if layer.mask_placement is not None:
            detached.mask_placement = deepcopy(layer.mask_placement)
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
            if key in ('pixels', 'mask', 'recipe_source'):
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
                for array in (layer.pixels, layer.mask, layer.recipe_source):
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
                                    if k not in ('pixels', 'mask', 'recipe_source')}),
                         pixels=layer.pixels, mask=layer.mask, recipe_source=layer.recipe_source)
        return self._edit('Add layer', lambda doc: doc.add_layer(detached, index))

    def remove_layer(self, layer_id: str) -> bool:
        def apply(doc):
            removed = {item.id for item in doc.layers if item.id == layer_id
                       or any(p.id == layer_id for p in doc.ancestors(item))}
            doc.layers = [item for item in doc.layers if item.id not in removed]
            for item in doc.layers:
                if item.mask_source_id in removed:
                    item.mask_source_id = None
            if doc.active_layer_id in removed:
                doc.active_layer_id = doc.layers[-1].id if doc.layers else None
        return self._edit('Remove layer', apply)

    def group_layers(self, layer_ids, name='Group') -> bool:
        """Place sibling layers into a pass-through folder as one history edit."""
        def apply(doc):
            ids = set(layer_ids)
            if not ids:
                raise ValueError('Choose layers to group')
            members = [doc.layer(item) for item in ids]
            parents = {item.parent_id for item in members}
            if len(parents) != 1:
                raise ValueError('Only layers in the same folder can be grouped')
            siblings = [item for item in doc.layers if item.parent_id == members[0].parent_id]
            positions = [i for i, item in enumerate(siblings) if item.id in ids]
            if max(positions) - min(positions) + 1 != len(ids):
                raise ValueError('Choose adjacent sibling layers to preserve their stacking order')
            group = Layer(name, size=(doc.width, doc.height), is_group=True,
                          parent_id=members[0].parent_id)
            doc.layers.insert(min(doc.index_of(item.id) for item in members), group)
            for item in members:
                item.parent_id = group.id
            doc.active_layer_id = group.id
        return self._edit('Group layers', apply)

    def ungroup_layer(self, layer_id) -> bool:
        def apply(doc):
            folder = doc.layer(layer_id)
            if not folder.is_group:
                raise ValueError('Select a folder to ungroup')
            if folder.mask is not None or folder.opacity != 1 or not folder.visible:
                raise ValueError('Reset the folder mask, visibility and opacity before ungrouping')
            ordered = doc.ordered_layers()
            for item in ordered:
                if item.parent_id == folder.id:
                    item.parent_id = folder.parent_id
            doc.layers = [item for item in ordered if item.id != layer_id]
            doc.active_layer_id = next((item.id for item in doc.layers if item.parent_id == folder.parent_id), None)
        return self._edit('Ungroup layers', apply)

    def selection_mask(self, layer_id, selection, mode='Replace') -> bool:
        """Project a document-space selection into the layer's native mask."""
        from .selections import project_selection, combine
        import cv2
        def apply(doc):
            layer = doc.layer(layer_id)
            if selection.shape != (doc.height, doc.width):
                raise ValueError('Selection does not match the document')
            projected = project_selection(selection, layer)
            old = (np.full(projected.shape, 255, np.uint8) if layer.mask is None else
                   cv2.resize(layer.mask, (projected.shape[1], projected.shape[0])))
            placement = layer.mask_placement  # combining keeps where the mask sits
            layer.set_mask(combine(old, projected, mode), True)
            layer.mask_placement = placement
        return self._edit('Selection to mask', apply)

    def duplicate_layer(self, layer_id: str) -> bool:
        """Insert an independent copy directly above the source and select it."""
        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            if layer.is_group:
                members = [item for item in doc.ordered_layers()
                           if item.id == layer_id or any(p.id == layer_id for p in doc.ancestors(item))]
                mapping = {item.id: new_id() for item in members}
                insert = doc.index_of(layer_id) + 1
                for original in members:
                    duplicate = copy(original)
                    duplicate.id = mapping[original.id]
                    duplicate.parent_id = mapping.get(original.parent_id, original.parent_id)
                    duplicate.mask_source_id = mapping.get(original.mask_source_id, original.mask_source_id)
                    duplicate.recipe_settings = deepcopy(original.recipe_settings)
                    duplicate.adjustment = deepcopy(original.adjustment)
                    duplicate.text, duplicate.shape = deepcopy(original.text), deepcopy(original.shape)
                    duplicate.name = original.name.encode('utf-8')[:16_379].decode('utf-8', errors='ignore') + ' copy'
                    doc.layers.insert(insert, duplicate)
                    insert += 1
                doc.active_layer_id = mapping[layer_id]
                return
            duplicate = copy(layer)
            duplicate.id = new_id()
            # A maximal valid name must remain valid after adding the suffix.
            suffix = ' copy'
            duplicate.name = (layer.name.encode('utf-8')[:16_384 - len(suffix)]
                              .decode('utf-8', errors='ignore') + suffix)
            duplicate.text = deepcopy(layer.text)
            duplicate.shape = deepcopy(layer.shape)
            duplicate.recipe_settings = deepcopy(layer.recipe_settings)
            duplicate.adjustment = deepcopy(layer.adjustment)
            duplicate.effects = deepcopy(layer.effects)
            duplicate.mask_placement = deepcopy(layer.mask_placement)
            doc.add_layer(duplicate, doc.index_of(layer_id) + 1)
        return self._edit('Duplicate layer', apply)

    def merge_plan(self, layer_id: str, selected_ids=()):
        """What a merge would combine, or None (upstream ``LayerMerge.mergePlan``).

        Several chosen layers merge together with everything their folders
        hold; a folder merges its contents and the folder goes; one layer
        merges with the nearest sibling beneath it, unless that is a folder.
        Returns ``(ids, removed, name, parent_id, anchor_id, action)`` with
        ``ids`` in stacking order.
        """
        doc = self._states[self._cursor].document
        layers = doc.layers
        active = doc.layer(layer_id)

        def inside(folder_id):
            return {item.id for item in layers
                    if any(p.id == folder_id for p in doc.ancestors(item))}

        selected = [item for item in dict.fromkeys(selected_ids or ()) if item]
        if len(selected) > 1:
            picked = set(selected)
            for item in selected:
                doc.layer(item)
                picked |= inside(item)
            ordered = [item for item in layers if item.id in picked]
            tops = [item for item in ordered if item.id in selected]
            if not any(not item.is_group for item in ordered) or not tops:
                return None
            top = tops[-1]
            return ([item.id for item in ordered], picked, top.name, top.parent_id,
                    top.id, 'Merge Layers')
        if active.is_group:
            members = inside(active.id)
            if not any(item.id in members and not item.is_group for item in layers):
                return None
            ids = [item.id for item in layers if item.id in members or item.id == active.id]
            return ids, set(ids), active.name, active.parent_id, active.id, 'Merge Group'
        index = doc.index_of(active.id)
        below = next((item for item in reversed(layers[:index])
                      if item.parent_id == active.parent_id), None)
        if below is None or below.is_group:
            return None
        return ([below.id, active.id], {below.id, active.id}, below.name,
                active.parent_id, active.id, 'Merge Down')

    def merge_layers(self, layer_id: str, selected_ids=()) -> bool:
        """Bake the planned layers, as the canvas shows them, into one pixel layer.

        Port of upstream ``LayerMerge.mergeLayers``: only the merged layers
        are composited (blend modes, opacity, masks, clipping, adjustments and
        effects baked in), cut loose from anything outside the merge, over a
        transparent canvas; the result is trimmed to its non-transparent
        bounds and takes the stack slot of the anchor as one undo step.
        Layers outside the merge that clipped to a merged layer now clip to
        the result.
        """
        from .composite import composite
        plan = self.merge_plan(layer_id, selected_ids)
        if plan is None:
            raise ValueError('Nothing to merge: choose a layer with a layer (not a folder) '
                             'beneath it in the same folder, a folder, or several layers')
        ids, removed, name, parent_id, anchor_id, action = plan

        def apply(doc: Document) -> None:
            kept = set(ids)
            subset = []
            for item in doc.layers:
                if item.id not in kept:
                    continue
                cut = copy(item)
                if cut.parent_id is not None and cut.parent_id not in kept:
                    cut.parent_id = None
                if cut.mask_source_id is not None and cut.mask_source_id not in kept:
                    cut.mask_source_id = None
                subset.append(cut)
            flat = Document(doc.width, doc.height, subset, resolution=doc.resolution)
            full = composite(flat)
            # PixelFilter.trimmed / brush_alpha_bounds: crop to alpha > 0; an
            # empty result keeps the whole canvas.
            rows = np.flatnonzero(full[..., 3].any(axis=1))
            cols = np.flatnonzero(full[..., 3].any(axis=0))
            if rows.size:
                top, bottom = int(rows[0]), int(rows[-1]) + 1
                left, right = int(cols[0]), int(cols[-1]) + 1
            else:
                top, bottom, left, right = 0, doc.height, 0, doc.width
            merged = Layer(name, pixels=np.ascontiguousarray(full[top:bottom, left:right]),
                           origin=(left, top), parent_id=parent_id)
            slot = doc.index_of(anchor_id)
            insertion = slot - sum(1 for item in doc.layers[:slot] if item.id in removed)
            remaining = [item for item in doc.layers if item.id not in removed]
            for item in remaining:
                if item.mask_source_id in removed:
                    item.mask_source_id = merged.id
            remaining.insert(min(max(0, insertion), len(remaining)), merged)
            doc.layers = remaining
            doc.active_layer_id = merged.id
        return self._edit(action, apply)

    def invert_mask(self, layer_id: str) -> bool:
        """Invert stored coverage, preserving mask enablement and source pixels.

        An absent mask means full coverage; its inverse is a compact black mask.
        """
        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            mask = (np.zeros((1, 1), np.uint8) if layer.mask is None
                    else 255 - layer.mask)
            placement = layer.mask_placement  # inversion keeps where the mask sits
            layer.set_mask(mask, layer.mask_enabled)
            layer.mask_placement = placement
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

    def resize_layer(self, layer_id: str, width, height, *, keep_aspect=False,
                     axis='Width') -> bool:
        """Resize the rendered rectangle; optionally derive one dimension.

        The current rendered aspect ratio is preserved, with half-up rounding
        to whole pixels and a one-pixel minimum. Native pixels/masks stay intact.
        """
        if axis not in ('Width', 'Height') or not isinstance(keep_aspect, bool):
            raise ValueError('Invalid aspect ratio controls')

        def dimension(value):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value != int(value) or value < 1):
                raise ValueError('Layer dimensions must be positive whole pixels')
            return int(value)

        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            if keep_aspect:
                old_width, old_height = layer.size
                if axis == 'Width':
                    new_width = dimension(width)
                    new_height = max(1, (new_width * old_height + old_width // 2) // old_width)
                else:
                    new_height = dimension(height)
                    new_width = max(1, (new_height * old_width + old_height // 2) // old_height)
            else:
                new_width, new_height = dimension(width), dimension(height)
            layer.size = (new_width, new_height)
        return self._edit('Resize layer', apply)

    def update_layer(self, layer_id: str, **changes) -> bool:
        """Set layer properties together as one undoable edit."""
        allowed = {'name', 'visible', 'opacity', 'blend_mode', 'origin',
                   'size', 'flip_x', 'flip_y', 'sampling', 'mask_enabled',
                   'rotation', 'parent_id', 'mask_source_id', 'adjustment',
                   'effects'}
        if set(changes) - allowed:
            raise ValueError('unsupported layer properties')

        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            for key, value in changes.items():
                setattr(layer, key, deepcopy(value))
        return self._edit('Change layer', apply)

    def fit_layer(self, layer_id: str) -> bool:
        """Fit the current layer proportions inside the canvas and center it.

        Rounded to whole pixels, with any odd spare pixel on the right/bottom.
        Size and placement change together; native pixels and masks are shared.
        """
        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            width, height = layer.size
            if layer.rotation % 360:
                from .composite import _rotation_geometry
                angle = math.radians(layer.rotation % 360)
                c, s = abs(math.cos(angle)), abs(math.sin(angle))
                factor = min(doc.width / (c * width + s * height),
                             doc.height / (s * width + c * height))
                # Reserve a pixel for centering/rotation rounding when needed.
                fitted_width = max(1, int(math.floor(width * factor)))
                fitted_height = max(1, int(math.floor(height * factor)))
                while True:
                    layer.size = (fitted_width, fitted_height)
                    layer.origin = ((doc.width - fitted_width) // 2,
                                    (doc.height - fitted_height) // 2)
                    _, bounds, (x, y) = _rotation_geometry(layer)
                    if x >= 0 and y >= 0 and x + bounds[0] <= doc.width and y + bounds[1] <= doc.height:
                        return
                    if fitted_width == fitted_height == 1:
                        raise ValueError('Canvas is too small to contain the rotated layer')
                    ratio = min((fitted_width - 1) / fitted_width if fitted_width > 1 else 1,
                                (fitted_height - 1) / fitted_height if fitted_height > 1 else 1)
                    fitted_width, fitted_height = max(1, int(fitted_width * ratio)), max(1, int(fitted_height * ratio))
            if width * doc.height >= height * doc.width:
                fitted_width = doc.width
                fitted_height = max(1, (doc.width * height + width // 2) // width)
            else:
                fitted_height = doc.height
                fitted_width = max(1, (doc.height * width + height // 2) // height)
            layer.size = (fitted_width, fitted_height)
            layer.origin = ((doc.width - fitted_width) // 2,
                            (doc.height - fitted_height) // 2)
        return self._edit('Fit layer to canvas', apply)

    def set_mask(self, layer_id: str, mask: Optional[np.ndarray],
                 enabled: bool = True, placement: Optional[dict] = None) -> bool:
        """``placement`` re-attaches an unlinked mask placement that painting
        and other coverage edits keep, unlike a fresh :meth:`set_mask`."""
        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            layer.set_mask(mask, enabled)
            layer.mask_placement = placement
        return self._edit('Edit mask', apply)

    def replace_pixels(self, layer_id: str, pixels: np.ndarray,
                       label: str = 'Replace pixels') -> bool:
        return self._edit(label, lambda doc: doc.layer(layer_id).replace_pixels(pixels))

    def update_shape(self, layer_id: str, style: dict) -> bool:
        """Redraw a shape layer from a new style as one undoable edit.

        The pixels and the ``shape`` style change together, so undo returns
        both (upstream's redraw is likewise part of the size edit).
        """
        from .shapes import redraw_shape, validate_shape_style
        validate_shape_style(style)

        def apply(doc: Document) -> None:
            layer = doc.layer(layer_id)
            pixels = redraw_shape(layer, style)
            if pixels is None:
                raise ValueError('Select a shape layer first')
            layer.pixels = pixels
            layer.shape = deepcopy(style)
        return self._edit('Change shape', apply)

    def copy_selection_as_layer(self, selection, source_id=None, *, cut=False) -> bool:
        """Copy the visible composite inside a selection as a new layer above
        the active one; ``cut`` also clears the selected pixels of ``source_id``.

        Upstream Cmd-J (Layer via Copy) draws the selection's pixels through the
        selection clip — a feathered edge becomes partial alpha
        (SelectionClipboard.swift, renderSelectedPixels/renderMergedPixels) —
        and Cmd-X Cut is that copy plus clearSelectedPixels on the active
        layer, leaving transparent holes, never a content fill. The region is
        the selection's whole-pixel bounds, clamped to the canvas by the
        selection mask itself. One undo step; the edit drops the selection,
        as upstream's addPixelLayer does. An absent or empty selection raises,
        like crop-to-selection.
        """
        from .composite import composite
        from .layer_paint import selection_on_layer
        from .selections import selection_bbox

        def apply(doc: Document) -> None:
            if (selection is None or selection.shape != (doc.height, doc.width)
                    or not selection.any()):
                raise ValueError('Create a selection first')
            box = selection_bbox(selection)
            if box is None:
                raise ValueError('Create a selection first')
            x, y, w, h = box
            copy_pixels = composite(doc)[y:y + h, x:x + w].copy()
            # Straight alpha scaled by the selection coverage, feather kept.
            weight = selection[y:y + h, x:x + w].astype(np.float32) / 255.0
            copy_pixels[..., 3] = np.rint(
                copy_pixels[..., 3] * weight).astype(np.uint8)
            copy_pixels[copy_pixels[..., 3] == 0, :3] = 0
            if cut:
                source = doc.layer(source_id)
                if source.is_group or source.adjustment is not None:
                    raise ValueError('Folders and adjustment layers have no '
                                     'pixels to cut. Select an image layer first.')
                if source.text is not None or source.shape is not None:
                    raise ValueError('Cutting would flatten this text or shape '
                                     'layer. Copy it instead, or add a paint layer.')
                if source.pixels is None:
                    raise ValueError('This layer is empty; there is nothing to cut.')
                coverage = selection_on_layer(selection, source)
                cleared = source.pixels.copy()
                cleared[..., 3] = np.rint(
                    cleared[..., 3] * (1.0 - coverage)).astype(np.uint8)
                cleared[cleared[..., 3] == 0, :3] = 0
                source.pixels = rgba_from(cleared)
                # The recipe source no longer matches the edited pixels.
                source.recipe_source = source.recipe_settings = None
            names = {item.name for item in doc.layers}
            number = 1
            while f'Layer {number}' in names:
                number += 1
            new = Layer(f'Layer {number}', pixels=copy_pixels, origin=(x, y))
            active = doc.layer(doc.active_layer_id) if doc.active_layer_id else None
            if active is not None:
                new.parent_id = active.id if active.is_group else active.parent_id
                index = doc.index_of(active.id) + 1
            else:
                index = None
            doc.add_layer(new, index)
        return self._edit('Cut selection as layer' if cut else 'Copy selection as layer', apply)

    def replace_recipe_result(self, layer_id, pixels, source, settings):
        """One undo step; preserve the target's mask, placement and properties."""
        def apply(doc):
            layer = doc.layer(layer_id)
            layer.replace_pixels(pixels)
            layer.recipe_source = rgba_from(source)
            layer.recipe_settings = deepcopy(settings)
        return self._edit('Regenerate recipe layer', apply)

    def canvas_resize(self, width, height, *, anchor=4, content_offset=None,
                      label='Canvas Size') -> bool:
        """Re-square the canvas; layers, masks and guides keep their pixels.

        ``anchor`` is the 3x3 numpad position the existing content pins to
        (upstream ``ResizeViewModel.anchor``); ``content_offset`` overrides it
        with an explicit document-space shift. A no-change call records
        nothing. Layers fully outside the new canvas are kept, shifted.
        """
        for name, value in (('width', width), ('height', height)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or not 1 <= value <= MAX_SIDE):
                raise ValueError(f'Canvas {name} must be a whole number of '
                                 f'pixels within 1..{MAX_SIDE}')
        if width * height > MAX_SURFACE_PIXELS:
            raise ValueError('Canvas exceeds the surface budget')
        if content_offset is None:
            offset = None
        else:
            if (len(tuple(content_offset)) != 2
                    or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                               and math.isfinite(v) and abs(v) <= 1_000_000
                               for v in content_offset)):
                raise ValueError('Content offset must be two finite offsets '
                                 'within ±1,000,000 pixels')
            offset = tuple(int(v) for v in content_offset)
        if not (isinstance(anchor, int) and not isinstance(anchor, bool)
                and 0 <= anchor <= 8):
            raise ValueError('Anchor must be a numpad position 0..8')

        def apply(doc: Document) -> None:
            shift = offset if offset is not None else (
                ((width - doc.width) * (anchor % 3)) // 2,
                ((height - doc.height) * (anchor // 3)) // 2)
            if (doc.width, doc.height) == (width, height) and shift == (0, 0):
                return
            doc.width, doc.height = width, height
            for layer in doc.layers:
                layer.origin = (layer.origin[0] + shift[0],
                                layer.origin[1] + shift[1])
                if layer.mask_placement is not None:
                    origin = list(layer.mask_placement['origin'])
                    origin[0] += shift[0]
                    origin[1] += shift[1]
                    layer.mask_placement['origin'] = origin
            if doc.guides:
                for guide in doc.guides:
                    guide['position'] += (shift[0] if guide['axis'] == 'vertical'
                                          else shift[1])
        return self._edit(label, apply)

    def crop_document(self, x, y, width, height) -> bool:
        """Crop to a document-space rectangle as one undo step."""
        for name, value in (('x', x), ('y', y)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or abs(value) > 1_000_000):
                raise ValueError(f'Crop {name} must be finite and within ±1,000,000')
        for name, value in (('width', width), ('height', height)):
            if (not isinstance(value, (int, float)) or isinstance(value, bool)
                    or not math.isfinite(value) or value != int(value)
                    or not 1 <= int(value) <= MAX_SIDE):
                raise ValueError(f'Crop {name} must be a whole number of '
                                 f'pixels within 1..{MAX_SIDE}')
        if int(width) * int(height) > MAX_SURFACE_PIXELS:
            raise ValueError('Crop exceeds the surface budget')
        return self.canvas_resize(int(width), int(height),
                                  content_offset=(-int(round(x)), -int(round(y))),
                                  label='Crop')

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
