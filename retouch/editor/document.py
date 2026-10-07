"""Document and layer records for the editor core.

Mirrors the subset of Compositor's ``ProjectManifest`` / ``ProjectLayerRecord``
(``Compositor/IO/ProjectStore.swift``) that milestone 2 composites: ungrouped
pixel and blank layers, bottom-to-top order, visibility, opacity, blend-mode
name, an untransformed placement (integer origin, flips) and an 8-bit raster
mask. Anything else is refused with :class:`UnsupportedFeature` rather than
dropped. Pixel and mask arrays are read-only: an edit replaces the array, so
the same buffers can later back undo history without copies.

Derived from Compositor (MIT, Copyright (c) 2026 Wonder Assembly LLC), pinned
at 11d8d7a; see ``third_party/compositor/``.
"""
import math
import uuid
from dataclasses import dataclass, field

import numpy as np

# DocumentLimits.swift. The per-document pixel budget upstream scales with the
# Mac (200-800 MP); 200 MP is its floor, so every upstream build opens what
# this writes. Images and masks are counted separately, as ProjectStore does.
MAX_SIDE = 30_000
MAX_SURFACE_PIXELS = 200_000_000
MAX_LAYERS = 10_000

# Every name the format accepts (docs/writing-comp-files.md). Only Normal is
# composited so far; the rest are stored and round-tripped unchanged.
BLEND_MODES = (
    'Normal', 'Darken', 'Multiply', 'Color Burn', 'Linear Burn', 'Lighten',
    'Screen', 'Color Dodge', 'Linear Dodge (Add)', 'Overlay', 'Soft Light',
    'Hard Light', 'Vivid Light', 'Linear Light', 'Pin Light', 'Hard Mix',
    'Difference', 'Exclusion', 'Subtract', 'Divide', 'Hue', 'Saturation',
    'Color', 'Luminosity',
)
SAMPLING = ('High quality', 'Smooth', 'Nearest')


class UnsupportedFeature(ValueError):
    """A valid Compositor construct this port cannot represent or render yet."""


def new_id():
    """An uppercase UUID string, the form the format requires."""
    return str(uuid.uuid4()).upper()


def _frozen(array, ndim, channels=None):
    array = np.asarray(array)
    if array.dtype != np.uint8 or array.ndim != ndim or (
            channels is not None and array.shape[2] != channels):
        shape = 'HxWx%d' % channels if channels else 'HxW'
        raise ValueError(f'expected a uint8 {shape} array, got {array.dtype} {array.shape}')
    if not (1 <= array.shape[0] <= MAX_SIDE and 1 <= array.shape[1] <= MAX_SIDE):
        raise ValueError('array side outside 1..%d px' % MAX_SIDE)
    if not _immutable(array):
        array = np.array(array, order='C', copy=True)
        array.flags.writeable = False
    return array


def _immutable(array):
    """True when no array in the view chain can still write these bytes."""
    if not array.flags.c_contiguous:
        return False
    while isinstance(array, np.ndarray):
        if array.flags.writeable:
            return False
        array = array.base
    return True


def rgba_from(pixels):
    """uint8 RGB or RGBA (H, W, C) -> read-only straight-alpha RGBA."""
    pixels = np.asarray(pixels)
    if pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3:
        alpha = np.full(pixels.shape[:2] + (1,), 255, np.uint8)
        pixels = np.concatenate([pixels, alpha], axis=2)
    return _frozen(pixels, 3, 4)


@dataclass(eq=False)
class Layer:
    """One record of the stack. ``pixels`` is straight-alpha sRGB RGBA."""
    name: str
    pixels: object = None          # read-only uint8 (H, W, 4) or None (blank)
    id: str = field(default_factory=new_id)
    visible: bool = True
    opacity: float = 1.0
    blend_mode: str = 'Normal'
    origin: tuple = (0, 0)         # document pixels, top-left
    size: tuple = None             # transform size; image layers: image size
    flip_x: bool = False
    flip_y: bool = False
    sampling: str = 'High quality'
    mask: object = None            # read-only uint8 (h, w): white reveals
    mask_enabled: bool = True
    # Kept verbatim: their PNG is authoritative. Dropped when pixels change.
    text: dict = None
    shape: dict = None

    def __post_init__(self):
        if self.pixels is not None:
            self.pixels = rgba_from(self.pixels)
            if self.size is None:
                self.size = (self.pixels.shape[1], self.pixels.shape[0])
        if self.mask is not None:
            self.mask = _frozen(self.mask, 2)
        if self.size is None:
            raise ValueError('a blank layer needs a transform size')
        self.origin = tuple(int(v) for v in self.origin)
        self.size = tuple(int(v) for v in self.size)

    def replace_pixels(self, pixels):
        """Swap in new pixel content (same placement); drops text/shape styles."""
        pixels = rgba_from(pixels)
        if (pixels.shape[1], pixels.shape[0]) != self.size:
            raise ValueError('replacement pixels must match the layer size')
        self.pixels, self.text, self.shape = pixels, None, None

    def set_mask(self, mask, enabled=True):
        """Attach an 8-bit coverage mask (None removes it)."""
        self.mask = None if mask is None else _frozen(mask, 2)
        self.mask_enabled = bool(enabled)

    def validate(self):
        if not self.name.strip() or len(self.name.encode('utf-8')) > 16_384:
            raise ValueError('layer name must be non-blank and at most 16 KiB')
        if not (math.isfinite(self.opacity) and 0 <= self.opacity <= 1):
            raise ValueError('opacity must be finite and within 0..1')
        if self.blend_mode not in BLEND_MODES:
            raise ValueError(f'unknown blend mode {self.blend_mode!r}')
        if self.sampling not in SAMPLING:
            raise ValueError(f'unknown sampling {self.sampling!r}')
        if any(abs(v) > 1_000_000 for v in self.origin):
            raise ValueError('layer origin out of range')
        if not all(1 <= v <= 300_000 for v in self.size):
            raise ValueError('layer size out of range')
        if self.pixels is not None and (
                (self.pixels.shape[1], self.pixels.shape[0]) != self.size):
            # A stretched layer needs resampling that has not been ported.
            raise UnsupportedFeature(f'layer {self.name!r} is scaled; only 1:1 placement '
                                     'is supported')


@dataclass(eq=False)
class Document:
    """A canvas and its layers, bottom to top."""
    width: int
    height: int
    layers: list = field(default_factory=list)
    resolution: float = 72.0
    document_id: str = field(default_factory=new_id)
    active_layer_id: str = None
    guides: list = None            # stored verbatim (format v8+)

    def layer(self, layer_id):
        for layer in self.layers:
            if layer.id == layer_id:
                return layer
        raise KeyError(layer_id)

    def index_of(self, layer_id):
        return self.layers.index(self.layer(layer_id))

    def add_layer(self, layer, index=None):
        """Insert ``layer`` (top of the stack by default) and make it active."""
        if any(existing.id == layer.id for existing in self.layers):
            raise ValueError(f'duplicate layer id {layer.id}')
        self.layers.insert(len(self.layers) if index is None else index, layer)
        self.active_layer_id = layer.id
        return layer

    def add_image(self, pixels, name, origin=(0, 0), index=None):
        return self.add_layer(Layer(name=name, pixels=pixels, origin=origin), index)

    def remove_layer(self, layer_id):
        layer = self.layer(layer_id)
        self.layers.remove(layer)
        if self.active_layer_id == layer_id:
            self.active_layer_id = self.layers[-1].id if self.layers else None
        return layer

    def move_layer(self, layer_id, index):
        """Move a layer to ``index`` in the bottom-to-top order."""
        layer = self.layer(layer_id)
        self.layers.remove(layer)
        self.layers.insert(max(0, min(index, len(self.layers))), layer)

    def validate(self):
        if not (isinstance(self.width, int) and not isinstance(self.width, bool)
                and isinstance(self.height, int) and not isinstance(self.height, bool)
                and 1 <= self.width <= MAX_SIDE and 1 <= self.height <= MAX_SIDE):
            raise ValueError('canvas side outside 1..%d px' % MAX_SIDE)
        try:
            uuid.UUID(self.document_id)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError('document id must be a UUID string') from exc
        if not (math.isfinite(self.resolution) and 1 <= self.resolution <= 9600):
            raise ValueError('resolution must be within 1..9600 pixels/inch')
        if len(self.layers) > MAX_LAYERS:
            raise ValueError('more than %d layers' % MAX_LAYERS)
        ids = set()
        image_pixels = mask_pixels = 0
        for layer in self.layers:
            layer.validate()
            uuid.UUID(layer.id)
            if layer.id != layer.id.upper() or layer.id in ids:
                raise ValueError('layer ids must be unique uppercase UUIDs')
            ids.add(layer.id)
            if layer.pixels is not None:
                image_pixels += layer.pixels.shape[0] * layer.pixels.shape[1]
            if layer.mask is not None:
                mask_pixels += layer.mask.size
        if image_pixels > MAX_SURFACE_PIXELS or mask_pixels > MAX_SURFACE_PIXELS:
            raise ValueError('document exceeds the %d MP layer budget'
                             % (MAX_SURFACE_PIXELS // 1_000_000))
        if self.active_layer_id is not None and self.active_layer_id not in ids:
            raise ValueError('active layer is not in the document')
