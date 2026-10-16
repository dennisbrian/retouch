"""Document and layer records for the editor core.

Mirrors the subset of Compositor's ``ProjectManifest`` / ``ProjectLayerRecord``
(``Compositor/IO/ProjectStore.swift``): pixel, blank, folder and supported
adjustment layers, live clipping links, hierarchy and order, visibility, opacity,
blend-mode name, integer placement, resizing, flips, center rotation and
8-bit raster masks. Anything else is refused with :class:`UnsupportedFeature` rather than
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

# Every name the format accepts (docs/writing-comp-files.md). Rendering supports
# the subset in composite.RENDER_BLEND_MODES; the rest round-trip unchanged.
BLEND_MODES = (
    'Normal', 'Darken', 'Multiply', 'Color Burn', 'Linear Burn', 'Lighten',
    'Screen', 'Color Dodge', 'Linear Dodge (Add)', 'Overlay', 'Soft Light',
    'Hard Light', 'Vivid Light', 'Linear Light', 'Pin Light', 'Hard Mix',
    'Difference', 'Exclusion', 'Subtract', 'Divide', 'Hue', 'Saturation',
    'Color', 'Luminosity',
)
SAMPLING = ('High quality', 'Smooth', 'Nearest')

# LayerEffects: the subset of Compositor's LayerEffects this port renders
# (``Compositor/Document/LayerEffects.swift``). Stroke, drop shadow, colour overlay,
# inner shadow, outer glow and inner glow are modelled; anything else is refused
# rather than dropped. ``enabled`` is upstream's optional flag: missing means
# visible, so it round-trips as absent unless a project turns it off.


class StrokeEffect:
    """A square-reach line inside or outside the layer's shown alpha."""
    __slots__ = ('enabled', 'size', 'red', 'green', 'blue', 'opacity', 'inside')

    def __init__(self, size=4, red=0, green=0, blue=0, opacity=1,
                 inside=False, enabled=True):
        self.enabled = enabled
        self.size = float(size)
        self.red = float(red)
        self.green = float(green)
        self.blue = float(blue)
        self.opacity = float(opacity)
        self.inside = inside

    def __eq__(self, other):
        return (isinstance(other, StrokeEffect)
                and all(getattr(self, key) == getattr(other, key) for key in self.__slots__))


class InnerShadowEffect:
    """A shadow cast inside the layer's own edges (upstream InnerShadowEffect)."""
    __slots__ = ('enabled', 'angle', 'distance', 'blur', 'red', 'green', 'blue', 'opacity')

    def __init__(self, angle=90, distance=10, blur=10,
                 red=0, green=0, blue=0, opacity=0.5, enabled=True):
        self.enabled = enabled
        self.angle = float(angle)
        self.distance = float(distance)
        self.blur = float(blur)
        self.red = float(red)
        self.green = float(green)
        self.blue = float(blue)
        self.opacity = float(opacity)

    @property
    def offset(self):
        """Where the shadow falls, in layer pixels (y grows downward)."""
        radians = math.radians(self.angle)
        return -math.cos(radians) * self.distance, math.sin(radians) * self.distance

    def __eq__(self, other):
        return (isinstance(other, InnerShadowEffect)
                and all(getattr(self, k) == getattr(other, k) for k in self.__slots__))


class _GlowEffect:
    """Shared shape for the omnidirectional glows (upstream has two structs)."""
    __slots__ = ('enabled', 'size', 'red', 'green', 'blue', 'opacity')

    def __init__(self, size, red, green, blue, opacity, enabled):
        self.enabled = enabled
        self.size = float(size)
        self.red = float(red)
        self.green = float(green)
        self.blue = float(blue)
        self.opacity = float(opacity)

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and all(getattr(self, k) == getattr(other, k) for k in self.__slots__))


class OuterGlowEffect(_GlowEffect):
    """A soft glow drawn around the outside of what the layer shows."""
    def __init__(self, size=20, red=1, green=1, blue=1, opacity=0.75, enabled=True):
        super().__init__(size, red, green, blue, opacity, enabled)


class InnerGlowEffect(_GlowEffect):
    """A glow emanating inward from the layer's boundary."""
    def __init__(self, size=10, red=1, green=1, blue=1, opacity=0.75, enabled=True):
        super().__init__(size, red, green, blue, opacity, enabled)


class ShadowEffect:
    """The layer's shape repeated behind it, offset and softened.

    ``angle`` is where the light comes from, counterclockwise from the right
    (90 is straight above), ``distance`` and ``blur`` are in layer pixels and
    ``opacity``/the colour are 0..1. Ranges match upstream ``isValid``.
    """
    __slots__ = ('enabled', 'angle', 'distance', 'blur', 'red', 'green', 'blue', 'opacity')

    def __init__(self, angle=90, distance=20, blur=20,
                 red=0, green=0, blue=0, opacity=0.5, enabled=True):
        self.enabled = enabled
        self.angle = float(angle)
        self.distance = float(distance)
        self.blur = float(blur)
        self.red = float(red)
        self.green = float(green)
        self.blue = float(blue)
        self.opacity = float(opacity)

    @property
    def offset(self):
        """Where the shadow sits, in layer pixels (y grows downward)."""
        radians = math.radians(self.angle)
        return -math.cos(radians) * self.distance, math.sin(radians) * self.distance

    def __eq__(self, other):
        return (isinstance(other, ShadowEffect)
                and self.__slots__ and all(getattr(self, k) == getattr(other, k)
                                           for k in self.__slots__))

    def __repr__(self):
        return ('ShadowEffect(angle=%r, distance=%r, blur=%r, red=%r, green=%r, '
                'blue=%r, opacity=%r, enabled=%r)'
                % (self.angle, self.distance, self.blur, self.red, self.green,
                   self.blue, self.opacity, self.enabled))


class ColorOverlayEffect:
    """A flat colour over everything the layer shows."""
    __slots__ = ('enabled', 'red', 'green', 'blue', 'opacity')

    def __init__(self, red=0, green=0, blue=0, opacity=1, enabled=True):
        self.enabled = enabled
        self.red = float(red)
        self.green = float(green)
        self.blue = float(blue)
        self.opacity = float(opacity)

    def __eq__(self, other):
        return (isinstance(other, ColorOverlayEffect)
                and all(getattr(self, k) == getattr(other, k) for k in self.__slots__))

    def __repr__(self):
        return ('ColorOverlayEffect(red=%r, green=%r, blue=%r, opacity=%r, enabled=%r)'
                % (self.red, self.green, self.blue, self.opacity, self.enabled))


class LayerEffects:
    """What a layer draws around itself; kept with the layer, never baked in."""
    __slots__ = ('stroke', 'shadow', 'colorOverlay', 'innerShadow', 'outerGlow', 'innerGlow')

    def __init__(self, stroke=None, shadow=None, colorOverlay=None,
                 innerShadow=None, outerGlow=None, innerGlow=None):
        self.stroke = stroke
        self.shadow = shadow
        self.colorOverlay = colorOverlay
        self.innerShadow = innerShadow
        self.outerGlow = outerGlow
        self.innerGlow = innerGlow

    def is_empty(self):
        return (self.stroke is None and self.shadow is None and self.colorOverlay is None
                and self.innerShadow is None and self.outerGlow is None
                and self.innerGlow is None)

    def __eq__(self, other):
        return (isinstance(other, LayerEffects)
                and self.stroke == other.stroke and self.shadow == other.shadow
                and self.colorOverlay == other.colorOverlay
                and self.innerShadow == other.innerShadow
                and self.outerGlow == other.outerGlow
                and self.innerGlow == other.innerGlow)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __bool__(self):
        return not self.is_empty()

    def __repr__(self):
        return ('LayerEffects(stroke=%r, shadow=%r, colorOverlay=%r, '
                'innerShadow=%r, outerGlow=%r, innerGlow=%r)'
                % (self.stroke, self.shadow, self.colorOverlay,
                   self.innerShadow, self.outerGlow, self.innerGlow))


def _effect_number(value, low, high):
    return (type(value) in (int, float) and not isinstance(value, bool)
            and math.isfinite(value) and low <= value <= high)


def validate_effects(effects):
    """Check a :class:`LayerEffects` against upstream's ``isValid`` ranges."""
    if not isinstance(effects, LayerEffects):
        raise ValueError('effects must be a LayerEffects record')
    stroke = effects.stroke
    if stroke is not None:
        if not isinstance(stroke, StrokeEffect):
            raise ValueError('stroke must be a StrokeEffect record')
        if (type(stroke.enabled) is not bool or type(stroke.inside) is not bool
                or not (_effect_number(stroke.size, 0, 500)
                        and _effect_number(stroke.opacity, 0, 1)
                        and all(_effect_number(v, 0, 1) for v in
                                (stroke.red, stroke.green, stroke.blue)))):
            raise ValueError('stroke settings out of range')
    shadow = effects.shadow
    if shadow is not None:
        if not isinstance(shadow, ShadowEffect):
            raise ValueError('shadow must be a ShadowEffect record')
        if (type(shadow.enabled) is not bool
                or not (_effect_number(shadow.angle, -360, 360)
                and _effect_number(shadow.distance, 0, 5000)
                and _effect_number(shadow.blur, 0, 500)
                and _effect_number(shadow.opacity, 0, 1)
                and all(_effect_number(v, 0, 1) for v in
                        (shadow.red, shadow.green, shadow.blue)))):
            raise ValueError('shadow settings out of range')
    overlay = effects.colorOverlay
    if overlay is not None:
        if not isinstance(overlay, ColorOverlayEffect):
            raise ValueError('colorOverlay must be a ColorOverlayEffect record')
        if (type(overlay.enabled) is not bool
                or not (_effect_number(overlay.opacity, 0, 1)
                and all(_effect_number(v, 0, 1) for v in
                        (overlay.red, overlay.green, overlay.blue)))):
            raise ValueError('colorOverlay settings out of range')
    inner_shadow = effects.innerShadow
    if inner_shadow is not None:
        if not isinstance(inner_shadow, InnerShadowEffect):
            raise ValueError('innerShadow must be an InnerShadowEffect record')
        if (type(inner_shadow.enabled) is not bool
                or not (_effect_number(inner_shadow.angle, -360, 360)
                and _effect_number(inner_shadow.distance, 0, 5000)
                and _effect_number(inner_shadow.blur, 0, 500)
                and _effect_number(inner_shadow.opacity, 0, 1)
                and all(_effect_number(v, 0, 1) for v in
                        (inner_shadow.red, inner_shadow.green, inner_shadow.blue)))):
            raise ValueError('innerShadow settings out of range')
    for glow, name in ((effects.outerGlow, 'outerGlow'), (effects.innerGlow, 'innerGlow')):
        if glow is not None:
            if not isinstance(glow, _GlowEffect):
                raise ValueError(f'{name} must be a glow effect record')
            if (type(glow.enabled) is not bool
                    or not (_effect_number(glow.size, 0, 500)
                    and _effect_number(glow.opacity, 0, 1)
                    and all(_effect_number(v, 0, 1) for v in
                            (glow.red, glow.green, glow.blue)))):
                raise ValueError(f'{name} settings out of range')

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
    effects: LayerEffects = None
    origin: tuple = (0, 0)         # document pixels, top-left
    size: tuple = None             # rendered rectangle; source pixels stay native
    flip_x: bool = False
    flip_y: bool = False
    sampling: str = 'High quality'
    mask: object = None            # read-only uint8 (h, w): white reveals
    mask_enabled: bool = True
    # A mask moved apart from its layer (upstream ``maskPlacement``): its own
    # document placement while it is unlinked. None = linked, spans the layer.
    mask_placement: dict = None
    mask_linked: bool = True
    # Kept verbatim: their PNG is authoritative. Dropped when pixels change.
    text: dict = None
    shape: dict = None
    rotation: float = 0           # clockwise about the unrotated rectangle center
    parent_id: str = None
    is_group: bool = False        # pass-through folder, never an isolated surface
    mask_source_id: str = None    # live alpha link; adjacent siblings form a clipping stack
    adjustment: dict = None
    recipe_source: object = None  # immutable original composite; Retouch extension
    recipe_settings: dict = None

    def __post_init__(self):
        if self.pixels is not None:
            self.pixels = rgba_from(self.pixels)
            if self.size is None:
                self.size = (self.pixels.shape[1], self.pixels.shape[0])
        if self.recipe_source is not None:
            self.recipe_source = rgba_from(self.recipe_source)
        if self.mask is not None:
            self.mask = _frozen(self.mask, 2)
        if self.size is None:
            raise ValueError('a blank layer needs a transform size')
        # Upstream keeps typed fractional placements exact (LayerTransform.swift:
        # "a fraction can still be asked for by hand"); only dragging snaps.
        self.origin = tuple(float(v) for v in self.origin)
        self.size = tuple(int(v) for v in self.size)

    def replace_pixels(self, pixels):
        """Swap in new pixel content (same placement); drops text/shape styles."""
        pixels = rgba_from(pixels)
        source_size = ((self.pixels.shape[1], self.pixels.shape[0])
                       if self.pixels is not None else self.size)
        if (pixels.shape[1], pixels.shape[0]) != source_size:
            raise ValueError('replacement pixels must match the source size')
        self.pixels, self.text, self.shape = pixels, None, None
        self.recipe_source = self.recipe_settings = None

    def set_mask(self, mask, enabled=True):
        """Attach an 8-bit coverage mask (None removes it).

        A fresh mask spans the layer's own rectangle again (upstream's
        placement collapses to nil), so any unlinked placement goes with it.
        """
        self.mask = None if mask is None else _frozen(mask, 2)
        self.mask_enabled = bool(enabled)
        self.mask_placement = None

    def validate(self):
        if (self.recipe_source is None) != (self.recipe_settings is None):
            raise ValueError('Recipe source and settings must be stored together')
        if self.recipe_settings is not None:
            from .recipe_state import validate_state
            validate_state(self.recipe_settings)
            if (self.pixels is None or self.is_group or self.adjustment is not None
                    or self.recipe_source.shape != self.pixels.shape
                    or not (self.recipe_source[..., 3] == 255).all()):
                raise ValueError('Recipe source must match an opaque raster layer')
        if (not isinstance(self.rotation, (int, float)) or isinstance(self.rotation, bool)
                or not math.isfinite(self.rotation)):
            raise ValueError('layer rotation must be finite')
        if not isinstance(self.is_group, bool):
            raise ValueError('is_group must be boolean')
        if self.is_group and (self.pixels is not None or self.adjustment is not None
                              or self.mask_source_id is not None):
            raise ValueError('folders cannot contain pixels, adjustments or clipping links')
        if self.mask_placement is not None:
            p = self.mask_placement
            if (not isinstance(p, dict)
                    or set(p) != {'origin', 'size', 'rotation', 'flipX', 'flipY', 'sampling'}
                    or len(tuple(p['origin'])) != 2 or len(tuple(p['size'])) != 2
                    or not all(isinstance(p[k], bool) for k in ('flipX', 'flipY'))
                    or not isinstance(p['sampling'], str)):
                raise ValueError('mask placement must be a full transform record')
            # Port-only guard: upstream clips through the layer's own grid
            # (LayerMask.clipImage) and never materializes the placement, but
            # this port resamples the mask to the placement size first, so
            # the placement itself must stay finite and inside the budget.
            if (not all(math.isfinite(v) for v in tuple(p['origin']) + tuple(p['size']))
                    or not math.isfinite(p['rotation'])
                    or not all(1 <= v <= 300_000 for v in tuple(p['size']))
                    or p['size'][0] * p['size'][1] > MAX_SURFACE_PIXELS):
                raise ValueError('mask placement out of range')
            if self.is_group:
                raise ValueError('folder masks cannot be unlinked')
        if self.adjustment is not None:
            from .adjustments import validate_adjustment
            validate_adjustment(self.adjustment)
            if self.pixels is not None:
                raise ValueError('adjustment layers cannot contain pixels')
        if self.effects is not None:
            validate_effects(self.effects)
            if self.is_group or self.adjustment is not None or self.mask_source_id is not None:
                raise ValueError('layer effects need a plain raster layer')
        if not self.name.strip() or len(self.name.encode('utf-8')) > 16_384:
            raise ValueError('layer name must be non-blank and at most 16 KiB')
        if not (math.isfinite(self.opacity) and 0 <= self.opacity <= 1):
            raise ValueError('opacity must be finite and within 0..1')
        if self.blend_mode not in BLEND_MODES:
            raise ValueError(f'unknown blend mode {self.blend_mode!r}')
        if self.sampling not in SAMPLING:
            raise ValueError(f'unknown sampling {self.sampling!r}')
        if not all(math.isfinite(v) for v in self.origin):
            raise ValueError('layer origin must be finite')
        if any(abs(v) > 1_000_000 for v in self.origin):
            raise ValueError('layer origin out of range')
        if not all(1 <= v <= 300_000 for v in self.size):
            raise ValueError('layer size out of range')
        if (self.pixels is not None or self.mask is not None) and any(v > MAX_SIDE for v in self.size):
            raise ValueError('rendered layer side exceeds %d px' % MAX_SIDE)


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
        # Removing a folder keeps its children and their order.
        for child in self.layers:
            if child.parent_id == layer_id:
                child.parent_id = layer.parent_id
            if child.mask_source_id == layer_id:
                child.mask_source_id = None
        self.layers.remove(layer)
        if self.active_layer_id == layer_id:
            self.active_layer_id = self.layers[-1].id if self.layers else None
        return layer

    def ancestors(self, layer):
        by_id = {item.id: item for item in self.layers}
        result = []
        while layer.parent_id is not None:
            layer = by_id[layer.parent_id]
            result.append(layer)
        return result

    def ordered_layers(self):
        children = {}
        for layer in self.layers:
            children.setdefault(layer.parent_id, []).append(layer)
        result = []
        def visit(parent):
            for layer in children.get(parent, []):
                result.append(layer)
                if layer.is_group:
                    visit(layer.id)
        visit(None)
        return result

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
        image_pixels = mask_pixels = rendered_pixels = 0
        for layer in self.layers:
            layer.validate()
            uuid.UUID(layer.id)
            if layer.id != layer.id.upper() or layer.id in ids:
                raise ValueError('layer ids must be unique uppercase UUIDs')
            ids.add(layer.id)
            if layer.pixels is not None:
                image_pixels += layer.pixels.shape[0] * layer.pixels.shape[1]
                from .composite import _rotation_geometry
                _, bounds, _ = _rotation_geometry(layer)
                rendered_pixels += bounds[0] * bounds[1]
            if layer.recipe_source is not None:
                image_pixels += layer.recipe_source.shape[0] * layer.recipe_source.shape[1]
            if layer.mask is not None:
                mask_pixels += layer.mask.size
                if layer.pixels is None:
                    from .composite import _rotation_geometry
                    _, bounds, _ = _rotation_geometry(layer)
                    rendered_pixels += bounds[0] * bounds[1]
        by_id = {layer.id: layer for layer in self.layers}
        for layer in self.layers:
            seen = {layer.id}
            parent = layer.parent_id
            while parent is not None:
                if parent in seen or len(seen) > 64 or parent not in by_id or not by_id[parent].is_group:
                    raise ValueError('invalid folder hierarchy')
                seen.add(parent)
                parent = by_id[parent].parent_id
            seen = {layer.id}
            source = layer.mask_source_id
            while source is not None:
                if source in seen or len(seen) > 256 or source not in by_id:
                    raise ValueError('invalid or cyclic clipping link')
                node = by_id[source]
                if node.is_group or node.adjustment is not None:
                    raise ValueError('clipping source must be a pixel or blank layer')
                seen.add(source)
                source = node.mask_source_id
        if max(image_pixels, mask_pixels, rendered_pixels) > MAX_SURFACE_PIXELS:
            raise ValueError('document exceeds the %d MP layer budget'
                             % (MAX_SURFACE_PIXELS // 1_000_000))
        if self.active_layer_id is not None and self.active_layer_id not in ids:
            raise ValueError('active layer is not in the document')
