"""Shape layers: port of Compositor's ShapeTool.swift (11d8d7a).

A shape layer's pixels are an ordinary raster (so they clip, mask, blend and
filter like any layer); the ``shape`` style dict is kept beside them on the
Layer so the shape can be redrawn — natively at a new size, here at any
parameter change. Rendering mirrors ``EditorSession.shapeImage``: fill for
Rectangle/Ellipse, a round-capped stroke for Line. Pillow's ImageDraw has no
anti-aliasing, so shapes draw at ``SUPERSAMPLE`` scale and come down with a
box filter, standing in for CGContext's own anti-aliasing.

Derived from Compositor (MIT, Copyright (c) 2026 Wonder Assembly LLC), pinned
at 11d8d7a; see ``third_party/compositor/``.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

SHAPE_KINDS = ('Rectangle', 'Ellipse', 'Line')
# AA quality knob: 4x supersampling keeps ellipse/line edges within ~1/4 of a
# pixel of a reference CGContext render without an area-coverage integrator.
SUPERSAMPLE = 4

# Sentinel so redraw_shape(layer) (redraw from the layer's own style) stays
# distinct from redraw_shape(layer, None) (an explicit no-style probe).
_UNSET = object()

_KEYS = {'kind', 'red', 'green', 'blue', 'cornerRadius'}
_LINE_KEYS = {'lineWidth', 'start', 'end'}


def shape_style(kind, red, green, blue, *, corner_radius=0.0, line_width=None,
                start=None, end=None):
    """Build a ``LayerShapeStyle`` record as the .comp manifest stores it."""
    if kind not in SHAPE_KINDS:
        raise ValueError('shape kind must be one of ' + ', '.join(SHAPE_KINDS))
    if kind == 'Line':
        if line_width is None or start is None or end is None:
            raise ValueError('a line needs lineWidth, start and end')
        style = {'kind': kind, 'red': float(red), 'green': float(green),
                 'blue': float(blue), 'cornerRadius': 0.0,
                 'lineWidth': float(line_width), 'start': list(start), 'end': list(end)}
    else:
        if line_width is not None:
            raise ValueError('only lines carry a lineWidth')
        style = {'kind': kind, 'red': float(red), 'green': float(green),
                 'blue': float(blue), 'cornerRadius': float(corner_radius)}
    validate_shape_style(style)
    return style


def validate_shape_style(style):
    """Match ProjectStore's gate: any dict survives; ours stays fully typed."""
    if not isinstance(style, dict):
        raise ValueError('shape style must be a record')
    if not {'kind', 'red', 'green', 'blue', 'cornerRadius'} <= set(style):
        raise ValueError('shape style needs kind, red, green, blue, cornerRadius')
    if style['kind'] not in SHAPE_KINDS:
        raise ValueError('unknown shape kind %r' % (style['kind'],))
    for key in ('red', 'green', 'blue'):
        value = style[key]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not 0 <= value <= 1):
            raise ValueError('shape colour channels are 0-1 floats')
    radius = style['cornerRadius']
    if (not isinstance(radius, (int, float)) or isinstance(radius, bool)
            or not math.isfinite(radius) or radius < 0):
        raise ValueError('shape cornerRadius must be a non-negative number')
    if style['kind'] == 'Line':
        if 'lineWidth' in style:
            width = style['lineWidth']
            if (not isinstance(width, (int, float)) or isinstance(width, bool)
                    or not math.isfinite(width) or width < 0):
                raise ValueError('shape lineWidth must be a non-negative number')
        for key in ('start', 'end'):
            if key in style:
                pair = style[key]
                if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                        or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                   and math.isfinite(v) for v in pair)):
                    raise ValueError(f'shape {key} is a pair of finite fractions')
    return style


def shape_image(style, width, height, *, supersample=SUPERSAMPLE, filled=True):
    """The shape filling its box as straight-alpha uint8 RGBA (h, w, 4).

    Port of ``EditorSession.shapeImage``: an ellipse is the ellipse of the
    box, a rectangle rounds by ``cornerRadius`` (clamped to half the shorter
    side, so a large radius makes a pill), and a line strokes between its
    stored end fractions with round caps at ``lineWidth`` thickness.
    """
    validate_shape_style(style)
    if not (isinstance(width, int) and isinstance(height, int)
            and 1 <= width and 1 <= height):
        raise ValueError('shape size must be positive integers')
    factor = max(1, int(supersample))
    w, h = width * factor, height * factor
    image = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(image)
    color = (int(round(style['red'] * 255)), int(round(style['green'] * 255)),
             int(round(style['blue'] * 255)))
    if style['kind'] == 'Line':
        # Port note: upstream strokes a possibly sub-pixel path; ImageDraw
        # rounds to whole pixels, so the port draws at supersample scale
        # where that quantization is factor-times finer.
        thickness = max(1, round(style.get('lineWidth', 0) * factor))
        inset = (min(thickness, w) / 2.0, min(thickness, h) / 2.0)
        default_start = (inset[0], inset[1])
        default_end = (w - inset[0], h - inset[1])
        start = tuple(p * w if i == 0 else p * h for i, p in
                      enumerate(style.get('start') or ())) if style.get('start') \
            else default_start
        end = tuple(p * w if i == 0 else p * h for i, p in
                    enumerate(style.get('end') or ())) if style.get('end') \
            else default_end
        draw.line([start, end], fill=255, width=thickness)
        radius = thickness / 2.0
        for point in (start, end):
            draw.ellipse([point[0] - radius, point[1] - radius,
                          point[0] + radius, point[1] + radius], fill=255)
    else:
        radius = 0.0
        stroke = None
        if not filled:
            stroke = max(1, round(style.get('lineWidth', 1) * factor)) \
                if style.get('lineWidth') else max(1, factor)
        if style['kind'] == 'Rectangle':
            radius = min(style.get('cornerRadius', 0.0) * factor, w / 2.0, h / 2.0)
        box = (0, 0, w - 1, h - 1)
        if stroke is None:
            if radius > 0:
                draw.rounded_rectangle(box, radius=radius, fill=255)
            elif style['kind'] == 'Ellipse':
                draw.ellipse(box, fill=255)
            else:
                draw.rectangle(box, fill=255)
        else:
            if radius > 0:
                draw.rounded_rectangle(box, radius=radius, outline=255, width=stroke)
            elif style['kind'] == 'Ellipse':
                draw.ellipse(box, outline=255, width=stroke)
            else:
                draw.rectangle(box, outline=255, width=stroke)
    coverage = np.asarray(image, np.float32)
    if factor > 1:
        coverage = coverage.reshape(height, factor, width, factor)
        coverage = coverage.mean(axis=(1, 3))
    alpha = np.clip(np.round(coverage), 0, 255).astype(np.uint8)
    rgb = np.empty((height, width, 3), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = color
    out = np.dstack((rgb, alpha))
    out.setflags(write=False)
    return out


def shape_layer(name, style, origin=(0, 0), **layer_kwargs):
    """A ``Layer`` holding the rendered shape plus its style record."""
    from .document import Layer
    size = layer_kwargs.pop('size', None)
    if size is None:
        raise ValueError('a shape layer needs its size')
    pixels = shape_image(style, size[0], size[1])
    return Layer(name, pixels=pixels, origin=origin, shape=style, **layer_kwargs)


def redraw_shape(layer, style=_UNSET):
    """Redraw a shape layer's pixels from its (or the given) style.

    Returns the new pixels array, or None when ``layer`` is not a live shape:
    it must carry a ``shape`` record and its pixels must still span the
    layer's own rectangle (upstream ``liveShape``: pixels once edited some
    other way are plain pixels and the style no longer rides along).
    """
    if style is _UNSET:
        style = layer.shape
        if style is None:
            return None
    if layer.pixels is None or layer.shape is None:
        return None
    if (layer.pixels.shape[1], layer.pixels.shape[0]) != tuple(layer.size):
        return None
    return shape_image(style, layer.size[0], layer.size[1])
