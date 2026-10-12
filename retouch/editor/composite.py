"""CPU reference compositor: separable blending with opacity and raster masks.

Semantics follow Compositor's export path (``IO/ImageExporter.swift`` and
``Rendering/LayerRenderer.swift``): a transparent sRGB canvas, layers drawn
bottom to top with source-over in gamma-encoded sRGB (no linearisation),
coverage = layer alpha x enabled mask x opacity, and a straight-alpha RGBA
result. Core Graphics keeps the canvas as premultiplied 8-bit and rounds after
every layer; this renderer keeps float32 between layers and rounds once, so it
can differ from the app by about one level per stacked translucent layer.
Opaque, fully revealed or fully hidden pixels match exactly.

Work happens in horizontal bands, so the float working set is bounded by
``band_rows`` x canvas width whatever the layer count or canvas height.

Derived from Compositor (MIT, Copyright (c) 2026 Wonder Assembly LLC), pinned
at 11d8d7a; see ``third_party/compositor/``.
"""
import os
import uuid
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .document import MAX_SURFACE_PIXELS, UnsupportedFeature

RENDER_BLEND_MODES = ('Normal', 'Multiply', 'Screen', 'Overlay', 'Difference')


def composite(document, band_rows=256):
    """Flatten ``document`` to a uint8 straight-alpha RGBA (H, W, 4) array."""
    document.validate()
    width, height = document.width, document.height
    if width * height > MAX_SURFACE_PIXELS:
        raise ValueError('canvas exceeds the %d MP export limit'
                         % (MAX_SURFACE_PIXELS // 1_000_000))
    renderer = _BandRenderer(document)
    out = np.zeros((height, width, 4), np.uint8)
    if not isinstance(band_rows, int) or isinstance(band_rows, bool) or band_rows < 1:
        raise ValueError('band_rows must be a positive integer')
    for top in range(0, height, band_rows):
        bottom = min(height, top + band_rows)
        rgb, alpha = renderer.render(top, bottom)
        out[top:bottom] = _straight(rgb, alpha)
    return out


def _over(rgb, alpha, source, a, mode):
    if mode != 'Normal':
        backdrop = rgb / np.where(alpha > 0, alpha, 1.0)[..., None]
        source = (1 - alpha[..., None]) * source + alpha[..., None] * _blend(backdrop, source, mode)
    rgb *= (1 - a)[..., None]
    rgb += source * a[..., None]
    alpha *= 1 - a
    alpha += a


class _BandRenderer:
    """Pass-through hierarchy and live clipping, with only band-sized scratch arrays."""
    def __init__(self, document):
        self.doc = document
        self.by_id = {l.id: l for l in document.layers}
        self.parents = {l.id: document.ancestors(l) for l in document.layers}
        self.visible = [l for l in document.ordered_layers() if not l.is_group
                        and l.visible and all(p.visible for p in self.parents[l.id])]
        self.prepared = {}
        self.stacks, self.stacked = {}, set()
        for i, base in enumerate(self.visible):
            if base.mask_source_id is not None or base.adjustment is not None:
                continue
            children = []
            for child in self.visible[i + 1:]:
                if child.mask_source_id != base.id or child.parent_id != base.parent_id:
                    break
                children.append(child)
            if children:
                self.stacks[base.id] = children
                self.stacked.update(l.id for l in children)
        for l in self.visible:
            if l.blend_mode not in RENDER_BLEND_MODES:
                raise UnsupportedFeature(f'blend mode {l.blend_mode!r} is not composited yet')

    def opacity(self, layer):
        value = layer.opacity
        for parent in self.parents[layer.id]:
            value *= parent.opacity
        return value

    def own(self, layer, top, bottom, mask_only=False):
        shape = (bottom - top, self.doc.width)
        rgb = np.zeros(shape + (3,), np.float32)
        alpha = np.zeros(shape, np.float32)
        if mask_only:
            if layer.mask is None or not layer.mask_enabled:
                return rgb, np.ones(shape, np.float32)
            key = (layer.id, True)
            if key not in self.prepared:
                self.prepared[key] = _prepare_mask(layer)
            pixels, origin = self.prepared[key]
            coverage = None
        else:
            if layer.pixels is None:
                return rgb, alpha
            key = (layer.id, False)
            if key not in self.prepared:
                self.prepared[key] = _prepare(layer)
            pixels, coverage, origin, _, _ = self.prepared[key]
        x0, y0 = origin
        h, w = pixels.shape[:2]
        ys, ye = max(top, y0), min(bottom, y0 + h)
        xs, xe = max(0, x0), min(self.doc.width, x0 + w)
        if ys < ye and xs < xe:
            src = pixels[ys - y0:ye - y0, xs - x0:xe - x0]
            region = (slice(ys - top, ye - top), slice(xs, xe))
            if mask_only:
                alpha[region] = src.astype(np.float32) / 255
            else:
                rgb[region] = src[..., :3].astype(np.float32) / 255
                a = src[..., 3].astype(np.float32) * (self.opacity(layer) / 255)
                if coverage is not None:
                    a *= coverage[ys-y0:ye-y0, xs-x0:xe-x0].astype(np.float32) / 255
                alpha[region] = a
        return rgb, alpha

    def folder_clip(self, layer, top, bottom):
        clip = np.ones((bottom - top, self.doc.width), np.float32)
        for parent in self.parents[layer.id]:
            if parent.mask is not None and parent.mask_enabled:
                clip *= self.own(parent, top, bottom, True)[1]
        return clip

    def render(self, top, bottom):
        shape = (bottom - top, self.doc.width)
        rgb, alpha = np.zeros(shape + (3,), np.float32), np.zeros(shape, np.float32)
        coverage_cache = {}
        def coverage(layer):
            if layer.id not in coverage_cache:
                a = self.own(layer, top, bottom)[1]
                if layer.mask_source_id:
                    a *= coverage(self.by_id[layer.mask_source_id])
                coverage_cache[layer.id] = a
            return coverage_cache[layer.id]
        def adjust(layer, dest, a, in_stack=False):
            from .adjustments import grade
            # Adjust colors in place; alpha remains exactly the captured coverage.
            straight = dest / np.where(a > 0, a, 1)[..., None]
            changed = grade(straight, layer.adjustment)
            if layer.blend_mode != 'Normal':
                changed = _blend(straight, changed, layer.blend_mode)
            weight = self.opacity(layer) * self.own(layer, top, bottom, True)[1]
            if not in_stack:
                weight *= self.folder_clip(layer, top, bottom)
            if layer.mask_source_id and not in_stack:
                weight *= coverage(self.by_id[layer.mask_source_id])
            dest[:] = (straight + (changed - straight) * weight[..., None]) * a[..., None]
        for layer in self.visible:
            if layer.id in self.stacked:
                continue
            if layer.adjustment is not None:
                adjust(layer, rgb, alpha)
                continue
            source, a = self.own(layer, top, bottom)
            if layer.id in self.stacks:
                # Native clipping stacks make the base opaque for internal blending,
                # then restore its alpha before placing the stack on the backdrop.
                stack_rgb, stack_alpha = source.copy(), np.ones(shape, np.float32)
                for child in self.stacks[layer.id]:
                    if child.adjustment is not None:
                        adjust(child, stack_rgb, stack_alpha, in_stack=True)
                    else:
                        child_rgb, child_a = self.own(child, top, bottom)
                        _over(stack_rgb, stack_alpha, child_rgb, child_a, child.blend_mode)
                source = stack_rgb
            elif layer.mask_source_id:
                a *= coverage(self.by_id[layer.mask_source_id])
            a *= self.folder_clip(layer, top, bottom)
            _over(rgb, alpha, source, a, layer.blend_mode)
        return rgb, alpha


def export_png(document, path):
    """Write the flattened document as an RGBA PNG (replaced atomically)."""
    path = Path(path)
    pixels = composite(document)
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex[:8]}.tmp')
    try:
        Image.fromarray(pixels, 'RGBA').save(
            temporary, format='PNG',
            dpi=(document.resolution, document.resolution))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _contributes(layer):
    if not layer.visible or layer.pixels is None or layer.opacity == 0:
        return False
    if layer.blend_mode not in RENDER_BLEND_MODES:
        raise UnsupportedFeature(f'blend mode {layer.blend_mode!r} (layer {layer.name!r}) is not composited yet')
    return True


def _prepare(layer):
    """Pixels and coverage in document orientation, plus placement."""
    pixels = layer.pixels
    if (pixels.shape[1], pixels.shape[0]) != layer.size:
        sampling = {'Nearest': Image.Resampling.NEAREST,
                    'Smooth': Image.Resampling.BILINEAR,
                    'High quality': Image.Resampling.LANCZOS}[layer.sampling]
        # Pillow resizes RGBA through premultiplied alpha, avoiding dark halos.
        pixels = np.asarray(Image.fromarray(pixels).resize(layer.size, sampling))
    coverage = None
    if layer.mask is not None and layer.mask_enabled:
        coverage = _fit_mask(layer.mask, pixels.shape[1], pixels.shape[0], layer.sampling)
    # A flip mirrors the layer's local rectangle; its mask follows the layer.
    if layer.flip_y:
        pixels = pixels[::-1]
        coverage = None if coverage is None else coverage[::-1]
    if layer.flip_x:
        pixels = pixels[:, ::-1]
        coverage = None if coverage is None else coverage[:, ::-1]
    pixels, coverage, origin = _rotate(pixels, coverage, layer)
    return pixels, coverage, origin, layer.opacity, layer.blend_mode


def _blend(backdrop, source, mode):
    """W3C separable formulas in the document's gamma-encoded sRGB space."""
    if mode == 'Multiply':
        return backdrop * source
    if mode == 'Screen':
        return backdrop + source - backdrop * source
    if mode == 'Overlay':
        return np.where(backdrop <= 0.5, 2 * backdrop * source,
                        1 - 2 * (1 - backdrop) * (1 - source))
    if mode == 'Difference':
        return np.abs(backdrop - source)
    raise UnsupportedFeature(f'blend mode {mode!r} is not composited yet')


def _rotation_geometry(layer):
    angle = np.deg2rad(layer.rotation % 360)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    # Snap cardinal angles to avoid an extra row from floating-point trig.
    c, s = (round(v) if abs(v - round(v)) < 1e-12 else v for v in (c, s))
    w, h = layer.size
    cx, cy = layer.origin[0] + w / 2, layer.origin[1] + h / 2
    ex, ey = (abs(c) * w + abs(s) * h) / 2, (abs(s) * w + abs(c) * h) / 2
    x, y = int(np.floor(cx - ex)), int(np.floor(cy - ey))
    width, height = int(np.ceil(cx + ex)) - x, int(np.ceil(cy + ey)) - y
    if max(width, height) > 30000 or width * height > MAX_SURFACE_PIXELS:
        raise ValueError('Rotated layer exceeds the rendered surface budget')
    matrix = np.array([[c, -s, cx - c * (w - 1) / 2 + s * (h - 1) / 2 - x - .5],
                       [s, c, cy - s * (w - 1) / 2 - c * (h - 1) / 2 - y - .5]], np.float64)
    return matrix, (width, height), (x, y)


def _rotate(pixels, coverage, layer):
    if layer.rotation % 360 == 0:
        return pixels, coverage, layer.origin
    matrix, size, origin = _rotation_geometry(layer)
    interpolation = cv2.INTER_NEAREST if layer.sampling == 'Nearest' else cv2.INTER_LINEAR
    if pixels.ndim == 2:
        return cv2.warpAffine(pixels, matrix, size, flags=interpolation), None, origin
    rgba = pixels.astype(np.float32) / 255
    rgba[..., :3] *= rgba[..., 3:4]
    warped = cv2.warpAffine(rgba, matrix, size, flags=interpolation)
    alpha = warped[..., 3]
    warped[..., :3] /= np.where(alpha > 0, alpha, 1)[..., None]
    warped[alpha == 0, :3] = 0
    pixels = np.rint(np.clip(warped, 0, 1) * 255).astype(np.uint8)
    if coverage is not None:
        coverage = cv2.warpAffine(np.asarray(coverage), matrix, size, flags=interpolation)
    return pixels, coverage, origin


def _prepare_mask(layer):
    mask = _fit_mask(layer.mask, *layer.size, layer.sampling)
    if layer.flip_y:
        mask = mask[::-1]
    if layer.flip_x:
        mask = mask[:, ::-1]
    mask, _, origin = _rotate(np.asarray(mask), None, layer)
    return mask, origin


def _fit_mask(mask, width, height, sampling):
    """A mask spans its layer's rectangle whatever its own pixel size."""
    if mask.shape == (height, width):
        return mask
    if mask.size == 1:
        return np.broadcast_to(mask.reshape(1, 1), (height, width))
    interpolation = cv2.INTER_NEAREST if sampling == 'Nearest' else cv2.INTER_LINEAR
    return cv2.resize(mask, (width, height), interpolation=interpolation)


def _straight(rgb, alpha):
    """Premultiplied float band -> straight-alpha uint8 RGBA."""
    a8 = np.rint(alpha * 255.0)
    safe = np.where(alpha > 0, alpha, 1.0)
    colour = np.rint(np.clip(rgb / safe[..., None], 0.0, 1.0) * 255.0)
    colour[a8 == 0] = 0
    return np.dstack([colour, a8]).astype(np.uint8)
