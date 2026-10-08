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
    layers = [_prepare(layer) for layer in document.layers if _contributes(layer)]
    out = np.zeros((height, width, 4), np.uint8)
    for top in range(0, height, band_rows):
        bottom = min(height, top + band_rows)
        rgb = np.zeros((bottom - top, width, 3), np.float32)   # premultiplied
        alpha = np.zeros((bottom - top, width), np.float32)
        for pixels, coverage, (x0, y0), opacity, mode in layers:
            h, w = pixels.shape[:2]
            ys, ye = max(top, y0), min(bottom, y0 + h)
            xs, xe = max(0, x0), min(width, x0 + w)
            if ys >= ye or xs >= xe:
                continue
            src = pixels[ys - y0:ye - y0, xs - x0:xe - x0]
            a = src[..., 3].astype(np.float32) * np.float32(opacity / 255.0)
            if coverage is not None:
                a *= coverage[ys - y0:ye - y0, xs - x0:xe - x0].astype(np.float32) / 255.0
            keep = 1.0 - a
            region = rgb[ys - top:ye - top, xs:xe]
            under = alpha[ys - top:ye - top, xs:xe]
            if mode != 'Normal':
                source = src[..., :3].astype(np.float32) / 255.0
                backdrop = region / np.where(under > 0, under, 1.0)[..., None]
                # Blend straight colors only in the overlapping coverage.
                # W3C compositing-1: Cs' = (1 - ab) Cs + ab B(Cb, Cs).
                source = (1.0 - under[..., None]) * source + under[..., None] * (
                    _blend(backdrop, source, mode))
            region *= keep[..., None]
            if mode == 'Normal':
                region += src[..., :3].astype(np.float32) * (a[..., None] / 255.0)
            else:
                region += source * a[..., None]
            under *= keep
            under += a
        out[top:bottom] = _straight(rgb, alpha)
    return out


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
    return pixels, coverage, layer.origin, layer.opacity, layer.blend_mode


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
