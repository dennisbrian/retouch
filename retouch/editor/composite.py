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

RENDER_BLEND_MODES = (
    'Normal', 'Darken', 'Multiply', 'Color Burn', 'Linear Burn', 'Lighten',
    'Screen', 'Color Dodge', 'Linear Dodge (Add)', 'Overlay', 'Soft Light',
    'Hard Light', 'Vivid Light', 'Linear Light', 'Pin Light', 'Hard Mix',
    'Difference', 'Exclusion', 'Subtract', 'Divide',
)

# --- Resampling -----------------------------------------------------------
#
# Native behaviour (Compositor's ``Rendering/DownsampleCache.swift`` and
# ``Rendering/LayerRenderer.swift``):
#   * Large reductions are prefiltred through a chain of exactly-2x halvings
#     made with high-quality (Lanczos) resampling, leaving Core Graphics only
#     the last, at-most-2x step. Each halving rounds up, so level ``k`` pixel
#     ``i`` covers source pixels ``i * 2**k .. (i + 1) * 2**k``: a piece of an
#     image reduced on its own lines up with the whole image reduced.
#   * An upright layer drawn pixel for pixel is copied across (nearest); a
#     shrink uses a low-quality filter (``LayerRenderer.interpolation``); an
#     enlarge and every rotated draw honour the layer's own sampling.
#   * Core Graphics' "low" filter is a 4-tap cubic, not bilinear: its impulse
#     response is 15/63/192/240 out of 512, i.e. a cubic with a ~= 0.5-class
#     kernel. OpenCV's bilinear (and area) kernels differ, which is why this
#     renderer uses ``cv2.INTER_CUBIC`` for the final step.
MAX_DOWNSAMPLE_LEVEL = 6


def _downsample_level(factor):
    """Halvings to draw from when an image lands ``factor`` output pixels per
    source pixel: the most that still leave the copy at least that large
    (0 from half size up)."""
    if not np.isfinite(factor) or not 0 < factor < 0.5:
        return 0
    return min(MAX_DOWNSAMPLE_LEVEL, int(np.floor(np.log2(1.0 / factor))))


def _halve(array, mask=False):
    """Exactly half the size, rounded up, Lanczos resampling.

    Colour is resampled premultiplied (as ``DownsampleCache.halve`` does with
    premultiplied ARGB) so a straight-alpha edge cannot pick up a dark halo,
    and ringing past a pixel's alpha is clamped so edges do not glow. Gray
    masks repeat an odd last row or column; colour pads with transparent
    pixels, matching upstream."""
    height, width = array.shape[:2]
    next_h, next_w = (height + 1) // 2, (width + 1) // 2
    if mask:
        padded = np.zeros((next_h * 2, next_w * 2), np.uint8)
        padded[:height, :width] = array
        if next_w * 2 > width:
            padded[:height, width] = array[:, width - 1]
        if next_h * 2 > height:
            padded[height, :next_w * 2] = padded[height - 1, :next_w * 2]
        return cv2.resize(padded, (next_w, next_h), interpolation=cv2.INTER_LANCZOS4)
    pad = 8
    channels = array.shape[2:]
    padded = np.zeros((height + 2 * pad, width + 2 * pad) + channels, np.float32)
    placed = array.astype(np.float32)
    if channels == (4,):
        placed = placed.copy()
        placed[..., :3] *= placed[..., 3:4] / 255.0
    padded[pad:pad + height, pad:pad + width] = placed
    reduced = cv2.resize(padded, ((width + 2 * pad) // 2, (height + 2 * pad) // 2),
                         interpolation=cv2.INTER_LANCZOS4)
    cropped = reduced[pad // 2:pad // 2 + next_h, pad // 2:pad // 2 + next_w]
    if channels == (4,):
        alpha = cropped[..., 3:4]
        # Premultiplied colour cannot exceed its alpha; clamp ringing, as
        # ``rgba_clamp_premultiplied`` does, then un-premultiply.
        np.clip(cropped[..., :3], 0, alpha, out=cropped[..., :3])
        cropped[..., :3] *= np.where(alpha > 0, 255.0 / np.maximum(alpha, 1e-6), 0)
    return np.rint(np.clip(cropped, 0, 255)).astype(np.uint8)


def _downsample(array, factor, mask=False):
    """``array`` reduced by ``_downsample_level(factor)`` halvings, stopping if
    the copy would reach one pixel. Returns the copy and its applied level."""
    level = _downsample_level(factor)
    reduced = array
    applied = 0
    while applied < level and (reduced.shape[0] > 1 or reduced.shape[1] > 1):
        reduced = _halve(reduced, mask=mask)
        applied += 1
    return reduced, applied


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


def export_png(document, path, *, bit_depth=8, icc_profile=None):
    """Write the flattened document with the working sRGB profile embedded.

    The compositor renders 8-bit sRGB straight alpha. Exporting through the
    shared colour-managed writer tags the file with the sRGB ICC profile so a
    colour-managed viewer reproduces the editor's intent instead of guessing,
    and optionally widens to 16-bit PNG/TIFF (each output level scaled by 257,
    the exact 8->16-bit lift) for downstream high-bit editing. ``bit_depth``
    16 is only honoured for PNG/TIFF; others stay 8-bit, matching the native
    Compositor exporter, which is 8-bit sRGB only.
    """
    from ..io import get_working_srgb_icc, write_image_with_icc

    path = Path(path)
    pixels = composite(document)
    if icc_profile is None:
        icc_profile = get_working_srgb_icc()
    temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex[:8]}.tmp')
    # write_image_with_icc writes through its own atomic temp beside the target;
    # stage under our temp name so the final replace stays on one filesystem.
    staged = temporary.with_name(temporary.name + path.suffix)
    try:
        write_image_with_icc(
            staged, cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGRA),
            icc_profile=icc_profile, bit_depth=bit_depth, float_range='byte',
            dpi=(document.resolution, document.resolution),
            resolution=document.resolution)
        os.replace(staged, path)
    finally:
        for leftover in (temporary, staged):
            if leftover.exists():
                leftover.unlink()
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
        pixels, _ = _resample(pixels, layer.size, layer.sampling, upright=layer.rotation % 360 == 0)
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
    """Separable blend formulas in gamma-encoded sRGB space.

    Component modes (Hue, Saturation, Color and Luminosity) need a
    non-separable colour-space pass and remain explicitly unsupported.
    """
    backdrop = np.asarray(backdrop, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    if mode == 'Darken':
        return np.minimum(backdrop, source)
    if mode == 'Multiply':
        return backdrop * source
    if mode == 'Color Burn':
        return np.where(source <= 0, 0, 1 - np.minimum(1, (1 - backdrop) / source))
    if mode == 'Linear Burn':
        return np.maximum(0, backdrop + source - 1)
    if mode == 'Lighten':
        return np.maximum(backdrop, source)
    if mode == 'Screen':
        return backdrop + source - backdrop * source
    if mode == 'Color Dodge':
        return np.where(source >= 1, 1, np.minimum(1, backdrop / np.maximum(1 - source, 1e-12)))
    if mode == 'Linear Dodge (Add)':
        return np.minimum(1, backdrop + source)
    if mode == 'Overlay':
        return np.where(backdrop <= 0.5, 2 * backdrop * source,
                        1 - 2 * (1 - backdrop) * (1 - source))
    if mode == 'Soft Light':
        d = np.where(backdrop <= 0.25,
                     ((16 * backdrop - 12) * backdrop + 4) * backdrop,
                     np.sqrt(backdrop))
        return np.where(source <= 0.5,
                        backdrop - (1 - 2 * source) * backdrop * (1 - backdrop),
                        backdrop + (2 * source - 1) * (d - backdrop))
    if mode == 'Hard Light':
        return np.where(source <= 0.5, 2 * backdrop * source,
                        1 - 2 * (1 - backdrop) * (1 - source))
    if mode == 'Vivid Light':
        burn = np.where(source <= 0, 0,
                        1 - np.minimum(1, (1 - backdrop) / np.maximum(2 * source, 1e-12)))
        dodge_source = 2 * (source - 0.5)
        dodge = np.where(dodge_source >= 1, 1,
                         np.minimum(1, backdrop / np.maximum(1 - dodge_source, 1e-12)))
        return np.where(source <= 0.5, burn, dodge)
    if mode == 'Linear Light':
        return np.clip(backdrop + 2 * source - 1, 0, 1)
    if mode == 'Pin Light':
        return np.where(source <= 0.5,
                        np.minimum(backdrop, 2 * source),
                        np.maximum(backdrop, 2 * source - 1))
    if mode == 'Hard Mix':
        vivid = _blend(backdrop, source, 'Vivid Light')
        return (vivid >= 0.5).astype(np.float32)
    if mode == 'Difference':
        return np.abs(backdrop - source)
    if mode == 'Exclusion':
        return backdrop + source - 2 * backdrop * source
    if mode == 'Subtract':
        return np.maximum(0, backdrop - source)
    if mode == 'Divide':
        return np.where(source <= 0, 1, np.minimum(1, backdrop / np.maximum(source, 1e-12)))
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
    # Core Graphics resamples the layer image and its coverage as separate
    # passes: the mask is a clip with its own filter, not a premultiplied twin
    # of the pixels. Nearest stays nearest for both; otherwise the mask uses
    # the cubic filter of the low quality a rotated draw applies, and the
    # pixels use the layer's own quality.
    pixel_interpolation = cv2.INTER_NEAREST if layer.sampling == 'Nearest' else cv2.INTER_CUBIC
    mask_interpolation = cv2.INTER_NEAREST if layer.sampling == 'Nearest' else cv2.INTER_CUBIC
    if pixels.ndim == 2:
        return cv2.warpAffine(pixels, matrix, size, flags=mask_interpolation), None, origin
    rgba = pixels.astype(np.float32) / 255
    rgba[..., :3] *= rgba[..., 3:4]
    warped = cv2.warpAffine(rgba, matrix, size, flags=pixel_interpolation)
    alpha = warped[..., 3]
    warped[..., :3] /= np.where(alpha > 0, alpha, 1)[..., None]
    warped[alpha == 0, :3] = 0
    pixels = np.rint(np.clip(warped, 0, 1) * 255).astype(np.uint8)
    if coverage is not None:
        coverage = cv2.warpAffine(np.asarray(coverage), matrix, size, flags=mask_interpolation)
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
    resized, _ = _resample(mask, (width, height), sampling, upright=True, mask=True)
    return resized


def _resample(array, size, sampling, upright=True, mask=False):
    """Resample ``array`` to ``size`` with the native rules.

    ``LayerRenderer`` (``Rendering/LayerRenderer.swift``) draws an upright
    layer pixel for pixel (nearest) when it is not rescaled, prefilters large
    reductions through the halving pyramid and leaves Core Graphics only the
    last reduction, and otherwise honours the layer's own sampling. A rotated
    layer never takes the upright pixel-for-pixel shortcut, so enlarging and
    rotating keep the chosen quality while shrinking still prefilters.

    Returns the resampled array and the pyramid halvings applied, so callers
    can keep mask and pixels in step."""
    width, height = size
    # A 2-D array is coverage, never colour, whatever the caller passed.
    mask = mask or array.ndim == 2
    sx = width / max(1, array.shape[1])
    sy = height / max(1, array.shape[0])
    if sampling == 'Nearest':
        return cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST), 0
    if upright and abs(sx - 1) < 1e-3 and abs(sy - 1) < 1e-3:
        if array.shape[:2] == (height, width):
            return array, 0
        # Same size, different shape (a mask spanning its layer): copy across.
        return cv2.resize(array, (width, height), interpolation=cv2.INTER_NEAREST), 0
    factor = min(sx, sy)
    reduced, level = _downsample(array, factor, mask=mask)
    current = (reduced.shape[1], reduced.shape[0])
    if current == (width, height):
        return reduced, level
    # The halving chain took the large reduction; only the last step is left.
    # A further shrink averages (area); an upscale, and every mask, uses the
    # 4-tap cubic Core Graphics' low quality applies.
    if reduced.shape[1] <= width and reduced.shape[0] <= height:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA
    return cv2.resize(reduced, (width, height), interpolation=interpolation), level


def _straight(rgb, alpha):
    """Premultiplied float band -> straight-alpha uint8 RGBA."""
    a8 = np.rint(alpha * 255.0)
    safe = np.where(alpha > 0, alpha, 1.0)
    colour = np.rint(np.clip(rgb / safe[..., None], 0.0, 1.0) * 255.0)
    colour[a8 == 0] = 0
    return np.dstack([colour, a8]).astype(np.uint8)
