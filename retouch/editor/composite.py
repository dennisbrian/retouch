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
import math
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
    'Hue', 'Saturation', 'Color', 'Luminosity',
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
_MAX_EFFECTS_WINDOW_PIXELS = 80_000_000  # upstream LayerEffectsSurface.swift:47

# Full-canvas effect planes are cached at 16 B/px; cap admission so a
# renderer holds at most ~384 MB per effects layer. Larger canvases fall
# back to the per-band window path above.
_MAX_EFFECTS_CACHE_PIXELS = 24_000_000


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
        self.effect_planes = {}
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
            changed = grade(straight, layer.adjustment, y0=float(top))
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
            if layer.effects is not None and not layer.effects.is_empty():
                if (self.doc.width * self.doc.height <= _MAX_EFFECTS_CACHE_PIXELS
                        and layer.id not in self.stacks):
                    # Effects are computed once over the whole canvas and
                    # cached: per-band recomputation re-blurs the same margin
                    # overlap band_rows times over, which dominates the run at
                    # small band_rows (25x at band_rows=1). The plane costs
                    # 16 B/px, so admission is capped at 24 MP; stack bases
                    # stay per-band (their source is mutated in place).
                    if layer.id not in self.effect_planes:
                        plane_src, plane_a = self.own(layer, 0, self.doc.height)
                        self.effect_planes[layer.id] = _apply_effects_plane(
                            plane_src, plane_a, layer.effects)
                    source = self.effect_planes[layer.id][0][top:bottom]
                    a = self.effect_planes[layer.id][1][top:bottom]
                else:
                    margin = _effect_margin(layer.effects)
                    ext_top = max(0, top - margin)
                    ext_bottom = min(self.doc.height, bottom + margin)
                    # Port-only guard: upstream budgets the effects surface
                    # (LayerEffectsSurface.swift:47, 80 MP); this port's band
                    # window is its equivalent, so budget it the same way.
                    if (ext_bottom - ext_top) * self.doc.width > _MAX_EFFECTS_WINDOW_PIXELS:
                        raise ValueError(
                            'layer effects window exceeds the %d MP live budget'
                            % (_MAX_EFFECTS_WINDOW_PIXELS // 1_000_000))
                    if ext_top != top or ext_bottom != bottom:
                        ext_source, ext_a = self.own(layer, ext_top, ext_bottom)
                    else:
                        ext_source, ext_a = source, a
                    eff_rgb, eff_a = _apply_effects(ext_source, ext_a, layer.effects)
                    source = eff_rgb[top - ext_top:bottom - ext_top]
                    a = eff_a[top - ext_top:bottom - ext_top]
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



# --- Layer Effects -------------------------------------------------------------
#
# Compositor's LayerEffects subset: stroke, drop shadow and color overlay.
#
# Effects are drawn on a padded copy of the source (``_effects_source``): the
# shadow's own silhouette, offset and blurred inside that padding, then the
# layer's pixels through its blend mode, then the flat colour over them. The
# coverage is taken from the unadjusted silhouette, as it is for adjustment
# layers, so grading never feeds back into the shadow's shape.


def _effect_margin(effects):
    """``LayerEffectsRenderer.margin`` for the subset this port renders."""
    margin = 0.0
    if effects is not None and effects.stroke is not None:
        stroke = effects.stroke
        # Both stroke kinds need the kernel's full reach in context: an inside
        # stroke erodes the silhouette (context above), an outside one dilates.
        if stroke.enabled and stroke.opacity > 0:
            margin = max(margin, stroke.size)
    if effects is not None and effects.outerGlow is not None:
        glow = effects.outerGlow
        if glow.enabled and glow.size > 0 and glow.opacity > 0:
            margin = max(margin, glow.size * 3)
    if effects is not None and effects.shadow is not None:
        shadow = effects.shadow
        if shadow.enabled and shadow.opacity > 0:
            margin = max(margin, shadow.distance + shadow.blur * 3)
    # Upstream's margin stops here: inner effects draw inside the silhouette.
    # But this port renders in bands, and each band's extended window must
    # carry enough of the layer's own alpha for the inner effects' blur to
    # see real coverage instead of the window's zero padding, or their
    # coverage would differ per band split (band-invariance contract).
    if effects is not None and effects.innerShadow is not None:
        inner_shadow = effects.innerShadow
        if inner_shadow.enabled and inner_shadow.opacity > 0:
            margin = max(margin, inner_shadow.distance + inner_shadow.blur * 3)
    if effects is not None and effects.innerGlow is not None:
        inner_glow = effects.innerGlow
        if inner_glow.enabled and inner_glow.size > 0 and inner_glow.opacity > 0:
            margin = max(margin, inner_glow.size * 3)
    return int(math.ceil(margin)) + 2


def _shift(array, dx, dy):
    """``array`` moved by ``dx`` columns and ``dy`` rows, filling with zeros.

    Plain slicing, not ``np.roll``: a roll would wrap the far edge of the
    layer's silhouette around into the padding the shadow is read from.
    """
    out = np.zeros_like(array)
    height, width = array.shape[0], array.shape[1]
    sx, tx = (0, dx) if dx >= 0 else (-dx, 0)
    sy, ty = (0, dy) if dy >= 0 else (-dy, 0)
    span_x, span_y = width - abs(dx), height - abs(dy)
    if span_x > 0 and span_y > 0:
        out[ty:ty + span_y, tx:tx + span_x] = array[sy:sy + span_y, sx:sx + span_x]
    return out


def _shift_float(array, dx, dy):
    """Bilinear move by a fractional pixel amount, upstream's smooth shift.

    ``effects_shift`` (Metal) and the fractionally placed CPU draw both blend
    neighbours when the shadow's offset lands between pixels
    (MetalLayerEffects.swift:278-289), so the same is done here on the
    float coverage; the integer ``_shift`` stays for hard pixel moves.
    """
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    size = (array.shape[1], array.shape[0])  # the caller's buffer is already padded
    return cv2.warpAffine(np.asarray(array, np.float32), matrix, size,
                          flags=cv2.INTER_LINEAR)


def _blur(mask, radius):
    """Gaussian soften with upstream's blur radius (``coverage(... blur:)``)."""
    if radius <= 0:
        return mask
    sigma = radius / 2.0
    kernel = max(3, int(math.ceil(sigma * 6)) | 1)
    return cv2.GaussianBlur(mask, (kernel, kernel), sigma, borderType=cv2.BORDER_CONSTANT)


def _stroke_coverage(alpha, stroke):
    """Upstream square-reach morphology ring for an inside/outside stroke."""
    # Upstream rounds half away from zero (Swift .rounded()), not banker's.
    reach = max(1, int(math.floor(stroke.size + 0.5)))
    kernel = np.ones((reach * 2 + 1, reach * 2 + 1), np.uint8)
    if stroke.inside:
        moved = cv2.erode(alpha, kernel, borderType=cv2.BORDER_CONSTANT,
                          borderValue=0)
        return np.maximum(0, alpha - moved)
    moved = cv2.dilate(alpha, kernel, borderType=cv2.BORDER_CONSTANT,
                       borderValue=0)
    return np.maximum(0, moved - alpha)


def _fill_over(rgb, alpha, color, coverage):
    """Flat straight RGB ``color`` through ``coverage``, source-over."""
    out_a = coverage + alpha * (1 - coverage)
    safe = np.where(out_a > 0, out_a, 1)[..., np.newaxis]
    color = np.asarray(color, np.float32)
    out_rgb = np.where(
        out_a[..., np.newaxis] > 0,
        (color * coverage[..., np.newaxis]
         + rgb * (alpha * (1 - coverage))[..., np.newaxis]) / safe,
        0.0)
    return out_rgb, out_a


_EFFECT_PLANE_CHUNK_ROWS = 512


def _apply_effects_plane(rgb, alpha, effects):
    """Full-canvas effects plane, row-chunked inside margin windows.

    Every effect step is row-local within ``_effect_margin`` (Gaussian kernel
    reach, morphology reach and the bilinear shift all stay inside it), so a
    chunk's margin context is bit-identical to the full plane's and the
    outputs are exact — this only bounds the ``_apply_effects`` transient,
    which peaks near 1 GB at 12 MP unchunked.
    """
    height = rgb.shape[0]
    if height <= _EFFECT_PLANE_CHUNK_ROWS * 2:
        return _apply_effects(rgb, alpha, effects)
    margin = _effect_margin(effects)
    out_rgb, out_alpha = np.empty_like(rgb), np.empty_like(alpha)
    top = 0
    while top < height:
        bottom = min(height, top + _EFFECT_PLANE_CHUNK_ROWS)
        c_top, c_bottom = max(0, top - margin), min(height, bottom + margin)
        chunk_rgb, chunk_alpha = _apply_effects(
            rgb[c_top:c_bottom], alpha[c_top:c_bottom], effects)
        out_rgb[top:bottom] = chunk_rgb[top - c_top:bottom - c_top]
        out_alpha[top:bottom] = chunk_alpha[top - c_top:bottom - c_top]
        top = bottom
    return out_rgb, out_alpha


def _apply_effects(rgb, alpha, effects):
    """Composite layer effects onto (rgb, alpha) straight float32 arrays.

    Matches ``LayerEffectsRenderer.render`` (upstream):
      1. Shadow sits behind the layer's own pixels (source-over).
      2. Layer's own pixels drawn over the shadow.
      3. Flat colour overlay sits over the layer, masked by the unshadowed
         silhouette (``coverage(shown, blur: 0)`` upstream).
    """
    layer_rgb, layer_alpha = rgb, alpha
    accum_rgb, accum_alpha = np.zeros_like(rgb), np.zeros_like(alpha)

    # 1. Shadow sits behind the layer's own pixels.
    shadow = effects.shadow
    if shadow is not None and shadow.enabled and shadow.opacity > 0:
        dx, dy = shadow.offset
        moved = _blur(_shift_float(layer_alpha, dx, dy), shadow.blur)
        shadow_a = moved * shadow.opacity
        shadow_c = np.array([shadow.red, shadow.green, shadow.blue], np.float32)
        accum_rgb = np.broadcast_to(shadow_c, accum_rgb.shape).astype(np.float32)
        accum_alpha = shadow_a

    # 2. Outer glow: blurred coverage, halo only outside the layer shape,
    #    drawn over the shadow but still behind the layer's own pixels.
    outer_glow = effects.outerGlow
    if (outer_glow is not None and outer_glow.enabled
            and outer_glow.size > 0 and outer_glow.opacity > 0):
        halo = _blur(layer_alpha, outer_glow.size) * (1 - layer_alpha) * outer_glow.opacity
        accum_rgb, accum_alpha = _fill_over(
            accum_rgb, accum_alpha, (outer_glow.red, outer_glow.green, outer_glow.blue), halo)

    # 3. Outside stroke sits behind the layer's own pixels.
    stroke = effects.stroke
    if (stroke is not None and stroke.enabled and not stroke.inside
            and stroke.size > 0 and stroke.opacity > 0):
        coverage = _stroke_coverage(layer_alpha, stroke) * stroke.opacity
        accum_rgb, accum_alpha = _fill_over(
            accum_rgb, accum_alpha, (stroke.red, stroke.green, stroke.blue), coverage)

    # 4. Layer's own pixels drawn over the shadow/glow/stroke (source-over).
    if layer_alpha.any():
        l_over = layer_alpha[..., np.newaxis]
        out_a = layer_alpha + accum_alpha * (1 - layer_alpha)
        safe = np.where(out_a > 0, out_a, 1)[..., np.newaxis]
        accum_rgb = np.where(
            out_a[..., np.newaxis] > 0,
            (layer_rgb * l_over + accum_rgb * (accum_alpha[..., np.newaxis] * (1 - l_over))) / safe,
            0.0
        )
        accum_alpha = out_a

    # 5. Flat colour overlay sits over the layer; its shape is the unshadowed
    # silhouette (``coverage(shown, blur: 0)`` upstream), so it never tints the shadow.
    overlay = effects.colorOverlay
    if overlay is not None and overlay.enabled and overlay.opacity > 0 and layer_alpha.any():
        o_a = layer_alpha * overlay.opacity
        o_over = o_a[..., np.newaxis]
        out_a = o_a + accum_alpha * (1 - o_a)
        safe = np.where(out_a > 0, out_a, 1)[..., np.newaxis]
        over_c = np.array([overlay.red, overlay.green, overlay.blue], np.float32)
        accum_rgb = np.where(
            out_a[..., np.newaxis] > 0,
            (over_c * o_over + accum_rgb * (accum_alpha[..., np.newaxis] * (1 - o_over))) / safe,
            0.0
        )
        accum_alpha = out_a

    # 6. Inner glow: coverage's blur complement, kept inside the shape.
    inner_glow = effects.innerGlow
    if (inner_glow is not None and inner_glow.enabled
            and inner_glow.size > 0 and inner_glow.opacity > 0):
        coverage = layer_alpha * (1 - _blur(layer_alpha, inner_glow.size)) * inner_glow.opacity
        accum_rgb, accum_alpha = _fill_over(
            accum_rgb, accum_alpha, (inner_glow.red, inner_glow.green, inner_glow.blue), coverage)

    # 7. Inner shadow: the outside's softness, kept to the layer's own shape.
    inner_shadow = effects.innerShadow
    if (inner_shadow is not None and inner_shadow.enabled
            and inner_shadow.opacity > 0):
        dx, dy = inner_shadow.offset
        moved = _blur(_shift_float(layer_alpha, dx, dy), inner_shadow.blur)
        coverage = layer_alpha * (1 - moved) * inner_shadow.opacity
        accum_rgb, accum_alpha = _fill_over(
            accum_rgb, accum_alpha, (inner_shadow.red, inner_shadow.green, inner_shadow.blue), coverage)

    # 8. Inside stroke is drawn last, over the layer and overlay.
    if (stroke is not None and stroke.enabled and stroke.inside
            and stroke.size > 0 and stroke.opacity > 0):
        coverage = _stroke_coverage(layer_alpha, stroke) * stroke.opacity
        accum_rgb, accum_alpha = _fill_over(
            accum_rgb, accum_alpha, (stroke.red, stroke.green, stroke.blue), coverage)

    return accum_rgb.astype(np.float32), accum_alpha.astype(np.float32)


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
        if layer.mask_placement is not None:
            coverage = _place_mask(coverage, layer)
    # A flip mirrors the layer's local rectangle; its mask follows the layer.
    if layer.flip_y:
        pixels = pixels[::-1]
        coverage = None if coverage is None else coverage[::-1]
    if layer.flip_x:
        pixels = pixels[:, ::-1]
        coverage = None if coverage is None else coverage[:, ::-1]
    pixels, coverage, origin = _rotate(pixels, coverage, layer)
    return pixels, coverage, origin, layer.opacity, layer.blend_mode


def _lum(rgb):
    """W3C/PDF relative luminance weights Core Image's blends use."""
    return rgb[..., 0] * 0.3 + rgb[..., 1] * 0.59 + rgb[..., 2] * 0.11


def _sat(rgb):
    """W3C ``Sat``: max - min per pixel."""
    return np.max(rgb, axis=-1) - np.min(rgb, axis=-1)


def _clip_color(rgb):
    """W3C ``ClipColor``: pull out-of-gamut pixels back onto the gamut hull."""
    lum = _lum(rgb)
    n = np.min(rgb, axis=-1)
    x = np.max(rgb, axis=-1)
    out = rgb.copy()
    below = n < 0
    if below.any():
        scale = np.where(below, lum / np.where(below, lum - n, 1), 1.0)[..., np.newaxis]
        out = lum[..., np.newaxis] + (out - lum[..., np.newaxis]) * scale
    x = np.max(out, axis=-1)
    above = x > 1
    if above.any():
        scale = np.where(above, (1 - lum) / np.where(above, x - lum, 1), 1.0)[..., np.newaxis]
        out = lum[..., np.newaxis] + (out - lum[..., np.newaxis]) * scale
    return out


def _set_lum(rgb, lum):
    return _clip_color(rgb + (lum - _lum(rgb))[..., np.newaxis])


def _set_sat(rgb, sat):
    """W3C ``SetSat``: rescale chroma to ``sat``, keeping channel order."""
    lo = np.min(rgb, axis=-1, keepdims=True)
    hi = np.max(rgb, axis=-1, keepdims=True)
    span = np.where(hi > lo, hi - lo, 1.0)
    return np.where(hi > lo, (rgb - lo) * sat[..., np.newaxis] / span, 0.0)


def _blend(backdrop, source, mode):
    """Blend formulas in gamma-encoded sRGB space.

    The component modes (Hue, Saturation, Color, Luminosity) follow the
    W3C/PDF-32000 non-separable formulas behind Core Image's blend filters.
    """
    backdrop = np.asarray(backdrop, dtype=np.float32)
    source = np.asarray(source, dtype=np.float32)
    if mode == 'Hue':
        # Source hue over backdrop saturation and luminance.
        return _set_lum(_set_sat(source, _sat(backdrop)), _lum(backdrop))
    if mode == 'Saturation':
        # Source saturation over backdrop hue and luminance.
        return _set_lum(_set_sat(backdrop, _sat(source)), _lum(backdrop))
    if mode == 'Color':
        return _set_lum(source, _lum(backdrop))
    if mode == 'Luminosity':
        return _set_lum(backdrop, _lum(source))
    if mode == 'Darken':
        return np.minimum(backdrop, source)
    if mode == 'Multiply':
        return backdrop * source
    if mode == 'Color Burn':
        # Core Image kernel corner: Cs==0 -> (Cb==1 ? 1 : 0); interior
        # unchanged. Verified against CIColorBurnBlendMode, null color space.
        return np.where(source <= 0, (backdrop == 1).astype(np.float32),
                        1 - np.minimum(1, (1 - backdrop) / source))
    if mode == 'Linear Burn':
        return np.maximum(0, backdrop + source - 1)
    if mode == 'Lighten':
        return np.maximum(backdrop, source)
    if mode == 'Screen':
        return backdrop + source - backdrop * source
    if mode == 'Color Dodge':
        # Core Image kernel corner: Cs==1 -> (Cb>0 ? 1 : 0); interior
        # unchanged. Verified against CIColorDodgeBlendMode, null color space.
        return np.where(source >= 1, (backdrop > 0).astype(np.float32),
                        np.minimum(1, backdrop / np.maximum(1 - source, 1e-12)))
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
        # Inherits both CI kernel corners from burn/dodge (see above).
        burn = np.where(source <= 0, (backdrop == 1).astype(np.float32),
                        1 - np.minimum(1, (1 - backdrop) / np.maximum(2 * source, 1e-12)))
        dodge_source = 2 * (source - 0.5)
        dodge = np.where(dodge_source >= 1, (backdrop > 0).astype(np.float32),
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


def _subpixel_shift(array, fx, fy, nearest):
    """Translate by a fractional pixel amount, upstream's subpixel placement.

    Core Graphics draws a fractionally placed 1:1 layer with antialiasing, so
    each destination pixel blends its neighbours by the fractional coverage;
    Nearest sampling disables that and duplicates/drops hard columns instead.
    RGBA goes through premultiplied colour, as the rotation warp does.
    """
    matrix = np.float32([[1, 0, fx], [0, 1, fy]])
    size = (array.shape[1] + 2, array.shape[0] + 2)  # one pixel of headroom each way
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if array.ndim == 2:
        return cv2.warpAffine(np.asarray(array), matrix, size, flags=flags)
    rgba = array.astype(np.float32) / 255
    rgba[..., :3] *= rgba[..., 3:4]
    warped = cv2.warpAffine(rgba, matrix, size, flags=flags)
    alpha = warped[..., 3]
    warped[..., :3] /= np.where(alpha > 0, alpha, 1)[..., None]
    warped[alpha == 0, :3] = 0
    return np.rint(np.clip(warped, 0, 1) * 255).astype(np.uint8)


def _rotate(pixels, coverage, layer):
    if layer.rotation % 360 == 0:
        origin = layer.origin
        fx, fy = origin[0] % 1, origin[1] % 1
        base = (int(math.floor(origin[0])), int(math.floor(origin[1])))
        if fx == 0 and fy == 0:
            return pixels, coverage, base
        # A fractionally placed upright layer: floor the canvas anchor and
        # shift the layer content by the residual inside the prepared buffer.
        nearest = layer.sampling == 'Nearest'
        pixels = _subpixel_shift(pixels, fx, fy, nearest)
        if coverage is not None:
            coverage = _subpixel_shift(np.asarray(coverage), fx, fy, nearest)
        return pixels, coverage, base
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
        # A full-coverage mask must stay full after the rotation warp: the
        # bbox corners would otherwise read border-zero and clip the layer's
        # alpha, so replicate the edge (Core Graphics clips to the layer's own
        # extent, it never samples a border beyond it).
        coverage = cv2.warpAffine(np.asarray(coverage), matrix, size,
                                  flags=mask_interpolation,
                                  borderMode=cv2.BORDER_REPLICATE)
    return pixels, coverage, origin


def _prepare_mask(layer):
    mask = _fit_mask(layer.mask, *layer.size, layer.sampling)
    if layer.mask_placement is not None:
        mask = _place_mask(mask, layer)
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


def _doc_matrix(origin, size, rotation, flip_x, flip_y):
    """The 2x3 affine taking a rectangle's own pixels into document space.

    Mirrors ``BrushRaster.pixelToDocument`` upstream: a pixel at ``p`` lands at
    ``origin + center + R*F*(p - center)`` — flip about the centre, rotate
    clockwise about it, place the rectangle's top-left at ``origin``.
    """
    w, h = size
    centre = np.array([w / 2, h / 2], np.float64)
    angle = np.deg2rad(rotation % 360)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    rot = np.array([[c, -s], [s, c]], np.float64)
    fx, fy = (-1.0 if flip_x else 1.0, -1.0 if flip_y else 1.0)
    a = rot @ np.array([[fx, 0], [0, fy]], np.float64)
    b = np.array(origin, np.float64) + centre - a @ centre
    return np.array([[a[0, 0], a[0, 1], b[0]],
                     [a[1, 0], a[1, 1], b[1]]], np.float64)


def _place_mask(mask, layer):
    """Warp an unlinked mask into its layer's own pixel grid.

    Upstream ``LayerMask.clipImage`` resamples the placed mask through
    ``pixelToDocument(placement) * pixelToDocument(layer).inverted()`` into a
    layer-grid clip before the layer draws with it, filling everything outside
    the mask's rect with the dominant gray of its border (a reveal-all mask
    keeps revealing, a hide-all one keeps hiding). The warped grid then rides
    the layer's own transform exactly like a linked mask.
    """
    placement = layer.mask_placement
    pw, ph = placement['size']
    base = _fit_mask(np.asarray(mask), int(round(pw)), int(round(ph)), layer.sampling)
    mh, mw = base.shape[:2]
    lw, lh = layer.size
    # Mask pixel -> document, composed with layer pixel -> document inverted.
    m_doc = _doc_matrix(placement['origin'], placement['size'],
                        placement['rotation'], placement['flipX'], placement['flipY'])
    l_doc = _doc_matrix(layer.origin, layer.size, layer.rotation,
                        layer.flip_x, layer.flip_y)
    l_inv = np.vstack([l_doc, [0, 0, 1]])
    try:
        l_inv = np.linalg.inv(l_inv)[:2]
    except np.linalg.LinAlgError:
        return np.full((lh, lw), 255, np.uint8)
    # cv2.warpAffine maps source -> destination, so feed the forward
    # composition: mask pixels -> document -> layer pixels. _doc_matrix is
    # center-convention (pixel centers at +0.5) while warpAffine samples
    # from pixel corners, so apply the same half-pixel translation _rotate
    # performs — without it every rotated/flipped placement is off by the
    # sub-pixel difference between the two conventions.
    warp = (np.vstack([l_inv, [0, 0, 1]]) @ np.vstack([m_doc, [0, 0, 1]]))[:2].astype(np.float32)
    warp[:, 2] += (warp[:, :2] @ np.array([0.5, 0.5], np.float32)) - 0.5
    # Dominant border gray: white when most border pixels reveal.
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    background = 255.0 if border.mean() >= 127.5 else 0.0
    nearest = placement['sampling'] == 'Nearest' or layer.sampling == 'Nearest'
    flags = cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC
    return np.rint(cv2.warpAffine(base.astype(np.float32), warp, (lw, lh),
                                  flags=flags, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=background)).astype(np.uint8)


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
    if mask or reduced.ndim != 3 or reduced.shape[2] != 4:
        return cv2.resize(reduced, (width, height), interpolation=interpolation), level
    # Native draws RGBA through a premultiplied surface. Interpolating straight
    # RGB leaks hidden transparent colours into visible edge pixels.
    rgba = reduced.astype(np.float32) / 255.0
    rgba[..., :3] *= rgba[..., 3:4]
    resized = cv2.resize(rgba, (width, height), interpolation=interpolation)
    alpha = resized[..., 3:4]
    np.clip(resized[..., :3], 0, alpha, out=resized[..., :3])
    resized[..., :3] = np.divide(
        resized[..., :3], np.where(alpha > 0, alpha, 1),
        out=np.zeros_like(resized[..., :3]), where=alpha > 0)
    return np.rint(np.clip(resized, 0, 1) * 255).astype(np.uint8), level


def _straight(rgb, alpha):
    """Premultiplied float band -> straight-alpha uint8 RGBA."""
    a8 = np.rint(alpha * 255.0)
    safe = np.where(alpha > 0, alpha, 1.0)
    colour = np.rint(np.clip(rgb / safe[..., None], 0.0, 1.0) * 255.0)
    colour[a8 == 0] = 0
    return np.dstack([colour, a8]).astype(np.uint8)
