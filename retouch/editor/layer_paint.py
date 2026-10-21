"""Paint colour onto, or erase, a raster layer's own pixels with a normal brush.

Port of the colour path of upstream ``Document/BrushStroke.swift``: a stroke
is grayscale coverage, capped stroke-wide at the brush opacity (overlapping
dabs never exceed it), and each painted pixel is recomposed as

    original + colour x coverage x opacity          (premultiplied source-over)

in sRGB, 8 bits per channel. Erasing takes ``coverage x opacity`` out of the
layer's alpha (Core Graphics ``destinationOut``) and leaves colour alone.
An optional selection clips the stroke, as upstream's ``selectionClip`` does.

Differences from upstream, all deliberate:

* The browser rasterises the brush (Gradio's ImageEditor), so the tip is a
  hard round brush; ``softness`` blurs the stroke's coverage to give a soft
  edge instead of upstream's per-dab Gaussian falloff.
* Painting stays inside the layer's own rectangle. Upstream grows a painted
  pixel layer to the union with the canvas; the editor's brush canvas only
  shows the layer's rectangle, so there is nowhere outside it to paint.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from .commands import DocumentHistory

PAINT_MODES = ('Paint', 'Erase')


def _layer_grid(layer) -> tuple[int, int]:
    """(width, height) of the pixel grid a stroke lands on."""
    if layer.pixels is not None:
        return layer.pixels.shape[1], layer.pixels.shape[0]
    return int(round(layer.size[0])), int(round(layer.size[1]))


def stroke_coverage(canvas: Any, layer: Layer, stroke_opacity: float = 1.0) -> Optional[np.ndarray]:
    """Brush coverage on the layer's own pixel grid, in 0..1 per stroke.

    Each canvas layer is one native stroke: its alpha is scaled by
    ``stroke_opacity`` (native sets alpha per stroke at publish time, so two
    50% strokes stack toward 75%), then strokes combine by screen
    accumulation — 1-(1-a1)(1-a2) — which caps at 1 per stroke while letting
    separate strokes build up, matching BrushStroke.swift's per-stroke fill
    composition. Within one canvas layer the browser has already flattened
    overlapping dabs.

    Stroke RGB is ignored: the colour comes from the colour picker.
    """
    width, height = _layer_grid(layer)
    coverage = None
    for stroke in (canvas or {}).get('layers', []) if isinstance(canvas, dict) else []:
        stroke = np.asarray(stroke)
        if stroke.dtype != np.uint8 or stroke.ndim != 3 or stroke.shape[2] != 4:
            raise ValueError('Invalid brush data')
        if not stroke[..., 3].any():
            continue
        alpha = np.asarray(Image.fromarray(np.ascontiguousarray(stroke[..., 3])).resize(
            (width, height), Image.Resampling.BILINEAR)).astype(np.float32) / 255.0
        if stroke_opacity != 1.0:
            alpha = alpha * np.float32(stroke_opacity)
        coverage = alpha if coverage is None else coverage + alpha - coverage * alpha
    if coverage is None or not coverage.any():
        return None
    if layer.flip_y:
        coverage = coverage[::-1]
    if layer.flip_x:
        coverage = coverage[:, ::-1]
    return np.ascontiguousarray(coverage)


def _grid_to_doc(layer, width: int, height: int) -> np.ndarray:
    """3x3 affine from a ``width x height`` grid over the layer to document
    pixels, in cv2's convention (pixel centres on integers)."""
    from .composite import _doc_matrix
    to_doc = np.vstack([_doc_matrix(layer.origin, layer.size, layer.rotation,
                                    layer.flip_x, layer.flip_y), [0, 0, 1]])
    scale = np.diag([layer.size[0] / width, layer.size[1] / height, 1.0])
    half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], np.float64)
    return np.linalg.inv(half) @ to_doc @ scale @ half


def selection_on_layer(selection: np.ndarray, layer) -> np.ndarray:
    """A document-space uint8 selection resampled onto the layer's pixel grid."""
    width, height = _layer_grid(layer)
    warped = cv2.warpAffine(selection.astype(np.float32) / 255.0,
                            _grid_to_doc(layer, width, height)[:2], (width, height),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
    return np.clip(warped, 0.0, 1.0)


def paint_canvas_background(document, layer, preview_side: int = 1024) -> dict:
    """The brush canvas for ``layer``: the visible composite under the layer's
    rectangle, at preview size, in the layer's displayed orientation.

    Showing the composite (not the layer alone) lets a new transparent paint
    layer be painted in context. Outside the canvas reads as mid grey.
    """
    from .composite import composite
    scale = min(1, preview_side / max(layer.size))
    width, height = (max(1, round(v * scale)) for v in layer.size)
    pixels = composite(document)
    rgb = pixels[..., :3].astype(np.float32)
    alpha = pixels[..., 3:4].astype(np.float32) / 255.0
    # Transparent areas show over white, as the preview does.
    flat = np.rint(rgb * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    if scale < 1 and layer.rotation % 360 == 0:
        # Area-average down first so the preview doesn't alias, then place.
        flat = cv2.resize(flat, (max(1, round(document.width * scale)),
                                 max(1, round(document.height * scale))), interpolation=interpolation)
        shrink = np.diag([flat.shape[1] / document.width, flat.shape[0] / document.height, 1.0])
        half = np.array([[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]], np.float64)
        grid_to_view = np.linalg.inv(half) @ shrink @ half @ _grid_to_doc(layer, width, height)
    else:
        grid_to_view = _grid_to_doc(layer, width, height)
    view = cv2.warpAffine(flat, grid_to_view[:2], (width, height),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    # Source grid -> displayed orientation (the compositor flips the layer).
    if layer.flip_y:
        view = view[::-1]
    if layer.flip_x:
        view = view[:, ::-1]
    image = np.dstack([view, np.full((height, width), 255, np.uint8)])
    return {'background': image, 'layers': [], 'composite': image}


def compose_paint(pixels: np.ndarray, coverage: np.ndarray, color: Sequence[float],
                  opacity: float, erase: bool = False) -> np.ndarray:
    """Recompose straight-alpha RGBA uint8 under ``coverage x opacity``.

    Paint is premultiplied source-over of a solid colour; erase scales alpha
    by ``1 - coverage x opacity``. Pixels with zero coverage are returned
    bit-identical.
    """
    amount = np.clip(coverage, 0.0, 1.0).astype(np.float32) * np.float32(opacity)
    out = pixels.copy()
    touched = amount > 0
    if not touched.any():
        return out
    a = amount[touched][:, None]
    src = pixels[touched].astype(np.float32) / 255.0
    base_alpha = src[:, 3:4]
    if erase:
        new_alpha = base_alpha * (1.0 - a)
        rgb = src[:, :3]
    else:
        paint = np.asarray(color, np.float32)[None, :3]
        new_alpha = a + base_alpha * (1.0 - a)
        premultiplied = paint * a + src[:, :3] * base_alpha * (1.0 - a)
        rgb = np.where(new_alpha > 0, premultiplied / np.maximum(new_alpha, 1e-12), 0.0)
    result = np.concatenate([rgb, new_alpha], axis=1)
    result = np.rint(np.clip(result, 0.0, 1.0) * 255.0).astype(np.uint8)
    # Fully transparent pixels carry no colour (straight-alpha convention).
    result[result[:, 3] == 0, :3] = 0
    out[touched] = result
    return out


def paint_layer(history: DocumentHistory, layer_id: str, canvas: Any, *,
                color: Sequence[float] = (0.0, 0.0, 0.0), opacity: float = 100,
                mode: str = 'Paint', softness: float = 0,
                selection: Optional[np.ndarray] = None) -> bool:
    """Apply the canvas strokes to the layer's pixels as one undoable edit.

    Returns ``False`` when there is nothing painted. Raises ``ValueError`` for
    layers that have no paintable pixels and for out-of-range settings.
    """
    if mode not in PAINT_MODES:
        raise ValueError('Choose Paint or Erase.')
    if (isinstance(opacity, bool) or not isinstance(opacity, (int, float))
            or not math.isfinite(opacity) or not 1 <= opacity <= 100):
        raise ValueError('Brush opacity must be between 1 and 100.')
    if (isinstance(softness, bool) or not isinstance(softness, (int, float))
            or not math.isfinite(softness) or not 0 <= softness <= 200):
        raise ValueError('Brush softness must be between 0 and 200 pixels.')
    if (len(color) != 3 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                or not math.isfinite(v) or not 0 <= v <= 1 for v in color)):
        raise ValueError('Choose a paint colour.')
    layer = history.document.layer(layer_id)
    if layer.is_group or layer.adjustment is not None:
        raise ValueError('Folders and adjustment layers have no pixels to paint. '
                         'Select an image layer, or add a new paint layer.')
    if layer.text is not None or layer.shape is not None:
        raise ValueError('Painting would flatten this text or shape layer. '
                         'Add a new paint layer above it instead.')
    if layer.pixels is None and mode == 'Erase':
        raise ValueError('This layer is empty; there is nothing to erase.')
    coverage = stroke_coverage(canvas, layer, stroke_opacity=opacity / 100.0)
    if coverage is None:
        return False
    if softness > 0:
        # Softness is given in native layer pixels: a Gaussian whose 3 sigma
        # equals it, so the edge fades over about that distance.
        sigma = softness / 3.0
        coverage = cv2.GaussianBlur(coverage, (0, 0), sigmaX=sigma, sigmaY=sigma,
                                    borderType=cv2.BORDER_CONSTANT)
    if selection is not None:
        doc = history.document
        if selection.dtype != np.uint8 or selection.shape != (doc.height, doc.width):
            raise ValueError('Selection does not match the current document.')
        coverage = coverage * selection_on_layer(selection, layer)
    # Quantise to 8-bit coverage, as upstream's grayscale coverage tiles are.
    coverage = np.rint(np.clip(coverage, 0.0, 1.0) * 255.0) / 255.0
    if not coverage.any():
        raise ValueError('The strokes do not overlap the current selection.')
    width, height = _layer_grid(layer)
    base = (layer.pixels if layer.pixels is not None
            else np.zeros((height, width, 4), np.uint8))
    painted = compose_paint(base, coverage, color, 1.0, erase=mode == 'Erase')
    if layer.pixels is not None and np.array_equal(painted, layer.pixels):
        raise ValueError('The strokes made no visible change.')
    return history.replace_pixels(layer.id, painted,
                                  'Erase on layer' if mode == 'Erase' else 'Paint on layer')
