"""Transient document-space selections and projection into native layer masks."""
import json
import math

import cv2
import numpy as np

from .document import MAX_SURFACE_PIXELS


def combine(old, new, mode):
    if old.shape != new.shape or old.dtype != np.uint8 or new.dtype != np.uint8:
        raise ValueError('Selection masks must be matching uint8 arrays')
    if mode == 'Replace':
        return new.copy()
    if mode == 'Add':
        return np.maximum(old, new)
    if mode == 'Subtract':
        return np.minimum(old, 255 - new)
    if mode == 'Intersect':
        return np.minimum(old, new)
    raise ValueError('Unknown selection combination')


def wand_selection(rgba, x, y, tolerance=32, radius=0, contiguous=True):
    """Magic-wand selection on a straight-alpha RGBA composite.

    Samples a (2r+1)^2 box around the seed, compares every pixel to the
    rounded box average per channel (alpha included), and either returns
    all matches or a 4-connected flood from the seed. Returns the HxW
    uint8 mask and the selected pixel count.
    """
    h, w = rgba.shape[:2]
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError('wand_selection needs an HxWx4 uint8 array')
    if h * w > MAX_SURFACE_PIXELS:
        raise ValueError('Selection exceeds the canvas budget')
    for v in (x, y):
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v):
            raise ValueError('Wand seed must be finite numbers')
    x, y = int(x), int(y)
    if not (0 <= x < w and 0 <= y < h):
        return np.zeros((h, w), np.uint8), 0
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) \
            or not math.isfinite(tolerance):
        raise ValueError('Tolerance must be a finite number')
    t = min(255, max(0, int(tolerance)))
    if isinstance(radius, bool) or not isinstance(radius, (int, float)) \
            or not math.isfinite(radius) or not 0 <= int(radius) <= 500:
        raise ValueError('Sample radius must be 0-500 pixels')
    r = int(radius)
    x0, x1 = max(0, x - r), min(w - 1, x + r)
    y0, y1 = max(0, y - r), min(h - 1, y + r)
    box = rgba[y0:y1 + 1, x0:x1 + 1].astype(np.int64)
    n = box.shape[0] * box.shape[1]
    # Round-half-up like the C average (WandPixels.c): sum + n/2 over n.
    ref = (box.sum(axis=(0, 1)) + n // 2) // n
    # int16 holds |0..255 - 0..255| exactly; int64 doubles the pass cost on
    # multi-megapixel canvases for the same bit-exact result.
    diff = np.abs(rgba.astype(np.int16) - ref.astype(np.int16))
    match = np.all(diff <= t, axis=2)
    if not contiguous:
        m = (match * np.uint8(255)).astype(np.uint8)
        return m, int(match.sum())
    if not match[y, x]:
        return np.zeros((h, w), np.uint8), 0
    # floodFill on the binary match image with FLOODFILL_MASK_ONLY;
    # 4-connectivity, fill value 255.
    fill = (match * np.uint8(255)).astype(np.uint8)
    ff = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY
    cv2.floodFill(fill, ff, (x, y), 255, loDiff=0, upDiff=0, flags=flags)
    m = ff[1:-1, 1:-1]
    return m, int((m > 0).sum())


def _color_list(value, name):
    """Parse '[[r, g, b], ...]' text (or an already-parsed list) to an Nx3 int array."""
    try:
        colors = json.loads(value) if isinstance(value, str) else value
        valid = (isinstance(colors, list)
                 and all(isinstance(c, (list, tuple)) and len(c) == 3
                         and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                 and math.isfinite(v) and 0 <= v <= 255 for v in c)
                         for c in colors))
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError(f'{name} needs RGB triplets like [[255, 0, 0]]')
    if not colors:
        return np.zeros((0, 3), np.int64)
    return np.rint(np.asarray(colors, np.float64)).astype(np.int64)


def color_range_selection(rgba, include, exclude, fuzziness):
    """Color-range mask matching the C kernel (WandPixels.c color_range_mask).

    A pixel matches when its straight RGB is within ``fuzziness`` of any
    include colour on every channel (per-channel box distance, not a
    vector norm), and not within range of any exclude colour. Fully
    transparent pixels never match. The mask is binary 0/255 like the
    native one; returns (mask, selected pixel count).
    """
    h, w = rgba.shape[:2]
    if rgba.ndim != 3 or rgba.shape[2] != 4 or rgba.dtype != np.uint8:
        raise ValueError('color_range_selection needs an HxWx4 uint8 array')
    if h * w > MAX_SURFACE_PIXELS:
        raise ValueError('Selection exceeds the canvas budget')
    if isinstance(fuzziness, bool) or not isinstance(fuzziness, (int, float)) \
            or not math.isfinite(fuzziness):
        raise ValueError('Fuzziness must be a finite number')
    f = min(255, max(0, int(fuzziness)))
    inc = _color_list(include, 'Include colours')
    exc = _color_list(exclude, 'Exclude colours')
    if inc.size == 0:
        return np.zeros((h, w), np.uint8), 0
    rgb = rgba[..., :3].astype(np.int64)
    near = np.zeros((h, w), bool)
    for c in inc:
        near |= np.all(np.abs(rgb - c) <= f, axis=2)
    for c in exc:
        near &= ~np.all(np.abs(rgb - c) <= f, axis=2)
    near &= rgba[..., 3] != 0
    return (near * np.uint8(255)).astype(np.uint8), int(near.sum())


def selection(width, height, kind, x=0, y=0, box_width=1, box_height=1, points='[]', feather=0):
    if width * height > MAX_SURFACE_PIXELS:
        raise ValueError('Selection exceeds the canvas budget')
    if any(not isinstance(v, (int, float)) or isinstance(v, bool)
           or not math.isfinite(v) or abs(v) > 1_000_000
           for v in (x, y, box_width, box_height, feather)):
        raise ValueError('Selection geometry must be finite')
    if not 0 <= feather <= 250:
        raise ValueError('Feather must be between 0 and 250 pixels')
    mask = np.zeros((height, width), np.uint8)
    if kind in ('Rectangle', 'Ellipse'):
        if box_width <= 0 or box_height <= 0:
            raise ValueError('Selection dimensions must be positive')
        if kind == 'Rectangle':
            x0, y0 = max(0, math.ceil(x)), max(0, math.ceil(y))
            x1, y1 = min(width, math.ceil(x + box_width)), min(height, math.ceil(y + box_height))
            mask[y0:max(y0, y1), x0:max(x0, x1)] = 255
        else:
            # Evaluate pixel centers in row bands rather than a full coordinate mesh.
            xs = (np.arange(width, dtype=np.float32) + .5 - x - box_width / 2) / (box_width / 2)
            for top in range(0, height, 256):
                ys = (np.arange(top, min(height, top + 256), dtype=np.float32)
                      + .5 - y - box_height / 2) / (box_height / 2)
                mask[top:top + len(ys)] = ((ys[:, None] ** 2 + xs[None, :] ** 2) <= 1) * np.uint8(255)
    elif kind == 'Polygon':
        try:
            vertices = json.loads(points) if isinstance(points, str) else points
            valid = (isinstance(vertices, list) and 3 <= len(vertices) <= 4096
                     and all(isinstance(p, (list, tuple)) and len(p) == 2
                             and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                     and math.isfinite(v) and abs(v) <= 1_000_000 for v in p)
                             for p in vertices))
        except (ValueError, TypeError):
            valid = False
        if not valid:
            raise ValueError('Polygon needs 3–4096 finite [x, y] points')
        cv2.fillPoly(mask, [np.rint(vertices).astype(np.int32)], 255)
    else:
        raise ValueError('Choose Rectangle, Ellipse or Polygon')
    if feather:
        # Upstream feathers with sigma = feather / 2 (Selection.swift:27) on
        # an extent-clamped image, so the fade extends either side of the
        # outline; BORDER_REPLICATE is the cv2 equivalent of the clamp.
        mask = cv2.GaussianBlur(mask, (0, 0), feather / 2, borderType=cv2.BORDER_REPLICATE)
    return mask


def _round_kernel(radius):
    """Disk-shaped uint8 structuring element of the given radius."""
    coords = np.arange(-radius, radius + 1)
    return ((coords[:, None] ** 2 + coords[None, :] ** 2) <= radius * radius).astype(np.uint8)


def _morph(mask, amount, iterations, op):
    # Upstream grows in fixed kernel passes; capping the radius keeps each
    # cv2 call inside its size limits while ceil-iterating stays exact.
    # erode defaults to a +inf border (edges never shrink); we pass 0 so
    # canvas edges erode like an unbounded canvas would.
    kernel = _round_kernel(min(amount, 16))
    return op(mask, kernel, iterations=iterations,
              borderType=cv2.BORDER_CONSTANT, borderValue=0)


def _check_amount(op_name, amount):
    if isinstance(amount, bool) or not isinstance(amount, (int, float)) \
            or not math.isfinite(amount) or not 1 <= amount <= 500:
        raise ValueError(f'{op_name} needs 1-500 pixels')


def expand(mask, amount):
    """Grow a selection by amount pixels (round-kernel dilate)."""
    _check_amount('Expand', amount)
    return _morph(mask, int(amount), math.ceil(int(amount) / 16), cv2.dilate)


def contract(mask, amount):
    """Shrink a selection by amount pixels (round-kernel erode).

    Canvas edges erode too, matching the parity of upstream's unbounded
    erode rather than treating off-canvas as selected.
    """
    _check_amount('Contract', amount)
    return _morph(mask, int(amount), math.ceil(int(amount) / 16), cv2.erode)


def move(mask, dx, dy):
    """Translate a selection, filling exposed canvas with zeros."""
    for v in (dx, dy):
        if isinstance(v, bool) or not isinstance(v, (int, float)) \
                or not math.isfinite(v) or abs(v) > 1_000_000:
            raise ValueError('Selection offsets must be finite numbers')
    matrix = np.float32([[1, 0, round(dx)], [0, 1, round(dy)]])
    return cv2.warpAffine(mask, matrix, (mask.shape[1], mask.shape[0]),
                          flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def border(mask, amount):
    """Ring of pixels within amount of the selection outline."""
    _check_amount('Border', amount)
    radius = int(amount)
    grown = _morph(mask, radius, math.ceil(radius / 16), cv2.dilate)
    shrunk = _morph(mask, radius, math.ceil(radius / 16), cv2.erode)
    return cv2.subtract(grown, shrunk)


def selection_bbox(mask):
    """Document-space (x, y, w, h) covered by a selection, or None if empty."""
    rows = np.nonzero(mask.any(axis=1))[0]
    if not rows.size:
        return None
    columns = np.nonzero(mask.any(axis=0))[0]
    return (int(columns[0]), int(rows[0]),
            int(columns[-1] - columns[0] + 1), int(rows[-1] - rows[0] + 1))


def project_selection(mask, layer):
    """Sample a canvas selection at source-pixel centers through the full transform."""
    w, h = ((layer.pixels.shape[1], layer.pixels.shape[0]) if layer.pixels is not None
            else layer.size)
    if w * h > MAX_SURFACE_PIXELS or max(w, h) > 30000:
        raise ValueError('Layer mask exceeds the surface budget')
    out = np.empty((h, w), np.uint8)
    angle = np.deg2rad(layer.rotation % 360)
    c, s = float(np.cos(angle)), float(np.sin(angle))
    cx, cy = layer.origin[0] + layer.size[0] / 2, layer.origin[1] + layer.size[1] / 2
    xs = (np.arange(w, dtype=np.float32) + .5) / w - .5
    xs *= layer.size[0] * (-1 if layer.flip_x else 1)
    for top in range(0, h, 256):
        ys = (np.arange(top, min(h, top + 256), dtype=np.float32) + .5) / h - .5
        ys *= layer.size[1] * (-1 if layer.flip_y else 1)
        mx = np.asarray(cx + c * xs[None, :] - s * ys[:, None] - .5, np.float32)
        my = np.asarray(cy + s * xs[None, :] + c * ys[:, None] - .5, np.float32)
        out[top:top + len(ys)] = cv2.remap(mask, mx, my, cv2.INTER_LINEAR,
                                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out
