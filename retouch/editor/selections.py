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
