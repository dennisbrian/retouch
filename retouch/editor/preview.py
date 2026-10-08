"""Bounded-resolution preview of a document for the editor canvas.

Layers keep their native pixels; the preview composites copies scaled so the
canvas's long side is at most ``max_side``. Scaled copies are cached by the
identity of the read-only source arrays, so an edit only rescales what it
replaced (a brushed mask, not every photo layer). Colour is scaled
premultiplied, so transparent edges do not pick up dark fringes.
"""
import cv2
import numpy as np

from .composite import composite
from .document import Document, Layer

PREVIEW_MAX_SIDE = 2048


def preview_scale(width, height, max_side=PREVIEW_MAX_SIDE):
    return min(1.0, float(max_side) / max(width, height))


def scaled_rect(origin, size, scale):
    """Layer rectangle in preview pixels (origin rounded, size at least 1)."""
    x, y = origin
    w, h = size
    return (int(round(x * scale)), int(round(y * scale)),
            max(1, int(round(w * scale))), max(1, int(round(h * scale))))


class PreviewRenderer:
    """Composite a document at preview scale, reusing scaled layer copies."""

    def __init__(self, max_side=PREVIEW_MAX_SIDE):
        self.max_side = max_side
        self._cache = {}

    def render(self, document):
        """RGBA uint8 preview of ``document`` and the scale it was drawn at."""
        scale = preview_scale(document.width, document.height, self.max_side)
        preview = Document(
            width=max(1, int(round(document.width * scale))),
            height=max(1, int(round(document.height * scale))),
            resolution=document.resolution, document_id=document.document_id)
        live = set()
        for layer in document.layers:
            if layer.pixels is None:
                continue
            x, y, w, h = scaled_rect(layer.origin, layer.size, scale)
            pixels = self._scaled(layer.pixels, w, h, live, rgba=True)
            mask = None
            if layer.mask is not None:
                mask = self._scaled(layer.mask, w, h, live, rgba=False)
            preview.layers.append(Layer(
                name=layer.name, pixels=pixels, id=layer.id, visible=layer.visible,
                opacity=layer.opacity, blend_mode=layer.blend_mode, origin=(x, y),
                flip_x=layer.flip_x, flip_y=layer.flip_y, sampling=layer.sampling,
                mask=mask, mask_enabled=layer.mask_enabled))
        # Forget copies of arrays this document no longer uses.
        self._cache = {key: value for key, value in self._cache.items() if key in live}
        return composite(preview), scale

    def mask(self, layer, scale):
        """A layer's mask at preview scale, in document orientation (flips applied)."""
        if layer.mask is None:
            return None
        _, _, w, h = scaled_rect(layer.origin, layer.size, scale)
        live = set(self._cache)
        mask = self._scaled(layer.mask, w, h, live, rgba=False)
        if layer.flip_y:
            mask = mask[::-1]
        if layer.flip_x:
            mask = mask[:, ::-1]
        return np.ascontiguousarray(mask)

    def _scaled(self, array, w, h, live, rgba):
        key = (id(array), w, h)
        live.add(key)
        hit = self._cache.get(key)
        if hit is not None and hit[0] is array:
            return hit[1]
        if array.shape[:2] == (h, w):
            small = array
        elif rgba:
            small = _resize_rgba(array, w, h)
        elif array.size == 1:
            small = np.full((h, w), array.reshape(-1)[0], np.uint8)
        else:
            small = _resize(array, w, h)
        small = np.ascontiguousarray(small)
        small.flags.writeable = False
        # Keep the source alive with its copy so its id cannot be reused.
        self._cache[key] = (array, small)
        return small


def _resize(array, w, h):
    shrinking = w < array.shape[1] and h < array.shape[0]
    return cv2.resize(array, (w, h),
                      interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


def _resize_rgba(pixels, w, h):
    alpha = pixels[..., 3]
    if alpha.min() == 255:
        rgb = _resize(np.ascontiguousarray(pixels[..., :3]), w, h)
        return np.dstack([rgb, np.full((h, w), 255, np.uint8)])
    a = alpha.astype(np.float32) / 255.0
    premultiplied = pixels[..., :3].astype(np.float32) * a[..., None]
    rgb = _resize(premultiplied, w, h)
    a_small = _resize(a, w, h)
    safe = np.where(a_small > 0, a_small, 1.0)
    colour = np.clip(np.rint(rgb / safe[..., None]), 0, 255)
    return np.dstack([colour, np.clip(np.rint(a_small * 255), 0, 255)]).astype(np.uint8)
