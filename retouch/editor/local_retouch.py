"""Brush-driven healing and dodge/burn as small, undoable overlay layers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from retouch.heal import _auto_radius, heal_region
from retouch.regions import apply_local_adjustment

from .commands import DocumentHistory
from .composite import composite
from .document import Layer


@dataclass(frozen=True)
class LocalCapture:
    document_id: str
    revision: int
    size: tuple[int, int]
    preview_size: tuple[int, int]


def capture_composite(history: DocumentHistory, preview_side: int = 1024) -> tuple[dict, LocalCapture]:
    """Only the server-rendered composite is authoritative; browser pixels aren't."""
    revision = history.revision
    doc = history.document
    pixels = composite(doc)
    if not (pixels[..., 3] == 255).all():
        raise ValueError('Local retouch currently requires an opaque composite.')
    factor = min(1, preview_side / max(doc.width, doc.height))
    size = (max(1, round(doc.width * factor)), max(1, round(doc.height * factor)))
    preview = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA) if factor < 1 else pixels
    if (history.document.document_id, history.revision) != (doc.document_id, revision):
        raise ValueError('Document changed while loading local retouch. Load the composite again.')
    return ({'background': preview, 'layers': [], 'composite': preview},
            LocalCapture(doc.document_id, revision, (doc.width, doc.height), size))


def painted_mask(canvas: Any, capture: LocalCapture) -> np.ndarray:
    """Use stroke alpha once, independent of RGB, at document resolution."""
    if not isinstance(canvas, dict) or not isinstance(canvas.get('layers'), (list, tuple)):
        raise ValueError('Paint a region on the local retouch canvas first.')
    width, height = capture.preview_size
    mask = np.zeros((height, width), np.uint8)
    for stroke in canvas['layers']:
        stroke = np.asarray(stroke)
        if stroke.dtype != np.uint8 or stroke.shape != (height, width, 4):
            raise ValueError('The local brush dimensions changed. Reload the composite and repaint.')
        np.maximum(mask, stroke[..., 3], out=mask)
    if capture.size != capture.preview_size:
        mask = cv2.resize(mask, capture.size, interpolation=cv2.INTER_LINEAR)
    if not mask.any():
        raise ValueError('Paint a region on the local retouch canvas first.')
    return mask


def apply_local_retouch(history: DocumentHistory, capture: LocalCapture, canvas: Any,
                        operation: str, opacity: float = 100, method: str = 'Telea',
                        selection: np.ndarray | None = None) -> str:
    """Bake the existing op once into a masked patch; preserve all source layers."""
    doc = history.document
    if not isinstance(capture, LocalCapture):
        raise ValueError('Load the composite into the local retouch canvas first.')
    if (capture.document_id, capture.revision, capture.size) != (
            doc.document_id, history.revision, (doc.width, doc.height)):
        raise ValueError('Document changed. Reload the composite and repaint before applying local retouch.')
    if operation not in ('Heal', 'Dodge', 'Burn'):
        raise ValueError('Choose Heal, Dodge or Burn.')
    if (isinstance(opacity, bool) or not isinstance(opacity, (int, float))
            or not math.isfinite(opacity) or not 0 < opacity <= 100):
        raise ValueError('Result opacity must be greater than 0 and at most 100.')
    if method not in ('Telea', 'Navier-Stokes'):
        raise ValueError('Choose Telea or Navier-Stokes healing.')
    mask = painted_mask(canvas, capture)
    if selection is not None:
        if selection.dtype != np.uint8 or selection.shape != mask.shape:
            raise ValueError('Selection does not match the current document.')
        mask = np.minimum(mask, selection)
        if not mask.any():
            raise ValueError('The strokes do not overlap the current selection.')
    pixels = composite(doc)
    if not (pixels[..., 3] == 255).all():
        raise ValueError('Local retouch currently requires an opaque composite.')
    x, y, width, height = cv2.boundingRect(mask)
    padding = max(32, 4 * _auto_radius(mask)) if operation == 'Heal' else 0
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(doc.width, x + width + padding), min(doc.height, y + height + padding)
    coverage = mask[top:bottom, left:right]
    original = pixels[top:bottom, left:right]
    bgr = cv2.cvtColor(original, cv2.COLOR_RGBA2BGR)
    if operation == 'Heal':
        result = heal_region(bgr, coverage.astype(np.float32) / 255,
                             method='telea' if method == 'Telea' else 'ns')
    else:
        result = apply_local_adjustment(bgr, coverage.astype(np.float32) / 255,
                                        operation.lower(), 1.0)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGBA)
    support = coverage > 0
    # LAB round trips and healer feathering may touch other pixels. Enforce the
    # painted support exactly and do not multiply its soft alpha a second time.
    result[~support] = original[~support]
    if np.array_equal(result, original):
        raise ValueError('The operation made no visible change. Try a different region or healing method.')
    if (history.document.document_id, history.revision) != (capture.document_id, capture.revision):
        raise ValueError('Document changed while applying local retouch. Reload the composite and repaint.')
    layer = Layer(operation + (' · ' + method if operation == 'Heal' else ''),
                  pixels=result, origin=(left, top), size=(right - left, bottom - top),
                  mask=support.astype(np.uint8) * 255, opacity=opacity / 100)
    history.add_layer(layer)
    return layer.id
