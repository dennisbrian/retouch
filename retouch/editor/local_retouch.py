"""Brush-driven healing, cloning and dodge/burn as undoable overlay layers."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import cv2
import numpy as np

from retouch.heal import _auto_radius, heal_region
from retouch.patchmatch import patchmatch_fill
from retouch.regions import apply_local_adjustment

from .commands import DocumentHistory
from .composite import composite
from .document import Layer

MAX_TEXTURE_REPAIR_PIXELS = 4096
MAX_TEXTURE_SEARCH_PIXELS = 512 * 512


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
                        selection: np.ndarray | None = None,
                        clone_offset: tuple[int, int] = (0, 0),
                        donor_selection: np.ndarray | None = None,
                        search_distance: int = 64) -> str:
    """Bake the existing op once into a masked patch; preserve all source layers."""
    doc = history.document
    if not isinstance(capture, LocalCapture):
        raise ValueError('Load the composite into the local retouch canvas first.')
    if (capture.document_id, capture.revision, capture.size) != (
            doc.document_id, history.revision, (doc.width, doc.height)):
        raise ValueError('Document changed. Reload the composite and repaint before applying local retouch.')
    if operation not in ('Heal', 'Clone', 'Dodge', 'Burn'):
        raise ValueError('Choose Heal, Clone, Dodge or Burn.')
    if (isinstance(opacity, bool) or not isinstance(opacity, (int, float))
            or not math.isfinite(opacity) or not 0 < opacity <= 100):
        raise ValueError('Result opacity must be greater than 0 and at most 100.')
    if method not in ('Telea', 'Navier-Stokes', 'Texture-aware'):
        raise ValueError('Choose Telea, Navier-Stokes or Texture-aware healing.')
    texture_heal = operation == 'Heal' and method == 'Texture-aware'
    if texture_heal:
        if (isinstance(search_distance, bool) or not isinstance(search_distance, (int, float))
                or not math.isfinite(search_distance) or search_distance != int(search_distance)
                or not 16 <= search_distance <= 256):
            raise ValueError('Texture search distance must be a whole number from 16 to 256 pixels.')
        if donor_selection is not None and (not isinstance(donor_selection, np.ndarray)
                or donor_selection.dtype != np.uint8 or donor_selection.shape != (doc.height, doc.width)):
            raise ValueError('Donor selection does not match the current document.')
    if operation == 'Clone':
        if (not isinstance(clone_offset, (tuple, list)) or len(clone_offset) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float))
                       or not math.isfinite(v) or v != int(v)
                       or abs(v) >= limit for v, limit in zip(clone_offset, capture.size))):
            raise ValueError('Clone source offsets must be whole pixels within the canvas dimensions.')
        dx, dy = map(int, clone_offset)
        if dx == dy == 0:
            raise ValueError('Set a nonzero clone source offset, then paint the destination.')
    mask = painted_mask(canvas, capture)
    if selection is not None:
        if selection.dtype != np.uint8 or selection.shape != mask.shape:
            raise ValueError('Selection does not match the current document.')
        mask = np.minimum(mask, selection)
        if not mask.any():
            raise ValueError('The strokes do not overlap the current selection.')
    x, y, width, height = cv2.boundingRect(mask)
    if operation == 'Clone' and (x + dx < 0 or y + dy < 0
            or x + width + dx > doc.width or y + height + dy > doc.height):
        raise ValueError('The clone source falls outside the canvas. Adjust the offset or repaint.')
    if texture_heal and np.count_nonzero(mask) > MAX_TEXTURE_REPAIR_PIXELS:
        raise ValueError('Texture-aware healing supports up to 4096 painted native pixels. '
                         'Use smaller strokes or the fast healing methods.')
    # Local strokes are brush-sized: the auto radius scales with the stroke's
    # bounding-box diagonal, which made a 40x100 preview stroke pad to a
    # ~470x710 inpaint region at radius 42 (2.3 s in cv2.inpaint alone).
    # Clamped, the heal still converges on brush-sized defects.
    heal_radius = min(_auto_radius(mask), 8)
    padding = (int(search_distance) if texture_heal else
               min(64, max(32, 4 * heal_radius)) if operation == 'Heal' else 0)
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(doc.width, x + width + padding), min(doc.height, y + height + padding)
    if texture_heal and (right - left) * (bottom - top) > MAX_TEXTURE_SEARCH_PIXELS:
        raise ValueError('Texture search area is too large. Reduce search distance or paint a smaller area.')
    pixels = composite(doc)
    if not (pixels[..., 3] == 255).all():
        raise ValueError('Local retouch currently requires an opaque composite.')
    coverage = mask[top:bottom, left:right]
    original = pixels[top:bottom, left:right]
    if operation == 'Clone':
        # Always sample the captured revision, including when source and target
        # overlap. Blend soft strokes once; the layer mask carries support only.
        donor = pixels[top + dy:bottom + dy, left + dx:right + dx]
        alpha = coverage[..., None].astype(np.uint32)
        result = ((donor.astype(np.uint32) * alpha
                   + original.astype(np.uint32) * (255 - alpha) + 127) // 255).astype(np.uint8)
    else:
        bgr = cv2.cvtColor(original, cv2.COLOR_RGBA2BGR)
        if texture_heal:
            donor = (None if donor_selection is None else donor_selection[top:bottom, left:right])
            if donor is not None and not np.any((donor > 0) & (coverage == 0)):
                raise ValueError('No donor pixels in the search area outside the strokes. '
                                 'Expand the donor selection or search distance.')
            report = {}
            result = patchmatch_fill(bgr, coverage, source_mask=donor, patch_size=7,
                                     iterations=3, seed=0, max_candidates=64, report=report)
            if report.get('executed') == 'abstain':
                raise ValueError('No suitable donor pixels. Change the selection or search distance.')
            alpha = coverage[..., None].astype(np.uint32)
            result = ((result.astype(np.uint32) * alpha
                       + bgr.astype(np.uint32) * (255-alpha) + 127) // 255).astype(np.uint8)
            if report.get('executed') != 'patchmatch':
                method += ' · donor-constrained fallback'
        elif operation == 'Heal':
            result = heal_region(bgr, coverage.astype(np.float32) / 255,
                                 method='telea' if method == 'Telea' else 'ns',
                                 radius=heal_radius)
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
