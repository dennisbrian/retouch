"""Revision-bound clone anchors on a server-rendered preview."""
from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from .local_retouch import LocalCapture, capture_composite


@dataclass(frozen=True)
class ClonePlacement:
    capture: LocalCapture
    preview: np.ndarray
    source: tuple[int, int] | None = None
    destination: tuple[int, int] | None = None


def _check_revision(history, capture):
    doc = history.document
    if not isinstance(capture, LocalCapture) or (
            capture.document_id, capture.revision, capture.size) != (
            doc.document_id, history.revision, (doc.width, doc.height)):
        raise ValueError('Document changed or no composite loaded. Reload local retouch and its clone picker.')


def prepare_clone_picker(history, capture):
    _check_revision(history, capture)
    canvas, _ = capture_composite(history)
    # Use exactly the brush preview dimensions, including custom preview sizes.
    preview = cv2.resize(canvas['background'], capture.preview_size, interpolation=cv2.INTER_AREA)
    _check_revision(history, capture)
    preview.setflags(write=False)
    return ClonePlacement(capture, preview)


def pick_clone_point(history, placement, point, target):
    if not isinstance(placement, ClonePlacement):
        raise ValueError('Load the clone picker first.')
    _check_revision(history, placement.capture)
    if target not in ('Source', 'Destination'):
        raise ValueError('Choose Source or Destination.')
    width, height = placement.capture.preview_size
    if (not isinstance(point, (tuple, list)) or len(point) != 2
            or any(type(v) is not int or not 0 <= v < limit
                   for v, limit in zip(point, (width, height)))):
        raise ValueError('Click inside the clone picker image.')
    native = tuple(min(limit - 1, max(0, round((v + .5) * limit / size - .5)))
                   for v, limit, size in zip(point, placement.capture.size, (width, height)))
    updated = replace(placement, **{target.lower(): native})
    image = updated.preview.copy()
    for anchor, color in ((updated.source, (0, 220, 80, 255)),
                          (updated.destination, (255, 40, 180, 255))):
        if anchor is not None:
            x, y = (round((v + .5) * size / limit - .5)
                    for v, size, limit in zip(anchor, (width, height), updated.capture.size))
            cv2.drawMarker(image, (x, y), color, cv2.MARKER_CROSS, 17, 2)
            cv2.circle(image, (x, y), 10, color, 1)
    offset = (0, 0)
    if updated.source is not None and updated.destination is not None:
        offset = tuple(s - d for s, d in zip(updated.source, updated.destination))
        message = (f'Source {updated.source}; destination {updated.destination}; offset {offset}. '
                   'Paint around the destination on the local retouch canvas, then apply Clone.')
    else:
        message = f'{target} set at {native}. Click the other anchor to calculate the offset.'
    return updated, image, offset, message
