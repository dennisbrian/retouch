"""Mask brush: dab spacing, tip falloff and stroke-wide opacity.

Follows Compositor's ``Document/BrushStroke.swift``: dabs are laid along the
stroke every ``diameter x 0.015`` (hard tip) or ``x 0.025`` (soft tip),
at least a quarter pixel apart, starting with one under the first point.
Hard tips combine with ``max`` (lighten), so overlaps never build up; soft
tips combine with screen, so a soft stroke builds coverage along its length
while keeping a feathered rim. A soft tip is fully on inside
``hardness x radius`` and fades with upstream's normalised Gaussian to zero
at the rim. The finished stroke coverage is applied once at the stroke
opacity: ``mask + (target - mask) x coverage x opacity``, so a stroke that
crosses itself never exceeds its opacity. Tips are snapped to whole pixels,
as upstream's grid tip is.

Coordinates are continuous layer pixels (pixel ``i`` spans ``[i, i + 1)``).
Only the stroke's bounding box is touched; the rest of the mask is shared.

Derived from Compositor (MIT, Copyright (c) 2026 Wonder Assembly LLC), pinned
at 11d8d7a; see ``third_party/compositor/``.
"""
import math

import numpy as np

MAX_DIAMETER = 2100.0          # BrushStroke.maxDiameter
MAX_DABS = 250_000             # refuse pathological strokes before allocating
_FALLOFF_K = 2.5


def spacing(diameter, hardness):
    """Distance between dabs in pixels (BrushStroke.spacingFraction)."""
    return max(0.25, diameter * (0.015 if hardness >= 1 else 0.025))


def falloff(u):
    """Soft-tip falloff for ``u`` = 0 at the hardness radius .. 1 at the rim."""
    u = np.asarray(u, np.float32)
    k = np.float32(_FALLOFF_K)
    value = (np.exp(-k * u * u) - np.exp(-k)) / (1 - np.exp(-k))
    return np.clip(value, 0.0, 1.0)


def tip(diameter, hardness):
    """The brush tip as float32 coverage on a square grid centred on its middle."""
    radius = diameter / 2.0
    size = int(math.ceil(diameter)) + 2
    centre = size / 2.0
    grid = np.arange(size, dtype=np.float32) + 0.5 - centre
    r = np.hypot(grid[None, :], grid[:, None])
    if hardness >= 1:
        # Antialiased silhouette, as Core Graphics fills the ellipse.
        return np.clip(radius + 0.5 - r, 0.0, 1.0).astype(np.float32)
    inner = radius * hardness
    u = (r - inner) / max(radius - inner, 1e-6)
    value = np.where(u <= 0, 1.0, falloff(np.clip(u, 0.0, 1.0)))
    value[u >= 1] = 0.0
    return value.astype(np.float32)


def dab_centres(points, diameter, hardness):
    """Evenly spaced dab positions along the polyline ``points``."""
    points = [(float(x), float(y)) for x, y in points]
    if not points:
        return []
    step = spacing(diameter, hardness)
    centres = [points[0]]
    to_next = step
    previous = points[0]
    for point in points[1:]:
        dx, dy = point[0] - previous[0], point[1] - previous[1]
        length = math.hypot(dx, dy)
        if length > 0:
            if (length - to_next) / step + len(centres) > MAX_DABS:
                raise ValueError('brush stroke is too long for this brush size')
            distance = to_next
            while distance <= length:
                centres.append((previous[0] + dx * distance / length,
                                previous[1] + dy * distance / length))
                distance += step
            to_next = distance - length
        previous = point
    return centres


def stroke_coverage(points, diameter, hardness, width, height):
    """Coverage of one stroke clipped to a ``width`` x ``height`` grid.

    Returns ``(coverage, (x0, y0))`` with float32 coverage in 0..1 for the
    stroke's bounding box, or ``(None, None)`` when nothing lands on the grid.
    """
    diameter, hardness = _check(diameter, hardness)
    centres = dab_centres(points, diameter, hardness)
    shape = tip(diameter, hardness)
    n = shape.shape[0]
    corners = [(int(round(x - n / 2.0)), int(round(y - n / 2.0))) for x, y in centres]
    corners = [(x, y) for x, y in corners
               if x < width and y < height and x + n > 0 and y + n > 0]
    if not corners:
        return None, None
    x0 = max(0, min(x for x, _ in corners))
    y0 = max(0, min(y for _, y in corners))
    x1 = min(width, max(x for x, _ in corners) + n)
    y1 = min(height, max(y for _, y in corners) + n)
    coverage = np.zeros((y1 - y0, x1 - x0), np.float32)
    hard = hardness >= 1
    for cx, cy in corners:
        sx, sy = max(cx, x0), max(cy, y0)
        ex, ey = min(cx + n, x1), min(cy + n, y1)
        region = coverage[sy - y0:ey - y0, sx - x0:ex - x0]
        stamp = shape[sy - cy:ey - cy, sx - cx:ex - cx]
        if hard:
            np.maximum(region, stamp, out=region)
        else:
            region += stamp * (1.0 - region)
    return coverage, (x0, y0)


def paint_mask(mask, points, *, diameter, hardness=1.0, opacity=1.0, reveal=True):
    """Return a new uint8 mask with one stroke painted white (reveal) or black.

    ``mask`` is not modified. Returns ``mask`` itself when the stroke misses it.
    """
    if not (math.isfinite(opacity) and 0.01 <= opacity <= 1):
        raise ValueError('brush opacity must be within 0.01..1')
    height, width = mask.shape
    coverage, origin = stroke_coverage(points, diameter, hardness, width, height)
    if coverage is None:
        return mask
    x0, y0 = origin
    h, w = coverage.shape
    result = np.array(mask, copy=True)
    region = result[y0:y0 + h, x0:x0 + w].astype(np.float32)
    target = 255.0 if reveal else 0.0
    region += (target - region) * (coverage * np.float32(opacity))
    result[y0:y0 + h, x0:x0 + w] = np.clip(np.rint(region), 0, 255).astype(np.uint8)
    return result


def _check(diameter, hardness):
    diameter, hardness = float(diameter), float(hardness)
    if not (math.isfinite(diameter) and 1 <= diameter <= MAX_DIAMETER):
        raise ValueError('brush size must be within 1..%d px' % MAX_DIAMETER)
    if not (math.isfinite(hardness) and 0 <= hardness <= 1):
        raise ValueError('brush hardness must be within 0..1')
    return diameter, hardness
