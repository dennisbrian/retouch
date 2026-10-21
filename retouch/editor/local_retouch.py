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
# Blur/Smudge/Liquify are dab-based warps: the working copy plus one dab
# scratch must fit three copies of the brushed region (upstream WarpStroke
# works at canvas size but only ever touches one stroke's rectangle).
MAX_WARP_SEARCH_PIXELS = 512 * 512
BLUR_MODES = ('Blur', 'Smudge', 'Liquify')


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


def _warp_settings(operation: str, brush_size: Any, strength: Any, hardness: Any = 0.5) -> tuple[int, float, float]:
    """Validate Blur/Smudge/Liquify settings; return (diameter, strength 0.01..1, hardness 0..0.98)."""
    if (isinstance(brush_size, bool) or not isinstance(brush_size, (int, float))
            or not math.isfinite(brush_size) or brush_size != int(brush_size)
            or not 2 <= brush_size <= 200):
        raise ValueError('Brush size must be a whole number from 2 to 200 pixels.')
    if (isinstance(strength, bool) or not isinstance(strength, (int, float))
            or not math.isfinite(strength) or not 1 <= strength <= 100):
        raise ValueError('Brush strength must be between 1 and 100.')
    if (isinstance(hardness, bool) or not isinstance(hardness, (int, float))
            or not math.isfinite(hardness) or not 0 <= hardness <= 100):
        raise ValueError('Brush hardness must be between 0 and 100.')
    return int(brush_size), strength / 100.0, min(0.98, hardness / 100.0)


def _weight(distance: np.ndarray, radius: float, hardness: float) -> np.ndarray:
    """Falloff at normalised distance u from a dab's centre (native WarpStroke.weight)."""
    u = distance / radius
    w = np.zeros_like(u)
    inside = u < 1
    core = inside & (u <= hardness)
    rim = inside & (u > hardness)
    w[core] = 1.0
    if rim.any():
        t = (1.0 - u[rim]) / (1.0 - hardness)
        w[rim] = t * t * (3.0 - 2.0 * t)
    return w


def _stroke_path(mask: np.ndarray, radius: int) -> list[tuple[float, float]]:
    """Dab centres from a painted mask, in traversal order.

    Gradio's SketchPad flattens layers, so per-dab direction is not available:
    the path is the mask's medial ridge — the crest of the distance
    transform, walked end to end — and each dab's direction is the local
    ridge tangent. `ponytail:` a pressure tablet UI with real stroke points
    should replace this with the recorded dab sequence.
    """
    dist = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    # One ridge walk from the global maximum of the distance transform is the
    # stroke's centreline; anything the walk misses (blobs, junctions) gets a
    # dab at its own local maximum, bounded so worst-case cost stays linear.
    y, x = np.unravel_index(np.argmax(dist), dist.shape)
    path = (_walk_ridge(dist, x, y) + list(reversed(_walk_ridge(dist, x, y, reverse=True)))
            if dist[y, x] >= 1 else [])
    step = max(1.0, radius * 0.25)
    points = [path[i] for i in range(0, len(path), max(1, int(step)))] if path else []
    # ponytail: blobs wider than the walk's coverage get one dab at their
    # centre of mass; full skeletonisation (skimage.morphology.skeletonize)
    # is the upgrade if junction-heavy strokes matter.
    return points


def _walk_ridge(dist: np.ndarray, x: int, y: int, reverse: bool = False) -> list[tuple[float, float]]:
    """One end of the ridge from (x, y): repeatedly step to the best 8-neighbour
    whose distance is within reach (crest points downhill to the ends)."""
    path = [(float(x), float(y))]
    height, width = dist.shape
    visited = {(x, y)}
    while True:
        best = None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < width and 0 <= ny < height) or (nx, ny) in visited:
                    continue
                if dist[ny, nx] <= 0:
                    continue
                # Downhill continues the ridge; slight uphill (<=1.5) bridges gaps.
                if best is None or dist[ny, nx] < dist[best[1], best[0]]:
                    best = (nx, ny)
        if best is None:
            break
        x, y = best
        visited.add((x, y))
        path.append((float(x), float(y)))
        if len(path) > 4 * (width + height):
            break
    if reverse:
        path = path[1:]
    return path


def _blur_region(rgba: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-softened copy of the region (upstream: CI GaussianBlur)."""
    out = cv2.GaussianBlur(rgba, (0, 0), sigmaX=sigma, sigmaY=sigma,
                           borderType=cv2.BORDER_REPLICATE)
    return out


def _warp_pixels(pixels: np.ndarray, mask: np.ndarray, operation: str,
                 radius: int, hardness: float, strength: float) -> np.ndarray:
    """Run native WarpStroke dabs over the region's pixels.

    Smudge picks up what was under the first dab and drags it along
    (native pickUp/smudge); Liquify pushes pixels with each dab's motion
    (native push, bilinear over a pre-dab scratch copy).
    """
    region = pixels.copy()
    height, width = region.shape[:2]
    points = _stroke_path(mask, radius)
    if not points:
        return pixels
    if operation == 'Smudge':
        spacing = max(1.0, 2 * radius * 0.005)
    else:
        spacing = max(1.0, 2 * radius * 0.025)
    # Native dabs every `spacing` px along the path.
    dabs: list[tuple[float, float]] = []
    last = None
    for point in points:
        if last is None or math.hypot(point[0] - last[0], point[1] - last[1]) >= spacing:
            dabs.append(point)
            last = point
    if len(dabs) < 2:
        # One dab (or all dabs coincident): Liquify has no motion to push
        # with and Smudge's carried square equals what is under it.
        raise ValueError('Drag to move pixels — a single click applies no warp.')
    if operation == 'Smudge':
        # Native pickUp: the carried square is what was under the first dab.
        cx, cy = int(round(dabs[0][0])), int(round(dabs[0][1]))
        carried = np.zeros((2 * radius + 1, 2 * radius + 1, 4), np.float32)
        y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
        sy0, sy1 = y0 - (cy - radius), y1 - (cy - radius)
        sx0, sx1 = x0 - (cx - radius), x1 - (cx - radius)
        carried[sy0:sy1, sx0:sx1] = region[y0:y1, x0:x1].astype(np.float32)
        keep = np.float32(strength)
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
        dist2 = np.sqrt(xx * xx + yy * yy)
        w = _weight(dist2, float(radius) + 0.5, hardness)[..., None]
        for (px, py) in dabs:
            cx, cy = int(round(px)), int(round(py))
            y0, y1 = max(0, cy - radius), min(height, cy + radius + 1)
            x0, x1 = max(0, cx - radius), min(width, cx + radius + 1)
            if y0 >= y1 or x0 >= x1:
                continue
            sy0, sy1 = y0 - (cy - radius), y1 - (cy - radius)
            sx0, sx1 = x0 - (cx - radius), x1 - (cx - radius)
            under = region[y0:y1, x0:x1].astype(np.float32)
            ww = w[sy0:sy1, sx0:sx1]
            painted = under + (carried[sy0:sy1, sx0:sx1] - under) * ww * keep
            region[y0:y1, x0:x1] = np.clip(np.rint(painted), 0, 255).astype(np.uint8)
            carried[sy0:sy1, sx0:sx1] = painted
    else:
        # Native push, dab by dab: sample the pre-dab copy from behind the motion.
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1].astype(np.float32)
        dist2 = np.sqrt(xx * xx + yy * yy)
        w = _weight(dist2, float(radius) + 0.5, hardness)
        previous = dabs[0]
        for (px, py) in dabs[1:]:
            dx, dy = px - previous[0], py - previous[1]
            previous = (px, py)
            if dx == dy == 0:
                continue
            move = np.array([dx, dy], np.float32) * strength
            margin = int(math.ceil(max(abs(move[0]), abs(move[1])))) + 2
            cx, cy = int(round(px)), int(round(py))
            x0, x1 = max(0, cx - radius - margin), min(width - 1, cx + radius + margin)
            y0, y1 = max(0, cy - radius - margin), min(height - 1, cy + radius + margin)
            if x0 > x1 or y0 > y1:
                continue
            scratch = region[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
            gy, gx = np.mgrid[y0:y1 + 1, x0:x1 + 1]
            d = np.sqrt((gx - px) ** 2 + (gy - py) ** 2).astype(np.float32)
            ww = _weight(d, float(radius) + 0.5, hardness)
            touched = ww > 0
            sx = np.clip(gx - x0 - move[0] * ww, 0, x1 - x0).astype(np.float32)
            sy = np.clip(gy - y0 - move[1] * ww, 0, y1 - y0).astype(np.float32)
            # cv2.remap samples the pre-dab scratch; untouched pixels sample themselves.
            remapped = cv2.remap(scratch, sx, sy, interpolation=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
            target = region[y0:y1 + 1, x0:x1 + 1]
            blended = target.astype(np.float32)
            m = touched[..., None].astype(np.float32)
            blended = remapped * m + blended * (1.0 - m)
            region[y0:y1 + 1, x0:x1 + 1] = np.clip(np.rint(blended), 0, 255).astype(np.uint8)
    return region


def apply_local_retouch(history: DocumentHistory, capture: LocalCapture, canvas: Any,
                        operation: str, opacity: float = 100, method: str = 'Telea',
                        selection: np.ndarray | None = None,
                        clone_offset: tuple[int, int] = (0, 0),
                        donor_selection: np.ndarray | None = None,
                        search_distance: int = 64,
                        brush_size: int = 40,
                        brush_strength: float = 70,
                        brush_hardness: float = 50) -> str:
    """Bake the existing op once into a masked patch; preserve all source layers."""
    doc = history.document
    if not isinstance(capture, LocalCapture):
        raise ValueError('Load the composite into the local retouch canvas first.')
    if (capture.document_id, capture.revision, capture.size) != (
            doc.document_id, history.revision, (doc.width, doc.height)):
        raise ValueError('Document changed. Reload the composite and repaint before applying local retouch.')
    if operation not in ('Heal', 'Clone', 'Dodge', 'Burn', *BLUR_MODES):
        raise ValueError('Choose Heal, Clone, Dodge, Burn, Blur, Smudge or Liquify.')
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
    if operation in BLUR_MODES:
        diameter, strength, hardness = _warp_settings(operation, brush_size, brush_strength,
                                                     brush_hardness)
        radius = int(math.ceil(diameter / 2))
        padding = (3 * max(1, radius) if operation == 'Blur'
                   else radius + int(math.ceil(diameter * strength)) + 2)
    else:
        radius = 0
        padding = (int(search_distance) if texture_heal else
                   min(64, max(32, 4 * heal_radius)) if operation == 'Heal' else 0)
    left, top = max(0, x - padding), max(0, y - padding)
    right, bottom = min(doc.width, x + width + padding), min(doc.height, y + height + padding)
    if texture_heal and (right - left) * (bottom - top) > MAX_TEXTURE_SEARCH_PIXELS:
        raise ValueError('Texture search area is too large. Reduce search distance or paint a smaller area.')
    if operation in BLUR_MODES and (right - left) * (bottom - top) > MAX_WARP_SEARCH_PIXELS:
        raise ValueError('The brushed warp area is too large. Paint a smaller area or use a smaller brush.')
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
    elif operation in BLUR_MODES:
        # Soft strokes set result opacity in the brush falloff, as native's
        # WarpStroke weight does; the visible edge comes from the stroke alpha.
        if operation == 'Blur':
            softened = _blur_region(original, sigma=float(radius))
            alpha = coverage[..., None].astype(np.uint32)
            result = ((softened.astype(np.uint32) * alpha
                       + original.astype(np.uint32) * (255 - alpha) + 127) // 255).astype(np.uint8)
        else:
            warped = _warp_pixels(original, coverage, operation, radius, hardness, strength)
            alpha = coverage[..., None].astype(np.uint32)
            result = ((warped.astype(np.uint32) * alpha
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
