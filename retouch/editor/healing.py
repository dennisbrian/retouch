"""Spot Healing and Clone Stamp for pixel layers.

Ports Compositor's ``Rendering/HealPixels.c`` (``spot_heal``, three modes) and
the Clone Stamp copy from ``Document/CloneStamp.swift`` (pinned 11d8d7a, MIT;
see ``third_party/compositor``). Both work on a layer's own pixels at their
own resolution and return a new straight-alpha RGBA array; callers record the
result as one undoable pixel replacement.

Differences from the native kernel, kept deliberately small:

- The membrane solve uses red-black successive over-relaxation (vectorised,
  same coarse-to-fine scheme, omega and iteration counts) instead of the C
  row-major Gauss-Seidel sweep, so healed pixels can differ by a level or two
  from the native app; the patch search, ring, grain and blend are the same.
- The native kernel runs over premultiplied bytes. Layers here are straight
  alpha, so pixels are premultiplied first and un-premultiplied after.
- A brush pass with several separate dabs heals each connected dab on its
  own (native heals one stroke at a time).
"""
import math

import cv2
import numpy as np

HEAL_MODES = ('Content-Aware', 'Create Texture', 'Proximity Match')

_OUTSIDE, _RING, _HOLE = 0, 1, 2


def _premultiply(rgba):
    out = rgba.astype(np.float64)
    out[..., :3] = np.floor(out[..., :3] * out[..., 3:4] / 255.0 + 0.5)
    return out.astype(np.uint8)


def _unpremultiply(rgba):
    out = rgba.astype(np.float64)
    alpha = out[..., 3:4]
    with np.errstate(divide='ignore', invalid='ignore'):
        rgb = np.where(alpha > 0, np.floor(out[..., :3] * 255.0 / np.maximum(alpha, 1) + 0.5), 0)
    out[..., :3] = np.clip(rgb, 0, 255)
    return out.astype(np.uint8)


def _hash(x):
    """``heal_hash`` on uint32 arrays (wrapping multiplies)."""
    x = np.asarray(x, dtype=np.uint32)
    with np.errstate(over='ignore'):
        x = x ^ (x >> np.uint32(16))
        x = x * np.uint32(0x7feb352d)
        x = x ^ (x >> np.uint32(15))
        x = x * np.uint32(0x846ca68b)
        x = x ^ (x >> np.uint32(16))
    return x


def _unit(key):
    return (key >> np.uint32(8)).astype(np.float64) / 16777216.0


def _lround(value):
    """C ``lround``: half away from zero."""
    return int(math.copysign(math.floor(abs(value) + 0.5), value))


def _score(image, ring_y, ring_x, wx0, wy0, ww, wh, dx, dy, width, height):
    if abs(dx) < ww and abs(dy) < wh:
        return math.inf
    if wx0 + dx < 0 or wy0 + dy < 0 or wx0 + ww + dx > width or wy0 + wh + dy > height:
        return math.inf
    target = image[wy0 + ring_y, wx0 + ring_x]
    source = image[wy0 + ring_y + dy, wx0 + ring_x + dx]
    diff = target - source
    return float((diff * diff).sum()) / len(ring_y)


def _relax(value, role, iterations, omega=1.8):
    """Red-black SOR over HOLE pixels; RING pixels are fixed, OUTSIDE ignored."""
    h, w = role.shape
    open_ = role != _OUTSIDE
    hole = role == _HOLE
    pad_open = np.pad(open_, 1).astype(np.float32)
    count = (pad_open[:-2, 1:-1] + pad_open[2:, 1:-1]
             + pad_open[1:-1, :-2] + pad_open[1:-1, 2:])
    yy, xx = np.indices((h, w))
    colours = [hole & (count > 0) & ((yy + xx) % 2 == parity) for parity in (0, 1)]
    weight = open_[..., None].astype(np.float32)
    safe = np.maximum(count, 1)[..., None]
    for _ in range(iterations):
        for chosen in colours:
            if not chosen.any():
                continue
            padded = np.pad(value * weight, ((1, 1), (1, 1), (0, 0)))
            total = (padded[:-2, 1:-1] + padded[2:, 1:-1]
                     + padded[1:-1, :-2] + padded[1:-1, 2:])
            update = value + omega * (total / safe - value)
            value[chosen] = update[chosen]


def _solve(value, role, depth=0):
    """``heal_solve``: smooth values over HOLE fixed to RING, coarse first."""
    h, w = role.shape
    iterations = 300
    if w > 32 and h > 32 and depth < 16:
        ch, cw = (h + 1) // 2, (w + 1) // 2
        pad_role = np.full((ch * 2, cw * 2), _OUTSIDE, np.uint8)
        pad_role[:h, :w] = role
        pad_value = np.zeros((ch * 2, cw * 2, 4), np.float32)
        pad_value[:h, :w] = value
        blocks_role = pad_role.reshape(ch, 2, cw, 2).transpose(0, 2, 1, 3).reshape(ch, cw, 4)
        blocks_value = pad_value.reshape(ch, 2, cw, 2, 4).transpose(0, 2, 1, 3, 4).reshape(ch, cw, 4, 4)
        known = (blocks_role == _RING)
        holes = (blocks_role == _HOLE)
        known_n = known.sum(axis=2)
        hole_n = holes.sum(axis=2)
        known_sum = (blocks_value * known[..., None]).sum(axis=2)
        hole_sum = (blocks_value * holes[..., None]).sum(axis=2)
        coarse_role = np.where(known_n > 0, _RING, np.where(hole_n > 0, _HOLE, _OUTSIDE)).astype(np.uint8)
        coarse = np.where((known_n > 0)[..., None], known_sum / np.maximum(known_n, 1)[..., None],
                          np.where((hole_n > 0)[..., None], hole_sum / np.maximum(hole_n, 1)[..., None], 0))
        coarse = coarse.astype(np.float32)
        _solve(coarse, coarse_role, depth + 1)
        up_role = np.repeat(np.repeat(coarse_role, 2, 0), 2, 1)[:h, :w]
        up_value = np.repeat(np.repeat(coarse, 2, 0), 2, 1)[:h, :w]
        take = (role == _HOLE) & (up_role == _HOLE)
        value[take] = up_value[take]
        iterations = 40
    _relax(value, role, iterations)


def _heal_one(rgba, coverage, opacity, mode, seed, origin=(0, 0), canvas=None):
    """``spot_heal`` over premultiplied uint8 RGBA, in place, one spot."""
    height, width = coverage.shape
    ys, xs = np.nonzero(coverage)
    if not len(ys):
        return
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    size = max(x1 - x0, y1 - y0)
    ring = min(16, max(2, size // 8))
    wx0, wy0 = max(0, x0 - ring), max(0, y0 - ring)
    wx1, wy1 = min(width, x1 + ring), min(height, y1 + ring)
    ww, wh = wx1 - wx0, wy1 - wy0

    hole = coverage[wy0:wy1, wx0:wx1] > 0
    near = cv2.dilate(hole.astype(np.uint8), np.ones((2 * ring + 1, 2 * ring + 1), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    role = np.where(hole, _HOLE, np.where(near, _RING, _OUTSIDE)).astype(np.uint8)
    ring_y, ring_x = np.nonzero(role == _RING)
    if not len(ring_y):
        return
    image = rgba.astype(np.float64)

    have_source, ox, oy = False, 0, 0
    if mode != 1:
        factors = (1.05, 1.35, 1.75, 2.25, 2.8)
        best = math.inf
        for f in range(2 if mode == 2 else 5):
            for a in range(24):
                angle = a * math.pi / 12.0
                dx = _lround(math.cos(angle) * factors[f] * ww)
                dy = _lround(math.sin(angle) * factors[f] * wh)
                score = _score(image, ring_y, ring_x, wx0, wy0, ww, wh, dx, dy, width, height)
                if not math.isfinite(score):
                    continue
                score *= (1.0 + 0.6 * f) if mode == 2 else (1.0 + 0.1 * f)  # nearer patches win ties
                if score < best:
                    best, ox, oy = score, dx, dy
        if math.isfinite(best):
            # Fine-tune the alignment so repeating texture lines up.
            cx, cy = ox, oy
            refined = _score(image, ring_y, ring_x, wx0, wy0, ww, wh, cx, cy, width, height)
            for j in range(-3, 4):
                for i in range(-3, 4):
                    score = _score(image, ring_y, ring_x, wx0, wy0, ww, wh, cx + i, cy + j, width, height)
                    if score < refined:
                        refined, ox, oy = score, cx + i, cy + j
            have_source = True

    box = image[wy0:wy1, wx0:wx1]
    source = image[wy0 + oy:wy1 + oy, wx0 + ox:wx1 + ox] if have_source else None
    is_ring = role == _RING
    value = np.zeros((wh, ww, 4), np.float32)
    diff = box - source if have_source else box
    value[is_ring] = diff[is_ring].astype(np.float32)
    mean = diff[is_ring].mean(axis=0)
    detail = np.zeros(3)
    if not have_source:
        # Fine detail around the spot: each pixel against its 4 neighbours' average.
        padded = np.pad(image[..., :3], ((1, 1), (1, 1), (0, 0)))
        inside = np.pad(np.ones((height, width), np.float64), 1)
        py, px = ring_y + wy0 + 1, ring_x + wx0 + 1
        around = (padded[py - 1, px] + padded[py + 1, px] + padded[py, px - 1] + padded[py, px + 1])
        n = (inside[py - 1, px] + inside[py + 1, px] + inside[py, px - 1] + inside[py, px + 1])[:, None]
        d = image[ring_y + wy0, ring_x + wx0, :3] - around / n
        detail = np.sqrt((d * d).sum(axis=0) / len(ring_y)) * 0.9
    value[role == _HOLE] = mean.astype(np.float32)
    _solve(value, role)

    hy, hx = np.nonzero(role == _HOLE)
    iy, ix = hy + wy0, hx + wx0
    target = image[iy, ix]
    amount = (coverage[iy, ix] / 255.0 * opacity)[:, None]
    healed = (source[hy, hx] if have_source else 0) + value[hy, hx].astype(np.float64)
    if not have_source:
        # The grain key uses whole-layer coordinates, as when healing the full layer.
        layer_width = canvas[0] if canvas else width
        key = _hash(np.uint32(seed) ^ _hash(((iy + origin[1]).astype(np.int64) * layer_width
                                              + ix + origin[0]).astype(np.uint32)))
        u1, u2 = _unit(key), _unit(key ^ np.uint32(0x68e31da4))
        grain = np.sqrt(-2.0 * np.log(1.0 - u1)) * np.cos(2.0 * math.pi * u2)
        healed[:, :3] += grain[:, None] * detail[None, :]
    out = target + (healed - target) * amount
    alpha = np.floor(np.clip(out[:, 3], 0, 255) + 0.5)
    rgb = np.floor(np.clip(out[:, :3], 0, alpha[:, None]) + 0.5)
    rgba[iy, ix, 3] = alpha.astype(np.uint8)
    rgba[iy, ix, :3] = rgb.astype(np.uint8)


def spot_heal(pixels, coverage, mode='Content-Aware', opacity=1.0, seed=0):
    """Heal straight-alpha RGBA ``pixels`` under 8-bit ``coverage``.

    ``mode``: Content-Aware copies texture from the nearby patch whose ring
    best matches the ring around the spot; Create Texture fills smoothly from
    the edges and adds grain matching the detail around it; Proximity Match
    takes the closest good patch. Each separate dab is healed on its own.
    Pixels outside the coverage are returned unchanged (bit-identical).
    """
    if mode not in HEAL_MODES:
        raise ValueError('Choose a healing mode: ' + ', '.join(HEAL_MODES))
    pixels = np.asarray(pixels)
    coverage = np.asarray(coverage)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError('expected uint8 RGBA pixels')
    if coverage.dtype != np.uint8 or coverage.shape != pixels.shape[:2]:
        raise ValueError('coverage must be uint8 and match the layer size')
    opacity = float(opacity)
    if not 0 <= opacity <= 1:
        raise ValueError('Healing opacity must be between 0 and 1.')
    if not coverage.any() or opacity == 0:
        return pixels
    count, labels, stats, _ = cv2.connectedComponentsWithStats((coverage > 0).astype(np.uint8), connectivity=8)
    height, width = coverage.shape
    out = np.array(pixels, copy=True)
    for label in range(1, count):
        x, y, w, h, _ = stats[label]
        # The patch search reaches at most 2.8 work boxes (spot + ring) away, plus
        # the 3 px fine-tune, so a window that wide sees everything the native
        # kernel could sample from the whole layer.
        reach = int(math.ceil(2.8 * (max(w, h) + 32))) + max(w, h) + 40
        rx0, ry0 = max(0, x - reach), max(0, y - reach)
        rx1, ry1 = min(width, x + w + reach), min(height, y + h + reach)
        spot = np.where(labels[ry0:ry1, rx0:rx1] == label, coverage[ry0:ry1, rx0:rx1], 0).astype(np.uint8)
        region = _premultiply(out[ry0:ry1, rx0:rx1])
        _heal_one(region, spot, opacity, HEAL_MODES.index(mode), seed + label, origin=(rx0, ry0), canvas=(width, height))
        changed = spot > 0
        window = out[ry0:ry1, rx0:rx1]
        window[changed] = _unpremultiply(region[changed][None])[0]
    return out


def clone_stamp(pixels, coverage, offset, opacity=1.0):
    """Copy the layer's own pixels from ``offset`` (dx, dy) away under ``coverage``.

    A painted pixel at (x, y) takes the pixel at (x + dx, y + dy), drawn
    source-over through the brush coverage like the native Clone Stamp;
    where the source falls outside the layer nothing is drawn.
    """
    pixels = np.asarray(pixels)
    coverage = np.asarray(coverage)
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 4:
        raise ValueError('expected uint8 RGBA pixels')
    if coverage.dtype != np.uint8 or coverage.shape != pixels.shape[:2]:
        raise ValueError('coverage must be uint8 and match the layer size')
    dx, dy = offset
    if not all(isinstance(v, int) and not isinstance(v, bool) for v in (dx, dy)):
        raise ValueError('Clone offsets must be whole pixels')
    if dx == 0 and dy == 0:
        raise ValueError('Set a clone source offset first')
    opacity = float(opacity)
    if not 0 <= opacity <= 1:
        raise ValueError('Clone opacity must be between 0 and 1.')
    height, width = coverage.shape
    ys, xs = np.nonzero(coverage)
    sy, sx = ys + dy, xs + dx
    inside = (sx >= 0) & (sy >= 0) & (sx < width) & (sy < height)
    ys, xs, sy, sx = ys[inside], xs[inside], sy[inside], sx[inside]
    out = np.array(pixels, copy=True)
    if not len(ys) or opacity == 0:
        return out
    dst = _premultiply(pixels[ys, xs][None])[0].astype(np.float64)
    src = _premultiply(pixels[sy, sx][None])[0].astype(np.float64)
    cov = (coverage[ys, xs] / 255.0 * opacity)[:, None]
    result = src * cov + dst * (1 - src[:, 3:4] / 255.0 * cov)
    result = np.floor(np.clip(result, 0, 255) + 0.5)
    result[:, :3] = np.minimum(result[:, :3], result[:, 3:4])
    out[ys, xs] = _unpremultiply(result.astype(np.uint8)[None])[0]
    return out
