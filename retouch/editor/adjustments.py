"""Supported Compositor adjustment records and alpha-preserving sRGB grading.

Derived from Compositor ImageAdjustments.swift and Levels.swift (MIT,
Copyright 2026 Wonder Assembly LLC), pinned at 11d8d7a; see third_party/compositor.
"""
from copy import deepcopy
import math

import numpy as np

from .document import UnsupportedFeature

KINDS = ('Exposure', 'Levels', 'Invert')
KINDS = ('Exposure', 'Levels', 'Invert', 'Saturation', 'Vibrance')
CHANNELS = ('RGB', 'Red', 'Green', 'Blue')
_KEYS = {'kind', 'hue', 'saturation', 'lightness', 'colorize', 'levels', 'curves',
         'exposureSettings', 'saturationSettings', 'vibranceSettings'}


def adjustment(kind='Exposure', exposure=0, offset=0, gamma=1,
               black=0, white=255, red_gamma=1, green_gamma=1, blue_gamma=1,
               saturation=0, vibrance=0):
    """Build a native-format record; all channels use identity by default."""
    if kind not in ('Levels', 'Saturation', 'Vibrance'):
        black, white = 0, 255
        red_gamma = green_gamma = blue_gamma = 1
    ranges = [dict(black=black, white=white, gamma=gamma if kind == 'Levels' else 1,
                   outputBlack=0, outputWhite=255)]
    ranges += [dict(black=0, white=255, gamma=g, outputBlack=0, outputWhite=255)
               for g in (red_gamma, green_gamma, blue_gamma)]
    record = dict(kind=kind, hue=0, saturation=saturation, lightness=0, colorize=False,
                  levels=dict(channel='RGB', ranges=ranges),
                  curves=dict(channel='RGB', channels=[
                      [dict(x=0, y=0), dict(x=255, y=255)] for _ in range(4)]))
    if kind == 'Exposure':
        record['exposureSettings'] = dict(exposure=exposure, offset=offset, gamma=gamma)
    if kind == 'Saturation':
        record['saturationSettings'] = dict(saturation=saturation)
    if kind == 'Vibrance':
        record['vibranceSettings'] = dict(vibrance=vibrance)
    validate_adjustment(record)
    return deepcopy(record)


def _number(value, lo, hi):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and lo <= value <= hi)


def validate_adjustment(record):
    if not isinstance(record, dict):
        raise ValueError('Adjustment must be a settings record')
    if record.get('kind') not in KINDS or set(record) - _KEYS:
        raise UnsupportedFeature('Supported adjustments are Exposure, Levels, Invert, Saturation and Vibrance')
    if any(not _number(record.get(k), -lim, lim) for k, lim in
           (('hue', 360), ('saturation', 100), ('lightness', 100))):
        raise ValueError('Invalid adjustment color settings')
    if not isinstance(record.get('colorize'), bool):
        raise ValueError('Invalid adjustment colorize setting')
    levels = record.get('levels')
    if (not isinstance(levels, dict) or set(levels) != {'channel', 'ranges'}
            or levels['channel'] not in CHANNELS or not isinstance(levels['ranges'], list)
            or len(levels['ranges']) != 4):
        raise ValueError('Invalid adjustment levels')
    for r in levels['ranges']:
        if (not isinstance(r, dict) or set(r) != {'black', 'white', 'gamma', 'outputBlack', 'outputWhite'}
                or not _number(r['gamma'], .1, 9.99)
                or any(not _number(r[k], 0, 255) for k in ('black', 'white', 'outputBlack', 'outputWhite'))
                or r['white'] < r['black'] + 1):
            raise ValueError('Invalid levels range')
    curves = record.get('curves')
    if (not isinstance(curves, dict) or set(curves) != {'channel', 'channels'}
            or curves['channel'] not in CHANNELS or not isinstance(curves['channels'], list)
            or len(curves['channels']) != 4):
        raise ValueError('Invalid adjustment curves')
    for points in curves['channels']:
        if (not isinstance(points, list) or not 2 <= len(points) <= 32
                or any(not isinstance(p, dict) or set(p) != {'x', 'y'}
                       or not _number(p['x'], 0, 255) or not _number(p['y'], 0, 255) for p in points)
                or points[0]['x'] != 0 or points[-1]['x'] != 255
                or any(a['x'] >= b['x'] for a, b in zip(points, points[1:]))):
            raise ValueError('Invalid adjustment curve points')
    exposure = record.get('exposureSettings')
    if exposure is not None and (not isinstance(exposure, dict)
            or set(exposure) != {'exposure', 'offset', 'gamma'}
            or not _number(exposure['exposure'], -20, 20)
            or not _number(exposure['offset'], -.5, .5)
            or not _number(exposure['gamma'], .01, 9.99)):
        raise ValueError('Invalid exposure settings')
    sat = record.get('saturationSettings')
    if sat is not None and (not isinstance(sat, dict) or set(sat) != {'saturation'}
            or not _number(sat['saturation'], -100, 100)):
        raise ValueError('Invalid saturation settings')
    vib = record.get('vibranceSettings')
    if vib is not None and (not isinstance(vib, dict) or set(vib) != {'vibrance'}
            or not _number(vib['vibrance'], -100, 100)):
        raise ValueError('Invalid vibrance settings')


def grade(rgb, record):
    """Grade straight encoded RGB floats, preserving alpha in the caller."""
    kind = record['kind']
    if kind == 'Invert':
        return 1 - rgb
    if kind == 'Saturation':
        s = record.get('saturationSettings', {}).get('saturation', 0)
        return _grade_saturation(rgb, s)
    if kind == 'Vibrance':
        v = record.get('vibranceSettings', {}).get('vibrance', 0)
        return _grade_vibrance(rgb, v)
    if kind == 'Exposure':
        s = record.get('exposureSettings') or dict(exposure=0, offset=0, gamma=1)
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055) ** 2.4)
        linear = np.maximum(0, linear * 2 ** s['exposure'] + s['offset']) ** (1 / s['gamma'])
        return np.clip(np.where(linear <= .0031308, 12.92 * linear,
                                1.055 * linear ** (1 / 2.4) - .055), 0, 1)
    def apply(values, r):
        values = np.clip((values * 255 - r['black']) / (r['white'] - r['black']), 0, 1)
        return (r['outputBlack'] + values ** (1 / r['gamma'])
                * (r['outputWhite'] - r['outputBlack'])) / 255
    ranges = record['levels']['ranges']
    # Native Levels applies each color channel, then the combined RGB channel.
    channels = np.stack([apply(rgb[..., i], ranges[i + 1]) for i in range(3)], axis=-1)
    return apply(channels, ranges[0])


def _rgb_to_hsl(rgb):
    """RGB float32 [0..1] → H (0..360), S, L each [0..1]. """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = np.maximum.reduce([r, g, b]), np.minimum.reduce([r, g, b])
    l = (mx + mn) / 2
    delta = mx - mn
    # Saturation
    s = np.where(delta == 0, 0, delta / np.where(l > 0.5, 2 - mx - mn, mx + mn))
    # Hue: 0=R, 120=G, 240=B  (standard HSL algorithm)
    # When two channels tie for max, HSL convention uses the first in RGB order:
    # yellow=R>G=B>0, magenta=R=B>G>0, cyan=G=B>R>0.
    h = np.zeros_like(l)
    safe = np.where(delta > 0, delta, 1)
    mask_g = (mx == g) & (delta > 0)  # green is second in RGB order
    h[mask_g] = ((b - r) / safe * 60 + 120)[mask_g]
    mask_b = (mx == b) & ~mask_g & (delta > 0)  # blue third; excludes R=B=G
    h[mask_b] = ((r - g) / safe * 60 + 240)[mask_b]
    return np.stack([h / 360, s, l], axis=-1)


def _hsl_to_rgb(hsl):
    """H [0..1], S [0..1], L [0..1] → RGB float32 [0..1]."""
    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = l - c / 2
    def channel(p):
        o = p + m
        return np.clip(o, 0, 1)
    h6 = h * 6
    r = np.where(h6 < 1, c, np.where(h6 < 2, x, 0))
    g = np.where(h6 < 1, x, np.where(h6 < 3, c, np.where(h6 < 4, 0, x)))
    b = np.where(h6 < 3, 0, np.where(h6 < 4, x, np.where(h6 < 5, c, 0)))
    return np.stack([channel(r), channel(g), channel(b)], axis=-1)


def _grade_saturation(rgb, amount):
    """Saturation: scales HSL saturation by (100 + amount) / 100.
    amount=0 → identity, amount=-100 → desaturate, amount=+100 → double.
    """
    if amount == 0:
        return rgb
    scale = (100 + amount) / 100.0
    hsl = _rgb_to_hsl(rgb)
    hsl[..., 1] = np.clip(hsl[..., 1] * scale, 0, 1)
    return _hsl_to_rgb(hsl)


def _grade_vibrance(rgb, amount):
    """Vibrance: logarithmic saturation boost that protects already-saturated pixels.
    amount=0 → identity; positive boosts; negative reduces.  Skin-tone protection
    comes from the log curve: saturated colours get less boost than neutral ones.
    """
    if amount == 0:
        return rgb
    hsl = _rgb_to_hsl(rgb)
    mx = np.maximum.reduce([rgb[..., 0], rgb[..., 1], rgb[..., 2]])
    mn = np.minimum.reduce([rgb[..., 0], rgb[..., 1], rgb[..., 2]])
    # 1 - (saturation at max channel) gives headroom; log distributes boost to low-sat.
    headroom = 1 - np.where(mx > 0, mx - mn, 0) / np.where(mx > 0, mx, 1)
    # Map [0..1] headroom → [0..1] boost: 1 at grey, 0 at fully saturated.
    boost = np.log1p(headroom * 9) / np.log1p(9) * (amount / 100.0)
    hsl[..., 1] = np.clip(hsl[..., 1] * (1 + boost), 0, 1)
    return _hsl_to_rgb(hsl)
