"""Supported Compositor adjustment records and alpha-preserving sRGB grading.

Derived from Compositor ImageAdjustments.swift and Levels.swift (MIT,
Copyright 2026 Wonder Assembly LLC), pinned at 11d8d7a; see third_party/compositor.
"""
from copy import deepcopy
import math

import numpy as np

from .document import UnsupportedFeature

KINDS = ('Exposure', 'Levels', 'Invert')
CHANNELS = ('RGB', 'Red', 'Green', 'Blue')
_KEYS = {'kind', 'hue', 'saturation', 'lightness', 'colorize', 'levels', 'curves',
         'exposureSettings'}


def adjustment(kind='Exposure', exposure=0, offset=0, gamma=1,
               black=0, white=255, red_gamma=1, green_gamma=1, blue_gamma=1):
    """Build a native-format record; all channels use identity by default."""
    if kind != 'Levels':
        black, white = 0, 255
        red_gamma = green_gamma = blue_gamma = 1
    ranges = [dict(black=black, white=white, gamma=gamma if kind == 'Levels' else 1,
                   outputBlack=0, outputWhite=255)]
    ranges += [dict(black=0, white=255, gamma=g, outputBlack=0, outputWhite=255)
               for g in (red_gamma, green_gamma, blue_gamma)]
    record = dict(kind=kind, hue=0, saturation=0, lightness=0, colorize=False,
                  levels=dict(channel='RGB', ranges=ranges),
                  curves=dict(channel='RGB', channels=[
                      [dict(x=0, y=0), dict(x=255, y=255)] for _ in range(4)]))
    if kind == 'Exposure':
        record['exposureSettings'] = dict(exposure=exposure, offset=offset, gamma=gamma)
    validate_adjustment(record)
    return deepcopy(record)


def _number(value, lo, hi):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and lo <= value <= hi)


def validate_adjustment(record):
    if not isinstance(record, dict):
        raise ValueError('Adjustment must be a settings record')
    if record.get('kind') not in KINDS or set(record) - _KEYS:
        raise UnsupportedFeature('Supported adjustments are Exposure, Levels and Invert')
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


def grade(rgb, record):
    """Grade straight encoded RGB floats, preserving alpha in the caller."""
    kind = record['kind']
    if kind == 'Invert':
        return 1 - rgb
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
