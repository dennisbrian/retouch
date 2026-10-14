"""Supported Compositor adjustment records and alpha-preserving sRGB grading.

Derived from Compositor ImageAdjustments.swift and Levels.swift (MIT,
Copyright 2026 Wonder Assembly LLC), pinned at 11d8d7a; see third_party/compositor.
"""
from copy import deepcopy
import math

import numpy as np

from .document import UnsupportedFeature

KINDS = ('Exposure', 'Levels', 'Curves', 'Invert', 'Saturation', 'Vibrance', 'Temperature', 'Tint', 'Brightness', 'Contrast', 'Hue')
CHANNELS = ('RGB', 'Red', 'Green', 'Blue')
_KEYS = {'kind', 'hue', 'saturation', 'lightness', 'colorize', 'levels', 'curves',
         'exposureSettings', 'saturationSettings', 'vibranceSettings', 'temperatureSettings', 'tintSettings', 'brightnessSettings', 'contrastSettings', 'hueSettings'}


def adjustment(kind='Exposure', exposure=0, offset=0, gamma=1,
               black=0, white=255, red_gamma=1, green_gamma=1, blue_gamma=1,
               saturation=0, vibrance=0, temperature=0, tint=0,
               brightness=0, contrast=0, hue=0):
    """Build a native-format record; all channels use identity by default."""
    if kind not in ('Levels', 'Saturation', 'Vibrance', 'Temperature', 'Tint', 'Brightness', 'Contrast', 'Hue'):
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
    if kind == 'Temperature':
        record['temperatureSettings'] = dict(temperature=temperature)
    if kind == 'Tint':
        record['tintSettings'] = dict(tint=tint)
    if kind == 'Brightness':
        record['brightnessSettings'] = dict(brightness=brightness)
    if kind == 'Contrast':
        record['contrastSettings'] = dict(contrast=contrast)
    if kind == 'Hue':
        record['hueSettings'] = dict(hue=hue)
    validate_adjustment(record)
    return deepcopy(record)


def _number(value, lo, hi):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and lo <= value <= hi)


def validate_adjustment(record):
    if not isinstance(record, dict):
        raise ValueError('Adjustment must be a settings record')
    if record.get('kind') not in KINDS or set(record) - _KEYS:
        raise UnsupportedFeature('Supported adjustments are Exposure, Levels, Invert, Saturation, Vibrance, Temperature, Tint, Brightness, Contrast and Hue')
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
    temp = record.get('temperatureSettings')
    if temp is not None and (not isinstance(temp, dict) or set(temp) != {'temperature'}
            or not _number(temp['temperature'], -100, 100)):
        raise ValueError('Invalid temperature settings')
    tnt = record.get('tintSettings')
    if tnt is not None and (not isinstance(tnt, dict) or set(tnt) != {'tint'}
            or not _number(tnt['tint'], -100, 100)):
        raise ValueError('Invalid tint settings')
    bright = record.get('brightnessSettings')
    if bright is not None and (not isinstance(bright, dict) or set(bright) != {'brightness'}
            or not _number(bright['brightness'], -100, 100)):
        raise ValueError('Invalid brightness settings')
    cont = record.get('contrastSettings')
    if cont is not None and (not isinstance(cont, dict) or set(cont) != {'contrast'}
            or not _number(cont['contrast'], -100, 100)):
        raise ValueError('Invalid contrast settings')
    hsett = record.get('hueSettings')
    if hsett is not None and (not isinstance(hsett, dict) or set(hsett) != {'hue'}
            or not _number(hsett['hue'], -180, 180)):
        raise ValueError('Invalid hue settings')


def _grade_hue(rgb, amount):
    """Hue: rotates the HSL hue wheel by `amount` degrees.
    amount=0 → identity; ±180 wraps correctly; hue is stored and handled in degrees.
    """
    if amount == 0:
        return rgb
    hsl = _rgb_to_hsl(rgb)
    hsl[..., 0] = (hsl[..., 0] + amount / 360.0) % 1.0
    return _hsl_to_rgb(hsl)


def _grade_brightness(rgb, amount):
    """Brightness: additive shift in sRGB space.
    amount=-100 → black, amount=+100 → white, amount=0 → identity.
    """
    if amount == 0:
        return rgb
    return np.clip(rgb + amount / 100.0, 0, 1)


def _grade_contrast(rgb, amount):
    """Contrast: midpoint-anchored S-curve (Photoshop/Lightroom style).
    amount=-100 → flat gray, amount=+100 → maximum contrast, amount=0 → identity.
    The midpoint 0.5 stays fixed; endpoints 0/1 also stay fixed.
    """
    if amount == 0:
        return rgb
    # Convert to [0..1] slope; identity (amount=0) → slope=1.
    # Using simple linear remapping: contrast=100 → slope=2 (steepest without clipping).
    slope = 1 + amount / 100.0
    midpoint = 0.5
    result = midpoint + (rgb - midpoint) * slope
    return np.clip(result, 0, 1)


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
    if kind == 'Temperature':
        t = record.get('temperatureSettings', {}).get('temperature', 0)
        return _grade_temperature(rgb, t)
    if kind == 'Tint':
        t = record.get('tintSettings', {}).get('tint', 0)
        return _grade_tint(rgb, t)
    if kind == 'Brightness':
        b = record.get('brightnessSettings', {}).get('brightness', 0)
        return _grade_brightness(rgb, b)
    if kind == 'Contrast':
        c = record.get('contrastSettings', {}).get('contrast', 0)
        return _grade_contrast(rgb, c)
    if kind == 'Hue':
        h = record.get('hueSettings', {}).get('hue', 0)
        return _grade_hue(rgb, h)
    if kind == 'Curves':
        return _grade_curves(rgb, record['curves'])
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


def _curve_value(x, points):
    """Monotone-cubic Hermite curve evaluation (Curves.swift:18-32)."""
    p = points
    i = min(len(p) - 2, max(0, max(j for j in range(len(p)) if p[j]['x'] <= x)))
    d = [(p[j + 1]['y'] - p[j]['y']) / (p[j + 1]['x'] - p[j]['x'])
         for j in range(len(p) - 1)]

    def slope(j):
        if j == 0:
            return d[0]
        if j == len(p) - 1:
            return d[-1]
        if d[j - 1] * d[j] <= 0:
            return 0
        return 2 / (1 / d[j - 1] + 1 / d[j])

    h = p[i + 1]['x'] - p[i]['x']
    t = (x - p[i]['x']) / h if h else 0
    t = min(1, max(0, t))
    y = ((2 * t ** 3 - 3 * t ** 2 + 1) * p[i]['y']
         + (t ** 3 - 2 * t ** 2 + t) * h * slope(i)
         + (-2 * t ** 3 + 3 * t ** 2) * p[i + 1]['y']
         + (t ** 3 - t ** 2) * h * slope(i + 1))
    return min(255, max(0, y))


def _identity_points(points):
    return (len(points) == 2 and points[0]['x'] == 0 and points[0]['y'] == 0
            and points[1]['x'] == 255 and points[1]['y'] == 255)


def _curves_lut(curves):
    """Composite 3x256 float64 tables: channel curve then RGB curve, baked
    at integers like Curves.apply (Curves.swift:37); /255 like levels_apply."""
    channels = curves['channels']
    if all(_identity_points(points) for points in channels):
        return None
    rgb = [_curve_value(i, channels[0]) for i in range(256)]
    luts = []
    for ch in range(1, 4):
        lut = np.empty(256, np.float64)
        for i in range(256):
            v = _curve_value(i, channels[ch])
            j = min(255, max(0, int(v // 1)))
            lut[i] = rgb[j] + (v - j) * (rgb[min(255, j + 1)] - rgb[j])
        luts.append(lut / 255)
    return luts


def _grade_curves(rgb, curves):
    """Apply baked LUTs with linear interpolation between entries, mirroring
    levels_apply (LevelsPixels.c:3-14) on the band's float pixels."""
    luts = _curves_lut(curves)
    if luts is None:
        return rgb
    out = np.empty_like(rgb)
    for i in range(3):
        t = luts[i]
        v = rgb[..., i] * 255
        j = np.clip(v, 0, 255).astype(np.int32)
        frac = v - j
        out[..., i] = t[j] + frac * (t[np.minimum(j + 1, 255)] - t[j])
    return out


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
    mask_r = (mx == r) & (delta > 0)  # red first: covers red and both red ties
    h[mask_r] = (((g - b) / safe * 60) % 360)[mask_r]
    mask_g = (mx == g) & ~mask_r  # green second in RGB order
    h[mask_g] = ((b - r) / safe * 60 + 120)[mask_g]
    mask_b = (mx == b) & ~mask_r & ~mask_g  # blue third; excludes R=B=G
    h[mask_b] = ((r - g) / safe * 60 + 240)[mask_b]
    return np.stack([h / 360, s, l], axis=-1)


def _hsl_to_rgb(hsl):
    """H [0..1], S [0..1], L [0..1] → RGB float32 [0..1].

    Standard piecewise HSL→RGB (W3C / CSS Algorithm 2).
    Six ordered hue segments with verified RGB channel assignments:
      [0,1): R=C, G=X, B=0  (red → yellow)
      [1,2): R=X, G=C, B=0  (yellow → green)
      [2,3): R=0, G=C, B=X  (green → cyan)
      [3,4): R=0, G=X, B=C  (cyan → blue)
      [4,5): R=X, G=0, B=C  (blue → magenta)
      [5,6): R=C, G=0, B=X  (magenta → red)
    Ties resolve to the first primary in RGB order:
      yellow=R>G=B, magenta=R=B>G, cyan=G=B>R.
    """
    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    c = (1 - np.abs(2 * l - 1)) * s
    x = c * (1 - np.abs((h * 6) % 2 - 1))
    m = l - c / 2
    h6 = (h * 6) % 6.0
    r = np.where(h6 < 1, c,
           np.where(h6 < 2, x,
           np.where(h6 < 4, 0.0,
           np.where(h6 < 5, x,
           np.where(h6 < 6, c, 0.0)))))
    g = np.where(h6 < 1, x,
           np.where(h6 < 3, c,
           np.where(h6 < 4, x, 0.0)))
    b = np.where(h6 < 2, 0.0,
           np.where(h6 < 3, x,
           np.where(h6 < 5, c,
           np.where(h6 < 6, x, 0.0))))
    return np.stack([np.clip(r + m, 0, 1),
                     np.clip(g + m, 0, 1),
                     np.clip(b + m, 0, 1)], axis=-1)


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


def _grade_temperature(rgb, amount):
    """Temperature: white balance toward amber (positive) or blue (negative).

    Photos are lit; a temperature slider re-balances the channels so a warm
    scene reads neutral and vice versa. Rec. 709 luminance is preserved, so
    brightness does not shift as the colour drifts.
    amount=-100 → cool, +100 → warm, 0 → identity.
    """
    if amount == 0:
        return rgb
    t = amount / 100.0
    gains = np.array([1 + 0.30 * t, 1.0, 1 - 0.30 * t], np.float32)
    return np.clip(rgb * _normalize_luminance(gains), 0, 1)


def _grade_tint(rgb, amount):
    """Tint: white balance along the green/magenta axis.

    Standard white-balance convention: positive pushes green, negative pushes
    magenta (Adobe Lightroom style). The adjustment is luminance-preserving
    via Rec. 709 weights so brightness stays constant.
    amount=-100 → magenta, +100 → green, 0 → identity.
    """
    if amount == 0:
        return rgb
    t = amount / 100.0
    # Green (G) is increased; R and B are reduced symmetrically to compensate.
    gains = np.array([1 - 0.15 * t, 1 + 0.30 * t, 1 - 0.15 * t], np.float32)
    return np.clip(rgb * _normalize_luminance(gains), 0, 1)


# Rec. 709 relative luminance weights: the eye is mostly green, least blue.
_LUMINANCE_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], np.float32)


def _normalize_luminance(gains):
    """Scale per-channel gains so a neutral grey keeps its brightness."""
    total = float(np.dot(gains, _LUMINANCE_WEIGHTS))
    return gains / total if total > 0 else np.ones(3, np.float32)
