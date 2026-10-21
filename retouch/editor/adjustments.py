"""Supported Compositor adjustment records and alpha-preserving sRGB grading.

Derived from Compositor ImageAdjustments.swift and Levels.swift (MIT,
Copyright 2026 Wonder Assembly LLC), pinned at 11d8d7a; see third_party/compositor.
"""
from copy import deepcopy
import math

import numpy as np

from .document import UnsupportedFeature

KINDS = ('Exposure', 'Levels', 'Curves', 'Invert', 'Saturation', 'Vibrance', 'Temperature', 'Tint', 'Brightness', 'Contrast', 'Hue', 'Gradient Map', 'Grain', 'Add Noise', 'Black & White', 'Color Balance')
CHANNELS = ('RGB', 'Red', 'Green', 'Blue')
_KIND_RANGES = ('master', 'reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas')
_COMMON_KEYS = {'kind', 'hue', 'saturation', 'lightness', 'colorize', 'range', 'invertRange', 'levels', 'curves'}
_SETTINGS_KEYS = {'exposureSettings', 'saturationSettings', 'vibranceSettings', 'temperatureSettings',
                  'tintSettings', 'brightnessSettings', 'contrastSettings', 'hueSettings',
                  'gradientMapSettings',
                  'grainSettings', 'noiseAmount', 'noiseGaussian', 'noiseMonochromatic', 'noiseSeed',
                  'blackWhiteSettings', 'colorBalanceSettings'}
# A record carries the union of every kind's keys (upstream stores one struct
# with every settings object optional), but each settings object belongs to
# one kind: only its own kind's settings may be present.
_KEYS = _COMMON_KEYS | _SETTINGS_KEYS
_KEYS_BY_KIND = {
    'Exposure': {'exposureSettings'}, 'Levels': set(), 'Curves': set(), 'Invert': set(),
    'Saturation': {'saturationSettings'}, 'Vibrance': {'vibranceSettings'},
    'Temperature': {'temperatureSettings'}, 'Tint': {'tintSettings'},
    'Brightness': {'brightnessSettings'}, 'Contrast': {'contrastSettings'},
    'Hue': {'hueSettings'},
    'Gradient Map': {'gradientMapSettings'},
    'Grain': {'grainSettings'},
    'Add Noise': {'noiseAmount', 'noiseGaussian', 'noiseMonochromatic', 'noiseSeed'},
    'Black & White': {'blackWhiteSettings'},
    'Color Balance': {'colorBalanceSettings'},
}


def adjustment(kind='Exposure', exposure=0, offset=0, gamma=1,
               black=0, white=255, red_gamma=1, green_gamma=1, blue_gamma=1,
               saturation=0, vibrance=0, temperature=0, tint=0,
               brightness=0, contrast=0, hue=0,
               shadows=(0, 0, 0), highlights=(1, 1, 1), reversed=False,
               amount=25, size=1.5, roughness=50, seed=0,
               gaussian=False, monochromatic=False,
               reds=40, yellows=60, greens=40, cyans=60, blues=20, magentas=80,
               bw_tint=False, tint_hue=40, tint_saturation=20,
               shadow_cyan_red=0, shadow_magenta_green=0, shadow_yellow_blue=0,
               mid_cyan_red=0, mid_magenta_green=0, mid_yellow_blue=0,
               highlight_cyan_red=0, highlight_magenta_green=0, highlight_yellow_blue=0,
               preserve_luminosity=True):
    """Build a native-format record; all channels use identity by default."""
    if kind not in ('Levels', 'Saturation', 'Vibrance', 'Temperature', 'Tint', 'Brightness', 'Contrast', 'Hue'):
        black, white = 0, 255
        red_gamma = green_gamma = blue_gamma = 1
    ranges = [dict(black=black, white=white, gamma=gamma if kind == 'Levels' else 1,
                   outputBlack=0, outputWhite=255)]
    ranges += [dict(black=0, white=255, gamma=g, outputBlack=0, outputWhite=255)
               for g in (red_gamma, green_gamma, blue_gamma)]
    record = dict(kind=kind, hue=0, saturation=saturation, lightness=0, colorize=False,
                  range='master', invertRange=False,
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
    if kind == 'Gradient Map':
        try:
            shadow_color, highlight_color = tuple(shadows), tuple(highlights)
            if len(shadow_color) != 3 or len(highlight_color) != 3:
                raise IndexError
            record['gradientMapSettings'] = dict(
                shadows=dict(red=shadow_color[0], green=shadow_color[1], blue=shadow_color[2]),
                highlights=dict(red=highlight_color[0], green=highlight_color[1], blue=highlight_color[2]),
                reversed=reversed)
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError('Invalid gradient map settings') from exc
    if kind == 'Grain':
        record['grainSettings'] = dict(amount=amount, size=size, roughness=roughness, seed=seed)
    if kind == 'Add Noise':
        record.update(noiseAmount=amount, noiseGaussian=gaussian,
                      noiseMonochromatic=monochromatic, noiseSeed=seed)
    if kind == 'Black & White':
        record['blackWhiteSettings'] = dict(
            reds=reds, yellows=yellows, greens=greens, cyans=cyans, blues=blues,
            magentas=magentas, tint=bw_tint, tintHue=tint_hue,
            tintSaturation=tint_saturation)
    if kind == 'Color Balance':
        record['colorBalanceSettings'] = dict(
            shadowCyanRed=shadow_cyan_red, shadowMagentaGreen=shadow_magenta_green,
            shadowYellowBlue=shadow_yellow_blue, midCyanRed=mid_cyan_red,
            midMagentaGreen=mid_magenta_green, midYellowBlue=mid_yellow_blue,
            highlightCyanRed=highlight_cyan_red,
            highlightMagentaGreen=highlight_magenta_green,
            highlightYellowBlue=highlight_yellow_blue,
            preserveLuminosity=preserve_luminosity)
    validate_adjustment(record)
    return deepcopy(record)


def _number(value, lo, hi):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and lo <= value <= hi)


def validate_adjustment(record):
    if not isinstance(record, dict):
        raise ValueError('Adjustment must be a settings record')
    if record.get('kind') not in KINDS or set(record) - _KEYS:
        raise UnsupportedFeature('Supported adjustments are Exposure, Levels, Invert, '
                                 'Saturation, Vibrance, Temperature, Tint, Brightness, '
                                 'Contrast, Hue, Black & White and Color Balance')
    if any(not _number(record.get(k), -lim, lim) for k, lim in
           (('hue', 360), ('saturation', 100), ('lightness', 100))):
        raise ValueError('Invalid adjustment color settings')
    if not isinstance(record.get('colorize'), bool):
        raise ValueError('Invalid adjustment colorize setting')
    if record.get('range') not in _KIND_RANGES:
        raise ValueError('Invalid adjustment range setting')
    if not isinstance(record.get('invertRange'), bool):
        raise ValueError('Invalid adjustment invertRange setting')
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
    gmap = record.get('gradientMapSettings')
    if gmap is not None and (not isinstance(gmap, dict)
            or set(gmap) != {'shadows', 'highlights', 'reversed'}
            or not isinstance(gmap['reversed'], bool)
            or any(not isinstance(gmap[end], dict)
                   or set(gmap[end]) != {'red', 'green', 'blue'}
                   or any(not _number(gmap[end][c], 0, 1)
                          for c in ('red', 'green', 'blue'))
                   for end in ('shadows', 'highlights'))
            or not isinstance(gmap.get('shadows'), dict)):
        raise ValueError('Invalid gradient map settings')
    grain = record.get('grainSettings')
    if grain is not None and (not isinstance(grain, dict)
            or set(grain) != {'amount', 'size', 'roughness', 'seed'}
            or not _number(grain['amount'], 0, 100)
            or not _number(grain['size'], .5, 20)
            or not _number(grain['roughness'], 0, 100)
            or not isinstance(grain['seed'], int) or isinstance(grain['seed'], bool)
            or not 0 <= grain['seed'] <= 0xFFFFFFFF):
        raise ValueError('Invalid grain settings')
    if 'noiseAmount' in record and not _number(record.get('noiseAmount'), .1, 400):
        raise ValueError('Invalid noise settings')
    if 'noiseGaussian' in record and not isinstance(record.get('noiseGaussian'), bool):
        raise ValueError('Invalid noise settings')
    if 'noiseMonochromatic' in record and not isinstance(record.get('noiseMonochromatic'), bool):
        raise ValueError('Invalid noise settings')
    if 'noiseSeed' in record and (not isinstance(record.get('noiseSeed'), int)
            or isinstance(record['noiseSeed'], bool) or not 0 <= record['noiseSeed'] <= 0xFFFFFFFF):
        raise ValueError('Invalid noise settings')
    bw = record.get('blackWhiteSettings')
    if bw is not None and (
            not isinstance(bw, dict)
            or set(bw) != {'reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas',
                           'tint', 'tintHue', 'tintSaturation'}
            or any(not _number(bw[k], -200, 300) for k in
                   ('reds', 'yellows', 'greens', 'cyans', 'blues', 'magentas'))
            or not isinstance(bw['tint'], bool)
            or not _number(bw['tintHue'], 0, 360)
            or not _number(bw['tintSaturation'], 0, 100)):
        raise ValueError('Invalid black & white settings')
    cb = record.get('colorBalanceSettings')
    if cb is not None and (
            not isinstance(cb, dict)
            or set(cb) != {'shadowCyanRed', 'shadowMagentaGreen', 'shadowYellowBlue',
                           'midCyanRed', 'midMagentaGreen', 'midYellowBlue',
                           'highlightCyanRed', 'highlightMagentaGreen',
                           'highlightYellowBlue', 'preserveLuminosity'}
            or any(not _number(cb[k], -100, 100) for k in (
                'shadowCyanRed', 'shadowMagentaGreen', 'shadowYellowBlue',
                'midCyanRed', 'midMagentaGreen', 'midYellowBlue',
                'highlightCyanRed', 'highlightMagentaGreen', 'highlightYellowBlue'))
            or not isinstance(cb['preserveLuminosity'], bool)):
        raise ValueError('Invalid color balance settings')
    # Every settings object belongs to one kind; a record may only carry its own.
    if set(record) & _SETTINGS_KEYS - _KEYS_BY_KIND.get(record.get('kind'), set()):
        raise UnsupportedFeature('Adjustment settings do not match its kind')


# --- Hue/Saturation ------------------------------------------------------------
#
# Native HueSaturation.swift port: one (hue, saturation, lightness) triple over
# a hue band (trapezoid weight), or colorize; applied through a 33^3
# trilinear cube built once per record and cached, because grade() runs per
# composite band (composite.py) and a ~0.15 s rebuild per band would dominate.
_HUE_DIM = 33
_HUE_STEP = _HUE_DIM - 1
_HUE_CHUNK = 1_500_000  # pixels per apply slice, bounding trilerp scratch
_HUE_BANDS = {
    'master': (0, 0, 360, 360),
    'reds': (315, 345, 15, 45),
    'yellows': (15, 45, 75, 105),
    'greens': (75, 105, 135, 165),
    'cyans': (135, 165, 195, 225),
    'blues': (195, 225, 255, 285),
    'magentas': (255, 285, 315, 345),
}
# Cache key -> flat float32 cube, channel-major: ((channel*33 + blue)*33 + green)*33 + red.
_HUE_CUBE_CACHE = {}
_HUE_CUBE_CACHE_MAX = 16


def _fwd(a, b):
    return (b - a) % 360.0


def _band_weight(band, hue):
    """Trapezoid: linear ramp in, plateau, linear ramp out, per HueBand.swift."""
    fs, rs, re_, fe = band
    span = _fwd(fs, fe)
    if span <= 0:
        return 1.0
    pos = _fwd(fs, hue)
    if pos > span:
        return 0.0
    ramp_in = _fwd(fs, rs)
    plateau_end = _fwd(fs, re_)
    if pos < ramp_in:
        return pos / ramp_in if ramp_in > 0 else 1.0
    if pos <= plateau_end:
        return 1.0
    ramp_out = span - plateau_end
    return (span - pos) / ramp_out if ramp_out > 0 else 1.0


def _hue_response(hue, saturation, lightness, rng, invert):
    """Per-degree weighted (shift, sat, light) table, as hueResponse() upstream."""
    table = []
    for deg in range(361):
        w = 1.0 if rng == 'master' else _band_weight(_HUE_BANDS[rng], float(deg))
        if invert and rng != 'master':
            w = 1.0 - w
        table.append((hue * w, saturation * w, lightness * w))
    return table


def _to_hsl(r, g, b):
    high, low = max(r, g, b), min(r, g, b)
    lightness = (high + low) / 2
    delta = high - low
    if delta <= 0:
        return (0.0, 0.0, lightness)
    sat = delta / (1 - abs(2 * lightness - 1))
    if high == r:
        hue = (g - b) / delta
    elif high == g:
        hue = (b - r) / delta + 2
    else:
        hue = (r - g) / delta + 4
    hue *= 60
    if hue < 0:
        hue += 360
    return (hue, min(1.0, sat), lightness)


def _to_rgb(hue, sat, lightness):
    if sat <= 0:
        return (lightness, lightness, lightness)
    chroma = (1 - abs(2 * lightness - 1)) * sat
    sector = hue / 60
    second = chroma * (1 - abs(sector % 2 - 1))
    base = lightness - chroma / 2
    s = int(sector)
    rgb = ((chroma, second, 0), (second, chroma, 0), (0, chroma, second),
           (0, second, chroma), (second, 0, chroma), (chroma, 0, second))[min(s, 5)]
    return tuple(min(1, max(0, c + base)) for c in rgb)


def _adjusted_saturation(saturation, amount):
    amount = min(1, max(-1, amount / 100))
    if amount <= 0:
        return max(0.0, saturation * (1 + amount))
    if amount >= 1:
        return 1.0 if saturation > 0 else 0.0
    return min(1.0, saturation / (1 - amount))


def _adjust_pixel(r, g, b, table, colorize, hue, saturation, lightness):
    hue_, sat, lightness_ = _to_hsl(r, g, b)
    if colorize:
        hue_ = hue % 360.0
        sat = min(1, max(0, saturation / 100))
        light_amount = lightness / 100
    else:
        idx = min(360, max(0, int(math.floor(hue_ + 0.5))))  # Int(hue.rounded())
        shift, s_amt, l_amt = table[idx]
        light_amount = l_amt / 100
        hue_ = (hue_ + shift) % 360.0
        sat = _adjusted_saturation(sat, s_amt)
    amount = min(1, max(-1, light_amount))
    lightness_ = lightness_ + (1 - lightness_) * amount if amount >= 0 \
        else lightness_ * (1 + amount)
    return _to_rgb(hue_, sat, min(1, max(0, lightness_)))


def _build_hue_cube(key):
    """Channel-major flat cube: ((channel*33 + blue)*33 + green)*33 + red.
    Channel-major puts each channel's plane contiguously, so the per-channel
    gather loop in ``_apply_hue_cube`` streams one plane instead of striding
    by 4 (this layout alone cut 12 MP apply from ~2.1 s to ~0.7 s)."""
    hue, saturation, lightness, colorize, rng, invert = key
    table = None if colorize else _hue_response(hue, saturation, lightness, rng, invert)
    grid = np.empty((4, _HUE_DIM, _HUE_DIM, _HUE_DIM), np.float32)
    for blue in range(_HUE_DIM):
        for green in range(_HUE_DIM):
            for red in range(_HUE_DIM):
                c = _adjust_pixel(red / _HUE_STEP, green / _HUE_STEP, blue / _HUE_STEP,
                                  table, colorize, hue, saturation, lightness)
                grid[:3, blue, green, red] = c
                grid[3, blue, green, red] = 1.0
    return grid.reshape(-1)


# Neighbour offsets of a cube cell in the flat channel-major layout.
_HUE_OFFS = (0, 1, _HUE_DIM, _HUE_DIM + 1,
             _HUE_DIM ** 2, _HUE_DIM ** 2 + 1,
             _HUE_DIM ** 2 + _HUE_DIM, _HUE_DIM ** 2 + _HUE_DIM + 1)


def _hue_cube(key):
    cube = _HUE_CUBE_CACHE.get(key)
    if cube is None:
        if len(_HUE_CUBE_CACHE) >= _HUE_CUBE_CACHE_MAX:
            _HUE_CUBE_CACHE.pop(next(iter(_HUE_CUBE_CACHE)))
        cube = _HUE_CUBE_CACHE[key] = _build_hue_cube(key)
    return cube


def _apply_hue_cube(rgb, cube):
    """Trilinear cube lookup on float RGB (..., 3), chunked to bound scratch.

    Per channel: eight (N,) corner gathers weighted-accumulated — one output
    column at a time keeps every temporary (N,) sized, so the whole apply is
    a stream of 1-D passes.
    """
    flat = np.ascontiguousarray(rgb, dtype=np.float32).reshape(-1, 3)
    out = np.empty_like(flat)
    plane = _HUE_DIM ** 3
    for start in range(0, len(flat), _HUE_CHUNK):
        f = flat[start:start + _HUE_CHUNK]
        p = np.clip(f * np.float32(_HUE_STEP), 0.0, np.float32(_HUE_STEP))
        lo = p.astype(np.int32)
        np.clip(lo, 0, _HUE_DIM - 2, out=lo)
        frac = p - lo
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
        ux, uy, uz = 1.0 - fx, 1.0 - fy, 1.0 - fz
        weights = (ux * uy * uz, fx * uy * uz, ux * fy * uz, fx * fy * uz,
                   ux * uy * fz, fx * uy * fz, ux * fy * fz, fx * fy * fz)
        cell = (lo[:, 2].astype(np.int64) * _HUE_DIM + lo[:, 1]) * _HUE_DIM + lo[:, 0]
        sl = slice(start, start + len(f))
        for channel in range(3):
            base = cell + channel * plane
            acc = cube[base + _HUE_OFFS[0]] * weights[0]
            for offset, weight in zip(_HUE_OFFS[1:], weights[1:]):
                acc += cube[base + offset] * weight
            out[sl, channel] = acc
    return out.reshape(rgb.shape)


def _grade_hue_sat(rgb, hue, saturation, lightness, colorize=False,
                   rng='master', invert=False):
    """Hue/Saturation: hue shift in degrees, saturation and lightness in
    [-100, 100] over one hue band (or colorize). Identity when everything is
    zero/neutral. Applied through the 33^3 trilinear cube, as native does."""
    if not colorize and hue == 0 and saturation == 0 and lightness == 0:
        return rgb
    cube = _hue_cube((float(hue), float(saturation), float(lightness),
                      bool(colorize), rng, bool(invert)))
    return _apply_hue_cube(rgb, cube)


def _grade_hue(rgb, amount):
    """Back-compat single-argument hue shift (master band)."""
    return _grade_hue_sat(rgb, amount, 0, 0)


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


# --- Grain and Add Noise -----------------------------------------------------------
#
# Ports of Compositor's C kernels ``adjust_grain`` (Rendering/AdjustPixels.c)
# and ``noise_add_at`` (Rendering/NoisePixels.c), both derived from the same
# hash chain (``mix32``). Lattice placement stays in float64 (the C kernels
# compute their field coordinates in double), everything after in float32;
# every hash wraps modulo 2**32 exactly like C uint32 arithmetic.

_U32 = np.uint32


def _u32(values):
    """Truncate to uint32, as C unsigned arithmetic wraps (two's complement)."""
    return (np.asarray(values, np.int64) & np.int64(0xFFFFFFFF)).astype(_U32)


def _mix32(x):
    x = x.astype(_U32, copy=False)
    x = _u32(x ^ (x >> _U32(16)))
    x = _u32(x * _U32(0x7feb352d))
    x = _u32(x ^ (x >> _U32(15)))
    x = _u32(x * _U32(0x846ca68b))
    return _u32(x ^ (x >> _U32(16)))


def _grain_field(u, v, scale, seed):
    """Smooth value-noise lattice, C ``grain_field`` in double coords/float32 math."""
    fx = np.floor(u / scale)
    fy = np.floor(v / scale)
    ix, iy = fx.astype(np.int64), fy.astype(np.int64)
    tx = (u / scale - fx).astype(np.float32)
    ty = (v / scale - fy).astype(np.float32)
    tx = tx * tx * (np.float32(3) - np.float32(2) * tx)
    ty = ty * ty * (np.float32(3) - np.float32(2) * ty)

    def lattice(ix, iy):
        h = _mix32(_u32(ix.astype(_U32) * _U32(0x9E3779B1))
                   ^ _mix32(_u32(iy.astype(_U32) * _U32(0x85EBCA77)) ^ _U32(seed)))
        return h.astype(np.uint16).astype(np.float32) / np.float32(65535.0) \
            + (h >> _U32(16)).astype(np.float32) / np.float32(65535.0) - np.float32(1.0)

    n00, n10 = lattice(ix, iy), lattice(ix + 1, iy)
    n01, n11 = lattice(ix, iy + 1), lattice(ix + 1, iy + 1)
    top = n00 + (n10 - n00) * tx
    bottom = n01 + (n11 - n01) * tx
    return (top + (bottom - top) * ty) * np.float32(1.6)


def _grade_grain(rgb, a, settings, x0=0.0, y0=0.0, units_per_pixel=1.0):
    """Film grain: two lattice octaves weighted by tonal level, C ``adjust_grain``.

    ``rgb`` is straight [0..1] float32, ``a`` the matching coverage in [0..1]
    float32; the kernel runs in the C byte domain (where its rounding lives),
    through the same 255/alpha unpremultiply-repremultiply chain. ``x0``/``y0``
    place the band in document space so the pattern stays fixed however the
    canvas splits its drawing; grain only touches covered pixels.
    """
    amount, size, roughness, seed = (settings['amount'], settings['size'],
                                     settings['roughness'], settings['seed'])
    if amount <= 0 or units_per_pixel <= 0:
        return rgb
    height, width = rgb.shape[:2]
    strength = np.float32(1.0 if amount > 100 else amount / 100.0) \
        * np.float32(0.35) * np.float32(255.0)
    rough = np.float32(0.0 if roughness < 0 else 1.0 if roughness > 100
                      else roughness / 100.0)
    fine_seed = int(_mix32(np.array([_U32(seed) ^ _U32(0xA511E9B3)], _U32))[0])
    detail_size = max(0.5, size * 0.35)
    u = x0 + (np.arange(width, dtype=np.float64) + 0.5) * units_per_pixel
    v = y0 + (np.arange(height, dtype=np.float64) + 0.5) * units_per_pixel
    coords = u[None, :], v[:, None]
    smooth = _grain_field(*coords, size, seed)
    fine = _grain_field(*coords, detail_size, fine_seed)
    noise = smooth + (fine - smooth) * rough
    a8 = np.rint(a * np.float32(255.0)).astype(np.uint8)
    a32 = a8.astype(np.float32)
    unpre = np.where(a8 == 255, np.float32(1.0), np.float32(255.0) / a32)[..., None]
    s = np.rint(np.clip(rgb, 0, 1) * np.float32(255.0)).astype(np.uint8).astype(np.float32)
    r = s * unpre
    level = (np.float32(0.2126) * r[..., 0] + np.float32(0.7152) * r[..., 1]
             + np.float32(0.0722) * r[..., 2]) / np.float32(255.0)
    level = np.minimum(level, np.float32(1.0))
    delta = noise * strength * (np.float32(0.4)
                                + np.float32(2.4) * level * (np.float32(1.0) - level))
    val = np.clip(r + delta[..., None], 0, 255)
    coverage = a32[..., None] / np.float32(255.0)
    res = (val * coverage + np.float32(0.5)).astype(np.uint8)
    # The band's coverage rounds back to exactly the a it was given.
    mask = (a > 0)[..., None]
    return np.where(mask, res.astype(np.float32) / np.float32(255.0), rgb)


def _grade_noise(rgb, a, settings, x0=0, y0=0):
    """Add Noise: per-pixel hash noise, uniform or Box-Muller Gaussian, C ``noise_add_at``.

    Same byte-domain discipline as ``_grade_grain``: the C kernel rounds in the
    premultiplied domain, so the band's coverage goes through the identical
    255/alpha unpremultiply-repremultiply chain.
    """
    amount = np.float32(settings['noiseAmount'])
    spread = amount / np.float32(100.0) * np.float32(127.5)
    height, width = rgb.shape[:2]
    seed = _U32(settings['noiseSeed'])
    px = _u32(x0 + np.arange(width, dtype=np.int64))[None, :]
    py = _u32(y0 + np.arange(height, dtype=np.int64))[:, None]
    base = _mix32(seed ^ _mix32(_u32(px * _U32(0x9e3779b9))
                                ^ _mix32(_u32(py * _U32(0x85ebca6b)))))
    base = np.broadcast_to(base, (height, width))

    def unit(key):
        return (_mix32(key) >> _U32(8)).astype(np.float32) * np.float32(1.0 / 16777216.0)

    a8 = np.rint(a * np.float32(255.0)).astype(np.uint8)
    a32 = a8.astype(np.float32)
    straight = np.rint(np.clip(rgb, 0, 1) * np.float32(255.0)).astype(np.uint8).astype(np.float32)
    out = rgb.copy()
    for c in range(3):
        key = base if settings['noiseMonochromatic'] \
            else _u32(base + _u32(c * 0x9e3779b9))
        if settings['noiseGaussian']:
            u1, u2 = unit(key), unit(_u32(key ^ _U32(0x68e31da4)))
            n = np.sqrt(np.float32(-2.0) * np.log(np.float32(1.0) - u1)) \
                * np.cos(np.float32(6.2831853) * u2) * spread * np.float32(2.0 / 3.0)
        else:
            n = (unit(key) * np.float32(2.0) - np.float32(1.0)) * spread
        value = straight[..., c] * np.float32(255.0) / a32 + n
        value = np.clip(value, 0, 255)
        pre = value * a32 / np.float32(255.0)
        out[..., c] = np.where(a > 0, np.floor(pre + np.float32(0.5)) / np.float32(255.0),
                               rgb[..., c])
    return out


def _grade_black_white(rgb, a, settings):
    """Black & White, C ``adjust_black_white`` (AdjustPixels.c).

    Photoshop's split: a color is its gray floor plus the secondary range
    (cyan/magenta/yellow) between its two darkest channels and the primary
    (red/green/blue) of its brightest, each range weighted per record. The
    kernel runs in the byte domain through the unpremultiply-repremultiply
    chain, like the other C ports.
    """
    weights = np.array([settings.get('reds', 40), settings.get('yellows', 60),
                        settings.get('greens', 40), settings.get('cyans', 60),
                        settings.get('blues', 20), settings.get('magentas', 80)],
                       np.float32) / np.float32(100)
    tint = bool(settings.get('tint', False))
    tint_hue = float(settings.get('tintHue', 40))
    tint_saturation = float(settings.get('tintSaturation', 20)) / 100.0
    a8 = np.rint(a * np.float32(255.0)).astype(np.uint8)
    a32 = a8.astype(np.float32)
    unpre = np.where(a8 == 255, np.float32(1.0), np.float32(255.0) / a32)
    c = np.rint(np.clip(rgb, 0, 1) * np.float32(255.0)).astype(np.uint8) \
        .astype(np.float32) * (unpre / np.float32(255.0))[..., None]
    c = np.minimum(c, np.float32(1.0))
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    md = r + g + b - mx - mn
    # primary/secondary indices exactly as the C kernel picks them; ties
    # resolve to the earlier branch (mx == r checked first, g >= b etc.).
    primary = np.where(mx == r, 0, np.where(mx == g, 2, 4))
    secondary = np.where(
        mx == r, np.where(g >= b, 1, 5),
        np.where(mx == g, np.where(r >= b, 1, 3), np.where(g >= r, 3, 5)))
    gray = mn + (md - mn) * weights[secondary] + (mx - md) * weights[primary]
    gray = np.minimum(np.float32(1.0), np.maximum(np.float32(0.0), gray))
    if tint and tint_saturation > 0:
        # The gray becomes the lightness of a color at the chosen hue; the
        # kernel computes the tint in float64 from the float32 gray.
        chroma = (1.0 - np.abs(2.0 * gray.astype(np.float64) - 1.0)) * tint_saturation
        hp = np.mod(tint_hue, 360.0) / 60.0
        xx = chroma * (1.0 - np.abs(np.mod(hp, 2.0) - 1.0))
        z = np.zeros_like(chroma)
        r1 = np.where(hp < 1, chroma, np.where(hp < 2, xx,
              np.where(hp < 4, z, np.where(hp < 5, xx, chroma))))
        g1 = np.where(hp < 1, xx, np.where(hp < 3, chroma,
              np.where(hp < 4, xx, z)))
        b1 = np.where(hp < 2, z, np.where(hp < 3, xx,
              np.where(hp < 5, chroma, xx)))
        m = gray.astype(np.float64) - chroma / 2.0
        out = np.stack([r1 + m, g1 + m, b1 + m], axis=-1)
        out = np.clip(out, 0.0, 1.0).astype(np.float32)
    else:
        out = np.repeat(gray[..., None], 3, axis=-1)
    val = np.clip(out * a32[..., None], 0, a32[..., None])
    res = np.floor(val + np.float32(0.5))  # C roundf: half away from zero
    mask = (a > 0)[..., None]
    return np.where(mask, res / np.float32(255.0), rgb)


def _tonal_weights(v):
    """C ``tonal_weights``: shadow/mid/highlight band weights over one channel."""
    a = np.float32(0.25)
    b = np.float32(0.333)
    scale = np.float32(0.7)
    s = np.clip((v - b) / -a + np.float32(0.5), 0, 1)
    h = np.clip((v + b - np.float32(1.0)) / a + np.float32(0.5), 0, 1)
    m1 = np.clip((v - b) / a + np.float32(0.5), 0, 1)
    m2 = np.clip((v + b - np.float32(1.0)) / -a + np.float32(0.5), 0, 1)
    return s * scale, m1 * m2 * scale, h * scale


def _grade_color_balance(rgb, a, settings):
    """Color Balance, C ``adjust_color_balance`` (AdjustPixels.c).

    Per-channel shadow/mid/highlight offsets (cyan/red, magenta/green,
    yellow/blue) weighted by tonal band membership, optionally restoring each
    pixel's 601 luminance afterwards. Byte-domain kernel like the others.
    """
    shadows = np.array([settings.get('shadowCyanRed', 0),
                        settings.get('shadowMagentaGreen', 0),
                        settings.get('shadowYellowBlue', 0)], np.float32) / np.float32(100)
    midtones = np.array([settings.get('midCyanRed', 0),
                         settings.get('midMagentaGreen', 0),
                         settings.get('midYellowBlue', 0)], np.float32) / np.float32(100)
    highlights = np.array([settings.get('highlightCyanRed', 0),
                           settings.get('highlightMagentaGreen', 0),
                           settings.get('highlightYellowBlue', 0)], np.float32) / np.float32(100)
    keep = bool(settings.get('preserveLuminosity', True))
    a8 = np.rint(a * np.float32(255.0)).astype(np.uint8)
    a32 = a8.astype(np.float32)
    unpre = np.where(a8 == 255, np.float32(1.0), np.float32(255.0) / a32)
    c = np.rint(np.clip(rgb, 0, 1) * np.float32(255.0)).astype(np.uint8) \
        .astype(np.float32) * (unpre / np.float32(255.0))[..., None]
    c = np.minimum(c, np.float32(1.0))
    if keep:
        before = (np.float32(0.299) * c[..., 0] + np.float32(0.587) * c[..., 1]
                  + np.float32(0.114) * c[..., 2])
    for i in range(3):
        s, m, h = _tonal_weights(c[..., i])
        c[..., i] = np.clip(c[..., i] + shadows[i] * s + midtones[i] * m
                            + highlights[i] * h, 0, 1)
    if keep:
        after = (np.float32(0.299) * c[..., 0] + np.float32(0.587) * c[..., 1]
                 + np.float32(0.114) * c[..., 2])
        ratio = np.where(after > np.float32(0.0001), before / np.where(
            after > np.float32(0.0001), after, np.float32(1.0)), np.float32(1.0))
        c = np.clip(c * ratio[..., None], 0, 1)
    val = np.clip(c * a32[..., None], 0, a32[..., None])
    res = np.floor(val + np.float32(0.5))  # C roundf: half away from zero
    mask = (a > 0)[..., None]
    return np.where(mask, res / np.float32(255.0), rgb)


# --- Gradient Map ------------------------------------------------------------
#
# Native GradientMapSettings (ImageAdjustments.swift) + C kernel
# ``adjust_gradient_map`` (Rendering/AdjustPixels.c): a 256-entry RGB LUT
# lerped dark→light in straight sRGB (no smoothstep, no linearisation),
# indexed by the Rec. 709 luminance of each pixel's bytes.

def _grade_gradient_map(rgb, settings):
    """Gradient Map: luminance picks a colour between the dark and light ends.

    Byte-domain port of the C kernel: input floats are rounded to bytes
    exactly as the kernel reads them, the LUT is the Swift ``channel()``
    lerp with round-to-nearest, and the index is the kernel's integer
    luminance ``(2126r + 7152g + 722b + 5000) / 10000`` clamped to 255.
    Alpha stays with the caller, as everywhere else in grade().
    """
    ends = ((settings['highlights'], settings['shadows'])
            if settings.get('reversed') else (settings['shadows'], settings['highlights']))
    dark, light = ends
    t = np.arange(256, dtype=np.float64) / 255.0
    channels = []
    for c in ('red', 'green', 'blue'):
        value = dark[c] + (light[c] - dark[c]) * t
        channels.append(np.clip(np.rint(value * 255.0), 0, 255).astype(np.uint8))
    lut = np.stack(channels, axis=-1)
    s = np.rint(np.clip(rgb, 0, 1) * 255.0).astype(np.uint32)
    level = (2126 * s[..., 0] + 7152 * s[..., 1] + 722 * s[..., 2] + 5000) // 10000
    np.minimum(level, 255, out=level)
    return lut[level].astype(np.float32) / np.float32(255.0)


def grade(rgb, record, x0=0.0, y0=0.0, units_per_pixel=1.0, alpha=None):
    """Grade straight encoded RGB floats, preserving alpha in the caller.

    ``x0``/``y0``/``units_per_pixel`` place the band in document space; only
    the field adjustments (Grain, Add Noise) read them, so the pattern stays
    fixed however the canvas splits its drawing. ``alpha`` is the band's
    coverage for the byte-domain kernels (Black & White, Color Balance);
    opaque when omitted, as their callers before this parameter passed.
    """
    kind = record['kind']
    a = np.ones(rgb.shape[:-1], np.float32) if alpha is None else alpha
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
        s = record.get('hueSettings') or {}
        # Colorize takes its hue from the top-level `hue` field (0..360, as
        # native does); the band shift lives in hueSettings (-180..180).
        hue = record.get('hue', 0) if record.get('colorize') else s.get('hue', 0)
        return _grade_hue_sat(rgb, hue, record.get('saturation', 0),
                              record.get('lightness', 0),
                              colorize=record.get('colorize', False),
                              rng=record.get('range', 'master'),
                              invert=record.get('invertRange', False))
    if kind == 'Gradient Map':
        settings = record.get('gradientMapSettings') or dict(
            shadows=dict(red=0, green=0, blue=0),
            highlights=dict(red=1, green=1, blue=1), reversed=False)
        return _grade_gradient_map(rgb, settings)
    if kind == 'Curves':
        return _grade_curves(rgb, record['curves'])
    if kind == 'Grain':
        settings = record.get('grainSettings') or dict(amount=25, size=1.5, roughness=50, seed=0)
        return _grade_grain(rgb, np.ones(rgb.shape[:2], np.float32), settings,
                            x0, y0, units_per_pixel)
    if kind == 'Add Noise':
        settings = dict(noiseAmount=record.get('noiseAmount', 10),
                        noiseGaussian=record.get('noiseGaussian', False),
                        noiseMonochromatic=record.get('noiseMonochromatic', False),
                        noiseSeed=record.get('noiseSeed', 0))
        return _grade_noise(rgb, np.ones(rgb.shape[:2], np.float32), settings, int(x0), int(y0))
    if kind == 'Black & White':
        settings = record.get('blackWhiteSettings') or {}
        return _grade_black_white(rgb, a, settings)
    if kind == 'Color Balance':
        settings = record.get('colorBalanceSettings') or {}
        if all(settings.get(k, 0) == 0 for k in
               ('shadowCyanRed', 'shadowMagentaGreen', 'shadowYellowBlue',
                'midCyanRed', 'midMagentaGreen', 'midYellowBlue',
                'highlightCyanRed', 'highlightMagentaGreen', 'highlightYellowBlue')):
            return rgb
        return _grade_color_balance(rgb, a, settings)
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
