"""Translate editable curve tables and texture controls to native records."""
from __future__ import annotations
from copy import deepcopy
from typing import Any

import numpy as np

from .adjustments import UnsupportedFeature
from .adjustments import _curve_value, adjustment, validate_adjustment


def curve_tables(settings: dict) -> list:
    """Return detached tables so editing controls cannot mutate history."""
    channels = settings.get('curves', {}).get('channels')
    if channels is None:
        channels = adjustment('Curves')['curves']['channels']
    return [[[p['x'], p['y']] for p in points] for points in channels]


def curve_channels(tables: Any) -> list:
    """Validate tables before any document mutation, including blank cells."""
    try:
        channels = []
        if len(tables) != 4:
            raise ValueError('Expected four channels')
        for table in tables:
            values = np.asarray(table, dtype=np.float64)
            if values.ndim != 2 or values.shape[1] != 2:
                raise ValueError('Expected input/output pairs')
            channels.append([dict(x=float(x), y=float(y)) for x, y in values])
        record = adjustment('Curves')
        record['curves']['channels'] = channels
        validate_adjustment(record)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Each curve needs 2–32 input/output points from 0 to 255, '
                         'with increasing inputs starting at 0 and ending at 255.') from exc
    return channels


def gradient_color(value: Any) -> dict:
    """A ``#rrggbb`` picker value into a native AdjustmentColor record."""
    text = str(value or '').lstrip('#')
    if len(text) == 3:
        text = ''.join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError('Choose the gradient map colours as #rrggbb.')
    try:
        red, green, blue = (int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError as exc:
        raise ValueError('Choose the gradient map colours as #rrggbb.') from exc
    return dict(red=red, green=green, blue=blue)


def controls_adjustment(*args: Any, existing: dict | None = None) -> dict:
    """Build a complete record; retain imported Curves' ancillary settings."""
    record = adjustment(*args[:16])
    if len(args) <= 16:
        return record
    # Callers pass the curve tables grouped as args[16] (a list of 4) with the
    # texture controls flat after it, then Tonal Contrast, Bloom, B&W, Color
    # Balance and the two Gradient Map colour pickers — exactly the extras
    # order the UI registers for add_grade/update_grade.
    (tables, amount, size, roughness, seed, noise_amount, gaussian,
     monochromatic, *rest) = args[16:]
    tonal = rest[:5]
    bloom = rest[5:7]
    bw, cb, gradient = rest[7:16], rest[16:26], rest[26:]
    if len(rest) >= 26 and record['kind'] not in ('Black & White', 'Color Balance',
                                                  'Gradient Map', 'Tonal Contrast', 'Bloom') and (
            tuple(bw) != (40, 60, 40, 60, 20, 80, False, 40, 20)
            or tuple(cb) != (0, 0, 0, 0, 0, 0, 0, 0, 0, True)
            or tuple(tonal) != (50, 16, 40, 60, 30)
            or tuple(bloom) != (40, 24)):
        # Other kinds may only carry the controls' default values; a record
        # with moved B&W / Color-Balance sliders would silently drop them.
        raise UnsupportedFeature('Adjustment settings do not match its kind')
    if record['kind'] == 'Curves':
        if existing and existing['kind'] == 'Curves':
            record = deepcopy(existing)
        record['curves']['channels'] = curve_channels(tables)
    elif record['kind'] in ('Grain', 'Add Noise'):
        if (isinstance(seed, bool) or not isinstance(seed, (int, float))
                or not np.isfinite(seed) or seed != int(seed) or not 0 <= seed <= 0xFFFFFFFF):
            raise ValueError('Seed must be an integer from 0 to 4294967295.')
        if record['kind'] == 'Grain':
            record['grainSettings'] = dict(amount=amount, size=size, roughness=roughness, seed=int(seed))
        else:
            record.update(noiseAmount=noise_amount, noiseGaussian=gaussian,
                          noiseMonochromatic=monochromatic, noiseSeed=int(seed))
    elif record['kind'] == 'Tonal Contrast':
        record['tonalSettings'] = dict(amount=tonal[0], radius=tonal[1],
                                       shadows=tonal[2], midtones=tonal[3],
                                       highlights=tonal[4])
    elif record['kind'] == 'Bloom':
        record['bloomSettings'] = dict(amount=bloom[0], radius=bloom[1])
    elif record['kind'] == 'Black & White' and len(bw) >= 9:
        record['blackWhiteSettings'] = dict(
            reds=bw[0], yellows=bw[1], greens=bw[2], cyans=bw[3],
            blues=bw[4], magentas=bw[5], tint=bw[6], tintHue=bw[7],
            tintSaturation=bw[8])
    elif record['kind'] == 'Color Balance' and len(cb) >= 10:
        record['colorBalanceSettings'] = dict(
            shadowCyanRed=cb[0], shadowMagentaGreen=cb[1],
            shadowYellowBlue=cb[2], midCyanRed=cb[3],
            midMagentaGreen=cb[4], midYellowBlue=cb[5],
            highlightCyanRed=cb[6], highlightMagentaGreen=cb[7],
            highlightYellowBlue=cb[8], preserveLuminosity=cb[9])
    elif record['kind'] == 'Gradient Map' and len(gradient) >= 3:
        record['gradientMapSettings'] = dict(shadows=gradient_color(gradient[0]),
                                             highlights=gradient_color(gradient[1]),
                                             reversed=bool(gradient[2]))
    validate_adjustment(record)
    return record


def curve_graph(*tables: Any) -> str:
    """Graph the same interpolation as the renderer; never interpolate HTML input."""
    try:
        channels = curve_channels(tables)
    except ValueError as exc:
        return '<p>' + str(exc) + '</p>'
    paths = []
    for points, color in zip(channels, ('#555555', '#c93333', '#258340', '#3366dd')):
        path = ' '.join(f'{"M" if x == 0 else "L"}{x + 20},{275 - _curve_value(x, points):.2f}'
                        for x in range(256))
        paths.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
    return ('<svg viewBox="0 0 300 300" width="100%" style="max-width:340px" '
            'role="img" aria-label="RGB, red, green and blue curves; input left to right, output bottom to top">'
            '<rect x="20" y="20" width="255" height="255" fill="#fafafa" stroke="#aaa"/>'
            '<path d="M20,275 L275,20" stroke="#bbb" stroke-dasharray="4"/>'
            + ''.join(paths) + '</svg>')
