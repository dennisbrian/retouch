"""Translate editable curve tables and texture controls to native records."""
from __future__ import annotations
from copy import deepcopy
from typing import Any

import numpy as np

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


def controls_adjustment(*args: Any, existing: dict | None = None) -> dict:
    """Build a complete record; retain imported Curves' ancillary settings."""
    record = adjustment(*args[:16])
    if len(args) <= 16:
        return record
    tables, amount, size, roughness, seed, noise_amount, gaussian, monochromatic = args[16:]
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
