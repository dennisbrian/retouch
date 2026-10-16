"""Validated, versioned provenance for regenerating a recipe raster layer."""
import hashlib
import math


def validate_overrides(values):
    if not isinstance(values, dict) or set(values) - {'smooth', 'whiten'}:
        raise ValueError('Recipe overrides support smoothing and whitening only')
    for value in values.values():
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not 0 <= value <= 100):
            raise ValueError('Recipe skin settings must be between 0 and 100')


def validate_state(state):
    if (not isinstance(state, dict)
            or set(state) != {'version', 'recipe', 'overrides', 'source_revision'}
            or type(state['version']) is not int or state['version'] != 1
            or not isinstance(state['recipe'], str) or not 1 <= len(state['recipe']) <= 200
            or type(state['source_revision']) is not int or state['source_revision'] < 0):
        raise ValueError('Invalid saved recipe settings')
    validate_overrides(state['overrides'])


def pixel_digest(pixels):
    return hashlib.sha256(pixels.tobytes()).hexdigest()
