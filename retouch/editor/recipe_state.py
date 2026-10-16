"""Validated, versioned provenance for regenerating a recipe raster layer."""
import hashlib
import math

from retouch.params import PROCESSING_PARAMS


# UI labels are editor-specific; ranges and defaults belong to the engine registry.
EXTRA_RECIPE_CONTROLS = {
    'iris_pop': 'Iris pop',
    'catchlight': 'Eye catchlights',
    'blush': 'Cheek blush',
    'nose_tip_blush': 'Nose tip blush',
    'face_shadow_fill': 'Face shadow fill',
    'texture_opacity': 'Skin detail retention',
}
RECIPE_OVERRIDE_SPECS = {
    spec.name: spec for spec in PROCESSING_PARAMS
    if spec.name in {'smooth', 'whiten', *EXTRA_RECIPE_CONTROLS}
}


def validate_overrides(values):
    if not isinstance(values, dict) or set(values) - RECIPE_OVERRIDE_SPECS.keys():
        raise ValueError('Unsupported recipe override')
    for name, value in values.items():
        spec = RECIPE_OVERRIDE_SPECS[name]
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not math.isfinite(value) or not spec.min_val <= value <= spec.max_val):
            raise ValueError(f'Recipe {name} must be between {spec.min_val} and {spec.max_val}')


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
