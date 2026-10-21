"""Fuzz-round-2 regressions: malformed .comp fields must raise ProjectError.

Raw ValueError/TypeError escaping load_project (effects float coercion,
non-dict adjustment) meant mutated packages crashed the reader instead of
being refused; these pin the wrapped behaviour.
"""
import pytest

from retouch.editor.document import UnsupportedFeature
from retouch.editor.project_store import ProjectError, load_project
from tests import compositor_fixtures as fx

EFFECTS = {
    'stroke': {'size': 4, 'red': 0, 'green': 0, 'blue': 0, 'opacity': 1},
    'shadow': {'angle': 90, 'distance': 20, 'blur': 20,
               'red': 0, 'green': 0, 'blue': 0, 'opacity': 0.5},
    'colorOverlay': {'red': 0, 'green': 0, 'blue': 0, 'opacity': 1},
    'innerShadow': {'angle': 90, 'distance': 10, 'blur': 10,
                    'red': 0, 'green': 0, 'blue': 0, 'opacity': 0.5},
    'outerGlow': {'size': 20, 'red': 1, 'green': 1, 'blue': 1, 'opacity': 0.75},
    'innerGlow': {'size': 10, 'red': 1, 'green': 1, 'blue': 1, 'opacity': 0.75},
}


@pytest.mark.parametrize('field,value', [
    ('size', 'x'), ('size', None), ('size', [1]), ('red', {}), ('opacity', True),
])
@pytest.mark.parametrize('kind', list(EFFECTS))
def test_effect_numeric_field_rejected_as_project_error(tmp_path, kind, field,
                                                        value):
    effects = {kind: dict(EFFECTS[kind], **{field: value})}
    record = fx.layer_record(0, 'x', 2, 2, effects=effects)
    path = fx.write_package(tmp_path / 'm.comp', 2, 2, [record],
                            images={0: fx.solid(2, 2, (1, 2, 3, 255))})
    with pytest.raises(ProjectError):
        load_project(path)


@pytest.mark.parametrize('bad', ['bright', [], 5])
def test_non_dict_adjustment_rejected_as_project_error(tmp_path, bad):
    record = fx.layer_record(0, 'x', 2, 2, image=False, adjustment=bad)
    path = fx.write_package(tmp_path / 'm.comp', 2, 2, [record],
                            images={0: fx.solid(2, 2, (1, 2, 3, 255))})
    with pytest.raises(ProjectError):
        load_project(path)


def test_unsupported_adjustment_kind_still_unsupported_feature(tmp_path):
    # The ProjectError wrapper around validate_adjustment must let the
    # UnsupportedFeature subclass through untouched.
    record = fx.layer_record(0, 'x', 2, 2, image=False,
                             adjustment={'kind': 'NoSuchKind'})
    path = fx.write_package(tmp_path / 'm.comp', 2, 2, [record],
                            images={0: fx.solid(2, 2, (1, 2, 3, 255))})
    with pytest.raises(UnsupportedFeature):
        load_project(path)
