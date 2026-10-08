"""Mask brush (retouch/editor/brush.py).

Several cases translate Compositor's ``CompositorTests/BrushTests.swift``
(MIT, Copyright (c) 2026 Wonder Assembly LLC, pinned at 11d8d7a) to the
mask-only brush: stroke-wide opacity cap, soft-stroke build-up, no ripple
from dab spacing, and opacity on mask painting.
"""
import numpy as np
import pytest

from retouch.editor import brush


def _white(width, height):
    return np.full((height, width), 255, np.uint8)


def _full(points, diameter, hardness, width, height):
    """Stroke coverage on the whole grid."""
    coverage, (x0, y0) = brush.stroke_coverage(points, diameter, hardness, width, height)
    full = np.zeros((height, width), np.float32)
    full[y0:y0 + coverage.shape[0], x0:x0 + coverage.shape[1]] = coverage
    return full


def test_opacity_caps_the_whole_stroke_even_where_it_overlaps_itself():
    # BrushTests.opacityCapsTheWholeStrokeEvenWhereItOverlapsItself
    mask = _white(200, 80)
    points = [(20, 40), (180, 40), (20, 40), (180, 40), (20, 40), (100, 40)]
    result = brush.paint_mask(mask, points, diameter=40, hardness=1.0, opacity=0.5, reveal=False)
    assert abs(int(result[40, 100]) - 128) <= 1
    assert result[0, 100] == 255          # outside the brush: untouched


def test_soft_stroke_builds_coverage_while_keeping_a_feathered_rim():
    # BrushTests.softStrokeBuildsCoverageWhileKeepingItsFeatheredRim
    lone = _full([(100, 40)], 40, 0.0, 200, 80)
    stroke = _full([(20, 40), (180, 40)], 40, 0.0, 200, 80)
    assert stroke[50, 100] * 255 > lone[50, 100] * 255 + 60
    assert stroke.max() <= 1.0
    # The rim stays soft: coverage keeps falling towards the edge.
    assert stroke[40, 100] > stroke[54, 100] > stroke[59, 100] >= 0


@pytest.mark.parametrize('hardness', [0.0, 0.5, 1.0])
def test_spaced_dabs_leave_no_visible_ripple_along_the_stroke(hardness):
    # BrushTests.spacedDabsLeaveNoVisibleRippleAlongTheStroke
    full = _full([(100, 150), (800, 150)], 120, hardness, 900, 300)
    for offset in (0, 30, 50):
        run = np.rint(full[150 + offset, 300:601] * 255)
        assert run.max() - run.min() <= 16, (hardness, offset)


def test_opacity_applies_to_mask_painting():
    # BrushTests.opacityAppliesToMaskPainting: a reveal-all mask, painted
    # black at 50% with a 20 px brush.
    mask = _white(80, 80)
    result = brush.paint_mask(mask, [(40, 40), (42, 40)], diameter=20, opacity=0.5, reveal=False)
    assert abs(int(result[40, 40]) - 128) <= 1
    assert result[5, 5] == 255


def test_reveal_paints_white_and_hard_edges_are_antialiased():
    mask = np.zeros((60, 60), np.uint8)
    result = brush.paint_mask(mask, [(30.0, 30.0)], diameter=20, hardness=1.0, reveal=True)
    assert result[30, 30] == 255
    assert result[30, 50] == 0
    edge = result[30, 19:22]
    assert ((edge > 0) & (edge < 255)).any()   # a partial pixel on the rim


def test_input_mask_is_never_modified_and_a_miss_returns_it():
    mask = _white(50, 50)
    mask.flags.writeable = False
    result = brush.paint_mask(mask, [(10, 10)], diameter=8, reveal=False)
    assert result is not mask and mask.min() == 255 and result[10, 10] == 0
    assert brush.paint_mask(mask, [(500, 500), (600, 600)], diameter=8, reveal=False) is mask


def test_only_the_stroke_bounds_are_touched():
    mask = _white(400, 400)
    result = brush.paint_mask(mask, [(100, 100), (120, 100)], diameter=10, reveal=False)
    changed = np.argwhere(result != mask)
    assert changed[:, 0].min() >= 94 and changed[:, 0].max() <= 106
    assert changed[:, 1].min() >= 94 and changed[:, 1].max() <= 126


def test_dabs_follow_upstream_spacing():
    centres = brush.dab_centres([(0, 0), (100, 0)], 100, 1.0)
    # 1.5 px apart for a 100 px hard brush, first dab under the first point.
    assert centres[0] == (0.0, 0.0)
    assert centres[1][0] == pytest.approx(1.5)
    assert len(centres) == 67
    assert brush.spacing(4, 0.0) == 0.25


@pytest.mark.parametrize('kwargs', [
    {'diameter': 0}, {'diameter': 5000}, {'diameter': float('nan')},
    {'diameter': 10, 'hardness': 1.5}, {'diameter': 10, 'opacity': 0},
    {'diameter': 10, 'opacity': 2},
])
def test_invalid_settings_are_refused(kwargs):
    with pytest.raises(ValueError):
        brush.paint_mask(_white(20, 20), [(5, 5)], **kwargs)


def test_pathological_strokes_are_refused():
    with pytest.raises(ValueError):
        brush.dab_centres([(0, 0), (1_000_000, 0)], 1, 1.0)
