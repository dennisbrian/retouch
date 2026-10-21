"""Scanlines adjustment kind: record, kernel, whole-plane band invariance."""
import numpy as np
import pytest
from pathlib import Path

from retouch.editor import (Document, DocumentHistory, Layer, UnsupportedFeature,
                            composite, load_project)
from retouch.editor.adjustments import adjustment, grade, validate_adjustment
from retouch.editor.selections import selection


def solid(color, width=4, height=4, alpha=255):
    a = np.empty((height, width, 4), np.uint8)
    a[..., :3], a[..., 3] = color, alpha
    return a

# --- Scanlines -----------------------------------------------------------------
#
# Port of upstream's Filter › Scanlines (Scanlines.swift scanlinesSettings +
# Rendering/DitherPixels.c scanlines_apply at 11d8d7a, v1.5.3). The painter
# walks each column with horizon occlusion, so the kind cannot band-split by
# rows: composite() admits it through the whole-plane path instead, and the
# band-invariance contract below proves that admission is exact.

DEFAULTS = dict(lineSpacing=4, thickness=70, glow=0, dots=0, wobble=0, displace=0,
                smoothness=50, threshold=0, split=0, density=0, contrast=0,
                blackLevel=0, colors='Black & White')


def scan_kwargs(**over):
    kwargs = dict(DEFAULTS)
    kwargs.update(over)
    return kwargs


def test_scanlines_record_round_trip():
    settings = adjustment('Scanlines', **scan_kwargs(lineSpacing=8, displace=-40,
                                                     colors='Original', split=3, glow=20))
    assert settings['kind'] == 'Scanlines'
    assert settings['scanlinesSettings'] == dict(DEFAULTS, lineSpacing=8, displace=-40,
                                                 colors='Original', split=3, glow=20)
    validate_adjustment(settings)
    # settings object belongs to its own kind only
    other = adjustment('Exposure')
    other['scanlinesSettings'] = settings['scanlinesSettings']
    with pytest.raises(UnsupportedFeature):
        validate_adjustment(other)


@pytest.mark.parametrize('kwargs', [
    dict(lineSpacing=1), dict(lineSpacing=33), dict(thickness=101), dict(glow=-1),
    dict(dots=101), dict(wobble=65), dict(displace=101), dict(displace=-101),
    dict(smoothness=101), dict(threshold=101), dict(split=17), dict(density=101),
    dict(contrast=-101), dict(blackLevel=101), dict(colors='Sepia'),
])
def test_scanlines_validation_rejects_out_of_range(kwargs):
    with pytest.raises(ValueError):
        adjustment('Scanlines', **scan_kwargs(**kwargs))


def test_scanlines_lines_of_light():
    gray = np.full((32, 32, 3), .5, np.float32)
    out = grade(gray, adjustment('Scanlines', **scan_kwargs(lineSpacing=8)))
    column = out[:, 8, 0]
    assert column.max() > .6, 'bright lines every spacing'
    assert column.min() < .1, 'dark gaps between them'
    peaks = [y for y in range(1, 31) if column[y] > column[y - 1] and column[y] >= column[y + 1]]
    assert peaks and all(b - a == 8 for a, b in zip(peaks, peaks[1:])), peaks
    # brighter picture -> brighter lines
    brighter = grade(np.full((32, 32, 3), .9, np.float32),
                     adjustment('Scanlines', **scan_kwargs(lineSpacing=8)))
    assert brighter[:, 8, 0].max() > column.max()


def test_scanlines_threshold_and_black_level():
    dark = grade(np.full((32, 32, 3), .3, np.float32),
                 adjustment('Scanlines', **scan_kwargs(lineSpacing=8, threshold=50)))
    assert dark[:, 8, 0].max() < .1, 'below threshold the screen stays dark'
    lifted = grade(np.zeros((32, 32, 3), np.float32),
                   adjustment('Scanlines', **scan_kwargs(lineSpacing=8, blackLevel=50)))
    assert lifted[:, 8, 0].max() > .3, 'black level lifts lines on black'


def test_scanlines_alpha_and_transparent_untouched():
    rng = np.random.default_rng(11)
    rgb = rng.uniform(0, 1, (24, 24, 3)).astype(np.float32)
    a = (rng.uniform(0, 1, (24, 24)) * 255).astype(np.uint8).astype(np.float32) / 255
    a[0] = 0  # a fully transparent row
    settings = adjustment('Scanlines', **scan_kwargs(lineSpacing=4, split=2, glow=30))
    from retouch.editor.adjustments import _grade_scanlines
    out = _grade_scanlines(rgb, a, settings['scanlinesSettings'])
    assert np.array_equal(out[a == 0], rgb[a == 0]), 'transparent pixels untouched'
    assert out.min() >= 0 and out.max() <= 1
    # grade()'s straight-alpha contract: rgb values may exceed their coverage
    # (they are straight), but never leave [0, 1].


def test_scanlines_premultiplied_semantics_match_byte_domain():
    # A half-covered pixel must land as the C kernel's premultiplied write,
    # unpremultiplied: byte round of (straight*1.35*shown*a) / a.
    rgb = np.full((8, 8, 3), .6, np.float32)
    a = np.full((8, 8), .5, np.float32)
    from retouch.editor.adjustments import _grade_scanlines
    settings = adjustment('Scanlines', **scan_kwargs(lineSpacing=4))
    out = _grade_scanlines(rgb, a, settings['scanlinesSettings'])
    a8 = 128
    straight = int(round(.6 * 255))  # 153
    pre = min(straight, a8)
    # column-agnostic check: the kernel's outputs are quantized to the byte
    # grid through the a8 premultiplied domain
    values = np.unique(np.rint(out * (a8 / 255.0) * 255))
    assert np.allclose(values % 1, 0), 'outputs live on the byte grid after premultiply'


def _scanlines_document():
    rng = np.random.default_rng(4)
    base = np.zeros((53, 16, 4), np.uint8)
    for c in range(3):
        base[..., c] = rng.integers(0, 256, (53, 16)).astype(np.uint8)
    base[..., 3] = (128 + rng.integers(0, 128, (53, 16))).astype(np.uint8)
    return base


@pytest.mark.parametrize('kwargs', [
    scan_kwargs(),
    scan_kwargs(lineSpacing=8, thickness=90),
    scan_kwargs(displace=40, smoothness=70),  # displacing: whole painter + lift
    scan_kwargs(displace=-40, smoothness=0),
    scan_kwargs(wobble=3, dots=60),
    scan_kwargs(split=3),
    scan_kwargs(glow=40),
    scan_kwargs(threshold=30, blackLevel=20, density=-30, contrast=40),
    scan_kwargs(colors='Original'),
])
def test_scanlines_band_invariance(kwargs):
    # Whole-plane admission: any band split must equal the single-band render
    # bit for bit, because grade() only ever sees the full canvas.
    base = _scanlines_document()
    doc = Document(16, 53, [Layer('Base', base),
                            Layer('Scanlines', size=(16, 53),
                                  adjustment=adjustment('Scanlines', **kwargs))])
    full = composite(doc)
    for band_rows in (256, 16, 7, 1):
        np.testing.assert_array_equal(full, composite(doc, band_rows=band_rows))


def test_scanlines_adjustment_layer_round_trips(tmp_path):
    base = Layer('Base', solid((80, 100, 120), alpha=128))
    settings = adjustment('Scanlines', **scan_kwargs(lineSpacing=6, split=2))
    adj = Layer('Scanlines', size=(4, 4), adjustment=settings,
                mask=selection(4, 4, 'Rectangle', 0, 0, 2, 4), opacity=.5)
    h = DocumentHistory(Document(4, 4, [base]))
    before = composite(h.document)
    h.add_layer(adj)
    image = composite(h.document)
    np.testing.assert_array_equal(image[..., 3], before[..., 3])  # alpha preserved
    np.testing.assert_array_equal(image[:, 2:], before[:, 2:])  # mask hides right half
    assert not np.array_equal(image[:, :2], before[:, :2])
    h.save(tmp_path / 'scanlines.comp')
    again = load_project(tmp_path / 'scanlines.comp')
    assert again.layers[-1].adjustment == settings
    np.testing.assert_array_equal(composite(again), image)
    h.undo()
    np.testing.assert_array_equal(composite(h.document), before)


def test_scanlines_whole_plane_over_limit_refused():
    # The whole-plane cache is capped like the effects plane (24 MP); a bigger
    # canvas with a scanlines layer refuses rather than blowing memory.
    from retouch.editor import composite as _
    big = Layer('Big', solid((10, 10, 10), width=5000, height=5000))
    doc = Document(5000, 5000, [big, Layer('S', size=(5000, 5000),
                                            adjustment=adjustment('Scanlines'))])
    with pytest.raises(ValueError, match='whole-plane'):
        composite(doc)


def test_scanlines_two_whole_plane_layers_refused():
    base = Layer('Base', solid((80, 100, 120)))
    one = Layer('One', size=(4, 4), adjustment=adjustment('Scanlines'))
    two = Layer('Two', size=(4, 4), adjustment=adjustment('Scanlines'))
    doc = Document(4, 4, [base, one, two])
    with pytest.raises(UnsupportedFeature, match='whole-plane'):
        composite(doc)


def test_scanlines_still_grades_inside_clipping_stack():
    # A scanlines adjustment inside a clipping stack stays per-band (its source
    # is the stack base's own pixels, mutated in place) and still renders.
    base = Layer('Base', solid((200, 200, 200)))
    stack = Layer('Stack', size=(4, 4), adjustment=adjustment('Scanlines'),
                  mask_source_id=base.id)
    doc = Document(4, 4, [base, stack])
    out = composite(doc)
    assert out[..., 3].min() == 255
    assert out.max() <= 255
    full = composite(doc, band_rows=256)
    np.testing.assert_array_equal(out, full)
