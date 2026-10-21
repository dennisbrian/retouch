"""Colour-range selection: color_range_selection contracts + UI wiring.

Parity target: WandPixels.c color_range_mask — per-channel RGB box
distance (not HSV, not a graded mask), include union minus exclude,
fully transparent pixels never match, fuzziness slider 0–200.
"""
import numpy as np
import pytest
import gradio as gr

from retouch.editor import Document
from retouch.editor.selections import color_range_selection, combine
from retouch.editor.ui import EditorSession, build_editor_tab


@pytest.fixture
def callbacks():
    with gr.Blocks() as app:
        with gr.Tabs():
            build_editor_tab()
    return [value.fn for value in app.fns.values() if value.fn]


def edits(callbacks):
    return {fn.__kwdefaults__['operation']: fn for fn in callbacks if fn.__name__ == 'edit'}


def swatch_image(h=10, w=40):
    """Left third red, middle third grey, right third blue, solid alpha."""
    img = np.zeros((h, w, 4), np.uint8)
    img[..., 3] = 255
    img[:, :w // 3, 0], img[:, :w // 3, 1], img[:, :w // 3, 2] = 200, 30, 30
    img[:, w // 3:2 * w // 3, :3] = 128
    img[:, 2 * w // 3:, 0], img[:, 2 * w // 3:, 1], img[:, 2 * w // 3:, 2] = 30, 30, 200
    return img


# 1. Exact-match distance semantics: fuzziness 0 selects only exact RGB.
def test_zero_fuzziness_exact_match():
    m, c = color_range_selection(swatch_image(), '[[200, 30, 30]]', '[]', 0)
    assert c == 10 * (40 // 3)
    assert (m[:, :40 // 3] == 255).all() and (m[:, 40 // 3:] == 0).all()


# 2. Per-channel box distance, not vector norm: (200,30,30) vs (200,80,30)
#    is inside fuzziness 50 channelwise but its vector distance is 50 (equal);
#    (220,80,50) has channel diffs 20/50/20 so it matches at 50 but a
#    Euclidean distance of ~57 would reject it.
def test_per_channel_box_distance_not_norm():
    img = np.zeros((1, 3, 4), np.uint8)
    img[..., 3] = 255
    img[0, 0, :3] = (200, 30, 30)
    img[0, 1, :3] = (220, 80, 50)   # diffs 20/50/20 -> box-match at 50
    img[0, 2, :3] = (251, 80, 50)   # red diff 51 -> no match at 50
    m, c = color_range_selection(img, '[[200, 30, 30]]', '[]', 50)
    assert c == 2
    assert m[0, 0] == m[0, 1] == 255 and m[0, 2] == 0


# 3. RGB not HSV: hue-equal but dark pixels do not match (an HSV-hue
#    implementation would select them).
def test_rgb_not_hue():
    img = np.zeros((1, 2, 4), np.uint8)
    img[..., 3] = 255
    img[0, 0, :3] = (255, 0, 0)     # bright red
    img[0, 1, :3] = (60, 0, 0)      # same hue, much darker
    m, c = color_range_selection(img, '[[255, 0, 0]]', '[]', 40)
    assert c == 1 and m[0, 0] == 255 and m[0, 1] == 0


# 4. Include union: two include colours select both halves.
def test_include_union():
    img = swatch_image()
    m, c = color_range_selection(img, '[[200, 30, 30], [30, 30, 200]]', '[]', 0)
    assert c == 10 * (40 // 3) + 10 * (40 - 2 * (40 // 3))
    assert (m[:, :40 // 3] == 255).all() and (m[:, 2 * (40 // 3):] == 255).all()
    assert (m[:, 40 // 3:2 * (40 // 3)] == 0).all()


# 5. Exclude subtraction: excluded colour wins over an include match.
def test_exclude_subtraction():
    img = swatch_image()
    # Include everything in range of grey, exclude the grey itself.
    m, c = color_range_selection(img, '[[128, 128, 128]]', '[[128, 128, 128]]', 0)
    assert c == 0 and not m.any()
    # Exclude applied per-channel: excluding (128,128,128) at fuzziness 0
    # leaves a wide include still able to select red and blue halves.
    m, c = color_range_selection(img, '[[128, 128, 128], [200, 30, 30]]',
                                 '[[128, 128, 128]]', 0)
    assert c == 10 * (40 // 3)
    assert (m[:, :40 // 3] == 255).all() and (m[:, 40 // 3:] == 0).all()


# 6. Fuzziness boundaries: 0 exact, max reachable range selects everything
#    with a 255-distance include colour (native slider caps at 200; the
#    kernel itself clamps 0..255).
def test_fuzziness_boundaries():
    img = swatch_image()
    _, c0 = color_range_selection(img, '[[201, 30, 30]]', '[]', 0)
    assert c0 == 0                                  # one-off colour: exact is empty
    _, c1 = color_range_selection(img, '[[201, 30, 30]]', '[]', 1)
    assert c1 == 10 * (40 // 3)                     # one-off colour: tolerance 1 covers it
    # fuzziness 200 from mid-grey reaches every swatch channelwise.
    _, call = color_range_selection(img, '[[128, 128, 128]]', '[]', 200)
    assert call == 10 * 40                          # native maximum covers the image


# 7. Binary mask, never graded: every value is exactly 0 or 255.
def test_binary_mask_values():
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (16, 24, 4), np.uint8)
    m, _ = color_range_selection(img, '[[128, 128, 128]]', '[[10, 10, 10]]', 90)
    assert set(np.unique(m)) <= {0, 255}


# 8. Alpha-zero pixels never match, even on colour.
def test_transparent_never_matches():
    img = np.zeros((2, 2, 4), np.uint8)
    img[..., :3] = (200, 30, 30)
    img[0, 0, 3] = 255                              # only one opaque pixel
    m, c = color_range_selection(img, '[[200, 30, 30]]', '[]', 0)
    assert c == 1 and m[0, 0] == 255 and m.sum() == 255


# 9a. Empty include returns an empty mask (native hasColors guard), no raise.
def test_empty_include_empty_mask():
    m, c = color_range_selection(swatch_image(), '[]', '[]', 40)
    assert c == 0 and not m.any() and m.shape == (10, 40)


# 9. Validation: non-triplet junk and NaN fuzziness raise.
@pytest.mark.parametrize('include, exclude, fuzz, message', [
    ('[[1, 2]]', '[]', 40, 'Include'),
    ('[[255, 0, 0]]', 'banana', 40, 'Exclude'),
    ('[[255, 0, 0]]', '[]', float('nan'), 'Fuzziness'),
])
def test_validation_raises(include, exclude, fuzz, message):
    with pytest.raises(ValueError, match=message):
        color_range_selection(swatch_image(), include, exclude, fuzz)


# 10. Combine modes on a colour-range result.
def test_combine_modes():
    new, _ = color_range_selection(swatch_image(), '[[200, 30, 30]]', '[]', 0)
    old = np.zeros_like(new)
    old[:5, :5] = 255
    assert (combine(old, new, 'Add') == np.maximum(old, new)).all()
    assert (combine(old, new, 'Subtract') == np.minimum(old, 255 - new)).all()
    assert (combine(old, new, 'Intersect') == np.minimum(old, new)).all()
    assert (combine(old, new, 'Replace') == new).all()


# 11. UI wiring: the color_range op composites the document and sets the
#     selection; an empty Replace deselects like the wand branch.
def test_ui_color_range_sets_selection(callbacks):
    fn = edits(callbacks)
    assert 'color_range' in fn
    pixels = np.zeros((30, 40, 3), np.uint8)
    pixels[..., :3] = (200, 30, 30)
    pixels[:, 20:] = (30, 30, 200)
    doc = Document(40, 30)
    doc.add_image(pixels, 'Base')
    session = EditorSession.create(doc)
    try:
        fn['color_range'](session, session.selected_id, None,
                          '[[200, 30, 30]]', '[]', 10, 'Replace')
        assert session.selection is not None
        assert (session.selection[:, :20] == 255).all()
        assert (session.selection[:, 20:] == 0).all()
        # Include colour present nowhere: empty Replace deselects.
        fn['color_range'](session, session.selected_id, None,
                          '[[1, 2, 3]]', '[]', 0, 'Replace')
        assert session.selection is None
    finally:
        session.close()


# 12. Selection is ephemeral: a layer op + undo leaves no history entry
#     for the colour-range selection itself (count stays 1 revision older).
def test_selection_is_ephemeral_no_undo_entry(callbacks):
    fn = edits(callbacks)
    doc = Document(40, 30)
    doc.add_image(np.full((30, 40, 3), 100, np.uint8), 'Base')
    session = EditorSession.create(doc)
    try:
        revision = session.history.revision
        fn['color_range'](session, session.selected_id, None,
                          '[[100, 100, 100]]', '[]', 0, 'Replace')
        assert session.selection is not None
        assert session.history.revision == revision   # no undo step recorded
    finally:
        session.close()
