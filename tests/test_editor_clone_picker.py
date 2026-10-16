"""Clone anchors map reduced previews to native pixels without editing history."""
import numpy as np
import pytest

from retouch.editor import Document, DocumentHistory, composite
from retouch.editor.clone_picker import prepare_clone_picker, pick_clone_point
from retouch.editor.local_retouch import capture_composite, apply_local_retouch, painted_mask


def setup():
    doc = Document(96, 80)
    pixels = np.zeros((80, 96, 3), np.uint8)
    pixels[:, :48] = [120, 60, 30]
    doc.add_image(pixels, 'Original')
    history = DocumentHistory(doc, saved=True)
    _, capture = capture_composite(history, preview_side=48)
    return history, capture


def test_anchor_mapping_markers_and_actual_clone():
    history, capture = setup()
    placement = prepare_clone_picker(history, capture)
    before = placement.preview.copy()
    placement, marked, offset, _ = pick_clone_point(history, placement, [10, 10], 'Source')
    assert placement.source == (20, 20) and offset == (0, 0)
    assert not np.array_equal(marked, before)
    placement, marked, offset, message = pick_clone_point(history, placement, [30, 10], 'Destination')
    assert placement.destination == (60, 20) and offset == (-40, 0)
    assert 'Paint around the destination' in message
    assert history.revision == capture.revision and len(history.document.layers) == 1
    np.testing.assert_array_equal(placement.preview, before)
    # Preview annotations never become clone pixels.
    stroke = np.zeros((40, 48, 4), np.uint8)
    stroke[10:12, 30:32] = [255, 0, 0, 255]
    coverage = painted_mask({'layers': [stroke]}, capture)
    expected = np.zeros((80, 96, 4), np.uint8)
    expected[..., 3] = 255
    expected[:, :48, :3] = [120, 60, 30]
    ys, xs = np.nonzero(coverage)
    expected[ys, xs, :3] = np.rint(coverage[ys, xs, None] / 255 * np.array([120, 60, 30]))
    lid = apply_local_retouch(history, capture, {'layers': [stroke]}, 'Clone', clone_offset=offset)
    assert history.document.layer(lid).name == 'Clone'
    np.testing.assert_array_equal(composite(history.document), expected)


@pytest.mark.parametrize('point', [None, [True, 0], [0.5, 0], [-1, 0], [48, 0], [0, 40], [1]])
def test_invalid_clicks_rejected(point):
    history, capture = setup()
    placement = prepare_clone_picker(history, capture)
    with pytest.raises(ValueError, match='inside'):
        pick_clone_point(history, placement, point, 'Source')
    assert history.revision == capture.revision


def test_stale_picker_and_capture_refused():
    history, capture = setup()
    placement = prepare_clone_picker(history, capture)
    history.update_layer(history.document.layers[0].id, name='Edited')
    with pytest.raises(ValueError, match='Document changed'):
        pick_clone_point(history, placement, [10, 10], 'Source')
    with pytest.raises(ValueError, match='Document changed'):
        prepare_clone_picker(history, capture)


def test_native_edge_coordinates_and_reselecting_anchors():
    history, capture = setup()
    placement = prepare_clone_picker(history, capture)
    placement, _, _, _ = pick_clone_point(history, placement, [47, 39], 'Destination')
    placement, _, offset, _ = pick_clone_point(history, placement, [0, 0], 'Source')
    assert all(0 <= v < limit for v, limit in zip(placement.destination, capture.size))
    assert offset == (-94, -78)
    placement, _, offset, _ = pick_clone_point(history, placement, [47, 39], 'Source')
    assert offset == (0, 0)  # the apply action explicitly refuses zero offsets
