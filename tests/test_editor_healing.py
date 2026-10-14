"""Spot Healing and Clone Stamp: upstream test contracts, native kernel parity, UI."""
import ctypes
import shutil
import subprocess
import tarfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pytest

from retouch.editor import Document, composite
from retouch.editor.healing import (
    HEAL_MODES,
    _premultiply,
    _unpremultiply,
    clone_stamp,
    spot_heal,
)
from retouch.editor.ui import EditorSession, build_editor_tab

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'third_party' / 'compositor' / 'source-11d8d7a.tar.gz'


def blemished():
    """Upstream SpotHealingTests: 2 px gray stripes with a 10x10 red blemish."""
    image = np.zeros((80, 120, 4), np.uint8)
    gray = np.where(np.arange(120) % 4 < 2, 100, 112).astype(np.uint8)
    image[..., :3] = gray[None, :, None]
    image[..., 3] = 255
    image[35:45, 55:65] = (230, 20, 20, 255)
    return image


def dab(shape, center, radius):
    coverage = np.zeros(shape, np.uint8)
    cv2.circle(coverage, center, radius, 255, -1)
    return coverage


@pytest.mark.parametrize('mode', HEAL_MODES)
def test_heals_the_blemish_under_the_brush_and_nothing_else(mode):
    original = blemished()
    coverage = dab(original.shape[:2], (60, 40), 12)
    healed = spot_heal(original, coverage, mode)
    for x, y in [(60, 40), (56, 36), (64, 44)]:
        r, g, _, a = (int(v) for v in healed[y, x])
        assert r - g < 30 and 80 <= g <= 130 and a == 255, (mode, x, y, healed[y, x])
    untouched = coverage == 0
    np.testing.assert_array_equal(healed[untouched], original[untouched])


def test_heal_opacity_blends_and_rejects_bad_settings():
    original = blemished()
    coverage = dab(original.shape[:2], (60, 40), 12)
    full = spot_heal(original, coverage, 'Content-Aware', 1.0).astype(int)
    half = spot_heal(original, coverage, 'Content-Aware', 0.5).astype(int)
    centre = original[40, 60].astype(int)
    assert abs(half[40, 60, 1] - (centre[1] + full[40, 60, 1]) / 2) <= 1
    np.testing.assert_array_equal(spot_heal(original, coverage, 'Content-Aware', 0), original)
    with pytest.raises(ValueError):
        spot_heal(original, coverage, 'Patch')
    with pytest.raises(ValueError):
        spot_heal(original, coverage[:-1], 'Content-Aware')


def test_separate_dabs_heal_on_their_own():
    original = blemished()
    original[10:16, 10:16] = (230, 20, 20, 255)
    coverage = np.maximum(dab(original.shape[:2], (60, 40), 12), dab(original.shape[:2], (13, 13), 6))
    healed = spot_heal(original, coverage, 'Content-Aware')
    for x, y in [(60, 40), (13, 13)]:
        assert int(healed[y, x, 0]) - int(healed[y, x, 1]) < 30


def test_transparent_layers_keep_straight_alpha_round_trip():
    pixels = np.zeros((40, 40, 4), np.uint8)
    pixels[..., :3] = (200, 120, 60)
    pixels[..., 3] = 128
    pixels[18:22, 18:22] = (10, 10, 10, 128)
    healed = spot_heal(pixels, dab((40, 40), (20, 20), 5), 'Create Texture')
    assert healed[20, 20, 3] == 128
    assert abs(int(healed[20, 20, 0]) - 200) <= 4


def test_clone_copies_the_source_and_keeps_the_offset():
    """Upstream CloneStampTests: red left half with a green square, blue right half."""
    image = np.zeros((40, 80, 4), np.uint8)
    image[..., 3] = 255
    image[:, :40, 0] = 255
    image[:, 40:, 2] = 255
    image[15:25, 10:20] = (0, 255, 0, 255)
    # Source (15, 20) painted at (60, 20): offset (-45, 0).
    first = clone_stamp(image, dab((40, 80), (60, 20), 3), (-45, 0))
    assert list(first[20, 60]) == [0, 255, 0, 255]
    assert list(first[5, 70]) == [0, 0, 255, 255]
    # Aligned: the next stroke keeps (-45, 0), so (66, 20) copies red from (21, 20).
    second = clone_stamp(first, dab((40, 80), (66, 20), 3), (-45, 0))
    assert list(second[20, 66]) == [255, 0, 0, 255]


def test_clone_outside_the_layer_draws_nothing_and_requires_an_offset():
    image = np.zeros((20, 20, 4), np.uint8)
    image[..., 3] = 255
    image[..., 1] = 90
    np.testing.assert_array_equal(clone_stamp(image, dab((20, 20), (2, 10), 2), (-15, 0)), image)
    with pytest.raises(ValueError, match='offset'):
        clone_stamp(image, dab((20, 20), (10, 10), 2), (0, 0))


@pytest.fixture(scope='module')
def native_heal(tmp_path_factory):
    """Upstream HealPixels.c compiled from the pinned source archive."""
    compiler = shutil.which('cc') or shutil.which('gcc') or shutil.which('clang')
    if compiler is None or not ARCHIVE.exists():
        pytest.skip('needs a C compiler and the pinned Compositor source archive')
    work = tmp_path_factory.mktemp('heal')
    with tarfile.open(ARCHIVE) as archive:
        for member in archive.getmembers():
            if member.name.endswith(('Rendering/HealPixels.c', 'Rendering/HealPixels.h')):
                member.name = Path(member.name).name
                archive.extract(member, work)
    library = work / 'libheal.so'
    subprocess.run([compiler, '-O2', '-shared', '-fPIC', '-o', str(library),
                    str(work / 'HealPixels.c'), '-lm'], check=True, capture_output=True)
    lib = ctypes.CDLL(str(library))
    lib.spot_heal.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t,
                              ctypes.c_size_t, ctypes.c_float, ctypes.c_int, ctypes.c_uint32]

    def run(rgba, coverage, mode, opacity, seed):
        pixels = np.ascontiguousarray(_premultiply(rgba))
        coverage = np.ascontiguousarray(coverage)
        assert lib.spot_heal(pixels.ctypes.data, coverage.ctypes.data, pixels.shape[1],
                             pixels.shape[0], pixels.shape[1] * 4, opacity, mode, seed) == 0
        return _unpremultiply(pixels)
    return run


@pytest.mark.parametrize('mode', range(3))
def test_matches_the_native_kernel(native_heal, mode):
    original = blemished()
    coverage = dab(original.shape[:2], (60, 40), 12)
    ported = spot_heal(original, coverage, HEAL_MODES[mode], seed=0)  # one dab: label 1
    native = native_heal(original, coverage, mode, 1.0, 1)
    diff = np.abs(ported.astype(int) - native.astype(int))
    if mode == 1:
        # Create Texture: red-black vs row-major relaxation over a striped edge.
        assert diff.max() <= 16 and diff[coverage > 0].mean() < 3
    else:
        np.testing.assert_array_equal(ported, native)
    # A smooth surface: the membranes agree within a level.
    yy, xx = np.mgrid[0:200, 0:300]
    ramp = np.zeros((200, 300, 4), np.uint8)
    ramp[..., 0], ramp[..., 1], ramp[..., 2], ramp[..., 3] = xx * 0.6, yy * 0.9, 120, 255
    ramp[80:120, 130:170] = (250, 0, 250, 255)
    coverage = dab(ramp.shape[:2], (150, 100), 30)
    diff = np.abs(spot_heal(ramp, coverage, HEAL_MODES[mode], seed=0).astype(int)
                  - native_heal(ramp, coverage, mode, 1.0, 1).astype(int))
    assert diff.max() <= 1


@pytest.fixture
def session():
    doc = Document(120, 80)
    doc.add_image(blemished(), 'Surface')
    result = EditorSession.create(doc)
    yield result
    result.close()


def brushed(center, radius, flip_x=False):
    stroke = np.zeros((80, 120, 4), np.uint8)
    cv2.circle(stroke, center, radius, (255, 51, 102, 255), -1)
    if flip_x:
        stroke = stroke[:, ::-1].copy()
    return {'layers': [stroke]}


def test_session_heal_is_one_undo_step(session):
    before = composite(session.history.document)
    revision = session.history.revision
    assert session.paint_pixels(brushed((60, 40), 12), 'heal')
    assert session.history.undo_label == 'Spot Healing'
    healed = session.selected().pixels
    assert int(healed[40, 60, 0]) - int(healed[40, 60, 1]) < 30
    session.history.undo()
    assert session.history.revision == revision
    np.testing.assert_array_equal(composite(session.history.document), before)


def test_session_heal_follows_a_flipped_layer(session):
    session.history.update_layer(session.selected_id, flip_x=True)
    # The canvas shows the flipped layer, so the blemish is drawn at x = 119 - 60.
    session.paint_pixels(brushed((60, 40), 12, flip_x=True), 'heal')
    healed = session.selected().pixels
    assert int(healed[40, 60, 0]) - int(healed[40, 60, 1]) < 30


def test_session_rejects_layers_without_pixels_and_empty_strokes(session):
    with pytest.raises(ValueError, match='Brush'):
        session.paint_pixels({'layers': []}, 'heal')
    group = session.history.document.layers[0]
    session.history.group_layers([group.id])
    session.selected_id = next(l.id for l in session.history.document.layers if l.is_group)
    with pytest.raises(ValueError, match='pixel layer'):
        session.paint_pixels(brushed((60, 40), 12), 'heal')


def test_heal_and_clone_buttons_accept_pending_strokes(session):
    with gr.Blocks() as app, gr.Tabs():
        build_editor_tab()
    edits = {value.fn.__kwdefaults__['operation']: value.fn for value in app.fns.values()
             if value.fn and value.fn.__name__ == 'edit'}
    edits['heal'](session, session.selected_id, brushed((60, 40), 12), 'Proximity Match', 100)
    assert session.history.undo_label == 'Spot Healing'
    edits['clone'](session, session.selected_id, brushed((100, 10), 3), 100, -38, 30)
    assert session.history.undo_label == 'Clone Stamp'
    np.testing.assert_array_equal(session.selected().pixels[10, 100], session.selected().pixels[40, 62])
    with pytest.raises(gr.Error, match='offset'):
        edits['clone'](session, session.selected_id, brushed((100, 10), 3), 100, 0, 0)
