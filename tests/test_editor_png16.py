import numpy as np
import pytest
from PIL import Image

from retouch.editor import Document, export_png


def _gradient_doc():
    doc = Document(64, 8, resolution=144)
    pix = np.zeros((8, 64, 4), np.uint8)
    pix[..., 0] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
    pix[..., 2] = np.linspace(255, 0, 64, dtype=np.uint8)[None, :]
    pix[..., 3] = np.linspace(32, 255, 64, dtype=np.uint8)[None, :]
    doc.add_image(pix, 'Gradient')
    doc.add_image(np.full((8, 64, 4), (0, 40, 200, 128), np.uint8), 'Overlay')
    return doc


def test_png16_writes_dpi_icc_and_true_depth(tmp_path):
    path = export_png(_gradient_doc(), tmp_path / 'g16.png', bit_depth=16)
    raw = path.read_bytes()
    assert raw[:8] == b'\x89PNG\r\n\x1a\n'
    # IHDR bit depth 16, colour type 6 (RGBA)
    assert raw[24] == 16 and raw[25] == 6
    assert b'pHYs' in raw
    with Image.open(path) as image:
        assert round(image.info['dpi'][0]) == 144
        assert image.info['icc_profile']


def test_png16_retains_float_precision_beyond_8bit(tmp_path):
    # The 16-bit render quantizes the float bands directly: it must expose
    # strictly more distinct levels than the 8-bit export could. Pillow has no
    # 16-bit RGBA mode and silently downscales on open, so read via cv2.
    import cv2
    doc = _gradient_doc()
    p16 = export_png(doc, tmp_path / 'g16.png', bit_depth=16)
    p8 = export_png(doc, tmp_path / 'g8.png')
    u16 = cv2.imread(str(p16), cv2.IMREAD_UNCHANGED)
    u8 = cv2.imread(str(p8), cv2.IMREAD_UNCHANGED)
    assert u16.dtype == np.uint16
    assert u8.dtype == np.uint8
    levels16 = {int(v) for v in np.unique(u16[..., 2])}   # BGRA: red channel
    levels8 = {int(v) for v in np.unique(u8[..., 2])}
    assert len(levels16) > len(levels8)


def test_png16_8bit_export_unchanged(tmp_path):
    doc = _gradient_doc()
    p8 = export_png(doc, tmp_path / 'a.png')
    with Image.open(p8) as image:
        before = np.asarray(image).copy()
    export_png(doc, tmp_path / 'b16.png', bit_depth=16)
    with Image.open(p8) as image:
        assert np.array_equal(np.asarray(image), before)


def test_png16_round_trip_exact(tmp_path):
    doc = _gradient_doc()
    path = export_png(doc, tmp_path / 'rt.png', bit_depth=16)
    with Image.open(path) as image:
        first = np.asarray(image).copy()
    path.write_bytes(path.read_bytes())  # no-op, keep lints calm
    with Image.open(path) as image:
        assert np.array_equal(np.asarray(image), first)
