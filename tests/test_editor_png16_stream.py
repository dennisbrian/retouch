"""Streamed 16-bit PNG export: pixel identity, ICC/pHYs parity, memory guard.

export_png's bit_depth==16 branch feeds 256-row slices to a zlib compressobj
instead of a full-canvas cv2.cvtColor BGRA copy + cv2.imwrite (the old path
peaked ~3.1 GB resident on a 200 MP export). PNG is lossless and filter 0
stores raw big-endian samples, so decoded pixels are bit-identical.
"""
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from retouch.editor.composite import composite_16, export_png
from retouch.editor.document import Document
from retouch.io import get_working_srgb_icc

PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def make_document(w, h, rng, resolution=144.0):
    doc = Document(width=w, height=h, resolution=resolution)
    lw, lh = max(1, w), max(1, h)
    doc.add_image(rng.integers(0, 256, size=(lh, lw, 4), dtype=np.uint8), 'base')
    return doc


def read16(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert img is not None and img.dtype == np.uint16 and img.shape[2] == 4
    return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)


def chunks_of(path):
    raw = path.read_bytes()
    pos = len(PNG_SIGNATURE)
    out = []
    while pos < len(raw):
        length = struct.unpack_from('>I', raw, pos)[0]
        kind = raw[pos + 4:pos + 8]
        out.append((kind, raw[pos + 8:pos + 8 + length]))
        pos += 12 + length
    return out


@pytest.mark.parametrize('w,h', [(1, 1), (17, 1), (1, 9), (13, 13), (256, 256),
                                 (300, 257), (513, 255), (640, 383), (1023, 257)])
def test_streamed_pixels_match_composite_16(tmp_path, w, h):
    rng = np.random.default_rng(w * 100 + h)
    doc = make_document(w, h, rng)
    out = tmp_path / 'stream.png'
    assert export_png(doc, out, bit_depth=16) == out
    expected = composite_16(doc).astype(np.uint16)
    np.testing.assert_array_equal(read16(out), expected)


def test_icc_profile_round_trips(tmp_path):
    rng = np.random.default_rng(1)
    p = tmp_path / 'icc.png'
    export_png(make_document(64, 64, rng), p, bit_depth=16)
    with Image.open(p) as im:
        assert im.info.get('icc_profile') == get_working_srgb_icc()


def test_chunk_order_and_phys(tmp_path):
    rng = np.random.default_rng(2)
    doc = make_document(32, 32, rng, resolution=300.0)
    p = tmp_path / 'chunks.png'
    export_png(doc, p, bit_depth=16)
    kinds = [k for k, _ in chunks_of(p)]
    assert kinds[0] == b'IHDR' and kinds[1] == b'iCCP'
    assert kinds[2] == b'pHYs' and kinds[-1] == b'IEND'
    assert all(k == b'IDAT' for k in kinds[3:-1])
    ihdr = chunks_of(p)[0][1]
    width, height, depth, color = struct.unpack('>IIBB', ihdr[:10])[:4]
    assert (width, height, depth, color) == (32, 32, 16, 6)
    phys = dict(chunks_of(p))[b'pHYs']
    px, py, unit = struct.unpack('>IIB', phys)
    assert unit == 1 and abs(px - round(300 * 39.3701)) <= 1 and px == py


def test_8_bit_path_untouched(tmp_path):
    """The 8-bit branch still routes through write_image_with_icc."""
    from retouch.editor.composite import composite
    rng = np.random.default_rng(3)
    doc = make_document(20, 10, rng)
    p8 = tmp_path / 'eight.png'
    export_png(doc, p8, bit_depth=8)
    img = cv2.imread(str(p8), cv2.IMREAD_UNCHANGED)
    assert img.dtype == np.uint8
    np.testing.assert_array_equal(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA),
                                  composite(doc))


def test_large_export_memory(tmp_path):
    """Peak RSS on a ~40 MP 16-bit export stays far below the old cv2 path
    (which measured ~2.5 GB); a streamed failure regresses toward that."""
    import resource
    import subprocess
    import sys
    W, H = 6400, 6208
    code = ('import sys;'
            f'sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r});'
            'import numpy as np;'
            'from retouch.editor.composite import export_png;'
            'from retouch.editor.document import Document;'
            f'doc = Document(width={W}, height={H});'
            'doc.add_image(np.random.default_rng(11).integers('
            f'0, 256, size=({H}, {W}, 4), dtype=np.uint8), "base");'
            f'export_png(doc, {str(tmp_path / "big.png")!r}, bit_depth=16);'
            'import resource;'
            'print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)')
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    mb = int(r.stdout.strip().splitlines()[-1]) / 1e6  # macOS bytes
    assert mb < 2200, f'streamed writer peaked at {mb:.0f} MB'
    del resource
