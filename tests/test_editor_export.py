"""export_jpeg: encoded-sRGB flatten over background, colour-managed JPEG write.

Expected values verified against the prototype report (proto_export_jpeg.py):
semi-transparent red 50% over white at q85 lands at (255,128,127)+-3, empty
corners stay white, the container is JFIF with an embedded sRGB ICC profile
and the document resolution as DPI.
"""
import numpy as np
import pytest
from PIL import Image

from retouch.editor import Document, export_jpeg, export_png
from tests import compositor_fixtures as fx


def _red_doc():
    # Full-canvas 50% red over white: no spatial edges, so JPEG stays analytic.
    doc = Document(2, 2, resolution=144)
    doc.add_image(fx.solid(2, 2, (255, 0, 0, 128)), 'Red')
    return doc


def _corner_doc():
    # Opaque red pixel in one corner; the far corner (6 blocks away) stays pure
    # background — JPEG ringing falls off well inside that distance.
    doc = Document(16, 16, resolution=144)
    doc.add_image(np.array([[[255, 0, 0, 255]]], np.uint8), 'Red', origin=(0, 0))
    return doc


def test_jpeg_flattens_encoded_srgb_over_white(tmp_path):
    path = export_jpeg(_red_doc(), tmp_path / 'flat.jpg')
    with Image.open(path) as image:
        assert image.mode == 'RGB'
        pixel = np.asarray(image)[0, 0]
        np.testing.assert_allclose(pixel, (255, 128, 127), atol=3)


def test_jpeg_empty_corner_stays_background(tmp_path):
    path = export_jpeg(_corner_doc(), tmp_path / 'flat.jpg')
    with Image.open(path) as image:
        corner = np.asarray(image)[15, 15]
        np.testing.assert_allclose(corner, (255, 255, 255), atol=3)


def test_jpeg_carries_jfif_icc_and_dpi(tmp_path):
    path = export_jpeg(_red_doc(), tmp_path / 'flat.jpg')
    head = path.read_bytes()[:32]
    assert head[:2] == b'\xff\xd8'           # SOI
    assert b'JFIF' in head                   # JFIF APP0
    with Image.open(path) as image:
        assert image.info['icc_profile']
        dpi = image.info.get('dpi', (72, 72))
        assert round(dpi[0]) == 144 and round(dpi[1]) == 144
    assert [p.name for p in tmp_path.iterdir()] == ['flat.jpg']


def test_jpeg_smaller_than_png_on_noise(tmp_path):
    rng = np.random.default_rng(7)
    doc = Document(64, 64, resolution=144)
    doc.add_image(rng.integers(0, 256, (64, 64, 4), dtype=np.uint8), 'Noise')
    jpeg = export_jpeg(doc, tmp_path / 'noise.jpg')
    png = export_png(doc, tmp_path / 'noise.png')
    assert jpeg.stat().st_size < png.stat().st_size


def test_jpeg_quality_validated(tmp_path):
    with pytest.raises(ValueError):
        export_jpeg(_red_doc(), tmp_path / 'bad.jpg', quality=1.5)


def test_jpeg_leaves_no_temp_files_on_success(tmp_path):
    export_jpeg(_red_doc(), tmp_path / 'ok.jpg')
    assert [p.name for p in tmp_path.iterdir()] == ['ok.jpg']


def test_jpeg_leaves_no_temp_files_on_write_failure(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise OSError('disk full')
    # export_jpeg imports the writer from retouch.io at call time, so patch
    # the io module's attribute.
    import retouch.io as io_mod
    monkeypatch.setattr(io_mod, 'write_image_with_icc', boom)
    with pytest.raises(OSError):
        export_jpeg(_red_doc(), tmp_path / 'fail.jpg')
    assert list(tmp_path.iterdir()) == []     # no target, no temp leftovers
