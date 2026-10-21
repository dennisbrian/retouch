"""Format v12-14 read-side compatibility rehearsal (scratch-only battery).

v12: 16-bit PNG layer images load via upstream dither_quantize16 semantics
     (DeepLayer's dithered 8-bit copy), not plain rounding.
v13: .exr layer imageFile refused with UnsupportedFeature in _check_layer,
     before any file bytes are opened.
v14: shape 'smoothing' key refused below v14, loaded-but-ignored on v14.
Gates: v15 refused; 16-bit in v11 refused; .exr in v12 refused; smoothing
     in v13 refused. Save of a v12-loaded project writes v11 with the same
     (dithered) pixels.
"""
import numpy as np
import pytest
from PIL import Image

from retouch.editor import (
    ProjectError,
    UnsupportedFeature,
    load_project,
    save_project,
)
from tests import compositor_fixtures as fx

from retouch.editor.project_store import _dither_quantize16

# --- local fixture helpers on top of the repo's write_package/png16 ----------

def _write_v12(path, rgba16, version=12, layer=None):
    """A package whose single layer image is a 16-bit RGBA PNG."""
    height, width = rgba16.shape[:2]
    record = layer or fx.layer_record(0, 'Ramp 16', width, height)
    path = fx.write_package(path, width, height, [record], version=version)
    fx.png16(path / 'images' / (fx.IDS[0] + '.png'), width, height,
             rgba16.astype('>u4').astype('>u2').tolist()
             if rgba16.dtype.kind == 'f' else
             np.asarray(rgba16, '>u2').reshape(-1).tolist())
    return path


EXR_BYTES = (b'\x76\x2f\x31\x01'      # OpenEXR magic + version
             b'\x02\x00\x00\x00' + b'\x00' * 64)


def _write_v13(path, version=13):
    record = fx.layer_record(0, 'HDR', 4, 4)
    record['imageFile'] = fx.IDS[0] + '.exr'
    path = fx.write_package(path, 4, 4, [record], version=version)
    (path / 'images' / (fx.IDS[0] + '.exr')).write_bytes(EXR_BYTES)
    return path


def _write_v14(path, version=14, smoothing=True):
    record = fx.layer_record(0, 'Shape', 4, 4,
                             shape={'kind': 'Rectangle', 'red': 0.1, 'green': 0.5,
                                    'blue': 0.9, 'cornerRadius': 4,
                                    'smoothing': smoothing})
    path = fx.write_package(path, 4, 4, [record], version=version,
                            images={0: fx.solid(4, 4, (10, 200, 30, 255))})
    return path




def _dithered_upstream_8bit(rgba16):
    """Upstream's 8-bit copy of a 16-bit layer: premultiply, dither_quantize16."""
    alpha16 = rgba16[..., 3:4].astype(np.float64)
    premult = np.rint(rgba16.astype(np.float64) * (alpha16 / 65535.0)).astype(np.uint16)
    return _dither_quantize16(premult)


def _ramp16(width=256, height=4, alpha=65535):
    ramp = np.tile((np.arange(width) * 65535 // (width - 1)).astype(np.uint16),
                   (height, 1))
    return np.stack([ramp, ramp, ramp,
                     np.full((height, width), alpha, np.uint16)], axis=-1)


class TestV12SixteenBit:
    def test_loads_dithered_not_plain_rounding(self, tmp_path):
        rgba16 = _ramp16()
        path = _write_v12(tmp_path / 'r12.comp', rgba16)
        layer = load_project(path).layers[0]
        assert layer.pixels.dtype == np.uint8

        # Opaque ramp: straight == premultiplied, so bit-exact vs the replica.
        np.testing.assert_array_equal(
            layer.pixels, _dithered_upstream_8bit(rgba16))

        # And the dither is genuinely different from plain rounding.
        plain = np.rint(rgba16[..., :3].astype(np.float64)
                        * (255.0 / 65535.0)).astype(np.uint8)
        diff = np.abs(layer.pixels[..., :3].astype(int) - plain.astype(int))
        assert diff.max() >= 1, 'dither must differ from plain rounding'
        assert diff.max() <= 1, 'dither stays within one step'

    def test_semitransparent_round_trips_premultiplied(self, tmp_path):
        alpha = (np.arange(8) * 65535 // 7).astype(np.uint16)
        rgba16 = np.stack([np.full((4, 8), 65535, np.uint16),
                           np.zeros((4, 8), np.uint16),
                           np.zeros((4, 8), np.uint16),
                           np.tile(alpha, (4, 1))], axis=-1)
        path = _write_v12(tmp_path / 's12.comp', rgba16)
        got = np.asarray(load_project(path).layers[0].pixels)
        want = _dithered_upstream_8bit(rgba16)
        np.testing.assert_array_equal(got[..., 3], want[..., 3])
        a8 = got[..., 3:4].astype(np.float64)
        back_premult = np.rint(got[..., :3].astype(np.float64) * a8 / 255.0)
        rd = np.abs(back_premult - want[..., :3].astype(np.float64))
        assert rd.max() <= 1

    def test_composite_matches_dithered_pixels(self, tmp_path):
        rgba16 = _ramp16()
        path = _write_v12(tmp_path / 'c12.comp', rgba16)
        from retouch.editor import composite
        flat = composite(load_project(path))
        want = _dithered_upstream_8bit(rgba16)
        fdiff = np.abs(flat.astype(int)[..., :3] - want[..., :3].astype(int))
        assert fdiff.max() <= 1

    def test_save_writes_v11_preserving_pixels(self, tmp_path):
        rgba16 = _ramp16()
        path = _write_v12(tmp_path / 'rt12.comp', rgba16)
        doc = load_project(path)
        expected = np.asarray(doc.layers[0].pixels)
        out = save_project(doc, tmp_path / 'rt11.comp')
        assert out.exists()
        import json
        manifest = json.loads((out / 'manifest.json').read_text())
        assert manifest['version'] == 11
        header = (out / 'images' / (fx.IDS[0] + '.png')).read_bytes()[:26]
        assert header[24] == 8, 'saved asset must be 8-bit'
        reloaded = load_project(out)
        np.testing.assert_array_equal(
            np.asarray(reloaded.layers[0].pixels), expected)


class TestV13ExrRefusal:
    def test_exr_refused_cleanly(self, tmp_path):
        path = _write_v13(tmp_path / 'exr13.comp')
        with pytest.raises(UnsupportedFeature, match='HDR'):
            load_project(path)

    def test_exr_in_v12_refused_as_invalid(self, tmp_path):
        path = _write_v13(tmp_path / 'exr12.comp', version=12)
        with pytest.raises(ProjectError, match='EXR layer assets'):
            load_project(path)


class TestV14Smoothing:
    def test_smoothing_loaded_and_ignored_on_v14(self, tmp_path):
        path = _write_v14(tmp_path / 'sm14.comp')
        layer = load_project(path).layers[0]
        assert layer.pixels is not None
        assert layer.shape is not None

    def test_smoothing_in_v13_refused(self, tmp_path):
        path = _write_v14(tmp_path / 'sm13.comp', version=13)
        with pytest.raises(ProjectError, match='smoothing'):
            load_project(path)

    def test_smoothing_false_needs_no_v14(self, tmp_path):
        path = _write_v14(tmp_path / 'sm13f.comp', version=13,
                                    smoothing=False)
        assert load_project(path).layers[0].shape is not None


class TestVersionGates:
    def test_v15_refused(self, tmp_path):
        rgba16 = _ramp16()
        path = _write_v12(tmp_path / 'v15.comp', rgba16)
        (path / 'manifest.json').write_text(
            (path / 'manifest.json').read_text().replace('"version": 12',
                                                          '"version": 15'))
        with pytest.raises(ProjectError, match='15'):
            load_project(path)

    def test_sixteen_bit_in_v11_refused(self, tmp_path):
        rgba16 = _ramp16()
        path = _write_v12(tmp_path / 'r11.comp', rgba16, version=11)
        with pytest.raises(ProjectError, match='only 8-bit'):
            load_project(path)

    def test_sixteen_bit_mask_refused(self, tmp_path):
        rgba16 = _ramp16()
        record = fx.layer_record(0, 'M', 256, 4)
        record['imageFile'] = None
        record['maskFile'] = fx.IDS[0] + '.mask.png'
        path = _write_v12(tmp_path / 'm12.comp', rgba16, layer=record)
        # The write_v12_package helper wrote the layer PNG; turn it into a
        # 16-bit mask asset instead (mask files are grayscale-or-RGBA PNGs,
        # only the depth gate matters here).
        images = path / 'images'
        mask_path = images / (fx.IDS[0] + '.mask.png')
        layer_png = images / (fx.IDS[0] + '.png')
        layer_png.rename(mask_path)
        with pytest.raises(ProjectError, match='only 8-bit'):
            load_project(path)
