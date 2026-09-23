"""RAF ``RawExposureBias`` is undone at the default rawpy decode.

Fujifilm records the RAW under-exposure used for highlight headroom in the RAF
header's metadata directory (record 0x9650, ExifTool ``RAF:RawExposureBias``).
The default decode applies ``2**-bias`` in linear light (clamped to 0..+3 EV,
with a ratio-preserving highlight shoulder) before IEC sRGB encoding.
"""

import logging
import struct

import numpy as np
import pytest

from retouch import io as image_io
from retouch.io import (
    RAW_EXPOSURE_BIAS_MAX_GAIN_EV,
    RAW_EXPOSURE_SHOULDER_KNEE,
    apply_raw_exposure_gain,
    imread_engine_with_context,
    color_context_for_path,
    read_image_16bit,
    read_raf_exposure_bias,
    raw_exposure_decode_info,
)


def _srgb(linear):
    linear = np.asarray(linear, dtype=np.float64)
    return np.where(linear <= 0.0031308, linear * 12.92, 1.055 * linear ** (1 / 2.4) - 0.055)


def _write_raf(path, records):
    """Write a minimal RAF: magic, meta-dir offset/length at 92, then records."""
    body = struct.pack(">I", len(records))
    for tag, payload in records:
        body += struct.pack(">HH", tag, len(payload)) + payload
    offset = 160
    header = bytearray(offset)
    header[:16] = b"FUJIFILMCCD-RAW "
    struct.pack_into(">II", header, 92, offset, len(body))
    path.write_bytes(bytes(header) + body)
    return path


def _bias_record(num, den=100):
    return (0x9650, struct.pack(">hh", num, den))


# Linear samples: shadow, mid-tone, near-knee, highlight, white.
_SAMPLES = np.array(
    [[[0, 3277, 6554], [6554, 9830, 655]],
     [[16384, 16384, 16384], [65535, 65535, 65535]]],
    dtype=np.uint16,
)


@pytest.fixture
def mock_rawpy(monkeypatch):
    rawpy = pytest.importorskip("rawpy")

    class Raw:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def postprocess(self, **kwargs):
            return _SAMPLES.copy()

    monkeypatch.setattr(rawpy, "imread", lambda _path: Raw())
    return rawpy


# --- tag parser --------------------------------------------------------------

def test_parser_reads_raw_exposure_bias_after_other_records(tmp_path):
    path = _write_raf(tmp_path / "a.RAF", [
        (0x0100, b"\x00" * 4),
        (0x0141, b"\x00\x0e\x00\x2a"),
        _bias_record(-172),
        (0x9651, b"\xff" * 4),
    ])
    assert read_raf_exposure_bias(path) == pytest.approx(-1.72)


@pytest.mark.parametrize("case", [
    "empty", "bad_magic", "truncated_dir", "missing_record", "zero_den", "not_raf",
])
def test_parser_returns_none_on_bad_input(tmp_path, case):
    if case == "empty":
        path = tmp_path / "a.raf"
        path.touch()
    elif case == "bad_magic":
        path = _write_raf(tmp_path / "a.raf", [_bias_record(-172)])
        data = bytearray(path.read_bytes())
        data[:4] = b"XXXX"
        path.write_bytes(bytes(data))
    elif case == "truncated_dir":
        path = _write_raf(tmp_path / "a.raf", [_bias_record(-172)])
        path.write_bytes(path.read_bytes()[:-3])
    elif case == "missing_record":
        path = _write_raf(tmp_path / "a.raf", [(0x0100, b"\x00" * 4)])
    elif case == "zero_den":
        path = _write_raf(tmp_path / "a.raf", [_bias_record(-172, 0)])
    else:
        path = _write_raf(tmp_path / "a.dng", [_bias_record(-172)])
    assert read_raf_exposure_bias(path) is None


# --- decode ------------------------------------------------------------------

def test_default_decode_applies_bias_gain_to_midtones(tmp_path, mock_rawpy):
    path = _write_raf(tmp_path / "a.RAF", [_bias_record(-172)])
    out = read_image_16bit(path)

    gain = 2 ** 1.72
    linear = _SAMPLES.astype(np.float64) / 65535
    below_knee = (linear * gain).max(axis=-1) <= RAW_EXPOSURE_SHOULDER_KNEE
    expected = (_srgb(linear * gain) * 255)[..., ::-1]
    # Every pixel whose brightest gained channel is under the knee gets the
    # exact 2**-bias gain (old code: no gain -> this fails by ~20-60 levels).
    assert below_knee[0].all()
    np.testing.assert_allclose(out[0], expected[0], atol=1e-3)
    # Highlights are rolled off, never pushed past white.
    assert out.max() <= 255.0 + 1e-4
    assert out[1, 1] == pytest.approx([255.0, 255.0, 255.0], abs=1e-3)
    assert (out[1, 0] > _srgb(0.25) * 255).all()  # still brightened


def test_context_records_bias_and_applied_gain(tmp_path, mock_rawpy):
    path = _write_raf(tmp_path / "a.RAF", [_bias_record(-172)])
    _image, context = imread_engine_with_context(path)
    assert context.raw_exposure_bias_ev == pytest.approx(-1.72)
    assert context.raw_exposure_gain_ev == pytest.approx(1.72)
    assert context.to_dict()["raw_exposure_gain_ev"] == pytest.approx(1.72)


def test_path_context_records_bias_for_batch_exports(tmp_path):
    path = _write_raf(tmp_path / "a.RAF", [_bias_record(-172)])
    context = color_context_for_path(path)
    assert context.raw_exposure_bias_ev == pytest.approx(-1.72)
    assert context.raw_exposure_gain_ev == pytest.approx(1.72)


def test_path_context_reuses_exact_raw_decode_info(tmp_path, monkeypatch):
    path = _write_raf(tmp_path / "a.RAF", [_bias_record(-172)])
    decode_info = {
        "raw_exposure_bias_ev": -1.72,
        "raw_exposure_gain_ev": 0.0,
    }

    def unexpected_reread(*_args, **_kwargs):
        raise AssertionError("context should reuse the completed decode metadata")

    monkeypatch.setattr("retouch.io.raw_exposure_decode_info", unexpected_reread)
    context = color_context_for_path(path, raw_exposure_info=decode_info)
    assert context.raw_exposure_bias_ev == pytest.approx(-1.72)
    assert context.raw_exposure_gain_ev == 0.0


def test_exposure_decode_info_records_bias_when_opted_out(tmp_path):
    path = _write_raf(tmp_path / "a.RAF", [_bias_record(-172)])
    assert raw_exposure_decode_info(path, apply_exposure_bias=False) == {
        "raw_exposure_bias_ev": pytest.approx(-1.72),
        "raw_exposure_gain_ev": 0.0,
    }


def test_linear_raw_cli_path_applies_and_reports_bias(tmp_path, monkeypatch):
    path = _write_raf(tmp_path / "a.RAF", [_bias_record(-172)])

    class FakeDeveloper:
        def load_raw(self, _path):
            return np.full((2, 2, 3), 0.1, dtype=np.float32), {}

        def develop(self, image, exposure=0.0, contrast=1.0):
            return image

    monkeypatch.setattr("retouch.raw_develop.RAWDeveloper", FakeDeveloper)
    from cli import _linear_raw_to_engine_bgr

    applied_info = {}
    applied = _linear_raw_to_engine_bgr(path, decode_info=applied_info)
    expected = _srgb(0.1 * 2 ** 1.72) * 255
    np.testing.assert_allclose(applied, expected, atol=1e-4)
    assert applied_info["raw_exposure_bias_ev"] == pytest.approx(-1.72)
    assert applied_info["raw_exposure_gain_ev"] == pytest.approx(1.72)

    disabled_info = {}
    disabled = _linear_raw_to_engine_bgr(
        path, apply_exposure_bias=False, decode_info=disabled_info
    )
    np.testing.assert_allclose(disabled, _srgb(0.1) * 255, atol=1e-4)
    assert disabled_info["raw_exposure_bias_ev"] == pytest.approx(-1.72)
    assert disabled_info["raw_exposure_gain_ev"] == 0.0


def test_legacy_uint8_raw_path_also_gains(tmp_path, mock_rawpy):
    path = _write_raf(tmp_path / "a.RAF", [_bias_record(-172)])
    image, context = image_io.imread_exif_with_context(path)
    assert image.dtype == np.uint8
    assert context.raw_exposure_gain_ev == pytest.approx(1.72)
    expected = _srgb(3277 / 65535 * 2 ** 1.72) * 255
    assert image[0, 0, 1] == pytest.approx(expected, abs=0.51)


def test_opt_out_decodes_without_gain(tmp_path, mock_rawpy):
    path = _write_raf(tmp_path / "a.RAF", [_bias_record(-172)])
    image, context = imread_engine_with_context(path, apply_exposure_bias=False)
    expected = (_srgb(_SAMPLES / 65535) * 255)[..., ::-1]
    np.testing.assert_allclose(image, expected, atol=1e-3)
    assert context.raw_exposure_gain_ev == 0.0


def test_missing_tag_means_no_gain_and_debug_log(tmp_path, mock_rawpy, caplog):
    path = _write_raf(tmp_path / "a.RAF", [(0x0100, b"\x00" * 4)])
    with caplog.at_level(logging.DEBUG, logger="retouch.io"):
        image, context = imread_engine_with_context(path)
    expected = (_srgb(_SAMPLES / 65535) * 255)[..., ::-1]
    np.testing.assert_allclose(image, expected, atol=1e-3)
    assert context.raw_exposure_bias_ev is None
    assert context.raw_exposure_gain_ev == 0.0
    assert any("No RawExposureBias" in r.getMessage() for r in caplog.records)


@pytest.mark.parametrize("num,expected_gain", [
    (-450, RAW_EXPOSURE_BIAS_MAX_GAIN_EV),  # -4.5 EV -> clamped to +3 EV
    (100, 0.0),                             # positive bias would darken -> 0
])
def test_gain_is_clamped_and_logged(tmp_path, mock_rawpy, caplog, num, expected_gain):
    path = _write_raf(tmp_path / "a.RAF", [_bias_record(num)])
    with caplog.at_level(logging.WARNING, logger="retouch.io"):
        _image, context = imread_engine_with_context(path)
    assert context.raw_exposure_bias_ev == pytest.approx(num / 100)
    assert context.raw_exposure_gain_ev == pytest.approx(expected_gain)
    assert any("clamped" in r.getMessage() for r in caplog.records)


def test_fuji_match_source_is_not_bias_gained(tmp_path, mock_rawpy, monkeypatch):
    path = _write_raf(tmp_path / "a.RAF", [_bias_record(-172)])
    seen = {}
    monkeypatch.setattr(
        image_io, "read_raf_with_raf2jpeg",
        lambda *a, **k: np.zeros((2, 2, 3), np.uint8),
    )
    import retouch.fuji_match as fuji_match

    def fake_calibrate(source, preview, strength, inplace=False):
        seen["source"] = source.copy()
        return source

    monkeypatch.setattr(fuji_match, "calibrate_to_fuji_preview", fake_calibrate)
    imread_engine_with_context(path, raw_decoder="rawpy-fuji-match")
    expected = (_srgb(_SAMPLES / 65535) * 255)[..., ::-1]
    np.testing.assert_allclose(seen["source"], expected, atol=1e-3)


# --- shoulder ----------------------------------------------------------------

def test_shoulder_is_monotonic_bounded_and_exact_below_knee():
    ramp = np.linspace(0.0, 1.0, 4097, dtype=np.float32)
    rgb = np.stack([ramp, ramp * 0.6, ramp * 0.3], axis=-1)[None]
    gain_ev = 2.72
    out = apply_raw_exposure_gain(rgb.copy(), gain_ev)
    gain = 2 ** gain_ev
    peak = out[0, :, 0]
    assert np.all(np.diff(out[0], axis=0) >= -1e-7)  # every channel monotonic
    assert out.max() <= 1.0
    assert peak[-1] == pytest.approx(1.0, abs=1e-5)
    under = ramp * gain <= RAW_EXPOSURE_SHOULDER_KNEE
    np.testing.assert_allclose(out[0, under], rgb[0, under] * gain, rtol=1e-6)
    # Just above the knee the channel ratios (hue) are (nearly) preserved ...
    near = ~under & (ramp * gain <= RAW_EXPOSURE_SHOULDER_KNEE * 1.05)
    assert near.any()
    np.testing.assert_allclose(out[0, near, 1] / out[0, near, 0], 0.6, rtol=0.01)
    # ... and gained white (a sensor-clipped cast) is driven to neutral.
    np.testing.assert_allclose(out[0, -1], [1.0, 1.0, 1.0], atol=1e-5)
