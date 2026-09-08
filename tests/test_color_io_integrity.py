"""Color I/O regression anchors independent of Retouch's inverse helpers."""

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageCms, ImageOps

from retouch import io as image_io
from retouch.raw_develop import RAWDeveloper


@pytest.mark.parametrize("mode,profile_name,color", [
    ("CMYK", "Generic CMYK Profile.icc", (0, 180, 180, 0)),
    ("L", "Generic Gray Gamma 2.2 Profile.icc", 90),
    ("LAB", None, (140, 160, 110)),
])
def test_native_profile_mode_is_preserved_until_transform(tmp_path, mode, profile_name, color):
    if profile_name:
        profile_path = Path("/System/Library/ColorSync/Profiles") / profile_name
        if not profile_path.exists():
            pytest.skip("Optional system ICC fixture unavailable")
        source_icc = profile_path.read_bytes()
    else:
        source_icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    source = Image.new(mode, (4, 3), color)
    path = tmp_path / "native.tif"
    source.save(path, icc_profile=source_icc)
    expected = ImageCms.profileToProfile(
        source, ImageCms.ImageCmsProfile(io.BytesIO(source_icc)),
        ImageCms.ImageCmsProfile(io.BytesIO(image_io.get_working_srgb_icc())),
        outputMode="RGB", renderingIntent=ImageCms.Intent.PERCEPTUAL,
    )
    actual, context = image_io.imread_engine_with_context(path)
    np.testing.assert_array_equal(actual[..., ::-1], np.asarray(expected))
    assert context.source_profile == source_icc


@pytest.mark.parametrize("extension", ["png", "tif"])
@pytest.mark.parametrize("channels", [1, 3])
def test_uint16_ingest_scales_full_range_and_records_downgrade(tmp_path, extension, channels):
    ramp = np.array([[0, 1, 256, 257, 32768, 65535]], dtype=np.uint16)
    samples = ramp if channels == 1 else np.stack((ramp, ramp[:, ::-1], ramp), axis=-1)
    path = tmp_path / ("ramp." + extension)
    assert cv2.imwrite(str(path), samples)
    actual, context = image_io.imread_engine_with_context(path)
    expected = np.rint(samples.astype(np.float64) / 257).astype(np.uint8)
    if channels == 1:
        expected = np.repeat(expected[..., None], 3, axis=-1)
    # RGB16 TIFF's Pillow decoder may select the high byte (<= 1 code error).
    np.testing.assert_allclose(actual, expected, atol=1)
    assert context.to_dict()["source_bit_depth"] == 16
    assert context.to_dict()["working_bit_depth"] == 8


@pytest.mark.parametrize("orientation", range(1, 9))
def test_uint16_png_orientation_applied_once(tmp_path, orientation):
    samples = (np.arange(18, dtype=np.uint16).reshape(2, 3, 3) * 3000)
    path = tmp_path / "oriented.png"
    exif = Image.Exif()
    exif[274] = orientation
    image_io.write_image_with_icc(path, samples, bit_depth=16)
    # Add source orientation without an 8-bit re-encode.
    raw = path.read_bytes()
    path.write_bytes(raw[:33] + image_io._png_chunk(b"eXIf", exif.tobytes()[6:]) + raw[33:])
    actual, _ = image_io.imread_engine_with_context(path)
    expected = Image.fromarray(np.rint(samples[..., ::-1] / 257).astype(np.uint8))
    expected.info["exif"] = exif.tobytes()
    np.testing.assert_array_equal(actual[..., ::-1], np.asarray(ImageOps.exif_transpose(expected)))


@pytest.mark.parametrize("prefer_16bit", [True, False])
def test_raw_ingest_uses_linear_decode_then_iec_srgb(tmp_path, monkeypatch, prefer_16bit):
    # rawpy is an optional package; the core test matrix intentionally does
    # not install the ``raw`` extra, so keep this RAW-specific contract
    # runnable when the optional decoder is present and skipped otherwise.
    rawpy = pytest.importorskip("rawpy")
    calls = []
    samples = np.array([[[0, 205, 11796], [32768, 65535, 655]]], dtype=np.uint16)

    class Raw:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def postprocess(self, **kwargs):
            calls.append(kwargs)
            return samples

    monkeypatch.setattr(rawpy, "imread", lambda _: Raw())
    path = tmp_path / "sample.raf"
    path.touch()
    actual, _ = image_io.imread_engine_with_context(path, prefer_16bit=prefer_16bit)
    assert calls[0]["gamma"] == (1, 1)
    assert calls[0]["output_bps"] == 16
    linear = samples.astype(np.float64) / 65535
    expected = np.where(linear <= .0031308, linear * 12.92,
                        1.055 * linear ** (1 / 2.4) - .055)[..., ::-1] * 255
    np.testing.assert_allclose(actual, expected, atol=0.51 if not prefer_16bit else 0.0001)


def _write_rgb16_tiff(path, arr, orientation=1):
    from PIL import TiffImagePlugin, TiffTags
    h, w, _ = arr.shape
    prefix = b"II*\x00\x08\x00\x00\x00"
    pixel_bytes = arr.astype("<u2").tobytes()
    ifd = TiffImagePlugin.ImageFileDirectory_v2(prefix)
    for tag, value, tagtype in (
        (256, w, TiffTags.LONG), (257, h, TiffTags.LONG),
        (258, (16, 16, 16), TiffTags.SHORT), (259, 1, TiffTags.SHORT),
        (262, 2, TiffTags.SHORT), (273, 0, TiffTags.LONG),
        (277, 3, TiffTags.SHORT), (278, h, TiffTags.LONG),
        (279, len(pixel_bytes), TiffTags.LONG), (284, 1, TiffTags.SHORT),
        (274, orientation, TiffTags.SHORT),
    ):
        ifd[tag] = value
        ifd.tagtype[tag] = tagtype
    Path(path).write_bytes(prefix + ifd.tobytes(8) + pixel_bytes)


@pytest.mark.parametrize("orientation", [1, 6, 8])
def test_rgb16_tiff_orientation_not_double_applied(tmp_path, orientation):
    arr = np.zeros((3, 4, 3), dtype=np.uint16)
    for y in range(3):
        for x in range(4):
            arr[y, x] = [y * 1000, x * 1000, (y + x) * 500]
    path = tmp_path / "rgb16.tif"
    _write_rgb16_tiff(path, arr, orientation=orientation)
    actual, _ = image_io.imread_engine_with_context(path)
    # Pillow's libtiff RGB16 reduction right-shifts by 8 bits (>>8), which can
    # differ from rint(x/257) by 1 code; orientation correctness (not exact
    # quantization) is what this regression checks.
    single_applied = ImageOps.exif_transpose(
        Image.fromarray((arr >> 8).astype(np.uint8), "RGB").copy()
        if orientation == 1 else
        _tagged(Image.fromarray((arr >> 8).astype(np.uint8), "RGB"), orientation)
    )
    np.testing.assert_allclose(actual[..., ::-1], np.asarray(single_applied), atol=1)


def _tagged(img, orientation):
    exif = Image.Exif()
    exif[274] = orientation
    img.info["exif"] = exif.tobytes()
    return img


def test_rgba_alpha_flattened_in_linear_light_against_white(tmp_path):
    from retouch.white_balance import linear_to_srgb, srgb_to_linear
    arr = np.array([[[255, 0, 0, 0], [0, 255, 0, 128], [0, 0, 255, 255]]], dtype=np.uint8)
    path = tmp_path / "rgba.png"
    Image.fromarray(arr, "RGBA").save(path)
    actual, context = image_io.imread_engine_with_context(path)
    assert context.to_dict()["alpha_mode"] == "flattened-white"

    alpha = arr[..., 3:4].astype(np.float32) / 255.0
    linear = srgb_to_linear(arr[..., :3].astype(np.float32) / 255.0)
    expected_rgb = np.rint(linear_to_srgb(linear * alpha + (1.0 - alpha)) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(actual[..., ::-1], expected_rgb)
    # Fully transparent source pixel must flatten to exact white, never leak
    # its discarded color as a fringe.
    np.testing.assert_array_equal(actual[0, 0], [255, 255, 255])


def test_rgba_fully_opaque_alpha_is_not_flattened(tmp_path):
    arr = np.array([[[10, 20, 30, 255], [40, 50, 60, 255]]], dtype=np.uint8)
    path = tmp_path / "opaque_rgba.png"
    Image.fromarray(arr, "RGBA").save(path)
    actual, context = image_io.imread_engine_with_context(path)
    assert context.to_dict()["alpha_mode"] == "opaque"
    np.testing.assert_array_equal(actual, arr[..., :3][..., ::-1])


def test_tagged_rgba_alpha_survives_icc_transform_then_flattens(tmp_path):
    profile_path = Path("/System/Library/ColorSync/Profiles/Generic Gray Gamma 2.2 Profile.icc")
    if not profile_path.exists():
        pytest.skip("Optional system ICC fixture unavailable")
    # A distinct (non-working) tagged profile forces the ICC-transform branch;
    # this checks alpha is captured before that branch's outputMode="RGB".
    arr = np.array([[[200, 0], [0, 200]]], dtype=np.uint8)
    la = np.dstack((arr, np.array([[0, 255]], dtype=np.uint8)))
    path = tmp_path / "tagged_la.png"
    Image.fromarray(la, "LA").save(path, icc_profile=profile_path.read_bytes())
    actual, context = image_io.imread_engine_with_context(path)
    assert context.to_dict()["alpha_mode"] == "flattened-white"
    np.testing.assert_array_equal(actual[0, 0], [255, 255, 255])


def test_float_tiff_ingest_rejected_instead_of_silently_clipped(tmp_path):
    arr = np.array([[0.1, 0.5], [0.9, 1.0]], dtype=np.float32)
    path = tmp_path / "float32.tif"
    Image.fromarray(arr, mode="F").save(path)
    with pytest.raises(ValueError, match="Float TIFF"):
        image_io.imread_engine_with_context(path)


def test_embedded_profile_conversion_records_intent_and_bpc(tmp_path):
    profile_path = Path("/System/Library/ColorSync/Profiles/Generic Gray Gamma 2.2 Profile.icc")
    if not profile_path.exists():
        pytest.skip("Optional system ICC fixture unavailable")
    source = Image.new("L", (4, 3), 90)
    path = tmp_path / "gray.tif"
    source.save(path, icc_profile=profile_path.read_bytes())
    _, context = image_io.imread_engine_with_context(path)
    d = context.to_dict()
    assert d["transform_intent"] == "perceptual"
    assert d["black_point_compensation"] is False


@pytest.mark.parametrize("bits", [16, 32])
@pytest.mark.parametrize("compress", [False, True])
def test_linear_tiff_keeps_rgb_samples_and_has_linear_profile(tmp_path, bits, compress):
    rgb = np.tile(np.array([[[1., .2, .05], [.18, .18, .18]]], np.float32), (4, 1, 1))
    path = tmp_path / "linear.tif"
    RAWDeveloper().export_linear(rgb, path, bit_depth=bits, compress=compress)
    bgr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    assert bgr.dtype == (np.uint16 if bits == 16 else np.float32)
    decoded = bgr[..., ::-1] / (65535 if bits == 16 else 1)
    np.testing.assert_allclose(decoded, rgb, atol=1 / 65535 if bits == 16 else 1e-7)
    # Pillow can read RGB16 TIFF; the float TIFF profile is checked via its IFD below.
    from retouch.io import get_linear_srgb_icc
    profile = get_linear_srgb_icc()
    assert profile in path.read_bytes()
    cms = ImageCms.ImageCmsProfile(io.BytesIO(profile))
    assert "linear" in ImageCms.getProfileDescription(cms).lower()
    transformed = ImageCms.profileToProfile(
        Image.new("RGB", (1, 1), (46, 46, 46)), cms,
        ImageCms.ImageCmsProfile(io.BytesIO(image_io.get_working_srgb_icc())),
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )
    assert all(abs(c - 118) <= 1 for c in transformed.getpixel((0, 0)))
