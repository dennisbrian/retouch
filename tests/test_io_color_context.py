"""Focused tests for the explicit ingest/delivery color contract."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from retouch.color_context import SOURCE_ASSUMED_SRGB, SOURCE_EMBEDDED_ICC
from retouch.io import (
    _HAS_IMAGECMS,
    color_context_for_path,
    convert_image_colorspace,
    get_working_srgb_icc,
    imread_exif,
    imread_exif_with_context,
    imread_engine_with_context,
    prepare_color_managed_export,
    read_icc_profile,
    write_image_with_color_context,
)


ADOBE_RGB_PROFILE = Path("/System/Library/ColorSync/Profiles/AdobeRGB1998.icc")


def _adobe_rgb_icc() -> bytes:
    if not _HAS_IMAGECMS or not ADOBE_RGB_PROFILE.is_file():
        pytest.skip("Adobe RGB (1998) ICC profile is not available")
    return ADOBE_RGB_PROFILE.read_bytes()


def _synth_patch() -> np.ndarray:
    """Return RGB patch columns spanning dark, skin-like, and saturated values."""
    colors = np.array(
        [
            [8, 40, 100],
            [32, 96, 160],
            [64, 128, 192],
            [96, 160, 224],
            [180, 90, 30],
            [245, 180, 60],
        ],
        dtype=np.uint8,
    )
    return np.tile(colors[None, :, :], (12, 1, 1))


def test_embedded_adobergb_patch_is_converted_to_engine_srgb(tmp_path: Path) -> None:
    """Tagged Adobe RGB samples must not be interpreted as raw sRGB values."""
    adobe_icc = _adobe_rgb_icc()
    working_icc = get_working_srgb_icc()
    assert working_icc is not None

    source_rgb = _synth_patch()
    source_path = tmp_path / "adobe-rgb-patch.png"
    Image.fromarray(source_rgb, mode="RGB").save(source_path, icc_profile=adobe_icc)

    loaded_bgr, context = imread_exif_with_context(source_path)
    expected_bgr = np.rint(
        convert_image_colorspace(
            cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR),
            adobe_icc,
            working_icc,
        )
        * 255.0
    ).astype(np.uint8)
    naive_srgb_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)

    np.testing.assert_array_equal(loaded_bgr, expected_bgr)
    np.testing.assert_array_equal(imread_exif(source_path), expected_bgr)
    engine_bgr, engine_context = imread_engine_with_context(source_path)
    np.testing.assert_array_equal(engine_bgr, expected_bgr)
    assert engine_context.source_profile == adobe_icc
    assert np.abs(loaded_bgr.astype(np.int16) - naive_srgb_bgr.astype(np.int16)).mean() > 5.0
    assert context.source_kind == SOURCE_EMBEDDED_ICC
    assert context.is_tagged is True
    assert context.assumed_srgb is False
    assert context.conversion_applied is True
    assert context.source_profile == adobe_icc
    assert context.to_dict()["source_profile_name"] == "Adobe RGB (1998)"


def test_untagged_input_is_explicitly_assumed_srgb(tmp_path: Path) -> None:
    source_rgb = _synth_patch()
    source_path = tmp_path / "untagged-patch.png"
    Image.fromarray(source_rgb, mode="RGB").save(source_path)

    loaded_bgr, context = imread_exif_with_context(source_path)

    np.testing.assert_array_equal(loaded_bgr, cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR))
    assert context.source_kind == SOURCE_ASSUMED_SRGB
    assert context.assumed_srgb is True
    assert context.source_profile is None
    assert context.conversion_applied is False
    assert context.to_dict()["source_profile_sha256"] is None

    prepared, output_profile = prepare_color_managed_export(loaded_bgr, context)
    np.testing.assert_array_equal(prepared, loaded_bgr)
    assert output_profile == context.working_profile


def test_legacy_array_exporters_can_recover_the_ingest_context(tmp_path: Path) -> None:
    source_path = tmp_path / "untagged-source.png"
    Image.fromarray(_synth_patch(), mode="RGB").save(source_path)

    context = color_context_for_path(source_path)

    assert context.source_kind == SOURCE_ASSUMED_SRGB
    assert context.source_profile is None
    assert context.working_profile == get_working_srgb_icc()


def test_untagged_ingest_is_byte_identical_to_plain_read(tmp_path: Path) -> None:
    """Untagged pixels must traverse the exact legacy path: byte-identical read."""
    source_rgb = _synth_patch()
    source_path = tmp_path / "untagged-legacy.png"
    Image.fromarray(source_rgb, mode="RGB").save(source_path)

    loaded_bgr, context = imread_exif_with_context(source_path)

    assert context.assumed_srgb is True
    assert context.conversion_applied is False
    with Image.open(source_path) as plain:
        np.testing.assert_array_equal(
            loaded_bgr,
            cv2.cvtColor(np.array(plain.convert("RGB")), cv2.COLOR_RGB2BGR),
        )
        np.testing.assert_array_equal(imread_exif(source_path), loaded_bgr)


def test_tagged_ingest_shifts_pixels_that_differ_between_profiles(tmp_path: Path) -> None:
    """Ingest must transform pixels, not merely flag conversion_applied.

    Pure red is gamut-stable under Adobe RGB -> sRGB, but the mid-saturation
    blue patch [64, 128, 192] has its red channel pulled hard toward zero.
    """
    adobe_icc = _adobe_rgb_icc()
    source_rgb = _synth_patch()
    source_path = tmp_path / "adobe-shift.png"
    Image.fromarray(source_rgb, mode="RGB").save(source_path, icc_profile=adobe_icc)

    loaded_bgr, context = imread_exif_with_context(source_path)
    naive_bgr = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2BGR)

    assert context.conversion_applied is True
    # Patch column 2 (BGR): raw-tagged red channel is 64; converted must drop
    # far below it, proving a real transform rather than a pass-through flag.
    assert int(loaded_bgr[0, 2, 2]) < 20
    assert int(loaded_bgr[0, 2, 2]) < int(naive_bgr[0, 2, 2]) - 40
    # Green mid value survives approximately (128 -> ~129).
    assert abs(int(loaded_bgr[0, 2, 1]) - 128) <= 2
    np.testing.assert_array_equal(loaded_bgr, imread_exif(source_path))


def test_export_uses_working_profile_by_default_and_restores_source_only_explicitly(
    tmp_path: Path,
) -> None:
    adobe_icc = _adobe_rgb_icc()
    working_icc = get_working_srgb_icc()
    assert working_icc is not None

    source_rgb = _synth_patch()
    source_path = tmp_path / "source-adobe.png"
    Image.fromarray(source_rgb, mode="RGB").save(source_path, icc_profile=adobe_icc)
    working_bgr, context = imread_exif_with_context(source_path)

    default_path = tmp_path / "default-working-profile.png"
    write_image_with_color_context(default_path, working_bgr, context)
    assert read_icc_profile(default_path) == working_icc
    assert read_icc_profile(default_path) != adobe_icc

    preserved_path = tmp_path / "explicit-source-profile.png"
    write_image_with_color_context(
        preserved_path,
        working_bgr,
        context,
        preserve_source_profile=True,
    )
    assert read_icc_profile(preserved_path) == adobe_icc

    # The explicit source-profile export is converted back to Adobe RGB, but
    # decoding it through the same ingest contract returns sRGB working pixels.
    roundtrip_bgr, roundtrip_context = imread_exif_with_context(preserved_path)
    np.testing.assert_allclose(roundtrip_bgr, working_bgr, atol=1)
    assert roundtrip_context.source_profile == adobe_icc
    assert roundtrip_context.conversion_applied is True
