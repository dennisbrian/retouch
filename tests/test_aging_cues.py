"""Tests for the measurement-only P8 R1 cue readout."""

import json
from types import SimpleNamespace

import numpy as np

from retouch.aging_cues import measure_p8_cues


def _regions(shape=(160, 160)):
    h, w = shape
    skin = np.ones(shape, np.float32)
    lips = np.zeros(shape, np.float32)
    lips[104:124, 54:106] = 1.0
    left_eye = np.zeros(shape, np.float32)
    left_eye[52:66, 38:62] = 1.0
    right_eye = np.zeros(shape, np.float32)
    right_eye[52:66, 98:122] = 1.0
    left_brow = np.zeros(shape, np.float32)
    left_brow[38:46, 34:64] = 1.0
    right_brow = np.zeros(shape, np.float32)
    right_brow[38:46, 96:126] = 1.0
    return SimpleNamespace(
        skin=skin,
        lips=lips,
        left_eye=left_eye,
        right_eye=right_eye,
        left_eyebrow=left_brow,
        right_eyebrow=right_brow,
    )


def _face(regions, *, dark_lips=False):
    image = np.full((160, 160, 3), (110, 145, 185), dtype=np.uint8)
    if dark_lips:
        image[regions.lips > 0.5] = (70, 55, 105)
    return image


def test_readout_preserves_signed_feature_components_and_units():
    regions = _regions()
    readout = measure_p8_cues(_face(regions, dark_lips=True), regions)

    lips = readout.feature_contrast["lips"]
    assert lips.support.status == "available"
    assert lips.values is not None
    assert len(lips.values["lab_delta"]) == 3
    assert lips.values["lab_delta"][0] < 0.0
    assert "LAB" in lips.units["lab_delta"]
    assert "feature_contrast.lips" in readout.available_descriptors
    assert "age" not in readout.to_dict()


def test_missing_skin_is_unavailable_not_zero_measurement():
    regions = _regions()
    regions.skin = None

    readout = measure_p8_cues(_face(_regions()), regions)

    assert readout.chroma_variation.support.status == "missing_support"
    assert readout.chroma_variation.values is None
    assert all(item.support.status == "missing_support" for item in readout.feature_contrast.values())
    assert all(item.values is None for item in readout.feature_contrast.values())


def test_missing_feature_region_is_explicit_but_other_features_remain_available():
    regions = _regions()
    regions.lips = None

    readout = measure_p8_cues(_face(_regions()), regions)

    assert readout.feature_contrast["lips"].support.status == "missing_support"
    assert readout.feature_contrast["lips"].values is None
    assert readout.feature_contrast["eyes"].support.status == "available"
    assert readout.chroma_variation.support.status == "available"


def test_parser_style_skin_exclusion_does_not_hide_feature_pixels():
    regions = _regions()
    for name in ("lips", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow"):
        regions.skin[getattr(regions, name) > 0.5] = 0.0

    readout = measure_p8_cues(_face(regions, dark_lips=True), regions)

    assert readout.feature_contrast["lips"].support.status == "available"
    assert readout.feature_contrast["lips"].values is not None
    assert readout.feature_contrast["eyes"].support.status == "available"


def test_small_support_is_insufficient_and_does_not_become_zero():
    regions = _regions()
    regions.skin = np.zeros_like(regions.skin)
    regions.skin[70:75, 70:75] = 1.0

    readout = measure_p8_cues(_face(_regions()), regions)

    assert readout.chroma_variation.support.status == "insufficient_support"
    assert readout.chroma_variation.values is None
    assert readout.chroma_variation.support.pixel_count == 25


def test_float_mask_overshoot_is_clipped_not_divided_as_8_bit():
    regions = _regions()
    for name in (
        "skin", "lips", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow",
    ):
        setattr(regions, name, getattr(regions, name) * 1.0001)

    readout = measure_p8_cues(_face(regions, dark_lips=True), regions)

    assert readout.chroma_variation.support.status == "available"
    assert readout.feature_contrast["lips"].support.status == "available"


def test_shape_mismatch_is_invalid_support_and_json_is_strictly_serializable():
    regions = _regions()
    regions.lips = np.ones((8, 8), dtype=np.float32)

    readout = measure_p8_cues(_face(_regions()), regions)
    payload = json.loads(readout.to_json(indent=2))

    assert readout.feature_contrast["lips"].support.status == "invalid_support"
    assert readout.feature_contrast["lips"].values is None
    assert payload["descriptor_revision"] == "p8-r1-appearance-v1"
    assert payload["chroma_variation"]["values"] is not None


def test_invalid_image_and_min_pixels_fail_at_api_boundary():
    regions = _regions()
    with np.testing.assert_raises(ValueError):
        measure_p8_cues(np.zeros((10, 10), dtype=np.uint8), regions)
    with np.testing.assert_raises(ValueError):
        measure_p8_cues(_face(regions), regions, min_pixels=0)
