"""Focused tests for pure native-resolution inspection contracts."""

from __future__ import annotations

import numpy as np
import pytest

from retouch.gui_inspection import (
    MODE_100_PERCENT,
    MODE_FACE,
    MODE_FIT,
    MODE_ROI,
    RevisionMismatchError,
    StaleInspectionError,
    build_download_disabled_preview_state,
    build_inspection_contract,
    clamp_native_crop,
    compare_render_revisions,
    crop_from_inspection_contract,
    crop_native_image,
    select_face_crop,
    select_roi_crop,
    validate_inspection_contract,
)


def _contract(mode, *, render_revision=12, current_revision=12, **kwargs):
    return build_inspection_contract(
        mode=mode,
        native_size=(100, 80),
        render_revision=render_revision,
        current_revision=current_revision,
        **kwargs,
    )


def test_revision_mismatch_and_stale_render_are_rejected_fail_closed():
    mismatch = compare_render_revisions(11, 12, 12)
    assert mismatch["valid"] is False
    assert mismatch["stale"] is True
    assert mismatch["status"] == "revision_mismatch"
    with pytest.raises(RevisionMismatchError):
        _contract(MODE_FIT, render_revision=11, current_revision=12)

    stale = compare_render_revisions(11, 11, 12)
    assert stale["valid"] is False
    assert stale["status"] == "stale"
    with pytest.raises(StaleInspectionError):
        _contract(
            MODE_FIT,
            render_revision=11,
            current_revision=12,
            requested_revision=11,
        )


def test_native_crop_bounds_clamp_and_never_resize():
    bounds = clamp_native_crop((-10, -5, 50, 30), (100, 80))
    assert bounds.to_tuple() == (0, 0, 40, 25)
    assert bounds.to_xyxy() == (0, 0, 40, 25)

    image = np.arange(80 * 100, dtype=np.uint16).reshape(80, 100)
    cropped = crop_native_image(image, bounds)
    assert cropped.shape == (25, 40)
    assert cropped.dtype == image.dtype
    assert cropped[0, 0] == image[0, 0]
    assert cropped[-1, -1] == image[24, 39]

    fit = _contract(MODE_FIT)
    assert fit["crop"]["x"] == 0
    assert fit["crop"]["y1"] == 80
    assert fit["native"]["crop_is_resized"] is False
    assert fit["display"]["scale"] == "fit"

    one_to_one = _contract("100%")
    assert one_to_one["mode"] == MODE_100_PERCENT
    assert one_to_one["crop"]["width"] == 100
    assert one_to_one["crop"]["height"] == 80
    assert one_to_one["display"]["scale_factor"] == 1.0


def test_face_selection_uses_bbox_and_clamps_padding_to_native_bounds():
    faces = [
        {"bbox": (90, 70, 30, 30)},
        {"face_data": {"bbox": (10, 20, 40, 50)}},
    ]
    selected = select_face_crop(faces, 0, (100, 80), padding=10)
    assert selected.source_bbox.to_tuple() == (90, 70, 30, 30)
    assert selected.crop.to_tuple() == (80, 60, 20, 20)

    contract = _contract(
        MODE_FACE,
        faces=faces,
        face_index=1,
        face_padding=5,
    )
    assert contract["selection"]["kind"] == "face"
    assert contract["selection"]["index"] == 1
    assert contract["crop"]["x"] == 5
    assert contract["crop"]["y"] == 15
    assert contract["crop"]["width"] == 50
    assert contract["crop"]["height"] == 60


def test_roi_selection_accepts_native_pixel_mapping_and_clamps():
    roi = select_roi_crop(
        {"left": 85.2, "top": 70.4, "right": 120.1, "bottom": 95.9},
        (100, 80),
    )
    assert roi.to_tuple() == (85, 70, 15, 10)

    contract = _contract(MODE_ROI, roi=(20, 10, 40, 30))
    assert contract["selection"]["kind"] == "roi"
    assert contract["crop"]["x1"] == 60
    assert contract["crop"]["y1"] == 40
    assert contract["crop"]["coordinate_space"] == "native_pixels"


def test_preview_state_is_explicitly_non_downloadable_and_validated():
    state = build_download_disabled_preview_state(7, mode=MODE_ROI)
    assert state["preview_only"] is True
    assert state["download_enabled"] is False
    assert state["download"]["allowed"] is False
    assert state["download"]["available"] is False
    assert state["download"]["href"] is None

    contract = _contract(MODE_FIT)
    assert contract["preview"]["download_enabled"] is False
    assert validate_inspection_contract(contract, 12)["valid"] is True

    tampered = dict(contract)
    tampered["preview"] = dict(contract["preview"])
    tampered["preview"]["download_enabled"] = True
    report = validate_inspection_contract(tampered, 12)
    assert report["valid"] is False
    assert "download_disabled_preview" in report["errors"]


def test_contract_crop_rechecks_revision_before_returning_native_pixels():
    contract = _contract(MODE_ROI, roi=(10, 10, 20, 15))
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    cropped = crop_from_inspection_contract(image, contract, 12)
    assert cropped.shape == (15, 20, 3)

    with pytest.raises(StaleInspectionError):
        crop_from_inspection_contract(image, contract, 13)
