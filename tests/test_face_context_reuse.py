"""Cached parsed regions are reused only when built from the same parser input.

Regression: the GUI preview cache key omits reshape, preprocessing and
``mask_feather_mode``. On a warm cache the small/fast path reused regions
parsed from the pre-reshape face, and ignored a feather-mode change entirely
(DSCF4463 at 1600px: slimming 0→60 warm vs cold differed on 2,616 px, max 45;
gaussian→guided warm output equalled the old mode). See
docs/plans/RESEARCH_RETOUCH_TARGET_AND_PREVIEW_PARITY_2026_09_23.md §7.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from retouch.detection import FaceContext, FaceData
from retouch.engine import RetouchEngine

match = RetouchEngine._cached_regions_match


def _fc(face_image=None, mode=None):
    return FaceContext(
        face_data=FaceData(landmarks=None, bbox=(0, 0, 4, 4), ied=2.0),
        regions=None,
        face_image=face_image,
        mask_feather_mode=mode,
    )


def _ctx(mode="gaussian"):
    return SimpleNamespace(mask_feather_mode=mode)


CROP = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)


def test_identical_crop_and_mode_reuses():
    assert match([_fc(CROP.copy(), "gaussian")], [CROP], _ctx("gaussian"))


def test_changed_parser_pixels_rejects_reuse():
    reshaped = CROP.copy()
    reshaped[1, 1, 0] += 1
    assert not match([_fc(CROP.copy(), "gaussian")], [reshaped], _ctx("gaussian"))


def test_changed_crop_shape_rejects_reuse():
    assert not match([_fc(CROP.copy(), "gaussian")], [CROP[:3]], _ctx("gaussian"))


def test_changed_feather_mode_rejects_reuse():
    assert not match([_fc(CROP.copy(), "gaussian")], [CROP], _ctx("guided"))


def test_face_count_mismatch_rejects_reuse():
    assert not match([_fc(CROP.copy())], [CROP, CROP], _ctx())


def test_legacy_context_without_parser_evidence_is_trusted():
    # e.g. tests/golden_face_fixture.py builds contexts without face_image.
    assert match([_fc(None, None)], [CROP], _ctx("guided"))
