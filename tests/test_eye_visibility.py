"""Regression tests for per-eye occlusion gating."""

from types import SimpleNamespace

import numpy as np
import pytest

from retouch.eye_enhancement import EyeEnhancer as EyeEnhancerV0
from retouch.eye_visibility import (
    _EAR_INDICES,
    _MIN_CONTRAST,
    _contrast_for_side,
    _ear_for_side,
    gate_occluded_eye_regions,
)
from retouch.eyes import EyeEnhancer as LegacyEyeEnhancer


def _regions() -> SimpleNamespace:
    regions = SimpleNamespace()
    for side, x1, x2 in (("left", 12, 28), ("right", 36, 52)):
        eye = np.zeros((64, 64), dtype=np.float32)
        iris = np.zeros((64, 64), dtype=np.float32)
        sclera = np.zeros((64, 64), dtype=np.float32)
        eye[20:40, x1:x2] = 1.0
        iris[26:34, x1 + 4:x1 + 12] = 1.0
        sclera[20:40, x1:x2] = 1.0
        setattr(regions, f"{side}_eye", eye)
        setattr(regions, f"{side}_iris", iris)
        setattr(regions, f"{side}_sclera", sclera)
    regions.hair = np.zeros((64, 64), dtype=np.float32)
    return regions


def _landmarks_with_eye_shapes(left_open: bool = True, right_open: bool = True):
    """Build normalized landmarks with independently controlled EAR shapes."""
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    def set_shape(indices, x1, x2, open_eye):
        y_top = 0.45 if open_eye else 0.5
        y_bottom = 0.55 if open_eye else 0.5
        values = (
            (x1, 0.5),
            (x2, 0.5),
            (x1 + (x2 - x1) * 0.25, y_top),
            (x1 + (x2 - x1) * 0.75, y_top),
            (x1 + (x2 - x1) * 0.25, y_bottom),
            (x1 + (x2 - x1) * 0.75, y_bottom),
        )
        for index, (x, y) in zip(indices, values):
            points[index].x = x
            points[index].y = y

    set_shape(_EAR_INDICES["left"], 0.20, 0.40, left_open)
    set_shape(_EAR_INDICES["right"], 0.60, 0.80, right_open)
    return SimpleNamespace(landmark=points)


def test_visible_eyes_are_left_unchanged():
    regions = _regions()
    gated = gate_occluded_eye_regions(regions)

    assert gated is regions
    assert np.array_equal(gated.left_eye, regions.left_eye)
    assert np.array_equal(gated.right_iris, regions.right_iris)


def test_missing_eye_evidence_gates_only_that_iris():
    regions = _regions()
    regions.right_eye.fill(0.0)
    original_left = regions.left_iris.copy()
    original_right = regions.right_iris.copy()

    gated = gate_occluded_eye_regions(regions)

    assert gated is not regions
    assert np.array_equal(regions.right_iris, original_right)
    assert np.array_equal(gated.left_iris, original_left)
    assert np.count_nonzero(gated.right_eye) == 0
    assert np.count_nonzero(gated.right_iris) == 0
    assert np.count_nonzero(gated.right_sclera) == 0


def test_hair_overlap_gates_only_occluded_eye():
    regions = _regions()
    regions.hair[20:40, 36:52] = 1.0

    gated = gate_occluded_eye_regions(regions)

    assert np.count_nonzero(gated.left_iris) > 0
    assert np.count_nonzero(gated.right_iris) == 0


def test_ear_uses_canonical_canthus_and_lid_points():
    landmarks = _landmarks_with_eye_shapes(left_open=True)

    ear = _ear_for_side(landmarks, _EAR_INDICES["left"], 100, 100)

    assert ear == pytest.approx(0.5)


def test_low_ear_gates_only_closed_eye():
    regions = _regions()
    landmarks = _landmarks_with_eye_shapes(left_open=False, right_open=True)

    gated = gate_occluded_eye_regions(
        regions,
        landmarks=landmarks,
        width=64,
        height=64,
    )

    assert np.count_nonzero(gated.left_iris) == 0
    assert np.count_nonzero(gated.right_iris) > 0
    assert np.count_nonzero(regions.left_iris) > 0


def test_landmark_decision_is_reused_by_downstream_reapplication():
    from retouch.parsing import FaceRegions

    template = _regions()
    regions = FaceRegions()
    for field in (
        "left_eye", "right_eye", "left_iris", "right_iris",
        "left_sclera", "right_sclera", "hair",
    ):
        setattr(regions, field, getattr(template, field))
    landmarks = _landmarks_with_eye_shapes(left_open=False, right_open=True)
    first = gate_occluded_eye_regions(
        regions,
        landmarks=landmarks,
        img_bgr=np.full((64, 64, 3), 180, dtype=np.uint8),
    )

    # Downstream enhancers receive a new image array but the same gated
    # regions object. They must retain the original landmark decision instead
    # of recalculating contrast after an earlier stage changed pixels.
    second = gate_occluded_eye_regions(
        first,
        img_bgr=np.zeros((64, 64, 3), dtype=np.uint8),
    )

    assert second is first
    assert np.count_nonzero(second.left_iris) == 0
    assert np.count_nonzero(second.right_iris) > 0


def test_contrast_signal_is_tone_adaptive_and_imported():
    regions = _regions()
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    image[26:34, 16:24] = 30

    contrast = _contrast_for_side(image, regions.left_eye, regions.left_iris)

    assert contrast is not None
    assert contrast > _MIN_CONTRAST


def test_low_contrast_gates_only_that_eye_when_ear_unavailable():
    regions = _regions()
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    image[26:34, 16:24] = 30
    image[26:34, 40:48] = 165

    gated = gate_occluded_eye_regions(regions, img_bgr=image)

    assert np.count_nonzero(gated.left_iris) > 0
    assert np.count_nonzero(gated.right_iris) == 0


def test_legacy_enhancer_does_not_edit_landmark_only_occluded_eye():
    regions = _regions()
    regions.right_eye.fill(0.0)
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    image[26:34, 40:48] = (90, 120, 180)

    enhancer = LegacyEyeEnhancer()
    both = enhancer.enhance(image, regions, strength=80)

    left_only = _regions()
    left_only.right_eye.fill(0.0)
    left_only.right_iris.fill(0.0)
    left_only.right_sclera.fill(0.0)
    expected = enhancer.enhance(image, left_only, strength=80)

    assert np.array_equal(both, expected)


def test_v0_enhancer_does_not_edit_landmark_only_occluded_eye():
    regions = _regions()
    regions.right_eye.fill(0.0)
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    image[26:34, 40:48] = (90, 120, 180)

    enhancer = EyeEnhancerV0()
    both = enhancer.enhance(image, regions, iris_saturate=80)

    left_only = _regions()
    left_only.right_eye.fill(0.0)
    left_only.right_iris.fill(0.0)
    left_only.right_sclera.fill(0.0)
    expected = enhancer.enhance(image, left_only, iris_saturate=80)

    assert np.array_equal(both, expected)
