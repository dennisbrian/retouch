"""Regression coverage for source-adaptive eye artifact backoff."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from retouch.eye_artifact_safety import (
    _MIN_CHROMA_SCALE,
    assess_eye_artifact_scales,
    resolve_eye_scale,
)
from retouch.eye_enhancement import EyeEnhancer as EyeEnhancerV0
from retouch.eyes import EyeEnhancer as LegacyEyeEnhancer


def _circle(size: int, center: tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=np.float32)
    cv2.circle(mask, center, radius, 1.0, -1)
    return mask


def _regions(size: int = 128, left_radius: int = 24, right_radius: int = 24):
    left_center = (40, 64)
    right_center = (88, 64)
    left_iris = _circle(size, left_center, left_radius)
    right_iris = _circle(size, right_center, right_radius)
    left_eye = _circle(size, left_center, left_radius + 7)
    right_eye = _circle(size, right_center, right_radius + 7)
    return SimpleNamespace(
        left_eye=left_eye,
        right_eye=right_eye,
        left_iris=left_iris,
        right_iris=right_iris,
        left_sclera=np.clip(left_eye - left_iris, 0.0, 1.0),
        right_sclera=np.clip(right_eye - right_iris, 0.0, 1.0),
        hair=np.zeros((size, size), dtype=np.float32),
    )


def _image_with_iris_saturation(
    regions,
    left_saturation: int,
    right_saturation: int,
) -> np.ndarray:
    hsv = np.zeros((*regions.left_iris.shape, 3), dtype=np.uint8)
    hsv[:, :, 2] = 150
    # Keep iris luminance below the surrounding eye so the independent
    # visibility gate reads both synthetic eyes as open, not low-contrast.
    hsv[regions.left_iris > 0.25] = (70, left_saturation, 55)
    hsv[regions.right_iris > 0.25] = (105, right_saturation, 55)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def test_large_low_chroma_irises_keep_full_strength():
    regions = _regions()
    image = _image_with_iris_saturation(regions, 30, 30)

    decisions = assess_eye_artifact_scales(image, regions)

    assert resolve_eye_scale(decisions, "left") == pytest.approx(1.0)
    assert resolve_eye_scale(decisions, "right") == pytest.approx(1.0)
    assert decisions["left"]["reason"] == "full_headroom"


def test_small_native_iris_is_smoothly_damped():
    regions = _regions(left_radius=8, right_radius=24)
    image = _image_with_iris_saturation(regions, 30, 30)

    decisions = assess_eye_artifact_scales(image, regions)

    assert 0.0 < resolve_eye_scale(decisions, "left") < 0.2
    assert resolve_eye_scale(decisions, "right") == pytest.approx(1.0)
    assert decisions["left"]["reason"] == "small_iris"


def test_saturated_source_iris_preserves_chroma_headroom():
    regions = _regions()
    image = _image_with_iris_saturation(regions, 210, 30)

    decisions = assess_eye_artifact_scales(image, regions)

    assert resolve_eye_scale(decisions, "left") == pytest.approx(_MIN_CHROMA_SCALE)
    assert resolve_eye_scale(decisions, "right") == pytest.approx(1.0)
    assert "limited_chroma_headroom" in decisions["left"]["reason"]


def test_missing_or_malformed_evidence_fails_open():
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    regions = SimpleNamespace(
        left_iris=None,
        right_iris=np.full((64, 64), np.nan, dtype=np.float32),
    )

    decisions = assess_eye_artifact_scales(image, regions)

    assert resolve_eye_scale(decisions, "left") == 1.0
    assert resolve_eye_scale(decisions, "right") == 1.0


def test_assessment_does_not_mutate_image_or_masks():
    regions = _regions(left_radius=8)
    image = _image_with_iris_saturation(regions, 210, 30)
    image_before = image.copy()
    iris_before = regions.left_iris.copy()

    assess_eye_artifact_scales(image, regions)

    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(regions.left_iris, iris_before)


def test_legacy_eye_stack_obeys_per_eye_scale():
    regions = _regions()
    image = _image_with_iris_saturation(regions, 210, 30)
    decisions = assess_eye_artifact_scales(image, regions)
    enhancer = LegacyEyeEnhancer()

    full = enhancer.enhance(
        image,
        regions,
        strength=70,
        catchlight_strength=0,
    )
    guarded = enhancer.enhance(
        image,
        regions,
        strength=70,
        catchlight_strength=0,
        eye_scales=decisions,
    )

    left = regions.left_iris > 0.25
    full_delta = float(np.mean(np.abs(full[left].astype(np.float32) - image[left])))
    guarded_delta = float(np.mean(np.abs(guarded[left].astype(np.float32) - image[left])))
    assert guarded_delta < full_delta * 0.7


def test_v0_eye_stack_obeys_per_eye_scale():
    regions = _regions()
    image = _image_with_iris_saturation(regions, 210, 30)
    decisions = assess_eye_artifact_scales(image, regions)
    enhancer = EyeEnhancerV0()

    full = enhancer.enhance(
        image,
        regions,
        sclera_brighten=50,
        iris_saturate=80,
        iris_brightness=50,
    )
    guarded = enhancer.enhance(
        image,
        regions,
        sclera_brighten=50,
        iris_saturate=80,
        iris_brightness=50,
        eye_scales=decisions,
    )

    left = regions.left_eye > 0.25
    full_delta = float(np.mean(np.abs(full[left].astype(np.float32) - image[left])))
    guarded_delta = float(np.mean(np.abs(guarded[left].astype(np.float32) - image[left])))
    assert guarded_delta < full_delta * 0.7


def test_synthetic_catchlight_is_opt_in(monkeypatch):
    regions = _regions(left_radius=16, right_radius=0)
    regions.right_eye.fill(0.0)
    regions.right_iris.fill(0.0)
    regions.right_sclera.fill(0.0)
    image = np.full((128, 128, 3), 50, dtype=np.uint8)
    enhancer = LegacyEyeEnhancer()

    # Isolate the catchlight branch from iris sculpting so a uniform source
    # has no naturally detected highlight.
    monkeypatch.setattr(enhancer, "_sculpt_iris", lambda img, _mask, _strength: img)

    default = enhancer.enhance(
        image,
        regions,
        strength=1,
        catchlight_strength=100,
    )
    explicit_off = enhancer.enhance(
        image,
        regions,
        strength=1,
        catchlight_strength=100,
        synthetic_catchlight=False,
    )
    explicit_on = enhancer.enhance(
        image,
        regions,
        strength=1,
        catchlight_strength=100,
        synthetic_catchlight=True,
    )

    np.testing.assert_array_equal(default, explicit_off)
    assert int(explicit_on.max()) > int(default.max())
