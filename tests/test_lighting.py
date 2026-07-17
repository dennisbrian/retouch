"""Tests for the shared, non-generative key-light estimator."""

from __future__ import annotations

from types import SimpleNamespace

import cv2
import numpy as np

from retouch.lighting import estimate_light_direction


def _circle_mask(shape, center, radius):
    mask = np.zeros(shape, dtype=np.float32)
    cv2.circle(mask, center, radius, 1.0, -1)
    return mask


def test_prefers_agreeing_paired_catchlights():
    h, w = 120, 180
    img = np.full((h, w, 3), 45, dtype=np.uint8)
    left = _circle_mask((h, w), (58, 62), 16)
    right = _circle_mask((h, w), (122, 62), 16)
    # Both highlights are up and right of their iris centers.
    cv2.circle(img, (65, 57), 3, (245, 245, 245), -1)
    cv2.circle(img, (129, 57), 3, (245, 245, 245), -1)
    regions = SimpleNamespace(left_iris=left, right_iris=right, skin=None)

    result = estimate_light_direction(img, regions)

    assert result.source == "catchlights"
    assert result.confidence > 0.35
    assert result.direction[0] > 0.6
    assert result.direction[1] < -0.3


def test_falls_back_to_low_frequency_skin_shading():
    h, w = 140, 200
    ramp = np.linspace(35, 215, w, dtype=np.uint8)[None, :].repeat(h, axis=0)
    img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    regions = SimpleNamespace(
        left_iris=None,
        right_iris=None,
        skin=_circle_mask((h, w), (w // 2, h // 2), 60),
    )

    result = estimate_light_direction(img, regions, face_width=120.0)

    assert result.source == "shading"
    assert result.confidence >= 0.05
    assert result.direction[0] > 0.98
    assert abs(result.direction[1]) < 0.05


def test_uniform_frame_reports_unknown_instead_of_inventing_light():
    img = np.full((96, 96, 3), 127, dtype=np.uint8)

    result = estimate_light_direction(img)

    assert result.source == "unknown"
    assert result.confidence == 0.0
    assert result.direction == (0.0, 0.0)


def test_float_input_matches_uint8_direction():
    h, w = 96, 128
    ramp = np.linspace(25, 225, w, dtype=np.uint8)[None, :].repeat(h, axis=0)
    img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
    regions = SimpleNamespace(left_iris=None, right_iris=None, skin=np.ones((h, w), dtype=np.float32))

    u8 = estimate_light_direction(img, regions)
    f32 = estimate_light_direction(img.astype(np.float32) / 255.0, regions)

    assert u8.source == f32.source == "shading"
    assert np.allclose(u8.direction, f32.direction, atol=1e-4)
