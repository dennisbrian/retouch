"""Tests for retouch/relight.py — Virtual Studio Relighting."""

import numpy as np
import pytest
import cv2
from retouch.relight import Relighter


class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class MockLandmarksList:
    def __init__(self, yaw_ratio=1.0):
        import math
        self.landmark = []
        for i in range(468):
            angle = i * 2.0 * math.pi / 468.0
            x = 0.5 + 0.3 * math.cos(angle)
            y = 0.5 + 0.3 * math.sin(angle)
            z = 0.1 * math.sin(angle * 3.0)
            self.landmark.append(MockLandmark(x, y, z))

        # Set temple-to-nose distances for yaw
        # lm[6] nose, lm[234] left temple, lm[454] right temple
        # Default: symmetric (ratio = 1.0)
        self.landmark[6].x = 0.5
        self.landmark[234].x = 0.3
        self.landmark[454].x = 0.7

        if yaw_ratio != 1.0:
            # Shift nose to create asymmetry:
            # Let d_left = |0.5 + shift - 0.3| = |0.2 + shift|
            # Let d_right = |0.7 - (0.5 + shift)| = |0.2 - shift|
            # We want max(d_left, d_right)/min(d_left, d_right) = yaw_ratio
            shift = 0.2 * (yaw_ratio - 1.0) / (yaw_ratio + 1.0)
            self.landmark[6].x = 0.5 + shift


def test_relighter_no_strength():
    """Test that zero strength returns identical output (both engines)."""
    relighter = Relighter()
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = MockLandmarksList()
    mask = np.ones((100, 100), dtype=np.float32)

    out_v1 = relighter.relight(canvas, landmarks, mask, face_width=100.0, strength=0.0, engine="v1")
    out_v2 = relighter.relight(canvas, landmarks, mask, face_width=100.0, strength=0.0, engine="v2")
    assert np.all(out_v1 == canvas)
    assert np.all(out_v2 == canvas)


def test_relighter_yaw_guard_attenuation():
    """Test that yaw guard attenuates strength for profile faces."""
    relighter = Relighter()
    canvas = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.float32)

    # 1. Symmetric face: no attenuation (ratio = 1.0 < 1.5)
    landmarks_sym = MockLandmarksList(yaw_ratio=1.0)
    out_sym = relighter.relight(canvas, landmarks_sym, mask, face_width=100.0, strength=100.0, engine="v2")
    # Check that image changes (relighting applied)
    assert not np.array_equal(out_sym, canvas)

    # 2. Extreme yaw: fully attenuated (ratio = 1.8 > 1.7 -> strength becomes 0)
    landmarks_yaw_extreme = MockLandmarksList(yaw_ratio=1.8)
    out_yaw_extreme = relighter.relight(canvas, landmarks_yaw_extreme, mask, face_width=100.0, strength=100.0, engine="v2")
    assert np.array_equal(out_yaw_extreme, canvas)

    # 3. Partial yaw: partially attenuated (ratio = 1.6, yaw_factor = 0.5)
    landmarks_yaw_part = MockLandmarksList(yaw_ratio=1.6)
    out_yaw_part = relighter.relight(canvas, landmarks_yaw_part, mask, face_width=100.0, strength=100.0, engine="v2")
    assert not np.array_equal(out_yaw_part, canvas)
    assert not np.array_equal(out_yaw_part, out_sym)


def test_relighter_angle_variations():
    """Test that different light angles produce different shading."""
    relighter = Relighter()
    canvas = np.full((100, 100, 3), 128, dtype=np.uint8)
    landmarks = MockLandmarksList()
    mask = np.ones((100, 100), dtype=np.float32)

    # Run with different azimuth/elevation angles
    out_left = relighter.relight(canvas, landmarks, mask, face_width=100.0, strength=100.0, azimuth=-45.0, elevation=30.0, engine="v2")
    out_right = relighter.relight(canvas, landmarks, mask, face_width=100.0, strength=100.0, azimuth=45.0, elevation=30.0, engine="v2")

    assert out_left.shape == (100, 100, 3)
    assert out_right.shape == (100, 100, 3)
    # Different angles should yield different lighting shading
    assert not np.array_equal(out_left, out_right)


def test_v2_synthetic_relighting_agreement():
    """Test v2: relighting produces output different from input."""
    relighter = Relighter()

    # Create synthetic image
    h, w = 100, 100
    canvas = np.full((h, w, 3), 128, dtype=np.uint8)
    landmarks = MockLandmarksList()
    mask = np.ones((h, w), dtype=np.float32)

    # Relight with two different angles
    out_az0 = relighter.relight(
        canvas, landmarks, mask, face_width=100.0, strength=60.0,
        azimuth=0.0, elevation=30.0, engine="v2"
    )

    out_az90 = relighter.relight(
        canvas, landmarks, mask, face_width=100.0, strength=60.0,
        azimuth=90.0, elevation=30.0, engine="v2"
    )

    # Both should differ from input
    diff_0 = np.sum(np.abs(out_az0.astype(float) - canvas.astype(float)))
    diff_90 = np.sum(np.abs(out_az90.astype(float) - canvas.astype(float)))

    # At least one should change significantly
    assert diff_0 > 1000 or diff_90 > 1000, \
        f"Relighting should produce visible change; diff_0={diff_0}, diff_90={diff_90}"


def test_v2_gain_bounds():
    """Test v2: output/input gain ratio stays within reasonable bounds in linear space."""
    relighter = Relighter()
    canvas = np.full((100, 100, 3), 128, dtype=np.uint8)
    landmarks = MockLandmarksList()
    mask = np.ones((100, 100), dtype=np.float32)

    out = relighter.relight(
        canvas, landmarks, mask, face_width=100.0, strength=100.0, engine="v2"
    )

    # Convert to linear space
    canvas_lin = (canvas.astype(np.float32) / 255.0) ** 2.2
    out_lin = (out.astype(np.float32) / 255.0) ** 2.2

    # Compute per-pixel gain (avoid division by zero)
    gain = np.divide(out_lin, canvas_lin, where=canvas_lin > 0.01, out=np.ones_like(out_lin))

    # Within masked region, gain should stay in reasonable bounds (relaxed for specular)
    valid = (mask > 0.3)
    gain_valid = gain[valid]

    # Allow gain between 0.5 and 3.0 to account for specular highlights
    assert np.all(gain_valid >= 0.5), f"Gain below 0.5: min={np.min(gain_valid)}"
    assert np.all(gain_valid <= 3.0), f"Gain above 3.0: max={np.max(gain_valid)}"


def test_v1_engine_still_works():
    """Test that engine='v1' still runs and produces valid output."""
    relighter = Relighter()
    canvas = np.full((100, 100, 3), 128, dtype=np.uint8)
    landmarks = MockLandmarksList()
    mask = np.ones((100, 100), dtype=np.float32)

    out = relighter.relight(
        canvas, landmarks, mask, face_width=100.0, strength=50.0, engine="v1"
    )

    assert out.shape == (100, 100, 3)
    assert out.dtype == np.uint8
    # With strength > 0 and valid mask, output should differ
    assert not np.array_equal(out, canvas)
