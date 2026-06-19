"""Tests for retouch/relight.py — Virtual Studio Relighting."""

import numpy as np
import pytest
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
    relighter = Relighter()
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    landmarks = MockLandmarksList()
    mask = np.ones((100, 100), dtype=np.float32)
    
    out = relighter.relight(canvas, landmarks, mask, face_width=100.0, strength=0.0)
    assert np.all(out == canvas)


def test_relighter_yaw_guard_attenuation():
    relighter = Relighter()
    canvas = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.float32)
    
    # 1. Symmetric face: no attenuation (ratio = 1.0 < 1.5)
    landmarks_sym = MockLandmarksList(yaw_ratio=1.0)
    out_sym = relighter.relight(canvas, landmarks_sym, mask, face_width=100.0, strength=100.0)
    # Check that image changes (relighting applied)
    assert not np.array_equal(out_sym, canvas)
    
    # 2. Extreme yaw: fully attenuated (ratio = 1.8 > 1.7 -> strength becomes 0)
    landmarks_yaw_extreme = MockLandmarksList(yaw_ratio=1.8)
    out_yaw_extreme = relighter.relight(canvas, landmarks_yaw_extreme, mask, face_width=100.0, strength=100.0)
    assert np.array_equal(out_yaw_extreme, canvas)
    
    # 3. Partial yaw: partially attenuated (ratio = 1.6, yaw_factor = 0.5)
    landmarks_yaw_part = MockLandmarksList(yaw_ratio=1.6)
    out_yaw_part = relighter.relight(canvas, landmarks_yaw_part, mask, face_width=100.0, strength=100.0)
    assert not np.array_equal(out_yaw_part, canvas)
    assert not np.array_equal(out_yaw_part, out_sym)


def test_relighter_angle_variations():
    relighter = Relighter()
    canvas = np.full((100, 100, 3), 128, dtype=np.uint8)
    landmarks = MockLandmarksList()
    mask = np.ones((100, 100), dtype=np.float32)
    
    # Run with different azimuth/elevation angles
    out_left = relighter.relight(canvas, landmarks, mask, face_width=100.0, strength=100.0, azimuth=-45.0, elevation=30.0)
    out_right = relighter.relight(canvas, landmarks, mask, face_width=100.0, strength=100.0, azimuth=45.0, elevation=30.0)
    
    assert out_left.shape == (100, 100, 3)
    assert out_right.shape == (100, 100, 3)
    # Different angles should yield different lighting shading
    assert not np.array_equal(out_left, out_right)
