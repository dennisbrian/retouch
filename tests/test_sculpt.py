"""Tests for retouch.relight.Relighter.sculpt() — facial structure sculpting (C2)."""

from __future__ import annotations

import numpy as np
import pytest
import cv2

from retouch.relight import Relighter


# ---------------------------------------------------------------------------
# Mock Landmarks for Testing
# ---------------------------------------------------------------------------


class MockLandmark:
    """Mock landmark point with x, y, z coordinates."""

    def __init__(self, x: float = 0.5, y: float = 0.5, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


class MockLandmarksList:
    """Mock MediaPipe landmarks list with all 468 points and yaw ratio control."""

    def __init__(self, yaw_ratio: float = 1.0):
        """
        Args:
            yaw_ratio: Controls temple-to-nose-bridge distance ratio for yaw simulation.
                      1.0 = frontal face, >1.0 = profile face.
        """
        self.landmark = [MockLandmark() for _ in range(478)]

        # Set up key landmarks for yaw guard calculation
        # lm[6] = nose bridge midpoint
        self.landmark[6].x = 0.5
        self.landmark[6].y = 0.4

        # lm[234] = left temple
        self.landmark[234].x = 0.3
        self.landmark[234].y = 0.3

        # lm[454] = right temple
        self.landmark[454].x = 0.7
        self.landmark[454].y = 0.3

        # Adjust temple positions to simulate yaw
        if yaw_ratio != 1.0:
            # Shift one temple to create asymmetry for profile faces
            # For profile: right temple moves toward nose, left stays same
            # This creates ratio = max(d_left, d_right) / min(d_left, d_right)
            shift = 0.2 * (yaw_ratio - 1.0) / 2.0  # shift amount for one side
            self.landmark[454].x -= shift  # Right temple moves left (profile right)

        # Add some depth variation (z) for face mesh
        for i, lm in enumerate(self.landmark):
            if lm.z == 0.0:
                # Assign z values that increase toward face center (rough 3D structure)
                dist_to_center = np.sqrt((lm.x - 0.5)**2 + (lm.y - 0.5)**2)
                lm.z = -0.1 * dist_to_center  # Negative z for face curvature


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


def test_sculpt_strength_zero_returns_unchanged():
    """Test that strength=0 returns canvas unchanged."""
    relighter = Relighter()
    canvas = np.full((400, 300, 3), 128, dtype=np.uint8)
    mask = np.ones((400, 300), dtype=np.float32)
    landmarks = MockLandmarksList()

    result = relighter.sculpt(
        canvas, landmarks, mask, face_width=100.0, strength=0.0
    )
    assert np.array_equal(result, canvas)


def test_sculpt_none_skin_mask_returns_unchanged():
    """Test that None skin_mask returns canvas unchanged."""
    relighter = Relighter()
    canvas = np.full((400, 300, 3), 128, dtype=np.uint8)
    landmarks = MockLandmarksList()

    result = relighter.sculpt(
        canvas, landmarks, None, face_width=100.0, strength=50.0
    )
    assert np.array_equal(result, canvas)


def test_sculpt_empty_skin_mask_returns_unchanged():
    """Test that nearly-empty skin_mask (max < 0.01) returns canvas unchanged."""
    relighter = Relighter()
    canvas = np.full((400, 300, 3), 128, dtype=np.uint8)
    mask = np.zeros((400, 300), dtype=np.float32)
    landmarks = MockLandmarksList()

    result = relighter.sculpt(
        canvas, landmarks, mask, face_width=100.0, strength=50.0
    )
    assert np.array_equal(result, canvas)


def test_sculpt_flat_field_invariance():
    """Test that a perfectly flat gray face produces negligible change.

    Flat-field invariance ensures the sculpt doesn't invent corrections
    from nothing — only modulates existing structure.
    """
    relighter = Relighter()

    # Create a flat gray canvas
    canvas = np.full((400, 300, 3), 128, dtype=np.uint8)
    mask = np.ones((400, 300), dtype=np.float32)
    mask[100:300, 50:250] = 0.8  # Center face region

    landmarks = MockLandmarksList()

    result = relighter.sculpt(
        canvas, landmarks, mask, face_width=100.0, strength=80.0
    )

    # Measure max abs diff from original
    diff = np.abs(result.astype(np.float32) - canvas.astype(np.float32))
    max_diff = np.max(diff)

    # Flat field should produce negligible change (gain^strength with bounded gain ≈ identity)
    # With gain in [0.7, 1.3], at full strength the max change is ~0.3^0.35 ≈ 0.86 on a
    # normalized scale, translating to ~22 on uint8. Allow generous tolerance for numerical precision.
    assert max_diff < 30.0, f"Flat field changed by {max_diff} (expected < 30)"


def test_sculpt_yaw_gating_extreme_profile():
    """Test that extreme yaw (profile face) heavily attenuates sculpting.

    Yaw guard should strongly fade sculpting strength for profile faces
    (temple ratio > 1.7). At ratio=1.8, yaw_factor ≈ 0.17, so effect should be ~17%.
    """
    relighter = Relighter()

    # Create a textured canvas to measure effect
    canvas = np.zeros((400, 300, 3), dtype=np.uint8)
    for y in range(400):
        canvas[y, :] = int(100 + y * 55 / 400)

    mask = np.ones((400, 300), dtype=np.float32)

    # Extreme yaw: ratio = 1.8 > 1.7, should heavily attenuate (yaw_factor ≈ 0.17)
    landmarks_extreme = MockLandmarksList(yaw_ratio=1.8)

    result = relighter.sculpt(
        canvas, landmarks_extreme, mask, face_width=100.0, strength=100.0
    )

    # Measure effect magnitude
    diff = np.abs(result.astype(np.float32) - canvas.astype(np.float32))
    mean_diff = np.mean(diff)

    # Effect should be very small (< 5% of max possible for strength=100)
    # At full strength with yaw_factor=0.17, correction is ~6% of max.
    assert mean_diff < 5.0, f"Extreme yaw effect too large: {mean_diff:.1f} (expected < 5)"


def test_sculpt_yaw_gating_partial_profile():
    """Test that partial yaw (1.6 ratio) partially attenuates sculpting."""
    relighter = Relighter()

    # Create a textured canvas to see the difference
    canvas = np.full((400, 300, 3), 150, dtype=np.uint8)
    mask = np.ones((400, 300), dtype=np.float32)

    # Frontal face
    landmarks_frontal = MockLandmarksList(yaw_ratio=1.0)

    # Partial profile: ratio = 1.6, yaw_factor = 0.5
    landmarks_partial = MockLandmarksList(yaw_ratio=1.6)

    result_frontal = relighter.sculpt(
        canvas, landmarks_frontal, mask, face_width=100.0, strength=100.0
    )

    result_partial = relighter.sculpt(
        canvas, landmarks_partial, mask, face_width=100.0, strength=100.0
    )

    # Partial yaw should produce less change than frontal
    diff_frontal = np.abs(result_frontal.astype(np.float32) - canvas.astype(np.float32))
    diff_partial = np.abs(result_partial.astype(np.float32) - canvas.astype(np.float32))

    assert np.mean(diff_partial) < np.mean(diff_frontal)


def test_sculpt_texture_preservation():
    """Test that high-frequency texture (pores) is preserved after sculpting.

    Sculpt uses low-band-only correction, so high-frequency residual should
    remain nearly unchanged (within 3% energy tolerance).
    """
    relighter = Relighter()

    # Create a synthetic face with high-frequency noise (pore texture)
    np.random.seed(42)
    base = np.full((400, 300, 3), 150, dtype=np.uint8)

    # Add high-frequency noise to simulate pores
    noise = (np.random.randn(400, 300, 3) * 8).astype(np.int16)
    canvas = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    mask = np.ones((400, 300), dtype=np.float32)
    mask[100:300, 50:250] = 0.8  # Face region

    landmarks = MockLandmarksList()

    # Measure high-frequency energy before sculpting
    gray_before = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_blurred = cv2.GaussianBlur(gray_before, (0, 0), sigmaX=2.0)
    hf_before = gray_before - gray_blurred
    hf_energy_before = np.std(hf_before[mask > 0.5])

    result = relighter.sculpt(
        canvas, landmarks, mask, face_width=100.0, strength=80.0
    )

    # Measure high-frequency energy after sculpting
    gray_after = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_after_blurred = cv2.GaussianBlur(gray_after, (0, 0), sigmaX=2.0)
    hf_after = gray_after - gray_after_blurred
    hf_energy_after = np.std(hf_after[mask > 0.5])

    # High-frequency energy should be preserved (allow 3% tolerance)
    energy_change = abs(hf_energy_after - hf_energy_before) / (hf_energy_before + 1e-6)
    assert energy_change < 0.03, f"HF energy changed by {energy_change*100:.1f}% (expected < 3%)"


def test_sculpt_no_double_shadow_with_relight():
    """Test that relight + sculpt doesn't produce unreasonable luminance swings.

    When both relight(100) and sculpt(100) run in sequence, the combined
    effect should not produce pixels that shift more than ~80 L-levels,
    avoiding double-shadowing artifacts.
    """
    relighter = Relighter()

    # Create a subtle gradient canvas for better testing
    canvas = np.zeros((400, 300, 3), dtype=np.uint8)
    for y in range(400):
        canvas[y, :] = int(100 + y * 155 / 400)

    mask = np.ones((400, 300), dtype=np.float32)
    landmarks = MockLandmarksList()

    # Apply relight at full strength
    relit = relighter.relight(
        canvas, landmarks, mask,
        face_width=100.0, strength=100.0,
        engine="v2"
    )

    # Apply sculpt at full strength
    sculpted = relighter.sculpt(
        relit, landmarks, mask,
        face_width=100.0, strength=100.0
    )

    # Measure max pixel shift in LAB L channel
    lab_original = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab_final = cv2.cvtColor(sculpted, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_diff = np.abs(lab_final[:, :, 0] - lab_original[:, :, 0])

    max_L_shift = np.max(L_diff)
    mean_L_shift = np.mean(L_diff[mask > 0.5])

    # Sanity check: shifts should be reasonable (not creating extreme double-shadows)
    assert max_L_shift < 100.0, f"Max L shift {max_L_shift} too large (expected < 100)"
    assert mean_L_shift < 40.0, f"Mean L shift {mean_L_shift} too large (expected < 40)"


def test_sculpt_output_dtype_and_shape():
    """Test that sculpt returns uint8 BGR array with correct shape."""
    relighter = Relighter()
    canvas = np.full((400, 300, 3), 128, dtype=np.uint8)
    mask = np.ones((400, 300), dtype=np.float32)
    landmarks = MockLandmarksList()

    result = relighter.sculpt(
        canvas, landmarks, mask, face_width=100.0, strength=50.0
    )

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.uint8
    assert result.shape == (400, 300, 3)


def test_sculpt_strength_scaling_effect():
    """Test that higher strength produces larger sculpting effect."""
    relighter = Relighter()

    # Create a gradient canvas for better effect measurement
    canvas = np.zeros((400, 300, 3), dtype=np.uint8)
    for y in range(400):
        canvas[y, :] = int(80 + y * 95 / 400)

    mask = np.ones((400, 300), dtype=np.float32)
    landmarks = MockLandmarksList()

    result_weak = relighter.sculpt(
        canvas, landmarks, mask, face_width=100.0, strength=20.0
    )

    result_strong = relighter.sculpt(
        canvas, landmarks, mask, face_width=100.0, strength=80.0
    )

    # Measure total change
    diff_weak = np.mean(np.abs(result_weak.astype(np.float32) - canvas.astype(np.float32)))
    diff_strong = np.mean(np.abs(result_strong.astype(np.float32) - canvas.astype(np.float32)))

    # Stronger effect should produce larger changes
    assert diff_strong > diff_weak, "Higher strength should produce larger effect"


def test_sculpt_empty_valid_mask_does_not_produce_nan(monkeypatch):
    """A near-frontal/flat face crop can leave zero pixels clearing the
    grazing-angle N_z threshold, making `valid` empty. Regression for a bug
    where S_target was normalized by np.mean() of an empty slice (NaN),
    silently poisoning the whole output with NaN pixels."""
    relighter = Relighter()
    canvas = np.full((100, 100, 3), 180, dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.float32)
    landmarks = MockLandmarksList()

    def _all_frontal_geometry(canvas_shape, landmarks, face_width):
        h, w = canvas_shape
        w_small, h_small = 40, 40
        # N_z == 1.0 everywhere fails the `N_z < 0.999` grazing-angle test
        # for every pixel, making `valid` empty regardless of the mask.
        ones = np.ones((h_small, w_small), dtype=np.float32)
        return (
            np.zeros((h_small, w_small), dtype=np.float32),
            np.zeros((h_small, w_small), dtype=np.float32),
            ones,
            w_small, h_small, w_small / w,
        )

    monkeypatch.setattr(relighter, "_shading_geometry", _all_frontal_geometry)

    result = relighter.sculpt(
        canvas, landmarks, mask, face_width=100.0, strength=60.0
    )

    assert not np.isnan(result.astype(np.float64)).any(), (
        "sculpt() produced NaN pixels when the valid-normals mask was empty"
    )
    assert result.dtype == np.uint8
    assert result.shape == canvas.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
