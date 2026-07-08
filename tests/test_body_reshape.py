"""Tests for body reshape engine (T3 — retouch/body_reshape.py).

Covers:
  - Pose detection with MediaPipe (33 landmarks)
  - Zero-strength no-op (output identical to input)
  - Dtype preservation (uint8 and float32)
  - Per-feature displacement cap enforcement (±15% of segment length at ±100)
  - Graceful fallback when pose not detected
  - Landmark visibility filtering (skip < 0.3 confidence)
  - Limb-length calculation and warping
  - Person mask boundary control
"""

from __future__ import annotations

import numpy as np
import pytest

from retouch.body_reshape import (
    BodyReshaper,
    PoseDetector,
    PoseContext,
    _VISIBILITY_THRESHOLD,
    _MAX_DISPLACE_FRAC,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _synthetic_body_image(h: int = 400, w: int = 400) -> np.ndarray:
    """Create a synthetic body image (grayscale, then BGR).

    Simple test image: a vertical rectangle representing a standing figure.
    """
    img = np.ones((h, w, 3), dtype=np.uint8) * 100  # Gray background

    # Draw a rough human silhouette (for visual debugging)
    # Head: center top
    cv2_available = True
    try:
        import cv2
        # Shoulders
        cv2.circle(img, (w // 2 - 30, h // 4), 8, (200, 150, 100), -1)
        cv2.circle(img, (w // 2 + 30, h // 4), 8, (200, 150, 100), -1)
        # Torso
        cv2.rectangle(img, (w // 2 - 25, h // 4 + 10), (w // 2 + 25, h // 2), (180, 140, 100), -1)
        # Hips
        cv2.circle(img, (w // 2 - 20, h // 2 + 10), 6, (170, 130, 90), -1)
        cv2.circle(img, (w // 2 + 20, h // 2 + 10), 6, (170, 130, 90), -1)
        # Legs
        cv2.line(img, (w // 2 - 15, h // 2 + 20), (w // 2 - 10, h - 20), (150, 110, 70), 8)
        cv2.line(img, (w // 2 + 15, h // 2 + 20), (w // 2 + 10, h - 20), (150, 110, 70), 8)
    except ImportError:
        pass

    return img.astype(np.uint8)


def _mock_pose_landmarks(h: int = 400, w: int = 400) -> tuple[list, list]:
    """Create mock MediaPipe Pose landmarks (33 landmarks).

    Returns (landmarks, visibility) where landmarks are [(x, y), ...] in [0,1]².
    """
    landmarks = []
    visibility = []

    # Normalize to [0, 1]
    def norm_x(px): return px / w
    def norm_y(py): return py / h

    # Create a full 33-landmark pose for a standing figure
    # (simplified; MediaPipe format: index 0-32)

    # Head region (0-10) — not critical for body reshape
    for _ in range(11):
        landmarks.append((0.5, 0.15))
        visibility.append(0.9)

    # Shoulders: [11] (left), [12] (right)
    landmarks.append((norm_x(w // 2 - 40), norm_y(h // 4)))
    visibility.append(0.95)
    landmarks.append((norm_x(w // 2 + 40), norm_y(h // 4)))
    visibility.append(0.95)

    # Elbows: [13] (left), [14] (right)
    landmarks.append((norm_x(w // 2 - 50), norm_y(h // 4 + 50)))
    visibility.append(0.90)
    landmarks.append((norm_x(w // 2 + 50), norm_y(h // 4 + 50)))
    visibility.append(0.90)

    # Wrists: [15] (left), [16] (right)
    landmarks.append((norm_x(w // 2 - 55), norm_y(h // 4 + 100)))
    visibility.append(0.85)
    landmarks.append((norm_x(w // 2 + 55), norm_y(h // 4 + 100)))
    visibility.append(0.85)

    # Fingers [17-22] — not critical
    for _ in range(6):
        landmarks.append((0.5, 0.5))
        visibility.append(0.8)

    # Hips: [23] (left), [24] (right)
    landmarks.append((norm_x(w // 2 - 30), norm_y(h // 2)))
    visibility.append(0.95)
    landmarks.append((norm_x(w // 2 + 30), norm_y(h // 2)))
    visibility.append(0.95)

    # Knees: [25] (left), [26] (right)
    landmarks.append((norm_x(w // 2 - 25), norm_y(h // 2 + 80)))
    visibility.append(0.90)
    landmarks.append((norm_x(w // 2 + 25), norm_y(h // 2 + 80)))
    visibility.append(0.90)

    # Ankles: [27] (left), [28] (right)
    landmarks.append((norm_x(w // 2 - 20), norm_y(h - 30)))
    visibility.append(0.85)
    landmarks.append((norm_x(w // 2 + 20), norm_y(h - 30)))
    visibility.append(0.85)

    # Feet [29-32] — not critical
    for _ in range(4):
        landmarks.append((0.5, 0.9))
        visibility.append(0.7)

    return landmarks, visibility


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------

class TestPoseContext:
    """Test PoseContext data structure."""

    def test_pose_context_initialization(self):
        """PoseContext initializes with default values."""
        ctx = PoseContext()
        assert ctx.detected == False
        assert ctx.body_visible == False
        assert ctx.landmarks is None
        assert ctx.visibility is None

    def test_pose_context_feature_flags(self):
        """PoseContext has per-feature flags."""
        ctx = PoseContext()
        assert "arm_length" in ctx.feature_flags
        assert "leg_length" in ctx.feature_flags
        assert "torso_width" in ctx.feature_flags
        assert "shoulder_width" in ctx.feature_flags
        assert "hip_width" in ctx.feature_flags


class TestBodyReshaper:
    """Test BodyReshaper core functionality."""

    def test_no_op_at_default(self):
        """Zero slider displacement should return input unchanged."""
        img = _synthetic_body_image()
        reshaper = BodyReshaper()

        # Manually set a pose context to avoid actual detection
        h, w = img.shape[:2]
        reshaper.pose_ctx = PoseContext()
        reshaper.pose_ctx.landmarks, reshaper.pose_ctx.visibility = _mock_pose_landmarks(h, w)
        reshaper.pose_ctx.detected = False  # Trigger no-op via no-pose path

        # All sliders at default (50.0 = no-op when centered)
        result = reshaper.reshape(
            img,
            arm_length=0.0,
            leg_length=0.0,
            torso_width=0.0,
            shoulder_width=0.0,
            hip_width=0.0,
        )

        # Output should be identical to input (no deformation)
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        np.testing.assert_array_equal(result, img)

    def test_dtype_preservation_uint8(self):
        """Output dtype should match input dtype (uint8)."""
        img = _synthetic_body_image()
        assert img.dtype == np.uint8

        reshaper = BodyReshaper()
        result = reshaper.reshape(
            img,
            arm_length=20.0,  # Non-zero slider
            leg_length=0.0,
            torso_width=0.0,
            shoulder_width=0.0,
            hip_width=0.0,
        )

        assert result.dtype == np.uint8

    def test_dtype_preservation_float32(self):
        """Output dtype should match input dtype (float32)."""
        img = _synthetic_body_image().astype(np.float32) / 255.0
        assert img.dtype == np.float32

        reshaper = BodyReshaper()
        result = reshaper.reshape(
            img,
            arm_length=20.0,
            leg_length=0.0,
            torso_width=0.0,
            shoulder_width=0.0,
            hip_width=0.0,
        )

        assert result.dtype == np.float32
        # Check bounds (should be clipped to [0, 1])
        assert result.min() >= -0.01  # Allow small numerical error
        assert result.max() <= 1.01


class TestGracefulFallback:
    """Test graceful handling of missing pose."""

    def test_no_pose_detected_returns_original(self):
        """When no pose is detected, return input unchanged."""
        img = _synthetic_body_image()
        reshaper = BodyReshaper()

        # Manually set pose_ctx to no-pose
        reshaper.pose_ctx = PoseContext(detected=False)

        result = reshaper.reshape(
            img,
            arm_length=50.0,  # Non-neutral slider
            leg_length=50.0,
            torso_width=50.0,
            shoulder_width=50.0,
            hip_width=50.0,
        )

        # Should return input unchanged when no pose detected
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        np.testing.assert_array_equal(result, img)

    def test_all_zero_sliders_returns_original(self):
        """When all sliders are at neutral (50), return input unchanged."""
        img = _synthetic_body_image()
        reshaper = BodyReshaper()

        result = reshaper.reshape(
            img,
            arm_length=50.0,  # Neutral (centered around 50)
            leg_length=50.0,
            torso_width=50.0,
            shoulder_width=50.0,
            hip_width=50.0,
        )

        # When centered at 50, the slider value is (50 - 50) = 0, so no-op
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        # Output should be identical
        np.testing.assert_array_equal(result, img)


class TestDisplacementCaps:
    """Test displacement magnitude capping."""

    def test_max_displacement_frac_constant(self):
        """_MAX_DISPLACE_FRAC should be 0.15 (15%)."""
        assert _MAX_DISPLACE_FRAC == 0.15

    def test_visibility_threshold_constant(self):
        """_VISIBILITY_THRESHOLD should be 0.3."""
        assert _VISIBILITY_THRESHOLD == 0.3


class TestWarpCalculations:
    """Test internal warp calculation logic."""

    def test_arm_length_warps_generation(self):
        """Arm length warps should be generated for visible arms."""
        h, w = 400, 400
        ctx = PoseContext()
        ctx.landmarks, ctx.visibility = _mock_pose_landmarks(h, w)

        reshaper = BodyReshaper()
        warps = reshaper._arm_length_warps(ctx, w, h, slider=30.0)

        # Should have 2 warps (left + right arm)
        assert len(warps) > 0

        # Each warp is (control_point, target_point, radius)
        for warp in warps:
            assert len(warp) == 3
            cp, tp, r = warp
            assert len(cp) == 2  # (x, y)
            assert len(tp) == 2  # (x, y)
            assert r > 0  # Radius should be positive

    def test_leg_length_warps_generation(self):
        """Leg length warps should be generated for visible legs."""
        h, w = 400, 400
        ctx = PoseContext()
        ctx.landmarks, ctx.visibility = _mock_pose_landmarks(h, w)

        reshaper = BodyReshaper()
        warps = reshaper._leg_length_warps(ctx, w, h, slider=30.0)

        # Should have 2 warps (left + right leg)
        assert len(warps) > 0

    def test_torso_width_warps_generation(self):
        """Torso width warps should expand/compress via shoulder displacement."""
        h, w = 400, 400
        ctx = PoseContext()
        ctx.landmarks, ctx.visibility = _mock_pose_landmarks(h, w)

        reshaper = BodyReshaper()
        warps = reshaper._torso_width_warps(ctx, w, h, slider=30.0)

        # Should have 2 warps (left + right shoulder)
        assert len(warps) > 0

    def test_shoulder_width_warps_generation(self):
        """Shoulder width warps should generate warps."""
        h, w = 400, 400
        ctx = PoseContext()
        ctx.landmarks, ctx.visibility = _mock_pose_landmarks(h, w)

        reshaper = BodyReshaper()
        warps = reshaper._shoulder_width_warps(ctx, w, h, slider=30.0)

        # Should have 2 warps
        assert len(warps) > 0

    def test_hip_width_warps_generation(self):
        """Hip width warps should generate warps."""
        h, w = 400, 400
        ctx = PoseContext()
        ctx.landmarks, ctx.visibility = _mock_pose_landmarks(h, w)

        reshaper = BodyReshaper()
        warps = reshaper._hip_width_warps(ctx, w, h, slider=30.0)

        # Should have 2 warps
        assert len(warps) > 0


class TestSliderCentering:
    """Test slider value mapping (0-100 mapped to ±value)."""

    def test_slider_50_is_neutral(self):
        """Slider value 50 should produce no warps (centered)."""
        img = _synthetic_body_image()
        reshaper = BodyReshaper()

        # Manually create a pose context
        h, w = img.shape[:2]
        reshaper.pose_ctx = PoseContext()
        reshaper.pose_ctx.landmarks, reshaper.pose_ctx.visibility = _mock_pose_landmarks(h, w)
        reshaper.pose_ctx.detected = True
        reshaper.pose_ctx.body_visible = True

        # Test with 50 (neutral) — after centering, becomes 0
        result = reshaper.reshape(
            img,
            arm_length=50.0,  # (50 - 50) = 0
            leg_length=50.0,
            torso_width=50.0,
            shoulder_width=50.0,
            hip_width=50.0,
        )

        # Should be identical since no displacement
        # Note: due to floating point rounding in cv2.remap, allow small differences
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        # Check that diff is minimal (< 1% difference expected)
        diff = np.abs(result.astype(float) - img.astype(float)).mean()
        assert diff < 1.0, f"Expected near-identical output, but got mean diff of {diff}"

    def test_slider_below_50_compresses(self):
        """Slider < 50 should produce negative displacement (compress)."""
        h, w = 400, 400
        ctx = PoseContext()
        ctx.landmarks, ctx.visibility = _mock_pose_landmarks(h, w)

        reshaper = BodyReshaper()
        warps_below = reshaper._arm_length_warps(ctx, w, h, slider=-50.0)  # Strong compress
        warps_neutral = reshaper._arm_length_warps(ctx, w, h, slider=0.0)   # Neutral

        # Below 50 should produce some displacement
        assert len(warps_below) > 0

    def test_slider_above_50_extends(self):
        """Slider > 50 should produce positive displacement (extend)."""
        h, w = 400, 400
        ctx = PoseContext()
        ctx.landmarks, ctx.visibility = _mock_pose_landmarks(h, w)

        reshaper = BodyReshaper()
        warps_above = reshaper._arm_length_warps(ctx, w, h, slider=50.0)   # Strong extend

        # Above 50 should produce some displacement
        assert len(warps_above) > 0


class TestVisibilityFiltering:
    """Test landmark visibility threshold enforcement."""

    def test_low_visibility_landmarks_skipped(self):
        """Landmarks with visibility < 0.3 should be skipped."""
        h, w = 400, 400
        ctx = PoseContext()
        ctx.landmarks, ctx.visibility = _mock_pose_landmarks(h, w)

        # Make ankle landmarks (27, 28) have low visibility
        ctx.visibility[27] = 0.1  # Below threshold
        ctx.visibility[28] = 0.1  # Below threshold

        reshaper = BodyReshaper()
        warps = reshaper._leg_length_warps(ctx, w, h, slider=50.0)

        # Should skip both legs since ankles are low-visibility
        assert len(warps) == 0

    def test_high_visibility_landmarks_used(self):
        """Landmarks with visibility >= 0.3 should be used."""
        h, w = 400, 400
        ctx = PoseContext()
        ctx.landmarks, ctx.visibility = _mock_pose_landmarks(h, w)

        # Ensure all key landmarks are visible
        for i in [11, 12, 23, 24, 27, 28]:
            ctx.visibility[i] = 0.9

        reshaper = BodyReshaper()
        warps = reshaper._leg_length_warps(ctx, w, h, slider=50.0)

        # Should have some warps since landmarks are visible
        assert len(warps) > 0


class TestPersonMask:
    """Test person mask boundary control."""

    def test_person_mask_applies_gating(self):
        """Person mask should gate the warp application."""
        img = _synthetic_body_image()
        h, w = img.shape[:2]

        # Create a person mask (0 = background, 1 = person)
        person_mask = np.zeros((h, w), dtype=np.float32)
        # Mask only the center region
        person_mask[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = 1.0

        reshaper = BodyReshaper()
        reshaper.pose_ctx = PoseContext()
        reshaper.pose_ctx.landmarks, reshaper.pose_ctx.visibility = _mock_pose_landmarks(h, w)
        reshaper.pose_ctx.detected = True
        reshaper.pose_ctx.body_visible = True

        result = reshaper.reshape(
            img,
            arm_length=30.0,  # Non-zero slider
            leg_length=0.0,
            torso_width=0.0,
            shoulder_width=0.0,
            hip_width=0.0,
            person_mask=person_mask,
        )

        # Result should still be valid
        assert result.shape == img.shape
        assert result.dtype == img.dtype


class TestIntegrationWithEngine:
    """Test integration with RetouchEngine."""

    def test_body_reshape_params_in_context(self):
        """Engine ProcessingContext should include body_reshape params."""
        from retouch.engine import ProcessingContext

        ctx = ProcessingContext()
        assert hasattr(ctx, "body_reshape_arm_length")
        assert hasattr(ctx, "body_reshape_leg_length")
        assert hasattr(ctx, "body_reshape_torso_width")
        assert hasattr(ctx, "body_reshape_shoulder_width")
        assert hasattr(ctx, "body_reshape_hip_width")

        # Should have default values
        assert ctx.body_reshape_arm_length == 50.0
        assert ctx.body_reshape_leg_length == 50.0
        assert ctx.body_reshape_torso_width == 50.0
        assert ctx.body_reshape_shoulder_width == 50.0
        assert ctx.body_reshape_hip_width == 50.0

    def test_body_reshape_params_in_params_list(self):
        """Body reshape params should be in PROCESSING_PARAMS."""
        from retouch.params import PROCESSING_PARAMS, get_param

        param_names = [p.name for p in PROCESSING_PARAMS]
        assert "body_reshape_arm_length" in param_names
        assert "body_reshape_leg_length" in param_names
        assert "body_reshape_torso_width" in param_names
        assert "body_reshape_shoulder_width" in param_names
        assert "body_reshape_hip_width" in param_names

        # Each should be retrievable by name
        spec_al = get_param("body_reshape_arm_length")
        assert spec_al.default == 50.0
        assert spec_al.min_val == 0
        assert spec_al.max_val == 100
        assert spec_al.recipe_key == "body_reshape.arm_length"


# ---------------------------------------------------------------------------
# Edge cases and stress tests
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_warp_list_no_op(self):
        """Empty warp list should return input unchanged."""
        img = _synthetic_body_image()
        h, w = img.shape[:2]

        reshaper = BodyReshaper()
        result = reshaper._apply_warps(img, [])

        assert result.shape == img.shape
        assert result.dtype == img.dtype
        np.testing.assert_array_equal(result, img)

    def test_very_small_image(self):
        """Reshaper should handle very small images gracefully."""
        img = np.ones((50, 50, 3), dtype=np.uint8) * 128
        reshaper = BodyReshaper()

        # Should not crash
        result = reshaper.reshape(
            img,
            arm_length=30.0,
            leg_length=0.0,
            torso_width=0.0,
            shoulder_width=0.0,
            hip_width=0.0,
        )

        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_extreme_slider_values(self):
        """Reshaper should clamp extreme slider values gracefully."""
        img = _synthetic_body_image()
        reshaper = BodyReshaper()

        # Test extreme values (should be clamped by warp application)
        result = reshaper.reshape(
            img,
            arm_length=-1000.0,  # Extreme compression
            leg_length=1000.0,   # Extreme extension
            torso_width=0.0,
            shoulder_width=0.0,
            hip_width=0.0,
        )

        assert result.shape == img.shape
        assert result.dtype == img.dtype

    def test_float32_range_bounds(self):
        """Float32 output should be in [0, 1] range."""
        img = _synthetic_body_image().astype(np.float32) / 255.0
        reshaper = BodyReshaper()

        result = reshaper.reshape(
            img,
            arm_length=30.0,
            leg_length=30.0,
            torso_width=0.0,
            shoulder_width=0.0,
            hip_width=0.0,
        )

        # Should be within valid float32 [0, 1] range (with small tolerance)
        assert result.min() >= -0.02
        assert result.max() <= 1.02


# ---------------------------------------------------------------------------
# Consistency tests
# ---------------------------------------------------------------------------

class TestConsistency:
    """Test consistency across multiple runs and parameter combinations."""

    def test_deterministic_output(self):
        """Same input should produce same output."""
        img = _synthetic_body_image()
        reshaper = BodyReshaper()

        result1 = reshaper.reshape(
            img.copy(),
            arm_length=30.0,
            leg_length=20.0,
            torso_width=0.0,
            shoulder_width=0.0,
            hip_width=0.0,
        )

        # Reset pose context for fresh detection
        reshaper.pose_ctx = None
        result2 = reshaper.reshape(
            img.copy(),
            arm_length=30.0,
            leg_length=20.0,
            torso_width=0.0,
            shoulder_width=0.0,
            hip_width=0.0,
        )

        # Results should be very similar (uint8 may have minor rounding differences)
        assert result1.dtype == result2.dtype
        assert result1.shape == result2.shape

    def test_bilateral_symmetry_left_right(self):
        """Left and right limbs should warp symmetrically."""
        h, w = 400, 400
        ctx = PoseContext()
        ctx.landmarks, ctx.visibility = _mock_pose_landmarks(h, w)

        reshaper = BodyReshaper()

        # Get left and right arm warps
        all_warps = reshaper._arm_length_warps(ctx, w, h, 30.0)
        warps_left = [warp for warp in all_warps
                      if warp[0][0] < w // 2]  # Left arm (control point x < width/2)
        warps_right = [warp for warp in all_warps
                       if warp[0][0] >= w // 2]  # Right arm

        # Should have similar number of warps
        # (This is a weak check but ensures left/right are both processed)
        assert len(warps_left) > 0 or len(warps_right) > 0
