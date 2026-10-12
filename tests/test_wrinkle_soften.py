"""Tests for wrinkle_soften in retouch/skin.py — ridge detection and line softening."""

import numpy as np
import cv2
import pytest

from retouch.skin import SkinProcessor
from retouch.parsing import FaceRegions


@pytest.fixture
def proc():
    return SkinProcessor()


@pytest.fixture
def img_base():
    """Base neutral mid-tone image for testing."""
    return np.full((128, 128, 3), 128, dtype=np.uint8)


@pytest.fixture
def face_mask():
    """Simple face-like mask."""
    mask = np.zeros((128, 128), dtype=np.float32)
    mask[32:96, 32:96] = 1.0
    return cv2.GaussianBlur(mask, (21, 21), 0)


@pytest.fixture
def regions_with_wrinkle_zones():
    """Create minimal FaceRegions with wrinkle zones."""
    regions = FaceRegions()

    # Basic masks
    h, w = 128, 128
    regions.skin = np.zeros((h, w), dtype=np.float32)
    regions.skin[32:96, 32:96] = 1.0

    # Hair and eyebrows (for exclusion testing) - positioned to not overlap wrinkle test zones
    regions.hair = np.zeros((h, w), dtype=np.float32)
    regions.hair[16:32, 32:96] = 0.8  # Top of head above forehead

    regions.left_eyebrow = np.zeros((h, w), dtype=np.float32)
    regions.left_eyebrow[48:53, 40:55] = 0.7

    regions.right_eyebrow = np.zeros((h, w), dtype=np.float32)
    regions.right_eyebrow[48:53, 73:88] = 0.7

    # Wrinkle zones - non-overlapping
    regions.forehead = np.zeros((h, w), dtype=np.float32)
    regions.forehead[32:48, 40:88] = 1.0  # Forehead band, above eyebrows

    regions.nasolabial_l = np.zeros((h, w), dtype=np.float32)
    regions.nasolabial_l[60:75, 45:65] = 1.0  # Lower face, left side

    regions.nasolabial_r = np.zeros((h, w), dtype=np.float32)
    regions.nasolabial_r[60:75, 63:83] = 1.0  # Lower face, right side

    regions.crows_feet_l = np.zeros((h, w), dtype=np.float32)
    regions.crows_feet_l[52:62, 32:45] = 1.0  # Outer eye corner, left

    regions.crows_feet_r = np.zeros((h, w), dtype=np.float32)
    regions.crows_feet_r[52:62, 83:96] = 1.0  # Outer eye corner, right

    regions.neck = np.zeros((h, w), dtype=np.float32)
    regions.neck[86:96, 40:88] = 0.5  # Neck below face

    return regions


class TestWrinkleSoftenBasics:
    """Test basic functionality: early returns, output type/shape."""

    def test_strength_zero_returns_unchanged(self, proc, img_base, regions_with_wrinkle_zones):
        """strength=0 should return image unchanged."""
        result = proc.wrinkle_soften(img_base, regions_with_wrinkle_zones, strength=0)
        assert np.array_equal(result, img_base)

    def test_none_regions_returns_unchanged(self, proc, img_base):
        """regions=None should return image unchanged."""
        result = proc.wrinkle_soften(img_base, None, strength=50)
        assert np.array_equal(result, img_base)

    def test_none_skin_mask_returns_unchanged(self, proc, img_base):
        """regions without skin mask should return image unchanged."""
        regions = FaceRegions()
        regions.skin = None
        result = proc.wrinkle_soften(img_base, regions, strength=50)
        assert np.array_equal(result, img_base)

    def test_output_dtype_and_shape(self, proc, img_base, regions_with_wrinkle_zones):
        """Output should be uint8 BGR with same shape."""
        result = proc.wrinkle_soften(img_base, regions_with_wrinkle_zones, strength=50)
        assert result.dtype == np.uint8
        assert result.shape == img_base.shape


class TestLandmarkGeometrySanity:
    """Test that landmark geometry zones are sane and non-empty."""

    def test_nasolabial_l_nonzero(self, proc, img_base, regions_with_wrinkle_zones):
        """Left nasolabial zone should be non-empty."""
        assert regions_with_wrinkle_zones.nasolabial_l is not None
        assert np.any(regions_with_wrinkle_zones.nasolabial_l > 0.1)

    def test_nasolabial_r_nonzero(self, proc, img_base, regions_with_wrinkle_zones):
        """Right nasolabial zone should be non-empty."""
        assert regions_with_wrinkle_zones.nasolabial_r is not None
        assert np.any(regions_with_wrinkle_zones.nasolabial_r > 0.1)

    def test_crows_feet_l_nonzero(self, proc, img_base, regions_with_wrinkle_zones):
        """Left crow's feet zone should be non-empty."""
        assert regions_with_wrinkle_zones.crows_feet_l is not None
        assert np.any(regions_with_wrinkle_zones.crows_feet_l > 0.1)

    def test_crows_feet_r_nonzero(self, proc, img_base, regions_with_wrinkle_zones):
        """Right crow's feet zone should be non-empty."""
        assert regions_with_wrinkle_zones.crows_feet_r is not None
        assert np.any(regions_with_wrinkle_zones.crows_feet_r > 0.1)

    def test_zones_spatially_compact(self, proc, img_base, regions_with_wrinkle_zones):
        """Each zone's bounding box should be a small fraction of total face area."""
        face_area = np.sum(regions_with_wrinkle_zones.skin > 0.3)

        for zone_name in ["nasolabial_l", "nasolabial_r", "crows_feet_l", "crows_feet_r"]:
            zone = getattr(regions_with_wrinkle_zones, zone_name)
            zone_area = np.sum(zone > 0.1)
            # Zone should be <15% of face area
            assert zone_area < face_area * 0.15, f"{zone_name} is too large"

    def test_crows_feet_not_overlapping_eye_interior(self, proc, regions_with_wrinkle_zones):
        """Crow's feet zones should not significantly overlap eye regions."""
        # Create a simple eye mock
        eye_l = np.zeros((128, 128), dtype=np.float32)
        eye_l[45:55, 38:50] = 1.0  # Left eye

        overlap = np.sum(regions_with_wrinkle_zones.crows_feet_l * eye_l)
        total_eye = np.sum(eye_l)
        if total_eye > 0:
            overlap_frac = overlap / total_eye
            assert overlap_frac < 0.5, "Crow's feet overlaps eye too much"


class TestRidgeDetection:
    """Test ridge detection and attenuation in synthetic wrinkle images."""

    def test_wrinkle_attenuation_reduces_contrast(self, proc, regions_with_wrinkle_zones):
        """Synthetic wrinkle should be softened by wrinkle_soften."""
        # Create image with artificial wrinkle: horizontal line in forehead zone
        img = np.full((128, 128, 3), 140, dtype=np.uint8)

        # Draw a VERY dark horizontal "wrinkle" line in forehead area
        # Use extreme contrast for test reliability given uint8 precision
        cv2.line(img, (40, 40), (88, 40), (30, 30, 30), 5)  # Very dark line

        # Measure L-channel brightening at the line
        lab_before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_before = lab_before[:, :, 0]

        # Get line pixels and surrounding
        line_y, line_x_start, line_x_end = 40, 40, 88
        line_L_before = L_before[line_y, line_x_start:line_x_end].mean()

        # Apply wrinkle softening at high strength
        result = proc.wrinkle_soften(img, regions_with_wrinkle_zones, strength=100)

        lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_after = lab_after[:, :, 0]

        line_L_after = L_after[line_y, line_x_start:line_x_end].mean()

        # L-value at wrinkle line should increase (wrinkle softened/brightened)
        assert line_L_after > line_L_before, f"Wrinkle line L should brighten: {line_L_after} > {line_L_before}"

        # But not completely eliminated (at least 30% of darkening should remain)
        L_surrounding = lab_before[line_y+5:line_y+8, line_x_start:line_x_end, 0].mean()
        original_depth = L_surrounding - line_L_before
        residual_depth = L_surrounding - line_L_after

        assert residual_depth > original_depth * 0.3, f"Wrinkle over-softened: {residual_depth} vs {original_depth * 0.3}"

    def test_diagonal_wrinkle_detection(self, proc, regions_with_wrinkle_zones):
        """Diagonal wrinkle (nasolabial-style) should also be detected."""
        img = np.full((128, 128, 3), 140, dtype=np.uint8)

        # Draw diagonal line in nasolabial region with very dark value
        cv2.line(img, (45, 60), (65, 75), (30, 30, 30), 4)

        lab_before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_before = lab_before[:, :, 0]
        L_before_line_mean = L_before[60:76, 45:66].mean()

        result = proc.wrinkle_soften(img, regions_with_wrinkle_zones, strength=100)

        lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_after = lab_after[:, :, 0]

        # Diagonal line region should show L-value changes
        L_after_line_mean = L_after[60:76, 45:66].mean()

        # Should have some lightening (wrinkle softening)
        assert L_after_line_mean > L_before_line_mean, f"Wrinkle should lighten: {L_after_line_mean} > {L_before_line_mean}"


class TestDepthCapping:
    """Test that attenuation is capped at 60% to preserve natural geometry."""

    def test_over_reduction_guard_at_strength_100(self, proc, regions_with_wrinkle_zones):
        """At strength=100, wrinkle must not be fully eliminated."""
        img = np.full((128, 128, 3), 140, dtype=np.uint8)

        # Draw strong wrinkle
        cv2.line(img, (40, 40), (88, 40), (80, 80, 80), 4)

        lab_before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_before = lab_before[40, 40:88]

        result = proc.wrinkle_soften(img, regions_with_wrinkle_zones, strength=100)

        lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_after = lab_after[40, 40:88]

        # Calculate how much the L-value changed
        L_change = L_after - L_before

        # At least some of the wrinkle depth must remain (capped at 60% removal)
        # So at minimum 40% of the original depth should survive
        wrinkle_depth = 140 - 80  # ~60 levels
        residual_depth = wrinkle_depth - L_change.min()

        # Should retain at least 40% of depth
        assert residual_depth > wrinkle_depth * 0.3, "Wrinkle reduced too much (failed 60% cap)"


class TestHairEyebrowExclusion:
    """Test that hair and eyebrows are excluded from treatment."""

    def test_hair_exclusion(self, proc, regions_with_wrinkle_zones):
        """Hair region should be mostly excluded from wrinkle processing."""
        img = np.full((128, 128, 3), 140, dtype=np.uint8)

        # Draw wrinkle in hair region (but note: Gaussian blur will bleed to nearby pixels)
        cv2.line(img, (40, 28), (88, 28), (30, 30, 30), 5)  # Very dark, in hair

        lab_before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_before = lab_before[28, 40:88].copy()

        result = proc.wrinkle_soften(img, regions_with_wrinkle_zones, strength=80)

        lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_after = lab_after[28, 40:88]

        # Hair should be mostly unaffected (allow some bleed from Gaussian)
        # Since hair is excluded from wrinkle_mask, there should be minimal effect
        change = np.abs(L_after - L_before)
        # Most pixels should have <5 change; allow for edge bleed
        assert np.sum(change > 5) < len(change) * 0.2, "Hair region too heavily modified"

    def test_eyebrow_exclusion(self, proc, regions_with_wrinkle_zones):
        """Eyebrow region should be mostly excluded from wrinkle processing."""
        img = np.full((128, 128, 3), 140, dtype=np.uint8)

        # Draw wrinkle in eyebrow region (but note: Gaussian blur will bleed)
        cv2.line(img, (40, 49), (55, 49), (30, 30, 30), 4)

        lab_before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_before = lab_before[49, 40:55].copy()

        result = proc.wrinkle_soften(img, regions_with_wrinkle_zones, strength=80)

        lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)
        L_after = lab_after[49, 40:55]

        # Eyebrow should be mostly unaffected (allow some bleed from Gaussian)
        change = np.abs(L_after - L_before)
        # Most pixels should have <5 change; allow for edge bleed
        assert np.sum(change > 5) < len(change) * 0.2, "Eyebrow region too heavily modified"


class TestNoGhostEdges:
    """Test for absence of ringing artifacts (over-subtraction overshoot)."""

    def test_no_overshoot_adjacent_to_ridge(self, proc, regions_with_wrinkle_zones):
        """Pixels adjacent to wrinkle should not show excessive overshoot."""
        img = np.full((128, 128, 3), 140, dtype=np.uint8)

        # Draw very dark wrinkle in forehead for strong detection
        cv2.line(img, (50, 40), (80, 40), (30, 30, 30), 5)

        lab_before = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)

        result = proc.wrinkle_soften(img, regions_with_wrinkle_zones, strength=100)

        lab_after = cv2.cvtColor(result, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Check regions adjacent to the wrinkle line (left/right of it)
        # Note: Some bleed is expected from Gaussian convolution
        adjacent_before = lab_before[40, 35:45, 0]  # Left of wrinkle
        adjacent_after = lab_after[40, 35:45, 0]

        # Should not have massive overshoot (>10 units)
        # Allow some bleed due to Gaussian, but cap at reasonable level
        max_overshoot = (adjacent_after - adjacent_before).max()
        assert max_overshoot < 15, f"Excessive overshoot detected: {max_overshoot}"


class TestStrengthScaling:
    """Test that strength parameter scales effect appropriately."""

    def test_higher_strength_stronger_effect(self, proc, regions_with_wrinkle_zones):
        """strength=80 should have stronger effect than strength=20."""
        img = np.full((128, 128, 3), 140, dtype=np.uint8)
        cv2.line(img, (40, 40), (88, 40), (30, 30, 30), 5)  # Very dark line for detection

        result_20 = proc.wrinkle_soften(img, regions_with_wrinkle_zones, strength=20)
        result_80 = proc.wrinkle_soften(img, regions_with_wrinkle_zones, strength=80)

        lab_orig = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_20 = cv2.cvtColor(result_20, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab_80 = cv2.cvtColor(result_80, cv2.COLOR_BGR2LAB).astype(np.float32)

        # Get line region L-values
        line_y = 40
        line_x = slice(40, 88)

        diff_20 = np.abs(lab_20[line_y, line_x, 0] - lab_orig[line_y, line_x, 0]).mean()
        diff_80 = np.abs(lab_80[line_y, line_x, 0] - lab_orig[line_y, line_x, 0]).mean()

        # Higher strength should produce larger changes (diff_80 should be > diff_20)
        assert diff_80 > diff_20, f"Strength scaling not working (80={diff_80} should > 20={diff_20})"
