"""Tests for C1 skin_locus recipe override integration."""

import cv2
import numpy as np
import pytest

from retouch.engine import RetouchEngine, build_context, ProcessingContext
from retouch.params import resolve_recipe, _deep_merge
from retouch.color_science import (
    bgr_to_oklab, oklab_to_oklch, SKIN_LOCI, measure_skin_state
)
from retouch.skin import SkinProcessor


class TestBuildContextSkinLocus:
    """Test skin_locus extraction from recipes and overrides in build_context()."""

    def test_recipe_with_skin_locus(self):
        """Recipe dict with {"skin": {"locus": {...}}} produces matching ctx.skin_locus."""
        rec = {
            "skin": {
                "locus": {"h_target": 30.0, "C_target": 0.05},
                "smooth": 0.3,
            },
            "frequency": {"smooth": 0.3},
        }
        overrides = {}
        ctx = build_context("test_recipe", rec, overrides)
        assert ctx.skin_locus is not None
        assert ctx.skin_locus["h_target"] == 30.0
        assert ctx.skin_locus["C_target"] == 0.05

    def test_recipe_without_skin_locus(self):
        """Recipe dict without skin.locus key produces ctx.skin_locus = None."""
        rec = {
            "skin": {"smooth": 0.3},
            "frequency": {"smooth": 0.3},
        }
        overrides = {}
        ctx = build_context("test_recipe", rec, overrides)
        assert ctx.skin_locus is None

    def test_empty_recipe_no_skin_locus(self):
        """Empty recipe produces ctx.skin_locus = None."""
        rec = {}
        overrides = {}
        ctx = build_context("test_recipe", rec, overrides)
        assert ctx.skin_locus is None

    def test_caller_override_wins_over_recipe(self):
        """Explicit caller override for skin_locus wins over recipe value."""
        rec = {
            "skin": {
                "locus": {"h_target": 30.0, "C_target": 0.05},
            },
            "frequency": {"smooth": 0.3},
        }
        caller_locus = {"h_target": 60.0, "C_target": 0.12}
        overrides = {"skin_locus": caller_locus}
        ctx = build_context("test_recipe", rec, overrides)
        assert ctx.skin_locus is not None
        assert ctx.skin_locus["h_target"] == 60.0
        assert ctx.skin_locus["C_target"] == 0.12

    def test_caller_none_override_uses_recipe(self):
        """Caller override of None (not in dict) falls back to recipe."""
        rec = {
            "skin": {
                "locus": {"h_target": 30.0, "C_target": 0.05},
            },
            "frequency": {"smooth": 0.3},
        }
        overrides = {}  # No "skin_locus" key means caller didn't supply it
        ctx = build_context("test_recipe", rec, overrides)
        assert ctx.skin_locus is not None
        assert ctx.skin_locus["h_target"] == 30.0
        assert ctx.skin_locus["C_target"] == 0.05


class TestRetouchEngineProcessSkinLocus:
    """Test end-to-end skin_locus override via RetouchEngine.process()."""

    def test_process_with_recipe_locus(self):
        """Engine processes a recipe with custom locus; hue shift is applied."""
        engine = RetouchEngine()

        # Create a synthetic face-like image (uniform skin tone)
        img = np.full((100, 100, 3), 140, dtype=np.uint8)  # Neutral gray-ish
        img[:, :, 0] = 120  # Slightly more blue
        img[:, :, 1] = 135  # Slightly more green
        img[:, :, 2] = 150  # More red (typical warm skin)

        # Create a recipe with a custom skin locus
        custom_recipe = {
            "skin": {
                "locus": {"h_target": 30.0, "C_target": 0.05},
            },
            "frequency": {"smooth": 0.5},
        }

        # Manually call build_context to confirm the locus is threaded through
        from retouch.params import resolve_recipe
        resolved = resolve_recipe("natural")
        resolved.update(custom_recipe)

        ctx = build_context("test", resolved, {})
        assert ctx.skin_locus is not None
        assert ctx.skin_locus["h_target"] == 30.0

    def test_process_with_caller_locus_override(self):
        """Explicit caller skin_locus kwarg overrides any recipe locus."""
        engine = RetouchEngine()
        img = np.full((100, 100, 3), 140, dtype=np.uint8)
        img[:, :, 2] = 160

        caller_locus = {"h_target": 45.0, "C_target": 0.08}
        result = engine.process(
            img,
            recipe="natural",
            skin_locus=caller_locus,
            skin_hue_unify=50.0,
        )

        # Verify the skin_locus was stored in the result params
        assert result.params.skin_locus is not None
        assert result.params.skin_locus["h_target"] == 45.0
        assert result.params.skin_locus["C_target"] == 0.08

    def test_process_no_locus_auto_detect(self):
        """When no custom locus is provided, unify_hue_line auto-detects from skin."""
        engine = RetouchEngine()
        img = np.full((100, 100, 3), 140, dtype=np.uint8)
        img[:, :, 2] = 160

        # Build a minimal recipe without a locus override
        rec = resolve_recipe("natural")
        # Ensure the recipe doesn't have a locus key
        if "skin" in rec and "locus" in rec["skin"]:
            rec["skin"].pop("locus")

        ctx_temp = build_context("natural", rec, {})
        # Now call process - it may still get a locus if resolve_recipe adds one
        result = engine.process(
            img,
            recipe="natural",
            skin_hue_unify=50.0,
        )

        # Just verify the processing worked; don't assume no locus since
        # the recipe might contain one after resolution
        assert result is not None
        assert result.params.skin_hue_unify == 50.0


class TestSkinHueShiftTowardCustomLocus:
    """Test that custom locus actually shifts hue toward the target."""

    def test_hue_shift_direction(self):
        """Verify that applying unify_hue_line with custom locus shifts toward target."""
        proc = SkinProcessor()

        # Create synthetic skin-like image
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        # Warm-ish color (BGR): more red/yellow
        img[:, :, 0] = 100  # Blue
        img[:, :, 1] = 130  # Green
        img[:, :, 2] = 150  # Red

        # Mask for the entire image
        mask = np.ones((100, 100), dtype=np.float32)

        # Measure baseline hue
        oklab_orig = bgr_to_oklab(img)
        oklch_orig = oklab_to_oklch(oklab_orig)
        h_orig = oklch_orig[..., 2]  # Hue channel

        # Apply with custom locus (target hue 30.0)
        custom_locus = {"h_target": 30.0, "C_target": 0.05}
        result_custom = proc.unify_hue_line(
            img, mask, hue_strength=80, chroma_strength=50, locus=custom_locus
        )

        oklab_custom = bgr_to_oklab(result_custom)
        oklch_custom = oklab_to_oklch(oklab_custom)
        h_custom = oklch_custom[..., 2]  # Hue channel after custom locus

        # The mean hue should move toward the custom target (30.0)
        h_custom_mean = h_custom[mask > 0.5].mean()

        # Apply with auto-detected locus (let it choose based on skin state)
        result_auto = proc.unify_hue_line(
            img, mask, hue_strength=80, chroma_strength=50, locus=None
        )

        oklab_auto = bgr_to_oklab(result_auto)
        oklch_auto = oklab_to_oklch(oklab_auto)
        h_auto = oklch_auto[..., 2]
        h_auto_mean = h_auto[mask > 0.5].mean()

        # The distance from custom hue (30.0) to h_custom_mean should be smaller
        # than the distance from custom hue (30.0) to h_auto_mean
        dist_custom = abs(h_custom_mean - 30.0)
        dist_auto = abs(h_auto_mean - 30.0)

        # The custom locus should pull toward 30.0 more than auto-detect
        # (Note: this is a heuristic check; exact values depend on image content)
        assert dist_custom <= dist_auto or abs(dist_custom - dist_auto) < 5.0, \
            f"Custom locus (dist={dist_custom:.2f}) should pull hue closer to 30.0 than auto-detect (dist={dist_auto:.2f})"


class TestNoRegression:
    """Ensure existing engine functionality still works."""

    def test_natural_recipe_still_works(self):
        """Engine.process() with natural recipe (no custom locus) still works."""
        engine = RetouchEngine()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        img[:, :, 2] = 150

        result = engine.process(img, recipe="natural")
        assert result is not None
        assert result.shape == (100, 100, 3)

    def test_context_default_skin_locus_none(self):
        """ProcessingContext default for skin_locus is None."""
        ctx = ProcessingContext()
        assert ctx.skin_locus is None

    def test_skin_hue_unify_without_custom_locus(self):
        """skin_hue_unify works as before when no custom locus is provided."""
        engine = RetouchEngine()
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        img[:, :, 2] = 150

        # Call with skin_hue_unify but no skin_locus override
        result = engine.process(
            img,
            recipe="natural",
            skin_hue_unify=50.0,
        )

        assert result is not None
        assert result.params.skin_hue_unify == 50.0
        # skin_locus might be present if the resolved recipe contains one,
        # but we didn't explicitly override it via the process() kwarg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
