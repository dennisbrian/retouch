"""Tests for recipe-reachable post-effects wiring (halation, grain, lut, auto_exposure, color_transfer_intensity).

Covers the build_context() logic that translates recipe dict + caller overrides
into a ProcessingContext, specifically for the five post-effects that were
historically caller-only but are now recipe-reachable.

See CHANGE 1 in engine.py build_context() and CHANGE 3 in recipes.py for context.
"""

import pytest

from retouch.engine import build_context
from retouch.params import resolve_recipe


class TestPostEffectsRecipeValues:
    """Test that build_context correctly extracts post-effect values from recipes."""

    def test_halation_from_recipe(self):
        """Recipe supplies halation: 0.25 → ctx.halation == 0.25."""
        rec = {"halation": 0.25}
        ctx = build_context("test", rec, {})
        assert ctx.halation == 0.25

    def test_grain_from_recipe(self):
        """Recipe supplies grain: 0.05 → ctx.grain == 0.05."""
        rec = {"grain": 0.05}
        ctx = build_context("test", rec, {})
        assert ctx.grain == 0.05

    def test_lut_from_recipe(self):
        """Recipe supplies lut: "kodak" → ctx.lut == "kodak"."""
        rec = {"lut": "kodak"}
        ctx = build_context("test", rec, {})
        assert ctx.lut == "kodak"

    def test_auto_exposure_true_from_recipe(self):
        """Recipe supplies auto_exposure: True → ctx.auto_exposure is True."""
        rec = {"auto_exposure": True}
        ctx = build_context("test", rec, {})
        assert ctx.auto_exposure is True

    def test_auto_exposure_false_from_recipe(self):
        """Recipe supplies auto_exposure: False → ctx.auto_exposure is False."""
        rec = {"auto_exposure": False}
        ctx = build_context("test", rec, {})
        assert ctx.auto_exposure is False

    def test_color_transfer_intensity_from_recipe(self):
        """Recipe supplies color_transfer_intensity: 0.5 → ctx.color_transfer_intensity == 0.5."""
        rec = {"color_transfer_intensity": 0.5}
        ctx = build_context("test", rec, {})
        assert ctx.color_transfer_intensity == 0.5


class TestCallerOverridesWinOverRecipe:
    """Test that caller overrides win over recipe values for post-effects."""

    def test_halation_caller_overrides_recipe(self):
        """Caller override wins: recipe 0.25, override 0.6 → ctx.halation == 0.6."""
        rec = {"halation": 0.25}
        overrides = {"halation": 0.6}
        ctx = build_context("test", rec, overrides)
        assert ctx.halation == 0.6

    def test_grain_caller_overrides_recipe(self):
        """Caller override wins: recipe 0.05, override 0.1 → ctx.grain == 0.1."""
        rec = {"grain": 0.05}
        overrides = {"grain": 0.1}
        ctx = build_context("test", rec, overrides)
        assert ctx.grain == 0.1

    def test_lut_caller_overrides_recipe(self):
        """Caller override wins: recipe "kodak", override "fuji" → ctx.lut == "fuji"."""
        rec = {"lut": "kodak"}
        overrides = {"lut": "fuji"}
        ctx = build_context("test", rec, overrides)
        assert ctx.lut == "fuji"

    def test_auto_exposure_caller_false_overrides_recipe_true(self):
        """Caller override False wins over recipe True → ctx.auto_exposure is False."""
        rec = {"auto_exposure": True}
        overrides = {"auto_exposure": False}
        ctx = build_context("test", rec, overrides)
        assert ctx.auto_exposure is False

    def test_auto_exposure_caller_true_overrides_recipe_false(self):
        """Caller override True wins over recipe False → ctx.auto_exposure is True."""
        rec = {"auto_exposure": False}
        overrides = {"auto_exposure": True}
        ctx = build_context("test", rec, overrides)
        assert ctx.auto_exposure is True

    def test_color_transfer_intensity_caller_overrides_recipe(self):
        """Caller override wins: recipe 0.5, override 0.8 → ctx.color_transfer_intensity == 0.8."""
        rec = {"color_transfer_intensity": 0.5}
        overrides = {"color_transfer_intensity": 0.8}
        ctx = build_context("test", rec, overrides)
        assert ctx.color_transfer_intensity == 0.8


class TestPostEffectsHistoricalDefaults:
    """Test that absent recipe + caller values fall back to historical defaults."""

    def test_halation_default_is_none(self):
        """Empty recipe and no override → ctx.halation is None."""
        ctx = build_context("test", {}, {})
        assert ctx.halation is None

    def test_grain_default_is_none(self):
        """Empty recipe and no override → ctx.grain is None."""
        ctx = build_context("test", {}, {})
        assert ctx.grain is None

    def test_lut_default_is_none(self):
        """Empty recipe and no override → ctx.lut is None."""
        ctx = build_context("test", {}, {})
        assert ctx.lut is None

    def test_auto_exposure_default_is_false(self):
        """Empty recipe and no override → ctx.auto_exposure is False."""
        ctx = build_context("test", {}, {})
        assert ctx.auto_exposure is False

    def test_color_transfer_intensity_default_is_one(self):
        """Empty recipe and no override → ctx.color_transfer_intensity == 1.0."""
        ctx = build_context("test", {}, {})
        assert ctx.color_transfer_intensity == 1.0


class TestBuiltInRecipesPostEffects:
    """Test that built-in recipes carry the expected post-effect values."""

    def test_creative_grade_v1_post_effects(self):
        """creative_grade_v1 recipe supplies halation, grain, lut."""
        rec = resolve_recipe("creative_grade_v1")
        ctx = build_context("creative_grade_v1", rec, {})
        assert ctx.halation == 0.25
        assert ctx.grain == 0.05
        assert ctx.lut == "kodak"

    def test_auto_clean_v1_post_effects(self):
        """auto_clean_v1 recipe supplies auto_exposure."""
        rec = resolve_recipe("auto_clean_v1")
        ctx = build_context("auto_clean_v1", rec, {})
        assert ctx.auto_exposure is True

    def test_matsuri_glow_v1_post_effects(self):
        """matsuri_glow_v1 keeps its reviewed, restrained event finish."""
        rec = resolve_recipe("matsuri_glow_v1")
        ctx = build_context("matsuri_glow_v1", rec, {})
        assert ctx.halation == 0.08
        assert ctx.grain == 0.03
        assert ctx.mv2_ombre is False
