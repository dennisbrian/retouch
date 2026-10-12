"""Tests for new parameter wiring (Q3 redness_even, Q4 whiten_hue_stable)."""

from __future__ import annotations

import pytest


class TestParamRegistry:
    """Verify new params exist in registry."""

    def test_redness_even_param_exists(self):
        from retouch.params import _SKIN_PARAMS
        from retouch.engine import _DEFAULTS
        names = [p.name for p in _SKIN_PARAMS]
        assert "redness_even" in names
        assert _DEFAULTS["redness_even"] == 0

    def test_whiten_hue_stable_param_exists(self):
        from retouch.params import _SKIN_PARAMS
        from retouch.engine import _DEFAULTS
        names = [p.name for p in _SKIN_PARAMS]
        assert "whiten_hue_stable" in names
        assert _DEFAULTS["whiten_hue_stable"] == 0

    def test_redness_even_recipe_key(self):
        from retouch.params import _SKIN_PARAMS
        ps = [p for p in _SKIN_PARAMS if p.name == "redness_even"]
        assert len(ps) == 1
        assert ps[0].recipe_key == "skin.redness_even"
        assert ps[0].min_val == 0
        assert ps[0].max_val == 100

    def test_whiten_hue_stable_recipe_key(self):
        from retouch.params import _SKIN_PARAMS
        ps = [p for p in _SKIN_PARAMS if p.name == "whiten_hue_stable"]
        assert len(ps) == 1
        assert ps[0].recipe_key == "skin.whiten_hue_stable"
        assert ps[0].conversion == "bool_flag"

    def test_both_params_in_skin_list(self):
        from retouch.params import _SKIN_PARAMS
        names = [p.name for p in _SKIN_PARAMS]
        assert "redness_even" in names
        assert "whiten_hue_stable" in names


class TestProcessingContext:
    """Verify ProcessingContext has new fields."""

    def test_context_has_redness_even(self):
        from retouch.engine import ProcessingContext
        ctx = ProcessingContext()
        assert hasattr(ctx, "redness_even")
        assert ctx.redness_even == 0

    def test_context_has_whiten_hue_stable(self):
        from retouch.engine import ProcessingContext
        ctx = ProcessingContext()
        assert hasattr(ctx, "whiten_hue_stable")
        assert ctx.whiten_hue_stable == 0

    def test_context_has_micro_dodge_burn(self):
        from retouch.engine import ProcessingContext
        ctx = ProcessingContext()
        assert hasattr(ctx, "micro_dodge_burn")
        assert ctx.micro_dodge_burn == 0.0

    def test_context_has_skinhueunify(self):
        from retouch.engine import ProcessingContext
        ctx = ProcessingContext()
        assert hasattr(ctx, "skin_hue_unify")

    def test_context_has_skinchromaeven(self):
        from retouch.engine import ProcessingContext
        ctx = ProcessingContext()
        assert hasattr(ctx, "skin_chroma_even")


class TestGUIWiring:
    """Verify GUI _process_inputs and PROCESS_INPUT_KEYS include new params."""

    def test_process_input_keys_count(self):
        import gui
        assert len(gui.PROCESS_INPUT_KEYS) >= 92

    def test_redness_even_in_keys(self):
        import gui
        assert "redness_even" in gui.PROCESS_INPUT_KEYS

    def test_whiten_hue_stable_in_keys(self):
        import gui
        assert "whiten_hue_stable" in gui.PROCESS_INPUT_KEYS

    def test_micro_dodge_burn_in_keys(self):
        import gui
        assert "micro_dodge_burn" in gui.PROCESS_INPUT_KEYS

    def test_recipe_keys_count_updated(self):
        from tests.test_gui import EXPECTED_RECIPE_KEY_COUNT, EXPECTED_RECIPE_KEYS
        assert EXPECTED_RECIPE_KEY_COUNT >= 83
        assert EXPECTED_RECIPE_KEY_COUNT == len(EXPECTED_RECIPE_KEYS)
