"""Tests for gui_smart.py — Smart Color card handlers (T2a).

Only tests the pure handler functions (apply_color_macros,
on_smart_mode_toggle). GUI component wiring itself is exercised indirectly
by tests/test_gui.py::TestProcessInputKeys-style drift guards passing
unchanged after gui.py's T2a edit (proving Classic component identity is
undisturbed) -- not re-tested here.
"""

import gradio as gr
import pytest

import gui_smart
from retouch.smart_intents import COLOR_INTENT


class TestApplyColorMacros:
    def _call(self, amount=0.0, warmth=0.0, contrast=0.0, **base_overrides):
        bases = dict(
            base_vibrance=0.0,
            base_saturation=0.0,
            base_kelvin=6500.0,
            base_tint=0.0,
            base_contrast=0.0,
            base_highlights=0.0,
            base_shadows=0.0,
        )
        bases.update(base_overrides)
        return gui_smart.apply_color_macros(amount, warmth, contrast, **bases)

    def test_all_zero_amounts_returns_bases_unchanged(self):
        result = self._call(amount=0.0, warmth=0.0, contrast=0.0)
        values = [update["value"] for update in result]
        assert values == [0.0, 0.0, 6500.0, 0.0, 0.0, 0.0, 0.0]

    def test_returns_seven_gr_updates_in_fixed_order(self):
        result = self._call()
        assert len(result) == 7
        for update in result:
            assert isinstance(update, dict)
            assert "value" in update

    def test_amount_one_moves_vibrance_and_saturation_to_target_positive(self):
        amount_macro = next(m for m in COLOR_INTENT.macro_controls if m.macro_id == "amount")
        vibrance_write, saturation_write = amount_macro.writes
        result = self._call(amount=1.0)
        assert result[0]["value"] == vibrance_write.target_positive
        assert result[1]["value"] == saturation_write.target_positive
        # warmth/contrast outputs (kelvin/tint/contrast/highlights/shadows) untouched
        assert result[2]["value"] == 6500.0
        assert result[4]["value"] == 0.0

    def test_warmth_negative_one_moves_kelvin_and_tint_to_target_negative(self):
        warmth_macro = next(m for m in COLOR_INTENT.macro_controls if m.macro_id == "warmth")
        kelvin_write, tint_write = warmth_macro.writes
        result = self._call(warmth=-1.0)
        assert result[2]["value"] == kelvin_write.target_negative
        assert result[3]["value"] == tint_write.target_negative

    def test_contrast_macro_moves_three_owned_params(self):
        contrast_macro = next(m for m in COLOR_INTENT.macro_controls if m.macro_id == "contrast")
        contrast_write, highlights_write, shadows_write = contrast_macro.writes
        result = self._call(contrast=1.0)
        assert result[4]["value"] == contrast_write.target_positive
        assert result[5]["value"] == highlights_write.target_positive
        assert result[6]["value"] == shadows_write.target_positive

    def test_base_values_are_read_not_guessed(self):
        # Non-zero, non-default bases must be respected at amount=0.
        result = self._call(
            amount=0.0, warmth=0.0, contrast=0.0,
            base_vibrance=17.0, base_kelvin=7200.0, base_contrast=-5.0,
        )
        assert result[0]["value"] == 17.0
        assert result[2]["value"] == 7200.0
        assert result[4]["value"] == -5.0


class TestOnSmartModeToggle:
    def test_classic_hides_smart_group(self):
        result = gui_smart.on_smart_mode_toggle("Classic")
        assert result["visible"] is False

    def test_smart_color_preview_shows_smart_group(self):
        result = gui_smart.on_smart_mode_toggle("Smart Color (preview)")
        assert result["visible"] is True
