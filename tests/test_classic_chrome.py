"""Tests for the Classic Chrome Fuji film simulation preset.

Verifies that the preset file exists, is valid JSON, loads via
``retouch.grading.load_preset``, and contains the field values that
encode the Classic Chrome design intent (low saturation, lifted blacks,
cool greens, warm highlight split-tone, fine grain).
"""

import json
from pathlib import Path

import pytest

from retouch.grading import PRESETS, load_preset


PRESET_PATH = Path(__file__).resolve().parent.parent / "presets" / "classic_chrome.json"
PRESET_NAME = "classic_chrome"


# ---------------------------------------------------------------------------
# File / JSON validity
# ---------------------------------------------------------------------------

class TestPresetFile:
    def test_preset_file_exists(self):
        assert PRESET_PATH.exists(), f"Preset file not found: {PRESET_PATH}"

    def test_preset_file_is_valid_json(self):
        data = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    def test_preset_has_description(self):
        data = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
        assert "description" in data
        assert isinstance(data["description"], str)
        assert len(data["description"]) > 0
        assert "Classic Chrome" in data["description"]

    def test_preset_has_name_field(self):
        data = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
        assert "name" in data
        assert data["name"] == "Classic Chrome"


# ---------------------------------------------------------------------------
# Loader integration
# ---------------------------------------------------------------------------

class TestPresetLoading:
    def test_load_preset_returns_dict(self):
        result = load_preset(PRESET_NAME)
        assert isinstance(result, dict)

    def test_load_preset_matches_file_contents(self):
        result = load_preset(PRESET_NAME)
        expected = json.loads(PRESET_PATH.read_text(encoding="utf-8"))
        assert result == expected

    def test_preset_registered_in_presets_dict(self):
        assert PRESET_NAME in PRESETS

    def test_presets_dict_entry_is_dict(self):
        assert isinstance(PRESETS[PRESET_NAME], dict)


# ---------------------------------------------------------------------------
# Required fields present
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = (
    "tonal_curve_strength",
    "skin_protect_strength",
    "highlight_rolloff_strength",
    "grain_strength",
    "lut",
    "curves",
    "calibration",
    "hsl_adjustments",
    "saturation_boost",
    "warmth",
    "split_tone_three_way",
    "shadow_lift",
    "vignette",
)


class TestRequiredFields:
    @pytest.fixture
    def preset(self):
        return load_preset(PRESET_NAME)

    @pytest.mark.parametrize("field", REQUIRED_FIELDS)
    def test_field_present(self, preset, field):
        assert field in preset, f"Missing required field: {field}"


# ---------------------------------------------------------------------------
# Engine-level Fuji foundation parameters
# ---------------------------------------------------------------------------

class TestFujiFoundationParams:
    @pytest.fixture
    def preset(self):
        return load_preset(PRESET_NAME)

    def test_tonal_curve_strength_in_unit_range(self, preset):
        v = preset["tonal_curve_strength"]
        assert isinstance(v, (int, float))
        assert 0.0 <= v <= 1.0
        assert v >= 0.5, "Classic Chrome needs a strong S-curve (>= 0.5)"

    def test_skin_protect_strength_in_unit_range(self, preset):
        v = preset["skin_protect_strength"]
        assert isinstance(v, (int, float))
        assert 0.0 <= v <= 1.0
        assert v >= 0.3, "Skin protect should be moderate-to-strong to guard skin during desaturation"

    def test_highlight_rolloff_strength_in_unit_range(self, preset):
        v = preset["highlight_rolloff_strength"]
        assert isinstance(v, (int, float))
        assert 0.0 <= v <= 1.0
        assert v > 0.0, "Classic Chrome rolls off highlights for the 'creamy' look"

    def test_grain_strength_in_unit_range(self, preset):
        v = preset["grain_strength"]
        assert isinstance(v, (int, float))
        assert 0.0 <= v <= 1.0
        assert 0.05 <= v <= 0.30, "Classic Chrome grain is subtle (0.05-0.30)"

    def test_lut_is_non_empty_string(self, preset):
        v = preset["lut"]
        assert isinstance(v, str)
        assert v.strip() != ""
        assert v.lower() != "none"

    def test_lut_points_to_known_lut(self, preset):
        from retouch.lut import list_available_luts
        available = list_available_luts()
        assert preset["lut"] in available, (
            f"LUT {preset['lut']!r} not found. Available: {available}"
        )


# ---------------------------------------------------------------------------
# Tonal curve: lifted blacks + gentle highlight rolloff
# ---------------------------------------------------------------------------

class TestTonalCurve:
    @pytest.fixture
    def curve_l(self):
        return load_preset(PRESET_NAME)["curves"]["L"]

    def test_curve_has_anchor_points(self, curve_l):
        assert isinstance(curve_l, list)
        assert len(curve_l) >= 3
        for pt in curve_l:
            assert isinstance(pt, (list, tuple))
            assert len(pt) == 2

    def test_curve_lifted_blacks(self, curve_l):
        first_in, first_out = curve_l[0]
        assert first_in == 0
        assert 0 < first_out <= 20, (
            f"Black point {first_out} should be lifted (>0) but not crushed"
        )

    def test_curve_preserves_white_point(self, curve_l):
        last_in, last_out = curve_l[-1]
        assert last_in == 255
        assert last_out <= 255
        assert last_out >= 230, "Highlight end should be gentle (rolled off, not crushed)"

    def test_curve_has_s_shape(self, curve_l):
        """Classic Chrome = strong S-curve. The midtone should sit slightly
        above the identity line, and the shoulder should fall below it."""
        points = {int(p[0]): int(p[1]) for p in curve_l}
        assert 128 in points
        mid = points[128]
        assert mid >= 128, f"Midpoint {mid} should be at/above identity for S-curve"

    def test_curve_monotonic(self, curve_l):
        """Tone curve must be monotonically non-decreasing."""
        for (x0, y0), (x1, y1) in zip(curve_l, curve_l[1:]):
            assert x0 < x1, f"X must increase: {x0} -> {x1}"
            assert y0 <= y1, f"Y must be non-decreasing: ({x0},{y0}) -> ({x1},{y1})"


# ---------------------------------------------------------------------------
# Color design: low saturation, cool greens, desaturated reds
# ---------------------------------------------------------------------------

class TestColorDesign:
    @pytest.fixture
    def preset(self):
        return load_preset(PRESET_NAME)

    def test_saturation_boost_is_negative(self, preset):
        """The defining characteristic of Classic Chrome: LOW saturation."""
        v = preset["saturation_boost"]
        assert isinstance(v, (int, float))
        assert v < 0.0, f"saturation_boost must be negative, got {v}"
        assert -0.30 <= v <= -0.05, (
            f"saturation_boost {v} outside expected Classic Chrome range [-0.30, -0.05]"
        )

    def test_warmth_is_neutral_to_cool(self, preset):
        """Research: slight cool cast (-150 to -300K WB shift)."""
        v = preset["warmth"]
        assert isinstance(v, (int, float))
        assert v <= 0.05, f"warmth {v} should be neutral-to-cool"

    def test_shadow_lift_positive(self, preset):
        """Lifted blacks are a defining editorial feature."""
        v = preset["shadow_lift"]
        assert isinstance(v, (int, float))
        assert v > 0, f"shadow_lift must be > 0 for lifted blacks, got {v}"

    def test_calibration_has_all_three_hues(self, preset):
        cal = preset["calibration"]
        for hue in ("red", "green", "blue"):
            assert hue in cal, f"calibration missing {hue}"
            assert "hue" in cal[hue]
            assert "sat" in cal[hue]

    def test_calibration_green_rotates_toward_teal(self, preset):
        """The most brand-distinctive part of Classic Chrome:
        greens pushed toward teal/cyan."""
        green_hue = preset["calibration"]["green"]["hue"]
        green_sat = preset["calibration"]["green"]["sat"]
        assert green_hue > 150, f"green hue {green_hue} should rotate toward teal (>150)"
        assert green_sat < 0, f"green sat {green_sat} should be desaturated"

    def test_calibration_red_is_desaturated(self, preset):
        """Reds slightly desaturated for the editorial look."""
        red_sat = preset["calibration"]["red"]["sat"]
        assert red_sat < 0, f"red sat {red_sat} should be desaturated"

    def test_hsl_adjustments_desaturate_reds_and_oranges(self, preset):
        """Skin-hue protection: desaturate reds and oranges."""
        hsl = preset["hsl_adjustments"]
        assert hsl["red"]["sat_shift"] < 0
        assert hsl["orange"]["sat_shift"] < 0

    def test_hsl_adjustments_push_green_toward_teal(self, preset):
        hsl = preset["hsl_adjustments"]
        green_hue_shift = hsl["green"]["hue_shift"]
        assert green_hue_shift > 0, (
            f"green hue_shift {green_hue_shift} should rotate toward teal (positive)"
        )


# ---------------------------------------------------------------------------
# Three-way split toning
# ---------------------------------------------------------------------------

class TestSplitToneThreeWay:
    @pytest.fixture
    def preset(self):
        return load_preset(PRESET_NAME)

    def test_has_all_three_zones(self, preset):
        st = preset["split_tone_three_way"]
        for zone in ("shadows", "midtones", "highlights"):
            assert zone in st, f"split_tone_three_way missing {zone}"
            assert "hue" in st[zone]
            assert "sat" in st[zone]

    def test_shadows_are_cool(self, preset):
        """Editorial moody look: cool shadows."""
        shadows = preset["split_tone_three_way"]["shadows"]
        assert 150 <= shadows["hue"] <= 240, (
            f"shadow hue {shadows['hue']} should be in cool range [150, 240]"
        )
        assert shadows["sat"] > 0, "shadow sat should be > 0 for a visible cast"

    def test_highlights_are_warm(self, preset):
        """Magazine look: warm highlights."""
        highlights = preset["split_tone_three_way"]["highlights"]
        assert 10 <= highlights["hue"] <= 50, (
            f"highlight hue {highlights['hue']} should be warm [10, 50]"
        )
        assert highlights["sat"] > 0, "highlight sat should be > 0 for a visible cast"

    def test_midtones_are_near_neutral(self, preset):
        mid = preset["split_tone_three_way"]["midtones"]
        assert abs(mid["sat"]) < 10, f"midtone sat {mid['sat']} should be near-neutral"


# ---------------------------------------------------------------------------
# Vignette
# ---------------------------------------------------------------------------

class TestVignette:
    def test_vignette_is_mild(self):
        preset = load_preset(PRESET_NAME)
        v = preset["vignette"]
        assert 0.0 < v <= 0.25, f"vignette {v} should be mild editorial strength"
