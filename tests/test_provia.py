"""Tests for the Provia Fuji film simulation preset.

Provia is Fuji's standard / neutral preset. Key contracts:
- Preset JSON is valid and loadable via ``load_preset``.
- ``lut`` is ``null`` (Provia is the only sim that uses no LUT).
- ``grain_strength`` is 0 or extremely low.
- ``skin_protect_strength`` is the lowest of the three sims (< 0.4).
- Saturation is only slightly punched (between -0.1 and 0.15).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from retouch.grading import load_preset


PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"
PROVIA_PATH = PRESETS_DIR / "provia.json"


def test_provia_preset_file_exists():
    """The Provia preset JSON file must exist on disk."""
    assert PROVIA_PATH.exists(), f"Missing preset: {PROVIA_PATH}"


def test_provia_preset_is_valid_json():
    """The Provia preset file must parse as valid JSON."""
    with open(PROVIA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_load_preset_returns_provia_dict():
    """``load_preset('provia')`` returns a dict with expected top-level keys."""
    result = load_preset("provia")
    assert isinstance(result, dict)
    assert "description" in result
    assert "curves" in result
    assert isinstance(result["description"], str)
    assert len(result["description"]) > 0


def test_provia_has_no_lut():
    """Provia is the only film sim without a LUT — preserve original colors."""
    result = load_preset("provia")
    assert "lut" in result, "Provia preset must declare 'lut' key"
    assert result["lut"] is None, (
        f"Provia must use no LUT, got: {result['lut']!r}"
    )


def test_provia_grain_is_minimal():
    """Provia should not add visible grain (strength == 0 or < 0.05)."""
    result = load_preset("provia")
    assert "grain_strength" in result
    grain = float(result["grain_strength"])
    assert grain < 0.05, f"Provia grain must be < 0.05, got {grain}"


def test_provia_skin_protect_is_low():
    """Provia's skin protection must be the lowest of the Fuji sims (< 0.4)."""
    result = load_preset("provia")
    assert "skin_protect_strength" in result
    skin = float(result["skin_protect_strength"])
    assert skin < 0.4, (
        f"Provia skin_protect_strength must be < 0.4 (lowest of sims), got {skin}"
    )


def test_provia_saturation_is_slight():
    """Provia's saturation boost is only slightly punched (-0.1 to 0.15)."""
    result = load_preset("provia")
    assert "saturation_boost" in result
    sat = float(result["saturation_boost"])
    assert -0.1 <= sat <= 0.15, (
        f"Provia saturation_boost must be in [-0.1, 0.15], got {sat}"
    )


def test_provia_tonal_curve_has_linear_middle():
    """The L curve should preserve a linear middle (128 -> ~128)."""
    result = load_preset("provia")
    points = result["curves"]["L"]
    midpoints = [(x, y) for (x, y) in points if 100 <= x <= 160]
    assert midpoints, "Curve must have a midpoint in [100, 160]"
    for x, y in midpoints:
        assert abs(y - x) <= 8, (
            f"Provia curve middle must be near-linear, got ({x}, {y})"
        )


def test_provia_no_shadow_lift():
    """Provia does not lift shadows — accurate reproduction only."""
    result = load_preset("provia")
    shadow_lift = float(result.get("shadow_lift", 0))
    assert shadow_lift == 0.0, (
        f"Provia must not lift shadows, got shadow_lift={shadow_lift}"
    )


def test_provia_warmth_is_neutral():
    """Provia has no intentional color cast — warmth must be ~0."""
    result = load_preset("provia")
    warmth = float(result.get("warmth", 0))
    assert abs(warmth) <= 0.02, (
        f"Provia warmth must be neutral (|w| <= 0.02), got {warmth}"
    )


def test_provia_has_name_field():
    """Preset should declare a name field for identification."""
    result = load_preset("provia")
    assert result.get("name") == "Provia", (
        f"Provia preset name must be 'Provia', got {result.get('name')!r}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
