"""Tests for the Astia Fuji film simulation preset (Phase 1.c)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from retouch.grading import load_preset


PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"
ASTIA_PATH = PRESETS_DIR / "astia.json"

FUJI_SIM_NAMES = ("astia", "provia", "classic_chrome")


def _load_raw() -> Dict[str, Any]:
    """Read the raw JSON file directly (no grader round-trip)."""
    with open(ASTIA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_all_fuji_sims() -> Dict[str, Dict[str, Any]]:
    """Return name -> settings for any sibling Fuji sims that exist on disk.

    Missing sims are silently skipped so this test stays green during the
    staggered rollout of Phase 1.c (Astia lands first; Provia may already
    exist; Classic Chrome may not yet).
    """
    sims: Dict[str, Dict[str, Any]] = {}
    for name in FUJI_SIM_NAMES:
        p = PRESETS_DIR / f"{name}.json"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                sims[name] = json.load(f)
    return sims


class TestAstiaFile:
    """Sanity checks for the raw astia.json artifact on disk."""

    def test_preset_file_exists(self):
        assert ASTIA_PATH.exists(), f"Missing preset file: {ASTIA_PATH}"

    def test_preset_file_is_valid_json(self):
        data = _load_raw()
        assert isinstance(data, dict)
        assert data, "Astia preset is empty"

    def test_preset_name_and_description(self):
        data = _load_raw()
        assert data.get("name") == "Astia"
        desc = data.get("description", "")
        assert isinstance(desc, str) and len(desc) > 0
        assert "portrait" in desc.lower() or "skin" in desc.lower()


class TestAstiaLoading:
    """Verify the preset is discoverable through the standard loader."""

    def test_load_preset_returns_astia(self):
        preset = load_preset("astia")
        assert isinstance(preset, dict)
        assert preset.get("name") == "Astia"

    def test_load_preset_via_list_available(self):
        from retouch.grading import list_available_presets
        assert "astia" in list_available_presets()

    def test_load_preset_via_load_all(self):
        from retouch.grading import load_all_presets
        all_presets = load_all_presets()
        assert "astia" in all_presets
        assert isinstance(all_presets["astia"], dict)


class TestAstiaDesignConstraints:
    """The values must satisfy the Phase 1.c Astia design brief."""

    def test_skin_protect_strength_is_high(self):
        """Astia is the skin-friendly portrait sim; protection must be very strong."""
        data = _load_raw()
        assert "skin_protect_strength" in data, "skin_protect_strength key missing"
        val = float(data["skin_protect_strength"])
        assert val >= 0.8, (
            f"skin_protect_strength must be >= 0.8 for Astia (got {val})"
        )

    def test_skin_protect_strength_is_highest_among_fuji_sims(self):
        """Astia must have the highest skin_protect_strength of all 3 Fuji sims.

        Skipped silently if Provia / Classic Chrome have not been merged yet
        (Phase 1.c is staggered across multiple workers)."""
        data = _load_raw()
        astia_skin = float(data["skin_protect_strength"])
        siblings = _load_all_fuji_sims()
        for name, sim in siblings.items():
            if name == "astia":
                continue
            other_skin = float(sim.get("skin_protect_strength", 0.0))
            assert astia_skin >= other_skin, (
                f"Astia skin_protect_strength ({astia_skin}) must be >= "
                f"{name} ({other_skin})"
            )

    def test_warm_midtones(self):
        """Slight warm bias overall (Astia warms skin, +0.05 is the design target)."""
        data = _load_raw()
        warmth = float(data.get("warmth", 0.0))
        assert warmth > 0.0, f"warmth must be > 0 for Astia (got {warmth})"
        assert warmth <= 0.15, f"warmth must stay gentle (got {warmth})"

    def test_grain_is_very_subtle(self):
        """Astia is a portrait sim; grain must be barely visible."""
        data = _load_raw()
        grain = float(data.get("grain_strength", 1.0))
        assert grain < 0.15, f"grain_strength must be < 0.15 for Astia (got {grain})"
        assert grain >= 0.0, f"grain_strength must be >= 0 (got {grain})"

    def test_saturation_is_gentle(self):
        """Slight positive saturation, not punchy."""
        data = _load_raw()
        sat = float(data.get("saturation_boost", 0.0))
        assert 0.0 < sat < 0.15, (
            f"saturation_boost must be in (0, 0.15) for Astia (got {sat})"
        )

    def test_skin_friendly_hsl_signature(self):
        """Astia must exhibit the documented skin-hue signature.

        Per FUJI_COLOR_RESEARCH.md §4.3:
          - "Red saturation is reduced." (reds desaturated by 10-20%)
          - Orange hues are warmed/brightened (skin-brightening move).

        This test is schema-agnostic: it accepts both the nested form
        ``hsl_adjustments.<color>.sat_shift`` and the flat form
        ``hsl_adjustments.saturation.<color>``.
        """
        data = _load_raw()
        hsl = data.get("hsl_adjustments", {}) or {}

        def _read(attr: str, color: str) -> float:
            """Read <attr>.<color> from either nested or flat hsl schema."""
            nested = (hsl.get(color) or {}).get(
                {"hue": "hue_shift", "saturation": "sat_shift", "luminance": "lum_shift"}[attr]
            )
            if nested is not None:
                return float(nested)
            flat = (hsl.get(attr) or {}).get(color)
            return float(flat) if flat is not None else 0.0

        red_sat = _read("saturation", "red")
        assert red_sat <= 0.0, (
            f"Astia must desaturate reds (red.sat <= 0); got {red_sat}"
        )
        assert red_sat >= -30.0, (
            f"red desaturation must stay moderate (>= -30); got {red_sat}"
        )

        orange_lum = _read("luminance", "orange")
        assert orange_lum > 0.0, (
            f"Astia must lift orange luminance to brighten skin (orange.lum > 0); "
            f"got {orange_lum}"
        )

    def test_curves_have_soft_toe_and_shoulder(self):
        """L curve: toe lifted (>0 at input 0) and shoulder rolled off (<255 at input 255)."""
        data = _load_raw()
        curve = data["curves"]["L"]
        assert isinstance(curve, list) and len(curve) >= 2
        points = sorted((int(x), int(y)) for x, y in curve)
        first_in, first_out = points[0]
        last_in, last_out = points[-1]
        assert first_in == 0 and first_out > 0, (
            f"Curve toe should lift shadows (got ({first_in},{first_out}))"
        )
        assert last_in == 255 and last_out < 255, (
            f"Curve shoulder should roll off highlights (got ({last_in},{last_out}))"
        )

    def test_no_external_lut(self):
        """Astia uses parametric color management (no LUT) per research doc §4.5."""
        data = _load_raw()
        assert data.get("lut") in (None, "", "none"), (
            f"lut should be null/empty for Astia (got {data.get('lut')!r})"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
