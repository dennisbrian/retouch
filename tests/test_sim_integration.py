"""Integration tests for the three official Fuji film simulations.

Verifies that the Phase 1.c sims (``classic_chrome``, ``astia``, ``provia``)
are wired into the engine, discoverable via the standard recipe API, and
carry the four new Phase 1.a foundation params (tonal curve, skin
protection, highlight rolloff, film grain).

The sim JSON files in ``presets/`` are tested structurally; the engine
entries in :data:`retouch.recipes.RECIPES` are tested for the same fields
under the recipe-key names that ``build_context`` actually reads
(``skin_protect`` and ``highlight_rolloff`` per ``retouch/params.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from retouch.engine import build_context, resolve_recipe
from retouch.recipes import FUJI_SIM_NAMES, RECIPES, list_fuji_sims


SIM_NAMES = ("classic_chrome", "astia", "provia")

# Top-level keys that each sim must carry in its RECIPES entry.
# These are the recipe-key names from retouch/params.py — the names
# build_context actually reads from a recipe dict.
REQUIRED_RECIPE_KEYS = (
    "tonal_curve_strength",
    "skin_protect",
    "highlight_rolloff",
    "grain_strength",
)


# ---------------------------------------------------------------------------
# Helper: load sim JSON from presets/ (structural-only validation)
# ---------------------------------------------------------------------------


PRESETS_DIR = Path(__file__).resolve().parent.parent / "presets"


def _load_sim_json(name: str) -> Dict[str, Any]:
    """Load the JSON descriptor for a sim, or skip the test if absent."""
    path = PRESETS_DIR / f"{name}.json"
    if not path.exists():
        pytest.skip(f"Sim JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# RECIPES dict membership
# ---------------------------------------------------------------------------


class TestSimRegistration:
    def test_all_three_sims_in_recipes_dict(self):
        for name in SIM_NAMES:
            assert name in RECIPES, f"{name!r} missing from RECIPES"

    def test_provia_in_recipes_dict(self):
        assert "provia" in RECIPES

    def test_astia_in_recipes_dict(self):
        assert "astia" in RECIPES

    def test_classic_chrome_in_recipes_dict(self):
        assert "classic_chrome" in RECIPES

    def test_fuji_sim_names_constant_matches(self):
        assert set(FUJI_SIM_NAMES) == {
            "classic_chrome", "astia", "provia",
            "velvia", "classic_neg", "nostalgic_neg",
            "pro_neg_hi", "pro_neg_std",
            "eterna", "eterna_bleach_bypass",
            "acros", "monochrome", "sepia",
            "reala_ace",
        }


# ---------------------------------------------------------------------------
# Engine recipe resolution
# ---------------------------------------------------------------------------


class TestSimResolution:
    def test_resolve_classic_chrome_returns_non_empty_dict(self):
        resolved = resolve_recipe("classic_chrome")
        assert isinstance(resolved, dict)
        assert len(resolved) > 0

    def test_resolve_astia_returns_non_empty_dict(self):
        resolved = resolve_recipe("astia")
        assert isinstance(resolved, dict)
        assert len(resolved) > 0

    def test_resolve_provia_returns_non_empty_dict(self):
        resolved = resolve_recipe("provia")
        assert isinstance(resolved, dict)
        assert len(resolved) > 0

    def test_resolved_sim_has_required_foundation_keys(self):
        """Every sim must carry the four Phase 1.a foundation keys."""
        for name in SIM_NAMES:
            resolved = resolve_recipe(name)
            for key in REQUIRED_RECIPE_KEYS:
                assert key in resolved, (
                    f"Resolved {name!r} missing required key: {key}"
                )

    def test_resolved_sim_inherits_standard_sections(self):
        """Sims extend ``natural`` so they must resolve the standard
        pipeline sections (frequency, skin, eyes, lips, hair, bloom)."""
        for name in SIM_NAMES:
            resolved = resolve_recipe(name)
            for section in ("frequency", "skin", "eyes", "lips", "hair", "bloom"):
                assert section in resolved, (
                    f"Resolved {name!r} missing standard section: {section}"
                )


# ---------------------------------------------------------------------------
# Foundation param ranges
# ---------------------------------------------------------------------------


class TestSimFoundationParamRanges:
    """Foundation params are 0-1 strengths; verify each sim is in range."""

    @pytest.mark.parametrize("sim_name", SIM_NAMES)
    def test_tonal_curve_strength_in_range(self, sim_name):
        resolved = resolve_recipe(sim_name)
        v = resolved["tonal_curve_strength"]
        assert 0.0 <= v <= 1.0, f"{sim_name}.tonal_curve_strength={v}"

    @pytest.mark.parametrize("sim_name", SIM_NAMES)
    def test_skin_protect_in_range(self, sim_name):
        resolved = resolve_recipe(sim_name)
        v = resolved["skin_protect"]
        assert 0.0 <= v <= 1.0, f"{sim_name}.skin_protect={v}"

    @pytest.mark.parametrize("sim_name", SIM_NAMES)
    def test_highlight_rolloff_in_range(self, sim_name):
        resolved = resolve_recipe(sim_name)
        v = resolved["highlight_rolloff"]
        assert 0.0 <= v <= 1.0, f"{sim_name}.highlight_rolloff={v}"

    @pytest.mark.parametrize("sim_name", SIM_NAMES)
    def test_grain_strength_in_range(self, sim_name):
        resolved = resolve_recipe(sim_name)
        v = resolved["grain_strength"]
        assert 0.0 <= v <= 1.0, f"{sim_name}.grain_strength={v}"

    def test_astia_has_strongest_skin_protection(self):
        """Astia is the portrait specialist — its skin_protect should
        be the highest of the three sims."""
        values = {
            name: resolve_recipe(name)["skin_protect"] for name in SIM_NAMES
        }
        assert values["astia"] == max(values.values()), (
            f"astia should have the highest skin_protect, got {values}"
        )

    def test_provia_has_no_grain(self):
        resolved = resolve_recipe("provia")
        assert resolved["grain_strength"] == 0.0

    def test_classic_chrome_has_grain(self):
        resolved = resolve_recipe("classic_chrome")
        assert resolved["grain_strength"] > 0.0


# ---------------------------------------------------------------------------
# build_context integration
# ---------------------------------------------------------------------------


class TestSimBuildContext:
    """``build_context`` should successfully build a ProcessingContext
    from each sim's resolved recipe (no exceptions, fields populated)."""

    @pytest.mark.parametrize("sim_name", SIM_NAMES)
    def test_build_context_succeeds(self, sim_name):
        rec = resolve_recipe(sim_name)
        ctx = build_context(sim_name, rec, {})
        assert ctx is not None
        assert ctx.active_recipe == sim_name

    @pytest.mark.parametrize("sim_name", SIM_NAMES)
    def test_build_context_populates_foundation_fields(self, sim_name):
        rec = resolve_recipe(sim_name)
        ctx = build_context(sim_name, rec, {})
        assert 0.0 <= ctx.tonal_curve_strength <= 1.0
        assert 0.0 <= ctx.skin_protect_strength <= 1.0
        assert 0.0 <= ctx.highlight_rolloff_strength <= 1.0
        assert 0.0 <= ctx.grain_strength <= 1.0

    def test_build_context_classic_chrome_strong_curve(self):
        rec = resolve_recipe("classic_chrome")
        ctx = build_context("classic_chrome", rec, {})
        assert ctx.tonal_curve_strength > 0.5

    def test_build_context_astia_strong_skin_protect(self):
        rec = resolve_recipe("astia")
        ctx = build_context("astia", rec, {})
        assert ctx.skin_protect_strength > 0.5


# ---------------------------------------------------------------------------
# CLI helper: list_fuji_sims
# ---------------------------------------------------------------------------


class TestListFujiSims:
    def test_list_fuji_sims_returns_list(self):
        result = list_fuji_sims()
        assert isinstance(result, list)

    def test_list_fuji_sims_contains_all_three(self):
        result = list_fuji_sims()
        assert {"classic_chrome", "astia", "provia"}.issubset(set(result))

    def test_list_fuji_sims_contains_all_fourteen(self):
        result = list_fuji_sims()
        assert set(result) == {
            "classic_chrome", "astia", "provia",
            "velvia", "classic_neg", "nostalgic_neg",
            "pro_neg_hi", "pro_neg_std",
            "eterna", "eterna_bleach_bypass",
            "acros", "monochrome", "sepia",
            "reala_ace",
        }

    def test_list_fuji_sims_are_sorted_strings(self):
        result = list_fuji_sims()
        for name in result:
            assert isinstance(name, str)
        assert result == sorted(result)

    def test_list_fuji_sims_no_extras(self):
        """Helper must not include non-Fuji sim recipes (e.g. fuji_porcelain)."""
        result = list_fuji_sims()
        assert "fuji_porcelain" not in result
        assert "natural" not in result


# ---------------------------------------------------------------------------
# JSON preset files (structural validation)
# ---------------------------------------------------------------------------


class TestSimPresetFiles:
    """Verify the JSON files in ``presets/`` carry the foundation params
    that the engine expects. These files are documentation/share-format;
    the engine consumes the in-Python RECIPES entries."""

    @pytest.mark.parametrize("sim_name", SIM_NAMES)
    def test_preset_file_is_valid_json(self, sim_name):
        path = PRESETS_DIR / f"{sim_name}.json"
        if not path.exists():
            pytest.skip(f"Sim JSON not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(data, dict)

    @pytest.mark.parametrize("sim_name", SIM_NAMES)
    def test_preset_has_required_foundation_params(self, sim_name):
        data = _load_sim_json(sim_name)
        # The JSON files use the *_strength naming convention (human-facing).
        for key in (
            "tonal_curve_strength",
            "skin_protect_strength",
            "highlight_rolloff_strength",
            "grain_strength",
        ):
            assert key in data, (
                f"presets/{sim_name}.json missing required key: {key}"
            )

    @pytest.mark.parametrize("sim_name", SIM_NAMES)
    def test_preset_foundation_params_in_range(self, sim_name):
        data = _load_sim_json(sim_name)
        for key in (
            "tonal_curve_strength",
            "skin_protect_strength",
            "highlight_rolloff_strength",
            "grain_strength",
        ):
            v = data[key]
            assert 0.0 <= v <= 1.0, (
                f"presets/{sim_name}.json: {key}={v} not in [0, 1]"
            )
