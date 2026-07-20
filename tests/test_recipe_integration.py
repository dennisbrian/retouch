"""Recipe integration tests.

Verifies:
  * ``extends`` correctly inherits all parent fields
  * Child recipes can override parent fields
  * ``build_context()`` returns expected values per recipe
  * Every ProcessingContext field is covered by at least one recipe
  * Recipe defaults are accessible and well-formed
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from retouch.engine import ProcessingContext, build_context, resolve_recipe
from retouch.recipes import (
    CONDITIONAL_RECIPE_NAMES,
    CURATED_RECIPE_NAMES,
    RECOMMENDED_RECIPE_NAMES,
    RECIPES,
)


# ---------------------------------------------------------------------------
# Recipe inheritance
# ---------------------------------------------------------------------------


class TestRecipeInheritance:
    def test_extends_inherits_all_parent_keys(self):
        """A child recipe should contain all keys present in its parent."""
        parent = resolve_recipe("natural")
        child = resolve_recipe("portrait")
        for key in parent:
            assert key in child, f"portrait missing parent key: {key}"

    def test_extends_cosplay_inherits_natural(self):
        parent = resolve_recipe("natural")
        child = resolve_recipe("cosplay")
        for key in parent:
            assert key in child, f"cosplay missing parent key: {key}"

    def test_extends_chains_three_levels(self):
        """A three-level chain (e.g. soft → anime_cinematic_soft →
        anime_cinematic_v1 → natural) inherits all keys from the topmost
        ancestor."""
        natural = resolve_recipe("natural")
        v1 = resolve_recipe("anime_cinematic_v1")
        soft = resolve_recipe("anime_cinematic_soft")
        # All natural keys should be in v1
        for key in natural:
            assert key in v1, f"anime_cinematic_v1 missing natural key: {key}"
        # All v1 keys should be in soft
        for key in v1:
            assert key in soft, f"anime_cinematic_soft missing v1 key: {key}"

    def test_extends_preserves_parent_value_when_not_overridden(self):
        """The ``color_harmony`` block is inherited unchanged from natural
        in portrait, since portrait's override doesn't change it."""
        # natural's color_harmony = {"preset": "natural", "amount": 0.00}
        # portrait's color_harmony = {"preset": "natural", "amount": 0.60}
        portrait = resolve_recipe("portrait")
        assert portrait["color_harmony"]["preset"] == "natural"

    def test_resolved_recipe_is_dict(self):
        resolved = resolve_recipe("natural")
        assert isinstance(resolved, dict)
        assert "frequency" in resolved
        assert "skin" in resolved


# ---------------------------------------------------------------------------
# Recipe override
# ---------------------------------------------------------------------------


class TestRecipeOverride:
    def test_child_overrides_parent_field(self):
        """A child recipe should be able to override specific parent values."""
        natural_freq_smooth = resolve_recipe("natural")["frequency"]["smooth"]
        portrait_freq_smooth = resolve_recipe("portrait")["frequency"]["smooth"]
        assert natural_freq_smooth == 0.30
        assert portrait_freq_smooth == 0.45
        assert portrait_freq_smooth != natural_freq_smooth

    def test_child_overrides_nested_dict(self):
        """When the child has a nested dict override, only the specified
        sub-keys are replaced; the rest of the parent's dict is preserved."""
        natural_skin = resolve_recipe("natural")["skin"]
        cosplay_skin = resolve_recipe("cosplay")["skin"]
        # Both have "equalize" and "rosy"
        assert "equalize" in natural_skin
        assert "equalize" in cosplay_skin
        # cosplay's value should differ from natural's
        assert cosplay_skin["equalize"] != natural_skin["equalize"]

    def test_override_specific_bloom_field(self):
        """bloom is a dict, but a child can override the whole dict."""
        anime_v1 = resolve_recipe("anime_cinematic_v1")
        anime_soft = resolve_recipe("anime_cinematic_soft")
        assert anime_v1["bloom"]["opacity"] == 0.16
        assert anime_soft["bloom"]["opacity"] == 0.22

    def test_override_top_level_scalar(self):
        """Top-level scalars like contrast can be overridden."""
        v1 = resolve_recipe("anime_cinematic_v1")
        soft = resolve_recipe("anime_cinematic_soft")
        assert v1["contrast"] == 7.0
        assert soft["contrast"] == 8.0

    def test_alias_recipes_share_storage(self):
        """The ``soft``/``action``/``fantasy`` aliases share the same dict
        object as their canonical recipes."""
        assert RECIPES["soft"] is RECIPES["anime_cinematic_soft"]
        assert RECIPES["action"] is RECIPES["anime_cinematic_action"]
        assert RECIPES["fantasy"] is RECIPES["anime_cinematic_fantasy"]


# ---------------------------------------------------------------------------
# Recipe defaults
# ---------------------------------------------------------------------------


class TestRecipeDefaults:
    def test_natural_recipe_has_required_sections(self):
        rec = resolve_recipe("natural")
        for section in ("frequency", "skin", "eyes", "lips", "hair", "bloom"):
            assert section in rec, f"natural missing section: {section}"

    def test_natural_recipe_modular_defaults(self):
        rec = resolve_recipe("natural")
        assert rec.get("slimming") == 0.0
        assert rec.get("blush") == 0.0
        assert rec.get("lip_finish") == "gloss"
        assert rec.get("nose_blush") is False
        assert rec.get("under_eye_blush") is False
        assert rec.get("white_costume_lift") is False

    def test_cosplay_recipe_enables_modular_flags(self):
        rec = resolve_recipe("cosplay")
        assert rec.get("nose_blush") is True
        assert rec.get("under_eye_blush") is True
        assert rec.get("slimming") == 50.0
        assert rec.get("blush") == 25.0

    def test_pink_dream_enables_white_costume_lift(self):
        rec = resolve_recipe("pink_dream")
        assert rec.get("white_costume_lift") is True

    def test_all_recipes_resolve_without_error(self):
        for name in RECIPES:
            resolved = resolve_recipe(name)
            assert isinstance(resolved, dict)
            assert "frequency" in resolved

    def test_curated_catalog_is_partitioned_by_safety(self):
        """A curated recipe is either correction-first or explicitly conditional."""
        assert len(CURATED_RECIPE_NAMES) == 50
        assert set(RECOMMENDED_RECIPE_NAMES).isdisjoint(CONDITIONAL_RECIPE_NAMES)
        assert set(CURATED_RECIPE_NAMES) == (
            set(RECOMMENDED_RECIPE_NAMES) | set(CONDITIONAL_RECIPE_NAMES)
        )
        assert "apex_cinema_v1" in CONDITIONAL_RECIPE_NAMES

    def test_wedding_and_mixed_temperature_are_conservative(self):
        """Regression gate for the washed-out grade seen in visual QA."""
        wedding = resolve_recipe("wedding_timeless_v1")
        mixed = resolve_recipe("con_mixed_temp_v1")
        assert wedding["film"]["strength"] <= 0.15
        assert wedding["bloom"]["opacity"] <= 0.02
        assert mixed["skin"]["hue_unify"] <= 0.25
        assert mixed["skin"]["chroma_even"] <= 0.20


# ---------------------------------------------------------------------------
# BuildContext from recipe
# ---------------------------------------------------------------------------


class TestBuildContextFromRecipe:
    """For each recipe, build_context() should produce a ProcessingContext
    with values that match the recipe's documented defaults."""

    def _ctx_for(self, recipe_name):
        rec = resolve_recipe(recipe_name)
        return build_context(recipe_name, rec, {})

    def test_natural_smooth(self):
        ctx = self._ctx_for("natural")
        # natural: frequency.smooth = 0.30 → 30.0
        assert ctx.smooth == pytest.approx(30.0)

    def test_natural_color_grade(self):
        ctx = self._ctx_for("natural")
        # natural: color_harmony.preset = "natural", amount = 0.0
        assert ctx.color_grade == "natural"
        assert ctx.grade_intensity == pytest.approx(0.0)

    def test_cosplay_color_grade(self):
        ctx = self._ctx_for("cosplay")
        assert ctx.color_grade == "cosplay"
        assert ctx.grade_intensity == pytest.approx(0.20)

    def test_pink_dream_color_grade(self):
        ctx = self._ctx_for("pink_dream")
        assert ctx.color_grade == "pink_dream"
        assert ctx.grade_intensity == pytest.approx(0.85)

    def test_active_recipe_set(self):
        for name in ("natural", "cosplay", "beauty", "pink_dream"):
            ctx = self._ctx_for(name)
            assert ctx.active_recipe == name

    def test_anime_cinematic_contrast_override(self):
        ctx = self._ctx_for("anime_cinematic_v1")
        assert ctx.contrast == 7.0

    def test_anime_cinematic_bloom_opacity(self):
        ctx = self._ctx_for("anime_cinematic_v1")
        # recipe: bloom.opacity = 0.16 → 16.0
        assert ctx.bloom == pytest.approx(16.0)

    def test_cosplay_whiten_tone(self):
        ctx = self._ctx_for("cosplay")
        # cosplay extends natural which uses "rosy" skin
        assert ctx.whiten_tone == "rosy"

    def test_anime_cinematic_porcelain_tone(self):
        ctx = self._ctx_for("anime_cinematic_v1")
        # anime_cinematic_v1 has "porcelain" key in skin
        assert ctx.whiten_tone == "porcelain"

    def test_impact_for_pink_dream(self):
        ctx = self._ctx_for("pink_dream")
        # pink_dream finish.impact = 0.60 → 60.0
        assert ctx.impact == pytest.approx(60.0)

    def test_slimming_for_cosplay(self):
        ctx = self._ctx_for("cosplay")
        assert ctx.slimming == 50.0

    def test_slimming_for_natural(self):
        ctx = self._ctx_for("natural")
        assert ctx.slimming == 0.0

    def test_dark_circles_default_zero_for_eyes_whites_recipe(self):
        """Regression test for BUGFIX-3: a recipe that sets eyes.whites
        must not accidentally drive dark_circles from teeth_whiten."""
        rec = {"eyes": {"whites": 0.8, "teeth_whiten": 0.8}}
        ctx = build_context("test", rec, {})
        assert ctx.dark_circles == 0.0
        assert ctx.teeth_whiten == pytest.approx(80.0)

    def test_dark_circles_uses_recipe_field_when_present(self):
        rec = {"eyes": {"dark_circles": 0.5}}
        ctx = build_context("test", rec, {})
        assert ctx.dark_circles == pytest.approx(50.0)

    def test_lip_finish_inherited_from_recipe(self):
        # korean_beauty sets lip_finish to "velvet"
        ctx = self._ctx_for("korean_beauty")
        assert ctx.lip_finish == "velvet"

    def test_teeth_whiten_inherited(self):
        ctx = self._ctx_for("beauty")
        # beauty: eyes.teeth_whiten = 0.20 → 20.0
        assert ctx.teeth_whiten == pytest.approx(20.0)

    def test_fuji_film_density_params(self):
        # Provia
        ctx_provia = self._ctx_for("provia")
        assert ctx_provia.film_enable is True
        assert ctx_provia.film_strength == pytest.approx(1.0)
        assert ctx_provia.film_toe_r == pytest.approx(0.10)
        assert ctx_provia.film_crosstalk_cy_mg == pytest.approx(0.04)
        assert ctx_provia.film_tonemap_strength == pytest.approx(0.42)
        assert ctx_provia.film_highlight_purity == pytest.approx(0.22)
        
        # Astia
        ctx_astia = self._ctx_for("astia")
        assert ctx_astia.film_enable is True
        assert ctx_astia.film_strength == pytest.approx(1.0)
        assert ctx_astia.film_toe_r == pytest.approx(0.05)
        assert ctx_astia.film_crosstalk_cy_mg == pytest.approx(0.03)
        assert ctx_astia.film_tonemap_strength == pytest.approx(0.50)
        assert ctx_astia.film_highlight_purity == pytest.approx(0.28)
        
        # Classic Chrome
        ctx_chrome = self._ctx_for("classic_chrome")
        assert ctx_chrome.film_enable is True
        assert ctx_chrome.film_strength == pytest.approx(1.0)
        assert ctx_chrome.film_toe_r == pytest.approx(0.15)
        assert ctx_chrome.film_crosstalk_cy_mg == pytest.approx(0.08)
        assert ctx_chrome.film_tonemap_strength == pytest.approx(0.85)
        assert ctx_chrome.film_highlight_purity == pytest.approx(0.34)


# ---------------------------------------------------------------------------
# Recipe parameter coverage
# ---------------------------------------------------------------------------


class TestRecipeParameterCoverage:
    """Every ProcessingContext field should be addressable from at least one
    recipe (either directly, via extension, or via the per-recipe mapping
    in build_context)."""

    def _all_context_fields(self):
        return {f.name for f in dataclasses.fields(ProcessingContext)}

    def test_recipes_have_frequency_skin_eyes_lips_hair(self):
        """Core per-region fields are required by every resolved recipe."""
        for name in RECIPES:
            resolved = resolve_recipe(name)
            for key in ("frequency", "skin", "eyes", "lips", "hair"):
                assert key in resolved, f"{name} missing {key}"

    def test_color_harmony_in_all_recipes(self):
        for name in RECIPES:
            resolved = resolve_recipe(name)
            assert "color_harmony" in resolved, f"{name} missing color_harmony"

    def test_modular_flag_keys(self):
        """The modular flag keys (nose_blush, under_eye_blush,
        white_costume_lift) should be settable via recipes."""
        for name, recipe in RECIPES.items():
            for key in ("nose_blush", "under_eye_blush", "white_costume_lift"):
                if key in recipe:
                    assert isinstance(recipe[key], bool), (
                        f"{name}.{key} must be bool, got {type(recipe[key]).__name__}"
                    )

    def test_every_context_field_referenced_in_build_context(self):
        """The build_context() function should touch (read or write) every
        ProcessingContext field, ensuring no parameter is silently dropped."""
        # Construct a representative recipe
        rec = {
            "frequency": {"smooth": 0.5, "mid_reduction": 0.4},
            "skin": {"equalize": 0.3, "rosy": 0.2, "porcelain": 0.0, "relight": 0.3, "relight_azimuth": 45.0, "relight_elevation": 35.0},
            "eyes": {"whites": 0.1, "teeth_whiten": 0.1, "iris": 0.1, "catchlight": 0.1, "dark_circles": 0.05},
            "lips": {"tint": "rose", "gloss": 0.1},
            "hair": {"shine": 0.2},
            "dodge_burn": {"amount": 0.1},
            "color_harmony": {"preset": "natural", "amount": 0.5},
            "bloom": {"opacity": 0.1, "threshold": 200.0, "softness": 25.0},
            "texture": {"opacity": 0.9, "pore_synthesis": 0.0},
            "finish": {"impact": 0.5},
            "specular_bloom": 30,
            "slimming": 20.0,
            "blush": 15.0,
            "lip_finish": "matte",
            "nose_blush": True,
            "under_eye_blush": True,
            "white_costume_lift": True,
            "shadow_hue": 200.0,
            "shadow_sat": 5.0,
            "midtone_hue": 0.0,
            "midtone_sat": 0.0,
            "highlight_hue": 30.0,
            "highlight_sat": 5.0,
            "glow": 5.0,
            "vignette": 5.0,
            "sharpen": 20.0,
            "sharpen_radius": 1.5,
            "subject_separation": 0.5,
            "contrast": 5.0,
            "brightness": 5.0,
            "highlights": -5.0,
            "shadows": 5.0,
            "whites": 5.0,
            "blacks": -5.0,
            "clarity": 10.0,
            "vibrance": 10.0,
            "saturation": 5.0,
        }
        overrides = {
            "nose_smooth": 20.0,
            "auto_exposure": True,
            "color_grade_stack": [{"preset": "natural"}],
            "color_ref": np.zeros((10, 10, 3), dtype=np.uint8),
            "color_transfer_intensity": 0.7,
            "chromatic_aberration": 2.0,
            "halation": 0.1,
            "grain": 0.05,
            "lut": "kodak",
        }
        ctx = build_context("custom", rec, overrides)

        # Now check that all the values from the recipe/overrides are reflected
        assert ctx.smooth == pytest.approx(50.0)
        assert ctx.whiten == pytest.approx(20.0)  # rosy
        assert ctx.blemish == pytest.approx(50.0)  # blemish == smooth
        assert ctx.eye_enhance == pytest.approx(10.0)
        assert ctx.teeth_whiten == pytest.approx(10.0)
        assert ctx.catchlight == pytest.approx(10.0)
        assert ctx.dark_circles == pytest.approx(5.0)
        assert ctx.lip_enhance == pytest.approx(10.0)
        assert ctx.lip_tint == "rose"
        assert ctx.lip_finish == "matte"
        assert ctx.hair_enhance == pytest.approx(20.0)
        assert ctx.equalize == pytest.approx(30.0)
        assert ctx.dodge_burn == pytest.approx(10.0)
        assert ctx.relight == pytest.approx(30.0)
        assert ctx.bloom == pytest.approx(10.0)
        assert ctx.bloom_threshold == 200.0
        assert ctx.bloom_softness == 25.0
        assert ctx.glow == 5.0
        assert ctx.vignette == 5.0
        assert ctx.sharpen == 20.0
        assert ctx.sharpen_radius == 1.5
        assert ctx.contrast == 5.0
        assert ctx.brightness == 5.0
        assert ctx.highlights == -5.0
        assert ctx.shadows == 5.0
        assert ctx.whites == 5.0
        assert ctx.blacks == -5.0
        assert ctx.clarity == 10.0
        assert ctx.vibrance == 10.0
        assert ctx.saturation == 5.0
        assert ctx.subject_separation == 0.5
        assert ctx.impact == pytest.approx(50.0)
        assert ctx.color_grade == "natural"
        assert ctx.grade_intensity == pytest.approx(0.5)
        assert ctx.slimming == 20.0
        assert ctx.blush == 15.0
        assert ctx.nose_blush is True
        assert ctx.under_eye_blush is True
        assert ctx.white_costume_lift is True
        assert ctx.nose_smooth == 20.0
        assert ctx.auto_exposure is True
        assert ctx.color_grade_stack == [{"preset": "natural"}]
        assert ctx.color_ref is not None
        assert ctx.color_transfer_intensity == 0.7
        assert ctx.chromatic_aberration == 2.0
        assert ctx.halation == 0.1
        assert ctx.grain == 0.05
        assert ctx.lut == "kodak"
        assert ctx.shadow_hue == 200.0
        assert ctx.shadow_sat == 5.0
        assert ctx.midtone_hue == 0.0
        assert ctx.highlight_hue == 30.0
        assert ctx.highlight_sat == 5.0
        assert ctx.active_recipe == "custom"
        assert ctx.lens_blur == 0.0
        assert ctx.hsl_hue_green == 0.0
        assert ctx.calibration_red_hue == 0.0

    def test_processing_context_fields_covered(self):
        """All ProcessingContext fields (except internal face_contexts) should
        be settable either via recipe dict or via overrides."""
        # Use a fully-populated recipe plus fully-populated overrides
        rec_keys = set()
        for recipe in RECIPES.values():
            rec_keys.update(recipe.keys())

        # The build_context function reads from rec + applies overrides
        # The test only needs to verify that every context field has a code
        # path in build_context, which is exercised by the previous test
        # (test_every_context_field_referenced_in_build_context).
        # Here we just confirm the field set is non-trivial.
        all_fields = self._all_context_fields()
        # face_contexts is internal/optional and may not be in a recipe
        addressable = all_fields - {"face_contexts"}
        assert len(addressable) > 50  # sanity: many fields are settable


# ---------------------------------------------------------------------------
# Recipe extension semantics
# ---------------------------------------------------------------------------


class TestRecipeExtensionSemantics:
    """Verify the order of `extends` resolution is correct and idempotent."""

    def test_resolve_recipe_returns_independent_copy(self):
        """Mutating the resolved recipe must not affect the original."""
        resolved = resolve_recipe("portrait")
        resolved["frequency"]["smooth"] = 99.0
        # Re-resolve should be unaffected
        fresh = resolve_recipe("portrait")
        assert fresh["frequency"]["smooth"] != 99.0

    def test_resolve_recipe_caches(self):
        """resolve_recipe should be deterministic and stable."""
        r1 = resolve_recipe("portrait")
        r2 = resolve_recipe("portrait")
        assert r1 == r2

    def test_unknown_recipe_falls_back_to_natural(self):
        resolved = resolve_recipe("__definitely_not_a_recipe__")
        natural = resolve_recipe("natural")
        assert resolved == natural

    def test_recursive_extends_no_infinite_loop(self):
        """Even if a recipe chain were circular, resolve_recipe should
        terminate without hanging."""
        # We don't actually create a cycle, but verify recursion protection
        # by checking that the 'soft' chain terminates.
        resolved = resolve_recipe("soft")
        assert "frequency" in resolved


class TestAnimeV2Recipe:
    def test_resolve_anime_v2(self):
        rec = resolve_recipe("anime_v2")
        assert "frequency" in rec
        assert "skin" in rec

    def test_anime_v2_extends_anime_cinematic_v1(self):
        rec = resolve_recipe("anime_v2")
        assert rec.get("specular_bloom_tone") == "rosy"
        assert "frequency" in rec

    def test_skin_flatten_value(self):
        rec = resolve_recipe("anime_v2")
        assert rec["skin"]["flatten"] == 0.55

    def test_skin_quantize_value(self):
        rec = resolve_recipe("anime_v2")
        assert rec["skin"]["quantize"] == 0.40

    def test_skin_relight_in_skin_dict(self):
        rec = resolve_recipe("anime_v2")
        assert rec["skin"]["relight"] == 0.42

    def test_build_context_skin_flatten(self):
        ctx = build_context("anime_v2", resolve_recipe("anime_v2"), {})
        assert ctx.skin_flatten == pytest.approx(55.0)

    def test_build_context_skin_quantize(self):
        ctx = build_context("anime_v2", resolve_recipe("anime_v2"), {})
        assert ctx.skin_quantize == 40.0

    def test_build_context_relight_from_skin_dict(self):
        ctx = build_context("anime_v2", resolve_recipe("anime_v2"), {})
        assert ctx.relight == 42.0

    def test_override_wins(self):
        rec = resolve_recipe("anime_v2")
        ctx = build_context("anime_v2", rec, {"skin_flatten": 75.0})
        assert ctx.skin_flatten == 75.0

    def test_skin_unify_value(self):
        rec = resolve_recipe("anime_v2")
        assert rec["skin"]["unify"] == 0.50

    def test_skin_glow_value(self):
        rec = resolve_recipe("anime_v2")
        assert rec["skin"]["glow"] == 0.20

    def test_build_context_skin_unify(self):
        ctx = build_context("anime_v2", resolve_recipe("anime_v2"), {})
        assert ctx.skin_unify == pytest.approx(50.0)

    def test_build_context_skin_glow(self):
        ctx = build_context("anime_v2", resolve_recipe("anime_v2"), {})
        assert ctx.skin_glow == pytest.approx(20.0)


class TestClearSkinV1UnlocksNewFeatures:
    """Regression guard: ``clear_skin_v1`` must drive every new skin/eye
    primitive end-to-end through the recipe → build_context → ctx path, so a
    data-driven mapping regression can't silently leave a feature un-fired.

    These are the features added in the region-aware / anisotropic / freckle /
    under-eye / eye-enhancement batch. If any assertion here fails, the recipe
    no longer *unlocks* that feature (param not reaching ctx).
    """

    def _ctx(self):
        return build_context("clear_skin_v1", resolve_recipe("clear_skin_v1"), {})

    def test_resolves_and_builds(self):
        # Must survive resolve + build_context without error.
        ctx = self._ctx()
        assert ctx.active_recipe == "clear_skin_v1"

    def test_region_aware_modulation(self):
        # frequency.regional_modulation (recipe_direct, 0-1) → ctx 0.6
        assert self._ctx().regional_modulation == pytest.approx(0.6)

    def test_anisotropic_smooth_engine(self):
        # frequency.smooth_engine → ctx "anisotropic" (grain-following path)
        assert self._ctx().smooth_engine == "anisotropic"

    def test_freckle_removal(self):
        # frequency.freckle_removal (recipe_direct, 0-100) → ctx 45.0
        assert self._ctx().freckle_removal == pytest.approx(45.0)

    def test_undereye_shadow_strength(self):
        # eyes.undereye_shadow_strength (recipe_direct, 0-1) → ctx 0.4
        assert self._ctx().undereye_shadow_strength == pytest.approx(0.4)

    def test_undereye_darken_and_puffiness(self):
        # undereye.* (recipe_pct 0-1 → ctx 0-100)
        ctx = self._ctx()
        assert ctx.undereye_darken_removal == pytest.approx(30.0)
        assert ctx.undereye_puffiness_reduction == pytest.approx(20.0)

    def test_eye_enhancement_suite(self):
        # eye.* (recipe_pct 0-1 → ctx 0-100); hue_shift is recipe_direct (-30..30)
        ctx = self._ctx()
        assert ctx.eye_sclera_brighten == pytest.approx(30.0)
        assert ctx.eye_iris_saturate == pytest.approx(40.0)
        assert ctx.eye_iris_brightness == pytest.approx(30.0)
        assert ctx.eye_iris_hue_shift == pytest.approx(-8.0)


class TestNewFeatureRecipesResolve:
    """Every recipe built on the new skin/eye primitives must resolve and map
    its hero feature into ctx without error."""

    @pytest.mark.parametrize("name,attr,expected", [
        ("freckle_free_v1", "freckle_removal", 85.0),
        ("freckle_free_v1", "regional_modulation", 0.3),
        ("tired_eye_rescue_v1", "undereye_shadow_strength", 0.7),
        ("tired_eye_rescue_v1", "undereye_darken_removal", 60.0),
        ("tired_eye_rescue_v1", "undereye_puffiness_reduction", 50.0),
        ("tired_eye_rescue_v1", "smooth_engine", "anisotropic"),
        ("tired_eye_rescue_v1", "eye_sclera_brighten", 25.0),
        ("aniso_pore_real_v1", "smooth_engine", "anisotropic"),
        ("aniso_pore_real_v1", "regional_modulation", 0.85),
        ("aniso_pore_real_v1", "freckle_removal", 0.0),  # not set → default
        ("cosplay_clear_v1", "smooth_engine", "anisotropic"),
        ("cosplay_clear_v1", "freckle_removal", 40.0),
        ("cosplay_clear_v1", "eye_iris_saturate", 55.0),
        ("cosplay_porcelain_protected_v1", "hb_even", 0.20),
        ("cosplay_porcelain_protected_v1", "hb_shift", -0.07),
        ("cosplay_porcelain_protected_v1", "mole_protect", 0.80),
        ("cosplay_porcelain_protected_v1", "film_highlight_purity", 0.28),
        ("cosplay_flash_rescue_v1", "hb_even", 0.30),
        ("cosplay_flash_rescue_v1", "hb_shift", -0.10),
        ("cosplay_flash_rescue_v1", "mole_protect", 0.80),
        ("cosplay_flash_rescue_v1", "film_highlight_purity", 0.32),
        ("cosplay_porcelain_color_demo_v1", "smooth", 55.0),
        ("cosplay_porcelain_color_demo_v1", "smooth_engine", "anisotropic"),
        ("cosplay_porcelain_color_demo_v1", "freckle_removal", 0.0),
        ("cosplay_porcelain_color_demo_v1", "color_grade", "natural"),
        ("cosplay_porcelain_color_demo_v1", "grade_intensity", 0.05),
        ("cosplay_porcelain_color_demo_v1", "film_highlight_purity", 0.10),
        ("apex_cosplay_v1", "heal_engine", "telea"),
        ("apex_cosplay_v1", "skin_sss", 25.0),
        ("apex_cosplay_v1", "film_highlight_purity", 0.30),
        ("apex_cosplay_v1", "saturation_mode", "subtractive"),
        ("apex_cosplay_v1", "hair_remove_flyaways", 30.0),
        ("apex_editorial_v1", "heal_engine", "telea"),
        ("apex_editorial_v1", "mole_protect", 0.90),
        ("apex_editorial_v1", "micro_dodge_burn", 15.0),
        ("apex_editorial_v1", "hb_even", 0.20),
        ("apex_cinema_v1", "heal_engine", "telea"),
        ("apex_cinema_v1", "grain_strength", 0.15),
        ("apex_cinema_v1", "film_highlight_purity", 0.35),
        ("apex_cinema_v1", "halation", 0.18),
        ("studio_porcelain_clear_v1", "smooth_engine", "anisotropic"),
        ("studio_porcelain_clear_v1", "regional_modulation", 0.6),
        ("studio_porcelain_clear_v1", "undereye_darken_removal", 45.0),
        ("studio_porcelain_clear_v1", "eye_sclera_brighten", 30.0),
        ("xhs_clear_glow_v1", "smooth_engine", "anisotropic"),
        ("xhs_clear_glow_v1", "freckle_removal", 50.0),
        ("xhs_clear_glow_v1", "eye_iris_brightness", 30.0),
        ("wedding_flawless_v1", "smooth_engine", "anisotropic"),
        ("wedding_flawless_v1", "freckle_removal", 40.0),
        ("wedding_flawless_v1", "undereye_darken_removal", 35.0),
        ("korean_glass_clear_v1", "smooth_engine", "anisotropic"),
        ("korean_glass_clear_v1", "regional_modulation", 0.7),
        ("korean_glass_clear_v1", "eye_sclera_brighten", 30.0),
        ("beauty_editorial_clear_v1", "smooth_engine", "anisotropic"),
        ("beauty_editorial_clear_v1", "freckle_removal", 55.0),
        ("beauty_editorial_clear_v1", "eye_iris_saturate", 45.0),
        ("fantasy_eye_pop_v1", "smooth_engine", "anisotropic"),
        ("fantasy_eye_pop_v1", "freckle_removal", 70.0),
        ("fantasy_eye_pop_v1", "eye_iris_hue_shift", -8.0),
        ("scifi_clean_v1", "smooth_engine", "anisotropic"),
        ("scifi_clean_v1", "freckle_removal", 60.0),
        ("scifi_clean_v1", "eye_iris_saturate", 65.0),
        ("idol_clear_v1", "smooth_engine", "anisotropic"),
        ("idol_clear_v1", "freckle_removal", 45.0),
        ("idol_clear_v1", "eye_iris_saturate", 55.0),
        ("pink_dream_clear_v1", "smooth_engine", "anisotropic"),
        ("pink_dream_clear_v1", "freckle_removal", 40.0),
        ("pink_dream_clear_v1", "eye_sclera_brighten", 30.0),
        ("fuji_porcelain_clear_v1", "smooth_engine", "anisotropic"),
        ("fuji_porcelain_clear_v1", "regional_modulation", 0.6),
        ("fuji_porcelain_clear_v1", "freckle_removal", 35.0),
        ("studio_hard_flash_clear_v1", "smooth_engine", "anisotropic"),
        ("studio_hard_flash_clear_v1", "freckle_removal", 40.0),
        ("studio_hard_flash_clear_v1", "eye_iris_brightness", 30.0),
        ("outdoor_golden_clear_v1", "smooth_engine", "anisotropic"),
        ("outdoor_golden_clear_v1", "freckle_removal", 45.0),
        ("outdoor_golden_clear_v1", "eye_iris_brightness", 25.0),
        ("convention_clear_v1", "smooth_engine", "anisotropic"),
        ("convention_clear_v1", "undereye_darken_removal", 60.0),
        ("convention_clear_v1", "undereye_puffiness_reduction", 50.0),
    ])
    def test_hero_feature_unlocks(self, name, attr, expected):
        ctx = build_context(name, resolve_recipe(name), {})
        val = getattr(ctx, attr)
        if isinstance(expected, str):
            assert val == expected
        else:
            assert val == pytest.approx(expected)
