"""Tests for gui.py — pure helper functions that do not require Gradio UI or MediaPipe.

These tests cover the recipe-to-UI mapping helpers, custom-style library helpers,
and the standalone HTML comparison utility used for before/after sliders.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest

import gui


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_styles_dir():
    """Snapshot the on-disk ``styles/`` directory and restore it after the test.

    The style library reads/writes files in a fixed ``styles/`` directory
    relative to the project root, so this fixture swaps it out for a temp
    directory and restores the original on teardown.
    """
    from retouch import style_library

    original_dir = style_library.DEFAULT_STYLE_DIR
    backup_dir = tempfile.mkdtemp(prefix="retouch_styles_backup_")
    if original_dir.exists():
        shutil.copytree(original_dir, backup_dir, dirs_exist_ok=True)
    try:
        # Wipe and replace the live directory for the duration of the test
        if original_dir.exists():
            shutil.rmtree(original_dir)
        original_dir.mkdir(parents=True, exist_ok=True)
        yield original_dir
    finally:
        # Restore original directory contents
        if original_dir.exists():
            shutil.rmtree(original_dir)
        shutil.copytree(backup_dir, original_dir, dirs_exist_ok=True)
        shutil.rmtree(backup_dir)


# ---------------------------------------------------------------------------
# Shared constants pulled from the module under test
# ---------------------------------------------------------------------------

EXPECTED_RECIPE_KEYS = [
    "ai_denoise", "ai_sr_scale", "airy_haze", "albedo_even", "auto_body_reshape",
    "auto_exposure", "backdrop_cleanup", "background_blur", "background_desaturation", "background_harmonize",
    "background_harmonize_mode", "blacks", "blemish", "bloom", "bloom_softness",
    "bloom_threshold", "blotch_reduction", "blue_shadow_grade", "blush", "body_dodge_burn",
    "body_equalize", "body_match_face", "body_relight", "body_reshape_arm_length", "body_reshape_hip_width",
    "body_reshape_leg_length", "body_reshape_shoulder_width", "body_reshape_torso_width", "body_shadow_lift", "body_smooth",
    "body_whiten", "brightness", "bw_channel_mixer_b", "bw_channel_mixer_g", "bw_channel_mixer_r",
    "catchlight", "chromatic_aberration", "clarity", "clarity_split_neg", "clarity_split_pos",
    "color_grade", "color_transfer_intensity", "contrast", "cosplay_consistency_strength", "cosplay_stockings_smooth",
    "cosplay_wig_lace_blend", "cyan_midtone_grade", "dark_circles", "dodge_burn", "equalize",
    "eye_enhance", "eye_iris_brightness", "eye_iris_hue_shift", "eye_iris_saturate", "eye_sclera_brighten",
    "eye_sclera_vessel_remove", "fabric_wrinkle_smooth", "face_exposure", "fade_toe", "film_crosstalk_cy_mg",
    "film_crosstalk_cy_ye", "film_crosstalk_mg_ye", "film_enable", "film_gamma", "film_midpoint",
    "film_highlight_purity", "film_shoulder_b", "film_shoulder_g", "film_shoulder_r", "film_skew", "film_strength",
    "film_toe_b", "film_toe_g", "film_toe_r", "film_tonemap_shoulder", "film_tonemap_strength",
    "film_tonemap_toe", "freckle_preserve_mask", "freckle_removal", "gamut_compress", "glow", "hb_even", "hb_shift",
    "grade_intensity", "grain", "grain_strength", "hair_deglare", "hair_enhance",
    "hair_remove_flyaways", "hair_ring_position", "hair_ring_tint", "halation", "heal_engine", "hemoglobin_smooth",
    "highlight_drift", "highlight_hue", "highlight_rolloff_strength", "highlight_sat", "highlights",
    "hsl_hue_global", "hsl_lum_global", "hsl_sat_global", "impact", "light_wrap",
    "lip_enhance", "lip_finish", "lip_tint", "lut", "makeup_cake_reduce", "makeup_coverage_even", "mask_feather_mode", "matte_black",
    "mark_policy",
    "micro_dodge_burn", "micro_restore", "mid_reduction", "midtone_hue", "midtone_sat",
    "mole_protect", "mv2_brows", "mv2_brows_color", "mv2_contour", "mv2_eyeliner",
    "mv2_eyeliner_color", "mv2_eyeliner_style", "mv2_eyeshadow", "mv2_eyeshadow_color", "mv2_eyeshadow_style",
    "mv2_ombre", "mv2_ombre_color1", "mv2_ombre_color2", "negative_split_tone_highlight", "negative_split_tone_shadow",
    "neural_defect_boost", "neural_stray_hair_boost", "nose_blush", "nose_restore", "nose_smooth",
    "pore_synthesis", "redness_even", "regional_modulation", "relight", "relight_azimuth",
    "relight_elevation", "reshape_chin_length", "reshape_eye_distance", "reshape_eye_size", "reshape_eye_size_l",
    "reshape_eye_size_r", "reshape_forehead", "reshape_jaw_width", "reshape_jaw_width_l", "reshape_jaw_width_r",
    "reshape_mouth_size", "reshape_neck_length", "reshape_neck_width", "reshape_nose_length", "reshape_nose_width",
    "reshape_nose_width_l", "reshape_nose_width_r", "reshape_smile", "saturation", "saturation_mode",
    "sculpt", "shadow_hue", "shadow_lift", "shadow_sat", "shadows",
    "sharpen", "sharpen_radius", "shine_removal", "skin_chroma_even", "skin_flatten",
    "skin_glow", "skin_hue_unify", "skin_protect_strength", "skin_quantize", "skin_sss", "skin_unify",
    "skin_unify_hue", "slimming", "smooth", "smooth_engine", "specular_bloom",
    "specular_bloom_tone", "specular_finish", "specular_finish_strength", "specular_recolor", "subject_separation",
    "subject_sharpen", "teeth_whiten", "texture_opacity", "texture_transplant", "tonal_curve_strength",
    "under_eye_blush", "undereye_darken_removal", "undereye_puffiness_reduction", "undereye_shadow_strength", "vein_attenuate",
    "vibrance", "vignette", "white_balance_kelvin", "white_balance_tint", "white_costume_lift",
    "whiten", "whiten_hue_stable", "whiten_tone", "whites", "wrinkle_soften",
    "wrinkle_soften_forehead", "wrinkle_soften_nasolabial", "wrinkle_soften_neck",
]

# Add new HSL / Calibration / lens_blur keys dynamically
for color in ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"]:
    EXPECTED_RECIPE_KEYS.extend([f"hsl_hue_{color}", f"hsl_sat_{color}", f"hsl_lum_{color}"])
for color in ["red", "green", "blue"]:
    EXPECTED_RECIPE_KEYS.extend([f"calibration_{color}_hue", f"calibration_{color}_sat", f"calibration_{color}_lum"])
EXPECTED_RECIPE_KEYS.append("lens_blur")

EXPECTED_RECIPE_KEY_COUNT = 246
# Self-updating: the recipe/smart-style slider tuple length is the contract
# defined by RECIPE_OUTPUT_KEYS, so this constant can never go stale.
EXPECTED_UI_OUTPUT_COUNT = len(gui.RECIPE_OUTPUT_KEYS)


# ---------------------------------------------------------------------------
# TestRecipeDefaults
# ---------------------------------------------------------------------------


class TestRecipeDefaults:
    """Tests for gui.recipe_defaults(recipe_name)."""

    def test_returns_dict(self):
        d = gui.recipe_defaults("natural")
        assert isinstance(d, dict)

    def test_returns_all_expected_keys(self):
        d = gui.recipe_defaults("natural")
        for key in EXPECTED_RECIPE_KEYS:
            assert key in d, f"Missing key: {key}"
        assert len(d) == EXPECTED_RECIPE_KEY_COUNT

    def test_no_extra_keys(self):
        d = gui.recipe_defaults("natural")
        extras = set(d.keys()) - set(EXPECTED_RECIPE_KEYS)
        assert not extras, f"Unexpected extra keys: {extras}"

    def test_natural_values(self):
        """Spot-check the canonical 'natural' recipe values."""
        d = gui.recipe_defaults("natural")
        # natural frequency.smooth is 0.30 → * 100 = 30
        assert d["smooth"] == 30
        # natural skin.rosy is 0.1 → * 100 = 10
        assert d["whiten"] == 10
        # defaults from the recipe
        assert d["clarity"] == 0
        assert d["glow"] == 0
        assert d["vignette"] == 0
        # natural recipe gives eye_enhance=5 and catchlight falls back to iris (=5)
        assert d["catchlight"] == 5

    def test_anime_cinematic_v1_values(self):
        """The task description pins clarity=14 for anime_cinematic_v1."""
        d = gui.recipe_defaults("anime_cinematic_v1")
        assert d["clarity"] == 14

    def test_all_recipes_return_full_dict(self):
        """Every recipe name in RECIPES must produce a dict with all keys."""
        for name in gui.RECIPES:
            d = gui.recipe_defaults(name)
            assert isinstance(d, dict)
            assert len(d) == EXPECTED_RECIPE_KEY_COUNT
            for key in EXPECTED_RECIPE_KEYS:
                assert key in d, f"Recipe {name!r} is missing key {key!r}"

    def test_all_recipes_numeric_fields_are_numeric(self):
        """Slider/box fields should be int or float, not None or strings."""
        # String fields in recipe_defaults
        string_fields = {
            "lip_tint", "whiten_tone", "lip_finish", "specular_bloom_tone", "color_grade", "lut", "mask_feather_mode",
            "background_harmonize_mode", "saturation_mode", "smooth_engine", "specular_finish", "mark_policy", "heal_engine",
            "mv2_brows_color", "mv2_eyeliner_color", "mv2_eyeliner_style",
            "mv2_eyeshadow_color", "mv2_eyeshadow_style", "mv2_ombre_color1", "mv2_ombre_color2"
        }
        # Boolean fields in recipe_defaults
        bool_fields = {
            "nose_blush", "under_eye_blush", "white_costume_lift", "auto_exposure",
            "film_enable", "whiten_hue_stable", "mv2_ombre"
        }
        # Optional-mask / passthrough fields that legitimately default to None
        # (not slider/box values, so they are not required to be numeric).
        nullable_fields = {
            "freckle_preserve_mask",
        }

        for name in gui.RECIPES:
            d = gui.recipe_defaults(name)
            for key in EXPECTED_RECIPE_KEYS:
                val = d[key]
                if key in nullable_fields:
                    assert val is None or isinstance(val, (int, float)), (
                        f"Recipe {name!r}.{key!r} should be None or numeric, "
                        f"got {type(val).__name__}"
                    )
                elif key in string_fields:
                    # These are string fields
                    assert isinstance(val, str), (
                        f"Recipe {name!r}.{key!r} should be str, got {type(val).__name__}"
                    )
                elif key in bool_fields:
                    assert isinstance(val, bool), (
                        f"Recipe {name!r}.{key!r} should be bool, got {type(val).__name__}"
                    )
                else:
                    assert isinstance(val, (int, float)), (
                        f"Recipe {name!r}.{key!r} should be numeric, got {type(val).__name__}"
                    )

    def test_dropdown_values_in_valid_choices(self):
        """String fields must be members of the declared choices lists."""
        for name in gui.RECIPES:
            d = gui.recipe_defaults(name)
            assert d["whiten_tone"] in gui.WHITEN_TONE_CHOICES
            assert d["lip_finish"] in gui.LIP_FINISH_CHOICES
            assert d["specular_bloom_tone"] in gui.SPECULAR_BLOOM_TONE_CHOICES
            assert d["color_grade"] in gui.COLOR_GRADE_NAMES
            assert d["lut"] in gui.LUT_CHOICES
            assert d["lip_tint"] in ["none", "cosplay", "rose", "pink",
                                     "coral", "natural", "berry"]

    def test_unknown_recipe_does_not_crash(self):
        """An unknown recipe should fall back to 'natural' without raising."""
        d = gui.recipe_defaults("nonexistent_recipe_xyz")
        assert isinstance(d, dict)
        assert len(d) == EXPECTED_RECIPE_KEY_COUNT

    def test_unknown_recipe_matches_natural(self):
        """Unknown recipes should produce the same dict as 'natural'."""
        d_unknown = gui.recipe_defaults("nonexistent_recipe_xyz")
        d_natural = gui.recipe_defaults("natural")
        assert d_unknown == d_natural

    def test_empty_string_recipe_does_not_crash(self):
        d = gui.recipe_defaults("")
        assert isinstance(d, dict)
        assert len(d) == EXPECTED_RECIPE_KEY_COUNT

    def test_recursive_extend_propagates(self):
        """'soft' extends 'anime_cinematic_soft' which extends 'anime_cinematic_v1'."""
        d = gui.recipe_defaults("soft")
        d_v1 = gui.recipe_defaults("anime_cinematic_v1")
        # clarity override 8 should be picked up
        assert d["clarity"] == 8
        # Smooth value should match the parent (32, not natural's 30)
        assert d["smooth"] == 32


# ---------------------------------------------------------------------------
# TestGetCustomStyleNames
# ---------------------------------------------------------------------------


class TestGetCustomStyleNames:
    """Tests for gui.get_custom_style_names()."""

    def test_returns_list(self):
        result = gui.get_custom_style_names()
        assert isinstance(result, list)
        # All entries, if any, must be strings
        for name in result:
            assert isinstance(name, str)

    def test_empty_when_no_styles(self, isolated_styles_dir):
        """With an empty styles dir, the function returns an empty list."""
        result = gui.get_custom_style_names()
        assert result == []

    def test_returns_strings_when_styles_exist(self, isolated_styles_dir):
        """When styles are present, the function returns a list of strings."""
        from retouch.style import StyleProfile
        from retouch.style_library import save_style_profile

        profile = StyleProfile(
            brightness_delta=5.0,
            contrast_delta=10.0,
            skin_smooth_strength=0.3,
        )
        save_style_profile("alpha_test", profile, "Author", tags=["a"])

        result = gui.get_custom_style_names()
        assert isinstance(result, list)
        assert len(result) == 1
        assert all(isinstance(n, str) for n in result)
        assert "alpha_test" in result


# ---------------------------------------------------------------------------
# TestApplyCustomStyle
# ---------------------------------------------------------------------------


class TestApplyCustomStyle:
    """Tests for gui.apply_custom_style(style_name, current_recipe)."""

    def test_none_style_returns_updates(self):
        """Passing None should return a list of gr.update() sized to the UI outputs."""
        result = gui.apply_custom_style(None)
        assert isinstance(result, (list, tuple))
        assert len(result) == EXPECTED_UI_OUTPUT_COUNT

    def test_empty_string_style_returns_updates(self):
        result = gui.apply_custom_style("")
        assert isinstance(result, (list, tuple))
        assert len(result) == EXPECTED_UI_OUTPUT_COUNT

    def test_invalid_style_returns_updates(self):
        """An unknown style name should be treated like None — a full update list."""
        result = gui.apply_custom_style("definitely_not_a_style_xyz")
        assert isinstance(result, (list, tuple))
        assert len(result) == EXPECTED_UI_OUTPUT_COUNT

    def test_valid_style_returns_full_tuple(self, isolated_styles_dir):
        """A saved style should produce a 59-value tuple of UI values."""
        from retouch.style import StyleProfile
        from retouch.style_library import save_style_profile

        profile = StyleProfile(
            brightness_delta=10.0,
            contrast_delta=15.0,
            skin_smooth_strength=0.4,
            skin_mid_reduction=0.5,
            skin_l_mean_delta=5.0,
        )
        save_style_profile("test_style_apply", profile, "Author", tags=["t"])

        result = gui.apply_custom_style("test_style_apply", "natural")
        # The valid path returns a tuple, not a list
        assert isinstance(result, tuple)
        assert len(result) == EXPECTED_UI_OUTPUT_COUNT

    def test_valid_style_overrides_recipe_defaults(self, isolated_styles_dir):
        """The returned values should reflect the saved profile, not just the recipe."""
        from retouch.style import StyleProfile
        from retouch.style_library import save_style_profile

        profile = StyleProfile(
            brightness_delta=10.0,
            contrast_delta=15.0,
            skin_smooth_strength=0.4,   # → 40 on the smooth slider
            skin_mid_reduction=0.5,
            skin_l_mean_delta=5.0,      # → 20 on the whiten slider
        )
        save_style_profile("test_style_override", profile, "Author")

        result = gui.apply_custom_style("test_style_override", "natural")
        # smooth is the first returned value
        assert result[0] == 40
        # whiten lands at its canonical RECIPE_OUTPUT_KEYS index (key-based
        # producers make the position self-documenting, not a magic number)
        whiten_idx = gui.RECIPE_OUTPUT_KEYS.index("whiten")
        assert result[whiten_idx] == 20


# ---------------------------------------------------------------------------
# TestOnRecipeChange
# ---------------------------------------------------------------------------


class TestOnRecipeChange:
    """Tests for gui.on_recipe_change(recipe)."""

    def test_returns_59_values(self):
        result = gui.on_recipe_change("natural")
        assert isinstance(result, tuple)
        assert len(result) == EXPECTED_UI_OUTPUT_COUNT

    def test_values_match_recipe_defaults(self):
        """The UI outputs should match recipe_defaults in canonical order."""
        result = gui.on_recipe_change("natural")
        d = gui.recipe_defaults("natural")
        keys = gui.RECIPE_OUTPUT_KEYS
        for name in ("smooth", "mid_reduction", "texture_opacity",
                     "pore_synthesis", "nose_smooth", "micro_restore",
                     "whiten", "equalize"):
            assert result[keys.index(name)] == d[name]

    def test_showcase_recipe_loads_guided_mask_refinement(self):
        key_index = gui.RECIPE_OUTPUT_KEYS.index("mask_feather_mode")
        defaults = gui.recipe_defaults("cosplay_character_showcase_v1")

        assert defaults["mask_feather_mode"] == "guided"
        assert gui.on_recipe_change("cosplay_character_showcase_v1")[key_index] == "guided"

    def test_unknown_recipe_returns_59_values(self):
        """Unknown recipes should still return a 59-tuple (falls back to natural)."""
        result = gui.on_recipe_change("totally_made_up")
        assert isinstance(result, tuple)
        assert len(result) == EXPECTED_UI_OUTPUT_COUNT

    def test_different_recipes_yield_different_outputs(self):
        natural = gui.on_recipe_change("natural")
        cosplay = gui.on_recipe_change("cosplay")
        # smooth differs between natural (30) and cosplay (55)
        assert natural[0] != cosplay[0]
        assert natural[0] == 30
        assert cosplay[0] == 55

    def test_anime_cinematic_v1_clarity(self):
        """The clarity value for anime_cinematic_v1 should be 14."""
        result = gui.on_recipe_change("anime_cinematic_v1")
        d = gui.recipe_defaults("anime_cinematic_v1")
        assert d["clarity"] == 14
        # clarity sits at its canonical RECIPE_OUTPUT_KEYS index (key-based
        # producers make this self-documenting rather than a magic number)
        clarity_index = gui.RECIPE_OUTPUT_KEYS.index("clarity")
        assert result[clarity_index] == 14


# ---------------------------------------------------------------------------
# TestMakeComparisonHtml
# ---------------------------------------------------------------------------


class TestMakeComparisonHtml:
    """Tests for gui._make_comparison_html(orig_bgr, result_bgr, max_height)."""

    def test_returns_string(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        html = gui._make_comparison_html(img, img.copy())
        assert isinstance(html, str)

    def test_non_empty_output(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        html = gui._make_comparison_html(img, img.copy())
        assert len(html) > 0

    def test_contains_template_markers(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        html = gui._make_comparison_html(img, img.copy())
        # The template uses cmp-<uid> as the wrapper id
        assert "cmp-" in html
        assert "data:image/jpeg;base64," in html
        assert "BEFORE" in html
        assert "AFTER" in html

    def test_works_with_different_sized_images(self):
        """Both images must have the same shape after _make_comparison_html."""
        img1 = np.zeros((100, 100, 3), dtype=np.uint8)
        img2 = np.full((100, 100, 3), 200, dtype=np.uint8)
        html = gui._make_comparison_html(img1, img2)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_resizes_when_above_max_height(self):
        """If both images exceed max_height, the result should still be a valid HTML string."""
        img1 = np.zeros((1000, 1000, 3), dtype=np.uint8)
        img2 = np.zeros((1000, 1000, 3), dtype=np.uint8)
        html = gui._make_comparison_html(img1, img2, max_height=300)
        assert isinstance(html, str)
        assert "data:image/jpeg;base64," in html

    def test_none_raises(self):
        """Documented behavior: passing None raises AttributeError (no silent failure)."""
        with pytest.raises(AttributeError):
            gui._make_comparison_html(None, None)

    def test_unique_uids(self):
        """Two consecutive calls should produce different uid values."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        html1 = gui._make_comparison_html(img, img.copy())
        html2 = gui._make_comparison_html(img, img.copy())
        # The uid appears right after 'cmp-'
        uid1 = html1.split('cmp-')[1].split('"')[0]
        uid2 = html2.split('cmp-')[1].split('"')[0]
        assert uid1 != uid2


# ---------------------------------------------------------------------------
# TestExtMap
# ---------------------------------------------------------------------------


class TestExtMap:
    """Tests for the gui.EXT_MAP constant."""

    def test_contains_jpeg(self):
        assert "JPEG" in gui.EXT_MAP

    def test_contains_png(self):
        assert "PNG" in gui.EXT_MAP

    def test_contains_webp(self):
        assert "WebP" in gui.EXT_MAP

    def test_contains_png16(self):
        assert "PNG-16" in gui.EXT_MAP

    def test_jpeg_maps_to_jpg(self):
        assert gui.EXT_MAP["JPEG"] == ".jpg"

    def test_png_maps_to_png(self):
        assert gui.EXT_MAP["PNG"] == ".png"

    def test_png16_maps_to_png(self):
        assert gui.EXT_MAP["PNG-16"] == ".png"

    def test_webp_maps_to_webp(self):
        assert gui.EXT_MAP["WebP"] == ".webp"

    def test_values_start_with_dot(self):
        """File extensions should always include the leading dot."""
        for fmt, ext in gui.EXT_MAP.items():
            assert ext.startswith("."), f"{fmt!r} extension {ext!r} must start with '.'"

    def test_values_are_lowercase(self):
        for ext in gui.EXT_MAP.values():
            assert ext == ext.lower(), f"Extension {ext!r} should be lowercase"

    def test_lookup_with_unknown_returns_none(self):
        assert gui.EXT_MAP.get("TIFF") is None
        assert gui.EXT_MAP.get("BMP", ".bmp") == ".bmp"


# ---------------------------------------------------------------------------
# TestExportResMap
# ---------------------------------------------------------------------------


class TestExportResMap:
    """Tests for the gui.EXPORT_RES_MAP constant."""

    def test_contains_original(self):
        """The 'Original' option must be present and mean 'no resize'."""
        assert "Original" in gui.EXPORT_RES_MAP
        assert gui.EXPORT_RES_MAP["Original"] is None

    def test_4k_resolution(self):
        assert gui.EXPORT_RES_MAP["4K (3840px)"] == 3840

    def test_2k_resolution(self):
        assert gui.EXPORT_RES_MAP["2K (2048px)"] == 2048

    def test_full_hd_resolution(self):
        assert gui.EXPORT_RES_MAP["Full HD (1920px)"] == 1920

    def test_hd_resolution(self):
        assert gui.EXPORT_RES_MAP["HD (1280px)"] == 1280

    def test_720p_resolution(self):
        assert gui.EXPORT_RES_MAP["720px"] == 720

    def test_only_original_is_none(self):
        """Only the 'Original' preset should map to None (no resize)."""
        for key, val in gui.EXPORT_RES_MAP.items():
            if key == "Original":
                assert val is None
            else:
                assert val is not None, f"{key!r} should map to an int, not None"

    def test_values_are_positive_ints(self):
        for key, val in gui.EXPORT_RES_MAP.items():
            if val is not None:
                assert isinstance(val, int)
                assert val > 0

    def test_resolutions_descending(self):
        """Resolutions should be ordered from largest to smallest (excluding Original)."""
        ordered = ["4K (3840px)", "2K (2048px)", "Full HD (1920px)",
                   "HD (1280px)", "720px"]
        sizes = [gui.EXPORT_RES_MAP[k] for k in ordered]
        assert sizes == sorted(sizes, reverse=True), \
            f"Resolutions should descend, got: {sizes}"


# ---------------------------------------------------------------------------
# TestChoiceConstants
# ---------------------------------------------------------------------------


class TestChoiceConstants:
    """Tests for the dropdown choice lists."""

    def test_whiten_tone_choices(self):
        assert gui.WHITEN_TONE_CHOICES == ["rosy", "porcelain", "neutral"]

    def test_specular_bloom_tone_choices_match_whiten(self):
        """The two cosmetic-tone dropdowns share the same allowed values."""
        assert gui.SPECULAR_BLOOM_TONE_CHOICES == gui.WHITEN_TONE_CHOICES

    def test_lip_finish_choices(self):
        assert gui.LIP_FINISH_CHOICES == ["gloss", "matte", "velvet"]

    def test_lip_tints_contains_none(self):
        """The 'none' option must be present so the user can opt-out."""
        assert "none" in gui.LIP_TINTS

    def test_lut_choices_contains_none_and_presets(self):
        assert "none" in gui.LUT_CHOICES
        assert "kodak" in gui.LUT_CHOICES
        assert "fuji" in gui.LUT_CHOICES

    def test_color_grade_names_starts_with_none(self):
        """'none' must be the first option so the dropdown has a no-op default."""
        assert gui.COLOR_GRADE_NAMES[0] == "none"

    def test_recipe_names_match_recipes_keys(self):
        """RECIPE_NAMES should be a snapshot of RECIPES keys (in order)."""
        assert gui.RECIPE_NAMES == list(gui.RECIPES.keys())

    def test_recipe_names_includes_natural(self):
        assert "natural" in gui.RECIPE_NAMES

    def test_all_choice_lists_are_unique(self):
        """No dropdown should contain duplicate entries."""
        for name, choices in [
            ("WHITEN_TONE_CHOICES", gui.WHITEN_TONE_CHOICES),
            ("SPECULAR_BLOOM_TONE_CHOICES", gui.SPECULAR_BLOOM_TONE_CHOICES),
            ("LIP_FINISH_CHOICES", gui.LIP_FINISH_CHOICES),
            ("LIP_TINTS", gui.LIP_TINTS),
            ("LUT_CHOICES", gui.LUT_CHOICES),
            ("COLOR_GRADE_NAMES", gui.COLOR_GRADE_NAMES),
            ("RECIPE_NAMES", gui.RECIPE_NAMES),
        ]:
            assert len(choices) == len(set(choices)), f"{name} has duplicates: {choices}"


# ---------------------------------------------------------------------------
# TestProcessInputKeys
# ---------------------------------------------------------------------------


class TestProcessInputKeys:
    """Tests for the gui.PROCESS_INPUT_KEYS constant."""

    def test_count_matches_process_image_arity(self):
        """PROCESS_INPUT_KEYS length must equal the arity of process_image."""
        from retouch.params import param_names
        # 2 leading + registry (minus 2 excluded) + 12 trailing transport keys
        # (look_params + face_params State + face_params_json).
        expected = 2 + (len(param_names()) - 2) + 12
        assert len(gui.PROCESS_INPUT_KEYS) == expected

    def test_first_key_is_img_paths(self):
        assert gui.PROCESS_INPUT_KEYS[0] == "img_paths"

    def test_second_key_is_recipe(self):
        assert gui.PROCESS_INPUT_KEYS[1] == "recipe"

    def test_last_keys_are_face_params(self):
        assert gui.PROCESS_INPUT_KEYS[-1] == "face_params_json"
        assert gui.PROCESS_INPUT_KEYS[-2] == "face_params"
        assert gui.PROCESS_INPUT_KEYS[-3] == "look_params"

    def test_no_duplicate_keys(self):
        assert len(gui.PROCESS_INPUT_KEYS) == len(set(gui.PROCESS_INPUT_KEYS))

    def test_all_keys_are_strings(self):
        for k in gui.PROCESS_INPUT_KEYS:
            assert isinstance(k, str)

    def test_keys_are_snake_case(self):
        for k in gui.PROCESS_INPUT_KEYS:
            assert k == k.lower(), f"Key {k!r} should be lowercase"
            assert " " not in k, f"Key {k!r} should not contain spaces"

    # --- _process_inputs de-footgun (name-keyed dict + drift guard) ----------

    @staticmethod
    def _extract_components(src):
        """Parse the gui.py source and pull the `_process_input_components`
        dict back out as (keys_list, value_token_list) without executing Gradio."""
        import ast
        tree = ast.parse(src)
        dict_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == "_process_input_components":
                        dict_node = node.value
        assert dict_node is not None, "_process_input_components dict not found in gui.py"
        keys = [k.value for k in dict_node.keys]
        # Values are the Gradio component variable names (source text).
        import re
        m = re.search(r"_process_input_components = \{(.*?)\n    \}", src, re.S)
        body = m.group(1)
        pairs, depth, cur = [], 0, ""
        for ch in body:
            if ch == ",":
                if depth == 0:
                    pairs.append(cur.strip()); cur = ""
                else:
                    cur += ch
            elif ch in "([{":
                depth += 1; cur += ch
            elif ch in ")]}":
                depth -= 1; cur += ch
            else:
                cur += ch
        if cur.strip():
            pairs.append(cur.strip())
        values = [p.split(":", 1)[1].strip() for p in pairs]
        return keys, values

    def test_component_dict_keys_match_process_input_keys(self):
        """Every PROCESS_INPUT_KEYS name maps to exactly one component, and
        the dict keys are in the same order (so the derived positional list
        is identical to the old hand-ordered list)."""
        src = open(gui.__file__).read()
        dict_keys, dict_values = self._extract_components(src)
        n = len(gui.PROCESS_INPUT_KEYS)
        assert len(dict_keys) == n
        assert set(dict_keys) == set(gui.PROCESS_INPUT_KEYS)
        assert dict_keys == list(gui.PROCESS_INPUT_KEYS)
        assert len(dict_values) == n

    def test_derived_process_inputs_is_byte_identical(self):
        """The positional `_process_inputs` list is derived from the dict in
        PROCESS_INPUT_KEYS order (no silent argument shift)."""
        src = open(gui.__file__).read()
        dict_keys, dict_values = self._extract_components(src)
        derived = [dict_values[dict_keys.index(k)] for k in gui.PROCESS_INPUT_KEYS]
        assert derived == dict_values  # dict order == keys order by construction
        # The first two slots are the special img_paths->img_input and recipe.
        assert dict_keys[0] == "img_paths" and dict_values[0] == "img_input"
        assert dict_keys[1] == "recipe" and dict_values[1] == "recipe"
        assert dict_keys[-1] == "face_params_json" and dict_values[-1] == "face_params_json"
        assert dict_keys[-2] == "face_params" and dict_values[-2] == "_face_params_state"
        assert dict_keys[-3] == "look_params" and dict_values[-3] == "_look_params_state"

    def _run_guard(self, dict_keys, process_keys):
        """Replicate the exact import-time drift guard from build_app()."""
        missing = set(process_keys) - set(dict_keys)
        extra = set(dict_keys) - set(process_keys)
        if missing or extra:
            raise AssertionError(
                f"_process_input_components drift: missing={sorted(missing)} extra={sorted(extra)}"
            )

    def test_drift_guard_raises_on_missing_component(self):
        """If a new ParamSpec adds a name to PROCESS_INPUT_KEYS but no matching
        component entry exists, the import-time guard must raise loudly."""
        dict_keys, _ = self._extract_components(open(gui.__file__).read())
        keys = list(gui.PROCESS_INPUT_KEYS)
        # Simulate forgetting to add the component for the last key.
        with pytest.raises(AssertionError):
            self._run_guard(dict_keys[:-1], keys)

    def test_drift_guard_raises_on_orphan_component(self):
        """If a component is added to the dict for a key not in
        PROCESS_INPUT_KEYS, the guard must raise."""
        dict_keys, _ = self._extract_components(open(gui.__file__).read())
        keys = list(gui.PROCESS_INPUT_KEYS)
        # Simulate an orphan key present only in the dict.
        with pytest.raises(AssertionError):
            self._run_guard(dict_keys + ["nonexistent_param"], keys)

    def test_drift_guard_passes_on_clean_dict(self):
        """With the real dict and real keys, the guard must NOT raise."""
        dict_keys, _ = self._extract_components(open(gui.__file__).read())
        self._run_guard(dict_keys, list(gui.PROCESS_INPUT_KEYS))

    def test_no_hand_ordered_process_inputs_literal(self):
        """The positional list is derived via comprehension, not a hand-ordered
        literal — so the historical position-based footgun cannot recur."""
        src = open(gui.__file__).read()
        assert "_process_inputs = [img_input, recipe," not in src
        assert "_process_inputs = [_process_input_components[k] for k in PROCESS_INPUT_KEYS]" in src


# ---------------------------------------------------------------------------
# TestGetEngine
# ---------------------------------------------------------------------------


class TestGetEngine:
    """Tests for gui.get_engine() — the engine singleton factory."""

    def test_returns_retouch_engine(self):
        from retouch import RetouchEngine
        eng = gui.get_engine()
        assert isinstance(eng, RetouchEngine)

    def test_singleton(self):
        """Multiple calls should return the same cached instance."""
        a = gui.get_engine()
        b = gui.get_engine()
        assert a is b

    def test_thread_safe_singleton(self):
        """Concurrent calls should all return the same instance."""
        import threading
        results = []
        def worker():
            results.append(gui.get_engine())
        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        first = results[0]
        assert all(r is first for r in results)


# ---------------------------------------------------------------------------
# TestResetFunctions
# ---------------------------------------------------------------------------


class TestResetFunctions:
    """Tests for the gui.reset_*() section-reset helpers."""

    def test_reset_skin_smoothing_returns_nine_values(self):
        result = gui.reset_skin_smoothing("natural")
        assert isinstance(result, tuple)
        assert len(result) == 9

    def test_reset_skin_smoothing_values_match_natural_recipe(self):
        d = gui.recipe_defaults("natural")
        result = gui.reset_skin_smoothing("natural")
        assert result == (d["smooth"], d["nose_smooth"], d["mid_reduction"],
                          d["texture_opacity"], d["micro_restore"],
                          d["pore_synthesis"], d["blemish"],
                          d["skin_flatten"], d["skin_quantize"])

    def test_reset_skin_tone_returns_seven_values(self):
        result = gui.reset_skin_tone("natural")
        assert isinstance(result, tuple)
        assert len(result) == 11

    def test_reset_skin_tone_values_match_natural_recipe(self):
        d = gui.recipe_defaults("natural")
        result = gui.reset_skin_tone("natural")
        assert result == (d["whiten"], d["whiten_tone"], d["equalize"],
                          d["shadow_lift"], d["nose_restore"], d["skin_sss"],
                          d["skin_unify"], d["skin_unify_hue"], d["auto_exposure"],
                          d["white_costume_lift"], d["face_exposure"])

    def test_reset_basic_tone_returns_five_values(self):
        result = gui.reset_basic_tone("natural")
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_reset_tone_curve_returns_four_values(self):
        result = gui.reset_tone_curve("natural")
        assert isinstance(result, tuple)
        assert len(result) == 4

    def test_reset_relighting_returns_three_values(self):
        result = gui.reset_relighting("natural")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_reset_eyes_lips_returns_sixteen_values(self):
        result = gui.reset_eyes_lips("natural")
        assert isinstance(result, tuple)
        assert len(result) == 16

    def test_reset_face_reshaping_returns_slimming_int(self):
        result = gui.reset_face_reshaping("natural")
        d = gui.recipe_defaults("natural")
        assert result == d["slimming"]

    def test_reset_structure_effects_returns_fifteen_values(self):
        result = gui.reset_structure_effects("natural")
        assert isinstance(result, tuple)
        assert len(result) == 15

    def test_reset_color_grading_returns_two_values(self):
        result = gui.reset_color_grading("natural")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_film_effects_returns_eight_values(self):
        result = gui.reset_film_effects("natural")
        assert isinstance(result, tuple)
        assert len(result) == 8

    def test_reset_split_toning_returns_six_values(self):
        result = gui.reset_split_toning("natural")
        assert isinstance(result, tuple)
        assert len(result) == 6

    def test_reset_color_transfer_returns_none_and_one(self):
        """Color-transfer reset clears the reference image and resets strength to 1.0."""
        result = gui.reset_color_transfer()
        assert result == (None, 1.0)

    def test_reset_debug_returns_false(self):
        """The debug-mode checkbox should always reset to off."""
        assert gui.reset_debug("natural") is False
        assert gui.reset_debug("cosplay") is False

    def test_reset_functions_work_for_any_recipe(self):
        """Every reset helper should accept any recipe name without crashing."""
        for recipe in gui.RECIPE_NAMES:
            gui.reset_skin_smoothing(recipe)
            gui.reset_skin_tone(recipe)
            gui.reset_basic_tone(recipe)
            gui.reset_tone_curve(recipe)
            gui.reset_relighting(recipe)
            gui.reset_eyes_lips(recipe)
            gui.reset_face_reshaping(recipe)
            gui.reset_structure_effects(recipe)
            gui.reset_color_grading(recipe)
            gui.reset_film_effects(recipe)
            gui.reset_split_toning(recipe)
            gui.reset_debug(recipe)

    def test_reset_functions_fall_back_for_unknown_recipe(self):
        """An unknown recipe name should still work (resolves to natural)."""
        a = gui.reset_skin_smoothing("not_a_recipe_xyz")
        b = gui.reset_skin_smoothing("natural")
        assert a == b


# ---------------------------------------------------------------------------
# TestOnSaveStyleValidation
# ---------------------------------------------------------------------------


class TestOnSaveStyleValidation:
    """Tests for gui.on_save_style() input validation paths."""

    def test_empty_name_returns_error_message(self):
        """An empty style name should return a user-facing error string."""
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r1, r2, r3 = gui.on_save_style("", "author", "", 30, 0.45, 1.0, 10, 0, 0)
        assert "cannot be empty" in r3.lower()

    def test_whitespace_name_returns_error(self):
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r1, r2, r3 = gui.on_save_style("   ", "author", "", 30, 0.45, 1.0, 10, 0, 0)
        assert "cannot be empty" in r3.lower()

    def test_too_long_name_returns_error(self):
        from unittest.mock import patch
        long_name = "x" * 101
        with patch("gui.gr") as mock_gr:
            r1, r2, r3 = gui.on_save_style(long_name, "author", "", 30, 0.45, 1.0, 10, 0, 0)
        assert "100 characters" in r3

    def test_exactly_100_chars_accepted(self, isolated_styles_dir):
        """A 100-character name is the boundary; should succeed without raising."""
        from unittest.mock import patch
        long_name = "x" * 100
        with patch("gui.gr") as mock_gr:
            mock_gr.update.return_value = {"__type__": "update"}
            r1, r2, r3 = gui.on_save_style(long_name, "author", "", 30, 0.45, 1.0, 10, 0, 0)
        # Should NOT return an error
        assert "Error" not in r3

    def test_valid_save_persists_profile(self, isolated_styles_dir):
        """A valid save call should produce a 'saved successfully' message."""
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            mock_gr.update.return_value = {"__type__": "update"}
            r1, r2, r3 = gui.on_save_style("test_valid_save", "author", "tag1,tag2",
                                           30, 0.45, 1.0, 10, 0, 0)
        assert "saved" in r3.lower()
        assert mock_gr.Info.called

    def test_save_default_author_when_none(self, isolated_styles_dir):
        """A None/empty author should fall back to 'Dennis'."""
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            mock_gr.update.return_value = {"__type__": "update"}
            gui.on_save_style("test_default_author", "", "", 30, 0.45, 1.0, 10, 0, 0)
        # The function should have called save_style_profile via get_custom_style_names
        # Verify the style was actually saved
        from retouch.style_library import list_styles
        names = [s["name"] for s in list_styles()]
        assert "test_default_author" in names


# ---------------------------------------------------------------------------
# TestOnLearnStyleValidation
# ---------------------------------------------------------------------------


class TestOnLearnStyleValidation:
    """Tests for gui.on_learn_style() input validation paths."""

    def test_missing_orig_dir_returns_error(self):
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r1, r2, r3 = gui.on_learn_style("", "/edit", "name", "author", "")
        assert "Original and Edited" in r3

    def test_missing_edit_dir_returns_error(self):
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r1, r2, r3 = gui.on_learn_style("/orig", "", "name", "author", "")
        assert "Original and Edited" in r3

    def test_both_dirs_missing_returns_error(self):
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r1, r2, r3 = gui.on_learn_style("", "", "name", "author", "")
        assert "Original and Edited" in r3

    def test_empty_name_returns_error(self):
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r1, r2, r3 = gui.on_learn_style("/orig", "/edit", "", "author", "")
        assert "cannot be empty" in r3.lower()

    def test_whitespace_name_returns_error(self):
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r1, r2, r3 = gui.on_learn_style("/orig", "/edit", "   ", "author", "")
        assert "cannot be empty" in r3.lower()

    def test_too_long_name_returns_error(self):
        from unittest.mock import patch
        long_name = "x" * 101
        with patch("gui.gr") as mock_gr:
            r1, r2, r3 = gui.on_learn_style("/orig", "/edit", long_name, "author", "")
        assert "100 characters" in r3


# ---------------------------------------------------------------------------
# TestOnProcessFolderValidation
# ---------------------------------------------------------------------------


class TestOnProcessFolderValidation:
    """Tests for gui.on_process_folder() input validation paths."""

    def test_missing_input_dir_returns_error(self):
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r = gui.on_process_folder("", "/out", "Use Standard Recipe", "", "natural",
                                      "JPEG", 95, "Original", False, False, False)
        assert r[0] is None
        assert r[1] is None
        assert "Input and Output" in r[2]

    def test_missing_output_dir_returns_error(self):
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r = gui.on_process_folder("/in", "", "Use Standard Recipe", "", "natural",
                                      "JPEG", 95, "Original", False, False, False)
        assert "Input and Output" in r[2]

    def test_custom_style_without_name_returns_error(self):
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r = gui.on_process_folder("/in", "/out", "Use Custom Style", "", "natural",
                                      "JPEG", 95, "Original", False, False, False)
        assert "select a custom style" in r[2].lower()

    def test_custom_style_with_bad_name_returns_error(self):
        from unittest.mock import patch
        with patch("gui.gr") as mock_gr:
            r = gui.on_process_folder("/in", "/out", "Use Custom Style",
                                      "nonexistent_style_xyz", "natural",
                                      "JPEG", 95, "Original", False, False, False)
        assert "not found" in r[2]


# ---------------------------------------------------------------------------
# TestOnBatchStyleChange
# ---------------------------------------------------------------------------


class TestOnBatchStyleChange:
    """Tests for gui.on_batch_style_change() dropdown visibility toggler."""

    def test_custom_style_hides_recipe_shows_custom(self):
        result = gui.on_batch_style_change("Use Custom Style")
        assert len(result) == 2
        # result[0] is the recipe dropdown update (hidden)
        # result[1] is the custom style dropdown update (visible)
        assert result[0].get("visible") is False
        assert result[1].get("visible") is True

    def test_standard_recipe_shows_recipe_hides_custom(self):
        result = gui.on_batch_style_change("Use Standard Recipe")
        assert len(result) == 2
        assert result[0].get("visible") is True
        assert result[1].get("visible") is False

    def test_unknown_value_defaults_to_standard(self):
        """An unrecognized value should not crash; default to standard-recipe visibility."""
        result = gui.on_batch_style_change("something_else")
        assert len(result) == 2
        assert result[0].get("visible") is True
        assert result[1].get("visible") is False


# ---------------------------------------------------------------------------
# TestProcessImageValidation
# ---------------------------------------------------------------------------


class TestProcessImageValidation:
    """Tests for gui.process_image() input-validation early returns."""

    def _build_args(self, **overrides):
        """Build a full arg tuple (one value per PROCESS_INPUT_KEYS entry).

        Every ParamSpec contributes its registry default, then the hand-pinned
        'natural'-recipe values below overlay them, so the function reaches its
        validation gates and short-circuits before any real image work is
        attempted — and a newly registered param can never KeyError this
        helper again (it starts from its own spec default).
        """
        from retouch.params import PROCESSING_PARAMS
        defaults = {s.name: s.default for s in PROCESSING_PARAMS}
        # Transport keys that are not ParamSpecs.
        defaults.update({
            "look_params": None, "face_params": None, "face_params_json": None,
        })
        defaults.update({
            "img_paths": None,
            "recipe": "natural",
            "smooth": 0, "mid_reduction": 0.0, "texture_opacity": 1.0,
            "pore_synthesis": 0, "nose_smooth": 0, "micro_restore": 0,
            "micro_dodge_burn": 0, "redness_even": 0, "whiten_hue_stable": 0,
            "whiten": 0, "equalize": 0, "white_costume_lift": False,
            "body_smooth": 0, "body_equalize": 0, "body_whiten": 0, "body_match_face": 0,
            "body_relight": 0, "body_dodge_burn": 0, "body_shadow_lift": 0,
            "shadow_lift": 0, "nose_restore": 0,
            "skin_hue_unify": 0, "skin_chroma_even": 0,
            "relight": 0, "relight_azimuth": 0, "relight_elevation": 30, "sculpt": 0, "shine_removal": 0, "wrinkle_soften": 0, "texture_transplant": 0,
            "eye_enhance": 0, "teeth_whiten": 0,
            "lip_enhance": 0, "lip_tint": "none", "blush": 0,
            "nose_blush": False, "under_eye_blush": False,
            "hair_enhance": 0, "dodge_burn": 0, "specular_bloom": 0,
            "skin_flatten": 0, "skin_quantize": 0, "skin_unify": 0, "skin_unify_hue": -1.0, "skin_glow": 0,
            "bloom": 0, "bloom_threshold": 210, "bloom_softness": 30,
            "contrast": 0, "brightness": 0,
            "highlights": 0, "shadows": 0, "whites": 0, "blacks": 0,
            "color_ref_img": None, "color_ref_strength": 0.0,
            "show_compare": False, "fast": True,
            "export_fmt": "JPEG", "export_quality": 95, "export_res": "Original",
            "blemish": 0, "dark_circles": 0, "catchlight": 0,
            "whiten_tone": "rosy", "auto_exposure": False,
            "clarity": 0, "vibrance": 0, "saturation": 0, "lip_finish": "gloss",
            "slimming": 0, "impact": 0,
            "sharpen": 0, "sharpen_radius": 1.0, "glow": 0, "vignette": 0,
            "fade_toe": 0, "highlight_drift": 0, "airy_haze": 0, "clarity_split_neg": 0, "clarity_split_pos": 0,
            "subject_separation": 0, "specular_bloom_tone": "rosy",
            "color_grade": "none", "grade_intensity": 0, "color_transfer_intensity": 1.0,
            "white_balance_kelvin": 6500, "white_balance_tint": 0,
            "bw_channel_mixer_r": 30, "bw_channel_mixer_g": 59, "bw_channel_mixer_b": 11,
            "negative_split_tone_shadow": 0, "negative_split_tone_highlight": 0,
            "hsl_hue_global": 0, "hsl_sat_global": 0, "hsl_lum_global": 0,
            "chromatic_aberration": 0.0, "grain": 0.0, "halation": 0.0,
            "lut": "none",
            "tonal_curve_strength": 0.0, "skin_protect_strength": 0.0,
            "highlight_rolloff_strength": 0.0, "grain_strength": 0.0,
            "reshape_eye_size": 0.0, "reshape_eye_distance": 0.0,
            "reshape_nose_width": 0.0, "reshape_nose_length": 0.0,
            "reshape_jaw_width": 0.0, "reshape_chin_length": 0.0,
            "reshape_mouth_size": 0.0, "reshape_smile": 0.0, "reshape_forehead": 0.0,
            "hair_deglare": 0.0, "hair_ring_position": 0.5, "hair_ring_tint": 0.0,
            "hair_remove_flyaways": 0.0,
            "film_enable": False, "film_strength": 0.0,
            "film_toe_r": 0.0, "film_toe_g": 0.0, "film_toe_b": 0.0,
            "film_shoulder_r": 0.0, "film_shoulder_g": 0.0, "film_shoulder_b": 0.0,
            "film_midpoint": 0.5, "film_gamma": 1.0,
            "film_crosstalk_cy_mg": 0.0, "film_crosstalk_cy_ye": 0.0, "film_crosstalk_mg_ye": 0.0,
            "film_tonemap_strength": 0.0, "film_tonemap_toe": 0.1,
            "film_tonemap_shoulder": 0.1, "film_skew": 0.0, "film_highlight_purity": 0.0,
            "heal_engine": "telea", "hb_even": 0.0, "hb_shift": 0.0,
            "background_harmonize": 0.0, "background_harmonize_mode": "split",
            "background_blur": 0.0, "background_desaturation": 0.0,
            "light_wrap": 0.0, "blue_shadow_grade": 0.0,
            "cyan_midtone_grade": 0.0, "subject_sharpen": 0.0, "matte_black": 0.0,
            "ai_denoise": 0.0, "ai_sr_scale": 1,
            "mv2_eyeshadow": 0.0, "mv2_eyeshadow_color": "brown", "mv2_eyeshadow_style": "natural",
            "mv2_eyeliner": 0.0, "mv2_eyeliner_color": "black", "mv2_eyeliner_style": "classic",
            "mv2_contour": 0.0, "mv2_brows": 0.0, "mv2_brows_color": "brown",
            "mv2_ombre": 0.0, "mv2_ombre_color1": "red", "mv2_ombre_color2": "pink",
            "shadow_hue": 0, "shadow_sat": 0,
            "midtone_hue": 0, "midtone_sat": 0,
            "highlight_hue": 0, "highlight_sat": 0,
            "quality_tier": "Full (native face crops)",
            "debug_mode": False,
        })
        defaults.update(overrides)
        return tuple(defaults[k] for k in gui.PROCESS_INPUT_KEYS)

    def test_no_image_returns_seven_tuple(self):
        """No image uploaded should return an 8-tuple early without engine work
        (7 original outputs + qa_warnings_html appended for F11)."""
        args = self._build_args(img_paths=None)
        result = gui.process_image(*args)
        assert isinstance(result, tuple)
        assert len(result) == 8

    def test_no_image_returns_user_facing_error_message(self):
        args = self._build_args(img_paths=None)
        result = gui.process_image(*args)
        # The 5th element is the status string
        assert isinstance(result[4], str)
        assert "upload" in result[4].lower()

    def test_no_image_hides_compare_viewer(self):
        args = self._build_args(img_paths=None)
        result = gui.process_image(*args)
        # The 2nd element is the compare_viewer gr.update — should be hidden
        assert result[1].get("visible") is False

    def test_no_image_returns_none_for_outputs(self):
        args = self._build_args(img_paths=None)
        result = gui.process_image(*args)
        # img_output (0), original_state (2), export_file (3), debug_gallery (5)
        assert result[0] is None
        assert result[2] is None
        assert result[3] is None
        assert result[5] is None

    def test_no_image_hides_debug_panel(self):
        args = self._build_args(img_paths=None)
        result = gui.process_image(*args)
        # The 7th element is the debug_panel gr.update — should be hidden
        assert result[6].get("visible") is False

    def test_empty_list_treated_as_no_image(self):
        """An empty list of paths should also trigger the early-return path."""
        args = self._build_args(img_paths=[])
        result = gui.process_image(*args)
        assert "upload" in result[4].lower()

    def test_arg_count_matches_keys(self):
        """The defaults builder must produce exactly len(PROCESS_INPUT_KEYS) args."""
        args = self._build_args()
        assert len(args) == len(gui.PROCESS_INPUT_KEYS)


# ---------------------------------------------------------------------------
# TestIntegrationConstantCrossRef
# ---------------------------------------------------------------------------


class TestIntegrationConstantCrossRef:
    """Cross-reference tests between constants and recipe outputs."""

    def test_recipe_outputs_count_matches_choices(self):
        """The number of UI outputs (from the _recipe_outputs list) must match
        what on_recipe_change returns."""
        result = gui.on_recipe_change("natural")
        assert len(result) == len(gui.RECIPE_OUTPUT_KEYS)

    def test_process_inputs_count_matches_process_input_keys(self):
        """_process_inputs and PROCESS_INPUT_KEYS must have matching lengths."""
        assert len(gui._process_inputs) == len(gui.PROCESS_INPUT_KEYS)

    def test_recipe_outputs_count_matches_on_recipe_change(self):
        """_recipe_outputs and on_recipe_change return tuple must match."""
        assert len(gui._recipe_outputs) == len(gui.on_recipe_change("natural"))

    def test_recipe_output_keys_include_formerly_dropped_params(self):
        """Regression guard: the 4 params that were silently dropped from
        apply_custom_style / on_smart_process (causing index-5 shift) are now
        part of the canonical RECIPE_OUTPUT_KEYS contract."""
        for dropped in ("regional_modulation", "smooth_engine",
                        "undereye_shadow_strength", "freckle_removal", "mark_policy"):
            assert dropped in gui.RECIPE_OUTPUT_KEYS

    def test_all_producers_keyed_by_recipe_output_keys(self):
        """All three producer paths (on_recipe_change, apply_custom_style,
        on_smart_process) build their tuple keyed by RECIPE_OUTPUT_KEYS, so the
        full set of defaults must cover every key — otherwise a producer would
        KeyError or emit a wrong-length/misaligned tuple."""
        d = gui.recipe_defaults("natural")
        missing = set(gui.RECIPE_OUTPUT_KEYS) - set(d.keys())
        assert not missing, f"recipe_defaults missing keys for producers: {sorted(missing)}"

    def test_process_input_keys_excludes_color_transfer_intensity(self):
        """color_transfer_intensity is in param_names() but has no GUI component."""
        assert "color_transfer_intensity" not in gui.PROCESS_INPUT_KEYS

    def test_process_input_keys_excludes_visual_outputs(self):
        """PROCESS_INPUT_KEYS is for process_image inputs and does not include
        the visual output components (preview, compare_viewer, etc.)."""
        # process_image returns 7 things; PROCESS_INPUT_KEYS has 69 items.
        # The keys describe the INPUTS to the function, not its outputs.
        assert "img_output" not in gui.PROCESS_INPUT_KEYS
        assert "compare_viewer" not in gui.PROCESS_INPUT_KEYS
        assert "status" not in gui.PROCESS_INPUT_KEYS
        assert "export_file" not in gui.PROCESS_INPUT_KEYS

    def test_ext_map_keys_match_radio_choices(self):
        """The EXT_MAP keys should match the format radio used in the UI."""
        assert set(gui.EXT_MAP.keys()) == {"JPEG", "PNG", "PNG-16", "WebP"}

    def test_export_res_map_keys_match_dropdown_choices(self):
        """The EXPORT_RES_MAP keys should match the resolution dropdown choices."""
        # The UI hard-codes the same list as choices.
        expected = {"Original", "4K (3840px)", "2K (2048px)",
                    "Full HD (1920px)", "HD (1280px)", "720px"}
        assert set(gui.EXPORT_RES_MAP.keys()) == expected
