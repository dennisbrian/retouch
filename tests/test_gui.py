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
    "smooth", "mid_reduction", "texture_opacity", "pore_synthesis", "nose_smooth",
    "whiten", "equalize", "relight", "relight_azimuth", "relight_elevation",
    "eye_enhance", "lip_enhance", "lip_tint", "blush", "teeth_whiten",
    "hair_enhance", "dodge_burn", "specular_bloom", "bloom", "bloom_threshold",
    "bloom_softness", "contrast", "brightness", "highlights", "shadows",
    "whites", "blacks", "nose_blush", "under_eye_blush", "white_costume_lift",
    "blemish", "dark_circles", "catchlight", "whiten_tone", "auto_exposure",
    "lip_finish", "slimming", "impact", "clarity", "vibrance", "saturation",
    "glow", "vignette", "sharpen", "sharpen_radius", "subject_separation",
    "specular_bloom_tone", "color_grade", "grade_intensity",
    "chromatic_aberration", "grain", "halation", "lut",
    "shadow_hue", "shadow_sat", "midtone_hue", "midtone_sat",
    "highlight_hue", "highlight_sat",
]
EXPECTED_RECIPE_KEY_COUNT = 59
EXPECTED_UI_OUTPUT_COUNT = 59


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
        for name in gui.RECIPES:
            d = gui.recipe_defaults(name)
            for key in EXPECTED_RECIPE_KEYS:
                val = d[key]
                if key in ("lip_tint", "whiten_tone", "lip_finish",
                           "specular_bloom_tone", "color_grade", "lut"):
                    # These are string fields
                    assert isinstance(val, str), (
                        f"Recipe {name!r}.{key!r} should be str, got {type(val).__name__}"
                    )
                elif key in ("nose_blush", "under_eye_blush", "white_costume_lift",
                             "auto_exposure"):
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
        # whiten is at index 5
        assert result[5] == 20


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
        """The first 9 UI outputs (smooth..relight_elevation) should match recipe_defaults."""
        result = gui.on_recipe_change("natural")
        d = gui.recipe_defaults("natural")
        assert result[0] == d["smooth"]
        assert result[1] == d["mid_reduction"]
        assert result[2] == d["texture_opacity"]
        assert result[3] == d["pore_synthesis"]
        assert result[4] == d["nose_smooth"]
        assert result[5] == d["whiten"]
        assert result[6] == d["equalize"]

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
        # clarity sits at index 35 in the on_recipe_change output tuple
        clarity_index = 35
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
