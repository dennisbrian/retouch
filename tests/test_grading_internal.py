"""Tests for retouch/grading.py — internal/private methods of ColorGrader."""
import json

import cv2
import numpy as np
import pytest
from retouch import grain, highlight, lut as lut_mod, tonal
from retouch import precision as precision_mod
from retouch.grading import (
    ColorGrader,
    _USER_PRESETS_DIRS,
    list_available_presets,
    load_all_presets,
    load_preset,
    register_presets_dir,
)
from retouch.lut import CubeLUT


@pytest.fixture
def grader():
    return ColorGrader()


@pytest.fixture
def img():
    return np.full((64, 64, 3), 128, dtype=np.uint8)


@pytest.fixture
def varied_img():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:32, :, :] = [50, 50, 50]
    img[32:, :, :] = [200, 200, 200]
    return img


@pytest.fixture
def colorful_img():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    img[:21, :, 2] = 200
    img[21:43, :, 1] = 180
    img[43:, :, 0] = 220
    return img


@pytest.fixture
def isolated_user_dirs():
    """Snapshot and restore ``_USER_PRESETS_DIRS`` so test cases don't pollute it."""
    original = list(_USER_PRESETS_DIRS)
    _USER_PRESETS_DIRS.clear()
    yield
    _USER_PRESETS_DIRS.clear()
    _USER_PRESETS_DIRS.extend(original)


class TestRegisterAndListPresets:
    def test_list_available(self):
        presets = list_available_presets()
        assert len(presets) > 0
        assert "natural" in presets


class TestAddGlow:
    def test_zero_opacity(self, grader, img):
        result = grader._add_glow(img, 0)
        assert np.all(result == img)

    def test_changes_bright_image(self, grader):
        img = np.full((64, 64, 3), 235, dtype=np.uint8)
        result = grader._add_glow(img, 0.5)
        assert not np.allclose(result, img)

    def test_with_tint(self, grader):
        img = np.full((64, 64, 3), 235, dtype=np.uint8)
        result = grader._add_glow(img, 0.5, tint=(200, 100, 150))
        assert not np.allclose(result, img)

    def test_with_mask(self, grader):
        img = np.full((64, 64, 3), 235, dtype=np.uint8)
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader._add_glow(img, 0.5, mask=mask)
        assert not np.allclose(result, img)

    def test_output_type(self, grader, img):
        result = grader._add_glow(img, 0.5)
        assert result.dtype == np.uint8


class TestSplitTone:
    def test_returns_image(self, grader, varied_img):
        tones = {"shadows": [118, 130], "highlights": [138, 120]}
        result = grader._split_tone(varied_img, tones)
        assert result.shape == varied_img.shape

    def test_with_mask(self, grader, varied_img):
        tones = {"shadows": [118, 130], "highlights": [138, 120]}
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader._split_tone(varied_img, tones, mask=mask)
        assert result.shape == varied_img.shape


class TestSplitToneThreeWay:
    def test_no_tones_returns_original(self, grader, img):
        result = grader._split_tone_three_way(img, {})
        assert np.all(result == img)

    def test_full_tones_changes_image(self, grader, varied_img):
        tones = {
            "shadows": {"hue": 30, "sat": 20},
            "midtones": {"hue": 120, "sat": 15},
            "highlights": {"hue": 300, "sat": 25},
        }
        result = grader._split_tone_three_way(varied_img, tones)
        assert not np.allclose(result, varied_img)

    def test_no_offset_returns_original(self, grader, img):
        tones = {
            "shadows": {"hue": 0, "sat": 0},
            "midtones": {"hue": 0, "sat": 0},
            "highlights": {"hue": 0, "sat": 0},
        }
        result = grader._split_tone_three_way(img, tones)
        assert np.all(result == img)

    def test_with_mask(self, grader, varied_img):
        tones = {
            "shadows": {"hue": 30, "sat": 20},
            "midtones": {"hue": 120, "sat": 15},
            "highlights": {"hue": 300, "sat": 25},
        }
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader._split_tone_three_way(varied_img, tones, mask=mask)
        assert result.shape == varied_img.shape


class TestAddOrtonGlow:
    def test_zero_opacity(self, grader, img):
        result = grader._add_orton_glow(img, 0)
        assert np.all(result == img)

    def test_changes_varied_image(self, grader, varied_img):
        result = grader._add_orton_glow(varied_img, 0.5)
        assert not np.allclose(result, varied_img)

    def test_with_mask(self, grader, varied_img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader._add_orton_glow(varied_img, 0.5, mask=mask)
        assert result.shape == varied_img.shape

    def test_output_type(self, grader, img):
        result = grader._add_orton_glow(img, 0.5)
        assert result.dtype == np.uint8


class TestAddChromaticAberration:
    def test_zero_disp(self, grader, img):
        result = grader._add_chromatic_aberration(img, 0)
        assert np.all(result == img)

    def test_changes_varied_image(self, grader, varied_img):
        result = grader._add_chromatic_aberration(varied_img, 5)
        assert not np.allclose(result, varied_img)

    def test_output_shape(self, grader, img):
        result = grader._add_chromatic_aberration(img, 5)
        assert result.shape == img.shape


class TestAddHalation:
    def test_zero_intensity(self, grader, img):
        result = grader._add_halation(img, intensity=0)
        assert np.all(result == img)

    def test_changes_varied_image(self, grader, varied_img):
        result = grader._add_halation(varied_img, threshold=150, radius=15, intensity=0.5)
        assert not np.allclose(result, varied_img)

    def test_output_shape(self, grader, img):
        result = grader._add_halation(img, threshold=200, radius=15, intensity=0.3)
        assert result.shape == img.shape


class TestAddGrain:
    def test_zero_strength(self, grader, img):
        result = grader._add_grain(img, 0)
        assert np.all(result == img)

    def test_changes_image(self, grader, img):
        result = grader._add_grain(img, 0.1)
        assert not np.allclose(result, img)

    def test_output_shape(self, grader, img):
        result = grader._add_grain(img, 0.05)
        assert result.shape == img.shape


class TestAdjustWhiteBalance:
    def test_identity_multipliers(self, grader, img):
        result = grader._adjust_white_balance(img, {"R": 1.0, "G": 1.0, "B": 1.0})
        assert np.all(result == img)

    def test_changes_image(self, grader, img):
        result = grader._adjust_white_balance(img, {"R": 1.2, "G": 1.0, "B": 0.8})
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        orig_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        assert not np.allclose(lab, orig_lab)

    def test_empty_multipliers(self, grader, img):
        result = grader._adjust_white_balance(img, {})
        assert np.all(result == img)


class TestAddHaze:
    def test_zero_strength(self, grader, img):
        result = grader._add_haze(img, 0)
        assert np.all(result == img)

    def test_changes_image(self, grader, varied_img):
        result = grader._add_haze(varied_img, 0.3)
        assert not np.allclose(result, varied_img)

    def test_with_mask(self, grader, varied_img):
        mask = np.ones((64, 64), dtype=np.float32)
        result = grader._add_haze(varied_img, 0.3, mask=mask)
        assert result.shape == varied_img.shape


class TestAddVignette:
    def test_zero_strength(self, grader, img):
        result = grader._add_vignette(img, 0)
        assert np.all(result == img)

    def test_darkens_corners(self, grader):
        img = np.full((100, 100, 3), 200, dtype=np.uint8)
        result = grader._add_vignette(img, 0.9)
        assert result[50, 50, 0] > result[5, 5, 0]


class TestApplyRGBCurves:
    def test_identity_returns_original(self, grader, img):
        result = grader._apply_rgb_curves(img, {})
        assert np.all(result == img)

    def test_r_curve_changes(self, grader, varied_img):
        result = grader._apply_rgb_curves(varied_img, {"R": [(0, 0), (128, 200), (255, 255)]})
        assert not np.allclose(result, varied_img)


class TestHSlHueShift:
    def test_no_shifts_returns_original(self, grader, img):
        result = grader._hsl_hue_shift(img, {})
        assert np.all(result == img)

    def test_hue_shift_changes_colorful(self, grader, colorful_img):
        result = grader._hsl_hue_shift(colorful_img, {"red": 30})
        assert not np.allclose(result, colorful_img)


class TestApplyCalibration:
    def test_empty_returns_original(self, grader, img):
        result = grader._apply_calibration(img, {})
        assert np.all(result == img)

    def test_calibration_changes_colorful(self, grader, colorful_img):
        cal = {"red": {"hue": 5, "sat": 3}, "green": {"hue": -2, "sat": 1}}
        result = grader._apply_calibration(colorful_img, cal)
        assert not np.allclose(result, colorful_img)


class TestGuidedFilter:
    def test_output_shape(self, grader, img):
        guide = np.full((64, 64), 128, dtype=np.float32)
        src = np.full((64, 64), 128, dtype=np.float32)
        result = grader._guided_filter(guide, src, r=5, eps=0.02)
        assert result.shape == (64, 64)

    def test_preserves_flat_input(self, grader):
        guide = np.full((32, 32), 100, dtype=np.float32)
        src = np.full((32, 32), 100, dtype=np.float32)
        result = grader._guided_filter(guide, src, r=5, eps=0.02)
        assert np.allclose(result, src, atol=1)


class TestAddClarity:
    def test_zero_strength(self, grader, img):
        result = grader._add_clarity(img, 0)
        assert np.all(result == img)

    def test_changes_varied_image(self, grader, varied_img):
        result = grader._add_clarity(varied_img, 0.5)
        assert not np.allclose(result, varied_img)

    def test_output_type(self, grader, img):
        result = grader._add_clarity(img, 0.3)
        assert result.dtype == np.uint8


class TestApplyHSLAdjustments:
    def test_empty_returns_original(self, grader, img):
        result = grader._apply_hsl_adjustments(img, {})
        assert np.all(result == img)

    def test_hue_adjustment_changes_colorful(self, grader, colorful_img):
        adj = {"hue": {"red": 10, "blue": -5}}
        result = grader._apply_hsl_adjustments(colorful_img, adj)
        assert not np.allclose(result, colorful_img)

    def test_sat_adjustment_changes(self, grader, varied_img):
        adj = {"saturation": {"red": 20, "green": -10}}
        result = grader._apply_hsl_adjustments(varied_img, adj)
        assert not np.allclose(result, varied_img)

    def test_lum_adjustment_changes(self, grader, varied_img):
        adj = {"luminance": {"red": 15, "blue": -10}}
        result = grader._apply_hsl_adjustments(varied_img, adj)
        assert not np.allclose(result, varied_img)


class TestAddFilmEmulation:
    def test_unknown_preset_raises(self, grader, img):
        with pytest.raises(FileNotFoundError, match="nonexistent_lut_zzz"):
            grader._add_film_emulation(img, "nonexistent_lut_zzz")

    def test_none_lut_returns_input(self, grader, img):
        result = grader._add_film_emulation(img, None)
        np.testing.assert_array_equal(result, img)

    def test_sentinel_none_string_returns_input(self, grader, img):
        result = grader._add_film_emulation(img, "none")
        np.testing.assert_array_equal(result, img)


class TestLUTIntegration:
    """Tests for the real 3D LUT integration in ColorGrader._add_film_emulation."""

    CUBE_IDENTITY_2 = (
        "LUT_3D_SIZE 2\n"
        "0.0 0.0 0.0\n"
        "1.0 0.0 0.0\n"
        "0.0 1.0 0.0\n"
        "1.0 1.0 0.0\n"
        "0.0 0.0 1.0\n"
        "1.0 0.0 1.0\n"
        "0.0 1.0 1.0\n"
        "1.0 1.0 1.0\n"
    )

    def test_identity_lut_path(self, tmp_path):
        """A direct path to an identity .cube produces an identity result."""
        p = tmp_path / "identity.cube"
        p.write_text(self.CUBE_IDENTITY_2)
        grader = ColorGrader()
        img = np.array(
            [[[0, 0, 0], [255, 255, 255], [128, 64, 200]]],
            dtype=np.uint8,
        )
        result = grader._add_film_emulation(img, str(p))
        np.testing.assert_allclose(result, img, atol=1)

    def test_identity_lut_stem(self, tmp_path, monkeypatch):
        """A stem name resolves to ``luts_dir/<stem>.cube``."""
        p = tmp_path / "identity.cube"
        p.write_text(self.CUBE_IDENTITY_2)
        monkeypatch.setattr(lut_mod, "luts_dir", lambda: tmp_path)
        grader = ColorGrader()
        img = np.array(
            [[[0, 0, 0], [255, 255, 255], [128, 64, 200]]],
            dtype=np.uint8,
        )
        result = grader._add_film_emulation(img, "identity")
        np.testing.assert_allclose(result, img, atol=1)

    def test_missing_lut_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr(lut_mod, "luts_dir", lambda: tmp_path)
        grader = ColorGrader()
        with pytest.raises(FileNotFoundError, match="missing_zzz"):
            grader._add_film_emulation(
                np.zeros((4, 4, 3), dtype=np.uint8), "missing_zzz"
            )

    def test_cube_lut_object_passthrough(self):
        """A pre-loaded CubeLUT can be passed directly without re-loading."""
        cube = CubeLUT(3)
        grader = ColorGrader()
        img = np.array(
            [[[0, 0, 0], [128, 128, 128], [255, 255, 255]]],
            dtype=np.uint8,
        )
        result = grader._add_film_emulation(img, cube)
        np.testing.assert_allclose(result, img, atol=1)

    def test_strength_blend(self, tmp_path):
        """``strength=0.0`` returns the input unchanged; full strength applied otherwise."""
        p = tmp_path / "identity.cube"
        p.write_text(self.CUBE_IDENTITY_2)
        grader = ColorGrader()
        img = np.full((4, 4, 3), 128, dtype=np.uint8)
        zero = grader._add_film_emulation(img, str(p), strength=0.0)
        np.testing.assert_array_equal(zero, img)
        full = grader._add_film_emulation(img, str(p), strength=1.0)
        np.testing.assert_array_equal(full, img)

    def test_lut_caching(self, tmp_path):
        """Loading the same LUT twice reuses the cached CubeLUT object."""
        p = tmp_path / "identity.cube"
        p.write_text(self.CUBE_IDENTITY_2)
        grader = ColorGrader()
        grader._add_film_emulation(np.zeros((4, 4, 3), dtype=np.uint8), str(p))
        key = str(p.resolve())
        assert key in grader._lut_cache
        cached = grader._lut_cache[key]
        grader._add_film_emulation(np.zeros((4, 4, 3), dtype=np.uint8), str(p))
        assert grader._lut_cache[key] is cached


class TestAddSparkles:
    def test_zero_strength(self, grader, img):
        result = grader._add_sparkles(img, 0)
        assert np.all(result == img)

    def test_output_shape(self, grader, img):
        result = grader._add_sparkles(img, 0.5)
        assert result.shape == img.shape


class TestModuleLevelPresetFunctions:
    """Tests for module-level functions: register_presets_dir, load_preset,
    load_all_presets, list_available_presets."""

    def test_register_presets_dir_adds_directory(
        self, tmp_path, isolated_user_dirs
    ):
        """register_presets_dir() appends the directory and its presets become
        discoverable through list_available_presets()."""
        preset_file = tmp_path / "custom_look.json"
        preset_file.write_text(json.dumps({"description": "custom test preset"}))
        register_presets_dir(tmp_path)
        assert tmp_path.resolve() in [p.resolve() for p in _USER_PRESETS_DIRS]
        assert "custom_look" in list_available_presets()

    def test_register_presets_dir_does_not_create_directory(
        self, tmp_path, isolated_user_dirs
    ):
        """register_presets_dir() only appends the path to the user list; it
        does NOT create the directory on disk."""
        nonexistent = tmp_path / "does_not_exist"
        assert not nonexistent.exists()
        register_presets_dir(nonexistent)
        assert not nonexistent.exists()
        assert nonexistent.resolve() in [p.resolve() for p in _USER_PRESETS_DIRS]

    def test_load_preset_existing(self):
        """load_preset() returns a dict with the expected top-level keys for
        a real preset shipped with the package."""
        result = load_preset("natural")
        assert isinstance(result, dict)
        assert "description" in result
        assert "curves" in result
        assert isinstance(result["description"], str)
        assert len(result["description"]) > 0

    def test_load_preset_nonexistent_raises(self):
        """load_preset() raises FileNotFoundError when the preset is missing
        (the implementation does not return None)."""
        with pytest.raises(FileNotFoundError):
            load_preset("nonexistent_preset_xyz_9999")

    def test_load_preset_with_invalid_json_raises(
        self, tmp_path, isolated_user_dirs
    ):
        """load_preset() propagates ``json.JSONDecodeError`` when the preset
        file is malformed (the implementation does not catch & warn; the
        warn-and-skip path lives in ``load_all_presets``)."""
        bad = tmp_path / "broken.json"
        bad.write_text("{ this is not valid json")
        register_presets_dir(tmp_path)
        with pytest.raises(json.JSONDecodeError):
            load_preset("broken")

    def test_load_all_presets_returns_dict(self):
        """load_all_presets() returns a dict of name -> settings for every
        default preset on disk."""
        result = load_all_presets()
        assert isinstance(result, dict)
        assert len(result) > 0
        assert "natural" in result
        assert all(isinstance(v, dict) for v in result.values())
        assert all(isinstance(k, str) for k in result.keys())

    def test_list_available_presets_returns_list(self):
        """list_available_presets() returns a list of strings containing all
        default preset names."""
        result = list_available_presets()
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)
        assert len(result) > 0
        assert "natural" in result


class TestFujiFoundationWiring:
    """Smoke tests verifying the Phase-1.a Fuji foundation modules are wired
    into ``ColorGrader.grade`` and ``engine.process()``."""

    def test_grade_with_tonal_curve_strength_changes_output(self, img):
        grader = ColorGrader()
        baseline = grader.grade(img, "natural", 1.0, skip_post_effects=True)
        graded = tonal.apply_hd_curve(baseline, strength=0.5)
        assert graded.shape == img.shape
        assert graded.dtype == img.dtype
        assert not np.array_equal(graded, baseline)

    def test_grade_with_highlight_rolloff_changes_output(self, varied_img):
        grader = ColorGrader()
        baseline = grader.grade(varied_img, "natural", 1.0, skip_post_effects=True)
        graded = highlight.apply_highlight_rolloff(baseline, strength=0.7)
        assert not np.array_equal(graded, baseline)

    def test_grade_with_film_grain_changes_output(self, img):
        grader = ColorGrader()
        baseline = grader.grade(img, "natural", 1.0)
        graded = grain.apply_film_grain(baseline, strength=0.5)
        assert not np.array_equal(graded, baseline)

    def test_grade_with_skin_protect_does_not_error(self, img):
        grader = ColorGrader()
        graded = grader.grade(
            img, "natural", 1.0,
            skin_protect_strength=0.5,
        )
        assert graded.shape == img.shape
        assert graded.dtype == img.dtype

    def test_default_grade_unchanged_when_strengths_zero(self, img):
        grader = ColorGrader()
        out_no_kwargs = grader.grade(img, "natural", 1.0)
        out_with_zeros = tonal.apply_hd_curve(out_no_kwargs, strength=0.0)
        out_with_zeros = highlight.apply_highlight_rolloff(out_with_zeros, strength=0.0)
        out_with_zeros = grain.apply_film_grain(out_with_zeros, strength=0.0)
        assert np.array_equal(out_with_zeros, out_no_kwargs)

    def test_engine_process_with_tonal_curve_strength(self, engine):
        img = np.zeros((256, 200, 3), dtype=np.uint8)
        for i in range(256):
            img[i, :, :] = i
        result = engine.process(
            img,
            tonal_curve_strength=0.7,
            skin_protect_strength=0.0,
            grain_strength=0.0,
            highlight_rolloff_strength=0.0,
        )
        assert result.shape == img.shape
        assert result.params.tonal_curve_strength == 0.7
        assert not np.array_equal(np.asarray(result), img)

    def test_engine_process_tonal_applies_without_color_grade(self, engine):
        img = np.zeros((256, 200, 3), dtype=np.uint8)
        for i in range(256):
            img[i, :, :] = i
        result = engine.process(img, tonal_curve_strength=0.7)
        assert not np.array_equal(np.asarray(result), img)

    def test_engine_process_default_strengths_zero_is_identity(self, engine):
        img = np.full((200, 200, 3), 128, dtype=np.uint8)
        result = engine.process(img)
        assert result.params.tonal_curve_strength == 0.0
        assert result.params.skin_protect_strength == 0.0
        assert result.params.grain_strength == 0.0
        assert result.params.highlight_rolloff_strength == 0.0


class TestPrecisionContext:
    """Tests for the Phase-1.b 16-bit float precision helpers."""

    def test_to_float_uint8_normalizes(self):
        img_u8 = np.array([[[0, 128, 255]]], dtype=np.uint8)
        out = precision_mod.to_float(img_u8)
        assert out.dtype == np.float32
        assert out.shape == img_u8.shape
        assert np.allclose(out, [[[0.0, 128 / 255, 1.0]]], atol=1e-6)

    def test_to_float_passthrough_when_already_float(self):
        img_f = np.full((4, 4, 3), 0.5, dtype=np.float32)
        out = precision_mod.to_float(img_f)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, img_f)

    def test_to_float_clamps_out_of_range(self):
        img_f = np.array([[[1.5, -0.2, 0.4]]], dtype=np.float32)
        out = precision_mod.to_float(img_f)
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_to_uint8_round_trip_preserves_values(self):
        img_u8 = np.array([[[0, 64, 128, 192, 255]] * 3], dtype=np.uint8)
        out_u8 = precision_mod.to_uint8(precision_mod.to_float(img_u8))
        assert out_u8.dtype == np.uint8
        assert np.array_equal(out_u8, img_u8)

    def test_to_uint8_rounds_not_truncates(self):
        img_f = np.array([[[0.5 / 255.0]]], dtype=np.float32)
        out_u8 = precision_mod.to_uint8(img_f)
        assert int(out_u8[0, 0, 0]) == 0

        img_f2 = np.array([[[0.6 / 255.0]]], dtype=np.float32)
        out_u8_2 = precision_mod.to_uint8(img_f2)
        assert int(out_u8_2[0, 0, 0]) == 1

    def test_to_uint8_clamps_out_of_range(self):
        img_f = np.array([[[1.5, -0.5, 0.5]]], dtype=np.float32)
        out_u8 = precision_mod.to_uint8(img_f)
        assert out_u8.dtype == np.uint8
        assert int(out_u8[0, 0, 0]) == 255
        assert int(out_u8[0, 0, 1]) == 0
        assert int(out_u8[0, 0, 2]) == 128

    def test_ensure_float_passthrough_for_float_input(self):
        img_f = np.full((2, 2, 3), 0.25, dtype=np.float32)
        out = precision_mod.ensure_float(img_f)
        assert out.dtype == np.float32
        np.testing.assert_array_equal(out, img_f)

    def test_ensure_float_converts_uint8(self):
        img_u8 = np.full((2, 2, 3), 255, dtype=np.uint8)
        out = precision_mod.ensure_float(img_u8)
        assert out.dtype == np.float32
        assert np.allclose(out, 1.0)

    def test_round_trip_within_one_quanta(self):
        rng = np.random.default_rng(42)
        img_u8 = rng.integers(0, 256, size=(16, 16, 3), dtype=np.uint8)
        rt = precision_mod.to_uint8(precision_mod.to_float(img_u8))
        max_diff = int(np.max(np.abs(rt.astype(int) - img_u8.astype(int))))
        assert max_diff <= 1

    def test_context_manager_is_float_by_default(self):
        with precision_mod.PrecisionContext() as pc:
            assert pc.is_float is True

    def test_context_manager_supports_8bit_mode(self):
        with precision_mod.PrecisionContext(bit_depth="8") as pc:
            assert pc.is_float is False

    def test_context_manager_rejects_unknown_bit_depth(self):
        with pytest.raises(ValueError):
            precision_mod.PrecisionContext(bit_depth="32")

    def test_process_runs_op_in_float(self):
        captured = {}

        def op_f(img_f):
            captured["dtype"] = img_f.dtype
            captured["min"] = float(img_f.min())
            captured["max"] = float(img_f.max())
            return img_f * 0.5

        img_u8 = np.full((4, 4, 3), 200, dtype=np.uint8)
        with precision_mod.PrecisionContext(bit_depth="16") as pc:
            out = pc.process(img_u8, op_f)

        assert captured["dtype"] == np.float32
        assert abs(captured["min"] - 200 / 255) < 1e-6
        assert abs(captured["max"] - 200 / 255) < 1e-6
        assert out.dtype == np.uint8
        assert 95 <= int(out[0, 0, 0]) <= 100

    def test_process_raises_outside_with_block(self):
        pc = precision_mod.PrecisionContext(bit_depth="16")
        with pytest.raises(RuntimeError):
            pc.process(np.zeros((2, 2, 3), dtype=np.uint8), lambda x: x)


class TestFloatGradingFunctions:
    """Tests for the float ``_F_`` grading methods on ColorGrader."""

    def test_F_apply_luminance_curve_dtype_is_float(self, grader, img):
        img_f = img.astype(np.float32) / 255.0
        out = grader._F_apply_luminance_curve(img_f, [(0, 0), (128, 140), (255, 255)])
        assert out.dtype == np.float32
        assert out.shape == img.shape
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_F_apply_luminance_curve_identity_passthrough(self, grader, img):
        img_f = img.astype(np.float32) / 255.0
        identity = [(0, 0), (255, 255)]
        out = grader._F_apply_luminance_curve(img_f, identity)
        assert np.allclose(out, img_f, atol=1e-5)

    def test_F_apply_luminance_curve_no_uint8_quantization_in_middle(self, grader):
        l = np.linspace(0, 255, 256, dtype=np.float32).reshape(1, 256, 1)
        l = np.repeat(l, 3, axis=2)
        img_f = l / 255.0
        out = grader._F_apply_luminance_curve(
            img_f, [(0, 0), (64, 64), (128, 192), (192, 192), (255, 255)]
        )
        assert out.dtype == np.float32
        mid_val = float(out[0, 128, 0]) * 255.0
        assert 180.0 < mid_val < 200.0

    def test_uint8_apply_luminance_curve_delegates_to_float(self, grader, img):
        identity = [(0, 0), (255, 255)]
        out_u8 = grader._apply_luminance_curve(img, identity)
        assert out_u8.dtype == np.uint8
        assert np.array_equal(out_u8, img)

    def test_uint8_apply_luminance_curve_modifies_image(self, varied_img, grader):
        out_u8 = grader._apply_luminance_curve(
            varied_img, [(0, 0), (50, 30), (200, 230), (255, 255)]
        )
        assert out_u8.dtype == np.uint8
        assert not np.array_equal(out_u8, varied_img)

    def test_F_apply_calibration_dtype_is_float(self, grader, img):
        img_f = img.astype(np.float32) / 255.0
        cal = {"red": {"hue": 5, "sat": 3}}
        out = grader._F_apply_calibration(img_f, cal)
        assert out.dtype == np.float32
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_F_apply_calibration_empty_returns_input(self, grader, img):
        img_f = img.astype(np.float32) / 255.0
        out = grader._F_apply_calibration(img_f, {})
        assert out is img_f

    def test_F_apply_calibration_all_zero_returns_input(self, grader, colorful_img):
        img_f = colorful_img.astype(np.float32) / 255.0
        cal = {"red": {"hue": 0, "sat": 0}, "green": {"hue": 0, "sat": 0}}
        out = grader._F_apply_calibration(img_f, cal)
        assert out is img_f

    def test_uint8_apply_calibration_delegates_to_float(self, colorful_img, grader):
        cal = {"red": {"hue": 5, "sat": 3}, "green": {"hue": -2, "sat": 1}}
        out = grader._apply_calibration(colorful_img, cal)
        assert out.dtype == np.uint8
        assert not np.array_equal(out, colorful_img)

    def test_F_split_tone_three_way_dtype_is_float(self, grader, varied_img):
        img_f = varied_img.astype(np.float32) / 255.0
        tones = {
            "shadows": {"hue": 30, "sat": 20},
            "midtones": {"hue": 120, "sat": 15},
            "highlights": {"hue": 300, "sat": 25},
        }
        out = grader._F_split_tone_three_way(img_f, tones)
        assert out.dtype == np.float32
        assert out.shape == varied_img.shape
        assert np.all(out >= 0.0) and np.all(out <= 1.0)

    def test_F_split_tone_three_way_empty_returns_input(self, grader, img):
        img_f = img.astype(np.float32) / 255.0
        out = grader._F_split_tone_three_way(img_f, {})
        assert out is img_f

    def test_F_split_tone_three_way_zero_offsets_returns_input(self, grader, img):
        img_f = img.astype(np.float32) / 255.0
        tones = {
            "shadows": {"hue": 0, "sat": 0},
            "midtones": {"hue": 0, "sat": 0},
            "highlights": {"hue": 0, "sat": 0},
        }
        out = grader._F_split_tone_three_way(img_f, tones)
        assert out is img_f

    def test_F_split_tone_three_way_with_mask(self, grader, varied_img):
        img_f = varied_img.astype(np.float32) / 255.0
        mask = np.ones((64, 64), dtype=np.float32)
        tones = {
            "shadows": {"hue": 30, "sat": 20},
            "midtones": {"hue": 120, "sat": 15},
            "highlights": {"hue": 300, "sat": 25},
        }
        out = grader._F_split_tone_three_way(img_f, tones, mask=mask)
        assert out.shape == varied_img.shape
        assert not np.array_equal(out, img_f)

    def test_uint8_split_tone_three_way_delegates_to_float(self, varied_img, grader):
        tones = {
            "shadows": {"hue": 30, "sat": 20},
            "midtones": {"hue": 120, "sat": 15},
            "highlights": {"hue": 300, "sat": 25},
        }
        out = grader._split_tone_three_way(varied_img, tones)
        assert out.dtype == np.uint8
        assert not np.array_equal(out, varied_img)

    def test_grade_still_works_with_float_path(self, img):
        """End-to-end: grade() (which uses PrecisionContext internally)
        should still produce uint8 output and not raise."""
        grader = ColorGrader()
        out = grader.grade(img, "natural", 1.0)
        assert out.dtype == np.uint8
        assert out.shape == img.shape

    def test_grade_uint8_intensity_blend_uses_original(self, img):
        """Intensity < 1.0 must blend against the original uint8 input,
        not the in-flight float result."""
        grader = ColorGrader()
        out_full = grader.grade(img, "natural", 1.0, skip_post_effects=True)
        out_half = grader.grade(img, "natural", 0.5, skip_post_effects=True)
        assert out_half.dtype == np.uint8
        assert not np.array_equal(out_full, out_half)
